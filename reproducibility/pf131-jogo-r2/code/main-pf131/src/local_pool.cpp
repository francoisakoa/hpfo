#include "internal.hpp"
#include "linear.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <filesystem>
#include <exception>

#if defined(__unix__) || defined(__APPLE__)
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <spawn.h>
#include <fcntl.h>
extern char** environ;
#endif

namespace pf::detail {
namespace {
void add_counters(Counters& a,const Counters& b) {
    a.objective_pf+=b.objective_pf;a.objective_local+=b.objective_local;
    a.gradient+=b.gradient;a.hessian+=b.hessian;a.invalid_objective+=b.invalid_objective;
    a.local_calls+=b.local_calls;a.local_converged+=b.local_converged;
    a.local_failed+=b.local_failed;a.local_limited+=b.local_limited;a.local_improvements+=b.local_improvements;
    a.linear_analyses+=b.linear_analyses;a.linear_factorizations+=b.linear_factorizations;a.linear_solves+=b.linear_solves;a.linear_cg_fallbacks+=b.linear_cg_fallbacks;
    a.linear_metis_analyses+=b.linear_metis_analyses;a.mumps_user_ordering_analyses+=b.mumps_user_ordering_analyses;
    a.hessian_products+=b.hessian_products;a.newton_iterations+=b.newton_iterations;
    a.newton_accepted+=b.newton_accepted;a.newton_rejected+=b.newton_rejected;a.negative_curvature+=b.negative_curvature;
}
#if defined(__unix__) || defined(__APPLE__)
bool transfer(int fd,void* data,std::size_t size,bool sending) {
    auto* p=static_cast<char*>(data);
    while(size) {
        ssize_t n;
        if(sending) {
#ifdef MSG_NOSIGNAL
            n=send(fd,p,size,MSG_NOSIGNAL);
#else
            n=send(fd,p,size,0);
#endif
        } else n=recv(fd,p,size,0);
        if(n<0&&errno==EINTR) continue;
        if(n<=0) return false;
        size-=static_cast<std::size_t>(n);p+=n;
    }
    return true;
}
template<class T> bool send_data(int fd,const T& value) {return transfer(fd,const_cast<T*>(&value),sizeof(value),true);}
template<class T> bool read_data(int fd,T& value) {return transfer(fd,&value,sizeof(value),false);}
void send_vector(int fd,const Vector& x) {
    if(!transfer(fd,const_cast<double*>(x.data()),sizeof(double)*x.size(),true)) throw std::runtime_error("Worker socket write failed");
}
bool read_vector(int fd,Vector& x) {return transfer(fd,x.data(),sizeof(double)*x.size(),false);}
struct Request {std::uint64_t allowance;double max_seconds, ball_radius;};
struct Startup {
    int atoms, dimension, local_method, max_iterations, exact_hessian, print_level, ball_constraints;
    double tolerance, newton_radius;
    int newton_cg_max_iterations;
    char linear_solver[128], newton_linear_solver[32], hsl_library[4096], newton_linear_ordering[16];
};
struct Response {
    Counters counters;
    double objective, best, work_seconds;
    int stop; // 0 complete, 1 time, 2 evaluation, 3 error
    char error[512];
};
#endif
}
struct LocalPool::Impl {
    Context& context;
#if defined(__unix__) || defined(__APPLE__)
    struct Worker {pid_t pid;int fd;};
    std::vector<Worker> workers;
    void cleanup() noexcept {
        for(auto w:workers) {close(w.fd);kill(w.pid,SIGTERM);}
        for(auto w:workers) {int status;while(waitpid(w.pid,&status,0)<0&&errno==EINTR) {}}
        workers.clear();
    }
#endif
    explicit Impl(Context& c):context(c) {
        if(context.options.local_workers<=1||context.options.local_method==LocalMethod::none||newton_uses_threads(context.options)) return;
#if defined(__unix__) || defined(__APPLE__)
        const auto* lj=dynamic_cast<const LennardJones*>(&context.problem);
        if(!lj) throw std::invalid_argument("Process workers currently support LennardJones; use local_workers=1 for custom problems");
        std::string executable=context.options.worker_executable;
        if(executable.empty()) {
            char path[4096];ssize_t n=readlink("/proc/self/exe",path,sizeof(path)-1);
            if(n<=0) throw std::invalid_argument("Set worker_executable to the absolute pf_local_worker path");
            path[n]='\0';executable=(std::filesystem::path(path).parent_path()/"pf_local_worker").string();
        }
        Startup startup{};startup.atoms=lj->atoms();startup.dimension=lj->spatial_dimension();
        startup.local_method=static_cast<int>(context.options.local_method);
        startup.max_iterations=context.options.local_max_iterations;startup.exact_hessian=context.options.ipopt_exact_hessian;
        startup.print_level=context.options.ipopt_print_level;startup.tolerance=context.options.local_tolerance;
        startup.ball_constraints=context.options.ipopt_ball_constraints;
        startup.newton_radius=context.options.newton_initial_radius;
        startup.newton_cg_max_iterations=context.options.newton_cg_max_iterations;
        if(context.options.newton_linear_solver.size()>=sizeof(startup.newton_linear_solver)||context.options.hsl_library.size()>=sizeof(startup.hsl_library))throw std::invalid_argument("Worker linear backend name or HSL path too long");
        std::strcpy(startup.newton_linear_solver,context.options.newton_linear_solver.c_str());
        if(context.options.newton_linear_ordering.size()>=sizeof(startup.newton_linear_ordering))throw std::invalid_argument("Worker ordering name too long");
        std::strcpy(startup.newton_linear_ordering,context.options.newton_linear_ordering.c_str());
        std::strcpy(startup.hsl_library,context.options.hsl_library.c_str());
        if(context.options.ipopt_linear_solver.size()>=sizeof(startup.linear_solver)) throw std::invalid_argument("Ipopt solver name too long");
        std::strcpy(startup.linear_solver,context.options.ipopt_linear_solver.c_str());
        std::vector<std::string> environment;
        for(char** e=environ;*e;++e) {
            std::string entry=*e;
            if(entry.rfind("OMP_NUM_THREADS=",0)&&entry.rfind("OPENBLAS_NUM_THREADS=",0)&&entry.rfind("MKL_NUM_THREADS=",0)&&entry.rfind("BLIS_NUM_THREADS=",0)) environment.push_back(entry);
        }
        environment.insert(environment.end(),{"OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","BLIS_NUM_THREADS=1"});
        std::vector<char*> envp;for(auto& s:environment) envp.push_back(s.data());envp.push_back(nullptr);
        // posix_spawn + exec: safe even after earlier OpenMP/BLAS calls in the parent.
        try {
            for(int i=0;i<std::min(context.options.local_workers,context.options.particles);++i) {
                int sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)!=0) throw std::runtime_error("socketpair failed");
                fcntl(sockets[0],F_SETFD,FD_CLOEXEC);fcntl(sockets[1],F_SETFD,FD_CLOEXEC);
                posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
                posix_spawn_file_actions_addclose(&actions,sockets[0]);
                posix_spawn_file_actions_adddup2(&actions,sockets[1],3);
                if(sockets[1]!=3) posix_spawn_file_actions_addclose(&actions,sockets[1]);
                char fd_arg[]="--fd=3";char* argv[]={executable.data(),fd_arg,nullptr};
                pid_t pid=0;int error=posix_spawn(&pid,executable.c_str(),&actions,nullptr,argv,envp.data());
                posix_spawn_file_actions_destroy(&actions);
                if(error) {close(sockets[0]);close(sockets[1]);throw std::runtime_error("Cannot start pf_local_worker: "+std::string(std::strerror(error)));}
                close(sockets[1]);workers.push_back({pid,sockets[0]});
                if(!send_data(sockets[0],startup)) throw std::runtime_error("Worker initialization failed");
                send_vector(sockets[0],context.original_bounds.lower);send_vector(sockets[0],context.original_bounds.upper);
            }
        } catch(...) {cleanup();throw;}
#else
        throw std::invalid_argument("local_workers > 1 requires POSIX process isolation (Linux/WSL/macOS)");
#endif
    }
    ~Impl() {
#if defined(__unix__) || defined(__APPLE__)
        cleanup();
#endif
    }
};
LocalPool::LocalPool(Context& c):impl_(std::make_unique<Impl>(c)) {}
LocalPool::~LocalPool()=default;
void LocalPool::shutdown() {impl_.reset();}
void LocalPool::refine_selected(Population& p,const std::vector<std::size_t>& indices,const Bounds& b,std::vector<double>* values) {
    auto& c=impl_->context;
    if(indices.empty()||c.options.local_method==LocalMethod::none) return;
    if(values&&values->size()!=p.size()) throw std::invalid_argument("Local value cache dimension mismatch");
    const auto begin=std::chrono::steady_clock::now();
    auto record=[&]{c.local_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();};
    try {
        if(newton_uses_threads(c.options)&&c.options.local_workers>1) {
            struct Task {LocalResult result{{},infinity,-1};Vector best;double best_value=infinity,seconds=0;Counters counters;std::string stop;std::exception_ptr error;};
            for(std::size_t first=0;first<indices.size();) {
                c.check(true);
                std::size_t count=std::min(std::size_t(c.options.local_workers),indices.size()-first);
                std::uint64_t remaining=0;
                if(c.options.max_objective_evaluations) {
                    remaining=c.options.max_objective_evaluations-c.counters.objective_total();
                    count=std::min(count,std::size_t(remaining));
                }
                std::vector<Task> tasks(count);
                #pragma omp parallel for schedule(static) num_threads(count)
                for(std::int64_t k=0;k<static_cast<std::int64_t>(count);++k) {
                    auto options=c.options;options.local_workers=1;options.batch_threads=1;
                    options.max_objective_evaluations=remaining?remaining/count+(std::size_t(k)<remaining%count):0;
                    Context local(c.problem,options,c.original_bounds);
                    local.start=c.start;local.active_ball_radius=c.active_ball_radius;
                    auto& task=tasks[k];
                    LocalCandidate candidate{b,{},infinity};
                    try {task.result=refine(local,p.at(indices[first+k]),b,&candidate);}
                    catch(const Stop& s){task.stop=s.reason;task.result={std::move(candidate.x),candidate.objective,5};}
                    catch(...){task.error=std::current_exception();}
                    task.best=local.best_x;task.best_value=local.best;
                    task.counters=local.counters;task.seconds=local.local_work_seconds;
                }
                bool timeout=false;
                // Fixed merge order preserves results independently of scheduling.
                for(std::size_t k=0;k<count;++k) {
                    auto& task=tasks[k];add_counters(c.counters,task.counters);c.local_work_seconds+=task.seconds;
                    if(task.best.size())c.observe(task.best,task.best_value);
                    if(task.error)std::rethrow_exception(task.error);
                    if(task.stop=="time_limit")timeout=true;
                    if(task.stop.empty()) {
                        p.at(indices[first+k])=std::move(task.result.x);
                        if(values)values->at(indices[first+k])=task.result.objective;
                    } else if(task.result.x.size()&&b.contains(task.result.x)&&std::isfinite(task.result.objective)) {
                        p.at(indices[first+k])=std::move(task.result.x);
                        if(values)values->at(indices[first+k])=task.result.objective;
                    }
                }
                first+=count;if(timeout)throw Stop("time_limit");
            }
        } else
#if defined(__unix__) || defined(__APPLE__)
        if(!impl_->workers.empty()) {
            for(std::size_t first=0;first<indices.size();) {
                c.check(true);
                std::size_t count=std::min(impl_->workers.size(),indices.size()-first);
                std::uint64_t remaining=0;
                if(c.options.max_objective_evaluations) {
                    remaining=c.options.max_objective_evaluations-c.counters.objective_total();
                    count=std::min(count,static_cast<std::size_t>(remaining));
                }
                for(std::size_t w=0;w<count;++w) {
                    Request request{remaining?remaining/count+(w<remaining%count?1:0):0,std::max(1e-9,c.options.max_seconds-c.seconds()),c.active_ball_radius};
                    int fd=impl_->workers[w].fd;
                    if(!send_data(fd,request)) throw std::runtime_error("Worker request failed");
                    send_vector(fd,p.at(indices[first+w]));send_vector(fd,b.lower);send_vector(fd,b.upper);
                }
                bool timeout=false;
                for(std::size_t w=0;w<count;++w) {
                    Response r{};int fd=impl_->workers[w].fd;
                    Vector x(c.problem.dimension()),best(c.problem.dimension());
                    if(!read_data(fd,r)||!read_vector(fd,x)||!read_vector(fd,best)) throw std::runtime_error("Local worker terminated unexpectedly");
                    add_counters(c.counters,r.counters);c.local_work_seconds+=r.work_seconds;
                    c.observe(best,r.best);
                    if(r.stop==3) throw std::runtime_error(std::string("Local worker: ")+r.error);
                    if(r.stop==1) timeout=true;
                    if(r.stop==0) {p.at(indices[first+w])=std::move(x);if(values)values->at(indices[first+w])=r.objective;}
                    else if(b.contains(x)&&std::isfinite(r.objective)) {p.at(indices[first+w])=std::move(x);if(values)values->at(indices[first+w])=r.objective;}
                }
                first+=count;
                if(timeout) throw Stop("time_limit");
            }
        } else
#endif
        {
            for(auto i:indices) {
                LocalCandidate candidate{b,{},infinity};
                try {auto r=refine(c,p.at(i),b,&candidate);p.at(i)=std::move(r.x);if(values)values->at(i)=r.objective;}
                catch(const Stop&) {
                    if(candidate.x.size()) {p.at(i)=std::move(candidate.x);if(values)values->at(i)=candidate.objective;}
                    throw;
                }
            }
        }
    } catch(...) {record();throw;}
    record();
}
int local_worker_main(int fd) {
#if defined(__unix__) || defined(__APPLE__)
    Startup startup{};
    if(!read_data(fd,startup)) return 2;
    LennardJones problem(startup.atoms,startup.dimension);const auto dimension=problem.dimension();
    Bounds original{Vector(dimension),Vector(dimension)};
    if(!read_vector(fd,original.lower)||!read_vector(fd,original.upper)) return 2;
    for(;;) {
        Request request{};if(!read_data(fd,request)) return 0;
        Vector x(dimension);Bounds b{Vector(dimension),Vector(dimension)};
        if(!read_vector(fd,x)||!read_vector(fd,b.lower)||!read_vector(fd,b.upper)) return 2;
        Options o;o.local_workers=1;o.batch_threads=1;
        o.local_method=static_cast<LocalMethod>(startup.local_method);o.local_max_iterations=startup.max_iterations;
        o.ipopt_exact_hessian=startup.exact_hessian;o.ipopt_print_level=startup.print_level;
        o.ipopt_ball_constraints=startup.ball_constraints;o.radius=request.ball_radius;
        o.newton_initial_radius=startup.newton_radius;o.newton_cg_max_iterations=startup.newton_cg_max_iterations;
        o.newton_linear_solver=startup.newton_linear_solver;o.hsl_library=startup.hsl_library;
        o.newton_linear_ordering=startup.newton_linear_ordering;
        o.local_tolerance=startup.tolerance;o.ipopt_linear_solver=startup.linear_solver;
        o.max_objective_evaluations=request.allowance;o.max_seconds=request.max_seconds;
        Context local(problem,o,original);
        Response response{};response.objective=infinity;Vector result=x;
        LocalCandidate candidate{b,{},infinity};
        try {auto r=refine(local,x,b,&candidate);result=r.x;response.objective=r.objective;}
        catch(const Stop& s) {
            response.stop=s.reason=="time_limit"?1:2;
            if(candidate.x.size()) {result=std::move(candidate.x);response.objective=candidate.objective;}
        }
        catch(const std::exception& e) {response.stop=3;std::strncpy(response.error,e.what(),sizeof(response.error)-1);}
        catch(...) {response.stop=3;std::strncpy(response.error,"Unknown worker failure",sizeof(response.error)-1);}
        response.counters=local.counters;response.best=local.best;response.work_seconds=local.local_work_seconds;
        Vector best=local.best_x.size()==dimension?local.best_x:x;
        if(!send_data(fd,response)) return 2;
        try {send_vector(fd,result);send_vector(fd,best);} catch(...) {return 2;}
    }
#else
    (void)fd;return 2;
#endif
}
} // namespace pf::detail
