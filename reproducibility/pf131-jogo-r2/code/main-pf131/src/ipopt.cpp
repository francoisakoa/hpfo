#include "internal.hpp"
#include <cmath>
#include <exception>
#include <memory>

#ifdef PF_WITH_IPOPT
#include <IpStdCInterface.h>
static_assert(std::is_same_v<Number,double>,"Ipopt must use binary64 double precision");

namespace pf::detail {
namespace {
struct Data {
    Context& context;
    Bounds bounds;
    Vector best_x;
    double best=infinity;
    Vector last_x;
    double last_value=infinity;
    std::exception_ptr error;
};
template<class F> Bool guard(Data& d,F&& f) noexcept {
    if(d.error) return false;
    try {f();return true;} catch(...) {d.error=std::current_exception();return false;}
}
Bool value_cb(Index n,Number* x,Bool,Number* value,UserDataPtr ptr) {
    auto& d=*static_cast<Data*>(ptr);
    return guard(d,[&] {
        Vector point=Eigen::Map<Vector>(x,n);
        d.context.check();
        if(d.last_x.size()==n && (point.array()==d.last_x.array()).all()) *value=d.last_value;
        else {*value=d.context.evaluate(point,true);d.last_x=point;d.last_value=*value;}
        if(!std::isfinite(*value)) throw std::runtime_error("Nonfinite Ipopt objective");
        if(d.bounds.contains(point) && d.context.local_feasible(point) && *value<d.best) {d.best=*value;d.best_x=std::move(point);}
    });
}
Bool gradient_cb(Index n,Number* x,Bool,Number* g,UserDataPtr ptr) {
    auto& d=*static_cast<Data*>(ptr);
    return guard(d,[&] {
        Vector result=d.context.gradient(Eigen::Map<Vector>(x,n));
        if(!result.allFinite()) throw std::runtime_error("Nonfinite Ipopt gradient");
        Eigen::Map<Vector>(g,n)=result;
    });
}
Bool constraints_cb(Index,Number* x,Bool,Index m,Number* g,UserDataPtr ptr) {
    auto& d=*static_cast<Data*>(ptr);
    return guard(d,[&] {
        const int dim=d.context.ball_dimension();
        for(Index atom=0;atom<m;++atom) {
            double norm2=0;for(int j=0;j<dim;++j) norm2+=x[atom*dim+j]*x[atom*dim+j];
            g[atom]=norm2;
        }
    });
}
Bool jacobian_cb(Index n,Number* x,Bool,Index m,Index,Index* row,Index* col,Number* values,UserDataPtr ptr) {
    auto& d=*static_cast<Data*>(ptr);
    return guard(d,[&] {
        if(!m) return;
        const int dim=d.context.ball_dimension();
        for(Index i=0;i<n;++i) {
            if(values) values[i]=2*x[i];
            else {row[i]=i/dim;col[i]=i;}
        }
    });
}
Bool hessian_cb(Index n,Number* x,Bool,Number factor,Index m,Number* lambda,Bool,Index,Index* row,Index* col,Number* values,UserDataPtr ptr) {
    auto& d=*static_cast<Data*>(ptr);
    return guard(d,[&] {
        if(!values) {Index k=0;for(Index i=0;i<n;++i) for(Index j=0;j<=i;++j) {row[k]=i;col[k]=j;++k;}return;}
        Matrix h=d.context.hessian(Eigen::Map<Vector>(x,n));
        if(!h.allFinite()) throw std::runtime_error("Nonfinite Ipopt Hessian");
        const int dim=d.context.ball_dimension();
        Index k=0;for(Index i=0;i<n;++i) for(Index j=0;j<=i;++j)
            values[k++]=factor*h(i,j)+(m&&i==j?2*lambda[i/dim]:0);
    });
}
Bool intermediate_cb(Index,Index,Number,Number,Number,Number,Number,Number,Number,Number,Index,UserDataPtr ptr) {
    auto& d=*static_cast<Data*>(ptr);return guard(d,[&]{d.context.check();});
}
char* mutable_string(const char* s) {return const_cast<char*>(s);}
void str(IpoptProblem p,const char* key,const std::string& value) {
    if(!AddIpoptStrOption(p,mutable_string(key),const_cast<char*>(value.c_str()))) throw std::runtime_error(std::string("Invalid Ipopt option: ")+key);
}
void num(IpoptProblem p,const char* key,double value) {
    if(!AddIpoptNumOption(p,mutable_string(key),value)) throw std::runtime_error(std::string("Invalid Ipopt option: ")+key);
}
void integer(IpoptProblem p,const char* key,int value) {
    if(!AddIpoptIntOption(p,mutable_string(key),value)) throw std::runtime_error(std::string("Invalid Ipopt option: ")+key);
}
}
LocalResult refine_ipopt(Context& c,const Vector& initial,const Bounds& bounds,double initial_value) {
    c.check(true);
    const Eigen::Index dimension=initial.size();
    if(dimension>30000) throw std::invalid_argument("Exact dense Ipopt interface supports at most 30000 variables");
    const Index n=static_cast<Index>(dimension);
    Vector x=bounds.project(initial),lower=bounds.lower,upper=bounds.upper;
    const bool exact=c.options.ipopt_exact_hessian&&c.problem.has_hessian();
    const int ball_dim=c.ball_dimension();
    const Index m=ball_dim?n/ball_dim:0;
    Vector g_lower=Vector::Constant(m,-1e20),g_upper=Vector::Constant(m,c.active_ball_radius*c.active_ball_radius);
    Data data{c,bounds,x,c.local_feasible(x)?initial_value:infinity,x,initial_value,{}};
    IpoptProblem raw=CreateIpoptProblem(n,lower.data(),upper.data(),m,m?g_lower.data():nullptr,m?g_upper.data():nullptr,m?n:0,exact?n*(n+1)/2:0,0,
                                        value_cb,constraints_cb,gradient_cb,jacobian_cb,hessian_cb);
    if(!raw) throw std::runtime_error("CreateIpoptProblem failed");
    std::unique_ptr<IpoptProblemInfo,decltype(&FreeIpoptProblem)> problem(raw,FreeIpoptProblem);
    str(raw,"option_file_name","");str(raw,"sb","yes");
    str(raw,"linear_solver",c.options.ipopt_linear_solver);
    if(!c.options.hsl_library.empty())str(raw,"hsllib",c.options.hsl_library);
    str(raw,"hessian_approximation",exact?"exact":"limited-memory");
    num(raw,"tol",c.options.local_tolerance);
    num(raw,"bound_relax_factor",0); // returned incumbent must satisfy the requested box
    if(m) {num(raw,"constr_viol_tol",1e-10);num(raw,"acceptable_constr_viol_tol",1e-10);}
    integer(raw,"max_iter",c.options.local_max_iterations);
    integer(raw,"print_level",c.options.ipopt_print_level);
    if(!SetIntermediateCallback(raw,intermediate_cb)) throw std::runtime_error("Ipopt callback registration failed");
    Number f=infinity;
    int status=static_cast<int>(IpoptSolve(raw,x.data(),nullptr,&f,nullptr,nullptr,nullptr,&data));
    if(data.error) std::rethrow_exception(data.error);
    if(status==-12||status==-11||status==-10||status<=-100)
        throw std::runtime_error("Ipopt configuration/runtime failure, status="+std::to_string(status)+", linear_solver="+c.options.ipopt_linear_solver);
    // Retain only callback-evaluated coordinates; no unchecked terminal objective.
    return {data.best_x,data.best,status};
}
} // namespace pf::detail
namespace pf {
bool ipopt_available() {return true;}
std::string ipopt_version() {return IPOPT_VERSION;}
}
#else
namespace pf {
bool ipopt_available() {return false;}
std::string ipopt_version() {return "unavailable";}
}
namespace pf::detail {
LocalResult refine_ipopt(Context&,const Vector&,const Bounds&,double) {throw std::runtime_error("Ipopt unavailable in this build");}
}
#endif
