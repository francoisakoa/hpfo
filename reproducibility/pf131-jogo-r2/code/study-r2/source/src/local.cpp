#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <exception>

namespace pf::detail {
int Context::ball_dimension() const {
    const auto* lj=dynamic_cast<const LennardJones*>(&problem);
    return lj && (options.local_method==LocalMethod::ipopt||options.local_method==LocalMethod::lj_newton) && options.ipopt_ball_constraints ? lj->spatial_dimension() : 0;
}
bool Context::local_feasible(const Vector& x) const {
    if(x.size()!=problem.dimension() || !x.allFinite()) return false;
    const int d=ball_dimension();
    if(!d) return true;
    const double limit=active_ball_radius*active_ball_radius;
    for(Eigen::Index i=0;i<x.size();i+=d)
        if(x.segment(i,d).squaredNorm()>limit+1e-10*std::max(1.0,limit)) return false;
    return true;
}
double Context::seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); }
void Context::check(bool objective_request) const {
    if(seconds()>=options.max_seconds) throw Stop("time_limit");
    if(objective_request && options.max_objective_evaluations && counters.objective_total()>=options.max_objective_evaluations)
        throw Stop("evaluation_limit");
}
void Context::observe(const Vector& x,double f) {
    if(std::isfinite(f)&&f<best&&original_bounds.contains(x) && (!require_local_feasibility||local_feasible(x))) {best=f;best_x=x;}
}
double Context::evaluate(const Vector& x,bool local) {
    check(true);
    if(local) ++counters.objective_local;else ++counters.objective_pf;
    double f=problem.value(x);
    if(!std::isfinite(f)) {++counters.invalid_objective;return infinity;}
    if(!local||local_feasible(x)) {
        observe(x,f);
        // Trial acceptance and local convergence do not determine whether a
        // feasible, evaluated improvement is useful to the particle filter.
        if(local&&local_candidate&&f<local_candidate->objective&&
           original_bounds.contains(x)&&local_candidate->bounds.contains(x)) {
            local_candidate->x=x;local_candidate->objective=f;
        }
    }
    return f;
}
std::vector<double> Context::batch(const Population& p) {
    check(true);
    std::size_t count=p.size();
    if(options.max_objective_evaluations) count=std::min(count,static_cast<std::size_t>(options.max_objective_evaluations-counters.objective_total()));
    std::vector<double> result(p.size(),infinity);std::vector<std::exception_ptr> errors(count);
    #pragma omp parallel for schedule(static) num_threads(options.batch_threads) if(count>8)
    for(std::int64_t i=0;i<static_cast<std::int64_t>(count);++i) {
        try {result[i]=problem.value(p[i]);} catch(...) {errors[i]=std::current_exception();}
    }
    counters.objective_pf+=count;
    for(std::size_t i=0;i<count;++i) {
        if(errors[i]) std::rethrow_exception(errors[i]);
        if(!std::isfinite(result[i])) {result[i]=infinity;++counters.invalid_objective;}
        observe(p[i],result[i]);
    }
    if(count<p.size()) throw Stop("evaluation_limit");
    check();return result;
}
Vector Context::gradient(const Vector& x) {
    check();++counters.gradient;Vector result=problem.gradient(x);
    if(result.size()!=problem.dimension()) throw std::runtime_error("Problem returned an incorrectly sized gradient");
    return result;
}
Matrix Context::hessian(const Vector& x) {
    check();++counters.hessian;Matrix result=problem.hessian(x);
    if(result.rows()!=problem.dimension()||result.cols()!=problem.dimension()) throw std::runtime_error("Problem returned an incorrectly sized Hessian");
    return result;
}
Vector projected_gradient(const Vector& x,const Vector& g,const Bounds& b) {
    Vector p=g;
    for(Eigen::Index i=0;i<x.size();++i) if((x[i]<=b.lower[i]+1e-12&&g[i]>0)||(x[i]>=b.upper[i]-1e-12&&g[i]<0)) p[i]=0;
    return p;
}
namespace {
class CandidateScope {
    Context& context;
    LocalCandidate* previous;
public:
    CandidateScope(Context& c,LocalCandidate& candidate):context(c),previous(c.local_candidate) {c.local_candidate=&candidate;}
    ~CandidateScope() {context.local_candidate=previous;}
};
LocalResult projected_lbfgs(Context& c,const Vector& initial,const Bounds& b,double initial_value) {
    Vector x=b.project(initial);double f=initial_value;Vector g=c.gradient(x);
    std::deque<Vector> sh,yh;std::deque<double> rho;int status=-1;
    for(int iteration=0;iteration<c.options.local_max_iterations;++iteration) {
        c.check();
        Vector pg=projected_gradient(x,g,b);
        if(!g.allFinite()||!std::isfinite(f)) {status=-2;break;}
        if(pg.lpNorm<Eigen::Infinity>()<=c.options.local_tolerance) {status=0;break;}
        Vector q=pg;std::vector<double> alpha(sh.size());
        for(std::size_t k=sh.size();k-->0;) {alpha[k]=rho[k]*sh[k].dot(q);q-=alpha[k]*yh[k];}
        double scaling=sh.empty()?1.0:sh.back().dot(yh.back())/yh.back().squaredNorm();
        Vector direction=std::clamp(scaling,1e-20,1e20)*q;
        for(std::size_t k=0;k<sh.size();++k) direction+=sh[k]*(alpha[k]-rho[k]*yh[k].dot(direction));
        direction=-direction;
        if(g.dot(b.project(x+direction)-x)>=0) {direction=-pg;sh.clear();yh.clear();rho.clear();}
        double step=1;bool accepted=false;Vector trial;double ft=infinity;
        for(int ls=0;ls<60;++ls) {
            trial=b.project(x+step*direction);Vector displacement=trial-x;
            if(displacement.lpNorm<Eigen::Infinity>()<1e-15) break;
            ft=c.evaluate(trial,true);
            if(std::isfinite(ft)&&ft<=f+1e-4*g.dot(displacement)) {accepted=true;break;}
            step*=0.5;
        }
        if(!accepted) {status=-2;break;}
        Vector gt=c.gradient(trial),s=trial-x,y=gt-g;double curvature=s.dot(y);
        if(curvature>1e-12*s.norm()*y.norm()&&curvature>0) {
            if(sh.size()==10) {sh.pop_front();yh.pop_front();rho.pop_front();}
            sh.push_back(s);yh.push_back(y);rho.push_back(1/curvature);
        }
        x=std::move(trial);g=std::move(gt);f=ft;
    }
    return {x,f,status};
}
}
LocalResult refine(Context& c,const Vector& x,const Bounds& b,LocalCandidate* retained) {
    if(c.options.local_method==LocalMethod::none) return {x,c.evaluate(x),0};
    auto begin=std::chrono::steady_clock::now();++c.counters.local_calls;
    LocalCandidate owned{b,{},infinity};LocalCandidate& candidate=retained?*retained:owned;
    candidate.x.resize(0);candidate.objective=infinity;CandidateScope scope(c,candidate);
    double before=infinity;
    auto record_improvement=[&] {if(candidate.objective<before)++c.counters.local_improvements;};
    LocalResult result;
    try {
        before=c.evaluate(x,true);
        if(c.options.local_method==LocalMethod::ipopt) result=refine_ipopt(c,x,b,before);
        else if(c.options.local_method==LocalMethod::lj_newton) result=refine_lj_newton(c,x,b,before);
        else result=projected_lbfgs(c,x,b,before);
        if(!b.contains(result.x)||!c.local_feasible(result.x)||!std::isfinite(result.objective)||
           (c.local_feasible(x)&&result.objective>before)) result={x,before,-2};
        // Preserve the backend status separately from the best admissible trial.
        // In particular, iteration_limit or line-search failure is not a veto.
        if(candidate.x.size()&&(!c.local_feasible(result.x)||!b.contains(result.x)||
           !std::isfinite(result.objective)||candidate.objective<result.objective)) {
            result.x=candidate.x;result.objective=candidate.objective;
        }
        record_improvement();
    } catch(const Stop&) {
        record_improvement(); // Context::observe already retained the incumbent.
        ++c.counters.local_limited;
        c.local_work_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        throw;
    } catch(...) {
        record_improvement();
        c.local_work_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        throw;
    }
    c.local_work_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    if(result.status==0||result.status==1) ++c.counters.local_converged;
    else if(result.status==-1||result.status==5) ++c.counters.local_limited;
    else ++c.counters.local_failed;
    // Backends return an evaluated point; never trust an unevaluated EnKF proposal.
    if(c.local_feasible(result.x)) c.observe(result.x,result.objective);
    return result;
}
} // namespace pf::detail
