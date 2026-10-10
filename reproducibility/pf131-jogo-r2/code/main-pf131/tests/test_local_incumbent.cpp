#include "../src/internal.hpp"
#include <cmath>
#include <iostream>

namespace {
int checks=0;
void check(bool value,const std::string& message) {++checks;if(!value)throw std::runtime_error(message);}
pf::Bounds box(int n,double r=5) {return {pf::Vector::Constant(n,-r),pf::Vector::Constant(n,r)};}

// Smooth, deliberately ill-conditioned objective with its exact derivative.
// Its feasible endpoint improves f but fails Armijo for all 60 trial steps.
class Oscillatory final:public pf::Problem {
public:
    Eigen::Index dimension()const override {return 1;}
    double value(const pf::Vector& x)const override {return x[0]+1e-3*std::sin(1e30*x[0]);}
    pf::Vector gradient(const pf::Vector& x)const override {return pf::Vector::Constant(1,1+1e27*std::cos(1e30*x[0]));}
};

void lj_case(pf::LocalMethod method,const std::string& backend,const std::string& ordering) {
    pf::LennardJones pair(2);auto b=box(6);
    pf::Vector x(6);x<<-.4,0,0,.4,0,0;
    const double initial=pair.value(x);
    pf::Options o;o.local_method=method;o.newton_linear_solver=backend;o.newton_linear_ordering=ordering;
    o.particles=2;o.min_particles=2;o.local_max_iterations=1;o.local_tolerance=1e-12;o.max_seconds=30;
    o.ipopt_linear_solver="mumps";o.final_refinement=false;
    pf::detail::Context direct(pair,o,b);
    const auto r=pf::detail::refine(direct,x,b);
    const std::string label=method==pf::LocalMethod::ipopt?"ipopt/mumps":("newton/"+backend+"/"+ordering);
    check(r.status!=0&&r.status!=1,label+": intentionally nonconverged");
    check(direct.counters.local_limited==1,label+": iteration limit recorded");
    check(b.contains(r.x)&&direct.local_feasible(r.x),label+": feasible returned point");
    check(r.objective<initial&&r.objective==pair.value(r.x),label+": improving evaluated point retained");
    check(direct.counters.local_improvements==1,label+": improvement counted independently");
    const auto per_call_budget=direct.counters.objective_total();
    for(int workers:{1,2}) {
        o.local_workers=workers;
        pf::detail::Context context(pair,o,b);pf::Population p{x,-x};std::vector<double> values(2,initial);
        {pf::detail::LocalPool pool(context);pool.refine_selected(p,{0,1},b,&values);}
        check(context.counters.local_converged==0&&context.counters.local_limited==2,label+": worker limits preserved");
        for(std::size_t i=0;i<p.size();++i) {
            check(b.contains(p[i])&&context.local_feasible(p[i]),label+": worker point feasible");
            check(values[i]<initial&&values[i]==pair.value(p[i]),label+": nonconverged worker improves particle");
        }
        check(context.best<initial&&context.best==pair.value(context.best_x),label+": coordinator retains best");

        o.particles=3;o.local_max_iterations=500;o.max_objective_evaluations=per_call_budget*workers;
        pf::detail::Context budget_context(pair,o,b);pf::Population partial{x,-x,x};std::vector<double> partial_values(3,initial);
        bool budget_stop=false;
        try {pf::detail::LocalPool pool(budget_context);pool.refine_selected(partial,{0,1,2},b,&partial_values);}
        catch(const pf::detail::Stop& s){budget_stop=s.reason=="evaluation_limit";}
        check(budget_stop,label+": interrupted pool propagates budget");
        for(int i=0;i<workers;++i) {
            check(partial_values[i]<initial&&partial_values[i]==pair.value(partial[i]),label+": interrupted particle and cache retain improvement");
            check(b.contains(partial[i])&&budget_context.local_feasible(partial[i]),label+": interrupted particle feasible");
        }

        // Both starts have an equal, strict share of the evaluation budget.
        // The extra unevaluated start forces multistart to report the exhausted budget.
        o.particles=3;o.local_max_iterations=500;o.max_objective_evaluations=3+per_call_budget*workers;
        const auto limited=pf::Solver(pair,b,o).multistart({x,-x,x});
        check(limited.status=="evaluation_limit",label+": budget stop propagated");
        check(limited.counters.objective_total()<=o.max_objective_evaluations,label+": strict budget");
        check(limited.counters.local_converged==0,label+": budget does not imply convergence");
        check(limited.objective<initial&&limited.objective==pair.value(limited.x),label+": budget preserves improvement");
        check(b.contains(limited.x)&&context.local_feasible(limited.x),label+": budget incumbent feasible");
        check(limited.counters.local_improvements>0,label+": interrupted improvement counted");
        o.particles=2;o.local_max_iterations=1;o.max_objective_evaluations=0;
    }
    std::cout<<"PASS "<<label<<" iteration/evaluation limits, serial/workers\n";
}
}

int main() {
    try {
        Oscillatory problem;auto b=box(1,1);pf::Options o;o.local_method=pf::LocalMethod::projected_lbfgs;
        pf::detail::Context c(problem,o,b);auto x=pf::Vector::Zero(1).eval();
        const auto failed=pf::detail::refine(c,x,b);
        check(failed.status==-2&&c.counters.local_failed==1,"line-search failure remains visible");
        check(failed.objective<problem.value(x)&&failed.objective==problem.value(failed.x),"failed line search returns improving trial");
        check(b.contains(failed.x)&&c.counters.local_improvements==1,"failed improvement feasible and counted");
        check(c.local_candidate==nullptr,"candidate scope restored after failure");
        o.max_objective_evaluations=2;
        pf::detail::Context interrupted(problem,o,b);bool stopped=false;
        try {pf::detail::refine(interrupted,x,b);}catch(const pf::detail::Stop& s){stopped=s.reason=="evaluation_limit";}
        check(stopped&&interrupted.local_candidate==nullptr,"candidate scope restored on budget exception");
        check(interrupted.best<problem.value(x)&&interrupted.counters.local_improvements==1,"interrupted line search retains improving trial");

        lj_case(pf::LocalMethod::lj_newton,"cg","auto");
        lj_case(pf::LocalMethod::lj_newton,"ldl","auto");
        if(pf::metis_available())lj_case(pf::LocalMethod::lj_newton,"ldl","metis");
        if(pf::mumps_available()) {
            lj_case(pf::LocalMethod::lj_newton,"mumps","auto");
            if(pf::metis_available())lj_case(pf::LocalMethod::lj_newton,"mumps","metis");
        }
        if(pf::ipopt_available())lj_case(pf::LocalMethod::ipopt,"cg","auto");
        std::cout<<"PASS "<<checks<<" local incumbent checks\n";
    }catch(const std::exception& e){std::cerr<<"FAIL after "<<checks<<": "<<e.what()<<'\n';return 1;}
}
