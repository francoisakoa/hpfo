#include "pf/pf.hpp"
#include "../src/internal.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
int checks=0;
#ifdef _OPENMP
constexpr int parallel_threads=2;
#else
constexpr int parallel_threads=1;
#endif
void check(bool condition,const std::string& message) {++checks;if(!condition)throw std::runtime_error(message);}
void near(double a,double b,double tolerance,const std::string& message) {check(std::abs(a-b)<=tolerance*std::max({1.0,std::abs(a),std::abs(b)}),message);}
pf::Bounds bounds(int n,double radius=5) {return {pf::Vector::Constant(n,-radius),pf::Vector::Constant(n,radius)};}
class Constant final:public pf::Problem {
public:
    Eigen::Index dimension()const override{return 6;}
    double value(const pf::Vector&)const override{return 0;}
    pf::Vector gradient(const pf::Vector&)const override{return pf::Vector::Zero(6);}
};
}
int main() {
    try {
        pf::LennardJones pair(2);pf::Vector x(6);x<<0,0,0,std::sqrt(1.0-1e-8),0,0;
        near(pair.value(x),-1,1e-15,"LJ pair minimum");check(pair.gradient(x).norm()<3e-14,"LJ pair gradient");
        pf::Vector collision=pf::Vector::Zero(6);check(std::isfinite(pair.value(collision)),"collision cap finite");
        pf::LennardJones problem(5);pf::Vector y(15);y<<0,0,0,1.12,0.1,0.02,0.3,1.14,0.2,-0.15,0.15,1.2,1.2,1.1,1.0;
        auto g=problem.gradient(y);auto h=problem.hessian(y);const double eps=1e-6;
        for(int i=0;i<y.size();++i) {
            auto plus=y,minus=y;plus[i]+=eps;minus[i]-=eps;
            near(g[i],(problem.value(plus)-problem.value(minus))/(2*eps),2e-7,"finite difference gradient");
            pf::Vector column=(problem.gradient(plus)-problem.gradient(minus))/(2*eps);
            check((column-h.col(i)).lpNorm<Eigen::Infinity>()<2e-5,"finite difference Hessian");
        }
        check((h-h.transpose()).norm()<1e-12,"Hessian symmetry");
        auto translated=y;for(int i=0;i<5;++i)translated.segment(3*i,3)+=pf::Vector::Constant(3,1.234);
        near(problem.value(y),problem.value(translated),1e-13,"translation invariance");
        pf::Vector reversed=y;for(int i=0;i<5;++i)reversed.segment(3*i,3)=y.segment(3*(4-i),3);
        near(problem.value(y),problem.value(reversed),1e-13,"permutation invariance");
        pf::Population p{pf::Vector::Constant(3,0),pf::Vector::Constant(3,1),pf::Vector::Constant(3,2)};
        auto w=pf::compute_weights({1e96,1e90,-5},5,p,{},1,0.01,0.01);
        double sum=0;for(double v:w.weights){check(std::isfinite(v)&&v>=0,"finite weight");sum+=v;}
        near(sum,1,1e-15,"weight normalization");check(w.ess>=1&&w.ess<=3.000001,"ESS bounds");
        auto bad=pf::compute_weights({pf::infinity,pf::infinity,pf::infinity},1,p,{},1,0,0);
        near(bad.ess,3,1e-15,"all invalid weights recovery");
        auto ids=pf::stratified_indices({0.2,0.3,0.5},{0,0.99999999999999,0.99999999999999});
        check(ids==std::vector<std::size_t>({0,2,2}),"stratified resampling boundary");
        check(pf::stratified_indices({0.25,0.25,0.25,0.25},{0,0,0,0})==std::vector<std::size_t>({0,0,1,2}),"source searchsorted left at exact boundary");
        near(pf::sharing_factors(p,1)[0],1,0,"sharing includes self");
        near(pf::next_beta(0,1,10,5,1e-4,1,1e-4,5),1e-4,1e-15,"single-iteration beta");
        pf::Options o;o.particles=8;o.min_particles=2;o.max_iterations=3;o.refine_top_k=2;o.final_refinement=false;
        o.local_max_iterations=150;o.kmeans_restarts=2;o.max_seconds=60;
        if(!pf::ipopt_available())o.local_method=pf::LocalMethod::projected_lbfgs;
        pf::LennardJones small(3);auto b=bounds(9);pf::Random rng(45);auto initial=pf::uniform_population(b,8,rng);
        pf::Solver serial(small,b,o);auto r1=serial.solve(initial),r2=serial.solve(initial);
        check(std::isfinite(r1.objective),"solver finite incumbent");near(r1.objective,small.value(r1.x),1e-13,"objective-coordinate consistency");
        check((r1.x-r2.x).norm()==0&&r1.objective==r2.objective,"repeated solve deterministic");
        check(r1.counters.objective_total()==r2.counters.objective_total(),"repeated counters deterministic");
        if(pf::ipopt_available()) near(r1.objective,-3,1e-6,"LJ3 known minimum");
        o.batch_threads=parallel_threads;o.local_workers=2;pf::Solver parallel(small,b,o);auto rp=parallel.solve(initial);
        near(rp.objective,r1.objective,1e-12,"serial/parallel objective");
        check((rp.x-r1.x).lpNorm<Eigen::Infinity>()<1e-9,"serial/parallel coordinates");
        check(rp.counters.objective_total()==r1.counters.objective_total(),"serial/parallel evaluations");
        o.max_objective_evaluations=17;auto limited=pf::Solver(small,b,o).solve(initial);
        check(limited.status=="evaluation_limit","evaluation limit status");check(limited.counters.objective_total()<=17,"hard evaluation budget");
        check(limited.x.size()==9&&std::isfinite(limited.objective),"budget returns incumbent");
        o.max_objective_evaluations=100;auto local_limit=pf::Solver(small,b,o).solve(initial);
        check(local_limit.counters.objective_total()<=100,"parallel local budget");
        o.max_objective_evaluations=0;o.max_seconds=1e-9;
        auto timeout=pf::Solver(small,b,o).solve(initial);check(timeout.status=="time_limit","time limit status");
        o.max_seconds=60;o.local_workers=1;o.target=-3;o.target_tolerance=1e-5;
        auto target=pf::Solver(small,b,o).solve(initial);
        if(pf::ipopt_available()) {check(target.target_reached,"target reached flag");check(target.iterations==int(target.history.size()),"actual iteration count");}
        pf::Options cluster_options;cluster_options.kmeans_restarts=2;cluster_options.batch_threads=parallel_threads;
        pf::Population groups;for(int i=0;i<10;++i)groups.push_back(pf::Vector::Constant(3,(i<5?-4:4)+0.01*i));
        for(auto method:{pf::Clustering::kmeans,pf::Clustering::dbscan,pf::Clustering::gmm}) {
            pf::Random crng(10);auto labels=pf::cluster(groups,method,crng,cluster_options,2);
            check(labels[0]==labels[4]&&labels[5]==labels[9]&&labels[0]!=labels[5],"cluster separation");
        }
        bool rejected=false;try {auto invalid=o;invalid.particles=1;pf::Solver fail(small,b,invalid);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"invalid options rejected");
        Constant constant;pf::Options branch;branch.local_method=pf::LocalMethod::none;
        branch.particles=50;branch.min_particles=2;branch.max_iterations=40;branch.batch_threads=parallel_threads;
        branch.kmeans_restarts=1;branch.refine_top_k=0;branch.final_refinement=false;
        pf::Population flat(50,pf::Vector::Zero(6));auto branches=pf::Solver(constant,bounds(6),branch).solve(flat);
        int kicks=0,islands=0,resampled=0,injected=0;
        for(const auto& row:branches.history){kicks+=row.kick;islands+=row.island_exchange;resampled+=row.resampled;injected+=row.injected;}
        check(kicks>=2,"stagnation kick branch");check(islands==3,"periodic island branch");
        check(resampled>=1,"low ESS resampling branch");check(injected>0,"diversity injection branch");
        branch.flow=pf::Flow::mutation;branch.covariance_mutation=true;branch.max_iterations=3;
        auto mutation=pf::Solver(constant,bounds(6),branch).solve(flat);check(mutation.iterations==3,"covariance mutation branch");
        branch.max_iterations=40;branch.no_improve_stop=12;
        auto stagnant=pf::Solver(constant,bounds(6),branch).solve(flat);
        check(stagnant.status=="stagnation_limit"&&stagnant.iterations==13,"global stagnation survives a kick");
        if(pf::ipopt_available()) {
            pf::Options local;local.particles=2;local.min_particles=2;local.local_max_iterations=300;
            auto wide=bounds(6);pf::Vector start(6);start<<.3,.3,.3,-.3,-.3,-.3;
            for(int workers:{1,2}) {
                local.local_workers=workers;local.radius=.35;
                pf::detail::Context context(pair,local,wide);pf::detail::LocalPool pool(context);
                for(double radius:{.35,.25}) {
                    context.active_ball_radius=radius;pf::Population candidates{start,-start};
                    pool.refine_selected(candidates,{0,1},wide);
                    double q=1/(4*radius*radius+1e-8),q3=q*q*q;
                    for(const auto& point:candidates) {
                        check(context.local_feasible(point),"local atom balls feasible, including worker radius update");
                        near(pair.value(point),q3*(q3-2),1e-7,"constrained pair analytic optimum despite lower infeasible start");
                    }
                }
                auto constrained=pf::Solver(pair,wide,local).multistart({start,-start});
                check(constrained.max_atom_radius<=local.radius+1e-9,"multistart excludes infeasible initial incumbent");
            }
        }
        std::cout<<"PASS "<<checks<<" checks; Ipopt="<<pf::ipopt_available()<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}
}
