#include "internal.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace pf {
namespace {
Population choose(const Population& p,std::size_t count,Random& rng,bool replacement=false) {
    if(!replacement&&count>=p.size()) return p;
    Population result;result.reserve(count);
    for(auto i:rng.choice(p.size(),count,replacement)) result.push_back(p[i]);
    return result;
}
Population sample_exact(const Population& p,std::size_t count,Random& rng) {
    Population result;result.reserve(count);
    for(auto i:rng.choice(p.size(),count,p.size()<count)) result.push_back(p[i]);
    return result;
}
void jitter(Population& p,const Bounds& b,Random& rng,double scale) {
    for(auto& x:p) {
        for(Eigen::Index j=0;j<x.size();++j) x[j]+=scale*(b.upper[j]-b.lower[j])*rng.normal();
        x=b.project(x);
    }
}
std::vector<std::size_t> best_indices(const std::vector<double>& f,int count) {
    std::vector<std::size_t> order(f.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](std::size_t a,std::size_t b){return f[a]<f[b];});
    order.resize(std::min(order.size(),static_cast<std::size_t>(count)));return order;
}
Population flow_population(detail::Context& c,const Population& p,const Bounds& b,Random& rng,
                           const Options& o,int iteration,Population& archive) {
    if(o.flow==Flow::mutation) {
        double sigma=o.mutation_sigma*std::max(0.1,1.0-double(iteration)/o.max_iterations);
        Matrix lower;
        if(o.covariance_mutation) {
            Vector mean=Vector::Zero(p[0].size());for(const auto& x:p) mean+=x;mean/=double(p.size());
            Matrix covariance=Matrix::Zero(mean.size(),mean.size());
            for(const auto& x:p) {Vector v=x-mean;covariance.noalias()+=v*v.transpose();}
            covariance/=double(p.size()-1);covariance.diagonal().array()+=1e-8;
            Eigen::LLT<Matrix> cholesky(covariance);
            if(cholesky.info()!=Eigen::Success) throw std::runtime_error("Mutation covariance factorization failed");
            lower=cholesky.matrixL();
        }
        Population out;out.reserve(p.size());
        for(const auto& x:p) {
            double r=rng.uniform();Vector y=x;
            if(r<0.4&&p.size()>=3) {
                const auto ids=rng.choice(p.size(),2);const auto i=ids[0],j=ids[1];
                y+=0.8*(p[i]-p[j]);
            } else if(r<0.7) {
                // Source solver.best_particle is never assigned; directed term is zero.
            } else if(r<0.85) {
                Vector random=uniform_population(b,1,rng)[0];double alpha=rng.uniform();y=alpha*x+(1-alpha)*random;
            } else if(r<0.95) {
                Vector partner=archive.empty()?uniform_population(b,1,rng)[0]:archive[rng.index(archive.size())];y=0.5*(x+partner);
            }
            Vector noise(x.size());for(Eigen::Index j=0;j<x.size();++j) noise[j]=rng.normal();
            if(o.covariance_mutation) y+=sigma*(lower*noise);else y+=sigma*noise;
            out.push_back(b.project(y));
        }
        return out;
    }
    // Python consumes one Generator.integers draw and gives sklearn a separate RNG.
    auto cluster_rng=Random::sklearn(static_cast<std::uint32_t>(rng.index(10000)));
    auto labels=cluster(p,o.clustering,cluster_rng,o);auto f=c.batch(p);
    std::map<int,std::vector<std::size_t>> groups;
    for(std::size_t i=0;i<p.size();++i) groups[labels[i]].push_back(i);
    Population out;out.reserve(p.size());
    const double alpha=ipf_step(o,iteration);
    for(const auto& group:groups) {
        const auto& ids=group.second;auto best=ids[0];for(auto i:ids) if(f[i]<f[best]) best=i;
        for(auto i:ids) out.push_back(o.ipf_enabled?
            b.project(map_particle(p[i],p[best],alpha,o.flow==Flow::fast?Mapping::linear:o.mapping)):p[i]);
    }
    jitter(out,b,rng,o.flow==Flow::fast?0.04:0.02);return out;
}
void check_initial(const Population& p,const Bounds& b,int size) {
    if(static_cast<int>(p.size())!=size) throw std::invalid_argument("Initial population row count must equal --particles");
    for(const auto& x:p) if(!b.contains(x)) throw std::invalid_argument("Initial population is nonfinite, outside bounds, or incorrectly sized");
}
void finalize(Result& result,detail::Context& c,const Bounds& b) {
    result.x=c.best_x;result.objective=c.best;result.counters=c.counters;
    result.local_seconds=c.local_seconds;result.local_work_seconds=c.local_work_seconds;
    result.active_local_radius=c.active_ball_radius;
    if(result.x.size()) {
        // Independent reporting audit, deliberately separate from optimization budgets.
        double audited=c.problem.value(result.x);result.audit_objective_evaluations=1;
        if(!b.contains(result.x)||!std::isfinite(audited)||std::abs(audited-result.objective)>1e-10*std::max(1.0,std::abs(audited)))
            throw std::runtime_error("Final incumbent audit failed");
        result.objective=audited;
        result.projected_gradient_inf=detail::projected_gradient(result.x,c.problem.gradient(result.x),b).lpNorm<Eigen::Infinity>();
        if(const auto* lj=dynamic_cast<const LennardJones*>(&c.problem))
            for(int i=0;i<lj->atoms();++i) result.max_atom_radius=std::max(result.max_atom_radius,result.x.segment(i*lj->spatial_dimension(),lj->spatial_dimension()).norm());
    }
    result.target_reached=std::isfinite(c.options.target)&&result.objective<=c.options.target+c.options.target_tolerance;
    if(result.target_reached && (result.status=="iteration_limit"||result.status=="stagnation_limit")) result.status="target_reached";
    result.seconds=c.seconds();
}
}
Solver::Solver(const Problem& p,Bounds b,Options o):problem_(p),bounds_(std::move(b)),options_(std::move(o)) {options_.validate(problem_,bounds_);}
Result Solver::solve(const Population& initial,const std::function<void(const Iteration&)>& observer) const {
    detail::Context c(problem_,options_,bounds_);Result result;result.status="iteration_limit";
    Random rng(options_.seed);
    // Source island_exchange/tune_parameters create unseeded independent RNGs.
    // Their historical entropy cannot be recovered. Use explicit deterministic
    // auxiliary subsequences without consuming any draws from the PF stream.
    Random island_rng(options_.seed),injection_rng(options_.seed);
    island_rng.advance(std::uint64_t(1)<<48);
    injection_rng.advance(std::uint64_t(1)<<49);
    Population p;
    if(initial.empty()) p=uniform_population(bounds_,options_.particles,rng);
    else {
        p=initial;
        rng.advance(std::uint64_t(options_.particles)*std::uint64_t(problem_.dimension()));
    }
    check_initial(p,bounds_,options_.particles);
    Bounds bounds=bounds_;Population archive;
    const double sigma_share=0.1*(bounds_.upper-bounds_.lower).norm();
    const double diversity_threshold=0.05*(bounds_.upper-bounds_.lower).norm();
    double beta=options_.beta_initial,beta_factor=1,radius=options_.radius;
    double best_for_stagnation=infinity,previous=infinity,kick_factor=10,sigma=options_.mutation_sigma;
    int stagnant=0,global_stagnant=0;
    double best_ever=infinity;
    detail::LocalPool pool(c);
    try {
        // Evaluate initial inputs so a tiny budget can still return a valid incumbent.
        // This also preserves a good supplied point which the source could discard before evaluation.
        c.batch(p);
        for(int t=0;t<options_.max_iterations;++t) {
            c.check(true);Iteration record;record.iteration=t+1;record.beta_used=beta;
            int current=std::max(options_.min_particles,int(options_.particles*(1-double(t)/options_.max_iterations)));
            p=choose(p,current,rng);
            if(options_.islands && options_.island_exchange && t>0 && t%10==0) {
                auto f=c.batch(p);Population merged=p;
                for(std::size_t island=0;island<3;++island) {
                    std::size_t best=p.size();
                    for(std::size_t i=island;i<p.size();i+=3) if(best==p.size()||f[i]<f[best]) best=i;
                    if(best!=p.size()) merged.push_back(p[best]);
                }
                p=sample_exact(merged,options_.particles,island_rng);record.island_exchange=true;
            }
            if(options_.islands && t>0 && t%20==0) {
                auto fixed_rng=Random::sklearn(42);auto labels=cluster(p,Clustering::kmeans,fixed_rng,options_,3);auto f=c.batch(p);
                Population reps;
                for(int k=0;k<3;++k) {
                    std::size_t best=p.size();
                    for(std::size_t i=0;i<p.size();++i) if(labels[i]==k&&(best==p.size()||f[i]<f[best])) best=i;
                    if(best!=p.size()) reps.push_back(p[best]);
                }
                p.insert(p.end(),reps.begin(),reps.end());
            }
            record.ipf_alpha=ipf_step(options_,t);
            p=flow_population(c,p,bounds,rng,options_,t,archive);
            auto pre=c.batch(p);
            auto selected=best_indices(pre,options_.refine_all?int(p.size()):options_.refine_top_k);
            c.active_ball_radius=radius;
            pool.refine_selected(p,selected,bounds,&pre);
            auto f=std::move(pre);
            auto best=static_cast<std::size_t>(std::min_element(f.begin(),f.end())-f.begin());
            if(f[best]<best_for_stagnation) {
                best_for_stagnation=f[best];archive.push_back(p[best]);if(archive.size()>10) archive.erase(archive.begin());
            }
            auto weight=compute_weights(f,beta,p,archive,sigma_share,options_.lambda_share,options_.lambda_novelty,options_.batch_threads);
            record.particles=static_cast<int>(p.size());record.current_best=f[best];record.best=c.best;
            record.objectives=f;record.weights=weight.weights;record.ess=weight.ess;
            record.diversity_before=diversity(p,options_.batch_threads);
            bool target=std::isfinite(options_.target)&&c.best<=options_.target+options_.target_tolerance;
            bool stagnation_stop=false;
            if(!target) {
                if(weight.ess<0.1*options_.particles) {
                    std::vector<double> draws(p.size());for(auto& u:draws) u=rng.uniform();
                    auto ids=stratified_indices(weight.weights,draws);Population resampled;resampled.reserve(p.size());
                    for(auto i:ids) resampled.push_back(p[i]);
                    jitter(resampled,bounds,rng,0.01);p=std::move(resampled);record.resampled=true;
                    if(options_.ess_rejuvenation && weight.ess<0.05*options_.particles) {
                        int count=int(0.1*options_.particles);
                        if(count) {auto extra=uniform_population(bounds,count,rng);p.insert(p.end(),extra.begin(),extra.end());record.injected+=count;}
                        if(p.size()!=static_cast<std::size_t>(options_.particles)) p=sample_exact(p,options_.particles,rng);
                    }
                }
                if(options_.adaptive_annealing) beta=next_beta(t,options_.max_iterations,options_.particles,weight.ess,beta,beta_factor,options_.beta_initial,options_.beta_max);
                if(std::abs(best_for_stagnation-previous)<1e-6) {++stagnant;kick_factor+=0.5;}
                else {stagnant=0;kick_factor=10.5;}
                if(best_for_stagnation<best_ever-1e-6) {best_ever=best_for_stagnation;global_stagnant=0;}
                else ++global_stagnant;
                stagnation_stop=options_.no_improve_stop>0&&global_stagnant>=options_.no_improve_stop;
                if(!stagnation_stop) {
                double d=record.resampled?diversity(p,options_.batch_threads):record.diversity_before;
                if(stagnant>=10) sigma=std::min(1.2*sigma,1.0);
                else sigma=options_.mutation_sigma*std::max(0.1,1-double(t)/options_.max_iterations);
                if(options_.adaptive_annealing) beta_factor=weight.ess<0.05*options_.particles?std::min(beta_factor*1.1,5.0):std::max(beta_factor*0.95,1.0);
                if(options_.diversity_injection&&d<diversity_threshold) {
                    int count=int(0.2*options_.particles);
                    if(count) {auto extra=uniform_population(bounds,count,injection_rng);p.insert(p.end(),extra.begin(),extra.end());record.injected+=count;}
                }
                previous=best_for_stagnation;
                if(options_.kicks&&stagnant>=10) {
                    kick_factor=std::min(100.0,kick_factor+0.5*stagnant+0.01*problem_.dimension());
                    radius=std::max(0.98*radius,3.0);
                    // Preserve the user's feasible box if it is narrower/asymmetric.
                    bounds.lower=bounds_.lower.cwiseMax(Vector::Constant(problem_.dimension(),-radius));
                    bounds.upper=bounds_.upper.cwiseMin(Vector::Constant(problem_.dimension(),radius));
                    if((bounds.lower.array()>=bounds.upper.array()).any()) bounds=bounds_;
                    for(auto& x:p) {for(Eigen::Index j=0;j<x.size();++j) x[j]+=(4*rng.uniform()-2)*kick_factor*0.38;x=bounds.project(x);}
                    stagnant=0;record.kick=true;
                }
                }
            }
            record.beta_next=beta;record.mutation_sigma=sigma;record.radius=radius;
            record.diversity_after=(record.resampled||record.injected||record.kick)?diversity(p,options_.batch_threads):record.diversity_before;
            record.objective_evaluations=c.counters.objective_total();record.seconds=c.seconds();
            result.history.push_back(record);result.iterations=t+1;if(observer) observer(result.history.back());
            if(target) {result.status="target_reached";break;}
            if(stagnation_stop) {result.status="stagnation_limit";break;}
        }
        c.active_ball_radius=radius;
        if(options_.final_refinement&&options_.local_method!=LocalMethod::none&&result.status!="target_reached"&&options_.refine_top_k>0) {
            auto f=c.batch(p);pool.refine_selected(p,best_indices(f,options_.refine_top_k),bounds);
        }
    } catch(const detail::Stop& stop) {result.status=stop.reason;}
    if(c.best_x.size()==0 && result.status=="iteration_limit") result.status="no_finite_candidate";
    pool.shutdown();finalize(result,c,bounds_);return result;
}
Result Solver::multistart(const Population& initial) const {
    check_initial(initial,bounds_,options_.particles);
    detail::Context c(problem_,options_,bounds_);Result r;r.status="multistart_complete";
    c.require_local_feasibility=true;
    Population p=initial;detail::LocalPool pool(c);
    try {
        c.batch(p);std::vector<std::size_t> ids(p.size());std::iota(ids.begin(),ids.end(),0);
        pool.refine_selected(p,ids,bounds_);
    } catch(const detail::Stop& stop) {r.status=stop.reason;}
    if(c.best_x.size()==0&&r.status=="multistart_complete") r.status="no_feasible_candidate";
    pool.shutdown();finalize(r,c,bounds_);return r;
}
} // namespace pf
