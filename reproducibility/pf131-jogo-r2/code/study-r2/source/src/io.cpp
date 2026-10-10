#include "pf/pf.hpp"
#include "linear.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace pf {
namespace {
std::ofstream open_output(const std::string& path) {
    std::ofstream out(path,std::ios::out|std::ios::trunc);
    if(!out) throw std::runtime_error("Cannot write "+path);
    out.exceptions(std::ios::badbit|std::ios::failbit);out<<std::setprecision(17);return out;
}
void number(std::ostream& out,double x) {if(std::isfinite(x)) out<<x;else out<<"null";}
void quoted(std::ostream& out,const std::string& s) {
    out<<'"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\') out<<'\\'<<c;
        else if(c<32) {auto flags=out.flags();out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c);out.flags(flags);out<<std::setfill(' ');}
        else out<<c;
    }
    out<<'"';
}
template<class T> void array(std::ostream& out,const T& values) {
    out<<'[';for(std::size_t i=0;i<static_cast<std::size_t>(values.size());++i){if(i)out<<',';number(out,values[i]);}out<<']';
}
void commit(const std::string& temp,const std::string& path) {std::filesystem::rename(temp,path);}
}
void write_initialization_json(const std::string& path,const PythonInitialization& initial,bool external) {
    const auto& seeds=initial.seeds;
    auto out=open_output(path);
    out<<"{\n\"schema_version\":1,\"seed_protocol\":\"md5-first8hex-offset0-1000\",\n";
    out<<"\"generator\":\"NumPy-PCG64-SeedSequence\",\"numpy_algorithm_reference\":\"2.3.5\",\n";
    out<<"\"base_seed\":"<<seeds.base<<",\"atoms\":"<<seeds.atoms<<",\"run\":"<<seeds.run;
    out<<",\"seed_init\":"<<seeds.init<<",\"seed_pf\":"<<seeds.pf;
    out<<",\"x0_margin_fraction\":0.1,\"x0_used_by_pf\":false,\"x0\":";array(out,initial.x0);
    out<<",\n\"population_source\":\""<<(external?"external-pfpop":"python-pf-box")<<'"';
    out<<",\"population_file\":\"initial.pfpop\",\"particles\":"<<initial.population.size();
    out<<",\"dimension\":"<<initial.x0.size();
    out<<",\"pf_raw_draws_before_loop\":"<<std::uint64_t(initial.population.size())*std::uint64_t(initial.x0.size());
    out<<",\"periodic_kmeans_seed\":42,\"auxiliary_streams\":{\"seed\":"<<seeds.pf;
    out<<",\"islands_advance\":281474976710656,\"diversity_advance\":562949953421312}";
    out<<",\"full_python_trajectory_parity\":false\n}\n";
    out.close();
}
void save_population(const std::string& path,const Population& p) {
    if(p.empty()) throw std::invalid_argument("Cannot save empty population");
    auto out=open_output(path);out<<"PFPOP1 "<<p.size()<<' '<<p[0].size()<<'\n';
    for(const auto& x:p) {
        if(x.size()!=p[0].size()||!x.allFinite()) throw std::invalid_argument("Invalid population");
        for(Eigen::Index j=0;j<x.size();++j) out<<(j?" ":"")<<x[j];
        out<<'\n';
    }
    out.close();
}
Population load_population(const std::string& path,Eigen::Index dimension) {
    std::ifstream in(path);std::string magic;std::int64_t n=0,d=0;
    if(!(in>>magic>>n>>d)||magic!="PFPOP1"||n<2||n>100000||d!=dimension||d<1||n*d>100000000)
        throw std::invalid_argument("Invalid PFPOP1 header in "+path);
    Population p;p.reserve(static_cast<std::size_t>(n));
    for(std::int64_t i=0;i<n;++i) {Vector x(d);for(std::int64_t j=0;j<d;++j) if(!(in>>x[j])||!std::isfinite(x[j])) throw std::invalid_argument("Invalid PFPOP1 coordinate");p.push_back(std::move(x));}
    std::string extra;if(in>>extra) throw std::invalid_argument("Trailing data in population file");return p;
}
void write_result_json(const std::string& path,const Result& r,const Options& o,int atoms,int run,const std::string& algorithm) {
    const std::string temp=path+".tmp";auto out=open_output(temp);
    out<<"{\n\"schema_version\":1,\"version\":\""<<version<<"\",\"source_sha256\":\"6d21a05eb7b31d2a42f0041e40de72effc45d7db13bbad03ac1d5256113ee328\",\n";
    out<<"\"algorithm\":";quoted(out,algorithm);out<<",\"atoms\":"<<atoms<<",\"run\":"<<run;
    out<<",\"status\":";quoted(out,r.status);out<<",\"objective\":";number(out,r.objective);
    out<<",\"target_reached\":"<<(r.target_reached?"true":"false")<<",\"iterations\":"<<r.iterations;
    out<<",\"seconds\":"<<r.seconds<<",\"local_wall_seconds\":"<<r.local_seconds<<",\"local_work_seconds\":"<<r.local_work_seconds;
    out<<",\"projected_gradient_inf\":";number(out,r.projected_gradient_inf);
    out<<",\"lj_r2_epsilon\":1e-8,\"max_atom_radius\":"<<r.max_atom_radius<<",\"active_local_radius\":"<<r.active_local_radius;
    out<<",\"audit_objective_evaluations\":"<<r.audit_objective_evaluations<<",\"audit_gradient_evaluations\":"<<(r.x.size()?1:0);
    out<<",\"objective_evaluations\":"<<r.counters.objective_total()<<",\"objective_pf\":"<<r.counters.objective_pf;
    out<<",\"objective_local\":"<<r.counters.objective_local<<",\"gradient_evaluations\":"<<r.counters.gradient;
    out<<",\"hessian_evaluations\":"<<r.counters.hessian<<",\"invalid_objectives\":"<<r.counters.invalid_objective;
    out<<",\"local_calls\":"<<r.counters.local_calls<<",\"local_converged\":"<<r.counters.local_converged;
    out<<",\"local_failed\":"<<r.counters.local_failed<<",\"local_limited\":"<<r.counters.local_limited;
    out<<",\"local_improvements\":"<<r.counters.local_improvements;
    out<<",\"linear_analyses\":"<<r.counters.linear_analyses<<",\"linear_factorizations\":"<<r.counters.linear_factorizations;
    out<<",\"linear_solves\":"<<r.counters.linear_solves<<",\"linear_cg_fallbacks\":"<<r.counters.linear_cg_fallbacks;
    out<<",\"linear_metis_analyses\":"<<r.counters.linear_metis_analyses<<",\"mumps_user_ordering_analyses\":"<<r.counters.mumps_user_ordering_analyses;
    out<<",\"hessian_products\":"<<r.counters.hessian_products<<",\"newton_iterations\":"<<r.counters.newton_iterations;
    out<<",\"newton_accepted\":"<<r.counters.newton_accepted<<",\"newton_rejected\":"<<r.counters.newton_rejected;
    out<<",\"negative_curvature\":"<<r.counters.negative_curvature;
    out<<",\n\"options\":{\"seed\":"<<o.seed;
    out<<",\"random_generator\":\"NumPy-PCG64-SeedSequence\"";
    out<<",\"auxiliary_rng_policy\":\"seed_pf-advance-2^48-islands-2^49-diversity\"";
    if(o.run_seeds) {
        out<<",\"base_seed\":"<<o.run_seeds->base<<",\"seed_init\":"<<o.run_seeds->init;
        out<<",\"seed_pf\":"<<o.run_seeds->pf;
    }
    out<<",\"particles\":"<<o.particles<<",\"min_particles\":"<<o.min_particles;
    out<<",\"max_iterations\":"<<o.max_iterations<<",\"refine_top_k\":"<<o.refine_top_k<<",\"refine_all\":"<<(o.refine_all?"true":"false");
    out<<",\"final_refinement\":"<<(o.final_refinement?"true":"false")<<",\"radius\":"<<o.radius<<",\"batch_threads\":"<<o.batch_threads<<",\"local_workers\":"<<o.local_workers;
    out<<",\"local_method\":";quoted(out,o.local_method==LocalMethod::ipopt?"ipopt":o.local_method==LocalMethod::lj_newton?"lj-newton":o.local_method==LocalMethod::projected_lbfgs?"projected-lbfgs":"none");
    out<<",\"flow\":";quoted(out,o.flow==Flow::invertible?"invertible":o.flow==Flow::fast?"fast":"mutation");
    out<<",\"mapping\":";quoted(out,o.mapping==Mapping::linear?"linear":o.mapping==Mapping::residual?"residual":o.mapping==Mapping::affine?"affine":"cnf");
    out<<",\"clustering\":";quoted(out,o.clustering==Clustering::kmeans?"kmeans":o.clustering==Clustering::dbscan?"dbscan":"gmm");
    out<<",\"covariance_mutation\":"<<(o.covariance_mutation?"true":"false")<<",\"kicks\":"<<(o.kicks?"true":"false");
    out<<",\"islands\":"<<(o.islands?"true":"false")<<",\"diversity_injection\":"<<(o.diversity_injection?"true":"false");
    out<<",\"ipf_enabled\":"<<(o.ipf_enabled?"true":"false");
    out<<",\"ipf_schedule\":\""<<(o.ipf_endpoint_schedule?"linear-endpoints":"pf131-legacy")<<'"';
    out<<",\"ipf_alpha_initial\":"<<o.ipf_alpha_initial<<",\"ipf_alpha_final\":"<<o.ipf_alpha_final;
    out<<",\"adaptive_annealing\":"<<(o.adaptive_annealing?"true":"false");
    out<<",\"island_exchange\":"<<(o.island_exchange?"true":"false");
    out<<",\"ess_rejuvenation\":"<<(o.ess_rejuvenation?"true":"false");
    out<<",\"mutation_sigma\":"<<o.mutation_sigma<<",\"beta_initial\":"<<o.beta_initial<<",\"beta_max\":"<<o.beta_max;
    out<<",\"lambda_share\":"<<o.lambda_share<<",\"lambda_novelty\":"<<o.lambda_novelty;
    out<<",\"target\":";number(out,o.target);out<<",\"target_tolerance\":"<<o.target_tolerance;
    out<<",\"max_seconds\":"<<o.max_seconds<<",\"max_objective_evaluations\":"<<o.max_objective_evaluations;
    out<<",\"no_improve_stop\":"<<o.no_improve_stop<<",\"ipopt_ball_constraints\":"<<(o.ipopt_ball_constraints?"true":"false");
    out<<",\"gmm_max_iterations\":"<<o.gmm_max_iterations;
    out<<",\"local_max_iterations\":"<<o.local_max_iterations<<",\"local_tolerance\":"<<o.local_tolerance;
    out<<",\"newton_linear_solver\":";quoted(out,o.newton_linear_solver);
    out<<",\"newton_linear_ordering\":";quoted(out,o.newton_linear_ordering);
    out<<",\"hsl_library\":";quoted(out,o.hsl_library);
    out<<",\"newton_cg_max_iterations\":"<<o.newton_cg_max_iterations<<",\"newton_initial_radius\":"<<o.newton_initial_radius;
    out<<",\"local_execution\":\""<<(detail::newton_uses_threads(o)?"threads":"processes")<<'"';
    out<<",\"ipopt_linear_solver\":";quoted(out,o.ipopt_linear_solver);out<<",\"ipopt_exact_hessian\":"<<(o.ipopt_exact_hessian?"true":"false");
    out<<",\"ipopt_print_level\":"<<o.ipopt_print_level<<",\"kmeans_restarts\":"<<o.kmeans_restarts;
    out<<",\"cluster_max_iterations\":"<<o.cluster_max_iterations<<",\"dbscan_epsilon\":"<<o.dbscan_epsilon<<",\"dbscan_min_samples\":"<<o.dbscan_min_samples<<"},\n";
    out<<"\"build\":{\"ipopt\":"<<(ipopt_available()?"true":"false");
    out<<",\"arithmetic\":\"IEEE754-binary64\"";
    out<<",\"native_mumps\":"<<(mumps_available()?"true":"false");
    out<<",\"metis\":"<<(metis_available()?"true":"false")<<",\"metis_api\":"<<metis_api_version();
    out<<",\"ipopt_version\":";quoted(out,ipopt_version());
    out<<",\"eigen_version\":\""<<EIGEN_WORLD_VERSION<<'.'<<EIGEN_MAJOR_VERSION<<'.'<<EIGEN_MINOR_VERSION<<'"';
#ifdef _OPENMP
    out<<",\"openmp\":true";
#else
    out<<",\"openmp\":false";
#endif
    out<<",\"compiler\":";quoted(out,__VERSION__);out<<"},\n\"x\":";array(out,r.x);
    out<<",\n\"history\":[";
    for(std::size_t i=0;i<r.history.size();++i) {
        if(i)out<<',';
        const auto& h=r.history[i];out<<"{\"iteration\":"<<h.iteration<<",\"objectives\":";array(out,h.objectives);
        out<<",\"weights\":";array(out,h.weights);out<<",\"beta_used\":"<<h.beta_used<<",\"beta_next\":"<<h.beta_next;
        out<<",\"ipf_alpha\":"<<h.ipf_alpha;
        out<<",\"ess\":"<<h.ess<<",\"best\":";number(out,h.best);out<<",\"kick\":"<<(h.kick?"true":"false")<<'}';
    }
    out<<"]\n}\n";out.close();commit(temp,path);
}
void write_history_csv(const std::string& path,const Result& r) {
    auto out=open_output(path);
    out<<"iteration,best,current_best,particles,beta_used,beta_next,ess,diversity_before,diversity_after,sigma,radius,resampled,kick,island_exchange,injected,objective_evaluations,seconds,ipf_alpha\n";
    for(const auto& h:r.history) out<<h.iteration<<','<<h.best<<','<<h.current_best<<','<<h.particles<<','<<h.beta_used<<','<<h.beta_next<<','<<h.ess<<','<<h.diversity_before<<','<<h.diversity_after<<','<<h.mutation_sigma<<','<<h.radius<<','<<h.resampled<<','<<h.kick<<','<<h.island_exchange<<','<<h.injected<<','<<h.objective_evaluations<<','<<h.seconds<<','<<h.ipf_alpha<<'\n';
    out.close();
}
void write_xyz(const std::string& path,const Result& r,int atoms) {
    if(r.x.size()!=3*atoms) return;
    auto out=open_output(path);out<<atoms<<"\nPF C++ E="<<r.objective<<" status="<<r.status<<'\n';
    for(int i=0;i<atoms;++i) out<<"Ar "<<r.x[3*i]<<' '<<r.x[3*i+1]<<' '<<r.x[3*i+2]<<'\n';
    out.close();
}
} // namespace pf
