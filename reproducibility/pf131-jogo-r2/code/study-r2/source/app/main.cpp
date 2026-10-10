#include "pf/pf.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace {
int parse_int(const std::string& s) {std::size_t n=0;int x=std::stoi(s,&n);if(n!=s.size())throw std::invalid_argument("Invalid integer: "+s);return x;}
double parse_double(const std::string& s) {std::size_t n=0;double x=std::stod(s,&n);if(n!=s.size()||!std::isfinite(x))throw std::invalid_argument("Invalid number: "+s);return x;}
std::uint64_t parse_unsigned(const std::string& s) {if(s.empty()||s[0]=='-')throw std::invalid_argument("Invalid unsigned integer");std::size_t n=0;auto x=std::stoull(s,&n);if(n!=s.size())throw std::invalid_argument("Invalid unsigned integer");return x;}
void help() {
    std::cout<<R"(PF C++ 1.3.1 article-study-r1 — experimental controls, separate from archived PF131
Usage: pf_solver --atoms 38 [75 98 ...] [options]
  --runs N                 Independent runs (default 1)
  --run-start N            First run ID for MD5 seeding (default 1)
  --no-ipf                 Remove transport only; preserve clustering/order and jitter
  --ipf-alpha-start A      Actual initial alpha in [0,1]; enables endpoint schedule
  --ipf-alpha-end A        Actual final alpha in [0,1]; enables endpoint schedule
  --ipf-schedule S         legacy | linear (linear interpolates endpoints)
  --no-adaptive-annealing  Keep beta equal to --beta-initial throughout the run
  --no-island-exchange     Disable only every-10 exchange; retain every-20 injection
  --no-ess-rejuvenation    Disable only ESS-triggered uniform injection
  --study-schema          Print study interface identifier and exit
  --particles N            Default max(3*atoms+56,50)
  --max-iter N             PF iterations (1000)
  --no-improve-stop N      Stop after N stagnant iterations; 0 disables (400)
  --max-time SEC           Solve budget including local and final refinement (20000)
  --max-evals N            All objective evaluations, including local (0 unlimited)
  --seed N                 Base seed (2024); MD5 seeds at offsets 0/1000 saved in JSON
  --threads N              OpenMP batch/clustering threads (1)
  --workers N              Local concurrency: threads for CG/LDL, processes for MA27/MUMPS/Ipopt (1)
  --worker-executable PATH Absolute path if worker is not beside the executable
  --local-refine METHOD    ipopt | lj-newton | projected-lbfgs | none (ipopt)
  --local-max-iter N       Local iteration limit (5000)
  --local-tol VALUE        Local tolerance (1e-8)
  --ipopt-linear-solver S  mumps, ma27, ... if present in your Ipopt (mumps)
  --ipopt-print-level N    Ipopt output 0..12 (0)
  --no-ipopt-ball          Box-only local refinement (Ipopt or lj-newton)
  --newton-linear-solver S cg | ldl | ma27 | mumps (cg)
  --newton-linear-ordering O auto | metis (ldl or mumps; auto by default)
  --hsl-library PATH      HSL shared library for native MA27 and Ipopt
  --newton-cg-max-iter N   CG cap (0: problem dimension)
  --newton-radius R       Initial trust radius / sqrt(atoms) (.2)
  --ipopt-limited-memory   Use quasi-Newton instead of analytic Hessian
  --refine-top-k N         Locally refine top N particles (5)
  --refine-all             Locally refine all particles
  --no-final-refine        Disable final top-k refinement
  --flow METHOD            invertible | fast | mutation
  --mapping METHOD         linear | residual | affine | cnf
  --clustering METHOD      kmeans | dbscan | gmm
  --cma                    Empirical covariance for mutation mode
  --radius R               Initial box and local Ipopt atom radius (5)
  --min-particles N        Minimum population size (5)
  --mutation-sigma S       Initial mutation standard deviation (.05)
  --beta-initial B         Initial beta (.0001)
  --beta-max B             Maximum beta (5, as in supplied code)
  --lambda-share L         Sharing weight (.01)
  --lambda-novelty L       Novelty weight (.01)
  --kmeans-restarts N      Greedy K-means++ starts (1)
  --cluster-max-iter N     K-means iteration limit (300)
  --gmm-max-iter N         GMM iteration limit (100)
  --dbscan-eps E           DBSCAN radius (.5)
  --dbscan-min-samples N   DBSCAN minimum group density (2)
  --no-kicks --no-islands --no-diversity-injection   Ablations
  --target F --target-tol E   Known reference; default source LJ table / 1e-4
  --no-target              Disable target stopping
  --population FILE       Frozen PFPOP1 initial population (one atom count only)
  --generate-only         Save Python x0, seeds and PF population without solving
  --algorithm NAME        pf | ipopt-multistart | both (pf)
  --include-ipopt          Equivalent to --algorithm both
  --output-dir DIR        New directory only (default results_<timestamp>)
  --quiet                 Suppress per-iteration output
  --version --help
No CUDA backend; computation is native C++ double precision.
)";
}
}
int main(int argc,char** argv) {
    try {
        pf::Options options;std::vector<int> atoms{38};int runs=1,run_start=1;bool explicit_particles=false,explicit_target=false,no_target=false;
        bool generate_only=false,quiet=false;std::string population_file,output_dir,algorithm="pf";
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            auto value=[&]()->std::string {if(i+1>=argc)throw std::invalid_argument("Missing value for "+key);return argv[++i];};
            if(key=="--help"||key=="-h") {help();return 0;}
            else if(key=="--version") {
                std::cout<<pf::version<<" ipopt="<<pf::ipopt_available()<<" float64=1 native_mumps="<<pf::mumps_available()<<" metis="<<pf::metis_available()<<" metis_api="<<pf::metis_api_version();
#ifdef _OPENMP
                std::cout<<" openmp=1\n";
#else
                std::cout<<" openmp=0\n";
#endif
                return 0;
            }
            else if(key=="--atoms") {atoms.clear();while(i+1<argc&&std::string(argv[i+1]).rfind("--",0)!=0)atoms.push_back(parse_int(argv[++i]));if(atoms.empty())throw std::invalid_argument("--atoms requires a number");}
            else if(key=="--study-schema") {std::cout<<"PF131_ARTICLE_STUDY_R1\n";return 0;}
            else if(key=="--run-start") run_start=parse_int(value());
            else if(key=="--no-ipf") options.ipf_enabled=false;
            else if(key=="--ipf-alpha-start") {options.ipf_alpha_initial=parse_double(value());options.ipf_endpoint_schedule=true;}
            else if(key=="--ipf-alpha-end") {options.ipf_alpha_final=parse_double(value());options.ipf_endpoint_schedule=true;}
            else if(key=="--ipf-schedule") {auto v=value();if(v!="legacy"&&v!="linear")throw std::invalid_argument("IPF schedule: legacy | linear");options.ipf_endpoint_schedule=v=="linear";}
            else if(key=="--no-adaptive-annealing") options.adaptive_annealing=false;
            else if(key=="--no-island-exchange") options.island_exchange=false;
            else if(key=="--no-ess-rejuvenation") options.ess_rejuvenation=false;
            else if(key=="--runs") runs=parse_int(value());
            else if(key=="--particles") {options.particles=parse_int(value());explicit_particles=true;}
            else if(key=="--max-iter") options.max_iterations=parse_int(value());
            else if(key=="--no-improve-stop") options.no_improve_stop=parse_int(value());
            else if(key=="--max-time") options.max_seconds=parse_double(value());
            else if(key=="--max-evals") options.max_objective_evaluations=parse_unsigned(value());
            else if(key=="--seed") options.seed=parse_unsigned(value());
            else if(key=="--threads") options.batch_threads=parse_int(value());
            else if(key=="--workers") options.local_workers=parse_int(value());
            else if(key=="--worker-executable") options.worker_executable=value();
            else if(key=="--local-max-iter") options.local_max_iterations=parse_int(value());
            else if(key=="--newton-cg-max-iter") options.newton_cg_max_iterations=parse_int(value());
            else if(key=="--newton-radius") options.newton_initial_radius=parse_double(value());
            else if(key=="--local-tol") options.local_tolerance=parse_double(value());
            else if(key=="--newton-linear-solver")options.newton_linear_solver=value();
            else if(key=="--newton-linear-ordering")options.newton_linear_ordering=value();
            else if(key=="--hsl-library")options.hsl_library=value();
            else if(key=="--ipopt-linear-solver") options.ipopt_linear_solver=value();
            else if(key=="--ipopt-print-level") options.ipopt_print_level=parse_int(value());
            else if(key=="--no-ipopt-ball"||key=="--no-local-ball") options.ipopt_ball_constraints=false;
            else if(key=="--ipopt-limited-memory") options.ipopt_exact_hessian=false;
            else if(key=="--refine-top-k") options.refine_top_k=parse_int(value());
            else if(key=="--refine-all"||key=="--ipopt-refine-all") options.refine_all=true;
            else if(key=="--no-final-refine") options.final_refinement=false;
            else if(key=="--min-particles") options.min_particles=parse_int(value());
            else if(key=="--radius") options.radius=parse_double(value());
            else if(key=="--mutation-sigma") options.mutation_sigma=parse_double(value());
            else if(key=="--beta-initial") options.beta_initial=parse_double(value());
            else if(key=="--beta-max") options.beta_max=parse_double(value());
            else if(key=="--lambda-share") options.lambda_share=parse_double(value());
            else if(key=="--lambda-novelty") options.lambda_novelty=parse_double(value());
            else if(key=="--kmeans-restarts") options.kmeans_restarts=parse_int(value());
            else if(key=="--cluster-max-iter") options.cluster_max_iterations=parse_int(value());
            else if(key=="--gmm-max-iter") options.gmm_max_iterations=parse_int(value());
            else if(key=="--dbscan-eps") options.dbscan_epsilon=parse_double(value());
            else if(key=="--dbscan-min-samples") options.dbscan_min_samples=parse_int(value());
            else if(key=="--cma") options.covariance_mutation=true;
            else if(key=="--no-kicks") options.kicks=false;
            else if(key=="--no-islands") options.islands=false;
            else if(key=="--no-diversity-injection") options.diversity_injection=false;
            else if(key=="--target") {options.target=parse_double(value());explicit_target=true;}
            else if(key=="--target-tol") options.target_tolerance=parse_double(value());
            else if(key=="--no-target") no_target=true;
            else if(key=="--population") population_file=value();
            else if(key=="--output-dir") output_dir=value();
            else if(key=="--generate-only") generate_only=true;
            else if(key=="--quiet") quiet=true;
            else if(key=="--algorithm") algorithm=value();
            else if(key=="--include-ipopt") algorithm="both";
            else if(key=="--local-refine") {auto v=value();if(v=="ipopt")options.local_method=pf::LocalMethod::ipopt;else if(v=="lj-newton")options.local_method=pf::LocalMethod::lj_newton;else if(v=="projected-lbfgs")options.local_method=pf::LocalMethod::projected_lbfgs;else if(v=="none")options.local_method=pf::LocalMethod::none;else throw std::invalid_argument("Unknown local method: "+v);}
            else if(key=="--flow") {auto v=value();if(v=="invertible")options.flow=pf::Flow::invertible;else if(v=="fast")options.flow=pf::Flow::fast;else if(v=="mutation")options.flow=pf::Flow::mutation;else throw std::invalid_argument("Unknown flow: "+v);}
            else if(key=="--mapping") {auto v=value();if(v=="linear")options.mapping=pf::Mapping::linear;else if(v=="residual")options.mapping=pf::Mapping::residual;else if(v=="affine")options.mapping=pf::Mapping::affine;else if(v=="cnf")options.mapping=pf::Mapping::cnf;else throw std::invalid_argument("Unknown mapping: "+v);}
            else if(key=="--clustering") {auto v=value();if(v=="kmeans")options.clustering=pf::Clustering::kmeans;else if(v=="dbscan")options.clustering=pf::Clustering::dbscan;else if(v=="gmm")options.clustering=pf::Clustering::gmm;else throw std::invalid_argument("Unknown clustering: "+v);}
            else throw std::invalid_argument("Unknown option: "+key);
        }
        if(runs<1||run_start<1||std::int64_t(run_start)+runs-1>100000) throw std::invalid_argument("Invalid run range");
        if(algorithm!="pf"&&algorithm!="both"&&algorithm!="ipopt-multistart") throw std::invalid_argument("Unknown algorithm");
        if(!population_file.empty()&&atoms.size()!=1) throw std::invalid_argument("A frozen population requires exactly one atom count");
        const std::map<int,double> known{{38,-173.928427},{75,-397.492331},{76,-402.894866},{77,-409.083517},{98,-543.665361},{102,-569.363653},{103,-575.766131},{104,-582.086642}};
        if(output_dir.empty()) output_dir="results_"+std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        if(std::filesystem::exists(output_dir)) throw std::invalid_argument("Output directory already exists: "+output_dir);
        std::filesystem::create_directories(output_dir);
        const auto base_seed=options.seed;
        for(int n:atoms) for(int run=run_start;run<run_start+runs;++run) {
            pf::LennardJones problem(n);auto o=options;
            if(!explicit_particles)o.particles=std::max(3*n+56,50);
            o.seed=pf::derive_seed(base_seed,n,run,1000);
            if(no_target)o.target=-pf::infinity;else if(!explicit_target&&known.count(n))o.target=known.at(n);
            pf::Bounds bounds{pf::Vector::Constant(problem.dimension(),-o.radius),pf::Vector::Constant(problem.dimension(),o.radius)};
            pf::Population initial;
            if(!population_file.empty()) {
                initial=pf::load_population(population_file,problem.dimension());
                if(!explicit_particles)o.particles=static_cast<int>(initial.size());
                if(initial.size()!=static_cast<std::size_t>(o.particles))throw std::invalid_argument("Initial population row count must equal --particles");
                for(const auto& x:initial)if(!bounds.contains(x))throw std::invalid_argument("Initial population outside bounds");
            }
            o.validate(problem,bounds);
            auto initialization=pf::python_initialization(bounds,o.particles,base_seed,n,run);
            o.run_seeds=initialization.seeds;
            if(population_file.empty())initial=initialization.population;
            auto folder=std::filesystem::path(output_dir)/("LJ"+std::to_string(n)+"_run"+std::to_string(run));
            std::filesystem::create_directory(folder);pf::save_population((folder/"initial.pfpop").string(),initial);
            pf::write_initialization_json((folder/"initialization.json").string(),initialization,!population_file.empty());
            if(generate_only) {std::cout<<"INITIAL_POPULATION="<<(folder/"initial.pfpop")<<'\n';continue;}
            auto save=[&](const pf::Result& r,const pf::Options& actual,const std::string& name) {
                pf::write_result_json((folder/(name+".json")).string(),r,actual,n,run,name);
                pf::write_history_csv((folder/(name+"_iterations.csv")).string(),r);
                pf::write_xyz((folder/(name+".xyz")).string(),r,n);
                std::cout<<"RESULT algorithm="<<name<<" atoms="<<n<<" run="<<run<<" status="<<r.status<<" f="<<std::setprecision(14)<<r.objective<<" iterations="<<r.iterations<<" evals="<<r.counters.objective_total()<<" seconds="<<r.seconds<<" workers="<<actual.local_workers<<std::endl;
            };
            if(algorithm=="pf"||algorithm=="both") {
                pf::Solver solver(problem,bounds,o);
                auto r=solver.solve(initial,[&](const pf::Iteration& h) {if(!quiet)std::cout<<"PF iter="<<h.iteration<<" f="<<std::setprecision(11)<<h.best<<" particles="<<h.particles<<" beta="<<h.beta_used<<" ESS="<<h.ess<<" kick="<<h.kick<<" evals="<<h.objective_evaluations<<" time="<<h.seconds<<std::endl;});
                save(r,o,"pf");
            }
            if(algorithm=="ipopt-multistart"||algorithm=="both") {
                o.local_method=pf::LocalMethod::ipopt;pf::Solver baseline(problem,bounds,o);save(baseline.multistart(initial),o,"ipopt_multistart");
            }
        }
        std::cout<<"OUTPUT="<<std::filesystem::absolute(output_dir)<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
