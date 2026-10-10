#include "../src/internal.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc,char** argv) {
    try {
        const auto setup_start=std::chrono::steady_clock::now();
        int atoms=38,starts=8,run=1;std::uint64_t base=2024;
        std::string method="lj-newton",population,output;
        pf::Options o;o.local_method=pf::LocalMethod::lj_newton;
        auto integer=[](const std::string& value){std::size_t used;int x=std::stoi(value,&used);if(used!=value.size())throw std::invalid_argument("Invalid integer");return x;};
        auto real=[](const std::string& value){std::size_t used;double x=std::stod(value,&used);if(used!=value.size())throw std::invalid_argument("Invalid number");return x;};
        bool explicit_starts=false;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];
            auto value=[&](){if(i+1>=argc)throw std::invalid_argument("Missing value for "+key);return std::string(argv[++i]);};
            if(key=="--help") {
                std::cout<<"pf_local_bench --atoms N --starts K --method lj-newton|ipopt --workers W --output FILE\n"
                         <<"Options: --population FILE --seed N --run N --local-tol T --local-max-iter N\n"
                         <<"         --max-time SEC --radius R --no-ipopt-ball --ipopt-linear-solver NAME\n"
                         <<"         --newton-linear-solver cg|ldl|ma27|mumps --hsl-library FILE\n"
                         <<"         --newton-linear-ordering auto|metis (ldl or mumps only)\n"
                         <<"Every start is retained and audited; initialization and output are outside core timing.\n";return 0;
            } else if(key=="--atoms")atoms=integer(value());
            else if(key=="--starts"){starts=integer(value());explicit_starts=true;}
            else if(key=="--workers")o.local_workers=integer(value());
            else if(key=="--method")method=value();
            else if(key=="--local-tol")o.local_tolerance=real(value());
            else if(key=="--local-max-iter")o.local_max_iterations=integer(value());
            else if(key=="--max-time")o.max_seconds=real(value());
            else if(key=="--radius")o.radius=real(value());
            else if(key=="--run")run=integer(value());
            else if(key=="--seed"){auto s=value();std::size_t used;base=std::stoull(s,&used);if(s.empty()||s[0]=='-'||used!=s.size())throw std::invalid_argument("Invalid seed");}
            else if(key=="--population")population=value();
            else if(key=="--output")output=value();
            else if(key=="--no-ipopt-ball")o.ipopt_ball_constraints=false;
            else if(key=="--newton-linear-solver")o.newton_linear_solver=value();
            else if(key=="--newton-linear-ordering")o.newton_linear_ordering=value();
            else if(key=="--hsl-library")o.hsl_library=value();
            else if(key=="--ipopt-linear-solver")o.ipopt_linear_solver=value();
            else throw std::invalid_argument("Unknown option: "+key);
        }
        if(method=="ipopt")o.local_method=pf::LocalMethod::ipopt;
        else if(method!="lj-newton")throw std::invalid_argument("Unknown local method");
        if(output.empty()||std::filesystem::exists(output)||run<1)throw std::invalid_argument("Provide a new --output file and a positive run");
        const std::string initial_file=output+".initial.pfpop",final_file=output+".final.pfpop";
        if(std::filesystem::exists(initial_file)||std::filesystem::exists(final_file))throw std::invalid_argument("Benchmark companion files already exist");
        pf::LennardJones problem(atoms);
        pf::Bounds bounds{pf::Vector::Constant(3*atoms,-o.radius),pf::Vector::Constant(3*atoms,o.radius)};
        pf::Population initial;
        if(!population.empty()) {
            initial=pf::load_population(population,3*atoms);
            if(explicit_starts&&starts!=int(initial.size()))throw std::invalid_argument("--starts differs from population row count");
            starts=int(initial.size());
            for(const auto& x:initial)if(!bounds.contains(x))throw std::invalid_argument("Initial population outside bounds");
        }
        o.particles=starts;o.min_particles=2;o.validate(problem,bounds);
        if(initial.empty())initial=pf::python_initialization(bounds,starts,base,atoms,run).population;
        auto parent=std::filesystem::path(output).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
        pf::save_population(initial_file,initial);
        auto particles=initial;std::vector<std::size_t> ids(starts);std::iota(ids.begin(),ids.end(),0);
        const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_start).count();
        pf::detail::Context context(problem,o,bounds);context.require_local_feasibility=true;
        const auto begin=std::chrono::steady_clock::now();std::string status="complete";
        pf::detail::LocalPool pool(context);
        try {pool.refine_selected(particles,ids,bounds);}catch(const pf::detail::Stop& s){status=s.reason;}
        pool.shutdown();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        pf::save_population(final_file,particles);
        std::ofstream out(output+".tmp");if(!out)throw std::runtime_error("Cannot write benchmark JSON");
        out.exceptions(std::ios::badbit|std::ios::failbit);out<<std::setprecision(17);
        out<<"{\"version\":\""<<pf::version<<"\",\"atoms\":"<<atoms<<",\"starts\":"<<starts;
        out<<",\"method\":\""<<method<<"\",\"workers\":"<<o.local_workers<<",\"status\":\""<<status<<'"';
        out<<",\"setup_seconds\":"<<setup_seconds<<",\"core_seconds\":"<<seconds;
        out<<",\"local_work_seconds\":"<<context.local_work_seconds<<",\"local_tolerance\":"<<o.local_tolerance;
        out<<",\"arithmetic\":\"IEEE754-binary64\"";
        out<<",\"base_seed\":"<<base<<",\"run\":"<<run<<",\"radius\":"<<o.radius;
        out<<",\"external_population\":"<<(population.empty()?"false":"true");
        out<<",\"ball_constraints\":"<<(o.ipopt_ball_constraints?"true":"false");
        out<<",\"objective_evaluations\":"<<context.counters.objective_total();
        out<<",\"gradient_evaluations\":"<<context.counters.gradient<<",\"hessian_evaluations\":"<<context.counters.hessian;
        out<<",\"hessian_products\":"<<context.counters.hessian_products<<",\"newton_iterations\":"<<context.counters.newton_iterations;
        out<<",\"local_converged\":"<<context.counters.local_converged<<",\"local_failed\":"<<context.counters.local_failed;
        out<<",\"linear_analyses\":"<<context.counters.linear_analyses<<",\"linear_factorizations\":"<<context.counters.linear_factorizations;
        out<<",\"linear_solves\":"<<context.counters.linear_solves<<",\"linear_cg_fallbacks\":"<<context.counters.linear_cg_fallbacks;
        out<<",\"linear_metis_analyses\":"<<context.counters.linear_metis_analyses<<",\"mumps_user_ordering_analyses\":"<<context.counters.mumps_user_ordering_analyses;
        out<<",\"newton_linear_ordering\":\""<<o.newton_linear_ordering<<'"';
        out<<",\"newton_linear_solver\":\""<<o.newton_linear_solver<<"\",\"ipopt_linear_solver\":\""<<o.ipopt_linear_solver<<'"';
        out<<",\"local_limited\":"<<context.counters.local_limited;
        out<<",\"outside_audit_objective_evaluations\":"<<2*starts<<",\"outside_audit_gradient_evaluations\":"<<starts;
        out<<",\"points\":[";int passed=0;
        for(int i=0;i<starts;++i) {
            pf::Vector gradient=problem.gradient(particles[i]);
            pf::Vector residual=particles[i]-pf::detail::local_projection(particles[i]-gradient,bounds,o.ipopt_ball_constraints?3:0,o.radius);
            const double stationarity=residual.lpNorm<Eigen::Infinity>();
            const bool feasible=bounds.contains(particles[i])&&context.local_feasible(particles[i]);
            const bool pass=feasible&&stationarity<=o.local_tolerance;passed+=pass;
            if(i)out<<',';
            out<<"{\"index\":"<<i<<",\"initial_objective\":"<<problem.value(initial[i]);
            out<<",\"objective\":"<<problem.value(particles[i])<<",\"projected_stationarity_inf\":"<<stationarity;
            out<<",\"feasible\":"<<(feasible?"true":"false")<<",\"audit_pass\":"<<(pass?"true":"false")<<'}';
        }
        out<<"],\"audit_pass_count\":"<<passed<<"}\n";out.close();std::filesystem::rename(output+".tmp",output);
        std::cout<<"BENCH atoms="<<atoms<<" method="<<method<<" workers="<<o.local_workers<<" seconds="<<seconds<<" audit="<<passed<<'/'<<starts<<" output="<<output<<'\n';
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
