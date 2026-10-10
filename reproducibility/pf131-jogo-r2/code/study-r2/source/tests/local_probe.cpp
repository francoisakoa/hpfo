#include "../src/lj_newton.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) {
    try {
        if(argc<5)throw std::invalid_argument("usage: pf_local_probe atoms starts method workers [tol] [iterations]");
        int n=std::stoi(argv[1]),count=std::stoi(argv[2]);std::string method=argv[3];
        pf::LennardJones problem(n);pf::Options o;o.particles=count;o.min_particles=2;
        o.local_workers=std::stoi(argv[4]);o.local_method=method=="ipopt"?pf::LocalMethod::ipopt:pf::LocalMethod::lj_newton;
        o.local_tolerance=argc>5?std::stod(argv[5]):1e-6;o.local_max_iterations=argc>6?std::stoi(argv[6]):5000;
        pf::Bounds bounds{pf::Vector::Constant(3*n,-5),pf::Vector::Constant(3*n,5)};
        auto initial=pf::python_initialization(bounds,count,2024,n,1).population;
        auto p=initial;pf::detail::Context c(problem,o,bounds);c.require_local_feasibility=true;
        const auto start=std::chrono::steady_clock::now();pf::detail::LocalPool pool(c);
        std::vector<std::size_t> ids(count);for(int i=0;i<count;++i)ids[i]=i;
        pool.refine_selected(p,ids,bounds);pool.shutdown();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<std::setprecision(17)<<"{\"atoms\":"<<n<<",\"method\":\""<<method<<"\",\"workers\":"<<o.local_workers;
        std::cout<<",\"seconds\":"<<seconds<<",\"work_seconds\":"<<c.local_work_seconds;
        std::cout<<",\"evaluations\":"<<c.counters.objective_total()<<",\"hvp\":"<<c.counters.hessian_products<<",\"iterations\":"<<c.counters.newton_iterations;
        std::cout<<",\"converged\":"<<c.counters.local_converged<<",\"failed\":"<<c.counters.local_failed<<",\"limited\":"<<c.counters.local_limited<<",\"points\":[";
        for(int i=0;i<count;++i) {
            auto g=problem.gradient(p[i]);
            pf::Vector pg=p[i]-pf::detail::local_projection(p[i]-g,bounds,3,5);
            if(i)std::cout<<',';
            std::cout<<"{\"f\":"<<problem.value(p[i])<<",\"stationarity\":"<<pg.lpNorm<Eigen::Infinity>()<<",\"feasible\":"<<(c.local_feasible(p[i])?"true":"false")<<'}';
        }
        std::cout<<"]}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
