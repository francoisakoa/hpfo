#include "pf/pf.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
class TrackingProblem final:public pf::Problem {
public:
    mutable pf::Population calls;
    Eigen::Index dimension()const override{return 6;}
    double value(const pf::Vector& x)const override{calls.push_back(x);return x.squaredNorm();}
    pf::Vector gradient(const pf::Vector& x)const override{return 2*x;}
};
}
int main() {
    try {
        std::cout<<std::setprecision(17);
        std::string line;
        while(std::getline(std::cin,line)) {
            std::istringstream in(line);std::string mode;std::uint64_t seed;in>>mode>>seed;
            if(mode=="seed") {int n,m,offset;in>>n>>m>>offset;std::cout<<pf::derive_seed(seed,n,m,offset)<<'\n';continue;}
            pf::Random rng(seed);std::size_t count;
            if(mode=="flow") {
                TrackingProblem p;pf::Bounds b{pf::Vector::Constant(6,-5),pf::Vector::Constant(6,5)};
                pf::Options o;o.seed=seed;o.particles=12;o.min_particles=2;o.max_iterations=3;
                o.local_method=pf::LocalMethod::none;o.refine_top_k=0;o.final_refinement=false;
                o.clustering=pf::Clustering::dbscan;o.dbscan_epsilon=1000;
                o.islands=false;o.kicks=false;o.diversity_injection=false;
                pf::Solver solver(p,b,o);
                solver.solve({},[&](const pf::Iteration& h) {
                    std::cout<<h.particles;
                    for(auto it=p.calls.end()-h.particles;it!=p.calls.end();++it)
                        for(auto value:*it)std::cout<<' '<<value;
                    std::cout<<'\n';
                });
                continue;
            }
            if(mode=="advance") {
                in>>count;std::cout<<rng.index(10000)<<' ';
                rng.advance(count);std::cout<<rng.index(10000)<<' '<<rng.raw()<<'\n';continue;
            }
            if(mode=="choice") {
                std::size_t size;bool replacement;in>>size>>count>>replacement;
                for(auto x:rng.choice(size,count,replacement))std::cout<<x<<' ';
                std::cout<<'\n'<<rng.raw()<<'\n';continue;
            }
            if(mode=="integers") {
                std::size_t size;in>>size>>count;
                for(std::size_t i=0;i<count;++i)std::cout<<rng.index(size)<<' ';
                std::cout<<'\n'<<rng.raw()<<'\n';continue;
            }
            in>>count;
            if(mode=="legacy") rng=pf::Random::sklearn(static_cast<std::uint32_t>(seed));
            for(std::size_t i=0;i<count;++i) {
                if(mode=="raw")std::cout<<rng.raw();
                else if(mode=="uniform")std::cout<<rng.uniform();
                else if(mode=="normal")std::cout<<rng.normal();
                else if(mode=="legacy")std::cout<<rng.uniform()<<' '<<rng.index(170);
                else if(mode=="mixed")std::cout<<rng.index(10000)<<' '<<rng.normal()<<' '<<rng.uniform();
                else return 2;
                std::cout<<' ';
            }
            std::cout<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
