#include "pf/pf.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Allocate through real library APIs, then resize/destroy in this consumer TU.
// This exercises the ABI boundary that broke when only libpf used -march=native.
int main() {
    try {
        int checks=0;
        for(int dimension:{1,2,3,4,6,8,9,15,16,17,32,63,64,65,127,128,257}) {
            pf::Bounds bounds{pf::Vector::Constant(dimension,-2),pf::Vector::Constant(dimension,2)};
            pf::Vector input=pf::Vector::LinSpaced(dimension,-3,3);
            for(int repeat=0;repeat<16;++repeat) {
                auto output=bounds.project(input); // allocated in libpf
                for(int j=0;j<dimension;++j)
                    if(output[j]!=std::clamp(input[j],-2.0,2.0)) throw std::runtime_error("Projection mismatch");
                output.conservativeResize(dimension+1); // reallocates and frees in client
                output[dimension]=42;
                pf::Random rng(10+repeat);
                auto population=pf::uniform_population(bounds,7,rng); // library allocations
                for(auto& point:population) {
                    if(!bounds.contains(point)) throw std::runtime_error("Invalid generated point");
                    auto mapped=pf::map_particle(point,output.head(dimension),.5,pf::Mapping::linear);
                    if(!mapped.allFinite()) throw std::runtime_error("Invalid mapped point");
                    point=std::move(mapped); // releases old library-owned storage in client
                    point.conservativeResize(dimension+3);
                }
                population.clear(); // releases returned allocations in client
                ++checks;
            }
        }
        pf::LennardJones lj(3);
        pf::Vector x(9);x<<0,0,0,1.1,0,0,.4,.9,.1;
        for(int i=0;i<32;++i) {
            auto gradient=lj.gradient(x);auto hessian=lj.hessian(x);
            if(!gradient.allFinite()||!hessian.allFinite()) throw std::runtime_error("Nonfinite derivative");
            gradient.conservativeResize(19);hessian.conservativeResize(13,13);
            ++checks;
        }
        std::cout<<"PASS "<<checks<<" Eigen allocation/resize/destruction cycles; alignment="
                 <<EIGEN_DEFAULT_ALIGN_BYTES<<" malloc_already_aligned="<<EIGEN_MALLOC_ALREADY_ALIGNED<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
