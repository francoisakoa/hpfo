#include "pf/pf.hpp"
#include <iomanip>
#include <iostream>
template<class T> void row(const T& v) {for(Eigen::Index i=0;i<static_cast<Eigen::Index>(v.size());++i)std::cout<<(i?" ":"")<<v[i];std::cout<<'\n';}
int main(int argc,char**) {
    std::cout<<std::setprecision(17);
    if(argc>1) {
        int n,d;if(!(std::cin>>n>>d)) return 2;
        pf::LennardJones model(n,d);pf::Vector point(model.dimension());
        for(Eigen::Index i=0;i<point.size();++i) if(!(std::cin>>point[i])) return 2;
        std::cout<<model.value(point)<<'\n';row(model.gradient(point));
        const auto h=model.hessian(point);
        for(Eigen::Index i=0;i<h.rows();++i) row(h.row(i));
        return 0;
    }
    pf::Vector x(6),b(6);x<<0.2,-0.1,0.3,1.3,0.4,0.7;b<<0.7,0.1,-0.2,0.9,0.6,0.3;
    pf::LennardJones lj(2);std::cout<<lj.value(x)<<'\n';
    for(auto method:{pf::Mapping::linear,pf::Mapping::residual,pf::Mapping::affine,pf::Mapping::cnf})row(pf::map_particle(x,b,0.73,method));
    pf::Population p{x,b,2*x,2*b};pf::Population archive{x+b};
    auto w=pf::compute_weights({-4,-2,1,7},1.8,p,archive,3.1,0.01,0.01);row(w.weights);std::cout<<w.ess<<'\n';
    for(int t=0;t<5;++t)std::cout<<pf::next_beta(t,5,50,17,0.9,1.3,1e-4,5)<<' ';
    std::cout<<'\n';
    row(pf::stratified_indices({0.1,0.2,0.3,0.4},{0.2,0.5,0.1,0.9}));
    row(pf::sharing_factors(p,3.1));std::cout<<pf::diversity(p)<<'\n';
}
