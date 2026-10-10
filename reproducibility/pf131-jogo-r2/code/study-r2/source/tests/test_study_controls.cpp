#include "pf/pf.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void check(bool c,const char* m){if(!c)throw std::runtime_error(m);}
struct Flat:pf::Problem{
    Eigen::Index dimension() const override{return 9;}
    double value(const pf::Vector&) const override{return 1.0;}
    pf::Vector gradient(const pf::Vector& x)const override{return pf::Vector::Zero(x.size());}
};
int main(){try{
    pf::Options o;o.local_method=pf::LocalMethod::none;o.particles=12;o.min_particles=5;
    o.refine_top_k=0;o.max_iterations=31;o.no_improve_stop=0;o.max_seconds=30;
    o.ipf_endpoint_schedule=true;
    for(double a:{0.95,0.98,0.99,1.0}){
        o.ipf_alpha_initial=a;
        check(std::abs(pf::ipf_step(o,0)-a)<1e-15,"initial alpha");
        check(std::abs(pf::ipf_step(o,30)-0.05)<1e-15,"final alpha");
    }
    o.ipf_endpoint_schedule=false;check(std::abs(pf::ipf_step(o,0)-1.03)<1e-15,"legacy trajectory");
    o.ipf_endpoint_schedule=true;o.ipf_enabled=false;check(pf::ipf_step(o,0)==0,"identity transport");
    pf::Vector x=pf::Vector::Ones(9),best=pf::Vector::Zero(9);
    check(pf::map_particle(x,best,0,pf::Mapping::linear)==x,"identity mapping");
    check(pf::map_particle(x,best,1,pf::Mapping::linear)==best,"alpha=1 collapse");
    Flat f;pf::Bounds b{pf::Vector::Constant(9,-5),pf::Vector::Constant(9,5)};
    o.adaptive_annealing=false;o.island_exchange=false;o.kicks=false;
    auto r=pf::Solver(f,b,o).solve();
    check(r.iterations==31,"iterations");check(r.counters.local_calls==0,"no local calls");
    check(r.counters.objective_local==0,"no local evals");
    for(auto& h:r.history){
        check(h.beta_used==o.beta_initial&&h.beta_next==o.beta_initial,"fixed beta");
        check(h.ipf_alpha==0&&!h.kick&&!h.island_exchange,"disabled components");
    }
    // These other controls remain active in the full protocol.
    o.ipf_enabled=true;o.ipf_alpha_initial=.98;o.island_exchange=true;o.kicks=true;o.adaptive_annealing=true;
    auto all=pf::Solver(f,b,o).solve();bool kick=false,island=false;
    for(auto& h:all.history){kick=kick||h.kick;island=island||h.island_exchange;}
    check(kick&&island,"positive mechanism controls");
    std::cout<<"PASS endpoints, identity/collapse, legacy alpha, fixed beta, no local/kick/island and positive controls\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
