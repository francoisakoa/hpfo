#include "../src/lj_newton.hpp"
#include <iostream>

namespace {
int checks=0;
void check(bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);}
pf::Bounds box(int d,double r=5){return {pf::Vector::Constant(d,-r),pf::Vector::Constant(d,r)};}
}
int main() {
    try {
        // Exact cached H*v must agree with the independent assembled Hessian.
        pf::LennardJones model(5);pf::Vector x(15);
        x<<0,0,0,1.12,.1,.02,.3,1.14,.2,-.15,.15,1.2,1.2,1.1,1;
        pf::detail::LJCurvature curvature;curvature.update(x);
        check((curvature.dense()-model.hessian(x)).norm()<1e-10,"exact assembled pair Hessian");
        check((curvature.gradient-model.gradient(x)).norm()<1e-11,"exact gradient");
        pf::Random rng(95);
        for(int k=0;k<10;++k) {
            pf::Vector v(15);for(auto& value:v)value=rng.normal();
            pf::Vector product=curvature.product(v),assembled=model.hessian(x)*v;
            check((product-assembled).norm()<1e-12*std::max(1.0,assembled.norm()),"exact Hessian product");
            const double eps=1e-6;
            pf::Vector fd=(model.gradient(x+eps*v)-model.gradient(x-eps*v))/(2*eps);
            check((product-fd).norm()<1e-6*std::max(1.0,fd.norm()),"Hessian product finite differences");
        }
        pf::Vector translation=pf::Vector::Ones(15);
        check(curvature.product(translation).norm()==0,"translation null mode");
        pf::Options o;o.local_method=pf::LocalMethod::lj_newton;o.local_tolerance=1e-8;o.particles=8;o.min_particles=2;
        pf::LennardJones pair(2);auto b=box(6);pf::Vector initial(6);initial<<.3,.2,.2,-.1,-.3,-.1;
        for(double radius:{5.,.35,.25}) {
            o.radius=radius;pf::detail::Context c(pair,o,b);
            auto result=pf::detail::refine(c,initial,b);
            check(c.local_feasible(result.x),"active atom balls feasible");
            const double distance=std::min(2*radius,std::sqrt(1-1e-8));
            const double q=1/(distance*distance+1e-8),q3=q*q*q;
            check(std::abs(result.objective-q3*(q3-2))<1e-7*std::max(1.,std::abs(result.objective)),"constrained pair analytic optimum");
            check(result.status==0,"pair stationary convergence");
        }
        o.radius=5;o.local_max_iterations=1000;o.max_iterations=3;o.final_refinement=false;
        pf::detail::Context collision_context(pair,o,b);
        auto separated=pf::detail::refine(collision_context,pf::Vector::Zero(6),b);
        check(std::abs(separated.objective+1)<1e-8&&separated.status==0,"escape coincident stationary maximum");
        check(collision_context.counters.negative_curvature>0,"coincident negative curvature detected");
        pf::LennardJones small(4);auto bounds=box(12);
        auto population=pf::python_initialization(bounds,8,2024,4,1).population;
        auto serial=pf::Solver(small,bounds,o).solve(population);
        check(serial.counters.hessian_products>0&&serial.counters.newton_iterations>0,"Newton path executed");
        check(std::abs(serial.objective+6)<1e-6,"LJ4 minimum");
#ifdef _OPENMP
        o.local_workers=3;
        auto parallel=pf::Solver(small,bounds,o).solve(population);
        check((serial.x.array()==parallel.x.array()).all(),"thread replay exact coordinates");
        check(serial.counters.objective_total()==parallel.counters.objective_total(),"thread replay exact counters");
        check(serial.counters.hessian_products==parallel.counters.hessian_products,"thread replay exact Hessian products");
#endif
        o.max_objective_evaluations=31;
        auto limited=pf::Solver(small,bounds,o).solve(population);
        check(limited.status=="evaluation_limit"&&limited.counters.objective_total()<=31,"strict shared budget");
        o.max_objective_evaluations=0;o.max_seconds=1e-9;
        check(pf::Solver(small,bounds,o).solve(population).status=="time_limit","deadline");
        o.max_seconds=30;o.local_workers=1;
        o.ipopt_ball_constraints=false;
        pf::detail::Context box_context(pair,o,b);auto box_result=pf::detail::refine(box_context,initial,b);
        check(std::abs(box_result.objective+1)<1e-8,"box-only Newton");
        bool rejected=false;
        try {pf::LennardJones invalid(3,2);pf::Solver solver(invalid,box(6),o);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"unsupported geometry rejected");
        std::cout<<"PASS "<<checks<<" LJ Newton checks\n";
    }catch(const std::exception& e){std::cerr<<"FAIL after "<<checks<<": "<<e.what()<<'\n';return 1;}
}
