#include "pf/pf.hpp"
#include <iostream>

class Sphere final : public pf::Problem {
public:
    Eigen::Index dimension() const override { return 6; }
    double value(const pf::Vector& x) const override { return x.squaredNorm(); }
    pf::Vector gradient(const pf::Vector& x) const override { return 2*x; }
};
int main() {
    Sphere problem;
    pf::Bounds bounds{pf::Vector::Constant(6,-5),pf::Vector::Constant(6,5)};
    pf::Options options;
    options.local_method=pf::LocalMethod::projected_lbfgs;
    options.local_workers=1; // custom callbacks stay in the parent process
    options.batch_threads=2;
    options.max_iterations=10;
    options.target=0;
    options.target_tolerance=1e-10;
    const auto result=pf::Solver(problem,bounds,options).solve();
    std::cout<<result.status<<" objective="<<result.objective<<'\n';
}
