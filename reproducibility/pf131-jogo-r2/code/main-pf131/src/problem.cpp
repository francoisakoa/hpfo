#include "pf/pf.hpp"
#include <cmath>

namespace pf {
void Bounds::validate(Eigen::Index dimension) const {
    if (dimension < 1 || lower.size() != dimension || upper.size() != dimension ||
        !lower.allFinite() || !upper.allFinite() || (lower.array() >= upper.array()).any())
        throw std::invalid_argument("Bounds must be finite, correctly sized and strictly ordered");
}
Vector Bounds::project(const Vector& x) const {
    if (x.size() != lower.size()) throw std::invalid_argument("Projection dimension mismatch");
    return x.cwiseMax(lower).cwiseMin(upper);
}
bool Bounds::contains(const Vector& x, double tolerance) const {
    return x.size() == lower.size() && x.allFinite() &&
        (x.array() >= lower.array() - tolerance).all() &&
        (x.array() <= upper.array() + tolerance).all();
}
Matrix Problem::hessian(const Vector&) const { throw std::logic_error("Problem has no exact Hessian"); }
LennardJones::LennardJones(int atoms, int spatial_dimension)
    : atoms_(atoms), spatial_dimension_(spatial_dimension) {
    if (atoms < 2 || spatial_dimension < 1 || atoms > 100000 || spatial_dimension > 100)
        throw std::invalid_argument("Invalid Lennard-Jones dimensions");
}
Eigen::Index LennardJones::dimension() const { return Eigen::Index(atoms_) * spatial_dimension_; }
void LennardJones::check(const Vector& x) const {
    if (x.size() != dimension()) throw std::invalid_argument("Lennard-Jones coordinate dimension mismatch");
    if (!x.allFinite()) throw std::invalid_argument("Coordinates must be finite");
}
double LennardJones::value(const Vector& x) const {
    check(x);
    double total = 0, compensation = 0;
    for (int i=0; i<atoms_; ++i) for (int j=0; j<i; ++j) {
        double r2 = (x.segment(i*spatial_dimension_,spatial_dimension_) -
                     x.segment(j*spatial_dimension_,spatial_dimension_)).squaredNorm();
        double q = 1.0/(r2 + 1e-8), q3=q*q*q;
        double term = q3*(q3-2.0) - compensation;
        double sum = total + term;
        compensation = (sum-total)-term;
        total = sum;
    }
    return total;
}
Vector LennardJones::gradient(const Vector& x) const {
    check(x);
    Vector g=Vector::Zero(dimension());
    for (int i=0; i<atoms_; ++i) for (int j=0; j<i; ++j) {
        const int ii=i*spatial_dimension_,jj=j*spatial_dimension_;
        double r2=0;
        for(int k=0;k<spatial_dimension_;++k) { double dx=x[ii+k]-x[jj+k];r2+=dx*dx; }
        double q=1.0/(r2 + 1e-8), q3=q*q*q;
        const double factor=12*q*q3*(1-q3);
        for(int k=0;k<spatial_dimension_;++k) {
            double v=factor*(x[ii+k]-x[jj+k]);g[ii+k]+=v;g[jj+k]-=v;
        }
    }
    return g;
}
Matrix LennardJones::hessian(const Vector& x) const {
    check(x);
    Matrix h=Matrix::Zero(dimension(),dimension());
    for (int i=0; i<atoms_; ++i) for (int j=0; j<i; ++j) {
        const int ii=i*spatial_dimension_,jj=j*spatial_dimension_;
        double r2=0;
        for(int k=0;k<spatial_dimension_;++k) { double dx=x[ii+k]-x[jj+k];r2+=dx*dx; }
        double q=1.0/(r2 + 1e-8),q3=q*q*q;
        double a=12*q*q3*(1-q3), b=q*q*q3*(168*q3-96);
        for(int k=0;k<spatial_dimension_;++k) for(int l=0;l<spatial_dimension_;++l) {
            double v=b*(x[ii+k]-x[jj+k])*(x[ii+l]-x[jj+l])+(k==l?a:0);
            h(ii+k,ii+l)+=v;h(jj+k,jj+l)+=v;h(ii+k,jj+l)-=v;h(jj+k,ii+l)-=v;
        }
    }
    return h;
}
void Options::validate(const Problem& problem,const Bounds& bounds) const {
    bounds.validate(problem.dimension());
#ifndef _OPENMP
    if(batch_threads!=1) throw std::invalid_argument("This build has no OpenMP; use batch_threads=1 or rebuild with OpenMP");
#endif
    if (max_iterations<1 || no_improve_stop<0 || particles<2 || min_particles<2 || min_particles>particles ||
        refine_top_k<0 || batch_threads<1 || local_workers<1 || local_max_iterations<1 ||
        !std::isfinite(local_tolerance) || local_tolerance<=0 || !std::isfinite(max_seconds) || max_seconds<=0 ||
        !std::isfinite(radius) || radius<=0 || !std::isfinite(mutation_sigma) || mutation_sigma<0 ||
        !std::isfinite(beta_initial) || beta_initial<=0 || !std::isfinite(beta_max) || beta_max<beta_initial ||
        !std::isfinite(target_tolerance) || target_tolerance<0 || std::isnan(target) || target==infinity ||
        !std::isfinite(lambda_share) || lambda_share<0 || !std::isfinite(lambda_novelty) || lambda_novelty<0 ||
        kmeans_restarts<1 || cluster_max_iterations<1 || gmm_max_iterations<1 || !std::isfinite(dbscan_epsilon) || dbscan_epsilon<=0 ||
        dbscan_min_samples<1 || ipopt_print_level<0 || ipopt_print_level>12)
        throw std::invalid_argument("Invalid solver options");
    if(newton_cg_max_iterations<0||!std::isfinite(newton_initial_radius)||newton_initial_radius<=0)
        throw std::invalid_argument("Invalid Newton options");
    if(newton_linear_solver!="cg"&&newton_linear_solver!="ldl"&&newton_linear_solver!="ma27"&&newton_linear_solver!="mumps")throw std::invalid_argument("Unknown Newton linear solver: "+newton_linear_solver);
    if(newton_linear_ordering!="auto"&&newton_linear_ordering!="metis")throw std::invalid_argument("Newton ordering must be auto or metis");
    if(newton_linear_ordering=="metis") {
        if(newton_linear_solver!="ldl"&&newton_linear_solver!="mumps")throw std::invalid_argument("METIS ordering requires Newton ldl or mumps");
        if(!metis_available())throw std::invalid_argument("METIS was not compiled in; rebuild with PF_WITH_METIS=ON");
    }
    if(ipopt_linear_solver=="ldl")throw std::invalid_argument("Ipopt does not expose standalone LDL. Use --newton-linear-solver ldl with lj-newton, or --ipopt-linear-solver ma27|mumps");
    if(local_method==LocalMethod::lj_newton) {
        if(newton_linear_solver=="mumps"&&!mumps_available())throw std::invalid_argument("Native MUMPS was not compiled in; rebuild with PF_WITH_MUMPS=ON");
        const auto* lj=dynamic_cast<const LennardJones*>(&problem);
        if(!lj||lj->spatial_dimension()!=3)throw std::invalid_argument("lj-newton requires a 3D LennardJones problem");
        if(ipopt_ball_constraints&&((bounds.lower.array()>-radius).any()||(bounds.upper.array()<radius).any()))
            throw std::invalid_argument("lj-newton atom balls must be contained in the box");
#ifndef _OPENMP
        if(local_workers>1&&(newton_linear_solver=="cg"||newton_linear_solver=="ldl"))throw std::invalid_argument("Threaded lj-newton requires OpenMP");
#endif
    }
    if (mapping==Mapping::affine && flow==Flow::invertible && problem.dimension()%2)
        throw std::invalid_argument("Source affine mapping requires an even dimension; choose linear/residual/cnf");
    if (local_method==LocalMethod::ipopt && !ipopt_available())
        throw std::invalid_argument("Ipopt was not compiled in; rebuild with PF_WITH_IPOPT=ON or select projected-lbfgs explicitly");
}
Population uniform_population(const Bounds& bounds,int count,Random& rng) {
    bounds.validate(bounds.lower.size());
    if (count<1) throw std::invalid_argument("Population size must be positive");
    Population p; p.reserve(count);
    for(int i=0;i<count;++i) {
        Vector x(bounds.lower.size());
        for(Eigen::Index j=0;j<x.size();++j) x[j]=bounds.lower[j]+rng.uniform()*(bounds.upper[j]-bounds.lower[j]);
        p.push_back(std::move(x));
    }
    return p;
}
PythonInitialization python_initialization(const Bounds& bounds,int count,std::uint64_t base,int atoms,int run) {
    bounds.validate(bounds.lower.size());
    RunSeeds seeds{base,atoms,run,derive_seed(base,atoms,run,0),derive_seed(base,atoms,run,1000)};
    Random initial_rng(seeds.init),pf_rng(seeds.pf);
    Vector x0(bounds.lower.size());
    for(Eigen::Index j=0;j<x0.size();++j) {
        const double margin=0.1*(bounds.upper[j]-bounds.lower[j]);
        const double low=bounds.lower[j]+margin,high=bounds.upper[j]-margin;
        x0[j]=low+(high-low)*initial_rng.uniform();
    }
    return {seeds,std::move(x0),uniform_population(bounds,count,pf_rng)};
}
} // namespace pf
