#include "pf/pf.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace pf {
double ipf_step(const Options& o,int iteration) {
    if(!o.ipf_enabled) return 0.0;
    if(o.ipf_endpoint_schedule) {
        const double t=o.max_iterations>1?double(iteration)/(o.max_iterations-1):0.0;
        // alpha(0)=alpha_initial and alpha(T-1)=alpha_final, with no +0.05 offset.
        return o.ipf_alpha_initial+(o.ipf_alpha_final-o.ipf_alpha_initial)*std::clamp(t,0.0,1.0);
    }
    return 0.98*(1-double(iteration)/o.max_iterations)+0.05; // unchanged PF131
}
Vector map_particle(const Vector& x,const Vector& best,double alpha,Mapping mapping) {
    if(x.size()!=best.size() || !std::isfinite(alpha)) throw std::invalid_argument("Invalid mapping arguments");
    switch(mapping) {
    case Mapping::linear: return (1-alpha)*x+alpha*best;
    case Mapping::residual: return x+alpha*(best-x).array().tanh().matrix();
    case Mapping::affine: {
        if(x.size()%2) throw std::invalid_argument("Affine mapping needs an even dimension");
        Vector y=x;Eigen::Index mid=x.size()/2;
        for(Eigen::Index i=0;i<mid;++i) y[mid+i]=x[mid+i]*std::exp(std::clamp(alpha*(best[i]-x[i]),-2.0,2.0))+alpha*(best[mid+i]-x[mid+i]);
        return y;
    }
    case Mapping::cnf: {
        // Exact solution of dy/dt=alpha*tanh(best-y), t in [0,1].
        Vector y(x.size());
        for(Eigen::Index i=0;i<x.size();++i) {
            double delta=x[i]-best[i];
            double a=std::abs(delta);
            y[i]=best[i]+(a>350?std::copysign(std::max(0.0,a-alpha),delta):std::asinh(std::exp(-alpha)*std::sinh(delta)));
        }
        return y;
    }
    }
    throw std::invalid_argument("Unknown mapping");
}
std::vector<double> sharing_factors(const Population& p,double sigma,int threads) {
    if(!std::isfinite(sigma)||sigma<=0) throw std::invalid_argument("Invalid sharing radius");
    std::vector<double> s(p.size(),p.size()<2?0.0:1.0);
    // backend_ops includes the self contribution; singleton input returns zero.
    #pragma omp parallel for schedule(static) num_threads(threads) if(p.size()>64)
    for(std::int64_t i=0;i<static_cast<std::int64_t>(p.size());++i)
        for(std::size_t j=0;j<p.size();++j) if(static_cast<std::size_t>(i)!=j) s[i]+=std::max(0.0,1-(p[i]-p[j]).norm()/sigma);
    return s;
}
std::vector<double> novelty_scores(const Population& p,const Population& archive,int threads) {
    std::vector<double> v(p.size(),0);
    if(archive.empty()) return v;
    #pragma omp parallel for schedule(static) num_threads(threads) if(p.size()>64)
    for(std::int64_t i=0;i<static_cast<std::int64_t>(p.size());++i) {
        for(const auto& a:archive) v[i]+=(p[i]-a).norm();
        v[i]/=static_cast<double>(archive.size());
    }
    return v;
}
WeightResult compute_weights(const std::vector<double>& f,double beta,const Population& p,
                             const Population& archive,double sigma,double share,double novelty,int threads) {
    if(f.empty()||f.size()!=p.size()||!std::isfinite(beta)||beta<0) throw std::invalid_argument("Invalid weights input");
    // A zero weight really removes the metric's work, not just its contribution.
    auto s=share==0?std::vector<double>(p.size(),0):sharing_factors(p,sigma,threads);
    auto n=novelty==0?std::vector<double>(p.size(),0):novelty_scores(p,archive,threads);
    std::vector<double> effective(f.size());
    double minimum=infinity,scale=1e-8;
    std::size_t valid=0;
    for(std::size_t i=0;i<f.size();++i) {
        effective[i]=f[i]+share*s[i]-novelty*n[i];
        if(std::isfinite(effective[i])) { minimum=std::min(minimum,effective[i]);scale=std::max(scale,std::abs(effective[i]));++valid; }
    }
    WeightResult r{std::vector<double>(f.size(),0),0};
    if(!valid) { std::fill(r.weights.begin(),r.weights.end(),1.0/static_cast<double>(f.size()));r.ess=static_cast<double>(f.size());return r; }
    double mean=0,var=0;
    for(double v:effective) if(std::isfinite(v)) mean+=(v/scale)/static_cast<double>(valid);
    for(double v:effective) if(std::isfinite(v)) { double z=v/scale-mean;var+=z*z/static_cast<double>(valid); }
    const double std_scaled=std::max(std::sqrt(var),1e-8/scale);
    double sum=0;
    for(std::size_t i=0;i<f.size();++i) if(std::isfinite(effective[i])) {
        double z=(effective[i]/scale-minimum/scale)/std_scaled;
        r.weights[i]=std::max(1e-8,std::exp(-beta*z));sum+=r.weights[i];
    }
    double square=0;
    for(double& w:r.weights) { w/=sum;square+=w*w; }
    r.ess=1/square;return r;
}
double diversity(const Population& p,int threads) {
    if(p.size()<2) return infinity;
    std::vector<double> rows(p.size(),0);
    #pragma omp parallel for schedule(static) num_threads(threads) if(p.size()>64)
    for(std::int64_t i=0;i<static_cast<std::int64_t>(p.size());++i) for(std::int64_t j=0;j<i;++j) rows[i]+=(p[i]-p[j]).norm();
    double sum=std::accumulate(rows.begin(),rows.end(),0.0);
    return sum/(static_cast<double>(p.size())*static_cast<double>(p.size()-1)/2);
}
double next_beta(int iteration,int max_iterations,int initial_particles,double ess,double beta,
                 double factor,double initial,double maximum) {
    // self.particles is never assigned in the source: diversity_factor is exactly 1.
    double progress=max_iterations>1?double(iteration)/(max_iterations-1):0;
    double ess_factor=std::clamp(ess/(0.5*initial_particles),0.1,1.0);
    return std::clamp(beta+factor*(maximum-beta)*progress*ess_factor,initial*(1-0.5*progress),maximum);
}
std::vector<std::size_t> stratified_indices(const std::vector<double>& weights,const std::vector<double>& draws) {
    if(weights.empty()||weights.size()!=draws.size()) throw std::invalid_argument("Invalid resampling dimensions");
    double sum=0;
    for(double w:weights) { if(!std::isfinite(w)||w<0) throw std::invalid_argument("Invalid resampling weight");sum+=w; }
    if(sum<=0) throw std::invalid_argument("Resampling needs positive mass");
    std::vector<std::size_t> indices(weights.size());
    std::size_t j=0;double cumulative=weights[0]/sum;
    for(std::size_t i=0;i<weights.size();++i) {
        if(draws[i]<0||draws[i]>=1||!std::isfinite(draws[i])) throw std::invalid_argument("Invalid uniform draw");
        double position=(static_cast<double>(i)+draws[i])/static_cast<double>(weights.size());
        while(j+1<weights.size() && position>cumulative) cumulative+=weights[++j]/sum;
        indices[i]=j;
    }
    return indices;
}
} // namespace pf
