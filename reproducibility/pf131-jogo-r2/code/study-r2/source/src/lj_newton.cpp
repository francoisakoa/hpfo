#include "lj_newton.hpp"
#include "linear.hpp"
#include <algorithm>
#include <cmath>

namespace pf::detail {
Vector local_projection(const Vector& x,const Bounds& b,int dim,double radius) {
    Vector y=b.project(x);
    if(dim) {
        // Exact projection: ball contained in box, so box projection is redundant.
        y=x;
        for(Eigen::Index i=0;i<y.size();i+=dim) {
            const double norm=y.segment(i,dim).norm();
            if(norm>radius)y.segment(i,dim)*=radius/norm;
        }
    }
    return y;
}
namespace {
double boundary_step(const Vector& s,const Vector& d,double radius) {
    const double sd=s.dot(d),dd=d.squaredNorm(),slack=std::max(0.0,radius*radius-s.squaredNorm());
    const double root=std::sqrt(sd*sd+dd*slack);
    return sd>0?slack/(root+sd):(-sd+root)/dd;
}
struct Tangent {
    const Vector& x;
    std::vector<bool> fixed,balls;
    Vector lambda;
    Tangent(const Vector& point,const Vector& g,const Bounds& b,int dim,double radius):x(point),fixed(x.size(),false),balls(x.size()/3,false),lambda(Vector::Zero(x.size())) {
        if(dim) {
            for(Eigen::Index i=0;i<x.size();i+=3) {
                const double norm2=x.segment<3>(i).squaredNorm(),dot=x.segment<3>(i).dot(g.segment<3>(i));
                if(norm2>=radius*radius*(1-1e-10)&&dot<0) {
                    balls[i/3]=true;lambda.segment<3>(i).setConstant(-dot/norm2);
                }
            }
        } else for(Eigen::Index i=0;i<x.size();++i)
            fixed[i]=(x[i]<=b.lower[i]+1e-12&&g[i]>0)||(x[i]>=b.upper[i]-1e-12&&g[i]<0);
    }
    Vector project(Vector v)const {
        for(Eigen::Index i=0;i<v.size();++i)if(fixed[i])v[i]=0;
        for(std::size_t i=0;i<balls.size();++i)if(balls[i]) {
            const auto p=x.segment<3>(3*i);
            v.segment<3>(3*i)-=p*(p.dot(v.segment<3>(3*i))/p.squaredNorm());
        }
        return v;
    }
};
Vector trust_cg(Context& c,const LJCurvature& h,const Tangent& tangent,double radius) {
    Vector residual=tangent.project(h.gradient),step=Vector::Zero(residual.size());
    Vector z=tangent.project(residual.cwiseQuotient(h.diagonal+tangent.lambda));
    Vector direction=-z;
    double rz=residual.dot(z);
    const double norm=residual.norm(),tol=std::min(0.1,std::sqrt(norm))*norm;
    const int maxiter=c.options.newton_cg_max_iterations?c.options.newton_cg_max_iterations:int(residual.size());
    for(int k=0;k<maxiter&&rz>0;++k) {
        c.check();++c.counters.hessian_products;
        Vector hd=tangent.project(h.product(direction)+tangent.lambda.cwiseProduct(direction));
        const double curvature=direction.dot(hd);
        if(curvature<=0) {
            ++c.counters.negative_curvature;
            return step+boundary_step(step,direction,radius)*direction;
        }
        const double alpha=rz/curvature;
        Vector next=step+alpha*direction;
        if(next.squaredNorm()>=radius*radius)return step+boundary_step(step,direction,radius)*direction;
        step=std::move(next);residual+=alpha*hd;
        if(residual.norm()<=tol)break;
        z=tangent.project(residual.cwiseQuotient(h.diagonal+tangent.lambda));
        const double next_rz=residual.dot(z);
        direction=-z+(next_rz/rz)*direction;rz=next_rz;
    }
    return step;
}
Vector trust_direct(Context& c,const LJCurvature& h,const Tangent& tangent,double radius,
                    SymmetricFactor& factor,double& damping) {
    // Hessian of the Lagrangian restricted to the current feasible tangent space.
    Matrix model=h.dense();model.diagonal()+=tangent.lambda;
    for(Eigen::Index j=0;j<model.cols();++j)model.col(j)=tangent.project(model.col(j));
    for(Eigen::Index i=0;i<model.rows();++i)model.row(i)=tangent.project(model.row(i).transpose()).transpose();
    const double scale=std::max(1.0,model.cwiseAbs().maxCoeff());
    // Give excluded normal directions positive curvature; the RHS is tangential.
    for(Eigen::Index j=0;j<model.cols();++j) {
        Vector e=Vector::Zero(model.rows());e[j]=1;
        model.col(j)+=scale*(e-tangent.project(e));
    }
    model=(0.5*(model+model.transpose())).eval();
    const Vector rhs=-tangent.project(h.gradient);
    const double floor=1e-12*scale;
    double shift=std::max(floor,damping*0.25),upper=floor;
    for(Eigen::Index i=0;i<model.rows();++i)
        upper=std::max(upper,model.row(i).cwiseAbs().sum()-std::abs(model(i,i))-model(i,i)+floor);
    for(int attempt=0;attempt<8;++attempt) {
        c.check();Matrix regularized=model;regularized.diagonal().array()+=shift;
        ++c.counters.linear_factorizations;
        if(factor.factor(regularized)) {
            ++c.counters.linear_solves;Vector step=factor.solve(rhs);
            if(step.allFinite()&&(regularized*step-rhs).norm()<=1e-7*std::max(1.0,rhs.norm())) {
                step=tangent.project(step);
                const double norm=step.norm();if(norm>radius)step*=radius/norm;
                damping=shift;return step;
            }
        }
        // Last attempt uses a strictly diagonally dominant positive definite model.
        shift=attempt==6?std::max(upper,shift*10):std::max(1e-6*scale,shift*10);
    }
    ++c.counters.linear_cg_fallbacks;
    return trust_cg(c,h,tangent,radius);
}
}
LocalResult refine_lj_newton(Context& c,const Vector& initial,const Bounds& b,double initial_value) {
    const auto* lj=dynamic_cast<const LennardJones*>(&c.problem);
    if(!lj||lj->spatial_dimension()!=3)throw std::invalid_argument("lj-newton requires 3D LennardJones");
    const int dim=c.ball_dimension();const double radius=c.active_ball_radius;
    if(dim&&((b.lower.array()>-radius).any()||(b.upper.array()<radius).any()))
        throw std::invalid_argument("lj-newton atom balls must be contained in the local box");
    auto project=[&](const Vector& point){return local_projection(point,b,dim,radius);};
    Vector x=project(initial);
    double f=(x.array()==initial.array()).all()?initial_value:c.evaluate(x,true);
    double delta=c.options.newton_initial_radius*std::sqrt(double(lj->atoms()));
    const double max_delta=2*std::sqrt(double(lj->atoms()));
    std::unique_ptr<SymmetricFactor> factor;
    if(c.options.newton_linear_solver!="cg") {
        factor=std::make_unique<SymmetricFactor>(int(x.size()),c.options);
        c.counters.linear_analyses+=factor->analyses();
        if(factor->uses_metis())c.counters.linear_metis_analyses+=factor->analyses();
        if(factor->uses_metis()&&factor->mumps_ordering()==1)c.counters.mumps_user_ordering_analyses+=factor->analyses();
    }
    double damping=0;
    LJCurvature h;bool update=true;int status=-1;
    for(int iteration=0;iteration<c.options.local_max_iterations;++iteration) {
        c.check();++c.counters.newton_iterations;
        if(update) {h.update(x);++c.counters.gradient;++c.counters.hessian;update=false;}
        const Vector& g=h.gradient;
        if(!g.allFinite()||!std::isfinite(f)){status=-2;break;}
        const Vector pg=x-project(x-g);
        if(pg.lpNorm<Eigen::Infinity>()<=c.options.local_tolerance) {
            Vector escape=h.collision_direction();
            if(escape.size()) {
                ++c.counters.hessian_products;
                if(escape.dot(h.product(escape))<0) {
                    escape*=delta/escape.norm();
                    Vector trial=project(x+escape);double ft=c.evaluate(trial,true);
                    if(ft<f) {++c.counters.negative_curvature;++c.counters.newton_accepted;x=std::move(trial);f=ft;update=true;continue;}
                }
                status=-2;break; // Never label a coincident repulsive maximum as converged.
            }
            status=0;break;
        }
        Tangent tangent(x,g,b,dim,radius);
        Vector direction=factor?trust_direct(c,h,tangent,delta,*factor,damping):trust_cg(c,h,tangent,delta);
        Vector step=project(x+direction)-x;
        ++c.counters.hessian_products;
        double predicted=-g.dot(step)-0.5*step.dot(h.product(step));
        // A projected Cauchy step safeguards constrained/indefinite models.
        Vector descent=-g.cwiseQuotient(h.diagonal);
        if(descent.norm()>delta)descent*=delta/descent.norm();
        Vector cauchy=project(x+descent)-x;
        ++c.counters.hessian_products;
        double hc=cauchy.dot(h.product(cauchy)),gc=g.dot(cauchy);
        if(hc>0&&-gc<hc)cauchy*=std::max(0.0,-gc/hc);
        ++c.counters.hessian_products;
        const double cp=-g.dot(cauchy)-0.5*cauchy.dot(h.product(cauchy));
        if(cp>predicted||!std::isfinite(predicted)){step=std::move(cauchy);predicted=cp;}
        if(!(predicted>0)||!step.allFinite()) {status=-2;break;}
        Vector trial=project(x+step);step=trial-x;
        const double ft=c.evaluate(trial,true),actual=f-ft;
        const double ratio=actual/predicted;
        // Close to a stationary point objective cancellation can erase reduction.
        const double roundoff=8*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(f));
        bool accept=std::isfinite(ft)&&actual>0&&ratio>1e-4;
        if(!accept&&std::isfinite(ft)&&std::abs(actual)<=roundoff&&predicted<=roundoff) {
            Vector gt=c.gradient(trial);
            accept=(trial-project(trial-gt)).lpNorm<Eigen::Infinity>()<0.5*pg.lpNorm<Eigen::Infinity>();
        }
        if(accept) {
            ++c.counters.newton_accepted;x=std::move(trial);f=ft;update=true;
            if(ratio>0.75&&step.norm()>0.8*delta)delta=std::min(2*delta,max_delta);
        } else ++c.counters.newton_rejected;
        if(ratio<0.25||!std::isfinite(ratio))delta*=0.25;
        if(delta<1e-13) {status=-2;break;}
    }
    return {x,f,status};
}
} // namespace pf::detail
