#pragma once
#include "internal.hpp"
#include <Eigen/Geometry>

namespace pf::detail {
// Cached exact pair curvature, assembled once per accepted iterate.
class LJCurvature {
    struct Pair {int i,j;Eigen::Vector3d delta;double a,b;};
    std::vector<Pair> pairs_;
public:
    Vector gradient,diagonal;
    void update(const Vector& x) {
        const int atoms=int(x.size()/3);
        pairs_.clear();pairs_.reserve(std::size_t(atoms)*(atoms-1)/2);
        gradient=Vector::Zero(x.size());diagonal=Vector::Zero(x.size());
        for(int i=0;i<atoms;++i)for(int j=0;j<i;++j) {
            Eigen::Vector3d d=x.segment<3>(3*i)-x.segment<3>(3*j);
            const double q=1/(d.squaredNorm()+1e-8),q3=q*q*q;
            const double a=12*q*q3*(1-q3),b=q*q*q3*(168*q3-96);
            pairs_.push_back({3*i,3*j,d,a,b});
            const Eigen::Vector3d force=a*d;
            gradient.segment<3>(3*i)+=force;gradient.segment<3>(3*j)-=force;
            const double bound=std::abs(a)+std::abs(b)*d.squaredNorm();
            diagonal.segment<3>(3*i).array()+=bound;diagonal.segment<3>(3*j).array()+=bound;
        }
        diagonal=diagonal.cwiseMax(1e-6);
    }
    Vector product(const Vector& v)const {
        Vector result=Vector::Zero(v.size());
        for(const auto& pair:pairs_) {
            const Eigen::Vector3d d=v.segment<3>(pair.i)-v.segment<3>(pair.j);
            const Eigen::Vector3d value=pair.a*d+(pair.b*pair.delta.dot(d))*pair.delta;
            result.segment<3>(pair.i)+=value;result.segment<3>(pair.j)-=value;
        }
        return result;
    }
    Matrix dense()const {
        Matrix matrix=Matrix::Zero(gradient.size(),gradient.size());
        for(const auto& pair:pairs_) {
            Eigen::Matrix3d block=pair.a*Eigen::Matrix3d::Identity()+pair.b*(pair.delta*pair.delta.transpose());
            matrix.block<3,3>(pair.i,pair.i)+=block;matrix.block<3,3>(pair.j,pair.j)+=block;
            matrix.block<3,3>(pair.i,pair.j)-=block;matrix.block<3,3>(pair.j,pair.i)-=block;
        }
        return matrix;
    }
    Vector collision_direction()const {
        // Exact coincidences are stationary maxima of the epsilon-smoothed pair.
        // A deterministic antisymmetric displacement exposes their negative mode.
        for(const auto& pair:pairs_)if(pair.delta.squaredNorm()<1e-24) {
            Vector v=Vector::Zero(gradient.size());v[pair.i]=1;v[pair.j]=-1;return v;
        }
        return {};
    }
};
}
