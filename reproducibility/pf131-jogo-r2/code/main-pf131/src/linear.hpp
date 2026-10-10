#pragma once
#include "internal.hpp"
#include <memory>
namespace pf::detail {
class SymmetricFactor {
public:
    SymmetricFactor(int dimension,const Options& options);
    ~SymmetricFactor();
    bool factor(const Matrix& matrix); // true only for positive definite inertia
    Vector solve(const Vector& rhs);
    std::uint64_t analyses()const;
    bool uses_metis()const;
    int mumps_ordering()const; // INFOG(7), -1 for non-MUMPS
    const std::vector<int>& permutation()const; // new position -> original variable
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
bool newton_uses_threads(const Options& o);
}
