#pragma once
#include <vector>
namespace pf::detail {
// The exact LJ Hessian is structurally complete, including currently zero entries.
// Both vectors use zero-based indices: perm[new]=old, inverse[old]=new.
struct MetisPermutation { std::vector<int> perm, inverse; };
MetisPermutation metis_complete_ordering(int dimension);
}
