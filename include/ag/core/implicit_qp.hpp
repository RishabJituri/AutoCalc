
#ifndef AG_IMPLICIT_QP_HPP
#define AG_IMPLICIT_QP_HPP

#include "variables.hpp"
#include <cstddef>
#include <vector>

namespace ag {

// Solve equality-constrained QP (dense, small):
//   minimize  1/2 y^T H y + q^T y
//   subject to A y = b
// Shapes: H [N,N], q [N], A [M,N] (M can be 0), b [M]
// Returns y* as a Variable. Backward uses implicit differentiation (KKT) to produce grads for H,q,A,b.
Variable QPSolveEq(const Variable& H, const Variable& q,
                   const Variable& A, const Variable& b,
                   double eps_pd = 1e-8);

} // namespace ag

#endif // AG_IMPLICIT_QP_HPP
