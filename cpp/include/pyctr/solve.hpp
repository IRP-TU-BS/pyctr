// Levenberg-Marquardt for small nonlinear systems F(x) = 0 (shooting methods), forward difference Jacobian.
// The Python implementation uses scipy least_squares / root (MINPACK).
#pragma once

#include <cmath>
#include <limits>

#include "pyctr/core.hpp"

namespace pyctr {

struct SolveResult {
  VecX x;
  VecX residual;
  bool success = false;
  int evaluations = 0;
};

// Marquardt scaling with diag(J^T J). Stops when |F|_inf <= ftol or the relative step is below xtol;
// success only if |F|_inf <= ftol (a small step can also mean a local minimum of |F|).
template <class F>
SolveResult levenberg_marquardt(F&& f, VecX x, double ftol = 1e-12, double xtol = 1e-12, int max_evaluations = 2000) {
  SolveResult result;
  VecX r = f(x);
  result.evaluations = 1;
  double lambda = 1e-3;
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  bool small_step = false;
  while (result.evaluations < max_evaluations) {
    if (!r.allFinite() || r.lpNorm<Eigen::Infinity>() <= ftol) break;
    MatX J(r.size(), x.size());
    for (int j = 0; j < x.size(); ++j) {
      VecX xh = x;
      const double h = eps * std::max(1.0, std::abs(x(j)));
      xh(j) += h;
      J.col(j) = (f(xh) - r) / h;
    }
    result.evaluations += static_cast<int>(x.size());
    const MatX JtJ = J.transpose() * J;
    const VecX g = J.transpose() * r;
    VecX D = JtJ.diagonal().cwiseMax(1e-30);
    bool improved = false;
    while (result.evaluations < max_evaluations) {
      const MatX A = JtJ + lambda * MatX(D.asDiagonal());
      const VecX dx = -A.ldlt().solve(g);
      const VecX xn = x + dx;
      const VecX rn = f(xn);
      ++result.evaluations;
      if (rn.allFinite() && rn.squaredNorm() < r.squaredNorm()) {
        small_step = dx.norm() <= xtol * (xtol + x.norm());
        x = xn;
        r = rn;
        lambda = std::max(lambda / 10, 1e-12);
        improved = true;
        break;
      }
      lambda *= 10;
      if (lambda > 1e16) break;
    }
    if (!improved || small_step) break;
  }
  result.success = r.allFinite() && r.lpNorm<Eigen::Infinity>() <= ftol;
  result.x = x;
  result.residual = r;
  return result;
}

}  // namespace pyctr
