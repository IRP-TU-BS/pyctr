#include "pyctr/rod_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <unsupported/Eigen/NonLinearOptimization>

namespace pyctr {

using RMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

DynamicCosseratRod::DynamicCosseratRod(const TubeParameters& params, double dt, double bdf_alpha)
    : params(params), N(static_cast<int>(std::ceil(params.L / params.s)) + 1), dt(dt) {
  rhoJ_ = params.rho * Vec3(params.I(), params.I(), params.J()).asDiagonal();
  c0_ = (1.5 + bdf_alpha) / (dt * (1 + bdf_alpha));
  c1_ = -2 / dt;
  c2_ = (0.5 + bdf_alpha) / (dt * (1 + bdf_alpha));
  d1_ = bdf_alpha / (1 + bdf_alpha);
}

void DynamicCosseratRod::set_initial_conditions(const Vec3& p0, const Mat3& R0) {
  p0_ = p0;
  R0_ = R0;
  has_initial_conditions_ = true;
}

void DynamicCosseratRod::integrate(const Vec6& guess, double c0, const RowMat& z_h, RowMat& y, RowMat& z) const {
  const double ds = params.L / (N - 1);
  const Mat3 Kse = params.Kse(), Kbt = params.Kbt();
  const Mat3 Kse_inv = (Kse + c0 * params.Bse).inverse();
  const Mat3 Kbt_inv = (Kbt + c0 * params.Bbt).inverse();
  const Vec3 Kse_vstar = Kse.col(2);
  const double rhoA = params.rho * params.A();
  const Vec3 rhoAg = rhoA * params.gravity;

  y.setZero(N, 24);  // q = w = 0 at the clamped base
  z.resize(N, 12);
  Eigen::Map<Vec3>(&y(0, 0)) = p0_;
  Eigen::Map<RMat3>(&y(0, 3)) = R0_;
  Eigen::Map<Vec6>(&y(0, 12)) = guess;
  for (int j = 0; j < N; ++j) {
    const Eigen::Map<const RMat3> R(&y(j, 3));
    const Eigen::Map<const Vec3> n(&y(j, 12)), m(&y(j, 15)), q(&y(j, 18)), w(&y(j, 21));
    const Eigen::Map<const Vec3> q_h(&z_h(j, 0)), w_h(&z_h(j, 3)), v_h(&z_h(j, 6)), u_h(&z_h(j, 9));
    const Vec3 u_star(0, j * ds > params.straight_length ? params.kappa : 0, 0);

    const Vec3 v = Kse_inv * (R.transpose() * n + Kse_vstar - params.Bse * v_h);
    const Vec3 u = Kbt_inv * (R.transpose() * m + Kbt * u_star - params.Bbt * u_h);
    z.row(j) << q.transpose(), w.transpose(), v.transpose(), u.transpose();
    if (j == N - 1) break;
    const Vec3 q_t = c0 * q + q_h, w_t = c0 * w + w_h, v_t = c0 * v + v_h, u_t = c0 * u + u_h;

    const Mat3 U = hat(u), W = hat(w);
    const Vec3 ps = R * v;
    const Mat3 Rs = R * U;
    const Vec3 ns = rhoA * R * (q_t + W * q) + R * params.C * q.cwiseProduct(q.cwiseAbs()) - rhoAg;
    const Vec3 ms = R * (rhoJ_ * w_t + W * rhoJ_ * w) - hat(ps) * n;
    const Vec3 qs = v_t - U * q + W * v;
    const Vec3 ws = u_t - U * w;
    Eigen::Map<Vec3>(&y(j + 1, 0)) = Eigen::Map<const Vec3>(&y(j, 0)) + ds * ps;
    Eigen::Map<RMat3>(&y(j + 1, 3)) = R + ds * Rs;
    Eigen::Map<Vec3>(&y(j + 1, 12)) = n + ds * ns;
    Eigen::Map<Vec3>(&y(j + 1, 15)) = m + ds * ms;
    Eigen::Map<Vec3>(&y(j + 1, 18)) = q + ds * qs;
    Eigen::Map<Vec3>(&y(j + 1, 21)) = w + ds * ws;
  }
}

// Residual of the shooting method for Eigen's MINPACK port: tip wrench of the integration minus the requested one.
// Forward difference Jacobian as in MINPACK lmdif, but with the step eps * max(1, |x|) instead of eps * |x|: the base
// wrench of a planar solution has components of 1e-32 that are zero in theory, and a step relative to them makes the
// Jacobian column rounding noise. MINPACK then stops at a point that is not a root, depending on the rounding.
struct DynamicCosseratRod::Shooting {
  using Scalar = double;
  using InputType = VecX;
  using ValueType = VecX;
  using JacobianType = MatX;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };
  const DynamicCosseratRod* rod;
  double c0;
  const RowMat* z_h;
  Vec6 tip_wrench;
  RowMat* y;
  RowMat* z;
  int inputs() const { return 6; }
  int values() const { return 6; }
  int operator()(const VecX& guess, VecX& r) const {
    rod->integrate(guess, c0, *z_h, *y, *z);
    r = y->row(rod->N - 1).segment<6>(12).transpose() - tip_wrench;
    return 0;
  }
  int df(const VecX& guess, MatX& J) const {
    const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
    VecX r0(6), r(6), xh = guess;
    (*this)(guess, r0);
    for (int j = 0; j < 6; ++j) {
      const double h = eps * std::max(1.0, std::abs(guess(j)));
      xh(j) = guess(j) + h;
      (*this)(xh, r);
      J.col(j) = (r - r0) / h;
      xh(j) = guess(j);
    }
    return 0;
  }
};

RodState DynamicCosseratRod::solve(double c0, const RowMat& z_h, const Vec6& tip_wrench) {
  if (!has_initial_conditions_) throw std::logic_error("call set_initial_conditions first");
  Shooting shooting{this, c0, &z_h, tip_wrench, &y, &z};
  VecX x = guess_;
  // MINPACK with the settings of scipy root(method="lm"): ftol = xtol = 1.49e-8, gtol = 0, factor = 100. maxfev counts
  // only the residual evaluations here, 200 allows as many iterations as the 1400 evaluations of lmdif (7 per iteration).
  Eigen::LevenbergMarquardt<Shooting> lm(shooting);
  lm.parameters.maxfev = 200;
  const auto status = lm.minimize(x);
  VecX r(6);
  shooting(x, r);  // also leaves y, z of the solution
  // MINPACK also reports success at a local minimum of |r| (small step), so check that it is a root. The scale
  // covers the rounding error of the Euler sums, which grows with the size of the wrenches.
  const double scale = 1 + std::max(x.lpNorm<Eigen::Infinity>(), tip_wrench.lpNorm<Eigen::Infinity>());
  if (status < 1 || status > 4 || !(r.lpNorm<Eigen::Infinity>() <= 1e-8 * scale))
    throw std::runtime_error("shooting method did not converge, MINPACK status " + std::to_string(status) +
                             ", max residual " + std::to_string(r.lpNorm<Eigen::Infinity>()));
  guess_ = x;
  return {y.leftCols(3), y.middleCols(3, 9), y.middleCols(12, 6)};
}

RodState DynamicCosseratRod::init_static(const Vec6& tip_wrench) {
  guess_.setZero();
  RodState out = solve(0.0, RowMat::Zero(N, 12), tip_wrench);
  z_old_ = z;
  z_h_ = (c1_ + c2_) * z;
  return out;
}

RodState DynamicCosseratRod::step(const Vec6& tip_wrench) {
  if (z_h_.rows() == 0) throw std::logic_error("call init_static first");
  RodState out = solve(c0_, z_h_, tip_wrench);
  z_h_ = c1_ * z + c2_ * z_old_ + d1_ * (c0_ * z + z_h_);
  z_old_ = z;
  return out;
}

}  // namespace pyctr
