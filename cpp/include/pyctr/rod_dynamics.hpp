// Dynamic Cosserat rod, port of pyctr.cosserat_rod_dynamics.DynamicCosseratRod
// (Till's 02_Dynamics/00_Cosserat_Rod_PDE: BDF-alpha in time, Euler in space, shooting method at every time step).
#pragma once

#include <tuple>

#include "pyctr/core.hpp"

namespace pyctr {

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
// positions (N, 3), orientations (N, 9) (R row by row), wrenches [n, m] (N, 6), all in the world frame
using RodState = std::tuple<RowMat, RowMat, RowMat>;

// Rod with clamped base (set_initial_conditions) and a tip wrench [n, m] (world frame) as boundary condition.
// Material damping Bse, Bbt and square law drag C come from the TubeParameters.
class DynamicCosseratRod {
 public:
  explicit DynamicCosseratRod(const TubeParameters& params, double dt = 2e-3, double bdf_alpha = -0.2);

  void set_initial_conditions(const Vec3& p0, const Mat3& R0);
  // Solve the static BVP and take it as rest state at t = 0
  RodState init_static(const Vec6& tip_wrench = Vec6::Zero());
  // Advance one time step dt. Call init_static first. Velocities q, w are in y.rightCols(6).
  RodState step(const Vec6& tip_wrench = Vec6::Zero());

  TubeParameters params;
  int N;  // number of nodes
  double dt;
  RowMat y;  // (N, 24): p, R, n, m, q, w of the last solution
  RowMat z;  // (N, 12): q, w, v, u of the last solution

 private:
  void integrate(const Vec6& guess, double c0, const RowMat& z_h, RowMat& y, RowMat& z) const;
  RodState solve(double c0, const RowMat& z_h, const Vec6& tip_wrench);
  struct Shooting;

  Mat3 rhoJ_;
  double c0_, c1_, c2_, d1_;  // BDF-alpha: x_t = c0 * x + x_h
  Vec3 p0_;
  Mat3 R0_;
  bool has_initial_conditions_ = false;
  Vec6 guess_ = Vec6::Zero();  // [n0, m0] of the last solution, warm start
  RowMat z_old_, z_h_;
};

}  // namespace pyctr
