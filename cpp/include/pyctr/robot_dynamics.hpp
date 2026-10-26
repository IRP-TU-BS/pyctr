// Dynamic concentric tube robot, port of pyctr.cosserat_rod_dynamics.DynamicConcentricTubeRobot
// (Till et al., "A Dynamic Model for Concentric Tube Robots", IEEE T-RO 36(6), 2020: BDF-alpha in time,
// midpoint rule in space, Newton's method on all nodes at every time step).
#pragma once

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "pyctr/core.hpp"

namespace pyctr {

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
// contact plane of every interval between nodes: normals, points (M - 1, 3), world frame
using Contacts = std::pair<RowMat, RowMat>;
// positions (M, 3), orientations (M, 9) (R row by row), wrenches [n, m^b] (M, 6) at the arc lengths s
using RobotState = std::tuple<RowMat, RowMat, RowMat>;

// Tubes are ordered inner to outer and curved about their y axis (u* = [0, kappa, 0]) after straight_length.
// alphas and betas may change between steps (rotate, translate). New betas give a new grid with the same number of
// nodes (fixed at init_static), see remesh; the axial velocity of the tubes is neglected (paper: beta_t = 0).
// The base plate may move: set p_init, R_init (pose at the end of the step) and base_velocity before each step.
// Tip wrench [F, M] (world frame) acts on the end of the inner tube. Contact: frictionless Hunt-Crossley penalty against one plane per interval.
// y per node: p, R (of tube 1), q, w, n (world), m^b_xy (total, tube 1 frame), m^b_z, theta, gamma (per tube)
// z per interval between nodes: q, w, u_xy (tube 1), u_z, gamma (per tube)
class DynamicConcentricTubeRobot {
 public:
  DynamicConcentricTubeRobot(std::vector<TubeParameters> tubes, double dt = 2e-3, double bdf_alpha = -0.2,
                             double contact_stiffness = 1e6, double contact_dissipation = 0.0,
                             const Mat3& R_init = Mat3::Identity(), const Vec3& p_init = Vec3::Zero());

  // reads the first k alphas, as Python
  void rotate(const VecX& alphas);
  // throws std::invalid_argument if tube 1 does not stick out or an inner tube would end inside an outer tube
  // (1e-12 m tolerance)
  void translate(const VecX& betas);
  // Solve the static BVP for the current alphas and betas and take it as rest state at t = 0
  RobotState init_static(const Vec6& tip_wrench = Vec6::Zero(), const std::optional<Contacts>& contacts = {});
  // Advance one time step dt with the current alphas and betas (calls remesh). Needs a successful init_static.
  RobotState step(const Vec6& tip_wrench = Vec6::Zero(), const std::optional<Contacts>& contacts = {});
  // y, z_old, z_h, alphas_old, alphas_h and the betas of the grid as one vector, the size does not change with betas
  VecX get_state() const;
  // inverse of get_state, also sets the betas (translate) and their grid
  void set_state(const VecX& x);
  // grid for new betas (translate) with the same number of nodes, does nothing if the betas did not change
  void remesh();
  bool initialized() const { return s.size() > 0; }

  // the grid (L, straight_length, s of tube 1, r_outer) is read from tubes in init_static and translate,
  // the material constants are taken in the constructor, as in Python
  std::vector<TubeParameters> tubes;
  const double dt;
  double contact_stiffness;    // N/m^2: force per robot length and per penetration depth
  double contact_dissipation;  // s/m
  Mat3 R_init;
  Vec3 p_init;
  std::pair<Vec3, Vec3> base_velocity{Vec3::Zero(), Vec3::Zero()};  // linear, angular, world frame
  VecX alphas, betas;

  // after init_static
  VecX s;                // arc length of the nodes (M)
  RowMat y;              // (M, 23 + 3k)
  VecX radius;           // outer radius per interval (M - 1)
  RowMat contact_force;  // force on every interval (N, world) of the last solve (M - 1, 3)

 private:
  // intervals < 0: node distance s of tube 1, otherwise that number of intervals distributed over the segments
  void grid(int intervals = -1);
  void ode(int j, const double* y, const double* z_h, double c0, const Contacts* contacts, double* ys, double* z,
           double* work) const;
  void residual(const VecX& x, double c0, const RowMat& z_h, const VecX& alphas_t, const Vec6& tip_wrench,
                const Contacts* contacts, VecX& r);
  void jacobian(const VecX& x, double c0, const RowMat& z_h, const Contacts* contacts);
  VecX newton_step(const VecX& r, const Vec6& tip_wrench);
  RobotState solve(double c0, const RowMat& z_h, const VecX& alphas_t, const Vec6& tip_wrench,
                   const std::optional<Contacts>& contacts);
  int ny() const { return 23 + 3 * k_; }
  int nz() const { return 8 + 2 * k_; }

  int k_;
  VecX EI_, GJ_, BB_, BT_, rhoI_, kappa_;
  double rhoA_;
  Vec3 f_;  // gravity per length
  double c0_, c1_, c2_, d1_;  // BDF-alpha: x_t = c0 * x + x_h

  RowMat active_, kappa_s_;  // per interval and tube: tube present, u*_y
  VecX h_;                   // interval lengths
  std::vector<int> ends_;    // node nearest to the end of tubes 2..k
  std::vector<int> base_idx_;
  VecX betas_;  // betas of the grid

  RowMat ys_, z_;  // ode at the midpoints of the last residual
  RowMat z_old_, z_h_;
  VecX a_old_, a_h_;
  bool history_ = false;  // z_old_, z_h_, a_old_, a_h_ belong to the current grid

  RowMat F_;  // df/dy of every interval ((M - 1) ny, ny)
  RowMat U_;  // pivot rows of newton_step
};

}  // namespace pyctr
