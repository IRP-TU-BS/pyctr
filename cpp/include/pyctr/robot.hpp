// Static concentric tube continuum robot, port of pyctr.robots.ConcentricTubeContinuumRobot.
#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pyctr/core.hpp"

namespace pyctr {

using Vec6 = Eigen::Matrix<double, 6, 1>;

// Distributed force (fx, fy, 0) * exp(-a (s - mu)^2) along the robot (Python: ((fx, fy), a, mu)).
struct Gaussian {
  double fx = 0, fy = 0, a = 0, mu = 0;
};

// A boundary value that is not set (Python: KeyError of the boundary_values dict)
struct MissingBoundaryValue : std::out_of_range {
  using std::out_of_range::out_of_range;
};

// The module-level shooting functions of pyctr.robots
enum class ShootingFunction { force, gaussian_forces, tip_pose, tip_position };

struct ForwardResult {
  MatX state;  // (N, 18 + 2 n): p, R (row-major), n, m, u_z of each tube, theta of each tube
  MatX us;  // (N, 3) curvature of tube 1
  std::vector<std::pair<int, int>> seg_indexes;  // (tubes in segment, cumulative row count), without the leading 0
};

struct StaticResult {
  MatX positions, orientations, wrenches, us, uzs, thetas;
};

class ConcentricTubeContinuumRobot {
 public:
  // tubes sorted from inner to outer, longest to shortest
  explicit ConcentricTubeContinuumRobot(std::vector<TubeParameters> tubes, const Mat3& R_init = Mat3::Identity(),
                                        const Vec3& p_init = Vec3::Zero());

  std::vector<TubeParameters> tubes;
  Mat3 R_init;
  Vec3 p_init;
  VecX alphas;  // rotation of each tube
  VecX betas;  // retraction of each tube as fraction of its length
  std::map<std::string, MatX> boundary_values;
  double step_len = 0;

  // state of the last fwd_static
  MatX positions, orientations, wrenches, uzs, thetas;
  std::vector<std::pair<int, int>> seg_indexes;

  static Vec3 get_curvature_vector(double kappa) { return Vec3(0, kappa, 0); }
  // CurvedCosseratRod.is_curved_or_at_end: 0 straight, 1 curved, -1 past the tube end
  int is_curved_or_at_end(size_t tube, double s) const;
  std::vector<double> get_ordered_segments(bool only_positive = true) const;
  std::vector<int> tube_tip_indices() const;  // -1 for a tube without a segment
  void rotate(const VecX& alphas);
  void translate(const VecX& betas);

  ForwardResult calc_forward(const Vec6& wrench, double step_len, const VecX& uzs0 = VecX());
  MatX calc_us(const std::vector<double>& s, const MatX& thetas_batch, const MatX& m_batch, const VecX& u1z_batch) const;
  Vec3 calc_uxz(const VecX& thetas, const Vec3& m, double s, double u1z) const;
  VecX cosserate_rod_ode(double s, const VecX& state) const;
  Vec3 external_gauss_forces(double s) const;

  void set_boundary_condition(const std::vector<std::string>& names, const std::vector<MatX>& values);
  void remove_boundary_condition(const std::string& name);
  void remove_all_boundary_condition() { boundary_values.clear(); }

  StaticResult fwd_static(const Vec6& wrench, double step_size = 0.01, const VecX& uzs0 = VecX());
  StaticResult fwd_kinematic(double step_size = 0.01) { return fwd_static(Vec6::Zero(), step_size); }
  StaticResult fwd_static_with_boundarys(const VecX& init_wrench, ShootingFunction shooting_function,
                                         double step_size = 0.01);
  StaticResult push_end(const Vec6& wrench, double step_size = 0.01);
  StaticResult push_end_to_position(const Vec3& pos, double step_size = 0.01);
  StaticResult fwd_external_gaussian_forces(double step_size = 0.001);
  void set_gaussians(std::vector<Gaussian> gaussians) { gaussians_ = std::move(gaussians); }
  void remove_gaussians() { gaussians_.clear(); }
  std::vector<double> get_tube_outer_radii() const;

 private:
  bool check_beta_validity(const VecX& betas) const;
  std::vector<size_t> curr_calc_tubes_;  // tubes of the segment that is integrated
  std::vector<Gaussian> gaussians_;
};

VecX shooting_function_force(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size);
VecX shooting_function_gaussian_forces(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size);
VecX shooting_function_tip_pose(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size);
VecX shooting_function_tip_position(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size);
VecX shooting_function(ShootingFunction f, const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size);

}  // namespace pyctr
