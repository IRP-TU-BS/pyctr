// Shared types of the C++ port of pyctr: vectors, rotations and the parameters of one tube.
#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace pyctr {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using VecX = Eigen::VectorXd;
using MatX = Eigen::MatrixXd;

constexpr double pi = 3.14159265358979323846;

// hat(u) v = u x v
inline Mat3 hat(const Vec3& u) {
  Mat3 m;
  m << 0, -u(2), u(1), u(2), 0, -u(0), -u(1), u(0), 0;
  return m;
}

// inverse of hat for the skew-symmetric part [m(2,1), m(0,2), m(1,0)]
inline Vec3 invhat(const Mat3& m) { return Vec3(m(2, 1), m(0, 2), m(1, 0)); }

inline Mat3 Rz(double theta) {
  Mat3 m;
  m << std::cos(theta), -std::sin(theta), 0, std::sin(theta), std::cos(theta), 0, 0, 0, 1;
  return m;
}

// Parameters of one tube, SI units. Defaults are those of pyctr.cosserat_rod.CosseratRod.
struct TubeParameters {
  double E = 339889091.1107947;  // Young's modulus
  double G = 97898219.42862949;  // shear modulus
  double r_inner = 0.0;
  double r_outer = 1.4e-3;
  double rho = 8000;  // density
  Vec3 gravity = Vec3(0, 0, -9.81);  // gravity vector in the world frame (pyctr: params["g"] = [0, 0, -g])
  double L = 0.15;  // length
  double straight_length = 0.0;  // the tube is straight from 0 to straight_length, curved after it
  double kappa = 0.0;  // precurvature of the curved part, u* = [0, kappa, 0]
  double s = 1e-3;  // distance between nodes along the tube
  Mat3 Bse = Mat3::Zero();  // material damping, shear and extension (dynamic rod)
  Mat3 Bbt = Mat3::Zero();  // material damping, bending and torsion (dynamic models)
  Mat3 C = Mat3::Zero();  // square law drag (dynamic rod)

  double A() const { return pi * (r_outer * r_outer - r_inner * r_inner); }
  double I() const { return pi / 4 * (std::pow(r_outer, 4) - std::pow(r_inner, 4)); }
  double J() const { return pi / 2 * (std::pow(r_outer, 4) - std::pow(r_inner, 4)); }  // polar moment
  Mat3 Kse() const { return Vec3(G * A(), G * A(), E * A()).asDiagonal(); }
  Mat3 Kbt() const { return Vec3(E * I(), E * I(), G * J()).asDiagonal(); }
};

}  // namespace pyctr
