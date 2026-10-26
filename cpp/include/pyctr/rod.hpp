// Static Cosserat rod models, port of pyctr/cosserat_rod.py and pyctr/cosserat_rod_force_along.py.
// The ODE state is [p, R (row-major), n, m] or, for the curvature integration, [p, R, n, u], as in Python.
#pragma once

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pyctr/core.hpp"

namespace pyctr {

// Errors that Python raises with these types, the bindings translate them.
struct NotImplementedError : std::logic_error {
  using std::logic_error::logic_error;
};
struct AttributeError : std::logic_error {
  using std::logic_error::logic_error;
};
struct ZeroDivisionError : std::domain_error {
  using std::domain_error::domain_error;
};

// Output of apply_force: one row per arc length point, R as 9 columns (row-major), wrench = [n, m].
struct RodStates {
  MatX p, R, wrench;
};

class CosseratRod {
 public:
  explicit CosseratRod(const TubeParameters& params = TubeParameters()) : params(params) {}
  virtual ~CosseratRod() = default;

  // A, I, J, Kse and Kbt are always derived from these (TubeParameters methods), Python computes them once
  TubeParameters params;
  double beta = 0.0;  // retracted fraction of L, set by the robot
  double alpha = 0.0;  // rotation, set by the robot
  Vec3 e3 = Vec3::UnitZ();  // Python _e3: third column of R0
  std::map<std::string, MatX> bounding_values;

  void set_initial_conditions(const Vec3& p0, const Mat3& R0);
  void set_bounding_values(const std::vector<std::string>& names, const std::vector<MatX>& values);
  Vec3 get_u_div(const Mat3& R, const Vec3& n, const Vec3& m, const Vec3& u) const;

  virtual VecX cosserate_rod_ode(const VecX& state, double s) const;
  virtual VecX cosserat_rod_ode_curvature(const VecX& state, double s) const;
  virtual Vec3 external_gauss_forces(double /*s*/) const { return Vec3::Zero(); }

  VecX shooting_function(const VecX& guess) const;
  VecX shooting_function_force(const VecX& guess, bool curvature_integration);
  RodStates push_end(const VecX& wrench, bool curvature_integration = false);
  RodStates move_end(const Vec3& p, const Mat3& R);

  // R: (N, 9) row-major rotations, u and m: (N, 3)
  virtual MatX calc_m_from_curvature(const MatX& R, const MatX& u) const;
  MatX calc_u_from_m(const MatX& R, const MatX& m) const;

  virtual RodStates apply_force(const VecX& wrench, int steps, bool curvature_integration);
  // with the default arguments of the Python class of the object
  RodStates apply_force(const VecX& wrench, int steps) {
    return apply_force(wrench, steps, default_curvature_integration());
  }
  RodStates apply_force(const VecX& wrench) { return apply_force(wrench, default_steps()); }

  // 0 straight, -1 behind the tip, nullopt in the curved part (Python returns None there)
  virtual std::optional<int> is_curved_or_at_end(double s) const;
  Vec3 get_kappa() const;
  void set_kappa(double kappa) { cur_kappa_ = kappa; }

 protected:
  virtual int default_steps() const { return 20; }
  virtual bool default_curvature_integration() const { return true; }
  // initial state [p0, R0, n, x] with x = m or u
  VecX initial_state(const Vec3& p0, const Mat3& R0, const Vec3& n, const Vec3& x) const;
  // integrates cosserate_rod_ode (or cosserat_rod_ode_curvature) and returns one state per row
  MatX integrate(const VecX& y0, const std::vector<double>& s, bool curvature) const;

  std::optional<Vec3> p0_;
  Mat3 R0_ = Mat3::Identity();
  std::optional<double> cur_kappa_;
};

class CurvedCosseratRod : public CosseratRod {
 public:
  CurvedCosseratRod() { params.kappa = 0.16; }  // Python: params=None
  explicit CurvedCosseratRod(const TubeParameters& params) : CosseratRod(params) {}

  VecX cosserate_rod_ode(const VecX& state, double s) const override;
  VecX cosserat_rod_ode_curvature(const VecX& state, double s) const override;  // raises NotImplementedError
  MatX calc_m_from_curvature(const MatX& R, const MatX& u) const override;
  using CosseratRod::apply_force;
  // straight part with kappa 0, then curved part, both moment based (curvature_integration raises)
  RodStates apply_force(const VecX& wrench, int steps, bool curvature_integration) override;
  std::optional<int> is_curved_or_at_end(double s) const override;

 protected:
  int default_steps() const override { return 100; }
  bool default_curvature_integration() const override { return false; }
};

enum class Force_Model { GAUSSIAN = 1, FOURIER = 2 };

// f * exp(-c (s - mu)^2) in x and y, Python tuple ((fx, fy), c, mu)
struct Gaussian {
  double fx = 0, fy = 0, c = 0, mu = 0;
};

// Python mixin ExtForceRod: distributed forces along the rod, in the world frame
template <class Rod>
class ExtForceRod : public Rod {
 public:
  explicit ExtForceRod(Force_Model force_model = Force_Model::GAUSSIAN) : force_model(force_model) {}
  ExtForceRod(const TubeParameters& params, Force_Model force_model = Force_Model::GAUSSIAN)
      : Rod(params), force_model(force_model) {}

  Force_Model force_model;
  std::vector<Gaussian> gaussians{Gaussian{}};  // one zero term, as in Python
  std::optional<VecX> ax, ay, bx, by;  // Fourier coefficients, unset as in Python

  Vec3 external_gauss_forces(double s) const override {
    Vec3 f = Vec3::Zero();
    for (const Gaussian& g : gaussians) {
      const double w = std::exp(-g.c * (s - g.mu) * (s - g.mu));
      f(0) += g.fx * w;
      f(1) += g.fy * w;
    }
    return f;
  }

  // bx and by are also sine terms in Python
  Vec3 external_fourier_forces(double s) const {
    if (!ax || !ay || !bx || !by) throw AttributeError("Fourier coefficients _ax, _ay, _bx, _by are not set");
    auto sines = [&](const VecX& a) {
      double f = 0;
      for (int k = 1; k < a.size(); ++k) f += a(k) * std::sin(s * pi * k / this->params.L);
      return f;
    };
    if (ax->size() == 0 || ay->size() == 0) throw std::out_of_range("_ax and _ay need the constant term");
    return Vec3((*ax)(0) + sines(*ax) + sines(*bx), (*ay)(0) + sines(*ay) + sines(*by), 0);
  }

  Vec3 external_forces(double s) const {
    return force_model == Force_Model::GAUSSIAN ? external_gauss_forces(s) : external_fourier_forces(s);
  }

  RodStates apply_external_forces();
  VecX shooting_function_external_force(const VecX& guess, int s_l = 100) {
    return -this->apply_force(guess, s_l).wrench.bottomRows(1).transpose();
  }
};

// Straight rod with forces along its body. Without gravity and with e3 = z in the moment based ODE, as in Python.
class StraightCosseratRod : public ExtForceRod<CosseratRod> {
 public:
  using ExtForceRod<CosseratRod>::ExtForceRod;
  VecX cosserate_rod_ode(const VecX& state, double s) const override;
};

class CurvedCosseratRodExt : public ExtForceRod<CurvedCosseratRod> {
 public:
  using ExtForceRod<CurvedCosseratRod>::ExtForceRod;
  VecX cosserate_rod_ode(const VecX& state, double s) const override;
};

extern template class ExtForceRod<CosseratRod>;
extern template class ExtForceRod<CurvedCosseratRod>;

}  // namespace pyctr
