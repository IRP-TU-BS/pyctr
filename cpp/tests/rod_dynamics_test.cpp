// Steel cantilever of tests/test_dynamics.py against beam theory: static tip deflection and first bending mode.
#undef NDEBUG  // Release builds define NDEBUG, the checks must run anyway
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "pyctr/rod_dynamics.hpp"

int main() {
  using namespace pyctr;
  TubeParameters t;
  t.E = 200e9;
  t.G = 80e9;
  t.r_outer = 1e-3;
  t.gravity = Vec3::Zero();
  t.L = 0.5;
  t.s = 0.0125;
  DynamicCosseratRod rod(t, 2e-3, -0.2);
  rod.set_initial_conditions(Vec3::Zero(), Mat3::Identity());
  const double EI = t.Kbt()(0, 0), rhoA = t.rho * t.A(), L = 0.5, F = 0.02;

  Vec6 tip = Vec6::Zero();
  tip(0) = F;
  const RowMat p = std::get<0>(rod.init_static(tip));
  const double deflection = F * L * L * L / (3 * EI);
  assert(std::abs(p(rod.N - 1, 0) - deflection) < 0.03 * deflection);

  // upward zero crossings of the tip after releasing the load
  std::vector<double> crossings;
  double x_old = p(rod.N - 1, 0);
  for (int i = 0; i < 200; ++i) {
    const double x = std::get<0>(rod.step())(rod.N - 1, 0);
    if (x_old < 0 && x >= 0) crossings.push_back(i * rod.dt);
    x_old = x;
  }
  assert(crossings.size() >= 2);
  const double period = (crossings.back() - crossings.front()) / (crossings.size() - 1);
  const double expected = 2 * pi / (1.8751 * 1.8751) * std::sqrt(rhoA * L * L * L * L / EI);
  assert(std::abs(period - expected) < 0.03 * expected);
  std::printf("rod_dynamics_test passed: deflection %.4g (beam %.4g), period %.4g s (beam %.4g s)\n",
              p(rod.N - 1, 0), deflection, period, expected);
  return 0;
}
