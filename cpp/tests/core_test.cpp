// Checks of the shared foundation: integrator against an analytic solution, solver on a small system.
#undef NDEBUG  // Release builds define NDEBUG, the checks must run anyway
#include <cassert>
#include <cmath>
#include <cstdio>

#include "pyctr/integrate.hpp"
#include "pyctr/solve.hpp"

int main() {
  using namespace pyctr;
  // harmonic oscillator y'' = -y, y(0) = 1: y = cos t
  std::vector<double> t;
  for (int i = 0; i <= 20; ++i) t.push_back(0.5 * i);
  const auto y = dopri5([](double, const VecX& v) { return VecX((VecX(2) << v(1), -v(0)).finished()); },
                        (VecX(2) << 1, 0).finished(), t);
  for (size_t i = 0; i < t.size(); ++i) assert(std::abs(y[i](0) - std::cos(t[i])) < 1e-8);

  // x^2 = 2, x y = 1
  const auto r = levenberg_marquardt(
      [](const VecX& x) { return VecX((VecX(2) << x(0) * x(0) - 2, x(0) * x(1) - 1).finished()); },
      (VecX(2) << 1, 1).finished());
  assert(r.success && std::abs(r.x(0) - std::sqrt(2.0)) < 1e-10 && std::abs(r.x(1) - 1 / std::sqrt(2.0)) < 1e-10);

  // x^2 + 1 = 0 has no root: the step becomes small at the minimum x = 0, this is not a success
  const auto none = levenberg_marquardt([](const VecX& x) { return VecX((VecX(1) << x(0) * x(0) + 1).finished()); },
                                        (VecX(1) << 1).finished());
  assert(!none.success && std::abs(none.residual(0) - 1) < 1e-6);

  assert((hat(Vec3(1, 2, 3)) * Vec3(4, 5, 6) - Vec3(1, 2, 3).cross(Vec3(4, 5, 6))).norm() < 1e-15);
  assert((invhat(hat(Vec3(1, 2, 3))) - Vec3(1, 2, 3)).norm() < 1e-15);
  std::puts("core_test passed");
  return 0;
}
