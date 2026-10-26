// Unloaded precurved rod is a circular arc of radius 1/kappa; push_end meets the tip wrench.
#undef NDEBUG  // Release builds define NDEBUG, the checks must run anyway
#include <cassert>
#include <cmath>
#include <cstdio>

#include "pyctr/rod.hpp"

int main() {
  using namespace pyctr;
  TubeParameters t;
  t.L = 0.1;
  t.kappa = 8.0;
  t.gravity = Vec3::Zero();
  CurvedCosseratRod rod(t);
  rod.set_initial_conditions(Vec3::Zero(), Mat3::Identity());
  const RodStates arc = rod.apply_force(VecX::Zero(6), 400);
  for (int i = 0; i < arc.p.rows(); ++i)
    assert(std::abs((arc.p.row(i).transpose() - Vec3(1 / 8.0, 0, 0)).norm() - 1 / 8.0) < 1e-9);

  CosseratRod straight;
  straight.set_initial_conditions(Vec3::Zero(), Mat3::Identity());
  VecX wrench(6);
  wrench << 0.01, -0.02, 0.003, 1e-4, -2e-4, 5e-5;
  for (bool curvature_integration : {false, true}) {
    const RodStates s = straight.push_end(wrench, curvature_integration);
    assert((s.wrench.bottomRows(1).transpose() - wrench).norm() < 1e-10);
  }
  std::puts("rod_test passed");
  return 0;
}
