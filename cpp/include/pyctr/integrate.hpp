// Adaptive Dormand-Prince 5(4) integration with outputs at given points, used by the static models
// (the Python implementation uses scipy odeint / solve_ivp with LSODA).
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "pyctr/core.hpp"

namespace pyctr {

// Integrates dy/dt = f(t, y) from t[0] with y(t[0]) = y0 and returns y at every t (increasing).
// The steps end exactly at every output point, so no interpolation is needed.
template <class F>
std::vector<VecX> dopri5(F&& f, const VecX& y0, const std::vector<double>& t, double rtol = 1e-10,
                         double atol = 1e-12) {
  static const double c2 = 1. / 5, c3 = 3. / 10, c4 = 4. / 5, c5 = 8. / 9;
  static const double a21 = 1. / 5, a31 = 3. / 40, a32 = 9. / 40, a41 = 44. / 45, a42 = -56. / 15, a43 = 32. / 9;
  static const double a51 = 19372. / 6561, a52 = -25360. / 2187, a53 = 64448. / 6561, a54 = -212. / 729;
  static const double a61 = 9017. / 3168, a62 = -355. / 33, a63 = 46732. / 5247, a64 = 49. / 176,
                      a65 = -5103. / 18656;
  static const double b1 = 35. / 384, b3 = 500. / 1113, b4 = 125. / 192, b5 = -2187. / 6784, b6 = 11. / 84;
  // error coefficients: b - b* (b* of the embedded 4th order method)
  static const double e1 = 71. / 57600, e3 = -71. / 16695, e4 = 71. / 1920, e5 = -17253. / 339200,
                      e6 = 22. / 525, e7 = -1. / 40;

  std::vector<VecX> out;
  out.reserve(t.size());
  if (t.empty()) return out;
  VecX y = y0;
  out.push_back(y);
  double h = 0;  // 0: choose the first step from the interval length
  VecX k1 = f(t[0], y);
  for (size_t i = 1; i < t.size(); ++i) {
    double tc = t[i - 1];
    const double tend = t[i];
    if (tend < tc) throw std::invalid_argument("dopri5: output points must be increasing");
    if (h <= 0) h = std::max((tend - tc) * 0.1, 1e-12);
    int iterations = 0;
    while (tc < tend) {
      if (++iterations > 1000000) throw std::runtime_error("dopri5: too many steps");
      const bool last = tc + h >= tend;
      const double hs = last ? tend - tc : h;
      const VecX k2 = f(tc + c2 * hs, y + hs * a21 * k1);
      const VecX k3 = f(tc + c3 * hs, y + hs * (a31 * k1 + a32 * k2));
      const VecX k4 = f(tc + c4 * hs, y + hs * (a41 * k1 + a42 * k2 + a43 * k3));
      const VecX k5 = f(tc + c5 * hs, y + hs * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
      const VecX k6 = f(tc + hs, y + hs * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));
      const VecX ynew = y + hs * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);
      const VecX k7 = f(tc + hs, ynew);
      const VecX err = hs * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7);
      const VecX scale = (atol + rtol * y.cwiseAbs().cwiseMax(ynew.cwiseAbs()).array()).matrix();
      const double e = std::sqrt((err.array() / scale.array()).square().mean());
      if (e <= 1.0 || hs < 1e-14) {  // accept
        tc = last ? tend : tc + hs;
        y = ynew;
        k1 = k7;
      }
      const double factor = e == 0 ? 5.0 : std::clamp(0.9 * std::pow(e, -0.2), 0.2, 5.0);
      if (!(last && e <= 1.0)) h = hs * factor;  // keep the step size of the last full step otherwise
    }
    out.push_back(y);
  }
  return out;
}

}  // namespace pyctr
