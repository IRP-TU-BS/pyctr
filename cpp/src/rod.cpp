#include "pyctr/rod.hpp"

#include <cmath>
#include <string>

#include "pyctr/integrate.hpp"
#include "pyctr/solve.hpp"

namespace pyctr {

namespace {

using RowMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

Mat3 rotation(const VecX& state) { return Eigen::Map<const RowMat3>(state.data() + 3); }
Mat3 rotation_row(const MatX& R, int i) {
  RowMat3 r;
  for (int k = 0; k < 9; ++k) r.data()[k] = R(i, k);
  return r;
}

// numpy.linspace
std::vector<double> linspace(double a, double b, int n) {
  if (n < 0) throw std::invalid_argument("Number of samples, " + std::to_string(n) + ", must be non-negative.");
  std::vector<double> s(std::max(n, 0));
  for (int i = 0; i < n; ++i) s[i] = n > 1 ? a + i * ((b - a) / (n - 1)) : a;
  if (n > 1) s.back() = b;
  return s;
}

void check_state(const VecX& y) {
  if (y.size() != 18) throw std::invalid_argument("state must have 18 elements [p, R, n, m or u]");
}

// [ps, Rs, ns, ms] of the moment based ODE with u = Kbt^-1 R^T m + u_star and distributed force f (dn/ds = -f)
VecX moment_ode(const VecX& y, const Vec3& e3, const Mat3& Kbt, const Vec3& u_star, const Vec3& f) {
  check_state(y);
  const Mat3 R = rotation(y);
  const Vec3 n = y.segment<3>(12), m = y.segment<3>(15);
  const Vec3 u = (R.transpose() * m).cwiseQuotient(Kbt.diagonal()) + u_star;  // Kbt is diagonal
  const Vec3 ps = R * e3;
  VecX dy(18);
  dy.head<3>() = ps;
  Eigen::Map<RowMat3>(dy.data() + 3) = R * hat(u);
  dy.segment<3>(12) = -f;
  dy.segment<3>(15) = -ps.cross(n);
  return dy;
}

// [ps, Rs, ns, us] of the curvature based ODE
VecX curvature_ode(const VecX& y, const Vec3& e3, const Mat3& Kbt, const Vec3& u_star, const Vec3& f) {
  check_state(y);
  const Mat3 R = rotation(y);
  const Vec3 n = y.segment<3>(12), u = y.segment<3>(15);
  VecX dy(18);
  dy.head<3>() = R * e3;
  Eigen::Map<RowMat3>(dy.data() + 3) = R * hat(u);
  dy.segment<3>(12) = -f;
  dy.segment<3>(15) = -(hat(u) * (Kbt * (u - u_star)) + hat(e3) * R.transpose() * n).cwiseQuotient(Kbt.diagonal());
  return dy;
}

Vec3 gravity_force(const TubeParameters& t) { return t.rho * t.A() * t.gravity; }

void check_wrench(const VecX& wrench) {
  if (wrench.size() != 6) throw std::invalid_argument("wrench must have 6 elements [n, m]");
}

// solve with Levenberg-Marquardt from a zero guess, as the Python least_squares(method="lm") calls
template <class F>
VecX solve_from_zero(F&& f) {
  return levenberg_marquardt(f, VecX::Zero(6), 1e-10, 1e-12).x;
}

}  // namespace

void CosseratRod::set_initial_conditions(const Vec3& p0, const Mat3& R0) {
  p0_ = p0;
  R0_ = R0;
  e3 = R0.col(2);
}

void CosseratRod::set_bounding_values(const std::vector<std::string>& names, const std::vector<MatX>& values) {
  if (values.size() < names.size()) throw std::invalid_argument("set_bounding_values: fewer values than names");
  for (size_t i = 0; i < names.size(); ++i) bounding_values[names[i]] = values[i];
}

Vec3 CosseratRod::get_u_div(const Mat3& R, const Vec3&, const Vec3& m, const Vec3&) const {
  return params.Kbt().inverse() * R.transpose() * m;
}

VecX CosseratRod::cosserate_rod_ode(const VecX& state, double) const {
  return moment_ode(state, e3, params.Kbt(), Vec3::Zero(), gravity_force(params));
}

VecX CosseratRod::cosserat_rod_ode_curvature(const VecX& state, double) const {
  return curvature_ode(state, e3, params.Kbt(), Vec3::Zero(), gravity_force(params));
}

VecX CosseratRod::initial_state(const Vec3& p0, const Mat3& R0, const Vec3& n, const Vec3& x) const {
  VecX y(18);
  y.head<3>() = p0;
  Eigen::Map<RowMat3>(y.data() + 3) = R0;
  y.segment<3>(12) = n;
  y.segment<3>(15) = x;
  return y;
}

MatX CosseratRod::integrate(const VecX& y0, const std::vector<double>& s, bool curvature) const {
  auto f = [&](double t, const VecX& y) { return curvature ? cosserat_rod_ode_curvature(y, t) : cosserate_rod_ode(y, t); };
  const std::vector<VecX> ys = dopri5(f, y0, s);
  MatX out(ys.size(), y0.size());
  for (size_t i = 0; i < ys.size(); ++i) out.row(i) = ys[i].transpose();
  return out;
}

VecX CosseratRod::shooting_function(const VecX& guess) const {
  check_wrench(guess);
  if (!p0_) throw AttributeError("set_initial_conditions has not been called");
  const MatX states = integrate(initial_state(*p0_, R0_, guess.head<3>(), guess.segment<3>(3)), linspace(0, params.L, 100), false);
  const VecX last = states.bottomRows(1).transpose();
  const MatX& pL = bounding_values.at("pL");
  const MatX& RL = bounding_values.at("RL");
  if (pL.size() != 3 || RL.rows() != 3 || RL.cols() != 3) throw std::invalid_argument("pL needs 3 elements, RL 3x3");
  const Mat3 A = RL.transpose() * rotation(last);
  VecX r(6);
  r << pL.reshaped() - last.head<3>(), invhat(A - A.transpose());
  return r;
}

VecX CosseratRod::shooting_function_force(const VecX& guess, bool curvature_integration) {
  const MatX& tip = bounding_values.at("tip_wrench");
  if (tip.size() != 6) throw std::invalid_argument("tip_wrench must have 6 elements");
  const int steps = static_cast<int>(std::ceil(params.L / params.s));
  return tip.reshaped() - apply_force(guess, steps, curvature_integration).wrench.bottomRows(1).transpose();
}

RodStates CosseratRod::push_end(const VecX& wrench, bool curvature_integration) {
  check_wrench(wrench);
  set_bounding_values({"tip_wrench"}, {wrench});
  const VecX x = solve_from_zero([&](const VecX& g) { return shooting_function_force(g, curvature_integration); });
  return apply_force(x, static_cast<int>(std::ceil(params.L / params.s)), curvature_integration);
}

RodStates CosseratRod::move_end(const Vec3& p, const Mat3& R) {
  set_bounding_values({"pL", "RL"}, {MatX(p.transpose()), MatX(R)});
  return apply_force(solve_from_zero([&](const VecX& g) { return shooting_function(g); }));
}

MatX CosseratRod::calc_m_from_curvature(const MatX& R, const MatX& u) const {
  MatX m(R.rows(), 3);
  for (int i = 0; i < R.rows(); ++i) m.row(i) = (rotation_row(R, i) * params.Kbt() * u.row(i).transpose()).transpose();
  return m;
}

MatX CosseratRod::calc_u_from_m(const MatX& R, const MatX& m) const {
  MatX u(R.rows(), 3);
  const Mat3 Kbt_inv = params.Kbt().inverse();
  for (int i = 0; i < R.rows(); ++i) u.row(i) = (Kbt_inv * rotation_row(R, i).transpose() * m.row(i).transpose()).transpose();
  return u;
}

RodStates CosseratRod::apply_force(const VecX& wrench, int steps, bool curvature_integration) {
  check_wrench(wrench);
  if (!p0_) throw AttributeError("set_initial_conditions has not been called");
  const Vec3 n0 = wrench.head<3>(), m0 = wrench.tail<3>();
  const Vec3 x0 = curvature_integration ? Vec3(params.Kbt().inverse() * R0_.transpose() * m0) : m0;
  const MatX states = integrate(initial_state(*p0_, R0_, n0, x0), linspace(0, params.L, steps), curvature_integration);
  RodStates out{states.leftCols(3), states.middleCols(3, 9), states.rightCols(6)};
  if (curvature_integration) out.wrench.rightCols(3) = calc_m_from_curvature(out.R, states.rightCols(3));
  return out;
}

std::optional<int> CosseratRod::is_curved_or_at_end(double s) const {
  if (s <= params.straight_length - params.L * beta) return 0;
  if (s > params.L - params.L * beta) return -1;
  return std::nullopt;
}

Vec3 CosseratRod::get_kappa() const {
  if (!cur_kappa_) throw AttributeError("cur_kappa is not set, call set_kappa or apply_force first");
  return Vec3(0, *cur_kappa_, 0);
}

VecX CurvedCosseratRod::cosserate_rod_ode(const VecX& state, double) const {
  return moment_ode(state, e3, params.Kbt(), get_kappa(), gravity_force(params));
}

// Python returns 24 values here ((3, 1) u minus (3,) kappa broadcasts to 3x3), so apply_force refuses it as well
VecX CurvedCosseratRod::cosserat_rod_ode_curvature(const VecX&, double) const {
  throw NotImplementedError("curvature integration is not implemented for CurvedCosseratRod");
}

MatX CurvedCosseratRod::calc_m_from_curvature(const MatX& R, const MatX& u) const {
  MatX m(R.rows(), 3);
  const Vec3 kappa = get_kappa();
  for (int i = 0; i < R.rows(); ++i)
    m.row(i) = (rotation_row(R, i) * params.Kbt() * (u.row(i).transpose() - kappa)).transpose();
  return m;
}

RodStates CurvedCosseratRod::apply_force(const VecX& wrench, int steps, bool curvature_integration) {
  if (curvature_integration)
    throw NotImplementedError(
        "curvature integration is not implemented for CurvedCosseratRod; "
        "use the moment-based integration (curvature_integration=False).");
  check_wrench(wrench);
  if (!p0_) throw AttributeError("set_initial_conditions has not been called");
  if (steps == 0) throw ZeroDivisionError("float division by zero");  // Python: _step_size = L_eff / steps
  // retracting by beta shortens the exposed length and moves the straight/curved boundary by L * beta
  const double L = params.L;
  const double L_eff = L - L * beta;
  const double straight_eff = std::max(params.straight_length - L * beta, 0.0);
  const int steps_straight = L_eff > 0 ? static_cast<int>(steps * straight_eff / L_eff) : 0;
  const int steps_curved = steps - steps_straight;

  VecX y0 = initial_state(*p0_, R0_, wrench.head<3>(), wrench.tail<3>());
  MatX straight(0, 18);
  double start = 0.0;
  if (steps_straight > 0) {
    set_kappa(0);
    straight = integrate(y0, linspace(0, straight_eff, steps_straight), false);
    y0 = straight.bottomRows(1).transpose();
    start = straight_eff;
  }
  set_kappa(params.kappa);
  const MatX curved = integrate(y0, linspace(start, L_eff, steps_curved), false);
  // the first curved row repeats the last straight row
  const int skip = steps_straight > 0 ? std::min<int>(1, curved.rows()) : 0;
  MatX states(straight.rows() + curved.rows() - skip, 18);
  states << straight, curved.bottomRows(curved.rows() - skip);
  return RodStates{states.leftCols(3), states.middleCols(3, 9), states.rightCols(6)};
}

std::optional<int> CurvedCosseratRod::is_curved_or_at_end(double s) const {
  const auto r = CosseratRod::is_curved_or_at_end(s);
  return r ? *r : 1;
}

template <class Rod>
RodStates ExtForceRod<Rod>::apply_external_forces() {
  return this->apply_force(solve_from_zero([&](const VecX& g) { return shooting_function_external_force(g); }));
}

template class ExtForceRod<CosseratRod>;
template class ExtForceRod<CurvedCosseratRod>;

VecX StraightCosseratRod::cosserate_rod_ode(const VecX& state, double s) const {
  return moment_ode(state, Vec3::UnitZ(), params.Kbt(), Vec3::Zero(), external_forces(s));
}

VecX CurvedCosseratRodExt::cosserate_rod_ode(const VecX& state, double s) const {
  return moment_ode(state, e3, params.Kbt(), get_kappa(), gravity_force(params) + external_forces(s));
}

}  // namespace pyctr
