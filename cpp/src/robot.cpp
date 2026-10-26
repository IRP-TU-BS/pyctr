#include "pyctr/robot.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "pyctr/integrate.hpp"

namespace pyctr {

namespace {

using RowMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

// Python: solve_ivp(method="LSODA", rtol=1e-8, atol=1e-10). Dormand-Prince with the same tolerances has a
// 10 to 100 times larger error than LSODA here, with these it is below the error of LSODA.
constexpr double ode_rtol = 1e-10;
constexpr double ode_atol = 1e-12;

// numpy.linspace(start, stop, num)
std::vector<double> linspace(double start, double stop, int num) {
  std::vector<double> t(std::max(num, 0));
  if (num == 1) t[0] = start;
  if (num < 2) return t;
  const double step = (stop - start) / (num - 1);
  for (int i = 0; i < num; ++i) t[i] = i * step + start;
  t.back() = stop;
  return t;
}

const MatX& boundary(const ConcentricTubeContinuumRobot& ctr, const std::string& name, Eigen::Index size) {
  const auto it = ctr.boundary_values.find(name);
  if (it == ctr.boundary_values.end()) throw MissingBoundaryValue("boundary value '" + name + "' is not set");
  if (it->second.size() != size)
    throw std::invalid_argument("boundary value '" + name + "' must have " + std::to_string(size) + " entries");
  return it->second;
}

// initial wrench [n0, m0] of a shooting guess
Vec6 guess_wrench(const VecX& guess) {
  if (guess.size() < 6) throw std::invalid_argument("the shooting guess needs at least the 6 entries of the wrench");
  return guess.head<6>();
}

// tip force error, world-frame tip moment error and (with_torsion) u_iz(L_i) of every tube
VecX wrench_residual(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size, const Vec6& tip_wrench,
                     bool with_torsion) {
  const VecX uzs0 = guess.size() > 6 ? VecX(guess.tail(guess.size() - 6)) : VecX();
  const StaticResult r = ctr.fwd_static(guess_wrench(guess), step_size, uzs0);
  const Eigen::Index last = r.wrenches.rows() - 1;
  const Vec3 n_tip = r.wrenches.row(last).head<3>().transpose();
  const Vec3 m_body_tip = r.wrenches.row(last).tail<3>().transpose();
  const Mat3 R_tip = Eigen::Map<const RowMat3>(r.orientations.row(last).eval().data());
  const size_t n = ctr.tubes.size();
  VecX res(6 + (with_torsion ? n : 0));
  res.head<3>() = tip_wrench.head<3>() - n_tip;
  res.segment<3>(3) = tip_wrench.tail<3>() - R_tip * m_body_tip;
  if (with_torsion) {
    const std::vector<int> tips = ctr.tube_tip_indices();
    // A tube without a segment (fully retracted) keeps u_iz(0); Python fails there.
    for (size_t i = 0; i < n; ++i) res(6 + i) = r.uzs(std::max(tips[i], 0), i);
  }
  return res;
}

// MINPACK dogleg: minimizes |R x - qtb| inside |diag x| <= delta
VecX dogleg(const MatX& R, const VecX& diag, const VecX& qtb, double delta) {
  const Eigen::Index n = R.rows();
  const double epsmch = std::numeric_limits<double>::epsilon();
  VecX x(n);
  for (Eigen::Index j = n - 1; j >= 0; --j) {  // Gauss-Newton direction, a zero pivot is replaced as in MINPACK
    double temp = R(j, j);
    if (temp == 0) {
      temp = epsmch * R.col(j).head(j + 1).cwiseAbs().maxCoeff();
      if (temp == 0) temp = epsmch;
    }
    x(j) = (qtb(j) - R.row(j).tail(n - j - 1).dot(x.tail(n - j - 1))) / temp;
  }
  const double qnorm = (diag.array() * x.array()).matrix().norm();
  if (qnorm <= delta) return x;
  // scaled gradient direction
  VecX g = (R.transpose() * qtb).array() / diag.array();
  const double gnorm = g.norm();
  double sgnorm = 0, alpha = delta / qnorm;
  if (gnorm != 0) {
    g = (g.array() / gnorm / diag.array()).matrix();
    const double temp = (R * g).norm();
    sgnorm = gnorm / temp / temp;
    alpha = 0;
    if (sgnorm < delta) {  // point on the dogleg at which the quadratic is minimized
      const double bnorm = qtb.norm(), dq = delta / qnorm, sd = sgnorm / delta;
      double t = bnorm / gnorm * (bnorm / qnorm) * sd;
      t = t - dq * sd * sd + std::sqrt((t - dq) * (t - dq) + (1 - dq * dq) * (1 - sd * sd));
      alpha = dq * (1 - sd * sd) / t;
    }
  }
  return (1 - alpha) * std::min(sgnorm, delta) * g + alpha * x;
}

// Powell hybrid method of MINPACK hybrd (scipy.optimize.root method "hybr") with the default options of
// scipy: forward difference Jacobian, diag from the column norms (mode 1), factor 100. The Broyden
// update is applied to J itself and Q R is recomputed, instead of the QR update of MINPACK (same result
// up to rounding, n is small).
template <class F>
VecX hybrd(F&& f, VecX x, double xtol = 1.49012e-8) {
  const Eigen::Index n = x.size();
  const double epsmch = std::numeric_limits<double>::epsilon();
  const double eps = std::sqrt(epsmch);
  const int maxfev = 200 * static_cast<int>(n + 1);
  VecX fvec = f(x);
  if (fvec.size() != n) throw std::invalid_argument("hybr needs as many residuals as unknowns");
  int nfev = 1;
  double fnorm = fvec.norm();
  VecX diag;
  double delta = 0;
  bool first = true;
  int nslow1 = 0, nslow2 = 0;
  int ncsuc = 0, ncfail = 0;  // not reset with a new Jacobian, as in MINPACK
  MatX J(n, n);
  while (true) {
    for (Eigen::Index j = 0; j < n; ++j) {  // fdjac1
      double h = eps * std::abs(x(j));
      if (h == 0) h = eps;
      VecX xh = x;
      xh(j) += h;
      J.col(j) = (f(xh) - fvec) / h;
    }
    nfev += static_cast<int>(n);
    const VecX acnorm = J.colwise().norm().transpose();
    if (first) {
      diag = acnorm.unaryExpr([](double v) { return v == 0 ? 1.0 : v; });
      const double xnorm = (diag.array() * x.array()).matrix().norm();
      delta = xnorm == 0 ? 100.0 : 100.0 * xnorm;
    }
    diag = diag.cwiseMax(acnorm);
    bool jeval = true;
    while (true) {
      Eigen::HouseholderQR<MatX> qr(J);
      const MatX R = qr.matrixQR().triangularView<Eigen::Upper>();
      const VecX qtf = qr.householderQ().transpose() * fvec;
      const VecX p = -dogleg(R, diag, qtf, delta);
      const VecX xnew = x + p;
      const double pnorm = (diag.array() * p.array()).matrix().norm();
      if (first) delta = std::min(delta, pnorm);
      const VecX fnew = f(xnew);
      ++nfev;
      const double fnorm1 = fnew.norm();
      const double actred = fnorm1 < fnorm ? 1 - (fnorm1 / fnorm) * (fnorm1 / fnorm) : -1;
      const double prenorm = (qtf + R * p).norm();
      const double prered = prenorm < fnorm ? 1 - (prenorm / fnorm) * (prenorm / fnorm) : 0;
      const double ratio = prered > 0 ? actred / prered : 0;
      if (ratio < 0.1) {
        ncsuc = 0;
        ++ncfail;
        delta *= 0.5;
      } else {
        ncfail = 0;
        ++ncsuc;
        if (ratio >= 0.5 || ncsuc > 1) delta = std::max(delta, pnorm / 0.5);
        if (std::abs(ratio - 1) <= 0.1) delta = pnorm / 0.5;
      }
      const VecX fold = fvec;
      if (ratio >= 1e-4) {
        x = xnew;
        fvec = fnew;
        fnorm = fnorm1;
        first = false;
      }
      ++nslow1;
      if (actred >= 0.001) nslow1 = 0;
      if (jeval) ++nslow2;
      if (actred >= 0.1) nslow2 = 0;
      const double xnorm = (diag.array() * x.array()).matrix().norm();
      if (delta <= xtol * xnorm || fnorm == 0 || nfev >= maxfev || 0.1 * std::max(0.1 * delta, pnorm) <= epsmch * xnorm ||
          nslow2 == 5 || nslow1 == 10)
        return x;
      if (ncfail == 2) break;
      // Broyden rank-1 update with the scaled step D^2 p / |D p|^2, also after a rejected step
      J += (fnew - fold - J * p) * (diag.array().square() * p.array()).matrix().transpose() / (pnorm * pnorm);
      jeval = false;
    }
  }
}

}  // namespace

ConcentricTubeContinuumRobot::ConcentricTubeContinuumRobot(std::vector<TubeParameters> tubes_, const Mat3& R_init_,
                                                           const Vec3& p_init_)
    : tubes(std::move(tubes_)),
      R_init(R_init_),
      p_init(p_init_),
      alphas(VecX::Zero(tubes.size())),
      betas(VecX::Zero(tubes.size())) {
  if (tubes.empty()) throw std::invalid_argument("ConcentricTubeContinuumRobot needs at least one tube");
}

int ConcentricTubeContinuumRobot::is_curved_or_at_end(size_t tube, double s) const {
  const TubeParameters& t = tubes[tube];
  if (s <= t.straight_length - t.L * betas(tube)) return 0;
  if (s > t.L - t.L * betas(tube)) return -1;
  return 1;
}

std::vector<double> ConcentricTubeContinuumRobot::get_ordered_segments(bool only_positive) const {
  std::vector<double> ends{0.0};
  for (size_t i = 0; i < tubes.size(); ++i) {
    ends.push_back(tubes[i].straight_length - tubes[i].L * betas(i));
    ends.push_back(tubes[i].L - tubes[i].L * betas(i));
  }
  std::sort(ends.begin(), ends.end());
  ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
  if (only_positive) ends.erase(ends.begin(), std::lower_bound(ends.begin(), ends.end(), 0.0));
  return ends;
}

std::vector<int> ConcentricTubeContinuumRobot::tube_tip_indices() const {
  std::vector<int> tips(tubes.size(), -1);
  for (size_t i = 0; i < tubes.size(); ++i)
    for (const auto& [count, end] : seg_indexes)
      if (count >= static_cast<int>(i) + 1) tips[i] = end - 1;
  return tips;
}

void ConcentricTubeContinuumRobot::rotate(const VecX& alphas_) {
  if (alphas_.size() != static_cast<Eigen::Index>(tubes.size()))
    throw std::invalid_argument("rotate: one alpha per tube expected");
  alphas = alphas_;
}

void ConcentricTubeContinuumRobot::translate(const VecX& betas_) {
  if (betas_.size() != static_cast<Eigen::Index>(tubes.size()))
    throw std::invalid_argument("translate: one beta per tube expected");
  if (!check_beta_validity(betas_))
    throw std::invalid_argument("Parameter Error: The beta values do not correspond to the specifications!");
  betas = betas_;
}

// the inner tube must stick out; 1e-12 m tolerance, so that tube ends aligned in exact arithmetic pass
bool ConcentricTubeContinuumRobot::check_beta_validity(const VecX& b) const {
  if (!(tubes[0].L * (1 - b(0)) > 0)) return false;
  for (Eigen::Index i = 1; i < b.size(); ++i)
    if (!(tubes[i].L - b(i) * tubes[i].L <= tubes[i - 1].L - b(i - 1) * tubes[i - 1].L + 1e-12)) return false;
  return true;
}

ForwardResult ConcentricTubeContinuumRobot::calc_forward(const Vec6& wrench, double step_len_, const VecX& uzs0) {
  const size_t n = tubes.size();
  if (uzs0.size() != 0 && uzs0.size() != static_cast<Eigen::Index>(n))
    throw std::invalid_argument("calc_forward: one initial u_z per tube expected");
  if (!(step_len_ > 0) || !std::isfinite(step_len_)) throw std::invalid_argument("calc_forward: step_len must be positive");
  // Tube 1 defines the reference frame: alpha_1 rotates the whole robot, theta_i(0) = alpha_i - alpha_1.
  const double alpha_1 = alphas(0);
  step_len = step_len_;

  VecX y(18 + 2 * n);
  y.head<3>() = p_init;
  Eigen::Map<RowMat3>(y.data() + 3) = R_init * Rz(alpha_1);
  y.segment<6>(12) = wrench;
  y.segment(18, n) = VecX::Zero(n);
  if (uzs0.size()) y.segment(18, n) = uzs0;
  y.tail(n) = (alphas.array() - alpha_1).matrix();

  const std::vector<double> segments = get_ordered_segments();
  if (segments.size() < 2) throw std::invalid_argument("calc_forward: no segment of positive length");
  std::vector<MatX> states, us;
  ForwardResult result;
  int current_seg_point = 0;
  for (size_t i = 1; i < segments.size(); ++i) {
    curr_calc_tubes_.clear();
    for (size_t t = 0; t < n; ++t)
      if (is_curved_or_at_end(t, segments[i]) >= 0) curr_calc_tubes_.push_back(t);

    const double steps = std::ceil((segments[i] - segments[i - 1]) / step_len);
    if (steps > std::numeric_limits<int>::max()) throw std::invalid_argument("calc_forward: step_len is too small");
    const std::vector<double> s = linspace(segments[i - 1], segments[i], static_cast<int>(steps));
    // With a single output point (segment shorter than one step) the state is not advanced, as in Python.
    const std::vector<VecX> ys =
        dopri5([this](double si, const VecX& yi) { return cosserate_rod_ode(si, yi); }, y, s, ode_rtol, ode_atol);
    MatX block(ys.size(), y.size());
    for (size_t r = 0; r < ys.size(); ++r) block.row(r) = ys[r].transpose();
    y = ys.back();
    us.push_back(calc_us(s, block.rightCols(n), block.middleCols<3>(15), block.col(18)));
    states.push_back(std::move(block));
    current_seg_point += static_cast<int>(ys.size());
    result.seg_indexes.emplace_back(static_cast<int>(curr_calc_tubes_.size()), current_seg_point);
  }

  result.state.resize(current_seg_point, y.size());
  result.us.resize(current_seg_point, 3);
  for (size_t i = 0, row = 0; i < states.size(); row += states[i].rows(), ++i) {
    result.state.middleRows(row, states[i].rows()) = states[i];
    result.us.middleRows(row, us[i].rows()) = us[i];
  }
  return result;
}

MatX ConcentricTubeContinuumRobot::calc_us(const std::vector<double>& s, const MatX& thetas_batch,
                                           const MatX& m_batch, const VecX& u1z_batch) const {
  const Eigen::Index rows = static_cast<Eigen::Index>(s.size());
  if (thetas_batch.rows() != rows || m_batch.rows() != rows || m_batch.cols() != 3 || u1z_batch.size() != rows)
    throw std::invalid_argument("calc_us: one row of thetas, m (3 columns) and u_1z per s expected");
  MatX u(rows, 3);
  for (size_t i = 0; i < s.size(); ++i)
    u.row(i) = calc_uxz(thetas_batch.row(i).transpose(), m_batch.row(i).transpose(), s[i], u1z_batch(i)).transpose();
  return u;
}

Vec3 ConcentricTubeContinuumRobot::calc_uxz(const VecX& thetas, const Vec3& m, double s, double u1z) const {
  // u_1|xy = (sum_j E_j I_j)^-1 (m|xy + sum_j E_j I_j (R_z(theta_j) u*_j)|xy), Rucker 2010 eq. (17)
  if (curr_calc_tubes_.empty()) throw std::invalid_argument("calc_uxz: call calc_forward first");
  if (thetas.size() < static_cast<Eigen::Index>(curr_calc_tubes_.size()))
    throw std::invalid_argument("calc_uxz: one theta per tube of the current segment expected");
  double EIk = 0;
  Vec3 sum = Vec3::Zero();
  for (size_t i = 0; i < curr_calc_tubes_.size(); ++i) {
    const size_t t = curr_calc_tubes_[i];
    const double EIj = tubes[t].Kbt()(0, 0);
    EIk += EIj;
    const double curved_part = std::max(is_curved_or_at_end(t, s), 0);
    sum += EIj * Rz(thetas(i)) * get_curvature_vector(tubes[t].kappa * curved_part);
  }
  Vec3 u1 = (m + sum) / EIk;
  u1(2) = u1z;  // torsion is an integrated state, not the bending average
  return u1;
}

VecX ConcentricTubeContinuumRobot::cosserate_rod_ode(double s, const VecX& state) const {
  const Eigen::Index n = static_cast<Eigen::Index>(tubes.size());
  const Eigen::Index k = static_cast<Eigen::Index>(curr_calc_tubes_.size());
  if (state.size() != 18 + 2 * n) throw std::invalid_argument("cosserate_rod_ode: state of size 18 + 2 n expected");
  const Mat3 R = Eigen::Map<const RowMat3>(state.data() + 3);
  const Vec3 nf = state.segment<3>(12);
  const Vec3 m = state.segment<3>(15);
  const VecX thetas = state.segment(18 + n, k);
  const double u1z = state(18);
  const Vec3 u1 = calc_uxz(thetas, m, s, u1z);
  const Vec3 e3(0, 0, 1);

  VecX dy = VecX::Zero(state.size());
  dy.head<3>() = R * e3;
  Eigen::Map<RowMat3>(dy.data() + 3) = R * hat(u1);
  dy.segment<3>(12) = -external_gauss_forces(s);  // dn/ds = -f, f in the world frame
  dy.segment<3>(15) = -u1.cross(m) - e3.cross(R.transpose() * nf);
  for (Eigen::Index i = 0; i < k; ++i) {
    const TubeParameters& tube = tubes[curr_calc_tubes_[i]];
    // u_iz' = (E_i I_i / G_i J_i) (u_ix u*_iy - u_iy u*_ix) with u_i = R_z(theta_i)^T u_1, Rucker 2010 eq. (22)
    const Vec3 ui = Rz(thetas(i)).transpose() * u1;
    const double curved_part = std::max(is_curved_or_at_end(curr_calc_tubes_[i], s), 0);
    const Vec3 ui_star = get_curvature_vector(tube.kappa * curved_part);
    const Mat3 Kbt = tube.Kbt();
    dy(18 + i) = Kbt(0, 0) / Kbt(2, 2) * (ui(0) * ui_star(1) - ui(1) * ui_star(0));
    dy(18 + n + i) = state(18 + i) - u1z;
  }
  return dy;
}

Vec3 ConcentricTubeContinuumRobot::external_gauss_forces(double s) const {
  Vec3 f = Vec3::Zero();
  for (const Gaussian& g : gaussians_) {
    const double e = std::exp(-g.a * (s - g.mu) * (s - g.mu));
    f(0) += g.fx * e;
    f(1) += g.fy * e;
  }
  return f;
}

void ConcentricTubeContinuumRobot::set_boundary_condition(const std::vector<std::string>& names,
                                                          const std::vector<MatX>& values) {
  if (values.size() < names.size()) throw std::invalid_argument("set_boundary_condition: one value per name expected");
  for (size_t i = 0; i < names.size(); ++i) boundary_values[names[i]] = values[i];
}

void ConcentricTubeContinuumRobot::remove_boundary_condition(const std::string& name) {
  if (!boundary_values.erase(name)) throw MissingBoundaryValue("boundary value '" + name + "' is not set");
}

StaticResult ConcentricTubeContinuumRobot::fwd_static(const Vec6& wrench, double step_size, const VecX& uzs0) {
  ForwardResult f = calc_forward(wrench, step_size, uzs0);
  const Eigen::Index n = static_cast<Eigen::Index>(tubes.size());
  seg_indexes = std::move(f.seg_indexes);
  positions = f.state.leftCols<3>();
  orientations = f.state.middleCols<9>(3);
  wrenches = f.state.middleCols<6>(12);
  uzs = f.state.middleCols(18, n);
  thetas = f.state.rightCols(n);
  return {positions, orientations, wrenches, f.us, uzs, thetas};
}

StaticResult ConcentricTubeContinuumRobot::fwd_static_with_boundarys(const VecX& init_wrench,
                                                                     ShootingFunction shooting_function,
                                                                     double step_size) {
  // Like Python, the result is used also if the solver did not converge.
  const VecX x =
      hybrd([&](const VecX& xi) { return pyctr::shooting_function(shooting_function, xi, *this, step_size); }, init_wrench);
  return fwd_static(x.head<6>(), step_size, x.size() > 6 ? VecX(x.tail(x.size() - 6)) : VecX());
}

StaticResult ConcentricTubeContinuumRobot::push_end(const Vec6& wrench, double step_size) {
  // unknowns n0 (3), m0 (3), u_iz(0) of each tube; residuals n(L) = F, sum m(L) = M, u_iz(L_i) = 0
  set_boundary_condition({"tip_wrench"}, {MatX(wrench)});
  StaticResult r = fwd_static_with_boundarys(VecX::Zero(6 + tubes.size()), ShootingFunction::force, step_size);
  remove_boundary_condition("tip_wrench");
  return r;
}

StaticResult ConcentricTubeContinuumRobot::push_end_to_position(const Vec3& pos, double step_size) {
  set_boundary_condition({"pL", "RL"}, {MatX(pos), MatX(Mat3::Identity())});
  StaticResult r = fwd_static_with_boundarys(VecX::Zero(6), ShootingFunction::tip_pose, step_size);
  remove_boundary_condition("pL");
  remove_boundary_condition("RL");
  return r;
}

StaticResult ConcentricTubeContinuumRobot::fwd_external_gaussian_forces(double step_size) {
  return fwd_static_with_boundarys(VecX::Zero(6), ShootingFunction::gaussian_forces, step_size);
}

std::vector<double> ConcentricTubeContinuumRobot::get_tube_outer_radii() const {
  std::vector<double> r;
  for (const TubeParameters& t : tubes) r.push_back(t.r_outer);
  return r;
}

VecX shooting_function_force(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size) {
  const Vec6 tip_wrench = Eigen::Map<const Vec6>(boundary(ctr, "tip_wrench", 6).data());
  return wrench_residual(guess, ctr, step_size, tip_wrench, true);
}

VecX shooting_function_gaussian_forces(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size) {
  return wrench_residual(guess, ctr, step_size, Vec6::Zero(), guess.size() > 6);
}

VecX shooting_function_tip_pose(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size) {
  const Vec3 pL = Eigen::Map<const Vec3>(boundary(ctr, "pL", 3).data());
  const Mat3 RL = boundary(ctr, "RL", 9).reshaped(3, 3);
  const StaticResult r = ctr.fwd_static(guess_wrench(guess), step_size);
  const Eigen::Index last = r.positions.rows() - 1;
  const Mat3 R = Eigen::Map<const RowMat3>(r.orientations.row(last).eval().data());
  const Mat3 A = R.transpose() * RL;
  VecX res(6);
  res << r.positions.row(last).transpose() - pL, invhat(A - A.transpose());
  return res;
}

VecX shooting_function_tip_position(const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size) {
  const Vec3 pL = Eigen::Map<const Vec3>(boundary(ctr, "pL", 3).data());
  const StaticResult r = ctr.fwd_static(guess_wrench(guess), step_size);
  VecX res = VecX::Zero(6);
  res.head<3>() = r.positions.row(r.positions.rows() - 1).transpose() - pL;
  return res;
}

VecX shooting_function(ShootingFunction f, const VecX& guess, ConcentricTubeContinuumRobot& ctr, double step_size) {
  switch (f) {
    case ShootingFunction::force:
      return shooting_function_force(guess, ctr, step_size);
    case ShootingFunction::gaussian_forces:
      return shooting_function_gaussian_forces(guess, ctr, step_size);
    case ShootingFunction::tip_pose:
      return shooting_function_tip_pose(guess, ctr, step_size);
    case ShootingFunction::tip_position:
      return shooting_function_tip_position(guess, ctr, step_size);
  }
  throw std::invalid_argument("unknown shooting function");
}

}  // namespace pyctr
