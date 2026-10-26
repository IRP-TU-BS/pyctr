#include "pyctr/robot_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace pyctr {

using RowMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

constexpr double TOL = 1e-6;  // m, smallest distance between segment borders (shorter intervals stop Newton)


DynamicConcentricTubeRobot::DynamicConcentricTubeRobot(std::vector<TubeParameters> tubes_, double dt_,
                                                       double bdf_alpha, double contact_stiffness_,
                                                       double contact_dissipation_, const Mat3& R_init_,
                                                       const Vec3& p_init_)
    : tubes(std::move(tubes_)),
      dt(dt_),
      contact_stiffness(contact_stiffness_),
      contact_dissipation(contact_dissipation_),
      R_init(R_init_),
      p_init(p_init_),
      k_(static_cast<int>(tubes.size())) {
  if (k_ == 0) throw std::invalid_argument("DynamicConcentricTubeRobot needs at least one tube");
  alphas = betas = VecX::Zero(k_);
  EI_.resize(k_), GJ_.resize(k_), BB_.resize(k_), BT_.resize(k_), rhoI_.resize(k_), kappa_.resize(k_);
  rhoA_ = 0;
  f_.setZero();
  for (int i = 0; i < k_; ++i) {
    const TubeParameters& t = tubes[i];
    EI_(i) = t.Kbt()(0, 0), GJ_(i) = t.Kbt()(2, 2), BB_(i) = t.Bbt(0, 0), BT_(i) = t.Bbt(2, 2);
    rhoI_(i) = t.rho * t.I(), kappa_(i) = t.kappa;
    rhoA_ += t.rho * t.A();
    f_ += t.rho * t.A() * t.gravity;
  }
  c0_ = (1.5 + bdf_alpha) / (dt * (1 + bdf_alpha)), c1_ = -2 / dt;
  c2_ = (0.5 + bdf_alpha) / (dt * (1 + bdf_alpha)), d1_ = bdf_alpha / (1 + bdf_alpha);
}

void DynamicConcentricTubeRobot::rotate(const VecX& a) {
  if (a.size() < k_) throw std::out_of_range("rotate: one alpha per tube");  // IndexError in Python
  alphas = a.head(k_);
}

void DynamicConcentricTubeRobot::translate(const VecX& b) {
  if (b.size() != k_) throw std::invalid_argument("translate: one beta per tube");
  // the inner tube must stick out; 1e-12 m tolerance, so that tube ends aligned in exact arithmetic pass
  bool valid = tubes[0].L * (1 - b(0)) > 0;
  for (int i = 1; i < k_; ++i)
    valid = valid && tubes[i].L - b(i) * tubes[i].L <= tubes[i - 1].L - b(i - 1) * tubes[i - 1].L + 1e-12;
  if (!valid) throw std::invalid_argument("The beta values do not correspond to the specifications!");
  betas = b;
}

namespace {

// lengths.sum() of numpy: pairwise summation, blocks of 8
double numpy_sum(const double* a, size_t n) {
  if (n < 8) {
    double r = 0;
    for (size_t i = 0; i < n; ++i) r += a[i];
    return r;
  }
  if (n > 128) {
    const size_t n2 = n / 2 - n / 2 % 8;
    return numpy_sum(a, n2) + numpy_sum(a + n2, n - n2);
  }
  double r[8];
  std::copy(a, a + 8, r);
  size_t i = 8;
  for (; i < n - n % 8; i += 8)
    for (int j = 0; j < 8; ++j) r[j] += a[i + j];
  double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
  for (; i < n; ++i) res += a[i];
  return res;
}

// np.interp: linear, fp[0] and fp[n - 1] outside of xp
VecX interp(const VecX& x, const VecX& xp, const VecX& fp) {
  const Eigen::Index n = xp.size();
  if (n == 0) throw std::invalid_argument("interp: no sample points");  // ValueError in Python
  VecX out(x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    const Eigen::Index j = std::upper_bound(xp.data(), xp.data() + n, x(i)) - xp.data() - 1;
    if (j < 0)
      out(i) = fp(0);
    else if (j >= n - 1)
      out(i) = fp(n - 1);
    else if (xp(j) == x(i))
      out(i) = fp(j);
    else
      out(i) = (fp(j + 1) - fp(j)) / (xp(j + 1) - xp(j)) * (x(i) - xp(j)) + fp(j);
  }
  return out;
}

}  // namespace

// all checks before the members change
void DynamicConcentricTubeRobot::grid(int intervals) {
  const int k = k_;
  VecX end(k), curve(k);
  for (int i = 0; i < k; ++i) {
    end(i) = tubes[i].L * (1 - betas(i));
    curve(i) = tubes[i].straight_length - tubes[i].L * betas(i);
  }
  std::vector<double> values;
  for (int i = 0; i < k; ++i) values.push_back(end(i)), values.push_back(curve(i));
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  std::vector<double> borders{0.0};  // borders closer than TOL are merged, roundoff would give intervals of 1e-17 m
  for (double b : values)
    if (borders.back() + TOL < b && b < end(0) - TOL) borders.push_back(b);
  borders.push_back(end(0));

  const int segments = static_cast<int>(borders.size()) - 1;
  std::vector<double> lengths(segments);
  for (int b = 0; b < segments; ++b) lengths[b] = borders[b + 1] - borders[b];
  std::vector<int> counts(segments);
  if (intervals < 0) {
    const double ds = tubes[0].s;
    if (!(ds > 0 && std::isfinite(ds))) throw std::invalid_argument("node spacing s of tube 1 must be positive");
    for (int b = 0; b < segments; ++b) {
      const double nd = std::ceil(lengths[b] / ds);
      if (!(nd < INT_MAX)) throw std::overflow_error("too many nodes, node spacing s of tube 1 is too small");
      counts[b] = static_cast<int>(nd);
    }
  } else if (intervals < segments) {
    char msg[96];
    std::snprintf(msg, sizeof msg, "%d intervals are not enough for %d segments", intervals, segments);
    throw std::invalid_argument(msg);
  } else {  // largest remainder
    const double total = numpy_sum(lengths.data(), segments);
    // Python divides by zero here
    if (!(total > 0)) throw std::invalid_argument("no segment for the intervals, tube 1 is retracted completely");
    std::vector<double> rest(segments);
    int sum = 0;
    for (int b = 0; b < segments; ++b) {
      const double share = static_cast<double>(intervals - segments) * lengths[b] / total;
      counts[b] = 1 + static_cast<int>(std::floor(share));
      rest[b] = std::floor(share) - share;
      sum += counts[b];
    }
    std::vector<int> order(segments);
    for (int b = 0; b < segments; ++b) order[b] = b;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return rest[a] < rest[b]; });
    for (int i = 0; i < std::min(intervals - sum, segments); ++i) ++counts[order[i]];
  }

  std::vector<double> s_{0.0};
  std::vector<double> active, kappa;
  for (int b = 0; b < segments; ++b) {
    const double s0 = borders[b], s1 = borders[b + 1];
    const int n = counts[b];
    const double step = (s1 - s0) / n;  // as np.linspace(s0, s1, n + 1)
    for (int i = 1; i <= n; ++i) {
      s_.push_back(i == n ? s1 : i * step + s0);
      for (int t = 0; t < k; ++t) {
        const double a = end(t) >= s1 - TOL;
        active.push_back(a);
        kappa.push_back(kappa_(t) * a * (curve(t) < 0.5 * (s0 + s1)));
      }
    }
  }
  const int M = static_cast<int>(s_.size());
  ends_.clear();
  for (int t = 1; t < k; ++t) {  // nearest node, the first one of equal distance (argmin)
    int best = 0;
    for (int i = 1; i < M; ++i)
      if (std::abs(s_[i] - end(t)) < std::abs(s_[best] - end(t))) best = i;
    ends_.push_back(best);
  }
  s = Eigen::Map<VecX>(s_.data(), M);
  h_ = s.tail(M - 1) - s.head(M - 1);
  active_ = Eigen::Map<RowMat>(active.data(), M - 1, k);
  kappa_s_ = Eigen::Map<RowMat>(kappa.data(), M - 1, k);
  radius = VecX::Zero(M - 1);
  for (int j = 0; j < M - 1; ++j)
    for (int t = 0; t < k; ++t) radius(j) = std::max(radius(j), active_(j, t) * tubes[t].r_outer);
  base_idx_.clear();
  for (int c = 0; c < 18; ++c) base_idx_.push_back(c);
  for (int c = 23 + k; c < 23 + 3 * k; ++c) base_idx_.push_back(c);
}

// Nodes (start of Newton) and history terms are interpolated in s, the history of u_z and gamma of a tube only from
// where the tube was before (none for a tube that was fully retracted).
void DynamicConcentricTubeRobot::remesh() {
  if (!history_) throw std::logic_error("call init_static first (it did not succeed)");
  if (betas == betas_) return;
  const int k = k_, M = static_cast<int>(s.size()), nz = this->nz();
  const VecX s_old = s, mid = 0.5 * (s.tail(M - 1) + s.head(M - 1));
  std::vector<std::vector<Eigen::Index>> there(k);  // old intervals of tube i
  for (int i = 1; i < k; ++i)
    for (Eigen::Index j = 0; j < M - 1; ++j)
      if (active_(j, i) > 0) there[i].push_back(j);
  grid(M - 1);
  betas_ = betas;
  const VecX new_mid = 0.5 * (s.tail(M - 1) + s.head(M - 1));
  for (Eigen::Index c = 0; c < y.cols(); ++c) y.col(c) = interp(s, s_old, y.col(c));
  for (RowMat* z : {&z_old_, &z_h_}) {
    RowMat out(M - 1, nz);
    for (int c = 0; c < nz; ++c) out.col(c) = interp(new_mid, mid, z->col(c));
    for (int i = 1; i < k; ++i) {
      const VecX xp = mid(there[i]);
      for (int c : {8 + i, 8 + k + i})  // u_z and gamma of tube i
        if (there[i].empty())
          out.col(c).setZero();
        else
          out.col(c) = interp(new_mid, xp, VecX(z->col(c)(there[i]))).cwiseProduct(active_.col(i));
    }
    *z = std::move(out);
  }
}

// eq. (13) with x_t = c0 * x + x_h on interval j, y and z_h of the interval midpoint. work: 3k doubles.
void DynamicConcentricTubeRobot::ode(int j, const double* y, const double* z_h, double c0, const Contacts* contacts,
                                     double* ys, double* z, double* work) const {
  const int k = k_;
  const double* active = &active_(j, 0);
  const double* kappa = &kappa_s_(j, 0);
  const Eigen::Map<const RowMat3> R(y + 3);
  const Eigen::Map<const Vec3> q(y + 12), w(y + 15), n(y + 18);
  const double *mz = y + 23, *th = y + 23 + k, *ga = y + 23 + 2 * k;
  const Eigen::Map<const Vec3> q_h(z_h), w_h(z_h + 3);
  const double *uxy_h = z_h + 6, *uz_h = z_h + 8, *ga_h = z_h + 8 + k;
  double *cos = work, *sin = work + k, *uz = work + 2 * k;

  double EIs = 0, BBs = 0, rhoIs = 0, mzs = 0, rhoIga = 0, bx = 0, by = 0;
  for (int i = 0; i < k; ++i) {
    const double a = active[i], EI = EI_(i) * a;
    cos[i] = std::cos(th[i]), sin[i] = std::sin(th[i]);
    uz[i] = (mz[i] * a - BT_(i) * uz_h[i]) / (GJ_(i) + c0 * BT_(i)) * a;  // (16), u*_z = 0
    EIs += EI, BBs += BB_(i) * a, rhoIs += rhoI_(i) * a, mzs += mz[i] * a, rhoIga += rhoI_(i) * a * ga[i] * a;
    bx -= EI * kappa[i] * sin[i], by += EI * kappa[i] * cos[i];
  }
  const double den = EIs + c0 * BBs;  // (17)
  const double ux1 = (y[21] - BBs * uxy_h[0] + bx) / den, uy1 = (y[22] - BBs * uxy_h[1] + by) / den;
  const Vec3 q_t = c0 * q + q_h, w_t = c0 * w + w_h;
  const double uxt = c0 * ux1 + uxy_h[0], uyt = c0 * uy1 + uxy_h[1], uzt1 = c0 * uz[0] + uz_h[0];
  const Vec3 u(ux1, uy1, uz[0]), u_t(uxt, uyt, uzt1);

  const Vec3 nb = R.transpose() * n;
  const Vec3 w_x_e3(w(1), -w(0), 0), e3_x_nb(-nb(1), nb(0), 0);
  const Vec3 mbs = -u.cross(Vec3(y[21], y[22], mzs)) - e3_x_nb + rhoIs * w_t;

  Vec3 f = f_;
  if (contacts) {
    const Eigen::Map<const Vec3> normal(&contacts->first(j, 0)), point(&contacts->second(j, 0));
    const double depth = radius(j) - normal.dot(Eigen::Map<const Vec3>(y) - point);
    const double v_normal = normal.dot(R * q);
    f += contact_stiffness * std::max(depth, 0.0) * std::max(0.0, 1 - contact_dissipation * v_normal) * normal;
  }

  auto vec = [](double* p) { return Eigen::Map<Vec3>(p); };
  vec(ys) = R.col(2);
  Eigen::Map<RowMat3>{ys + 3} = R * hat(u);
  vec(ys + 12) = -u.cross(q) + w_x_e3;
  vec(ys + 15) = u_t - u.cross(w);
  vec(ys + 18) = -f + rhoA_ * (R * (w.cross(q) + q_t));
  const double spin = rhoIs * w(2) + rhoIga;
  ys[21] = mbs(0) + w_x_e3(0) * spin;
  ys[22] = mbs(1) + w_x_e3(1) * spin;
  for (int i = 0; i < k; ++i) {
    // curvature and bending moment of tube i in its own frame, (4) and (14)
    const double a = active[i], EI = EI_(i) * a, BB = BB_(i) * a, c = cos[i], sn = sin[i];
    const double ux = c * ux1 + sn * uy1, uy = c * uy1 - sn * ux1;
    const double mx = EI * ux + BB * (c * uxt + sn * uyt), my = EI * (uy - kappa[i]) + BB * (c * uyt - sn * uxt);
    const double uz_t = c0 * uz[i] + uz_h[i], ga_t = (c0 * ga[i] * a + ga_h[i]) * a;
    ys[23 + i] = -(ux * my - uy * mx) + 2 * rhoI_(i) * a * (w_t(2) + ga_t);
    ys[23 + k + i] = (uz[i] - uz[0]) * a;
    ys[23 + 2 * k + i] = (uz_t - uzt1) * a;
  }
  if (z) {
    vec(z) = q, vec(z + 3) = w;
    z[6] = ux1, z[7] = uy1;
    for (int i = 0; i < k; ++i) z[8 + i] = uz[i], z[8 + k + i] = ga[i] * active[i];
  }
}

// midpoint rule between nodes (25), base (19), free torsion at the end of outer tubes, tip wrench (21)
void DynamicConcentricTubeRobot::residual(const VecX& x, double c0, const RowMat& z_h, const VecX& alphas_t,
                                          const Vec6& tip_wrench, const Contacts* contacts, VecX& r) {
  const int k = k_, M = static_cast<int>(s.size()), ny = this->ny();
  const Eigen::Map<const RowMat> Y(x.data(), M, ny);
  r.resize(M * ny);
  std::vector<double> mid(ny), work(3 * k);
  for (int j = 0; j < M - 1; ++j) {
    for (int c = 0; c < ny; ++c) mid[c] = 0.5 * (Y(j + 1, c) + Y(j, c));
    ode(j, mid.data(), &z_h(j, 0), c0, contacts, &ys_(j, 0), &z_(j, 0), work.data());
    for (int c = 0; c < ny; ++c) r(j * ny + c) = (Y(j + 1, c) - Y(j, c)) / h_(j) - ys_(j, c);
  }
  const Mat3 R0 = R_init * Rz(alphas(0));
  VecX base(18 + 2 * k);
  Eigen::Map<RowMat3>(base.data() + 3) = R0;
  base.head(3) = p_init;
  base.segment(12, 3) = R0.transpose() * base_velocity.first;
  base.segment(15, 3) = R0.transpose() * base_velocity.second + Vec3(0, 0, alphas_t(0));
  base.segment(18, k) = alphas.array() - alphas(0);
  base.segment(18 + k, k) = alphas_t.array() - alphas_t(0);
  int row = (M - 1) * ny;
  for (size_t b = 0; b < base_idx_.size(); ++b) r(row++) = Y(0, base_idx_[b]) - base(b);
  for (int t = 1; t < k; ++t) r(row++) = Y(ends_[t - 1], 23 + t);
  const Vec3 Mb = Eigen::Map<const RowMat3>(&Y(M - 1, 3)).transpose() * tip_wrench.tail<3>();
  for (int a = 0; a < 3; ++a) r(row++) = Y(M - 1, 18 + a) - tip_wrench(a);
  r(row++) = Y(M - 1, 21) - Mb(0);
  r(row++) = Y(M - 1, 22) - Mb(1);
  r(row++) = Y(M - 1, 23) - Mb(2);
}

// F = df/dy at every interval midpoint by forward differences (ys_ of the last residual is f there)
void DynamicConcentricTubeRobot::jacobian(const VecX& x, double c0, const RowMat& z_h, const Contacts* contacts) {
  const int k = k_, M = static_cast<int>(s.size()), ny = this->ny();
  const Eigen::Map<const RowMat> Y(x.data(), M, ny);
  F_.resize((M - 1) * ny, ny);
  std::vector<double> mid(ny), f1(ny), work(3 * k);
  for (int j = 0; j < M - 1; ++j) {
    for (int c = 0; c < ny; ++c) mid[c] = 0.5 * (Y(j + 1, c) + Y(j, c));
    for (int c = 0; c < ny; ++c) {
      const double m = mid[c], d = 1e-8 * (1 + std::abs(m));
      mid[c] = m + d;
      ode(j, mid.data(), &z_h(j, 0), c0, contacts, f1.data(), nullptr, work.data());
      mid[c] = m;
      for (int r = 0; r < ny; ++r) F_(j * ny + r, c) = (f1[r] - ys_(j, r)) / d;
    }
  }
}

namespace {

// Gaussian elimination with partial pivoting of the first n columns of the rows [0, m) of W. The right hand side
// is a column of W, so the multipliers are not kept.
void eliminate(RowMat& W, int m, int n) {
  const int w = static_cast<int>(W.cols());
  for (int i = 0; i < n; ++i) {
    int p;
    W.col(i).segment(i, m - i).cwiseAbs().maxCoeff(&p);
    p += i;
    if (!(W(p, i) != 0)) throw std::runtime_error("Newton's method: singular Jacobian");
    if (p != i) W.row(p).swap(W.row(i));
    const double inv = 1 / W(i, i);
    const double* pivot = &W(i, i + 1);
    for (int r = i + 1; r < m; ++r) {
      const double l = W(r, i) * inv;
      if (l == 0) continue;  // the interval rows are sparse
      double* row = &W(r, i + 1);
      for (int c = 0; c < w - i - 1; ++c) row[c] -= l * pivot[c];
    }
  }
}

}  // namespace

// Solves J dx = r for the Jacobian of residual(): interval j rows are -I / h - 0.5 F_j (node j) and
// I / h - 0.5 F_j (node j + 1), the boundary rows are linear in y apart from R^T M. The ny columns of node j are
// eliminated from the rows that are not zero in them (rows carried from node j - 1, interval j, ends at node j),
// the remaining rows are carried to node j + 1. That is partial pivoting on the whole matrix, without the zeros.
VecX DynamicConcentricTubeRobot::newton_step(const VecX& r, const Vec6& tip_wrench) {
  const int k = k_, M = static_cast<int>(s.size()), ny = this->ny(), last = (M - 1) * ny;
  const int nb = static_cast<int>(base_idx_.size()), rhs = 2 * ny;
  RowMat W(2 * ny, 2 * ny + 1);  // columns: node j, node j + 1, right hand side
  U_.resize(last, 2 * ny + 1);   // pivot rows of every node for the back substitution
  int c = 0;                     // rows in W
  auto unit_row = [&](int col, int row_of_r) {
    W.row(c).setZero();
    W(c, col) = 1, W(c, rhs) = r(row_of_r);
    ++c;
  };
  auto add_ends = [&](int node) {
    for (int t = 1; t < k; ++t)
      if (ends_[t - 1] == node) unit_row(23 + t, last + nb + t - 1);
  };
  for (int b = 0; b < nb; ++b) unit_row(base_idx_[b], last + b);
  for (int j = 0; j < M - 1; ++j) {
    const double ih = 1 / h_(j);
    for (int i = 0; i < ny; ++i, ++c) {
      W.row(c).head(ny) = W.row(c).segment(ny, ny) = -0.5 * F_.row(j * ny + i);
      W(c, i) -= ih, W(c, ny + i) += ih, W(c, rhs) = r(j * ny + i);
    }
    add_ends(j);
    eliminate(W, c, ny);
    U_.middleRows(j * ny, ny) = W.topRows(ny);
    c -= ny;
    for (int i = 0; i < c; ++i) {
      W.row(i).head(ny) = W.row(ny + i).segment(ny, ny);
      W.row(i).segment(ny, ny).setZero();
      W(i, rhs) = W(ny + i, rhs);
    }
  }
  add_ends(M - 1);
  for (int a = 0; a < 3; ++a) unit_row(18 + a, last + nb + k - 1 + a);
  for (int a = 0; a < 3; ++a) {  // m^b_xy and m^b_z of tube 1 minus R^T M
    unit_row(a < 2 ? 21 + a : 23, last + nb + k + 2 + a);
    for (int b = 0; b < 3; ++b) W(c - 1, 3 + 3 * b + a) = -tip_wrench(3 + b);
  }
  eliminate(W, ny, ny);
  VecX dx(M * ny);
  dx.tail(ny) = W.topLeftCorner(ny, ny).triangularView<Eigen::Upper>().solve(W.col(rhs).head(ny));
  for (int j = M - 2; j >= 0; --j) {
    const auto U = U_.middleRows(j * ny, ny);
    dx.segment(j * ny, ny) = U.leftCols(ny).triangularView<Eigen::Upper>().solve(
        U.col(rhs) - U.middleCols(ny, ny) * dx.segment((j + 1) * ny, ny));
  }
  return dx;
}

RobotState DynamicConcentricTubeRobot::solve(double c0, const RowMat& z_h, const VecX& alphas_t,
                                             const Vec6& tip_wrench, const std::optional<Contacts>& contacts) {
  const int k = k_, M = static_cast<int>(s.size()), ny = this->ny();
  if (contacts && (contacts->first.rows() != M - 1 || contacts->first.cols() != 3 ||
                   contacts->second.rows() != M - 1 || contacts->second.cols() != 3))
    throw std::invalid_argument("contacts: normals and points must be (M - 1, 3)");
  const Contacts* cp = contacts ? &*contacts : nullptr;
  ys_.resize(M - 1, ny);
  z_.resize(M - 1, nz());
  VecX x = Eigen::Map<const VecX>(y.data(), y.size()), r;
  double rmax = 0;
  bool converged = false;
  for (int it = 0; it < 20; ++it) {
    residual(x, c0, z_h, alphas_t, tip_wrench, cp, r);
    if (!r.allFinite()) {
      rmax = std::numeric_limits<double>::quiet_NaN();  // maxCoeff ignores NaN
      break;
    }
    rmax = r.cwiseAbs().maxCoeff();
    if (rmax < 1e-9) {
      converged = true;
      break;
    }
    jacobian(x, c0, z_h, cp);
    x -= newton_step(r, tip_wrench);
  }
  if (!converged) {
    char msg[96];
    std::snprintf(msg, sizeof msg, "Newton's method did not converge, max residual %.1e", rmax);
    throw std::runtime_error(msg);
  }
  y = Eigen::Map<const RowMat>(x.data(), M, ny);
  contact_force = RowMat::Zero(M - 1, 3);
  if (cp) {
    const double K = contact_stiffness, D = contact_dissipation;
    for (int j = 0; j < M - 1; ++j) {
      const VecX mid = 0.5 * (y.row(j + 1) + y.row(j)).transpose();
      const Eigen::Map<const Vec3> normal(&cp->first(j, 0)), point(&cp->second(j, 0));
      const double depth = radius(j) - normal.dot(mid.head<3>() - point);
      const double v_normal = normal.dot(Eigen::Map<const RowMat3>(mid.data() + 3) * mid.segment<3>(12));
      contact_force.row(j) = K * std::max(depth, 0.0) * std::max(0.0, 1 - D * v_normal) * h_(j) * normal;
    }
  }
  RowMat wrenches(M, 6);
  wrenches.leftCols(5) = y.middleCols(18, 5);
  wrenches.col(5) = y.middleCols(23, k).rowwise().sum();
  return {y.leftCols(3), y.middleCols(3, 9), wrenches};
}

RobotState DynamicConcentricTubeRobot::init_static(const Vec6& tip_wrench, const std::optional<Contacts>& contacts) {
  history_ = false;  // step needs the history of this grid, also after a failed solve
  grid();
  grid(static_cast<int>(s.size()) - 1);  // the same grid as remesh gives for these betas
  betas_ = betas;
  const int k = k_, M = static_cast<int>(s.size());
  const Mat3 R0 = R_init * Rz(alphas(0));
  y = RowMat::Zero(M, ny());  // straight robot as first guess
  for (int i = 0; i < M; ++i) {
    y.row(i).head<3>() = (p_init + s(i) * R0.col(2)).transpose();
    Eigen::Map<RowMat3>(&y(i, 3)) = R0;
    y.row(i).segment(23 + k, k) = (alphas.array() - alphas(0)).transpose();
  }
  RobotState out = solve(0.0, RowMat::Zero(M - 1, nz()), VecX::Zero(k), tip_wrench, contacts);
  z_old_ = z_, z_h_ = (c1_ + c2_) * z_;
  a_old_ = alphas, a_h_ = (c1_ + c2_) * alphas;
  history_ = true;
  return out;
}

RobotState DynamicConcentricTubeRobot::step(const Vec6& tip_wrench, const std::optional<Contacts>& contacts) {
  remesh();
  RobotState out = solve(c0_, z_h_, c0_ * alphas + a_h_, tip_wrench, contacts);
  // history term of the next step, _advance in Python
  z_h_ = c1_ * z_ + c2_ * z_old_ + d1_ * (c0_ * z_ + z_h_), z_old_ = z_;
  a_h_ = c1_ * alphas + c2_ * a_old_ + d1_ * (c0_ * alphas + a_h_), a_old_ = alphas;
  return out;
}

VecX DynamicConcentricTubeRobot::get_state() const {
  if (!history_) throw std::logic_error("call init_static first (it did not succeed)");
  VecX x(y.size() + 2 * z_h_.size() + 3 * k_);
  x << Eigen::Map<const VecX>(y.data(), y.size()), Eigen::Map<const VecX>(z_old_.data(), z_old_.size()),
      Eigen::Map<const VecX>(z_h_.data(), z_h_.size()), a_old_, a_h_, betas_;
  return x;
}

void DynamicConcentricTubeRobot::set_state(const VecX& x) {
  if (!history_) throw std::logic_error("call init_static first (it did not succeed)");
  const Eigen::Index size = y.size() + 2 * z_h_.size() + 3 * k_;
  if (x.size() != size)
    throw std::invalid_argument("state has " + std::to_string(x.size()) + " values, this robot needs " +
                                std::to_string(size));
  const VecX b = x.tail(k_);
  if (b != betas) translate(b);
  if (b != betas_) {
    grid(static_cast<int>(s.size()) - 1);
    betas_ = b;
  }
  const double* p = x.data();
  y = Eigen::Map<const RowMat>(p, y.rows(), y.cols()), p += y.size();
  z_old_ = Eigen::Map<const RowMat>(p, z_old_.rows(), z_old_.cols()), p += z_old_.size();
  z_h_ = Eigen::Map<const RowMat>(p, z_h_.rows(), z_h_.cols()), p += z_h_.size();
  a_old_ = Eigen::Map<const VecX>(p, k_), a_h_ = Eigen::Map<const VecX>(p + k_, k_);
}

}  // namespace pyctr
