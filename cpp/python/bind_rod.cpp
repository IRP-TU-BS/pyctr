#include "convert.hpp"
#include "pyctr/rod.hpp"

namespace py = pybind11;
using namespace pyctr;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// (N, cols) from an array with N rows, e.g. rotations as (N, 9) or (N, 3, 3), row-major as numpy
MatX rows_of(const Array& a, int cols) {
  const py::ssize_t n = a.ndim() == 0 ? 1 : a.shape(0);
  if (a.size() != n * cols) throw py::value_error("expected " + std::to_string(cols) + " values per row");
  return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(a.data(), n, cols);
}

Vec3 vec3_of(const Array& a) {
  if (a.size() != 3) throw py::value_error("expected 3 elements");
  return Eigen::Map<const Vec3>(a.data());
}

py::tuple states_to_python(const RodStates& s) { return py::make_tuple(s.p, s.R, s.wrench); }

// keys of the Python params dict; A, I, J, Kse, Kbt are derived from the current values
py::dict params_to_python(const CosseratRod& rod) {
  const TubeParameters& t = rod.params;
  py::dict d;
  d["E"] = t.E;
  d["G"] = t.G;
  d["r_inner"] = t.r_inner;
  d["r_outer"] = t.r_outer;
  d["rho"] = t.rho;
  d["g"] = MatX(t.gravity.transpose());  // (1, 3) as in Python
  d["L"] = t.L;
  d["straight_length"] = t.straight_length;
  d["s"] = t.s;
  d["kappa"] = t.kappa;
  d["beta"] = rod.beta;
  d["alpha"] = rod.alpha;
  d["A"] = t.A();
  d["I"] = t.I();
  d["J"] = t.J();
  d["Kse"] = t.Kse();
  d["Kbt"] = t.Kbt();
  return d;
}

// Python: rod.params.update(d). Unknown keys are ignored as in tube_from_python.
void set_params(CosseratRod& rod, const py::dict& d) {
  for (const char* key : {"A", "I", "J", "Kse", "Kbt"})
    if (d.contains(key)) throw py::value_error(std::string(key) + " is derived from E, G, r_inner and r_outer");
  py::dict all = params_to_python(rod);
  for (auto item : d) all[item.first] = item.second;
  rod.params = tube_from_python(all);
  rod.beta = all["beta"].cast<double>();
  rod.alpha = all["alpha"].cast<double>();
}

TubeParameters tube_of(const py::object& params) {
  return params.is_none() ? TubeParameters() : tube_from_python(params);
}

// accepts pyctr_cpp.Force_Model and pyctr.cosserat_rod_force_along.Force_Model (same values)
Force_Model force_model_of(const py::object& o) { return static_cast<Force_Model>(o.attr("value").cast<int>()); }

bool same_force_model(Force_Model a, const py::object& b) {
  return py::hasattr(b, "value") && py::int_(static_cast<int>(a)).equal(b.attr("value"));
}

template <class T, class Base>
void bind_ext(py::class_<T, Base>& c) {
  c.def("external_fourier_forces",
        [](const T& r, double s) { return MatX(r.external_fourier_forces(s)); }, py::arg("s"))
      .def("apply_external_forces", [](T& r) { return states_to_python(r.apply_external_forces()); })
      .def("shooting_function_external_force", &T::shooting_function_external_force, py::arg("guess"),
           py::arg("s_l") = 100)
      .def_property(
          "_gaussians",
          // a tuple, so that in-place changes (append, item assignment) raise instead of changing a copy
          [](const T& r) {
            py::list l;
            for (const Gaussian& g : r.gaussians) l.append(py::make_tuple(py::make_tuple(g.fx, g.fy), g.c, g.mu));
            return py::tuple(l);
          },
          [](T& r, py::iterable l) {
            r.gaussians.clear();
            for (py::handle g : l) {
              auto t = g.cast<py::tuple>();
              auto f = t[0].cast<py::tuple>();
              r.gaussians.push_back({f[0].cast<double>(), f[1].cast<double>(), t[1].cast<double>(), t[2].cast<double>()});
            }
          },
          "((fx, fy), c, mu) terms: force f * exp(-c (s - mu)^2) in the world frame, assign the whole list to change it")
      // read-only copies, assign the whole array to change a coefficient
      .def_readwrite("_ax", &T::ax)
      .def_readwrite("_ay", &T::ay)
      .def_readwrite("_bx", &T::bx)
      .def_readwrite("_by", &T::by)
      .def_property(
          "_force_model", [](const T& r) { return r.force_model; },
          [](T& r, py::object o) { r.force_model = force_model_of(o); });
}

}  // namespace

PYBIND11_MODULE(_rod, m) {
  py::register_exception<NotImplementedError>(m, "_NotImplementedError", PyExc_NotImplementedError);
  py::register_exception<AttributeError>(m, "_AttributeError", PyExc_AttributeError);
  py::register_exception<ZeroDivisionError>(m, "_ZeroDivisionError", PyExc_ZeroDivisionError);

  py::enum_<Force_Model> force_model(m, "Force_Model");
  force_model.value("GAUSSIAN", Force_Model::GAUSSIAN).value("FOURIER", Force_Model::FOURIER);
  // equal to pyctr.cosserat_rod_force_along.Force_Model members with the same value; assigned instead of def,
  // because def would only add an overload behind the __eq__ of enum_, which matches any object first
  force_model.attr("__eq__") =
      py::cpp_function([](Force_Model a, py::object b) { return same_force_model(a, b); }, py::is_method(force_model));
  force_model.attr("__ne__") =
      py::cpp_function([](Force_Model a, py::object b) { return !same_force_model(a, b); }, py::is_method(force_model));

  py::class_<CosseratRod>(m, "CosseratRod", "Static Cosserat rod, port of pyctr.cosserat_rod.CosseratRod")
      .def(py::init([](py::object params) { return CosseratRod(tube_of(params)); }), py::arg("params") = py::none())
      .def_property_readonly("params", &params_to_python,
                             "copy of the parameters, change them with set_params (writing into the dict has no effect)")
      .def("set_params", &set_params, py::arg("params"))
      .def(
          "set_initial_conditions",
          [](CosseratRod& r, const Array& p0, const Mat3& R0) { r.set_initial_conditions(vec3_of(p0), R0); },
          py::arg("p0"), py::arg("R0"))
      .def("set_bounding_values", &CosseratRod::set_bounding_values, py::arg("names"), py::arg("values"))
      .def(
          "get_u_div",
          [](const CosseratRod& r, const Mat3& R, const Array& n, const Array& m, const Array& u) -> py::object {
            const Vec3 du = r.get_u_div(R, vec3_of(n), vec3_of(m), vec3_of(u));
            return m.ndim() == 1 ? py::cast(du) : py::cast(MatX(du));  // shape of m, (3,) or (3, 1) as in Python
          },
          py::arg("R"), py::arg("n"), py::arg("m"), py::arg("u"))
      .def("cosserate_rod_ode", &CosseratRod::cosserate_rod_ode, py::arg("state"), py::arg("s"))
      .def("cosserat_rod_ode_curvature", &CosseratRod::cosserat_rod_ode_curvature, py::arg("state"), py::arg("s"))
      .def(
          "external_gauss_forces", [](const CosseratRod& r, double s) { return MatX(r.external_gauss_forces(s)); },
          py::arg("s"), "(3, 1)")
      .def("shooting_function", &CosseratRod::shooting_function, py::arg("guess"))
      .def("shooting_function_force", &CosseratRod::shooting_function_force, py::arg("guess"),
           py::arg("curvature_integration"))
      .def(
          "push_end",
          [](CosseratRod& r, const VecX& wrench, bool ci) { return states_to_python(r.push_end(wrench, ci)); },
          py::arg("wrench"), py::arg("curvature_integration") = false)
      .def(
          "move_end",
          [](CosseratRod& r, const Array& p, const Mat3& R) { return states_to_python(r.move_end(vec3_of(p), R)); },
          py::arg("p"), py::arg("R"))
      .def(
          "calc_m_from_curvature",
          [](const CosseratRod& r, const Array& R, const Array& u) {
            return r.calc_m_from_curvature(rows_of(R, 9), rows_of(u, 3));
          },
          py::arg("R"), py::arg("u"))
      .def(
          "calc_u_from_m",
          [](const CosseratRod& r, const Array& R, const Array& m, py::object) {
            return r.calc_u_from_m(rows_of(R, 9), rows_of(m, 3));
          },
          py::arg("R"), py::arg("m"), py::arg("u_star") = py::none())
      .def(
          "apply_force",
          [](CosseratRod& r, const VecX& w, int s_l, bool ci) { return states_to_python(r.apply_force(w, s_l, ci)); },
          py::arg("wrench"), py::arg("s_l") = 20, py::arg("curvature_integration") = true)
      .def("is_curved_or_at_end", &CosseratRod::is_curved_or_at_end, py::arg("s"))
      .def("get_kappa", &CosseratRod::get_kappa)
      .def("set_kappa", &CosseratRod::set_kappa, py::arg("kappa"));

  py::class_<CurvedCosseratRod, CosseratRod>(m, "CurvedCosseratRod",
                                             "Precurved rod, port of pyctr.cosserat_rod.CurvedCosseratRod")
      .def(py::init([](py::object params) {
             return params.is_none() ? CurvedCosseratRod() : CurvedCosseratRod(tube_from_python(params));
           }),
           py::arg("params") = py::none())
      .def(
          "apply_force",
          [](CurvedCosseratRod& r, const VecX& w, int steps, bool ci) {
            return states_to_python(r.apply_force(w, steps, ci));
          },
          py::arg("wrench"), py::arg("steps") = 100, py::arg("curvature_integration") = false);

  py::class_<StraightCosseratRod, CosseratRod> straight(
      m, "StraightCosseratRod", "Straight rod with forces along the body, port of pyctr.cosserat_rod_force_along");
  straight.def(py::init([](py::object params, py::object fm) {
                 return StraightCosseratRod(tube_of(params), force_model_of(fm));
               }),
               py::arg("params") = py::none(), py::arg("force_model") = Force_Model::GAUSSIAN);
  bind_ext(straight);

  py::class_<CurvedCosseratRodExt, CurvedCosseratRod> curved_ext(
      m, "CurvedCosseratRodExt", "Precurved rod with forces along the body, port of pyctr.cosserat_rod_force_along");
  curved_ext.def(py::init([](py::object params, py::object fm) {
                   return params.is_none() ? CurvedCosseratRodExt(force_model_of(fm))
                                           : CurvedCosseratRodExt(tube_from_python(params), force_model_of(fm));
                 }),
                 py::arg("params") = py::none(), py::arg("force_model") = Force_Model::GAUSSIAN);
  bind_ext(curved_ext);
}
