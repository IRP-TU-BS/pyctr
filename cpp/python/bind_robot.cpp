// Python bindings of the static concentric tube robot, same names and return values as pyctr.robots.
#include <string>

#include "convert.hpp"
#include "pyctr/robot.hpp"

namespace py = pybind11;
using namespace pyctr;
using Robot = ConcentricTubeContinuumRobot;

namespace {

// any array-like with 3 entries, e.g. the Python default p_init of shape (3, 1)
Vec3 vec3_from_python(py::handle obj) {
  py::array_t<double, py::array::c_style | py::array::forcecast> a(py::reinterpret_borrow<py::object>(obj));
  if (a.size() != 3) throw py::value_error("3 entries expected");
  return Eigen::Map<const Vec3>(a.data());
}

// 1-D arrays become column vectors
MatX matrix_from_python(py::handle obj) {
  py::array_t<double, py::array::c_style | py::array::forcecast> a(py::reinterpret_borrow<py::object>(obj));
  const bool two_d = a.ndim() == 2;
  const Eigen::Index rows = two_d ? a.shape(0) : a.size(), cols = two_d ? a.shape(1) : 1;
  return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(a.data(), rows, cols);
}

py::object matrix_to_python(const MatX& m) {
  if (m.cols() == 1) return py::cast(VecX(m));
  return py::cast(m);
}

// None before the first fwd_static, as in Python
py::object state_to_python(const MatX& m) { return m.size() ? py::cast(m) : py::object(py::none()); }

// [0, (tubes in segment, cumulative row count), ...] as in Python
py::list seg_indexes_to_python(const std::vector<std::pair<int, int>>& seg) {
  py::list out;
  out.append(0);
  for (const auto& [count, end] : seg) out.append(py::make_tuple(count, end));
  return out;
}

py::tuple static_to_python(const StaticResult& r) {
  return py::make_tuple(r.positions, r.orientations, r.wrenches, r.us, r.uzs, r.thetas);
}

// ShootingFunction, its name ("force" or "shooting_function_force"), or a function with that __name__
// (the functions of pyctr.robots and of this module)
ShootingFunction shooting_from_python(py::handle f) {
  if (py::isinstance<ShootingFunction>(f)) return f.cast<ShootingFunction>();
  std::string name = py::isinstance<py::str>(f) ? f.cast<std::string>() : py::getattr(f, "__name__").cast<std::string>();
  const std::string prefix = "shooting_function_";
  if (name.rfind(prefix, 0) == 0) name = name.substr(prefix.size());
  if (name == "force") return ShootingFunction::force;
  if (name == "gaussian_forces") return ShootingFunction::gaussian_forces;
  if (name == "tip_pose") return ShootingFunction::tip_pose;
  if (name == "tip_position") return ShootingFunction::tip_position;
  throw py::value_error("unknown shooting function '" + name + "'");
}

VecX optional_vec(py::handle obj) { return obj.is_none() ? VecX() : obj.cast<VecX>(); }

}  // namespace

PYBIND11_MODULE(_robot, m) {
  m.doc() = "Static concentric tube continuum robot (C++ port of pyctr.robots)";

  py::register_local_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const MissingBoundaryValue& e) {
      PyErr_SetString(PyExc_KeyError, e.what());
    }
  });

  py::enum_<ShootingFunction>(m, "ShootingFunction")
      .value("force", ShootingFunction::force)
      .value("gaussian_forces", ShootingFunction::gaussian_forces)
      .value("tip_pose", ShootingFunction::tip_pose)
      .value("tip_position", ShootingFunction::tip_position);

  py::class_<Robot>(m, "ConcentricTubeContinuumRobot")
      .def(py::init([](py::iterable tubes, const Mat3& R_init, py::object p_init) {
             return Robot(tubes_from_python(tubes), R_init, p_init.is_none() ? Vec3::Zero() : vec3_from_python(p_init));
           }),
           py::arg("tubes"), py::arg("R_init") = Mat3(Mat3::Identity()), py::arg("p_init") = py::none())
      .def_property_readonly("tubes",
                             [](const Robot& r) {
                               py::list out;
                               for (const TubeParameters& t : r.tubes) out.append(tube_to_python(t));
                               return out;
                             })
      // R_init, p_init, alphas and betas are numpy views on the members: in-place writes change the robot,
      // as with the numpy attributes in Python
      .def_property(
          "R_init", [](Robot& r) -> Mat3& { return r.R_init; }, [](Robot& r, const Mat3& R) { r.R_init = R; })
      .def_property(
          "p_init",  // (3, 1) as in Python
          [](py::object self) {
            Vec3& p = self.cast<Robot&>().p_init;
            return py::array_t<double>({3, 1}, {sizeof(double), sizeof(double)}, p.data(), self);
          },
          [](Robot& r, py::handle p) { r.p_init = vec3_from_python(p); })
      .def_property(
          "alphas", [](Robot& r) -> VecX& { return r.alphas; }, &Robot::rotate)
      .def_property(
          "betas", [](Robot& r) -> VecX& { return r.betas; },
          [](Robot& r, const VecX& betas) {  // without the validity check of translate, as in Python
            if (betas.size() != r.betas.size()) throw py::value_error("one beta per tube expected");
            r.betas = betas;
          })
      .def_readonly("step_len", &Robot::step_len)
      .def_property_readonly("boundary_values",
                             [](const Robot& r) {
                               py::dict d;
                               for (const auto& [name, value] : r.boundary_values) d[name.c_str()] = matrix_to_python(value);
                               return d;
                             })
      .def_property_readonly("positions", [](const Robot& r) { return state_to_python(r.positions); })
      .def_property_readonly("orientations", [](const Robot& r) { return state_to_python(r.orientations); })
      .def_property_readonly("wrenches", [](const Robot& r) { return state_to_python(r.wrenches); })
      .def_property_readonly("uzs", [](const Robot& r) { return state_to_python(r.uzs); })
      .def_property_readonly("thetas", [](const Robot& r) { return state_to_python(r.thetas); })
      .def_property_readonly("seg_indexes",
                             [](const Robot& r) {
                               return r.positions.size() ? py::object(seg_indexes_to_python(r.seg_indexes))
                                                         : py::object(py::none());
                             })
      .def_static("get_curvature_vector", &Robot::get_curvature_vector, py::arg("kappa"))
      .def("get_ordered_segments", &Robot::get_ordered_segments, py::arg("only_positive") = true)
      .def("tube_tip_indices",
           [](const Robot& r) {
             py::list out;
             for (int i : r.tube_tip_indices()) out.append(i < 0 ? py::object(py::none()) : py::object(py::int_(i)));
             return out;
           })
      .def("rotate", &Robot::rotate, py::arg("alphas"))
      .def("translate", &Robot::translate, py::arg("betas"))
      .def(
          "calc_forward",
          [](Robot& r, const Vec6& wrench, double step_len, py::object uzs0) {
            const ForwardResult f = r.calc_forward(wrench, step_len, optional_vec(uzs0));
            return py::make_tuple(f.state, f.us, seg_indexes_to_python(f.seg_indexes));
          },
          py::arg("wrench"), py::arg("step_len"), py::arg("uzs0") = py::none())
      .def(
          "calc_us",
          [](const Robot& r, const std::vector<double>& s, const MatX& thetas_batch, const MatX& m_batch,
             const MatX& uz_batch) {
            if (uz_batch.cols() < 1) throw py::value_error("calc_us: uz_batch needs at least one column");
            return r.calc_us(s, thetas_batch, m_batch, uz_batch.col(0));
          },
          py::arg("s"), py::arg("thetas_batch"), py::arg("m_batch"), py::arg("uz_batch"))
      .def(
          "calc_uxz",
          [](const Robot& r, const VecX& thetas, const Vec3& m, double s, double u1z) {
            return MatX(r.calc_uxz(thetas, m, s, u1z).transpose());  // (1, 3) as in Python
          },
          py::arg("thetas"), py::arg("m"), py::arg("s"), py::arg("u1z"))
      .def("cosserate_rod_ode", &Robot::cosserate_rod_ode, py::arg("s"), py::arg("state"))
      .def(
          "external_gauss_forces", [](const Robot& r, double s) { return MatX(r.external_gauss_forces(s)); },
          py::arg("s"))
      .def(
          "set_boundary_condition",
          [](Robot& r, const std::vector<std::string>& names, py::sequence values) {
            std::vector<MatX> v;
            for (py::handle value : values) v.push_back(matrix_from_python(value));
            r.set_boundary_condition(names, v);
          },
          py::arg("names"), py::arg("values"))
      .def("remove_boundary_condition", &Robot::remove_boundary_condition, py::arg("name"))
      .def("remove_all_boundary_condition", &Robot::remove_all_boundary_condition)
      .def(
          "fwd_static",
          [](Robot& r, const Vec6& wrench, double step_size, py::object uzs0) {
            return static_to_python(r.fwd_static(wrench, step_size, optional_vec(uzs0)));
          },
          py::arg("wrench"), py::arg("step_size") = 0.01, py::arg("uzs0") = py::none())
      .def(
          "fwd_kinematic", [](Robot& r, double step_size) { return static_to_python(r.fwd_kinematic(step_size)); },
          py::arg("step_size") = 0.01)
      .def(
          "fwd_static_with_boundarys",
          [](Robot& r, const VecX& init_wrench, py::handle shooting_function, double step_size) {
            return static_to_python(
                r.fwd_static_with_boundarys(init_wrench, shooting_from_python(shooting_function), step_size));
          },
          py::arg("init_wrench"), py::arg("shooting_function"), py::arg("step_size") = 0.01)
      .def(
          "push_end",
          [](Robot& r, const Vec6& wrench, double step_size) { return static_to_python(r.push_end(wrench, step_size)); },
          py::arg("wrench"), py::arg("step_size") = 0.01)
      .def(
          "push_end_to_position",
          [](Robot& r, py::handle pos, double step_size) {
            return static_to_python(r.push_end_to_position(vec3_from_python(pos), step_size));
          },
          py::arg("pos"), py::arg("step_size") = 0.01)
      .def(
          "fwd_external_gaussian_forces",
          [](Robot& r, double step_size) { return static_to_python(r.fwd_external_gaussian_forces(step_size)); },
          py::arg("step_size") = 0.001)
      .def(
          "set_gaussians",
          [](Robot& r, py::iterable gaussians) {
            std::vector<Gaussian> g;
            for (py::handle item : gaussians) {  // ((fx, fy), a, mu)
              py::sequence t = py::reinterpret_borrow<py::sequence>(item);
              py::sequence f = t[0];
              g.push_back({py::float_(f[0]), py::float_(f[1]), py::float_(t[1]), py::float_(t[2])});
            }
            r.set_gaussians(std::move(g));
          },
          py::arg("gaussians"))
      .def("remove_gaussians", &Robot::remove_gaussians)
      .def("get_tube_outer_radii", &Robot::get_tube_outer_radii);

  m.def("shooting_function_force", &shooting_function_force, py::arg("guess"), py::arg("ctr"), py::arg("step_size"));
  m.def("shooting_function_gaussian_forces", &shooting_function_gaussian_forces, py::arg("guess"), py::arg("ctr"),
        py::arg("step_size"));
  m.def("shooting_function_tip_pose", &shooting_function_tip_pose, py::arg("guess"), py::arg("ctr"),
        py::arg("step_size"));
  m.def("shooting_function_tip_position", &shooting_function_tip_position, py::arg("guess"), py::arg("ctr"),
        py::arg("step_size"));
}
