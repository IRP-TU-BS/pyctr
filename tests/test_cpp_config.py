import os
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from pyctr.yaml_to_model import load_continous_ctcr_model, setup_tubes

try:
    import pyctr_cpp

    HAS_CPP = hasattr(pyctr_cpp, "setup_tubes") and hasattr(pyctr_cpp, "load_tubes")
except ImportError:
    HAS_CPP = False

EXAMPLE_ROBOT = os.path.join(os.path.dirname(__file__), "../example_robots/ctr_robot.yaml")

# Parameters that Python and C++ compute with the same IEEE operations in the same order: compared exactly.
EXACT = ("E", "G", "r_inner", "r_outer", "rho", "L", "straight_length", "s", "kappa")
# Derived from the radii with a different order of operations (J = pi (2r)^4 / 32 in Python, pi r^4 / 2 in C++):
# a few rounding errors of 1e-16 each.
DERIVED = ("A", "I", "J", "Kse", "Kbt")

TUBE_FILE = """
L: 200
curved_len: 50.5
kappa: '(1 + 2) * -3e-1 / 4'
E: 2E+11
G: 8 * 1e10
r_inner: 0.5 * 1e-3
r_outer: '- -1.  /  1000'
rho: .5e4
g: 9.81
"""

INLINE_CONFIG = """
tubes:
    inner: 'tubes/tube.yaml'
    1:
        L: 100.
        curved_len: 100
        kappa: 2 * (3 + 4) / 7
        r_inner: 1.5e-3
        r_outer: (2.5) * 1e-3
        unused: 1 + 1
    outer: {L: 80, curved_len: 0, g: -2.5}
"""


@unittest.skipIf(not HAS_CPP, "pyctr_cpp with setup_tubes and load_tubes is not built")
class TestCppConfig(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "tubes"))
        self.write("tubes/tube.yaml", TUBE_FILE)
        self.inline = self.write("robot.yaml", INLINE_CONFIG)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def assert_same_raw(self, path):
        expected = setup_tubes(path)
        result = pyctr_cpp.setup_tubes(path)
        # same doubles: both parse decimal numbers correctly rounded and evaluate in the same order
        self.assertEqual(result, expected)
        return result

    def assert_same_tubes(self, path):
        robot = load_continous_ctcr_model(path)
        tubes = pyctr_cpp.load_tubes(path)
        self.assertEqual(len(tubes), len(robot.tubes))
        for d, (_, rod) in zip(tubes, robot.tubes):
            p = rod.params
            for key in EXACT:
                self.assertEqual(d[key], p[key], key)
            np.testing.assert_array_equal(d["g"], p["g"].flatten())
            for key in DERIVED:
                np.testing.assert_allclose(d[key], p[key], rtol=1e-14, atol=0, err_msg=key)
        return tubes

    def test_example_robot(self):
        self.assert_same_raw(EXAMPLE_ROBOT)
        tubes = self.assert_same_tubes(EXAMPLE_ROBOT)
        self.assertEqual(tubes[0]["L"], 0.185)
        np.testing.assert_array_equal(tubes[0]["g"], [0, 0, 0])

    def test_inline_tubes_and_expressions(self):
        raw = self.assert_same_raw(self.inline)
        self.assertEqual([r["L"] for r in raw], [200, 100, 80])  # file order, not key order
        self.assertEqual(raw[0]["kappa"], (1 + 2) * -3e-1 / 4)
        self.assertEqual(raw[0]["r_outer"], 1e-3)
        self.assertEqual(raw[1]["unused"], 2)
        tubes = self.assert_same_tubes(self.inline)
        np.testing.assert_array_equal(tubes[0]["g"], [0, 0, -9.81])
        np.testing.assert_array_equal(tubes[2]["g"], [0, 0, 2.5])
        self.assertEqual(tubes[1]["straight_length"], 0.0)

    def test_defaults_of_missing_parameters(self):
        path = self.write("minimal.yaml", "tubes:\n  0: {L: 150, curved_len: 30}\n")
        self.assert_same_raw(path)
        self.assert_same_tubes(path)

    def test_path_types_and_relative_paths(self):
        expected = setup_tubes(self.inline)
        self.assertEqual(pyctr_cpp.setup_tubes(pathlib.Path(self.inline)), expected)
        cwd = os.getcwd()
        try:
            os.chdir(self.dir)  # tube files are relative to the config file, the config file to the cwd
            self.assertEqual(pyctr_cpp.setup_tubes("robot.yaml"), expected)
            os.chdir(os.path.join(self.dir, "tubes"))
            self.assertEqual(pyctr_cpp.setup_tubes("../robot.yaml"), expected)
        finally:
            os.chdir(cwd)

    def test_empty_tubes(self):
        path = self.write("empty.yaml", "tubes: {}\n")
        self.assertEqual(pyctr_cpp.setup_tubes(path), setup_tubes(path))
        self.assertEqual(pyctr_cpp.load_tubes(path), [])

    def test_invalid_expressions(self):
        # rejected by Python's eval too
        for value in ["1 +", "(1", "1)", "1 / 0", "abc", "1e", "1e+", "inf", "nan", ". * 2", "1 2", "''"]:
            path = self.write("bad.yaml", f"tubes:\n  0: {{L: 1, curved_len: {value}}}\n")
            with self.subTest(value=value):
                with self.assertRaises(Exception):
                    setup_tubes(path)
                with self.assertRaisesRegex(ValueError, "curved_len"):
                    pyctr_cpp.setup_tubes(path)
        # accepted by Python's eval, which runs any code, but not by the C++ evaluator
        for value in ["2 ** 3", "abs(-1)", "1_000 * 2", "[1, 2][0] + 0", "__import__('os').getpid() * 0"]:
            path = self.write("bad.yaml", f"tubes:\n  0:\n    curved_len: \"{value}\"\n")
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "cannot evaluate"):
                    pyctr_cpp.setup_tubes(path)

    def test_python_literals_and_whitespace(self):
        # YAML values (escapes in double quotes) that both accept, with the same result
        accepted = ["00", "017.5", "017e1", "'00 * 3'", ".inf", "-.inf", "+.INF", '"1\\f+ 1"', '"(1\\n+ 1)"',
                    '"\\n1 + 1\\n"', "|\n      1 + 1\n", "'" + "-" * 3001 + "1'", "'" + "(" * 200 + "1" + ")" * 200 + "'"]
        for value in accepted:
            path = self.write("ok.yaml", f"tubes:\n  0:\n    L: 1\n    curved_len: {value}\n")
            with self.subTest(value=value[:20]):
                self.assert_same_raw(path)
        path = self.write("nan.yaml", "tubes:\n  0: {L: 1, curved_len: .NaN}\n")
        self.assertTrue(np.isnan(setup_tubes(path)[0]["curved_len"]))
        self.assertTrue(np.isnan(pyctr_cpp.setup_tubes(path)[0]["curved_len"]))
        # unquoted 017 is octal 15 for PyYAML (YAML 1.1), C++ rejects it instead of reading 17
        path = self.write("octal.yaml", "tubes:\n  0: {L: 1, curved_len: 017}\n")
        self.assertEqual(setup_tubes(path)[0]["curved_len"], 15)
        with self.assertRaisesRegex(ValueError, "leading zeros"):
            pyctr_cpp.setup_tubes(path)
        # rejected by both: leading zeros, quoted .inf, vertical tab, a newline outside parentheses, more than 200
        # nested parentheses (C++ must not overflow the stack)
        rejected = ["09", "'017'", "1.0 * 017", "'.inf'", '"1\\v+ 1"', '"1\\n+ 1"',
                    "'" + "(" * 201 + "1" + ")" * 201 + "'", "'" + "(" * 200000 + "1" + ")" * 200000 + "'"]
        for value in rejected:
            path = self.write("bad.yaml", f"tubes:\n  0:\n    L: 1\n    curved_len: {value}\n")
            with self.subTest(value=value[:20]):
                with self.assertRaises(Exception):
                    setup_tubes(path)
                with self.assertRaisesRegex(ValueError, "curved_len"):
                    pyctr_cpp.setup_tubes(path)

    def test_merge_keys(self):
        # own keys override merged keys, earlier merged mappings override later ones, nested merges, and an
        # overridden invalid value is never evaluated
        config = """
base: &base {E: 1e9, G: '2 * 1e8', rho: 1000, kappa: abc}
stiff: &stiff {<<: *base, E: 3e9, r_outer: 2e-3}
tubes:
    0: {<<: *base, L: 100, curved_len: 20, kappa: 5}
    1:
        <<: [*stiff, {r_outer: 9, r_inner: 1e-3, rho: 7}]
        L: 120
        curved_len: 10
        kappa: 1
"""
        path = self.write("merge.yaml", config)
        raw = self.assert_same_raw(path)
        self.assertEqual((raw[1]["E"], raw[1]["r_outer"], raw[1]["rho"]), (3e9, 2e-3, 1000))
        self.assert_same_tubes(path)
        path = self.write("bad.yaml", "tubes:\n  0: {<<: 3, L: 1, curved_len: 1}\n")
        with self.assertRaises(Exception):
            setup_tubes(path)
        with self.assertRaisesRegex(ValueError, "'<<' must be a mapping"):
            pyctr_cpp.setup_tubes(path)

    def test_invalid_files(self):
        with self.assertRaises(FileNotFoundError):
            setup_tubes(os.path.join(self.dir, "missing.yaml"))
        with self.assertRaisesRegex(RuntimeError, "missing.yaml"):
            pyctr_cpp.setup_tubes(os.path.join(self.dir, "missing.yaml"))
        path = self.write("missing_tube.yaml", "tubes:\n  0: tubes/missing.yaml\n")
        with self.assertRaisesRegex(RuntimeError, "missing.yaml"):
            pyctr_cpp.setup_tubes(path)
        for text in ["", "other: 1\n", "tubes: 3\n", "tubes: [a]\n", "tubes:\n  0: [1, 2]\n", "tubes:\n  0: {L: }\n",
                     "tubes:\n  0: {L: [1, 2]}\n"]:
            path = self.write("bad.yaml", text)
            with self.subTest(text=text):
                with self.assertRaises(Exception):
                    load_continous_ctcr_model(path)
                with self.assertRaises(ValueError):
                    pyctr_cpp.load_tubes(path)
        path = self.write("no_length.yaml", "tubes:\n  0: {curved_len: 10}\n")
        with self.assertRaises(KeyError):
            load_continous_ctcr_model(path)
        with self.assertRaisesRegex(ValueError, "'L' is missing"):
            pyctr_cpp.load_tubes(path)

    def test_dicts_accepted_by_cpp_rods(self):
        rod_class = getattr(pyctr_cpp, "CurvedCosseratRod", None)
        if rod_class is None:
            self.skipTest("pyctr_cpp.CurvedCosseratRod is not built")
        for d in pyctr_cpp.load_tubes(self.inline):
            rod_class(d)


if __name__ == "__main__":
    unittest.main()
