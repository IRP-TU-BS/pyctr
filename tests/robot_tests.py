import unittest

import numpy as np
import yaml

from pyctcr import cosserat_rod_force_along
from pyctcr.robots import ConcentricTubeContinuumRobot

def convert_config_to_tuple(config_dict):
    """
    A aid function to deconstruct a dict with tube parameters into a list
    :param config_dict:
    :return:
    """
    tubes = config_dict['tubes']
    tubes_config = []
    for i, key in enumerate(tubes.keys()):
        tubes_config.append([])
        for prop in ['len', 'arc_len', 'u_0', 'flex_rigidity', 'tor_rigidity']: # keep order
            tubes_config[i].append(tubes[key][prop])

    return tubes_config

def load_yaml_file_to_dict(path):
    """
    A simple wrapper function to load a yaml file to a dictionary
    :param path:
    :return:
    """
    with open(path) as f:
        dict_data = yaml.load(f, Loader=yaml.FullLoader)
    return dict_data

def convert_tuple_config(config_dict, prefix = "../"):
    tubes = config_dict['tubes']
    tubes_config = []
    for i, key in enumerate(tubes.keys()):
        if isinstance(tubes[key], str):
            tube = load_yaml_file_to_dict(prefix+tubes[key])
        else:
            tube = tubes[key]
        for k in tube.keys():
            if isinstance( tube[k], str ):
                tube[k] = eval(tube[k])
        tubes_config.append(tube)

    return tubes_config


def load_tube_yaml(config_dict):
    tubes = load_yaml_file_to_dict(config_dict)['tubes']
    tubes_config = []
    for i, key in enumerate(tubes.keys()):
        if isinstance(tubes[key], str):
            tubes_config.append(load_yaml_file_to_dict(tubes[key]))
        else:
            tubes_config.append(tubes[key])

    return tubes_config

def setup_tubes(config_path):
    config_dict = load_yaml_file_to_dict(config_path)
    return convert_tuple_config(config_dict)


class RobotTestCase(unittest.TestCase):
    def test_something(self):
        tube_conf = setup_tubes("../example_robots/ctr_robot.yaml")

        rods = []
        tubes_lenghts = []

        for rod_conf in tube_conf:
            rod_conf['s'] = 1 * 1e-3
            rod_len = rod_conf['L'] * 1e-3
            rod_conf['L'] = rod_len
            rod_conf['straight_length'] = rod_len - rod_conf['curved_len'] * 1e-3
            rod_conf['curved_len'] = rod_conf['curved_len'] * 1e-3
            tubes_lenghts.append(rod_len)
            rod = cosserat_rod_force_along.CurvedCosseratRodExt(rod_conf)
            p0 = np.array([[0, 0, 0]])
            R0 = np.eye(3)
            rod.set_initial_conditions(p0, R0)
            rods.append(rod)
        ctr = ConcentricTubeContinuumRobot(rods)
        # test fwd kin
        ctr.fwd_kinematic()
        ctr.fwd_static([0,0.1,0,0,0,0])
        ctr.push_end([0,0,0,0,0,0])

if __name__ == '__main__':
    unittest.main()
