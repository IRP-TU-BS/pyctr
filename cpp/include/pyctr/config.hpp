// Loading robots from YAML files, port of pyctr/yaml_to_model.py.
#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "pyctr/core.hpp"

namespace pyctr {

// Raw parameters of every tube in the order of the config file, in the units of the files (no conversion).
// Tube entries are file names relative to the directory of the config file, or inline mappings.
// String values are arithmetic expressions such as "0.5 * 1e-3" (numbers, + - * /, unary + -, parentheses).
// YAML merge keys (<<: *base) in tube mappings are resolved as by PyYAML.
std::vector<std::map<std::string, double>> setup_tubes(const std::filesystem::path& config_path);

// Tube parameters with the conversions of load_continous_ctcr_model: s = 1 mm, L and curved_len in mm,
// straight_length = L - curved_len, scalar g -> gravity [0, 0, -g]. Missing parameters keep the defaults.
std::vector<TubeParameters> load_tubes(const std::filesystem::path& config_path);

}  // namespace pyctr
