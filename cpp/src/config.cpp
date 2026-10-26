#include "pyctr/config.hpp"

#include <yaml-cpp/yaml.h>

#include <cctype>
#include <charconv>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>

namespace pyctr {
namespace {

// Replaces Python's eval of string values, which would run any code. Recursive descent:
//   expr = term {("+" | "-") term}, term = factor {("*" | "/") factor},
//   factor = {"+" | "-"} ("(" expr ")" | number)
// Left associative with the precedence of Python, so the results are the same doubles as with eval, as long as
// every integer operand and intermediate result is exactly representable as double (Python's int is exact).
// Whitespace as in Python: space, tab and form feed; newlines only inside parentheses and around the expression.
class Expression {
 public:
  explicit Expression(const std::string& text) : text_(text) {}

  double value() {
    skip_space(true);
    const double v = expr();
    skip_space(true);
    if (pos_ != text_.size()) fail("unexpected character");
    return v;
  }

 private:
  double expr() {
    double v = term();
    while (true) {
      if (accept('+')) {
        v += term();
      } else if (accept('-')) {
        v -= term();
      } else {
        return v;
      }
    }
  }

  double term() {
    double v = factor();
    while (true) {
      if (accept('*')) {
        v *= factor();
      } else if (accept('/')) {
        const double d = factor();
        if (d == 0) fail("division by zero");  // Python raises ZeroDivisionError
        v /= d;
      } else {
        return v;
      }
    }
  }

  // Unary signs in a loop and a limit on the parentheses, so that deep nesting cannot overflow the stack
  double factor() {
    bool negative = false;
    while (true) {
      if (accept('-')) {
        negative = !negative;
      } else if (!accept('+')) {
        break;
      }
    }
    double v = 0;
    if (accept('(')) {
      if (++parens_ > 200) fail("more than 200 nested parentheses");  // the limit of the Python parser
      v = expr();
      if (!accept(')')) fail("expected ')'");
      --parens_;
    } else {
      v = number();
    }
    return negative ? -v : v;
  }

  // Python float literal without underscores: digits [. digits] [(e|E) [+|-] digits], at least one digit before
  // the exponent. Words such as inf or nan are rejected as by eval.
  double number() {
    skip_space(parens_ > 0);
    const size_t start = pos_;
    size_t n = digits();
    bool integer = true;
    if (peek('.')) {
      ++pos_;
      n += digits();
      integer = false;
    }
    if (n == 0) fail("expected a number");
    if (peek('e') || peek('E')) {
      ++pos_;
      if (peek('+') || peek('-')) ++pos_;
      if (digits() == 0) fail("expected the digits of the exponent");
      integer = false;
    }
    // Python rejects 017 (unquoted, PyYAML reads it as octal 15), but accepts 00 and 017.5
    if (integer && text_[start] == '0' && text_.find_first_not_of('0', start) < pos_) fail("leading zeros in an integer");
    double v = 0;
    const auto r = std::from_chars(text_.data() + start, text_.data() + pos_, v);
    if (r.ec != std::errc()) fail("number out of the range of double");  // Python gives inf or 0 here
    return v;
  }

  size_t digits() {
    const size_t start = pos_;
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    return pos_ - start;
  }

  bool peek(char c) const { return pos_ < text_.size() && text_[pos_] == c; }

  bool accept(char c) {
    skip_space(parens_ > 0);
    if (!peek(c)) return false;
    ++pos_;
    return true;
  }

  void skip_space(bool newlines) {
    const char* space = newlines ? " \t\f\n\r" : " \t\f";
    while (pos_ < text_.size() && text_[pos_] != '\0' && std::strchr(space, text_[pos_])) ++pos_;
  }

  [[noreturn]] void fail(const std::string& reason) const {
    throw std::invalid_argument("cannot evaluate '" + text_ + "': " + reason + " at position " +
                                std::to_string(pos_) + " (allowed: numbers, + - * /, parentheses)");
  }

  const std::string& text_;
  size_t pos_ = 0;
  int parens_ = 0;
};

YAML::Node load_file(const std::filesystem::path& path) {
  try {
    return YAML::LoadFile(path.string());
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(path.string() + ": " + e.what());
  }
}

// Unquoted numbers and strings are both evaluated as expressions. Only the unquoted YAML 1.1 words for infinity
// and NaN are numbers for PyYAML that eval would reject.
double parameter_value(const YAML::Node& node, const std::string& key) {
  if (!node.IsScalar()) throw std::invalid_argument("tube parameter '" + key + "' must be a number or an expression");
  const std::string& s = node.Scalar();
  if (node.Tag() != "!") {  // "!" is the tag of quoted scalars
    static const std::regex inf(R"([-+]?\.(inf|Inf|INF))");
    static const std::regex nan(R"(\.(nan|NaN|NAN))");
    if (std::regex_match(s, inf)) return (s[0] == '-' ? -1 : 1) * std::numeric_limits<double>::infinity();
    if (std::regex_match(s, nan)) return std::numeric_limits<double>::quiet_NaN();
  }
  try {
    return Expression(s).value();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument("tube parameter '" + key + "': " + e.what());
  }
}

// YAML 1.1 merge keys (<<: *base) as in PyYAML: the own keys of a mapping override the merged keys, earlier merged
// mappings override later ones. Values are evaluated only after merging, as in Python.
void collect(const YAML::Node& map, std::map<std::string, YAML::Node>& values) {
  std::map<std::string, YAML::Node> own;
  for (const auto& p : map) {
    if (p.first.Scalar() == "<<") continue;
    own.erase(p.first.Scalar());  // the last duplicate key wins; YAML::Node::operator= would change the document
    own.emplace(p.first.Scalar(), p.second);
  }
  values.insert(own.begin(), own.end());
  const YAML::Node merge = map["<<"];
  if (!merge) return;
  if (merge.IsMap()) return collect(merge, values);
  if (!merge.IsSequence()) throw std::invalid_argument("'<<' must be a mapping or a list of mappings");
  for (const auto& m : merge) {
    if (!m.IsMap()) throw std::invalid_argument("'<<' must be a mapping or a list of mappings");
    collect(m, values);
  }
}

}  // namespace

std::vector<std::map<std::string, double>> setup_tubes(const std::filesystem::path& config_path) {
  const YAML::Node config = load_file(config_path);
  const YAML::Node tubes = config.IsMap() && config["tubes"] ? config["tubes"] : YAML::Node();
  if (!tubes.IsMap()) throw std::invalid_argument(config_path.string() + ": 'tubes' must be a mapping");
  std::vector<std::map<std::string, double>> out;
  for (const auto& entry : tubes) {
    YAML::Node tube = entry.second;
    if (tube.IsScalar()) tube = load_file(config_path.parent_path() / tube.Scalar());
    if (!tube.IsMap()) {
      throw std::invalid_argument(config_path.string() + ": tube '" + entry.first.Scalar() +
                                  "' must be a file name or a mapping of parameters");
    }
    std::map<std::string, YAML::Node> values;
    collect(tube, values);
    std::map<std::string, double> params;
    for (const auto& [key, value] : values) params[key] = parameter_value(value, key);
    out.push_back(params);
  }
  return out;
}

std::vector<TubeParameters> load_tubes(const std::filesystem::path& config_path) {
  std::vector<TubeParameters> out;
  for (const auto& raw : setup_tubes(config_path)) {
    auto required = [&](const char* key) {
      const auto it = raw.find(key);
      if (it == raw.end()) throw std::invalid_argument(config_path.string() + ": tube parameter '" + key + "' is missing");
      return it->second;
    };
    auto optional = [&](const char* key, double& value) {
      const auto it = raw.find(key);
      if (it != raw.end()) value = it->second;
    };
    // other keys are ignored by the models, as in Python
    TubeParameters t;
    optional("E", t.E);
    optional("G", t.G);
    optional("r_inner", t.r_inner);
    optional("r_outer", t.r_outer);
    optional("rho", t.rho);
    optional("kappa", t.kappa);
    const auto g = raw.find("g");
    if (g != raw.end()) t.gravity = Vec3(0, 0, -g->second);
    t.s = 1 * 1e-3;
    t.L = required("L") * 1e-3;
    t.straight_length = t.L - required("curved_len") * 1e-3;
    out.push_back(t);
  }
  return out;
}

}  // namespace pyctr
