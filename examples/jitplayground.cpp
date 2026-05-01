#include <mfem.hpp>

#include "../fem/dfem/util.hpp"

#include <proteus/CppJitModule.h>

#include "jitplayground.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace util
{
constexpr std::string_view Dirname(std::string_view path)
{
   const size_t last_sep = path.find_last_of("/\\");
   if (last_sep == std::string_view::npos) { return {}; }
   return path.substr(0, last_sep);
}

constexpr std::string_view thisFileDir = Dirname(__FILE__);
}

template <typename T>
static std::string TypeNameString()
{
   return std::string(mfem::future::get_type_name<T>());
}

template <typename Tuple, size_t... Is>
static auto ParamTypeStringsImpl(std::index_sequence<Is...>)
{
   return std::array<std::string, sizeof...(Is)>
   {
      TypeNameString<std::remove_reference_t<decltype(mfem::future::get<Is>(std::declval<Tuple&>()))>>()...
   };
}

template <typename Tuple>
static auto ParamTypeStrings()
{
   return ParamTypeStringsImpl<Tuple>(
             std::make_index_sequence<mfem::future::tuple_size<Tuple>::value> {});
}

static std::string_view Trim(std::string_view s)
{
   size_t begin = 0;
   while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])))
   {
      ++begin;
   }
   size_t end = s.size();
   while (end > begin &&
          std::isspace(static_cast<unsigned char>(s[end - 1])))
   {
      --end;
   }
   return s.substr(begin, end - begin);
}

static bool IsValidIdentifier(std::string_view s)
{
   if (s.empty()) { return false; }
   const unsigned char c0 = static_cast<unsigned char>(s[0]);
   if (!(std::isalpha(c0) || c0 == '_')) { return false; }
   for (size_t i = 1; i < s.size(); ++i)
   {
      const unsigned char c = static_cast<unsigned char>(s[i]);
      if (!(std::isalnum(c) || c == '_')) { return false; }
   }
   return true;
}

static bool ParseJitDirective(std::string_view line,
                              std::string &type,
                              std::string &var,
                              std::string &kind)
{
   const size_t jit_pos = line.find("$JIT");
   if (jit_pos == std::string_view::npos) { return false; }

   const size_t open = line.find('[', jit_pos);
   const size_t close = line.find(']', jit_pos);
   MFEM_VERIFY(open != std::string_view::npos &&
               close != std::string_view::npos &&
               close > open,
               "malformed $JIT directive (expected brackets): " << line);

   const std::string_view payload = line.substr(open + 1, close - open - 1);
   const size_t comma1 = payload.find(',');
   const size_t comma2 = (comma1 == std::string_view::npos)
                         ? std::string_view::npos
                         : payload.find(',', comma1 + 1);
   MFEM_VERIFY(comma1 != std::string_view::npos &&
               comma2 != std::string_view::npos,
               "malformed $JIT directive (expected 3 comma-separated fields): "
               << line);

   const std::string_view f0 = Trim(payload.substr(0, comma1));
   const std::string_view f1 = Trim(payload.substr(comma1 + 1,
                                                   comma2 - comma1 - 1));
   const std::string_view f2 = Trim(payload.substr(comma2 + 1));
   MFEM_VERIFY(!f0.empty() && !f1.empty() && !f2.empty(),
               "malformed $JIT directive (empty field): " << line);

   type.assign(f0);
   var.assign(f1);
   kind.assign(f2);
   return true;
}

static std::string ReadFileOrEmpty(const std::string &fn)
{
   std::ifstream file(fn);
   if (!file.is_open())
   {
      std::cerr << "could not open file " << fn << "\n";
      return {};
   }
   std::stringstream buffer;
   buffer << file.rdbuf();
   return buffer.str();
}

static std::vector<std::string> ExtractJitVarNames(const std::string
                                                   &kernel_code)
{
   std::stringstream ss(kernel_code);
   std::string line;
   std::vector<std::string> var_names;
   std::unordered_set<std::string> seen_vars;

   while (std::getline(ss, line))
   {
      std::string type, var, kind;
      if (ParseJitDirective(line, type, var, kind))
      {
         MFEM_VERIFY(IsValidIdentifier(var),
                     "$JIT variable must be a valid identifier: " << var);
         MFEM_VERIFY(seen_vars.insert(var).second,
                     "duplicate $JIT variable name: " << var);
         var_names.push_back(var);
      }
   }
   return var_names;
}

static std::string RewriteKernelForJit(std::string kernel_code,
                                       const std::vector<std::string> &jit_values)
{
   std::stringstream ss(kernel_code);
   std::string line;

   std::string out;
   out.reserve(kernel_code.size() + 128);

   bool have_pending = false;
   size_t pending_index = 0;
   std::string pending_type;
   std::string pending_var;
   std::unordered_set<std::string> seen_vars;

   while (std::getline(ss, line))
   {
      line.push_back('\n');

      if (have_pending)
      {
         MFEM_VERIFY(pending_index < jit_values.size(),
                     "not enough JIT values provided");
         const size_t indent_end = line.find_first_not_of(" \t");
         const std::string indent =
            (indent_end == std::string::npos) ? std::string() :
            line.substr(0, indent_end);
         out += indent + "const " + pending_type + " " + pending_var + " = " +
                jit_values[pending_index] + ";\n";
         have_pending = false;
         ++pending_index;
         continue;
      }

      std::string type, var, kind;
      if (ParseJitDirective(line, type, var, kind))
      {
         MFEM_VERIFY(IsValidIdentifier(var),
                     "$JIT variable must be a valid identifier: " << var);
         MFEM_VERIFY(kind == "generic",
                     "unsupported $JIT kind: " << kind);
         MFEM_VERIFY(seen_vars.insert(var).second,
                     "duplicate $JIT variable name: " << var);

         pending_type = std::move(type);
         pending_var = std::move(var);
         have_pending = true;
         continue; // drop directive line
      }

      out += line;
   }

   MFEM_VERIFY(!have_pending,
               "$JIT directive must annotate a following line");
   MFEM_VERIFY(jit_values.size() == pending_index,
               "JIT value count must match number of $JIT directives");
   return out;
}

static std::string GeneratedOutputPath(std::string_view original_path)
{
   const size_t last_sep = original_path.find_last_of("/\\");
   const size_t dot = original_path.find_last_of('.');
   const bool dot_in_filename =
      (dot != std::string_view::npos) &&
      (last_sep == std::string_view::npos || dot > last_sep);

   const std::string_view base =
      dot_in_filename ? original_path.substr(0, dot) : original_path;
   return std::string(base) + "_generated.hpp";
}

static void WriteFileOrWarn(const std::string &path,
                            const std::string &contents)
{
   std::ofstream out(path);
   if (!out.is_open())
   {
      std::cerr << "could not write generated file " << path << "\n";
      return;
   }
   out << contents;
}

class JitQFunction
{
public:
   template <typename ImplT, size_t N>
   JitQFunction(ImplT, const std::string &fn,
                const std::array<bool, N> &activity_map)
   {
      using qf_signature = typename
                           mfem::future::get_function_signature<
                           decltype(&ImplT::operator())>::type;
      using qf_param_ts = typename qf_signature::parameter_ts;
      constexpr size_t nparams = mfem::future::tuple_size<qf_param_ts>::value;
      static_assert(N == nparams, "activity_map size must match qfunc arity");

      this->fn = fn;
      this->nparams = nparams;
      this->activity_map.reserve(N);
      for (size_t i = 0; i < N; ++i)
      {
         this->activity_map.push_back(activity_map[i]);
      }
      {
         const auto param_types_arr = ParamTypeStrings<qf_param_ts>();
         this->param_types.assign(param_types_arr.begin(), param_types_arr.end());
      }
      this->return_type = TypeNameString<typename qf_signature::return_t>();
      this->return_is_void = std::is_same_v<typename qf_signature::return_t, void>;
      this->impl_type_name = TypeNameString<ImplT>();
      this->jit_var_names = ExtractJitVarNames(ReadFileOrEmpty(fn));
   }

   template <typename ReturnT, typename... Args>
   ReturnT run(std::string_view name,
               std::initializer_list<std::pair<std::string_view, std::string_view>> jit_values,
               Args&&... args)
   {
      auto ordered_values = MatchJitValues(jit_values);
      auto &mod = GetOrCreateModule(ordered_values);
      auto &instance = mod.instantiate(std::string(name), std::string());
      return instance.template run<ReturnT>(std::forward<Args>(args)...);
   }

   template <typename ReturnT, typename... Args>
   ReturnT run_primal(
      std::initializer_list<std::pair<std::string_view, std::string_view>> jit_values,
      Args&&... args)
   {
      return run<ReturnT>(qfunc_name, jit_values,
                          std::forward<Args>(args)...);
   }

   template <typename ReturnT, typename... Args>
   ReturnT run_derivative(
      std::initializer_list<std::pair<std::string_view, std::string_view>> jit_values,
      Args&&... args)
   {
      return run<ReturnT>(qfunc_name + "_fwddiff", jit_values,
                          std::forward<Args>(args)...);
   }

private:
   std::vector<std::string_view> MatchJitValues(
      std::initializer_list<std::pair<std::string_view, std::string_view>>
      named_values) const
   {
      std::unordered_map<std::string_view, std::string_view> value_map;
      for (const auto &[name, value] : named_values)
      {
         value_map[name] = value;
      }

      std::vector<std::string_view> ordered_values;
      ordered_values.reserve(jit_var_names.size());
      for (const auto &var_name : jit_var_names)
      {
         auto it = value_map.find(var_name);
         MFEM_VERIFY(it != value_map.end(),
                     "missing JIT value for variable: " << var_name);
         ordered_values.push_back(it->second);
      }

      MFEM_VERIFY(ordered_values.size() == named_values.size(),
                  "provided " << named_values.size() << " JIT values but expected "
                  << jit_var_names.size());
      return ordered_values;
   }


   std::string BuildModuleCode(const std::vector<std::string> &jit_values) const
   {
      std::string module_code =
         RewriteKernelForJit(ReadFileOrEmpty(fn), jit_values);
      module_code += "\n\n";
      module_code += "// --- generated ---\n";
      module_code +=
         "template <typename return_type, typename... Args>\n"
         "return_type __enzyme_fwddiff(Args...);\n"
         "\n"
         "extern int enzyme_const;\n"
         "extern int enzyme_dup;\n"
         "\n";

      // Generate a primal wrapper with the requested symbol name, so the kernel
      // header can just define the qfunc as a functor.
      //
      // Note: Proteus instantiates entrypoints via `qfunc_wrapper<>(...)` even
      // when there are no user template args, so keep the wrapper itself a
      // template (with a default parameter) while still doing literal `$JIT`
      // replacements in the kernel code.
      module_code += "template <typename = void>\n";
      module_code += return_type + " " +
                     std::string(qfunc_name) + "(";
      bool first = true;
      for (size_t i = 0; i < nparams; ++i)
      {
         if (!first) { module_code += ", "; }
         first = false;
         module_code += param_types[i] + " Arg" + std::to_string(i);
      }
      module_code += ")\n";
      module_code += "{\n";
      module_code += "   " + impl_type_name + " qf;\n";
      if (return_is_void)
      {
         module_code += "   ";
      }
      else
      {
         module_code += "   return ";
      }
      module_code += "qf(";
      for (size_t i = 0; i < nparams; ++i)
      {
         if (i) { module_code += ", "; }
         module_code += "Arg" + std::to_string(i);
      }
      module_code += ");\n";
      module_code += "}\n\n";

      module_code += "template <typename = void>\n";
      module_code += return_type + " " +
                     std::string(qfunc_name) + "_fwddiff(";

      first = true;
      for (size_t i = 0; i < nparams; ++i)
      {
         if (!first) { module_code += ", "; }
         first = false;
         module_code += param_types[i] + " Arg" + std::to_string(i);
         if (activity_map[i])
         {
            module_code += ", " + param_types[i] + " dArg" + std::to_string(i);
         }
      }
      module_code += ")\n";
      module_code += "{\n";
      if (return_is_void)
      {
         module_code += "   __enzyme_fwddiff<void>(\n";
      }
      else
      {
         module_code += "   return __enzyme_fwddiff<" +
                        return_type + ">(\n";
      }
      module_code += "      (void*)" + std::string(qfunc_name) + "<>";
      module_code += ",\n";
      for (size_t i = 0; i < nparams; ++i)
      {
         if (activity_map[i])
         {
            module_code += "      enzyme_dup, Arg" + std::to_string(i) +
                           ", dArg" + std::to_string(i);
         }
         else
         {
            module_code += "      enzyme_const, Arg" + std::to_string(i);
         }
         module_code += (i + 1 == nparams) ? ");\n" : ",\n";
      }
      module_code += "}\n";

      WriteFileOrWarn(GeneratedOutputPath(fn), module_code);
      return module_code;
   }

   proteus::CppJitModule &GetOrCreateModule(
      const std::vector<std::string_view> &jit_values)
   {
      std::string key;
      for (const auto &val : jit_values)
      {
         if (!key.empty()) { key += ","; }
         key += val;
      }

      auto it = modules.find(key);
      if (it != modules.end())
      {
         return *it->second;
      }

      std::vector<std::string> values(jit_values.begin(), jit_values.end());
      std::string code = BuildModuleCode(values);
      auto mod = std::make_unique<proteus::CppJitModule>("host", code,
                                                         DefaultExtraArgs());
      auto [inserted, ok] = modules.emplace(key, std::move(mod));
      MFEM_VERIFY(ok, "failed to cache JIT module");
      return *inserted->second;
   }

   static std::vector<std::string> DefaultExtraArgs()
   {
      return {"-fplugin=/Users/andrej1/local/enzyme/lib/ClangEnzyme-20.dylib"};
   }

   std::string qfunc_name = "qfunc_wrapper";
   std::string fn;
   size_t nparams = 0;
   std::vector<bool> activity_map;
   std::vector<std::string> param_types;
   std::string return_type;
   bool return_is_void = false;
   std::string impl_type_name;
   std::vector<std::string> jit_var_names;
   std::unordered_map<std::string, std::unique_ptr<proteus::CppJitModule>> modules;
};

int main()
{
   const size_t N = 4;
   const size_t M = 5;
   const double A = 123.4;

   std::vector<double> X(N);
   std::vector<double> Y(N);
   for (size_t i = 0; i < N; ++i)
   {
      X[i] = static_cast<double>(i + 1);
      Y[i] = static_cast<double>(N - i);
   }

   // // >>> user interface calls
   // const std::string kernel_path = std::string(util::thisFileDir) +
   //                                 "/jitplayground.hpp";
   // JitQFunction qf(daxpy_op{}, kernel_path, std::array{false, true, false});
   // // <<< user interface calls

   // // this will happen internally in dFEM

   daxpy_op op;
   printf("\n\nfunction call\n");
   op(&A, X.data(), Y.data(), &N);

   // reset X for the derivative test
   for (size_t i = 0; i < N; ++i)
   {
      X[i] = static_cast<double>(i + 1);
      Y[i] = static_cast<double>(N - i);
   }

   std::vector<double> dX(N, 1.0);
   printf("\n\nforward diff call\n");
   daxpy_op_fwddiff(&A, X.data(), dX.data(), Y.data(), &N);

   std::vector<double> dX_manual(N, A);

   printf("\n\nderivative checks\n");
   std::cout << "dX: ";
   for (size_t i = 0; i < N; ++i)
   {
      std::cout << dX[i] << (i + 1 == N ? '\n' : ' ');
   }

   std::cout << "dX_manual: ";
   for (size_t i = 0; i < N; ++i)
   {
      std::cout << dX_manual[i] << (i + 1 == N ? '\n' : ' ');
   }

   double max_abs_err = 0.0;
   for (size_t i = 0; i < N; ++i)
   {
      max_abs_err = std::max(max_abs_err, std::abs(dX[i] - dX_manual[i]));
   }
   std::cout << "max |dX - dX_manual| = " << max_abs_err << "\n";

   return 0;
}
