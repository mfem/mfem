#include <mfem.hpp>
#include "nlohmann/json.hpp"
#include "minja.hpp"
#include "myqfunction.hpp"

using namespace mfem;
using namespace mfem::future;

template<class T>
struct remove_cvref
{
   using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename qf_t>
auto process(qf_t qf)
{
   using qfsig = typename create_function_signature<qf_t>::type;
   using qfpar_t = typename qfsig::parameter_ts;
   using qfout_t = typename qfsig::return_t;

   auto qfparams = decay_tuple<qfpar_t> {};

   auto in_str = apply([](auto&&... arg)
   {
      return std::vector<std::string>
      {
         std::string(get_type_name<typename remove_cvref<decltype(arg)>::type>())...
      };
   }, qfparams);

   std::vector<std::string> out_str
   {
      std::string(get_type_name<typename remove_cvref<qfout_t>::type>())
   };

   return std::tuple{in_str, out_str};
}

int main()
{
   // load the kernel template
   std::ifstream
   kernel_istream("/Users/andrej1/repos/mfem/examples/kernel_skeleton.jinja");
   if (!kernel_istream.is_open())
   {
      std::cerr << "error opening jinja template file" << std::endl;
      return 1;
   }
   std::stringstream buffer;
   buffer << kernel_istream.rdbuf();

   std::string fileContent = buffer.str();
   auto kernel_tmpl = minja::Parser::parse(buffer.str(), /* options= */ {});

   auto [in_str, out_str] = process(myqfunction0);

   for (auto &v : in_str)
   {
      std::cout << v << " ";
   }
   std::cout << std::endl;

   const size_t DUMMY_STRIDE = 64*32*32;
   const size_t basis_p_1d = 2;

   json context_json{};
   context_json["kernel_name"] = "demo";

   context_json["spaces"].push_back(
   {
      {"P_1D", basis_p_1d},
      {"dim", 3},
      {"needs_value", true},
      {"needs_grad", true},
   });

   context_json["spaces"].push_back(
   {
      {"P_1D", basis_p_1d},
   });

   context_json["inputs"].push_back(
   {
      {"name", "potential"},
      {"space_idx", 0},
      {"num_comp", 1},
      {"comp_stride", DUMMY_STRIDE},
      {"eval_grad", true},
   });

   context_json["inputs"].push_back(
   {
      {"name", "weights"},
      {"space_idx", 0},
      {"num_comp", 1},
      {"comp_stride", DUMMY_STRIDE},
      {"is_qdata", true},
   });

   context_json["outputs"].push_back(
   {
      {"name", "solution"},
      {"space_idx", 0},
      {"num_comp", 1},
      {"comp_stride", DUMMY_STRIDE},
      {"eval_grad", true},
   });

   const size_t nqf = 1;
   const std::vector<std::string> qfunc_names = {"myqfunction0"};
   const std::vector<std::vector<size_t>> qfunc_inputs = {{0, 1, 2}};

   for (size_t i = 0; i < nqf; i++)
   {
      json inarr = json::array();
      for (size_t j = 0; j < qfunc_inputs[i].size(); j++)
      {
         inarr.push_back(
         {
            {"index", j},
            {"datatype", in_str[j]}
         });
      }

      context_json["qfuncs"].push_back(
      {
         {"name", qfunc_names[i]},
         {"inputs", inarr}
      });
   }

   std::cout << context_json.dump(2) << std::endl;

   auto context = minja::Context::make(context_json);
   auto kernel_source = kernel_tmpl->render(context);

   std::cout << ">>> generated kernel source\n"
             << kernel_source
             << "\n<<< generated kernel source\n"
             << std::endl;

   {
      // test casting
      std::vector<real_t> d(4);
      int i = 0;
      for (auto &v : d)
      {
         v = ++i;
      }

      mfem::future::tensor<real_t, 2, 2> *dudxi =
         reinterpret_cast<mfem::future::tensor<real_t, 2, 2> *>(d.data());

      std::cout << *dudxi << std::endl;
   }

   return 0;
}
