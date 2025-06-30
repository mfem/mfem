#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <unistd.h>

#define NVTX_COLOR ::gpu::nvtx::kMagenta
#include "incompressible_navier_nvtx.hpp"

///////////////////////////////////////////////////////////////////////////////
int navier(int argc, char *argv[], double &u, double &p, double &Ψ);

///////////////////////////////////////////////////////////////////////////////
template <class T>
std::enable_if_t<!std::numeric_limits<T>::is_integer, bool>
AlmostEq(T x, T y, T tolerance = 100.0 * std::numeric_limits<T>::epsilon())
{
   const T neg = std::abs(x - y);
   constexpr T min = std::numeric_limits<T>::min();
   constexpr T eps = std::numeric_limits<T>::epsilon();
   const T min_abs = std::min(std::abs(x), std::abs(y));
   if (std::abs(min_abs) == 0.0) { return neg < eps; }
   return (neg / (1.0 + std::max(min, min_abs))) < tolerance;
}

///////////////////////////////////////////////////////////////////////////////
using char_uptr = std::unique_ptr<char[]>;
using args_ptr_t = std::vector<char_uptr>;
using args_t = std::vector<char *>;

///////////////////////////////////////////////////////////////////////////////
struct Results
{
   double u{}, p{}, Ψ {};
};

///////////////////////////////////////////////////////////////////////////////
struct Test
{
   static constexpr const char *binary = "incompNS_2Dtest ";
   static constexpr const char *common = "-no-vis -no-pv";
   const std::string options;
   const Results results;
   Test(const char *args, const Results &res):
      options(std::string(args) + " " + common), results(res)
   {
      dbg("options: {}", options.c_str());
      dbg("results: U={:.15e}, P={:.15e}, Ψ={:.15e}",
          results.u, results.p, results.Ψ);
   }
   std::string Command() const { return binary + options; }
};

///////////////////////////////////////////////////////////////////////////////
#if 1 // dot product reduction (miniapps/navier/incompNS_2Dtest.cpp#L182)
static const Test gold[] =
{
   {
      "-nx 9 -ny 3 -sr 0",
      { 3.056430866716070e-06, 1.504027632950462e-01, 7.132601171183242e-08 }
   },
   {
      "-nx 16 -ny 8 -sr 0",
      { 1.409287729554512e-05, 5.718053801962010e-01, 1.904938419441012e-07 }
   },
   // {
   //    "-nx 9 -ny 3 -sr 1",
   //    { 1.23258426138828e-05, 5.18207619597952e-01, 1.956175418199867e-07 }
   // },
   // {
   //    "-nx 9 -ny 3 -sr 2",
   //    { 4.74381190869396e-05, 1.80653923294262e+00, 5.90778654333642e-07 }
   // },
};
#else // Norml2 reduction
///////////////////////////////////////////////////////////////////////////////
const Test runs[] =
{
   {
      "-nx 9 -ny 3 -sr 0",
      { 1.746844586767688e-03, 3.874421956807944e-01, 2.670296315544763e-04 }
   },
   // 1.748265101955712e-03, 3.878179512284757e-01, 2.670693013280680e-04 //
   // Release { "-nx 9 -ny 3 -sr 1",
   //  { 1.232584261388279e-05, 5.182076195979519e-01, 1.956175418199866e-07 }
   //  },
   // { "-nx 9 -ny 3 -sr 2",
   //  { 4.743811908693962e-05, 1.806539232942622e+00, 5.907786543336419e-07 }
   //  },
};
#endif

///////////////////////////////////////////////////////////////////////////////
int NavierTest(const int k, const Test &run)
{
   dbg();
   static args_ptr_t args_ptr;
   args_t args;

   std::istringstream iss(run.Command());

   auto add_arg = [&](std::string token) -> char_uptr
   {
      auto arg_ptr = std::make_unique<char[]>(token.size() + 1);
      std::memcpy(arg_ptr.get(), token.c_str(), token.size() + 1);
      arg_ptr[token.size()] = '\0';
      return arg_ptr;
   };

   std::string token;
   while (iss >> token)
   {
      auto arg_ptr = add_arg(token);
      args.push_back(arg_ptr.get());
      args_ptr.emplace_back(std::move(arg_ptr));
   }
   args.push_back(nullptr);

   auto launch = [&args, &run, &k]() -> int
   {
      // dbg("Launching test #{}: \x1B[33m{}\x1B[m", k, gold.Command().c_str());
      Results res{};
      navier(args.size() - 1, args.data(), res.u, res.p, res.Ψ);
      // dbg("Results: U={:.15e}, P={:.15e}, Ψ={:.15e}", res.u, res.p, res.Ψ);

      const bool u = AlmostEq(res.u, run.results.u);
      const bool p = AlmostEq(res.p, run.results.p);
      const bool Ψ = AlmostEq(res.Ψ, run.results.Ψ);

      constexpr auto ok = [](bool ok) -> int { return ok ? 32 : 31; };
      constexpr auto to_string = [](args_t &args) -> std::string
      {
         std::string args_str;
         for (auto &arg : args)
         {
            if (!arg) { break; }
            args_str += std::string(arg) + " ";
         }
         return args_str;
      };

      dbg("#{} \x1B[33m{}\x1B[m", k, to_string(args).c_str());
      dbg("U: \x1B[33m{:.15e} \x1B[{}m{:.15e}", run.results.u, ok(u), res.u );
      dbg("P: \x1B[33m{:.15e} \x1B[{}m{:.15e}", run.results.p, ok(p), res.p);
      dbg("Ψ: \x1B[33m{:.15e} \x1B[{}m{:.15e}", run.results.Ψ, ok(Ψ), res.Ψ);

      if (u && p && Ψ) { return std::cout << "✅" << std::endl, EXIT_SUCCESS; }
      else { return std::cout << "❌" << std::endl, EXIT_FAILURE; }
   };

   // first launch with default arguments
   if (launch() != EXIT_SUCCESS) { return EXIT_FAILURE; }

   // second launch with the same arguments, but with -pa
   args.pop_back(); // nullptr
   auto arg_pa_ptr = add_arg("-pa");
   args.push_back(arg_pa_ptr.get());
   args_ptr.emplace_back(std::move(arg_pa_ptr));
   args.push_back(nullptr);
   if (launch() != EXIT_SUCCESS) { return EXIT_FAILURE; }

   return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
try
{
   dbg();

   int opt;
   int test = -1;
   auto show_usage = [](const int ret = EXIT_FAILURE)
   {
      printf("Usage: program [-a <arg>] [-b <arg>] [-h]\n");
      printf("  -t <test>  Optional test number \n");
      printf("  -h         Show this help message\n");
      exit(ret);
   };

   while ((opt = getopt(argc, argv, "t:h")) != -1)
   {
      switch (opt)
      {
         case 't': test = std::atoi(optarg); break;
         case 'h': show_usage(EXIT_SUCCESS);
         default:  show_usage(EXIT_FAILURE);
      }
   }

   constexpr int N_TESTS = sizeof(gold) / sizeof(Test);
   if (test >= 0 && test < N_TESTS) { return NavierTest(test, gold[test]); }

   int k = 0;
   for (auto &run : gold)
   {
      if (NavierTest(k++, run) != EXIT_SUCCESS) { return EXIT_FAILURE; }
   }
   return EXIT_SUCCESS;
}
catch (std::exception &e)
{
   std::cerr << "\033[31m..xxxXXX[ERROR]XXXxxx.." << std::endl;
   std::cerr << "\033[31m{}" << e.what() << std::endl;
   return EXIT_FAILURE;
}
