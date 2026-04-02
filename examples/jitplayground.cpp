#include <proteus/CppJitModule.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class JitQFunction
{
public:
   JitQFunction(const std::string &fn)
   {
      std::ifstream t(fn);
      if (!t.is_open())
      {
         std::cerr << "could not open file " << fn << "\n";
      }
      std::stringstream buffer;
      buffer << t.rdbuf();
      code = buffer.str();
   }

   const std::string &GetCode() const
   {
      return code;
   }

private:
   std::string code, dcode;
};

static int parse_int(const char *s, int default_value)
{
   if (!s) { return default_value; }
   try { return std::stoi(s); }
   catch (...) { return default_value; }
}

static int parse_sizet(const char *s, size_t default_value)
{
   if (!s) { return default_value; }
   try { return std::stoull(s); }
   catch (...) { return default_value; }
}

static double parse_double(const char *s, double default_value)
{
   if (!s) { return default_value; }
   try { return std::stod(s); }
   catch (...) { return default_value; }
}

int main(int argc, char *argv[])
{
   std::string filename;
   if (argc < 2)
   {
      std::cerr << "Usage: " << argv[0] << " <kernel-file> [N] [A]\n";
      return 1;
   }
   filename = argv[1];

   JitQFunction qf(filename);

   const size_t N = parse_sizet(argc >= 3 ? argv[2] : nullptr, 64);
   const double A = parse_double(argc >= 4 ? argv[3] : nullptr, 123.0);

   std::vector<double> X(N);
   std::vector<double> Y(N);
   for (size_t i = 0; i < N; ++i)
   {
      X[i] = static_cast<double>(i + 1);
      Y[i] = static_cast<double>(N - i);
   }

   std::vector<std::string> extra_args
   {
      "-fplugin=/Users/andrej1/local/enzyme/lib/ClangEnzyme-20.dylib"
   };

   proteus::CppJitModule mod{"host", qf.GetCode(), extra_args};

   auto &instance = mod.instantiate("daxpy_wrapper", std::to_string(N));
   instance.run<void>(static_cast<const double*>(&A), &X, &Y);

   auto &dinstance = mod.instantiate("ddaxpy_wrapper", std::to_string(N));
   std::vector<double> dX(N, 1.0);
   std::vector<double> dY(N, 0.0);
   dinstance.run<void>(
      &A,
      &X, &dX,
      &Y);

   std::vector<double> dX_manual(N, A);

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
