#include "mfem.hpp"
#include <ctime>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Parse command-line options.
   bool print = false;
   int dim = 1;
   int order = 1;
   int rbfType = 0;
   int distNorm = 2;
   int numPoints = 10;
   int evals = 1;
   double h = 4.01;
   double x = 0.2;
   double y = 0.7;
   double z = 0.1;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim",
                  "dimension");
   args.AddOption(&order, "-o", "--order",
                  "RK order or -1 for RBF");
   args.AddOption(&rbfType, "-f", "--func",
                  "(0) Gaussian, (1) Multiquadric, (2) Inverse multiquadric, (3) Wendland31");
   args.AddOption(&distNorm, "-s", "--dist",
                  "Which Lp norm to use for distance");
   args.AddOption(&numPoints, "-n", "--points",
                  "number of points in 1d");
   args.AddOption(&h, "-m", "--smoothing",
                  "smoothing parameter (units of distance)");
   args.AddOption(&x, "-x", "--xpos",
                  "evaluation position, x");
   args.AddOption(&y, "-y", "--ypos",
                  "evaluation position, y");
   args.AddOption(&z, "-z", "--zpos",
                  "evaluation position, z");
   args.AddOption(&evals, "-e", "--evals",
                  "number of evaluations");
   args.AddOption(&print, "-p", "--print", "-no-p",
                  "--no-print", "Print out full matrices");
   args.Parse();
   if (!args.Good())
   {
      args.PrintError(cout);
   }
}
