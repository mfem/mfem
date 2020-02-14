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
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Get collection
   const KernelFECollection *fec = new KernelFECollection(dim, numPoints, h,
                                                          rbfType, distNorm, order);
   int geomType = TensorBasisElement::GetTensorProductGeometry(dim);
   cout << fec->Name() << endl;
   // Get element
   const FiniteElement *fe = fec->FiniteElementForGeometry(geomType);
   
   // Initialize integration point
   IntegrationPoint ip;
   ip.x = x;
   ip.y = y;
   ip.z = z;

   // Evaluate value at integration point, possibly multiple times for timing
   int dof = fe->GetDof();
   Vector shape(dof);
   DenseMatrix dshape(dof, dim);
   for (int i = 0; i < evals; ++i)
   {
      fe->CalcShape(ip, shape);
      fe->CalcDShape(ip, dshape);
   }

   // Print the values of each function at the specified point
   if (print)
   {
      const IntegrationRule &nodes = fe->GetNodes();
      for (int i = 0; i < dof; ++i)
      {
         cout << nodes.IntPoint(i).x << "\t";
         if (dim > 1) cout << nodes.IntPoint(i).y << "\t";
         if (dim > 2) cout << nodes.IntPoint(i).z << "\t";
         cout << shape(i) << "\t";
         for (int d = 0; d < dim; ++d)
         {
            cout << dshape(i, d) << "\t";
         }
         cout << endl;
      }
   }

   // Make sure the value sums to 1 and the derivative sums to 0
   double sum = 0.0;
   vector<double> dsum(3, 0.0);
   for (int i = 0; i < dof; ++i)
   {
      sum += shape(i);
      for (int d = 0; d < dim; ++d)
      {
         dsum[d] += dshape(i, d);
      }
   }
   cout << "sum:\t" << sum << endl;
   cout << "dsum:\t";
   for (int d = 0; d < dim; ++d)
   {
      cout << dsum[d] << "\t";
   }
   cout << endl;

   
   
   // Free memory
   delete fec;
}
