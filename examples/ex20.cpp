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
   int order = -1;
   int funcType = 0;
   int distType = 0;
   int numPoints = 10;
   int evals = 1;
   double h = 4.01;
   double x = 0.5;
   double y = 0.5;
   double z = 0.5;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim",
                  "dimension");
   args.AddOption(&order, "-o", "--order",
                  "RK order or -1 for RBF");
   args.AddOption(&funcType, "-f", "--func",
                  "(0) Gaussian, (1) Multiquadric, (2) Inverse multiquadric");
   args.AddOption(&distType, "-s", "--dist",
                  "(0) Euclidean, (1) Manhattan");
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

   // Get RBF
   RBFFunction *func;
   switch (funcType)
   {
   case 0:
      func = new GaussianRBF();
      break;
   case 1:
      func = new MultiquadricRBF();
      break;
   case 2:
      func = new InvMultiquadricRBF();
      break;
   default:
      MFEM_ABORT("unknown RBF");
   }
   
   // Get distance
   DistanceMetric *dist;
   switch (distType)
   {
   case 0:
      dist = new EuclideanDistance(dim);
      break;
   case 1:
      dist = new ManhattanDistance(dim);
      break;
   default:
      MFEM_ABORT("unknown dist");
   }
   
   // Get collection
   KernelFECollection *fec = new KernelFECollection(dim, numPoints, h,
                                                    func, dist, order);

   Geometry::Type geomType;
   switch (dim)
   {
   case 1:
      geomType = Geometry::SEGMENT;
      break;
   case 2:
      geomType = Geometry::SQUARE;
      break;
   case 3:
      geomType = Geometry::CUBE;
      break;
   default:
      MFEM_ABORT("unknown dim");
   }

   // Get element
   const FiniteElement *fe = fec->FiniteElementForGeometry(geomType);
   
   // Initialize integration point
   int dof = fe->GetDof();
   IntegrationPoint ip;
   Vector shape(dof);
   DenseMatrix dshape(dof, dim);
   ip.x = x;
   ip.y = y;
   ip.z = z;

   // Evaluate integrals
   for (int i = 0; i < evals; ++i)
   {
      fe->CalcShape(ip, shape);
      fe->CalcDShape(ip, dshape);
   }
   
   if (print)
   {
      for (int i = 0; i < dof; ++i)
      {
         cout << shape(i) << "\t";
         for (int d = 0; d < dim; ++d)
         {
            cout << dshape(i, d) << "\t";
         }
         cout << endl;
      }
   }

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
