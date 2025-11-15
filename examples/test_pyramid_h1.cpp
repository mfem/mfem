#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double testFunc(const Vector &p)
{
   const double x = p[0];
   const double y = p[1];
   const double z = p[2];

   return cos(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
}

int main(int argc, char *argv[])
{
   int e = (int)Element::PYRAMID;
   int nx = 1;
   int r = 3;
   int o = 1;
   // int pyrtype = 0;
   double d = 0.0;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&e, "-e", "--elem-type", "Element Type: [4,7]");
   args.AddOption(&nx, "-n", "--n", "Num elems in 1D");
   args.AddOption(&r, "-r", "--refine", "Number of refinements");
   args.AddOption(&o, "-o", "--order", "Number of refinements");
   // args.AddOption(&pyrtype, "-p", "--pyramid-type", "0-Bergot, 1-Fuentes");
   args.AddOption(&d, "-d", "--deformation", "Mesh deformation [0,1)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   FunctionCoefficient testCoef(testFunc);

   H1_FECollection fec(o, 3, BasisType::GaussLobatto);

   Vector errs(r+1); errs = -1.0;
   Vector conv(r);
   for (int i = 0; i <= r; i++)
   {
      int n = nx * pow(2, i);
      Mesh mesh = Mesh::MakeCartesian3D(n,n,n,(Element::Type)e);

      if (d > 0.0)
      {
         const double max = (double)(RAND_MAX) + 1.0;
         const double h = 1.0 / n;

         Vector disp(3*mesh.GetNV());
         for (int j=0; j<disp.Size(); j++)
         {
            disp[j] = (2.0 * rand()/max - 1.0) * h * d;
         }

         mesh.MoveVertices(disp);
      }

      FiniteElementSpace fes(&mesh, &fec);

      GridFunction x(&fes);
      x.ProjectCoefficient(testCoef);
      errs[i] = x.ComputeL2Error(testCoef);
      cout << "DoFs / L2 Error / Conv: " << fes.GetNDofs() << " / " << errs[i];
      if (i > 0)
      {
         conv[i-1] = errs[i-1] / errs[i];
         cout << " / " << conv[i-1];
      }
      cout << endl;

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << x << flush;
      }
   }
}
