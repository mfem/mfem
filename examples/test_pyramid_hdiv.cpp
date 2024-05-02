#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void irrTestFunc(const Vector &p, Vector &V)
{
   const double x = p[0];
   const double y = p[1];
   const double z = p[2];

   V.SetSize(3);
   V[0] = sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
   V[1] = cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
   V[2] = cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
}

void solTestFunc(const Vector &p, Vector &V)
{
   const double x = p[0];
   const double y = p[1];
   const double z = p[2];

   V.SetSize(3);
   V[0] = -cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
   V[1] =  sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
   V[2] =  0.0;
}

real_t ComputeL2ErrorHDivPyramid(GridFunction &x, VectorCoefficient &exsol);

int main(int argc, char *argv[])
{
   int e = (int)Element::PYRAMID;
   int nx = 1;
   int r = 3;
   int o = 1;
   double d = 0.0;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&e, "-e", "--elem-type", "Element Type: [4,7]");
   args.AddOption(&nx, "-n", "--n", "Num elems in 1D");
   args.AddOption(&r, "-r", "--refine", "Number of refinements");
   args.AddOption(&o, "-o", "--order", "Number of refinements");
   args.AddOption(&d, "-d", "--deformation", "Mesh deformation [0,1)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   VectorFunctionCoefficient irrTestCoef(3, irrTestFunc);
   VectorFunctionCoefficient solTestCoef(3, solTestFunc);

   // RT0_3DFECollection fec1;
   RT_FECollection fec(o-1,3);
   const RT_FuentesPyramidElement *pyr =
      dynamic_cast<const RT_FuentesPyramidElement *>(fec.FiniteElementForGeometry(
                                                        Geometry::PYRAMID));
   // FiniteElementCollection &fec = (o == 1) ?
   //                               (FiniteElementCollection&)fec1 : (FiniteElementCollection&)fec2;

   Vector errsIrr(r+1); errsIrr = -1.0;
   Vector errsSol(r+1); errsSol = -1.0;
   Vector convIrr(r);
   Vector convSol(r);
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

      GridFunction irr(&fes);
      irr.ProjectCoefficient(irrTestCoef);
      if (true)
      {
         errsIrr[i] = irr.ComputeL2Error(irrTestCoef);
      }
      else
      {
         errsIrr[i] = ComputeL2ErrorHDivPyramid(irr, irrTestCoef);
      }

      GridFunction sol(&fes);
      sol.ProjectCoefficient(solTestCoef);
      if (true)
      {
         errsSol[i] = sol.ComputeL2Error(solTestCoef);
      }
      else
      {
         errsSol[i] = ComputeL2ErrorHDivPyramid(sol, solTestCoef);
      }

      cout << "DoFs / L2 Error / Conv: " << fes.GetNDofs() << " / "
           << errsIrr[i] << " " << errsSol[i];
      if (i > 0)
      {
         convIrr[i-1] = errsIrr[i-1] / errsIrr[i];
         convSol[i-1] = errsSol[i-1] / errsSol[i];
         cout << " / " << convIrr[i-1] << " " << convSol[i-1];
      }
      cout << endl;

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << irr << flush;
      }
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << sol << flush;
      }
   }
   std::cout << "Maximum Zeta: " << pyr->GetZetaMax() << std::endl;
}

real_t ComputeL2ErrorHDivPyramid(GridFunction &x, VectorCoefficient &exsol)
{
   FiniteElementSpace *fes = x.FESpace();

   real_t error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;
   Vector F(3);

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      int intorder = 2*fe->GetOrder() + 3; // <----------
      const IntegrationRule *ir;
      ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      T = fes->GetElementTransformation(i);
      x.GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      vals.Norm2(loc_errs);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         real_t u = ip.x;
         real_t v = ip.y;
         real_t w = ip.z;
         vals.GetColumn(j, F);
         real_t rt_err2 = (pow(F(0)*(1.0-w), 2) +
                           pow(F(1)*(1.0-w), 2) +
                           pow(F(2)-u*F(0)-v*F(1), 2)
                          ) / pow(1.0-w, 4);

         real_t loc_err = (false) ? (loc_errs(j) * loc_errs(j)) : rt_err2;

         std::cout << u << ", " << v << ", " << w
                   << ", V = " << exact_vals(0, j) << ", " << exact_vals(1,
                                                                         j) << ", " << exact_vals(2,
                                                                               j) << ", loc_err = " << loc_errs(j) << " or " << sqrt(rt_err2)
                   << ", F = ";
         F.Print(std::cout);
         error += ip.weight * T->Weight() * loc_err;
      }
   }

   return (error < 0.0) ? -sqrt(-error) : sqrt(error);
}
