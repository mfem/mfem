#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class HDivBasisCoef : public VectorCoefficient
{
private:
   int p, ndof;

   RT_FuentesPyramidElement pyr;

   Vector dofs;
   mutable DenseMatrix shape;

   bool log;
   ofstream ofs;

public:
   HDivBasisCoef(int p_, bool log_ = false)
      : VectorCoefficient(3),
        p(p_), ndof(p*(3*(p-1)*(p + 1) + 5)),
        pyr(p-1), dofs(ndof), shape(ndof, 3), log(log_)
   { dofs = 0.0; ofs.open("hdiv.dat"); ofs.precision(16); }

   void SetDoF(int dof) { dofs = 0.0; dofs(dof) = 1.0; }
   int GetNDoF() const { return ndof; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      V.SetSize(3);

      real_t u_data[3];
      Vector u(u_data, 3);
      T.Transform(ip, u);

      IntegrationPoint gip; gip.Set(u_data, 3);

      pyr.CalcRawVShape(gip, shape);
      shape.MultTranspose(dofs, V);

      if (log)
      {
         ofs << "{"<< u(0) << ", " << u(1) << ", " << u(2) << "}"<< std::endl;
         ofs << "{";
         for (int i=0; i<ndof; i++)
         {
            ofs << "{" << shape(i, 0) << ", " << shape(i, 1) << ", " << shape(i, 2) << "}";
            if (i < ndof-1) { ofs << "," << std::endl; }
         }
         ofs << "}\n" << std::endl;
      }
   }
};

/**
   Looks reasonable:  Q;  0,  1,  2,  3,
                     T1;  4,  5,
                     T2;  7,  8,
                     T3; 10, 11,
                     T4; 13, 14,
                    III; 20,
                     IV; 21, 22, 23, 24,
                      V; 25,
                     VI; 26,
                    VII; 27

   Looks questionable: T1; 6, // probably ok
                       T2; 9, // probably ok
                       T3; 12, // probably ok
                       T4; 15, // probably ok
                        I; 16, 17,
                       II; 18, 19
*/

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/ref-pyramid-2tet.mesh";
   int ref_levels = 2;
   int order = 2;
   int p = 1;
   int i0 = 0;
   int i1 = -1;
   bool disc = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&p, "-p", "--basis-order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&i0, "-i0", "--start",
                  "Finite element order (polynomial degree)");
   args.AddOption(&i1, "-i1", "--end",
                  "Finite element order (polynomial degree)");
   args.AddOption(&disc, "-disc", "--discontinuous", "-cont",
                  "--continuous",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection fec_h1(order, dim);
   L2_FECollection fec_l2(order, dim);
   FiniteElementCollection *fec = &fec_h1;
   if (disc)
   {
      fec = &fec_l2;
   }
   FiniteElementSpace vfes(&mesh, fec, dim);

   GridFunction v(&vfes);

   HDivBasisCoef pyrVCoef(p, true);

   if (i1 < i0) { i1 = pyrVCoef.GetNDoF()-1; }

   // for (int i=0; i<pyrVCoef.GetNDoF(); i++)
   for (int i=i0; i<=i1; i++)
   {
      pyrVCoef.SetDoF(i);
      v.ProjectCoefficient(pyrVCoef);
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << v
                  << "window_title 'V" << i << "'\n"
                  << "keys cvv\n" << flush;
      }

   }
}
