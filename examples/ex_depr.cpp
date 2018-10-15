#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class OldStyleCoefficient : public Coefficient
{
public:
   OldStyleCoefficient() {}

#ifdef MFEM_DEPRECATED
   /** If MFEM_DEPRECATED is on then we use the existing old-style
       implementation of Eval */
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      double x[3]; Vector transip(x, 3); T.Transform(ip, transip);
      return transip.Norml2();
   }
#else
   /** If MFEM_DEPRECATED is off then we need to update existing
       implementations of Eval. */
   virtual double Eval(const ElementTransformation &T) const
   {
      double x[3]; Vector transip(x, 3); T.Transform(T.GetIntPoint(), transip);
      return transip.Norml2();
   }
#endif

};

class NewStyleCoefficient : public Coefficient
{
public:
   NewStyleCoefficient() {}

#ifdef MFEM_DEPRECATED
   /** Newly implemented coefficients will need to supply the following
       override if MFEM_DEPRECATED is on. */
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   { return Eval(T); }
#endif

   /** Newly implemented coefficients should use the new style of Eval */
   virtual double Eval(const ElementTransformation &T) const
   {
      double x[3]; Vector transip(x, 3); T.Transform(T.GetIntPoint(), transip);
      return transip.Norml2();
   }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   OldStyleCoefficient oCoef;
   NewStyleCoefficient nCoef;

   GridFunction xo(fespace);
   GridFunction xn(fespace);

   xo.ProjectCoefficient(oCoef);
   xn.ProjectCoefficient(nCoef);

   // 12. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream old_ofs("xo.gf");
   ofstream new_ofs("xn.gf");
   old_ofs.precision(8);
   new_ofs.precision(8);
   xo.Save(old_ofs);
   xn.Save(new_ofs);

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream old_sock(vishost, visport);
      socketstream new_sock(vishost, visport);
      old_sock.precision(8);
      new_sock.precision(8);
      old_sock << "solution\n" << *mesh << xo
               << "window_title 'Old Style Coefficient'"
               << "window_geometry 0 0 350 350"<< flush;
      new_sock << "solution\n" << *mesh << xn
               << "window_title 'New Style Coefficient'"
               << "window_geometry 360 0 350 350"<< flush;
   }

   // 14. Free the used memory.
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}
