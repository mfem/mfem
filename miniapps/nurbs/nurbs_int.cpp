//                                MFEM Example interpolation -- modified for NURBS FE
//
// Compile with: make nurbs_int
//
// Sample runs:  nurbs_int -m ../../data/pipe-nurbs-2d.mesh -o 2


//
// Description:  This example code illustrates usage of mixed finite element
//               spaces, with three variants:
//               0) H(div)
//               1) H(div)
//               2) H(curl)


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void vfun(const Vector &x, Vector &v)
{
      v(0) = sin(6.28 * x(0));
      v(1) = sin(6.28 * x(1));
      if (x.Size() == 3) { v(2) = sin(6.28 * x(2)); }
}


int dim;
real_t freq = 1.0, kappa;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int dim = 2;
   int ref_levels = 0;
   int order = 1;
   bool NURBS = true;
   int prob = 0;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim",
                  "Mesh dimensionto use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&NURBS, "-n", "--nurbs", "-nn","--no-nurbs",
                  "NURBS.");
   args.AddOption(&prob, "-p", "--problem-type",
                  "Choose between 0: h1, 1: div, 2: curl");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   kappa = freq * M_PI;

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = nullptr;
   if (dim == 2)
   {
      mesh = new Mesh("../../data/square-nurbs.mesh", 1, 1);
   }
   else
   {
      mesh = new Mesh("../../data/cube-nurbs.mesh", 1, 1);
   }
   dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use Nedelec or
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *fec = nullptr;
   NURBSExtension *NURBSext = nullptr;
   FiniteElementSpace *fes = nullptr;
   if (!mesh->NURBSext)
   {
      mfem_error("");
   }

   NURBSext  = new NURBSExtension(mesh->NURBSext, order);
   if (prob == 0)
   {
      fec  = new NURBSFECollection(order);
      fes = new FiniteElementSpace(mesh, NURBSext, fec, dim);
      cout << "Number of H1 finite element unknowns: " << fes->GetTrueVSize() << endl;
   }
   else if (prob == 1)
   {
      fec = new NURBS_HDivFECollection(order, dim);
      fes = new FiniteElementSpace(mesh, NURBSext, fec);
      cout << "Number of H(div) finite element unknowns: " << fes->GetTrueVSize() <<endl;
   }
   else
   {
      fec = new NURBS_HCurlFECollection(order, dim);
      fes = new FiniteElementSpace(mesh, NURBSext, fec);
      cout << "Number of H(curl) finite element unknowns: " << fes->GetTrueVSize() <<endl;
   }
   mfem::out<<"Create NURBS fec and ext"<<std::endl;

   // 6. Define the solution vector as a finite element grid function
   //    corresponding to the trial fespace.
   GridFunction gf(fes);

   VectorFunctionCoefficient vcoeff(dim, vfun);

   gf.ProjectCoefficient(vcoeff);

   gf.SetTrueVector();
   gf.SetFromTrueVector();

   // 11. Compute and print the L_2 norm of the error.
   real_t errProj = gf.ComputeL2Error(vcoeff);

   cout << " Interpolation error: || v_h - v ||_{L_2} = " << errProj << '\n' << endl;

   // 12. Save the solution.
   VisItDataCollection visit_dc("nurbs_int", mesh);
   visit_dc.RegisterField("solution", &gf);
   visit_dc.Save();

   // 14. Free the used memory.
   delete fec;
   delete fes;
   delete mesh;

   return 0;
}

