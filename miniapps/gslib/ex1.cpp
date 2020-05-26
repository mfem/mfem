//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m1 ../../data/square-disc.mesh -m2 inner.mesh
//
// Description:  Overlapping grids with MFEM:
//               This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Method to use FindPointsGSLIB to determine the boundary points of a mesh
// that are interior to another mesh.
void GetInterdomainBoundaryPoints(FindPointsGSLIB &finder1,
                                  FindPointsGSLIB &finder2,
                                  Vector &vxyz1,                  Vector &vxyz2,
                                  Array<int> ess_tdof_list1,      Array<int> ess_tdof_list2,
                                  Array<int> &ess_tdof_list1_int, Array<int> &ess_tdof_list2_int,
                                  const int dim)
{
   int nb1 = ess_tdof_list1.Size(),
       nb2 = ess_tdof_list2.Size(),
       nt1 = vxyz1.Size()/dim,
       nt2 = vxyz2.Size()/dim;

   Vector bnd1(nb1*dim);
   for (int i = 0; i < nb1; i++)
   {
      int idx = ess_tdof_list1[i];
      for (int d = 0; d < dim; d++) { bnd1(i+d*nb1) = vxyz1(idx + d*nt1); }
   }

   Vector bnd2(nb2*dim);
   for (int i = 0; i < nb2; i++)
   {
      int idx = ess_tdof_list2[i];
      for (int d = 0; d < dim; d++) { bnd2(i+d*nb2) = vxyz2(idx + d*nt2); }
   }

   finder1.FindPoints(bnd2);
   finder2.FindPoints(bnd1);

   const Array<unsigned int> &code_out1 = finder1.GetCode();
   const Array<unsigned int> &code_out2 = finder2.GetCode();

   //Setup ess_tdof_list_int
   for (int i = 0; i < nb1; i++)
   {
      int idx = ess_tdof_list1[i];
      if (code_out2[i] != 2) { ess_tdof_list1_int.Append(idx); }
   }

   for (int i = 0; i < nb2; i++)
   {
      int idx = ess_tdof_list2[i];
      if (code_out1[i] != 2) { ess_tdof_list2_int.Append(idx); }
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file_1   = "../../data/square-disc.mesh";
   const char *mesh_file_2   = "inner.mesh";
   int order                 = 2;
   const char *device_config = "cpu";
   bool visualization        = true;
   int r1_levels             = 0;
   int r2_levels             = 0;
   double rel_tol            = 1.e-8;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file_1, "-m1", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_file_2, "-m2", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&r1_levels, "-r1", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&r2_levels, "-r2", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rel_tol, "-rt", "--relative tolerance",
                  "Tolerance for Schwarz iteration convergence criterion.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   const int nmeshes = 2;

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Array <Mesh*> mesharr(2);
   mesharr[0] = new Mesh(mesh_file_1, 1, 1);
   mesharr[1] = new Mesh(mesh_file_2, 1, 1);
   int dim = mesharr[0]->Dimension();


   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   for (int lev = 0; lev < r1_levels; lev++) { mesharr[0]->UniformRefinement(); }
   for (int lev = 0; lev < r2_levels; lev++) { mesharr[1]->UniformRefinement(); }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   Array <FiniteElementCollection*> fecarr(nmeshes);
   Array <FiniteElementSpace*> fespacearr(nmeshes);
   for (int i = 0; i< nmeshes; i ++)
   {
      if (order > 0)
      {
         fecarr[i] =  new H1_FECollection(order, dim);
      }
      else if (mesharr[i]->GetNodes())
      {
         fecarr[i] =  mesharr[i]->GetNodes()->OwnFEC();
         cout << "Using isoparametric FEs: " << fecarr[0]->Name() << endl;
      }
      else
      {
         fecarr[i] = new H1_FECollection(order = 1, dim);
      }
      fespacearr[i] = new FiniteElementSpace(mesharr[i], fecarr[i]);
   }

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list1, ess_tdof_list2;
   if (mesharr[0]->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesharr[0]->bdr_attributes.Max());
      ess_bdr = 1;
      fespacearr[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list1);
   }

   if (mesharr[1]->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesharr[1]->bdr_attributes.Max());
      ess_bdr = 1;
      fespacearr[1]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list2);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace1.
   ConstantCoefficient one(1.0);
   Array<LinearForm*> b_ar(nmeshes);
   for (int i = 0; i< nmeshes; i ++)
   {
      b_ar[i] = new LinearForm(fespacearr[i]);
      b_ar[i]->AddDomainIntegrator(new DomainLFIntegrator(one));
      b_ar[i]->Assemble();
   }

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace1. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x1(fespacearr[0]), x2(fespacearr[1]);
   x1 = 0;
   x2 = 0;

   // 9. Setup FindPointsGSLIB and determine points on each mesh's boundary that
   //    are interior to another mesh.
   mesharr[0]->SetCurvature(order, false, dim, Ordering::byNODES);
   Vector vxyz1 = *mesharr[0]->GetNodes();
   mesharr[1]->SetCurvature(order, false, dim, Ordering::byNODES);
   Vector vxyz2 = *mesharr[1]->GetNodes();

   FindPointsGSLIB finder1, finder2;
   finder1.Setup(*mesharr[0]);
   finder2.Setup(*mesharr[1]);

   Array<int> ess_tdof_list1_int, ess_tdof_list2_int;
   GetInterdomainBoundaryPoints(finder1, finder2, vxyz1, vxyz2,
                                ess_tdof_list1, ess_tdof_list2,
                                ess_tdof_list1_int, ess_tdof_list2_int, dim);

   // 10. Use FindPointsGSLIB to interpolate the solution at interdomain boundary
   //     points.
   const int nb1 = ess_tdof_list1_int.Size(),
             nb2 = ess_tdof_list2_int.Size(),
             nt1 = vxyz1.Size()/dim,
             nt2 = vxyz2.Size()/dim;

   MFEM_VERIFY(nb1!=0 || nb2!=0, " Please use overlapping grids.");

   Vector bnd1(nb1*dim);
   for (int i = 0; i < nb1; i++)
   {
      int idx = ess_tdof_list1_int[i];
      for (int d = 0; d < dim; d++) { bnd1(i+d*nb1) = vxyz1(idx + d*nt1); }
   }

   Vector bnd2(nb2*dim);
   for (int i = 0; i < nb2; i++)
   {
      int idx = ess_tdof_list2_int[i];
      for (int d = 0; d < dim; d++) { bnd2(i+d*nb2) = vxyz2(idx + d*nt2); }
   }

   Vector interp_vals1(nb1), interp_vals2(nb2);
   finder1.Interpolate(bnd2, x1, interp_vals2);
   finder2.Interpolate(bnd1, x2, interp_vals1);

   // 11. Set up the bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   Array <BilinearForm*> a_ar(2);
   a_ar[0] = new BilinearForm(fespacearr[0]);
   a_ar[1] = new BilinearForm(fespacearr[1]);
   a_ar[0]->AddDomainIntegrator(new DiffusionIntegrator(one));
   a_ar[1]->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   a_ar[0]->Assemble();
   a_ar[1]->Assemble();

   delete b_ar[0];
   delete b_ar[1];

   // 13. Use simultaneous Schwarz iterations to iteratively solve the PDE
   //     and interpolate interdomain boundary data to impose Dirichlet
   //     boundary conditions.

   int NiterSchwarz = 100;
   for (int schwarz = 0; schwarz < NiterSchwarz; schwarz++)
   {
      for (int i = 0; i< nmeshes; i ++)
      {
         b_ar[i] = new LinearForm(fespacearr[i]);
         b_ar[i]->AddDomainIntegrator(new DomainLFIntegrator(one));
         b_ar[i]->Assemble();
      }

      OperatorPtr A1, A2;
      Vector B1, X1, B2, X2;

      a_ar[0]->FormLinearSystem(ess_tdof_list1, x1, *b_ar[0], A1, X1, B1);
      a_ar[1]->FormLinearSystem(ess_tdof_list2, x2, *b_ar[1], A2, X2, B2);

      // 11. Solve the linear system A X = B.
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M1((SparseMatrix&)(*A1));
      PCG(*A1, M1, B1, X1, 0, 200, 1e-12, 0.0);
      GSSmoother M2((SparseMatrix&)(*A2));
      PCG(*A2, M2, B2, X2, 0, 200, 1e-12, 0.0);

      // 12. Recover the solution as a finite element grid function.
      a_ar[0]->RecoverFEMSolution(X1, *b_ar[0], x1);
      a_ar[1]->RecoverFEMSolution(X2, *b_ar[1], x2);

      // Interpolate boundary condition
      finder1.Interpolate(x1, interp_vals2);
      finder2.Interpolate(x2, interp_vals1);

      double dxmax = std::numeric_limits<float>::min();
      double x1inf = x1.Normlinf();
      double x2inf = x2.Normlinf();
      for (int i = 0; i < nb1; i++)
      {
         int idx = ess_tdof_list1_int[i];
         double dx = std::abs(x1(idx)-interp_vals1(i))/x1inf;
         if (dx > dxmax) { dxmax = dx; }
         x1(idx) = interp_vals1(i);
      }

      for (int i = 0; i < nb2; i++)
      {
         int idx = ess_tdof_list2_int[i];
         double dx = std::abs(x2(idx)-interp_vals2(i))/x2inf;
         if (dx > dxmax) { dxmax = dx; }
         x2(idx) = interp_vals2(i);
      }

      delete b_ar[0];
      delete b_ar[1];

      std::cout << std::setprecision(8)    <<
                "Iteration: "           << schwarz <<
                ", Relative residual: " << dxmax   << endl;
      if (dxmax < rel_tol) { break; }
   }

   // 14. Send the solution by socket to a GLVis server.
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      for (int ip = 0; ip<mesharr.Size(); ++ip)
      {
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "parallel " << mesharr.Size() << " " << ip << "\n";
         if (ip==0) { sol_sock << "solution\n" << *mesharr[ip] << x1 << flush; }
         if (ip==1) { sol_sock << "solution\n" << *mesharr[ip] << x2 << flush; }
         sol_sock << "window_title 'Overlapping grid solution'\n"
                  << "window_geometry "
                  << 0 << " " << 0 << " " << 450 << " " << 350 << "\n"
                  << "keys jmcA]]]" << endl;
      }
   }

   // 15. Free the used memory.
   finder1.FreeData();
   finder2.FreeData();
   for (int i = 0; i < nmeshes; i++)
   {
      delete a_ar[i];
      delete fespacearr[i];
      if (order > 0) { delete fecarr[i]; }
      delete mesharr[i];
   }

   return 0;
}
