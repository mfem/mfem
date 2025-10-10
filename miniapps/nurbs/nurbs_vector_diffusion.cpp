// Simple test application that solves a diffusion using vector functions
//
// Compile with: make nurbs_vector_diffusion
//
// Sample runs:  nurbs_vector_diffusion

// Description:  This example code solves a simple 2D/3D vector poisson problem
//               corresponding to the system
//
//                          -k*Delta*u = f
//
//               with weak or strong essential boundary conditions  u= given

//               NURBS-based H(div) spaces only implemented for meshes
//               consisting of a single patch.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_h1_ex(const Vector & x, Vector & u)
{
   u[0] = sin(M_PI*x[0])*sin(M_PI*x[1]);
   u[1] = 2*u[0];

   if (x.Size() == 3)
   {
      u[0] *= sin(M_PI*x[2]);
      u[1] *= sin(M_PI*x[2]);
      u[2] = 2*u[1];
   }
}

void fFun_h1(const Vector & x, Vector & f)
{
   f[0] = M_PI*M_PI*sin(M_PI*x[0])*sin(M_PI*x[1]);
   f[1] = 2*f[0];
   if (x.Size() == 3)
   {
      f[0] *= sin(M_PI*x[2]);
      f[1] *= sin(M_PI*x[2]);
      f[2] = 2*f[1];
   }
   f *= x.Size();
}

void uFun_hdiv_ex(const Vector & x, Vector & u)
{
   u[0] = sin(M_PI*x[0]);
   u[1] = sin(M_PI*x[1]);

   if (x.Size() == 3)
   {
      u[2] = sin(M_PI*x[2]);
   }
}

void fFun_hdiv(const Vector & x, Vector & f)
{
   f[0] = M_PI*M_PI*sin(M_PI*x[0]);
   f[1] = M_PI*M_PI*sin(M_PI*x[1]);
   if (x.Size() == 3)
   {
      f[2] = M_PI*M_PI*sin(M_PI*x[2]);
   }
}


void uFun_hcurl_ex(const Vector & x, Vector & u)
{
   u[0] = sin(M_PI*x[1]);
   u[1] = sin(M_PI*x[0]);

   if (x.Size() == 3)
   {
      u[0] *= sin(M_PI*x[2]);
      u[1] *= sin(M_PI*x[2]);
      u[2] = sin(M_PI*x[0])*sin(M_PI*x[1]);
   }
}

void fFun_hcurl(const Vector & x, Vector & f)
{
   f[0] = M_PI*M_PI*sin(M_PI*x[1]);
   f[1] = M_PI*M_PI*sin(M_PI*x[0]);
   if (x.Size() == 3)
   {
      f[0] *= sin(M_PI*x[2]);
      f[1] *= sin(M_PI*x[2]);
      f[2] = M_PI*M_PI*sin(M_PI*x[0])*sin(M_PI*x[1]);
   }
   f *= (x.Size()-1);
}

int main(int argc, char *argv[])
{
   StopWatch chrono;
   Device device("cpu");
   device.Print();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int ref_levels = -1;
   int order = 1;
   bool hdiv = true;
   bool h1BC = false;
   bool weakBC = true;
   bool visualization = true;
   bool visit = false;
   real_t penalty = -1.0;
   int maxiter = 200;
   real_t rtol = 1.e-6;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&hdiv, "-div", "--hdiv", "-curl", "--hcurl",
                  "Select which vector FE to use.");
   args.AddOption(&h1BC, "-h1-bc", "--H1-bc", "-conf-bc", "--conforming-bc",
                  "Use Full H1 BCs or the conforming BCs:\n"
                  " - Normal for Div conforming elements\n"
                  " - Tangential for Curl conforming elements).");
   args.AddOption(&weakBC, "-w", "--wbc", "-s", "--sbc",
                  "Weak boundary conditions.");
   args.AddOption(&penalty, "-p", "--lambda",
                  "Penalty parameter for enforcing weak Dirichlet boundary conditions.");
   args.AddOption(&rtol, "-lt", "--linear-tolerance",
                  "Relative tolerance for the GMRES solver.");
   args.AddOption(&maxiter, "-li", "--linear-itermax",
                  "Maximum iteration count for the GMRES solver.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (penalty < 0)
   {
      penalty = 10*(order+1)*(order+1);
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   if (!mesh->NURBSext)
   {
      mfem_error("Gradient of vector shape functions only defined for NURBS elements.");
   }
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      if (ref_levels < 0)
      {
         ref_levels =
            (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
   }

   // 4. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *vfe_coll = nullptr;
   VectorFunctionCoefficient *fcoeff = nullptr;
   VectorFunctionCoefficient *ucoeff = nullptr;

   switch (2*hdiv + h1BC)
   {
      case 0:
         vfe_coll = new NURBS_HCurlFECollection(order,dim);
         fcoeff = new VectorFunctionCoefficient(dim, fFun_hcurl);
         ucoeff = new VectorFunctionCoefficient(dim, uFun_hcurl_ex);
         break;
      case 1:
         vfe_coll = new NURBS_HCurlH1FECollection(order,dim);
         fcoeff = new VectorFunctionCoefficient(dim, fFun_h1);
         ucoeff = new VectorFunctionCoefficient(dim, uFun_h1_ex);
         mfem::out<<"Homogeneous BCs!!"<<std::endl;
         break;
      case 2:
         vfe_coll = new NURBS_HDivFECollection(order,dim);
         fcoeff = new VectorFunctionCoefficient(dim, fFun_hdiv);
         ucoeff = new VectorFunctionCoefficient(dim, uFun_hdiv_ex);
         break;
      case 3:
         vfe_coll = new NURBS_HDivH1FECollection(order,dim);
         fcoeff = new VectorFunctionCoefficient(dim, fFun_h1);
         ucoeff = new VectorFunctionCoefficient(dim, uFun_h1_ex);
         mfem::out<<"Homogeneous BCs!!"<<std::endl;
         break;
   }
   NURBSExtension *NURBSext  = new NURBSExtension(mesh->NURBSext, order);
   mfem::out<<"Create NURBS fecollection "<<vfe_coll->Name()<<std::endl;

   FiniteElementSpace space(mesh, NURBSext, vfe_coll);
   Array<int> ess_tdof_list;
   space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   std::cout << "***********************************************************\n";
   std::cout << "Number of dofs       = " << space.GetVSize() << endl;
   std::cout << "Number boundary dofs = " << ess_tdof_list.Size() << endl;
   std::cout << "***********************************************************\n";

   // 5. Define the coefficients, analytical solution.
   ConstantCoefficient k_c(1.0);

   MemoryType mt = device.GetMemoryType();
   Vector x(space.GetVSize(),mt);

   GridFunction u;
   u.MakeRef(&space, x, 0);
   u.ProjectCoefficient(*ucoeff);
   if (h1BC) { u.SetSubVector(ess_tdof_list, 0.0); }

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   real_t err_u  = u.ComputeL2Error(*ucoeff, irs);
   real_t norm_u = ComputeLpNorm(2., *ucoeff, *mesh, irs);

   // 6. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
   Vector rhs(space.GetVSize(),mt);

   LinearForm *fform(new LinearForm);
   fform->Update(&space, rhs, 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*fcoeff));
   if (weakBC) { fform->AddBdrFaceIntegrator(new VectorFEDGDirichletLFIntegrator(*ucoeff, k_c, -1.0, penalty)); }
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   // 7. Assemble the finite element matrix for the diffusion operator
   BilinearForm *kVarf(new BilinearForm(&space));
   kVarf->AddDomainIntegrator(new VectorFEDiffusionIntegrator(k_c));
   if (weakBC) { kVarf->AddBdrFaceIntegrator(new VectorFEDGDiffusionIntegrator(k_c, -1.0, penalty)); }
   kVarf->Assemble();
   if (!weakBC) { kVarf->EliminateEssentialBC(ess_bdr, u, *fform); }
   kVarf->Finalize();

   // 8. Construct the preconditioner
   SparseMatrix &K(kVarf->SpMat());
   DSmoother  invK(K);
   invK.iterative_mode = false;

   // 9. Solve the linear system with MINRES.
   //    Check the norm of the unpreconditioned residual.
   chrono.Clear();
   chrono.Start();
   MINRESSolver solver;
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxiter);
   solver.SetOperator(K);
   solver.SetPreconditioner(invK);
   solver.SetPrintLevel(1);

   solver.Mult(rhs, x);
   if (device.IsEnabled()) { x.HostRead(); }
   chrono.Stop();

   if (solver.GetConverged())
   {
      std::cout << "MINRES converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
   }
   else
   {
      std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
   }
   std::cout << "MINRES solver took " << chrono.RealTime() << "s.\n";

   // 10.  Print the previously computer interpolation errors.
   //      Compute and print the L2 error norms of the numerical solution.
   std::cout << "Projection error:\n";
   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";

   err_u  = u.ComputeL2Error(*ucoeff, irs);
   norm_u = ComputeLpNorm(2., *ucoeff, *mesh, irs);

   if (weakBC)
   {
      std::cout << "Error when solving with weak boundary conditions:\n";
   }
   else
   {
      std::cout << "Error when solving with strong boundary conditions:\n";
   }
   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";

   std::cout << "Projection error:\n";
   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";

   err_u  = u.ComputeL2Error(*ucoeff, irs);
   norm_u = ComputeLpNorm(2., *ucoeff, *mesh, irs);

   if (weakBC)
   {
      std::cout << "Error when solving with weak boundary conditions:\n";
   }
   else
   {
      std::cout << "Error when solving with strong boundary conditions:\n";
   }
   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";

   // 11. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("vdif.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream u_ofs("sol.gf");
      u_ofs.precision(8);
      u.Save(u_ofs);
   }

   // 12. Save data in the VisIt format
   //     Define a traditional NURBS space as long as ViSit does not read to new Vector FEs
   if (visit)
   {
      H1_FECollection h1_coll(order+1);
      FiniteElementSpace ui_space(mesh, space.StealNURBSext(), &h1_coll, dim);
      VectorGridFunctionCoefficient uh_cf(&u);
      GridFunction ui_gf(&ui_space);
      ui_gf.ProjectCoefficient(uh_cf);

      VisItDataCollection visit_dc("VectorDiffusion", mesh);
      visit_dc.RegisterField("solution", &ui_gf);
      visit_dc.Save();
   }

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << *mesh << u << "window_title 'Velocity'" << endl;
   }

   // 14. Free the used memory.
   delete fform;
   delete kVarf;
   delete vfe_coll;
   delete mesh;
   delete fcoeff;
   delete ucoeff;
   return 0;
}



