//                        MFEM EDG/HDG example
//
// Compile with: make advection
//
// Sample runs:  hdg_advection -o 1 -r 1 -tr 4 -no-vis
//            hdg_advection -o 5 -r 1 -tr 4 -no-vis
//            hdg_advection -o 1 -r 4 -tr 1
//            hdg_advection -o 5 -r 4 -tr 1
//            hdg_advection -o 1 -r 1 -tr 4 -no-vis -m ../data/inline-tri.mesh
//            hdg_advection -o 5 -r 1 -tr 4 -no-vis -m ../data/inline-tri.mesh
//            hdg_advection -o 1 -r 5 -tr 1 -m ../data/inline-tri.mesh
//            hdg_advection -o 5 -r 5 -tr 1 -m ../data/inline-tri.mesh
//
//
// Description:  This example code demonstrates the use of MFEM to define a
//               finite element discretization of the advection-reaction problem
//               mu u + a.grad(u) = f with inhomogeneous Neumann boundary conditions.
//               Specifically, we discretize using a HDG space of the
//               specified order.
//
// The weak form is: seek (u,ubar) such that for all (v, vbar)
//
// \mu (u,v)   + (v, a.grad(u) - < 1, [zeta a.n u v] >  + < ubar, [zeta a.n v] > = (f, w)
// < ubar, [zeta a.n v] >  +
//       < 1, [zeta a.n ubar vbar] > + < 1, [(1-zeta) a.n ubar vbar >_{\Gamma_N} = < g, vbar >
//
// where (.,.) is the d-dimensional L2 product, <.,.> is the d-1 dimensional L2 product,
// zeta = 1 for inflow boundaries, and 0 otherwise.
//
// The discretization is based on the paper:
//
// G. N. Wells, Analysis of an interface stabilized finite element method: the advection-diffusion-reaction equation, SIAM J. Numer. Anal., 2011, 49:1, 87--109.
//
// Contributed by: T. Horvath, Oakland University
//                 S. Rhebergen, A. Sivas, University of Waterloo


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//---------------------------------------------------------------------
// Exact solution and r.h.s.. See below for implementation.
double u_exact(const Vector &x);
double f_rhs  (const Vector &x);
void advection_function(const Vector &x, Vector &v);
int dim;
//---------------------------------------------------------------------
int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-tri.mesh";
   int order = 1;
   int total_ref_levels = 4;
   int initial_ref_levels = 0;
   bool visualization = true;
   bool save = true;
   bool hdg = true;
   double memA = 0.0;
   double memB = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree > 1).");
   args.AddOption(&initial_ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly for the initial calculation.");
   args.AddOption(&total_ref_levels, "-tr", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&save, "-save", "--save-files", "-no-save",
                  "--no-save-files",
                  "Enable or disable file saving.");
   args.AddOption(&hdg, "-hdg", "--hybrid", "-edg",
                  "--embedded",
                  "HDG / EDG option.");
   args.AddOption(&memA, "-memA", "--memoryA",
                  "Storage of A.");
   args.AddOption(&memB, "-memB", "--memoryB",
                  "Storage of B.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (order < 1)
   {
      cout << "Polynomial order should be > 0. Changing to order 1.";
      order = 1;
   }

   // memA, memB \in [0,1], memB <= memA
   if (memB > memA)
   {
      std::cout << "memB cannot be more than memA. Resetting to be equal" << std::endl
                << std::flush;
      memA = memB;
   }
   if (memA > 1.0)
   {
      std::cout << "memA cannot be more than 1. Resetting to 1" << std::endl <<
                std::flush;
      memA = 1.0;
   }
   else if (memA < 0.0)
   {
      std::cout << "memA cannot be less than 0. Resetting to 0." << std::endl <<
                std::flush;
      memA = 0.0;
   }
   if (memB > 1.0)
   {
      std::cout << "memB cannot be more than 1. Resetting to 1" << std::endl <<
                std::flush;
      memB = 1.0;
   }
   else if (memB < 0.0)
   {
      std::cout << "memB cannot be less than 0. Resetting to 0." << std::endl <<
                std::flush;
      memB = 0.0;
   }

   // 2. Read the mesh from the given mesh file. Refine it up to the initial_ref_levels.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   for (int ii=0; ii<initial_ref_levels; ii++)
   {
      mesh->UniformRefinement();
   }

   // 3. Define the vectors that will contain the errors and the iteration count at every refinement level
   Vector l2errors(total_ref_levels);
   Array<int> iterativeMethodIts(total_ref_levels);

   // 4. Define the finite element spaces on the mesh.
   FiniteElementCollection *Uh_fec(new DG_FECollection(order, dim));
   FiniteElementCollection *Uhbar_fec = NULL;
   if (hdg)
   {
      Uhbar_fec = new DG_Interface_FECollection(order, dim);
   }
   else
   {
      Uhbar_fec = new H1_Trace_FECollection(order, dim);
   }

   // Finite element spaces:
   // Uh_space is the DG space on elements
   // ubar_space is the DG space on faces
   FiniteElementSpace *Uh_space(new FiniteElementSpace(mesh, Uh_fec));
   FiniteElementSpace *Uhbar_space(new FiniteElementSpace(mesh, Uhbar_fec));

   // 5. Define the coefficients
   ConstantCoefficient mu(1.0); // reaction constant
   // Given boundary condition / exact solution
   FunctionCoefficient ucoeff(u_exact);
   // Given advection vector:
   VectorFunctionCoefficient advection(dim, advection_function);

   // 6. Define the different forms and gridfunctions.

   // We apply static condensation to the system
   //
   // [ A   B ] [  u   ] = [ F ]
   // [ C   D ] [ ubar ]   [ H ]
   //
   // Eliminating u we find the global system
   //
   // S ubar = G
   //
   // where S = - C A^{-1} B + D and G = -C A^{-1} F + H.
   // Having solved this system for ubar, we can compute u from
   //
   // u = A^{-1} (F - B ubar)

   // Set up the linear form fform(.) which corresponds to the right-hand
   // side of the linear system, which in this case is (f, phi_i) and
   // phi_i are the basis functions in the finite element Uh_space.
   FunctionCoefficient fcoeff(f_rhs);
   LinearForm *fform(new LinearForm);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   // Set up the linear form gform(.) which corresponds to the right-hand
   // side of the linear system, which in this case is <g, bar_phi_i>_{Gamma_N} and
   // bar_phi_i are the basis functions in the finite element Uhbar_space.
   LinearForm *gform(new LinearForm);
   gform->AddSktBoundaryNeumannIntegrator(new HDGInflowLFIntegrator(ucoeff,
                                                                    advection));

   // Set up the bilinear form for the whole system. HDGBilinearForm2 can compute
   // the Schur complement locally for a 2x2 problem.
   HDGBilinearForm *AVarf(new HDGBilinearForm(Uh_space, Uhbar_space));
   AVarf->AddHDGDomainIntegrator(
      new HDGDomainIntegratorAdvection(mu, advection));
   AVarf->AddHDGFaceIntegrator(
      new HDGFaceIntegratorAdvection(advection));

   GridFunction ubar(Uhbar_space);
   GridFunction u(Uh_space);

   for (int ref_levels = 0; ref_levels < total_ref_levels; ref_levels++)
   {
      // 7. Define the right hand side vectors
      int dimUh   = Uh_space->GetVSize();
      int dimUhbar = Uhbar_space->GetVSize();

      std::cout << "***********************************************************\n";
      std::cout << "dim(Uh) = " << dimUh << "\n";
      std::cout << "dim(Uhbar) = " << dimUhbar << "\n";
      std::cout << "***********************************************************\n";

      Vector rhs_F(dimUh);
      Vector rhs_G(dimUhbar);
      Vector UBAR(dimUhbar);

      // 8. Assemble the RHS and the bilinear forms
      fform->Update(Uh_space, rhs_F, 0);
      fform->Assemble(); // This is a vector

      gform->Update(Uhbar_space, rhs_G, 0);
      gform->Assemble(); // This is a vector

      // Compute and Finalize the Schur complement
      GridFunction *F = new GridFunction(Uh_space, rhs_F);
      AVarf->AssembleSC(F, memA, memB);
      AVarf->Finalize();

      SparseMatrix *SC = AVarf->SpMatSC();

      Vector *rhs_SC = AVarf->VectorSC();

      // AVarf->VectorSC() provides -C*A^{-1} F, but the RHS for the
      // Schur complement is  G - C*A^{-1} F
      *rhs_SC += rhs_G;

      // 9. Solve the Schur complement system
      const int maxIter(1000);
      const double rtol(1.e-15);
      const double atol(0.0);
      const int PrintLevel = -1;
      GSSmoother M(*SC, 1, 1);
      BiCGSTABSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*SC);
      solver.SetPrintLevel(PrintLevel);
      solver.SetPreconditioner(M);
      ubar = 0.0;
      chrono.Clear();
      chrono.Start();
      solver.Mult(*rhs_SC, ubar);
      chrono.Stop();

      if (solver.GetConverged())
         std::cout << "Iterative method converged in "
                   << solver.GetNumIterations()
                   << " iterations with a residual norm of "
                   << solver.GetFinalNorm() << ".\n";
      else
         std::cout << "Iterative method did not converge in "
                   << solver.GetNumIterations()
                   << " iterations. Residual norm is "
                   << solver.GetFinalNorm() << ".\n";

      std::cout << "Iterative method solver took "
                << chrono.RealTime() << "s. \n";
      iterativeMethodIts[ref_levels] = solver.GetNumIterations();

      // Delete the SC matrix to save memory
      SC = NULL;

      // 10. Reconstruction
      // Reconstruct the solution u from the facet solution ubar
      AVarf->Reconstruct(F, &ubar, &u);

      // 11. Compute the discretization error
      const int order_quad = max(2, 2*order+2);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      const double err_u  = u.ComputeL2Error(ucoeff, irs);
      l2errors(ref_levels) = fabs(
                                err_u); // fabs() to avoid negative values that ComputeL2Error can create

      // 12. Save the mesh and the solution.
      if (save)
      {
         ofstream mesh_ofs("refined.mesh");
         mesh_ofs.precision(8);
         mesh->Print(mesh_ofs);

         ofstream u_ofs("sol_u.gf");
         u_ofs.precision(8);
         u.Save(u_ofs);

         ofstream ubar_ofs("sol_lambda.gf");
         ubar_ofs.precision(8);
         ubar.Save(ubar_ofs);
      }

      // 13. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream u_sock(vishost, visport);
         u_sock.precision(8);
         u_sock << "solution\n" << *mesh << u << flush;
      }

      // 14. Refine the mesh to increase the resolution and update the spaces and the forms.
      mesh->UniformRefinement();

      Uh_space->Update(0);
      Uhbar_space->Update(0);

      AVarf->Update();

      u.Update();
      ubar.Update();

   }

   // 15. Print the results and compute the rates
   std::cout << "\n\n---------------------------------\n";
   std::cout << "level  l2errors  order iterations\n";
   std::cout << "---------------------------------\n";
   for (int ref_levels = 0; ref_levels < total_ref_levels; ref_levels++)
   {
      if (ref_levels == 0)
      {
         std::cout << "  " << ref_levels << "   "
                   << std::setprecision(2) << std::scientific
                   << l2errors(ref_levels)
                   << "  " << "-    " << "    "
                   << iterativeMethodIts[ref_levels] << std::endl;
      }
      else
      {
         const double order = log(l2errors(ref_levels)/l2errors(ref_levels-1))/log(0.5);
         std::cout << "  " << ref_levels << "   "
                   << std::setprecision(2) << std::scientific
                   << l2errors(ref_levels)
                   << "  " << std::setprecision(4) << std::fixed
                   << order << "   "
                   << iterativeMethodIts[ref_levels] << std::endl;
      }
   }
   std::cout << "\n\n";

   // 16. Free the used memory.
   delete mesh;
   delete Uh_fec;
   delete Uhbar_fec;
   delete Uh_space;
   delete Uhbar_space;
   delete fform;
   delete gform;
   delete AVarf;

   std::cout << "Done." << std::endl ;

   return 0;
}
//---------------------------------------------------------------------
// Exact solution
double u_exact(const Vector &x)
{
   double ue = 0.0;
   const double xx = x(0);
   const double yy = x(1);
   if (dim == 2)
   {
      ue = 1.0 + sin(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+yy));
   }
   else if (dim == 3)
   {
      const double zz = x(2);
      ue = 1.0 + sin(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+zz));
   }

   return ue;
}
//---------------------------------------------------------------------
// Rhs function
double f_rhs(const Vector &x)
{
   double rhs = 0.0;
   const double ax = 0.8;
   const double ay = 0.6;
   const double mu = 1.0;
   const double xx = x(0);
   const double yy = x(1);

   if (dim == 2)
   {
      const double uu = 1.0 + sin(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+yy));
      const double dudx = 0.125 * M_PI * (1.0+yy) * (1.0+yy)
                          * cos(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+yy));
      const double dudy =  0.25 * M_PI * (1.0+xx) * (1.0+yy)
                           * cos(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+yy));

      rhs = mu * uu + ax * dudx + ay * dudy;
   }

   if (dim == 3)
   {
      const double az = 0.7;
      const double zz = x(2);
      const double uu = 1.0 + sin(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+zz));
      const double dudx = 0.125 * M_PI * (1.0+yy) * (1.0+zz)
                          * cos(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+zz));

      const double dudy = 0.125 * M_PI * (1.0+xx) * (1.0+zz)
                          * cos(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+zz));

      const double dudz = 0.125 * M_PI * (1.0+xx) * (1.0+yy)
                          * cos(0.125 * M_PI * (1.0+xx) * (1.0+yy) * (1.0+zz));

      rhs = mu * uu + ax * dudx + ay * dudy + az * dudz;
   }

   return rhs;
}
//---------------------------------------------------------------------
// Advection vector
void advection_function(const Vector &x, Vector &v)
{
   if (dim == 3)
   {
      v(0) = 0.8;
      v(1) = 0.6;
      v(2) = 0.7;
   }
   else if (dim == 2)
   {
      v(0) = 0.8;
      v(1) = 0.6;
   }
}
