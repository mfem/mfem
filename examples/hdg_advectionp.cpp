//                        MFEM EDG/HDG example
//
// Compile with: make advectionp
//
// Sample runs:  mpirun -np 1 hdg_advectionp -o 1 -r 4 -tr 1 -no-vis // test scalability
//            mpirun -np 2 hdg_advectionp -o 1 -r 4 -tr 1 -no-vis // test scalability
//            mpirun -np 4 hdg_advectionp -o 1 -r 4 -tr 1 -no-vis // test scalability
//            mpirun -np 4 hdg_advectionp -o 1 -r 0 -tr 3 -no-vis // test conv rates
//            mpirun -np 4 hdg_advectionp -o 1 -r 2 -tr 1 -no-vis // test scalability
//            mpirun -np 2 hdg_advectionp -o 5 -r 4 -tr 1 -no-vis // test conv rates
//            mpirun -np 2 hdg_advectionp -o 5 -r 5 -tr 1 -m ../data/inline-tri.mesh -no-vis // test
//
// Description:  This example code demonstrates the use of MFEM to define a
//               finite element discretization of the advection-reaction problem
//               mu u + a.grad(u) = f with inhomogeneous Neumann boundary conditions.
//               Specifically, we discretize using a HDG space of the
//               specified order.
//
// The weak form is: seek (u,ubar) such that for all (v, vbar)
//
// \mu (u,v)   + (v, a.grad(u))  - < 1, [zeta a.n u v] >  + < ubar, [zeta a.n v] > = (f, w)
// < ubar, [zeta a.n v] >  +
//       < 1, [zeta a.n ubar vbar] > + < 1, [(1-zeta) a.n ubar vbar >_{\Gamma_N} = < g, vbar >
//
// where (.,.) is the d-dimensional L2 product, <.,.> is the d-1 dimensional L2 product.
//
// The discretization is based on the paper:
//
// G. N. Wells, Analysis of an interface stabilized finite element method: the advection-diffusion-reaction equation, SIAM J. Numer. Anal., 2011, 49:1, 87--109.
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//             University of Waterloo

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

   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   double assemblyTime, solveTime, reconstructTime;
   double GassemblyTime, GsolveTime, GreconstructTime;

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-tri.mesh";
   int order          = 1;
   int initial_ref_levels = 0;
   int total_ref_levels  = 1;
   bool visualization   = true;
   bool save          = true;
   bool hdg = true;
   double memA = 0.0;
   double memB = 0.0;
   bool petsc = false;
   bool verbose = (myid == 0);
   const char *petscrc_file = "";

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
   args.AddOption(&petsc, "-petsc", "--use-petsc",
                  "-no-petsc", "--no-use-petsc",
                  "Enable or disable SC solver.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

#ifdef MFEM_USE_PETSC
   // We initialize PETSc
   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
#endif

#ifndef MFEM_USE_PETSC
   if (petsc)
   {
      std::cout << "MFEM does not use PETSc. Change the solver to hypre" << std::endl
                << std::flush;
      petsc = false;
   }
#endif

   if (order < 1)
   {
      cout << "Polynomial order should be > 0. Changing to order 1.";
      order = 1;
   }

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

   // 3. Read the mesh from the given mesh file. Refine it up to the initial_ref_levels.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   for (int ii=0; ii<initial_ref_levels; ii++)
   {
      mesh->UniformRefinement();
   }

   // Define a parallel mesh. The serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 4. Define the vectors that will contain the errors and the iteration count at every refinement level
   Vector l2errors(total_ref_levels);
   Array<int> iterativeMethodIts(total_ref_levels);

   // Define parallel finite element spaces on the parallel mesh.
   // Uh_space is the DG space on elements
   // ubar_space is the DG space on faces

   if (order < 1)
   {
      cout << "Polynomial order should be > 0. Changing to order 1.";
      order = 1;
   }

   // 5. Define the finite element spaces on the mesh.
   FiniteElementCollection *Uh_fec(new DG_FECollection(order, dim));
   FiniteElementCollection *Uhbar_fec = NULL;
   if (hdg)
	   Uhbar_fec = new DG_Interface_FECollection(order, dim);
   else
	   Uhbar_fec = new H1_Trace_FECollection(order, dim);

   ParFiniteElementSpace *Uh_space   = new ParFiniteElementSpace(pmesh, Uh_fec);
   ParFiniteElementSpace *Uhbar_space = new ParFiniteElementSpace(pmesh,
                                                                  Uhbar_fec);

   // 6. Define the coefficients
   FunctionCoefficient fcoeff(f_rhs);
   FunctionCoefficient ucoeff(u_exact);
   ConstantCoefficient mu(1.0); // reaction constant

   // Given advection vector:
   VectorFunctionCoefficient advection(dim, advection_function);

   // 7. Define the different forms and gridfunctions.

   // Set up the linear form fform(.) which corresponds to the right-hand
   // side of the linear system, which in this case is (f, phi_i) and
   // phi_i are the basis functions in the finite element Uh_space.

   ParLinearForm *fform = new ParLinearForm(Uh_space);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   // Set up the linear form gform(.) which corresponds to the right-hand
   // side of the linear system, which in this case is <g, bar_phi_i>_{Gamma_N} and
   // bar_phi_i are the basis functions in the finite element Uhbar_space.

   ParLinearForm *gform = new ParLinearForm(Uhbar_space);
   gform->AddSktBoundaryNeumannIntegrator(new HDGInflowLFIntegrator(ucoeff,
                                                                    advection));

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

   // Set up the bilinear form for the whole system. ParHDGBilinearForm2 can compute
   // the Schur complement locally for a 2x2 problem.
   HDGBilinearForm *AVarf(new HDGBilinearForm(Uh_space, Uhbar_space, true));
   AVarf->AddHDGDomainIntegrator(
      new HDGDomainIntegratorAdvection(mu, advection));
   AVarf->AddHDGFaceIntegrator(
      new HDGFaceIntegratorAdvection(advection));

   ParGridFunction u(Uh_space);
   ParGridFunction ubar(Uhbar_space);

   for (int ref_levels = 0; ref_levels < total_ref_levels; ref_levels++)
   {
      // 8. Define the right hand side vectors
      HYPRE_Int dimUh   = Uh_space->GlobalTrueVSize();
      HYPRE_Int dimUhbar = Uhbar_space->GlobalTrueVSize();

      if (verbose)
      {
         std::cout << "****************************************************\n";
         std::cout << "dim(Uh)      = " << dimUh << "\n";
         std::cout << "dim(Uhbar)   = " << dimUhbar << "\n";
         std::cout << "dim(Uh+Uhbar) = " << dimUh + dimUhbar << "\n";
         std::cout << "****************************************************\n";
      }

      ubar = 0.0;
      HypreParVector *UBAR = ubar.ParallelProject();

      // 10. Assemble the RHS and the bilinear forms
      Vector rhs_F(Uh_space->GlobalVSize());
      Vector rhs_H(Uhbar_space->GlobalVSize());

      // Linear forms
      fform->Update(Uh_space);
      fform->Assemble();
      gform->Update(Uhbar_space);
      gform->Assemble();

      HypreParVector *trueF;
      trueF = fform->ParallelAssemble();

      HypreParVector *trueG;
      trueG = gform->ParallelAssemble();
      // Create a ParGridFunction from the right hand side.
      ParGridFunction *F = new ParGridFunction(Uh_space, trueF);

      chrono.Clear();
      chrono.Start();
      AVarf->AssembleSC(F, memA, memB);
      chrono.Stop();
      AVarf->Finalize();

      assemblyTime = chrono.RealTime();

      HypreParMatrix *SC = AVarf->ParallelAssembleSC();

      HypreParVector *rhs_SC = AVarf->ParallelVectorSC();
      // AVarf->ParallelVectorSC() provides -C*A^{-1} F, but the RHS for the
      // Schur complement is  G - C*A^{-1} F
      rhs_SC->Add(1.0, *trueG);

      double tol = 1.0e-12;
      int maxIter = 1000;
      int PrintLevel = -1;

      // 12. Solve the linear system
      if (petsc)
      {
#ifdef MFEM_USE_PETSC
         // Solver using PETSc
         //=======================
         PetscLinearSolver *petsc_solver;
         PetscPreconditioner *petsc_precon= NULL;
         petsc_solver = new PetscLinearSolver(MPI_COMM_WORLD, "solver_");
         petsc_precon = new PetscPreconditioner(MPI_COMM_WORLD,*SC,"solver_");
         petsc_solver->SetOperator(*SC);
         petsc_solver->SetPreconditioner(*petsc_precon);
         petsc_solver->SetTol(tol);
         petsc_solver->SetAbsTol(0.0);
         petsc_solver->SetMaxIter(maxIter);
         petsc_solver->SetPrintLevel(PrintLevel);
         chrono.Clear();
         chrono.Start();
         petsc_solver->Mult(*rhs_SC, *UBAR);
         chrono.Stop();

         if (verbose)
         {
            if (petsc_solver->GetConverged())
               std::cout << "Solver converged in " << petsc_solver->GetNumIterations()
                         << " iterations with a residual norm of " << petsc_solver->GetFinalNorm() <<
                         ".\n";
            else
               std::cout << "Solver did not converge in " << petsc_solver->GetNumIterations()
                         << " iterations. Residual norm is " << petsc_solver->GetFinalNorm() << ".\n";

            std::cout << "Solver solver took " << chrono.RealTime() << "s. \n";
         }
         delete petsc_solver;
         delete petsc_precon;
#endif
      }
      else
      {
         HypreSolver *pdiag = new HypreDiagScale(*SC);
         HypreGMRES *itsolver = new HypreGMRES(*SC);
         itsolver->SetTol(tol);
         itsolver->SetMaxIter(maxIter);
         itsolver->SetPrintLevel(PrintLevel);
         itsolver->SetPreconditioner(*pdiag);
         itsolver->SetZeroInintialIterate();
         chrono.Clear();
         chrono.Start();
         itsolver->Mult(*rhs_SC, *UBAR);
         chrono.Stop();

         int numIterations = 0;
         itsolver->GetNumIterations(numIterations);
         if (verbose)
         {
            std::cout << "\nIterative method converged in "
                      << numIterations << ".\n";

            iterativeMethodIts[ref_levels] = numIterations;
            std::cout << "Iterative solver took " << chrono.RealTime() << "s. \n";
         }
      }

      // Delete the SC matrix to save memory
      SC = NULL;
      solveTime = chrono.RealTime();

      ubar = ParGridFunction(Uhbar_space, UBAR);

      chrono.Clear();
      chrono.Start();
      AVarf->Reconstruct(F, &ubar, &u);
      chrono.Stop();

      reconstructTime = chrono.RealTime();

      const int order_quad = max(2, 2*order+2);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      const double err_u  = u.ComputeL2Error(ucoeff, irs);
      if (verbose)
      {
         std::cout << "\nL2 error " << err_u << ".\n";
      }
      l2errors(ref_levels) = fabs(
                                err_u); // fabs() to avoid negative values that ComputeL2Error can create

      // 14. Save the mesh and the solution.
      if (save)
      {
         ostringstream mesh_name, u_name, ubar_name;
         mesh_name << "mesh." << setfill('0') << setw(6) << myid;
         u_name << "sol_u." << setfill('0') << setw(6) << myid;
         ubar_name << "sol_ubar." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);

         ofstream u_ofs(u_name.str().c_str());
         u_ofs.precision(8);
         u.Save(u_ofs);
      }

      // 15. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream u_sock(vishost, visport);
         u_sock << "parallel " << num_procs << " " << myid << "\n";
         u_sock.precision(8);
         u_sock << "solution\n" << *pmesh << u << "window_title 'Velocity'"
                << endl;
         // Make sure all ranks have sent their 'u' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());
      }

      // 16. Refine the mesh to increase the resolution and update the spaces and the forms. Print the runtimes
      pmesh->UniformRefinement();

      Uh_space->Update(0);
      Uhbar_space->Update(0);

      AVarf->Update();

      u.Update();
      ubar.Update();
      MPI_Reduce(&assemblyTime,&GassemblyTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(&solveTime,&GsolveTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(&reconstructTime,&GreconstructTime,1,MPI_DOUBLE,MPI_MAX,0,
                 MPI_COMM_WORLD);

      if (verbose)
      {
         printf("\t Assembly time   = %.2f\n",GassemblyTime);
         printf("\t Solve time      = %.2f\n",GsolveTime);
         printf("\t Reconstruct time = %.2f\n",GreconstructTime);
      }
   }

   // 17. Print the results
   if (verbose)
   {
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
            const double order = log(l2errors(ref_levels)/l2errors(ref_levels-1))
                                 /log(0.5);
            std::cout << "  " << ref_levels << "   "
                      << std::setprecision(2) << std::scientific
                      << l2errors(ref_levels)
                      << "  " << std::setprecision(4) << std::fixed
                      << order << "   "
                      << iterativeMethodIts[ref_levels] << std::endl;
         }
      }
      std::cout << "\n\n";
   }


   // 19. Free the used memory.
   delete pmesh;
   delete Uh_fec;
   delete Uhbar_fec;
   delete Uh_space;
   delete Uhbar_space;
   delete fform;
   delete gform;
   delete AVarf;

#ifdef MFEM_USE_PETSC
   MFEMFinalizePetsc();
#endif
   MPI_Finalize();

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
