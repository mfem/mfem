//                  MFEM Example Hybridizable DG
//
// Compile with: make hdg_poissonp
//
// Sample runs:  mpirun -np 1 hdg_poissonp -o 1 -r 1 -tr 4 -no-vis
//         mpirun -np 2 hdg_poissonp -o 5 -r 1 -tr 4 -no-vis
//         mpirun -np 2 hdg_poissonp -o 1 -r 4 -tr 1
//         mpirun -np 3 hdg_poissonp -o 5 -r 4 -tr 1
//         mpirun -np 2 hdg_poissonp -o 1 -r 1 -tr 4 -no-vis -m ../data/inline-tri.mesh
//         mpirun -np 2 hdg_poissonp -o 5 -r 1 -tr 4 -no-vis -m ../data/inline-tri.mesh
//         mpirun -np 4 hdg_poissonp -o 1 -r 5 -tr 1 -m ../data/inline-tri.mesh
//         mpirun -np 2 hdg_poissonp -o 5 -r 5 -tr 1 -m ../data/inline-tri.mesh
//
// Description:  This example code solves the 2D/3D diffusion problem
//                -\nu Delta u = f
//               with Dirichlet boundary conditions, using HDG discretization.
//
// The methods approximates the solution u, the diffusive flux q = -\nu \nabla u,
// and the restriction of u to the faces, denoted by lambda.
//
// The weak form is: seek (q,u,\lambda) such that for all (v, w, \mu)
//
// -\nu^{-1}(q, v)       + (u, div(v))       - <\lambda, v \cdot n>      = 0
//  (div(q), w)          + <\tau u, w>       - <\tau \lambda, w>         = (f, w)
// -<[[q \cdot n]], \mu> - <[[\tau u]], \mu> + <[[(\tau \lambda]], \mu>  = 0
//
// where [[.]] is the jump operator, (.,.) is the d-dimensional L2 product,
// <.,.> is the d-1 dimensional L2 product.
//
// The discretization is based on the paper:
//
// N.C. Nguyen, J. Peraire, B. Cockburn, An implicit high-order hybridizable
// discontinuous Galerkin method for linear convection–diffusion equations,
// J. Comput. Phys., 2009, 228:9, 3232--3254.
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
double uFun_ex(const Vector & x);
void qFun_ex(const Vector & x, Vector & q);
double fFun(const Vector & x);
double diff;

// We can minimize the expression |\nu \nabla u_h^* + q_h |^2 over a single element K,
// for p+1 degree u_h^*, with the constraint \int_K u_h^* = \int_K u_h, so the mean
// of u_h^* is the same as the one of u_h.
//
// This results in the problem
//
// (nabla w_h, \nu \nabla u_h^*) = -(nabla w_h, q_h)
// (1, u_h^*)                    = (1, u_h)
//
// Since the fist equation on its own would generate a singular problem
// the last line of the system is rewritten by the second equation.
//
// This elementwise operation will provide a superconvergent solution
// \|u-u_h\|_{L^2} < C h^{p+2} |u|_{p+1}
class pHDGPostProcessing
{
private:
   ParGridFunction *q, *u;

   ParFiniteElementSpace *pfes;

   Coefficient *diffcoeff;
protected:
   const IntegrationRule *IntRule;

public:
   pHDGPostProcessing(ParFiniteElementSpace *f, ParGridFunction &_q,
                      ParGridFunction &_u, Coefficient &_diffcoeff)
      : q(&_q), u(&_u), pfes(f), diffcoeff(&_diffcoeff)
   {
      IntRule = NULL;
   }

   void Postprocessing(ParGridFunction &u_postprocessed) ;
};

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   double assemblyTime, solveTime, reconstructTime, pprocessTime;
   double GassemblyTime, GsolveTime, GreconstructTime, GpprocessTime;

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-tri.mesh";
   int order = 1;
   int initial_ref_levels = 0;
   int total_ref_levels = 1;
   bool visualization = true;
   bool verbose = (myid == 0);
   bool post = true;
   bool save = true;
   bool hdg = true;
   double memA = 0.0;
   double memB = 0.0;
   bool petsc = false;
   const char *petscrc_file = "";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&initial_ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly for the initial calculation.");
   args.AddOption(&total_ref_levels, "-tr", "--totalrefine",
                  "Number of times to refine the mesh uniformly to get the convergence rates.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&post, "-post", "--postprocessing",
                  "-no-post", "--no-postprocessing",
                  "Enable or disable postprocessing.");
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
      if (verbose)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (verbose)
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

   // 3. Read the mesh from the given mesh file. Refine it up to the initial_ref_levels.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int ii=0; ii<initial_ref_levels; ii++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 4. Vectors for the different discretization errors
   Vector u_l2errors(total_ref_levels), q_l2errors(total_ref_levels),
          mean_l2errors(total_ref_levels), u_star_l2errors(total_ref_levels);

   // 5. Define a finite element collections and spaces on the mesh.
   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *face = NULL;
   if (hdg)
   {
      face = new DG_Interface_FECollection(order, dim);
   }
   else
   {
      face = new H1_Trace_FECollection(order, dim);
   }

   // Finite element spaces:
   // V_space is the vector valued DG space on elements for q_h
   // W_space is the scalar DG space on elements for u_h
   // M_space is the DG space on faces for lambda_h
   ParFiniteElementSpace *V_space = new ParFiniteElementSpace(pmesh, dg_coll, dim);
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, dg_coll);
   ParFiniteElementSpace *M_space = new ParFiniteElementSpace(pmesh, face);

   // 6. Define the coefficients, the exact solutions, the right hand side and the diffusion coefficient along with the diffusion penalty parameter.
   FunctionCoefficient fcoeff(fFun);

   FunctionCoefficient ucoeff(uFun_ex);
   VectorFunctionCoefficient qcoeff(dim, qFun_ex);

   diff = 1.;
   ConstantCoefficient diffusion(diff); // diffusion constant
   double tau_D = 5.0;

   // 7. Define the different forms and gridfunctions.
   HDGBilinearForm *AVarf(new HDGBilinearForm(V_space, W_space, M_space, true));
   AVarf->AddHDGDomainIntegrator(new HDGDomainIntegratorDiffusion(diffusion));
   AVarf->AddHDGFaceIntegrator(new HDGFaceIntegratorDiffusion(tau_D));

   ParGridFunction lambda(M_space);
   ParGridFunction q_variable(V_space), u_variable(W_space);

   ParLinearForm *fform(new ParLinearForm);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   for (int ref_levels = initial_ref_levels;
        ref_levels < (initial_ref_levels + total_ref_levels); ref_levels++)
   {
      // 8. Compute the problem size and define the right hand side vectors
      HYPRE_Int dimV = V_space->GlobalTrueVSize();
      HYPRE_Int dimW = W_space->GlobalTrueVSize();
      HYPRE_Int dimM = M_space->GlobalTrueVSize();

      if (verbose)
      {
         std::cout << "***********************************************************\n";
         std::cout << "dim(V) = " << dimV << "\n";
         std::cout << "dim(W) = " << dimW << "\n";
         std::cout << "dim(M) = " << dimM << "\n";
         std::cout << "dim(V+W+M) = " << dimV + dimW + dimM << "\n";
         std::cout << "***********************************************************\n";
      }

      HypreParVector *trueR(new HypreParVector(V_space));
      *trueR = 0.0;

      HypreParVector *trueF;

      // 9. To eliminate the boundary conditions we project the BC to a grid function
      // defined for the facet unknowns.
      lambda.ProjectCoefficientSkeleton(ucoeff);

      HypreParVector *Lambda =  new HypreParVector(M_space);

      *Lambda = 0.0;

      Lambda->Add(1.0, lambda);

      lambda.Print();

      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;

      // 10. Assemble the RHS and the Schur complement
      fform->Update(W_space);
      fform->Assemble();

      trueF = fform->ParallelAssemble();

      // Creating a gridfunctions for the elimination of the boundary
      ParGridFunction *R = new ParGridFunction(V_space, trueR);
      ParGridFunction *F = new ParGridFunction(W_space, trueF);

      chrono.Clear();
      chrono.Start();
      AVarf->AssembleSC(R, F, ess_bdr, lambda, memA, memB);
      chrono.Stop();
      AVarf->Finalize();

      assemblyTime = chrono.RealTime();
      HypreParMatrix *SC = AVarf->ParallelAssembleSC();

      HypreParVector *rhs_SC = AVarf->ParallelVectorSC();
      // AVarf->ParallelVectorSC() provides -C*A^{-1} RF, the RHS for the
      // Schur complement is  L - C*A^{-1} RF, but L is zero for this case

      // 11. Solve the Schur complement system
      double tol = 1.0e-12;
      int maxIter = 1000;
      int PrintLevel = -1;

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
         petsc_solver->Mult(*rhs_SC, *Lambda);
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
         HypreBoomerAMG *amg = new HypreBoomerAMG(*SC);
         HyprePCG *pcg = new HyprePCG(*SC);
         pcg->SetTol(tol);
         pcg->SetMaxIter(maxIter);
         amg->SetPrintLevel(PrintLevel);
         pcg->SetPrintLevel(PrintLevel);
         pcg->SetPreconditioner(*amg);
         pcg->SetZeroInintialIterate();
         chrono.Clear();
         chrono.Start();
         pcg->Mult(*rhs_SC, *Lambda);
         chrono.Stop();

         int numIterations = 0;
         pcg->GetNumIterations(numIterations);

         if (verbose)
         {
            std::cout << "\nIterative method converged in "
                      << numIterations << ".\n";

            std::cout << "Iterative solver took " << chrono.RealTime() << "s. \n";
         }
      }

      // Delete the SC matrix to save memory
      SC = NULL;
      solveTime = chrono.RealTime();

      // 12. Reconstruction
      // Create a gridfunction from the right hand side.
      // It is mostly important for the parallel code,
      // here it is done this way to make the 2 codes more similar
      lambda = ParGridFunction(M_space, Lambda);

      lambda.Print();

      chrono.Clear();
      chrono.Start();
      AVarf->Reconstruct(R, F, &lambda, &q_variable, &u_variable);
      chrono.Stop();

      reconstructTime = chrono.RealTime();

      // 13. Compute the discretization error
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double err_u   = u_variable.ComputeL2Error(ucoeff, irs);
      double norm_u   = ComputeGlobalLpNorm(2., ucoeff, *pmesh, irs);
      double err_q   = q_variable.ComputeL2Error(qcoeff, irs);
      double norm_q   = ComputeGlobalLpNorm(2., qcoeff, *pmesh, irs);
      double err_mean  = u_variable.ComputeMeanLpError(2.0, ucoeff, irs);

      if (verbose)
      {
         std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
         std::cout << "|| q_h - q_ex || / || q_ex || = " << err_q / norm_q << "\n";
         std::cout << "|| u_h - u_ex || = " << err_u << "\n";
         std::cout << "|| q_h - q_ex || = " << err_q << "\n";
         std::cout << "|| mean(u_h) - mean(u_ex) || = " << err_mean << "\n";
      }

      u_l2errors(ref_levels-initial_ref_levels) = fabs(err_u);
      q_l2errors(ref_levels-initial_ref_levels) = fabs(err_q);
      mean_l2errors(ref_levels-initial_ref_levels) = fabs(err_mean);


      // 14. Save the mesh and the solution.
      if (save)
      {
         ostringstream mesh_name, u_name, q_name, lambda_name;
         mesh_name << "mesh." << setfill('0') << setw(6) << myid;
         u_name << "sol_u." << setfill('0') << setw(6) << myid;
         q_name << "sol_q." << setfill('0') << setw(6) << myid;
         lambda_name << "sol_lambda." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);

         ofstream u_ofs(u_name.str().c_str());
         u_ofs.precision(8);
         u_variable.Save(u_ofs);

         ofstream q_ofs(q_name.str().c_str());
         q_ofs.precision(8);
         q_variable.Save(q_ofs);

         ParGridFunction lambda_variable(M_space, Lambda);
         ofstream lambda_ofs(lambda_name.str().c_str());
         lambda_ofs.precision(8);
         lambda_variable.Save(lambda_ofs);
      }

      // 15. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream u_sock(vishost, visport);
         u_sock << "parallel " << num_procs << " " << myid << "\n";
         u_sock.precision(8);
         u_sock << "solution\n" << *pmesh << u_variable << "window_title 'U'"
                << endl;
         // Make sure all ranks have sent their 'u' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());
         socketstream q_sock(vishost, visport);
         q_sock << "parallel " << num_procs << " " << myid << "\n";
         q_sock.precision(8);
         q_sock << "solution\n" << *pmesh << q_variable << "window_title 'Q'"
                << endl;
      }

      // 16. Postprocessing
      if (post)
      {
         FiniteElementCollection *dg_coll_pstar(new DG_FECollection(order+1, dim));
         ParFiniteElementSpace *Vstar_space = new ParFiniteElementSpace(pmesh,
                                                                        dg_coll_pstar);

         ParGridFunction u_post(Vstar_space);

         pHDGPostProcessing *hdgpost(new pHDGPostProcessing(Vstar_space, q_variable,
                                                            u_variable, diffusion));

         chrono.Clear();
         chrono.Start();
         hdgpost->Postprocessing(u_post);
         chrono.Stop();

         pprocessTime = chrono.RealTime();

         order_quad = max(2, 2*order+5);
         const IntegrationRule *irs[Geometry::NumGeom];
         for (int i=0; i < Geometry::NumGeom; ++i)
         {
            irs[i] = &(IntRules.Get(i, order_quad));
         }
         double err_u_post   = u_post.ComputeL2Error(ucoeff, irs);

         u_star_l2errors(ref_levels-initial_ref_levels) = fabs(err_u_post);

         if (verbose)
         {
            std::cout << "|| u^*_h - u_ex || = " << err_u_post << "\n";
         }

         if (save)
         {
            ostringstream u_star_name;
            u_star_name << "sol_u_star." << setfill('0') << setw(6) << myid;
            ofstream u_star_ofs(u_star_name.str().c_str());
            u_star_ofs.precision(8);
            u_post.Save(u_star_ofs);
         }

         if (visualization)
         {
            char vishost[] = "localhost";
            int  visport   = 19916;

            MPI_Barrier(pmesh->GetComm());
            socketstream u_star_sock(vishost, visport);
            u_star_sock << "parallel " << num_procs << " " << myid << "\n";
            u_star_sock.precision(8);
            u_star_sock << "solution\n" << *pmesh << u_post << "window_title 'U_star'"
                        << endl;
         }

      }

      // 17. Refine the mesh to increase the resolution and update the spaces and the forms. Print the runtimes
      pmesh->UniformRefinement();

      V_space->Update(0);
      W_space->Update(0);
      M_space->Update(0);

      AVarf->Update();
      q_variable.Update();
      u_variable.Update();
      lambda.Update();
      MPI_Reduce(&assemblyTime,&GassemblyTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(&solveTime,&GsolveTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(&reconstructTime,&GreconstructTime,1,MPI_DOUBLE,MPI_MAX,0,
                 MPI_COMM_WORLD);
      MPI_Reduce(&pprocessTime,&GpprocessTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

      if (verbose)
      {
         printf("\t Assembly time   = %.2f\n",GassemblyTime);
         printf("\t Solve time     = %.2f\n",GsolveTime);
         printf("\t Reconstruct time = %.2f\n",GreconstructTime);
         printf("\t Postprocess time = %.2f\n",GpprocessTime);
      }
   }

   // 18. Print the results
   if (verbose)
   {
      std::cout << "\n\n-----------------------\n";
      std::cout <<
                "level  u_l2errors  order   q_l2errors  order   mean_l2errors  order u_star_l2errors   order\n";
      std::cout << "-----------------------\n";
      for (int ref_levels = 0; ref_levels < total_ref_levels; ref_levels++)
      {
         if (ref_levels == 0)
         {
            std::cout << "  " << ref_levels << "   "
                      << std::setprecision(2) << std::scientific << u_l2errors(ref_levels)
                      << "   " << " -      "
                      << std::setprecision(2) << std::scientific << q_l2errors(ref_levels)
                      << "    " << " -      "
                      << std::setprecision(2) << std::scientific << mean_l2errors(ref_levels)
                      << "    " << " -      "
                      << std::setprecision(2) << std::scientific << u_star_l2errors(ref_levels)
                      << "    " << " -      " << std::endl;
         }
         else
         {
            double u_order    = log(u_l2errors(ref_levels)/u_l2errors(ref_levels-1))/log(
                                   0.5);
            double q_order    = log(q_l2errors(ref_levels)/q_l2errors(ref_levels-1))/log(
                                   0.5);
            double mean_order   = log(mean_l2errors(ref_levels)/mean_l2errors(
                                         ref_levels-1))/log(0.5);
            double u_star_order = log(u_star_l2errors(ref_levels)/u_star_l2errors(
                                         ref_levels-1))/log(0.5);
            std::cout << "  " << ref_levels << "   "
                      << std::setprecision(2) << std::scientific << u_l2errors(ref_levels)
                      << "  " << std::setprecision(4) << std::fixed << u_order
                      << "   " << std::setprecision(2) << std::scientific << q_l2errors(ref_levels)
                      << "   " << std::setprecision(4) << std::fixed << q_order
                      << "   " << std::setprecision(2) << std::scientific << mean_l2errors(ref_levels)
                      << "   " << std::setprecision(4) << std::fixed << mean_order
                      << "   " << std::setprecision(2) << std::scientific << u_star_l2errors(
                         ref_levels)
                      << "   " << std::setprecision(4) << std::fixed << u_star_order << std::endl;
         }
      }
   }

   // 19. Free the used memory.
   delete pmesh;
   delete V_space;
   delete W_space;
   delete M_space;
   delete AVarf;
   delete fform;
   delete dg_coll;
   delete face;

   if (verbose)
   {
      std::cout << "\n\nDone." << std::endl ;
   }

#ifdef MFEM_USE_PETSC
   MFEMFinalizePetsc();
#endif
   MPI_Finalize();

   return 0;
}

double uFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   int dim = x.Size();

   switch (dim)
   {
      case 2:
      {
         return 1.;//1.0 + xi + sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi);
         break;
      }
      case 3:
      {
         double zi(x(2));
         return xi + sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi)*sin(2.0*M_PI*zi);
         break;
      }
   }

   return 0;
}

void qFun_ex(const Vector & x, Vector & q)
{
   double xi(x(0));
   double yi(x(1));
   int dim = x.Size();

   switch (dim)
   {
      case 2:
      {
         q(0) = 0.;// -diff*1.0 - diff*2.0*M_PI*cos(2.0*M_PI*xi)*sin(2.0*M_PI*yi);
         q(1) = 0.;// 0.0 - diff*2.0*M_PI*sin(2.0*M_PI*xi)*cos(2.0*M_PI*yi);
         break;
      }
      case 3:
      {
         double zi(x(2));
         q(0) = -diff*1.0 - diff*2.0*M_PI*cos(2.0*M_PI*xi)*sin(2.0*M_PI*yi)*sin(
                   2.0*M_PI*zi);
         q(1) =  0.0 - diff*2.0*M_PI*sin(2.0*M_PI*xi)*cos(2.0*M_PI*yi)*sin(2.0*M_PI*zi);
         q(2) =  0.0 - diff*2.0*M_PI*sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi)*cos(2.0*M_PI*zi);
         break;
      }
   }
}


double fFun(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   int dim = x.Size();

   switch (dim)
   {
      case 2:
      {
         return 0.0;//diff*8.0*M_PI*M_PI*sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi);
         break;
      }
      case 3:
      {
         double zi(x(2));
         return diff*12.0*M_PI*M_PI*sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi)*sin(2.0*M_PI*zi);
         break;
      }
   }

   return 0;

}

void pHDGPostProcessing::Postprocessing(ParGridFunction &u_postprocessed)
{
   Mesh *mesh = pfes->GetMesh();
   Array<int>  vdofs;
   Vector      elmat2, shape, RHS, to_RHS, vals, uvals;
   double      RHS2;
   DenseMatrix elmat, invdfdx, dshape, dshapedxt, qvals;

   int  ndofs;
   const FiniteElement *fe_elem;
   ElementTransformation *Trans;

   for (int i = 0; i < pfes->GetNE(); i++)
   {
      pfes->GetElementVDofs(i, vdofs);
      ndofs = vdofs.Size();
      vals.SetSize(ndofs);
      vals = 0.0;
      elmat.SetSize(ndofs);
      elmat2.SetSize(ndofs);
      shape.SetSize(ndofs);

      RHS.SetSize(ndofs);
      to_RHS.SetSize(ndofs);

      elmat = 0.0;
      elmat2 = 0.0;
      RHS = 0.0;
      RHS2 = 0.0;

      fe_elem = pfes->GetFE(i);
      int dim = fe_elem->GetDim();
      int spaceDim = dim;
      Vector qval_col;
      qval_col.SetSize(dim);
      invdfdx.SetSize(dim, spaceDim);
      dshape.SetSize(ndofs, spaceDim);
      dshapedxt.SetSize(ndofs, spaceDim);

      Trans = mesh->GetElementTransformation(i);

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int order = 2*fe_elem->GetOrder() + 2;
         ir = &IntRules.Get(fe_elem->GetGeomType(), order);
      }

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);

         fe_elem->CalcDShape(ip, dshape);
         fe_elem->CalcShape(ip, shape);

         Trans->SetIntPoint(&ip);
         // Compute invdfdx = / adj(J),      if J is square
         //            \ adj(J^t.J).J^t, otherwise
         CalcAdjugate(Trans->Jacobian(), invdfdx);
         double w = Trans->Weight();
         w = ip.weight / w;
         w *= diffcoeff->Eval(*Trans, ip);
         Mult(dshape, invdfdx, dshapedxt);

         AddMult_a_AAt(w, dshapedxt, elmat);

         dshapedxt *= ip.weight ;

         qval_col = 0.0;
         for (int ii = 0; ii<dim; ii++)
         {
            qval_col(ii) = q->GetValue(i, ip, (ii+1));
         }

         dshapedxt.Mult(qval_col, to_RHS);

         RHS -= to_RHS;

         shape *= (Trans->Weight() * ip.weight);
         elmat2 += shape;

         double uvalsj;
         uvalsj = u->GetValue(i, ip, 1);

         double rhs_weight = (Trans->Weight() * ip.weight);
         RHS2  += (uvalsj*rhs_weight);

      }

      // changing the last row and the last entry
      for (int j = 0; j < ndofs; j++)
      {
         elmat(ndofs-1,j) = elmat2(j);
      }
      RHS(ndofs-1) = RHS2;

      elmat.Invert();
      elmat.Mult(RHS, vals);
      u_postprocessed.SetSubVector(vdofs, vals);

   }
}
