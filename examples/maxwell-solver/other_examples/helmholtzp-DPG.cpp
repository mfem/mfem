//                       MFEM Example 8 - Parallel Version
//
// Compile with: make ex8p
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void helmholtz_solution(const Vector & x, double & p, Vector & dp, double & d2p);
double p_exact(const Vector &x);
void gradp_exact(const Vector &x, Vector & gradp);
double f_exact(const Vector &x);

int dim;
double omega;
int sol;
int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool visualization = 1;
   double k = 0.;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&sol, "-sol", "--solution",
                  "1: polynomial, 2: Plane wave (sin)");                  
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");                                          
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   // Angular frequency
   omega = 2.0 * M_PI * k;

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }
   pmesh.ReorientTetMesh();

   // 6. Define the trial, interfacial (trace) and test DPG spaces:
   //    - The trial space, x0_space, contains the non-interfacial unknowns and
   //      has the essential BC.
   //    - The interfacial space, xhat_space, contains the interfacial unknowns
   //      and does not have essential BC.
   //    - The test space, test_space, is an enriched space where the enrichment
   //      degree may depend on the spatial dimension of the domain, the type of
   //      the mesh and the trial space order.
   unsigned int trial_order = order;
   unsigned int trace_order = order - 1;
   unsigned int test_order  = order; /* reduced order, full order is
                                        (order + dim - 1) */
   if (dim == 2 && (order%2 == 0 || (pmesh.MeshGenerator() & 2 && order > 1)))
   {
      test_order++;
   }
   if (test_order < trial_order)
   {
      if (myid == 0)
      {
         cerr << "Warning, test space not enriched enough to handle primal"
              << " trial space\n";
      }
   }


   H1_FECollection x0_fec(trial_order, dim);
   RT_Trace_FECollection xhat_fec(trace_order, dim);
   L2_FECollection test_fec(test_order, dim);

   ParFiniteElementSpace x0_space(&pmesh, &x0_fec);
   ParFiniteElementSpace xhat_space(&pmesh, &xhat_fec);
   ParFiniteElementSpace test_space(&pmesh, &test_fec);

   HYPRE_Int glob_true_s0     =   x0_space.GlobalTrueVSize();
   HYPRE_Int glob_true_s1     = xhat_space.GlobalTrueVSize();
   HYPRE_Int glob_true_s_test = test_space.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "\nNumber of Unknowns:\n"
           << " Trial space,     X0   : " << glob_true_s0
           << " (order " << trial_order << ")\n"
           << " Interface space, Xhat : " << glob_true_s1
           << " (order " << trace_order << ")\n"
           << " Test space,      Y    : " << glob_true_s_test
           << " (order " << test_order << ")\n\n";
   }

   // 7. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.
   Coefficient * rhs = nullptr;
   if (sol)
   {
      rhs = new FunctionCoefficient(f_exact);
   }
   else
   {
      rhs = new ConstantCoefficient(1.0);
   }
   ParLinearForm F(&test_space);
   F.AddDomainIntegrator(new DomainLFIntegrator(*rhs));
   F.Assemble();


   // 8. Set up the mixed bilinear form for the primal trial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the inverse stiffness matrix on the discontinuous test space, Sinv,
   //    and the stiffness matrix on the continuous trial space, S0.
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_dof;
   x0_space.GetEssentialVDofs(ess_bdr, ess_dof);

   ParGridFunction x0(&x0_space);
   x0 = 0.0;
   FunctionCoefficient * x_ex = nullptr;
   if (sol) 
   {
      x_ex = new FunctionCoefficient(p_exact);
      // x0.ProjectBdrCoefficient(*x_ex,ess_bdr);
      x0.ProjectCoefficient(*x_ex);
   }

   ConstantCoefficient one(1.0);
   ConstantCoefficient omeg(omega*omega);
   ConstantCoefficient negomeg(-omega*omega);
   ParMixedBilinearForm B0(&x0_space,&test_space);
   B0.AddDomainIntegrator(new DiffusionIntegrator(one));
   B0.AddDomainIntegrator(new MassIntegrator(omeg));
   B0.Assemble();
   B0.EliminateEssentialBCFromTrialDofs(ess_dof, x0, F);
   B0.Finalize();

   ParMixedBilinearForm Bhat(&xhat_space,&test_space);
   Bhat.AddTraceFaceIntegrator(new TraceJumpIntegrator());
   Bhat.Assemble();
   Bhat.Finalize();


   // Gram matrix inverse (H1 inner product on the element)
   ParBilinearForm Sinv(&test_space);
   SumIntegrator *Sum = new SumIntegrator;
   Sum->AddIntegrator(new DiffusionIntegrator(one));
   Sum->AddIntegrator(new MassIntegrator(one));
   Sinv.AddDomainIntegrator(new InverseIntegrator(Sum));
   Sinv.Assemble();
   Sinv.Finalize();

   
   // Preconditioner
   ParBilinearForm S0(&x0_space);
   S0.AddDomainIntegrator(new DiffusionIntegrator(one));
   S0.AddDomainIntegrator(new MassIntegrator(omeg));
   S0.Assemble();
   S0.EliminateEssentialBC(ess_bdr);
   S0.Finalize();

   HypreParMatrix & matB0   = *B0.ParallelAssemble();    
   HypreParMatrix & matBhat = *Bhat.ParallelAssemble();  
   HypreParMatrix & matSinv = *Sinv.ParallelAssemble(); 
   HypreParMatrix & matS0   = *S0.ParallelAssemble();   

   // 9. Define the block structure of the problem, by creating the offset
   //    variables. Also allocate two BlockVector objects to store the solution
   //    and rhs.
   enum {x0_var, xhat_var, NVAR};

   int true_s0     = x0_space.TrueVSize();
   int true_s1     = xhat_space.TrueVSize();
   int true_s_test = test_space.TrueVSize();

   Array<int> true_offsets(NVAR+1);
   true_offsets[0] = 0;
   true_offsets[1] = true_s0;
   true_offsets[2] = true_s0+true_s1;

   Array<int> true_offsets_test(2);
   true_offsets_test[0] = 0;
   true_offsets_test[1] = true_s_test;

   BlockVector x(true_offsets), b(true_offsets);
   x = 0.0;
   b = 0.0;

   // 10. Set up the 1x2 block Least Squares DPG operator, B = [B0 Bhat],
   //     the normal equation operator, A = B^t Sinv B, and
   //     the normal equation right-hand-size, b = B^t Sinv F.
   BlockOperator B(true_offsets_test, true_offsets);
   B.SetBlock(0, 0, &matB0);
   B.SetBlock(0, 1, &matBhat);

   RAPOperator A(B, matSinv, B);

   HypreParVector &trueF = *F.ParallelAssemble();
   {
      HypreParVector SinvF(&test_space);
      matSinv.Mult(trueF, SinvF);
      B.MultTranspose(SinvF, b);
   }

   // 11. Set up a block-diagonal preconditioner for the 2x2 normal equation
   //
   //        [ S0^{-1}     0     ]
   //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
   //
   //     corresponding to the primal (x0) and interfacial (xhat) unknowns.
   //     Since the Shat operator is equivalent to an H(div) matrix reduced to
   //     the interfacial skeleton, we approximate its inverse with one V-cycle
   //     of the ADS preconditioner from the hypre library (in 2D we use AMS for
   //     the rotated H(curl) problem).
   HypreBoomerAMG *S0inv = new HypreBoomerAMG(matS0);
   S0inv->SetPrintLevel(0);

   HypreParMatrix &Shat = *RAP(&matSinv, &matBhat);
   // Vector diag;
   // Shat.GetDiag(diag);
   // Shat.ScaleRows(diag);
   
   // Shat *= omega*omega;
   HypreSolver *Shatinv;
   if (dim == 2) { Shatinv = new HypreAMS(Shat, &xhat_space); }
   else          { Shatinv = new HypreADS(Shat, &xhat_space); }



   BlockDiagonalPreconditioner P(true_offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);


   // 12. Solve the normal equation system using the PCG iterative solver.
   //     Check the weighted norm of residual for the DPG least square problem.
   //     Wrap the primal variable in a GridFunction for visualization purposes.
   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetOperator(A);
   pcg.SetPreconditioner(P);
   pcg.SetRelTol(1e-12);
   pcg.SetMaxIter(200);
   pcg.SetPrintLevel(1);
   pcg.Mult(b, x);

   {
      HypreParVector LSres(&test_space), tmp(&test_space);
      B.Mult(x, LSres);
      LSres -= trueF;
      matSinv.Mult(LSres, tmp);
      double res = sqrt(InnerProduct(LSres, tmp));
      if (myid == 0)
      {
         cout << "\n|| B0*x0 + Bhat*xhat - F ||_{S^-1} = " << res << endl;
      }
   }

   x0.Distribute(x.GetBlock(x0_var));
   x0.ProjectBdrCoefficient(*x_ex,ess_bdr);

   if (sol)
   {
      VectorFunctionCoefficient gradcoeff(dim,gradp_exact);
      double H1error = x0.ComputeH1Error(x_ex,&gradcoeff);
      if (myid == 0)
      {
         cout << "\n|| x0 - x_exact||_H^1 = " << H1error << endl;
      }
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x0 << flush;

      MPI_Barrier(MPI_COMM_WORLD);

      if (sol)
      {
         ParGridFunction u_ex(&x0_space);
         u_ex.ProjectCoefficient(*x_ex);
         socketstream exact_sock(vishost, visport);
         exact_sock << "parallel " << num_procs << " " << myid << "\n";
         exact_sock.precision(8);
         exact_sock << "solution\n" << pmesh << u_ex << flush;
      }

   }

   // 15. Free the used memory.
   delete rhs;
   delete Shatinv;
   delete S0inv;

   MPI_Finalize();

   return 0;
}


void helmholtz_solution(const Vector & x, double & p, Vector & dp, double & d2p)
{
   if (sol == 1) // polynomial
   {
      if (dim == 2)
      {
         p = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
         dp[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]);
         dp[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]);
         d2p = -2.0 * x[1]*(1.0 - x[1]) 
               -2.0 * x[0]*(1.0 - x[0]);
      }
      else
      {
         p = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
         dp[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
         dp[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]) * x[2]*(1.0 - x[2]);
         dp[2] = (1.0 - 2.0 *x[2]) * x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
         d2p = -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[1]) * x[1]
               -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[2]) * x[2]
               -2.0*(-1.0 + x[1]) * x[1] * (-1.0 + x[2]) * x[2];
      }
   }
   else if (sol==2)
   {
      double alpha;
      if (dim == 2)
      {
         alpha = omega/sqrt(2);
         p = cos(alpha * ( x(0) + x(1) ) );
         dp[0] = -alpha * sin(alpha * ( x(0) + x(1) ) );
         dp[1] = dp[0];
         d2p = -2.0 * alpha * alpha * p;
      }
      else
      {
         alpha = omega/sqrt(3);
         p = cos(alpha * ( x(0) + x(1) + x(2) ) );
         dp[0] = -alpha * sin(alpha * ( x(0) + x(1) + x(2) ) );
         dp[1] = dp[0];
         dp[2] = dp[0];
         d2p = -3.0 * alpha * alpha * p;
      }
   }
   else
   {
      MFEM_ABORT("Unknown exact solution choice");
   }

}

double p_exact(const Vector &x)
{
   double p, d2p;
   Vector dp(dim);
   helmholtz_solution(x, p, dp, d2p);
   return p;
}

void gradp_exact(const Vector &x, Vector & gradp)
{
   double p, d2p;
   helmholtz_solution(x, p, gradp, d2p);
}
double f_exact(const Vector &x)
{
   double p, d2p;
   Vector dp(dim);
   helmholtz_solution(x, p, dp, d2p);
   return -d2p + omega * omega * p;
}