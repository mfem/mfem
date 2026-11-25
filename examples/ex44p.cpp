//                        MFEM Example 44 - parallel version
//
// Compile with: make ex44p
//
// Sample runs: mpirun -np 4 ex44p
//
// Description: This example code demonstrates the use of MFEM to
//              solve the bound-constrained energy minimization problem
//
//              minimize 1/2 [λ (∇·u, ∇·u) + 2μ (ε(u), ε(u))] - (f,u)
//              subject to          u·ñ ≤ ϕ on Γ,
//
//              where ε(u) = (∇u + ∇uᵀ)/2 is the symmetric gradient of
//              the displacement u, λ and μ are the Lamé parameters,
//              f is a body force, ϕ is a gap function, and ñ is a
//              fixed approximation of the contact normal.
//
//              This corresponds to a unilateral Signorini-type contact
//              problem, where the displacement u on the contact boundary
//              Γ is constrained to not penetrate a rigid obstacle.
//
//              The problem is discretized and solved using the proximal
//              Galerkin finite element method (see Example 2 of [1]).
//
//              This example highlights the use of MFEM's SubMesh and
//              MixedBilinearForm features to construct a coupled
//              nonlinear system involving variables defined separately
//              over the domain and the boundary submesh.
//
// [1] Dokken, J. S., Farrell, P. E., Keith, B., Papadopoulos, I. P. A. and
//     Surowiec, T. M. (2025) The latent variable proximal point algorithm
//     for variational problems with inequality constraints. Computer
//     Methods in Applied Mechanics and Engineering, 445, 118181.

#include "mfem.hpp"
#include "ex44.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   int order = 1;
   int ref_levels = 0;
   real_t alpha = 1;
   real_t tol = 1e-6;
   bool visualization = false;

   int proximal_iterations = 20;
   int newton_iterations = 10;
   real_t min_newt_tol = 1e-8;
   real_t increment_u = -mfem::infinity();

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of uniform mesh refinements.");
   args.AddOption(&proximal_iterations, "-i", "--iterations",
                  "Maximum number of proximal iterations.");
   args.AddOption(&newton_iterations, "-ni", "--newton_iters",
                  "Maximum number of newton solver iterations.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha parameter for boundary condition.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Iteration tolerance.");
   args.AddOption(&min_newt_tol, "-mnt", "--min-newton-tol",
                  "Minimum Newton update tolerance.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (order < 1)
   {
      if (myid == 0) { mfem::err << "Require order >= 1" << endl; }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file.
   const char *mesh_file = "../data/wheel.msh";
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 4. Interpolate the geometry after refinement to control geometry error.
   int curvature_order = max(order+1, 2);
   mesh.SetCurvature(curvature_order);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 5a. Mark the outer surface of the wheel as the contact boundary.
   Array<int> contact_attr(1);
   contact_attr[0] = 1;

   // 5b. Extract the submesh corresponding to the contact boundary.
   ParSubMesh contact_mesh(ParSubMesh::CreateFromBoundary(pmesh, contact_attr));
   const int sub_dim = contact_mesh.Dimension();

   // 6. Define finite element spaces for the solutions.
   H1_FECollection H1fec(order+1, dim);
   ParFiniteElementSpace H1fes(&pmesh, &H1fec, dim, Ordering::byVDIM);

   L2_FECollection L2fec(order-1, sub_dim);
   ParFiniteElementSpace L2fes(&contact_mesh, &L2fec, 1);

   HYPRE_BigInt num_dofs_H1 = H1fes.GlobalTrueVSize();
   HYPRE_BigInt num_dofs_L2 = L2fes.GlobalTrueVSize();

   if (myid == 0)
   {
      mfem::out << "----------------------------------------\n"
                << "Number of H1 finite element unknowns: " << num_dofs_H1 << "\n"
                << "Number of L2 finite element unknowns: " << num_dofs_L2 << "\n"
                << "----------------------------------------\n";
   }

   Array<int> toffsets(3);
   toffsets[0] = 0;
   toffsets[1] = H1fes.GetTrueVSize();
   toffsets[2] = L2fes.GetTrueVSize();
   toffsets.PartialSum();

   // 7. Determine the list of essential boundary DOFs.
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0; ess_bdr[3] = 1;

   Array<int> ess_tdof_list;
   H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Prescribe the displacement on the inner surface of the wheel.
   Vector g(dim);
   g = 0.0; g[dim - 1] = -0.1;
   VectorConstantCoefficient g_coeff(g);

   ParGridFunction disp_lift(&H1fes);
   disp_lift = 0.0;
   disp_lift.ProjectBdrCoefficient(g_coeff, ess_bdr);
   Vector disp_lift_tv;
   disp_lift.GetTrueDofs(disp_lift_tv);

   // 8. Allocate grid functions and true-DOF block vectors.
   ParGridFunction u_gf(&H1fes);          // displacement
   ParGridFunction u_old_gf(&H1fes);
   ParGridFunction psi_gf(&L2fes);        // latent variable
   ParGridFunction psi_old_gf(&L2fes);
   ParGridFunction kappa_gf(&L2fes);      // dual variable

   u_old_gf = 0.0;
   psi_old_gf = 0.0;
   kappa_gf = 0.0;

   BlockVector tx(toffsets), residual(toffsets), delta(toffsets);
   tx = 0.0;

   // 9. Set up variables and coefficients.
   const real_t lambda = 1.0, mu = 1.0;   // Lamé constants

   Vector f(dim);
   f = 0.0; f(dim-1) = -0.1;              // body force

   Vector n_tilde(dim);
   n_tilde = 0.0; n_tilde(dim-1) = -1.0;  // contact normal

   ConstantCoefficient one(1.0);
   Vector zero_vec(dim); zero_vec = 0.0;
   VectorConstantCoefficient zero(zero_vec);

   // Force, contact normal, and gap coefficients.
   VectorConstantCoefficient f_coeff(f);
   VectorConstantCoefficient n_tilde_c(n_tilde);
   FunctionCoefficient gap_cf(GapFunction);

   // 10. Initialize displacement, latent variable, and the Newton state.
   VectorFunctionCoefficient init_u_c(dim, InitialGuess);
   u_gf.ProjectCoefficient(init_u_c);
   u_gf.ProjectBdrCoefficient(g_coeff, ess_bdr);
   u_old_gf = u_gf;

   // Create a temporary trace grid function to project the initial ψ
   // using ψ = -ln(ϕ - u·ñ).
   {
      H1_FECollection trace_fec(order+1, sub_dim);
      ParFiniteElementSpace trace_fes(&contact_mesh, &trace_fec, dim,
                                      H1fes.GetOrdering());

      ParGridFunction u_trace_gf(&trace_fes);
      contact_mesh.Transfer(u_gf, u_trace_gf);

      VectorGridFunctionCoefficient u_trace_cf(&u_trace_gf);

      InnerProductCoefficient udotn(u_trace_cf, n_tilde_c);
      TransformedCoefficient psi_init_cf(&gap_cf, &udotn, LogarithmicFunction);
      psi_gf.ProjectCoefficient(psi_init_cf);
      psi_old_gf = psi_gf;
   }

   u_gf.GetTrueDofs(tx.GetBlock(0));
   kappa_gf.GetTrueDofs(tx.GetBlock(1));

   // 11. Each Newton step solves the linearized saddle point system:
   //             | K    -B^T | | δu |   | R_u |
   //             | B     M   | | δκ | = | R_κ |.
   //     K is the elasticity stiffness matrix, B is the normal-trace coupling,
   //     and M is the boundary mass matrix weighted by α exp(-ψ).

   // 11a. Define contact quadrature and assemble linear forms.
   const IntegrationRule &ir = IntRules.Get(
                                  contact_mesh.GetTypicalElementGeometry(), 2*(order + 1));

   // Assemble (f,v).
   ParLinearForm b_force(&H1fes);
   b_force.AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));
   b_force.Assemble();
   Vector force(H1fes.GetTrueVSize());
   b_force.ParallelAssemble(force);

   // Assemble (exp(-ψ) - ϕ, q)_Γ, reassemble whenever ψ changes.
   GridFunctionCoefficient psi_cf(&psi_gf);
   TransformedCoefficient exp_neg_psi(&psi_cf, ExponentialFunction);
   SumCoefficient exp_minus_gap(exp_neg_psi, gap_cf, 1.0, -1.0);
   ParLinearForm b1(&L2fes);
   b1.AddDomainIntegrator(new DomainLFIntegrator(exp_minus_gap, &ir));

   // 11b. Assemble the fixed bilinear forms and set up the block Jacobian.
   // Assemble K, the matrix of the elasticity bilinear form a(u,v).
   ParBilinearForm a00(&H1fes);
   a00.AddDomainIntegrator(new ElasticityIntegrator(one, lambda, mu));
   a00.Assemble();
   a00.Finalize();
   HypreParMatrix *A00 = a00.ParallelAssemble();
   Vector disp_lift_elast(H1fes.GetTrueVSize());
   A00->Mult(disp_lift_tv, disp_lift_elast);
   A00->EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);

   // Assemble B from (u·ñ, q)_Γ, coupling the parent and boundary spaces.
   ParentToSubMixedBilinearForm a10(&H1fes, &L2fes);
   a10.AddBoundaryDomainIntegrator(new MixedFormIntegrator(n_tilde_c, dim));
   a10.Assemble();
   a10.Finalize();
   HypreParMatrix *A10 = a10.ParallelAssemble();
   Vector disp_lift_contact(L2fes.GetTrueVSize());
   A10->Mult(disp_lift_tv, disp_lift_contact);
   delete A10->EliminateCols(ess_tdof_list);

   // Store B^T.
   HypreParMatrix *A01 = A10->Transpose();

   // Newton Jacobian: J = [ K  -B^T; B  M ].
   // M = α A11, where A11 is the mass matrix weighted by exp(-ψ).
   // A11 is assembled inside the Newton loop.
   BlockOperator A(toffsets);
   A.SetBlock(0,0,A00);
   A.SetBlock(1,0,A10);
   A.SetBlock(0,1,A01,-1.0);

   // 12. Construct the block preconditioner P = [ K  0; B  S ].
   // Approximate the Schur complement by S = M + B D^(-1) B^T, D = diag(K).

   // Build the constant blocks.
   HypreParVector A00_diag(MPI_COMM_WORLD, A00->GetGlobalNumRows(),
                           A00->GetRowStarts());
   A00->GetDiag(A00_diag);

   HypreParMatrix *BDinvBt;
   {
      HypreParMatrix DinvBt(*A01);
      DinvBt.InvScaleRows(A00_diag);
      BDinvBt = ParMult(A10, &DinvBt);
   }

   // Construct the fixed AMG approximation P00 ≈ K^(-1).
   HypreBoomerAMG P00(*A00);
   P00.SetElasticityOptions(&H1fes);
   P00.SetPrintLevel(0);

   // Approximate the action of P^(-1) using P00, P11, and B.
   BlockLowerTriangularPreconditioner prec(toffsets);
   prec.SetDiagonalBlock(0,&P00);
   prec.SetBlock(1,0,A10);

   // 13. Configure the linear solver.
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetRelTol(1e-8);
   gmres.SetKDim(200);
   gmres.SetPrintLevel(0);
   gmres.SetMaxIter(2000);
   gmres.SetPreconditioner(prec);
   gmres.SetOperator(A);

   // 14. Set up GLVis visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream displacement_sock, pressure_sock;

   ParGridFunction pressure_gf(&L2fes);   // contact pressure p = -κ
   pressure_gf = kappa_gf;
   pressure_gf *= -1.0;

   // Visualize the initial solution.
   if (visualization)
   {
      displacement_sock.open(vishost, visport);
      displacement_sock.precision(8);

      displacement_sock << "parallel " << num_procs << " " << myid << "\n";
      displacement_sock << "solution\n" << pmesh << u_gf
                        << "window_title 'Displacement u'\n"
                        << "window_geometry 0 0 600 500\n"
                        << "keys aacb\n" << flush;

      MPI_Barrier(MPI_COMM_WORLD);

      pressure_sock.open(vishost, visport);
      pressure_sock.precision(8);

      pressure_sock << "parallel " << num_procs << " " << myid << "\n";
      pressure_sock << "solution\n" << contact_mesh << pressure_gf
                    << "window_title 'Contact pressure'\n"
                    << "window_geometry 620 0 600 500\n"
                    << "keys aac\n" << flush;
   }

   // 15. Perform proximal iterations, with an inner Newton solve.
   for (int k = 0; k < proximal_iterations; k++)
   {
      // Store the previous displacement to measure the update.
      ParGridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      if (myid == 0)
      {
         mfem::out << "\nOuter iteration: " << k+1
                   << "\n----------------------------------------" << endl;
      }

      for (int j = 0; j < newton_iterations; j++)
      {
         // Keep ψ_old fixed and recover ψ = ψ_old - ακ.
         u_gf.SetFromTrueDofs(tx.GetBlock(0));
         kappa_gf.SetFromTrueDofs(tx.GetBlock(1));
         add(psi_old_gf, -alpha, kappa_gf, psi_gf);

         // Assemble the nonlinear residual at the current (u, κ).
         // R_u = K u - B^T κ - f.
         A00->Mult(tx.GetBlock(0), residual.GetBlock(0));
         residual.GetBlock(0) += disp_lift_elast;
         A01->AddMult(tx.GetBlock(1), residual.GetBlock(0), -1.0);
         residual.GetBlock(0) -= force;
         for (int i = 0; i < ess_tdof_list.Size(); i++)
         {
            const int tdof = ess_tdof_list[i];
            residual.GetBlock(0)[tdof] =
               tx.GetBlock(0)[tdof] - disp_lift_tv[tdof];
         }

         // Reassemble the nonlinear boundary linearform.
         // R_κ = B u + assemble_Γ(exp(-ψ) - ϕ).
         b1.Assemble();
         b1.ParallelAssemble(residual.GetBlock(1));
         residual.GetBlock(1) += disp_lift_contact;
         A10->AddMult(tx.GetBlock(0), residual.GetBlock(1));

         // Reassemble the ψ-dependent bilinear form and Jacobian block.
         // M = α A11, with A11 the mass matrix weighted by exp(-ψ).
         ParBilinearForm a11(&L2fes);
         a11.AddDomainIntegrator(new MassIntegrator(exp_neg_psi, &ir));
         a11.Assemble();
         a11.Finalize();
         HypreParMatrix *A11 = a11.ParallelAssemble();
         A.SetBlock(1,1,A11,alpha);

         // Construct S = M + B D^(-1) B^T for the current Newton iterate.
         HypreParMatrix *S = Add(alpha, *A11, 1.0, *BDinvBt);

         {
            // Construct the AMG approximation P11 ≈ S^(-1).
            HypreBoomerAMG P11(*S);
            P11.SetPrintLevel(0);

            prec.SetDiagonalBlock(1,&P11);

            // Solve J (δu, δκ) = R, then subtract the correction.
            delta = 0.0;
            gmres.Mult(residual, delta);
            MFEM_VERIFY(gmres.GetConverged(), "Linear solver failed to converge");
            int num_iter = gmres.GetNumIterations();
            if (myid == 0)
            {
               mfem::out << "\tLinear solver iterations: " << num_iter << endl;
            }
            tx -= delta;
         }

         delete S;
         delete A11;

         u_gf.SetFromTrueDofs(tx.GetBlock(0));
         kappa_gf.SetFromTrueDofs(tx.GetBlock(1));
         add(psi_old_gf, -alpha, kappa_gf, psi_gf);

         // Check Newton convergence using the PG increment and tolerance floor.
         u_tmp -= u_gf;
         real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         if (myid == 0)
         {
            mfem::out << "\tNewton iteration: " << j+1
                      << ", Newton_update_size = " << Newton_update_size << endl;
         }

         if (Newton_update_size < std::max(increment_u, min_newt_tol))
         {
            break;
         }
      }

      // Update visualization.
      if (visualization)
      {
         displacement_sock << "parallel " << num_procs << " " << myid << "\n";
         displacement_sock << "solution\n" << pmesh << u_gf << flush;

         pressure_gf = kappa_gf; pressure_gf *= -1.0;
         pressure_sock << "parallel " << num_procs << " " << myid << "\n";
         pressure_sock << "solution\n" << contact_mesh << pressure_gf << flush;
      }

      // Check outer convergence and advance the proximal iterate.
      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      if (myid == 0)
      {
         mfem::out << "Increment (|| uₕ - uₕ_prvs ||) = " << increment_u << endl;
      }

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;
      alpha *= 2.0;

      if (increment_u < tol || k == proximal_iterations-1)
      {
         break;
      }

   }

   // 16. Release the pointers.
   delete BDinvBt;
   delete A00;
   delete A01;
   delete A10;

   return 0;
}
