//                       MFEM Example 40 - Parallel Version
//
// Compile with: make ex40p
//
// Sample runs: mpirun -np 4 ex40p -step 10.0 -gr 2.0
//              mpirun -np 4 ex40p -step 10.0 -gr 2.0 -o 3 -r 1
//              mpirun -np 4 ex40p -step 10.0 -gr 2.0 -r 4 -m ../data/l-shape.mesh
//              mpirun -np 4 ex40p -step 10.0 -gr 2.0 -r 2 -m ../data/fichera.mesh
//
// Description: This example code demonstrates how to use MFEM to solve the
//              eikonal equation,
//
//                      |∇𝑢| = 1 in Ω,  𝑢 = 0 on ∂Ω.
//
//              The viscosity solution of this problem coincides with the unique optimum
//              of the nonlinear program
//
//                   maximize ∫_Ω 𝑢 d𝑥 subject to |∇𝑢| ≤ 1 in Ω, 𝑢 = 0 on ∂Ω,    (⋆)
//
//              which is the foundation for method implemented below.
//
//              Following the proximal Galerkin methodology [1,2] (see also Example
//              36), we construct a Legendre function for the closed unit ball
//              𝐵₁ := {𝑥 ∈ Rⁿ | |𝑥| ≤ 1}. Our choice is the Hellinger entropy,
//
//                    R(𝑥) = −( 1 − |𝑥|² )^{1/2},
//
//              although other choices are possible, each leading to a slightly
//              different algorithm. We then adaptively regularize the optimization
//              problem (⋆) with the Bregman divergence of the Hellinger entropy,
//
//                 maximize  ∫_Ω 𝑢 d𝑥 - αₖ⁻¹ D(∇𝑢,∇𝑢ₖ₋₁)  subject to  𝑢 = 0 on Ω.
//
//              This results in a sequence of functions ( 𝜓ₖ , 𝑢ₖ ),
//
//                      𝑢ₖ → 𝑢,    𝜓ₖ/|𝜓ₖ| → ∇𝑢    as k → ∞,
//
//              defined by the nonlinear saddle-point problems
//
//               Find 𝜓ₖ ∈ H(div,Ω) and 𝑢ₖ ∈ L²(Ω) such that
//               ( (∇R)⁻¹(𝜓ₖ) , τ ) + ( 𝑢ₖ , ∇⋅τ ) = 0                     ∀ τ ∈ H(div,Ω)
//               ( ∇⋅𝜓ₖ , v )                     = ( ∇⋅𝜓ₖ₋₁ - αₖ , v )    ∀ v ∈ L²(Ω)
//
//              where (∇R)⁻¹(𝜓) = 𝜓 / ( 1 + |𝜓|² )^{1/2} and αₖ = α₀rᵏ, where r ≥ 1
//              is a prescribed growth rate. (r = 1 is the most stable.) The
//              saddle-point problems are solved using a damped quasi-Newton method
//              with a tunable regularization parameter 0 ≤ ϵ << 1. The solver is
//              also made more robust by additional safeguards described in the code.
//
//              [1] Keith, B. and Surowiec, T. (2024) Proximal Galerkin: A structure-
//                  preserving finite element method for pointwise bound constraints.
//                  Foundations of Computational Mathematics, 1–97.
//              [2] Dokken, J., Farrell, P., Keith, B., Papadopoulos, I., and
//                  Surowiec, T. (2025) The latent variable proximal point algorithm
//                  for variational problems with inequality constraints. (To appear.)

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class IsomorphismCoefficient : public VectorCoefficient
{
protected:
   ParGridFunction *psi;

public:
   IsomorphismCoefficient(int vdim, ParGridFunction &psi_)
      : VectorCoefficient(vdim), psi(&psi_) { }

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

class DIsomorphismCoefficient : public MatrixCoefficient
{
protected:
   ParGridFunction *psi;
   real_t eps;

public:
   DIsomorphismCoefficient(int height, ParGridFunction &psi_, real_t eps_ = 0.0)
      : MatrixCoefficient(height),  psi(&psi_), eps(eps_) { }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int max_it = 5;
   int ref_levels = 3;
   real_t alpha = 1.0;
   real_t growth_rate = 1.0;
   real_t newton_scaling = 0.8;
   real_t eps = 1e-6;
   real_t tol = 1e-4;
   real_t max_alpha = 1e2;
   real_t max_psi = 1e2;
   real_t eps2 = 1e-1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
   args.AddOption(&alpha, "-step", "--step",
                  "Initial size alpha");
   args.AddOption(&growth_rate, "-gr", "--growth-rate",
                  "Growth rate of the step size alpha");
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   MFEM_ASSERT(mesh.bdr_attributes.Size(),
               "This example does not support meshes"
               " without boundary attributes."
              )

   // 3. Postprocess the mesh.
   // 3A. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 3B. Interpolate the geometry after refinement to control geometry error.
   // NOTE: Minimum second-order interpolation is used to improve the accuracy.
   int curvature_order = max(order,2);
   mesh.SetCurvature(curvature_order);

   // 3C. Compute the maximum mesh size.
   real_t hmax = 0.0;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      hmax = max(mesh.GetElementSize(i, 1), hmax);
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 4. Define the necessary finite element spaces on the mesh.
   RT_FECollection RTfec(order, dim);
   ParFiniteElementSpace RTfes(&pmesh, &RTfec);

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2fes(&pmesh, &L2fec);

   int num_dofs_RT = RTfes.GlobalTrueVSize();
   int num_dofs_L2 = L2fes.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of H(div) dofs: "
           << num_dofs_RT << endl;
      cout << "Number of L² dofs: "
           << num_dofs_L2 << endl;
   }

   // 5. Define the offsets for the block matrices
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = RTfes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   Array<int> toffsets(3);
   toffsets[0] = 0;
   toffsets[1] = RTfes.GetTrueVSize();
   toffsets[2] = L2fes.GetTrueVSize();
   toffsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   BlockVector tx(toffsets), trhs(toffsets);
   tx = 0.0; trhs = 0.0;

   // 6. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   ParGridFunction u_gf, delta_psi_gf;
   delta_psi_gf.MakeRef(&RTfes,x,offsets[0]);
   u_gf.MakeRef(&L2fes,x,offsets[1]);

   ParGridFunction psi_old_gf(&RTfes);
   ParGridFunction psi_gf(&RTfes);
   ParGridFunction u_old_gf(&L2fes);

   // 7. Define initial guesses for the solution variables.
   delta_psi_gf = 0.0;
   psi_gf = 0.0;
   u_gf = 0.0;
   psi_old_gf = psi_gf;
   u_old_gf = u_gf;

   // 8. Prepare for glvis output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
   }

   // 9. Coefficients to be used later.
   ConstantCoefficient one_cf(1.0);
   ConstantCoefficient zero_cf(0.0);
   Vector zero_vec(sdim); zero_vec = 0.0;
   VectorConstantCoefficient zero_vec_cf(zero_vec);
   ConstantCoefficient neg_alpha_cf((real_t) -1.0*alpha);
   IsomorphismCoefficient Z(sdim, psi_gf);
   DIsomorphismCoefficient DZ(sdim, psi_gf, eps);
   ScalarVectorProductCoefficient neg_Z(-1.0, Z);
   DivergenceGridFunctionCoefficient div_psi_cf(&psi_gf);
   DivergenceGridFunctionCoefficient div_psi_old_cf(&psi_old_gf);
   SumCoefficient psi_old_minus_psi(div_psi_old_cf, div_psi_cf, 1.0, -1.0);

   // 10. Assemble constant matrices/vectors to avoid reassembly in the loop.
   ParLinearForm b0, b1;
   b0.MakeRef(&RTfes,rhs.GetBlock(0),0);
   b1.MakeRef(&L2fes,rhs.GetBlock(1),0);

   b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(neg_Z));
   b1.AddDomainIntegrator(new DomainLFIntegrator(neg_alpha_cf));
   b1.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));

   ParBilinearForm a00(&RTfes);
   a00.AddDomainIntegrator(new VectorFEMassIntegrator(DZ));

   ParMixedBilinearForm a10(&RTfes,&L2fes);
   a10.AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   a10.Assemble();
   a10.Finalize();
   HypreParMatrix *A10 = a10.ParallelAssemble();

   HypreParMatrix *A01 = A10->Transpose();

   ParLinearForm vol_form(&L2fes);
   ParGridFunction one_gf(&L2fes);
   one_gf = 1.0;
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
   vol_form.Assemble();
   real_t domain_volume = vol_form(one_gf);

   // 11. Iterate.
   int k;
   int total_iterations = 0;
   real_t increment_u = 0.1;
   ParGridFunction u_tmp(&L2fes);
   for (k = 0; k < max_it; k++)
   {
      u_tmp = u_old_gf;

      if (myid == 0)
      {
         mfem::out << "\nOUTER ITERATION " << k+1 << endl;
      }

      int j;
      for ( j = 0; j < 5; j++)
      {
         total_iterations++;

         b0.Assemble();
         b0.ParallelAssemble(trhs.GetBlock(0));

         b1.Assemble();
         b1.ParallelAssemble(trhs.GetBlock(1));

         a00.Assemble(false);
         a00.Finalize(false);
         HypreParMatrix *A00 = a00.ParallelAssemble();

         // Construct Schur-complement preconditioner
         HypreParVector A00_diag(MPI_COMM_WORLD, A00->GetGlobalNumRows(),
                                 A00->GetRowStarts());
         A00->GetDiag(A00_diag);
         HypreParMatrix S_tmp(*A01);
         S_tmp.InvScaleRows(A00_diag);
         HypreParMatrix *S = ParMult(A10, &S_tmp, true);

         BlockDiagonalPreconditioner prec(toffsets);
         HypreBoomerAMG P00(*A00);
         P00.SetPrintLevel(0);
         HypreBoomerAMG P11(*S);
         P11.SetPrintLevel(0);
         prec.SetDiagonalBlock(0,&P00);
         prec.SetDiagonalBlock(1,&P11);

         BlockOperator A(toffsets);
         A.SetBlock(0,0,A00);
         A.SetBlock(1,0,A10);
         A.SetBlock(0,1,A01);

         MINRESSolver minres(MPI_COMM_WORLD);
         minres.SetPrintLevel(-1);
         minres.SetRelTol(1e-12);
         minres.SetMaxIter(10000);
         minres.SetOperator(A);
         minres.SetPreconditioner(prec);
         minres.Mult(trhs,tx);
         delete S;
         delete A00;

         delta_psi_gf.SetFromTrueDofs(tx.GetBlock(0));
         u_gf.SetFromTrueDofs(tx.GetBlock(1));

         u_tmp -= u_gf;
         real_t Newton_update_size = u_tmp.ComputeL2Error(zero_cf);
         u_tmp = u_gf;

         // Damped Newton update
         psi_gf.Add(newton_scaling, delta_psi_gf);
         a00.Update();

         if (visualization)
         {
            sol_sock << "parallel " << num_procs << " " << myid << "\n";
            sol_sock << "solution\n" << pmesh << u_gf << "window_title 'Discrete solution'"
                     << flush;
         }

         if (myid == 0)
         {
            mfem::out << "Newton_update_size = " << Newton_update_size << endl;
         }

         if (newton_scaling*Newton_update_size < increment_u)
         {
            break;
         }
      }

      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero_cf);

      if (myid == 0)
      {
         mfem::out << "Number of Newton iterations = " << j+1 << endl;
         mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_u << endl;
      }

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;
      alpha *= max(growth_rate, 1_r);

      // Safeguard 1: Stop alpha from growing too large
      alpha = min(alpha, max_alpha);

      // Safeguard 2: Stop |ψ| from growing too large
      real_t norm_psi = psi_old_gf.ComputeL1Error(zero_vec_cf)/domain_volume;
      if (norm_psi > max_psi)
      {
         // Additional entropy regularization
         neg_alpha_cf.constant = -alpha/(1.0 + eps2 * alpha * hmax);
         psi_old_minus_psi.SetAlpha(1.0/(1.0 + eps2 * alpha * hmax));
      }
      else
      {
         neg_alpha_cf.constant = -alpha;
      }

      if (increment_u < tol || k == max_it-1)
      {
         break;
      }

   }

   // 12. Print stats.
   if (myid == 0)
   {
      mfem::out << "\n Outer iterations: " << k+1
                << "\n Total iterations: " << total_iterations
                << "\n Total dofs:       " << RTfes.GetTrueVSize() + L2fes.GetTrueVSize()
                << endl;
   }

   // 13. Free the used memory.
   delete A01;
   delete A10;
   return 0;
}

void IsomorphismCoefficient::Eval(Vector &V, ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");

   Vector psi_vals(vdim);
   psi->GetVectorValue(T, ip, psi_vals);
   real_t norm = psi_vals.Norml2();
   real_t phi = 1.0 / sqrt(1.0 + norm*norm);

   V = psi_vals;
   V *= phi;
}

void DIsomorphismCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");
   MFEM_ASSERT(eps >= 0, "eps is negative");

   Vector psi_vals(height);
   psi->GetVectorValue(T, ip, psi_vals);
   real_t norm = psi_vals.Norml2();
   real_t phi = 1.0 / sqrt(1.0 + norm*norm);

   K = 0.0;
   for (int i = 0; i < height; i++)
   {
      K(i,i) = phi + eps;
      for (int j = 0; j < height; j++)
      {
         K(i,j) -= psi_vals(i) * psi_vals(j) * pow(phi, 3);
      }
   }
}
