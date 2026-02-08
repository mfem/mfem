//                               MFEM Signorini
//
// Compile with: make regularp
//
// Sample runs:  mpirun -np 4 regularp
//
// Description:  This program solves Signorini's problem using MFEM. We aim to
//               solve the bound-constrained minimization problem
//
//                   minimize 1/2 ∫_Ω σ(u) : ε(u) dx − ∫_Ω f ⋅ u dx
//                                subject to u ∈ K,
//
//               where K = { v ∈ H¹(Ω,R³) : v = g on Γ_D, v ⋅ ñ ≤ ϕ on Γ_T },
//               where Γ_D ∪ Γ_T = Γ = ∂Ω, ϕ is a gap function defined on Γ_T,
//               and f ∈ L²(Ω,R³) is a given forcing term. We employ Bregman
//               proximal point with the Legendre function R_N : R → R defined
//               as
//
//               R_N(a) = R(a)                                                  for 1/N ≤ a ≤ N,
//                        R(N) + R'(N)(a - N) + R''(N)/2 (a - N)^2              for a > N,
//                        R(1/N) + R'(1/N)(a - 1/N) + R''(1/N)/2 (a - 1/N)^2    for a < 1/N,
//
//               where R(a) = a ln(a) - a and N is chosen sufficiently large.
//               With this choice,
//
//                     R_N'(a) = ln(a)                   for 1/N ≤ a ≤ N,
//                               ln(N) + 1/N (a - N)     for a > N,
//                               ln(1/N) + N (a - 1/N)   for a < 1/N,
//
//               and the Bregman proximal point iteration reads: Given uᵏ⁻¹ ∈ K
//               and an unsummable sequence {αₖ}ₖ > 0, find uᵏ ∈ H¹(Ω,R³) such
//               that
//
//                αₖ(σ(uᵏ),ε(v)) - (R_N'(ϕ − uᵏ ⋅ ñ),v ⋅ ñ)_{Γ_T}
//                                 = αₖ(f,v) - (R_N'(ϕ − uᵏ⁻¹ ⋅ ñ),v ⋅ ñ)_{Γ_T}
//
//               for all v ∈ H¹(Ω,R³). This is a nonlinear problem in uᵏ, so we
//               employ Newton's method. Defining Fₖ : H¹(Ω,R³) → (H¹(Ω,R³))'
//               as
//
//                 <Fₖ(u),v> =   αₖ(σ(u),ε(v)) - (R_N'(ϕ − u ⋅ ñ),v ⋅ ñ)_{Γ_T}
//                             - αₖ(f,v) + (R_N'(ϕ − uᵏ⁻¹ ⋅ ñ),v ⋅ ñ)_{Γ_T},
//
//               we solve the following problem:
//
//                            Find δuᵏ_j ∈ H¹(Ω,R³) such that
//                       Fₖ'(uᵏ_j) δuᵏ_j = -Fₖ(uᵏ_j)    in (H¹(Ω,R³))',
//
//               where
//
//               <Fₖ'(u) w,v>
//                      = αₖ(σ(w),ε(v)) + (R_N''(ϕ − u ⋅ ñ) w ⋅ ñ,v ⋅ ñ)_{Γ_T}.
//
//               The solution is then updated as uᵏ_{j+1} = uᵏ_j + δuᵏ_j until
//               convergence. The initial guess uᵏ_0 is taken as uᵏ⁻¹.
//
//               We take f ≡ (0,0,-2)^T, g ≡ 0, ñ ≡ (0,0,-1)^T on Ω, the unit
//               cube in R³.

#include "mfem.hpp"
#include "sfem/bilininteg.hpp"
#include "sfem/lininteg.hpp"
#include "sfem/coefficient.hpp"

using namespace std;
using namespace mfem;

void   InitDisplacement(const Vector &x, Vector &u);
real_t GapFunction(const Vector &x);
void   ForceFunction(const Vector &x, Vector &f);
real_t RegLogPrime(const real_t a);
real_t RegLogDoublePrime(const real_t a);

/**
 * @brief Returns a Coefficient object for R_N'(ϕ − u ⋅ ñ) for a given
 *        GridFunction u and unit vector field ñ.
 *
 * @param u GridFunction
 * @param n_tilde Unit vector field
 */
class RegLogPrimeCoefficient : public Coefficient
{
private:
   GridFunction *u;
   Vector &n_tilde;
   real_t sign;

public:
   RegLogPrimeCoefficient(GridFunction *_u, Vector &_n_tilde,
      real_t _sign = 1.0) : u(_u), n_tilde(_n_tilde), sign(_sign) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

/**
 * @brief Returns a Coefficient object for R_N''(ϕ − u ⋅ ñ) for a given
 *        GridFunction u and unit vector field ñ.
 *
 * @param u GridFunction
 * @param n_tilde Unit vector field
 */
class RegLogDoublePrimeCoefficient : public Coefficient
{
private:
   GridFunction *u;
   Vector &n_tilde;
   real_t sign;

public:
   RegLogDoublePrimeCoefficient(GridFunction *_u, Vector &_n_tilde,
      real_t _sign = 1.0) : u(_u), n_tilde(_n_tilde), sign(_sign) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

// We take the plane to be z = plane_g.
const real_t plane_g = 0.0;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options
   const char* mesh_file = "../../data/ref-cube.mesh";
   int order = 1;
   int ref_levels = 0;
   real_t alpha = 1.0;
   real_t lambda = 1.0;
   real_t mu = 1.0;
   int max_outer_iter = 30;
   int max_newton_iter = 15;
   real_t itol = 1e-9;
   real_t ntol = 1e-12;
   bool reorder_space = false;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of uniform mesh refinements.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha parameter for boundary condition.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lamé's first parameter.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lamé's second parameter.");
   args.AddOption(&max_outer_iter, "-i", "--outer-iterations",
                  "Maximum number of iterations.");
   args.AddOption(&max_newton_iter, "-n", "--newton-iterations",
                  "Maximum number of Newton iterations.");
   args.AddOption(&itol, "-tol", "--tolerance",
                  "Iteration tolerance.");
   args.AddOption(&reorder_space, "-nodes", "--by-nodes", "-vdim", "--by-vdim",
                  "Use byNODES ordering of vector space instead of byVDIM");
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
   if (mu <= 0.0 || lambda + 2.0/3.0 * mu <= 0.0)
   {
      MFEM_WARNING("Unphysical Lamé parameters.");
   }
   if (myid == 0)
   {
      args.PrintOptions(mfem::out);
   }

   // 2. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // 3. Postprocess the mesh.
   // 3A. Refine the serial mesh on all processors to increase the resolution. In
   //     this program we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 3B. Interpolate the geometry after refinement to control geometry error.
   // NOTE: Minimum second-order interpolation is used to improve the accuracy.
   int curvature_order = max(order,2);
   mesh.SetCurvature(curvature_order);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh = ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 5. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   if (pmesh.NURBSext)
   {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh.GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      if (reorder_space)
      {
         fespace = new ParFiniteElementSpace(&pmesh, fec, dim, Ordering::byNODES);
      }
      else
      {
         fespace = new ParFiniteElementSpace(&pmesh, fec, dim, Ordering::byVDIM);
      }
   }
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 6. Define coefficients for later.
   VectorFunctionCoefficient f_coeff(dim, ForceFunction);
   ScalarVectorProductCoefficient alpha_f_coeff(alpha, f_coeff);
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);

   Vector n_tilde(dim);
   n_tilde = 0.0;
   n_tilde(dim-1) = -1.0;
   {
      real_t n_tilde_norm = n_tilde.Norml2();
      if (n_tilde_norm != 1.0)
      {
         if (myid == 0)
         {
            MFEM_WARNING("n_tilde norm is not 1.0, normalizing it.");
         }
         n_tilde /= n_tilde_norm;
      }
   }

   // 7A. Define the solution vector u as a parallel finite element grid
   //     function corresponding to fespace. Initialize u with initial guess of
   //     u(x,y,z) = (0,0,plane_g), which satisfies the boundary conditions.
   ParGridFunction u_previous(fespace);
   ParGridFunction u_current(fespace);
   ParGridFunction delta_u(fespace);
   GridFunctionCoefficient u_previous_coeff(&u_previous);
   VectorFunctionCoefficient init_u(dim, InitDisplacement);
   u_previous.ProjectCoefficient(init_u);
   u_current = u_previous;
   delta_u = 0.0;

   // 7B. Define the regularized log prime and double prime coefficients present
   //     in the Newton iteration.
   StressGridFunctionCoefficient stress_u_curr_coeff(lambda, mu, &u_current);
   FlatVectorCoefficient nalpha_vstress_u_curr_coeff(stress_u_curr_coeff, -alpha);
   RegLogDoublePrimeCoefficient reg_log_dp_curr_coeff(&u_current, n_tilde);
   RegLogPrimeCoefficient reg_log_p_curr_coeff(&u_current, n_tilde);
   RegLogPrimeCoefficient nreg_log_p_prev_coeff(&u_previous, n_tilde, -1.0);

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs.
   Array<int> ess_bdr_x(pmesh.bdr_attributes.Max());
   Array<int> ess_bdr_y(pmesh.bdr_attributes.Max());
   Array<int> ess_bdr_z(pmesh.bdr_attributes.Max());
   ess_bdr_x = 0; ess_bdr_x[2] = 1; ess_bdr_x[4] = 1;
   ess_bdr_y = 0; ess_bdr_y[1] = 1; ess_bdr_y[3] = 1;
   ess_bdr_z = 0; ess_bdr_z[0] = 1;

   Array<int> ess_tdof_list_x, ess_tdof_list_y, ess_tdof_list_z;
   fespace->GetEssentialTrueDofs(ess_bdr_x, ess_tdof_list_x, 0);
   fespace->GetEssentialTrueDofs(ess_bdr_y, ess_tdof_list_y, 1);

   Array<int> ess_tdof_list;
   ess_tdof_list.Append(ess_tdof_list_x);
   ess_tdof_list.Append(ess_tdof_list_y);

   // 9. Set up GLVis visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   real_t newton_error, iter_error;

   if (myid == 0)
   {
      mfem::out << "\nk" << setw(3) << "j" << setw(14) << "newton_error"
                << setw(14) << "iter_error" << std::endl;
      mfem::out << "--------------------------------" << std::endl;
   }

   // 10. Iterate:
   for (int k = 1; k <= max_outer_iter; k++)
   {
      ConstantCoefficient alpha_coeff(alpha);

      // Newton loop
      for (int j = 0; j <= max_newton_iter; j++)
      {
         if (j == 0)
         {
            u_current = u_previous;
         }

         // Step 1: Set up the bilinear form a(⋅,⋅) on the finite element space.
         ParBilinearForm *a = new ParBilinearForm(fespace);
         a->AddDomainIntegrator(new ElasticityIntegrator(alpha_coeff,lambda,mu));
         a->AddBdrFaceIntegrator(
            new BoundaryProjectionIntegrator(reg_log_dp_curr_coeff,
            n_tilde), ess_bdr_z);
         a->Assemble();

         // Step 2: Set up the linear form b(⋅) on the finite element space.
         ParLinearForm *b = new ParLinearForm(fespace);
         b->AddDomainIntegrator(new VectorDomainLFStrainIntegrator(nalpha_vstress_u_curr_coeff));
         b->AddBdrFaceIntegrator(
            new BoundaryProjectionLFIntegrator(reg_log_p_curr_coeff, n_tilde),
            ess_bdr_z);
         b->AddDomainIntegrator(new VectorDomainLFIntegrator(alpha_f_coeff));
         b->AddBdrFaceIntegrator(
            new BoundaryProjectionLFIntegrator(nreg_log_p_prev_coeff, n_tilde),
            ess_bdr_z);
         b->Assemble();

         // Step 3: Form the linear system A X = B. This includes eliminating boundary
         // conditions, applying AMR constraints, and other transformations.
         HypreParMatrix A;
         Vector B, X;
         a->FormLinearSystem(ess_tdof_list, delta_u, *b, A, X, B);

         // Step 4: Solve the linear system.
         HypreBoomerAMG *amg = new HypreBoomerAMG(A);
         amg->SetElasticityOptions(fespace);
         amg->SetPrintLevel(0);
         HyprePCG *pcg = new HyprePCG(A);
         pcg->SetTol(1e-12);
         pcg->SetMaxIter(500);
         pcg->SetPrintLevel(0);
         pcg->SetPreconditioner(*amg);
         pcg->Mult(B, X);

         // Step 5: Recover the solution.
         a->RecoverFEMSolution(X, *b, delta_u);

         // Step 6: Update u_current.
         u_current += delta_u;

         // Step 7: Check for convergence.
         newton_error = delta_u.ComputeL2Error(zero);

         if (myid == 0)
         {
            mfem::out << setw(4) << j << setw(14) << newton_error << std::endl;
         }

         // Step 8: Free used memory.
         delete amg;
         delete pcg;
         delete a;
         delete b;

         if (newton_error < ntol)
         {
            break;
         }
      }

      // Step 9: Compute difference between current and previous solutions.
      iter_error = u_current.ComputeL2Error(u_previous_coeff);

      if (myid == 0)
      {
         mfem::out << k << setw(4) << " " << setw(13) << " " << setw(14)
                   << iter_error << setw(14) << std::endl;
      }

      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << u_current << std::flush;
      }

      // Step 10: Check for convergence.
      if (iter_error < itol)
      {
         if (myid == 0)
         {
            mfem::out << "\nConverged after " << k << " iterations." << std::endl;
         }
         break;
      }

      // Step 11: Update previous solution for next iteration.
      u_previous = u_current;
   }

   // 12. Free used memory.
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   return 0;
}

/**
 * @brief u(x,y,z) = (0,0,plane_g)
 */
void InitDisplacement(const Vector &x, Vector &u)
{
   u = 0.0;
   u(x.Size() - 1) = plane_g;
}

/**
 * @brief Computes the gap function φ based on the input vector x; represents
 *        the distance between a point x and the plane z = plane_g.
 *
 * @param  x Input vector
 * @return real_t Computed gap function value, φ(x)
 */
real_t GapFunction(const Vector &x)
{
   return x(x.Size() - 1) - plane_g;
}

/**
 * @brief Computes the force function based on the input vector x.
 *
 * @param x Input vector
 * @param f Output vector representing the downward force
 */
void ForceFunction(const Vector &x, Vector &f)
{
   const real_t force = -2.0;

   f = 0.0;
   f(x.Size() - 1) = force;
}

real_t RegLogPrime(const real_t a)
{
   // TODO: Make N a parameter
   const real_t N = 1e5;

   if (a > N)
   {
      return log(N) + 1.0/N * (a - N);
   }
   else if (a < 1.0/N)
   {
      return -log(N) + N * (a - 1.0/N);
   }
   else
   {
      return log(a);
   }
}

real_t RegLogDoublePrime(const real_t a)
{
   // TODO: Make N a parameter
   const real_t N = 1e5;

   if (a > N)
   {
      return 1.0 / N;
   }
   else if (a < 1.0/N)
   {
      return N;
   }
   else
   {
      return 1.0 / a;
   }
}

real_t RegLogPrimeCoefficient::Eval(ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   ParGridFunction *par_u = dynamic_cast<ParGridFunction*>(u);
   const int dim = T.GetSpaceDim();

   // Get current point coordinates
   Vector x(dim);
   T.Transform(ip, x);

   // Get value of u at the integration point
   Vector u_val(dim);
   if (par_u)
   {
      par_u->GetVectorValue(T, ip, u_val);
   }
   else
   {
      u->GetVectorValue(T, ip, u_val);
   }

   // Evaluate the gap function φ
   const real_t phi = GapFunction(x);

   return sign * RegLogPrime(phi - u_val * n_tilde);
}

real_t RegLogDoublePrimeCoefficient::Eval(ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   ParGridFunction *par_u = dynamic_cast<ParGridFunction*>(u);
   const int dim = T.GetSpaceDim();

   // Get current point coordinates
   Vector x(dim);
   T.Transform(ip, x);

   // Get value of u at the integration point
   Vector u_val(dim);
   if (par_u)
   {
      par_u->GetVectorValue(T, ip, u_val);
   }
   else
   {
      u->GetVectorValue(T, ip, u_val);
   }

   // Evaluate the gap function φ
   const real_t phi = GapFunction(x);

   return sign * RegLogDoublePrime(phi - u_val * n_tilde);
}
