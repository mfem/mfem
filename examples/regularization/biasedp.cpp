//                               MFEM Signorini
//
// Compile with: make biasedp
//
// Sample runs:  mpirun -np 4 biasedp -r 1 -f 3 -a 1 -i 10 -vis
//               mpirun -np 4 biasedp -f 5 -s 0 -i 7 -n 10 -m ../../data/wheel.msh -vis
//               mpirun -np 4 biasedp -f 5 -s 0 -i 7 -n 10 -m ../../data/wheel.msh -pv
//
// Description:  This program solves a biased Signorini's problem using MFEM.
//               We aim to solve the bound-constrained minimization problem
//
//                 minimize ∑ᵢ₌₁² 1/2 ∫_Ωⁱ σ(uⁱ) : ε(uⁱ) dx − ∫_Ωⁱ fⁱ ⋅ uⁱ dx
//                                      subject to u ∈ K,
//
//               where
//
//                 V = { v = (v¹,v²) ∈ H¹(Ω¹,R³) × H¹(Ω²,R³) : v = 0 on Γ_D},
//                               K = { v ∈ V : Bv ≤ ϕ on Γ¹_T },
//
//               B : V → L¹(Γ¹_T) is defined by Bv = (v¹ - v² ∘ Π) ⋅ n,
//               Γ_D ∪ Γ_T = ∂Ω, ϕ is a gap function defined on Γ¹_T,
//               and fⁱ ∈ L²(Ωⁱ,R³) are given forcing terms. The gap function ϕ
//               is defined according to the contact pairing mapping
//               Π : Γ¹_T → Γ²_T, ϕ(x) = (Π(x) - x) · n(x). The vector function
//               n : Γ¹_T → R³ is defined by
//
//                  n(x) = sign((Π(x) - x) ⋅ n¹(x)) / |Π(x) - x| (Π(x) - x)
//
//               if (Π(x) - x) ⋅ n¹(x) ≠ 0, and n(x) = n¹(x) otherwise. We employ
//               Bregman proximal point with the Legendre function R_N : R → R
//               defined as
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
//               and the Bregman proximal point iteration reads: Given uₖ₋₁ ∈ K
//               and an unsummable sequence {αₖ}ₖ > 0, find uₖ ∈ V such that
//
//                αₖ ∑ᵢ₌₁² (σ(uₖⁱ),ε(vₖⁱ)) - (R_N'(ϕ − Buₖ),Bv)_{Γ¹_T}
//                                 = αₖ ∑ᵢ₌₁² (fⁱ,vⁱ) - (R_N'(ϕ - Buₖ₋₁),Bv)_{Γ¹_T}
//
//               for all v ∈ V. This is a nonlinear problem in uᵏ, so we
//               employ Newton's method. Defining Fₖ : V → V' as
//
//                  <Fₖ(u),v> = αₖ <E'(u),v> - (R_N'(ϕ − Bu),Bv)_{Γ¹_T}
//                                             + (R_N'(ϕ − Buₖ₋₁),Bv)_{Γ¹_T},
//
//               we solve the following problem:
//
//                 Find δuₖ_j ∈ V such that Fₖ'(uₖ,ⱼ) δuₖ,ⱼ = -Fₖ(uₖ,ⱼ) in V',
//
//               where
//
//                   <Fₖ'(u) w,v> = αₖ a(w,v) + (R_N''(ϕ - Bu) Bw,Bv)_{Γ¹_T},
//                              a(u,v) = ∑ᵢ₌₁² (σ(uₖⁱ),ε(vₖⁱ)).
//
//               The solution is then updated as uₖ_{j+1} = uₖ,ⱼ + δuₖ,ⱼ until
//               convergence. The initial guess uₖ,₀ is taken as uₖ₋₁.
//
//               We take f¹ ≡ (0,0,-2)^T, f² ≡ 0, and Ω² to be a slab.

#include "mfem.hpp"
#include "sfem/bilininteg.hpp"
#include "sfem/lininteg.hpp"
#include "sfem/coefficient.hpp"
#include <fstream>
#include <filesystem>

using namespace std;
using namespace mfem;

/**
 * @brief u(x,y,z) = (0,0,slab_g)
 */
void InitDisplacement(const Vector &x, Vector &u);

/**
 * @brief Computes the force function based on the input vector x.
 *
 * @param x Input vector
 * @param f Output vector representing the downward force
 */
void ForceFunction(const Vector &x, Vector &f);

/**
 * @brief Computes the contact pairing Π. For a slab Ω² with top surface at
 *        z = slab_g, Π is the projection onto that plane.
 *
 * @param x  Input vector
 * @param pi Output vector, Π(x)
 */
void Pi(const Vector &x, Vector &pi);

/**
 * @brief Returns a VectorCoefficient object for the vector function n.
 */
class NVectorCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient *Pi;
   VectorCoefficient *nt;

public:
   NVectorCoefficient(int dim, VectorCoefficient *_Pi)
      : VectorCoefficient(dim), Pi(_Pi), nt(NULL) { }

   NVectorCoefficient(int dim, VectorCoefficient *_Pi,
      VectorCoefficient *_nt) : VectorCoefficient(dim),
      Pi(_Pi), nt(_nt) { }

   virtual void Eval(Vector &N, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

/**
 * @brief Returns a Coefficient object for the gap function ϕ.
 */
class GapFunctionCoefficient : public Coefficient
{
private:
   VectorCoefficient *Pi;
   VectorCoefficient *n;

public:
   GapFunctionCoefficient(VectorCoefficient *_Pi, VectorCoefficient *_n)
      : Pi(_Pi), n(_n) { }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

/**
 * @brief Computes ϕ − Bu = ϕ − (u¹ − u² ∘ Π) · n
 */
class RegLogCoefficientBase : public Coefficient
{
protected:
   GridFunction *u1;
   GridFunction *u2_pi;
   Coefficient *gap;
   VectorCoefficient *n;
   real_t N;

   real_t EvalArg(ElementTransformation &T, const IntegrationPoint &ip)
   {
      ParGridFunction *par_u1 = dynamic_cast<ParGridFunction*>(u1);
      ParGridFunction *par_u2_pi = dynamic_cast<ParGridFunction*>(u2_pi);
      const int dim = T.GetSpaceDim();

      // Get value of u1 at x
      Vector u1_val(dim);
      if (par_u1) { par_u1->GetVectorValue(T, ip, u1_val); }
      else { u1->GetVectorValue(T, ip, u1_val); }

      // Get value of u2 ∘ Π at x. Because u2 has been pre-transferred onto
      // mesh1's fespace via FindPointsGSLIB, we evaluate it at (T, ip)
      // directly.
      Vector u2_pi_val(dim);
      if (par_u2_pi) { par_u2_pi->GetVectorValue(T, ip, u2_pi_val); }
      else { u2_pi->GetVectorValue(T, ip, u2_pi_val); }

      // Store u1 - u2 ∘ Π at x
      Vector diff(dim);
      subtract(u1_val, u2_pi_val, diff);

      // Get value of n at x
      Vector n_val(dim);
      n->Eval(n_val, T, ip);

      return gap->Eval(T, ip) - diff * n_val;
   }

public:
   RegLogCoefficientBase(GridFunction *_u1, GridFunction *_u2,
      Coefficient *_gap, VectorCoefficient *_n, real_t _N = 1e2)
      : u1(_u1), u2_pi(_u2), gap(_gap), n(_n), N(_N) {}
};

/**
 * @brief Computes ϕ − Bu = ϕ − (u¹ − u² ∘ Π) · n
 */
class RegLogCoefficientBase : public Coefficient
{
protected:
   GridFunction *u1;
   GridFunction *u2_pi;
   Coefficient *gap;
   VectorCoefficient *n;
   real_t N;

   real_t EvalArg(ElementTransformation &T, const IntegrationPoint &ip)
   {
      ParGridFunction *par_u1 = dynamic_cast<ParGridFunction*>(u1);
      ParGridFunction *par_u2_pi = dynamic_cast<ParGridFunction*>(u2_pi);
      const int dim = T.GetSpaceDim();

      // Get value of u1 at x
      Vector u1_val(dim);
      if (par_u1) { par_u1->GetVectorValue(T, ip, u1_val); }
      else { u1->GetVectorValue(T, ip, u1_val); }

      // Get value of u2 ∘ Π at x. Because u2 has been pre-transferred onto
      // mesh1's fespace via FindPointsGSLIB, we evaluate it at (T, ip)
      // directly.
      Vector u2_pi_val(dim);
      if (par_u2_pi) { par_u2_pi->GetVectorValue(T, ip, u2_pi_val); }
      else { u2_pi->GetVectorValue(T, ip, u2_pi_val); }

      // Store u1 - u2 ∘ Π at x
      Vector diff(dim);
      subtract(u1_val, u2_pi_val, diff);

      // Get value of n at x
      Vector n_val(dim);
      n->Eval(n_val, T, ip);

      return gap->Eval(T, ip) - diff * n_val;
   }

public:
   RegLogCoefficientBase(GridFunction *_u1, GridFunction *_u2,
      Coefficient *_gap, VectorCoefficient *_n, real_t _N = 1e2)
      : u1(_u1), u2_pi(_u2), gap(_gap), n(_n), N(_N) {}
};

/**
 * @brief Returns a Coefficient object for R_N'(ϕ − Bu) for given GridFunctions
 *        u1, u2.
 *
 * @param u1 GridFunction
 * @param u2 GridFunction
 * @param gap Coefficient
 * @param n VectorCoefficient
 * @param N Regularization parameter for the regularized log function (default: 1e2)
 * @param sign Sign to apply to the coefficient (default: 1.0)
 */
class RegLogPrimeCoefficient : public RegLogCoefficientBase
{
private:
   real_t sign;

public:
   RegLogPrimeCoefficient(GridFunction *u1, GridFunction *u2,
      Coefficient *gap, VectorCoefficient *n,
      real_t N = 1e2, real_t _sign = 1.0)
      : RegLogCoefficientBase(u1, u2, gap, n, N), sign(_sign) {}

   static real_t RegLogPrime(const real_t a, const real_t M);

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return sign * RegLogPrime(EvalArg(T, ip), N);
   }
};

/**
 * @brief Returns a Coefficient object for R_N''(ϕ − Bu) for given GridFunctions
 *        u1, u2.
 *
 * @param u1 GridFunction
 * @param u2 GridFunction
 * @param gap Coefficient
 * @param n VectorCoefficient
 * @param N Regularization parameter for the regularized log function (default: 1e2)
 */
class RegLogDoublePrimeCoefficient : public RegLogCoefficientBase
{
public:
   RegLogDoublePrimeCoefficient(GridFunction *u1, GridFunction *u2,
      Coefficient *gap, VectorCoefficient *n, real_t N = 1e2)
      : RegLogCoefficientBase(u1, u2, gap, n, N) { }

   static real_t RegLogDoublePrime(const real_t a, const real_t M);

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return RegLogDoublePrime(EvalArg(T, ip), N);
   }
};

// We take a slab with top surface at z = slab_g and the force to be a
// constant downward force of magnitude force_g.
real_t slab_g = -0.5;
real_t force_g = 2.0;

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
   real_t alpha = 0.5;
   real_t lambda1 = 1.0;
   real_t mu1 = 1.0;
   real_t lambda2 = 1e3;
   real_t mu2 = 1e3;
   int max_outer_iter = 30;
   int max_newton_iter = 15;
   real_t itol = 1e-8;
   real_t ntol = 1e-8;
   bool reorder_space = false;
   bool visualization = false;
   bool paraview_output = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of uniform mesh refinements.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha parameter for boundary condition.");
   args.AddOption(&lambda1, "-lambda1", "--lambda1",
                  "Lamé's first parameter for Ω¹.");
   args.AddOption(&mu1, "-mu1", "--mu1",
                  "Lamé's second parameter for Ω¹.");
   args.AddOption(&lambda2, "-lambda2", "--lambda2",
                  "Lamé's first parameter for Ω².");
   args.AddOption(&mu2, "-mu2", "--mu2",
                  "Lamé's second parameter for Ω².");
   args.AddOption(&slab_g, "-s", "--slab",
                  "Height of the slab for the Signorini condition.");
   args.AddOption(&force_g, "-f", "--force",
                  "Magnitude of the downward force for Ω¹.");
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
   args.AddOption(&paraview_output, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable ParaView output.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (mu1 <= 0.0 || lambda1 + 2.0/3.0 * mu1 <= 0.0)
   {
      if (myid == 0)
      {
         MFEM_WARNING("Unphysical Lamé parameters for Ω¹.");
      }
   }
   if (mu2 <= 0.0 || lambda2 + 2.0/3.0 * mu2 <= 0.0)
   {
      if (myid == 0)
      {
         MFEM_WARNING("Unphysical Lamé parameters for Ω².");
      }
   }
   if (myid == 0)
   {
      args.PrintOptions(mfem::out);
   }

   // 2A. Read the (serial) mesh from the given mesh file on all processors.  We
   //     can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //     and volume meshes with the same code.
   Mesh mesh1(mesh_file, 1, 1);
   filesystem::path mesh_path(mesh_file);
   string mesh_stem = mesh_path.stem().string();
   if (mesh_stem == "wheel")
   {
      mesh1.Transform([](const Vector &x_in, Vector &x_out) {
         x_out.SetSize(3);
         x_out(0) = x_in(0) - 0.1;
         x_out(1) = x_in(1) - 0.5;
         x_out(2) = x_in(2);
      });
   }
   const int dim1 = mesh1.Dimension();

   // 2B. Create the slab mesh.
   Mesh mesh2;
   {
      real_t thickness  = 0.25;  // slab thickness in z
      real_t half_width = 1.0;   // half the side-length of the square top face
      int    nx         = 2;     // elements in x
      int    ny         = 2;     // elements in y
      int    nz         = 2;     // elements in z
      real_t side = 2.0 * half_width;

      mesh2 = Mesh::MakeCartesian3D(
         nx, ny, nz, Element::HEXAHEDRON,
         side,       // x-extent
         side,       // y-extent
         thickness   // z-extent
      );

      // Shift so the slab is centred in x/y and its top sits at z = slab_g
      auto transform = [&](const Vector &x_in, Vector &x_out)
      {
         x_out.SetSize(3);
         x_out(0) = x_in(0) - half_width;              // centre x
         x_out(1) = x_in(1) - half_width;              // centre y
         x_out(2) = x_in(2) + (slab_g - thickness);    // top at slab_g
      };
      mesh2.Transform(transform);
   }
   const int dim2 = mesh2.Dimension();

   if (dim1 != dim2)
   {
      if (myid == 0)
      {
         MFEM_ABORT("dim(Ω¹) ≠ dim(Ω²).");
      }
   }

   // 3. Postprocess the meshes.
   // 3A. Refine the serial meshes on all processors to increase the resolution. In
   //     this program we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh1.UniformRefinement();
      mesh2.UniformRefinement();
   }

   // 3B. Interpolate the geometry after refinement to control geometry error.
   // NOTE: Minimum second-order interpolation is used to improve the accuracy.
   int curvature_order = max(order,2);
   mesh1.SetCurvature(curvature_order);
   mesh2.SetCurvature(curvature_order);

   // 4. Define parallel meshes by a partitioning of the serial meshes. Once the
   //    parallel meshes are defined, the serial meshes can be deleted.
   ParMesh pmesh1 = ParMesh(MPI_COMM_WORLD, mesh1);
   ParMesh pmesh2 = ParMesh(MPI_COMM_WORLD, mesh2);
   mesh1.Clear();
   mesh2.Clear();

   // 5. Define finite element spaces on the meshes. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace1;
   ParFiniteElementSpace *fespace2;
   if (pmesh1.NURBSext)
   {
      fec = NULL;
      fespace1 = (ParFiniteElementSpace *)pmesh1.GetNodes()->FESpace();
      fespace2 = (ParFiniteElementSpace *)pmesh2.GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim1);
      if (reorder_space)
      {
         fespace1 = new ParFiniteElementSpace(&pmesh1, fec, dim1, Ordering::byNODES);
         fespace2 = new ParFiniteElementSpace(&pmesh2, fec, dim2, Ordering::byNODES);
      }
      else
      {
         fespace1 = new ParFiniteElementSpace(&pmesh1, fec, dim1, Ordering::byVDIM);
         fespace2 = new ParFiniteElementSpace(&pmesh2, fec, dim2, Ordering::byVDIM);
      }
   }
   HYPRE_BigInt size = fespace1->GlobalTrueVSize() + fespace2->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 6. Define coefficients for later.
   Vector zero_vec(dim2); zero_vec = 0.0;
   ConstantCoefficient zero_coeff(0.0);
   ConstantCoefficient one(1.0);
   VectorFunctionCoefficient f1_coeff(dim1, ForceFunction);
   VectorConstantCoefficient f2_coeff(zero_vec);
   VectorFunctionCoefficient pi_coeff(dim1, Pi);

   Vector n_tilde(dim1);
   n_tilde = 0.0;
   n_tilde(dim1-1) = -1.0;
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
   VectorConstantCoefficient n_tilde_coeff(n_tilde);

   NVectorCoefficient n_coeff(dim1, &pi_coeff, &n_tilde_coeff);
   GapFunctionCoefficient gap_coeff(&pi_coeff, &n_coeff);

   // 7. Define the solution vector u as a parallel finite element grid
   //    function corresponding to fespace. Initialize u with initial guess of
   //    u(x,y,z) = (0,0,slab_g), which satisfies the boundary conditions.
   ParGridFunction u1_previous(fespace1), u2_previous(fespace2);
   ParGridFunction u1_current(fespace1), u2_current(fespace2);
   ParGridFunction delta_u1(fespace1), delta_u2(fespace2);
   VectorGridFunctionCoefficient u1_previous_coeff(&u1_previous);
   VectorGridFunctionCoefficient u2_previous_coeff(&u2_previous);
   VectorFunctionCoefficient init_u1(dim1, InitDisplacement);
   u1_previous.ProjectCoefficient(init_u1);
   u1_current = u1_previous;
   delta_u1 = 0.0;
   u2_previous = 0.0;
   u2_current = 0.0;
   delta_u2 = 0.0;

   // 8A. Determine the list of true (i.e. parallel conforming) essential
   //     boundary dofs.
   Array<int> ess_bdr_x(pmesh1.bdr_attributes.Max());
   Array<int> ess_bdr_y(pmesh1.bdr_attributes.Max());
   Array<int> ess_bdr_z(pmesh1.bdr_attributes.Max());
   ess_bdr_x = 0; ess_bdr_y = 0; ess_bdr_z = 0;

   // 8B. Apply boundary conditions for each mesh.
   if (mesh_stem == "ref-cube")
   {
      ess_bdr_x[2] = 1; ess_bdr_x[4] = 1;
      ess_bdr_y[1] = 1; ess_bdr_y[3] = 1;
      ess_bdr_z[0] = 1;
   }
   else if (mesh_stem == "wheel")
   {
      ess_bdr_x[1] = 1; ess_bdr_x[2] = 1; ess_bdr_x[3] = 1;
      ess_bdr_y[1] = 1; ess_bdr_y[2] = 1; ess_bdr_y[3] = 1;
      ess_bdr_z[0] = 1;
   }
   else if (mesh_stem == "hemisphere")
   {
      ess_bdr_x[1] = 1;
      ess_bdr_y[1] = 1;
      ess_bdr_z[0] = 1;
   }
   else
   {
      MFEM_ABORT("Unknown mesh file. Please specify essential boundary "
                 "conditions for this mesh.");
   }

   Array<int> ess_tdof_list_x, ess_tdof_list_y;
   fespace1->GetEssentialTrueDofs(ess_bdr_x, ess_tdof_list_x, 0);
   fespace1->GetEssentialTrueDofs(ess_bdr_y, ess_tdof_list_y, 1);

   Array<int> ess_tdof_list;
   ess_tdof_list.Append(ess_tdof_list_x);
   ess_tdof_list.Append(ess_tdof_list_y);

   // 8C. Essential boundary dofs for Ω² (slab). Clamp all DOFs on the bottom
   //     face. For MakeCartesian3D hex meshes: attribute 1 = bottom (z_min).
   Array<int> ess_bdr2(pmesh2.bdr_attributes.Max());
   ess_bdr2 = 1;   // clamp all faces of the slab
   ess_bdr2[5] = 0; // free the top face (attribute 6 = z_max, index 5)
   Array<int> ess_tdof_list2;
   fespace2->GetEssentialTrueDofs(ess_bdr2, ess_tdof_list2);

   // 9A. Set up GLVis visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 9B. Set up cross-mesh transfer from Ω² to Ω¹ via Π using FindPointsGSLIB.
   //     We precompute the Π-mapped DOF coordinates of fespace1, so that at each
   //     iteration we can transfer u2 (living on mesh2) to a GridFunction on
   //     fespace1 that represents u2 ∘ Π. This avoids cross-mesh evaluation in
   //     the coefficient Eval methods.
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(pmesh2);

   // Get physical coordinates of fespace1's scalar DOFs via identity projection
   ParGridFunction id_coords(fespace1);
   {
      VectorFunctionCoefficient id_func(dim1,
         [](const Vector &x, Vector &y) { y = x; });
      id_coords.ProjectCoefficient(id_func);
   }

   // Apply Π to each DOF coordinate; pack in byNODES layout for FindPointsGSLIB
   int scalar_ndofs = fespace1->GetNDofs();
   Vector pi_xyz(scalar_ndofs * dim1);
   for (int i = 0; i < scalar_ndofs; i++)
   {
      Vector xi(dim1);
      for (int d = 0; d < dim1; d++)
      {
         xi(d) = id_coords(fespace1->DofToVDof(i, d));
      }
      Vector pix(dim1);
      Pi(xi, pix);
      for (int d = 0; d < dim1; d++)
      {
         pi_xyz(d * scalar_ndofs + i) = pix(d);
      }
   }

   // Locate the Π-mapped points on mesh2 once (reused every transfer call)
   finder.FindPoints(pi_xyz, Ordering::byNODES);

   // Transferred grid functions: u2_curr/prev ∘ Π on fespace1
   ParGridFunction u2_curr_on_mesh1(fespace1);
   ParGridFunction u2_prev_on_mesh1(fespace1);
   u2_curr_on_mesh1 = 0.0;
   u2_prev_on_mesh1 = 0.0;

   // Helper: transfer a GridFunction from fespace2 to fespace1 via Π
   auto TransferU2 = [&](ParGridFunction &u2_src, ParGridFunction &u2_dst)
   {
      Vector u2_interp(scalar_ndofs * dim1);
      finder.Interpolate(u2_src, u2_interp);
      for (int i = 0; i < scalar_ndofs; i++)
      {
         for (int d = 0; d < dim1; d++)
         {
            u2_dst(fespace1->DofToVDof(i, d)) =
               u2_interp(d * scalar_ndofs + i);
         }
      }
   };

   // 9C. Set up ParaView output. We need one data collection per mesh so that
   //     both geometries appear in ParaView. Both write to the same prefix path;
   //     load both .pvd files in ParaView to see them in a single scene.
   ParaViewDataCollection paraview_dc1(mesh_stem, &pmesh1);
   ParaViewDataCollection paraview_dc2(mesh_stem + "-slab", &pmesh2);
   if (paraview_output)
   {
      paraview_dc1.SetPrefixPath("ParaView");
      paraview_dc1.SetLevelsOfDetail(curvature_order);
      paraview_dc1.SetDataFormat(VTKFormat::BINARY);
      paraview_dc1.SetHighOrderOutput(true);
      paraview_dc1.RegisterField("Displacement", &u1_previous);
      paraview_dc1.SetCycle(0);
      paraview_dc1.SetTime(0.0);
      paraview_dc1.Save();

      paraview_dc2.SetPrefixPath("ParaView");
      paraview_dc2.SetLevelsOfDetail(curvature_order);
      paraview_dc2.SetDataFormat(VTKFormat::BINARY);
      paraview_dc2.SetHighOrderOutput(true);
      paraview_dc2.RegisterField("Slab", &u2_previous);
      paraview_dc2.SetCycle(0);
      paraview_dc2.SetTime(0.0);
      paraview_dc2.Save();
   }

   real_t newton_error, iter_error;

   if (myid == 0)
   {
      mfem::out << "\nk" << setw(3) << "j" << setw(14) << "newton_error"
                << setw(14) << "iter_error" << std::endl;
      mfem::out << "--------------------------------" << std::endl;
   }

   // 10. Iterate:
   real_t N = 1e6;
   for (int k = 1; k <= max_outer_iter; k++)
   {
      alpha = 2 * k;
      u1_current = u1_previous;
      u2_current = u2_previous;

      // Transfer u2 from mesh2 to mesh1 via Π for this outer iteration
      TransferU2(u2_current, u2_curr_on_mesh1);
      TransferU2(u2_previous, u2_prev_on_mesh1);

      // Newton loop
      for (int j = 0; j <= max_newton_iter; j++)
      {
         delta_u1 = 0.0;
         delta_u2 = 0.0;

         ConstantCoefficient alpha_coeff(alpha);
         ScalarVectorProductCoefficient alpha_f1_coeff(alpha, f1_coeff);
         ScalarVectorProductCoefficient alpha_f2_coeff(alpha, f2_coeff);

         RegLogDoublePrimeCoefficient reg_log_dp_curr_coeff(&u1_current,
            &u2_curr_on_mesh1, &gap_coeff, &n_coeff, N);
         RegLogPrimeCoefficient reg_log_p_curr_coeff(&u1_current,
            &u2_curr_on_mesh1, &gap_coeff, &n_coeff, N);
         RegLogPrimeCoefficient nreg_log_p_prev_coeff(&u1_previous,
            &u2_prev_on_mesh1, &gap_coeff, &n_coeff, N, -1.0);
         StressGridFunctionCoefficient stress_u1_curr_coeff(lambda1, mu1, &u1_current);
         FlatVectorCoefficient nalpha_vstress_u1_curr_coeff(stress_u1_curr_coeff, -alpha);
         StressGridFunctionCoefficient stress_u2_curr_coeff(lambda2, mu2, &u2_current);
         FlatVectorCoefficient nalpha_vstress_u2_curr_coeff(stress_u2_curr_coeff, -alpha);

         // --- Body 1 (Ω¹): elasticity + contact boundary terms ---

         // Step 1A: Set up the bilinear form a₁(⋅,⋅) on fespace1.
         ParBilinearForm *a1 = new ParBilinearForm(fespace1);
         a1->AddDomainIntegrator(new ElasticityIntegrator(alpha_coeff,lambda1,mu1));
         a1->AddBdrFaceIntegrator(
            new BoundaryProjectionIntegrator(reg_log_dp_curr_coeff,
            n_coeff), ess_bdr_z);
         a1->Assemble();

         // Step 2A: Set up the linear form b₁(⋅) on fespace1.
         ParLinearForm *b1 = new ParLinearForm(fespace1);
         b1->AddDomainIntegrator(new VectorDomainLFStrainIntegrator(nalpha_vstress_u1_curr_coeff));
         b1->AddBdrFaceIntegrator(
            new BoundaryProjectionLFIntegrator(reg_log_p_curr_coeff, &n_coeff),
            ess_bdr_z);
         b1->AddDomainIntegrator(new VectorDomainLFIntegrator(alpha_f1_coeff));
         b1->AddBdrFaceIntegrator(
            new BoundaryProjectionLFIntegrator(nreg_log_p_prev_coeff, &n_coeff),
            ess_bdr_z);
         b1->Assemble();

         // Step 3A: Form and solve the linear system for δu¹.
         HypreParMatrix A1;
         Vector B1, X1;
         a1->FormLinearSystem(ess_tdof_list, delta_u1, *b1, A1, X1, B1);

         HypreBoomerAMG *amg1 = new HypreBoomerAMG(A1);
         amg1->SetElasticityOptions(fespace1);
         amg1->SetPrintLevel(0);
         HyprePCG *pcg1 = new HyprePCG(A1);
         pcg1->SetTol(1e-12);
         pcg1->SetMaxIter(500);
         pcg1->SetPrintLevel(0);
         pcg1->SetPreconditioner(*amg1);
         pcg1->Mult(B1, X1);

         a1->RecoverFEMSolution(X1, *b1, delta_u1);
         u1_current += delta_u1;

         // --- Body 2 (Ω²): elasticity only (biased formulation) ---

         // Step 1B: Set up the bilinear form a₂(⋅,⋅) on fespace2.
         ParBilinearForm *a2 = new ParBilinearForm(fespace2);
         a2->AddDomainIntegrator(new ElasticityIntegrator(alpha_coeff,lambda2,mu2));
         a2->Assemble();

         // Step 2B: Set up the linear form b₂(⋅) on fespace2.
         ParLinearForm *b2 = new ParLinearForm(fespace2);
         b2->AddDomainIntegrator(new VectorDomainLFStrainIntegrator(nalpha_vstress_u2_curr_coeff));
         b2->AddDomainIntegrator(new VectorDomainLFIntegrator(alpha_f2_coeff));
         b2->Assemble();

         // Step 3B: Form and solve the linear system for δu².
         HypreParMatrix A2;
         Vector B2, X2;
         a2->FormLinearSystem(ess_tdof_list2, delta_u2, *b2, A2, X2, B2);

         HypreBoomerAMG *amg2 = new HypreBoomerAMG(A2);
         amg2->SetElasticityOptions(fespace2);
         amg2->SetPrintLevel(0);
         HyprePCG *pcg2 = new HyprePCG(A2);
         pcg2->SetTol(1e-12);
         pcg2->SetMaxIter(500);
         pcg2->SetPrintLevel(0);
         pcg2->SetPreconditioner(*amg2);
         pcg2->Mult(B2, X2);

         a2->RecoverFEMSolution(X2, *b2, delta_u2);
         u2_current += delta_u2;

         // Re-transfer u2 to mesh1 so body 1 sees the updated u2 ∘ Π
         TransferU2(u2_current, u2_curr_on_mesh1);

         // Step 7: Check for convergence.
         newton_error = delta_u1.ComputeL2Error(f2_coeff)
                      + delta_u2.ComputeL2Error(f2_coeff);

         if (myid == 0)
         {
            mfem::out << setw(4) << j << setw(14) << newton_error << std::endl;
         }

         // Step 8: Free used memory.
         delete amg1; delete pcg1; delete a1; delete b1;
         delete amg2; delete pcg2; delete a2; delete b2;

         if (newton_error < ntol)
         {
            break;
         }
      }

      // Step 9: Compute difference between current and previous solutions.
      iter_error = u1_current.ComputeL2Error(u1_previous_coeff)
                 + u2_current.ComputeL2Error(u2_previous_coeff);

      if (myid == 0)
      {
         mfem::out << k << setw(4) << " " << setw(13) << " " << setw(14)
                   << iter_error << setw(14) << std::endl;
      }

      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh1 << u1_current << std::flush;
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
      u1_previous = u1_current;
      u2_previous = u2_current;
   }

   // 11. Save the final solution in ParaView format.
   if (paraview_output)
   {
      paraview_dc1.SetCycle(1);
      paraview_dc1.SetTime((real_t)1);
      paraview_dc1.Save();

      paraview_dc2.SetCycle(1);
      paraview_dc2.SetTime((real_t)1);
      paraview_dc2.Save();
   }

   // 12. Free used memory.
   finder.FreeData();
   if (fec)
   {
      delete fespace1;
      delete fespace2;
      delete fec;
   }
   return 0;
}

void InitDisplacement(const Vector &x, Vector &u)
{
   u = 0.0;
   u(x.Size() - 1) = slab_g;
}

void ForceFunction(const Vector &x, Vector &f)
{
   f = 0.0;
   f(x.Size() - 1) = -force_g;
}

void Pi(const Vector &x, Vector &pi)
{
   pi = x;
   pi(x.Size() - 1) = slab_g;
}

void NFunctionCoefficient::Eval(Vector &N, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   const int dim = T.GetSpaceDim();

   // Get current point coordinates
   Vector x(dim);
   T.Transform(ip, x);

   // Evaluate Π at x
   Vector pi(dim);
   Pi->Eval(pi, T, ip);

   // Store Π(x) - x, (Π(x) - x) / |Π(x) - x|
   Vector diff(dim), diff_unit(dim);
   subtract(pi,x,diff);
   diff_unit = diff;
   diff_unit /= diff.Norml2();

   // Get normal n¹ at x
   Vector normal1(dim);
   T.SetIntPoint(&Geometries.GetCenter(T.GetGeometryType()));
   CalcOrtho(T.Jacobian(), normal1);
   normal1 /= normal1.Norml2();

   Vector w;
   if (!nt) { w = normal1; }
   else { nt->Eval(w, T, ip); }

   const real_t val = diff * w;
   if (val > 0)
   {
      N = diff_unit;
   }
   else if (val < 0)
   {
      N.Set(-1.0, diff_unit);
   }
   else
   {
      N = w;
   }
}

real_t GapFunctionCoefficient::Eval(ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   const int dim = T.GetSpaceDim();
   Vector x(dim), pi(dim), diff(dim), n_val(dim);

   T.Transform(ip, x);
   Pi->Eval(pi, T, ip);
   subtract(pi, x, diff);

   n->Eval(n_val, T, ip);
   return diff * n_val;
}

real_t RegLogPrimeCoefficient::RegLogPrime(const real_t a, const real_t M)
{
   if (a > M)
   {
      return log(M) + 1.0/M * (a - M);
   }
   else if (a < 1.0/M)
   {
      return -log(M) + M * (a - 1.0/M);
   }
   else
   {
      return log(a);
   }
}

real_t RegLogDoublePrimeCoefficient::RegLogDoublePrime(const real_t a,
                                                       const real_t M)
{
   if (a > M)
   {
      return 1.0 / M;
   }
   else if (a < 1.0/M)
   {
      return M;
   }
   else
   {
      return 1.0 / a;
   }
}
