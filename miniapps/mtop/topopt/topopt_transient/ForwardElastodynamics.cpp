// =============================================================================
// Forward Elastodynamics Solver for Semi-Infinite Wave Propagation
// =============================================================================
//
// This solver implements transient elastodynamics with absorbing boundary
// conditions for semi-infinite domain problems. The implementation uses
// a parallel finite element discretization with MFEM.
//
// PHYSICS:
//   - Second-order elastodynamics: M(ρ) ü + C u̇ + K(ρ) u = f(t)
//   - Design-dependent mass M(ρ) and stiffness K(ρ) via SIMP interpolation
//   - Volumetric damping (sponge layers) for wave absorption
//   - First-order Robin absorbing boundary conditions
//   - Time-dependent surface loading
//
// STATE VECTOR ORDERING:
//   State x = [u, v] where u = displacement, v = velocity
//   System becomes first-order: [u̇, v̇] = [v, M^{-1}(-K u - C v + f)]
//
// DESIGN DEPENDENCE:
//   - Mass: M(ρ) = ∫ r_m(ρ̃) ρ₀ φᵢ·φⱼ dx
//   - Stiffness: K(ρ) = ∫ r_k(ρ̃) 𝓒₀ : ε(φᵢ) : ε(φⱼ) dx
//   - SIMP: r(ρ̃) = r_min + ρ̃^p (r_max - r_min)
//
// COMPILE:
//   make ForwardElastodynamics -j8
//
// RUN:
//   srun -n <nprocs> ./ForwardElastodynamics [options]
//   Options: -r <refine> -o <order> -tf <final_time> -dt <timestep> -pv
//
// LOCAL SWEET SPOT:
// mpirun -np 8 ./ForwardElastodynamics -r 0 -o 2 -tf 0.3 -dt 0.00005 -no-pv (IDEAL)
// mpirun -np 8 ./ForwardElastodynamics -r 0 -o 1 -tf 0.3 -dt 0.0001 -no-pv (CHEAP)
// =============================================================================

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

// =============================================================================
// SIMP MATERIAL INTERPOLATION
// =============================================================================
// Computes r(ρ̃) = r_min + ρ̃^p (r_max - r_min)
class SIMPCoefficient : public Coefficient
{
private:
   GridFunction *rho_filter;  // Filtered density ρ̃
   real_t r_min, r_max;
   real_t exponent;

public:
   SIMPCoefficient(GridFunction *rho_filt, real_t rmin, real_t rmax, real_t p)
      : rho_filter(rho_filt), r_min(rmin), r_max(rmax), exponent(p) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);  // Clamp to [0,1]
      real_t rho_pow = std::pow(rho_val, exponent);
      return r_min + rho_pow * (r_max - r_min);
   }
};

// =============================================================================
// GAUSSIAN DESIGN DISTRIBUTION
// =============================================================================
// Creates a 2D Gaussian design field: ρ̃(x,y) = ρ_min + (1-ρ_min) * exp(-r²/(2σ²))
// where r² = (x-x_c)² + (y-y_c)²
class GaussianDesignCoefficient : public Coefficient
{
private:
   real_t x_center, y_center;  // Center of Gaussian
   real_t sigma_x, sigma_y;    // Standard deviations
   real_t rho_min;             // Minimum density (at edges)
   real_t rho_max;             // Maximum density (at center)

public:
   GaussianDesignCoefficient(real_t xc, real_t yc, real_t sx, real_t sy,
                             real_t rmin = 0.3, real_t rmax = 1.0)
      : x_center(xc), y_center(yc), sigma_x(sx), sigma_y(sy),
        rho_min(rmin), rho_max(rmax) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);

      // Distance from center (normalized by sigma)
      real_t dx = (x(0) - x_center) / sigma_x;
      real_t dy = (x(1) - y_center) / sigma_y;
      real_t r_squared = dx * dx + dy * dy;

      // Gaussian: ρ = ρ_min + (ρ_max - ρ_min) * exp(-r²/2)
      real_t gaussian = std::exp(-0.5 * r_squared);
      real_t rho = rho_min + (rho_max - rho_min) * gaussian;

      return std::min(std::max(rho, 0.0), 1.0);  // Clamp to [0,1]
   }
};

// =============================================================================
// DAMPING PROFILE FOR SPONGE LAYERS
// =============================================================================
// Computes the integrated damping function φ(x) used to define spatially
// varying damping coefficients in absorbing boundary layers
class DampingProfile : public Coefficient
{
private:
   real_t thickness;
   real_t x_max, y_max;
   real_t phi_max;

public:
   DampingProfile(real_t thick, real_t xmax, real_t ymax)
      : thickness(thick), x_max(xmax), y_max(ymax)
   {
      phi_max = thickness * thickness / 2.0;
   }

   real_t GetPhiMax() const { return phi_max; }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);

      real_t phi = 0.0;
      real_t s = 0.0;

      // Left boundary layer
      if (x(0) < thickness)
      {
         s = thickness - x(0);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      // Right boundary layer
      if (x(0) > x_max - thickness)
      {
         s = x(0) - (x_max - thickness);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      // Bottom boundary layer
      if (x(1) < thickness)
      {
         s = thickness - x(1);
         real_t phi_local = thickness * s - 0.5 * s * s;
         phi = std::max(phi, phi_local);
      }

      return phi;
   }
};

// =============================================================================
// SPATIALLY-VARYING DAMPING COEFFICIENT
// =============================================================================
// Computes γ(x) = ρ * γ_max * F(η(x)) where η = φ(x)/φ_max
// F(η) is a smooth ramp function that transitions from 0 to 1
class SpatialDampingCoefficient : public Coefficient
{
private:
   DampingProfile *phi_coef;
   real_t phi_max;
   real_t gamma_max;
   real_t rho;
   real_t beta;  // Ramp function parameter
   int m;        // Polynomial order

public:
   SpatialDampingCoefficient(DampingProfile *phi, real_t gmax,
                              real_t density, real_t b = 2.0, int mp = 2)
      : phi_coef(phi), gamma_max(gmax), rho(density), beta(b), m(mp)
   {
      phi_max = phi_coef->GetPhiMax();
   }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t phi_val = phi_coef->Eval(T, ip);

      if (phi_max < 1e-12) return 0.0;

      real_t eta = phi_val / phi_max;
      eta = std::min(std::max(eta, 0.0), 1.0);

      real_t eta_pow = std::pow(eta, m);
      real_t F_eta = (std::exp(beta * eta_pow) - 1.0) / (std::exp(beta) - 1.0);

      return rho * gamma_max * F_eta;
   }
};

// =============================================================================
// ELASTODYNAMICS TIME-DEPENDENT OPERATOR
// =============================================================================
// Implements the right-hand side for the first-order system:
//   [u̇]   [                        v                          ]
//   [v̇] = [ M^{-1}(-K u - C_vol v - C_abs v + f(t)) ]
//
// where:
//   M       = mass matrix
//   K       = stiffness matrix
//   C_vol   = volumetric damping matrix
//   C_abs   = absorbing boundary condition matrix
//   f(t)    = time-dependent applied load
class ElastodynamicsOperator : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &fespace;
   ParBilinearForm *M, *K, *C_vol, *C_abs;
   HypreParMatrix *Mmat, *Kmat, *Cvol_mat, *Cabs_mat;
   HypreBoomerAMG *M_prec;
   CGSolver *M_solver;

   mutable ParGridFunction u_gf;  // For visualization
   mutable ParGridFunction v_gf;  // For visualization

   Array<int> ess_tdof_list;      // Essential DOFs (empty for semi-infinite)
   Array<int> block_true_offsets; // Block structure [u, v]
   int true_size;

   mutable Vector res;            // Work vector for residual
   mutable Vector tmp;            // Work vector for products

   mutable Vector load_params;    // [duration, amplitude]
   Array<int> load_bdr_markers;   // Boundary markers for loading

public:
   ElastodynamicsOperator(
      ParFiniteElementSpace &f,
      Coefficient &mass_coef,        // Design-dependent mass coefficient
      Coefficient &lambda_coef,      // Design-dependent Lamé λ
      Coefficient &mu_coef,          // Design-dependent Lamé μ
      real_t amplitude, real_t duration,
      SpatialDampingCoefficient *gamma_coef,
      real_t impedance,
      Array<int> &exterior_bdr_attr,
      Array<int> &ess_bdr_attr);

   void SetTime(real_t t) override { TimeDependentOperator::SetTime(t); }

   virtual void Mult(const Vector &x, Vector &y) const override;

   const Array<int>& GetEssentialTrueDofs() const { return ess_tdof_list; }

   ParGridFunction& GetDisplacement() { return u_gf; }
   ParGridFunction& GetVelocity() { return v_gf; }

   Array<int>& GetBlockOffsets() { return block_true_offsets; }

   virtual ~ElastodynamicsOperator();
};

ElastodynamicsOperator::ElastodynamicsOperator(
   ParFiniteElementSpace &f,
   Coefficient &mass_coef,
   Coefficient &lambda_coef,
   Coefficient &mu_coef,
   real_t amplitude, real_t duration,
   SpatialDampingCoefficient *gamma_coef,
   real_t impedance,
   Array<int> &exterior_bdr_attr,
   Array<int> &ess_bdr_attr)
   : TimeDependentOperator(2 * f.GetTrueVSize(), 0.0),
     fespace(f),
     u_gf(&fespace),
     v_gf(&fespace),
     true_size(f.GetTrueVSize()),
     res(true_size),
     tmp(true_size),
     load_params(2)
{
   int myid = Mpi::WorldRank();

   // Block structure: [displacement, velocity]
   block_true_offsets.SetSize(3);
   block_true_offsets[0] = 0;
   block_true_offsets[1] = true_size;
   block_true_offsets[2] = 2 * true_size;

   // Initialize grid functions and work vectors
   u_gf = 0.0;
   v_gf = 0.0;
   res = 0.0;
   tmp = 0.0;

   // Store loading parameters
   load_params(0) = duration;
   load_params(1) = amplitude;

   // Get essential DOFs (typically empty for semi-infinite domains)
   fespace.GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);

   if (myid == 0)
   {
      cout << "\n=== Elastodynamics Operator ===" << endl;
      cout << "DOFs per field: " << true_size << endl;
      cout << "Essential DOFs: " << ess_tdof_list.Size() << endl;
   }

   // Assemble design-dependent mass matrix: M(ρ)
   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new VectorMassIntegrator(mass_coef));
   M->Assemble();
   M->Finalize();
   Mmat = M->ParallelAssemble();

   // Assemble design-dependent stiffness matrix: K(ρ)
   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
   K->Assemble();
   K->Finalize();
   Kmat = K->ParallelAssemble();

   // Assemble volumetric damping matrix
   C_vol = new ParBilinearForm(&fespace);
   C_vol->AddDomainIntegrator(new VectorMassIntegrator(*gamma_coef));
   C_vol->Assemble();
   C_vol->Finalize();
   Cvol_mat = C_vol->ParallelAssemble();

   // Assemble absorbing boundary condition matrix
   C_abs = new ParBilinearForm(&fespace);
   ConstantCoefficient impedance_coef(impedance);
   C_abs->AddBoundaryIntegrator(new VectorMassIntegrator(impedance_coef), exterior_bdr_attr);
   C_abs->Assemble();
   C_abs->Finalize();
   Cabs_mat = C_abs->ParallelAssemble();

   // NNZ() is collective (global reduction over all ranks); every rank must call it
   // together. Compute on all ranks, then print on rank 0 only.
   HYPRE_BigInt m_nnz = Mmat->NNZ(), k_nnz = Kmat->NNZ(),
                cvol_nnz = Cvol_mat->NNZ(), cabs_nnz = Cabs_mat->NNZ();
   if (myid == 0)
   {
      cout << "Matrix assembly complete:" << endl;
      cout << "  Mass NNZ:     " << m_nnz << endl;
      cout << "  Stiffness NNZ: " << k_nnz << endl;
      cout << "  Damping NNZ:   " << cvol_nnz << endl;
      cout << "  ABC NNZ:       " << cabs_nnz << endl;
   }

   // Set up mass matrix solver (CG with AMG preconditioner)
   M_prec = new HypreBoomerAMG(*Mmat);
   M_prec->SetPrintLevel(0);

   M_solver = new CGSolver(fespace.GetComm());
   M_solver->SetPreconditioner(*M_prec);
   M_solver->SetOperator(*Mmat);
   M_solver->SetRelTol(1e-12);
   M_solver->SetAbsTol(0.0);
   M_solver->SetMaxIter(100);
   M_solver->SetPrintLevel(0);

   // Set up boundary markers for time-dependent loading
   ParMesh *pmesh = fespace.GetParMesh();
   int max_bdr_attr = pmesh->bdr_attributes.Max();
   load_bdr_markers.SetSize(max_bdr_attr);
   load_bdr_markers = 0;

   // Mark load boundaries (attributes 21-26 for Gmsh mesh)
   for (int attr = 21; attr <= 26; attr++)
   {
      if (attr <= max_bdr_attr)
      {
         load_bdr_markers[attr - 1] = 1;
      }
   }

   if (myid == 0)
   {
      cout << "\nTime-dependent loading:" << endl;
      cout << "  Type: Smooth Gaussian pulse" << endl;
      cout << "  Peak amplitude: " << amplitude << endl;
      cout << "  Duration: " << duration << " s" << endl;
      cout << "====================================\n" << endl;
   }
}

void ElastodynamicsOperator::Mult(const Vector &x, Vector &y) const
{
   real_t time = this->GetTime();

   // Initialize output vector (prevents uninitialized memory errors)
   y = 0.0;

   // Extract state blocks: x = [u, v], y = [u̇, v̇]
   BlockVector bx(const_cast<Vector&>(x), block_true_offsets);
   BlockVector by(y, block_true_offsets);

   // Create views into state blocks
   Vector u_true(bx.GetBlock(0).GetData(), true_size);
   Vector v_true(bx.GetBlock(1).GetData(), true_size);

   // First equation: u̇ = v
   by.GetBlock(0) = v_true;

   // Second equation: v̇ = M^{-1}(-K u - C_vol v - C_abs v + f(t))
   res = 0.0;

   // Elastic restoring force: -K u
   Kmat->Mult(u_true, tmp);
   res.Add(-1.0, tmp);

   // Volumetric damping: -C_vol v
   Cvol_mat->Mult(v_true, tmp);
   res.Add(-1.0, tmp);

   // Absorbing boundary damping: -C_abs v
   Cabs_mat->Mult(v_true, tmp);
   res.Add(-1.0, tmp);

   // Time-dependent applied load: f(t)
   real_t duration = load_params(0);
   real_t amplitude = load_params(1);

   // Smooth Gaussian pulse: g(t) = A * exp(-(t - t_c)^2 / (2 σ^2))
   real_t t_center = duration / 2.0;
   real_t sigma = duration / 4.0;
   real_t t_diff = time - t_center;
   real_t gauss_factor = exp(-t_diff * t_diff / (2.0 * sigma * sigma));
   real_t current_amplitude = amplitude * gauss_factor;

   // Define load coefficient
   class GaussianLoad : public VectorCoefficient
   {
   private:
      real_t amp;
   public:
      GaussianLoad(int dim, real_t a) : VectorCoefficient(dim), amp(a) {}
      void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override
      {
         V.SetSize(vdim);
         V = 0.0;
         V(1) = -amp;  // Downward load
      }
   };

   GaussianLoad load_coef(fespace.GetParMesh()->SpaceDimension(), current_amplitude);

   // Assemble boundary load vector
   ParLinearForm load_form(&fespace);
   load_form.AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(load_coef),
      const_cast<Array<int>&>(load_bdr_markers));
   load_form.Assemble();

   Vector load_vec(true_size);
   load_form.ParallelAssemble(load_vec);

   res.Add(1.0, load_vec);

   // Solve M v̇ = res
   // Check if RHS is effectively zero (avoid unnecessary solver calls)
   real_t local_res_norm_sq = res * res;
   real_t global_res_norm_sq;
   MPI_Allreduce(&local_res_norm_sq, &global_res_norm_sq, 1,
                 MPI_DOUBLE, MPI_SUM, fespace.GetComm());
   real_t global_res_norm = sqrt(global_res_norm_sq);

   if (global_res_norm < 1e-14)
   {
      by.GetBlock(1) = 0.0;
   }
   else
   {
      by.GetBlock(1) = 0.0;  // Initialize output
      M_solver->Mult(res, by.GetBlock(1));
   }
}

ElastodynamicsOperator::~ElastodynamicsOperator()
{
   delete M_solver;
   delete M_prec;
   delete Cabs_mat;
   delete Cvol_mat;
   delete Kmat;
   delete Mmat;
   delete C_abs;
   delete C_vol;
   delete K;
   delete M;
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();
   int myid = Mpi::WorldRank();

   Device device("cpu");

   // Command-line options
   int ref_levels = 3;
   int order = 2;
   real_t t_final = 0.3;
   real_t dt = 0.0005;
   int vis_steps = -1;
   bool paraview = true;
   bool enable_damping = true;
   bool gaussian_design = true;  // Use Gaussian vs. uniform design
   const char *mesh_file = "lamb-problem-damping-mesh-triangs.msh";
   real_t target_attenuation = 1e-4;
   real_t beta = 2.0;
   int m = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement level");
   args.AddOption(&order, "-o", "--order", "FE order");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time");
   args.AddOption(&dt, "-dt", "--time-step", "Time step");
   args.AddOption(&vis_steps, "-vs", "--vis-steps", "Vis frequency (-1: auto)");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview", "ParaView");
   args.AddOption(&enable_damping, "-damp", "--damping", "-no-damp", "--no-damping", "Damping");
   args.AddOption(&gaussian_design, "-gauss", "--gaussian-design",
                  "-uniform", "--uniform-design", "Gaussian vs uniform design");
   args.AddOption(&mesh_file, "-mesh", "--mesh-file", "Mesh file");
   args.AddOption(&target_attenuation, "-eps", "--epsilon", "Target attenuation");
   args.AddOption(&beta, "-beta", "--beta", "Damping parameter");
   args.AddOption(&m, "-m", "--m-power", "Damping exponent");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Auto-calculate visualization frequency
   if (vis_steps < 0)
   {
      int total_steps = static_cast<int>(t_final / dt);
      int max_output_files = 50;
      vis_steps = std::max(1, total_steps / max_output_files);
      if (myid == 0)
      {
         cout << "\nOutput frequency: every " << vis_steps << " steps" << endl;
      }
   }

   // Domain parameters
   real_t x_max = 1.5;
   real_t y_max = 0.75;

   // Load mesh
   if (myid == 0)
   {
      cout << "\n=== Loading Mesh ===" << endl;
      cout << "File: " << mesh_file << endl;
   }

   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "Error: Cannot open mesh file '" << mesh_file << "'" << endl;
      }
      return 1;
   }

   Mesh mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   if (myid == 0)
   {
      cout << "Refinement levels: " << ref_levels << endl;
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec, dim);

   HYPRE_BigInt total_dofs = fespace.GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "\n=== Problem Setup ===" << endl;
      cout << "Domain: " << x_max << " × " << y_max << " m" << endl;
      cout << "DOFs per field: " << total_dofs << endl;
      cout << "Total state size: " << 2*total_dofs << endl;
   }

   // ==========================================================================
   // FILTERED DESIGN FIELD: initialize ρ̃ in H1.
   // This forward-only driver has no raw control ρ; this is the density field
   // that enters the mass and stiffness coefficients.
   // ==========================================================================
   H1_FECollection design_fec(order, dim);
   ParFiniteElementSpace design_fes(&pmesh, &design_fec);
   ParGridFunction rho_filter(&design_fes);

   // GlobalTrueVSize() is collective: every rank must call it together. Calling it
   // only inside an if(myid==0) block (as the prints below do) deadlocks all other
   // ranks. Compute it here on all ranks, then just print the stored value.
   HYPRE_BigInt design_dofs = design_fes.GlobalTrueVSize();

   // Damping layer thickness (used for both sponge layers and design centering)
   real_t damping_thickness = 0.25;

   if (gaussian_design)
   {
      // -----------------------------------------------------------------------
      // Gaussian design: ρ̃(x,y) = ρ_min + (1-ρ_min) * exp(-r²/(2σ²))
      // -----------------------------------------------------------------------
      // Compute center of non-damped domain
      real_t x_center = x_max / 2.0;
      real_t y_center = y_max / 2.0;

      // Gaussian standard deviations (controls spread)
      // Use 1/3 of non-damped domain size so most mass is in center
      real_t sigma_x = (x_max - 2.0 * damping_thickness) / 3.0;
      real_t sigma_y = (y_max - damping_thickness) / 3.0;  // No top damping

      // Density range
      real_t rho_min_design = 0.3;   // Minimum density at edges
      real_t rho_max_design = 1.0;   // Maximum density at center

      // Create and project Gaussian design
      GaussianDesignCoefficient gaussian_coef(x_center, y_center, sigma_x, sigma_y,
                                               rho_min_design, rho_max_design);
      rho_filter.ProjectCoefficient(gaussian_coef);

      if (myid == 0)
      {
         cout << "\n=== Design Field: Gaussian Distribution ===" << endl;
         cout << "Center: (" << x_center << ", " << y_center << ")" << endl;
         cout << "Std dev: σ_x = " << sigma_x << ", σ_y = " << sigma_y << endl;
         cout << "Density range: [" << rho_min_design << ", " << rho_max_design << "]" << endl;
         cout << "Design DOFs: " << design_dofs << endl;
         cout << "Non-damped domain: [" << damping_thickness << ", "
              << x_max - damping_thickness << "] × [" << damping_thickness << ", "
              << y_max << "]" << endl;
      }
   }
   else
   {
      // -----------------------------------------------------------------------
      // Uniform design: ρ̃ = 1.0 (fully solid)
      // -----------------------------------------------------------------------
      rho_filter = 1.0;

      if (myid == 0)
      {
         cout << "\n=== Design Field: Uniform (Fully Solid) ===" << endl;
         cout << "Density: 1.0" << endl;
         cout << "Design DOFs: " << design_dofs << endl;
      }
   }

   // ==========================================================================
   // MATERIAL PROPERTIES with SIMP interpolation
   // ==========================================================================
   // Base material properties (when ρ̃ = 1)
   real_t rho_0 = 1.0;        // Base mass density
   real_t mu_0 = 1.0;         // Base shear modulus
   real_t lambda_0 = 2.0;     // Base Lamé parameter

   // SIMP interpolation parameters
   real_t r_min = 1e-6;       // Void property ratio
   real_t r_max = 1.0;        // Solid property ratio
   real_t simp_exponent = 3.0;

   // Create base constant coefficients
   ConstantCoefficient rho_0_coef(rho_0);
   ConstantCoefficient lambda_0_coef(lambda_0);
   ConstantCoefficient mu_0_coef(mu_0);

   // Create SIMP coefficients
   SIMPCoefficient simp_mass(&rho_filter, r_min, r_max, simp_exponent);
   SIMPCoefficient simp_stiff(&rho_filter, r_min, r_max, simp_exponent);

   // Design-dependent material coefficients
   // Mass: ρ(ρ̃) = r_m(ρ̃) * ρ₀
   ProductCoefficient mass_coef(simp_mass, rho_0_coef);

   // Stiffness: λ(ρ̃) = r_k(ρ̃) * λ₀, μ(ρ̃) = r_k(ρ̃) * μ₀
   ProductCoefficient lambda_coef(simp_stiff, lambda_0_coef);
   ProductCoefficient mu_coef(simp_stiff, mu_0_coef);

   // Wave speeds (for reference, using base material)
   real_t c_s = sqrt(mu_0 / rho_0);
   real_t c_p = sqrt((lambda_0 + 2*mu_0) / rho_0);

   if (myid == 0)
   {
      cout << "\n=== Material Properties ===" << endl;
      cout << "Base density ρ₀: " << rho_0 << endl;
      cout << "Base shear modulus μ₀: " << mu_0 << endl;
      cout << "Base Lamé λ₀: " << lambda_0 << endl;
      cout << "SIMP exponent: " << simp_exponent << endl;
      cout << "Property range: [" << r_min << ", " << r_max << "]" << endl;
      cout << "S-wave speed c_s: " << c_s << " m/s" << endl;
      cout << "P-wave speed c_p: " << c_p << " m/s" << endl;
   }

   // Damping configuration (damping_thickness already declared above)
   DampingProfile phi_profile(damping_thickness, x_max, y_max);
   real_t gamma_max = 0.0;

   if (enable_damping)
   {
      real_t I_F = 0.2136;  // For β=2, m=2
      if (abs(beta - 2.0) > 0.01 || m != 2)
      {
         I_F = 0.5 / beta;
      }
      gamma_max = (2.0 * c_p / I_F) * log(1.0 / target_attenuation);

      if (myid == 0)
      {
         cout << "\n=== Damping Configuration ===" << endl;
         cout << "γ_max: " << gamma_max << " s^{-1}" << endl;
         cout << "Target attenuation: " << target_attenuation << endl;
      }
   }

   SpatialDampingCoefficient gamma_coef(&phi_profile, gamma_max, rho_0, beta, m);

   // Loading parameters
   real_t pulse_duration = 0.005;
   real_t pulse_amplitude = 30.0;

   // Boundary conditions
   Array<int> exterior_bdr_attr(pmesh.bdr_attributes.Max());
   exterior_bdr_attr = 0;

   if (enable_damping)
   {
      if (pmesh.bdr_attributes.Max() >= 10) exterior_bdr_attr[9] = 1;
      if (pmesh.bdr_attributes.Max() >= 11) exterior_bdr_attr[10] = 1;
      if (pmesh.bdr_attributes.Max() >= 12) exterior_bdr_attr[11] = 1;
   }

   Array<int> empty_bdr_attr(pmesh.bdr_attributes.Max());
   empty_bdr_attr = 0;

   real_t impedance = enable_damping ? (rho_0 * c_p) : 0.0;

   // Create design-dependent elastodynamics operator
   ElastodynamicsOperator oper(
      fespace, mass_coef, lambda_coef, mu_coef,
      pulse_amplitude, pulse_duration,
      &gamma_coef, impedance, exterior_bdr_attr, empty_bdr_attr);

   // Initial conditions
   BlockVector state(oper.GetBlockOffsets());
   state = 0.0;

   // Time integrator
   RK4Solver ode_solver;
   ode_solver.Init(oper);

   // ParaView output
   ParaViewDataCollection *paraview_dc = nullptr;
   if (paraview)
   {
      // Create separate output directories for different design types
      std::string design_suffix = gaussian_design ? "_Gaussian" : "_Uniform";
      std::string collection_name = enable_damping
                                    ? "Elastodynamics_Damped" + design_suffix
                                    : "Elastodynamics" + design_suffix;
      std::string prefix_path;
      if (enable_damping)
      {
         prefix_path = gaussian_design ? "ParaView/Gaussian_Damped" : "ParaView/Uniform_Damped";
      }
      else
      {
         prefix_path = gaussian_design ? "ParaView/Gaussian" : "ParaView/Uniform";
      }

      paraview_dc = new ParaViewDataCollection(collection_name.c_str(), &pmesh);
      paraview_dc->SetPrefixPath(prefix_path.c_str());
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->RegisterField("displacement", &(oper.GetDisplacement()));
      paraview_dc->RegisterField("velocity", &(oper.GetVelocity()));
      paraview_dc->RegisterField("density", &rho_filter);  // Design field
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->Save();
   }

   // Time integration loop
   real_t t = 0.0;
   bool last_step = false;

   if (myid == 0) { cout << "\nTime integration:" << endl; }

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2) { last_step = true; }

      oper.SetTime(t + dt);
      ode_solver.Step(state, t, dt);

      BlockVector bstate(state, oper.GetBlockOffsets());

      if (ti % vis_steps == 0 || last_step)
      {
         real_t local_u_norm = bstate.GetBlock(0).Normlinf();
         real_t local_v_norm = bstate.GetBlock(1).Normlinf();

         real_t global_u_norm, global_v_norm;
         MPI_Allreduce(&local_u_norm, &global_u_norm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         MPI_Allreduce(&local_v_norm, &global_v_norm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

         if (myid == 0)
         {
            cout << "step " << ti << ", t=" << t
                 << ", |u|∞=" << global_u_norm
                 << ", |v|∞=" << global_v_norm << endl;
         }

         if (paraview)
         {
            oper.GetDisplacement().SetFromTrueDofs(bstate.GetBlock(0));
            oper.GetVelocity().SetFromTrueDofs(bstate.GetBlock(1));

            paraview_dc->SetCycle(ti);
            paraview_dc->SetTime(t);
            paraview_dc->Save();
         }
      }
   }

   if (paraview) { delete paraview_dc; }

   if (myid == 0)
   {
      cout << "\n=== Simulation Complete ===" << endl;
   }

   return 0;
}
