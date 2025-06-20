//                             MFEM Signorini MMS
//
// Compile with: make signorinip_mms
//
// Sample runs:  mpirun -np 4 signorinip_mms
//
// Description:  This program solves the Signorini problem using MFEM.
//               The problem is defined on a solid with a Dirichlet
//               boundary condition on the bottom face and a traction
//               boundary (Γₜ) condition on the top face. The traction
//               boundary condition is defined through a unit vector field
//               ñ. We aim to (iteravely) solve
//
//               (σ(uᵏ), ε(v)) = (f, v)                      for all v ∈ V
//               uᵏ · ñ = φ₁ + (uᵏ⁻¹ · ñ - φ₁) exp(αₖ (σ(uᵏ⁻¹)n · ñ)) on Γₜ
//
//               where σ is the stress tensor, ε is the strain tensor,
//               f is the body force, v is the test function, uᵏ is the
//               displacement at iteration k,
//               ϕ₁ is the prescribed displacement, αₖ is a step-size
//               parameter, and n is the normal vector to the boundary.

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

void ManufacturedSolution(const Vector &x, Vector &u);
void InitDisplacement(const Vector &x, Vector &u);
real_t GapFunction(const Vector &x);
void ForceFunction(const Vector &x, Vector &f);
void ComputeStress(const DenseMatrix &grad_u, const real_t lambda,
                   const real_t mu, DenseMatrix &sigma);

/**
 * @brief Implements the contact express boundary condition for the Signorini
 *        problem. The vector n_tilde representing ñ is assumed to be equal to
 *        (0,...,0,-1).
 *
 * @param dim Spatial dimension
 * @param u_prev Previous displacement vector
 * @param n_tilde Vector field
 * @param lambda First Lamé parameter
 * @param mu Second Lamé parameter
 * @param alpha Step-size parameter
 */
class TractionBoundary : public VectorCoefficient
{
private:
   GridFunction *u_prev;
   Vector n_tilde;
   real_t lambda, mu, alpha;

public:
   TractionBoundary(int _dim, GridFunction *_u_prev, Vector _n_tilde,
                    real_t _lambda, real_t _mu, real_t _alpha)
      : VectorCoefficient(_dim), u_prev(_u_prev), n_tilde(_n_tilde), lambda(_lambda),
        mu(_mu), alpha(_alpha) {}

   virtual void Eval(Vector &u, ElementTransformation &T,
                     const IntegrationPoint &ip) override
   {
      ParGridFunction *par_u_prev = dynamic_cast<ParGridFunction*>(u_prev);
      const int dim = T.GetSpaceDim();

      // Get current point coordinates
      Vector x(dim);
      T.Transform(ip, x);

      // Get value and Jacobian of previous solution
      Vector u_prev_val(dim);
      DenseMatrix grad_u_prev(dim,dim);
      if (par_u_prev)
      {
         par_u_prev->GetVectorValue(T, ip, u_prev_val);
         par_u_prev->GetVectorGradient(T, grad_u_prev);
      }
      else
      {
         u_prev->GetVectorValue(T, ip, u_prev_val);
         u_prev->GetVectorGradient(T, grad_u_prev);
      }
      // Evaluate the stress tensor σ(uᵏ⁻¹)
      DenseMatrix sigma(dim,dim);
      ComputeStress(grad_u_prev, lambda, mu, sigma);

      // Compute normal vector n
      Vector n(dim);
      CalcOrtho(T.Jacobian(), n);
      n /= n.Norml2();

      // Compute pressure σ(uᵏ⁻¹)n · ñ
      Vector sigma_n(dim);
      sigma.Mult(n, sigma_n);
      const real_t pressure = sigma_n * n_tilde;

      // Evaluate the gap function φ₁
      const real_t phi_1 = GapFunction(x);

      // Set the boundary condition
      // uᵏ · ñ = φ₁ + (uᵏ⁻¹ · ñ - φ₁) exp(αₖ (σ(uᵏ⁻¹)n · ñ))
      u.SetSize(dim);
      u = u_prev_val;
      u(dim-1) = phi_1 + (u_prev_val * n_tilde - phi_1) * exp(alpha * pressure);
      u(dim-1) /= n_tilde(dim-1);
   }
};

real_t lambda = 1.0;
real_t mu = 1.0;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char* mesh_file = "../data/ref-cube.mesh";
   int order = 1;
   real_t alpha = 1.0;
   int ref_levels = 0;
   int max_iterations = 7;
   real_t itol = 1e-6;
   bool reorder_space = false;
   bool visualization = false;
   bool paraview_output = false;
   bool logger = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha parameter for boundary condition.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lamé's first parameter.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lamé's second parameter.");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of uniform mesh refinements.");
   args.AddOption(&max_iterations, "-i", "--iterations",
                  "Maximum number of iterations.");
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
   args.AddOption(&logger, "-l", "--logger", "-no-log", "--no-logger",
                  "Enable or disable logging.");
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
      std::cerr << "Invalid Lamé parameters." << std::endl;
      return 3;
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
   int curvature_order = max(order, 2);
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

   // 10. Determine the list of true (i.e. parallel conforming) essential
   //     boundary dofs.
   Array<int> ess_bdr_contact(pmesh.bdr_attributes.Max());
   Array<int> ess_bdr_sym_x(pmesh.bdr_attributes.Max());
   Array<int> ess_bdr_sym_y(pmesh.bdr_attributes.Max());
   ess_bdr_contact = 0; ess_bdr_contact[0] = 1;
   ess_bdr_sym_x = 0; ess_bdr_sym_x[2] = 1; ess_bdr_sym_x[4] = 1;
   ess_bdr_sym_y = 0; ess_bdr_sym_y[1] = 1; ess_bdr_sym_y[3] = 1;

   Array<int> ess_tdof_list_contact, ess_tdof_list_sym_x, ess_tdof_list_sym_y;
   fespace->GetEssentialTrueDofs(ess_bdr_sym_x, ess_tdof_list_sym_x, 0);
   fespace->GetEssentialTrueDofs(ess_bdr_sym_y, ess_tdof_list_sym_y, 1);
   fespace->GetEssentialTrueDofs(ess_bdr_contact, ess_tdof_list_contact, 2);

   Array<int> ess_tdof_list_all;
   ess_tdof_list_all.Append(ess_tdof_list_contact);
   ess_tdof_list_all.Append(ess_tdof_list_sym_x);
   ess_tdof_list_all.Append(ess_tdof_list_sym_y);

   // 11. Define coefficients for later.
   VectorFunctionCoefficient u_exact_coeff(dim, ManufacturedSolution);
   VectorFunctionCoefficient f_coeff(dim, ForceFunction);
   Vector n_tilde(dim);
   n_tilde = 0.0;
   n_tilde(dim-1) = -1.0;
   if (n_tilde.Norml2() != 1.0)
   {
      std::cerr << "Vector field n_tilde is not normalized." << std::endl
                << "n_tilde norm: " << n_tilde.Norml2() << std::endl;
      return 3;
   }

   // 8. Set up the parallel linear form b(⋅) which corresponds to the
   //    right-hand side of the FEM linear system.
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));
   b->Assemble();

   // 9. Define the solution vector u as a parallel finite element grid
   //    function corresponding to fespace. Initialize u with initial guess of
   //    u(x,y,z) = (0,0,-0.1z), which satisfies the boundary conditions.
   ParGridFunction u_previous(fespace);
   ParGridFunction u_current(fespace);
   VectorGridFunctionCoefficient u_previous_coeff(&u_previous);

   VectorFunctionCoefficient init_u(dim, InitDisplacement);
   u_previous.ProjectCoefficient(init_u);

   // 10. Set up the bilinear form a(⋅,⋅) on the finite element space
   //     corresponding to the linear elasticity integrator with coefficients
   //     lambda and mu.
   ConstantCoefficient one(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(one,lambda,mu));
   a->Assemble();

   // 11. Set up GLVis visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 12. Initialize ParaView output.
   ParaViewDataCollection paraview_dc("signorini_mms", &pmesh);
   if (paraview_output)
   {
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("displacement",&u_previous);
      paraview_dc.Save();
   }

   real_t iter_error;
   real_t l2_error;

   if (myid == 0)
   {
      mfem::out << "\nk" << setw(14) << "iter_error" << setw(14) << "l2_error"
                << std::endl;
      mfem::out << "-----------------------------" << std::endl;
   }

   // 13. Iterate:
   for (int k = 1; k <= max_iterations; k++)
   {
      // Step 1: Reassemble the linear form b(⋅).
      b->Assemble();

      // Step 2: Create the boundary condition coefficient using previous solution.
      TractionBoundary trac_coeff(dim, &u_previous, n_tilde, lambda, mu, alpha);
      u_current.ProjectBdrCoefficient(trac_coeff, ess_bdr_contact);

      // Step 3: Form the linear system A X = B. This includes eliminating boundary
      // conditions, applying AMR constraints, and other transformations.
      HypreParMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, u_current, *b, A, X, B);

      // Step 4: Define and apply a parallel PCG solver for A X = B with the BoomerAMG
      // preconditioner from hypre.
      HypreBoomerAMG *amg = new HypreBoomerAMG(A);
      amg->SetElasticityOptions(fespace);
      amg->SetPrintLevel(0);
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(500);
      pcg->SetPrintLevel(0);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);

      // Free used memory.
      delete amg;
      delete pcg;

      // Step 5: Recover the solution.
      a->RecoverFEMSolution(X, *b, u_current);

      // Step 6: Compute difference between current and previous solutions.
      iter_error = u_current.ComputeL2Error(u_previous_coeff);
      l2_error = u_current.ComputeL2Error(u_exact_coeff);

      if (myid == 0)
      {
         mfem::out << k << setw(14) << iter_error << setw(14) << l2_error
                   << std::endl;
      }

      // Step 7: Send the above data by socket to a GLVis server. Use the "n"
      // and "b" keys in GLVis to visualize the displacements.
      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << u_current << std::flush;
      }

      // Step 8: Check for convergence.
      if (!logger)
      {
         if (iter_error < itol)
         {
            if (myid == 0)
            {
               mfem::out << "\nConverged after " << k << " iterations." << std::endl;
            }
            if (visualization)
            {
               sol_sock << "keys cFFF\n";
            }
            break;
         }
      }

      // Step 9: Update previous solution for next iteration.
      u_previous = u_current;
   }

   // 14. Save the final solution in ParaView format.
   if (paraview_output)
   {
      paraview_dc.SetCycle(1);
      paraview_dc.SetTime((real_t)1);
      paraview_dc.Save();
   }

   // 15. Free used memory.
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }

   return 0;
}

void ManufacturedSolution(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   const real_t z = x(dim-1);

   Vector f(dim);
   ForceFunction(x, f);
   real_t fz = f(dim-1);

   u = 0.0;
   u(dim-1) = -fz / (2 * (lambda + 2*mu)) * (z - 2.0) * z;
   u(dim-1) += -0.5;
}

/**
 * @brief Initializes the displacement vector u with an initial guess of
 *        u(x,y,z) = (0,0,-0.1z), which satisfies the boundary conditions.
 */
void InitDisplacement(const Vector &x, Vector &u)
{
   const real_t displacement = -0.1;
   const int dim = x.Size();

   u = 0.0;
   u(dim-1) = displacement*x(dim-1);
}

/**
 * @brief Computes the gap function φ₁ based on the input vector x; represents
 *        the distance between a point x and a plane.
 *
 * @param x Input vector
 * @return real_t Computed gap function value, φ₁(x)
 */
real_t GapFunction(const Vector &x)
{
   const int dim = x.Size();
   const real_t d = -0.5;

   return x(dim-1) - d;
}

/**
 * @brief Computes the force function based on the input vector x.
 *
 * @param x Input vector
 * @param f Output vector representing the downward force
 */
void ForceFunction(const Vector &x, Vector &f)
{
   const int dim = x.Size();
   const real_t force = -2.0;

   f = 0.0;
   f(dim-1) = force;
}

/**
 * @brief Computes the stress tensor σ(u) based on the gradient of the
 *        displacement field u and the Lamé parameters.
 *
 * @param grad_u Gradient of the displacement field
 * @param lambda First Lamé parameter
 * @param mu     Second Lamé parameter
 * @param sigma  Computed stress tensor
 */
void ComputeStress(const DenseMatrix &grad_u, const real_t lambda,
                   const real_t mu, DenseMatrix &sigma)
{
   const int dim = grad_u.Size();

   // Compute div(u): trace of Jacobian ∇u
   const real_t div_u = grad_u.Trace();

   // Compute strain: ε(u) = (∇u + ∇uᵀ)/2
   DenseMatrix epsilon = grad_u;
   epsilon.Symmetrize();

   // Compute stress: σ(u) = λ div(u) I + 2μ ε(u)
   DenseMatrix I;
   I.Diag(1.0, dim);
   sigma = 0.0;
   Add(lambda * div_u, I, 2 * mu, epsilon, sigma);
}

