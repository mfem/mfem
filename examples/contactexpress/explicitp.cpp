//                                MFEM Signorini
//
// Compile with: make explicitp
//
// Sample runs:  mpirun -np 4 explicitp
//               mpirun -np 4 explicitp -r 1 -p 0 -m ../../data/wheel.msh
//               mpirun -np 8 explicitp -r 1 -p 0 -m ../../data/hemisphere.msh
//               mpirun -np 8 explicitp -m ../../data/pipe.msh -f 0.5
//
// Description:  This program solves the Signorini problem using MFEM.
//               The problem is defined on a solid with a Dirichlet
//               boundary condition on the bottom face and a traction
//               boundary (Γₜ) condition on the top face. The traction
//               boundary condition is defined through a unit vector field
//               ñ. We aim to (iteratively) find uᵏ ∈ V such that
//
//               (σ(u), ε(v)) = (f, v)                     for all v ∈ V
//               uᵏ · ñ = φ₁ + (uᵏ⁻¹ · ñ - φ₁) exp(αₖ (σ(uᵏ)n · ñ)) on Γₜ
//
//               where σ is the stress tensor, ε is the strain tensor,
//               f is the body force, uᵏ is the displacement at iteration k, ϕ₁
//               is a prescribed gap function, αₖ is a positive sequence of
//               step-size parameters, and n is the normal vector to the
//               boundary.

#include "mfem.hpp"
#include "signorini.hpp"
#include <iostream>
#include <filesystem>

using namespace std;
using namespace mfem;

// We take the plane to be z = plane_g and the force to be a constant downward
// force of magnitude force_g.
real_t plane_g = -0.5;
real_t force_g = 2.0;

// Selects mesh-dependent problem data (initial displacement and forcing).
bool pipe_mesh = false;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char* mesh_file = "../../data/ref-cube.mesh";
   int order = 1;
   real_t alpha = 1.0;
   real_t lambda = 1.0;
   real_t mu = 1.0;
   int ref_levels = 0;
   int max_iterations = 9;
   real_t itol = 1e-6;
   bool reorder_space = false;
   bool visualization = true;
   bool paraview_output = false;

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
   args.AddOption(&plane_g, "-p", "--plane",
                  "Height of the plane for the Signorini condition.");
   args.AddOption(&force_g, "-f", "--force",
                  "Magnitude of the downward force.");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of uniform mesh refinements.");
   args.AddOption(&max_iterations, "-i", "--iterations",
                  "Maximum number of (outer) iterations.");
   args.AddOption(&itol, "-tol", "--tolerance",
                  "Outer iteration tolerance.");
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

   // Determine the mesh name (used to select boundary conditions and problem
   // data below).
   filesystem::path mesh_path(mesh_file);
   string mesh_stem = mesh_path.stem().string();
   pipe_mesh = (mesh_stem == "pipe");

   // 3. Postprocess the mesh.
   // 3A. Refine the serial mesh on all processors to increase the resolution. In
   //     this program we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 3B. Interpolate the geometry after refinement to control geometry error.
   // NOTE: Minimum second-order interpolation is used to improve the accuracy.
   int curvature_order = max(order, 2);
   mesh.SetCurvature(curvature_order);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. The boundary
   //    attributes provided by the mesh file are used directly (see section 7).
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

   // 6. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs, keyed off the mesh name.
   //
   //    ess_bdr_x / ess_bdr_y  : symmetry planes; the in-plane displacement
   //                             component is held at zero there.
   //    contact_bdr (= Γ_T)    : contact surface; the normal (z) component is
   //                             set by the TractionBoundary exp-update.
   Array<int> ess_bdr_x(pmesh.bdr_attributes.Max());
   Array<int> ess_bdr_y(pmesh.bdr_attributes.Max());
   Array<int> contact_bdr(pmesh.bdr_attributes.Max());
   ess_bdr_x = 0; ess_bdr_y = 0; contact_bdr = 0;

   if (mesh_stem == "ref-cube")
   {
      ess_bdr_x[2] = 1; ess_bdr_x[4] = 1;
      ess_bdr_y[1] = 1; ess_bdr_y[3] = 1;
      contact_bdr[0] = 1;
   }
   else if (mesh_stem == "wheel")
   {
      ess_bdr_x[1] = 1; ess_bdr_x[2] = 1; ess_bdr_x[3] = 1;
      ess_bdr_y[1] = 1; ess_bdr_y[2] = 1; ess_bdr_y[3] = 1;
      contact_bdr[0] = 1;
   }
   else if (mesh_stem == "hemisphere")
   {
      ess_bdr_x[1] = 1;
      ess_bdr_y[1] = 1;
      contact_bdr[0] = 1;
   }
   else if (mesh_stem == "pipe")
   {
      ess_bdr_x = 1;
      ess_bdr_y = 1;
      contact_bdr[0] = 1; contact_bdr[2] = 1; contact_bdr[3] = 1;
   }
   else
   {
      MFEM_ABORT("Unknown mesh file. Please specify essential boundary "
                 "conditions for this mesh.");
   }

   // The contact condition constrains only the normal (z) component on Γ_T;
   // the tangential components there remain free. The x/y symmetry planes
   // remove the remaining rigid-body modes.
   Array<int> ess_tdof_list_x, ess_tdof_list_y, ess_tdof_list_c;
   fespace->GetEssentialTrueDofs(ess_bdr_x,   ess_tdof_list_x, 0);
   fespace->GetEssentialTrueDofs(ess_bdr_y,   ess_tdof_list_y, 1);
   fespace->GetEssentialTrueDofs(contact_bdr, ess_tdof_list_c, dim-1);

   Array<int> ess_tdof_list;
   ess_tdof_list.Append(ess_tdof_list_x);
   ess_tdof_list.Append(ess_tdof_list_y);
   ess_tdof_list.Append(ess_tdof_list_c);

   // 7. Define coefficients for later.
   VectorFunctionCoefficient f_coeff(dim, ForceFunction);
   Vector n_tilde(dim);
   n_tilde = 0.0;
   n_tilde(dim-1) = -1.0;
   {
      real_t n_tilde_norm = n_tilde.Norml2();
      if (n_tilde_norm != 1.0)
      {
         if (myid == 0)
         {
            cout << "Warning: n_tilde norm is not 1.0, normalizing it." << endl;
         }
         n_tilde /= n_tilde_norm;
      }
   }

   // 9. Set up the parallel linear form b(⋅) which corresponds to the
   //    right-hand side of the FEM linear system.
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));
   if (myid == 0)
   {
      cout << "r.h.s. ... " << flush;
   }
   b->Assemble();

   // 10. Define the solution vector u as a parallel finite element grid
   //     function corresponding to fespace. Initialize u with initial guess of
   //     u(x,y,z) = (0,0,plane_g), which satisfies the boundary conditions.
   //     u_inner holds the current estimate of uᵏ during the inner iteration.
   ParGridFunction u_previous(fespace);
   ParGridFunction u_current(fespace);
   ParGridFunction u_inner(fespace);
   VectorGridFunctionCoefficient u_previous_coeff(&u_previous);

   VectorFunctionCoefficient init_u(dim, InitDisplacement);
   u_previous.ProjectCoefficient(init_u);
   u_current = u_previous;

   // 11. Set up the bilinear form a(⋅,⋅) on the finite element space
   //     corresponding to the linear elasticity integrator with coefficients
   //     lambda and mu.
   ConstantCoefficient one(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(one,lambda,mu));
   if (myid == 0)
   {
      cout << "matrix ... " << flush;
   }
   a->Assemble();

   // 12. Set up GLVis visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 13. Initialize ParaView output.
   ParaViewDataCollection paraview_dc("signorini", &pmesh);
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

   if (myid == 0)
   {
      mfem::out << "\nk" << setw(14) << "iter_error"
                << setw(10) << "inner_it" << setw(16) << "inner_err"
                << std::endl;
      mfem::out << "------------------------------------------------" << std::endl;
   }

   // 14. Iterate:
   for (int k = 1; k <= max_iterations; k++)
   {
      // Step 1: Reassemble the linear form b(⋅).
      b->Assemble();

      // Step 2: Create the boundary condition coefficient using previous solution.
      TractionBoundary trac_coeff(dim, &u_previous, &u_previous, n_tilde, lambda, mu,
                                  alpha);
      u_current.ProjectBdrCoefficient(trac_coeff, contact_bdr);

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

      if (myid == 0)
      {
         mfem::out << k << setw(14) << iter_error << std::endl;
      }

      // Step 7: Send the above data by socket to a GLVis server. Use the "n"
      // and "b" keys in GLVis to visualize the displacements.
      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << u_current << std::flush;
      }

      // Step 8: Check for convergence.
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

      // Step 9: Update previous solution for next iteration.
      u_previous = u_current;
   }

   // 15. Save the final solution in ParaView format.
   if (paraview_output)
   {
      paraview_dc.SetCycle(1);
      paraview_dc.SetTime((real_t)1);
      paraview_dc.Save();
   }

   // 16. Free used memory.
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }

   return 0;
}

void InitDisplacement(const Vector &x, Vector &u)
{
   u = 0.0;
   if (!pipe_mesh)
   {
      u(x.Size() - 1) = plane_g;
   }
}

void ForceFunction(const Vector &x, Vector &f)
{
   f = 0.0;
   const real_t F = -force_g;
   if (pipe_mesh)
   {
      if (x(0) > 8.0)
      {
         f(x.Size() - 1) = F;
      }
   }
   else
   {
      f(x.Size() - 1) = F;
   }
}

real_t GapFunction(const Vector &x)
{
   return x(x.Size() - 1) - plane_g;
}
