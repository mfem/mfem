//                                MFEM Signorini
//
// Compile with: make signorinip
//
// Sample runs:  mpirun -np 4 signorinip
//               mpirun -np 4 signorinip -m ../data/true-tetrahedron.mesh
//               mpirun -np 4 signorinip -m ../data/ball-nurbs.mesh -o 3
//
// Description:  This program solves the Signorini problem using MFEM.
//               The problem is defined on a solid with a Dirichlet
//               boundary condition on the bottom face and a traction
//               boundary condition on the top face. The traction
//               boundary condition is defined through a unit vector field
//               ñ. We aim to (iteravely) solve
//
//               (σ(u), ε(v)) = (f, v)                       for all v ∈ V
//               uᵏ ⋅ ñ = ϕ₁ − (ϕ₁ − uᵏ⁻¹) ⋅ ñ exp(αₖ σ(uᵏ⁻¹) n) ⋅ ñ) on Γₜ
//
//               where σ(u) is the stress tensor, ε(v) is the strain tensor,
//               f is the body force, v is the test function, uᵏ is the
//               displacement at iteration k, ñ is a unit vector field,
//               ϕ₁ is the prescribed displacement, αₖ is a step-size
//               parameter, and n is the normal vector to the boundary.

#include "mfem.hpp"
#include "signorini.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options
   const char* mesh_file = "../data/ref-cube.mesh";
   int order = 2;
   real_t alpha = 1;
   real_t lambda = 1.0;
   real_t mu = 1.0;
   int ref_levels = 0;
   int max_iterations = 50;
   real_t itol = 1e-6;
   bool visualization = true;
   bool paraview_output = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha parameter for boundary condition.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lame's first parameter.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lame's second parameter.");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of uniform mesh refinements.");
   args.AddOption(&max_iterations, "-i", "--iterations",
                  "Maximum number of iterations.");
   args.AddOption(&itol, "-tol", "--tolerance",
                  "Iteration tolerance.");
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
      std::cerr << "Invalid Lame parameters." << std::endl;
      return 3;
   }
   if (myid == 0)
   {
      args.PrintOptions(mfem::out);
   }

   // 3A. Load the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   ParMesh pmesh = ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear(); // the serial mesh is no longer needed

   // 3B. Refine the mesh.
   for (int l = 0; l < ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // 3C. Interpolate the geometry after refinement to control geometry error.
   int curvature_order = max(order, 2);
   mesh.SetCurvature(curvature_order);

   // 3D. Mark the bottom boundary of the solid as attribute 1, the rest as 2
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      Element *facet = pmesh.GetBdrElement(i);
      Array<int> vertices;
      facet->GetVertices(vertices);

      // Compute the centroid of the facet
      real_t z_centroid = 0.0;
      for (int j = 0; j < vertices.Size(); j++)
      {
         z_centroid += pmesh.GetVertex(vertices[j])[dim-1];
      }
      z_centroid /= vertices.Size();

      // if (z_centroid < 0) // Ball
      if (abs(z_centroid) < 1e-8) // Cube/Tetrahedron
      {
         facet->SetAttribute(1);
      }
      else
      {
         facet->SetAttribute(2);
      }
   }
   pmesh.SetAttributes();

   // 4. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (pmesh.NURBSext)
   {
      pmesh.DegreeElevate(order, order);
   }

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
      fespace = new ParFiniteElementSpace(&pmesh, fec, dim);
   }
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1; // boundary attribute 1 is Dirichlet

   // Get essential boundary DOFs
   Array<int> ess_tdof_list;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Define coefficients for later
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

   // 8A. Set up the linear form b(⋅)
   ParLinearForm b(fespace);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));
   b.Assemble();

   // 8B. Define the solution vectors
   ParGridFunction u_previous(fespace);
   ParGridFunction u_current(fespace);
   GridFunctionCoefficient u_previous_coeff(&u_previous);

   // 8C. Initialize 'u_previous' using the 'InitDisplacement' function.
   VectorFunctionCoefficient init_u(dim, InitDisplacement);
   u_previous.ProjectCoefficient(init_u);

   // 9. Set up the bilinear form a(⋅,⋅) on the finite element space
   //    corresponding to the linear elasticity integrator with coefficients
   //    lambda and mu.
   ConstantCoefficient one(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(one,lambda,mu));
   a->Assemble();

   // 10. Set up visualization
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 11. Initialize ParaView output
   ParaViewDataCollection paraview_dc("signorini", &pmesh);
   if (paraview_output)
   {
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("Displacement",&u_previous);
      paraview_dc.Save();
   }

   real_t iter_error;

   // 12. Iterate
   for (int iter = 1; iter <= max_iterations; iter++)
   {
      if (myid == 0)
      {
         mfem::out << "Iteration " << iter << "/" << max_iterations << std::endl;
      }

      // Reassemble the linear form b(⋅)
      b.Assemble();

      // Create the boundary condition coefficient using previous solution
      TractionBoundary trac_coeff(dim, &u_previous, n_tilde, lambda, mu, alpha);
      u_current.ProjectBdrCoefficient(trac_coeff, ess_bdr);

      // Form the linear system A X = B. This includes eliminating boundary
      // conditions, applying AMR constraints, and other transformations.
      HypreParMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, u_current, b, A, X, B);

      // 13. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG *amg = new HypreBoomerAMG(A);
      amg->SetElasticityOptions(fespace);
      amg->SetPrintLevel(0);
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-8);
      pcg->SetMaxIter(500);
      pcg->SetPrintLevel(0);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);

      // Recover the solution
      a->RecoverFEMSolution(X, b, u_current);

      // Compute difference between current and previous solutions
      iter_error = u_current.ComputeL2Error(u_previous_coeff);

      if (myid == 0)
      {
         mfem::out << "L2 iter difference: " << iter_error << std::endl;
      }

      // 14. Send the above data by socket to a GLVis server. Use the "n" and "b"
      //     keys in GLVis to visualize the displacements.
      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << u_current << std::flush;
      }

      // Check for convergence
      if (iter_error < itol)
      {
         if (myid == 0)
         {
            mfem::out << "Converged after " << iter << " iterations." << std::endl;
         }
         if (visualization)
         {
            sol_sock << "keys cFFF\n";
         }
         break;
      }

      // Update previous solution for next iteration
      u_previous = u_current;
   }

   // 15. Save the final solution in ParaView format
   if (paraview_output)
   {
      paraview_dc.SetCycle(1);
      paraview_dc.SetTime((real_t)1);
      paraview_dc.Save();
   }

   // 16. Free the used memory.
   delete a;
   if (fec)
   {
      delete fespace;
      delete fec;
   }

   Hypre::Finalize();
   Mpi::Finalize();

   return 0;
}
