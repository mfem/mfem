//                                MFEM Signorini
//
// Compile with: make signorini
//
// Sample runs:  signorini
//               signorini -m ../data/true-tetrahedron.mesh
//
// Description:  This program solves the Signorini problem using MFEM.
//               The problem is defined on a solid with a Dirichlet
//               boundary condition on the bottom face and a traction
//               boundary (Γₜ) condition on the top face. The traction
//               boundary condition is defined through a unit vector field
//               ñ. We aim to (iteravely) solve
//
//               (σ(u), ε(v)) = (f, v)                       for all v ∈ V
//               uᵏ ⋅ ñ = ϕ₁ − (ϕ₁ − uᵏ⁻¹) ⋅ ñ exp(αₖ σ(uᵏ⁻¹) n) ⋅ ñ) on Γₜ
//
//               where σ is the stress tensor, ε is the strain tensor,
//               f is the body force, v is the test function, uᵏ is the
//               displacement at iteration k,
//               ϕ₁ is the prescribed displacement, αₖ is a step-size
//               parameter, and n is the normal vector to the boundary.

#include "mfem.hpp"
#include "signorini.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
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
                  "Lamé's first parameter.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lamé's second parameter.");
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
      args.PrintUsage(cout);
      return 1;
   }
   if (mu <= 0.0 || lambda + 2.0/3.0 * mu <= 0.0)
   {
      std::cerr << "Invalid Lamé parameters." << std::endl;
      return 3;
   }
   args.PrintOptions(mfem::out);

   // 2. Read the mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // 3. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh.NURBSext)
   {
      mesh.DegreeElevate(order, order);
   }

   // 4. Refine the mesh processors to increase the resolution. In
   //    this program we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 5. Interpolate the geometry after refinement to control geometry error.
   int curvature_order = max(order, 2);
   mesh.SetCurvature(curvature_order);

   // 6. Mark the bottom boundary of the mesh as attribute 1 and the other
   //    boundary elements as attribute 2.
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      Element *facet = mesh.GetBdrElement(i);
      Array<int> vertices;
      facet->GetVertices(vertices);

      // Compute the z-centroid of the facet.
      real_t z_centroid = 0.0;
      for (int j = 0; j < vertices.Size(); j++)
      {
         z_centroid += mesh.GetVertex(vertices[j])[dim-1];
      }
      z_centroid /= vertices.Size();

      if (abs(z_centroid) < 1e-8)
      {
         facet->SetAttribute(1);
      }
      else
      {
         facet->SetAttribute(2);
      }
   }
   mesh.SetAttributes();

   // 7. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;
   if (mesh.NURBSext)
   {
      fec = NULL;
      fespace = (FiniteElementSpace *)mesh.GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new FiniteElementSpace(&mesh, fec, dim);
   }
   int size = fespace->GetTrueVSize();
   cout << "Number of finite element unknowns: " << size << endl;

   // 8. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this program, the boundary conditions are defined by
   //    marking only boundary attribute 1 from the mesh as essential and
   //    converting it to a list of true dofs.
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1; // boundary attribute 1 is Dirichlet

   Array<int> ess_tdof_list;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 9. Define coefficients for later.
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

   // 10. Set up the parallel linear form b(⋅) which corresponds to the
   //     right-hand side of the FEM linear system.
   LinearForm b(fespace);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));
   b.Assemble();

   // 11. Define the solution vector u as a parallel finite element grid
   //     function corresponding to fespace. Initialize u with initial guess of
   //     u(x) = (0,...,0,-0.1*x_d), which satisfies the boundary conditions.
   GridFunction u_previous(fespace);
   GridFunction u_current(fespace);
   GridFunctionCoefficient u_previous_coeff(&u_previous);

   VectorFunctionCoefficient init_u(dim, InitDisplacement);
   u_previous.ProjectCoefficient(init_u);

   // 12. Set up the bilinear form a(⋅,⋅) on the finite element space
   //     corresponding to the linear elasticity integrator with coefficients
   //     lambda and mu.
   ConstantCoefficient one(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(one,lambda,mu));
   a->Assemble();

   // 13. Set up GLVis visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 14. Initialize ParaView output.
   ParaViewDataCollection paraview_dc("signorini", &mesh);
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

   // 15. Iterate:
   for (int iter = 1; iter <= max_iterations; iter++)
   {
      mfem::out << "Iteration " << iter << "/" << max_iterations << std::endl;

      // Reassemble the linear form b(⋅)
      b.Assemble();

      // Create the boundary condition coefficient using previous solution
      TractionBoundary trac_coeff(dim, &u_previous, n_tilde, lambda, mu, alpha);
      u_current.ProjectBdrCoefficient(trac_coeff, ess_bdr);

      // Form the linear system A X = B. This includes eliminating boundary
      // conditions, applying AMR constraints, and other transformations.
      SparseMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, u_current, b, A, X, B);

      // 16. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
      GSSmoother M(A);
      PCG(A, M, B, X, 0, 200, 1e-12, 0.0);

      // Recover the solution.
      a->RecoverFEMSolution(X, b, u_current);

      // Compute and display the (L2) difference between current and previous
      // solutions.
      iter_error = u_current.ComputeL2Error(u_previous_coeff);
      mfem::out << "L2 iter difference: " << iter_error << std::endl;

      // 17. Send the above data by socket to a GLVis server. Use the "n" and "b"
      //     keys in GLVis to visualize the displacements.
      if (visualization)
      {
         sol_sock << "solution\n" << mesh << u_current << std::flush;
      }

      // Check for convergence.
      if (iter_error < itol)
      {
         mfem::out << "Converged after " << iter << " iterations." << std::endl;
         if (visualization)
         {
            sol_sock << "keys cFFF\n";
         }
         break;
      }

      // Update previous solution for next iteration.
      u_previous = u_current;
   }

   // 18. Save the final solution in ParaView format.
   if (paraview_output)
   {
      paraview_dc.SetCycle(1);
      paraview_dc.SetTime((real_t)1);
      paraview_dc.Save();
   }

   // 19. Free the used memory.
   delete a;
   if (fec)
   {
      delete fespace;
      delete fec;
   }

   return 0;
}
