//                                MFEM Example 43
//
// Compile with: make ex43
//
// Sample runs:  ex43 -m ../data/ball-nurbs.mesh -r 2
//               ex43 -m ../data/ref-cube.mesh -r 2
//               ex43 -m ../data/fichera.mesh
//               ex43 -m ../data/star.mesh
//
// Description:  This example code solves a linear elasticity problem using
//               Nitsche's method to enforce sliding boundary conditions. In
//               particular, we consider a linear elastic body that is displaced
//               in the normal direction on the entire boundary, but is free to
//               slide in the tangential direction. This is achieved by imposing
//               homogeneous Dirichlet boundary conditions on the normal
//               component of the displacement, while applying homogeneous
//               Neumann boundary conditions on the tangential components of the
//               displacement. By enforcing a uniform, constant normal
//               displacement on the boundary, we can simulate the effect of
//               compressing or expanding the elastic body uniformly. These
//               boundary conditions are applied weakly using Nitsche's method,
//               allowing for more flexibility in handling complex geometries in
//               either 2D or 3D.
//
//               The strong form is given by:
//
//                          −Div(σ(u)) = 0      in Ω
//                               u ⋅ n = g      on Γ
//                                σ(u) ⊥ n      on Γ
//
//               where σ(u) = λ tr(ε(u)) I + 2μ ε(u) is the stress tensor, ε(u)
//               is the strain tensor, λ and μ are the Lamé parameters, and g is
//               the prescribed displacement on the boundary. Here, n is the
//               outward normal on the boundary Γ = ∂Ω.
//
//               The weak form using Nitsche's method is:
//
//               Find u ∈ V     such that   a(u,v) = b(v)   for all v ∈ V
//
//               where
//
//                          a(u,v) := ∫_Ω σ(u) : ε(v) dx
//                                    - ∫_Γ (σ(u) n ⋅ n) (v ⋅ n) dS
//                                    - ∫_Γ (σ(v) n ⋅ n) (u ⋅ n) dS
//                                    + κ ∫_Γ h⁻¹ (λ + 2μ) (u ⋅ n) (v ⋅ n) dS,
//
//                            b(v) := - ∫_Γ σ(v) n ⋅ n g dS
//                                    + κ ∫_Γ h⁻¹ (λ + 2μ) (v ⋅ n) g dS,
//
//               with κ > 0 being a penalty parameter. Here, h is a
//               characteristic element size on the boundary. The function
//               space V is a vector H1-conforming finite element space.
//
//               This example can be viewed as an alternative to Example 28.
//               Whereas Example 28 imposes sliding boundary conditions using
//               the general-purpose constrained system solvers found in
//               mfem/linalg/constraints.hpp, this example employs Nitsche's
//               method to weakly enforce the same condition by modifying the
//               underlying variational formulation. Unlike Example 28, the
//               approach here is specialized to isotropic linear elasticity,
//               but it has the advantage of producing a well-conditioned SPD
//               stiffness matrix that can be readily preconditioned with
//               standard AMG. We recommend reviewing Example 2 before working
//               through this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   real_t displ_mag = 0.1;
   int order = 1;
   int ref_levels = 0;
   real_t lambda = 1.0;
   real_t mu = 1.0;
   real_t kappa = -1.0;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&displ_mag, "-g", "--displ",
                  "Magnitude of the normal displacement.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of uniform mesh refinements.");
   args.AddOption(&lambda, "-l", "--lambda", "First Lamé parameter.");
   args.AddOption(&mu, "-mu", "--mu", "Second Lamé parameter.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "The penalty parameter, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement.
   for (int i = 0; i < ref_levels; i++)
   {
      mesh->UniformRefinement();
   }

   // 5. Interpolate the geometry after refinement to control geometry error.
   int curvature_order = max(order, 2);
   mesh->SetCurvature(curvature_order);

   // 6. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;
   if (mesh->NURBSext)
   {
      fec = NULL;
      fespace = mesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new FiniteElementSpace(mesh, fec, dim);
   }
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl << "Assembling: " << flush;

   // 7. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with constant
   //    coefficients lambda and mu.
   ConstantCoefficient lambda_c(lambda);
   ConstantCoefficient mu_c(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_c,mu_c));
   a->AddBdrFaceIntegrator(
      new SlidingElasticityIntegrator(lambda_c, mu_c, kappa),
      ess_bdr);

   // 10. Set up the linear form b(.) corresponding to the Nitsche method
   //     to impose the Dirichlet boundary conditions. Here, we set the
   //     prescribed displacement on the Dirichlet boundary to be a constant
   //     normal displacement of magnitude 'displ_mag'.
   ConstantCoefficient g(displ_mag);

   LinearForm *b = new LinearForm(fespace);
   b->AddBdrFaceIntegrator(
      new SlidingElasticityLFIntegrator(
         g, lambda_c, mu_c, kappa), ess_bdr);
   b->Assemble();

   // 11. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   cout << "matrix ... " << flush;
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "done." << endl;

   cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
   // 12. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //     solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 500, 1e-8, 0.0);
#else
   // 12. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 13. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 14. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element. This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!mesh->NURBSext)
   {
      mesh->SetNodalFESpace(fespace);
   }

   // 15. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      GridFunction *nodes = mesh->GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Send the above data by socket to a GLVis server. Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 17. Free the used memory.
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete mesh;

   return 0;
}
