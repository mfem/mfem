//                                MFEM Example 43 - Parallel Version
//
// Compile with: make ex43p
//
// Sample runs:  mpirun -np 4 ex43p -m ../data/ball-nurbs.mesh -r 2
//               mpirun -np 4 ex43p -m ../data/ref-cube.mesh -r 2
//               mpirun -np 4 ex43p -m ../data/fichera.mesh
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
//               We recommend viewing Example 2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
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

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   real_t displ_mag = 0.1;
   int order = 1;
   int ref_levels = 0;
   real_t lambda = 1.0;
   real_t mu = 1.0;
   real_t kappa = -1.0;
   bool static_cond = false;
   bool reorder_space = false;
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
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   // 5. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement.
   for (int i = 0; i < ref_levels; i++)
   {
      mesh->UniformRefinement();
   }

   // 6. Interpolate the geometry after refinement to control geometry error.
   int curvature_order = max(order, 2);
   mesh->SetCurvature(curvature_order);

   // 7. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 8. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   const bool use_nodal_fespace = pmesh->NURBSext;
   if (use_nodal_fespace)
   {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      if (reorder_space)
      {
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byNODES);
      }
      else
      {
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
      }
   }
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   // 9. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   // 10. Define the solution vector x as a finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 11. Set up the bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with constant
   //     coefficients lambda and mu.
   ConstantCoefficient lambda_c(lambda);
   ConstantCoefficient mu_c(mu);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_c,mu_c));
   a->AddBdrFaceIntegrator(
      new SlidingElasticityIntegrator(lambda_c, mu_c, kappa),
      ess_bdr);

   // 12. Set up the linear form b(.) corresponding to the Nitsche method
   //     to impose the Dirichlet boundary conditions. Here, we set the
   //     prescribed displacement on the Dirichlet boundary to be a constant
   //     normal displacement of magnitude 'displ_mag'.
   ConstantCoefficient g(displ_mag);

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddBdrFaceIntegrator(
      new SlidingElasticityDirichletLFIntegrator(
         g, lambda_c, mu_c, kappa), ess_bdr);
   b->Assemble();

   // 13. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (myid == 0) { cout << "matrix ... " << flush; }
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   if (myid == 0)
   {
      cout << "done." << endl;
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 14. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
   //     preconditioner from hypre.
   HypreBoomerAMG *amg = new HypreBoomerAMG(A);
   if (!a->StaticCondensationIsEnabled())
   {
      amg->SetElasticityOptions(fespace);
   }
   else
   {
      amg->SetSystemsOptions(dim, reorder_space);
   }
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-8);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(B, X);

   // 15. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 16. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element.  This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!use_nodal_fespace)
   {
      pmesh->SetNodalFESpace(fespace);
   }

   // 17. Save in parallel the displaced mesh and the inverted solution (which
   //     gives the backward displacements to the original grid). This output
   //     can be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      GridFunction *nodes = pmesh->GetNodes();
      *nodes += x;
      x *= -1;

      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 18. Send the above data by socket to a GLVis server.  Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 19. Free the used memory.
   delete pcg;
   delete amg;
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete pmesh;

   return 0;
}
