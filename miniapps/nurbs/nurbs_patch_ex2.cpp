//               MFEM Example 2 - NURBS with patch-wise assembly
//
// Compile with: make nurbs_patch_ex2
//
// Sample runs:  nurbs_patch_ex2 -incdeg 3 -ref 2 -iro 8 -patcha
//               nurbs_patch_ex2 -incdeg 3 -ref 2 -iro 8 -patcha -pa
//               nurbs_patch_ex2 -incdeg 3 -ref 2 -iro 8 -patcha -fint
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)
//
//               This example is a specialization of ex2 which demonstrates
//               patch-wise matrix assembly and partial assembly on NURBS
//               meshes.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex-nurbs.mesh";
   // const char *mesh_file = "../../data/beam-hex.mesh";
   bool pa = false;
   bool patchAssembly = false;
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int ir_order = -1;
   bool reorder_space = false;
   int visport = 19916;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&patchAssembly, "-patcha", "--patch-assembly", "-no-patcha",
                  "--no-patch-assembly", "Enable patch-wise assembly.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&ir_order, "-iro", "--integration-order",
                  "Order of integration rule.");
   args.AddOption(&reorder_space, "-nodes", "--by-nodes", "-vdim", "--by-vdim",
                  "Use byNODES ordering of vector space instead of byVDIM");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }
   MFEM_VERIFY(!(patchAssembly && !pa), "Patch assembly must be used with -pa");

   // 2. Read the serial mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   bool isNURBS = mesh.NURBSext;

   // Verify mesh is valid for this problem
   MFEM_VERIFY(!(isNURBS && !mesh.GetNodes()), "NURBS mesh must have nodes");
   if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
      return 3;
   }

   // 3. Optionally, increase the NURBS degree.
   if (isNURBS && nurbs_degree_increase>0)
   {
      mesh.DegreeElevate(nurbs_degree_increase);
   }

   // 4. Refine the serial mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }
   // 5. Define a parallel mesh by a partitioning of the serial mesh.
   // ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   // serial_mesh.Clear(); // the serial mesh is no longer needed

   // 5. Define a finite element space on the mesh.
   // Node ordering is important
   const Ordering::Type fes_ordering =
      reorder_space ? Ordering::byNODES : Ordering::byVDIM;

   FiniteElementCollection * fec = nullptr;
   if (isNURBS)
   {
      fec = mesh.GetNodes()->OwnFEC();
   }
   else
   {
      fec = new H1_FECollection(1, dim);
   }

   // ParFiniteElementSpace *fespace = new ParFiniteElementSpace(&mesh, fec, dim, fes_ordering);
   FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec, dim, fes_ordering);
   // FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec);
   if (Mpi::Root())
   {
      cout << "Finite Element Collection: " << fec->Name() << endl;
      cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl;
   }

   // Print out some info on the size of spaces
   // const Operator *G;
   // G = fespace->GetElementRestriction()
   // cout << "G : " << G->Height() << " x " << G->Width() << endl;

   cout << "GetNE() = " << fespace->GetNE() << std::endl;
   cout << "GetVSize() = " << fespace->GetVSize() << std::endl; // Vsize = VDIM * ND
   cout << "GetNDofs() = " << fespace->GetNDofs() << std::endl;
   cout << "GetNP() = " << mesh.NURBSext->GetNP() << std::endl;


   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the linear form b(.)
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm b(fespace);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   if (Mpi::Root())
   {
      cout << "RHS ... " << flush;
   }
   b.Assemble();
   if (Mpi::Root())
   {
      cout << "done." << endl;
   }

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.

   // Lame parameters
   Vector lambda(mesh.attributes.Max());
   lambda = 2.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   cout << "lambda = " << endl;
   lambda.Print(cout);
   Vector mu(mesh.attributes.Max());
   mu = 3.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   // Bilinear integrator
   ElasticityIntegrator *ei = new ElasticityIntegrator(lambda_func, mu_func);
   if (patchAssembly)
   {
      ei->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
   }

   // Patch rule
   NURBSMeshRules *patchRule = nullptr;
   if (isNURBS)
   {
      if (ir_order == -1) { ir_order = 2*fec->GetOrder(); }
      if (Mpi::Root())
      {
         cout << "Using ir_order " << ir_order << endl;
      }

      patchRule = new NURBSMeshRules(mesh.NURBSext->GetNP(), dim);
      // Loop over patches and set a different rule for each patch.
      for (int p=0; p < mesh.NURBSext->GetNP(); ++p)
      {
         Array<const KnotVector*> kv(dim);
         mesh.NURBSext->GetPatchKnotVectors(p, kv);

         std::vector<const IntegrationRule*> ir1D(dim);
         const IntegrationRule *ir = &IntRules.Get(Geometry::SEGMENT, ir_order);

         // Construct 1D integration rules by applying the rule ir to each knot span.
         for (int i=0; i<dim; ++i)
         {
            ir1D[i] = ir->ApplyToKnotIntervals(*kv[i]);
         }

         patchRule->SetPatchRules1D(p, ir1D);
      }  // loop (p) over patches

      patchRule->Finalize(mesh);
      ei->SetNURBSPatchIntRule(patchRule);
   }


   // 10. Assemble and solve the linear system
   // Define and assemble bilinear form
   if (Mpi::Root())
   {
      cout << "Assemble a ... " << flush;
   }
   BilinearForm a(fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(ei);
   a.UseExternalIntegrators();
   a.Assemble();
   if (Mpi::Root())
   {
      cout << "done." << endl;
   }

   // Define linear system
   if (Mpi::Root())
   {
      cout << "Matrix ... " << flush;
   }
   OperatorPtr A;
   // SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   if (Mpi::Root())
   {
      cout << "done. " << "(size = " << fespace->GetTrueVSize() << ")" << endl;
   }

   // Solve the linear system A X = B.
   if (Mpi::Root())
   {
      cout << "Solving linear system ... " << endl;
   }
   // GSSmoother M(A);
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 1, 200, 1e-20, 0.0);

   if (Mpi::Root())
   {
      cout << "Done solving system." << endl;
   }

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);


   // 11. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element.
   if (!isNURBS)
   {
      mesh.SetNodalFESpace(fespace);
   }
   // 12. Save the displaced mesh and the inverted solution
   {
      GridFunction *nodes = mesh.GetNodes();
      *nodes += x;
      x *= -1;
   }

   // 13. Send the above data by socket to a GLVis server.
   if (visualization)
   {
      // send to socket
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x;
      sol_sock << "window_geometry " << 0 << " " << 0 << " "
               << 800 << " " << 800 << "\n"
               << "keys agc\n" << std::flush;
   }

   // 14. Free the used memory.
   // delete *mesh;
   delete fespace;
   // delete patchRule;

   return 0;
}
