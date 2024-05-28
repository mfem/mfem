//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/octahedron.mesh -o 1
//               mpirun -np 4 ex1p -m ../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               mpirun -np 4 ex1p -pa -d cuda
//               mpirun -np 4 ex1p -fa -d cuda
//               mpirun -np 4 ex1p -pa -d occa-cuda
//               mpirun -np 4 ex1p -pa -d raja-omp
//               mpirun -np 4 ex1p -pa -d ceed-cpu
//               mpirun -np 4 ex1p -pa -d ceed-cpu -o 4 -a
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * mpirun -np 4 ex1p -pa -d ceed-cuda
//             * mpirun -np 4 ex1p -pa -d ceed-hip
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/beam-tet.mesh -pa -d ceed-cpu
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

Mesh DividingPlaneMesh(bool tet_mesh, bool split, bool three_dim, real_t scale = 1.0)
{

   auto mesh = three_dim ? Mesh("../../data/ref-cube.mesh") : Mesh("../../data/ref-square.mesh");
   {
      Array<Refinement> refs;
      refs.Append(Refinement(0, Refinement::X));
      mesh.GeneralRefinement(refs);
   }
   delete mesh.ncmesh;
   mesh.ncmesh = nullptr;
   mesh.FinalizeTopology();
   mesh.Finalize(true, true);

   mesh.SetAttribute(0, 1);
   mesh.SetAttribute(1, split ? 2 : 1);

   // Introduce internal boundary elements
   const int new_attribute = mesh.bdr_attributes.Max() + 1;
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      mesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh.GetAttribute(e1) != mesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh.GetFace(f)->Duplicate(&mesh);
         new_elem->SetAttribute(new_attribute);
         mesh.AddBdrElement(new_elem);
      }
   }
   if (tet_mesh)
   {
      mesh = Mesh::MakeSimplicial(mesh);
   }
   mesh.FinalizeTopology();
   mesh.Finalize(true, true);

   for (int i = 0; i < mesh.GetNV(); i++)
   {
      mesh.GetVertex(i)[0] *= scale;
   }

   return mesh;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
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
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   // Mesh mesh(mesh_file, 1, 1);

   auto mesh = DividingPlaneMesh(false, true, true, 1.0);

   int dim = mesh.Dimension();

   mesh.EnsureNCMesh();
   mesh.UniformRefinement();
   // mesh.UniformRefinement();
   // Array<Refinement> refs(1);
   // refs[0].index = 0;
   // refs[0].ref_type = Refinement::XYZ;
   // mesh.GeneralRefinement(refs);
   // mesh.RandomRefinement(0.5);
   // delete mesh.ncmesh;
   // mesh.ncmesh = nullptr;


   // mesh.RandomRefinement(0.5);

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   // {
   //    int ref_levels =
   //       (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
   //    for (int l = 0; l < ref_levels; l++)
   //    {
   //       mesh.UniformRefinement();
   //    }
   // }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   // std::vector<int> partitioning(mesh.GetNE(), 0);
   // partitioning[1] = Mpi::WorldSize() > 1 ? 1 : 0;
   // std::cout << __FILE__ << ':' << __LINE__ << std::endl;
   // ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning.data());

   std::cout << "\n\n\nIn Ex1p\n\n";


   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // pmesh.UniformRefinement();
   // pmesh.UniformRefinement();

   // Array<Refinement> refs(1);
   // refs[0].index = 0;
   // refs[0].ref_type = Refinement::XYZ;
   // pmesh.GeneralRefinement(refs);
   // pmesh.GeneralRefinement(refs);
   // pmesh.RandomRefinement(0.5);
   // pmesh.RandomRefinement(0.5);

   // pmesh.Rebalance();

   mesh.Clear();


   std::cout << __FILE__ << ':' << __LINE__ << std::endl;
   std::cout << "MyRank " << pmesh.GetMyRank() << std::endl;

   std::cout << __FILE__ << ':' << __LINE__ << std::endl;
   if (pmesh.ncmesh)
   {
      std::cout << "nodes.Size() " << pmesh.ncmesh->nodes.Size() << std::endl;
      for (auto it = pmesh.ncmesh->nodes.begin(); it != pmesh.ncmesh->nodes.end(); ++it)
      {
         std::cout << "node " << it.index() << ' ' << it->p1 << ' ' << it->p2;
         std::cout << " vert_index " << it->vert_index << " edge_index " << it->edge_index;
         std::cout << std::endl;
      }
   }
   std::cout << "pmesh.GetNV() " << pmesh.GetNV() << std::endl;


   // std::cout << "R" << pmesh.GetMyRank() << " GetNE() " << pmesh.GetNE() << std::endl;
   // for (int i = 0; i < pmesh.pncmesh->elements.Size(); i++)
   // {
   //    std::cout << "R" << pmesh.GetMyRank() << " element " << i << " " << pmesh.pncmesh->elements[i].index << std::endl;
   // }

   // if (order > 0

   // {
   //    int par_ref_levels = 2;
   //    for (int l = 0; l < par_ref_levels; l++)
   //    {
   //       pmesh.UniformRefinement();
   //    }
   // }

   Array<int> subdomain_attributes(1);
   // subdomain_attributes[0] = 2;
   // auto psubmesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

   subdomain_attributes[0] = 2;
   auto psubmesh = ParSubMesh::CreateFromBoundary(pmesh, subdomain_attributes);

   // auto &psubmesh = pmesh;

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (psubmesh.GetNodes())
   {
      fec = psubmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   ParFiniteElementSpace fespace(&psubmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (psubmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(psubmesh.bdr_attributes.Max());
      ess_bdr = 1;
      for (auto x : psubmesh.bdr_attributes)
      {
         std::cout << "bdr " << x << std::endl;
      }
      std::cout << "psubmesh.bdr_attributes.Max() " << psubmesh.bdr_attributes.Max() << std::endl;
      std::cout << "ess_bdr.Size() " << ess_bdr.Size() << std::endl;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the
   //     Diffusion domain integrator.
   ParBilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use Jacobi smoothing, for now.
   Solver *prec = NULL;
   if (pa)
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            prec = new ceed::AlgebraicSolver(a, ess_tdof_list);
         }
         else
         {
            prec = new OperatorJacobiSmoother(a, ess_tdof_list);
         }
      }
   }
   else
   {
      prec = new HypreBoomerAMG;
   }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // Transfer the solution back to the original mesh.
   auto bk_fec = std::unique_ptr<H1_FECollection>(new H1_FECollection(order, pmesh.Dimension()));
   ParFiniteElementSpace bk_fespace(&pmesh, bk_fec.get());
   ParGridFunction bk_x(&bk_fespace);
   bk_x = 0.0;

   psubmesh.Transfer(x, bk_x);

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      psubmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << psubmesh << x << flush;

      socketstream sol_sock2(vishost, visport);
      sol_sock2 << "parallel " << num_procs << " " << myid << "\n";
      sol_sock2.precision(8);
      sol_sock2 << "solution\n" << pmesh << bk_x << flush;

   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
