//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/octahedron.mesh -o 1
//               ex1 -m ../data/periodic-annulus-sector.msh
//               ex1 -m ../data/periodic-torus-sector.msh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -fa -d cuda
//               ex1 -pa -d raja-cuda
//             * ex1 -pa -d raja-hip
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cpu -o 4 -a
//               ex1 -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * ex1 -pa -d ceed-cuda
//             * ex1 -pa -d ceed-hip
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
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
#include "mesh/submesh/ncsubmesh.hpp"


using namespace std;
using namespace mfem;

Mesh DividingPlaneMesh(bool tet_mesh, bool split, bool three_dim)
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
   return mesh;
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
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
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   // Mesh mesh(mesh_file, 1, 1);
   // Mesh mesh("../../data/ref-segment.mesh", 1, 1);

   // Solve Poisson on a pair of cubes fully coupled, then compare to solving the interface
   // then coupling the two domains using the 2D solution as the boundary condition.

   auto mesh = DividingPlaneMesh(true, true, true);
   mesh.EnsureNCMesh();
   // auto mesh = Mesh("../../data/ref-cube.mesh");

   // mesh.Finalize(true);
   // mesh.UniformRefinement();
   // mesh.EnsureNCMesh(true);
   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 7;

   auto refine_half = [](Mesh &mesh, int vattr, int battr, bool backwards = true)
   {
      Array<Refinement> refs(1);
      std::vector<int> ind(mesh.GetNBE());
      if (backwards)
      {
         std::iota(ind.rbegin(), ind.rend(), 0);
      }
      else
      {
         std::iota(ind.begin(), ind.end(), 0);
      }
      for (int e : ind)
      {
         std::cout << e << ' ';
         if (mesh.GetBdrAttribute(e) == battr)
         {
            int el, info;
            mesh.GetBdrElementAdjacentElement(e, el, info);
            if (mesh.GetAttribute(el) == vattr)
            {
               refs[0].index = el;
               refs[0].ref_type = Refinement::XYZ;
               break;
            }
         }
      }
      mesh.GeneralRefinement(refs);
   };

   int test = 5;
   mesh.EnsureNCMesh(true);
   switch (test)
   {
      case 0:
         break;
      case 1:
         mesh.UniformRefinement();
         break;
      case 2:
         {
            Array<Refinement> refs(1);
            refs[0].index = 0;
            refs[0].ref_type = Refinement::XYZ;
            mesh.GeneralRefinement(refs);
         }
         break;
      case 3 :
         {
            Array<Refinement> refs(1);
            refs[0].index = 1;
            refs[0].ref_type = Refinement::XYZ;
            mesh.GeneralRefinement(refs);
         }
         break;
      case 4 :
         {
            mesh.UniformRefinement();
            refine_half(mesh,1,subdomain_attributes[0], false);
            // refine_half(mesh,1,subdomain_attributes[0], true);
            // refine_half(mesh,1,subdomain_attributes[0], false);
         }
         break;
      case 5 :
         mesh.RandomRefinement(0.5, false, 1, 1);
         mesh.RandomRefinement(0.5, false, 1, 1);
         mesh.RandomRefinement(0.5, false, 1, 1);
         // mesh.RandomRefinement(0.5, false, 1, 1);
         break;
   }

   std::cout << "\n\n\nInEx1\n\n\n";

   auto print_ncmesh = [](NCMesh &ncmesh)
   {
      for (const auto &n : ncmesh.nodes)
      {
         const int id = ncmesh.nodes.FindId(n.p1, n.p2);
         std::cout << "id " << id << " n.p1 " << n.p1 << " n.p2 " << n.p2 << " Edge " << n.edge_index << " Vertex " << n.vert_index << std::endl;
      }

      const auto &face_list = ncmesh.GetFaceList();
      std::cout << "face_list.conforming.Size() " << face_list.conforming.Size() << std::endl;
      std::cout << "face_list.master.Size() " << face_list.masters.Size() << std::endl;
      std::cout << "face_list.slaves.Size() " << face_list.slaves.Size() << std::endl;

      const auto &edge_list = ncmesh.GetEdgeList();
      std::cout << "edge_list.conforming.Size() " << edge_list.conforming.Size() << std::endl;
      std::cout << "edge_list.master.Size() " << edge_list.masters.Size() << std::endl;
      std::cout << "edge_list.slaves.Size() " << edge_list.slaves.Size() << std::endl;

      const auto &vertex_list = ncmesh.GetVertexList();
      std::cout << "vertex_list.conforming.Size() " << vertex_list.conforming.Size() << std::endl;
      std::cout << "vertex_list.master.Size() " << vertex_list.masters.Size() << std::endl;
      std::cout << "vertex_list.slaves.Size() " << vertex_list.slaves.Size() << std::endl;


   };

   if (mesh.ncmesh)
   {
      print_ncmesh(*mesh.ncmesh);
   }

   std::cout << "\n\nSubMesh\n\n";

   auto submesh = SubMesh::CreateFromBoundary(mesh, subdomain_attributes);

   int dim = submesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   // {
   //    int ref_levels =
   //       (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
   //    for (int l = 0; l < ref_levels; l++)
   //    {
   //       mesh.RandomRefinement(0.5);
   //    }
   // }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (submesh.GetNodes())
   {
      fec = submesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&submesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (submesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(submesh.bdr_attributes.Max());
      ess_bdr = 1;
      for (auto x : submesh.bdr_attributes)
      {
         std::cout << "bdr " << x << std::endl;
      }
      std::cout << "submesh.bdr_attributes.Max() " << submesh.bdr_attributes.Max() << std::endl;
      std::cout << "ess_bdr.Size() " << ess_bdr.Size() << std::endl;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   std::cout << "ess_tdof_list.Size() " << ess_tdof_list.Size() << std::endl;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient f([dim](const Vector &x){
      // double c = M_PI * M_PI * dim;
      // for (int i = 0; i < x.Size(); ++i)
      // {
      //    c *= std::sin(M_PI * x(i));
      // }
      // return c;
      return 1.0;
   });
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&fespace);
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

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            ceed::AlgebraicSolver M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
         else
         {
            OperatorJacobiSmoother M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // Transfer the solution back to mesh

   auto bk_fec = std::unique_ptr<H1_FECollection>(new H1_FECollection(order, mesh.Dimension()));
   FiniteElementSpace bk_fespace(&mesh, bk_fec.get());
   GridFunction bk_x(&bk_fespace);
   bk_x = 0.0;
   std::cout << "bk_fespace.GetTrueVSize() " << bk_fespace.GetTrueVSize() << std::endl;

   submesh.Transfer(x, bk_x);
   Vector nval;
   bk_x.GetNodalValues(nval);


   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock2(vishost, visport);
      sol_sock2.precision(8);
      sol_sock2 << "solution\n" << submesh << x << flush;

      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << bk_x << flush;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
