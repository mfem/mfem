//                     ----------------------------------------
//                     Partial Assembly with Simplices Miniapp
//                     ----------------------------------------
//
// Compile with: make ex_pa_simplices
//
// Sample runs:  ex_pa_simplices
//               mpirun -np 2 ex_pa_tetrahedron --mesh "../../data/inline-tet.mesh" --bp_type bp1
//               mpirun -np 2 ex_pa_tetrahedron --mesh "../../data/inline-tet.mesh" --bp_type bp3
//
// Description: This miniapp illustrates the performance of partial assembly routines
// on simplices for the mass and diffusion integrators.
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;
using namespace mfem;

real_t m_func(const Vector &x);
real_t d_func(const Vector &x);
real_t f_func(const Vector &x);
real_t u_ex(const Vector &x);
void up_ex(const Vector &x, Vector &up);
void dm_func(const Vector &x, DenseMatrix &D);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command line options.
   string mesh_file = "../../data/inline-tet.mesh";
   string filename = "pa_simplices_results";
   int order = 4;
   int nrefs = 1;
   int max_iters = 2000;
   const char *device_config = "cpu";
   string bp_type = "bp1";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&nrefs, "-nr", "--nrefs", "Number of mesh refinements");
   args.AddOption(&max_iters, "-it", "--maxiters", "Maximum CG iterations");
   args.AddOption(&filename, "-f", "--filename", "Name of save file");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&bp_type, "-bp", "--bp_type",
                  "Type of BP, only bp1 or bp3 supported");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      // currently not doing any initial refinements
      int ref_levels = 0;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = nrefs;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }
   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   fec = new H1_FECollection(order, dim, BasisType::Positive);
   delete_fec = true;
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   HYPRE_BigInt nElems = pmesh.GetNE();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
      cout << "Number of elements: " << nElems << endl;
   }

   if (size > 150000000)
   {
      cout << "Too many DOFs!" << endl;
      return 0;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the external boundary attributes from the mesh as
   //    essential (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (bp_type == "bp3")
   {
      if (pmesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh.bdr_attributes.Max());
         ess_bdr = 0;
         // Apply boundary conditions on all external boundaries:
         pmesh.MarkExternalBoundaries(ess_bdr);
         // Boundary conditions can also be applied based on named attributes:
         // pmesh.MarkNamedBoundaries(set_name, ess_bdr)

         fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
   }
   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   const FiniteElement *el = fespace.GetFE(0);
   const IntegrationRule ir_b = IntRules.Get(el->GetGeomType(), 2*order);
   // FunctionCoefficient fcoeff(f_func);
   FunctionCoefficient fcoeff([dim, bp_type](const Vector &x)
   {
      if (dim == 2)
      {
         real_t xi(x(0));
         real_t yi(x(1));
         if (bp_type == "bp1")
         {
            return xi * (1.0 - xi) * yi * (1.0 - yi);
         }
         else if (bp_type == "bp3")
         {
            return 2.0 * (xi * (1.0-xi) +  yi * (1.0 - yi));
         }
         else
         {
            MFEM_ABORT("BP not implemented!")
         }
      }
      else if (dim == 3)
      {
         real_t xi(x(0));
         real_t yi(x(1));
         real_t zi(x(2));
         if (bp_type == "bp1")
         {
            return xi * (1.0 - xi) * yi * (1.0 - yi) * zi * (1.0 - zi);
         }
         else if (bp_type == "bp3")
         {
            return 2.0 * (xi * (1.0-xi) * yi * (1.0 - yi) + xi * (1.0 - xi) * zi *
                          (1.0 - zi) + yi * (1.0 - yi) * zi * (1.0 - zi));
         }
         else
         {
            MFEM_ABORT("BP not implemented!")
         }
      }
      else
      {
         MFEM_ABORT("Problem not implemented for this dimension!")
      }
   });
   b.AddDomainIntegrator(new DomainLFIntegrator(fcoeff, &ir_b));
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
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   if (bp_type == "bp3")
   {
      static FunctionCoefficient diffcoeff(d_func);
      a.AddDomainIntegrator(new DiffusionIntegrator(diffcoeff));
   }
   else if (bp_type == "bp1")
   {
      static FunctionCoefficient masscoeff(m_func);
      a.AddDomainIntegrator(new MassIntegrator(masscoeff));
   }
   else
   {
      MFEM_ABORT("Specified BP is not implemented!!");
   }

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   // if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use Jacobi smoothing, for now.
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(max_iters);
   cg.SetPrintLevel(0);
   cg.SetOperator(*A);
   MFEM_DEVICE_SYNC;
   auto start = std::chrono::steady_clock::now();
   cg.Mult(B, X);
   MFEM_DEVICE_SYNC;
   auto end = std::chrono::steady_clock::now();
   std::chrono::duration<double> diff = end - start;
   int num_iters = cg.GetNumIterations();

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // 15. Compute error.
   FunctionCoefficient uexact([dim](const Vector &x)
   {
      if (dim == 2)
      {
         real_t xi(x(0));
         real_t yi(x(1));
         return xi * (1.0 - xi) * yi * (1.0 - yi);
      }
      else if (dim == 3)
      {
         real_t xi(x(0));
         real_t yi(x(1));
         real_t zi(x(2));
         return xi * (1.0 - xi) * yi * (1.0 - yi) * zi * (1.0 - zi);
      }
      else
      {
         MFEM_ABORT("Problem not implemented for this dimension!")
      }
   });
   VectorFunctionCoefficient upexact(dim, [dim](const Vector &x, Vector &up)
   {
      if (dim == 2)
      {
         real_t xi(x(0));
         real_t yi(x(1));

         up(0) = (1.0 - 2.0 * xi) * yi * (1.0 - yi);
         up(1) = (1.0 - 2.0 * yi) * xi * (1.0 - xi);
      }
      else if (dim == 3)
      {
         real_t xi(x(0));
         real_t yi(x(1));
         real_t zi(x(2));

         up(0) = (1.0 - 2.0 * xi) * yi * (1.0 - yi) * zi * (1.0 - zi);
         up(1) = (1.0 - 2.0 * yi) * xi * (1.0 - xi) * zi * (1.0 - zi);
         up(2) = (1.0 - 2.0 * zi) * xi * (1.0 - xi) * yi * (1.0 - yi);
      }
      else
      {
         MFEM_ABORT("Problem not implemented for this dimension!")
      }
   });
   real_t L2errSol = x.ComputeL2Error(uexact);
   real_t H1errSol = x.ComputeGradError(&upexact);
   if (myid == 0)
   {
      cout << "L2 error: " << L2errSol << ", H1semi error: " << H1errSol <<
           ", time: " << diff.count() << ", iters: " << num_iters << ", DOFS/s: " <<
           size / (diff.count() / num_iters) << endl;

      fstream file;
      file.open(filename + ".txt", std::ios::app);
      // file << bp_type << " ";
      file << "p=" << order << ", time=" << diff.count() << ", iters=" << num_iters <<
           ", dofs=" << size << ", dofs/s=" << size / (diff.count() / num_iters);
      file << endl;
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}

real_t m_func(const Vector &x)
{
   return 1.0;
}

real_t d_func(const Vector &x)
{
   return 1.0;
}

// void dm_func(const Vector &x, DenseMatrix &D)
// {
//    real_t xi(x(0));
//    real_t yi(x(1));
//    real_t zi(x(2));

//    D(0,0) = 1.0;
//    D(0,1) = 0.0;
//    D(0,2) = 0.0;
//    D(1,0) = 0.0;
//    D(1,1) = 1.0;
//    D(1,2) = 0.0;
//    D(2,0) = 0.0;
//    D(2,1) = 0.0;
//    D(2,2) = 1.0;
// }