//                                MFEM Example 0
//
// Compile with: make ex_pa_tetrahedron
//
// Sample runs:  ex_pa_tetrahedron
//               mpirun -np 2 ex_pa_tetrahedron
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Poisson
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
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
   // string mesh_file = "../data/ref-tetrahedron.mesh";
   // string mesh_file = "../data/ref-cube.mesh";
   // string mesh_file = "../data/ref-square.mesh";
   // string mesh_file = "../data/inline-tri.mesh";
   string mesh_file = "../../data/inline-tet.mesh";
   // string mesh_file = "../data/inline-quad.mesh";
   string mesh_type;
   string filename;
   int order = 4;
   int nrefs = 1;
   int max_iters = 2000;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&nrefs, "-nr", "--nrefs", "Number of mesh refinements");
   args.AddOption(&mesh_type, "-t", "--type", "Type of mesh");
   args.AddOption(&max_iters, "-it", "--maxiters", "Type of mesh");
   args.AddOption(&filename, "-f", "--filename", "Name of save file");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   // Device device(device_config);
   // if (myid == 0) { device.Print(); }

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
      // int ref_levels =
      //    (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
      int ref_levels = 2;
      // ref_levels += nrefs;
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

   if (size > 1000000000)
   {
      cout << "Too many DOFs!" << endl;
      return 0;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the external boundary attributes from the mesh as
   //    essential (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
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

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   const FiniteElement *el = fespace.GetFE(0);
   const IntegrationRule ir_b = IntRules.Get(el->GetGeomType(), 2*order);
   FunctionCoefficient fcoeff(f_func);
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
   FunctionCoefficient diffcoeff(d_func);
   a.AddDomainIntegrator(new DiffusionIntegrator(diffcoeff));
   // FunctionCoefficient masscoeff(m_func);
   // a.AddDomainIntegrator(new MassIntegrator(masscoeff));

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
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(0);
   cg.SetOperator(*A);
   auto start = std::chrono::steady_clock::now();
   cg.Mult(B, X);
   auto end = std::chrono::steady_clock::now();
   std::chrono::duration<double> diff = end - start;
   int num_iters = cg.GetNumIterations();

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // 15. Compute error.
   FunctionCoefficient uexact(u_ex);
   VectorFunctionCoefficient upexact(3, up_ex);
   real_t L2errSol = x.ComputeL2Error(uexact);
   real_t H1errSol = x.ComputeGradError(&upexact);
   if (myid == 0)
   {
      cout << "L2 error: " << L2errSol << ", H1semi error: " << H1errSol <<
             ", time: " << diff.count() << ", iters: " << num_iters << ", time per iter: " <<
             diff.count() / num_iters << endl;

      fstream file;
      file.open(filename + ".txt", std::ios::app);
      file << "p=" << order << ", time=" << diff.count() << ", iters=" << num_iters << ", dofs=" << size;
      file << endl;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   // // test ragged tensor loop
   // real_t *Ftest = new real_t[p * p * p] {0};
   // for (int i=0; i < p; i++)
   // {
   //    for (int j=0; j < p; j++)
   //    {
   //       for (int k=0; k < p; k++)
   //       {

   //       }
   //    }
   // }

   return 0;
}

real_t m_func(const Vector &x)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(x(2));

   // return 1.0 / exp(xi + yi);
   return 1.0;
   // return xi * (1.0 - yi);
}

real_t d_func(const Vector &x)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(x(2));

   // return 1.0;
   return 1.0;
}

void dm_func(const Vector &x, DenseMatrix &D)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(x(2));

   D(0,0) = 1.0;
   D(0,1) = 0.0;
   D(0,2) = 0.0;
   D(1,0) = 0.0;
   D(1,1) = 1.0;
   D(1,2) = 0.0;
   D(2,0) = 0.0;
   D(2,1) = 0.0;
   D(2,2) = 1.0;
}

real_t f_func(const Vector &x)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(x(2));

   // return 50.0 * sin(5.0 * (xi + yi));
   // return xi * yi * zi * (1.0 - xi - yi - zi);
   // return 2.0 * (yi * zi + xi * (yi + zi));
   // return xi * (1.0 - xi) * yi * (1.0 - yi) * zi * (1.0 - zi);
   // return 2.0 * exp(xi + yi) * (xi*xi*(1.0+yi)*zi + yi*zi*(yi+zi) + xi*(yi*yi*zi + zi*zi + yi*(1.0 + 3.0*zi + zi*zi)));
   return 2.0 * (xi * (1.0-xi) * yi * (1.0 - yi) + xi * (1.0 - xi) * zi * (1.0 - zi) + yi * (1.0 - yi) * zi * (1.0 - zi));
   // return 2.0 * (xi + 2.0 * yi);
   // return 2.0 * (xi*xi*xi + 2.0*xi*yi*yi + (yi-1.0)*yi*yi + xi*xi*(2.0*yi - 1.0)) * cos(xi*yi) - (-2.0*yi + xi*xi*xi*xi*yi + xi*xi*xi*(yi-1.0)*yi + xi*xi*yi*yi*yi + xi*(-2.0 - yi*yi*yi + yi*yi*yi*yi)) * sin(xi*yi);
   // return 2.0*xi*xi - yi + 4.0*xi*yi + yi*yi;
   // return 0.0;
   // return -2.0;
   // return xi;
   // return 2.0 * (xi * (1.0 - xi) + 2.0 * yi * (1.0 - yi));
   // return xi * (1.0 - xi) * yi * (1.0 - yi);

   // return 12.0 * M_PI * M_PI * sin(2.0*M_PI*xi) * sin(2.0*M_PI*yi) * sin(2.0*M_PI*zi);

   // return xi * (1.0 - yi) * sin(2.0*M_PI*xi) * sin(2.0*M_PI*yi) - 2.0*M_PI * (-xi * sin(2.0*M_PI*xi) * sin(2.0*M_PI*yi) * 2.0*M_PI + cos(2.0*M_PI*xi) * sin(2.0*M_PI*yi) -
   //        exp(yi) * sin(2.0*M_PI*xi) * sin(2.0*M_PI*yi) * 2.0 * M_PI + exp(yi) * sin(2.0*M_PI*xi) * cos(2.0*M_PI*yi));
}

real_t u_ex(const Vector &x)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(x(2));

   // return sin(5.0 * (xi + yi));
   // return xi * yi * zi * (1.0 - xi - yi - zi);// * exp(xi + yi);
   return xi * (1.0 - xi) * yi * (1.0 - yi) * zi * (1.0 - zi);
   // return xi * yi * (1.0 - xi - yi) * sin(xi * yi);
   // return xi;
   // return xi * xi;
   // return xi * xi * xi;
   // return xi * (1.0 - xi) * yi * (1.0 - yi);
   // return sin(2.0*M_PI*xi) * sin(2.0*M_PI*yi) * sin(2.0*M_PI*zi);
}

void up_ex(const Vector &x, Vector &up)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(x(2));

   // up(0) = 5.0 * cos(5.0 * (xi + yi));
   // up(1) = 5.0 * cos(5.0 * (xi + yi));

   // up(0) = yi * (-xi*yi * (-1.0 + xi + yi) * cos(xi*yi) - (-1.0 + 2.0*xi + yi) * sin(xi*yi));
   // up(1) = xi * (-xi*yi * (-1.0 + xi + yi) * cos(xi*yi) - (-1.0 + xi + 2.0*yi) * sin(xi*yi));

   // up(0) = 1.0;
   // up(1) = 0.0;

   // up(0) = 2.0 * xi;
   // up(1) = 0.0;

   // up(0) = 3.0 * xi * xi;
   // up(1) = 0.0;

   // up(0) = (1.0 - 2.0*xi) * yi * (1.0 - yi);
   // up(1) = (1.0 - 2.0*yi) * xi * (1.0 - xi);

   up(0) = yi * zi * (-xi + (1.0 - xi - yi - zi));
   up(1) = xi * zi * (-yi + (1.0 - xi - yi - zi));
   up(2) = xi * yi * (-zi + (1.0 - xi - yi - zi));

   // up(0) = yi * zi * (-xi * exp(xi + yi) + (1.0 - xi - yi - zi) * exp(xi + yi) + xi * (1.0 - xi - yi - zi) * exp(xi + yi));
   // up(1) = xi * zi * (-yi * exp(xi + yi) + (1.0 - xi - yi - zi) * exp(xi + yi) + yi * (1.0 - xi - yi - zi) * exp(xi + yi));
   // up(2) = xi * yi * (-zi * exp(xi + yi) + (1.0 - xi - yi - zi) * exp(xi + yi) + zi * (1.0 - xi - yi - zi) * exp(xi + yi));

   // up(0) = cos(2.0*M_PI*xi) * sin(2.0*M_PI*yi) * 2.0 * M_PI;
   // up(1) = sin(2.0*M_PI*xi) * cos(2.0*M_PI*yi) * 2.0 * M_PI;
}
