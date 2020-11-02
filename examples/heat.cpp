//            MFEM Distance Function Solver - Parallel Version
//
// Compile with: make distance
//
// Sample runs:
//   Problem 0: exact boundary alignment:
//     mpirun -np 4 heat -m ../data/inline-segment.mesh -rs 3 -t 2.0
//     mpirun -np 4 heat -m ../data/inline-quad.mesh -rs 2 -t 2.0
//
//    K. Crane et al:
//    Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class GradientCoefficient : public VectorCoefficient
{
private:
   GridFunction &u;

public:
   GradientCoefficient(GridFunction &u_gf, int dim)
      : VectorCoefficient(dim), u(u_gf) { }

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);

      u.GetGradient(T, V);
      const double norm = V.Norml2() + 1e-12;
      V /= -norm;
   }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int problem = 0;
   int rs_levels = 0;
   int order = 2;
   double t_param = 1.0;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type:\n\t"
                  "0: exact alignment with the mesh boundary\n\t"
                  "1: zero level set enclosing a volume");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&t_param, "-t", "--t-param", "Diffusion time step");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
      MPI_Finalize();
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   // Compute average mesh size (assumes similar cells).
   double area = 0.0, dx;
   const int zones_cnt = mesh.GetNE();
   for (int i = 0; i < zones_cnt; i++) { area += mesh.GetElementVolume(i); }
   switch (mesh.GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT:
         dx = area / zones_cnt; break;
      case Geometry::SQUARE:
         dx = sqrt(area / zones_cnt); break;
      case Geometry::TRIANGLE:
         dx = sqrt(2.0 * area / zones_cnt); break;
      case Geometry::CUBE:
         dx = pow(area / zones_cnt, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         dx = pow(6.0 * area / zones_cnt, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   dx /= order;

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

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
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
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
   ParFiniteElementSpace fespace(&pmesh, fec);
   ParFiniteElementSpace fespace_vec(&pmesh, fec, dim);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (myid == 0) { cout << "Number of FE unknowns: " << size << endl; }

   // List of true essential boundary dofs.
   Array<int> ess_tdof_list;
   /*
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   */

   // Solution x with correct Dirichlet BC.
   ParGridFunction u0(&fespace);
   /*
   w = 0.0;
   Array<int> bdr(pmesh.bdr_attributes.Max()); bdr = 1;
   ConstantCoefficient one(1.0);
   w.ProjectBdrCoefficient(one, bdr);
   */

   DeltaCoefficient dc(0.5, 0.5, 1.0);
   u0.ProjectCoefficient(dc);

   // Set up RHS.
   ParLinearForm b(&fespace);
   b = u0;

   // Diffusion and mass terms in the LHS.
   ParBilinearForm a1(&fespace);
   ConstantCoefficient one(1.0);
   a1.AddDomainIntegrator(new MassIntegrator(one));
   const double dt = t_param * dx * dx;
   ConstantCoefficient t_coeff(dt);
   a1.AddDomainIntegrator(new DiffusionIntegrator(t_coeff));
   a1.Assemble();

   ParGridFunction u(&fespace);
   OperatorPtr A;
   Vector B, X;
   a1.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

   // Solve the linear system A X = B; CG + BoomerAMG.
   Solver *prec = new HypreBoomerAMG;

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(5000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a1.RecoverFEMSolution(X, b, u);

   GradientCoefficient grad_u(u, dim);
   ParGridFunction x(&fespace_vec);
   x.ProjectCoefficient(grad_u);

   ParLinearForm b2(&fespace);
   b2.AddDomainIntegrator(new DomainLFGradIntegrator(grad_u));
   b2.Assemble();

   ParBilinearForm a2(&fespace);
   a2.AddDomainIntegrator(new DiffusionIntegrator(one));
   a2.Assemble();

   ParGridFunction d(&fespace);
   d = 0.0;
   OperatorPtr A2;
   a2.FormLinearSystem(ess_tdof_list, d, b2, A2, X, B);

   CGSolver cg2(MPI_COMM_WORLD);
   cg2.SetRelTol(1e-12);
   cg2.SetMaxIter(5000);
   cg2.SetPrintLevel(1);
   Solver *prec2 = new HypreBoomerAMG;
   if (prec) { cg2.SetPreconditioner(*prec2); }
   cg2.SetOperator(*A2);
   cg2.Mult(B, X);
   delete prec2;
   a2.RecoverFEMSolution(X, b2, d);

   // Rescale the distance to have minimum at zero.
   double d_min_loc = d.Min();
   double d_min_glob;
   MPI_Allreduce(&d_min_loc, &d_min_glob, 1, MPI_DOUBLE,
                 MPI_MIN, fespace.GetComm());
   d -= d_min_glob;

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock_w(vishost, visport);
      sol_sock_w << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_w.precision(8);
      sol_sock_w << "solution\n" << pmesh << u0;
      sol_sock_w << "window_geometry " << 0 << " " << 0 << " "
                                       << 600 << " " << 600 << "\n"
                 << "window_title '" << "u0" << "'\n" << flush;

      socketstream sol_sock_u(vishost, visport);
      sol_sock_u << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_u.precision(8);
      sol_sock_u << "solution\n" << pmesh << u;
      sol_sock_u << "window_geometry " << 600 << " " << 0 << " "
                                       << 600 << " " << 600 << "\n"
                 << "window_title '" << "u" << "'\n" << flush;

      socketstream sol_sock_x(vishost, visport);
      sol_sock_x << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_x.precision(8);
      sol_sock_x << "solution\n" << pmesh << x;
      sol_sock_x << "window_geometry " << 1200 << " " << 0 << " "
                                       << 600 << " " << 600 << "\n"
                 << "window_title '" << "X" << "'\n" << flush;

      socketstream sol_sock_d(vishost, visport);
      sol_sock_d << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_d.precision(8);
      sol_sock_d << "solution\n" << pmesh << d;
      sol_sock_d << "window_geometry " << 600 << " " << 600 << " "
                                       << 600 << " " << 600 << "\n"
                 << "window_title '" << "Distance" << "'\n" << flush;
   }

   ParaViewDataCollection paraview_dc("Dist", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("w",&u0);
   paraview_dc.RegisterField("u",&u);
   paraview_dc.Save();

   if (delete_fec) { delete fec; }

   MPI_Finalize();
   return 0;
}
