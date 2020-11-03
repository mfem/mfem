//            MFEM Distance Function Solver - Parallel Version
//
// Compile with: make distance
//
// Sample runs:
//   Problem 0: point source
//     mpirun -np 4 heat -m ../data/inline-quad.mesh -rs 2 -t 1.0
//
//   Problem 1: level set
//      mpirun -np 4 heat -m ../data/inline-quad.mesh -rs 3 -o 2 -t 1.0 -p 1
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

double surface_level_set(const Vector &x)
{
   const double sine = 0.25 * std::sin(4 * M_PI * x(0));
   return (x(1) >= sine + 0.5) ? 0.0 : 1.0;
}

void DiffuseField(ParGridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator.
   ParBilinearForm *Lap = new ParBilinearForm(field.ParFESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();
   HypreParMatrix *A = Lap->ParallelAssemble();

   HypreSmoother *S = new HypreSmoother(*A,0,smooth_steps);
   S->iterative_mode = true;

   Vector tmp(A->Width());
   field.SetTrueVector();
   Vector fieldtrue = field.GetTrueVector();
   tmp = 0.0;
   S->Mult(tmp, fieldtrue);

   field.SetFromTrueDofs(fieldtrue);

   delete S;
   delete Lap;
}

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

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   ParFiniteElementSpace fespace_vec(&pmesh, &fec, dim);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (myid == 0) { cout << "Number of FE unknowns: " << size << endl; }

   // List of true essential boundary dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   OperatorPtr A;
   Vector B, X;

   // Position of the object of interest.
   ParGridFunction u0(&fespace);
   if (problem == 0)
   {
      DeltaCoefficient dc(0.75, 0.625, 1.0);
      u0.ProjectCoefficient(dc);
   }
   else
   {
      FunctionCoefficient dc(surface_level_set);
      u0.ProjectCoefficient(dc);
      DiffuseField(u0, 5);

      // Transform so that the peak is at 0.5.
      for (int i = 0; i < u0.Size(); i++)
      {
         const double x = u0(i);
         u0(i) = (x < 0.0 || x > 1.0) ? 0.0 : 4.0 * x * (1.0 - x);
      }
   }

   // Solution of the first diffusion step.
   ParGridFunction u(&fespace);
   // Final distance function solution.
   ParGridFunction d(&fespace);

   // Step 1 - diffuse.
   {
      // Set up RHS.
      ParLinearForm b1(&fespace);
      GridFunctionCoefficient u0_coeff(&u0);
      b1.AddDomainIntegrator(new DomainLFIntegrator(u0_coeff));
      b1.Assemble();

      // Diffusion and mass terms in the LHS.
      ParBilinearForm a1(&fespace);
      a1.AddDomainIntegrator(new MassIntegrator);
      const double dt = t_param * dx * dx;
      ConstantCoefficient t_coeff(dt);
      a1.AddDomainIntegrator(new DiffusionIntegrator(t_coeff));
      a1.Assemble();

      // Solve with Dirichlet BC.
      ParGridFunction u_dirichlet(&fespace);
      u_dirichlet = 0.0;
      a1.FormLinearSystem(ess_tdof_list, u_dirichlet, b1, A, X, B);
      Solver *prec = new HypreBoomerAMG;
      cg.SetPreconditioner(*prec);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a1.RecoverFEMSolution(X, b1, u_dirichlet);
      delete prec;

      // Diffusion and mass terms in the LHS.
      ParBilinearForm a_n(&fespace);
      a_n.AddDomainIntegrator(new MassIntegrator);
      a_n.AddDomainIntegrator(new DiffusionIntegrator(t_coeff));
      a_n.Assemble();

      // Solve with Neumann BC.
      ParGridFunction u_neumann(&fespace);
      ess_tdof_list.DeleteAll();
      a_n.FormLinearSystem(ess_tdof_list, u_neumann, b1, A, X, B);
      Solver *prec2 = new HypreBoomerAMG;
      cg.SetPreconditioner(*prec2);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a_n.RecoverFEMSolution(X, b1, u_neumann);
      delete prec2;

      for (int i = 0; i < u.Size(); i++)
      {
         //u(i) = u_neumann(i);
         //u(i) = u_dirichlet(i);
         u(i) = 0.5 * (u_neumann(i) + u_dirichlet(i));
      }
   }

   // Step 2 - normalize the gradient. The x here is only for visualization.
   GradientCoefficient grad_u(u, dim);
   ParGridFunction x(&fespace_vec);
   x.ProjectCoefficient(grad_u);

   // Step 3 - solve for the distance using the normalized gradient.
   {
      // RHS - normalized gradient.
      ParLinearForm b2(&fespace);
      b2.AddDomainIntegrator(new DomainLFGradIntegrator(grad_u));
      b2.Assemble();

      // LHS - diffusion.
      ParBilinearForm a2(&fespace);
      a2.AddDomainIntegrator(new DiffusionIntegrator);
      a2.Assemble();

      // No BC.
      ess_tdof_list.DeleteAll();

      a2.FormLinearSystem(ess_tdof_list, d, b2, A, X, B);

      Solver *prec2 = new HypreBoomerAMG;
      cg.SetPreconditioner(*prec2);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a2.RecoverFEMSolution(X, b2, d);
      delete prec2;
   }

   // Rescale the distance to have minimum at zero.
   double d_min_loc = d.Min();
   double d_min_glob;
   MPI_Allreduce(&d_min_loc, &d_min_glob, 1, MPI_DOUBLE,
                 MPI_MIN, fespace.GetComm());
   d -= d_min_glob;

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      int size = 500;
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock_w(vishost, visport);
      sol_sock_w << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_w.precision(8);
      sol_sock_w << "solution\n" << pmesh << u0;
      sol_sock_w << "window_geometry " << 0 << " " << 0 << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "u0" << "'\n" << flush;

      socketstream sol_sock_u(vishost, visport);
      sol_sock_u << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_u.precision(8);
      sol_sock_u << "solution\n" << pmesh << u;
      sol_sock_u << "window_geometry " << size << " " << 0 << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "u" << "'\n" << flush;

      socketstream sol_sock_x(vishost, visport);
      sol_sock_x << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_x.precision(8);
      sol_sock_x << "solution\n" << pmesh << x;
      sol_sock_x << "window_geometry " << 2*size << " " << 0 << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "X" << "'\n"
                 << "keys evvRj*******A\n" << flush;

      socketstream sol_sock_d(vishost, visport);
      sol_sock_d << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_d.precision(8);
      sol_sock_d << "solution\n" << pmesh << d;
      sol_sock_d << "window_geometry " << size << " " << size << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "Distance" << "'\n"
                 << "keys rRjmm*****\n" << flush;
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

   MPI_Finalize();
   return 0;
}
