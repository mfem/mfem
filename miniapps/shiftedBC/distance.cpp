//            MFEM Distance Function Solver - Parallel Version
//
// Compile with: make distance
//
// Sample runs:
//   Problem 0: point sources
//     mpirun -np 4 distance -m ./u5.mesh -rs 2 -t 50.0
//
//   Problem 1: level sets
//      mpirun -np 4 distance -m ../../data/inline-quad.mesh -rs 3 -o 2 -t 1.0 -p 1
//      mpirun -np 4 distance -m ../../data/periodic-square.mesh -rs 5 -o 2 -t 1.0 -p 2
//      mpirun -np 4 distance -m ../../data/periodic-cube.mesh -rs 3 -o 2 -t 1.0 -p 2
//
//
//    K. Crane et al:
//    Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow

#include "distfunction.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double surface_level_set(const Vector &x)
{
   const double sine = 0.25 * std::sin(4 * M_PI * x(0));
   return (x(1) >= sine + 0.5) ? 0.0 : 1.0;
}

double Gyroid(const Vector & xx)
{
   const double period = 4.0 * M_PI;
   double x=xx[0]*period;
   double y=xx[1]*period;
   double z=0.0;
   if(xx.Size()==3)
   {
      z=xx[2]*period;
   }
   return std::sin(x)*std::cos(y) +
          std::sin(y)*std::cos(z) +
          std::sin(z)*std::cos(x);
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

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   Coefficient *ls_coeff;
   int smooth_steps; bool transform;
   if (problem == 0)
   {
      ls_coeff = new DeltaCoefficient(0.0, 0.0, 1.0);
      smooth_steps = 0;
      transform = false;
   }
   else if (problem == 1)
   {
      ls_coeff = new FunctionCoefficient(surface_level_set);
      smooth_steps = 5;
      transform = true;
   }
   else
   {
      ls_coeff = new FunctionCoefficient(Gyroid);
      smooth_steps = 0;
      transform = true;
   }
   DistanceFunction dist_func(pmesh, order, t_param);
   ParGridFunction &distance = dist_func.ComputeDistance(*ls_coeff,
                                                         smooth_steps, transform);
   const ParGridFunction &src = dist_func.GetLastSourceGF(),
                         &diff_src = dist_func.GetLastDiffusedSourceGF();
   delete ls_coeff;

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace_vec(&pmesh, &fec, dim);
   GradientCoefficient grad_u(dist_func.GetLastDiffusedSourceGF(), dim);
   ParGridFunction x(&fespace_vec);
   x.ProjectCoefficient(grad_u);

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      int size = 500;
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock_w(vishost, visport);
      sol_sock_w << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_w.precision(8);
      sol_sock_w << "solution\n" << pmesh << src;
      sol_sock_w << "window_geometry " << 0 << " " << 0 << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "u0" << "'\n" << flush;

      socketstream sol_sock_u(vishost, visport);
      sol_sock_u << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_u.precision(8);
      sol_sock_u << "solution\n" << pmesh << diff_src;
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
      sol_sock_d << "solution\n" << pmesh << distance;
      sol_sock_d << "window_geometry " << size << " " << size << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "Distance" << "'\n"
                 << "keys rRjmm*****\n" << flush;
   }

   /*
   ParaViewDataCollection paraview_dc("Dist", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("w", &src);
   paraview_dc.RegisterField("u", &diff_src);
   paraview_dc.Save();
   */

   ConstantCoefficient zero(0.0);
   const double u0_norm = src.ComputeL2Error(zero),
                u_norm  = diff_src.ComputeL2Error(zero),
                d_norm  = distance.ComputeL2Error(zero);
   if (myid == 0)
   {
     std::cout <<  u0_norm << " "<< u_norm << " " << d_norm << std::endl;
   }

   MPI_Finalize();
   return 0;
}
