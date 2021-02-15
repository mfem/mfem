//            MFEM Distance Function Solver - Parallel Version
//
// Compile with: make distance
//
// Sample runs:
//   Problem 0: point source.
//     mpirun -np 4 distance -m ./corners.mesh -p 0 -rs 3 -t 200.0
//
//   Problem 1: zero level set - circle / sphere at the center of the mesh
//     mpirun -np 4 distance -m ../../data/inline-quad.mesh -rs 3 -o 2 -t 2.0 -p 1
//     mpirun -np 4 distance -m ../../data/periodic-cube.mesh -rs 2 -o 2 -t 1.0 -p 1 -s 1
//
//   Problem 2: zero level set - perturbed sine.
//     mpirun -np 4 distance -m ../../data/inline-quad.mesh -rs 3 -o 2 -t 2.0 -p 2
//
//   Problem 3: level set - Gyroid.
//      mpirun -np 4 distance -m ../../data/periodic-square.mesh -rs 5 -o 2 -t 1.0 -p 3
//      mpirun -np 4 distance -m ../../data/periodic-cube.mesh -rs 3 -o 2 -t 1.0 -p 3
//

#include <fstream>
#include <iostream>
#include "dist_solver.hpp"

using namespace std;
using namespace mfem;

double sine_ls(const Vector &x)
{
   const double sine = 0.25 * std::sin(4 * M_PI * x(0)) +
                       0.05 * std::sin(16 * M_PI * x(0));
   return (x(1) >= sine + 0.5) ? -1.0 : 1.0;
}

double sphere_ls(const Vector &x)
{
   const int dim = x.Size();
   if (dim == 2)
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      return (r >= 0.4) ? -1.0 : 1.0;
   }
   else if (dim == 3)
   {
      const double xc = x(0) - 0.0, yc = x(1) - 0.0, zc = x(2) - 0.0;
      const double r = sqrt(xc*xc + yc*yc + zc*zc);
      return (r >= 0.8) ? -1.0 : 1.0;
   }
   else
   {
      return (x(0) >= 0.5) ? -1.0 : 1.0;
   }
}

double Gyroid(const Vector &xx)
{
   const double period = 2.0 * M_PI;
   double x=xx[0]*period;
   double y=xx[1]*period;
   double z=0.0;
   if (xx.Size()==3)
   {
      z=xx[2]*period;
   }
   return std::sin(x)*std::cos(y) +
          std::sin(y)*std::cos(z) +
          std::sin(z)*std::cos(x);
}

double Sph(const mfem::Vector &xx)
{
   double R=0.4;
   mfem::Vector lvec(3);
   lvec=0.0;
   for (int i=0; i<xx.Size(); i++)
   {
      lvec[i]=xx[i];
   }

   return lvec[0]*lvec[0]+lvec[1]*lvec[1]+lvec[2]*lvec[2]-R*R;
}

void DGyroid(const mfem::Vector &xx, mfem::Vector &vals)
{
   vals.SetSize(xx.Size());
   vals=0.0;

   double pp=4*M_PI;

   mfem::Vector lvec(3);
   lvec=0.0;
   for (int i=0; i<xx.Size(); i++)
   {
      lvec[i]=xx[i]*pp;
   }

   vals[0]=cos(lvec[0])*cos(lvec[1])-sin(lvec[2])*sin(lvec[0]);
   vals[1]=-sin(lvec[0])*sin(lvec[1])+cos(lvec[1])*cos(lvec[2]);
   if (xx.Size()>2)
   {
      vals[2]=-sin(lvec[1])*sin(lvec[2])+cos(lvec[2])*cos(lvec[0]);
   }

   vals*=pp;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int solver_type = 0;
   int problem = 1;
   int rs_levels = 2;
   int order = 2;
   double t_param = 1.0;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&solver_type, "-s", "--solver",
                  "Solver type:\n\t"
                  "0: Heat\n\t"
                  "1: P-Laplacian");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type:\n\t"
                  "0: Point source\n\t"
                  "1: Circle / sphere level set in 2D / 3D\n\t"
                  "2: 2D sine-looking level set\n\t"
                  "3: Gyroid level set in 2D or 3D");
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
   int smooth_steps;
   if (problem == 0)
   {
      ls_coeff = new DeltaCoefficient(0.5, -0.5, 1000.0);
      smooth_steps = 0;
   }
   else if (problem == 1)
   {
      ls_coeff = new FunctionCoefficient(sphere_ls);
      smooth_steps = 5;
   }
   else if (problem == 2)
   {
      ls_coeff = new FunctionCoefficient(sine_ls);
      smooth_steps = 1;
   }
   else
   {
      ls_coeff = new FunctionCoefficient(Gyroid);
      smooth_steps = 0;
   }

   DistanceSolver *dist_solver = NULL;
   if (solver_type == 0)
   {
      auto ds = new HeatDistanceSolver(t_param);
      if (problem == 0)
      {
         ds->transform = false;
         ds->diffuse_iter = 1;
      }
      ds->smooth_steps = smooth_steps;
      ds->vis_glvis = false;
      dist_solver = ds;
   }
   else if (solver_type == 1)
   {
      const int p = 10;
      const int newton_iter = 50;
      auto ds = new PLapDistanceSolver(p, newton_iter);
      dist_solver = ds;
   }
   else { MFEM_ABORT("Wrong solver option."); }
   dist_solver->print_level = 1;

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes_s(&pmesh, &fec), pfes_v(&pmesh, &fec, dim);
   ParGridFunction distance_s(&pfes_s), distance_v(&pfes_v);
   dist_solver->ComputeScalarDistance(*ls_coeff, distance_s);
   dist_solver->ComputeVectorDistance(*ls_coeff, distance_v);

   ParGridFunction input_ls(distance_s.ParFESpace());
   input_ls.ProjectCoefficient(*ls_coeff);
   delete ls_coeff;

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      int size = 500;
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock_w(vishost, visport);
      sol_sock_w << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_w.precision(8);
      sol_sock_w << "solution\n" << pmesh << input_ls;
      sol_sock_w << "window_geometry " << 0 << " " << 0 << " "
                 << size << " " << size << "\n"
                 << "window_title '" << "Input Level Set" << "'\n" << flush;

      socketstream sol_sock_ds(vishost, visport);
      sol_sock_ds << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_ds.precision(8);
      sol_sock_ds << "solution\n" << pmesh << distance_s;
      sol_sock_ds << "window_geometry " << size << " " << 0 << " "
                  << size << " " << size << "\n"
                  << "window_title '" << "Distance" << "'\n"
                  << "keys rRjmm********\n" << flush;

      socketstream sol_sock_dv(vishost, visport);
      sol_sock_dv << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_dv.precision(8);
      sol_sock_dv << "solution\n" << pmesh << distance_v;
      sol_sock_dv << "window_geometry " << 2*size << " " << 0 << " "
                  << size << " " << size << "\n"
                  << "window_title '" << "Directions" << "'\n"
                  << "keys rRjmm********vve\n" << flush;
   }

   // Paraview output.
   ParaViewDataCollection dacol("ParaViewDistance", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("level_set", &input_ls);
   dacol.RegisterField("distance", &distance_s);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();

   ConstantCoefficient zero(0.0);
   const double d_norm  = distance_s.ComputeL2Error(zero);
   if (myid == 0)
   {
      cout << fixed << setprecision(10) << "Norm: " << d_norm << std::endl;
   }

   delete dist_solver;

   MPI_Finalize();
   return 0;
}
