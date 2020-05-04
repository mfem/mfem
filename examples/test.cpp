//                                MFEM Example 9
//
// Compile with: make serial_nogpu
//
// Description:  This code solves the time-dependent advection-diffusion
//               equation:
//               \frac(\partial u}{\partial t}
//                                 = \mathbf{a} \cdot \Nabla u - \nu \Nabla^2 u
//               where a is a given advection velocity, \nu is the diffusion
//               parameter, and u0(x) = u(0,x) is a given initial condition.
//
//               The demonstrates explicit time marching with H1 elements of
//               arbitrary order. Periodic boundary conditions are used through
//               periodic meshes. GLVis can be used for visualization of a
//               time-evolving solution.

#include <fstream>
#include <iostream>
#include <algorithm>

#include "mfem.hpp"
#include "mpi.h"

#include "evolver.hpp"

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &X, Vector &v);

// Initial condition
double u0_function(const Vector &X);

// Inflow boundary condition
double inflow_function(const Vector &X, const double t);

// Mesh bounding box
Vector bb_min, bb_max;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   problem = 3;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 3;
   const char *device_config = "cpu";
   int ode_solver_type = 22;
   double t_final = 3 * 2*M_PI;
   double dt = 0.01;
   bool glvis = false;
   bool paraview = false;
   int vis_steps = 5;

   double nu_val = 0.001;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&glvis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&nu_val, "-nu", "--nu-value",
                  "Value for \nu, the parameter that controls diffusion.");
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
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. If the mesh is of NURBS type, we convert it
   //    to a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define the finite element space of the given
   //    polynomial order on the refined mesh.
   H1_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // 8. Set up and assemble the bilinear and linear forms corresponding to the
   //    CG discretization.
   /// negative to move the diffusion terms to the right side
   ConstantCoefficient nu(-nu_val);
   ConstantCoefficient one(1.0);
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient u0(u0_function);

   ParBilinearForm *m_pa = new ParBilinearForm(fes);
   ParBilinearForm *k_pa = new ParBilinearForm(fes);
   m_pa->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   k_pa->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   /// create mass matrix
   m_pa->AddDomainIntegrator(new MassIntegrator(one));
   /// add advection terms to stiffness matrix
   k_pa->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   /// add diffusion terms to stiffness matrix
   k_pa->AddDomainIntegrator(new DiffusionIntegrator(nu));

   m_pa->Assemble();
   int skip_zeros = 0;
   k_pa->Assemble(skip_zeros);
   m_pa->Finalize();
   k_pa->Finalize(skip_zeros);

   ParBilinearForm *m = new ParBilinearForm(fes);
   ParBilinearForm *k = new ParBilinearForm(fes);
   /// create mass matrix
   m->AddDomainIntegrator(new MassIntegrator);
   /// add advection terms to stiffness matrix
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   /// add diffusion terms to stiffness matrix
   k->AddDomainIntegrator(new DiffusionIntegrator(nu));

   m->Assemble();
   k->Assemble(skip_zeros);
   m->Finalize();
   k->Finalize(skip_zeros);


   ParGridFunction *u = new ParGridFunction(fes);
   u->UseDevice(true);
   u->ProjectCoefficient(u0);


   HypreParVector *U = u->GetTrueDofs();

   ParSystemOperator pso(*m, *k);
   ParSystemOperator pso_pa(*m_pa, *k_pa);

   pso.setParameters(dt, U);
   pso_pa.setParameters(dt, U);

   mfem::Vector pso_r(U->Size());
   double t1 = MPI_Wtime();
   pso.Mult(*U, pso_r);
   double t2 = MPI_Wtime();
   if (myid == 0)
      std::cout << "FA Mult time: " << t2-t1 << std::endl;

   mfem::Vector pso_pa_r(U->Size());
   double t3 = MPI_Wtime();
   pso_pa.Mult(*U, pso_pa_r);
   double t4 = MPI_Wtime();
   if (myid == 0)
      std::cout << "PA Mult time: " << t4-t3 << std::endl;
   
   double local_mult_speedup = (t2-t1) / (t4-t3);
   double global_mult_speedup;
   MPI_Reduce(&local_mult_speedup, &global_mult_speedup, 1,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   if (myid == 0)
      std::cout << "PA speedup: " << global_mult_speedup / num_procs << endl;

   mfem::Vector diff_r(pso_pa_r);
   diff_r -= pso_r;
   // std::cout << "r diff: " << diff_r.Norml2() << std::endl;

   mfem::Operator &pso_jac = pso.GetGradient(*U);
   mfem::Operator &pso_pa_jac = pso_pa.GetGradient(*U);

   mfem::Vector pso_jac_r(U->Size());
   double t5 = MPI_Wtime();
   pso_jac.Mult(*U, pso_jac_r);
   double t6 = MPI_Wtime();
   if (myid == 0)
      std::cout << "FA Jac Mult time: " << t6-t5 << std::endl;

   mfem::Vector pso_pa_jac_r(U->Size());
   double t7 = MPI_Wtime();
   pso_pa_jac.Mult(*U, pso_pa_jac_r);
   double t8 = MPI_Wtime();
   if (myid == 0)
      std::cout << "PA Jac Mult time: " << t8-t7 << std::endl;

   double local_jac_speedup = (t6-t5) / (t8-t7);
   double global_jac_speedup;
   MPI_Reduce(&local_jac_speedup, &global_jac_speedup, 1,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if (myid == 0)
      std::cout << "PA Jac speedup: " << global_jac_speedup / num_procs << endl;


   mfem::Vector diff_jac(pso_pa_jac_r);
   diff_jac -= pso_jac_r;

   // std::cout << "jac diff: " << diff_jac.Norml2() << std::endl;

   *u = *U;

   // 13. Free the used memory.
   delete U;
   delete u;
   delete k;
   delete m;
   delete fes;
   delete pmesh;

   MPI_Finalize();
   return 0;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 3:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 0:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x, const double t)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
