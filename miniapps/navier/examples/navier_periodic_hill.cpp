// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// 3D flow over a cylinder benchmark example

#include "navier_solver.hpp"
#include "prandtl_kolmogorov.hpp"
#include "dist_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 4;
   double kin_vis = 1.0 / 5000.0;
   int nsteps = 5000;
   double dt = 1e-8;
   double dt_max = 5e-3;
   double cfl_target = 0.8;
   double t_final = 50.0;
   double Uavg = 1.0;
   double H = 1.0;
   double Lx = 9.0*H;
   double Ly = 3.035*H;
   double beta_x = 2.0;
   double beta_y = 2.4;
   double W = 1.929;
   bool les = false;
   bool rans = true;
} ctx;

// See Almeida et al. 1993 and
// https://turbmodels.larc.nasa.gov/Other_LES_Data/2Dhill_periodic/hill-geometry.dat

inline double hill_step(double x, double w, double h)
{
   double y, xs = x / w;
   if (xs <= 0.0)
   {
      y = h;
   }
   else if (xs > 0.0 && xs <= 9.0/54.0)
   {
      y = h*std::min(1.0,1.0+7.05575248e-1*pow(xs,2.0)-1.1947737203e1*pow(xs,3.0));
   }
   else if (xs > 9.0/54.0 && xs <= 14.0/54.0)
   {
      y = h*(0.895484248+1.881283544*xs-10.582126017*pow(xs,2.0)
             +10.627665327*pow(xs,3.0));
   }
   else if ((xs>14.0/54.0) && (xs<=20.0/54.0))
   {
      y = h*(0.92128609+1.582719366*xs-9.430521329*pow(xs,2.0)
             +9.147030728*pow(xs,3.0));
   }
   else if ((xs>20.0/54.0) && (xs<=30.0/54.0))
   {
      y = h*(1.445155365-2.660621763*xs+2.026499719*pow(xs,2.0)
             -1.164288215*pow(xs,3.0));
   }
   else if ((xs>30.0/54.0) && (xs<=40.0/54.0))
   {
      y = h*(0.640164762+1.6863274926*xs-5.798008941*pow(xs,2.0)
             +3.530416981*pow(xs,3.0));
   }
   else if ((xs>40.0/54.0) && (xs<=1.0))
   {
      y = h*(2.013932568-3.877432121*xs+1.713066537*pow(xs,2.0)
             +0.150433015*pow(xs,3.0));
   }
   else
   {
      y = 0.0;
   }
   return y;
}
inline double hill_height(double x, double Lx, double w, double h)
{
   double xx = 0.0;
   if (x < 0.0)
   {
      xx = Lx + std::fmod(x, Lx);
   }
   else if (x > Lx)
   {
      xx = std::fmod(x, Lx);
   }
   else
   {
      xx = x;
   }

   return hill_step(xx,w,h) + hill_step(Lx-xx,w,h);
}

void vel_inlet(const Vector &c, double t, Vector &u)
{
   double x = c(0);
   double y = c(1);

   u(0) = ctx.Uavg;
   u(1) = 0.0;
}

void vel_ic(const Vector &c, Vector &u)
{
   double x = c(0);
   double y = c(1);
   double z = 0.0;

   double amp = 0.2, ran = 0.0;

   ran = 3.e4*(x*sin(y)+z*cos(y));
   ran = 6.e3*sin(ran);
   ran = 3.e3*sin(ran);
   ran = cos(ran);
   u(0) = 1. + ran*amp;

   ran = (2+ran)*1.e4*(y*sin(z)+x*cos(z));
   ran = 2.e3*sin(ran);
   ran = 7.e3*sin(ran);
   ran = cos(ran);
   u(1) = ran*amp;
}

void velocity_wall(const Vector &c, const double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
}

bool in_between(const double x, const double x0, const double x1)
{
   double tol = 1e-8;
   return (fabs(x - x0) >= tol) && (fabs(x - x1) >= tol);
}

void flowrate(const Vector &, double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
}

class RANSInitCoefficient : public Coefficient
{
public:
   RANSInitCoefficient(const ParGridFunction &u, const double Re) : u(u)
   {
      turbulence_intensity = 0.15 * pow(Re, -1.0/8.0);
   }

   double  Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector val;
      u.GetVectorValue(T, ip, val);
      const double u_mag = val.Norml2();
      return 1.5 * u_mag * u_mag * turbulence_intensity * turbulence_intensity;
   }

private:
   const ParGridFunction &u;
   double turbulence_intensity;
};

class EddyViscosityCoefficient : public Coefficient
{
public:
   EddyViscosityCoefficient(const ParGridFunction &k,
                            const ParGridFunction &wall_distance,
                            const double L,
                            const double tau,
                            const double mu) :
      k(k),
      wall_distance(wall_distance),
      L(L),
      tau(tau),
      mu(mu) {}

   double  Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      const double k_q = k.GetValue(T, ip);
      const double wd_q = wall_distance.GetValue(T, ip);

      const double sqrtk = sqrt(.5*(k_q+abs(k_q)));
      const double d = 0.41*wd_q*sqrt(wd_q/L);
      const double l = 0.5*(d+sqrtk*sqrt(2)*tau-abs(d-sqrt(2)*sqrtk*tau));

      return mu * sqrtk * l;
   }

private:
   const ParGridFunction &k;
   const ParGridFunction &wall_distance;
   const double L, tau, mu;
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *device_config = "cpu";
   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // int serial_refinements = 1;
   // int nx = 11;
   // int ny = 8;
   int serial_refinements = 1;
   int nx = 8;
   int ny = 11;

   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL);

   mesh.EnsureNodes();
   H1_FECollection mesh_fec(ctx.order, mesh.Dimension());
   FiniteElementSpace mesh_fes(&mesh, &mesh_fec, mesh.Dimension());
   mesh.SetNodalFESpace(&mesh_fes);

   // See Nek5000 example. Decrease resolution in the high velocity regions
   // (increase CFL) and increase resolution near the wall.
   mesh.Transform([](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);

      u(0) = 0.5*(sinh(ctx.beta_x*(x-0.5))/sinh(ctx.beta_x*0.5) + 1.0);
      u(1) = 0.5*(tanh(ctx.beta_y*(2.0*y-1.0))/tanh(ctx.beta_y) + 1.0);
   });

   // Rescale to [0,Lx]x[0,Ly]
   mesh.Transform([](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);

      double xscale = ctx.Lx;
      double yscale = ctx.Ly;

      u(0) = xscale * x;
      u(1) = yscale * y;
   });

   // Shift points in x
   mesh.Transform([](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);

      double amp = 0.25;
      double xfac = 0.0;
      double yfac = pow((1.0-y/ctx.Ly), 3.0);

      if (x <= ctx.W/2.0)
      {
         xfac = -2.0/ctx.W * x;
      }
      else if ((x > ctx.W/2.0) && (x <= ctx.Lx-ctx.W/2.0))
      {
         xfac = 2.0/(ctx.Lx-ctx.W) * x - 1.0 - ctx.W/(ctx.Lx-ctx.W);
      }
      else if (x > (ctx.Lx-ctx.W/2.0))
      {
         xfac = -2.0/ctx.W * x + 2.0*ctx.Lx/ctx.W;
      }

      double shift = xfac*yfac;

      u(0) = x + amp * shift;
      u(1) = y;
   });

   mesh.Transform([](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);

      const double yh = hill_height(x, ctx.Lx, ctx.W, ctx.H);

      u(0) = x;
      u(1) = yh + y * (1.0 - yh / ctx.Ly);
   });


   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   Vector x_translation({ctx.Lx, 0.0});
   std::vector<Vector> translations = {x_translation};
   Mesh periodic_mesh = Mesh::MakePeriodic(mesh,
                                           mesh.CreatePeriodicVertexMapping(translations));

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   // char buffer[50];
   // sprintf(buffer,"phill_%d_%d_%d.mesh", nx, ny, serial_refinements);
   // mesh.Save(buffer);

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);

   // Create the flow solver.
   NavierSolver navier(pmesh, ctx.order, ctx.kin_vis);

   // auto kv_gf = navier.GetVariableViscosity();
   // ConstantCoefficient kv_coeff(ctx.kin_vis);
   // kv_gf->ProjectCoefficient(kv_coeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> wall_attr(pmesh->bdr_attributes.Max());
   wall_attr = 0;
   wall_attr[0] = 1;
   wall_attr[2] = 1;

   // Set the initial condition.
   ParGridFunction *u_ic = navier.GetCurrentVelocity();
   VectorFunctionCoefficient u_ic_coeff(pmesh->Dimension(), vel_ic);
   u_ic->ProjectCoefficient(u_ic_coeff);

   // Walls
   VectorFunctionCoefficient velocity_wall_coeff(pmesh->Dimension(),
                                                 velocity_wall);
   u_ic->ProjectBdrCoefficient(velocity_wall_coeff, wall_attr);

   navier.AddVelDirichletBC(velocity_wall, wall_attr);

   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   navier.AddAccelTerm(flowrate, domain_attr);

   double t = 0.0;
   double t_rans = t;
   double dt = ctx.dt;
   double dt_rans = dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   if (ctx.les)
   {
      navier.EnableFilter(NavierSolver::FilterMethod::HPFRT_LPOLY);
   }

   navier.Setup(dt);

   ParGridFunction *u_next_gf = nullptr;
   ParGridFunction *u_gf = navier.GetCurrentVelocity();
   ParGridFunction *p_gf = navier.GetCurrentPressure();

   std::shared_ptr<RANSModel> rans_model;
   std::shared_ptr<ODESolver> rans_ode;
   ParGridFunction *wall_distance_gf = nullptr, *k_gf = nullptr,
                    *nu_t_gf = nullptr;
   ConstantCoefficient zero_coeff(0.0);
   VectorGridFunctionCoefficient vel_coeff(navier.GetCurrentVelocity());
   GridFunctionCoefficient kv_coeff(navier.GetVariableViscosity());
   GridFunctionCoefficient *wall_distance_coeff = nullptr;
   EddyViscosityCoefficient *nu_t_coeff = nullptr;
   ParGridFunction kv_orig(*navier.GetVariableViscosity());
   Vector k_tdof;
   if (ctx.rans)
   {
      ParFiniteElementSpace &kfes = *navier.GetVariableViscosity()->ParFESpace();
      k_gf = new ParGridFunction(&kfes);
      *k_gf = 1.0;
      k_gf->ProjectBdrCoefficient(zero_coeff, wall_attr);
      k_tdof.SetSize(kfes.GetTrueVSize());
      k_tdof = 0.0;
      k_gf->ParallelProject(k_tdof);

      nu_t_gf = new ParGridFunction(&kfes);
      *nu_t_gf = 0.0;

      // Compute wall distance function
      wall_distance_gf = new ParGridFunction(&kfes);

      ParGridFunction ls_gf(&kfes);
      ls_gf = 1.0;
      ls_gf.ProjectBdrCoefficient(zero_coeff, wall_attr);

      GridFunctionCoefficient ls_coeff(&ls_gf);

      const int p = 5;
      const int newton_iter = 50;
      PLapDistanceSolver dist_solver(p, newton_iter);
      dist_solver.print_level = 0;
      dist_solver.ComputeScalarDistance(ls_coeff, *wall_distance_gf);

      wall_distance_coeff = new GridFunctionCoefficient(wall_distance_gf);

      const double mu_calibration_const = 0.55;

      rans_model.reset(new PrandtlKolmogorov(
                          kfes,
                          vel_coeff,
                          kv_coeff,
                          zero_coeff,
                          *wall_distance_coeff,
                          zero_coeff,
                          mu_calibration_const,
                          wall_attr));

      rans_ode.reset(new ARKStepSolver(MPI_COMM_WORLD, ARKStepSolver::IMEX));
      rans_ode->Init(*rans_model);

      int flag = 0;
      ARKStepSetPostprocessStepFn(static_cast<ARKStepSolver *>
                                  (rans_ode.get())->GetMem(),
                                  PrandtlKolmogorov::PostProcessCallback);
      MFEM_VERIFY(flag >= 0, "error in ARKStepSetPostprocessStepFn()");
      ARKStepSetPostprocessStageFn(static_cast<ARKStepSolver *>
                                   (rans_ode.get())->GetMem(),
                                   PrandtlKolmogorov::PostProcessCallback);
      MFEM_VERIFY(flag >= 0, "error in ARKStepSetPostprocessStageFn()");
      ARKStepSetStagePredictFn(static_cast<ARKStepSolver *>
                               (rans_ode.get())->GetMem(),
                               PrandtlKolmogorov::PostProcessCallback);
      MFEM_VERIFY(flag >= 0, "error in ARKStepSetStagePredictFn()");
      static_cast<ARKStepSolver *>(rans_ode.get())->SetIMEXTableNum(
         ARK324L2SA_ERK_4_2_3, ARK324L2SA_DIRK_4_2_3);

      nu_t_coeff = new EddyViscosityCoefficient(*k_gf, *wall_distance_gf, 9.0,
                                                ctx.dt_max, mu_calibration_const);
      nu_t_gf->ProjectCoefficient(*nu_t_coeff);
   }

   ParaViewDataCollection pvdc("output/phill", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   if (ctx.rans)
   {
      pvdc.RegisterField("wall_distance", wall_distance_gf);
      pvdc.RegisterField("k", k_gf);
      pvdc.RegisterField("nu_t", nu_t_gf);
   }
   pvdc.Save();

   bool rans_init = true;

   int step = 0;
   auto write_output = [&]()
   {
      if (Mpi::Root())
      {
         printf("Writing output...\n");
      }
      if (ctx.rans)
      {
         k_gf->Distribute(k_tdof);
      }
      pvdc.SetCycle(step);
      pvdc.SetTime(t);
      pvdc.Save();
   };

   for (step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      if (ctx.rans && t >= 5.0)
      {
         t_rans = t;
         // TODO: Calling Init here to reset (not implemented in the interface)
         if (rans_init)
         {
            RANSInitCoefficient init_coeff(*u_gf, 1.0/ctx.kin_vis);
            k_gf->ProjectCoefficient(init_coeff);
            k_gf->ParallelProject(k_tdof);
            write_output();
            rans_init = false;
         }
         rans_ode->Init(*rans_model);
         rans_ode->Step(k_tdof, t_rans, dt_rans);
         nu_t_gf->ProjectCoefficient(*nu_t_coeff);

         *navier.GetVariableViscosity() = kv_orig;
         *navier.GetCurrentVelocity() += *nu_t_gf;
      }

      navier.Step(t, dt, step, true);
      // Get a prediction for a stable timestep
      int ok = navier.PredictTimestep(1e-8, ctx.dt_max, ctx.cfl_target, dt);
      if (ok < 0)
      {
         // Reject the time step
         if (Mpi::Root())
         {
            std::cout
                  << "Step reached maximum CFL or predicted CFL, retrying with smaller step size."
                  << std::endl;
         }
      }
      else
      {
         // Queue new time step in the history array
         navier.UpdateTimestepHistory(dt);
         // Accept the time step
         t += dt;
      }

      // Compute the CFL
      double cfl = navier.ComputeCFL(*u_gf, dt);

      if (step % 50 == 0)
      {
         write_output();
      }

      if (Mpi::Root())
      {
         printf("%11s %11s %11s %11s\n", "Step", "Time", "dt", "CFL");
         printf("%11d %.5E %.5E %.5E\n", step, t, dt, cfl);
         fflush(stdout);
      }
   }

   navier.PrintTimingData();

   delete pmesh;

   return 0;
}
