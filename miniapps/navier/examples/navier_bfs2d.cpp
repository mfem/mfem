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
#include "smagorinsky.hpp"
#include "dist_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 2;
   double kin_vis = 1.0/36000.0;
   int nsteps = 5000;
   double dt = 1e-8;
   double dt_max = 1e-3;
   double cfl_target = 0.5;
   double t_final = 1.0;
   double Uavg = 1.0;
   double H = 0.03;
   bool les = true;
   bool rans = false;
} ctx;

void vel_ic(const Vector &c, double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
}

void vel_inlet(const Vector &c, double t, Vector &u)
{
   const double y = c(1);

   u(0) = 1.0;
   u(1) = 0.0;
}

double pres_outlet(const Vector &c, double t)
{
   return 0.0;
}

void velocity_wall(const Vector &c, const double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
}

double k_bdr(const Vector &c, double t)
{
   const double x = c(0), y = c(1);
   if (x == -0.06)
   {
      return 4.5e-5;
   }
   return 0.0;
}

class RANSInitCoefficient : public Coefficient
{
public:
   RANSInitCoefficient(const ParGridFunction &u, const double Re) : u(u)
   {
      turbulence_intensity = 0.15 * pow(Re, -1.0/8.0);
   }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
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

   int serial_refinements = 1;

   Mesh mesh("bfs2d.e");

   mesh.EnsureNodes();
   H1_FECollection mesh_fec(ctx.order, mesh.Dimension());
   FiniteElementSpace mesh_fes(&mesh, &mesh_fec, mesh.Dimension());
   mesh.SetNodalFESpace(&mesh_fes);

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   // Create the flow solver.
   NavierSolver navier(pmesh, ctx.order, ctx.kin_vis);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> inlet_attr(pmesh->bdr_attributes.Max());
   inlet_attr = 0;
   inlet_attr[0] = 1;

   Array<int> outlet_attr(pmesh->bdr_attributes.Max());
   outlet_attr = 0;
   outlet_attr[1] = 1;

   Array<int> wall_attr(pmesh->bdr_attributes.Max());
   wall_attr = 0;
   wall_attr[2] = 1;

   // Set the initial condition.
   ParGridFunction *u_ic = navier.GetCurrentVelocity();
   VectorFunctionCoefficient u_ic_coeff(pmesh->Dimension(), vel_ic);
   u_ic->ProjectCoefficient(u_ic_coeff);

   // Inlet
   VectorFunctionCoefficient velocity_inlet_coeff(pmesh->Dimension(),
                                                  vel_inlet);
   u_ic->ProjectBdrCoefficient(velocity_inlet_coeff, inlet_attr);
   navier.AddVelDirichletBC(vel_inlet, inlet_attr);

   // Outlet
   navier.AddPresDirichletBC(pres_outlet, outlet_attr);

   // Walls
   VectorFunctionCoefficient velocity_wall_coeff(pmesh->Dimension(),
                                                 velocity_wall);
   u_ic->ProjectBdrCoefficient(velocity_wall_coeff, wall_attr);
   navier.AddVelDirichletBC(velocity_wall, wall_attr);

   double t = 0.0;
   double t_rans = t;
   double dt = ctx.dt;
   double dt_rans = dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   if (ctx.les)
   {
      navier.EnableFilter(NavierSolver::FilterMethod::HPFRT_LPOLY);
      navier.SetCutoffModes(1);
      navier.SetFilterAlpha(10.0);
   }

   navier.Setup(dt);

   ParGridFunction *u_next_gf = nullptr;
   ParGridFunction *u_gf = navier.GetCurrentVelocity();
   ParGridFunction *p_gf = navier.GetCurrentPressure();

   std::shared_ptr<TurbulenceModel> les_sgs_model;
   std::shared_ptr<LESDelta> les_delta, geometric_delta;
   std::shared_ptr<ParGridFunction> nut_sgs_gf;
   ParGridFunction *wall_distance_gf = nullptr, *k_gf = nullptr,
                    *nu_t_gf = nullptr;
   ConstantCoefficient zero_coeff(0.0);
   GridFunctionCoefficient *wall_distance_coeff = nullptr;
   // Compute wall distance function
   wall_distance_gf = new ParGridFunction(
      navier.GetVariableViscosity()->ParFESpace());

   ParGridFunction ls_gf(navier.GetVariableViscosity()->ParFESpace());
   ls_gf = 1.0;
   ls_gf.ProjectBdrCoefficient(zero_coeff, wall_attr);

   GridFunctionCoefficient ls_coeff(&ls_gf);

   const int p = 10;
   const int newton_iter = 50;
   const double dx = AvgElementSize(*pmesh);
   HeatDistanceSolver dist_solver(dx * dx);
   // PLapDistanceSolver dist_solver(p, newton_iter);
   dist_solver.print_level = 0;
   dist_solver.ComputeScalarDistance(ls_coeff, *wall_distance_gf);

   wall_distance_coeff = new GridFunctionCoefficient(wall_distance_gf);

   if (ctx.les)
   {
      geometric_delta = std::make_shared<CurvedGeometricDelta>(ctx.order);
      les_delta = std::make_shared<VanDriestDelta>(*geometric_delta, *u_gf,
                                                   *wall_distance_gf, 1.0/ctx.kin_vis);
      les_sgs_model = std::make_shared<Smagorinsky>(*u_gf, *les_delta);
      nut_sgs_gf = std::make_shared<ParGridFunction>(*p_gf);
   }

   std::shared_ptr<PrandtlKolmogorov> rans_model;
   std::shared_ptr<ODESolver> rans_ode;

   VectorGridFunctionCoefficient vel_coeff(navier.GetCurrentVelocity());
   GridFunctionCoefficient kv_coeff(navier.GetVariableViscosity());
   PrandtlKolmogorov::EddyViscosityCoefficient *nu_t_coeff = nullptr;
   ParGridFunction kv_orig(*navier.GetVariableViscosity());
   RANSInitCoefficient *init_coeff = nullptr;
   Vector k_tdof;
   if (ctx.rans)
   {
      nu_t_gf = new ParGridFunction(navier.GetVariableViscosity()->ParFESpace());
      *nu_t_gf = 0.0;

      const double mu_calibration_const = 0.55;

      rans_model.reset(new PrandtlKolmogorov(
                          *pmesh, ctx.order, *u_gf, kv_orig, *wall_distance_gf));

      init_coeff = new RANSInitCoefficient(*u_gf, 1.0/ctx.kin_vis);
      auto inflow_coeff = new ConstantCoefficient(4.5e-5);
      rans_model->SetFixedValue(*init_coeff, {0, 2});
      rans_model->Setup();

      k_gf = &rans_model->GetScalar();
      *k_gf = 4.5e-5;
      // k_gf->ProjectCoefficient(*init_coeff);
      k_tdof.SetSize(k_gf->ParFESpace()->GetTrueVSize());
      k_gf->ParallelProject(k_tdof);

      rans_ode.reset(new ARKStepSolver(MPI_COMM_WORLD, ARKStepSolver::EXPLICIT));
      rans_ode->Init(*rans_model);
      ARKStepSolver *ark = static_cast<ARKStepSolver *>(rans_ode.get());
      ark->SetSStolerances(1e-5, 1e-5);
      ark->SetMaxStep(ctx.dt_max);
      ark->SetERKTableNum(0);

      nu_t_coeff = new PrandtlKolmogorov::EddyViscosityCoefficient(*k_gf,
                                                                   *wall_distance_gf,
                                                                   mu_calibration_const);
      nu_t_gf->ProjectCoefficient(*nu_t_coeff);
   }

   std::string outputdir_base("output/bfs2d");
   if (ctx.les && ctx.rans)
   {
      outputdir_base.append("_hybrid_les_rans");
   }
   else if (ctx.les)
   {
      outputdir_base.append("_les");
   }
   else if (ctx.rans)
   {
      outputdir_base.append("_rans");
   }

   ParaViewDataCollection pvdc(outputdir_base, pmesh);
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
      pvdc.RegisterField("nuPLUSnu_t", navier.GetVariableViscosity());
   }
   if (ctx.les)
   {
      pvdc.RegisterField("nu_t_sgs", nut_sgs_gf.get());
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

   write_output();

   for (step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      navier.Step(t, dt, step, true);
      // Get a prediction for a stable timestep
      double dt_next = dt;
      int ok = navier.PredictTimestep(1e-8, ctx.dt_max, ctx.cfl_target, dt_next);
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
         if (ctx.rans)
         {
            t_rans = t;
            dt_rans = dt;
            *navier.GetVariableViscosity() = kv_orig;
            rans_ode->Init(*rans_model);
            rans_ode->Step(k_tdof, t_rans, dt_rans);
            nu_t_gf->ProjectCoefficient(*nu_t_coeff);
            *navier.GetVariableViscosity() += *nu_t_gf;
         }

         if (ctx.les)
         {
            les_sgs_model->ComputeEddyViscosity(*nut_sgs_gf);

            *navier.GetVariableViscosity() = kv_orig;
            *navier.GetVariableViscosity() += *nut_sgs_gf;
         }

         // Queue new time step in the history array
         navier.UpdateTimestepHistory(dt);
         // Accept the time step
         t += dt;
         dt = dt_next;
      }

      // Compute the CFL
      double cfl = navier.ComputeCFL(*u_gf, dt);

      if (step % 5 == 0)
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
