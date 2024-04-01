// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// 3D Taylor-Green vortex benchmark example at Re=1600
// Unsteady flow of a decaying vortex is computed and compared against a known,
// analytical solution.

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int element_subdivisions = 1;
   int order = 4;
   real_t kinvis = 1.0 / 1600.0;
   real_t t_final = 10 * 1e-3;
   real_t dt = 1e-3;
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
} ctx;

void vel_tgv(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);
   real_t zi = x(2);

   u(0) = sin(xi) * cos(yi) * cos(zi);
   u(1) = -cos(xi) * sin(yi) * cos(zi);
   u(2) = 0.0;
}

class QuantitiesOfInterest
{
public:
   QuantitiesOfInterest(ParMesh *pmesh)
   {
      H1_FECollection h1fec(1);
      ParFiniteElementSpace h1fes(pmesh, &h1fec);

      onecoeff.constant = 1.0;
      mass_lf = new ParLinearForm(&h1fes);
      mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      mass_lf->Assemble();

      ParGridFunction one_gf(&h1fes);
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   };

   real_t ComputeKineticEnergy(ParGridFunction &v)
   {
      Vector velx, vely, velz;
      real_t integ = 0.0;
      const FiniteElement *fe;
      ElementTransformation *T;
      FiniteElementSpace *fes = v.FESpace();

      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         int intorder = 2 * fe->GetOrder();
         const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

         v.GetValues(i, *ir, velx, 1);
         v.GetValues(i, *ir, vely, 2);
         v.GetValues(i, *ir, velz, 3);

         T = fes->GetElementTransformation(i);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);

            real_t vel2 = velx(j) * velx(j) + vely(j) * vely(j)
                          + velz(j) * velz(j);

            integ += ip.weight * T->Weight() * vel2;
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&integ,
                    &global_integral,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      return 0.5 * global_integral / volume;
   };

   ~QuantitiesOfInterest() { delete mass_lf; };

private:
   ConstantCoefficient onecoeff;
   ParLinearForm *mass_lf;
   real_t volume;
};

template<typename T>
T sq(T x)
{
   return x * x;
}

// Computes Q = 0.5*(tr(\nabla u)^2 - tr(\nabla u \cdot \nabla u))
void ComputeQCriterion(ParGridFunction &u, ParGridFunction &q)
{
   FiniteElementSpace *v_fes = u.FESpace();
   FiniteElementSpace *fes = q.FESpace();

   // AccumulateAndCountZones
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   q = 0.0;

   // Local interpolation
   int elndofs;
   Array<int> v_dofs, dofs;
   Vector vals;
   Vector loc_data;
   int vdim = v_fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, dofs);
      v_fes->GetElementVDofs(e, v_dofs);
      u.GetSubVector(v_dofs, loc_data);
      vals.SetSize(dofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval
         // GetVectorGradientHat
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         real_t q_val = 0.5 * (sq(grad(0, 0)) + sq(grad(1, 1)) + sq(grad(2, 2)))
                        + grad(0, 1) * grad(1, 0) + grad(0, 2) * grad(2, 0)
                        + grad(1, 2) * grad(2, 1);

         vals(dof) = q_val;
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < dofs.Size(); j++)
      {
         int ldof = dofs[j];
         q(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = q.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(q.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(q.GetData());

   // Compute means
   for (int i = 0; i < q.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         q(i) /= nz;
      }
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.element_subdivisions,
                  "-es",
                  "--element-subdivisions",
                  "Number of 1d uniform subdivisions for each element.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.pa,
                  "-pa",
                  "--enable-pa",
                  "-no-pa",
                  "--disable-pa",
                  "Enable partial assembly.");
   args.AddOption(&ctx.ni,
                  "-ni",
                  "--enable-ni",
                  "-no-ni",
                  "--disable-ni",
                  "Enable numerical integration rules.");
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(
      &ctx.checkres,
      "-cr",
      "--checkresult",
      "-no-cr",
      "--no-checkresult",
      "Enable or disable checking of the result. Returns -1 on failure.");
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

   Mesh orig_mesh("../../data/periodic-cube.mesh");
   Mesh mesh = Mesh::MakeRefined(orig_mesh, ctx.element_subdivisions,
                                 BasisType::ClosedUniform);
   orig_mesh.Clear();

   mesh.EnsureNodes();
   GridFunction *nodes = mesh.GetNodes();
   *nodes *= M_PI;

   int nel = mesh.GetNE();
   if (Mpi::Root())
   {
      mfem::out << "Number of elements: " << nel << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);
   u_ic->ProjectCoefficient(u_excoeff);

   real_t t = 0.0;
   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   ParGridFunction w_gf(*u_gf);
   ParGridFunction q_gf(*p_gf);
   flowsolver.ComputeCurl3D(*u_gf, w_gf);
   ComputeQCriterion(*u_gf, q_gf);

   QuantitiesOfInterest kin_energy(pmesh);

   ParaViewDataCollection pvdc("tgv_output", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.RegisterField("vorticity", &w_gf);
   pvdc.RegisterField("qcriterion", &q_gf);
   pvdc.Save();

   real_t u_inf_loc = u_gf->Normlinf();
   real_t p_inf_loc = p_gf->Normlinf();
   real_t u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
   real_t p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
   real_t ke = kin_energy.ComputeKineticEnergy(*u_gf);

   std::string fname = "tgv_out_p_" + std::to_string(ctx.order) + ".txt";
   FILE *f = NULL;

   if (Mpi::Root())
   {
      int nel1d = static_cast<int>(std::round(pow(nel, 1.0 / 3.0)));
      int ngridpts = p_gf->ParFESpace()->GlobalVSize();
      printf("%11s %11s %11s %11s %11s\n", "Time", "dt", "u_inf", "p_inf", "ke");
      printf("%.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke);

      f = fopen(fname.c_str(), "w");
      fprintf(f, "3D Taylor Green Vortex\n");
      fprintf(f, "order = %d\n", ctx.order);
      fprintf(f, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
      fprintf(f, "dofs per component = %d\n", ngridpts);
      fprintf(f, "=================================================\n");
      fprintf(f, "        time                   kinetic energy\n");
      fprintf(f, "%20.16e     %20.16e\n", t, ke);
      fflush(f);
      fflush(stdout);
   }

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if ((step + 1) % 100 == 0 || last_step)
      {
         flowsolver.ComputeCurl3D(*u_gf, w_gf);
         ComputeQCriterion(*u_gf, q_gf);
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
      }

      u_inf_loc = u_gf->Normlinf();
      p_inf_loc = p_gf->Normlinf();
      u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
      p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
      ke = kin_energy.ComputeKineticEnergy(*u_gf);
      if (Mpi::Root())
      {
         printf("%.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke);
         fprintf(f, "%20.16e     %20.16e\n", t, ke);
         fflush(f);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   // Test if the result for the test run is as expected.
   if (ctx.checkres)
   {
      real_t tol = 2e-5;
      real_t ke_expected = 1.25e-1;
      if (fabs(ke - ke_expected) > tol)
      {
         if (Mpi::Root())
         {
            mfem::out << "Result has a larger error than expected."
                      << std::endl;
         }
         return -1;
      }
   }

   delete pmesh;

   return 0;
}
