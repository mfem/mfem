// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//            ------------------------------------------------
//            Extrapolation Miniapp: PDE-based extrapolation
//            ------------------------------------------------
//
// Compile with: make extrapolate
//
// Sample runs:
//     mpirun -np 4 extrapolate -o 3
//     mpirun -np 4 extrapolate -rs 3 -o 2 -p 1

#include <fstream>
#include <iostream>
#include "../common/mfem-common.hpp"
#include "marking.hpp"

using namespace std;
using namespace mfem;

int problem = 0;

const char vishost[] = "localhost";
const int  visport   = 19916;
const int wsize = 450; // glvis window size.

double domainLS(const Vector &coord)
{
   // Map from [0,1] to [-1,1].
   const int dim = coord.Size();
   const double x = coord(0)*2.0 - 1.0,
                y = coord(1)*2.0 - 1.0,
                z = (dim > 2) ? coord(2)*2.0 - 1.0 : 0.0;
   switch(problem)
   {
      case 0:
      {
         // 2d circle.
         return 0.75 - sqrt(x*x + y*y + 1e-12);
      }
      case 1:
      {
         // 2d star.
         return 0.60 - sqrt(x*x + y*y + 1e-12) +
                0.25 * (y*y*y*y*y + 5.0*x*x*x*x*y - 10.0*x*x*y*y*y) /
                       pow(x*x + y*y + 1e-12, 2.5);
      }
      default: MFEM_ABORT("Bad option for --problem!"); return 0.0;
   }
}

double solution0(const Vector &coord)
{
   // Map from [0,1] to [-1,1].
   const int dim = coord.Size();
   const double x = coord(0)*2.0 - 1.0,
                y = coord(1)*2.0 - 1.0,
                z = (dim > 2) ? coord(2)*2.0 - 1.0 : 0.0;

   if (domainLS(coord) > 0.0)
   {
      return std::cos(M_PI * x) * std::sin(M_PI * y);
   }
   else { return 0.0; }
}

class LevelSetNormalGradCoeff : public VectorCoefficient
{
private:
   const ParGridFunction &ls_gf;

public:
   LevelSetNormalGradCoeff(const ParGridFunction &ls) :
      VectorCoefficient(ls.ParFESpace()->GetMesh()->Dimension()), ls_gf(ls) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector grad_ls(vdim), n(vdim);
      ls_gf.GetGradient(T, grad_ls);
      const double norm_grad = grad_ls.Norml2();
      V = grad_ls;
      if (norm_grad > 0.0) { V /= norm_grad; }

      // Since positive level set values correspond to the known region, we
      // transport into the opposite direction of the gradient.
      V *= -1;
   }
};

void PrintNorm(int myid, Vector &v, std::string text)
{
   double norm = v.Norml1();
   MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   if (myid == 0)
   {
      std::cout << std::setprecision(12) << std::fixed
                << text << norm << std::endl;
   }
}

void PrintIntegral(int myid, ParGridFunction &g, std::string text)
{
   ConstantCoefficient zero(0.0);
   double norm = g.ComputeL1Error(zero);
   if (myid == 0)
   {
      std::cout << std::setprecision(12) << std::fixed
                << text << norm << std::endl;
   }
}

class GradComponentCoeff : public Coefficient
{
private:
   const ParGridFunction &u_gf;
   int comp;

public:
   GradComponentCoeff(const ParGridFunction &u, int c) : u_gf(u), comp(c) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector grad_u(T.GetDimension());
      u_gf.GetGradient(T, grad_u);
      return grad_u(comp);
   }
};

class NormalGradCoeff : public Coefficient
{
private:
   const ParGridFunction &u_gf;
   VectorCoefficient &n_coeff;

public:
   NormalGradCoeff(const ParGridFunction &u, VectorCoefficient &n)
      : u_gf(u), n_coeff(n) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const int dim = T.GetDimension();
      Vector n(dim), grad_u(dim);
      n_coeff.Eval(n, T, ip);
      u_gf.GetGradient(T, grad_u);
      return n * grad_u;
   }
};

class NormalGradComponentCoeff : public Coefficient
{
private:
   const ParGridFunction &du_dx, &du_dy;
   VectorCoefficient &n_coeff;

public:
   NormalGradComponentCoeff(const ParGridFunction &dx,
                            const ParGridFunction &dy, VectorCoefficient &n)
      : du_dx(dx), du_dy(dy), n_coeff(n) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const int dim = T.GetDimension();
      Vector n(dim), grad_u(dim);
      n_coeff.Eval(n, T, ip);
      grad_u(0) = du_dx.GetValue(T, ip);
      grad_u(1) = du_dy.GetValue(T, ip);
      return n * grad_u;
   }
};

class AdvectionOper : public TimeDependentOperator
{
private:
   Array<bool> &active_zones;
   ParBilinearForm &M, &K;
   const Vector &b;

public:
   AdvectionOper(Array<bool> &zones,
                 ParBilinearForm &Mbf, ParBilinearForm &Kbf, const Vector &rhs)
      : TimeDependentOperator(Mbf.Size()),
        active_zones(zones),
        M(Mbf), K(Kbf), b(rhs) { }

   virtual void Mult(const Vector &x, Vector &dx) const
   {
      ParFiniteElementSpace &pfes = *M.ParFESpace();
      const int NE = pfes.GetNE();
      const int nd = pfes.GetFE(0)->GetDof();

      Vector rhs(x.Size());
      HypreParMatrix *K_mat = K.ParallelAssemble(&K.SpMat());
      K_mat->Mult(x, rhs);
      rhs += b;

      Array<int> dofs(nd);
      DenseMatrix M_loc(nd);
      DenseMatrixInverse M_loc_inv(&M_loc);
      Vector rhs_loc(nd), dx_loc(nd);
      for (int k = 0; k < NE; k++)
      {
         pfes.GetElementDofs(k, dofs);

         if (active_zones[k] == false)
         {
            dx.SetSubVector(dofs, 0.0);
            continue;
         }

         rhs.GetSubVector(dofs, rhs_loc);
         M.SpMat().GetSubMatrix(dofs, dofs, M_loc);
         M_loc_inv.Factor();
         M_loc_inv.Mult(rhs_loc, dx_loc);
         dx.SetSubVector(dofs, dx_loc);
      }
   }
};

class Extrapolator
{
private:
   void TimeLoop(ParGridFunction &sltn, ODESolver &ode_solver,
                 double dt, int vis_x_pos, std::string vis_name)
   {
      socketstream sock;

      const int myid  = sltn.ParFESpace()->GetMyRank();
      const double t_final = 0.4;
      bool done = false;
      double t = 0.0;
      for (int ti = 0; !done;)
      {
         double dt_real = min(dt, t_final - t);
         ode_solver.Step(sltn, t, dt_real);
         ti++;

         done = (t >= t_final - 1e-8*dt);
         if (done || ti % vis_steps == 0)
         {
            if (myid == 0)
            {
               cout << "time step: " << ti << ", time: " << t << endl;
            }
            if (visualization)
            {
               common::VisualizeField(sock, vishost, visport, sltn,
                                      vis_name.c_str(), vis_x_pos, wsize+50,
                                      wsize, wsize, "rRjmm********A");
               MPI_Barrier(sltn.ParFESpace()->GetComm());
            }
         }
      }
   }

public:
   enum XtrapType {ASLAM = 0, BOCHKOV = 1} xtrap_type = ASLAM;
   int xtrap_order    = 1;
   bool visualization = false;
   int vis_steps      = 5;

   Extrapolator() { }

   // The known values taken from elements where level_set > 0, and extrapolated
   // to all other elements. The known values are not changed.
   void Extrapolate(Coefficient &level_set, const ParGridFunction &input,
                    ParGridFunction &xtrap)
   {
      ParMesh &pmesh = *input.ParFESpace()->GetParMesh();
      const int order = input.ParFESpace()->GetOrder(0),
                dim   = pmesh.Dimension(), NE = pmesh.GetNE();

      // Get a ParGridFunction and mark elements.
      H1_FECollection fec(order, dim);
      ParFiniteElementSpace pfes_H1(&pmesh, &fec);
      ParGridFunction ls_gf(&pfes_H1);
      ls_gf.ProjectCoefficient(level_set);
      if (visualization)
      {
         socketstream sock;
         common::VisualizeField(sock, vishost, visport, ls_gf,
                                "Domain level set", 0, 0, wsize, wsize,
                                "rRjmm********A");
         MPI_Barrier(pmesh.GetComm());
      }
      // Mark elements.
      Array<int> elem_marker;
      ShiftedFaceMarker marker(pmesh, pfes_H1, false);
      ls_gf.ExchangeFaceNbrData();
      marker.MarkElements(ls_gf, elem_marker);

      // The active zones are where we extrapolate (where the PDE is solved).
      Array<bool> active_zones(NE);
      for (int k = 0; k < NE; k++)
      {
         // Extrapolation is done in zones that are CUT or OUTSIDE.
         active_zones[k] =
            (elem_marker[k] == ShiftedFaceMarker::INSIDE) ? false : true;
      }

      // Setup a VectorCoefficient for n = - grad_ls / |grad_ls|.
      // The sign makes it point out of the known region.
      // The coefficient must be continuous to have well-defined transport.
      LevelSetNormalGradCoeff ls_n_coeff_L2(ls_gf);
      ParFiniteElementSpace pfes_H1_vec(&pmesh, &fec, dim);
      ParGridFunction lsn_gf(&pfes_H1_vec);
      ls_gf.ExchangeFaceNbrData();
      lsn_gf.ProjectDiscCoefficient(ls_n_coeff_L2, GridFunction::ARITHMETIC);
      VectorGridFunctionCoefficient ls_n_coeff(&lsn_gf);

      // Initial solution.
      // Trim to the known values (only elements inside the known region).
      Array<int> dofs;
      L2_FECollection fec_L2(order, dim);
      ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
      ParGridFunction u(&pfes_L2);
      u.ProjectGridFunction(input);
      for (int k = 0; k < NE; k++)
      {
         pfes_L2.GetElementDofs(k, dofs);
         if (elem_marker[k] != ShiftedFaceMarker::INSIDE)
         { u.SetSubVector(dofs, 0.0); }
      }
      if (visualization)
      {
         socketstream sock;
         common::VisualizeField(sock, vishost, visport, u,
                                "Fixed (known) u values", wsize, 0,
                                wsize, wsize, "rRjmm********A");
      }

      // Normal derivative function.
      ParGridFunction n_grad_u(&pfes_L2);
      NormalGradCoeff n_grad_u_coeff(u, ls_n_coeff);
      n_grad_u.ProjectCoefficient(n_grad_u_coeff);
      if (visualization)
      {
         socketstream sock;
         common::VisualizeField(sock, vishost, visport, n_grad_u,
                                "n.grad(u)", 2*wsize, 0, wsize, wsize,
                                "rRjmm********A");
      }

      // 2nd normal derivative function.
      ParGridFunction n_grad_n_grad_u(&pfes_L2);
      NormalGradCoeff n_grad_n_grad_u_coeff(n_grad_u, ls_n_coeff);
      n_grad_n_grad_u.ProjectCoefficient(n_grad_n_grad_u_coeff);
      if (visualization)
      {
         socketstream sock;
         common::VisualizeField(sock, vishost, visport, n_grad_n_grad_u,
                                "n.grad(n.grad(u))", 3*wsize, 0, wsize, wsize,
                                "rRjmm********A");
      }

      ParBilinearForm lhs_bf(&pfes_L2), rhs_bf(&pfes_L2);
      lhs_bf.AddDomainIntegrator(new MassIntegrator);
      const double alpha = -1.0;
      rhs_bf.AddDomainIntegrator(new ConvectionIntegrator(ls_n_coeff, alpha));
      auto trace_i = new NonconservativeDGTraceIntegrator(ls_n_coeff, alpha);
      rhs_bf.AddInteriorFaceIntegrator(trace_i);
      rhs_bf.KeepNbrBlock(true);

      ls_gf.ExchangeFaceNbrData();
      lhs_bf.Assemble();
      lhs_bf.Finalize();
      rhs_bf.Assemble(0);
      rhs_bf.Finalize(0);

      // Compute a CFL time step.
      double h_min = std::numeric_limits<double>::infinity();
      for (int k = 0; k < NE; k++)
      {
         h_min = std::min(h_min, pmesh.GetElementSize(k));
      }
      MPI_Allreduce(MPI_IN_PLACE, &h_min, 1, MPI_DOUBLE, MPI_MIN,
                    pfes_L2.GetComm());
      h_min /= order;
      // The propagation speed is 1.
      double dt = 0.25 * h_min / 1.0;

      // Time loops.
      Vector rhs(pfes_L2.GetVSize());
      AdvectionOper adv_oper(active_zones, lhs_bf, rhs_bf, rhs);
      RK2Solver ode_solver(1.0);
      ode_solver.Init(adv_oper);

      if (xtrap_type == ASLAM)
      {
         switch (xtrap_order)
         {
            case 0:
            {
               // Constant extrapolation of u.
               rhs = 0.0;
               TimeLoop(u, ode_solver, dt, wsize, "u - constant extrap");
               break;
            }
            case 1:
            {
               // Constant extrapolation of [n.grad_u].
               rhs = 0.0;
               TimeLoop(n_grad_u, ode_solver, dt, 2*wsize, "n.grad(u)");

               // Linear extrapolation of u.
               lhs_bf.Mult(n_grad_u, rhs);
               TimeLoop(u, ode_solver, dt, wsize, "u - linear Aslam extrap");
               break;
            }
            case 2:
            {
               // Constant extrapolation of [n.grad(n.grad(u))].
               rhs = 0.0;
               TimeLoop(n_grad_n_grad_u, ode_solver, dt, 3*wsize, "n.grad(n.grad(u))");

               // Linear extrapolation of [n.grad_u].
               lhs_bf.Mult(n_grad_n_grad_u, rhs);
               TimeLoop(n_grad_u, ode_solver, dt, 2*wsize, "n.grad(u)");

               // Quadratic extrapolation of u.
               lhs_bf.Mult(n_grad_u, rhs);
               TimeLoop(u, ode_solver, dt, wsize, "u - quadratic Aslam extrap");
               break;
            }
            default: MFEM_ABORT("Wrong extrapolation order.");
         }
      }
      else if (xtrap_type == BOCHKOV)
      {
         switch (xtrap_order)
         {
            case 0:
            {
               // Constant extrapolation of u.
               rhs = 0.0;
               TimeLoop(u, ode_solver, dt, wsize, "u - constant extrap");
               break;
            }
            case 1:
            {
               // Constant extrapolation of all grad(u) components.
               rhs = 0.0;
               ParGridFunction grad_u_0(&pfes_L2), grad_u_1(&pfes_L2);
               GradComponentCoeff grad_u_0_coeff(u, 0), grad_u_1_coeff(u, 1);
               grad_u_0.ProjectCoefficient(grad_u_0_coeff);
               grad_u_1.ProjectCoefficient(grad_u_1_coeff);
               TimeLoop(grad_u_0, ode_solver, dt, 2*wsize, "grad_u_0");
               TimeLoop(grad_u_1, ode_solver, dt, 3*wsize, "grad_u_1");

               // Linear extrapolation of u.
               ParLinearForm rhs_lf(&pfes_L2);
               NormalGradComponentCoeff grad_u_n(grad_u_0, grad_u_1, ls_n_coeff);
               rhs_lf.AddDomainIntegrator(new DomainLFIntegrator(grad_u_n));
               rhs_lf.Assemble();
               rhs = rhs_lf;
               TimeLoop(u, ode_solver, dt, wsize, "u - linear Bochkov extrap");
               break;
            }
            case 2:  MFEM_ABORT("Not implemented.");
            default: MFEM_ABORT("Wrong extrapolation order.");
         }
      }
      else { MFEM_ABORT("Wrong extrapolation type option"); }

      xtrap.ProjectGridFunction(u);
   }
};

int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi;
   int myid = mpi.WorldRank();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 2;
   Extrapolator::XtrapType xtrap_type = Extrapolator::ASLAM;
   int xtrap_order = 1;
   int order = 2;
   bool visualization = true;
   int vis_steps = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption((int*)&xtrap_type, "-et", "--extrap-type",
                  "Extrapolation type: Aslam (0) or Bochkov (1).");
   args.AddOption(&xtrap_order, "-eo", "--extrap-order",
                  "Extrapolation order: 0/1/2 for constant/linear/quadratic.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&problem, "-p", "--problem",
                  "0 - 2D circle,\n\t"
                  "1 - 2D star");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }


   // Refine the mesh and distribute.
   Mesh mesh(mesh_file, 1, 1);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   const int dim = pmesh.Dimension();

   // Input function.
   L2_FECollection fec_L2(order, dim);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
   ParGridFunction u(&pfes_L2);
   FunctionCoefficient u0_coeff(solution0);
   u.ProjectCoefficient(u0_coeff);

   // Extrapolate.
   Extrapolator xtrap;
   xtrap.xtrap_type  = xtrap_type;
   xtrap.xtrap_order = xtrap_order;
   xtrap.visualization = true;
   xtrap.vis_steps = 5;
   FunctionCoefficient ls_coeff(domainLS);
   ParGridFunction ux(&pfes_L2);
   xtrap.Extrapolate(ls_coeff, u, ux);

   PrintNorm(myid, ux, "final norm: ");

   // ParaView output.
   ParaViewDataCollection dacol("ParaViewExtrapolate", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("extrapolated sltn", &ux);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();

   return 0;
}
