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
//
// Navier MMS example
//
// A manufactured solution is defined as
//
// u = [pi * sin(t) * sin(pi * x)^2 * sin(2 * pi * y),
//      -(pi * sin(t) * sin(2 * pi * x)) * sin(pi * y)^2].
//
// p = cos(pi * x) * sin(t) * sin(pi * y)
//
// The solution is used to compute the symbolic forcing term (right hand side),
// of the equation. Then the numerical solution is computed and compared to the
// exact manufactured solution to determine the error.

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int ser_ref_levels = 2;
   int order = 4;
   double kinvis = 0.05;
   double t_final = 1.0;
   double dt = 1.0e-2;
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
   double a = 0.5;
   double ft = 2.0;
   double fx = 2.0;
   double Tbar = 1.0;
   double Kbar = 1.0;
} ctx;

double StructureInitialSolution(const Vector &pt)
{
   //void(pt);
   return 1.0;
}

double StructureInitialRate(const Vector &pt)
{
    double x = pt(0);
    return ctx.a * sin(ctx.fx * M_PI * x);
}

double StructureExactSolution(const Vector &pt, const double t)
{
    double x = pt(0);
    return ctx.a / (M_PI * ctx.ft) * sin(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t) + 1.0;
}

double StructureForcing(const Vector &pt, const double t)
{
    double x = pt(0);

    // Compute the normal vector
    double dEtaDx = ctx.a * ctx.fx / ctx.ft * cos(ctx.fx * M_PI * x) * sin(ctx.ft * M_PI * t);
    double n2 = 1/sqrt(1 + dEtaDx * dEtaDx);
    double n1 = -dEtaDx * n2;

    // Compute pressure, structure, velocity, and required derivatives.
    // Uses a tiny bit more memory, will be easier to debug.
    // We need to recall that y = eta.
    double eta = StructureExactSolution(pt,t);
    double p = cos(ctx.fx * M_PI * x) * cos(ctx.fx * M_PI * eta) * cos(ctx.ft * M_PI * t);
    double d2EtaDx2 = -ctx.fx * ctx.fx * M_PI * M_PI * (eta - 1);
    double d2EtaDt2 = -ctx.ft * ctx.ft * M_PI * M_PI * (eta - 1);
    double dUDy = -ctx.a * ctx.fx * M_PI * cos(ctx.fx * M_PI *x) * cos(ctx.ft * M_PI * t);
    double dVDy = -ctx.a * ctx.fx * M_PI * cos(ctx.fx * M_PI * x) * cos(ctx.ft * M_PI * t) * dEtaDx;
    double dVDx =  ctx.a * ctx.fx * M_PI * cos(ctx.fx * M_PI * x) * cos(ctx.ft * M_PI * t) * (1 + dEtaDx * dEtaDx);

    // We now have everything we need.
    return ctx.Tbar * d2EtaDt2 + ctx.Kbar * eta - ctx.Tbar * d2EtaDx2 - p*n2 
         + ctx.kinvis * ((dUDy + dVDx) * n1 + 2 * dVDy * n2);
}

void vel(const Vector &coords, double t, Vector &u)
{
   double x = coords(0);
   double y = coords(1);

   u(0) = -(ctx.a * cos(ctx.ft * M_PI * t) * cos(ctx.fx * M_PI * x)
            * sin(ctx.fx * M_PI
                  * (-1.0 + y
                     - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                     / (ctx.ft * M_PI))));
   u(1) = ctx.a * cos(ctx.ft * M_PI * t)
          * cos(ctx.fx * M_PI
                * (-1.0 + y
                   - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                   / (ctx.ft * M_PI)))
          * sin(ctx.fx * M_PI * x)
          - (pow(ctx.a, 2.0) * ctx.fx * cos(ctx.ft * M_PI * t)
             * pow(cos(ctx.fx * M_PI * x), 2.0) * sin(ctx.ft * M_PI * t)
             * sin(ctx.fx * M_PI
                   * (-1.0 + y
                      - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                      / (ctx.ft * M_PI))))
          / ctx.ft;
}

double p(const Vector &coords, double t)
{
   double x = coords(0);
   double y = coords(1);

   return cos(ctx.ft * M_PI * t) * cos(ctx.fx * M_PI * x)
          * cos(ctx.fx * M_PI * y);
}

void accel(const Vector &coords, double t, Vector &u)
{
   double x = coords(0);
   double y = coords(1);

   u(0) = (-(pow(ctx.ft, 2.0) * ctx.fx * M_PI * cos(ctx.ft * M_PI * t) * cos(
                ctx.fx * M_PI * y)
             * sin(ctx.fx * M_PI * x))
           + pow(ctx.a, 2.0) * pow(ctx.ft, 2.0) * ctx.fx * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * cos(ctx.fx * M_PI * x)
           * cos(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           * sin(ctx.fx * M_PI * x)
           - pow(ctx.a, 2.0) * pow(ctx.ft, 2.0) * ctx.fx * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * cos(ctx.fx * M_PI * x)
           * pow(cos(ctx.fx * M_PI
                     * (-1.0 + y
                        - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                        / (ctx.ft * M_PI))),
                 2.0)
           * sin(ctx.fx * M_PI * x)
           + 3.0 * pow(ctx.a, 2.0) * ctx.ft * pow(ctx.fx, 3.0) * ctx.kinvis * pow(M_PI,
                                                                                  2.0)
           * cos(ctx.ft * M_PI * t) * cos(ctx.fx * M_PI * x)
           * cos(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x)
           - 2.0 * ctx.a * pow(ctx.ft, 2.0) * pow(ctx.fx, 2.0) * ctx.kinvis * pow(M_PI,
                                                                                  2.0)
           * cos(ctx.ft * M_PI * t) * cos(ctx.fx * M_PI * x)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           + ctx.a * pow(ctx.ft, 3.0) * M_PI * cos(ctx.fx * M_PI * x) * sin(
              ctx.ft * M_PI * t)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           - pow(ctx.a, 3.0) * pow(ctx.fx, 4.0) * ctx.kinvis * pow(M_PI,
                                                                   2.0) * cos(ctx.ft * M_PI * t)
           * pow(cos(ctx.fx * M_PI * x), 3.0) * pow(sin(ctx.ft * M_PI * t), 2.0)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           - pow(ctx.a, 2.0) * pow(ctx.ft, 2.0) * ctx.fx * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * cos(ctx.fx * M_PI * x)
           * sin(ctx.fx * M_PI * x)
           * pow(sin(ctx.fx * M_PI
                     * (-1.0 + y
                        - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                        / (ctx.ft * M_PI))),
                 2.0))
          / pow(ctx.ft, 2.0);

   u(1) = (2.0 * ctx.a * pow(ctx.ft, 3.0) * pow(ctx.fx,
                                                2.0) * ctx.kinvis * pow(M_PI, 2.0)
           * cos(ctx.ft * M_PI * t)
           * cos(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           * sin(ctx.fx * M_PI * x)
           - ctx.a * pow(ctx.ft, 4.0) * M_PI
           * cos(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x)
           + pow(ctx.a, 3.0) * pow(ctx.ft, 2.0) * pow(ctx.fx, 2.0) * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * pow(cos(ctx.fx * M_PI * x), 2.0)
           * cos(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x)
           - pow(ctx.a, 3.0) * pow(ctx.ft, 2.0) * pow(ctx.fx, 2.0) * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * pow(cos(ctx.fx * M_PI * x), 2.0)
           * pow(cos(ctx.fx * M_PI
                     * (-1.0 + y
                        - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                        / (ctx.ft * M_PI))),
                 2.0)
           * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x)
           + 6.0 * pow(ctx.a, 3.0) * ctx.ft * pow(ctx.fx, 4.0) * ctx.kinvis * pow(M_PI,
                                                                                  2.0)
           * cos(ctx.ft * M_PI * t) * pow(cos(ctx.fx * M_PI * x), 2.0)
           * cos(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           * pow(sin(ctx.ft * M_PI * t), 2.0) * sin(ctx.fx * M_PI * x)
           - pow(ctx.ft, 3.0) * ctx.fx * M_PI * cos(ctx.ft * M_PI * t) * cos(
              ctx.fx * M_PI * x)
           * sin(ctx.fx * M_PI * y)
           - pow(ctx.a, 2.0) * pow(ctx.ft, 3.0) * ctx.fx * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * pow(cos(ctx.fx * M_PI * x), 2.0)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           - pow(ctx.a, 2.0) * pow(ctx.ft, 3.0) * ctx.fx * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * pow(cos(ctx.fx * M_PI * x), 2.0)
           * cos(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           - 5.0 * pow(ctx.a, 2.0) * pow(ctx.ft, 2.0) * pow(ctx.fx, 3.0) * ctx.kinvis
           * pow(M_PI, 2.0) * cos(ctx.ft * M_PI * t)
           * pow(cos(ctx.fx * M_PI * x), 2.0) * sin(ctx.ft * M_PI * t)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           + pow(ctx.a, 2.0) * pow(ctx.ft, 3.0) * ctx.fx * M_PI
           * pow(cos(ctx.fx * M_PI * x), 2.0) * pow(sin(ctx.ft * M_PI * t), 2.0)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           - pow(ctx.a, 4.0) * pow(ctx.fx, 5.0) * ctx.kinvis * pow(M_PI,
                                                                   2.0) * cos(ctx.ft * M_PI * t)
           * pow(cos(ctx.fx * M_PI * x), 4.0) * pow(sin(ctx.ft * M_PI * t), 3.0)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           + pow(ctx.a, 2.0) * pow(ctx.ft, 3.0) * ctx.fx * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * pow(sin(ctx.fx * M_PI * x), 2.0)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           - pow(ctx.a, 2.0) * pow(ctx.ft, 3.0) * ctx.fx * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0)
           * cos(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           * pow(sin(ctx.fx * M_PI * x), 2.0)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           + 3.0 * pow(ctx.a, 2.0) * pow(ctx.ft, 2.0) * pow(ctx.fx, 3.0) * ctx.kinvis
           * pow(M_PI, 2.0) * cos(ctx.ft * M_PI * t) * sin(ctx.ft * M_PI * t)
           * pow(sin(ctx.fx * M_PI * x), 2.0)
           * sin(ctx.fx * M_PI
                 * (-1.0 + y
                    - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                    / (ctx.ft * M_PI)))
           - 2.0 * pow(ctx.a, 3.0) * pow(ctx.ft, 2.0) * pow(ctx.fx, 2.0) * M_PI
           * pow(cos(ctx.ft * M_PI * t), 2.0) * pow(cos(ctx.fx * M_PI * x), 2.0)
           * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x)
           * pow(sin(ctx.fx * M_PI
                     * (-1.0 + y
                        - (ctx.a * sin(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x))
                        / (ctx.ft * M_PI))),
                 2.0))
          / pow(ctx.ft, 3.0);
}

/*class FluidForcingCoefficient : public VectorCoefficient
{
   public:
      FluidForcingCoefficient(ParGridFunction *, ParGridFunction *, Mesh *, ParMesh *);
      void Eval(Vector &, ElementTransformation &, const IntegrationPoint &);
      void Update();

   private:
      ParGridFunction *vel_gf;
      ParGridFunction *pres_gf;
      Mesh *s_mesh;
      ParMesh *f_mesh;
};

FluidForcingCoefficient::FluidForcingCoefficient(ParGridFunction *vel_gf_, ParGridFunction *pres_gf_, Mesh *s_mesh_, ParMesh *f_mesh_)
{
   vel_gf = vel_gf_;
   pres_gf = pres_gf;
   s_mesh = s_mesh_;
   f_mesh = f_mesh_;
}
void FluidForcingCoefficient::Eval(Vector &Qvec, ElementTransformation &T, const IntegrationPoint &ip)
{
   // Extract the eta value.
   double y, dEtaDx_val;
   double x = ip.x;
   {
   y = eta_gf.GetValue(T, ips[0]);
   eta_gf.GetGradient(T,dedx_vec);
   dEtaDx_val = dedx_vec[0];
   }

   // Now extract the required velocity derivative and pressure values.
   DenseMatrix points(fluid_mesh->Dimension(),1);
   Vector coords(2);
   coords[0] = x;
   coords[1] = y;
   points.SetCol(0,coords);

   Array<int> elem_ids;
   Array<IntegrationPoint> ips;

   fluid_mesh->FindPoints(points, elem_ids, ips); // Parallel issues??? Some condense/gather operation probably needed below

   switch (elem_ids[0])
   {
   case -2:
      // This point was found on another process. TODO: We will need to implement a MPI all reduce with a sum for the fluid_forcing vector inside shell operator.
      // Should never happen if we run with one process in the meantime.
      return 0.0;

   case -1:
      if (mpi.Root())
         std::cout << "Point not found on ANY process." << std::endl;
      return 0.0/0.0; // <- for some reason C++ would not let me directly return NaN here?
   
   default: // Do nothing, we must have found the point on the locally owned part of the mesh.
      break;
   }

   double p = p_gf->GetValue(elem_ids[0],ips[0]);

   DenseMatrix grad(2,2);
   ElementTransformation T; T.SetIntPoint(ips[0]);
   u_gf->GetVectorGradient(T,grad);
   double dudy = grad(0,1), dvdx = grad(1,0), dvdy = grad(1,1);
   double n2 = 1/sqrt(1 + dEtaDx_val * dEtaDx_val);
   double n1 = -dEtaDx_val * n2;

   return p*n2 - ctx.kinvis * ((dudy + dvdx) * n1 + 2 * dvdy * n2);
   
}*/

class ShellOperator : public SecondOrderTimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list; 

   BilinearForm *M, *K, *S;
   LinearForm forcing_LF;
   SparseMatrix Mmat, Mmat0, Kmat, Smat;
   Solver *invM, *invS;
   double current_dt, fac0old;
   Array<int> block_trueOffsets;
   FunctionCoefficient forcing_function;

   GMRESSolver M_solver, S_solver; // Krylov solver for inverting mass and overall system matricies.
   DSmoother M_prec, S_prec;  // Preconditioner for the mass matrix M

   double current_time, dt;

   mutable Vector z, V; // auxiliary vector

public:
   ShellOperator(FiniteElementSpace &f, Array<int> &ess_bdr, FunctionCoefficient &);

   using SecondOrderTimeDependentOperator::Mult;
   virtual void Mult(const Vector &u, const Vector &du_dt,
                     Vector &d2udt2) const;
   
   using SecondOrderTimeDependentOperator::ImplicitSolve;
   virtual void ImplicitSolve(const double fac0, const double fac1,
                              const Vector &u, const Vector &dudt, Vector &d2udt2);

   virtual ~ShellOperator();
};


ShellOperator::ShellOperator(FiniteElementSpace &f, Array<int> &ess_bdr, FunctionCoefficient &ff)
   : SecondOrderTimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL),
     K(NULL), block_trueOffsets(3), forcing_function(StructureForcing), z(height), V(height)
{
   //const double rel_tol = 1e-8;
   ConstantCoefficient one(1);

   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(one));
   K->Assemble();
   // Make the full system matrix and deal with boundary conditions manually.
   K->FormSystemMatrix(Array<int>(), Kmat);

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   // Make the full system matrix and deal with boundary conditions manually.
   M->FormSystemMatrix(Array<int>(), Mmat);

   forcing_function.SetTime(t);
   forcing_LF.Update(&fespace);
   forcing_LF.AddDomainIntegrator(new DomainLFIntegrator(forcing_function));
   forcing_LF.AddDomainIntegrator(new DomainLFIntegrator(ff));
   std::cout << "here" << std::endl;
   forcing_LF.Assemble();
   std::cout << "assembled rhs effectively" << std::endl;

   fac0old=0.0;

   // Make the matricies obey boundary conditions.
   for (int cond_ind = 0; cond_ind < ess_tdof_list.Size(); ++cond_ind){
      Kmat.EliminateRow(ess_tdof_list[cond_ind],DIAG_ONE);
      Mmat.EliminateRow(ess_tdof_list[cond_ind],DIAG_ONE);
   }
   //Kmat.PrintMatlab();

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(0.0);
   M_solver.SetAbsTol(1e-12);
   M_solver.SetMaxIter(1000);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);
}

void ShellOperator::Mult(const Vector &u, const Vector &du_dt,
                        Vector &d2udt2)  const
{
   // Compute:
   //    d2EtaDt2 = -Kbar / Tbar eta + M^-1(F - K eta)
   // BCs.
   V = u;
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      V[ess_tdof_list[i]] = 1.0;
   }
   Kmat.Mult(V, z);
   z *= ctx.Tbar;
   z += forcing_LF;
   M_solver.Mult(z, V);

   d2udt2 = u;
   d2udt2 *= -ctx.Kbar;
   d2udt2 += V;
   d2udt2 *= 1/ctx.Tbar;

   // Apply the inhomgenous boundary conditions.
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      d2udt2[ess_tdof_list[i]] = 0.0;
   }
}

void ShellOperator::ImplicitSolve(const double fac0, const double fac1,
                                 const Vector &u, const Vector &dudt, Vector &d2udt2)
{
   // Solve the equation
   //    d2udt2 = Smat^1*(-Kbar M eta - Tbar K eta + F).
   // We have the system matrix 
   //    Smat = ((1 + fac0 * Kbar / Tbar) M + fac0 * K).
   // Determine if we need to rebuild the system matrix.
   if (abs(fac0old) < 1e-16){
      // Initialize the bilinear form.
      ConstantCoefficient MassCoef(ctx.Tbar + fac0 * ctx.Kbar);
      ConstantCoefficient DiffCoef(fac0 * ctx.Tbar);
      S = new BilinearForm(&fespace);
      S->AddDomainIntegrator(new MassIntegrator(MassCoef));
      S->AddDomainIntegrator(new DiffusionIntegrator(DiffCoef));
      S->Assemble();
      S->FormSystemMatrix(Array<int>(),Smat);

      // Constraints
      for (int cond_ind = 0; cond_ind < ess_tdof_list.Size(); ++cond_ind){
         Smat.EliminateRow(ess_tdof_list[cond_ind],DIAG_ONE);
      }

      // Solver
      S_solver.iterative_mode = false;
      S_solver.SetRelTol(0);
      S_solver.SetAbsTol(1e-8);
      S_solver.SetMaxIter(100000);
      S_solver.SetPrintLevel(0);
      S_solver.SetPreconditioner(S_prec);
      S_solver.SetOperator(Smat);
   }
   else if (abs(fac0-fac0old) > 1e-14){
      // Reinitialize the bilinear form.
      ConstantCoefficient MassCoef(ctx.Tbar + fac0 * ctx.Kbar);
      ConstantCoefficient DiffCoef(fac0 * ctx.Tbar);
      S = new BilinearForm(&fespace);
      S->AddDomainIntegrator(new MassIntegrator(MassCoef));
      S->AddDomainIntegrator(new DiffusionIntegrator(DiffCoef));
      S->Assemble();
      S->FormSystemMatrix(Array<int>(),Smat);

      // Constraints
      for (int cond_ind = 0; cond_ind < ess_tdof_list.Size(); ++cond_ind){
         Smat.EliminateRow(ess_tdof_list[cond_ind],DIAG_ONE);
      }

      // Solver
      S_solver.iterative_mode = false;
      S_solver.SetRelTol(0);
      S_solver.SetAbsTol(1e-8);
      S_solver.SetMaxIter(100000);
      S_solver.SetPrintLevel(0);
      S_solver.SetPreconditioner(S_prec);
      S_solver.SetOperator(Smat);
   }
   
   // We first need to form the right hand side.
   forcing_function.SetTime(t);
   forcing_LF.Update();
   forcing_LF.Assemble();

   // Form the part of the rhs that comes from the current postion.
   Vector aux = u;
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      aux[ess_tdof_list[i]] = 1.0;
   }
   Mmat.Mult(aux,z);
   z *= -ctx.Kbar;
   z += forcing_LF;
   Kmat.Mult(aux,V);
   V *= -ctx.Tbar;
   z += V;

   // Apply the inhomgenous boundary conditions.
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      z[ess_tdof_list[i]] = 0.0;
      d2udt2[ess_tdof_list[i]] = 0.0;
   }

   // Solve the system.
   S_solver.Mult(z,d2udt2);

   // Apply the inhomgenous boundary conditions.
   // This really shouldn't be required if one is using a Krylov solver, but it can't hurt.
   for (int i = 0; i < ess_tdof_list.Size(); ++i){
      d2udt2[ess_tdof_list[i]] = 0.0;
   }
}

ShellOperator::~ShellOperator()
{
   delete M;
   delete K;
   delete S;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int vis_steps = 1;
   bool verbose = false;

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the fluid_mesh uniformly in serial.");
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
      if (mpi.Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(mfem::out);
   }

   // fluid_mesh *fluid_mesh = new fluid_mesh("../../data/periodic-square.fluid_mesh");
   Mesh *fluid_mesh = new Mesh("../../data/inline-quad.mesh");
   fluid_mesh->SetCurvature(ctx.order);
   GridFunction *fluid_nodes_gf = fluid_mesh->GetNodes();

   Mesh struct_mesh;
   struct_mesh = struct_mesh.MakeCartesian1D(4); // Use default interval [0,1].

   for (int i = 0; i < ctx.ser_ref_levels; ++i)
   {
      fluid_mesh->UniformRefinement();
      struct_mesh.UniformRefinement();
   }

   H1_FECollection fe_coll(ctx.order, 1);
   FiniteElementSpace fespace(&struct_mesh, &fe_coll);
   
   std::cout << "Number of structure finite element unknowns: "
        << fespace.GetTrueVSize() << std::endl;

   Array<int> ess_bdr(struct_mesh.bdr_attributes.Max());
   for (int i = 0; i < ess_bdr.Size(); ++i){
      ess_bdr[i] = 1;
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << fluid_mesh->GetNE() << std::endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *fluid_mesh);
   delete fluid_mesh;

   // Create the flow solver.
   NavierSolver naviersolver(pmesh, ctx.order, ctx.kinvis);
   naviersolver.EnablePA(ctx.pa);
   naviersolver.EnableNI(ctx.ni);
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;
   ParGridFunction *u_next_gf = nullptr;
   GridFunction *wg_gf = nullptr;

   // Set the initial condition.
   ParGridFunction *u_ic = naviersolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   u_ic->ProjectCoefficient(u_excoeff);

   FunctionCoefficient p_excoeff(p);

   GridFunction eta_gf(&fespace);
   GridFunction dEtaDt_gf(&fespace);

   FunctionCoefficient eta_0(StructureInitialSolution);
   eta_gf.ProjectCoefficient(eta_0);
   Vector eta;
   eta_gf.GetTrueDofs(eta);

   FunctionCoefficient dEtaDt_0(StructureInitialRate);
   dEtaDt_gf.ProjectCoefficient(dEtaDt_0);
   Vector dEtaDt;
   dEtaDt_gf.GetTrueDofs(dEtaDt);

   // And from fluid to structure transfer.
   FunctionCoefficient fluid_forcing_coef( [&](const Vector &c) -> double {
            double x = c(0);

            // Extract the eta value.
            double y, dEtaDx_val;
            {
            // Put this stuff in a disposable scope so that we don't have to deal with name collisions for very temporary things.
            DenseMatrix points(struct_mesh.Dimension(), 1);
            points.SetCol(0, x);

            Array<int> elem_ids;
            Array<IntegrationPoint> ips;

            struct_mesh.FindPoints(points, elem_ids, ips);
            y = eta_gf.GetValue(elem_ids[0], ips[0]);
            /*ElementTransformation* T = struct_mesh.GetElementTransformation(elem_ids[0]);
            T->SetIntPoint(&ips[0]);
            T->ElementNo = elem_ids[0];*/
            Vector dedx_vec(1);
            eta_gf.GetGradient(*struct_mesh.GetElementTransformation(elem_ids[0]),dedx_vec);
            dEtaDx_val = dedx_vec[0];
            //delete T;
            }

            // Now extract the required velocity derivative and pressure values.
            DenseMatrix points(pmesh->Dimension(),1);
            Vector coords(pmesh->Dimension());
            coords[0] = x;
            coords[1] = y;
            //coords.Print();
            points.SetCol(0,coords);
            //points.Print(); // <- Possible undefined behavior in points. If this line is //ed out then first structure solve doesn't work. If it is in, then it works.
            //std::cout << x << "  " << y << std::endl;

            Array<int> elem_ids;
            Array<IntegrationPoint> ips;

            pmesh->FindPoints(points, elem_ids, ips); // Parallel issues??? Some condense operation probably needed below or perhaps it is more safe to do so after RHS assembly.

            switch (elem_ids[0])
            {
            case -2:
               // This point was found on another process. TODO: We will need to implement a MPI all reduce with a sum for the fluid_forcing vector inside shell operator.
               // Should never happen if we run with one process in the meantime.
               std::cout << "Point not found on THIS process." << std::endl; // <- Should be removed when no longer testing in serial.
               return 0.0;

            case -1:
               if (mpi.Root())
                  //std::cout << "Point not found on ANY process." << std::endl; // <- Looks like mfem auto warns here.
               return 0.0/0.0; // <- for some reason C++ would not let me directly return NaN here?
            
            default: // Do nothing, we must have found the point on the locally owned part of the mesh.
               break;
            }

            double p = p_gf->GetValue(elem_ids[0],ips[0]);

            DenseMatrix grad(2,2);
            /*auto T = pmesh->GetElementTransformation(elem_ids[0]);
            T->SetIntPoint(&ips[0]);
            T->ElementNo = elem_ids[0];*/
            u_gf->GetVectorGradient(*pmesh->GetElementTransformation(elem_ids[0]),grad);
            double dudy = grad(0,1), dvdx = grad(1,0), dvdy = grad(1,1);
            double n2 = 1/sqrt(1 + dEtaDx_val * dEtaDx_val);
            double n1 = -dEtaDx_val * n2;

            return p*n2 - ctx.kinvis * ((dudy + dvdx) * n1 + 2 * dvdy * n2);
   });
   
   // Define the helper function that will transfer velocity information from the structure to fluid.
   VectorFunctionCoefficient shell_vel_coef(
        pmesh->Dimension(), [&](const Vector &c, Vector &u) {
            double x = c(0);
            double y = c(1);

            DenseMatrix points(struct_mesh.Dimension(), 1);
            points.SetCol(0, x);

            Array<int> elem_ids;
            Array<IntegrationPoint> ips;

            struct_mesh.FindPoints(points, elem_ids, ips);

            u(0) = 0.0;
            u(1) = dEtaDt_gf.GetValue(elem_ids[0], ips[0]);
        });
   
   // Define the helper function that will transfer location information from the structure to fluid.
   VectorFunctionCoefficient shell_loc_coef(
        pmesh->Dimension(), [&](const Vector &c, Vector &u) {
            double x = c(0);
            double y = c(1);

            DenseMatrix points(struct_mesh.Dimension(), 1);
            points.SetCol(0, x);

            Array<int> elem_ids;
            Array<IntegrationPoint> ips;

            struct_mesh.FindPoints(points, elem_ids, ips);

            u(0) = x;
            u(1) = eta_gf.GetValue(elem_ids[0], ips[0]) * y;
        });
   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the fluid_mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 1;
   attr[2] = 0;
   naviersolver.AddVelDirichletBC(vel, attr);
   attr = 0;
   attr[2] = 1;
   naviersolver.AddVelDirichletBC(&shell_vel_coef,attr);

   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   naviersolver.AddAccelTerm(accel, domain_attr);

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   naviersolver.SetMaxBDFOrder(2);

   naviersolver.Setup(dt);

   double errl2_u = 0.0;
   double errlinf_u = 0.0;
   double errl2_p = 0.0;
   double errlinf_p = 0.0;
   double err_eta = 0.0;
   u_next_gf = naviersolver.GetProvisionalVelocity();
   u_next_gf->ProjectCoefficient(u_excoeff);
   u_gf = naviersolver.GetCurrentVelocity();
   p_gf = naviersolver.GetCurrentPressure();
   wg_gf = naviersolver.GetCurrentMeshVelocity();

   // This is just to initialize some datastructres in navier
   p_gf->ProjectCoefficient(p_excoeff);
   naviersolver.MeanZero(*p_gf);
   *p_gf = 0.0;

   // Create the structure solver.
   ShellOperator oper(fespace, ess_bdr, fluid_forcing_coef);
   std::cout << "here" << std::endl;
   SecondOrderODESolver *ode_solver = new NewmarkSolver();
   std::cout << "here" << std::endl;
   ode_solver->Init(oper);
   std::cout << "here" << std::endl;

   ParGridFunction u_ex(*u_gf);
   ParGridFunction p_ex(*p_gf);
   double L2L2_err = 0.0;
   std::cout << "here" << std::endl;

   auto userchk = [&](int step)
   {
      u_ex.ProjectCoefficient(u_excoeff);
      for (int i = 0; i < u_ex.Size(); ++i)
      {
         u_ex[i] = (*u_next_gf)[i] - u_ex[i];
      }

      p_ex.ProjectCoefficient(p_excoeff);
      for (int i = 0; i < p_ex.Size(); ++i)
      {
         p_ex[i] = (*p_gf)[i] - p_ex[i];
      }

      // Compare against exact solution of velocity and pressure.
      errl2_u = naviersolver.NekNorm(u_ex, 0, true);
      errl2_p = naviersolver.NekNorm(p_ex, 0, false);
      errlinf_u = naviersolver.NekNorm(u_ex, 1, true);
      errlinf_p = naviersolver.NekNorm(p_ex, 1, false);

      if (mpi.Root())
      {
         if (step == -1)
         {
            printf("%11s %11s %11s %11s %11s %11s %11s errlog\n",
                   "time",
                   "dt",
                   "errl2_u",
                   "errl2_p",
                   "errlinf_u",
                   "errlinf_p",
                   "err_eta");
         }
         printf("%11.5E %11.5E %11.5E %11.5E %11.5E %11.5E %11.5E errlog\n",
                t,
                dt,
                errl2_u,
                errl2_p,
                errlinf_u,
                errlinf_p,
                err_eta);
         fflush(stdout);
      }
   };
   double L2L2_u = 0.0;

   userchk(-1);

   auto xi_0 = new GridFunction(*pmesh->GetNodes());
   std::cout << "here" << std::endl;

   VectorFunctionCoefficient
   mesh_nodes(2, [&](const Vector &cin, double t, Vector &cout)
   {
      double x = cin(0);
      double y = cin(1);
      double eta = ctx.a / (M_PI * ctx.ft) * sin(ctx.fx * M_PI * x)
                   * sin(ctx.ft * M_PI * t);
      cout(0) = x;
      cout(1) = (1.0 + eta) * y;
   });

   VectorFunctionCoefficient
   mesh_nodes_velocity(2, [&](const Vector &cin, double t, Vector &cout)
   {
      double x = cin(0);
      double y = cin(1);
      cout(0) = 0.0;
      cout(1) = ctx.a * y * cos(ctx.ft * M_PI * t) * sin(ctx.fx * M_PI * x);
   });

   ParaViewDataCollection paraview_dc("fsimms_coupled", pmesh);
   paraview_dc.SetLevelsOfDetail(ctx.order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(t);
   paraview_dc.RegisterField("velocity", u_gf);
   paraview_dc.RegisterField("pressure", p_gf);
   paraview_dc.RegisterField("velocity_error", &u_ex);
   paraview_dc.RegisterField("pressure_error", &p_ex);

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      ode_solver->Step(eta, dEtaDt, t, dt);

      if (last_step || (step % vis_steps) == 0)
      {
         std::cout << "\nstep " << step << ", t = " << t << std::endl;
      }

      eta_gf.SetFromTrueDofs(eta);
      dEtaDt_gf.SetFromTrueDofs(dEtaDt);

      int order_quad = std::max(2, 2*ctx.order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      FunctionCoefficient etacoeff (StructureExactSolution);
      GridFunctionCoefficient eta_approx_coeff(&eta_gf);
      etacoeff.SetTime(t);
      eta_gf.SetFromTrueDofs(eta);
      double err_eta  = eta_gf.ComputeL2Error(etacoeff, irs);
      double norm_eta = ComputeLpNorm(2., etacoeff, struct_mesh, irs);
      double norm_eta_a = ComputeLpNorm(2.,eta_approx_coeff,struct_mesh,irs);

      if (verbose){
         for (int i = 0; i < eta.Size(); ++i){
            std::cout << i << "  " << eta[i] << "\n";
         }
      }
      /*if(mpi.Root()){
         std::cout << "|| u_h - u_ex || = " << err_eta << "\n";
         std::cout << "|| u_ex || = " << norm_eta  << "\n";
         std::cout << "|| u_h || = " << norm_eta_a << "\n";
      }*/

      L2L2_err += err_eta * err_eta;
      // The ODE solver automatically steps time. We need to undo this for the fluid solve and then advance time AFTER the fluid solve.
      t -= dt;

      *pmesh->GetNodes() = *xi_0;

      mesh_nodes_velocity.SetTime(t + dt);
      wg_gf->ProjectCoefficient(shell_vel_coef);
      // wg_gf->ProjectCoefficient(mesh_nodes_velocity);

      mesh_nodes.SetTime(t + dt);
      //naviersolver.TransformMesh(mesh_nodes);
      naviersolver.TransformMesh(shell_loc_coef);

      naviersolver.Step(t, dt, step, true);
      t += dt;

      u_excoeff.SetTime(t);
      p_excoeff.SetTime(t);

      /*if (step < 5)
      {
         u_next_gf->ProjectCoefficient(u_excoeff);
         p_gf->ProjectCoefficient(p_excoeff);
      }*/

      userchk(step);
      L2L2_u += errl2_u * errl2_u;

      naviersolver.UpdateTimestepHistory(dt);

      paraview_dc.SetCycle(step + 1);
      paraview_dc.SetTime(t);
      paraview_dc.Save();
   }

   if (ctx.visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank()
             << "\n";
      u_sock << "solution\n" << *pmesh << u_ex << std::flush;

      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank()
             << "\n";
      p_sock << "solution\n" << *pmesh << p_ex << std::flush;
   }

   naviersolver.PrintTimingData();
   if (mpi.Root())
   {
      std::cout << "Overall Errors\n";
      std::cout << "|| eta_h - eta_ex ||_L2L2 = " << sqrt(dt*L2L2_err) << std::endl;
      std::cout << "|| u_h - u_ex ||_L2L2 = " << sqrt(dt*L2L2_u) << std::endl;
   }

   delete pmesh;
   delete ode_solver;

   return 0;
}