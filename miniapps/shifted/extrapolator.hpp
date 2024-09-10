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

#ifndef MFEM_EXTRAPOLATOR_HPP
#define MFEM_EXTRAPOLATOR_HPP

#include "mfem.hpp"

namespace mfem
{

class DiscreteUpwindLOSolver;
class FluxBasedFCT;

class AdvectionOper : public TimeDependentOperator
{
private:
   Array<bool> &active_zones;
   ParBilinearForm &M, &K;
   HypreParMatrix *K_mat;
   const Vector &b;

   DiscreteUpwindLOSolver *lo_solver;
   Vector *lumpedM;

   void ComputeElementsMinMax(const ParGridFunction &gf,
                              Vector &el_min, Vector &el_max) const;
   void ComputeBounds(const ParFiniteElementSpace &pfes,
                      const Vector &el_min, const Vector &el_max,
                      Vector &dof_min, Vector &dof_max) const;
   void ZeroOutInactiveZones(Vector &dx);

public:
   // HO is standard FE advection solve; LO is upwind diffusion.
   enum AdvectionMode {HO, LO} adv_mode = AdvectionOper::HO;

   AdvectionOper(Array<bool> &zones, ParBilinearForm &Mbf,
                 ParBilinearForm &Kbf, const Vector &rhs);

   ~AdvectionOper();

   void Mult(const Vector &x, Vector &dx) const override;
};

// Extrapolates through DG advection based on:
// [1] Aslam, "A Partial Differential Equation Approach to Multidimensional
// Extrapolation", JCP 193(1), 2004.
// [2] Bochkov, Gibou, "PDE-Based Multidimensional Extrapolation of Scalar
// Fields over Interfaces with Kinks and High Curvatures", SISC 42(4), 2020.
class Extrapolator
{
public:
   enum XtrapType {ASLAM, BOCHKOV} xtrap_type = ASLAM;
   AdvectionOper::AdvectionMode advection_mode = AdvectionOper::HO;
   int xtrap_degree   = 1;
   bool visualization = false;
   int vis_steps      = 5;

   Extrapolator() { }

   // The known values taken from elements where level_set > 0, and extrapolated
   // to all other elements. The known values are not changed.
   void Extrapolate(Coefficient &level_set, const ParGridFunction &input,
                    const real_t time_period, ParGridFunction &xtrap, int visport = 19916);

   // Errors in cut elements, given an exact solution.
   void ComputeLocalErrors(Coefficient &level_set, const ParGridFunction &exact,
                           const ParGridFunction &xtrap,
                           real_t &err_L1, real_t &err_L2, real_t &err_LI);

private:
   void TimeLoop(ParGridFunction &sltn, ODESolver &ode_solver, real_t t_final,
                 real_t dt, int vis_x_pos, std::string vis_name, int visport = 19916);
};

class LevelSetNormalGradCoeff : public VectorCoefficient
{
private:
   const ParGridFunction &ls_gf;

public:
   LevelSetNormalGradCoeff(const ParGridFunction &ls) :
      VectorCoefficient(ls.ParFESpace()->GetMesh()->Dimension()), ls_gf(ls) { }

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector grad_ls(vdim), n(vdim);
      ls_gf.GetGradient(T, grad_ls);
      const real_t norm_grad = grad_ls.Norml2();
      V = grad_ls;
      if (norm_grad > 0.0) { V /= norm_grad; }

      // Since positive level set values correspond to the known region, we
      // transport into the opposite direction of the gradient.
      V *= -1;
   }
};

class GradComponentCoeff : public Coefficient
{
private:
   const ParGridFunction &u_gf;
   int comp;

public:
   GradComponentCoeff(const ParGridFunction &u, int c) : u_gf(u), comp(c) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
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

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
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

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      const int dim = T.GetDimension();
      Vector n(dim), grad_u(dim);
      n_coeff.Eval(n, T, ip);
      grad_u(0) = du_dx.GetValue(T, ip);
      grad_u(1) = du_dy.GetValue(T, ip);
      return n * grad_u;
   }
};

class DiscreteUpwindLOSolver
{
public:
   DiscreteUpwindLOSolver(ParFiniteElementSpace &space, const SparseMatrix &adv,
                          const Vector &Mlump);

   void CalcLOSolution(const Vector &u, const Vector &rhs, Vector &du) const;

   Array<int> &GetKmap() { return K_smap; }

protected:
   ParFiniteElementSpace &pfes;
   const SparseMatrix &K;
   mutable SparseMatrix D;

   Array<int> K_smap;
   const Vector &M_lumped;

   void ComputeDiscreteUpwindMatrix() const;
   void ApplyDiscreteUpwindMatrix(ParGridFunction &u, Vector &du) const;
};

} // namespace mfem

#endif
