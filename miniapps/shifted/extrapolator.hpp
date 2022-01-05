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
   double dt = 0.0;

   DiscreteUpwindLOSolver *lo_solver;
   FluxBasedFCT *fct_solver;
   Vector *lumpedM;

   void ComputeElementsMinMax(const ParGridFunction &gf,
                              Vector &el_min, Vector &el_max) const;
   void ComputeBounds(const ParFiniteElementSpace &pfes,
                      const Vector &el_min, const Vector &el_max,
                      Vector &dof_min, Vector &dof_max) const;
   void ZeroOutInactiveZones(Vector &dx);

public:
   // 0 is stanadard HO; 1 is upwind diffusion; 2 is FCT.
   enum AdvectionMode {HO, LO, FCT} adv_mode = HO;

   AdvectionOper(Array<bool> &zones, ParBilinearForm &Mbf,
                 ParBilinearForm &Kbf, const Vector &rhs, AdvectionMode mode);

   ~AdvectionOper();

   virtual void Mult(const Vector &x, Vector &dx) const;

   void SetDt(double delta_t) { dt = delta_t; }
};

class Extrapolator
{
public:
   enum XtrapType {ASLAM, BOCHKOV} xtrap_type = ASLAM;
   AdvectionOper::AdvectionMode dg_mode = AdvectionOper::HO;
   int xtrap_order    = 1;
   bool visualization = false;
   int vis_steps      = 5;

   Extrapolator() { }

   // The known values taken from elements where level_set > 0, and extrapolated
   // to all other elements. The known values are not changed.
   void Extrapolate(Coefficient &level_set, const ParGridFunction &input,
                    ParGridFunction &xtrap);

   // Errors in cut elements, given an exact solution.
   void ComputeLocalErrors(Coefficient &level_set, const ParGridFunction &exact,
                           const ParGridFunction &xtrap,
                           double &err_L1, double &err_L2, double &err_LI);

private:
   void TimeLoop(ParGridFunction &sltn, ODESolver &ode_solver,
                 double dt, int vis_x_pos, std::string vis_name);
};

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

class DiscreteUpwindLOSolver
{
public:
   DiscreteUpwindLOSolver(ParFiniteElementSpace &space, const SparseMatrix &adv,
                          const Vector &Mlump);

   virtual void CalcLOSolution(const Vector &u, const Vector &rhs,
                               Vector &du) const;

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

class FluxBasedFCT
{
public:
   FluxBasedFCT(ParFiniteElementSpace &space, double &delta_t,
                const SparseMatrix &adv_mat, const Array<int> &adv_smap,
                const SparseMatrix &mass_mat)
      : pfes(space), dt(delta_t),
        K(adv_mat), M(mass_mat), K_smap(adv_smap), flux_ij(adv_mat),
        gp(&pfes), gm(&pfes) { }

   void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                        const Vector &du_ho, const Vector &du_lo,
                        const Vector &u_min, const Vector &u_max,
                        Vector &du) const;

protected:
   ParFiniteElementSpace &pfes;
   double &dt;

   const SparseMatrix &K, &M;
   const Array<int> &K_smap;

   // Temporary computation objects.
   mutable SparseMatrix flux_ij;
   mutable ParGridFunction gp, gm;

   void ComputeFluxMatrix(const ParGridFunction &u, const Vector &du_ho,
                          SparseMatrix &flux_mat) const;
   void AddFluxesAtDofs(const SparseMatrix &flux_mat,
                        Vector &flux_pos, Vector &flux_neg) const;
   void ComputeFluxCoefficients(const Vector &u, const Vector &du_lo,
      const Vector &m, const Vector &u_min, const Vector &u_max,
      Vector &coeff_pos, Vector &coeff_neg) const;
   void UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
      ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
      SparseMatrix &flux_mat, Vector &du) const;
};

} // namespace mfem

#endif
