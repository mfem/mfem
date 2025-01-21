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

#ifndef MFEM_DIST_SOLVER_HPP
#define MFEM_DIST_SOLVER_HPP

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace common
{

real_t AvgElementSize(ParMesh &pmesh);

class DistanceSolver
{
protected:
   void ScalarDistToVector(ParGridFunction &dist_s, ParGridFunction &dist_v);

public:
   // 0 = nothing, 1 = main solver only, 2 = full (solver + preconditioner).
   IterativeSolver::PrintLevel print_level;

   DistanceSolver() { }
   virtual ~DistanceSolver() { }

   // Computes a scalar ParGridFunction which is the length of the shortest path
   // to the zero level set of the given Coefficient. It is expected that the
   // given [distance] has a valid (scalar) ParFiniteElementSpace, and that the
   // result is computed in the same space. Some implementations may output a
   // "signed" distance, i.e., the distance has different signs on both sides of
   // the zero level set.
   virtual void ComputeScalarDistance(Coefficient &zero_level_set,
                                      ParGridFunction &distance) = 0;

   // Computes a vector ParGridFunction where the magnitude is the length of the
   // shortest path to the zero level set of the given Coefficient, and the
   // direction is the starting direction of the shortest path. It is expected
   // that the given [distance] has a valid (vector) ParFiniteElementSpace, and
   // that the result is computed in the same space.
   virtual void ComputeVectorDistance(Coefficient &zero_level_set,
                                      ParGridFunction &distance);
};


// K. Crane et al: "Geodesics in Heat: A New Approach to Computing Distance
// Based on Heat Flow", DOI:10.1145/2516971.2516977.
class HeatDistanceSolver : public DistanceSolver
{
public:
   HeatDistanceSolver(real_t diff_coeff)
      : DistanceSolver(), parameter_t(diff_coeff), smooth_steps(0),
        diffuse_iter(1), transform(true), vis_glvis(false) { }

   // The computed distance is not "signed". In addition to the standard usage
   // (with zero level sets), this function can be applied to point sources when
   // transform = false.
   void ComputeScalarDistance(Coefficient &zero_level_set,
                              ParGridFunction &distance);

   real_t parameter_t;
   int smooth_steps, diffuse_iter;
   bool transform, vis_glvis;
};

// A. Belyaev et al: "On Variational and PDE-based Distance Function
// Approximations", Section 6, DOI:10.1111/cgf.12611.
// This solver is computationally cheap, but is accurate for distance
// approximations only near the zero level set.
class NormalizationDistanceSolver : public DistanceSolver
{
private:

   class NormalizationCoeff : public Coefficient
   {
   private:
      ParGridFunction &u;

   public:
      NormalizationCoeff(ParGridFunction &u_gf) : u(u_gf) { }
      real_t Eval(ElementTransformation &T,
                  const IntegrationPoint &ip) override;
   };

public:
   NormalizationDistanceSolver() { }

   void ComputeScalarDistance(Coefficient& u_coeff, ParGridFunction& dist);
};


// A. Belyaev et al: "On Variational and PDE-based Distance Function
// Approximations", Section 7, DOI:10.1111/cgf.12611.
class PLapDistanceSolver : public DistanceSolver
{
public:
   PLapDistanceSolver(int maxp_ = 30, int newton_iter_ = 10,
                      real_t rtol = 1e-7, real_t atol = 1e-12)
      : maxp(maxp_), newton_iter(newton_iter_),
        newton_rel_tol(rtol), newton_abs_tol(atol) { }

   void SetMaxPower(int new_pp) { maxp = new_pp; }

   // The computed distance is "signed".
   void ComputeScalarDistance(Coefficient& func, ParGridFunction& fdist);

private:
   int maxp; // maximum value of the power p
   const int newton_iter;
   const real_t newton_rel_tol, newton_abs_tol;
};

class NormalizedGradCoefficient : public VectorCoefficient
{
private:
   const GridFunction &u;

public:
   NormalizedGradCoefficient(const GridFunction &u_gf, int dim)
      : VectorCoefficient(dim), u(u_gf) { }

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);

      u.GetGradient(T, V);
      const real_t norm = V.Norml2() + 1e-12;
      V /= -norm;
   }
};


// Product of the modulus of the first coefficient and the second coefficient
class PProductCoefficient : public Coefficient
{
private:
   Coefficient &basef, &corrf;

public:
   PProductCoefficient(Coefficient& basec_, Coefficient& corrc_)
      : basef(basec_), corrf(corrc_) { }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      real_t u = basef.Eval(T,ip);
      real_t c = corrf.Eval(T,ip);
      if (u<0.0) { u*=-1.0; }
      return u*c;
   }
};


// Formulation for the ScreenedPoisson equation. The positive part of the input
// coefficient supply unit volumetric loading, the negative part - negative unit
// volumetric loading. The parameter rh is the radius of a linear cone filter
// which will deliver similar smoothing effect as the Screened Poisson
// equation. It determines the length scale of the smoothing.
class ScreenedPoisson: public NonlinearFormIntegrator
{
protected:
   real_t diffcoef;
   Coefficient *func;

public:
   ScreenedPoisson(Coefficient &nfunc, real_t rh):func(&nfunc)
   {
      real_t rd=rh/(2*std::sqrt(3.0));
      diffcoef= rd*rd;
   }

   ~ScreenedPoisson() { }

   void SetInput(Coefficient &nfunc) { func = &nfunc; }

   real_t GetElementEnergy(const FiniteElement &el,
                           ElementTransformation &trans,
                           const Vector &elfun) override;

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &trans,
                              const Vector &elfun,
                              Vector &elvect) override;

   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &trans,
                            const Vector &elfun,
                            DenseMatrix &elmat) override;
};


class PUMPLaplacian: public NonlinearFormIntegrator
{

protected:
   Coefficient *func;
   VectorCoefficient *fgrad;
   bool ownership;
   real_t pp, ee;

public:
   // The VectorCoefficent should contain a vector with entries:
   // [0] - derivative with respect to x
   // [1] - derivative with respect to y
   // [2] - derivative with respect to z
   PUMPLaplacian(Coefficient *nfunc, VectorCoefficient *nfgrad,
                 bool ownership_=true)
      : func(nfunc), fgrad(nfgrad), ownership(ownership_), pp(2.0), ee(1e-7) { }

   void SetPower(real_t pp_) { pp = pp_; }
   void SetReg(real_t ee_)   { ee = ee_; }

   virtual ~PUMPLaplacian()
   {
      if (ownership)
      {
         delete func;
         delete fgrad;
      }
   }

   real_t GetElementEnergy(const FiniteElement &el,
                           ElementTransformation &trans,
                           const Vector &elfun) override;

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &trans,
                              const Vector &elfun,
                              Vector &elvect) override;

   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &trans,
                            const Vector &elfun,
                            DenseMatrix &elmat) override;
};

// Low-pass filter based on the Screened Poisson equation.
// B. S. Lazarov, O. Sigmund: "Filters in topology optimization based on
// Helmholtz-type differential equations", DOI:10.1002/nme.3072.
class PDEFilter
{
public:
   PDEFilter(ParMesh &mesh, real_t rh, int order = 2,
             int maxiter = 100, real_t rtol = 1e-12,
             real_t atol = 1e-15, int print_lv = 0)
      : rr(rh),
        fecp(order, mesh.Dimension()),
        fesp(&mesh, &fecp, 1),
        gf(&fesp)
   {
      sv = fesp.NewTrueDofVector();

      nf = new ParNonlinearForm(&fesp);
      prec = new HypreBoomerAMG();
      prec->SetPrintLevel(print_lv);

      gmres = new GMRESSolver(mesh.GetComm());

      gmres->SetAbsTol(atol);
      gmres->SetRelTol(rtol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(print_lv);
      gmres->SetPreconditioner(*prec);

      sint=nullptr;
   }

   ~PDEFilter()
   {
      delete gmres;
      delete prec;
      delete nf;
      delete sv;
   }

   void Filter(ParGridFunction &func, ParGridFunction &ffield)
   {
      GridFunctionCoefficient gfc(&func);
      Filter(gfc, ffield);
   }

   void Filter(Coefficient &func, ParGridFunction &ffield);

private:
   const real_t rr;
   H1_FECollection fecp;
   ParFiniteElementSpace fesp;
   ParGridFunction gf;

   ParNonlinearForm* nf;
   HypreBoomerAMG* prec;
   GMRESSolver *gmres;
   HypreParVector *sv;

   ScreenedPoisson* sint;
};

} // namespace common

} // namespace mfem

#endif // MFEM_USE_MPI
#endif
