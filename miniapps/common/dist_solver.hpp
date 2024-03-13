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

namespace mfem
{

namespace common
{

double AvgElementSize(Mesh &mesh);

class DistanceSolver
{
protected:
   void ScalarDistToVector(GridFunction *dist_s, GridFunction *dist_v);
#ifdef MFEM_USE_MPI
   void ScalarDistToVector(ParGridFunction &dist_s, ParGridFunction &dist_v);
#endif

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
                                      GridFunction *distance) = 0;

#ifdef MFEM_USE_MPI
   // The computed distance is not "signed". In addition to the standard usage
   // (with zero level sets), this function can be applied to point sources when
   // transform = false.
   virtual void ComputeScalarDistance(Coefficient &zero_level_set,
                                      ParGridFunction *distance) = 0;
#endif

   // Computes a vector ParGridFunction where the magnitude is the length of the
   // shortest path to the zero level set of the given Coefficient, and the
   // direction is the starting direction of the shortest path. It is expected
   // that the given [distance] has a valid (vector) ParFiniteElementSpace, and
   // that the result is computed in the same space.
   virtual void ComputeVectorDistance(Coefficient &zero_level_set,
                                      GridFunction *distance);
};

// K. Crane et al: "Geodesics in Heat: A New Approach to Computing Distance
// Based on Heat Flow", DOI:10.1145/2516971.2516977.
class HeatDistanceSolver : public DistanceSolver
{
public:
   HeatDistanceSolver(double diff_coeff)
      : DistanceSolver(), parameter_t(diff_coeff), smooth_steps(0),
        diffuse_iter(1), transform(true), vis_glvis(false) { }

#ifdef MFEM_USE_MPI
   // The computed distance is not "signed". In addition to the standard usage
   // (with zero level sets), this function can be applied to point sources when
   // transform = false.
   void ComputeScalarDistance(Coefficient &zero_level_set,
                              ParGridFunction *distance);
#endif
   void ComputeScalarDistance(Coefficient &zero_level_set,
                              GridFunction *distance);

   double parameter_t;
   int smooth_steps, diffuse_iter;
   bool transform, vis_glvis;
};

// Formulation for the ScreenedPoisson equation. The positive part of the input
// coefficient supply unit volumetric loading, the negative part - negative unit
// volumetric loading. The parameter rh is the radius of a linear cone filter
// which will deliver similar smoothing effect as the Screened Poisson
// equation. It determines the length scale of the smoothing.
class ScreenedPoisson: public NonlinearFormIntegrator
{
protected:
   double diffcoef;
   Coefficient *func;

public:
   ScreenedPoisson(Coefficient &nfunc, double rh):func(&nfunc)
   {
      double rd=rh/(2*std::sqrt(3.0));
      diffcoef= rd*rd;
   }

   ~ScreenedPoisson() { }

   void SetInput(Coefficient &nfunc) { func = &nfunc; }

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &trans,
                                   const Vector &elfun) override;

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect) override;

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat) override;
};

class PDEFilter
{
public:
#ifdef MFEM_USE_MPI
   PDEFilter(ParMesh &mesh, double rh, int order = 2,
             int maxiter = 100, double rtol = 1e-12,
             double atol = 1e-15, int print_lv = 0)
      : rr(rh),
        fecp(order, mesh.Dimension()),
        fesp(NULL),
        gf(NULL),
        nf(NULL),
        sv(NULL),
        pfesp(new ParFiniteElementSpace(&mesh, &fecp, 1)),
        pgf(new ParGridFunction(pfesp))
   {
      psv = pfesp->NewTrueDofVector();

      pnf = new ParNonlinearForm(pfesp);
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
#endif

   PDEFilter(Mesh &mesh, double rh, int order = 2,
             int maxiter = 100, double rtol = 1e-12,
             double atol = 1e-15, int print_lv = 0)
      : rr(rh),
        fecp(order, mesh.Dimension()),
        fesp(new FiniteElementSpace(&mesh, &fecp, 1)),
        gf(new GridFunction(fesp))
#ifdef MFEM_USE_MPI
      ,pfesp(NULL),
        pgf(NULL)
#endif
   {
      sv = new Vector(gf->GetTrueVector());
      nf = new NonlinearForm(fesp);

      gmres = new GMRESSolver;
      gmres->SetAbsTol(atol);
      gmres->SetRelTol(rtol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(print_lv);

      sint=nullptr;
   }

   ~PDEFilter()
   {
      if (gmres) { delete gmres; }
      if (nf) { delete nf; }
      if (sv) { delete sv; }
      if (gf) { delete gf; }
      if (fesp) { delete fesp; }
#ifdef MFEM_USE_MPI
      delete prec;
      delete pnf;
      delete psv;
      delete pgf;
      delete pfesp;
#endif
   }

#ifdef MFEM_USE_MPI
   void Filter(ParGridFunction &func, ParGridFunction &ffield)
   {
      GridFunctionCoefficient gfc(&func);
      Filter(gfc, ffield);
   }

   void Filter(Coefficient &func, ParGridFunction &ffield);
#endif

   void Filter(GridFunction &func, GridFunction &ffield)
   {
      GridFunctionCoefficient gfc(&func);
      Filter(gfc, ffield);
   }

   void Filter(Coefficient &func, GridFunction &ffield);

private:
   const double rr;
   H1_FECollection fecp;
   FiniteElementSpace *fesp;
   GridFunction *gf;
   NonlinearForm* nf;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfesp;
   ParGridFunction *pgf;
   ParNonlinearForm* pnf;
   HypreBoomerAMG* prec;
   HypreParVector *psv;
#endif
   GMRESSolver *gmres;
   Vector *sv;

   ScreenedPoisson* sint;
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
      const double norm = V.Norml2() + 1e-12;
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

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
      double u = basef.Eval(T,ip);
      double c = corrf.Eval(T,ip);
      if (u<0.0) { u*=-1.0; }
      return u*c;
   }
};

class PUMPLaplacian: public NonlinearFormIntegrator
{

protected:
   Coefficient *func;
   VectorCoefficient *fgrad;
   bool ownership;
   double pp, ee;

public:
   // The VectorCoefficent should contain a vector with entries:
   // [0] - derivative with respect to x
   // [1] - derivative with respect to y
   // [2] - derivative with respect to z
   PUMPLaplacian(Coefficient *nfunc, VectorCoefficient *nfgrad,
                 bool ownership_=true)
      : func(nfunc), fgrad(nfgrad), ownership(ownership_), pp(2.0), ee(1e-7) { }

   void SetPower(double pp_) { pp = pp_; }
   void SetReg(double ee_)   { ee = ee_; }

   virtual ~PUMPLaplacian()
   {
      if (ownership)
      {
         delete func;
         delete fgrad;
      }
   }

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &trans,
                                   const Vector &elfun) override;

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect) override;

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat) override;
};

// A. Belyaev et al: "On Variational and PDE-based Distance Function
// Approximations", Section 7, DOI:10.1111/cgf.12611.
class PLapDistanceSolver : public DistanceSolver
{
public:
   PLapDistanceSolver(int maxp_ = 30, int newton_iter_ = 10,
                      double rtol = 1e-7, double atol = 1e-12)
      : maxp(maxp_), newton_iter(newton_iter_),
        newton_rel_tol(rtol), newton_abs_tol(atol) { }

   void SetMaxPower(int new_pp) { maxp = new_pp; }

   // The computed distance is "signed".
   void ComputeScalarDistance(Coefficient& func, GridFunction* fdist);
#ifdef MFEM_USE_MPI
   // The computed distance is "signed".
   void ComputeScalarDistance(Coefficient& func, ParGridFunction* fdist);
#endif

private:
   int maxp; // maximum value of the power p
   const int newton_iter;
   const double newton_rel_tol, newton_abs_tol;
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
      GridFunction &u;

   public:
      NormalizationCoeff(GridFunction &u_gf) : u(u_gf) { }
      virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   };

public:
   NormalizationDistanceSolver() { }

   void ComputeScalarDistance(Coefficient& u_coeff, GridFunction* dist);

#ifdef MFEM_USE_MPI
   void ComputeScalarDistance(Coefficient& u_coeff, ParGridFunction* dist);
#endif
};

} // namespace common

} // namespace mfem

#endif
