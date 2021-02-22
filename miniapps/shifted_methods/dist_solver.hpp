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

#ifndef MFEM_DIST_FUNCTION_HPP
#define MFEM_DIST_FUNCTION_HPP

#include "mfem.hpp"

namespace mfem
{

class DistanceSolver
{
protected:
   void ScalarDistToVector(ParGridFunction &dist_s, ParGridFunction &dist_v);

public:
   // 0 is nothing / 1 is the main solver / 2 is full (solver + precond).
   int print_level = 0;

   DistanceSolver() { }
   virtual ~DistanceSolver() { }

   // Computes a scalar ParGridFunction which is the length of the shortest
   // path to the zero level set of the given Coefficient.
   // It is expected that [distance] has a valid (scalar) ParFiniteElementSpace,
   // and the result is computed in the same space.
   // Some implementations may output a "signed" distance, i.e., the distance
   // has different signs on both sides of the zeto level set.
   virtual void ComputeScalarDistance(Coefficient &zero_level_set,
                                      ParGridFunction &distance) = 0;

   // Computes a vector ParGridFunction where the magnitude is the length of
   // the shortest path to the zero level set of the given Coefficient, and
   // the direction is the starting direction of the shortest path.
   // It is expected that [distance] has a valid (vector) ParFiniteElementSpace,
   // and the result is computed in the same space.
   virtual void ComputeVectorDistance(Coefficient &zero_level_set,
                                      ParGridFunction &distance);
};

// K. Crane et al:
// Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow
class HeatDistanceSolver : public DistanceSolver
{
public:
   HeatDistanceSolver(double diff_coeff)
      : DistanceSolver(), parameter_t(diff_coeff), smooth_steps(0),
        diffuse_iter(1), transform(true), vis_glvis(false) { }

   // The computed distance is not "signed".
   // In addition to the standard usage (with zero level sets), this function
   // can be applied to point sources when transform = false.
   void ComputeScalarDistance(Coefficient &zero_level_set,
                              ParGridFunction &distance);

   int parameter_t, smooth_steps, diffuse_iter;
   bool transform, vis_glvis;
};

class NormalizedGradCoefficient : public VectorCoefficient
{
private:
   const GridFunction &u;

public:
   NormalizedGradCoefficient(const GridFunction &u_gf, int dim)
      : VectorCoefficient(dim), u(u_gf) { }

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);

      u.GetGradient(T, V);
      const double norm = V.Norml2() + 1e-12;
      V /= -norm;
   }
};

//Product of the modulus of the first coefficient and the second coefficient
class PProductCoefficient : public Coefficient
{
private:
   Coefficient *basef, *corrf;

public:
   PProductCoefficient(Coefficient& basec_,Coefficient& corrc_)
   {
      basef=&basec_;
      corrf=&corrc_;
   }

   virtual
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
      double u=basef->Eval(T,ip);
      double c=corrf->Eval(T,ip);
      if (u<0.0) { u*=-1.0;}
      return u*c;
   }
};

//Formulation for the  ScreenedPoisson equation
//The positive part of the input coefficient supply unit volumetric loading
//The negative part - negative unit volumetric loading
//The parameter rh is the radius of a linear cone filter which will deliver
//similar smoothing effect as the Screened Poisson euation
//It determines the length scale of the smoothing.
class ScreenedPoisson: public NonlinearFormIntegrator
{
protected:
   double diffcoef;
   mfem::Coefficient* func;

public:
   ScreenedPoisson(mfem::Coefficient& nfunc, double rh):func(&nfunc)
   {
      double rd=rh/(2*std::sqrt(3.0));
      diffcoef= rd*rd;
   }

   ~ScreenedPoisson() { }

   void SetInput(mfem::Coefficient& nfunc) { func = &nfunc; }

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


/// The VectorCoefficent should return a vector with entries:
/// [0] - derivative with respect to x
/// [1] - derivative with respect to y
/// [2] - derivative with respect to z
class PUMPLaplacian: public NonlinearFormIntegrator
{

protected:
   mfem::Coefficient *func;
   mfem::VectorCoefficient *fgrad;
   bool ownership;
   double pp, ee;

public:
   PUMPLaplacian(Coefficient* nfunc, VectorCoefficient* nfgrad,
                 bool ownership_=true)
   {
      func=nfunc;
      fgrad=nfgrad;
      ownership=ownership_;
      pp=2.0;
      ee=1e-7;
   }

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

class PDEFilter
{
public:
   PDEFilter(mfem::ParMesh& mesh, double rh, int order_=2,
             int maxiter=100, double rtol=1e-7, double atol=1e-15, int print_lv=0)
   {
      int dim=mesh.Dimension();
      lcom=mesh.GetComm();

      rr=rh;

      fecp=new mfem::H1_FECollection(order_,dim);
      fesp=new mfem::ParFiniteElementSpace(&mesh,fecp,1,mfem::Ordering::byVDIM);

      sv = fesp->NewTrueDofVector();
      bv = fesp->NewTrueDofVector();

      gf = new mfem::ParGridFunction(fesp);


      nf=new mfem::ParNonlinearForm(fesp);
      prec=new mfem::HypreBoomerAMG();
      prec->SetPrintLevel(print_lv);

      gmres = new mfem::GMRESSolver(lcom);

      gmres->SetAbsTol(atol);
      gmres->SetRelTol(rtol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(print_lv);
      gmres->SetPreconditioner(*prec);

      K=nullptr;
      sint=nullptr;
   }

   ~PDEFilter()
   {

      delete gmres;
      delete prec;
      delete nf;
      delete gf;
      delete bv;
      delete sv;
      delete fesp;
      delete fecp;
   }

   void Filter(mfem::ParGridFunction& func, mfem::ParGridFunction ffield)
   {
      mfem::GridFunctionCoefficient gfc(&func);
      Filter(gfc,ffield);

   }

   void Filter(mfem::Coefficient& func, mfem::ParGridFunction& ffield)
   {
      if (sint==nullptr)
      {
         sint=new mfem::ScreenedPoisson(func,rr);
         nf->AddDomainIntegrator(sint);
         *sv=0.0;
         K=&(nf->GetGradient(*sv));
         gmres->SetOperator(*K);

      }
      else
      {
         sint->SetInput(func);
      }

      //form RHS
      *sv=0.0;
      nf->Mult(*sv,*bv);
      //filter the input field
      gmres->Mult(*bv,*sv);

      gf->SetFromTrueDofs(*sv);

      mfem::GridFunctionCoefficient gfc(gf);
      ffield.ProjectCoefficient(gfc);
   }

private:
   MPI_Comm lcom;
   mfem::H1_FECollection* fecp;
   mfem::ParFiniteElementSpace* fesp;
   mfem::ParNonlinearForm* nf;
   mfem::HypreBoomerAMG* prec;
   mfem::GMRESSolver *gmres;
   mfem::HypreParVector *sv;
   mfem::HypreParVector *bv;

   mfem::ParGridFunction* gf;

   mfem::Operator* K;
   mfem::ScreenedPoisson* sint;
   double rr;

};

class PLapDistanceSolver : public DistanceSolver
{
public:
   PLapDistanceSolver(int maxp_ = 30, int newton_iter_ = 10,
                      double rtol = 1e-7, double atol = 1e-12, int print_lv = 0)
      : maxp(maxp_), newton_iter(newton_iter_),
        newton_rel_tol(rtol), newton_abs_tol(atol)
   {
      print_level = print_lv;
   }

   void SetMaxPower(int new_pp) { maxp = new_pp; }

   // Ths computed distance is "signed".
   void ComputeScalarDistance(Coefficient& func, ParGridFunction& fdist);

private:
   int maxp; //maximum value of the power p
   double newton_abs_tol;
   double newton_rel_tol;
   int newton_iter;
};

} // namespace mfem

#endif
