// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include <memory>

#include "mfem.hpp"

#if defined(__has_include) && __has_include("general/nvtx.hpp") && !defined(_WIN32)
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kChartreuse
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

using real_t = mfem::real_t;

///////////////////////////////////////////////////////////////////////////////
class IsoElasticyLambdaCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   IsoElasticyLambdaCoeff(mfem::Coefficient *E,
                          mfem::Coefficient *nu):
      E(E), nu(nu) { }

   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Lambda = [](const real_t E, const real_t ν)
      {
         return E * ν / (1.0 + ν) / (1.0 - 2.0 * ν);
      };
      return Lambda(EE, nn);
   }
};

///////////////////////////////////////////////////////////////////////////////
class IsoElasticySchearCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   IsoElasticySchearCoeff(mfem::Coefficient *E_, mfem::Coefficient *nu_):
      E(E_), nu(nu_) { }

   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Schear = [](const real_t E, const real_t ν)
      {
         return E / (2.0 * (1.0 + ν));
      };
      return Schear(EE, nn);
   }
};

///////////////////////////////////////////////////////////////////////////////
class IsoLinElasticSolver : public mfem::Operator
{
public:
   IsoLinElasticSolver(mfem::ParMesh *mesh, int vorder = 1,
                       bool pa = false, bool dfem = false);

   ~IsoLinElasticSolver();

   /// Set the Linear Solver
   void SetLinearSolver(const real_t rtol = 1e-8,
                        const real_t atol = 1e-12,
                        const int miter = 4000);

   /// Solves the forward problem.
   void FSolve();

   /// Forms the tangent matrix
   void AssembleTangent();

   /// Solves the adjoint with the provided rhs.
   void ASolve(mfem::Vector &rhs);

   /// Solves the forward problem with the provided rhs.
   void FSolve(mfem::Vector &rhs);

   /// Adds displacement BC in direction 0(x), 1(y), 2(z), or 4(all).
   void AddDispBC(const int id, const int dir, real_t val);

   /// Adds displacement BC in direction 0(x), 1(y), 2(z), or 4(all).
   void AddDispBC(const int id, const int dir, mfem::Coefficient &val);

   /// Clear all displacement BC
   void DelDispBC();

   /// Set the values of the volumetric force.
   void SetVolForce(real_t fx, real_t fy, real_t fz = 0.0);

   /// Add surface load
   void AddSurfLoad(int id, real_t fx, real_t fy, real_t fz = 0.0)
   {
      mfem::Vector vec;
      vec.SetSize(spaceDim);
      vec[0] = fx;
      vec[1] = fy;
      if (spaceDim == 3) { vec[2] = fz; }
      auto *vc = new mfem::VectorConstantCoefficient(vec);
      if (load_coeff.find(id) != load_coeff.end()) { delete load_coeff[id]; }
      load_coeff[id] = vc;
   }

   /// Add surface load
   void AddSurfLoad(int id, mfem::VectorCoefficient &ff)
   {
      surf_loads[id] = &ff;
   }

   /// Associates coefficient to the volumetric force.
   void SetVolForce(mfem::VectorCoefficient &ff);

   /// Returns the displacements.
   mfem::ParGridFunction &GetDisplacements()
   {
      fdisp.SetFromTrueDofs(sol);
      return fdisp;
   }

   /// Returns the adjoint displacements.
   mfem::ParGridFunction &GetADisplacements()
   {
      adisp.SetFromTrueDofs(adj);
      return adisp;
   }

   /// Returns the solution vector.
   mfem::Vector &GetSol() { return sol; }

   /// Returns the adjoint solution vector.
   mfem::Vector &GetAdj() { return adj; }

   void GetSol(mfem::ParGridFunction &sgf)
   {
      sgf.SetSpace(vfes);
      sgf.SetFromTrueDofs(sol);
   }

   void GetAdj(mfem::ParGridFunction &agf)
   {
      agf.SetSpace(vfes);
      agf.SetFromTrueDofs(adj);
   }

   /// Sets BC dofs, bilinear form, preconditioner and solver.
   /// Should be called before calling Mult of MultTranspose
   virtual void Assemble();

   /// Forward solve with given RHS. x is the RHS vector.
   /// The BC are set to zero.
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Adjoint solve with given RHS. x is the RHS vector.
   /// The BC are set to zero.
   void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Set material
   void SetMaterial(mfem::Coefficient &E_, mfem::Coefficient &nu_)
   {
      dbg();
      E = &E_;
      nu = &nu_;

      delete lambda;
      delete mu;
      delete bf;

      lambda = new IsoElasticyLambdaCoeff(E, nu);
      mu = new IsoElasticySchearCoeff(E, nu);

      dbg("new bf ParBilinearForm");
      bf = new mfem::ParBilinearForm(vfes);

      if (dfem) { AddDFemDomainIntegrator(); }
      else
      {
         bf->AddDomainIntegrator(new mfem::ElasticityIntegrator(*lambda, *mu));
         if (pa) { bf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL); }
      }
   }

   class NonTensorUniformParameterSpace : public
      mfem::future::UniformParameterSpace
   {
   public:
      NonTensorUniformParameterSpace(mfem::ParMesh &mesh,
                                     const mfem::IntegrationRule &ir,
                                     int vdim) :
         mfem::future::UniformParameterSpace(mesh, ir, vdim, false)
      {
         dtq.nqpt = ir.GetNPoints();
      }
   };

private:
   mfem::ParMesh *pmesh;
   const bool pa, dfem; // partial assembly, dFEM operator
   const int vorder, dim, spaceDim;

   // finite element collection for linear elasticity
   mfem::FiniteElementCollection *vfec;

   // finite element space for linear elasticity
   mfem::ParFiniteElementSpace *vfes;

   // solution true vector
   mutable mfem::Vector sol;
   // adjoint true vector
   mutable mfem::Vector adj;
   // RHS
   mutable mfem::Vector rhs;

   // forward solution
   mfem::ParGridFunction fdisp;
   // adjoint solution
   mfem::ParGridFunction adisp;

   // Linear solver parameters
   real_t linear_rtol;
   real_t linear_atol;
   int linear_iter;

   mfem::HypreBoomerAMG *prec; // preconditioner
   mfem::CGSolver *ls;         // linear solver

   // PA LOR preconditioner
   mfem::Array<int> lor_block_offsets;
   std::unique_ptr<mfem::Solver> lor_pa_prec;
   std::unique_ptr<mfem::ParLORDiscretization> lor_disc;
   std::unique_ptr<mfem::ElasticityIntegrator> lor_integrator;
   std::unique_ptr<mfem::ParFiniteElementSpace> lor_scalar_fespace;
   std::unique_ptr<mfem::BlockDiagonalPreconditioner> lor_blockDiag;
   std::vector<std::unique_ptr<mfem::ParBilinearForm>> lor_bilinear_forms;
   std::vector<std::unique_ptr<mfem::HypreParMatrix>> lor_block;
   std::vector<std::unique_ptr<mfem::HypreBoomerAMG>> lor_amg_blocks;

   /// Volumetric force created by the solver.
   mfem::VectorConstantCoefficient *lvforce;
   /// Volumetric force coefficient can point to the one
   /// created by the solver or to external vector coefficient.
   mfem::VectorCoefficient *volforce;

   // surface loads
   using VectorCoefficientPtrMap = std::map<int, mfem::VectorCoefficient *>;
   VectorCoefficientPtrMap load_coeff; // internaly generated load
   VectorCoefficientPtrMap surf_loads; // external vector coeeficients

   class SurfaceLoad;
   std::unique_ptr<SurfaceLoad> lcsurf_load; // localy generated surface loads
   std::unique_ptr<SurfaceLoad> glsurf_load; // global surface loads

   // boundary conditions for x,y, and z directions
   using ConstantCoefficientMap = std::map<int, mfem::ConstantCoefficient>;
   ConstantCoefficientMap bcx, bcy, bcz;

   // holds BC in coefficient form
   using CoefficientPtrMap = std::map<int, mfem::Coefficient*>;
   CoefficientPtrMap bccx, bccy, bccz;

   // holds the displacement contrained DOFs
   mfem::Array<int> ess_tdofv;

   // creates a list with essetial dofs
   // sets the values in the bsol vector
   // the list is written in ess_dofs
   void SetEssTDofs(mfem::Vector &bsol, mfem::Array<int> &ess_dofs);
   void SetEssTDofs(const int j, mfem::ParFiniteElementSpace& scalar_space,
                    mfem::Array<int> &ess_dofs);

   mfem::Coefficient *E;
   mfem::Coefficient *nu;

   mfem::Coefficient *lambda;
   mfem::Coefficient *mu;
   mfem::Coefficient *rho; // density

   mfem::ParBilinearForm *bf;
   mfem::ConstrainedOperator *Kc;
   std::unique_ptr<mfem::OperatorHandle> Kh;
   std::unique_ptr<mfem::HypreParMatrix> K, Ke;

   static constexpr int U = 0, Coords = 1, ECoeff = 2, NuCoeff = 3;
   const mfem::FiniteElement *fe;
   mfem::ParGridFunction *nodes;
   mfem::ParFiniteElementSpace *mfes;
   mfem::Array<int> domain_attributes;
   const mfem::IntegrationRule &ir;
   mfem::QuadratureSpace qs;
   NonTensorUniformParameterSpace E_ps, nu_ps;
   std::unique_ptr<mfem::CoefficientVector> E_cv, nu_cv;
   mfem::future::DifferentiableOperator dop;
   void AddDFemDomainIntegrator();

   mfem::ParLinearForm *lf;

   class SurfaceLoad: public mfem::VectorCoefficient
   {
      VectorCoefficientPtrMap *map;
   public:
      SurfaceLoad(int dim, VectorCoefficientPtrMap &cmap):
         mfem::VectorCoefficient(dim)
      {
         map = &cmap;
      }

      void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                const mfem::IntegrationPoint &ip) override
      {
         V.SetSize(GetVDim());
         V = 0.0;
         auto it = map->find(T.Attribute);
         if (it != map->end()) { it->second->Eval(V, T, ip); }
      }
   };
};

namespace PointwiseTrans
{

/*  Standrd "Heaviside" projection in topology optimization with threshold eta
 * and steepness of the projection beta.
 * */
inline real_t HProject(real_t rho, real_t eta, real_t beta)
{
   // tanh projection - Wang&Lazarov&Sigmund2011
   real_t a = std::tanh(eta * beta);
   real_t b = std::tanh(beta * (1.0 - eta));
   real_t c = std::tanh(beta * (rho - eta));
   real_t rez = (a + c) / (a + b);
   return rez;
}

/// Gradient of the "Heaviside" projection with respect to rho.
inline real_t HGrad(real_t rho, real_t eta, real_t beta)
{
   real_t c = std::tanh(beta * (rho - eta));
   real_t a = std::tanh(eta * beta);
   real_t b = std::tanh(beta * (1.0 - eta));
   real_t rez = beta * (1.0 - c * c) / (a + b);
   return rez;
}

/// Second derivative of the "Heaviside" projection with respect to rho.
inline real_t HHess(real_t rho, real_t eta, real_t beta)
{
   real_t c = std::tanh(beta * (rho - eta));
   real_t a = std::tanh(eta * beta);
   real_t b = std::tanh(beta * (1.0 - eta));
   real_t rez = -2.0 * beta * beta * c * (1.0 - c * c) / (a + b);
   return rez;
}

inline real_t FluidInterpolation(real_t rho, real_t q)
{
   return q * (1.0 - rho) / (q + rho);
}

inline real_t GradFluidInterpolation(real_t rho, real_t q)
{
   real_t tt = q + rho;
   return -q / tt - q * (1.0 - rho) / (tt * tt);
}

inline real_t SIMPInterpolation(real_t rho, real_t p)
{
   return std::pow(rho, p);
}

inline real_t GradSIMPInterpolation(real_t rho, real_t p)
{
   return p * std::pow(rho, p - 1.0);
}

} // namespace PointwiseTrans

class FilterOperator : public mfem::Operator
{
public:
   FilterOperator(real_t r_, mfem::ParMesh *pmesh_, int order_ = 2)
   {
      r = r_;
      order = order_;
      pmesh = pmesh_;
      int dim = pmesh->Dimension();
      sfec = new mfem::H1_FECollection(order, dim);
      sfes = new mfem::ParFiniteElementSpace(pmesh, sfec, 1);

      //ifec = new mfem::H1Pos_FECollection(order - 1, dim);
      //ifec=new mfem::H1_FECollection(order-1,dim);
      ifec=new mfem::L2_FECollection(order-1,dim);
      ifes = new mfem::ParFiniteElementSpace(pmesh, ifec, 1);

      dfes = ifes;
      SetSolver();

      K = nullptr;
      S = nullptr;
      A = nullptr;
      pcg = nullptr;
      prec = nullptr;

      mfem::Operator::width = dfes->GetTrueVSize();
      mfem::Operator::height = sfes->GetTrueVSize();
   }

   FilterOperator(real_t r_, mfem::ParMesh *pmesh_,
                  mfem::ParFiniteElementSpace *dfes_, int order_ = 2):
      dfes(dfes_)
   {
      r = r_;
      order = order_;
      pmesh = pmesh_;
      int dim = pmesh->Dimension();
      sfec = new mfem::H1_FECollection(order, dim);
      sfes = new mfem::ParFiniteElementSpace(pmesh, sfec, 1);

      ifec = nullptr;
      ifes = nullptr;

      SetSolver();

      K = nullptr;
      S = nullptr;
      A = nullptr;
      pcg = nullptr;
      prec = nullptr;

      mfem::Operator::width = dfes->GetTrueVSize();
      mfem::Operator::height = sfes->GetTrueVSize();
   }

   mfem::ParFiniteElementSpace *GetFilterFES() { return sfes; }
   mfem::ParFiniteElementSpace *GetDesignFES() { return dfes; }

   virtual ~FilterOperator()
   {
      delete pcg;
      delete prec;
      delete K;
      delete S;
      delete A;
      delete sfes;
      delete sfec;
      delete ifes;
      delete ifec;
   }

   void Update()
   {
      sfes->Update();
      dfes->Update();
      Assemble();
   }

   void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      // y=bdrc;
      tmpv.SetSize(y.Size());

      pcg->SetAbsTol(atol);
      pcg->SetRelTol(rtol);
      pcg->SetMaxIter(max_iter);
      pcg->SetPrintLevel(prt_level);
      S->Mult(x, tmpv);
      K->EliminateBC(*A, ess_tdofv, bdrc, tmpv);
      pcg->Mult(tmpv, y);
   }

   void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override
   {
      y = 0.0;
      rhsv.SetSize(x.Size());
      rhsv = x;
      tmpv.SetSize(x.Size());
      tmpv = 0.0;
      pcg->SetAbsTol(atol);
      pcg->SetRelTol(rtol);
      pcg->SetMaxIter(max_iter);
      pcg->SetPrintLevel(prt_level);
      K->EliminateBC(*A, ess_tdofv, tmpv, rhsv);
      pcg->Mult(rhsv, tmpv);
      S->MultTranspose(tmpv, y);
   }

   void SetSolver(real_t rtol_ = 1e-8, real_t atol_ = 1e-12, int miter_ = 1000,
                  int prt_level_ = 1)
   {
      rtol = rtol_;
      atol = atol_;
      max_iter = miter_;
      prt_level = prt_level_;
   }

   void AddBC(int id, real_t val)
   {
      bcr[id] = mfem::ConstantCoefficient(val);

      delete pcg;
      delete prec;
      delete K;
      delete S;
      delete A;

      pcg = nullptr;
      prec = nullptr;
      K = nullptr;
      S = nullptr;
      A = nullptr;
   }

   void Assemble()
   {
      delete pcg;
      delete prec;
      delete K;
      delete S;
      delete A;

      ess_tdofv.DeleteAll();
      bdrc.SetSize(sfes->GetTrueVSize());
      bdrc = 0.0;
      // set boundary conditions
      if (bcr.size() != 0)
      {
         mfem::ParGridFunction tmpgf(sfes);
         tmpgf = 0.0;
         for (auto it = bcr.begin(); it != bcr.end(); it++)
         {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr = 0;
            ess_bdr[it->first - 1] = 1;
            mfem::Array<int> ess_tdof_list;
            sfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
            ess_tdofv.Append(ess_tdof_list);
            tmpgf.ProjectBdrCoefficient(it->second, ess_bdr);
         }
         tmpgf.GetTrueDofs(bdrc);
      }

      real_t dr = r / (2.0 * sqrt(3.0));
      mfem::ConstantCoefficient dc(dr * dr);

      auto *bf = new mfem::ParBilinearForm(sfes);
      bf->AddDomainIntegrator(new mfem::MassIntegrator());
      bf->AddDomainIntegrator(new mfem::DiffusionIntegrator(dc));
      bf->Assemble();
      bf->Finalize();
      K = bf->ParallelAssemble();
      delete bf;

      A = K->EliminateRowsCols(ess_tdofv);
      K->EliminateZeroRows();

      // allocate the CG solver and the preconditioner
      prec = new mfem::HypreBoomerAMG(*K);
      pcg = new mfem::CGSolver(pmesh->GetComm());
      pcg->SetOperator(*K);
      pcg->SetPreconditioner(*prec);

      auto *mf = new mfem::ParMixedBilinearForm(dfes, sfes);
      mf->AddDomainIntegrator(new mfem::MassIntegrator());
      mf->Assemble();
      mf->Finalize();
      S = mf->ParallelAssemble();
      delete mf;
   }

   // forward filter
   void FFilter(mfem::Coefficient *coeff, mfem::ParGridFunction &gf)
   {
      gf.SetSpace(GetFilterFES());
      tmpv.SetSize(GetFilterFES()->TrueVSize());
      tmpv = 0.0;
      rhsv.SetSize(GetFilterFES()->TrueVSize());

      mfem::ParLinearForm lf(GetFilterFES());
      lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(*coeff));
      lf.Assemble();
      lf.ParallelAssemble(rhsv);

      pcg->SetAbsTol(atol);
      pcg->SetRelTol(rtol);
      pcg->SetMaxIter(max_iter);
      pcg->SetPrintLevel(prt_level);

      K->EliminateBC(*A, ess_tdofv, bdrc, rhsv);
      pcg->Mult(rhsv, tmpv);
      gf.SetFromTrueDofs(tmpv);
   }

   void AFilter(mfem::Coefficient *coeff, mfem::ParGridFunction &gf)
   {
      gf.SetSpace(GetDesignFES());

      tmpv.SetSize(GetFilterFES()->TrueVSize());
      tmpv = 0.0;
      rhsv.SetSize(GetFilterFES()->TrueVSize());

      mfem::ParLinearForm lf(GetFilterFES());
      lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(*coeff));
      lf.Assemble();
      lf.ParallelAssemble(rhsv);

      pcg->SetAbsTol(atol);
      pcg->SetRelTol(rtol);
      pcg->SetMaxIter(max_iter);
      pcg->SetPrintLevel(prt_level);

      K->EliminateBC(*A, ess_tdofv, tmpv, rhsv);
      pcg->Mult(rhsv, tmpv);

      rhsv.SetSize(GetDesignFES()->TrueVSize());
      S->MultTranspose(tmpv,rhsv);

      gf.SetFromTrueDofs(rhsv);
   }

private:
   mutable mfem::HypreParMatrix *S;
   mutable mfem::HypreParMatrix *K;
   mutable mfem::HypreParMatrix *A;
   mutable mfem::Solver *prec;
   mutable mfem::CGSolver *pcg;

   mfem::FiniteElementCollection *sfec;
   mfem::ParFiniteElementSpace *sfes;
   mfem::FiniteElementCollection *ifec;
   mfem::ParFiniteElementSpace *ifes;

   mfem::ParGridFunction sol;

   mutable mfem::Vector tmpv;
   mutable mfem::Vector bdrc;          // boundary conditions
   mutable mfem::Vector rhsv;          // RHS for the adjoint
   mutable mfem::Array<int> ess_tdofv; // boundary dofs

   real_t r;
   int order;

   mfem::ParMesh *pmesh;

   mfem::ParFiniteElementSpace *dfes;

   std::map<int, mfem::ConstantCoefficient> bcr;

   real_t atol;
   real_t rtol;
   int max_iter;
   int prt_level;
};

class IsoComplCoef : public mfem::Coefficient
{
public:
   IsoComplCoef(bool SIMP_=false,bool PROJ_=false)
   {

      eta=0.5;
      beta=8.0;
      p=1.0;

      SIMP=SIMP_;
      PROJ=PROJ_;

      rho=nullptr;
      sol=nullptr;

      dMu.reset(new DerivedCoef(this,&IsoComplCoef::EvalMu));
      dLambda.reset(new DerivedCoef(this,&IsoComplCoef::EvalLambda));
      dE.reset(new DerivedCoef(this,&IsoComplCoef::EvalE));
      dIsoCompl.reset(new DerivedCoef(this,&IsoComplCoef::EvalGrad));
      dnu.reset(new DerivedCoef(this,&IsoComplCoef::EvalNu));
   }

   IsoComplCoef(mfem::GridFunction *rho_, mfem::GridFunction *sol_,
                bool SIMP_ = false, bool PROJ_ = false)
   {
      eta = 0.5;
      beta = 8.0;
      p = 1.0;

      SIMP = SIMP_;
      PROJ = PROJ_;

      rho = rho_;
      sol = sol_;

      dMu.reset(new DerivedCoef(this,&IsoComplCoef::EvalMu));
      dLambda.reset(new DerivedCoef(this,&IsoComplCoef::EvalLambda));
      dE.reset(new DerivedCoef(this,&IsoComplCoef::EvalE));
      dIsoCompl.reset(new DerivedCoef(this,&IsoComplCoef::EvalGrad));
   }

   virtual ~IsoComplCoef() {}

   void SetGridFunctions(mfem::GridFunction *rho_, mfem::GridFunction *sol_)
   {
      rho = rho_;
      sol = sol_;
   }

   void SetDensity(mfem::GridFunction* rho_)
   {
      rho=rho_;
   }

   void SetDispl(mfem::GridFunction* sol_)
   {
      sol=sol_;
   }

   void SetMaterial(real_t Emin_, real_t Emax_, real_t nu_)
   {
      cEmin.constant = Emin_;
      cEmax.constant = Emax_;
      cnu.constant = nu_;
      SetMaterial(&cEmin, &cEmax, &cnu);
   }

   void SetMaterial(Coefficient *Emin_, Coefficient *Emax_, Coefficient *nu_)
   {
      Emin = Emin_;
      Emax = Emax_;
      nu = nu_;

      llmax = std::make_unique<IsoElasticyLambdaCoeff>(Emax, nu);
      llmin = std::make_unique<IsoElasticyLambdaCoeff>(Emin, nu);
      mmmax = std::make_unique<IsoElasticySchearCoeff>(Emax, nu);
      mmmin = std::make_unique<IsoElasticySchearCoeff>(Emin, nu);
   }

   void SetProj(real_t eta_, real_t beta_)
   {
      PROJ = true;
      eta = eta_;
      beta = beta_;
   }

   void SetSIMP(real_t p_)
   {
      SIMP = true;
      p = p_;
   }

   Coefficient *GetE() { return dE.get(); }
   Coefficient *GetLambda() { return dLambda.get(); }
   Coefficient *GetMu() { return dMu.get(); }
   Coefficient *GetGradIsoComp() { return dIsoCompl.get(); }
   Coefficient* GetNu() {return dnu.get();}

   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      real_t Lmax = llmax->Eval(T, ip);
      real_t Mmax = mmmax->Eval(T, ip);

      real_t Lmin = llmin->Eval(T, ip);
      real_t Mmin = mmmin->Eval(T, ip);

      sol->GetVectorGradient(T, grad);
      real_t div_u = grad.Trace();
      real_t density_max = Lmax * div_u * div_u;
      real_t density_min = Lmin * div_u * div_u;

      int dim = T.GetSpaceDim();
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            density_max += Mmax * grad(i, j) * (grad(i, j) + grad(j, i));
            density_min += Mmin * grad(i, j) * (grad(i, j) + grad(j, i));
         }
      }
      real_t val = rho->GetValue(T, ip);
      if (PROJ) { val = PointwiseTrans::HProject(val, eta, beta); }

      if (SIMP) { val = PointwiseTrans::SIMPInterpolation(val, p); }

      return val * density_max + density_min;
   }

private:

   real_t EvalNu(mfem::ElementTransformation &T,
                 const mfem::IntegrationPoint &ip)
   {
      return nu->Eval(T,ip);
   }
   real_t EvalGrad(mfem::ElementTransformation &T,
                   const mfem::IntegrationPoint &ip)
   {
      real_t Lmax = llmax->Eval(T, ip);
      real_t Mmax = mmmax->Eval(T, ip);

      sol->GetVectorGradient(T, grad);
      real_t div_u = grad.Trace();
      real_t density_max = Lmax * div_u * div_u;

      int dim = T.GetSpaceDim();
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            density_max += Mmax * grad(i, j) * (grad(i, j) + grad(j, i));
         }
      }
      real_t val = rho->GetValue(T, ip);
      real_t hvl = val;
      real_t gvl = 1.0;

      if (PROJ) { hvl = PointwiseTrans::HProject(val, eta, beta); }

      if (SIMP) { gvl = gvl * PointwiseTrans::GradSIMPInterpolation(hvl, p); }

      if (PROJ) { gvl = gvl * PointwiseTrans::HGrad(val, eta, beta); }

      return -gvl * density_max;
   }

   real_t EvalLambda(mfem::ElementTransformation &T,
                     const mfem::IntegrationPoint &ip)
   {
      real_t Lmax = llmax->Eval(T, ip);
      real_t Lmin = llmin->Eval(T, ip);

      real_t val = rho->GetValue(T, ip);
      if (PROJ) { val = PointwiseTrans::HProject(val, eta, beta); }

      if (SIMP) { val = PointwiseTrans::SIMPInterpolation(val, p); }

      return Lmax * val + Lmin;
   }

   real_t EvalMu(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
   {
      real_t Mmax = mmmax->Eval(T, ip);
      real_t Mmin = mmmin->Eval(T, ip);

      real_t val = rho->GetValue(T, ip);
      if (PROJ) { val = PointwiseTrans::HProject(val, eta, beta); }

      if (SIMP) { val = PointwiseTrans::SIMPInterpolation(val, p); }

      return Mmax * val + Mmin;
   }

   real_t EvalE(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
   {
      real_t vEmax = Emax->Eval(T, ip);
      real_t vEmin = Emin->Eval(T, ip);
      real_t val = rho->GetValue(T, ip);
      if (PROJ) { val = PointwiseTrans::HProject(val, eta, beta); }

      if (SIMP) { val = PointwiseTrans::SIMPInterpolation(val, p); }

      return vEmax * val + vEmin;
   }

   real_t eta;
   real_t beta;
   real_t p;

   bool SIMP = 0;
   bool PROJ = 0;

   Coefficient *Emax;
   Coefficient *Emin;
   Coefficient *nu;

   mfem::ConstantCoefficient cEmax;
   mfem::ConstantCoefficient cEmin;
   mfem::ConstantCoefficient cnu;

   std::unique_ptr<IsoElasticyLambdaCoeff> llmax;
   std::unique_ptr<IsoElasticySchearCoeff> mmmax;

   std::unique_ptr<IsoElasticyLambdaCoeff> llmin;
   std::unique_ptr<IsoElasticySchearCoeff> mmmin;

   mfem::GridFunction *sol;
   mfem::GridFunction *rho;

   mfem::DenseMatrix grad;

   [[maybe_unused]]
   real_t (IsoComplCoef::*ptr2grad)(mfem::ElementTransformation &T,
                                    const mfem::IntegrationPoint &ip) =
                                       &IsoComplCoef::EvalGrad;

   [[maybe_unused]]
   real_t (IsoComplCoef::*ptr2Mu)(mfem::ElementTransformation &T,
                                  const mfem::IntegrationPoint &ip) =
                                     &IsoComplCoef::EvalMu;

   [[maybe_unused]]
   real_t (IsoComplCoef::*ptr2Lambda)(mfem::ElementTransformation &T,
                                      const mfem::IntegrationPoint &ip) =
                                         &IsoComplCoef::EvalLambda;

   [[maybe_unused]]
   real_t (IsoComplCoef::*ptr2E)(mfem::ElementTransformation &T,
                                 const mfem::IntegrationPoint &ip) =
                                    &IsoComplCoef::EvalE;

   class DerivedCoef : public Coefficient
   {
   public:
      DerivedCoef(IsoComplCoef *obj_,
                  real_t (IsoComplCoef::*methodPtr)(mfem::ElementTransformation &T,
                                                    const mfem::IntegrationPoint &ip))
      {
         obj = obj_;
         ptr2eval = methodPtr;
      }

      real_t Eval(mfem::ElementTransformation &T,
                  const mfem::IntegrationPoint &ip) override
      {
         return (obj->*ptr2eval)(T, ip);
      }

   private:
      IsoComplCoef *obj;
      real_t (IsoComplCoef::*ptr2eval)(mfem::ElementTransformation &T,
                                       const mfem::IntegrationPoint &ip);
   };

   std::unique_ptr<DerivedCoef> dMu;
   std::unique_ptr<DerivedCoef> dE;
   std::unique_ptr<DerivedCoef> dLambda;
   std::unique_ptr<DerivedCoef> dIsoCompl;
   std::unique_ptr<DerivedCoef> dnu;
};
