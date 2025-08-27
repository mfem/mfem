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

#ifndef MFEM_DARCYOP
#define MFEM_DARCYOP

#include "mfem.hpp"
#include "../general/socketstream.hpp"

namespace mfem
{

class DarcyOperator : public TimeDependentOperator
{
public:
   enum class SolverType
   {
      Default = 0,
      LBFGS,
      LBB,
      Newton,
      KINSol,
   };

private:
   Array<int> offsets;
   const Array<int> &ess_flux_tdofs_list;
   DarcyForm *darcy;
   LinearForm *g, *f, *h;
#ifdef MFEM_USE_MPI
   ParDarcyForm *pdarcy {};
   ParLinearForm *pg{}, *pf{}, *ph{};
#endif
   const Array<Coefficient*> &coeffs;
   SolverType solver_type;
   bool btime_u, btime_p;
   real_t rtol{1e-6}, atol{1e-10};
   int max_iters{1000};

   FiniteElementSpace *trace_space{};

   real_t idt{};
   std::unique_ptr<Coefficient> idtcoeff;
   std::unique_ptr<BilinearForm> Mt0, Mq0;

   std::string lsolver_str;
   std::unique_ptr<Solver> prec, lin_prec;
   std::string prec_str, lin_prec_str;
   std::unique_ptr<IterativeSolver> solver;
   std::string solver_str;
   std::unique_ptr<IterativeSolverMonitor> monitor;
   int monitor_step{-1};

   mutable BlockVector x, rhs;

   class SchurPreconditioner : public Solver
   {
      const DarcyForm *darcy;
#ifdef MFEM_USE_MPI
      const ParDarcyForm *pdarcy {};
#endif
      const Operator *op {};
      bool nonlinear;

      const char *prec_str;
      mutable std::unique_ptr<BlockDiagonalPreconditioner> darcyPrec;
      mutable std::unique_ptr<SparseMatrix> S;
#ifdef MFEM_USE_MPI
      mutable std::unique_ptr<HypreParMatrix> hS;
#endif
      mutable bool reconstruct {};

      void Construct(const Vector &x) const;
#ifdef MFEM_USE_MPI
      void ConstructPar(const Vector &x) const;
#endif

   public:
      SchurPreconditioner(const DarcyForm *darcy, bool nonlinear = false);
#ifdef MFEM_USE_MPI
      SchurPreconditioner(const ParDarcyForm *darcy, bool nonlinear = false);
#endif

      const char *GetString() const { return prec_str; }

      void SetOperator(const Operator &op_) override
      { op = &op_; reconstruct = true; }

      void Mult(const Vector &x, Vector &y) const override;
   };

public:
   class SolutionController : public IterativeSolverController
   {
   public:
      enum class Type
      {
         Native,
         Flux,
         Potential
      };

   protected:
      DarcyForm &darcy;
      BlockVector &x;
      const BlockVector &rhs;
      Type type;
      real_t rtol;
      int it_prev{};
      Vector sol_prev;

      bool CheckSolution(const Vector &x, const Vector &y) const;
      virtual void ReduceValues(real_t diff[], int num) const { }

   public:
      SolutionController(DarcyForm &darcy, BlockVector &x, const BlockVector &rhs,
                         Type type, real_t rtol);

      void MonitorSolution(int it, real_t norm, const Vector &x,
                           bool final) override;

      bool RequiresUpdatedSolution() const override { return true; }
   };

#ifdef MFEM_USE_MPI
   class ParSolutionController : public SolutionController
   {
   protected:
      ParDarcyForm &pdarcy;

      void ReduceValues(real_t diff[], int num) const override;

   public:
      ParSolutionController(ParDarcyForm &pdarcy, BlockVector &x,
                            const BlockVector &rhs, Type type, real_t rtol);

      void MonitorSolution(int it, real_t norm, const Vector &x,
                           bool final) override;
   };
#endif //MFEM_USE_MPI

private:
   SolutionController::Type sol_type{SolutionController::Type::Native};

   class IterativeGLVis : public IterativeSolverMonitor
   {
   protected:
      DarcyForm &darcy;
      BlockVector &x;
      const BlockVector &rhs;
      int step;
      bool save_files;

      socketstream q_sock, t_sock;

      virtual void StreamPreamble(socketstream &ss) { }
      virtual std::string FormFilename(const char *base, int it,
                                       const char *suff = "gf");
   public:
      IterativeGLVis(DarcyForm &darcy, BlockVector &x, const BlockVector &rhs,
                     int step = 0, bool save_files = false);

      void MonitorSolution(int it, real_t norm, const Vector &x,
                           bool final) override;

      bool RequiresUpdatedSolution() const override { return true; }
   };

#ifdef MFEM_USE_MPI
   class ParIterativeGLVis : public IterativeGLVis
   {
      ParDarcyForm &pdarcy;

      void StreamPreamble(socketstream &ss) override;
      std::string FormFilename(const char *base, int it,
                               const char *suff = "gf") override;
   public:
      ParIterativeGLVis(ParDarcyForm &pdarcy_, BlockVector &x, const BlockVector &rhs,
                        int step = 0, bool save_files = false)
         : IterativeGLVis(pdarcy_, x, rhs, step), pdarcy(pdarcy_) { }
   };
#endif //MFEM_USE_MPI

   void SetupNonlinearSolver(real_t rtol, real_t atol, int iters);
   void SetupLinearSolver(real_t rtol, real_t atol, int iters);

public:
   DarcyOperator(const Array<int> &ess_flux_tdofs_list, DarcyForm *darcy,
                 LinearForm *g, LinearForm *f, LinearForm *h, const Array<Coefficient*> &coeffs,
                 SolverType stype = SolverType::LBFGS,  bool bflux_u = true,
                 bool btime_p = true);
#ifdef MFEM_USE_MPI
   DarcyOperator(const Array<int> &ess_flux_tdofs_list, ParDarcyForm *darcy,
                 ParLinearForm *g, ParLinearForm *f, ParLinearForm *h,
                 const Array<Coefficient*> &coeffs,
                 SolverType stype = SolverType::LBFGS,  bool bflux_u = true,
                 bool btime_p = true);
#endif

   ~DarcyOperator();

   void SetTolerance(real_t rtol_, real_t atol_ = 0.) { rtol = rtol_; atol = atol_; }
   void SetMaxIters(int iters_) { max_iters = iters_; }

   void EnableSolutionController(SolutionController::Type type) { sol_type = type; }
   void EnableIterationsVisualization(int vis_step = 0) { monitor_step = vis_step; }

   static Array<int> ConstructOffsets(const DarcyForm &darcy);
   inline const Array<int>& GetOffsets() const { return offsets; }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;
};

void RandomizeMesh(Mesh &mesh, real_t dr);

class VectorBlockDiagonalIntegrator : public BilinearFormIntegrator
{
   int numIntegs;
   std::vector<BilinearFormIntegrator*> integs;
   bool own_integs{true};

   std::vector<DenseMatrix> elmats;

   template<typename FType, typename... Args> void AssembleElementMat(
      FType f, DenseMatrix &elmat, Args&&... args);

   template<typename FType, typename... Args> void AssembleFaceMat(
      FType f, const FiniteElement &trial_fe1, const FiniteElement &test_fe1,
      const FiniteElement &trial_fe2, const FiniteElement &test_fe2,
      DenseMatrix &elmat, Args&&... args);

public:
   VectorBlockDiagonalIntegrator(int n) : numIntegs(n) { integs.resize(n); }
   VectorBlockDiagonalIntegrator(int n, BilinearFormIntegrator *integ_)
      : numIntegs(n) { integs.push_back(integ_); }

   ~VectorBlockDiagonalIntegrator()
   {
      if (own_integs)
      {
         for (BilinearFormIntegrator *bfi : integs) { delete bfi; }
      }
   }

   void SetIntegrator(int i, BilinearFormIntegrator *integ_) { integs[i] = integ_; }
   BilinearFormIntegrator *GetIntegrator(int i) const { return integs[i]; }

   inline int GetNumIntegrators() const { return numIntegs; }

   void UseExternalIntegrators() { own_integs = false; }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override
   {
      AssembleElementMat<>(&BilinearFormIntegrator::AssembleElementMatrix,
                           elmat, el, Trans);
   }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override
   {
      AssembleElementMat<>(&BilinearFormIntegrator::AssembleElementMatrix2,
                           elmat, trial_fe, test_fe, Trans);
   }

   void AssembleFaceMatrix(const FiniteElement &fe1,
                           const FiniteElement &fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override
   {
      using face_fx = void (BilinearFormIntegrator::*)(
                         const FiniteElement &, const FiniteElement &,
                         FaceElementTransformations &, DenseMatrix &);
      if (Trans.Elem2No >= 0)
         AssembleFaceMat<>(static_cast<face_fx>
                           (&BilinearFormIntegrator::AssembleFaceMatrix),
                           fe1, fe1, fe2, fe2,
                           elmat, fe1, fe2, Trans);
      else
         AssembleElementMat<>(static_cast<face_fx>
                              (&BilinearFormIntegrator::AssembleFaceMatrix),
                              elmat, fe1, fe2, Trans);

   }

   void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                           const FiniteElement &test_fe1,
                           const FiniteElement &trial_fe2,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override
   {
      using face_fx = void (BilinearFormIntegrator::*)(
                         const FiniteElement &, const FiniteElement &,
                         const FiniteElement &, const FiniteElement &,
                         FaceElementTransformations &, DenseMatrix &);
      if (Trans.Elem2No >= 0)
         AssembleFaceMat<>(static_cast<face_fx>
                           (&BilinearFormIntegrator::AssembleFaceMatrix),
                           trial_fe1, test_fe1, trial_fe2, test_fe2,
                           elmat, trial_fe1, test_fe1,
                           trial_fe2, test_fe2, Trans);
      else
         AssembleElementMat<>(static_cast<face_fx>
                              (&BilinearFormIntegrator::AssembleFaceMatrix),
                              elmat, trial_fe1, test_fe1,
                              trial_fe2, test_fe2, Trans);
   }
};

template<typename FType, typename... Args>
void VectorBlockDiagonalIntegrator::AssembleElementMat(
   FType f, DenseMatrix &elmat, Args&&... args)
{
   if (numIntegs > (int)integs.size())
   {
      elmats.resize(1);
      (integs[0]->*f)(args..., elmats[0]);
      const int w = elmats[0].Width();
      const int h = elmats[0].Height();

      elmat.SetSize(numIntegs * h, numIntegs * w);
      elmat = 0.0;

      for (int i = 0; i < numIntegs; i++)
      {
         elmat.CopyMN(elmats[0], i * h, i * w);
      }
   }
   else
   {
      elmats.resize(numIntegs);
      int w = 0, h = 0;
      for (int i = 0; i < numIntegs; i++)
      {
         if (!integs[i])
         {
            elmats[i].SetSize(0);
            continue;
         }
         (integs[i]->*f)(args..., elmats[i]);
         w += elmats[i].Width();
         h += elmats[i].Height();
      }

      elmat.SetSize(h, w);
      elmat = 0.0;

      int off_i = 0, off_j = 0;
      for (int i = 0; i < numIntegs; i++)
      {
         elmat.CopyMN(elmats[i], off_i, off_j);
         off_j += elmats[i].Width();
         off_i += elmats[i].Height();
      }
   }
}

template<typename FType, typename... Args>
void VectorBlockDiagonalIntegrator::AssembleFaceMat(
   FType f, const FiniteElement &trial_fe1, const FiniteElement &test_fe1,
   const FiniteElement &trial_fe2, const FiniteElement &test_fe2,
   DenseMatrix &elmat, Args&&... args)
{
   const int tr_ndof1 = trial_fe1.GetDof();
   const int te_ndof1 = test_fe1.GetDof();
   const int tr_ndof2 = trial_fe2.GetDof();
   const int te_ndof2 = test_fe2.GetDof();

   if (numIntegs > (int)integs.size())
   {
      elmats.resize(1);
      (integs[0]->*f)(args..., elmats[0]);
      const int w = elmats[0].Width();
      const int h = elmats[0].Height();
      const int tr_vdim = w / (tr_ndof1 + tr_ndof2);
      const int te_vdim = h / (te_ndof1 + te_ndof2);
      const int w1 = tr_ndof1 * tr_vdim;
      const int w2 = tr_ndof2 * tr_vdim;
      const int h1 = te_ndof1 * te_vdim;
      const int h2 = te_ndof2 * te_vdim;

      elmat.SetSize(numIntegs * h, numIntegs * w);
      elmat = 0.0;

      for (int i = 0; i < numIntegs; i++)
      {
         elmat.CopyMN(elmats[0],
                      h1, w1, 0,   0,
                      i * h1, i * w1);
         elmat.CopyMN(elmats[0],
                      h2, w1, h1,  0,
                      numIntegs * h1 + i * h1, i * w1);
         elmat.CopyMN(elmats[0],
                      h1, w2, 0,  w1,
                      i * h1, numIntegs * w1 + i * w2);
         elmat.CopyMN(elmats[0],
                      h2, w2, h1, w1,
                      numIntegs * h1 + i * h2, numIntegs * w1 + i * w2);
      }
   }
   else
   {
      elmats.resize(numIntegs);
      int w1 = 0, w2 = 0, h1 = 0, h2 = 0;
      for (int i = 0; i < numIntegs; i++)
      {
         if (!integs[i])
         {
            elmats[i].SetSize(0);
            continue;
         }
         (integs[i]->*f)(args..., elmats[i]);
         const int w = elmats[i].Width();
         const int h = elmats[i].Height();
         const int tr_vdim = w / (tr_ndof1 + tr_ndof2);
         const int te_vdim = h / (te_ndof1 + te_ndof2);
         w1 += tr_ndof1 * tr_vdim;
         w2 += tr_ndof2 * tr_vdim;
         h1 += te_ndof1 * te_vdim;
         h2 += te_ndof2 * te_vdim;
      }

      elmat.SetSize(h1 + h2, w1 + w2);
      elmat = 0.0;

      int off_i1 = 0, off_j1 = 0, off_i2 = h1, off_j2 = w1;
      for (int i = 0; i < numIntegs; i++)
      {
         const int w = elmats[i].Width();
         const int h = elmats[i].Height();
         const int tr_vdim = w / (tr_ndof1 + tr_ndof2);
         const int te_vdim = h / (te_ndof1 + te_ndof2);
         w1 = tr_ndof1 * tr_vdim;
         w2 = tr_ndof2 * tr_vdim;
         h1 = te_ndof1 * te_vdim;
         h2 = te_ndof2 * te_vdim;
         elmat.CopyMN(elmats[i], h1, w1,  0,  0, off_i1, off_j1);
         elmat.CopyMN(elmats[i], h2, w1, h1,  0, off_i2, off_j1);
         elmat.CopyMN(elmats[i], h1, w2,  0, w1, off_i1, off_j2);
         elmat.CopyMN(elmats[i], h2, w2, h1, w1, off_i2, off_j2);
         off_j1 += w1;
         off_j2 += w2;
         off_i1 += h1;
         off_i2 += h2;
      }
   }
}

}

#endif
