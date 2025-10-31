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

#ifndef MFEM_HDG_DARCYOP
#define MFEM_HDG_DARCYOP

#include "mfem.hpp"
#include "../../general/socketstream.hpp"
#include <array>
#include <vector>

namespace mfem
{
namespace hdg
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
   inline const DarcyForm& GetDarcyForm() const { return *darcy; }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;
};

void RandomizeMesh(Mesh &mesh, real_t dr);

class DarcyErrorEstimator : public ErrorEstimator
{
   BilinearFormIntegrator &bfi;
   const GridFunction &sol_tr, &sol_p;
   long current_sequence{-1};
   Vector error_estimates;
   real_t total_error{};

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = sol_tr.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   DarcyErrorEstimator(BilinearFormIntegrator &integ, const GridFunction &solr,
                       const GridFunction &solp)
      : bfi(integ), sol_tr(solr), sol_p(solp) { }

   real_t GetTotalError() const override { return total_error; }

   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   void Reset() override { current_sequence = -1; }
};

class VectorBlockDiagonalIntegrator : public BilinearFormIntegrator
{
   int numIntegs;
   std::vector<BilinearFormIntegrator*> integs;
   bool own_integs{true};

   std::vector<DenseMatrix> elmats;

   template<typename FType, typename... Args> void AssembleElementMat(
      FType f, DenseMatrix &elmat, Args&&... args)
   {
      AssembleMat<FType, 0, 0, Args...>(f, {}, {}, elmat, args...);
   }

   template<typename FType, typename... Args> void AssembleTraceMat(
      FType f, const FiniteElement &test_fe1, const FiniteElement &test_fe2,
      DenseMatrix &elmat, Args&&... args)
   {
      AssembleMat<FType, 0, 2, Args...>(f, {}, {&test_fe1, &test_fe2}, elmat,
                                        args...);
   }

   template<typename FType, typename... Args> void AssembleFaceMat(
      FType f, const FiniteElement &trial_fe1, const FiniteElement &test_fe1,
      const FiniteElement &trial_fe2, const FiniteElement &test_fe2,
      DenseMatrix &elmat, Args&&... args)
   {
      AssembleMat<FType, 2, 2, Args...>(f, {&trial_fe1, &trial_fe2}, {&test_fe1, &test_fe2},
                                        elmat, args...);
   }

   template<typename FType, typename... Args> void AssembleHDGFaceMat(
      FType f, const FiniteElement &trace_el, const FiniteElement &fe1,
      const FiniteElement &fe2, DenseMatrix &elmat, Args&&... args)
   {
      AssembleMat<FType, 3, 3, Args...>(f, {&fe1, &fe2, &trace_el}, {&fe1, &fe2, &trace_el},
                                        elmat, args...);
   }

   template<typename FType, int N, int M, typename... Args> void AssembleMat(
      FType f,
      std::array<const FiniteElement*,N> trial_fe,
      std::array<const FiniteElement*,M> test_fe,
      DenseMatrix &elmat, Args&&... args);

public:
   VectorBlockDiagonalIntegrator(int n) : numIntegs(n) { integs.resize(n); }
   VectorBlockDiagonalIntegrator(int n, BilinearFormIntegrator *integ_)
      : numIntegs(n) { integs.push_back(integ_); }

   VectorBlockDiagonalIntegrator(const std::vector<BilinearFormIntegrator*>
                                 &integs_)
      : numIntegs(integs_.size())
   {
      integs.reserve(numIntegs);
      for (BilinearFormIntegrator *bfi : integs_)
      {
         integs.push_back(bfi);
      }
   }

   template<int N>
   VectorBlockDiagonalIntegrator(BilinearFormIntegrator *integs_[N])
      : numIntegs(N)
   {
      integs.reserve(numIntegs);
      for (BilinearFormIntegrator *bfi : integs_)
      {
         integs.push_back(bfi);
      }
   }

   VectorBlockDiagonalIntegrator(
      std::initializer_list<BilinearFormIntegrator*> integs_)
      : numIntegs(integs_.size())
   {
      integs.reserve(numIntegs);
      for (BilinearFormIntegrator *bfi : integs_)
      {
         integs.push_back(bfi);
      }
   }

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
      face_fx fx = &BilinearFormIntegrator::AssembleFaceMatrix;
      if (Trans.Elem2No >= 0)
         AssembleFaceMat<>(fx, fe1, fe1, fe2, fe2,
                           elmat, fe1, fe2, Trans);
      else
      {
         AssembleElementMat<>(fx, elmat, fe1, fe2, Trans);
      }

   }

   void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                           const FiniteElement &test_fe1,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override
   {
      using face_fx = void (BilinearFormIntegrator::*)(
                         const FiniteElement &, const FiniteElement &, const FiniteElement &,
                         FaceElementTransformations &, DenseMatrix &);
      face_fx fx = &BilinearFormIntegrator::AssembleFaceMatrix;
      if (Trans.Elem2No >= 0)
         AssembleTraceMat<>(fx, test_fe1, test_fe2, elmat, trial_face_fe,
                            test_fe1, test_fe2, Trans);
      else
         AssembleElementMat<>(fx, elmat, trial_face_fe, test_fe1,
                              test_fe2, Trans);
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
      face_fx fx = &BilinearFormIntegrator::AssembleFaceMatrix;
      if (Trans.Elem2No >= 0)
         AssembleFaceMat<>(fx, trial_fe1, test_fe1, trial_fe2, test_fe2,
                           elmat, trial_fe1, test_fe1, trial_fe2, test_fe2,
                           Trans);
      else
         AssembleElementMat<>(fx, elmat, trial_fe1, test_fe1,
                              trial_fe2, test_fe2, Trans);
   }

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override
   {
      using face_fx = void (BilinearFormIntegrator::*)(
                         const FiniteElement &,
                         const FiniteElement &, const FiniteElement &,
                         FaceElementTransformations &, DenseMatrix &);
      face_fx fx = &BilinearFormIntegrator::AssembleHDGFaceMatrix;
      if (Trans.Elem2No >= 0)
         AssembleHDGFaceMat<>(fx, trace_el, el1, el2, elmat,
                              trace_el, el1, el2, Trans);
      else
         AssembleFaceMat<>(fx, el1, trace_el, el1, trace_el,
                           elmat, trace_el, el1, el2, Trans);
   }
};

template<typename FType, int N, int M, typename... Args>
void VectorBlockDiagonalIntegrator::AssembleMat(
   FType f,
   std::array<const FiniteElement*,N> trial_fes,
   std::array<const FiniteElement*,M> test_fes,
   DenseMatrix &elmat, Args&&... args)
{
   constexpr int NN = (N > 0)?(N):(1);
   constexpr int MM = (M > 0)?(M):(1);
   int tr_ndofs = 0, te_ndofs = 0;
   for (auto *fe : trial_fes) { tr_ndofs += fe->GetDof(); }
   for (auto *fe : test_fes) { te_ndofs += fe->GetDof(); }

   if (numIntegs > (int)integs.size())
   {
      // single integrator
      elmats.resize(1);
      (integs[0]->*f)(args..., elmats[0]);
      const int w = elmats[0].Width();
      const int h = elmats[0].Height();
      const int tr_vdim = (N > 0)?(w / tr_ndofs):(1);
      const int te_vdim = (M > 0)?(h / te_ndofs):(1);

      elmat.SetSize(numIntegs * h, numIntegs * w);
      elmat = 0.0;

      int off_n = 0;
      for (int n = 0; n < NN; n++)
      {
         const int w_n = (N != 0)?(trial_fes[n]->GetDof() * tr_vdim):(w);

         int off_m = 0;
         for (int m = 0; m < MM; m++)
         {
            const int h_m = (M != 0)?(test_fes[m]->GetDof() * te_vdim):(h);

            for (int i = 0; i < numIntegs; i++)
            {
               elmat.CopyMN(elmats[0],
                            h_m, w_n, off_m, off_n,
                            numIntegs * off_m + i * h_m,
                            numIntegs * off_n + i * w_n);
            }
            off_m += h_m;
         }
         off_n += w_n;
      }
   }
   else
   {
      // corresponding number of integrators
      elmats.resize(numIntegs);
      std::array<int, NN> ws{};
      std::array<int, MM> hs{};
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
         if (N > 0)
         {
            const int tr_vdim = w / tr_ndofs;
            for (int n = 0; n < N; n++)
            { ws[n] += trial_fes[n]->GetDof() * tr_vdim; }
         }
         else
         {
            ws[0] += w;
         }

         if (M > 0)
         {
            const int te_vdim = h / te_ndofs;
            for (int m = 0; m < M; m++)
            { hs[m] += test_fes[m]->GetDof() * te_vdim; }
         }
         else
         {
            hs[0] += h;
         }
      }

      int tot_w = 0;
      for (int w : ws) { tot_w += w; }
      int tot_h = 0;
      for (int h : hs) { tot_h += h; }

      elmat.SetSize(tot_h, tot_w);
      elmat = 0.0;

      std::array<int, NN> off_js{};
      std::array<int, MM> off_is{};
      for (int j = 0; j < NN-1; j++)
      {
         off_js[j+1] = off_js[j] + ws[j];
      }
      for (int i = 0; i < MM-1; i++)
      {
         off_is[i+1] = off_is[i] + hs[i];
      }
      for (int i = 0; i < numIntegs; i++)
      {
         const int w = elmats[i].Width();
         const int h = elmats[i].Height();
         const int tr_vdim = (N > 0)?(w / tr_ndofs):(1);
         const int te_vdim = (M > 0)?(h / te_ndofs):(1);
         int off_n = 0;
         for (int n = 0; n < NN; n++)
         {
            const int w_n = (N > 0)?(trial_fes[n]->GetDof() * tr_vdim):(w);
            int off_m = 0;
            for (int m = 0; m < MM; m++)
            {
               const int h_m = (M > 0)?(test_fes[m]->GetDof() * te_vdim):(h);
               elmat.CopyMN(elmats[i], h_m, w_n, off_m, off_n, off_is[m], off_js[n]);
               off_m += h_m;
               if (n == NN-1)
               {
                  off_is[m] += h_m;
               }
            }
            off_n += w_n;
            off_js[n] += w_n;
         }
      }
   }
}

} // namespace hdg
} // namespace mfem

#endif // MFEM_HDG_DARCYOP
