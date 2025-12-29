#pragma once

#include "linalg/densemat.hpp"
#include "mfem.hpp"
#include "./legendre.hpp"

namespace mfem
{

class InterpolationOperator : public Operator
{
   FiniteElementSpace &fes;
   QuadratureSpace &qs;
   std::unique_ptr<GridFunction> target_gf;
   mutable QuadratureFunction target_qf;
#ifdef MFEM_USE_MPI
   bool is_parallel = false;
#else
   constexpr bool is_parallel = false;
#endif
public:
   bool IsParallel() const { return is_parallel; }
   InterpolationOperator(FiniteElementSpace &fes_,
                         QuadratureSpace &qs_)
      : Operator(qs_.GetSize()*fes_.GetVectorDim(), fes_.GetTrueVSize())
      , fes(fes_)
      , qs(qs_)
      , target_qf(&qs_, nullptr, fes_.GetVectorDim())
   {
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
      if (pfes)
      {
         target_gf = std::make_unique<ParGridFunction>(pfes);
      }
#endif
      if (!target_gf)
      {
         target_gf = std::make_unique<GridFunction>(&fes);
      }
   }

   void Mult(const Vector &gf_tvec, Vector &qf_vec) const override
   {
      target_gf->SetFromTrueDofs(gf_tvec);
      qf_vec.SetSize(qs.GetSize()*fes.GetVectorDim());
      target_qf.SetData(qf_vec.GetData());
      target_qf.ProjectGridFunction(*target_gf);
      target_qf *= qs.GetWeights();
   }

   // Transpose of interpolation
   void MultTranspose(const Vector &qf_vec, Vector &gf_tvec) const override
   {
      const int vdim = fes.GetVectorDim();
      gf_tvec.SetSize(fes.GetTrueVSize());

      // We need a temporary GridFunction to handle the mapping to DOFs
      Vector el_vdofs; // Local DOFs for the element
      Array<int> vdofs;
      *target_gf = 0.0; // initialize to zero
      const Vector &weights = qs.GetWeights();

      Vector shape;
      for (int e=0; e < fes.GetNE(); e++)
      {
         const FiniteElement *fe = fes.GetFE(e);
         const int dof = fe->GetDof();
         const int nqp = qs.GetElementIntRule(e).GetNPoints();
         const int qs_offset = qs.Offset(e);
         shape.SetSize(dof);

         fes.GetElementVDofs(e, vdofs);
         el_vdofs.SetSize(vdofs.Size());
         el_vdofs = 0.0;

         // Matrix representing shape functions evaluated at all quadrature points in the element
         // Dimensions: nqp x dof
         const IntegrationRule &ir = qs.GetElementIntRule(e);
         ElementTransformation &Tr = *qs.GetTransformation(e);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            fe->CalcPhysShape(Tr, shape);
            const real_t w = weights[qs_offset + q];
            for (int d = 0; d < vdim; d++)
            {
               // The value at this quadrature point for this vector component
               // Note: indexing depends on how qf_vec stores components (usually block or interleaved)
               real_t val = qf_vec(qs_offset * vdim + q * vdim + d)*w;
               for (int i = 0; i < dof; i++)
               {
                  // Accumulate: Shape function value * quadrature value
                  el_vdofs(i*vdim + d) += shape(i) * val;
               }
            }
         }
         // Add local contributions to the global GridFunction
         target_gf->AddElementVector(vdofs, el_vdofs);
      }
      target_gf->GetTrueDofs(gf_tvec);
   }
};

class BoundingOperator : public Operator
{
   FiniteElementSpace &fes;
   QuadratureSpace &qs;
   std::unique_ptr<GridFunction> target_gf;
   mutable QuadratureFunction target_qf;
   PLBound bound;
   DenseMatrix L;
   Vector Lones;
#ifdef MFEM_USE_MPI
   bool is_parallel = false;
#else
   constexpr bool is_parallel = false;
#endif
public:
   bool IsParallel() const { return is_parallel; }
   BoundingOperator(FiniteElementSpace &fes_,
                    QuadratureSpace &qs_)
      : Operator(qs_.GetSize()*fes_.GetVectorDim(), fes_.GetTrueVSize())
      , fes(fes_)
      , qs(qs_)
      , target_qf(&qs_, nullptr, fes_.GetVectorDim())
      , bound(&fes_)
   {
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
      if (pfes)
      {
         target_gf = std::make_unique<ParGridFunction>(pfes);
      }
#endif
      if (!target_gf)
      {
         target_gf = std::make_unique<GridFunction>(&fes);
      }
      L = bound.GetUpperBoundMatrix(fes_.GetMesh()->Dimension(), &fes_);
      MFEM_VERIFY(L.Height() == qs_.GetElementIntRule(0).GetNPoints(),
                  "Bounding matrix and quadrature rule size mismatch.");
   }

   void Mult(const Vector &gf_tvec, Vector &qf_vec) const override
   {
      target_gf->SetFromTrueDofs(gf_tvec);
      Vector dof_vals;
      qf_vec.SetSize(qs.GetSize()*fes.GetVectorDim());
      Vector qf_local(fes.GetVectorDim());
      for (int i=0; i<qs.GetNE(); i++)
      {
         target_gf->GetElementDofValues(i, dof_vals);
         qf_local.SetData(qf_vec.GetData() + qs.Offset(i)*fes.GetVectorDim());
         L.Mult(dof_vals, qf_local);
      }
      qf_vec *= qs.GetWeights();
   }
   // Transpose of interpolation
   void MultTranspose(const Vector &qf_vec, Vector &gf_tvec) const override
   {
      const int vdim = fes.GetVectorDim();
      gf_tvec.SetSize(fes.GetTrueVSize());

      // We need a temporary GridFunction to handle the mapping to DOFs
      Vector el_vdofs; // Local DOFs for the element
      Array<int> vdofs;
      *target_gf = 0.0; // initialize to zero

      IntegrationRule ir = qs.GetElementIntRule(0);
      Vector qf_local(fes.GetVectorDim()*ir.GetNPoints());
      Vector wqf_local(fes.GetVectorDim()*ir.GetNPoints());
      Vector gf_local(fes.GetFE(0)->GetDof()*vdim);
      const Vector &w = qs.GetWeights();
      Vector w_local(L.Height());
      for (int i=0; i<qs.GetNE(); i++)
      {
         qf_local.SetData(qf_vec.GetData() + qs.Offset(i)*vdim);
         wqf_local = qf_local;
         // Extract local weights
         w_local.SetData(w.GetData() + qs.Offset(i));
         wqf_local *= w_local;
         L.MultTranspose(wqf_local, gf_local);
         fes.GetElementVDofs(i, vdofs);
         target_gf->AddElementVector(vdofs, gf_local);
      }
      target_gf->GetTrueDofs(gf_tvec);
   }
};

class PGOperator : public Operator
{
   Operator &A; // primal spd matrix
   Operator &B; // coupling matrix
   LegendreFunctional &entropy; // entropy function R
   Array<int> offsets;
   mutable std::unique_ptr<BlockOperator> hess;
   mutable std::unique_ptr<SparseMatrix> hess_sparse;
#ifdef MFEM_USE_MPI
   bool is_parallel = false;
   mutable std::unique_ptr<HypreParMatrix> hess_par;
#else
   constexpr bool is_parallel = false;
#endif
public:
   PGOperator(Operator &A_,
              Operator &B_,
              LegendreFunctional &entropy_)
      : Operator(A_.Height() + B_.Height())
      , A(A_), B(B_), entropy(entropy_)
   {
      offsets.SetSize(3);
      offsets[0] = 0;
      offsets[1] = A.Height();
      offsets[2] = B.Height();
      offsets.PartialSum();
#ifdef MFEM_USE_MPI
      {
         auto pA = dynamic_cast<HypreParMatrix*>(&A);
         auto pB = dynamic_cast<HypreParMatrix*>(&B);
         if (pA && pB)
         {
            hess = std::make_unique<BlockOperator>(offsets);
            hess->SetBlock(0, 0, pA);
            hess->SetBlock(1, 0, pB);
            hess->SetBlock(0, 1, pB->Transpose());
            hess->owns_blocks = false;
            is_parallel = true;
         }
      }
#endif
      if (!hess)
      {
         auto Amat = dynamic_cast<SparseMatrix*>(&A);
         auto Bmat = dynamic_cast<SparseMatrix*>(&B);
         if (Amat && Bmat)
         {
            hess = std::make_unique<BlockOperator>(offsets);
            hess->SetBlock(0, 0, &A);
            hess->SetBlock(1, 0, &B);
            hess->SetBlock(0, 1, Transpose(*Bmat));
            hess->owns_blocks = false;
         }
      }

   }

   ~PGOperator()
   {
      if (hess)
      {
         if (!hess->IsZeroBlock(0,1))
         {
            Operator& Bt = hess->GetBlock(0,1);
#ifdef MFEM_USE_MPI
            if (auto pBt = dynamic_cast<HypreParMatrix*>(&Bt))
            {
               delete pBt;
            }
            else { delete static_cast<SparseMatrix*>(&Bt); }
#else
            delete static_cast<SparseMatrix*>(&Bt);
#endif
         }
      }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector X(const_cast<Vector&>(x), offsets);
      BlockVector Y(y, offsets);
      A.Mult(X.GetBlock(0), Y.GetBlock(0));
      B.AddMultTranspose(X.GetBlock(1), Y.GetBlock(0));
      B.Mult(X.GetBlock(0), Y.GetBlock(1));
      entropy.Mult(X.GetBlock(1), Y.GetBlock(1));
   }

   Operator &GetGradient(const Vector &x) const override
   {
      MFEM_VERIFY(hess,
                  "Hessian not available for this PGOperator. A and B must be matrix types");
      BlockVector X(const_cast<Vector&>(x), offsets);
      hess->SetBlock(1, 1, &entropy.GetGradient(X.GetBlock(1)), -1.0);
      if (is_parallel)
      {
#ifdef MFEM_USE_MPI
         Array2D<const HypreParMatrix*> blocks(2, 2);
         blocks(0,0) = dynamic_cast<const HypreParMatrix*>(&hess->GetBlock(0,0));
         blocks(1,0) = dynamic_cast<const HypreParMatrix*>(&hess->GetBlock(1,0));
         blocks(0,1) = dynamic_cast<const HypreParMatrix*>(&hess->GetBlock(0,1));
         blocks(1,1) = dynamic_cast<const HypreParMatrix*>(&hess->GetBlock(1,1));
         Array2D<real_t> coeffs(2,2);
         coeffs = 1.0;
         coeffs(1,1) = -1.0;
         hess_par.reset(HypreParMatrixFromBlocks(blocks, &coeffs));
         return *hess_par;
#endif
      }
      else
      {
         BlockMatrix blocks(offsets);
         blocks.SetBlock(0,0, dynamic_cast<SparseMatrix*>(&hess->GetBlock(0,0)));
         blocks.SetBlock(1,0, dynamic_cast<SparseMatrix*>(&hess->GetBlock(1,0)));
         blocks.SetBlock(0,1, dynamic_cast<SparseMatrix*>(&hess->GetBlock(0,1)));
         blocks.SetBlock(1,1, dynamic_cast<SparseMatrix*>(&hess->GetBlock(1,1)));
         blocks.GetBlock(1,1) *= -1.0;
         hess_sparse.reset(blocks.CreateMonolithic());
         return *hess_sparse;
      }
   }
};

// Given a system
// Au + B^T lambda = f
// Bu - grad R^*(psi + alpha*lambda) = 0
// We condense the system
// BA^{-1}B^T lambda + grad R^*(psi + alpha*lambda) = BA^{-1}f
// to obtain a system for lambda only.
// Mult will return the action of the condensed operator on lambda.
// GetGradient will return the gradient of the condensed operator at
// the current psi and alpha.
class CondensedLocalPGOperator : public Operator
{
   const DenseMatrix &A; // primal spd matrix
   const DenseMatrix &B; // coupling matrix
   const LegendreFunction &entropy; // entropy function R
   const Vector &psi_k; // previous psi_k
   const real_t &alpha; // step size
   const Vector &w; // weights for dofs

   mutable Vector psi;

   DenseMatrixInverse inverter;
   DenseMatrix invA;
   DenseMatrix BinvA;
   DenseMatrix BinvABt;
   mutable DenseMatrix S;
   void UpdateSchur()
   {
      MFEM_VERIFY(A.Height() == A.Width(),
                  "Matrix A must be square.");
      MFEM_VERIFY(B.Width() == A.Width(),
                  "Matrix dimensions do not match for multiplication.");
      MFEM_VERIFY(B.Height() == psi_k.Size(),
                  "Matrix and vector dimensions do not match for multiplication.");
      MFEM_VERIFY(w.Size() == psi_k.Size(),
                  "Weight and vector dimensions do not match for multiplication.");
      MFEM_VERIFY(alpha > 0.0,
                  "Step size alpha must be positive.");

      using mfem::Mult;
      inverter.Factor(A);
      inverter.GetInverseMatrix(invA);
      BinvA.SetSize(B.Height(), A.Width());
      Mult(B, invA, BinvA);
      BinvABt.SetSize(B.Height(), B.Height());
      MultABt(BinvA, B, BinvABt);
      S = BinvABt;
      psi.SetSize(psi_k.Size());
   }
public:
   CondensedLocalPGOperator(const DenseMatrix &A_,
                            const DenseMatrix &B_,
                            const LegendreFunction &entropy_,
                            const Vector &psi_,
                            const real_t &alpha_,
                            const Vector &w_)
      : Operator(B_.Height())
      , A(A_), B(B_), entropy(entropy_), psi_k(psi_), alpha(alpha_), w(w_)
   {
      UpdateSchur();
   }
   void Update() { UpdateSchur(); }
   void Mult(const Vector &lambda, Vector &y) const override
   {
      MFEM_VERIFY(lambda.Size() == psi_k.Size(),
                  "Input vector size does not match latent variable size.");
      add(psi_k, alpha, lambda, psi);
      y.SetSize(lambda.Size());
      // y = BA^{-1}B^T lambda + grad R^*(psi + alpha*lambda)
      BinvABt.Mult(lambda, y);
      for (int i=0; i<psi.Size(); i++)
      {
         y(i) += entropy.gradinv(psi(i))*w(i);
      }
   }
   void RecoverPrimal(const Vector &F, const Vector &lambda, Vector &u) const
   {
      invA.Mult(F, u);
      BinvA.AddMultTranspose(lambda, u, -1.0);
   }
   void CondensedRHS(const Vector &F, Vector &rhs) const
   {
      rhs.SetSize(psi_k.Size());
      BinvA.Mult(F, rhs);
   }
   Operator &GetGradient(const Vector &lambda) const override
   {
      add(psi_k, alpha, lambda, psi);
      for (int i=0; i<psi.Size(); i++)
      {
         real_t hessval = entropy.hessinv(psi(i));
         S(i,i) += BinvABt(i, i) + alpha*hessval*w(i);
      }
      return S;
   }
};

class CondensedGlobalPGOperator : public Operator
{
   const Operator &invA; // primal spd matrix
   const Operator &B; // coupling matrix
   LegendreFunctional &entropy;

   mutable Vector psi;
   mutable Vector primal_rhs_vec;
   mutable Vector primal_sol_vec;
   Array<int> ess_tdof_list = {};

   mutable std::unique_ptr<SparseMatrix> S;
   mutable std::unique_ptr<SparseMatrix> BinvABt;
   mutable std::unique_ptr<HypreParMatrix> S_par;
   void UpdateSchur()
   {
      MFEM_VERIFY(invA.Height() == invA.Width(),
                  "Matrix A must be square.");
      MFEM_VERIFY(B.Width() == invA.Width(),
                  "Matrix dimensions do not match for multiplication.");
      MFEM_VERIFY(B.Height() == entropy.Height(),
                  "Matrix and vector dimensions do not match for multiplication.");
      psi.SetSize(B.Height());
      primal_sol_vec.SetSize(B.Width());
      primal_sol_vec = 0.0;
      primal_rhs_vec.SetSize(B.Width());
      primal_rhs_vec = 0.0;

      const SparseMatrix * invA_sp = dynamic_cast<const SparseMatrix*>(&invA);
      if (invA_sp)
      {
         out << "Using SparseMatrix for invA in CondensedGlobalPGOperator." << std::endl;
         const SparseMatrix * B_sp = dynamic_cast<const SparseMatrix*>(&B);
         if (B_sp)
         {
            using mfem::Mult;
            std::unique_ptr<SparseMatrix> BinvA(Mult(*B_sp, *invA_sp));
            std::unique_ptr<SparseMatrix> Bt(Transpose(*B_sp));
            out << "Computing BinvABt..." << std::endl;
            BinvABt.reset(Mult(*BinvA, *Bt));
            S.reset(new SparseMatrix(BinvABt->GetI(), BinvABt->GetJ(),
                                     new real_t[BinvABt->NumNonZeroElems()], B.Height(), B.Height(), false, true,
                                     true));
         }
      }
   }
public:
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list_)
   { ess_tdof_list = ess_tdof_list_; }
   CondensedGlobalPGOperator(const Operator &invA_,
                             const Operator &B_,
                             LegendreFunctional &entropy_)
      : Operator(B_.Height())
      , invA(invA_), B(B_), entropy(entropy_)
   {
      UpdateSchur();
   }
   void Update() { UpdateSchur(); }
   void Mult(const Vector &lambda, Vector &y) const override
   {
      // y = BA^{-1}B^T lambda + grad R^*(psi + alpha*lambda)
      y.SetSize(lambda.Size());

      // BinvABt.Mult(lambda, y);
      if (BinvABt)
      {
         BinvABt->Mult(lambda, y);
      }
      else
      {
         // Fallback if BinvABt is not available
         B.MultTranspose(lambda, primal_rhs_vec);
         invA.Mult(primal_rhs_vec, primal_sol_vec);
         B.Mult(primal_sol_vec, y);
      }

      // add grad R^*(psi + alpha*lambda)
      entropy.AddMult(lambda, y);
   }
   Operator &GetGradient(const Vector &lambda) const override
   {
      *S = *BinvABt;
      SparseMatrix * hess = dynamic_cast<SparseMatrix*>(&entropy.GetGradient(lambda));
      MFEM_VERIFY(hess != nullptr,
                  "Hessian must be a SparseMatrix.");
      *S += *hess;
      return *S;
   }
   void RecoverPrimal(const Vector &F, const Vector &lambda, Vector &u) const
   {
      // u = A^{-1}(f - B^T lambda)
      B.MultTranspose(lambda, primal_rhs_vec);
      primal_rhs_vec -= F;
      invA.Mult(primal_rhs_vec, u);
      u.Neg();
   }
   void CondensedRHS(const Vector &F, Vector &rhs) const
   {
      // rhs = -rhs + BA^{-1}f
      primal_sol_vec = 0.0;
      invA.Mult(F, primal_sol_vec);
      rhs.Neg();
      B.AddMult(primal_sol_vec, rhs);
   }
};

} // namespace mfem
