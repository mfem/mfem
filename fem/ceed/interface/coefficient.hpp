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

#ifndef MFEM_LIBCEED_COEFF
#define MFEM_LIBCEED_COEFF

#ifdef MFEM_USE_CEED

#include "../../../general/forall.hpp"
#include "../../../config/config.hpp"
#include "../../../linalg/vector.hpp"
#include "../../../linalg/dtensor.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../gridfunc.hpp"
#include "../../qfunction.hpp"
#include "util.hpp"
#include "ceed.hpp"

namespace mfem
{

class Mesh;
class IntegrationRule;
class Coefficient;
class VectorCoefficient;
class GridFunction;

namespace ceed
{

struct Coefficient
{
   const int ncomp;
   Coefficient(int ncomp_) : ncomp(ncomp_) { }
   virtual bool IsConstant() const { return true; }
   virtual ~Coefficient() { }
};

struct VariableCoefficient : Coefficient
{
   CeedVector coeffVector = nullptr;
   const CeedEvalMode emode;
   VariableCoefficient(int ncomp_, CeedEvalMode emode_)
      : Coefficient(ncomp_), emode(emode_) { }
   virtual bool IsConstant() const override { return false; }
   ~VariableCoefficient()
   {
      CeedVectorDestroy(&coeffVector);
   }
};

struct GridCoefficient : VariableCoefficient
{
   const mfem::GridFunction &gf;
   CeedBasis basis;
   CeedElemRestriction restr;
   GridCoefficient(const mfem::GridFunction &gf_)
      : VariableCoefficient(gf_.VectorDim(), CEED_EVAL_INTERP),
        gf(gf_)
   {
      InitVector(gf, coeffVector);
   }
};

struct QuadCoefficient : VariableCoefficient
{
   mfem::Vector coeff;
   CeedElemRestriction restr;
   QuadCoefficient(int ncomp_) : VariableCoefficient(ncomp_, CEED_EVAL_NONE) { }
};

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::Coefficient @a Q, an mfem::Mesh @a mesh, and an mfem::IntegrationRule
    @a ir.

    @param[in] Q is the coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`.
    @param[out] ctx is the Context associated to the QFunction. */
template <typename Context>
void InitCoefficient(mfem::Coefficient *Q, mfem::Mesh &mesh,
                     const mfem::IntegrationRule &ir,
                     Coefficient*& coeff_ptr, Context &ctx)
{
   if ( Q == nullptr )
   {
      Coefficient *ceedCoeff = new Coefficient(1);
      ctx.coeff = 1.0;
      coeff_ptr = ceedCoeff;
   }
   else if (ConstantCoefficient *const_coeff =
               dynamic_cast<ConstantCoefficient*>(Q))
   {
      Coefficient *ceedCoeff = new Coefficient(1);
      ctx.coeff = const_coeff->constant;
      coeff_ptr = ceedCoeff;
   }
   else if (GridFunctionCoefficient* gf_coeff =
               dynamic_cast<GridFunctionCoefficient*>(Q))
   {
      GridCoefficient *ceedCoeff =
         new GridCoefficient(*gf_coeff->GetGridFunction());
      coeff_ptr = ceedCoeff;
   }
   else if (QuadratureFunctionCoefficient *cQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(Q))
   {
      QuadCoefficient *ceedCoeff = new QuadCoefficient(1);
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      const mfem::QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == nq * ne,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(&ir == &qFun.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qFun.Read();
      ceedCoeff->coeff.MakeRef(const_cast<mfem::QuadratureFunction &>(qFun),0);
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
   else
   {
      QuadCoefficient *ceedCoeff = new QuadCoefficient(1);
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      ceedCoeff->coeff.SetSize(nq * ne);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), nq, ne);
      for (int e = 0; e < ne; ++e)
      {
         mfem::ElementTransformation &T = *mesh.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            C(q,e) = Q->Eval(T, ir.IntPoint(q));
         }
      }
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
}


/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::VectorCoefficient @a VQ, an mfem::Mesh @a mesh, and an
    mfem::IntegrationRule @a ir.

    @param[in] VQ is the vector coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`.
    @param[out] ctx is the Context associated to the QFunction. */
template <typename Context>
void InitCoefficient(mfem::VectorCoefficient *VQ, mfem::Mesh &mesh,
                     const mfem::IntegrationRule &ir,
                     Coefficient *&coeff_ptr, Context &ctx)
{
   if (VectorConstantCoefficient *const_coeff =
          dynamic_cast<VectorConstantCoefficient*>(VQ))
   {
      const int vdim = const_coeff->GetVDim();
      const mfem::Vector &val = const_coeff->GetVec();
      Coefficient *ceedCoeff = new Coefficient(vdim);
      for (int i = 0; i < vdim; i++)
      {
         ctx.coeff[i] = val[i];
      }
      coeff_ptr = ceedCoeff;
   }
   else if (VectorGridFunctionCoefficient* vgf_coeff =
               dynamic_cast<VectorGridFunctionCoefficient*>(VQ))
   {
      GridCoefficient *ceedCoeff =
         new GridCoefficient(*vgf_coeff->GetGridFunction());
      coeff_ptr = ceedCoeff;
   }
   else if (VectorQuadratureFunctionCoefficient *cQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(VQ))
   {
      QuadCoefficient *ceedCoeff = new QuadCoefficient(cQ->GetVDim());
      const int dim = mesh.Dimension();
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      const mfem::QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == dim * nq * ne,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(&ir == &qFun.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qFun.Read();
      ceedCoeff->coeff.MakeRef(const_cast<mfem::QuadratureFunction &>(qFun),0);
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
   else
   {
      const int dim = mesh.Dimension();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(dim);
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      ceedCoeff->coeff.SetSize(dim * nq * ne);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), dim, nq, ne);
      mfem::DenseMatrix Q_ir;
      for (int e = 0; e < ne; ++e)
      {
         mfem::ElementTransformation &T = *mesh.GetElementTransformation(e);
         VQ->Eval(Q_ir, T, ir);
         for (int q = 0; q < nq; ++q)
         {
            for (int i = 0; i < dim; ++i)
            {
               C(i,q,e) = Q_ir(i,q);
            }
         }
      }
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
}

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::Coefficient @a Q, an mfem::Mesh @a mesh, and an mfem::IntegrationRule
    @a ir for the elements given by the indices @a indices.

    @param[in] Q is the coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] nelem The number of elements.
    @param[in] indices The indices of the elements of same type in the
                       `FiniteElementSpace`.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`.
    @param[out] ctx is the Context associated to the QFunction. */
template <typename Context>
void InitCoefficientWithIndices(mfem::Coefficient *Q, mfem::Mesh &mesh,
                                const mfem::IntegrationRule &ir,
                                int nelem,
                                const int* indices,
                                Coefficient*& coeff_ptr, Context &ctx)
{
   if ( Q == nullptr )
   {
      Coefficient *ceedCoeff = new Coefficient(1);
      ctx.coeff = 1.0;
      coeff_ptr = ceedCoeff;
   }
   else if (ConstantCoefficient *const_coeff =
               dynamic_cast<ConstantCoefficient*>(Q))
   {
      Coefficient *ceedCoeff = new Coefficient(1);
      ctx.coeff = const_coeff->constant;
      coeff_ptr = ceedCoeff;
   }
   else if (GridFunctionCoefficient* gf_coeff =
               dynamic_cast<GridFunctionCoefficient*>(Q))
   {
      GridCoefficient *ceedCoeff =
         new GridCoefficient(*gf_coeff->GetGridFunction());
      coeff_ptr = ceedCoeff;
   }
   else if (QuadratureFunctionCoefficient *cQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(Q))
   {
      QuadCoefficient *ceedCoeff = new QuadCoefficient(1);
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      const mfem::QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == nq * ne,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(&ir == &qFun.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      ceedCoeff->coeff.SetSize(nq * nelem);
      Memory<int> m_indices((int*)indices, nelem, false);
      auto in = Reshape(qFun.Read(), nq, ne);
      auto d_indices = Read(m_indices, nelem);
      auto out = Reshape(ceedCoeff->coeff.Write(), nq, nelem);
      MFEM_FORALL(i, nelem * nq,
      {
         const int q = i%nq;
         const int sub_e = i/nq;
         const int e = d_indices[sub_e];
         out(q, sub_e) = in(q, e);
      });
      m_indices.DeleteDevice();
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
   else
   {
      QuadCoefficient *ceedCoeff = new QuadCoefficient(1);
      const int nq = ir.GetNPoints();
      ceedCoeff->coeff.SetSize(nq * nelem);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), nq, nelem);
      for (int i = 0; i < nelem; ++i)
      {
         const int e = indices[i];
         mfem::ElementTransformation &T = *mesh.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            C(q, i) = Q->Eval(T, ir.IntPoint(q));
         }
      }
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
}


/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::VectorCoefficient @a Q, an mfem::Mesh @a mesh, and an
    mfem::IntegrationRule @a ir for the elements given by the indices @a indices.

    @param[in] VQ is the vector coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] nelem The number of elements.
    @param[in] indices The indices of the elements of same type in the
                       `FiniteElementSpace`.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`.
    @param[out] ctx is the Context associated to the QFunction. */
template <typename Context>
void InitCoefficientWithIndices(mfem::VectorCoefficient *VQ, mfem::Mesh &mesh,
                                const mfem::IntegrationRule &ir,
                                int nelem,
                                const int* indices,
                                Coefficient *&coeff_ptr, Context &ctx)
{
   if (VectorConstantCoefficient *const_coeff =
          dynamic_cast<VectorConstantCoefficient*>(VQ))
   {
      const int vdim = const_coeff->GetVDim();
      const mfem::Vector &val = const_coeff->GetVec();
      Coefficient *ceedCoeff = new Coefficient(vdim);
      for (int i = 0; i < vdim; i++)
      {
         ctx.coeff[i] = val[i];
      }
      coeff_ptr = ceedCoeff;
   }
   else if (VectorGridFunctionCoefficient* vgf_coeff =
               dynamic_cast<VectorGridFunctionCoefficient*>(VQ))
   {
      GridCoefficient *ceedCoeff =
         new GridCoefficient(*vgf_coeff->GetGridFunction());
      coeff_ptr = ceedCoeff;
   }
   else if (VectorQuadratureFunctionCoefficient *cQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(VQ))
   {
      QuadCoefficient *ceedCoeff = new QuadCoefficient(cQ->GetVDim());
      const int dim = mesh.Dimension();
      const int ne = mesh.GetNE();
      const int nq = ir.GetNPoints();
      const mfem::QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == dim * nq * ne,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(&ir == &qFun.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      ceedCoeff->coeff.SetSize(dim * nq * nelem);
      Memory<int> m_indices((int*)indices, nelem, false);
      auto in = Reshape(qFun.Read(), dim, nq, ne);
      auto d_indices = Read(m_indices, nelem);
      auto out = Reshape(ceedCoeff->coeff.Write(), dim, nq, nelem);
      MFEM_FORALL(i, nelem * nq,
      {
         const int q = i%nq;
         const int sub_e = i/nq;
         const int e = d_indices[sub_e];
         for (int d = 0; d < dim; d++)
         {
            out(d, q, sub_e) = in(d, q, e);
         }
      });
      m_indices.DeleteDevice();
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
   else
   {
      const int dim = mesh.Dimension();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(dim);
      const int nq = ir.GetNPoints();
      ceedCoeff->coeff.SetSize(dim * nq * nelem);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), dim, nq, nelem);
      mfem::DenseMatrix Q_ir;
      for (int i = 0; i < nelem; ++i)
      {
         const int e = indices[i];
         mfem::ElementTransformation &T = *mesh.GetElementTransformation(e);
         VQ->Eval(Q_ir, T, ir);
         for (int q = 0; q < nq; ++q)
         {
            for (int d = 0; d < dim; ++d)
            {
               C(d, q, i) = Q_ir(d, q);
            }
         }
      }
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
}

template <typename Coeff, typename Context>
void InitCoefficient(Coeff *Q, mfem::Mesh &mesh,
                     const mfem::IntegrationRule &ir, int nelem,
                     const int* indices, Coefficient *&coeff_ptr, Context &ctx)
{
   if (indices)
   {
      InitCoefficientWithIndices(Q, mesh, ir, nelem, indices, coeff_ptr, ctx);
   }
   else
   {
      InitCoefficient(Q, mesh, ir, coeff_ptr, ctx);
   }
}

} // namespace ceed

} // namespace mfem

#endif

#endif // MFEM_LIBCEED_COEFF
