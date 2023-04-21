// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include "../../../linalg/vector.hpp"
#include "../../../linalg/dtensor.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../gridfunc.hpp"
#include "../../qfunction.hpp"
#include "util.hpp"
#include "ceed.hpp"

namespace mfem
{

namespace ceed
{

struct Coefficient
{
   CeedVector coeffVector = nullptr;
   const int ncomp;
   const CeedEvalMode emode;
   Coefficient(int ncomp_, CeedEvalMode emode_) : ncomp(ncomp_), emode(emode_) {}
   virtual ~Coefficient()
   {
      CeedVectorDestroy(&coeffVector);
   }
};

struct GridCoefficient : Coefficient
{
   const mfem::GridFunction &gf;
   CeedBasis basis;
   CeedElemRestriction restr;
   GridCoefficient(const mfem::GridFunction &gf_)
      : Coefficient(gf_.VectorDim(), CEED_EVAL_INTERP), gf(gf_)
   {
      InitVector(gf, coeffVector);
   }
};

struct QuadCoefficient : Coefficient
{
   mfem::Vector coeff;
   CeedElemRestriction restr;
   QuadCoefficient(int ncomp_) : Coefficient(ncomp_, CEED_EVAL_NONE) {}
};

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::Coefficient @a Q, an mfem::Mesh @a mesh, and an mfem::IntegrationRule
    @a ir.

    @param[in] Q is the coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficient(mfem::Coefficient *Q, mfem::Mesh &mesh,
                            const mfem::IntegrationRule &ir, bool use_bdr,
                            Coefficient *&coeff_ptr)
{
   if (Q == nullptr || dynamic_cast<ConstantCoefficient *>(Q))
   {
      // The constant coefficient case is handled by the QFunction context
      coeff_ptr = nullptr;
   }
   else if (mfem::GridFunctionCoefficient *gf_coeff =
               dynamic_cast<mfem::GridFunctionCoefficient *>(Q))
   {
      GridCoefficient *ceedCoeff =
         new GridCoefficient(*gf_coeff->GetGridFunction());
      coeff_ptr = ceedCoeff;
   }
   else if (mfem::QuadratureFunctionCoefficient *qf_coeff =
               dynamic_cast<mfem::QuadratureFunctionCoefficient *>(Q))
   {
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(1);
      const mfem::QuadratureFunction &qfunc = qf_coeff->GetQuadFunction();
      MFEM_VERIFY(qfunc.Size() == nq * ne,
                  "Incompatible QuadratureFunction dimension\n");
      MFEM_VERIFY(&ir == &qfunc.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qfunc.Read();
      ceedCoeff->coeff.MakeRef(const_cast<mfem::QuadratureFunction &>(qfunc), 0);
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
   else
   {
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(1);
      ceedCoeff->coeff.SetSize(nq * ne);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), nq, ne);
      for (int e = 0; e < ne; ++e)
      {
         auto &T = use_bdr ? *mesh.GetBdrElementTransformation(e) :
                   *mesh.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            C(q, e) = Q->Eval(T, ir.IntPoint(q));
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
                          `CeedOperator`. */
inline void InitCoefficient(mfem::VectorCoefficient *VQ, mfem::Mesh &mesh,
                            const mfem::IntegrationRule &ir, bool use_bdr,
                            Coefficient *&coeff_ptr)
{
   if (VQ == nullptr || dynamic_cast<mfem::VectorConstantCoefficient *>(VQ))
   {
      // The constant coefficient case is handled by the QFunction context
      coeff_ptr = nullptr;
   }
   else if (mfem::VectorGridFunctionCoefficient *vgf_coeff =
               dynamic_cast<mfem::VectorGridFunctionCoefficient *>(VQ))
   {
      GridCoefficient *ceedCoeff =
         new GridCoefficient(*vgf_coeff->GetGridFunction());
      coeff_ptr = ceedCoeff;
   }
   else if (mfem::VectorQuadratureFunctionCoefficient *vqf_coeff =
               dynamic_cast<mfem::VectorQuadratureFunctionCoefficient *>(VQ))
   {
      const int vdim = vqf_coeff->GetVDim();
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(vdim);
      const mfem::QuadratureFunction &qfunc = vqf_coeff->GetQuadFunction();
      MFEM_VERIFY(qfunc.Size() == vdim * nq * ne,
                  "Incompatible QuadratureFunction dimension\n");
      MFEM_VERIFY(&ir == &qfunc.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qfunc.Read();
      ceedCoeff->coeff.MakeRef(const_cast<mfem::QuadratureFunction &>(qfunc), 0);
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
   else
   {
      const int vdim = VQ->GetVDim();
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(vdim);
      ceedCoeff->coeff.SetSize(vdim * nq * ne);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), vdim, nq, ne);
      mfem::DenseMatrix Q_ir;
      for (int e = 0; e < ne; ++e)
      {
         auto &T = use_bdr ? *mesh.GetBdrElementTransformation(e) :
                   *mesh.GetElementTransformation(e);
         VQ->Eval(Q_ir, T, ir);
         for (int q = 0; q < nq; ++q)
         {
            for (int i = 0; i < vdim; ++i)
            {
               C(i, q, e) = Q_ir(i, q);
            }
         }
      }
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
}

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::MatrixCoefficient @a MQ, an mfem::Mesh @a mesh, and an
    mfem::IntegrationRule @a ir.

    @param[in] MQ is the matrix coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficient(mfem::MatrixCoefficient *MQ, mfem::Mesh &mesh,
                            const mfem::IntegrationRule &ir, bool use_bdr,
                            Coefficient *&coeff_ptr)
{
   if (MQ == nullptr || dynamic_cast<mfem::MatrixConstantCoefficient *>(MQ))
   {
      // The constant coefficient case is handled by the QFunction context
      coeff_ptr = nullptr;
   }
   else
   {
      // Assumes matrix coefficient is symmetric
      const int vdim = MQ->GetVDim();
      const int ncomp = vdim * (vdim + 1) / 2;
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(ncomp);
      ceedCoeff->coeff.SetSize(ncomp * nq * ne);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), ncomp, nq, ne);
      mfem::DenseMatrix Q_ip;
      for (int e = 0; e < ne; ++e)
      {
         auto &T = use_bdr ? *mesh.GetBdrElementTransformation(e) :
                   *mesh.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            MQ->Eval(Q_ip, T, ir.IntPoint(q));
            for (int j = 0; j < vdim; ++j)
            {
               for (int i = j; i < vdim; ++i)
               {
                  const int idx = (j * vdim) - (((j - 1) * j) / 2) + i - j;
                  C(idx, q, e) = Q_ip(i, j);  // Column-major
               }
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
                          `CeedOperator`.  */
inline void InitCoefficientWithIndices(mfem::Coefficient *Q,
                                       mfem::Mesh &mesh,
                                       const mfem::IntegrationRule &ir,
                                       bool use_bdr,
                                       int nelem,
                                       const int *indices,
                                       Coefficient *&coeff_ptr)
{
   if (Q == nullptr || dynamic_cast<mfem::ConstantCoefficient *>(Q))
   {
      // The constant coefficient case is handled by the QFunction context
      coeff_ptr = nullptr;
   }
   else if (GridFunctionCoefficient *gf_coeff =
               dynamic_cast<GridFunctionCoefficient *>(Q))
   {
      GridCoefficient *ceedCoeff =
         new GridCoefficient(*gf_coeff->GetGridFunction());
      coeff_ptr = ceedCoeff;
   }
   else if (QuadratureFunctionCoefficient *qf_coeff =
               dynamic_cast<QuadratureFunctionCoefficient *>(Q))
   {
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(1);
      ceedCoeff->coeff.SetSize(nq * nelem);
      const mfem::QuadratureFunction &qfunc = qf_coeff->GetQuadFunction();
      MFEM_VERIFY(qfunc.Size() == nq * ne,
                  "Incompatible QuadratureFunction dimension\n");
      MFEM_VERIFY(&ir == &qfunc.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      Memory<int> m_indices((int*)indices, nelem, false);
      auto in = Reshape(qfunc.Read(), nq, ne);
      auto d_indices = Read(m_indices, nelem);
      auto out = Reshape(ceedCoeff->coeff.Write(), nq, nelem);
      mfem::forall(nelem * nq, [=] MFEM_HOST_DEVICE (int i)
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
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(1);
      ceedCoeff->coeff.SetSize(nq * nelem);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), nq, nelem);
      for (int i = 0; i < nelem; ++i)
      {
         const int e = indices[i];
         auto &T = use_bdr ? *mesh.GetBdrElementTransformation(e) :
                   *mesh.GetElementTransformation(e);
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
                          `CeedOperator`. */
inline void InitCoefficientWithIndices(mfem::VectorCoefficient *VQ,
                                       mfem::Mesh &mesh,
                                       const mfem::IntegrationRule &ir,
                                       bool use_bdr,
                                       int nelem, const int *indices,
                                       Coefficient *&coeff_ptr)
{
   if (VQ == nullptr || dynamic_cast<mfem::VectorConstantCoefficient *>(VQ))
   {
      // The constant coefficient case is handled by the QFunction context
      coeff_ptr = nullptr;
   }
   else if (VectorGridFunctionCoefficient *vgf_coeff =
               dynamic_cast<VectorGridFunctionCoefficient *>(VQ))
   {
      GridCoefficient *ceedCoeff =
         new GridCoefficient(*vgf_coeff->GetGridFunction());
      coeff_ptr = ceedCoeff;
   }
   else if (VectorQuadratureFunctionCoefficient *vqf_coeff =
               dynamic_cast<VectorQuadratureFunctionCoefficient *>(VQ))
   {
      const int vdim = vqf_coeff->GetVDim();
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(vdim);
      ceedCoeff->coeff.SetSize(vdim * nq * nelem);
      const mfem::QuadratureFunction &qfunc = vqf_coeff->GetQuadFunction();
      MFEM_VERIFY(qfunc.Size() == vdim * nq * ne,
                  "Incompatible QuadratureFunction dimension\n");
      MFEM_VERIFY(&ir == &qfunc.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      Memory<int> m_indices((int*)indices, nelem, false);
      auto in = Reshape(qfunc.Read(), vdim, nq, ne);
      auto d_indices = Read(m_indices, nelem);
      auto out = Reshape(ceedCoeff->coeff.Write(), vdim, nq, nelem);
      mfem::forall(nelem * nq, [=] MFEM_HOST_DEVICE (int i)
      {
         const int q = i%nq;
         const int sub_e = i/nq;
         const int e = d_indices[sub_e];
         for (int d = 0; d < vdim; d++)
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
      const int vdim = VQ->GetVDim();
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(vdim);
      ceedCoeff->coeff.SetSize(vdim * nq * nelem);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), vdim, nq, nelem);
      mfem::DenseMatrix Q_ir;
      for (int i = 0; i < nelem; ++i)
      {
         const int e = indices[i];
         auto &T = use_bdr ? *mesh.GetBdrElementTransformation(e) :
                   *mesh.GetElementTransformation(e);
         VQ->Eval(Q_ir, T, ir);
         for (int q = 0; q < nq; ++q)
         {
            for (int d = 0; d < vdim; ++d)
            {
               C(d, q, i) = Q_ir(d, q);
            }
         }
      }
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
}

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::MatrixCoefficient @a Q, an mfem::Mesh @a mesh, and an
    mfem::IntegrationRule @a ir for the elements given by the indices @a indices.

    @param[in] MQ is the matrix coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] nelem The number of elements.
    @param[in] indices The indices of the elements of same type in the
                       `FiniteElementSpace`.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficientWithIndices(mfem::MatrixCoefficient *MQ,
                                       mfem::Mesh &mesh,
                                       const mfem::IntegrationRule &ir,
                                       bool use_bdr,
                                       int nelem, const int *indices,
                                       Coefficient *&coeff_ptr)
{
   if (MQ == nullptr || dynamic_cast<mfem::MatrixConstantCoefficient *>(MQ))
   {
      // The constant coefficient case is handled by the QFunction context
      coeff_ptr = nullptr;
   }
   else
   {
      // Assumes matrix coefficient is symmetric
      const int vdim = MQ->GetVDim();
      const int ncomp = vdim * (vdim + 1) / 2;
      const int nq = ir.GetNPoints();
      QuadCoefficient *ceedCoeff = new QuadCoefficient(ncomp);
      ceedCoeff->coeff.SetSize(ncomp * nq * nelem);
      auto C = Reshape(ceedCoeff->coeff.HostWrite(), ncomp, nq, nelem);
      mfem::DenseMatrix Q_ip;
      for (int i = 0; i < nelem; ++i)
      {
         const int e = indices[i];
         auto &T = use_bdr ? *mesh.GetBdrElementTransformation(e) :
                   *mesh.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            MQ->Eval(Q_ip, T, ir.IntPoint(q));
            for (int dj = 0; dj < vdim; ++dj)
            {
               for (int di = dj; di < vdim; ++di)
               {
                  const int idx = (dj * vdim) - (((dj - 1) * dj) / 2) + di - dj;
                  C(idx, q, e) = Q_ip(di, dj);  // Column-major
               }
            }
         }
      }
      InitVector(ceedCoeff->coeff, ceedCoeff->coeffVector);
      coeff_ptr = ceedCoeff;
   }
}

template <typename CoeffType>
inline void InitCoefficient(CoeffType *Q, mfem::Mesh &mesh,
                            const mfem::IntegrationRule &ir,
                            bool use_bdr,
                            int nelem,
                            const int *indices,
                            Coefficient *&coeff_ptr)
{
   if (indices)
   {
      InitCoefficientWithIndices(Q, mesh, ir, use_bdr, nelem, indices, coeff_ptr);
   }
   else
   {
      InitCoefficient(Q, mesh, ir, use_bdr, coeff_ptr);
   }
}

} // namespace ceed

} // namespace mfem

#endif

#endif // MFEM_LIBCEED_COEFF
