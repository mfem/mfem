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
   const int ncomp;
   const CeedEvalMode emode;
   CeedVector coeff_vector;
   Coefficient(int ncomp, CeedEvalMode emode)
      : ncomp(ncomp), emode(emode), coeff_vector(nullptr) {}
   virtual ~Coefficient()
   {
      CeedVectorDestroy(&coeff_vector);
   }
};

struct GridCoefficient : Coefficient
{
   const mfem::GridFunction &gf;
   CeedBasis basis;
   CeedElemRestriction restr;
   GridCoefficient(const mfem::GridFunction &gf, Ceed ceed)
      : Coefficient(gf.VectorDim(), CEED_EVAL_INTERP), gf(gf),
        basis(nullptr), restr(nullptr)
   {
      InitVector(gf, ceed, coeff_vector);
   }
};

struct QuadCoefficient : Coefficient
{
   mfem::Vector vector;
   CeedElemRestriction restr;
   QuadCoefficient(int ncomp)
      : Coefficient(ncomp, CEED_EVAL_NONE), restr(nullptr) {}
};

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::Coefficient @a Q, an mfem::Mesh @a mesh, and an mfem::IntegrationRule
    @a ir.

    @param[in] Q is the coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] use_bdr is a flag to construct the coefficient on mesh boundaries.
    @param[in] ceed The Ceed object.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficient(mfem::Coefficient &Q,
                            mfem::Mesh &mesh,
                            const mfem::IntegrationRule &ir,
                            bool use_bdr,
                            Ceed ceed,
                            Coefficient *&coeff_ptr)
{
   if (mfem::GridFunctionCoefficient *gf_coeff =
          dynamic_cast<mfem::GridFunctionCoefficient *>(&Q))
   {
      auto *ceed_coeff = new GridCoefficient(*gf_coeff->GetGridFunction(), ceed);
      coeff_ptr = ceed_coeff;
   }
   else if (mfem::QuadratureFunctionCoefficient *qf_coeff =
               dynamic_cast<mfem::QuadratureFunctionCoefficient *>(&Q))
   {
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      auto *ceed_coeff = new QuadCoefficient(1);
      const mfem::QuadratureFunction &qfunc = qf_coeff->GetQuadFunction();
      MFEM_VERIFY(qfunc.Size() == nq * ne,
                  "Incompatible QuadratureFunction dimension.");
      MFEM_VERIFY(&ir == &qfunc.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.");
      qfunc.Read();
      ceed_coeff->vector.MakeRef(const_cast<mfem::QuadratureFunction &>(qfunc), 0);
      InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
      coeff_ptr = ceed_coeff;
   }
   else
   {
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      auto *ceed_coeff = new QuadCoefficient(1);
      ceed_coeff->vector.SetSize(nq * ne);
      auto C = Reshape(ceed_coeff->vector.HostWrite(), nq, ne);
      mfem::IsoparametricTransformation T;
      for (int e = 0; e < ne; ++e)
      {
         if (use_bdr)
         {
            mesh.GetBdrElementTransformation(e, &T);
         }
         else
         {
            mesh.GetElementTransformation(e, &T);
         }
         for (int q = 0; q < nq; ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T.SetIntPoint(&ip);
            C(q, e) = Q.Eval(T, ip);
         }
      }
      InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
      coeff_ptr = ceed_coeff;
   }
}

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::VectorCoefficient @a VQ, an mfem::Mesh @a mesh, and an
    mfem::IntegrationRule @a ir.

    @param[in] VQ is the vector coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] use_bdr is a flag to construct the coefficient on mesh boundaries.
    @param[in] ceed The Ceed object.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficient(mfem::VectorCoefficient &VQ,
                            mfem::Mesh &mesh,
                            const mfem::IntegrationRule &ir,
                            bool use_bdr,
                            Ceed ceed,
                            Coefficient *&coeff_ptr)
{
   if (mfem::VectorGridFunctionCoefficient *vgf_coeff =
          dynamic_cast<mfem::VectorGridFunctionCoefficient *>(&VQ))
   {
      auto *ceed_coeff = new GridCoefficient(*vgf_coeff->GetGridFunction(), ceed);
      coeff_ptr = ceed_coeff;
   }
   else if (mfem::VectorQuadratureFunctionCoefficient *vqf_coeff =
               dynamic_cast<mfem::VectorQuadratureFunctionCoefficient *>(&VQ))
   {
      const int vdim = vqf_coeff->GetVDim();
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      auto *ceed_coeff = new QuadCoefficient(vdim);
      const mfem::QuadratureFunction &qfunc = vqf_coeff->GetQuadFunction();
      MFEM_VERIFY(qfunc.Size() == vdim * nq * ne,
                  "Incompatible QuadratureFunction dimension.");
      MFEM_VERIFY(&ir == &qfunc.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.");
      qfunc.Read();
      ceed_coeff->vector.MakeRef(const_cast<mfem::QuadratureFunction &>(qfunc), 0);
      InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
      coeff_ptr = ceed_coeff;
   }
   else
   {
      const int vdim = VQ.GetVDim();
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      auto *ceed_coeff = new QuadCoefficient(vdim);
      ceed_coeff->vector.SetSize(vdim * nq * ne);
      auto C = Reshape(ceed_coeff->vector.HostWrite(), vdim, nq, ne);
      mfem::IsoparametricTransformation T;
      mfem::DenseMatrix Q_ip;
      for (int e = 0; e < ne; ++e)
      {
         if (use_bdr)
         {
            mesh.GetBdrElementTransformation(e, &T);
         }
         else
         {
            mesh.GetElementTransformation(e, &T);
         }
         VQ.Eval(Q_ip, T, ir);
         for (int q = 0; q < nq; ++q)
         {
            for (int i = 0; i < vdim; ++i)
            {
               C(i, q, e) = Q_ip(i, q);
            }
         }
      }
      InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
      coeff_ptr = ceed_coeff;
   }
}

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::MatrixCoefficient @a MQ, an mfem::Mesh @a mesh, and an
    mfem::IntegrationRule @a ir.

    @param[in] MQ is the matrix coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] use_bdr is a flag to construct the coefficient on mesh boundaries.
    @param[in] ceed The Ceed object.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficient(mfem::MatrixCoefficient &MQ,
                            mfem::Mesh &mesh,
                            const mfem::IntegrationRule &ir,
                            bool use_bdr,
                            Ceed ceed,
                            Coefficient *&coeff_ptr)
{
   // Assumes matrix coefficient is symmetric
   const int vdim = MQ.GetVDim();
   const int ncomp = (vdim * (vdim + 1)) / 2;
   const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
   const int nq = ir.GetNPoints();
   auto *ceed_coeff = new QuadCoefficient(ncomp);
   ceed_coeff->vector.SetSize(ncomp * nq * ne);
   auto C = Reshape(ceed_coeff->vector.HostWrite(), ncomp, nq, ne);
   mfem::IsoparametricTransformation T;
   mfem::DenseMatrix Q_ip;
   for (int e = 0; e < ne; ++e)
   {
      if (use_bdr)
      {
         mesh.GetBdrElementTransformation(e, &T);
      }
      else
      {
         mesh.GetElementTransformation(e, &T);
      }
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         MQ.Eval(Q_ip, T, ip);
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
   InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
   coeff_ptr = ceed_coeff;
}

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::Coefficient @a Q, an mfem::Mesh @a mesh, and an mfem::IntegrationRule
    @a ir for the elements given by the indices @a indices.

    @param[in] Q is the coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] use_bdr is a flag to construct the coefficient on mesh boundaries.
    @param[in] nelem is the number of elements.
    @param[in] indices are the indices of the elements of same type in the
                       `FiniteElementSpace`.
    @param[in] ceed The Ceed object.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficientWithIndices(mfem::Coefficient &Q,
                                       mfem::Mesh &mesh,
                                       const mfem::IntegrationRule &ir,
                                       bool use_bdr,
                                       int nelem,
                                       const int *indices,
                                       Ceed ceed,
                                       Coefficient *&coeff_ptr)
{
   if (mfem::GridFunctionCoefficient *gf_coeff =
          dynamic_cast<mfem::GridFunctionCoefficient *>(&Q))
   {
      auto *ceed_coeff = new GridCoefficient(*gf_coeff->GetGridFunction(), ceed);
      coeff_ptr = ceed_coeff;
   }
   else if (mfem::QuadratureFunctionCoefficient *qf_coeff =
               dynamic_cast<mfem::QuadratureFunctionCoefficient *>(&Q))
   {
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      auto *ceed_coeff = new QuadCoefficient(1);
      ceed_coeff->vector.SetSize(nq * nelem);
      const mfem::QuadratureFunction &qfunc = qf_coeff->GetQuadFunction();
      MFEM_VERIFY(qfunc.Size() == nq * ne,
                  "Incompatible QuadratureFunction dimension.");
      MFEM_VERIFY(&ir == &qfunc.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.");
      Memory<int> m_indices((int*)indices, nelem, false);
      auto in = Reshape(qfunc.Read(), nq, ne);
      auto d_indices = Read(m_indices, nelem);
      auto out = Reshape(ceed_coeff->vector.Write(), nq, nelem);
      mfem::forall(nelem * nq, [=] MFEM_HOST_DEVICE (int i)
      {
         const int q = i%nq;
         const int sub_e = i/nq;
         const int e = d_indices[sub_e];
         out(q, sub_e) = in(q, e);
      });
      m_indices.DeleteDevice();
      InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
      coeff_ptr = ceed_coeff;
   }
   else
   {
      const int nq = ir.GetNPoints();
      auto *ceed_coeff = new QuadCoefficient(1);
      ceed_coeff->vector.SetSize(nq * nelem);
      auto C = Reshape(ceed_coeff->vector.HostWrite(), nq, nelem);
      mfem::IsoparametricTransformation T;
      for (int i = 0; i < nelem; ++i)
      {
         const int e = indices[i];
         if (use_bdr)
         {
            mesh.GetBdrElementTransformation(e, &T);
         }
         else
         {
            mesh.GetElementTransformation(e, &T);
         }
         for (int q = 0; q < nq; ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T.SetIntPoint(&ip);
            C(q, i) = Q.Eval(T, ip);
         }
      }
      InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
      coeff_ptr = ceed_coeff;
   }
}

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::VectorCoefficient @a VQ, an mfem::Mesh @a mesh, and an
    mfem::IntegrationRule @a ir for the elements given by the indices @a indices.

    @param[in] VQ is the vector coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] use_bdr is a flag to construct the coefficient on mesh boundaries.
    @param[in] nelem is the number of elements.
    @param[in] indices are the indices of the elements of same type in the
                       `FiniteElementSpace`.
    @param[in] ceed The Ceed object.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficientWithIndices(mfem::VectorCoefficient &VQ,
                                       mfem::Mesh &mesh,
                                       const mfem::IntegrationRule &ir,
                                       bool use_bdr,
                                       int nelem,
                                       const int *indices,
                                       Ceed ceed,
                                       Coefficient *&coeff_ptr)
{
   if (mfem::VectorGridFunctionCoefficient *vgf_coeff =
          dynamic_cast<mfem::VectorGridFunctionCoefficient *>(&VQ))
   {
      auto *ceed_coeff = new GridCoefficient(*vgf_coeff->GetGridFunction(), ceed);
      coeff_ptr = ceed_coeff;
   }
   else if (mfem::VectorQuadratureFunctionCoefficient *vqf_coeff =
               dynamic_cast<mfem::VectorQuadratureFunctionCoefficient *>(&VQ))
   {
      const int vdim = vqf_coeff->GetVDim();
      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      const int nq = ir.GetNPoints();
      auto *ceed_coeff = new QuadCoefficient(vdim);
      ceed_coeff->vector.SetSize(vdim * nq * nelem);
      const mfem::QuadratureFunction &qfunc = vqf_coeff->GetQuadFunction();
      MFEM_VERIFY(qfunc.Size() == vdim * nq * ne,
                  "Incompatible QuadratureFunction dimension.");
      MFEM_VERIFY(&ir == &qfunc.GetSpace()->GetIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different.");
      Memory<int> m_indices((int*)indices, nelem, false);
      auto in = Reshape(qfunc.Read(), vdim, nq, ne);
      auto d_indices = Read(m_indices, nelem);
      auto out = Reshape(ceed_coeff->vector.Write(), vdim, nq, nelem);
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
      InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
      coeff_ptr = ceed_coeff;
   }
   else
   {
      const int vdim = VQ.GetVDim();
      const int nq = ir.GetNPoints();
      auto *ceed_coeff = new QuadCoefficient(vdim);
      ceed_coeff->vector.SetSize(vdim * nq * nelem);
      auto C = Reshape(ceed_coeff->vector.HostWrite(), vdim, nq, nelem);
      mfem::IsoparametricTransformation T;
      mfem::DenseMatrix Q_ip;
      for (int i = 0; i < nelem; ++i)
      {
         const int e = indices[i];
         if (use_bdr)
         {
            mesh.GetBdrElementTransformation(e, &T);
         }
         else
         {
            mesh.GetElementTransformation(e, &T);
         }
         VQ.Eval(Q_ip, T, ir);
         for (int q = 0; q < nq; ++q)
         {
            for (int d = 0; d < vdim; ++d)
            {
               C(d, q, i) = Q_ip(d, q);
            }
         }
      }
      InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
      coeff_ptr = ceed_coeff;
   }
}

/** @brief Initializes an mfem::ceed::Coefficient @a coeff_ptr from an
    mfem::MatrixCoefficient @a MQ, an mfem::Mesh @a mesh, and an
    mfem::IntegrationRule @a ir for the elements given by the indices @a indices.

    @param[in] MQ is the matrix coefficient from the `Integrator`.
    @param[in] mesh is the mesh.
    @param[in] ir is the integration rule.
    @param[in] use_bdr is a flag to construct the coefficient on mesh boundaries.
    @param[in] nelem is the number of elements.
    @param[in] indices are the indices of the elements of same type in the
                       `FiniteElementSpace`.
    @param[in] ceed The Ceed object.
    @param[out] coeff_ptr is the structure to store the coefficient for the
                          `CeedOperator`. */
inline void InitCoefficientWithIndices(mfem::MatrixCoefficient &MQ,
                                       mfem::Mesh &mesh,
                                       const mfem::IntegrationRule &ir,
                                       bool use_bdr,
                                       int nelem,
                                       const int *indices,
                                       Ceed ceed,
                                       Coefficient *&coeff_ptr)
{
   // Assumes matrix coefficient is symmetric
   const int vdim = MQ.GetVDim();
   const int ncomp = (vdim * (vdim + 1)) / 2;
   const int nq = ir.GetNPoints();
   auto *ceed_coeff = new QuadCoefficient(ncomp);
   ceed_coeff->vector.SetSize(ncomp * nq * nelem);
   auto C = Reshape(ceed_coeff->vector.HostWrite(), ncomp, nq, nelem);
   mfem::IsoparametricTransformation T;
   mfem::DenseMatrix Q_ip;
   for (int i = 0; i < nelem; ++i)
   {
      const int e = indices[i];
      if (use_bdr)
      {
         mesh.GetBdrElementTransformation(e, &T);
      }
      else
      {
         mesh.GetElementTransformation(e, &T);
      }
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         MQ.Eval(Q_ip, T, ip);
         for (int dj = 0; dj < vdim; ++dj)
         {
            for (int di = dj; di < vdim; ++di)
            {
               const int idx = (dj * vdim) - (((dj - 1) * dj) / 2) + di - dj;
               C(idx, q, i) = Q_ip(di, dj);  // Column-major
            }
         }
      }
   }
   InitVector(ceed_coeff->vector, ceed, ceed_coeff->coeff_vector);
   coeff_ptr = ceed_coeff;
}

template <typename CoeffType>
inline void InitCoefficient(CoeffType &Q,
                            mfem::Mesh &mesh,
                            const mfem::IntegrationRule &ir,
                            bool use_bdr,
                            int nelem,
                            const int *indices,
                            Ceed ceed,
                            Coefficient *&coeff_ptr)
{
   if (indices)
   {
      InitCoefficientWithIndices(Q, mesh, ir, use_bdr, nelem, indices,
                                 ceed, coeff_ptr);
   }
   else
   {
      InitCoefficient(Q, mesh, ir, use_bdr, ceed, coeff_ptr);
   }
}

} // namespace ceed

} // namespace mfem

#endif

#endif // MFEM_LIBCEED_COEFF
