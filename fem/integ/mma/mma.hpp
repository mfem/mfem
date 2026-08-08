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

/** @file mma.hpp
    MMA partial-assembly backends and public entry flags.

    ## When MMA runs
    - Programmatic: ForceMMA(true) / MMAForce RAII
    - Env: MFEM_USE_MMA set and not "0" (see GetForceMMA)
    - UsesSimplexMMA: fixed-order H1/H1Pos tri/tet; Positive only if ForceMMA
    - UsesTensorMMA: ForceMMA + H1 GLL quad/hex, double, p >= 3

    ## Host apply tree
    - Tensor: PreferTensorDense → dense sum-fact vs Emulate shell
      (diffusion 2D may use lapack fat GEMM when LAPACK is on)
    - Simplex host: PreferMultiRhs(nq, ndof, NE) → lapack multi-RHS (size gate)
      else → dense / form simplex host path

    ## Device apply tree
    - TensorMmaEnabled → dmma (CUDA) / mfma (HIP); else blas Emulate

    ## Package map (fem/integ/mma/)
    - mma.hpp / mma.cpp   ForceMMA / Uses* / simplex helpers (this file)
    - mode/               backends: common, dispatch, dmma, mfma, blas, lapack, batch
    - form/               integrator-agnostic Apply engines
    - mass.hpp, …         operator drivers (QFn + Kernel registration)

    Entry points internal::Mma*Apply* are intentionally outside namespace mma.

    ## Adding a specialization:
    Edit shared tables in form/register.hpp (Mass/Diffusion simplex + tensors),
    or DomainLF's RegisterSimplexMmaKernels() in domain_lf.cpp.
    Order: DIM, then D1D, then QND/Q1D.
    Unregistered sizes use Fallback (runtime shell).
    See mode/README.md and form/README.md.
*/

// Backends + dispatch (common via dmma/mfma/blas)
#include "mode/dispatch.hpp"
#include "mode/lapack.hpp"

// Public Uses* / ForceMMA + simplex helpers
#include "../../fespace.hpp"   // FiniteElementSpace, ElementDofOrdering (pulls mesh)
#include "../../fe/fe_h1.hpp"
#include "../../fe/fe_pos.hpp"
#include "../../gridfunc.hpp"  // GetSimplexMeshNodesE

namespace mfem
{

/** @brief Prefer MMA PA when an MMA path exists for the space.
    Enables opt-in tensor MMA and Bernstein simplex MMA
    instead of their default SUM / Stroud paths.
    Also enabled when MFEM_USE_MMA is set to any value other than "0".
    @return Previous programmatic force flag (not including env). */
bool ForceMMA(bool enable = true);
bool GetForceMMA();

/** @brief RAII: ForceMMA(enable) for this scope, then restore the previous flag. */
class MMAForce
{
   const bool previous;
public:
   explicit MMAForce(bool enable) : previous(ForceMMA(enable)) { }
   ~MMAForce() { ForceMMA(previous); }
   MMAForce(const MMAForce &) = delete;
   MMAForce &operator=(const MMAForce &) = delete;
};

/// \cond DO_NOT_DOCUMENT

/** True if the typical FE is a nodal H1 or Positive H1 triangle/tet of the
    given mesh dimension (used by simplex MMA assemble asserts). */
inline bool IsSimplexMmaH1Element(const FiniteElement &el, int dim)
{
   if (dim == 2)
   {
      return dynamic_cast<const H1_TriangleElement *>(&el) ||
             dynamic_cast<const H1Pos_TriangleElement *>(&el);
   }
   return dynamic_cast<const H1_TetrahedronElement *>(&el) ||
          dynamic_cast<const H1Pos_TetrahedronElement *>(&el);
}

/** True if this space can use dense simplex PA (as opposed to Stroud
    sum-factorization on Positive bases).
    Backends: CUDA DMMA, HIP MFMA, or dense host GEMM/hand kernels on CPU.
    - GLL (`H1_*`): always eligible (CUDA / HIP / CPU).
    - Positive (`H1Pos_*`): only when forced via ForceMMA / MFEM_USE_MMA. */
inline bool UsesSimplexMMA(const FiniteElementSpace &fes)
{
   if (fes.IsVariableOrder()) { return false; }

   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   if (dim != 2 && dim != 3) { return false; }
   if (mesh->SpaceDimension() != dim) { return false; }
   if (mesh->GetNumGeometries(dim) != 1) { return false; }

   const FiniteElement &el = *fes.GetTypicalFE();
   if (dim == 2)
   {
      if (el.GetGeomType() != Geometry::TRIANGLE) { return false; }
   }
   else
   {
      if (el.GetGeomType() != Geometry::TETRAHEDRON) { return false; }
   }
   if (!IsSimplexMmaH1Element(el, dim)) { return false; }

   const bool positive =
      dynamic_cast<const H1Pos_TriangleElement *>(&el) ||
      dynamic_cast<const H1Pos_TetrahedronElement *>(&el);
   if (positive && !GetForceMMA()) { return false; }
   return true;
}

inline bool IsTensorsMmaH1Element(const FiniteElement &el, int dim)
{
   if (dim == 2)
   {
      return dynamic_cast<const H1_QuadrilateralElement *>(&el) != nullptr;
   }
   return dynamic_cast<const H1_HexahedronElement *>(&el) != nullptr;
}

/** Opt-in sum-factored tensor MMA for fixed-order H1 GLL quad/hex.
    GPU: MMA smem shell (Interp/Grad + dmma/mfma when TensorMmaEnabled, else
    fine-grained blas::Sumf / blas::GemmMbyK).
    CPU: 1D LAPACK GEMM when profitable (mass), else same MMA shell + dense blas_*.
    Unregistered (D1D,Q1D) Fallback is the runtime MMA shell.
    Requires ForceMMA / MFEM_USE_MMA; double precision only; p >= 3. */
inline bool UsesTensorMMA(const FiniteElementSpace &fes)
{
   if (!GetForceMMA()) { return false; }
   if (fes.IsVariableOrder()) { return false; }
#if defined(MFEM_USE_SINGLE)
   return false;
#else
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   if (dim != 2 && dim != 3) { return false; }
   if (mesh->SpaceDimension() != dim) { return false; }
   if (mesh->GetNumGeometries(dim) != 1) { return false; }
   const FiniteElement &el = *fes.GetTypicalFE();
   if (dim == 2)
   {
      if (el.GetGeomType() != Geometry::SQUARE) { return false; }
   }
   else
   {
      if (el.GetGeomType() != Geometry::CUBE) { return false; }
   }
   if (!IsTensorsMmaH1Element(el, dim)) { return false; }
   // m8n8k4 pad waste dominates at p=2 (D,Q)=(3,4); use stock SUM there.
   // Fragment math needs D1D >= 3; require p >= 3 for MMA competitiveness.
   if (el.GetOrder() < 3) { return false; }
   return true;
#endif
}

} // namespace mfem

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal
{

/** Restrict mesh nodes to a NATIVE E-vector: layout (ndof x sdim x NE). */
inline void GetSimplexMeshNodesE(Mesh &mesh, MemoryType mt, Vector &nodes_e,
                                 int &nd_n, int &sdim)
{
   mesh.EnsureNodes();
   const GridFunction *nodes = mesh.GetNodes();
   MFEM_VERIFY(nodes, "Mesh has no nodes");
   const FiniteElementSpace *nfes = nodes->FESpace();
   sdim = nfes->GetVDim();
   nd_n = nfes->GetTypicalFE()->GetDof();
   const Operator *nR =
      nfes->GetElementRestriction(ElementDofOrdering::NATIVE);
   MFEM_VERIFY(nR, "Missing mesh ElementRestriction");
   nodes_e.SetSize(nR->Height(), mt);
   nodes_e.UseDevice(true);
   nR->Mult(*nodes, nodes_e);
}

/** Build 2D Jacobian at (q,e) from mesh nodes E and GradP slice G. */
template <typename EAcc, typename GAcc>
MFEM_HOST_DEVICE inline void EvalSimplexJ2(EAcc E, GAcc G, const int q,
                                           const int e, const int ND,
                                           real_t &J11, real_t &J21,
                                           real_t &J12, real_t &J22)
{
   J11 = J21 = J12 = J22 = 0.0;
   for (int i = 0; i < ND; i++)
   {
      const real_t x = E(i, 0, e), y = E(i, 1, e);
      const real_t gx = G(q, 0, i), gy = G(q, 1, i);
      J11 += x * gx; J21 += y * gx;
      J12 += x * gy; J22 += y * gy;
   }
}

/** Build 3D Jacobian at (q,e) from mesh nodes E and GradP slice G. */
template <typename EAcc, typename GAcc>
MFEM_HOST_DEVICE inline void EvalSimplexJ3(EAcc E, GAcc G, const int q,
                                           const int e, const int ND,
                                           real_t &J11, real_t &J21, real_t &J31,
                                           real_t &J12, real_t &J22, real_t &J32,
                                           real_t &J13, real_t &J23, real_t &J33)
{
   J11 = J21 = J31 = J12 = J22 = J32 = J13 = J23 = J33 = 0.0;
   for (int i = 0; i < ND; i++)
   {
      const real_t x = E(i, 0, e), y = E(i, 1, e), z = E(i, 2, e);
      const real_t gx = G(q, 0, i), gy = G(q, 1, i), gz = G(q, 2, i);
      J11 += x * gx; J21 += y * gx; J31 += z * gx;
      J12 += x * gy; J22 += y * gy; J32 += z * gy;
      J13 += x * gz; J23 += y * gz; J33 += z * gz;
   }
}

void PADetJSetupSimplexFromNodes(const int dim,
                                 const int NE,
                                 const int NQ,
                                 const int ND,
                                 const bool by_val,
                                 const Array<real_t> &w,
                                 const Array<real_t> &g,
                                 const Vector &nodes_e,
                                 const Vector &c,
                                 Vector &d);

} // namespace mfem::internal

/// \endcond DO_NOT_DOCUMENT

