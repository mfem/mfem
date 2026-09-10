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

#include "darcyhybridization.hpp"
#include "bilininteg_hdg.hpp"
#include "../../linalg/batched/batched.hpp"
#include "../../general/forall.hpp"

#include <algorithm>
#include <cstring>
#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

#include "../../mesh/segment.hpp"
#include "../../mesh/triangle.hpp"
#include "../../mesh/quadrilateral.hpp"

namespace mfem
{

DarcyHybridization::DarcyHybridization(FiniteElementSpace *fes_u_,
                                       FiniteElementSpace *fes_p_,
                                       FiniteElementSpace *fes_c_,
                                       bool bsymmetrize)
   : Hybridization(fes_u_, fes_c_),
     fes_p(*fes_p_), bsym(bsymmetrize)
{
#ifdef MFEM_USE_MPI
   pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
   pfes_p = dynamic_cast<ParFiniteElementSpace*>(&fes_p);
   c_pfes = dynamic_cast<ParFiniteElementSpace*>(&c_fes);
#endif
   SetLocalNLSolver(LSsolveType::LBFGS);
   SetLocalNLPreconditioner(LPrecType::GMRES);
}

/** @brief The connectivity TraceAssemblyMode::Batched assembles against.

    Cached behind a pointer so that adding to it costs no rebuild anywhere;
    see the declaration. Every array here is a function of the mesh and the
    trace space, and BuildTraceHMap() carries the argument for why the
    PATTERN can be too. */
/** @brief The element-blocked face blocks LinearResidualBatched() multiplies,
    gathered ONCE rather than per residual evaluation, plus that routine's
    per-call workspace.

    **TWO costs sank this route, each about the size of the element loop's
    entire work, and removing either one alone left it a regression.** That is
    the finding, and it corrects an earlier note here that named only the
    first.

    The gather. Ct, E, G and H live at per-face offsets whose stride is NOT
    uniform -- a boundary face's Ct block is (na, nc) where an interior face's
    is (2na, nc) -- so a BatchedLinAlg tensor over them needs a gather, and
    the element-blocked (na, T) form needs one regardless because an element's
    faces are not adjacent in those arrays. Per call that is
    na*T*NE + 2*nd*T*NE + T*T*NE doubles moved, about 40 MB at order 3.

    The workspace. Eight Vectors of na*NE / nd*NE / T*NE, constructed and
    destroyed per evaluation -- 4.7 MB of new/delete at order 3 on 48x48
    quads. They are held here now and only SetSize()d.

    Measured on a linear NPC residual, 48x48, 20 evaluations, against the
    element loop (seconds; the ratio is the element loop over this route):

        order      cached+held   gather per call   element loop
        1 quad       0.0123          0.0633           0.1170     9.5x / 1.8x
        1 tri        0.0191          0.1183           0.1860     9.7x / 1.6x
        2 quad       0.0662          0.2608           0.1727     2.6x / 0.66x
        2 tri        0.0695          0.2865           0.2590     3.7x / 0.90x
        3 quad       0.1885          0.6883           0.2952     1.6x / 0.43x
        3 tri        0.1639          0.5432           0.3729     2.3x / 0.69x

    The middle column is this cache disabled with the workspace still held:
    a regression from order 2 up, which is what says the gather is real. The
    earlier numbers, 0.62x and 0.42x, were taken with BOTH costs present and
    were attributed here to the gather alone -- caching it moved them to
    0.60x and 0.45x, i.e. not at all, and only hoisting the allocations
    turned the route around. **When two costs are the same size, fixing one
    and re-measuring looks exactly like fixing nothing.**

    **Valid until Reset(), and that is exact rather than hopeful.** The route
    fires only when every integrator is linear, and in that case E, G and H
    are written by AssemblePotHDGFaces() at assembly time and by nothing
    afterwards -- ConstructGrad()'s writes to them are inside `if (c_nlfi_p)`
    and `if (c_nlfi)`, both of which CanBatchLinearResidual() refuses. Ct is
    built once in ConstructC(). Re-assembly goes through Reset(), which clears
    this. */
struct DarcyHybridization::ResidualCache
{
   int nf{0}, nc{0}, na{0}, nd{0};
   Array<int> face_map;              ///< BuildElementHFaceMap()'s, with H
   Vector Ct_all, E_all, G_all, H_all;

   /** @brief The element matrices of a "nonlinear" form whose integrators are
       all BilinearFormIntegrators, assembled ONCE.

       `BilinearFormIntegrator` derives from `NonlinearFormIntegrator`, and
       its AssembleElementVector() assembles the element matrix and multiplies
       -- "general but not efficient", says its own comment. So a caller that
       puts a VectorMassIntegrator on the flux mass NONLINEAR form, which
       convdiff and anisodiff both do, re-assembles a constant matrix on every
       residual evaluation. Assembled here once instead.

       Kept SEPARATE from Af_lin_data / Df_lin_data deliberately:
       ConstructGrad() and AddMultA()/AddMultDE() handle m_nlfi_u and
       m_nlfi_p on their own branch, so merging would double-count on every
       route that is not this one. Empty when the slot is absent.

       Worth, measured on the same fixture as the table above with the mass
       terms moved to the nonlinear slots -- which is `convdiff -nl`'s shape
       and where the route refused outright before: 14.2x, 13.4x, 4.5x, 5.4x,
       3.9x, 6.0x over orders 1-3 on quads and triangles. Larger than the
       linear case because the loop it replaces was re-assembling the element
       matrix as well as multiplying by it. */
   Vector Au_all, Dp_all;

   /// Per-call workspace, held here so a residual evaluation allocates
   /// nothing: eight Vectors of na*NE / nd*NE / T*NE, which at order 3 on
   /// 48x48 quads is 4.7 MB of new/delete per call.
   Vector u_all, p_all, bu_all, bp_all, x_all, ru_all, rp_all, y_all;
};

struct DarcyHybridization::TraceHMap
{
   int nf{0};        ///< faces per element, one count for the whole mesh
   int nc{0};        ///< trace vdofs per face, one count for the whole mesh
   int NF{0};        ///< faces in the mesh
   int NE{0};        ///< elements in the mesh
   int ncdofs{0};    ///< rows of H, and c_fes.GetVSize()
   int nnz{0};       ///< nonzeros of the structural pattern

   /// (NF*nc) the global trace vdof of each (face, local dof).
   Array<int> face_dofs;
   /// (NE*nf) the global face of each (element, local face).
   Array<int> el_face;
   /** @brief (ncdofs+1) the CSR row offsets, in row order, and (nnz) the
       column indices: the whole pattern, built once and copied into each
       matrix the mode assembles.

       **Both are built and kept on the HOST, deliberately, and only @a I ever
       reaches a kernel.** The pattern is built once per mesh, so making it a
       kernel would save nothing measurable -- and every consumer of the
       assembled matrix wants it on the host anyway.
       SparseMatrix::SetDiagIdentity() and EliminateRowCol(), both of which
       run on the way out of ComputeH(), index `I[i]`, `J[k]` and `A[k]`
       through Memory::operator[], which is a raw host access that neither
       syncs nor invalidates. So the pattern staying host-valid is what makes
       them correct; it is the VALUES that travel. */
   Array<int> I, J;
   /** @brief (NE*nf*nf) where element @a el's (f1 columns, f2 rows) block
       goes, as `2*slot + side`: @a slot is the position of f1 in f2's @a nb
       list, and @a side is 0 when @a el is the lowest-numbered element
       contributing to that (face, face) pair and 1 when it is the other one.
       Indexed `(el*nf + lf2)*nf + lf1`. */
   Array<int> el_slot;
};

DarcyHybridization::~DarcyHybridization()
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   if (own_m_nlfi_p) { delete m_nlfi_p; }
   if (own_m_nlfi) { delete m_nlfi; }
   if (!extern_bdr_constr_pot_integs)
   {
      for (size_t k=0; k < boundary_constraint_pot_integs.size(); k++)
      { delete boundary_constraint_pot_integs[k]; }
      for (size_t k=0; k < boundary_constraint_pot_nonlin_integs.size(); k++)
      { delete boundary_constraint_pot_nonlin_integs[k]; }
      for (size_t k=0; k < boundary_constraint_nonlin_integs.size(); k++)
      { delete boundary_constraint_nonlin_integs[k]; }
   }
}

/** @brief A LINEAR potential face constraint.

    This may be combined with a NONLINEAR potential mass, and used not to be.
    Both this and SetPotMassNonlinearIntegrator() carried a bare MFEM_VERIFY
    refusing the combination, with no reason given; the reason was that the
    constraint's contribution to D is assembled into Df_data at assembly time
    and ConstructGrad() zeroes Df_data to make room for the nonlinear mass's
    Jacobian, so the contribution was destroyed after the first gradient.
    Finalize() now backs it up into Df_lin_data on the PotNL branch and the
    three D consumers add it as a further term rather than an alternative --
    see the comment on that backup.

    The combination is not exotic. A caller whose potential mass is nonlinear
    must put it on a NonlinearForm, and the HDG face stabilization goes on the
    SAME form; `convdiff` installs the identical HDGDiffusionIntegrator either
    way. DarcyForm::EnableHybridization() now routes such a constraint here,
    which assembles E, G, H and D once instead of once per element per Newton
    evaluation and makes the batched face kernel reachable on a nonlinear
    problem. */
void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, BilinearFormIntegrator *c_pot_integ)
{
   c_bfi.reset(c_flux_integ);
   c_bfi_p.reset(c_pot_integ);
   c_nlfi_p.reset();
   c_nlfi.reset();
}

void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, NonlinearFormIntegrator *c_pot_integ)
{
   c_bfi.reset(c_flux_integ);
   c_bfi_p.reset();
   c_nlfi_p.reset(c_pot_integ);
   c_nlfi.reset();
}

void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, BlockNonlinearFormIntegrator *c_integ)
{
   c_bfi.reset(c_flux_integ);
   c_bfi_p.reset();
   c_nlfi_p.reset();
   c_nlfi.reset(c_integ);
}

void DarcyHybridization::SetFluxMassNonlinearIntegrator(
   NonlinearFormIntegrator *flux_integ, bool own)
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   own_m_nlfi_u = own;
   m_nlfi_u = flux_integ;
}

void DarcyHybridization::SetPotMassNonlinearIntegrator(NonlinearFormIntegrator
                                                       *pot_integ, bool own)
{
   // A linear constraint alongside this is supported; see
   // SetConstraintIntegrators(BilinearFormIntegrator*, BilinearFormIntegrator*).
   if (own_m_nlfi_p) { delete m_nlfi_p; }
   own_m_nlfi_p = own;
   m_nlfi_p = pot_integ;
}

void DarcyHybridization::SetBlockNonlinearIntegrator(
   BlockNonlinearFormIntegrator *block_integ, bool own)
{
   if (own_m_nlfi) { delete m_nlfi; }
   own_m_nlfi = own;
   m_nlfi = block_integ;
}

void DarcyHybridization::Init(const Array<int> &ess_flux_tdof_list)
{
   const int NE = fes.GetNE();

   if (Ct_data.Size()) { return; }

   /* **Force the trace space's face -> dof table, and it is a one-line fix to
      the largest allocation site in an HDG solve.**

      FiniteElementSpace::GetFaceDofs() has a fast path -- `if (face_dof &&
      variant == 0) { face_dof->GetRow(face, dofs); ... }` -- and a slow path
      that declares `Array<int> V, E, Eo;` fresh on every call. Nothing here
      built the table, so every GetFaceVDofs() took the slow path, and in 2D a
      face IS an edge, so Mesh::GetFaceEdges() does `edges.SetSize(1)` on two
      of those empty arrays: TWO four-byte new[]/delete[] per call. This class
      asks per (element, local face) per residual and per gradient, from
      MultNL(), ScatterElementH(), NPCReduce() and NPCRecover().

      Measured with DHAT on `convdiff -p 6 -nl -dg -hb -npc` at 256 elements:
      46,089 blocks of 90,949 -- more than half of every heap block in the run
      -- from that one four-byte array.

      The table is what the slow path would have computed, stored: its builder
      calls GetFaceDofs() itself, so the values including their sign encoding
      are identical by construction. FiniteElementSpace::Destroy() frees it, so
      an Update() or a mesh refinement invalidates it with no help from here.

      **Not for a variable-order space**, and that guard is the reason to read
      the member's own comment: `face_dof` holds VARIANT 0 only, and the slow
      path returns the face's own order from var_face_orders where the fast
      path returns the collection's. This branch has no per-face trace order --
      that is gf-hdg-p-adaptivity's -- but the guard costs nothing and the
      failure would be silent. */
   if (!c_fes.IsVariableOrder()) { c_fes.GetFaceToDofTable(); }

   // count the number of dofs in the discontinuous version of fes:
   Array<int> vdofs;
   int num_hat_dofs = 0;
   hat_offsets.SetSize(NE+1);
   hat_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      fes.GetElementVDofs(i, vdofs);
      num_hat_dofs += vdofs.Size();
      hat_offsets[i+1] = num_hat_dofs;
   }

   // Define the "free" (0) and "essential" (1) hat_dofs.
   // The "essential" hat_dofs are those that depend only on essential cdofs;
   // all other hat_dofs are "free".
   hat_dofs_marker.SetSize(num_hat_dofs);
   Array<int> free_tdof_marker;
   if (ParallelU())
   {
#ifdef MFEM_USE_MPI
      free_tdof_marker.SetSize(pfes->TrueVSize());
#endif
   }
   else
   {
      free_tdof_marker.SetSize(fes.GetConformingVSize());
   }

   free_tdof_marker = 1;
   for (int i = 0; i < ess_flux_tdof_list.Size(); i++)
   {
      free_tdof_marker[ess_flux_tdof_list[i]] = 0;
   }
   Array<int> free_vdofs_marker;
   if (ParallelU())
   {
#ifdef MFEM_USE_MPI
      HypreParMatrix *P = pfes->Dof_TrueDof_Matrix();
      free_vdofs_marker.SetSize(fes.GetVSize());
      P->BooleanMult(1, free_tdof_marker, 0, free_vdofs_marker);
#endif
   }
   else
   {
      const SparseMatrix *cP = fes.GetConformingProlongation();
      if (!cP)
      {
         free_vdofs_marker.MakeRef(free_tdof_marker);
      }
      else
      {
         free_vdofs_marker.SetSize(fes.GetVSize());
         cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
      }
   }

   for (int i = 0; i < NE; i++)
   {
      fes.GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         hat_dofs_marker[hat_offsets[i]+j] = ! free_vdofs_marker[vdofs[j]];
      }
   }

   free_tdof_marker.DeleteAll();
   free_vdofs_marker.DeleteAll();

   // Define Af_offsets and Af_f_offsets
   Af_offsets.SetSize(NE+1);
   Af_offsets[0] = 0;
   Af_f_offsets.SetSize(NE+1);
   Af_f_offsets[0] = 0;

   for (int i = 0; i < NE; i++)
   {
      int f_size = 0; // count the "free" hat_dofs in element i
      for (int j = hat_offsets[i]; j < hat_offsets[i+1]; j++)
      {
         if (hat_dofs_marker[j] != 1) { f_size++; }
      }
      Af_offsets[i+1] = Af_offsets[i] + f_size*f_size;
      Af_f_offsets[i+1] = Af_f_offsets[i] + f_size;
   }

   Af_data.SetSize(Af_offsets[NE]); Af_data = 0.;
   Af_ipiv.SetSize(Af_f_offsets[NE]);

   // Assemble the constraint matrix C
   ConstructC();

   // Define Bf_offsets, Df_offsets and Df_f_offsets
   Bf_offsets.SetSize(NE+1);
   Bf_offsets[0] = 0;
   Df_offsets.SetSize(NE+1);
   Df_offsets[0] = 0;
   Df_f_offsets.SetSize(NE+1);
   Df_f_offsets[0] = 0;
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Ae_offsets.SetSize(NE+1);
   Ae_offsets[0] = 0;
   Be_offsets.SetSize(NE+1);
   Be_offsets[0] = 0;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   for (int i = 0; i < NE; i++)
   {
      int f_size = Af_f_offsets[i+1] - Af_f_offsets[i];
      int d_size = fes_p.GetFE(i)->GetDof() * fes_p.GetVDim();
      Bf_offsets[i+1] = Bf_offsets[i] + f_size*d_size;
      Df_offsets[i+1] = Df_offsets[i] + d_size*d_size;
      Df_f_offsets[i+1] = Df_f_offsets[i] + d_size;
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      int a_size = hat_offsets[i+1] - hat_offsets[i];
      int e_size = a_size - f_size;
      Ae_offsets[i+1] = Ae_offsets[i] + e_size*a_size;
      Be_offsets[i+1] = Be_offsets[i] + e_size*d_size;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   }

   Bf_data.SetSize(Bf_offsets[NE]); Bf_data = 0.;
   // A nonlinear potential mass allocates D lazily, in ReducedGradient(),
   // because ConstructGrad() is what fills it. Anything assembled LINEARLY
   // into D needs it at assembly time instead, so it is allocated here when
   // we can see that coming.
   //
   // This condition is now a FAST PATH, not a correctness requirement, and
   // that distinction was paid for: `c_bfi_p` is visible here while the
   // linear potential mass form is not, so the test reads "a linear face
   // constraint exists" where the question is "linear D content exists".
   // A linear M_p with only a DOMAIN integrator, alongside a nonlinear
   // Mnl_p, satisfies neither half -- and segfaulted in
   // DenseMatrix::operator+= on an unsized Df_data. AssemblePotMassMatrix()
   // now allocates what it writes, so getting this test wrong costs one
   // branch per element and nothing else.
   if (!m_nlfi_p || c_bfi_p)
   {
      AllocD();
   }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Ae_data.SetSize(Ae_offsets[NE]); Ae_data = 0.;
   Be_data.SetSize(Be_offsets[NE]); Be_data = 0.;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   if (c_bfi_p)
   {
      AllocEG();
      if (NPCEnabled())
      {
         AllocH();
      }
   }
}

void DarcyHybridization::SetEssentialBC(const Array<int> &bdr_attr_is_ess)
{
   c_fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
}

void DarcyHybridization::SetEssentialVDofs(const Array<int> &ess_vdofs_list)
{
   if (c_fes.Conforming() && !ParallelC())
   {
      ess_vdofs_list.Copy(ess_tdof_list); // ess_vdofs_list --> ess_tdof_list
   }
   else
   {
      Array<int> ess_vdof_marker, ess_tdof_marker;
      FiniteElementSpace::ListToMarker(ess_vdofs_list, c_fes.GetVSize(),
                                       ess_vdof_marker);
      if (!ParallelC())
      {
         c_fes.ConvertToConformingVDofs(ess_vdof_marker, ess_tdof_marker);
      }
      else
      {
#ifdef MFEM_USE_MPI
         ess_tdof_marker.SetSize(c_pfes->GetTrueVSize());
         c_pfes->Dof_TrueDof_Matrix()->BooleanMultTranspose(1, ess_vdof_marker,
                                                            0, ess_tdof_marker);
#else
         MFEM_ABORT("internal MFEM error");
#endif
      }
      FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdof_list);
   }
}

/** @brief Assemble an element matrix of @a Mu.

    Accumulates, as AssemblePotMassMatrix() and AssembleDivMatrix() do, because
    an element receives its flux mass block in more than one pass: the domain
    integrators first, then any boundary face integrator of the flux mass form
    -- see DarcyForm::AssembleFluxMassBdrFaces(). The storage is therefore
    zeroed at Init() and by Reset(). */
void DarcyHybridization::AssembleFluxMassMatrix(int el, const DenseMatrix &A)
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   // The element matrix has to be (hat dofs) square, and until this guard it
   // was read to that extent WITHOUT being checked -- the MFEM_ASSERT below
   // checks the index total, is debug-only, and cannot see a wrongly SHAPED
   // argument at all. A caller who hands over the wrong shape therefore gets
   // an out-of-bounds READ: no fault where it happens, plausible garbage in
   // Af_data, and a crash or not depending on the heap.
   //
   // Reached, not hypothetical. A bare VectorMassIntegrator on a flux space
   // whose vdim is neq*dim takes its own vdim from the SPACE DIMENSION when
   // the coefficient is scalar, so it produces (nd*dim) square where this
   // wants (nd*neq*dim) -- 2x2 against 4x4 at order 0. It presented as a
   // SIGSEGV in an unrelated test file, under some Catch2 filters and not
   // others, and not under gdb. AddressSanitizer on this translation unit
   // named it in one run; three hours of bisection had not.
   MFEM_VERIFY(A.Height() == s && A.Width() == s,
               "flux mass element matrix is " << A.Height() << "x" << A.Width()
               << ", expected " << s << "x" << s
               << " -- see the note above on VectorMassIntegrator's vdim");
   int Af_el_idx = Af_offsets[el];
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   int Ae_el_idx = Ae_offsets[el];
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   for (int j = 0; j < s; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         for (int i = 0; i < s; i++)
         {
            Ae_data[Ae_el_idx++] += A(i, j);
         }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < s; i++)
      {
         if (hat_dofs_marker[o + i] == 1) { continue; }
         Af_data[Af_el_idx++] += A(i, j);
      }
   }
   MFEM_ASSERT(Af_el_idx == Af_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   MFEM_ASSERT(Ae_el_idx == Ae_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   A_empty = false;
}

void DarcyHybridization::AssemblePotMassMatrix(int el, const DenseMatrix &D)
{
   // **This routine allocates what it writes.** Init() also allocates D, but
   // it has to GUESS whether anything will be assembled into it: it cannot
   // see the forms, only c_bfi_p, so its condition `!m_nlfi_p || c_bfi_p`
   // reads "a linear face constraint exists" where the question is "linear
   // potential-mass content exists". A LINEAR M_p carrying only a DOMAIN
   // integrator, alongside a nonlinear Mnl_p, satisfies neither half and was
   // never allocated -- and this is the one place that then writes, through
   // an unsized Df_data. Measured: DarcyForm::Assemble() segfaulted in
   // DenseMatrix::operator+= on exactly that configuration, with a
   // three-arm control showing the crash needs only the two DOMAIN
   // integrators (a face integrator on Mnl_p is dropped either way and is a
   // separate matter). Both assembly routes reach D through here -- the
   // domain pass directly and ComputeAndAssemblePotFaceMatrix() at its tail
   // -- so this is the single choke point, and the guard fires at most once
   // because AllocD() sizes Df_data. Same idiom as ReducedGradient()'s three
   // `if (!Df_data.Size())` sites; the invariant is local rather than
   // distributed across a guess made in Init().
   if (!Df_data.Size()) { AllocD(); }

   const int s = Df_f_offsets[el+1] - Df_f_offsets[el];
   DenseMatrix D_i(&Df_data[Df_offsets[el]], s, s);
   MFEM_ASSERT(D.Size() == s, "Incompatible sizes");

   D_i += D;

   D_empty = false;
}

void DarcyHybridization::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int o = hat_offsets[el];
   const int w = hat_offsets[el+1] - o;
   const int h = Df_f_offsets[el+1] - Df_f_offsets[el];
   int Bf_el_idx = Bf_offsets[el];
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   int Be_el_idx = Be_offsets[el];
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   for (int j = 0; j < w; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         for (int i = 0; i < h; i++)
         {
            Be_data[Be_el_idx++] += B(i, j);
         }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < h; i++)
      {
         Bf_data[Bf_el_idx++] += B(i, j);
      }
   }
   MFEM_ASSERT(Bf_el_idx == Bf_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   MFEM_ASSERT(Be_el_idx == Be_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
}

void DarcyHybridization::ComputeAndAssemblePotFaceMatrix(
   int face, DenseMatrix &elmat1, DenseMatrix &elmat2, Array<int> &vdofs1,
   Array<int> &vdofs2, int skip_zeros)
{
   Mesh *mesh = fes_p.GetMesh();
   const int num_faces = mesh->GetNumFaces();
   const FiniteElement *tr_fe, *fe1, *fe2;
   DenseMatrix elmat;
   int ndof1, ndof2;
   bool save2 = false;

   tr_fe = c_fes.GetFaceElement(face);
   const int c_dof = tr_fe->GetDof() * c_fes.GetVDim();

   int el1, el2;
   mesh->GetFaceElements(face, &el1, &el2);
   fes_p.GetElementVDofs(el1, vdofs1);
   fe1 = fes_p.GetFE(el1);
   ndof1 = vdofs1.Size();
   FaceElementTransformations *ftr = NULL;

   int inf1, inf2, nc;
   if (mesh->Nonconforming())
   {
      mesh->GetFaceInfos(face, &inf1, &inf2, &nc);
      MFEM_ASSERT(nc < 0 || el2 >= 0 ||
                  inf2 >= 0, "Master face should not be integrated directly!");
   }
   else
   {
      inf1 = inf2 = nc = -1;
   }

   if (el2 >= 0)
   {
      ftr = mesh->GetFaceElementTransformations(face);
      save2 = true;
   }
#ifdef MFEM_USE_MPI
   else if (ParallelC())
   {
      ParMesh *pmesh = c_pfes->GetParMesh();
      if (pmesh->FaceIsTrueInterior(face))
      {
         ftr = pmesh->GetSharedFaceTransformationsByLocalIndex(face);
      }
   }
#endif

   if (save2)
   {
      fes_p.GetElementVDofs(ftr->Elem2No, vdofs2);
      fe2 = fes_p.GetFE(ftr->Elem2No);
      ndof2 = vdofs2.Size();
      c_bfi_p->AssembleHDGFaceMatrix(*tr_fe, *fe1, *fe2, *ftr, elmat);
   }
   else
   {
      if (!ftr) { ftr = mesh->GetFaceElementTransformations(face); }
      vdofs2.SetSize(0);
      ndof2 = 0;
      c_bfi_p->AssembleHDGFaceMatrix(0, *tr_fe, *fe1, *ftr, elmat);
   }

   MFEM_ASSERT(elmat.Width() == ndof1+ndof2+c_dof &&
               elmat.Height() == ndof1+ndof2+c_dof,
               "Size mismatch");

   // assemble D element matrices
   elmat1.CopyMN(elmat, ndof1, ndof1, 0, 0);
   AssemblePotMassMatrix(ftr->Elem1No, elmat1);
   if (save2)
   {
      elmat2.CopyMN(elmat, ndof2, ndof2, ndof1, ndof1);
      AssemblePotMassMatrix(ftr->Elem2No, elmat2);
   }

   // assemble E and G constraints
   if (nc >= 0 && face >= num_faces)
   {
      DenseMatrix E_f_1, G_f_1;
      E_f_1.CopyMN(elmat, ndof1, c_dof, 0, ndof1+ndof2);
      G_f_1.CopyMN(elmat, c_dof, ndof1, ndof1+ndof2, 0);
      AssembleNCSlaveEGFaceMatrix(face, E_f_1, G_f_1);
   }
   else
   {
      DenseMatrix E_f_1(&E_data[E_offsets[face]], ndof1, c_dof);
      DenseMatrix G_f_1(&G_data[G_offsets[face]], c_dof, ndof1);
      E_f_1.CopyMN(elmat, ndof1, c_dof, 0, ndof1+ndof2);
      G_f_1.CopyMN(elmat, c_dof, ndof1, ndof1+ndof2, 0);
   }
   if (save2)
   {
      if (nc >= 0)
      {
         // interior slave face
         DenseMatrix E_f_2, G_f_2;
         E_f_2.CopyMN(elmat, ndof2, c_dof, ndof1, ndof1+ndof2);
         G_f_2.CopyMN(elmat, c_dof, ndof2, ndof1+ndof2, ndof1);
         AssembleNCSlaveEGFaceMatrix(face, E_f_2, G_f_2);
      }
      else
      {
         DenseMatrix E_f_2(&E_data[E_offsets[face] + c_dof*ndof1], ndof2, c_dof);
         DenseMatrix G_f_2(&G_data[G_offsets[face] + c_dof*ndof1], c_dof, ndof2);
         E_f_2.CopyMN(elmat, ndof2, c_dof, ndof1, ndof1+ndof2);
         G_f_2.CopyMN(elmat, c_dof, ndof2, ndof1+ndof2, ndof1);
      }
   }

   // assemble H matrix. This is not merely allocation: it decides WHERE the
   // face H goes -- element-wise into H_data, which NPC reads through
   // GetHFaceMatrix(), or straight into the global sparse H, which only the
   // reduced solve reads. A linear form used to take the second branch
   // unconditionally, leaving GetHFaceMatrix() over a null pointer.
   if (NPCEnabled())
   {
      if (face < num_faces)
      {
         DenseMatrix H_f(&H_data[H_offsets[face]], c_dof, c_dof);
         H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
      }
      else
      {
         DenseMatrix H_f;
         H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
         AssembleNCSlaveHFaceMatrix(face, H_f);
      }
   }
   else if (face < num_faces)
   {
      Array<int> c_dofs;
      c_fes.GetFaceVDofs(face, c_dofs);

      if (!H) { H.reset(new SparseMatrix(c_fes.GetVSize())); }
      DenseMatrix H_f;
      H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
      H->AddSubMatrix(c_dofs, c_dofs, H_f, skip_zeros);
   }
   else
   {
      int face_master = -1;
      DenseMatrix H_f, ItHI_f;
      H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
      face_getter fx([this, &ItHI_f, &face_master](int f, DenseMatrix &m)
      {
         const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
         ItHI_f.SetSize(c_size);
         m.Reset(ItHI_f.GetData(), c_size, c_size);
         face_master = f;
      });
      AssembleNCSlaveFaceMatrix(face, face_getter(), NULL, face_getter(), NULL,
                                fx, &H_f);

      if (face_master < 0) { return; } // ghost master?
      if (!H) { H.reset(new SparseMatrix(c_fes.GetVSize())); }
      Array<int> c_dofs;
      c_fes.GetFaceVDofs(face_master, c_dofs);
      H->AddSubMatrix(c_dofs, c_dofs, ItHI_f, skip_zeros);
   }
}

/// The integrators the batched face assembly would be asked to apply.
void DarcyHybridization::PotFaceConstraintIntegrators(
   Array<BilinearFormIntegrator*> &integs) const
{
   integs.SetSize(0);
   BilinearFormIntegrator *cbfi = c_bfi_p.get();
   if (!cbfi) { return; }

   // **Looking THROUGH the SumIntegrator is what makes this reachable at
   // all.** DarcyForm::EnableHybridization() wraps a form's interior face
   // integrators in one unconditionally, even when there is exactly one, so a
   // dynamic_cast on c_bfi_p itself never matched for any caller that goes
   // through DarcyForm -- which is every caller in the tree.
   // AssemblyMode::Batched was therefore dead code, silently, and a timing
   // comparison did not show it: the run-to-run scatter on convdiff's
   // assembly is wider than the AtomicAdd cost the mode adds.
   if (auto *sum = dynamic_cast<SumIntegrator*>(cbfi))
   {
      for (int i = 0; i < sum->NumIntegrators(); i++)
      {
         integs.Append(sum->GetIntegrator(i));
      }
      return;
   }
   integs.Append(cbfi);
}

void DarcyHybridization::HatDofMaps(Array<int> &free_map, Array<int> &ess_map,
                                    Array<int> &ess_offsets) const
{
   const int NE = fes.GetNE();
   free_map.SetSize(Af_f_offsets.Last());
   ess_offsets.SetSize(NE + 1);
   ess_offsets[0] = 0;
   for (int el = 0; el < NE; el++)
   {
      const int o = hat_offsets[el];
      const int a = hat_offsets[el+1] - o;
      const int nf = Af_f_offsets[el+1] - Af_f_offsets[el];
      ess_offsets[el+1] = ess_offsets[el] + (a - nf);
   }
   ess_map.SetSize(ess_offsets.Last());

   for (int el = 0; el < NE; el++)
   {
      const int o = hat_offsets[el];
      const int a = hat_offsets[el+1] - o;
      int f = Af_f_offsets[el], e = ess_offsets[el];
      for (int i = 0; i < a; i++)
      {
         if (hat_dofs_marker[o + i] == 1) { ess_map[e++] = i; }
         else { free_map[f++] = i; }
      }
      MFEM_ASSERT(f == Af_f_offsets[el+1] && e == ess_offsets[el+1],
                  "Internal error.");
   }
}

bool DarcyHybridization::CanBatchElementMass(
   BilinearForm *M, const FiniteElementSpace &f) const
{
   if (asm_mode != AssemblyMode::Batched) { return false; }
   if (!M) { return false; }
   Array<BilinearFormIntegrator*> *dbfi = M->GetDBFI();
   if (!dbfi || dbfi->Size() == 0) { return false; }
   return HDGElementMassCanBatch(f, *dbfi);
}

bool DarcyHybridization::AssembleFluxMassMatricesBatched(BilinearForm *M_u)
{
   if (!CanBatchElementMass(M_u, fes)) { return false; }

   Vector emat;
   HDGElementMassBatched(fes, *M_u->GetDBFI(), emat);

   const int NE = fes.GetNE();
   Array<int> free_map, ess_map, ess_offsets;
   HatDofMaps(free_map, ess_map, ess_offsets);

   // The scatter is the mask AssembleFluxMassMatrix() applies per element,
   // written once: a free COLUMN goes to Af in its own compacted indexing, an
   // essential one goes to Ae with every row of the element. Both accumulate,
   // as the per-element routine does.
   Vector Afv, Aev;
   Afv.NewMemoryAndSize(Af_data.GetMemory(), Af_data.Size(), false);
   const auto d_M = emat.Read();
   const int *d_fm = free_map.Read(), *d_em = ess_map.Read();
   const int *d_eo = ess_offsets.Read();
   const int *d_ho = hat_offsets.Read();
   const int *d_ao = Af_offsets.Read(), *d_afo = Af_f_offsets.Read();
   real_t *d_Af = Afv.ReadWrite();
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Aev.NewMemoryAndSize(Ae_data.GetMemory(), Ae_data.Size(), false);
   const int *d_aeo = Ae_offsets.Read();
   real_t *d_Ae = Aev.ReadWrite();
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int a = d_ho[e+1] - d_ho[e];
      const int nf = d_afo[e+1] - d_afo[e];
      const int nes = d_eo[e+1] - d_eo[e];
      const real_t *M = d_M + a * a * e;

      for (int jj = 0; jj < nf; jj++)
      {
         const int j = d_fm[d_afo[e] + jj];
         for (int ii = 0; ii < nf; ii++)
         {
            const int i = d_fm[d_afo[e] + ii];
            d_Af[d_ao[e] + ii + nf * jj] += M[i + a * j];
         }
      }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      for (int jj = 0; jj < nes; jj++)
      {
         const int j = d_em[d_eo[e] + jj];
         for (int i = 0; i < a; i++)
         {
            d_Ae[d_aeo[e] + i + a * jj] += M[i + a * j];
         }
      }
#else
      MFEM_CONTRACT_VAR(nes);
      MFEM_CONTRACT_VAR(d_em);
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   });

   Af_data.GetMemory().Sync(Afv.GetMemory());
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Ae_data.GetMemory().Sync(Aev.GetMemory());
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   A_empty = false;

   // NO copy back here, and this line used to be one. DarcyForm::Assemble()
   // runs AssembleFluxMassBdrFaces() immediately after this, and that pass
   // may be a kernel too (AssembleFluxMassBdrMatricesBatched()); syncing
   // between them would pull Af and Ae to the host only to push them back.
   // DarcyForm::AssembleFluxMassBdrFaces() syncs once, after either pass,
   // exactly as AssemblePotHDGFaces() does for the potential group -- and it
   // must, because everything downstream reads these through raw pointers.
   //
   // The sync that was here was needed only because the boundary pass was
   // host code accumulating into Af through AssembleFluxMassMatrix(). Under
   // Device("debug") that read faults on an mprotected page; under CUDA it
   // silently reads a stale buffer. Measured with the boundary pass forced
   // back onto the host and this line removed: the fault lands in
   // AssembleFluxMassMatrix(), naming neither the array nor the routine that
   // left it there.
   return true;
}

void DarcyHybridization::FluxMassBdrWork(BilinearForm *M_u,
                                         Array<int> &bdr_els,
                                         Array<int> &integs,
                                         Array<int> &elems) const
{
   bdr_els.SetSize(0);
   integs.SetSize(0);
   elems.SetSize(0);
   if (!M_u) { return; }

   Array<BilinearFormIntegrator*> *bfbfi = M_u->GetBFBFI();
   if (!bfbfi || bfbfi->Size() == 0) { return; }
   Array<Array<int>*> &markers = *M_u->GetBFBFI_Marker();

   Mesh *mesh = fes.GetMesh();
   const int nattr = mesh->bdr_attributes.Size() ? mesh->bdr_attributes.Max() : 0;
   MFEM_CONTRACT_VAR(nattr);

   for (int b = 0; b < fes.GetNBE(); b++)
   {
      // A PERIODIC MESH KEEPS THE BOUNDARY ELEMENTS whose faces the
      // identification turned interior. This is the same
      // Mesh::GetBdrFaceTransformations() null test the loop in
      // DarcyForm::AssembleFluxMassBdrFaces() uses, so the kernel inherits
      // the guard rather than restating it -- and it must, or the two routes
      // would differ by one contribution per leftover element.
      FaceElementTransformations *ftr = mesh->GetBdrFaceTransformations(b);
      if (!ftr) { continue; }

      const int attr = mesh->GetBdrAttribute(b);
      for (int k = 0; k < bfbfi->Size(); k++)
      {
         const Array<int> *m = markers[k];
         MFEM_ASSERT(!m || m->Size() == nattr,
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         if (m && (*m)[attr-1] == 0) { continue; }
         bdr_els.Append(b);
         integs.Append(k);
         elems.Append(ftr->Elem1No);
      }
   }
}

bool DarcyHybridization::CanBatchFluxMassBdrFaces(BilinearForm *M_u) const
{
   if (asm_mode != AssemblyMode::Batched) { return false; }
   if (!M_u) { return false; }
   Array<BilinearFormIntegrator*> *bfbfi = M_u->GetBFBFI();
   if (!bfbfi || bfbfi->Size() == 0) { return false; }

   Array<int> bdr_els, integs, elems;
   FluxMassBdrWork(M_u, bdr_els, integs, elems);
   return bdr_els.Size() > 0;
}

bool DarcyHybridization::AssembleFluxMassBdrMatricesBatched(BilinearForm *M_u,
                                                            int skip_zeros)
{
   if (!CanBatchFluxMassBdrFaces(M_u)) { return false; }

   Array<int> bdr_els, integs, elems;
   FluxMassBdrWork(M_u, bdr_els, integs, elems);
   const int NC = bdr_els.Size();

   Array<BilinearFormIntegrator*> &bfbfi = *M_u->GetBFBFI();
   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   const int vd = fes.GetVDim();

   // THE OFFSETS, ON THE HOST, and this is not defensive -- it is the fault
   // the debug backend threw the first time this routine ran under a Device.
   // AssembleFluxMassMatricesBatched() ran just before it and handed these
   // same arrays to a kernel; Array<int>::Read() defaults to on_dev = true,
   // so they came back DEVICE-valid while the host half below indexes them
   // raw as hat_offsets[e+1]. Under Device("debug") that is a SIGSEGV inside
   // this function with an address and nothing else; under CUDA it would read
   // stale memory and size the blocks wrongly.
   //
   // **The general shape, and it is worth more than the fix**: the offset
   // arrays are SHARED between the passes, so the second kernel of a chain
   // has to host-read whatever the first one made device-valid. The element
   // pass never had to, being the first. Only the offsets, not the data --
   // Af_data and Ae_data stay device-resident, which is the whole point of
   // batching this pass at all.
   hat_offsets.HostRead();
   Af_f_offsets.HostRead();
   hat_dofs_marker.HostRead();

   // The contributions GROUPED BY ELEMENT, keeping the loop's order within
   // each element. One thread per element then sums that element's blocks in
   // exactly the order AssembleFluxMassMatrix() would have been called in, so
   // every entry sees the same sequence of additions and the result is
   // bit-for-bit rather than round-off. One thread per CONTRIBUTION would
   // have needed AtomicAdd -- a corner element carries two boundary faces --
   // and would then have been neither.
   Array<int> ecount(NE);
   ecount = 0;
   for (int c = 0; c < NC; c++) { ecount[elems[c]]++; }

   Array<int> wel, wbeg, fill(NE);
   int nslot = 0;
   for (int e = 0; e < NE; e++)
   {
      if (ecount[e] == 0) { continue; }
      wel.Append(e);
      wbeg.Append(nslot);
      fill[e] = nslot;
      nslot += ecount[e];
   }
   wbeg.Append(nslot);
   MFEM_ASSERT(nslot == NC, "internal error");

   Array<int> slot_of(NC);
   for (int c = 0; c < NC; c++) { slot_of[c] = fill[elems[c]]++; }

   // Where each slot's block starts. Sizes vary with the element, so this is
   // an offset array rather than one stride -- which costs nothing and is
   // what lets a mixed-order or mixed-geometry mesh through a gate that
   // otherwise would have had to refuse it.
   Array<int> poff(NC + 1);
   poff[0] = 0;
   for (int w = 0; w < wel.Size(); w++)
   {
      const int e = wel[w];
      const int a = hat_offsets[e+1] - hat_offsets[e];
      for (int s = wbeg[w]; s < wbeg[w+1]; s++) { poff[s+1] = poff[s] + a * a; }
   }

   Vector pack(poff[NC]);
   pack.UseDevice(true);

   // THE INTEGRATORS, on the host, in the loop's own (boundary element,
   // integrator) order -- the pack slot is permuted, the evaluation is not.
   // AssembleFaceMatrix() takes a FaceElementTransformations and a
   // FiniteElement, neither of which carries any MFEM_HOST_DEVICE, so there
   // is nothing to move here and no family to dispatch on; see the doxygen.
   {
      DenseMatrix elmat;
      real_t *pw = pack.HostWrite();
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      Array<int> vdofs;
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      for (int c = 0; c < NC; c++)
      {
         const int e = elems[c];
         const int a = hat_offsets[e+1] - hat_offsets[e];
         FaceElementTransformations *ftr =
            mesh->GetBdrFaceTransformations(bdr_els[c]);
         const FiniteElement *fe1 = fes.GetFE(e);
         // The second element is a dummy on a boundary face, as in the loop
         // this replaces: never used, but a null reference cannot be formed.
         bfbfi[integs[c]]->AssembleFaceMatrix(*fe1, *fe1, *ftr, elmat);
         MFEM_VERIFY(elmat.Height() == fe1->GetDof() * vd &&
                     elmat.Width() == elmat.Height() && elmat.Height() == a,
                     "the flux mass boundary face integrator must return the "
                     "block of the adjacent element alone");
         const int o = poff[slot_of[c]];
         for (int i = 0; i < a * a; i++) { pw[o + i] = elmat.GetData()[i]; }
#ifndef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         // The sparse flux mass, which the loop this replaces accumulates
         // into on the same pass. Host sparse work either way, and the
         // element kernels above simply drop it -- unreachable, since
         // MFEM_DARCY_HYBRIDIZATION_ELIM_BCS is defined unconditionally at
         // the top of darcyhybridization.hpp, but not a gap worth copying.
         fes.GetElementVDofs(e, vdofs);
         M_u->SpMat().AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
#endif //!MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      }
   }
   MFEM_CONTRACT_VAR(skip_zeros);

   Array<int> free_map, ess_map, ess_offsets;
   HatDofMaps(free_map, ess_map, ess_offsets);

   // The scatter is AssembleFluxMassMatrix()'s mask, the same one
   // AssembleFluxMassMatricesBatched() writes for the element blocks: a free
   // COLUMN goes to Af in its own compacted indexing, an essential one goes
   // to Ae with every row of the element.
   Vector Afv, Aev;
   Afv.NewMemoryAndSize(Af_data.GetMemory(), Af_data.Size(), false);
   const auto d_p = pack.Read();
   const int *d_we = wel.Read(), *d_wb = wbeg.Read(), *d_po = poff.Read();
   const int *d_fm = free_map.Read(), *d_em = ess_map.Read();
   const int *d_eo = ess_offsets.Read();
   const int *d_ho = hat_offsets.Read();
   const int *d_ao = Af_offsets.Read(), *d_afo = Af_f_offsets.Read();
   real_t *d_Af = Afv.ReadWrite();
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Aev.NewMemoryAndSize(Ae_data.GetMemory(), Ae_data.Size(), false);
   const int *d_aeo = Ae_offsets.Read();
   real_t *d_Ae = Aev.ReadWrite();
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   mfem::forall(wel.Size(), [=] MFEM_HOST_DEVICE (int w)
   {
      const int e = d_we[w];
      const int a = d_ho[e+1] - d_ho[e];
      const int nf = d_afo[e+1] - d_afo[e];
      const int nes = d_eo[e+1] - d_eo[e];

      for (int s = d_wb[w]; s < d_wb[w+1]; s++)
      {
         const real_t *M = d_p + d_po[s];

         for (int jj = 0; jj < nf; jj++)
         {
            const int j = d_fm[d_afo[e] + jj];
            for (int ii = 0; ii < nf; ii++)
            {
               const int i = d_fm[d_afo[e] + ii];
               d_Af[d_ao[e] + ii + nf * jj] += M[i + a * j];
            }
         }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         for (int jj = 0; jj < nes; jj++)
         {
            const int j = d_em[d_eo[e] + jj];
            for (int i = 0; i < a; i++)
            {
               d_Ae[d_aeo[e] + i + a * jj] += M[i + a * j];
            }
         }
#else
         MFEM_CONTRACT_VAR(nes);
         MFEM_CONTRACT_VAR(d_em);
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      }
   });

   Af_data.GetMemory().Sync(Afv.GetMemory());
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Ae_data.GetMemory().Sync(Aev.GetMemory());
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   A_empty = false;

   // No copy back, for the reason on the element pass above:
   // DarcyForm::AssembleFluxMassBdrFaces() syncs once after this.
   return true;
}

bool DarcyHybridization::AssemblePotMassMatricesBatched(BilinearForm *M_p)
{
   if (!CanBatchElementMass(M_p, fes_p)) { return false; }

   Vector emat;
   HDGElementMassBatched(fes_p, *M_p->GetDBFI(), emat);

   const int NE = fes_p.GetNE();
   const int N = fes_p.GetFE(0)->GetDof() * fes_p.GetVDim();

   Vector Dv;
   Dv.NewMemoryAndSize(Df_data.GetMemory(), Df_data.Size(), false);
   const auto d_M = emat.Read();
   const int *d_do = Df_offsets.Read();
   real_t *d_D = Dv.ReadWrite();
   const int n = N;

   // No mask here: the potential block keeps every element dof, so this is
   // AssemblePotMassMatrix()'s `D_i += D` for every element at once.
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const real_t *M = d_M + n * n * e;
      real_t *D = d_D + d_do[e];
      for (int i = 0; i < n * n; i++) { D[i] += M[i]; }
   });

   Df_data.GetMemory().Sync(Dv.GetMemory());
   D_empty = false;

   // As for the flux: AssemblePotHDGFaces() follows, and its boundary pass is
   // host code accumulating into D whenever the boundary kernel is refused --
   // which is most of the time.
   SyncLocalBlocksToHost();
   return true;
}

bool DarcyHybridization::CanBatchDiv(MixedBilinearForm *B) const
{
   if (asm_mode != AssemblyMode::Batched) { return false; }
   if (!B) { return false; }
   Array<BilinearFormIntegrator*> *dbfi = B->GetDBFI();
   if (!dbfi || dbfi->Size() == 0) { return false; }
   return HDGElementDivCanBatch(fes, fes_p, *dbfi);
}

bool DarcyHybridization::AssembleDivMatricesBatched(MixedBilinearForm *B)
{
   if (!CanBatchDiv(B)) { return false; }

   Vector emat;
   HDGElementDivBatched(fes, fes_p, *B->GetDBFI(), emat);

   const int NE = fes.GetNE();
   Array<int> free_map, ess_map, ess_offsets;
   HatDofMaps(free_map, ess_map, ess_offsets);

   Vector Bfv, Bev;
   Bfv.NewMemoryAndSize(Bf_data.GetMemory(), Bf_data.Size(), false);
   const auto d_M = emat.Read();
   const int *d_fm = free_map.Read(), *d_em = ess_map.Read();
   const int *d_eo = ess_offsets.Read();
   const int *d_ho = hat_offsets.Read();
   const int *d_bo = Bf_offsets.Read(), *d_afo = Af_f_offsets.Read();
   const int *d_dfo = Df_f_offsets.Read();
   real_t *d_Bf = Bfv.ReadWrite();
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Bev.NewMemoryAndSize(Be_data.GetMemory(), Be_data.Size(), false);
   const int *d_beo = Be_offsets.Read();
   real_t *d_Be = Bev.ReadWrite();
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int a = d_ho[e+1] - d_ho[e];        // hat (flux) dofs
      const int h = d_dfo[e+1] - d_dfo[e];      // potential dofs
      const int nf = d_afo[e+1] - d_afo[e];
      const int nes = d_eo[e+1] - d_eo[e];
      const real_t *M = d_M + h * a * e;

      for (int jj = 0; jj < nf; jj++)
      {
         const int j = d_fm[d_afo[e] + jj];
         for (int i = 0; i < h; i++)
         {
            d_Bf[d_bo[e] + i + h * jj] += M[i + h * j];
         }
      }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      for (int jj = 0; jj < nes; jj++)
      {
         const int j = d_em[d_eo[e] + jj];
         for (int i = 0; i < h; i++)
         {
            d_Be[d_beo[e] + i + h * jj] += M[i + h * j];
         }
      }
#else
      MFEM_CONTRACT_VAR(nes);
      MFEM_CONTRACT_VAR(d_em);
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   });

   Bf_data.GetMemory().Sync(Bfv.GetMemory());
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Be_data.GetMemory().Sync(Bev.GetMemory());
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   // AssembleDivLDGFaces() and the potential mass loop follow, both host code
   // reading these through raw pointers; see the flux mass.
   SyncLocalBlocksToHost();
   return true;
}

void DarcyHybridization::PotBdrFaceLists(
   std::vector<Array<int>> &lists, Array<int> &all,
   Array<BilinearFormIntegrator*> &integs) const
{
   Mesh *mesh = fes_p.GetMesh();
   const int nint = NumBdrPotConstraintIntegrators();

   integs.SetSize(0);
   for (int k = 0; k < nint; k++)
   {
      integs.Append(boundary_constraint_pot_integs[k]);
   }
   lists.assign(nint, Array<int>());
   all.SetSize(0);

   for (int b = 0; b < mesh->GetNBE(); b++)
   {
      // A PERIODIC MESH KEEPS THE BOUNDARY ELEMENTS whose faces the
      // identification turned interior, and GetBdrElementFaceIndex() then
      // hands back an interior face whose two-sided E, G and H the interior
      // pass has already filled. Mesh::GetBdrFaceTransformations() returning
      // null is how every other boundary loop in this class drops them, and
      // it is how this one does; see DarcyForm::AssemblePotHDGFaces().
      if (!mesh->GetBdrFaceTransformations(b)) { continue; }
      const int face = mesh->GetBdrElementFaceIndex(b);
      const int attr = mesh->GetBdrAttribute(b);
      bool any = false;
      for (int k = 0; k < nint; k++)
      {
         const Array<int> *m = boundary_constraint_pot_integs_marker[k];
         if (m && (*m)[attr-1] == 0) { continue; }
         lists[k].Append(face);
         any = true;
      }
      if (any) { all.Append(face); }
   }
}

bool DarcyHybridization::CanBatchPotBdrFaceAssembly() const
{
   if (asm_mode != AssemblyMode::Batched) { return false; }
   // The same two refusals the interior pass carries, and for the same
   // reasons: H's destination, and shared faces.
   if (!NPCEnabled()) { return false; }
   if (ParallelC()) { return false; }
   if (NumBdrPotConstraintIntegrators() == 0) { return false; }

   std::vector<Array<int>> lists;
   Array<int> all;
   Array<BilinearFormIntegrator*> integs;
   PotBdrFaceLists(lists, all, integs);
   if (all.Size() == 0) { return false; }
   return HDGBdrFaceScatterCanBatch(c_fes, fes_p, integs, lists);
}

bool DarcyHybridization::AssemblePotBdrFaceMatricesBatched()
{
   if (!CanBatchPotBdrFaceAssembly()) { return false; }

   std::vector<Array<int>> lists;
   Array<int> all;
   Array<BilinearFormIntegrator*> integs;
   PotBdrFaceLists(lists, all, integs);

   Vector Ev, Gv, Hv, Dv;
   Ev.NewMemoryAndSize(E_data.GetMemory(), E_data.Size(), false);
   Gv.NewMemoryAndSize(G_data.GetMemory(), G_data.Size(), false);
   Hv.NewMemoryAndSize(H_data.GetMemory(), H_data.Size(), false);
   Dv.NewMemoryAndSize(Df_data.GetMemory(), Df_data.Size(), false);

   HDGBdrFaceScatterBatched(c_fes, fes_p, integs, lists, all, E_offsets,
                            H_offsets, Df_offsets, Ev, Gv, Hv, Dv);

   E_data.GetMemory().Sync(Ev.GetMemory());
   G_data.GetMemory().Sync(Gv.GetMemory());
   H_data.GetMemory().Sync(Hv.GetMemory());
   Df_data.GetMemory().Sync(Dv.GetMemory());
   D_empty = false;

   return true;
}

int DarcyHybridization::NumPotFaceConstraintIntegrators() const
{
   Array<BilinearFormIntegrator*> integs;
   PotFaceConstraintIntegrators(integs);
   return integs.Size();
}

bool DarcyHybridization::CanBatchPotFaceAssembly() const
{
   if (asm_mode != AssemblyMode::Batched) { return false; }

   // The kernel writes H into H_data, which is where the per-face route puts
   // it ONLY under NPC; otherwise that route scatters H into the assembled
   // sparse matrix and nothing ever reads H_data. Taking the kernel there
   // would put the face term where the reduced solve does not look. Refused
   // rather than repaired here, because repairing it is a scatter the kernel
   // does not have; see ComputeAndAssemblePotFaceMatrix().
   if (!NPCEnabled()) { return false; }

   // A shared face is not interior by Mesh::FaceIsInterior(), which is what
   // the face list is built from, so in parallel the kernel would silently
   // drop every face on a partition boundary.
   if (ParallelC()) { return false; }

   Array<BilinearFormIntegrator*> integs;
   PotFaceConstraintIntegrators(integs);
   if (integs.Size() == 0) { return false; }

   Array<int> flist;
   InteriorFaceList(flist);
   return HDGFaceScatterCanBatch(c_fes, fes_p, integs, flist);
}

bool DarcyHybridization::CanBatchLocalResidual() const
{
   if (asm_mode != AssemblyMode::Batched) { return false; }

   // MultNlMode::AtFields is NPC's mode and the only one that evaluates the
   // local residual at supplied fields; every other mode SOLVES the local
   // problem, and a precomputed residual is no use to a solve.
   if (!NPCEnabled()) { return false; }

   // Parallel is refused, and the reason is NOT that the kernel needs
   // anything from a neighbour. It does not: NPCCheck() refuses a conforming
   // flux space, so every element's flux dofs are its own and this loop is
   // rank-local exactly as NPCResidual()'s own comment says. It is refused
   // because it has not been RUN on more than one rank -- this worktree is a
   // serial build -- and a claim about behaviour on this branch is measured
   // rather than argued from a code path. Lifting it is a test, not work.
   if (ParallelU() || ParallelP()) { return false; }

   if (!m_nlfi) { return false; }
   if (!HDGMixedConductionResidualCanBatch(fes, m_nlfi)) { return false; }

   // One flux block per element, and it has to be the element's WHOLE vdof
   // set: the kernel writes ndof*vdim entries per element into a flat array
   // in GetElementVDofs() order, while GetFDofs() drops essential flux dofs
   // -- which would shorten one element's slice and shift every slice after
   // it. An L2 flux has no essential dofs, so this is a check rather than a
   // restriction there; on an H(div) flux it is the restriction, and the
   // integrator gate refuses that space anyway.
   const int NE = fes.GetNE();
   if (NE == 0) { return false; }
   const int na = UniformBlockSize(Af_f_offsets, NE);
   return na > 0 && na == fes.GetFE(0)->GetDof() * fes.GetVDim();
}

void DarcyHybridization::NLFaceConstraintIntegrators(
   Array<NonlinearFormIntegrator*> &integs,
   Array<BlockNonlinearFormIntegrator*> &bintegs) const
{
   integs.SetSize(0);
   bintegs.SetSize(0);

   // Looking THROUGH the sum is what makes any of this reachable, exactly as
   // in PotFaceConstraintIntegrators(): DarcyForm::EnableHybridization()
   // wraps a form's interior face integrators in a SumNLFIntegrator or a
   // SumBlockNLFIntegrator unconditionally, even when there is one, so a
   // dynamic_cast on the slot itself never matches what is inside.
   if (NonlinearFormIntegrator *c = c_nlfi_p.get())
   {
      if (auto *sum = dynamic_cast<SumNLFIntegrator*>(c))
      {
         for (int i = 0; i < sum->NumIntegrators(); i++)
         { integs.Append(sum->GetIntegrator(i)); }
      }
      else { integs.Append(c); }
   }
   if (BlockNonlinearFormIntegrator *c = c_nlfi.get())
   {
      if (auto *sum = dynamic_cast<SumBlockNLFIntegrator*>(c))
      {
         for (int i = 0; i < sum->NumIntegrators(); i++)
         { bintegs.Append(sum->GetIntegrator(i)); }
      }
      else { bintegs.Append(c); }
   }
}

bool DarcyHybridization::CanBatchNLFaceGrad() const
{
   if (asm_mode != AssemblyMode::Batched) { return false; }

   // NPC only, and for a reason that is not the same as
   // CanBatchPotFaceAssembly()'s. The batched pass runs AFTER the element
   // loop, so it needs the element states to still exist -- and only
   // MultNlMode::GradAtFields has them, in darcy_u / darcy_p as Newton state.
   // Under MultNlMode::Grad the fields are produced by MultInvNL() inside the
   // loop and are local to it.
   if (!NPCEnabled()) { return false; }

   // A shared face is not Mesh::FaceIsInterior(), so in parallel the pair
   // list would silently drop every partition boundary.
   if (ParallelC()) { return false; }

   // A system's blocks are field-outermost, which is what byNODES gives and
   // what every integrator here indexes; byVDIM would interleave them.
   if (fes_p.GetVDim() > 1 && fes_p.GetOrdering() != Ordering::byNODES)
   { return false; }
   if (c_fes.GetVDim() > 1 && c_fes.GetOrdering() != Ordering::byNODES)
   { return false; }

   Array<NonlinearFormIntegrator*> integs;
   Array<BlockNonlinearFormIntegrator*> bintegs;
   NLFaceConstraintIntegrators(integs, bintegs);
   if (integs.Size() + bintegs.Size() == 0) { return false; }

   Array<int> flist;
   InteriorFaceList(flist);
   return HDGNLFaceGradCanBatch(c_fes, fes_p, &fes, integs, bintegs, flist);
}

/** @brief The nonlinear interior-face constraint's RESIDUAL for every face at
    once, added into the potential row of @a r_local and the trace row @a y.

    Tier 1 of the nonlinear extension, and the sibling of
    AssembleNLFaceGradBatched(): same face list, same pair indexing, same
    state gather, and HDGNLFaceResidualBatched() is the gradient kernel one
    tensor rank lower. What differs is only the destination -- a residual goes
    to the two rows rather than to E, G, H and D.

    @return false when the integrators do not admit it, leaving the caller's
            element loop to do the work. */
bool DarcyHybridization::CanBatchNLFaceResidual() const
{
   // The caller's choice, and the same gate CanBatchNLFaceGrad() carries:
   // the two are the residual and the Jacobian of one constraint and a
   // configuration that batches one should batch the other.
   if (asm_mode != AssemblyMode::Batched) { return false; }

   Array<NonlinearFormIntegrator*> integs;
   Array<BlockNonlinearFormIntegrator*> bintegs;
   NLFaceConstraintIntegrators(integs, bintegs);
   if (integs.Size() + bintegs.Size() == 0) { return false; }

   if (!NPCEnabled() || ParallelC()) { return false; }
   if (fes_p.GetVDim() > 1 && fes_p.GetOrdering() != Ordering::byNODES)
   { return false; }
   if (c_fes.GetVDim() > 1 && c_fes.GetOrdering() != Ordering::byNODES)
   { return false; }

   // **A boundary constraint is NOT a refusal, and asking for one was the
   // whole reason this predicate never fired.** Both skip sites -- MultNL()'s
   // face loop and LocalNLOperator::AddMultDE() -- test `FTr->Elem2No >= 0`
   // before skipping, so the boundary branch runs in the element loop exactly
   // as it did before, and this kernel covers the interior faces the loop
   // then leaves alone. The first draft refused
   // boundary_constraint_pot_nonlin_integs on the grounds that a half-applied
   // constraint drops the boundary term silently; that is a real failure mode
   // (BdrHyperbolicDirichletIntegrator) but it is not this code's, and the
   // gradient's CanBatchNLFaceGrad() -- verified against the per-pair loop --
   // admits the same fixtures with no such test. Measured: every one of the
   // 74 refusals across [Batched] was this condition, so the route was dead.

   Array<int> flist;
   InteriorFaceList(flist);
   if (flist.Size() == 0) { return false; }
   if (!HDGNLFaceResidualCanBatch(c_fes, fes_p, &fes, integs, bintegs, flist))
   { return false; }

   // The uniformity the pair gather needs, asked here rather than discovered
   // half way through the gather -- MultNL() has already SKIPPED the element
   // loop's interior face work by the time the routine runs, so a refusal
   // there would silently drop the term. **The predicate has to be the
   // routine's whole condition**, which is the lesson
   // CanBatchTraceAssembly() was rewritten for.
   const int LDD = Df_f_offsets[1] - Df_f_offsets[0];
   const int LDC = c_fes.GetFaceElement(flist[0])->GetDof() * c_fes.GetVDim();
   Mesh *mesh = fes_p.GetMesh();
   Array<int> p_dofs, c_dofs;
   for (int fi = 0; fi < flist.Size(); fi++)
   {
      const int f = flist[fi];
      c_fes.GetFaceVDofs(f, c_dofs);
      if (c_dofs.Size() != LDC) { return false; }
      int el1, el2;
      mesh->GetFaceElements(f, &el1, &el2);
      const int els[2] = { el1, el2 };
      for (int side = 0; side < 2; side++)
      {
         const int el = els[side];
         if (el < 0) { return false; }
         if (Df_f_offsets[el+1] - Df_f_offsets[el] != LDD) { return false; }
         fes_p.GetElementVDofs(el, p_dofs);
         if (p_dofs.Size() != LDD) { return false; }
      }
   }
   return true;
}

bool DarcyHybridization::AssembleNLFaceResidualBatched(
   const Vector &x, BlockVector &r_local, Vector &y, bool checked) const
{
   // **@a checked exists because the predicate is not cheap and MultNL()
   // has already paid for it.** It walks every interior face building four
   // dof lists per face, and MultNL() must ask BEFORE the element loop while
   // this runs after it -- so asking again here doubled the cost of a route
   // whose whole margin is the per-face frame it removes. Measured, 48x48,
   // 20 NPC residuals: 0.4811 -> 0.4298 s at order 1 on quads and
   // 2.0853 -> 1.8656 at order 3 on triangles, which moved the route from
   // 1.05x-1.24x against the per-face loop to 1.17x-1.36x. **A predicate
   // that walks the mesh is not free, and this one was being paid twice.**
   // A parameter and not a cached member, per this class's standing rule
   // that a new data member is a `make clean` in every tree.
   if (!checked && !CanBatchNLFaceResidual()) { return false; }

   Array<NonlinearFormIntegrator*> integs;
   Array<BlockNonlinearFormIntegrator*> bintegs;
   NLFaceConstraintIntegrators(integs, bintegs);
   Array<int> flist;
   InteriorFaceList(flist);
   const int NF = flist.Size();
   const int NP = 2 * NF;
   Mesh *mesh = fes_p.GetMesh();
   const int LDD = Df_f_offsets[1] - Df_f_offsets[0];
   const int LDC = c_fes.GetFaceElement(flist[0])->GetDof() * c_fes.GetVDim();

   // The same gather AssembleNLFaceGradBatched() does, and it should be
   // shared with it the moment either changes.
   Vector el_state(NP * LDD), tr_state(NP * LDC);
   Array<int> p_dofs, c_dofs;
   Vector p_l, x_f;
   Array<int> pair_el(NP), pair_face(NP);
   for (int fi = 0; fi < NF; fi++)
   {
      const int f = flist[fi];
      int el1, el2;
      mesh->GetFaceElements(f, &el1, &el2);
      const int els[2] = { el1, el2 };
      c_fes.GetFaceVDofs(f, c_dofs);
      x.GetSubVector(c_dofs, x_f);
      for (int side = 0; side < 2; side++)
      {
         const int p = 2 * fi + side;
         const int el = els[side];
         fes_p.GetElementVDofs(el, p_dofs);
         darcy_p.GetSubVector(p_dofs, p_l);
         for (int i = 0; i < LDD; i++) { el_state(p * LDD + i) = p_l(i); }
         for (int i = 0; i < LDC; i++) { tr_state(p * LDC + i) = x_f(i); }
         pair_el[p] = el;
         pair_face[p] = f;
      }
   }

   Vector r_el, r_tr;
   HDGNLFaceResidualBatched(c_fes, fes_p, &fes, integs, bintegs, flist,
                            el_state, tr_state, r_el, r_tr);

   // The scatter stays on the host: it is one AddElementVector per pair over
   // dof lists this routine already has, and the pairs of an element collide
   // on the potential row exactly as the element loop's faces do. Batching it
   // wants the (element, local face) maps LinearResidualBatched() uses, which
   // this route does not build -- a real next step, not a defect.
   const real_t *pre = r_el.HostRead(), *prt = r_tr.HostRead();
   Vector blk;
   for (int p = 0; p < NP; p++)
   {
      fes_p.GetElementVDofs(pair_el[p], p_dofs);
      blk.SetSize(LDD);
      for (int i = 0; i < LDD; i++) { blk(i) = pre[p * LDD + i]; }
      r_local.GetBlock(1).AddElementVector(p_dofs, blk);

      c_fes.GetFaceVDofs(pair_face[p], c_dofs);
      blk.SetSize(LDC);
      for (int i = 0; i < LDC; i++) { blk(i) = prt[p * LDC + i]; }
      y.AddElementVector(c_dofs, blk);
   }
   return true;
}

bool DarcyHybridization::AssembleNLFaceGradBatched(const Vector &x) const
{
   if (!CanBatchNLFaceGrad()) { return false; }

   Array<NonlinearFormIntegrator*> integs;
   Array<BlockNonlinearFormIntegrator*> bintegs;
   NLFaceConstraintIntegrators(integs, bintegs);

   Array<int> flist;
   InteriorFaceList(flist);
   const int NF = flist.Size();
   if (NF == 0) { return true; }
   const int NP = 2 * NF;

   Mesh *mesh = fes_p.GetMesh();
   const int LDD = Df_f_offsets[1] - Df_f_offsets[0];
   const int LDC = c_fes.GetFaceElement(flist[0])->GetDof() * c_fes.GetVDim();

   Array<int> D_off(NP), E_off(NP), G_off(NP), H_off(NP);
   Vector el_state(NP * LDD), tr_state(NP * LDC);

   Array<int> p_dofs, c_dofs;
   Vector p_l, x_f;
   for (int fi = 0; fi < NF; fi++)
   {
      const int f = flist[fi];
      int el1, el2;
      mesh->GetFaceElements(f, &el1, &el2);
      const int els[2] = { el1, el2 };

      c_fes.GetFaceVDofs(f, c_dofs);
      x.GetSubVector(c_dofs, x_f);
      MFEM_VERIFY(x_f.Size() == LDC, "trace block size is not uniform");

      for (int side = 0; side < 2; side++)
      {
         const int p = 2 * fi + side;
         const int el = els[side];
         MFEM_VERIFY(Df_f_offsets[el+1] - Df_f_offsets[el] == LDD,
                     "potential block size is not uniform");

         D_off[p] = Df_offsets[el];
         // Side 2's E and G blocks follow side 1's, which is the offset
         // AssembleHDGGrad() computes as c_dofs_size*d_dofs_size.
         const int eg = side ? (LDC * LDD) : 0;
         E_off[p] = E_offsets[f] + eg;
         G_off[p] = G_offsets[f] + eg;
         H_off[p] = H_offsets[f];

         fes_p.GetElementVDofs(el, p_dofs);
         MFEM_VERIFY(p_dofs.Size() == LDD, "potential vdof count is not "
                     "uniform: " << p_dofs.Size() << " against " << LDD);
         darcy_p.GetSubVector(p_dofs, p_l);
         for (int i = 0; i < LDD; i++) { el_state(p * LDD + i) = p_l(i); }
         for (int i = 0; i < LDC; i++) { tr_state(p * LDC + i) = x_f(i); }
      }
   }

   // D, E, G and H arrive here from the HOST element loop, which reached them
   // through raw pointers -- and a raw host write does not invalidate a
   // device copy. What makes the ReadWrite() below see them at all is that
   // SyncLocalBlocksToHost(), at the top of MultNL(), hands the host
   // OWNERSHIP and not merely a readable copy; see the note there, which
   // carries the measurement. Without that this kernel accumulated onto the
   // previous gradient's device data from the second evaluation onward.
   HDGNLFaceGradScatterBatched(c_fes, fes_p, &fes, integs, bintegs, flist,
                               el_state, tr_state, D_off, E_off, G_off, H_off,
                               Df_data, E_data, G_data, H_data);

   // The kernel took D, E, G and H through Vector::ReadWrite(), whose default
   // is on_dev = true, so they come back marked valid on the device -- and
   // ComputeH(), which runs next, indexes them raw as &Df_data[...] on the
   // host. Found by running it: under Device("debug") the batched route
   // faulted inside NPCGradient() at order 0 with an address and nothing
   // else, where the per-pair route got through. Same shape as the offsets
   // note on SyncLocalBlocksToHost() itself, and the same fix.
   SyncLocalBlocksToHost();
   return true;
}

/// The interior faces, which is what the batched face assembly covers.
void DarcyHybridization::InteriorFaceList(Array<int> &flist) const
{
   Mesh *mesh = fes_p.GetMesh();
   flist.SetSize(0);
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      if (mesh->FaceIsInterior(f)) { flist.Append(f); }
   }
}

bool DarcyHybridization::AssemblePotFaceMatricesBatched()
{
   if (!CanBatchPotFaceAssembly()) { return false; }

   Array<BilinearFormIntegrator*> integs;
   PotFaceConstraintIntegrators(integs);

   Array<int> flist;
   InteriorFaceList(flist);
   if (flist.Size() == 0) { return true; }

   // Vector views carrying the arrays' Memory -- not GetData(), for the
   // reason on InvertA(): a raw pointer pins the kernel to the host.
   Vector Ev, Gv, Hv, Dv;
   Ev.NewMemoryAndSize(E_data.GetMemory(), E_data.Size(), false);
   Gv.NewMemoryAndSize(G_data.GetMemory(), G_data.Size(), false);
   Hv.NewMemoryAndSize(H_data.GetMemory(), H_data.Size(), false);
   Dv.NewMemoryAndSize(Df_data.GetMemory(), Df_data.Size(), false);

   HDGFaceScatterBatched(c_fes, fes_p, integs, flist, E_offsets, H_offsets,
                         Df_offsets, Ev, Gv, Hv, Dv);

   E_data.GetMemory().Sync(Ev.GetMemory());
   G_data.GetMemory().Sync(Gv.GetMemory());
   H_data.GetMemory().Sync(Hv.GetMemory());
   Df_data.GetMemory().Sync(Dv.GetMemory());
   D_empty = false;

   // NO copy back here. The boundary pass runs next and may be a kernel too,
   // and syncing between them would pull D to the host only to push it back.
   // DarcyForm::AssemblePotHDGFaces() calls SyncLocalBlocksToHost() once,
   // after both -- which it must, because ComputeElementH() and every face
   // loop after it read these through raw pointers.
   //
   // That sync is not optional and its absence is not a warning. Under
   // Device("debug") the first host reader faults, in
   // DenseMatrix::operator+= inside AssemblePotMassMatrix(), naming neither
   // the array nor the routine that left it there; under CUDA it does not
   // fault at all and the answer comes back 60% wrong.
   return true;
}

void DarcyHybridization::ComputeAndAssemblePotBdrFaceMatrix(
   int bface, DenseMatrix &elmat1, Array<int> &vdofs, int skip_zeros)
{
   Mesh *mesh = fes_p.GetMesh();
   const FiniteElement *tr_fe, *fe;
   DenseMatrix elmat, elmat_aux, h_elmat;
   Array<int> c_dofs;

   const int face = mesh->GetBdrElementFaceIndex(bface);
   tr_fe = c_fes.GetFaceElement(face);
   c_fes.GetFaceVDofs(face, c_dofs);
   const int c_dof = c_dofs.Size();

   FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(face);
   fes_p.GetElementVDofs(ftr->Elem1No, vdofs);
   fe = fes_p.GetFE(ftr->Elem1No);
   const int ndof = fe->GetDof() * fes_p.GetVDim();

   MFEM_ASSERT(boundary_constraint_pot_integs.size() > 0,
               "No boundary constraint integrators");

   const int bdr_attr = mesh->GetBdrAttribute(bface);
   for (size_t i = 0; i < boundary_constraint_pot_integs.size(); i++)
   {
      if (boundary_constraint_pot_integs_marker[i]
          && (*boundary_constraint_pot_integs_marker[i])[bdr_attr-1] == 0) { continue; }

      boundary_constraint_pot_integs[i]->AssembleHDGFaceMatrix(*tr_fe, *fe, *fe, *ftr,
                                                               elmat_aux);

      if (elmat.Size() > 0)
      { elmat += elmat_aux; }
      else
      { elmat = elmat_aux; }
   }

   if (elmat.Size() == 0) { return; }

   MFEM_ASSERT(elmat.Width() == ndof+c_dof &&
               elmat.Height() == ndof+c_dof,
               "Size mismatch");

   // assemble D element matrices
   elmat1.CopyMN(elmat, ndof, ndof, 0, 0);
   AssemblePotMassMatrix(ftr->Elem1No, elmat1);

   // assemble E constraint
   DenseMatrix E_f_1(&E_data[E_offsets[face]], ndof, c_dof);
   E_f_1.CopyMN(elmat, ndof, c_dof, 0, ndof);

   // assemble G constraint
   DenseMatrix G_f(&G_data[G_offsets[face]], c_dof, ndof);
   G_f.CopyMN(elmat, c_dof, ndof, ndof, 0);

   // assemble H matrix -- the same choice of destination as the interior case
   if (NPCEnabled())
   {
      DenseMatrix H_f(&H_data[H_offsets[face]], c_dof, c_dof);
      H_f.CopyMN(elmat, c_dof, c_dof, ndof, ndof);
   }
   else
   {
      if (!H) { H.reset(new SparseMatrix(c_fes.GetVSize())); }
      h_elmat.CopyMN(elmat, c_dof, c_dof, ndof, ndof);
      H->AddSubMatrix(c_dofs, c_dofs, h_elmat, skip_zeros);
   }
}

/** Filtered IN PLACE, and neither the temporary nor the DeleteAll() this used
    to carry is an accident of style.

    It read `Array<int> vdofs; fes.GetElementVDofs(el, vdofs); fdofs.DeleteAll();
    ... fdofs.Append(...)`, which is two allocations per call and one free of
    the CALLER's buffer. The temporary starts at capacity zero, so
    GetElementVDofs() allocates it and DofsToVDofs() then grows it to vdim
    times that -- two malloc/free pairs -- and DeleteAll() guarantees the
    caller reallocates on the next element however carefully it hoisted its
    array. Measured over `convdiff -p 6 -nl -dg -hb -npc` at 256 elements:
    6,912 blocks of 90,949, from this routine and GetEDofs() together.

    The retained dofs are a subsequence of the full list, so the filter can run
    in place with one cursor and no second array. */
void DarcyHybridization::GetFDofs(int el, Array<int> &fdofs) const
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   fes.GetElementVDofs(el, fdofs);
   MFEM_ASSERT(fdofs.Size() == s, "Incompatible DOF sizes");
   int k = 0;
   for (int i = 0; i < s; i++)
   {
      if (hat_dofs_marker[i + o] != 1) { fdofs[k++] = fdofs[i]; }
   }
   fdofs.SetSize(k);
}

/// The eliminated dofs, filtered in place; see GetFDofs() for why in place.
void DarcyHybridization::GetEDofs(int el, Array<int> &edofs) const
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   fes.GetElementVDofs(el, edofs);
   MFEM_ASSERT(edofs.Size() == s, "Incompatible DOF sizes");
   int k = 0;
   for (int i = 0; i < s; i++)
   {
      if (hat_dofs_marker[i + o] == 1) { edofs[k++] = edofs[i]; }
   }
   edofs.SetSize(k);
}

FaceElementTransformations *DarcyHybridization::GetFaceTransformation(
   int f) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   FaceElementTransformations *FTr;
   if (el2 >= 0)
   {
      FTr = fes.GetMesh()->GetFaceElementTransformations(f);
   }
#ifdef MFEM_USE_MPI
   else if (ParallelC() && c_pfes->GetParMesh()->FaceIsTrueInterior(f))
   {
      FTr = c_pfes->GetParMesh()->GetSharedFaceTransformationsByLocalIndex(f);
   }
#endif
   else
   {
      FTr = fes.GetMesh()->GetFaceElementTransformations(f, 21);
   }

   return FTr;
}

void DarcyHybridization::BuildElementDofMaps() const
{
   const int NE = fes.GetNE();
   const int na = Af_f_offsets.Last(), nd = Df_f_offsets.Last();
   // el_c_dofs is built in this same pass, so this early-out covers it: if
   // the u/p maps are here then the trace map was either built or
   // deliberately emptied. Asking about its SIZE instead would rebuild the
   // whole thing every call on a mesh that cannot support it.
   if (el_u_dofs.Size() == na && el_p_dofs.Size() == nd) { return; }

   el_u_dofs.SetSize(na);
   el_p_dofs.SetSize(nd);

   Array<int> u_vdofs, p_dofs;
   for (int el = 0; el < NE; el++)
   {
      GetFDofs(el, u_vdofs);
      MFEM_ASSERT(u_vdofs.Size() == Af_f_offsets[el+1] - Af_f_offsets[el],
                  "Internal error.");
      std::copy(u_vdofs.begin(), u_vdofs.end(),
                el_u_dofs.begin() + Af_f_offsets[el]);

      fes_p.GetElementVDofs(el, p_dofs);
      MFEM_ASSERT(p_dofs.Size() == Df_f_offsets[el+1] - Df_f_offsets[el],
                  "Internal error.");
      std::copy(p_dofs.begin(), p_dofs.end(),
                el_p_dofs.begin() + Df_f_offsets[el]);
   }

   // The TRACE map, which is the third of the three and the one the batched
   // linear residual gathers x with. Only buildable when the mesh gives one
   // face count per element and one trace size per face -- the same
   // uniformity BuildElementHFaceMap() asks for, and for the same reason: the
   // element-blocked trace vector has to have one stride. Left EMPTY when it
   // does not hold, which is what CanBatchLinearResidual() tests.
   {
      Array<int> faces, c_dofs;
      GetElementFaces(0, faces);
      const int nf = faces.Size();
      const int vdim = c_fes.GetVDim();
      const int nc = (nf > 0)
                     ? c_fes.GetFaceElement(faces[0])->GetDof() * vdim : 0;
      bool uniform = (nf > 0 && nc > 0);
      if (uniform)
      {
         el_c_dofs.SetSize(NE*nf*nc);
         for (int el = 0; el < NE && uniform; el++)
         {
            GetElementFaces(el, faces);
            if (faces.Size() != nf) { uniform = false; break; }
            for (int lf = 0; lf < nf; lf++)
            {
               c_fes.GetFaceVDofs(faces[lf], c_dofs);
               if (c_dofs.Size() != nc) { uniform = false; break; }
               for (int j = 0; j < nc; j++)
               {
                  // A negative index is an orientation flip, which the gather
                  // kernels do not decode. Refused rather than mishandled;
                  // DG_Interface produces none.
                  if (c_dofs[j] < 0) { uniform = false; break; }
                  el_c_dofs[(el*nf + lf)*nc + j] = c_dofs[j];
               }
            }
         }
      }
      if (!uniform) { el_c_dofs.SetSize(0); }
      else { el_c_dofs.UseDevice(true); }
   }

   // The kernel side of GetSubVector()/SetSubVector() is chosen by
   // dofs.UseDevice() || elemvect.UseDevice(), so this is half of what sends
   // the gather to the device; the blocked vector at each call site is the
   // other half.
   el_u_dofs.UseDevice(true);
   el_p_dofs.UseDevice(true);
}

void DarcyHybridization::SyncLocalBlocksToHost() const
{
   // The OFFSET arrays as well as the data, and they are the sharper half.
   // The batched element mass hands these straight to a kernel, so
   // Array<int>::Read() -- whose default is on_dev = true -- marks them valid
   // there; and every host reader indexes them raw, as Af_offsets[el]. The
   // fault that follows names an address and nothing else. Same shape as the
   // Af_ipiv note on InvertA(), and found the same way.
   if (hat_offsets.Size()) { hat_offsets.HostRead(); }
   if (Af_offsets.Size()) { Af_offsets.HostRead(); }
   if (Af_f_offsets.Size()) { Af_f_offsets.HostRead(); }
   if (Bf_offsets.Size()) { Bf_offsets.HostRead(); }
   if (Df_offsets.Size()) { Df_offsets.HostRead(); }
   if (Df_f_offsets.Size()) { Df_f_offsets.HostRead(); }
   if (E_offsets.Size()) { E_offsets.HostRead(); }
   if (H_offsets.Size()) { H_offsets.HostRead(); }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   if (Ae_offsets.Size()) { Ae_offsets.HostRead(); }
   if (Be_offsets.Size()) { Be_offsets.HostRead(); }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   // HostReadWrite() AND NOT HostRead(), which is the whole of a defect that
   // cost two device-only wrong answers; the reason is on the declaration.
   // A read leaves BOTH copies valid, and every host loop below writes these
   // blocks through raw pointers, which does not invalidate the device side --
   // so the next kernel found a valid, stale device copy and skipped its
   // upload. Taking ownership here is what makes the host loop's work the
   // thing a kernel sees.
   if (Ct_data.Size()) { Ct_data.HostRead(); }
   if (E_data.Size()) { E_data.HostReadWrite(); }
   if (G_data.Size()) { G_data.HostReadWrite(); }
   if (H_data.Size()) { H_data.HostReadWrite(); }
   if (Af_data.Size()) { Af_data.HostReadWrite(); }
   if (Af_ipiv.Size()) { Af_ipiv.HostReadWrite(); }
   if (Bf_data.Size()) { Bf_data.HostRead(); }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   // The ELIMINATED blocks, which the batched flux mass writes alongside Af
   // and which EliminateVDofsInRHS() reads on the host. They were missing
   // from this list and the debug backend faulted on Ae the first time the
   // mass kernel ran with a Device configured.
   if (Ae_data.Size()) { Ae_data.HostRead(); }
   if (Be_data.Size()) { Be_data.HostRead(); }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   if (Bnl_data.Size()) { Bnl_data.HostReadWrite(); }
   if (Df_data.Size()) { Df_data.HostReadWrite(); }
   if (Df_lin_data.Size()) { Df_lin_data.HostReadWrite(); }
   if (Df_ipiv.Size()) { Df_ipiv.HostReadWrite(); }
   if (Sf_data.Size()) { Sf_data.HostReadWrite(); }
   if (Sf_ipiv.Size()) { Sf_ipiv.HostReadWrite(); }
   // Ct_data, Ae_data, Bf_data and Be_data stay on HostRead() above, and it
   // is a declaration of scope rather than a judgement that they are safe:
   // they are not `mutable`, so a const method cannot take ownership of
   // them, and unlike the blocks above nothing rewrites them on the host
   // BETWEEN two kernel passes -- Ct is built once in ConstructC() and the
   // other three at Assemble() time. A host loop that ever does rewrite one
   // of them between two device passes has this defect and will need the
   // same treatment.
}

void DarcyHybridization::BuildElementColouring() const
{
   if (colour_offsets.Size() > 0) { return; }

   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();

   Array<int> colours;
   mesh->GetElementColoring(colours);

   int ncol = 0;
   for (int el = 0; el < NE; el++) { ncol = std::max(ncol, colours[el] + 1); }

   // counting sort of the elements by colour
   colour_offsets.SetSize(ncol + 1);
   colour_offsets = 0;
   for (int el = 0; el < NE; el++) { colour_offsets[colours[el] + 1]++; }
   colour_offsets.PartialSum();

   Array<int> fill(colour_offsets);
   colour_order.SetSize(NE);
   for (int el = 0; el < NE; el++) { colour_order[fill[colours[el]]++] = el; }

   // Warm whatever the element loop would otherwise build lazily inside the
   // parallel region. These are all const accessors that fill a shared table
   // on first use, and a race there is not a wrong answer but a corrupt one.
   // Cheap insurance, once, and it is why this lives with the colouring
   // rather than beside it.
   if (NE > 0)
   {
      Array<int> dofs, faces, oris;
      switch (mesh->Dimension())
      {
         case 1: mesh->GetElementVertices(0, faces); break;
         case 2: mesh->GetElementEdges(0, faces, oris); break;
         case 3: mesh->GetElementFaces(0, faces, oris); break;
      }
      fes_p.GetElementVDofs(0, dofs);
      GetFDofs(0, dofs);
      for (int f = 0; f < faces.Size(); f++)
      {
         c_fes.GetFaceVDofs(faces[f], dofs);
         c_fes.GetFaceElement(faces[f]);
      }
   }
}

FaceElementTransformations *DarcyHybridization::GetFaceTransformation(
   int f, TransWorkspace &ws) const
{
   // The same three branches as the shared-cache overload above, through the
   // caller-allocated Mesh/ParMesh entry points. Kept as a separate function
   // rather than a defaulted argument so the serial path is provably the one
   // it always was.
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   if (el2 >= 0)
   {
      fes.GetMesh()->GetFaceElementTransformations(f, ws.face, ws.f1, ws.f2);
   }
#ifdef MFEM_USE_MPI
   else if (ParallelC() && c_pfes->GetParMesh()->FaceIsTrueInterior(f))
   {
      c_pfes->GetParMesh()->GetSharedFaceTransformationsByLocalIndex(
         f, ws.face, ws.f1, ws.f2);
   }
#endif
   else
   {
      fes.GetMesh()->GetFaceElementTransformations(f, ws.face, ws.f1, ws.f2, 21);
   }

   return &ws.face;
}

void DarcyHybridization::AssembleCtFaceMatrix(int face,
                                              const DenseMatrix &elmat)
{
   const Mesh *mesh = fes.GetMesh();
   const int num_faces = mesh->GetNumFaces();
   int el1, el2;
   mesh->GetFaceElements(face, &el1, &el2);

   const int hat_size_1 = hat_offsets[el1+1] - hat_offsets[el1];
   const int f_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
   const int c_size = c_fes.GetFaceElement(face)->GetDof() * c_fes.GetVDim();

   int inf1, inf2, nc;
   if (mesh->Nonconforming())
   {
      mesh->GetFaceInfos(face, &inf1, &inf2, &nc);
      MFEM_ASSERT(nc < 0 || el2 >= 0 ||
                  inf2 >= 0, "Master face should not be integrated directly!");
   }
   else
   {
      inf1 = inf2 = nc = -1;
   }

   //el1
   if (nc >= 0 && face >= num_faces)
   {
      // ghost slave
      DenseMatrix Ct_face(f_size_1, c_size);
      AssembleCtSubMatrix(el1, elmat, Ct_face);
      AssembleNCSlaveCtFaceMatrix(face, Ct_face);
      return;
   }
   DenseMatrix Ct_face_1(&Ct_data[Ct_offsets[face]], f_size_1, c_size);
   AssembleCtSubMatrix(el1, elmat, Ct_face_1);

   //el2
   if (el2 >= 0)
   {
      //const int hat_size_2 = hat_offsets[el2+1] - hat_offsets[el2];
      const int f_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];

      if (nc >= 0)
      {
         //interior slave
         DenseMatrix Ct_face(f_size_2, c_size);
         AssembleCtSubMatrix(el2, elmat, Ct_face, hat_size_1);
         AssembleNCSlaveCtFaceMatrix(face, Ct_face);
         return;
      }
      DenseMatrix Ct_face_2(&Ct_data[Ct_offsets[face] + f_size_1*c_size],
                            f_size_2, c_size);
      AssembleCtSubMatrix(el2, elmat, Ct_face_2, hat_size_1);
   }
}

void DarcyHybridization::AssembleCtSubMatrix(int el, const DenseMatrix &elmat,
                                             DenseMatrix &Ct_, int ioff)
{
   const int hat_offset = hat_offsets[el];
   const int hat_size = hat_offsets[el+1] - hat_offset;

   int row = 0;
   for (int i = 0; i < hat_size; i++)
   {
      if (hat_dofs_marker[hat_offset + i] == 1) { continue; }
      bool bzero = true;
      for (int j = 0; j < Ct_.Width(); j++)
      {
         const real_t val = elmat(i + ioff, j);
         if (val == 0.) { continue; }
         Ct_(row, j) = val;
         bzero = false;
      }
      if (!bzero)
      {
         //mark the hat dof as "boundary" if the row is non-zero
         hat_dofs_marker[hat_offset + i] = -1;
      }
      row++;
   }
   MFEM_ASSERT(row == Af_f_offsets[el+1] - Af_f_offsets[el], "Internal error.");
}

void DarcyHybridization::AssembleNCSlaveFaceMatrix(int f,
                                                   face_getter fx_Ct, const DenseMatrix *Ct_,
                                                   face_getter fx_C, const DenseMatrix *C_,
                                                   face_getter fx_H, const DenseMatrix *H_)
{
   const Mesh *mesh = fes.GetMesh();
#ifdef MFEM_DEBUG
   int el1, el2, inf1, inf2, nc;
   mesh->GetFaceElements(f, &el1, &el2);
   mesh->GetFaceInfos(f, &inf1, &inf2, &nc);
   MFEM_ASSERT(nc >= 0 && (el2 >= 0 || inf2 >= 0), "Not a slave face");
#endif

   const int dim = mesh->Dimension();
   const int num_faces = mesh->GetNumFaces();
   auto &nclist = mesh->ncmesh->GetNCList(dim-1);
   const FiniteElementCollection *c_fec = c_fes.FEColl();

   auto find = nclist.GetMeshIdAndType(f);
   MFEM_ASSERT(find.type == NCMesh::NCList::MeshIdType::SLAVE, "Not a slave face");
   const NCMesh::Slave &slave = static_cast<const NCMesh::Slave&>(*find.id);

   if (slave.master >= num_faces) { return; }

#ifdef MFEM_DEBUG
   mesh->GetFaceElements(slave.master, &el1, &el2);
   mesh->GetFaceInfos(slave.master, &inf1, &inf2, &nc);
   MFEM_ASSERT(nc >= 0 && el2 < 0, "Not a master face");
#endif
   const Geometry::Type geom_m = mesh->GetFaceGeometry(slave.master);
   const Geometry::Type geom_s = slave.Geom();

   IsoparametricTransformation T;
   DenseMatrix Ct_m, C_m, H_m, I, Io;

   // compound the master matrix from the slave ones
   if (fx_Ct)
   {
      fx_Ct(slave.master, Ct_m);
   }
   if (fx_C)
   {
      fx_C(slave.master, C_m);
   }
   if (fx_H)
   {
      fx_H(slave.master, H_m);
   }

   const FiniteElement *fe_m = c_fes.GetFaceElement(slave.master);
   switch (geom_m)
   {
      case Geometry::SQUARE:   T.SetFE(&QuadrilateralFE); break;
      case Geometry::TRIANGLE: T.SetFE(&TriangleFE); break;
      case Geometry::SEGMENT:  T.SetFE(&SegmentFE); break;
      default: MFEM_ABORT("unsupported geometry");
   }

   nclist.OrientedPointMatrix(slave, T.GetPointMat());
   const FiniteElement *fe_s = c_fes.GetFaceElement(slave.index);
   fe_s->GetTransferMatrix(*fe_m, T, I);

   // get master/slave orientation and DOFs ordering
   int ori_m = 0, ori_s = 0;

   if (dim == 2)
   {
      // In 2D, DOF ordering follows orientation of the edges, which is not
      // accounted for in the point matrix or face info.
      Array<int> edges_m, oris_m;

      // get master edge/face orientation
      mesh->GetFaceEdges(slave.master, edges_m, oris_m);
      ori_m = oris_m[0] > 0 ? 0 : 1;

      // get slave edge/face orientation
      if (slave.index < num_faces)
      {
         // regular slave
         Array<int> edges_s, oris_s;
         mesh->GetFaceEdges(slave.index, edges_s, oris_s);
         ori_s = oris_s[0] > 0 ? 0 : 1;
      }
      else
      {
         // ghost slave
         int verts[4], edges[4], oris[4];
         mesh->ncmesh->GetFaceVerticesEdges(slave, verts, edges, oris);
         ori_s = oris[0] > 0 ? 0 : 1;

         // check for inverted orientation
         int sinf1, sinf2;
         mesh->GetFaceInfos(slave.index, &sinf1, &sinf2);
         if (Mesh::DecodeFaceInfoOrientation(sinf2)) { ori_s ^= 1; }
      }
   }
   else if (dim == 3)
   {
      // In 3D, DOF ordering is simply given by the face info.
      int minf1, minf2;
      mesh->GetFaceInfos(slave.master, &minf1, &minf2);
      ori_m = Mesh::DecodeFaceInfoOrientation(minf1);

      // get slave face orientation
      int sinf1, sinf2;
      mesh->GetFaceInfos(slave.index, &sinf1, &sinf2);
      ori_s = Mesh::DecodeFaceInfoOrientation(sinf1);

      if (slave.index >= num_faces)
      {
         // check for inverted orientation
         if (Mesh::DecodeFaceInfoOrientation(sinf2)) { ori_s ^= 1; }
      }
   }

   if (ori_m || ori_s)
   {
      // reorder the transfer matrix
      Io.SetSize(I.Height(), I.Width());

      Array<int> dofs_m, dofs_s;
      c_fec->SubDofOrder(geom_m, Geometry::Dimension[geom_m], ori_m, dofs_m);
      c_fec->SubDofOrder(geom_s, Geometry::Dimension[geom_s], ori_s, dofs_s);

      for (int j = 0; j < I.Width(); j++)
         for (int i = 0; i < I.Height(); i++)
         {
            const int io_i = UnsignIndex(dofs_s[i]);
            const int io_j = UnsignIndex(dofs_m[j]);
            bool sign = false;
            if (dofs_s[i] < 0) { sign = !sign; }
            if (dofs_m[j] < 0) { sign = !sign; }
            Io(io_i, io_j) = (sign)?(-I(i,j)):(+I(i,j));
         }
   }
   else
   {
      // no reordering needed
      Io.Reset(I.GetData(), I.Height(), I.Width());
   }

   if (c_fes.GetVDim() > 0)
   {
      const int vdim = c_fes.GetVDim();
      const int dofs_in =  Io.Height();
      const int dofs_out = Io.Width();
      if (fx_Ct)
      {
         const int dofs_el = Ct_->Height();
         DenseMatrix Ct_d(dofs_el, dofs_in);
         DenseMatrix Ct_md(dofs_el, dofs_out);
         for (int d = 0; d < vdim; d++)
         {
            Ct_d.CopyMN(*Ct_, dofs_el, dofs_in, 0, d*dofs_in);
            mfem::Mult(Ct_d, Io, Ct_md);
            Ct_m.AddMatrix(Ct_md, 0, d*dofs_out);
         }
      }
      if (fx_C)
      {
         const int dofs_el = C_->Width();
         DenseMatrix C_d(dofs_in, dofs_el);
         DenseMatrix C_md(dofs_out, dofs_el);
         for (int d = 0; d < vdim; d++)
         {
            C_d.CopyMN(*C_, dofs_in, dofs_el, d*dofs_in, 0);
            mfem::MultAtB(Io, C_d, C_md);
            C_m.AddMatrix(C_md, d*dofs_out, 0);
         }
      }
      if (fx_H)
      {
         DenseMatrix H_d(dofs_in);
         DenseMatrix H_md(dofs_out);
         for (int di = 0; di < vdim; di++)
            for (int dj = 0; dj < vdim; dj++)
            {
               H_d.CopyMN(*H_, dofs_in, dofs_in, di*dofs_in, dj*dofs_in);
               RAP(H_d, Io, H_md);
               H_m.AddMatrix(H_md, di*dofs_out, dj*dofs_out);
            }
      }
   }
   else
   {
      if (fx_Ct)
      {
         mfem::AddMult(*Ct_, Io, Ct_m);
      }
      if (fx_C)
      {
         mfem::AddMultAtB(Io, *C_, C_m);
      }
      if (fx_H)
      {
         DenseMatrix H_ma(H_m.Height(), H_m.Width());
         RAP(*H_, Io, H_ma);
         H_m += H_ma;
      }
   }
}

void DarcyHybridization::AssembleNCSlaveCtFaceMatrix(int face,
                                                     const DenseMatrix &Ct_)
{
   AssembleNCSlaveFaceMatrix(face,
   [this](int f, DenseMatrix &m) { GetCtFaceMatrix(f, 0, m); }, &Ct_);
}

void DarcyHybridization::AssembleNCSlaveEGFaceMatrix(int face,
                                                     const DenseMatrix &E, const DenseMatrix &G)
{
   AssembleNCSlaveFaceMatrix(face,
   [this](int f, DenseMatrix &m) { GetEFaceMatrix(f, 0, m); }, &E,
   [this](int f, DenseMatrix &m) { GetGFaceMatrix(f, 0, m); }, &G);
}

void DarcyHybridization::AssembleNCSlaveHFaceMatrix(int face,
                                                    const DenseMatrix &H)
{
   AssembleNCSlaveFaceMatrix(face,
                             face_getter(), NULL,
                             face_getter(), NULL,
   [this](int f, DenseMatrix &m) { GetHFaceMatrix(f, m); }, &H);
}

void DarcyHybridization::ConstructC()
{
   Mesh *mesh = fes.GetMesh();
   int num_faces = mesh->GetNumFaces();

#if defined(MFEM_USE_DOUBLE)
   constexpr real_t mtol = 1e-12;
#elif defined(MFEM_USE_SINGLE)
   constexpr real_t mtol = 4e-6;
#else
#error "Only single and double precision are supported!"
   constexpr real_t mtol = 1.;
#endif

   // Define Ct_offsets and allocate Ct_data
   Ct_offsets.SetSize(num_faces+1);
   Ct_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      int el1, el2, inf1, inf2, nc = -1;
      mesh->GetFaceElements(f, &el1, &el2);
      if (mesh->Nonconforming())
      {
         mesh->GetFaceInfos(f, &inf1, &inf2, &nc);
      }

      int f_size = Af_f_offsets[el1+1] - Af_f_offsets[el1];
      if (el2 >= 0 && nc < 0)
      {
         f_size += Af_f_offsets[el2+1] - Af_f_offsets[el2];
      }
      const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
      Ct_offsets[f+1] = Ct_offsets[f] + c_size * f_size;
   }

   Ct_data.SetSize(Ct_offsets[num_faces]); Ct_data = 0.;

   // Assemble the constraint element matrices
   if (c_bfi)
   {
      DenseMatrix elmat;

      for (int f = 0; f < num_faces; f++)
      {
         FaceElementTransformations *FTr = mesh->GetInteriorFaceTransformations(f);
         if (!FTr) { continue; }

         const FiniteElement *fe1 = fes.GetFE(FTr->Elem1No);
         const FiniteElement *fe2 = fes.GetFE(FTr->Elem2No);

         c_bfi->AssembleFaceMatrix(*c_fes.GetFaceElement(f),
                                   *fe1, *fe2, *FTr, elmat);
         // zero-out small elements in elmat
         elmat.Threshold(mtol * elmat.MaxMaxNorm());

         // assemble the matrix
         AssembleCtFaceMatrix(f, elmat);
      }

#ifdef MFEM_USE_MPI
      if (ParallelU()) { pfes->ExchangeFaceNbrData(); }
      ParMesh *pmesh = NULL;
      if (pfes) { pmesh = pfes->GetParMesh(); }
      else if (c_pfes) { pmesh = c_pfes->GetParMesh(); }
      const int NE = mesh->GetNE();

      if (pmesh)
      {
         const int num_shared_faces = pmesh->GetNSharedFaces();
         for (int sf = 0; sf < num_shared_faces; sf++)
         {
            const int f = pmesh->GetSharedFace(sf);
            FaceElementTransformations *FTr = pmesh->GetSharedFaceTransformations(sf);

            const FiniteElement *fe1 = fes.GetFE(FTr->Elem1No);
            const FiniteElement *fe2 =
               (pfes)?(pfes->GetFaceNbrFE(FTr->Elem2No - NE)):(fe1);

            c_bfi->AssembleFaceMatrix(*c_fes.GetFaceElement(f),
                                      *fe1, *fe2, *FTr, elmat);
            // zero-out small elements in elmat
            elmat.Threshold(mtol * elmat.MaxMaxNorm());

            // assemble the matrix
            AssembleCtFaceMatrix(f, elmat);
         }
      }
#endif

      if (boundary_constraint_integs.size())
      {
         const FiniteElement *fe1, *fe2;
         const FiniteElement *face_el;

         // Which boundary attributes need to be processed?
         Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
         bdr_attr_marker = 0;
         for (size_t k = 0; k < boundary_constraint_integs.size(); k++)
         {
            if (boundary_constraint_integs_marker[k] == NULL)
            {
               bdr_attr_marker = 1;
               break;
            }
            Array<int> &bdr_marker = *boundary_constraint_integs_marker[k];
            MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                        "invalid boundary marker for boundary face integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < bdr_attr_marker.Size(); i++)
            {
               bdr_attr_marker[i] |= bdr_marker[i];
            }
         }

         for (int i = 0; i < fes.GetNBE(); i++)
         {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            FaceElementTransformations *FTr = mesh->GetBdrFaceTransformations(i);
            if (!FTr) { continue; }

            int iface = mesh->GetBdrElementFaceIndex(i);
            face_el = c_fes.GetFaceElement(iface);
            fe1 = fes.GetFE (FTr -> Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (size_t k = 0; k < boundary_constraint_integs.size(); k++)
            {
               if (boundary_constraint_integs_marker[k] &&
                   (*boundary_constraint_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               boundary_constraint_integs[k]->AssembleFaceMatrix(*face_el, *fe1, *fe2, *FTr,
                                                                 elmat);
               // zero-out small elements in elmat
               elmat.Threshold(mtol * elmat.MaxMaxNorm());

               // assemble the matrix
               AssembleCtFaceMatrix(iface, elmat);
            }
         }
      }
   }
   else
   {
      // Check if c_fes is really needed here.
      MFEM_ABORT("TODO: algebraic definition of C");
   }
}

void DarcyHybridization::AllocD() const
{
   Df_data.SetSize(Df_offsets.Last()); Df_data = 0.;
   Df_ipiv.SetSize(Df_f_offsets.Last());
}

void DarcyHybridization::AllocEG() const
{
   Mesh *mesh = fes.GetMesh();
   const int num_faces = mesh->GetNumFaces();

   // Define E_offsets and allocate E_data and G_data
   E_offsets.SetSize(num_faces+1);
   E_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      int el1, el2, inf1, inf2, nc = -1;
      mesh->GetFaceElements(f, &el1, &el2);
      if (mesh->Nonconforming())
      {
         mesh->GetFaceInfos(f, &inf1, &inf2, &nc);
      }

      int d_size = Df_f_offsets[el1+1] - Df_f_offsets[el1];
      if (el2 >= 0 && nc < 0)
      {
         d_size += Df_f_offsets[el2+1] - Df_f_offsets[el2];
      }
      const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
      E_offsets[f+1] = E_offsets[f] + c_size * d_size;
   }

   E_data.SetSize(E_offsets.Last()); E_data = 0.;
   G_data.SetSize(G_offsets.Last()); G_data = 0.;
}

void DarcyHybridization::AllocH() const
{
   Mesh *mesh = fes.GetMesh();
   int num_faces = mesh->GetNumFaces();

   // Define E_offsets and allocate E_data and G_data
   H_offsets.SetSize(num_faces+1);
   H_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
      H_offsets[f+1] = H_offsets[f] + c_size * c_size;
   }

   H_data.SetSize(H_offsets[num_faces]); H_data = 0.;
}

int DarcyHybridization::UniformBlockSize(const Array<int> &f_offsets, int NE)
{
   if (NE <= 0) { return -1; }

   const int n = f_offsets[1] - f_offsets[0];
   for (int el = 1; el < NE; el++)
   {
      if (f_offsets[el+1] - f_offsets[el] != n) { return -1; }
   }
   return n;
}

bool DarcyHybridization::CanBatchLocalFactor() const
{
   const int NE = fes.GetNE();

   // A zero common size is uniform and vacuous -- an absent D block, say --
   // so it is the -1 of "not all equal" that disqualifies, not a size of 0.
   return UniformBlockSize(Af_f_offsets, NE) >= 0 &&
          UniformBlockSize(Df_f_offsets, NE) >= 0;
}

void DarcyHybridization::SetLocalFactorMode(LocalFactorMode mode)
{
   lfac_mode = mode;
}

void DarcyHybridization::InvertA()
{
   const int NE = fes.GetNE();

   const int n = (lfac_mode == LocalFactorMode::Batched)
                 ? UniformBlockSize(Af_f_offsets, NE) : -1;
   if (n > 0)
   {
      // Af_data already has the memory order DenseTensor wants -- NE
      // contiguous n*n blocks, Af_offsets[el] == el*n*n -- and the tensor is
      // a non-owning view of it, so this factors the same array in place.
      // Af_ipiv is likewise el*n, which is the (n, NE) shape LUFactor writes.
      //
      // NewMemoryAndSize AND NOT THE RAW-POINTER CONSTRUCTOR, which is what
      // decides whether this can run on a device at all. DenseTensor(real_t*,
      // ...) goes through Memory::Wrap() and sets VALID_HOST with no device
      // type, so the batched backend -- which is an mfem::forall over
      // ReadWrite() -- would be pinned to the host however the Device is
      // configured. Passing the Memory carries its device state instead, and
      // own_mem is false because Af_data owns it.
      DenseTensor A;
      A.NewMemoryAndSize(Af_data.GetMemory(), n, n, NE, false);
      BatchedLinAlg::LUFactor(A, Af_ipiv);
      // The factors are Af_data's own memory, so whichever side the backend
      // left valid is the side Af_data now reports; the host readers below
      // and in MultInv() go through Array::operator[], which does not sync.
      //
      // Af_ipiv is the sharper half of that and is worth stating: it is
      // written through Array::Write(), whose default is on_dev = true, so on
      // a device it comes back device-valid, and LUFactors uses its entries
      // as ARRAY INDICES (ipiv[i] - 1). A host reader would index on
      // uninitialised memory, which a synthetic probe duly segfaults on.
      //
      // **THAT USED TO SAY NOTHING REACHES IT, AND THAT IS WITHDRAWN.** The
      // argument was: InvertA() runs only for LocalOpType::PotNL and FluxNL,
      // whose local solves are MultInvNL(), and a nonlinear local solve does
      // not run with a device configured at all -- it aborts in LBFGSSolver
      // on a NaN. Both halves are true and the conclusion does not follow,
      // because NPC has no local nonlinear solve: it reaches the host
      // MultInv() directly, on a PotNL problem, and indexes these pivots
      // there. Measured -- the NPC case in
      // tests/unit/miniapps/test_debug_device.cpp faults on exactly this
      // page under Device("debug"), and HostIsValid() reports 1 while it
      // does.
      //
      // SyncLocalBlocksToHost() is what closes it, called from MultNL() and
      // ComputeH(). The shape of the mistake is the one this branch keeps
      // finding: "X is unguarded" and "something reaches the gap" are two
      // claims, and the second was argued from one caller instead of
      // enumerated.
      Af_data.GetMemory().Sync(A.GetMemory());
      return;
   }

   for (int el = 0; el < NE; el++)
   {
      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];

      // Decompose A

      LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);

      LU_A.Factor(a_dofs_size);
   }
}

void DarcyHybridization::InvertD()
{
   const int NE = fes.GetNE();

#ifdef MFEM_DEBUG
   // Checked for every element whichever way the factorisation is done, so
   // that the two modes fail on the same inputs as well as agreeing on the
   // rest.
   for (int el = 0; el < NE; el++)
   {
      const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
      DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
      const real_t norm = D.MaxMaxNorm();
      if (norm == 0.)
      {
         MFEM_ABORT("Inverting an empty matrix!");
      }
      if (D.Rank(norm * 1e-12) < d_dofs_size)
      {
         MFEM_ABORT("Inverting a singular matrix!");
      }
   }
#endif

   const int n = (lfac_mode == LocalFactorMode::Batched)
                 ? UniformBlockSize(Df_f_offsets, NE) : -1;
   if (n > 0)
   {
      // Carrying the Memory rather than a raw pointer, for the reason on
      // InvertA(): the raw-pointer constructor pins the batched backend to
      // the host whatever the Device is configured as.
      DenseTensor D;
      D.NewMemoryAndSize(Df_data.GetMemory(), n, n, NE, false);
      BatchedLinAlg::LUFactor(D, Df_ipiv);
      Df_data.GetMemory().Sync(D.GetMemory());
      return;
   }

   for (int el = 0; el < NE; el++)
   {
      int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

      // Decompose D

      LUFactors LU_D(&Df_data[Df_offsets[el]], &Df_ipiv[Df_f_offsets[el]]);

      LU_D.Factor(d_dofs_size);
   }
}

void DarcyHybridization::GetElementFaces(int el, Array<int> &faces) const
{
   const Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();

#ifdef MFEM_THREAD_SAFE
   Array<int> oris;
#else
   static Array<int> oris;
#endif

   switch (dim)
   {
      case 1:
         mesh->GetElementVertices(el, faces);
         break;
      case 2:
         mesh->GetElementEdges(el, faces, oris);
         break;
      case 3:
         mesh->GetElementFaces(el, faces, oris);
         break;
   }
}

bool DarcyHybridization::GetBnlMatrix(int el, DenseMatrix &Bnl) const
{
   if (Bnl_empty || Bnl_data.Size() != Bf_offsets.Last()) { return false; }

   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   Bnl.UseExternalData(const_cast<real_t*>(&Bnl_data[Bf_offsets[el]]),
                       a_dofs_size, d_dofs_size);
   return true;
}

int DarcyHybridization::GetElementTraceSize(const Array<int> &faces) const
{
   int size = 0;
   for (int f = 0; f < faces.Size(); f++)
   {
      size += c_fes.GetFaceElement(faces[f])->GetDof() * c_fes.GetVDim();
   }
   return size;
}

int DarcyHybridization::AssemblyChunkSize(int NE) const
{
   // The chunk bounds the block buffer, so it must not grow with the mesh:
   // an order-2 hex has 54 trace dofs and therefore a 23 kB block, which at
   // one buffer for the whole mesh would be gigabytes on a mesh worth
   // threading. It must still be long enough that the serial scatter between
   // chunks is not the synchronisation point.
   int chunk = 256;
#ifdef MFEM_USE_OPENMP
   if (asm_mode == AssemblyMode::Threaded)
   {
      chunk = std::max(chunk, 8 * omp_get_max_threads());
   }
#endif
   return std::min(chunk, std::max(NE, 1));
}

/** @brief @a Bt <- @a sgn times the transpose of each of @a NE blocks of @a B,
    which are (@a nd, @a na) column-major and become (@a na, @a nd).

    A free function and not a member, and nvcc is why: an extended
    __host__ __device__ lambda may not be defined inside a member function with
    private or protected access, which FactorElementsBatched() has. The other
    batched kernels in fem/darcy are namespace-scope functions for the same
    reason; see bilininteg_hdg.cpp. */
void TransposeBlocksScaled(const Vector &B, int na, int nd, int NE,
                           real_t sgn, Vector &Bt)
{
   const int nb = na*nd;
   const auto d_B = B.Read();
   auto d_Bt = Bt.Write();
   // One thread per ELEMENT and the block transposed by inner loops. A thread
   // per entry needs three integer divisions to recover (i, j, el) and that
   // was measurable -- see HDGGatherFaceCols(), where the same shape of
   // kernel cost 0.19 s of a 0.95 s loop before it was written this way.
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int el)
   {
      const real_t *b = d_B + el*nb;
      real_t *bt = d_Bt + el*nb;
      // B's block is (nd, na) column-major, so B(j, i) sits at i*nd + j.
      for (int j = 0; j < nd; j++)
      {
         for (int i = 0; i < na; i++) { bt[j*na + i] = sgn * b[i*nd + j]; }
      }
   });
}

/** @brief Gather one (@a m, @a nc) column-major block per (element, local
    face) into the element-blocked (@a m, @a nf*nc, @a nel) tensor @a dst.

    @a map carries three offsets per (element, local face) -- into Ct_data,
    into E_data/G_data, and into H_data -- and @a comp picks which. It is
    built once for the whole mesh, so element @a e of the chunk is element
    @a el_0 + e of that map; that is what keeps it one array rather than one
    per chunk.

    A free function and not a member, for the nvcc reason spelled out on
    TransposeBlocksScaled(). */
void HDGGatherFaceCols(const Vector &src, const Array<int> &map, int comp,
                       int m, int nc, int nf, int el_0, int nel, Vector &dst)
{
   const int T = nf*nc;
   const auto d_src = src.Read();
   const auto d_map = map.Read();
   auto d_dst = dst.Write();
   // One thread per (element, face) and the block copied by inner loops, NOT
   // one thread per entry. The flat form needs four integer divisions per
   // output entry to recover (i, j, lf, e), and integer division is not a
   // cheap instruction: measured, that alone was 0.19 s of a 0.95 s face-pair
   // loop at n=128, for a copy that moves 28 MB. This leaves two divisions
   // per BLOCK and the copy contiguous on both sides.
   mfem::forall(nf*nel, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int lf = idx % nf;
      const int e  = idx / nf;
      const real_t *src = d_src + d_map[3*((el_0 + e)*nf + lf) + comp];
      real_t *dst = d_dst + e*m*T + lf*nc*m;
      for (int j = 0; j < nc; j++)
      {
         for (int i = 0; i < m; i++) { dst[j*m + i] = src[j*m + i]; }
      }
   });
}

/** @brief The same gather for a block stacked by ROWS: one (@a nc, @a n)
    block per (element, local face) into an (@a nf*nc, @a n, @a nel) tensor.

    G is the one stored that way -- (trace dofs, potential dofs) rather than
    the (potential dofs, trace dofs) of E -- so its element block is a column
    of face blocks where E's is a row of them. */
void HDGGatherFaceRows(const Vector &src, const Array<int> &map, int comp,
                       int nc, int n, int nf, int el_0, int nel, Vector &dst)
{
   const int T = nf*nc;
   const auto d_src = src.Read();
   const auto d_map = map.Read();
   auto d_dst = dst.Write();
   mfem::forall(nf*nel, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int lf = idx % nf;
      const int e  = idx / nf;
      const real_t *src = d_src + d_map[3*((el_0 + e)*nf + lf) + comp];
      real_t *dst = d_dst + e*T*n + lf*nc;
      for (int j = 0; j < n; j++)
      {
         for (int i = 0; i < nc; i++) { dst[j*T + i] = src[j*nc + i]; }
      }
   });
}

/** @brief Add each face's own (@a nc, @a nc) block to the matching DIAGONAL
    block of the element's (T, T) matrix, T = @a nf*nc.

    An offset of -1 says this element is not the face's first, which is how
    the element loop states "integrate the face contribution only on one
    side". */
void HDGAddFaceDiagBlocks(const Vector &src, const Array<int> &map, int comp,
                          int nc, int nf, int el_0, int nel, Vector &dst)
{
   const int T = nf*nc;
   const auto d_src = src.Read();
   const auto d_map = map.Read();
   auto d_dst = dst.ReadWrite();
   mfem::forall(nf*nel, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int lf = idx % nf;
      const int e  = idx / nf;
      const int o  = d_map[3*((el_0 + e)*nf + lf) + comp];
      if (o < 0) { return; }
      const real_t *src = d_src + o;
      real_t *dst = d_dst + e*T*T + lf*nc*T + lf*nc;
      for (int j = 0; j < nc; j++)
      {
         for (int i = 0; i < nc; i++) { dst[j*T + i] += src[j*nc + i]; }
      }
   });
}

/** @brief Repack the element's (T, T) matrix into the (f2, f1) BLOCK layout
    ComputeElementH() leaves behind and ScatterElementH() replays: f1 outer,
    f2 inner, each block contiguous and column-major.

    The two layouts differ -- the matrix runs whole columns of length T where
    the buffer runs one face block at a time -- so this is a real permutation
    and not a reinterpretation. Written as a kernel over the BUFFER, whose
    linear index decomposes into (i2, j1, f2, f1, e) with exactly the strides
    the buffer has; only the read is scattered. */
void HDGPackElementH(const Vector &Hfull, int nc, int nf, int nel,
                     Vector &Hel)
{
   const int T = nf*nc;
   const auto d_H = Hfull.Read();
   auto d_buf = Hel.Write();
   mfem::forall(nf*nf*nel, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int f2 = idx % nf;
      const int f1 = (idx / nf) % nf;
      const int e  = idx / (nf*nf);
      const real_t *src = d_H + e*T*T + f1*nc*T + f2*nc;
      real_t *dst = d_buf + e*T*T + (f1*nf + f2)*nc*nc;
      for (int j1 = 0; j1 < nc; j1++)
      {
         for (int i2 = 0; i2 < nc; i2++)
         { dst[j1*nc + i2] = src[j1*T + i2]; }
      }
   });
}

bool DarcyHybridization::FactorElementsBatched(ComputeHMode mode,
                                               Vector &AiBt_all) const
{
   if (lfac_mode != LocalFactorMode::Batched) { return false; }

   const int NE = fes.GetNE();
   const int na = UniformBlockSize(Af_f_offsets, NE);
   const int nd = UniformBlockSize(Df_f_offsets, NE);
   // The same STORAGE conditions CanBatchLocalSolve() asks for, and for the
   // same reason: what the DenseTensor views need is one block size and the
   // el*n*n layout, not a particular local operator. A zero size is uniform
   // and degenerate -- there is no Schur complement to form without a D.
   if (na <= 0 || nd <= 0) { return false; }
   if (Af_data.Size() != Af_offsets.Last() ||
       Df_data.Size() != Df_offsets.Last() ||
       Bf_data.Size() != Bf_offsets.Last()) { return false; }

   const bool gradient = (mode != ComputeHMode::Linear);
   // Where the Schur complement goes -- the question ComputeElementH() asks,
   // and for the reason written there: in FluxNL, Df_data holds the factored
   // LINEAR potential mass that the local solve needs, so the complement goes
   // to Sf_data instead.
   const bool to_S = (gradient && lop_type == LocalOpType::FluxNL);
   if (to_S && (Sf_data.Size() != Df_data.Size() ||
                Df_lin_data.Size() != Df_data.Size())) { return false; }

   // Decompose A. NewMemoryAndSize and not the raw-pointer constructor, for
   // the reason spelled out in InvertA().
   DenseTensor A;
   A.NewMemoryAndSize(Af_data.GetMemory(), na, na, NE, false);
   if (!gradient || lop_type != LocalOpType::PotNL)
   {
      BatchedLinAlg::LUFactor(A, Af_ipiv);
      Af_data.GetMemory().Sync(A.GetMemory());
   }

   // AiBt = A^-1 times the negated (0,1) block, one element's (na, nd) block
   // at Bf_offsets[el] -- the same slot B fills transposed, and the shape Bnl
   // is already stored in.
   AiBt_all.SetSize(na*nd*NE);
   AiBt_all.UseDevice(true);
   TransposeBlocksScaled(Bf_data, na, nd, NE, (bsym)?(1.):(-1.), AiBt_all);
   if (gradient && !Bnl_empty && Bnl_data.Size() == Bf_offsets.Last())
   {
      // The guard GetBnlMatrix() applies per element, applied once -- neither
      // half of it depends on the element.
      Vector Bnl_v;
      Bnl_v.NewMemoryAndSize(Bnl_data.GetMemory(), Bnl_data.Size(), false);
      Bnl_v.UseDevice(true);
      AiBt_all -= Bnl_v;
   }
   BatchedLinAlg::LUSolve(A, Af_ipiv, AiBt_all);

   // Construct and decompose the Schur complement
   DenseTensor B;
   B.NewMemoryAndSize(Bf_data.GetMemory(), nd, na, NE, false);
   Vector &S_store = (to_S)?(Sf_data):(Df_data);
   Array<int> &S_ipiv = (to_S)?(Sf_ipiv):(Df_ipiv);
   if (to_S)
   {
      Vector D_lin;
      D_lin.NewMemoryAndSize(Df_lin_data.GetMemory(), Df_lin_data.Size(),
                             false);
      D_lin.UseDevice(true);
      Sf_data.UseDevice(true);
      Sf_data = D_lin;
   }
   Vector S_v;
   S_v.NewMemoryAndSize(S_store.GetMemory(), S_store.Size(), false);
   S_v.UseDevice(true);
   // beta = 1, which is y + A x and not a fused subtraction; see
   // MultInvBatched() for why that distinction is worth stating.
   BatchedLinAlg::AddMult(B, AiBt_all, S_v, 1.0, 1.0);
   S_store.GetMemory().Sync(S_v.GetMemory());

   DenseTensor S;
   S.NewMemoryAndSize(S_store.GetMemory(), nd, nd, NE, false);
   BatchedLinAlg::LUFactor(S, S_ipiv);
   S_store.GetMemory().Sync(S.GetMemory());

   return true;
}

bool DarcyHybridization::BuildElementHFaceMap(int na, int nd, bool with_h,
                                              int &nf, int &nc,
                                              Array<int> &face_map) const
{
   const Mesh *mesh = fes.GetMesh();
   const int NE = mesh->GetNE();
   if (NE <= 0) { return false; }

   const int vdim = c_fes.GetVDim();

   Array<int> faces;
   GetElementFaces(0, faces);
   nf = faces.Size();
   if (nf <= 0) { return false; }
   nc = c_fes.GetFaceElement(faces[0])->GetDof() * vdim;
   if (nc <= 0) { return false; }

   face_map.SetSize(3*NE*nf);
   for (int el = 0; el < NE; el++)
   {
      GetElementFaces(el, faces);
      if (faces.Size() != nf) { return false; }
      for (int lf = 0; lf < nf; lf++)
      {
         const int f = faces[lf];
         if (c_fes.GetFaceElement(f)->GetDof() * vdim != nc) { return false; }

         int el1, el2;
         mesh->GetFaceElements(f, &el1, &el2);
         // The element loop's own test, and it is a test on the FACE rather
         // than on the element: side 0 is the face's first element. A shared
         // face of a ParMesh has el2 < 0 and its local element is always the
         // first, so nothing here is serial-only.
         const int side = (el1 != el) ? 1 : 0;

         const int k = 3*(el*nf + lf);
         face_map[k    ] = Ct_offsets[f] + side*na*nc;
         // E and G share E_offsets, and with one block size their second
         // side sits the same distance in: E is (nd, nc) and G is (nc, nd).
         face_map[k + 1] = E_offsets[f] + side*nd*nc;
         face_map[k + 2] = (with_h && side == 0) ? H_offsets[f] : -1;
      }
   }

   face_map.UseDevice(true);
   return true;
}

void DarcyHybridization::ComputeElementsHBatched(
   ComputeHMode mode, int el_0, int nel, int na, int nd, int nf, int nc,
   const Vector &AiBt_all, const Array<int> &face_map,
   ElementHWorkspace &ws, Vector &Hel) const
{
   const bool gradient = (mode != ComputeHMode::Linear);
   const bool with_eg = (c_bfi_p || mode == ComputeHMode::Gradient);
   // Where the Schur complement was left; the same question, and the same
   // answer, as FactorElementsBatched().
   const bool to_S = (gradient && lop_type == LocalOpType::FluxNL);
   const int T = nf*nc;

   if (nel <= 0 || T <= 0) { return; }

   // Chunk views of the element-blocked stores. An ALIAS Memory and not a raw
   // pointer, for the reason InvertA() gives at length: the raw-pointer
   // DenseTensor constructor goes through Memory::Wrap(), which sets
   // VALID_HOST with no device type, and that pins the batched kernels to the
   // host however the Device is configured.
   Memory<real_t> A_mem(Af_data.GetMemory(), el_0*na*na, nel*na*na);
   DenseTensor A;
   A.NewMemoryAndSize(A_mem, na, na, nel, false);
   Array<int> A_ipiv;
   A_ipiv.MakeRef(Af_ipiv.GetMemory(), el_0*na, nel*na);

   Memory<real_t> B_mem(Bf_data.GetMemory(), el_0*na*nd, nel*na*nd);
   DenseTensor B;
   B.NewMemoryAndSize(B_mem, nd, na, nel, false);

   Vector &S_store = (to_S) ? (Sf_data) : (Df_data);
   Array<int> &S_store_ipiv = (to_S) ? (Sf_ipiv) : (Df_ipiv);
   Memory<real_t> S_mem(S_store.GetMemory(), el_0*nd*nd, nel*nd*nd);
   DenseTensor S;
   S.NewMemoryAndSize(S_mem, nd, nd, nel, false);
   Array<int> S_ipiv;
   S_ipiv.MakeRef(S_store_ipiv.GetMemory(), el_0*nd, nel*nd);

   Memory<real_t> AiBt_mem(AiBt_all.GetMemory(), el_0*na*nd, nel*na*nd);
   Vector AiBt;
   AiBt.NewMemoryAndSize(AiBt_mem, nel*na*nd, false);
   AiBt.UseDevice(true);

   // The caller's scratch, and SetSize() on a Vector that is already big
   // enough is free; see ElementHWorkspace for what making these locals cost.
   Vector &Ct_el = ws.Ct, &AiCt = ws.AiCt, &BAiCt = ws.BAiCt,
           &CAiBt = ws.CAiBt, &Hfull = ws.Hfull;
   Ct_el.SetSize(na*T*nel);
   AiCt.SetSize(na*T*nel);
   BAiCt.SetSize(nd*T*nel);
   CAiBt.SetSize(T*nd*nel);
   Hfull.SetSize(T*T*nel);

   HDGGatherFaceCols(Ct_data, face_map, 0, na, nc, nf, el_0, nel, Ct_el);

   // A^-1 C^T, for every face of every element of the chunk in one solve.
   AiCt = Ct_el;
   BatchedLinAlg::LUSolve(A, A_ipiv, AiCt);

   // S^-1 (B A^-1 C^T - E). The subtraction is its own pass and is NOT folded
   // into the AddMult's beta: AddMult scales y before accumulating, so a
   // pre-loaded -E would have the product summed on top of it, where the
   // element loop sums the product and then subtracts. See MultInvBatched().
   BatchedLinAlg::AddMult(B, AiCt, BAiCt, 1.0, 0.0);
   if (with_eg)
   {
      Vector &E_el = ws.EG;
      E_el.SetSize(nd*T*nel);
      HDGGatherFaceCols(E_data, face_map, 1, nd, nc, nf, el_0, nel, E_el);
      BAiCt -= E_el;
   }
   BatchedLinAlg::LUSolve(S, S_ipiv, BAiCt);

   // The tensor views of the two products come after the writes that fill
   // them, and that is load-bearing on a device: a DenseTensor holds a COPY
   // of the Memory, so it carries the validity flags as they stood when it
   // was made. FactorElementsBatched() states the same thing from the other
   // end, where it has to Sync() back.
   DenseTensor Ct_t;
   Ct_t.NewMemoryAndSize(Ct_el.GetMemory(), na, T, nel, false);

   // -C A^-1 C^T
   BatchedLinAlg::AddMult(Ct_t, AiCt, Hfull, 1.0, 0.0, BatchedLinAlg::Op::T);
   Hfull.Neg();

   // C A^-1 B^T + G
   BatchedLinAlg::AddMult(Ct_t, AiBt, CAiBt, 1.0, 0.0, BatchedLinAlg::Op::T);
   if (with_eg)
   {
      Vector &G_el = ws.EG;
      G_el.SetSize(T*nd*nel);
      HDGGatherFaceRows(G_data, face_map, 1, nc, nd, nf, el_0, nel, G_el);
      CAiBt += G_el;
   }

   DenseTensor CAiBt_t;
   CAiBt_t.NewMemoryAndSize(CAiBt.GetMemory(), T, nd, nel, false);
   BatchedLinAlg::AddMult(CAiBt_t, BAiCt, Hfull, 1.0, 1.0);

   if (mode == ComputeHMode::Gradient)
   {
      HDGAddFaceDiagBlocks(H_data, face_map, 2, nc, nf, el_0, nel, Hfull);
   }

   HDGPackElementH(Hfull, nc, nf, nel, Hel);
}

/** @brief ScatterElementH()'s scratch: this element's face list, its faces'
    trace vdofs gathered once end to end, and two Array views onto them.

    **The face vdofs were being asked for nf*(nf+1) times per element** -- once
    in the f1 loop and once per f2 inside it -- and each ask is two four-byte
    malloc/free pairs, because FiniteElementSpace::GetFaceDofs() declares
    `Array<int> V, E, Eo;` fresh and in 2D Mesh::GetFaceEdges() does
    `edges.SetSize(1)` on them. Measured with DHAT on `convdiff -p 6 -nl -dg
    -hb -npc` at 256 elements: 21,504 blocks of 90,949, the single largest
    allocation site in an HDG solve, at 4 bytes each. Gathered once per
    element it is nf asks instead of nf*(nf+1). */
struct DarcyHybridization::SerialHWorkspace
{
   Array<int> faces;    ///< the element's faces
   Array<int> dofs;     ///< every face's trace vdofs, end to end
   Array<int> offs;     ///< where each face's block starts in @a dofs
   Array<int> v1, v2;   ///< views onto @a dofs; see the &rows != &cols note
   Array<int> scratch;  ///< GetFaceVDofs()'s destination during the gather
   /// ComputeElementH()'s dense temporaries, four per element before they
   /// were hoisted; DenseMatrix::SetSize() does not shrink, so they size once.
   DenseMatrix AiBt, AiCt, BAiCt, CAiBt;
};

void DarcyHybridization::ComputeElementH(int el, ComputeHMode mode,
                                         real_t *Hel,
                                         SerialHWorkspace &ws,
                                         const Vector *AiBt_all) const
{
   const bool assemble = (mode != ComputeHMode::GradientFactorOnly);
   const bool gradient = (mode != ComputeHMode::Linear);

   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

   // FactorElementsBatched() already did everything down to the Schur
   // complement's LU, for every element at once; all that is wanted here is
   // the views onto what it left.
   const bool prefactored = (AiBt_all != NULL);

   // Decompose A
   LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);
   if (!prefactored && (!gradient || lop_type != LocalOpType::PotNL))
   {
      LU_A.Factor(a_dofs_size);
   }

   // Construct Schur complement
   const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                       d_dofs_size, a_dofs_size);
   DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
   // From @a ws and not a local, here and for the three below: see
   // SerialHWorkspace. Four DenseMatrix objects per element per gradient,
   // 2,048 malloc/free pairs and 1.1 MB on a 256-element problem.
   DenseMatrix &AiBt = ws.AiBt;

   // AiBt is A^-1 times the negated (0,1) block, which everything below
   // -- the Schur complement and the C A^-1 B^T + G product -- is built
   // from. The (0,1) block is -/+B^T from the linear divergence form plus,
   // for a solution-dependent flux law, d(flux residual)/dp; subtracting
   // the latter here is what puts it into both.
   if (prefactored)
   {
      AiBt.UseExternalData(
         const_cast<real_t*>(AiBt_all->GetData()) + Bf_offsets[el],
         a_dofs_size, d_dofs_size);
   }
   else
   {
      AiBt.SetSize(a_dofs_size, d_dofs_size);
      AiBt.Transpose(B);
      if (!bsym) { AiBt.Neg(); }
      DenseMatrix Bnl;
      if (gradient && GetBnlMatrix(el, Bnl))
      {
         AiBt -= Bnl;
      }
      LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());
   }

   LUFactors LU_S;
   if (!gradient || lop_type != LocalOpType::FluxNL)
   {
      if (!prefactored)
      {
         mfem::AddMult(B, AiBt, D);
      }

      // Decompose Schur complement
      LU_S.data = D.GetData();
      LU_S.ipiv = &Df_ipiv[Df_f_offsets[el]];
      if (!prefactored) { LU_S.Factor(d_dofs_size); }
   }
   else
   {
      // FluxNL: Df_data holds the FACTORED LINEAR potential mass, which
      // LocalFluxNLOperator::SolveP() needs on every local iteration, so the
      // Schur complement cannot go there. It goes to Sf_data, which exists
      // for exactly this and which MultInv() reads whenever it applies the
      // Jacobian's blocks.
      //
      // It used to be built into a function-local temporary and discarded.
      // That made the matrix-free gradient unrepresentable in this mode --
      // hence a refusal that named GradientMode::MatrixFree, which was the
      // symptom rather than the cause -- and left MultInv() reading the
      // potential mass where it expected a Schur complement. With the default
      // Assembled gradient nothing complained and the answer was silently
      // wrong; see HDG-ORDERING-API.md finding 3.
      MFEM_VERIFY(Sf_data.Size() == Df_data.Size(),
                  "FluxNL Schur storage was not allocated; it is sized in "
                  "Finalize() where lop_type is decided.");
      DenseMatrix S_el(&Sf_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
      if (!prefactored)
      {
         const DenseMatrix D_lin(&Df_lin_data[Df_offsets[el]],
                                 d_dofs_size, d_dofs_size);
         S_el = D_lin;
         mfem::AddMult(B, AiBt, S_el);
      }

      // Decompose Schur complement
      LU_S.data = S_el.GetData();
      LU_S.ipiv = &Sf_ipiv[Df_f_offsets[el]];
      if (!prefactored) { LU_S.Factor(d_dofs_size); }
   }

   if (!assemble) { return; }

   Array<int> &faces = ws.faces;
   GetElementFaces(el, faces);

   DenseMatrix &AiCt = ws.AiCt, &BAiCt = ws.BAiCt, &CAiBt = ws.CAiBt;
   DenseMatrix H_l;   ///< a view, via UseExternalData(): never allocates
   real_t *Hp = Hel;

   // Mult C^T
   for (int f1 = 0; f1 < faces.Size(); f1++)
   {
      int el1_1, el1_2;
      fes.GetMesh()->GetFaceElements(faces[f1], &el1_1, &el1_2);
      DenseMatrix Ct1;
      GetCtFaceMatrix(faces[f1], el1_1 != el, Ct1);

      //A^-1 C^T
      AiCt.SetSize(Ct1.Height(), Ct1.Width());
      AiCt = Ct1;
      LU_A.Solve(Ct1.Height(), Ct1.Width(), AiCt.GetData());

      //S^-1 (B A^-1 C^T - E)
      BAiCt.SetSize(B.Height(), Ct1.Width());
      mfem::Mult(B, AiCt, BAiCt);

      if (c_bfi_p || mode == ComputeHMode::Gradient)
      {
         DenseMatrix E;
         GetEFaceMatrix(faces[f1], el1_1 != el, E);

         BAiCt -= E;
      }

      LU_S.Solve(BAiCt.Height(), BAiCt.Width(), BAiCt.GetData());

      for (int f2 = 0; f2 < faces.Size(); f2++)
      {
         int el2_1, el2_2;
         fes.GetMesh()->GetFaceElements(faces[f2], &el2_1, &el2_2);
         DenseMatrix Ct2;
         GetCtFaceMatrix(faces[f2], el2_1 != el, Ct2);

         // The block lands in the buffer rather than in H_ directly; the
         // arithmetic below is unchanged, and ScatterElementH() walks the
         // same two loops in the same order to add them.
         H_l.UseExternalData(Hp, Ct2.Width(), Ct1.Width());

         //- C A^-1 C^T
         mfem::MultAtB(Ct2, AiCt, H_l);
         H_l.Neg();

         //(C A^-1 B^T + G) S^-1 (B A^-1 C^T - E)
         CAiBt.SetSize(Ct2.Width(), B.Height());
         mfem::MultAtB(Ct2, AiBt, CAiBt);

         if (c_bfi_p || mode == ComputeHMode::Gradient)
         {
            DenseMatrix G;
            GetGFaceMatrix(faces[f2], el2_1 != el, G);

            CAiBt += G;
         }

         mfem::AddMult(CAiBt, BAiCt, H_l);

         if (f1 == f2)
         {
            //integrate the face contrbution only on one (first) side
            if (mode == ComputeHMode::Gradient && el2_1 == el)
            {
               DenseMatrix H_f;
               GetHFaceMatrix(faces[f1], H_f);
               H_l += H_f;
            }
         }

         Hp += Ct2.Width() * Ct1.Width();
      }
   }
}


void DarcyHybridization::ScatterElementH(int el, const real_t *Hel,
                                         SparseMatrix &H_,
                                         SerialHWorkspace &ws) const
{
   const int skip_zeros = 1;

   Array<int> &faces = ws.faces;
   GetElementFaces(el, faces);

   // The gather, once. Note ws.dofs must not reallocate between the MakeRef()
   // views below and their use, which is why it is filled completely first.
   const int nf = faces.Size();
   ws.offs.SetSize(nf + 1);
   ws.dofs.SetSize(0);
   ws.offs[0] = 0;
   for (int f = 0; f < nf; f++)
   {
      c_fes.GetFaceVDofs(faces[f], ws.scratch);
      ws.dofs.Append(ws.scratch);
      ws.offs[f+1] = ws.dofs.Size();
   }

   const real_t *Hp = Hel;
   DenseMatrix H_l;

   for (int f1 = 0; f1 < nf; f1++)
   {
      Array<int> &c_dofs_1 = ws.v1;
      c_dofs_1.MakeRef(ws.dofs.GetData() + ws.offs[f1],
                       ws.offs[f1+1] - ws.offs[f1]);

      for (int f2 = 0; f2 < nf; f2++)
      {
         Array<int> &c_dofs_2 = ws.v2;
         c_dofs_2.MakeRef(ws.dofs.GetData() + ws.offs[f2],
                          ws.offs[f2+1] - ws.offs[f2]);

         H_l.UseExternalData(const_cast<real_t*>(Hp), c_dofs_2.Size(),
                             c_dofs_1.Size());

         if (f1 == f2)
         {
            // Both index arrays are one object here, as they were when this
            // was a single loop, and that is load-bearing: AddSubMatrix()
            // reads &rows != &cols to decide whether skip_zeros may drop an
            // entry whose transpose is nonzero. Passing two equal but
            // distinct arrays would give H_ a different sparsity pattern.
            H_.AddSubMatrix(c_dofs_1, c_dofs_1, H_l, skip_zeros);
         }
         else
         {
            H_.AddSubMatrix(c_dofs_2, c_dofs_1, H_l, skip_zeros);
         }

         Hp += c_dofs_2.Size() * c_dofs_1.Size();
      }
   }

   MFEM_ASSERT(Hp - Hel == GetElementTraceSize(faces) *
               GetElementTraceSize(faces),
               "element H block buffer over- or under-run");
}

/** @brief Add one chunk's element blocks into the CSR data array.

    @a side selects the pass: 0 assigns, 1 accumulates. One thread per
    (element, f1, f2) BLOCK with the (i2, j1) loops inside, which leaves two
    integer divisions per thread rather than four per entry -- the arithmetic
    that was 40% of ComputeElementsHBatched()'s first version.

    The source block is the (f1 outer, f2 inner) layout ComputeElementH()
    leaves and ScatterElementH() replays: rows are f2's dofs, columns are
    f1's, column-major. */
void HDGScatterElementsH(const Array<int> &el_slot, const Array<int> &el_face,
                         const Array<int> &face_dofs, const Array<int> &I,
                         int el_0, int nel, int nf, int nc, int side,
                         const Vector &Hel, Vector &V)
{
   const int T = nf*nc;
   const auto d_slot = el_slot.Read();
   const auto d_elface = el_face.Read();
   const auto d_fdofs = face_dofs.Read();
   const auto d_I = I.Read();
   const auto d_Hel = Hel.Read();
   auto d_V = V.ReadWrite();
   mfem::forall(nel*nf*nf, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int lf1 = idx % nf;
      const int lf2 = (idx / nf) % nf;
      const int e = idx / (nf*nf);
      const int el = el_0 + e;
      const int code = d_slot[(el*nf + lf2)*nf + lf1];
      if ((code & 1) != side) { return; }
      // The columns of f1 sit at one slot of the row's neighbour list, nc of
      // them together, which is the whole reason a per-nonzero index can be
      // recovered from arithmetic instead of stored.
      const int coff = (code >> 1)*nc;
      const int f2 = d_elface[el*nf + lf2];
      const real_t *src = d_Hel + e*T*T + (lf1*nf + lf2)*nc*nc;
      for (int i2 = 0; i2 < nc; i2++)
      {
         const int k = d_I[d_fdofs[f2*nc + i2]] + coff;
         for (int j1 = 0; j1 < nc; j1++)
         {
            const real_t val = src[j1*nc + i2];
            if (side == 0) { d_V[k + j1] = val; }
            else { d_V[k + j1] += val; }
         }
      }
   });
}

/** @brief Whether @a nlfi is linear despite its type -- i.e. is really a
    BilinearFormIntegrator.

    `class BilinearFormIntegrator : public NonlinearFormIntegrator`
    (fem/bilininteg.hpp), so installing one on a NONLINEAR form compiles, and
    convdiff and anisodiff both do it: a VectorMassIntegrator on the flux mass
    and a ConservativeConvectionIntegrator on the potential mass. The default
    AssembleElementVector() (fem/bilininteg.cpp) then assembles the element
    matrix and multiplies -- "general but not efficient", says its own comment
    -- on EVERY residual evaluation, for a matrix that never changes.

    **A SumNLFIntegrator is looked THROUGH, and it has to be**:
    DarcyForm::EnableHybridization() wraps a nonlinear mass form's domain
    integrators in one unconditionally, even when there is exactly one, so
    nothing ever arrives here as a bare BilinearFormIntegrator. The first
    draft refused the wrapper on the grounds that nothing in the tree builds
    one on these two slots; measured, every convdiff `-nl` run builds one, and
    the refusal made the whole route dead code. That is the second time this
    exact wrapper has done that -- see SumNLFIntegrator::NumIntegrators(),
    which exists because AssemblyMode::Batched was dead for the same reason.

    The sum is linear only if EVERY member is, and a member that is itself a
    sum is looked through too. */
bool HDGIntegratorIsLinear(const NonlinearFormIntegrator *nlfi)
{
   if (!nlfi) { return false; }
   if (auto *snlfi = dynamic_cast<const SumNLFIntegrator*>(nlfi))
   {
      const int n = snlfi->NumIntegrators();
      if (n == 0) { return false; }
      for (int i = 0; i < n; i++)
      {
         if (!HDGIntegratorIsLinear(snlfi->GetIntegrator(i))) { return false; }
      }
      return true;
   }
   return dynamic_cast<const BilinearFormIntegrator*>(nlfi) != NULL;
}

/// Add @a nlfi's element matrices into @a p, an (n, n, NE) column-major store.
static void HDGAddLinearNLF(const NonlinearFormIntegrator *nlfi,
                            FiniteElementSpace &fes, int n, real_t *p)
{
   if (auto *snlfi = dynamic_cast<const SumNLFIntegrator*>(nlfi))
   {
      for (int i = 0; i < snlfi->NumIntegrators(); i++)
      { HDGAddLinearNLF(snlfi->GetIntegrator(i), fes, n, p); }
      return;
   }
   auto *bfi = const_cast<BilinearFormIntegrator*>(
                  dynamic_cast<const BilinearFormIntegrator*>(nlfi));
   MFEM_VERIFY(bfi, "not a bilinear integrator");
   const int NE = fes.GetNE();
   DenseMatrix elmat;
   IsoparametricTransformation Tr;
   for (int el = 0; el < NE; el++)
   {
      fes.GetMesh()->GetElementTransformation(el, &Tr);
      bfi->AssembleElementMatrix(*fes.GetFE(el), Tr, elmat);
      MFEM_VERIFY(elmat.Height() == n && elmat.Width() == n,
                  "a linear nonlinear-form integrator gave a "
                  << elmat.Height() << "x" << elmat.Width()
                  << " block where " << n << "x" << n << " was wanted");
      for (int c = 0; c < n; c++)
         for (int r = 0; r < n; r++)
         { p[el*n*n + c*n + r] += elmat(r, c); }
   }
}

/** @brief Assemble @a nlfi -- which HDGIntegratorIsLinear() has said is
    bilinear, possibly under a sum -- into one (@a n, @a n, NE) block store,
    once. */
void HDGAssembleLinearNLF(const NonlinearFormIntegrator *nlfi,
                          FiniteElementSpace &fes, int n, Vector &out)
{
   const int NE = fes.GetNE();
   out.SetSize(n*n*NE);
   out.UseDevice(true);
   real_t *p = out.HostWrite();
   std::fill(p, p + n*n*NE, 0.0);
   HDGAddLinearNLF(nlfi, fes, n, p);
}

bool DarcyHybridization::CanBatchLinearResidual() const
{
   // NPC only: this replaces MultNlMode::AtFields's element loop, which is
   // the mode whose fields are Newton state rather than produced by a local
   // solve.
   if (!NPCEnabled()) { return false; }
   // A shared face has one of its elements on another rank, so neither the
   // element-blocked gather nor the two-contributions trace scatter survives.
   if (ParallelC()) { return false; }

   // **Every nonlinearity is a refusal, and that is the whole scope of this
   // routine.** A nonlinear integrator of any kind means the local residual
   // is an integrator evaluation rather than a matrix product, and there is
   // nothing to batch with BatchedLinAlg. The batched LOCAL RESIDUAL
   // (CanBatchLocalResidual) is the other half of that story and covers the
   // element integrator; this covers the case where there is none.
   // A BLOCK nonlinearity is refused outright: its two rows couple and there
   // is no matrix to assemble once.
   if (m_nlfi) { return false; }
   // **But a "nonlinear" flux or potential mass whose integrators are all
   // BilinearFormIntegrators is LINEAR**, and that is what convdiff and
   // anisodiff install. Admitted, and assembled once; see
   // HDGNonlinearFormIsLinear() and the ResidualCache fields.
   if (m_nlfi_u && !HDGIntegratorIsLinear(m_nlfi_u)) { return false; }
   if (m_nlfi_p && !HDGIntegratorIsLinear(m_nlfi_p)) { return false; }
   // **And BOTH halves or neither.** Freezing a bilinear mass slot into
   // once-assembled blocks is only safe if the GRADIENT freezes with it, and
   // the two specialised local operators put that out of reach:
   // LocalOpType::PotNL factors Af_data in place and ConstructGrad() leaves
   // Df_data alone, so CopyLinearGradBlocks() cannot apply there and the
   // gradient stays live. Refusing here is what keeps the residual live to
   // match. Measured before this guard, on m_nlfi_p alone (which IS the
   // PotNL branch): the residual froze at its first parameter value while
   // the gradient tracked, 8.0e-3 to 4.0e-2 apart over five 1% steps. See
   // EnsureResidualCache().
   //
   // convdiff -nl is unaffected -- a nonlinear flux mass with a linear face
   // constraint lands on FullNL, where CopyLinearGradBlocks() does apply.
   if ((m_nlfi_u || m_nlfi_p)
       && (lop_type == LocalOpType::PotNL || lop_type == LocalOpType::FluxNL))
   { return false; }
   if (c_nlfi_p || c_nlfi) { return false; }
   if (!boundary_constraint_pot_nonlin_integs.empty()) { return false; }
   if (!boundary_constraint_nonlin_integs.empty()) { return false; }
   // The E, G and H terms are taken only when the face constraint is linear
   // and present; without it the element loop skips them and the arithmetic
   // below would add terms the loop does not. Refused rather than branched.
   if (!c_bfi_p) { return false; }

   // The blocks the products need, at one size each.
   const int NE = fes.GetNE();
   if (NE <= 0) { return false; }
   const int na = UniformBlockSize(Af_f_offsets, NE);
   const int nd = UniformBlockSize(Df_f_offsets, NE);
   if (na <= 0 || nd <= 0) { return false; }
   // **The two rows differ, and they differ because the element loop's two
   // routines differ.** AddMultA() is `if (m_nlfi_u) ... else if (!A_empty)`
   // -- an ALTERNATIVE, so a nonlinear flux mass REPLACES the linear one --
   // while AddMultDE() adds the linear D alongside m_nlfi_p whenever D is
   // non-empty, because Df_lin_data then holds linear content -- a face
   // constraint, a linear domain mass, or both -- that nothing else
   // supplies. So an empty linear A is fine when m_nlfi_u will stand in for
   // it, and an empty linear D never is.
   if (!m_nlfi_u && A_empty) { return false; }
   if (D_empty) { return false; }
   if (!m_nlfi_u && Af_lin_data.Size() != Af_offsets.Last()) { return false; }
   if (Df_lin_data.Size() != Df_offsets.Last()) { return false; }
   if (Bf_data.Size() != Bf_offsets.Last()) { return false; }

   // And the trace map, which is where the one-face-count-per-element and
   // one-trace-size-per-face conditions are actually tested.
   BuildElementDofMaps();
   if (el_c_dofs.Size() == 0) { return false; }
   return true;
}

/** @brief Add each element's own T-vector into @a y at the trace dofs
    @a el_c_dofs names, accumulating.

    **The one place here that needs an atomic, and the one place it is
    provably deterministic anyway.** A trace dof belongs to one face and a
    face has at most two elements, so every entry of @a y receives at most two
    contributions -- and MultNL() zeroes y before the loop, so the sum is
    `0 + a + b` whichever order they arrive in. IEEE addition of two terms is
    commutative to the bit and `0 + a` is exact, so the atomic costs no
    determinism. That argument fails the moment y is not zeroed or a dof can
    have three contributors, which is why it is written down rather than
    assumed. */
void HDGScatterTraceRows(const Vector &src, const Array<int> &map,
                         int nf, int nc, int NE, Vector &y)
{
   const int T = nf*nc;
   const auto d_src = src.Read();
   const auto d_map = map.Read();
   auto d_y = y.ReadWrite();
   mfem::forall(NE*nf, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int lf = idx % nf;
      const int e = idx / nf;
      const real_t *s = d_src + e*T + lf*nc;
      const int *m = d_map + (e*nf + lf)*nc;
      for (int j = 0; j < nc; j++) { AtomicAdd(d_y[m[j]], s[j]); }
   });
}

/** @brief Build ResidualCache if it is not built, and say whether it is
    usable. Shared by LinearResidualBatched() and CopyLinearGradBlocks(),
    which is not an accident: **the two have to be built from the same blocks
    or the residual and the gradient are of different operators.**

    That is not hypothetical. Tier 2 admitted a bilinear integrator on a
    NONLINEAR mass form and assembled it once here, while
    CopyLinearGradBlocks() went on refusing whenever m_nlfi_p was set -- so
    the gradient re-evaluated the integrator and tracked a moving coefficient
    while the residual stayed frozen at its first value. Measured on a
    parametric ConservativeConvectionIntegrator: the residual was wrong by
    8.0e-3, 1.6e-2, 2.4e-2, 3.2e-2, 4.0e-2 as the parameter moved by 1% steps,
    growing linearly, with the gradient correct to 1.1e-14 throughout and no
    warning either way. This branch's own record says what that costs: a
    hybridized Jacobian that does not match its residual gives no wrong
    answer, only a Newton that misbehaves, and it took LBFGS-against-Newton
    to find the last one.

    So the frozen blocks now feed BOTH, and a caller whose coefficients move
    calls InvalidateCoefficientCache().

    ### What this mechanism costs, and the one change that would delete it

    Four separate pieces of machinery exist only to support this cache, and
    they are worth listing together because they have a single cause:

    1. HDGIntegratorIsLinear() has to DISCOVER at run time that something
       typed as nonlinear is in fact bilinear, recursing through
       SumNLFIntegrator and asking dynamic_cast.
    2. That discovery is not sufficient on its own. CanBatchLinearResidual()
       must additionally refuse LocalOpType::PotNL and FluxNL, because those
       two layouts factor one block in place and cannot freeze the gradient
       to match -- so the residual must be kept live to match IT.
    3. A caller whose coefficients move must call
       InvalidateCoefficientCache() by hand, because no type distinguishes a
       bilinear integrator holding a frozen coefficient from one holding a
       live one.
    4. The residual and the gradient must be built from the same blocks, per
       the measurement above, which is why this function exists at all
       rather than each route building its own.

    All four follow from one fact: NonlinearFormIntegrator::
    AssembleElementVector() hands back a vector and says NOTHING about which
    part of it was state-independent. The cache is a run-time reconstruction
    of information the interface discarded, and every item above is a place
    where the reconstruction can be wrong.

    **An interpolatory formulation would make all four unnecessary rather
    than easier**, and that is the caller's observation, recorded here
    because it is the structural answer and this is where the cost is paid.
    If a nonlinear integrator were required to present itself as a FIXED
    shape matrix times a nodal factor -- the nonlinearity collocated at
    interpolation nodes, as in interpolatory HDG -- then the invariant part
    would be in the TYPE instead of being inferred: there is nothing to
    detect, because the shape matrix is constant by construction; nothing to
    invalidate when the state or a parameter moves, because only the geometry
    touches the shape matrix; and the residual and the gradient share that
    matrix by construction, so the defect measured above becomes
    unrepresentable rather than guarded against.

    **It is not a refactor, and must not be adopted as one.** Collocating a
    nonlinearity at nodes is a different quadrature from integrating it, so
    it changes the discrete problem -- every answer and every reference
    moves. The right order is to adopt it for its own merits, if the
    superconvergence and the reassembly saving justify it, and to take the
    deletion of items 1-4 as a consequence. Adopting it to delete cache code
    would be choosing a discretisation to suit an implementation.
    doc/HDG-NPC-PARAMETRIC-COEFFICIENTS-REPLY-TO-GFFP.md sec. 8 carries the
    open question. */
bool DarcyHybridization::EnsureResidualCache(int na, int nd) const
{
   const int NE = fes.GetNE();
   if (res_cache)
   {
      return res_cache->na == na && res_cache->nd == nd
             && res_cache->nf * res_cache->nc > 0;
   }
   {
      std::unique_ptr<ResidualCache> c(new ResidualCache);
      // with_h, because the trace row needs H and the map is what carries
      // the "first side only" convention as an offset of -1.
      if (!BuildElementHFaceMap(na, nd, true, c->nf, c->nc, c->face_map))
      { return false; }
      const int Tc = c->nf * c->nc;
      if (Tc <= 0) { return false; }
      c->na = na;
      c->nd = nd;
      c->Ct_all.SetSize(na*Tc*NE);
      c->E_all.SetSize(nd*Tc*NE);
      c->G_all.SetSize(Tc*nd*NE);
      c->H_all.SetSize(Tc*Tc*NE);
      c->Ct_all.UseDevice(true);
      c->E_all.UseDevice(true);
      c->G_all.UseDevice(true);
      c->H_all.UseDevice(true);
      // Ct comes out (na, T) per element, so ONE product Ct*x_el is the sum
      // over the element's faces -- the element loop's own face loop, without
      // the loop. Likewise E is (nd, T) and G is (T, nd).
      HDGGatherFaceCols(Ct_data, c->face_map, 0, na, c->nc, c->nf, 0, NE,
                        c->Ct_all);
      HDGGatherFaceCols(E_data, c->face_map, 1, nd, c->nc, c->nf, 0, NE,
                        c->E_all);
      HDGGatherFaceRows(G_data, c->face_map, 1, c->nc, nd, c->nf, 0, NE,
                        c->G_all);
      // H is DIAGONAL per face inside the element's (T, T) block and is added
      // rather than written, so it is zeroed first.
      c->H_all = 0.;
      HDGAddFaceDiagBlocks(H_data, c->face_map, 2, c->nc, c->nf, 0, NE,
                           c->H_all);
      // Tier 2: a "nonlinear" mass form whose integrators are all bilinear is
      // constant, so it is assembled ONCE here instead of per residual
      // evaluation by BilinearFormIntegrator::AssembleElementVector(). The
      // predicate has already established both are of that kind.
      if (m_nlfi_u)
      { HDGAssembleLinearNLF(m_nlfi_u, fes, na, c->Au_all); }
      if (m_nlfi_p)
      { HDGAssembleLinearNLF(m_nlfi_p, fes_p, nd, c->Dp_all); }
      res_cache = std::move(c);
   }
   return true;
}

bool DarcyHybridization::LinearResidualBatched(
   const Vector &x, const Vector &bu, const Vector &bp,
   BlockVector &r_local, Vector &y) const
{
   if (!CanBatchLinearResidual()) { return false; }

   const int NE = fes.GetNE();
   const int na = UniformBlockSize(Af_f_offsets, NE);
   const int nd = UniformBlockSize(Df_f_offsets, NE);

   if (!EnsureResidualCache(na, nd)) { return false; }
   const ResidualCache &rc = *res_cache;
   const int nf = rc.nf, nc = rc.nc;
   const int T = nf*nc;
   if (T <= 0 || rc.na != na || rc.nd != nd) { return false; }

   // ---- the gathers. GetSubVector() is already an mfem::forall over the dof
   // map (linalg/vector.cpp), so these are kernels; UseDevice(true) on the
   // blocked side is the other half of sending them there, per
   // BuildElementDofMaps().
   ResidualCache &wk = const_cast<ResidualCache&>(rc);
   Vector &u_all = wk.u_all, &p_all = wk.p_all, &bu_all = wk.bu_all;
   Vector &bp_all = wk.bp_all, &x_all = wk.x_all;
   u_all.SetSize(na*NE);
   p_all.SetSize(nd*NE);
   bu_all.SetSize(na*NE);
   bp_all.SetSize(nd*NE);
   x_all.SetSize(T*NE);
   u_all.UseDevice(true);
   p_all.UseDevice(true);
   bu_all.UseDevice(true);
   bp_all.UseDevice(true);
   x_all.UseDevice(true);
   darcy_u.GetSubVector(el_u_dofs, u_all);
   darcy_p.GetSubVector(el_p_dofs, p_all);
   bu.GetSubVector(el_u_dofs, bu_all);
   bp.GetSubVector(el_p_dofs, bp_all);
   x.GetSubVector(el_c_dofs, x_all);

   // ---- the face blocks come from the cache above, not from a gather here.
   DenseTensor A, B, D, Ct, E, G, H;
   if (!m_nlfi_u)
   {
      A.NewMemoryAndSize(const_cast<Array<real_t>&>(Af_lin_data).GetMemory(),
                         na, na, NE, false);
   }
   B.NewMemoryAndSize(Bf_data.GetMemory(), nd, na, NE, false);
   D.NewMemoryAndSize(Df_lin_data.GetMemory(), nd, nd, NE, false);
   Ct.NewMemoryAndSize(rc.Ct_all.GetMemory(), na, T, NE, false);
   E.NewMemoryAndSize(rc.E_all.GetMemory(), nd, T, NE, false);
   G.NewMemoryAndSize(rc.G_all.GetMemory(), T, nd, NE, false);
   H.NewMemoryAndSize(rc.H_all.GetMemory(), T, T, NE, false);

   // ---- the flux row.  ru = A u + sgn B^T p - bu + C^T x
   //
   // The signs are the element loop's, read off it rather than rederived:
   // LocalNLOperator::Mult() forms `bu = B^T p`, negates it under bsym, then
   // adds A u; and MultNL() forms `bu_l = bu - Ct x` before LocalResidual()
   // subtracts it. So the Ct term arrives with a PLUS here because it was
   // subtracted from something that is then subtracted.
   const real_t sgn = bsym ? -1.0 : 1.0;
   Vector &ru_all = wk.ru_all, &rp_all = wk.rp_all, &y_all = wk.y_all;
   ru_all.SetSize(na*NE);
   rp_all.SetSize(nd*NE);
   y_all.SetSize(T*NE);
   ru_all.UseDevice(true);
   rp_all.UseDevice(true);
   y_all.UseDevice(true);
   if (m_nlfi_u)
   {
      // The nonlinear flux mass form's constant part, INSTEAD of the linear
      // A -- see the predicate, and AddMultA(), which this mirrors.
      DenseTensor Au;
      Au.NewMemoryAndSize(const_cast<Vector&>(rc.Au_all).GetMemory(),
                          na, na, NE, false);
      BatchedLinAlg::Mult(Au, u_all, ru_all);                   // Au u
   }
   else
   {
      BatchedLinAlg::Mult(A, u_all, ru_all);                    // A u
   }
   BatchedLinAlg::AddMult(B, p_all, ru_all, sgn, 1.0,
                          BatchedLinAlg::Op::T);   // + sgn B^T p
   BatchedLinAlg::AddMult(Ct, x_all, ru_all, 1.0, 1.0);         // + Ct x
   ru_all -= bu_all;                                            // - bu

   // ---- the potential row. rp = B u + D p - sgn bp + E x, where the element
   // loop negates its bp_l under bsym before subtracting.
   BatchedLinAlg::Mult(B, u_all, rp_all);                       // B u
   BatchedLinAlg::AddMult(D, p_all, rp_all, 1.0, 1.0);          // + D p
   if (m_nlfi_p)
   {
      DenseTensor Dp;
      Dp.NewMemoryAndSize(const_cast<Vector&>(rc.Dp_all).GetMemory(),
                          nd, nd, NE, false);
      BatchedLinAlg::AddMult(Dp, p_all, rp_all, 1.0, 1.0);      // + Dp p
   }
   BatchedLinAlg::AddMult(E, x_all, rp_all, 1.0, 1.0);          // + E x
   rp_all.Add(-sgn, bp_all);                                    // - sgn bp

   // ---- the trace row. y = C u + G p + H x, H on the first side only, which
   // the -1 offsets in the map already arranged.
   BatchedLinAlg::AddMult(Ct, u_all, y_all, 1.0, 0.0,
                          BatchedLinAlg::Op::T);   // Ct^T u
   BatchedLinAlg::AddMult(G, p_all, y_all, 1.0, 1.0);           // + G p
   BatchedLinAlg::AddMult(H, x_all, y_all, 1.0, 1.0);           // + H x

   // ---- the scatters. The field rows are disjoint per element -- an L2
   // element owns its dofs -- so they need no accumulation and no atomic;
   // the trace row is the one that collides, and see HDGScatterTraceRows().
   r_local.GetBlock(0).UseDevice(true);
   r_local.GetBlock(1).UseDevice(true);
   r_local.GetBlock(0).SetSubVector(el_u_dofs, ru_all);
   r_local.GetBlock(1).SetSubVector(el_p_dofs, rp_all);
   r_local.GetBlock(0).SyncAliasMemory(r_local);
   r_local.GetBlock(1).SyncAliasMemory(r_local);

   y.UseDevice(true);
   HDGScatterTraceRows(y_all, el_c_dofs, nf, nc, NE, y);
   return true;
}

void DarcyHybridization::SetTraceAssemblyMode(TraceAssemblyMode mode)
{
   tasm_mode = mode;
}

const DarcyHybridization::TraceHMap *DarcyHybridization::BuildTraceHMap() const
{
   // A refusal is cached as a map with nf == 0, so the conditions below are
   // asked once and not once per linearisation.
   if (trace_h_map) { return (trace_h_map->nf > 0) ? trace_h_map.get() : NULL; }
   trace_h_map.reset(new TraceHMap);
   TraceHMap &m = *trace_h_map;

   // A shared face has one of its elements on another rank, so neither the
   // row ownership nor the two-contributions argument survives. The face
   // kernels refuse ParallelC() for the same reason.
   if (ParallelC()) { return NULL; }

   // The pattern this mode builds is the STRUCTURAL one, which is what the
   // host route's AddSubMatrix(skip_zeros) plus Finalize(0) comes to -- see
   // the declaration for the measurement. Finalize(skip_zeros=1), which the
   // other diagonal policies take, drops every exactly-zero entry instead,
   // and that pattern is a function of the values: it would need a
   // compaction pass per linearisation rather than a map built once.
   // Refused rather than approximated, and no caller in this tree asks for
   // it -- diag_policy is DIAG_ONE everywhere.
   if (diag_policy != DIAG_ONE && diag_policy != DIAG_ZERO) { return NULL; }

   const Mesh *mesh = fes.GetMesh();
   const int NE = mesh->GetNE();
   const int NF = mesh->GetNumFaces();
   if (NE <= 0 || NF <= 0) { return NULL; }

   Array<int> faces;
   GetElementFaces(0, faces);
   const int nf = faces.Size();
   if (nf <= 0) { return NULL; }
   const int vdim = c_fes.GetVDim();
   const int nc = c_fes.GetFaceElement(faces[0])->GetDof() * vdim;
   if (nc <= 0) { return NULL; }
   const int ncdofs = c_fes.GetVSize();

   // (face, local dof) -> trace vdof, and its inverse. Built rather than
   // computed: a face's dofs are NOT contiguous in 2-D, measured across
   // DG_Interface orders 0 to 3, so nothing here may assume f*nc + k.
   m.face_dofs.SetSize(NF*nc);
   Array<int> dof_face(ncdofs);
   dof_face = -1;
   Array<int> fdofs;
   for (int f = 0; f < NF; f++)
   {
      c_fes.GetFaceVDofs(f, fdofs);
      // One dof count for every face. A p-adaptive trace, or a mesh mixing
      // face geometries, fails here and takes the host route.
      //
      // BOTH counts are checked, and they are not the same question. The
      // element block buffer is sized by GetElementTraceSize(), which asks
      // GetFaceElement()->GetDof(), while the scatter indexes rows by
      // GetFaceVDofs(); the kernel's `e*T*T` stride is only right when the
      // two agree on every face.
      if (fdofs.Size() != nc) { return NULL; }
      if (c_fes.GetFaceElement(f)->GetDof() * vdim != nc) { return NULL; }
      for (int k = 0; k < nc; k++)
      {
         const int d = fdofs[k];
         // A negative index is an orientation flip, which would need the
         // value negated as AddSubMatrix does. DG_Interface produces none --
         // measured 0 of them at every order and dimension tried -- so this
         // is a refusal rather than a case to handle.
         if (d < 0 || d >= ncdofs) { return NULL; }
         // The row-ownership condition, and the one that rules out an H1
         // trace: a dof on two faces would be written by two threads.
         if (dof_face[d] >= 0) { return NULL; }
         m.face_dofs[f*nc + k] = d;
         dof_face[d] = f;
      }
   }
   // A dof on no face at all would leave an empty row that the diagonal
   // policy expects to exist.
   for (int d = 0; d < ncdofs; d++) { if (dof_face[d] < 0) { return NULL; } }

   // (element, local face) -> face, and the at-most-two elements of a face.
   m.el_face.SetSize(NE*nf);
   Array<int> f2e(2*NF);
   f2e = -1;
   for (int el = 0; el < NE; el++)
   {
      GetElementFaces(el, faces);
      if (faces.Size() != nf) { return NULL; }
      for (int lf = 0; lf < nf; lf++)
      {
         const int f = faces[lf];
         m.el_face[el*nf + lf] = f;
         // A face listed twice by ONE element -- a one-element-wide periodic
         // mesh does this -- gives that element four blocks in the (f, f)
         // pair, so the entry is a sum of four terms and the two-pass scatter
         // is wrong. Refused, and this is the reason to check it rather than
         // trust "a face has two elements".
         for (int lo = 0; lo < lf; lo++)
         { if (faces[lo] == f) { return NULL; } }
         if (f2e[2*f] < 0) { f2e[2*f] = el; }
         else if (f2e[2*f+1] < 0) { f2e[2*f+1] = el; }
         else { return NULL; }
      }
   }

   // The faces sharing an element with each face, ascending and unique --
   // which IS the row's column layout, one neighbour at a time. Locals and
   // not members: nothing reads them once J and el_slot are built, and a
   // cached array with no reader is a liability rather than documentation.
   Array<int> nb_offsets(NF+1), nb;
   nb_offsets[0] = 0;
   nb.SetSize(0);
   Array<int> nbf;
   for (int f = 0; f < NF; f++)
   {
      nbf.SetSize(0);
      for (int s = 0; s < 2; s++)
      {
         const int el = f2e[2*f + s];
         if (el < 0) { continue; }
         for (int lf = 0; lf < nf; lf++) { nbf.Append(m.el_face[el*nf + lf]); }
      }
      nbf.Sort();
      nbf.Unique();
      for (int i = 0; i < nbf.Size(); i++) { nb.Append(nbf[i]); }
      nb_offsets[f+1] = nb.Size();
   }

   // The CSR row offsets. Every row of a face has the same length, so this is
   // a prefix sum over rows of nc times the face's neighbour count.
   m.I.SetSize(ncdofs+1);
   m.I[0] = 0;
   for (int d = 0; d < ncdofs; d++)
   {
      const int f = dof_face[d];
      m.I[d+1] = m.I[d] + nc*(nb_offsets[f+1] - nb_offsets[f]);
   }
   m.nnz = m.I[ncdofs];

   // The column indices, once. A row of face f lists the dofs of each of f's
   // neighbour faces in turn, nc together -- which is what makes a
   // per-nonzero index recoverable as I[row] + slot*nc + j, and is also why
   // the columns are NOT ascending: a face's dofs are not contiguous, so the
   // blocks interleave. Nothing downstream requires sorted columns, and the
   // host route does not produce them either.
   m.J.SetSize(m.nnz);
   for (int f = 0; f < NF; f++)
   {
      const int begin = nb_offsets[f], end = nb_offsets[f+1];
      for (int i = 0; i < nc; i++)
      {
         int k = m.I[m.face_dofs[f*nc + i]];
         for (int s = begin; s < end; s++)
         {
            const int fj = nb[s];
            for (int j = 0; j < nc; j++) { m.J[k++] = m.face_dofs[fj*nc + j]; }
         }
         MFEM_ASSERT(k == m.I[m.face_dofs[f*nc + i]+1], "row length mismatch");
      }
   }

   // Where each element's (f1 columns, f2 rows) block lands, and which of the
   // pair's at most two elements this one is.
   m.el_slot.SetSize(NE*nf*nf);
   for (int el = 0; el < NE; el++)
   {
      for (int lf2 = 0; lf2 < nf; lf2++)
      {
         const int f2 = m.el_face[el*nf + lf2];
         const int b = nb_offsets[f2], e2 = nb_offsets[f2+1];
         for (int lf1 = 0; lf1 < nf; lf1++)
         {
            const int f1 = m.el_face[el*nf + lf1];
            int slot = -1;
            for (int s = b; s < e2; s++)
            { if (nb[s] == f1) { slot = s - b; break; } }
            MFEM_VERIFY(slot >= 0, "a face pair of an element is missing from "
                        "the face's own neighbour list");
            // Side 0 is the LOWEST-numbered element contributing to this
            // (f2, f1) pair, so counting the lower-numbered contributors is
            // the rank. The elements that can contribute are the two of f2,
            // since the row's face is f2.
            int side = 0;
            for (int s = 0; s < 2; s++)
            {
               const int other = f2e[2*f2 + s];
               if (other < 0 || other >= el) { continue; }
               for (int lf = 0; lf < nf; lf++)
               { if (m.el_face[other*nf + lf] == f1) { side++; break; } }
            }
            MFEM_VERIFY(side < 2, "a trace matrix entry with more than two "
                        "contributing elements");
            m.el_slot[(el*nf + lf2)*nf + lf1] = 2*slot + side;
         }
      }
   }

   // Only what the scatter kernel reads. I is on both lists: the kernel needs
   // it to find a row, and BuildTraceHPattern() copies it into the matrix on
   // the host -- which is safe for both because nothing ever writes it again,
   // so the first sync in each direction is the only one.
   m.face_dofs.UseDevice(true);
   m.el_face.UseDevice(true);
   m.I.UseDevice(true);
   m.el_slot.UseDevice(true);

   m.NF = NF;
   m.NE = NE;
   m.nc = nc;
   m.ncdofs = ncdofs;
   m.nf = nf;      // last: a nonzero nf is what marks the map as usable
   return &m;
}

bool DarcyHybridization::CanBatchTraceAssembly() const
{
   if (tasm_mode != TraceAssemblyMode::Batched) { return false; }
   if (!BuildTraceHMap()) { return false; }

   // **And the destination question, which is not the same as the map's and
   // was missing from a first version of this predicate.** The mode builds
   // the CSR, so it can only have a matrix that is empty when ComputeH() is
   // entered -- and on a NON-NPC problem it is not. The face-constraint
   // assembly reaches the sparse H directly during Assemble():
   // ComputeAndAssemblePotFaceMatrix() and its boundary twin write each
   // face's diagonal block to H_data under NPC and to `H->AddSubMatrix()`
   // otherwise, so by Finalize()'s ComputeH() call the reduced route already
   // holds an unfinalized linked-list matrix carrying those terms.
   //
   // This is the offload plan's item 9 -- "the kernel writes H_data and the
   // reduced route reads an assembled sparse H" -- reaching a second kernel,
   // and it is a scope limit rather than a defect: NPC is what this branch is
   // for. Lifting it means giving the face assembly a CSR destination too,
   // which is the same map and a different scatter.
   //
   // Asked of `H` and not of NPCEnabled() alone, because a problem with no
   // potential face term at all leaves H empty and is fine either way.
   return NPCEnabled() || !H;
}

void DarcyHybridization::BuildTraceHPattern(
   const TraceHMap &m, std::unique_ptr<SparseMatrix> &H_, Vector &V) const
{
   // Built finalized, which is the whole point: the host route allocates one
   // RowNode per nonzero every linearisation and walks the list twice to
   // compact it, and none of that structure depends on a value.
   H_.reset(new SparseMatrix);
   H_->OverrideSize(m.ncdofs, m.ncdofs);
   H_->GetMemoryI().New(m.ncdofs+1, H_->GetMemoryI().GetMemoryType());
   H_->GetMemoryJ().New(m.nnz, H_->GetMemoryJ().GetMemoryType());
   H_->GetMemoryData().New(m.nnz, H_->GetMemoryData().GetMemoryType());

   // The pattern, as a host copy of the cached one. A copy and not an alias:
   // a matrix aliasing the map's arrays would dangle the moment Reset()
   // dropped the map, and it is nnz ints against the nnz doubles the fill
   // writes anyway.
   H_->GetMemoryI().CopyFromHost(m.I.HostRead(), m.ncdofs+1);
   H_->GetMemoryJ().CopyFromHost(m.J.HostRead(), m.nnz);

   // The scatter ASSIGNS every entry it owns, so this zero is not what makes
   // the answer right -- it is what makes a structural entry no element
   // reaches read as zero rather than as whatever the allocator left.
   V.NewMemoryAndSize(H_->GetMemoryData(), m.nnz, false);
   V.UseDevice(true);
   V = 0.;
}

void DarcyHybridization::ScatterElementsHBatched(
   const TraceHMap &m, int el_0, int nel, const Vector &Hel, Vector &V) const
{
   if (nel <= 0) { return; }

   // Two passes and not one: an entry's two elements are in different passes
   // by construction, so neither pass has a write conflict and no atomic is
   // needed. Pass 0 assigns and pass 1 adds, and side 0 is always the lower
   // element index -- so the sum is (lower + higher) however the chunks fall,
   // which is the host loop's own order. A two-term IEEE sum is
   // order-independent regardless, so this is determinism rather than luck.
   for (int side = 0; side < 2; side++)
   {
      HDGScatterElementsH(m.el_slot, m.el_face, m.face_dofs, m.I, el_0, nel,
                          m.nf, m.nc, side, Hel, V);
   }
}

void DarcyHybridization::ComputeH(ComputeHMode mode,
                                  std::unique_ptr<SparseMatrix> &H_) const
{
   MFEM_ASSERT(mode != ComputeHMode::Linear || !NPCEnabled(),
               "Cannot assemble H matrix in the non-linear regime");

   // Still needed by the Finalize() below; ScatterElementH() carries its own.
   const int skip_zeros = 1;
   const int NE = fes.GetNE();

   // The factorisation below is what GradientMode::MatrixFree needs and all it
   // needs; the face loops after it are the assembly, which is what costs one
   // local solve per trace dof of the element. Both modes take the same first
   // half, which is the point of doing it here rather than in ConstructGrad():
   // that duplicate omitted the Jacobian's d(flux residual)/dp and so built a
   // different Schur complement from this one.
   const bool assemble = (mode != ComputeHMode::GradientFactorOnly);

   // The sparse third of this routine, and the last stage that had no device
   // path at all. It replaces both the per-element scatter and the Finalize()
   // below, so it decides how H_ is CREATED as well as how it is filled --
   // one is a linked list to be compacted, the other a CSR that already has
   // its pattern. Only usable on a matrix this routine owns from the start,
   // which on a non-NPC problem it does not: see CanBatchTraceAssembly(),
   // which asks that same question so the predicate cannot claim a route
   // this call did not take.
   const TraceHMap *thmap = (assemble && !H_ && CanBatchTraceAssembly())
                            ? BuildTraceHMap() : NULL;

   // The view of H_'s data the scatter writes through, created once for the
   // whole loop; see BuildTraceHPattern().
   Vector Hcsr;
   if (thmap) { BuildTraceHPattern(*thmap, H_, Hcsr); }
   else if (assemble && !H_) { H_.reset(new SparseMatrix(c_fes.GetVSize())); }

   // The loop runs in chunks: a chunk's element-local work may happen in any
   // order, and the scatter that follows it is replayed in element order. The
   // buffer is what separates the two, and both halves of that separation are
   // needed.
   //
   // Threading the scatter instead is not an option that was passed over.
   // AddSubMatrix() reaches an unfinalized SparseMatrix through SetColPtr(),
   // and that matrix has one current_row, one column-pointer scratch and one
   // RowNode allocator for the whole matrix -- so two threads adding to
   // *disjoint* rows still collide, and the failure is a hang rather than a
   // wrong answer. Element colouring buys disjoint rows and therefore does not
   // fix this by itself.
   //
   // The two modes then agree bit for bit -- but the ordering is not what
   // buys that, and it was worth measuring rather than asserting. A trace dof
   // lives on a face and a face has at most two elements, so each entry of H_
   // is a sum of at most two contributions, and IEEE addition of two terms
   // does not depend on their order. Scattering a chunk back-to-front instead
   // was tried: it leaves the assembled matrix's effect unchanged, at the same
   // 139 and 119 iterations and the same errors to every digit on the two
   // nonlinear cases that are sensitive enough to have drifted before.
   //
   // So element order is kept because it is free and deterministic, not
   // because exactness needs it. What exactness needs is that the arithmetic
   // above be per-element, which it is: nothing in it is reassociated by the
   // schedule, and the entries themselves are what the sum is over.
   const int chunk = AssemblyChunkSize(NE);

   // Every element's factorisation and Schur complement in one batch, when
   // that is asked for; the loop below then does the face pairs only.
   Vector AiBt_all;
   const bool prefactored = FactorElementsBatched(mode, AiBt_all);

   // And the FACE-PAIR loop batches with it, when the mesh gives one face
   // count per element and one trace size per face. That turns the whole of
   // ComputeElementH() after the factorisation into five BatchedLinAlg calls
   // and four gather/pack kernels per chunk -- see ComputeElementsHBatched().
   int na = 0, nd = 0, nf = 0, nc = 0;
   Array<int> face_map;
   bool batched_asm = false;
   if (prefactored && assemble)
   {
      na = UniformBlockSize(Af_f_offsets, NE);
      nd = UniformBlockSize(Df_f_offsets, NE);
      // The map is built on the host out of these three, and the batched FACE
      // assembly hands them to kernels, which leaves them device-valid; every
      // reader here indexes them raw. Same shape as the Af_ipiv note on
      // InvertA(), so the same repair, and it has to happen BEFORE the map is
      // built rather than in the sync at the end.
      if (Ct_offsets.Size()) { Ct_offsets.HostRead(); }
      if (E_offsets.Size()) { E_offsets.HostRead(); }
      if (H_offsets.Size()) { H_offsets.HostRead(); }
      batched_asm = (na > 0 && nd > 0) &&
                    BuildElementHFaceMap(na, nd,
                                         mode == ComputeHMode::Gradient,
                                         nf, nc, face_map);
   }

   if (prefactored && !batched_asm)
   {
      // THE copy back, and the only one this route needs. Everything that
      // consumes these blocks is host code reaching them through raw
      // pointers, which do not sync -- so without this it reads stale host
      // memory, the whole matrix and silently, or on the debug backend
      // segfaults on a host pointer whose device copy is the valid one.
      // It is the transfer a full-device path has to remove, and naming it in
      // one place is the point of having it here.
      //
      // **It is unconditional, and an earlier version made it conditional on
      // `assemble` on an argument that turned out to be wrong.**
      // ComputeHMode::GradientFactorOnly has no face loop after it, so it
      // looked like the one end of this chain with nothing to read back. But
      // that mode is what GradientMode::MatrixFree uses, and the apply that
      // follows -- MultNL(GradMult) -- calls the PER-ELEMENT MultInv(), which
      // reads Af_data, Bf_data and both sets of pivots exactly as the face
      // loop does. The device test in tests/unit/miniapps/test_debug_device.cpp
      // segfaults in LUFactors::Solve without this line; there is no
      // device-resident end here yet, and there will not be until the
      // matrix-free apply goes through MultInvBatched().
      AiBt_all.HostRead();
      SyncLocalBlocksToHost();
   }

   // The block buffer is a Vector and not an Array<real_t> because the
   // batched path WRITES it with a kernel; the scatter that reads it is host
   // code either way, and HostRead() below is where that is said.
   Vector Hel_data;
   // Device-resident when EITHER end of the chain is a kernel: the face-pair
   // loop writing it, or the trace scatter reading it. With only the latter
   // the host loop still fills it through HostWrite() and the scatter's
   // Read() uploads it, which is one transfer per chunk and the price of
   // taking half the chain.
   Hel_data.UseDevice(batched_asm || thmap != NULL);
   Array<int> Hel_offsets, faces;

   // Hoisted out of the loop, and deliberately: see ElementHWorkspace.
   ElementHWorkspace ws;
   // Likewise, and for the same reason one level down: see SerialHWorkspace.
   SerialHWorkspace hws;
   if (batched_asm)
   {
      ws.Ct.UseDevice(true);
      ws.AiCt.UseDevice(true);
      ws.EG.UseDevice(true);
      ws.BAiCt.UseDevice(true);
      ws.CAiBt.UseDevice(true);
      ws.Hfull.UseDevice(true);
   }

   for (int el_0 = 0; el_0 < NE; el_0 += chunk)
   {
      const int el_1 = std::min(el_0 + chunk, NE);
      const int nel = el_1 - el_0;

      // Sizing the chunk is serial and cheap, and it is also what lets the
      // loop below write into a plain array with no allocation of its own.
      Hel_offsets.SetSize(nel+1);
      Hel_offsets[0] = 0;
      for (int el = el_0; el < el_1; el++)
      {
         int size = 0;
         if (assemble)
         {
            GetElementFaces(el, faces);
            const int t_size = GetElementTraceSize(faces);
            size = t_size * t_size;
         }
         Hel_offsets[el-el_0+1] = Hel_offsets[el-el_0] + size;
      }
      Hel_data.SetSize(Hel_offsets[nel]);

      if (batched_asm)
      {
         MFEM_ASSERT(Hel_offsets[nel] == nel*nf*nc*nf*nc,
                     "the batched face-pair loop needs one trace size");
         ComputeElementsHBatched(mode, el_0, nel, na, nd, nf, nc, AiBt_all,
                                 face_map, ws, Hel_data);
      }
      else
      {
         real_t * const Hbuf =
            (Hel_data.Size() > 0) ? Hel_data.HostWrite() : NULL;

#ifdef MFEM_USE_OPENMP
         #pragma omp parallel if (asm_mode == AssemblyMode::Threaded)
#endif
         {
            // PER THREAD, and that is the whole reason this is a `parallel`
            // region with a `for` inside rather than a `parallel for`: the
            // workspace holds ComputeElementH()'s dense temporaries, and one
            // shared between threads is the race this branch has recorded
            // twice already (SumNLFIntegrator, VectorBlockDiagonalIntegrator).
            // Per CHUNK per thread rather than per element, which is the
            // point -- chunks are hundreds of elements.
            SerialHWorkspace ews;
#ifdef MFEM_USE_OPENMP
            #pragma omp for schedule(dynamic)
#endif
            for (int el = el_0; el < el_1; el++)
            {
               ComputeElementH(el, mode,
                               Hbuf ? Hbuf + Hel_offsets[el-el_0] : NULL,
                               ews, prefactored ? &AiBt_all : NULL);
            }
         }
      }

      if (!assemble) { continue; }

      if (thmap)
      {
         // No HostRead() here, and that is the transfer this mode exists to
         // remove: the blocks stay where the face-pair kernel left them and
         // the CSR data is written in place. Nothing per chunk is host work.
         ScatterElementsHBatched(*thmap, el_0, nel, Hel_data, Hcsr);
         continue;
      }

      // The transfer the scatter needs, and the only one this route takes per
      // chunk. It is what a device-resident assembly of H would have to
      // remove, and removing it means a different scatter -- see
      // SetAssemblyMode() on why the sparse one cannot even be threaded.
      const real_t * const Hbuf_r = Hel_data.HostRead();

      for (int el = el_0; el < el_1; el++)
      {
         ScatterElementH(el, Hbuf_r + Hel_offsets[el-el_0], *H_, hws);
      }
   }


   if (batched_asm)
   {
      // THE copy back, moved to after the loop rather than before it. The
      // face pairs read the local blocks on the device; what still needs them
      // on the host is everything downstream -- the per-element MultInv() the
      // reduced and NPC routes call, and the raw-pointer readers of the
      // offset arrays. The comment on the other branch is the whole argument
      // for why it is unconditional.
      //
      // **AiBt_all is deliberately NOT read back here, and it was.** It is a
      // local of this routine (declared above, dead at the closing brace) and
      // the only host reader of it anywhere is ComputeElementH(), which is
      // the OTHER branch -- so on this path the call moved na*nd*NE doubles
      // down for nobody. 5.3 MB per gradient at order 2 on 64x64. The
      // !batched_asm branch keeps its own read and needs it.
      SyncLocalBlocksToHost();
   }

   // Everything past here finalizes the assembled matrix, and there is none
   // in GradientMode::MatrixFree: the loop above did the factorisation, which
   // is the whole of what that mode wanted.
   if (!assemble) { return; }

   if (thmap)
   {
      // Hand H_'s own Memory the flags the scatter left on the view, then
      // bring the values down.
      H_->GetMemoryData().Sync(Hcsr.GetMemory());

      // **The values come down here, and an attempt to keep them on the
      // device SEGFAULTED under Device("debug").** The two host operations
      // that force it are SetDiagIdentity() below and the caller's
      // EliminateRow() loop, and both were rewritten as kernels
      // (HDGTraceEliminateRows, HDGTraceDiagIdentity) to remove exactly this
      // transfer -- so the kernels are not the missing piece. The consumers
      // are: cuDSS is fine (device pointers) and so is UMFPackSolver
      // (HostReadI/J/Data), but **GSSmoother is not** -- SparseMatrix's
      // Gauss-Seidel sweeps index I, J and A through Memory::operator[], a
      // raw host access that neither syncs nor invalidates, and the unit
      // tests precondition the trace solve with it. Moving the readback to
      // after the kernels was not enough either, so at least one more raw
      // reader is in the chain and it has not been found.
      //
      // **So this stays, and "structural" is the honest label**: making the
      // trace matrix device-resident is a change to SparseMatrix's host
      // methods or a caller contract, not something ComputeH() can do. The
      // debug backend is what established it; a host build passes either way,
      // which is why this needed running rather than reasoning about.
      //
      // ReadWrite and not Read, for SyncLocalBlocksToHost()'s reason.
      H_->HostReadWriteData();

      // Nothing to finalize: the matrix was built as a CSR. Nor is there a
      // diagonal to force -- a face is its own neighbour, so every row's
      // block for its own face is present and carries the diagonal, which is
      // what the SearchRow() loop below is for. BuildTraceHMap() refuses the
      // policies whose pattern is not the structural one.
      MFEM_ASSERT(diag_policy == DIAG_ONE || diag_policy == DIAG_ZERO, "");
   }
   else if (diag_policy == DIAG_ONE || diag_policy == DIAG_ZERO)
   {
      // put zeroes on the diagonal
      for (int i = 0; i < H_->Height(); i++)
      {
         H_->SearchRow(i, i);
      }
      H_->Finalize(0);
   }
   else
   {
      H_->Finalize(skip_zeros);
   }

   if (!ParallelC())
   {
      const SparseMatrix *cP = c_fes.GetConformingProlongation();
      if (cP)
      {
         if (H_->Height() != cP->Width())
         {
            SparseMatrix *cH = mfem::RAP(*cP, *H_, *cP);
            H_.reset(cH);
         }
      }

      // ensure diagonal is non-zero
      if (diag_policy == DIAG_ONE)
      {
         H_->SetDiagIdentity();
      }
   }
}

#ifdef MFEM_USE_MPI
void DarcyHybridization::ComputeParH(ComputeHMode mode,
                                     std::unique_ptr<SparseMatrix> &H_, OperatorHandle &pH_) const
{
   ComputeH(mode, H_);

   if (!ParallelC())
   {
      pH_.Reset(H_.get(), false);
   }
   else // parallel
   {
      OperatorHandle dH(pH_.Type()), pP(pH_.Type());
      dH.MakeSquareBlockDiag(c_pfes->GetComm(), c_pfes->GlobalVSize(),
                             c_pfes->GetDofOffsets(), H_.get());
      // TODO - construct Dof_TrueDof_Matrix directly in the pS format
      pP.ConvertFrom(c_pfes->Dof_TrueDof_Matrix());
      pH_.MakePtAP(dH, pP);
      dH.Clear();
      pP.Clear();
      H_.reset();

      if (diag_policy == DIAG_ONE)
      {
         MFEM_ASSERT(pH_.Type() == Operator::Hypre_ParCSR,
                     "Fix of the diagonal is implemented only for HypreParMatrix");
         pH_.As<HypreParMatrix>()->EliminateZeroRows();
      }
   }
}
#endif //MFEM_USE_MPI

void DarcyHybridization::GetCtFaceMatrix(
   int f, int side, DenseMatrix &Ct_) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
   const int f_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];

   if (side == 0)
   {
      Ct_.Reset(const_cast<real_t*>(&Ct_data[Ct_offsets[f]]), f_size_1, c_size);
   }
   else
   {
      MFEM_ASSERT(el2 >= 0, "Invalid element");
      const int f_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];
      Ct_.Reset(const_cast<real_t*>(&Ct_data[Ct_offsets[f] + f_size_1*c_size]),
                f_size_2, c_size);
   }
}

void DarcyHybridization::GetEFaceMatrix(
   int f, int side, DenseMatrix &E) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
   const int d_size_1 = Df_f_offsets[el1+1] - Df_f_offsets[el1];

   if (side == 0)
   {
      E.Reset(&E_data[E_offsets[f]], d_size_1, c_size);
   }
   else
   {
      MFEM_ASSERT(el2 >= 0, "Invalid element");
      const int d_size_2 = Df_f_offsets[el2+1] - Df_f_offsets[el2];
      E.Reset(&E_data[E_offsets[f] + d_size_1*c_size], d_size_2, c_size);
   }
}

void DarcyHybridization::GetGFaceMatrix(
   int f, int side, DenseMatrix &G) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
   const int d_size_1 = Df_f_offsets[el1+1] - Df_f_offsets[el1];

   if (side == 0)
   {
      G.Reset(G_data.GetData() + G_offsets[f], c_size, d_size_1);
   }
   else
   {
      MFEM_ASSERT(el2 >= 0, "Invalid element");
      const int d_size_2 = Df_f_offsets[el2+1] - Df_f_offsets[el2];
      G.Reset(G_data.GetData() + G_offsets[f] + d_size_1*c_size, c_size, d_size_2);
   }
}

void DarcyHybridization::GetHFaceMatrix(int f, DenseMatrix &H) const
{
   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();

   H.Reset(&H_data[H_offsets[f]], c_size, c_size);
}

void DarcyHybridization::GetCtSubMatrix(int el, const Array<int> &c_dofs,
                                        DenseMatrix &Ct_l) const
{
   const int hat_offset = hat_offsets[el  ];
   const int hat_size = hat_offsets[el+1] - hat_offset;
   const int f_size = Af_f_offsets[el+1] - Af_f_offsets[el];

   Array<int> vdofs;
   fes.GetElementVDofs(el, vdofs);

   Ct_l.SetSize(f_size, c_dofs.Size());
   Ct_l = 0.;

   int i = 0;
   for (int row = hat_offset; row < hat_offset + hat_size; row++)
   {
      if (hat_dofs_marker[row] == 1) { continue; }
      const int ncols = Ct->RowSize(row);
      const int *cols = Ct->GetRowColumns(row);
      const real_t *vals = Ct->GetRowEntries(row);
      for (int j = 0; j < c_dofs.Size(); j++)
      {
         const int cdof = (c_dofs[j]>=0)?(c_dofs[j]):(-1-c_dofs[j]);
         for (int col = 0; col < ncols; col++)
            if (cols[col] == cdof)
            {
               real_t val = vals[col];
               Ct_l(i,j) = (c_dofs[j] >= 0)?(+val):(-val);
               break;
            }
      }
      i++;
   }
}

void DarcyHybridization::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(bfin, "DarcyHybridization must be finalized");

   if (H)
   {
      H->Mult(x, y);
      return;
   }

   MultNL(MultNlMode::Mult, darcy_rhs, x, y);

   // Essential trace dofs. There is no assembled matrix on this path to move
   // columns out of, so the constraint is carried the way NonlinearForm
   // carries one: the values ride in @a x, the residual is zero on those rows,
   // the gradient has a unit diagonal there, and Newton therefore leaves them
   // alone. The reduced right-hand side is zeroed to match in
   // EliminateTraceTrueDofsInRHS(). Inert unless SetEssentialBC() was called.
   y.SetSubVector(ess_tdof_list, 0.);
}

Operator &DarcyHybridization::GetGradient(const Vector &x) const
{
   MFEM_VERIFY(bfin, "DarcyHybridization must be finalized");

   if (H) { return *H; }

   return ReducedGradient(MultNlMode::Grad, x);
}

Operator &DarcyHybridization::ReducedGradient(MultNlMode mode,
                                              const Vector &x_tr) const
{
   if (!Df_data.Size()) { AllocD(); }// D is resetted in ConstructGrad()
   if (!E_data.Size() || !G_data.Size()) { AllocEG(); }// E and G are rewritten
   if (!H_data.Size()) { AllocH(); }
   else if (c_nlfi_p || c_nlfi)
   {
      // H is resetted here for additive double side integration
      H_data = 0.;
   }

   Vector y;//dummy
   BlockVector zero_b;
   const BlockVector &b = (mode == MultNlMode::GradAtFields)
                          ? (ZeroLoad(zero_b, false), zero_b) : darcy_rhs;
   MultNL(mode, b, x_tr, y);

   if (grad_mode == GradientMode::Assembled)
   {
      //assemble gradient matrix
      Grad.reset();
      ComputeH(ComputeHMode::Gradient, Grad);
      // Rows only. The columns could be eliminated too, but they need not be --
      // the correction is zero on these dofs, so their columns contribute
      // nothing -- and EliminateRowCol() would demand a structurally symmetric
      // matrix, which the reduced gradient is not.
      //
      // As ONE kernel over the essential rows when the matrix came from the
      // batched trace assembly, which is what keeps it device-resident:
      // SparseMatrix::EliminateRow() reaches I, J and A through
      // Memory::operator[], so the per-row loop would drag the whole CSR to
      // the host and cuDSS would push it straight back. The rows are disjoint
      // so there is nothing to reduce. Same arithmetic, and the original is
      // named in HDGTraceEliminateRows() so the two can be diffed.
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         Grad->EliminateRow(ess_tdof_list[i], Matrix::DIAG_ONE);
      }
      return *Grad;
   }

   // Matrix-free: the same factorisation, none of the assembly. Gradient::Mult
   // then applies H - [C G] M^-1 [C^T; E] one element at a time.
   Grad.reset();
   std::unique_ptr<SparseMatrix> H_unused;
   ComputeH(ComputeHMode::GradientFactorOnly, H_unused);
   MarkEmptyTraceRows();
   pGrad.Reset(new Gradient(*this));
   return *pGrad;
}

void DarcyHybridization::MultNL(MultNlMode mode, const Vector &bu,
                                const Vector &bp, const Vector &x, Vector &y,
                                BlockVector *r_local) const
{
   MFEM_ASSERT(mode != MultNlMode::AtFields || r_local,
               "MultNlMode::AtFields has nowhere to put the local residual");
   // The element loop below is host dense work reaching the local blocks
   // through raw pointers, and LocalFactorMode::Batched may have left them on
   // the device. See SyncLocalBlocksToHost() -- and note that this is the
   // matrix-free gradient's whole route to them, which is where it was found.
   SyncLocalBlocksToHost();
   const int NE = fes.GetNE();
   const int dim = fes.GetMesh()->Dimension();

   BlockVector yb;
   if (mode == MultNlMode::Sol)
   {
      yb.Update(y, darcy_offsets);
   }
   else
   {
      y = 0.0;
   }

   if (f_2_b.Size() == 0)
   {
      f_2_b = fes.GetMesh()->GetFaceToBdrElMap();
   }

   // NPC's element integrator as ONE kernel for the whole mesh, before the
   // element loop rather than inside it. See CanBatchLocalResidual() for what
   // is admitted and what is not; the element loop below then reads its own
   // slice instead of calling the integrator.
   //
   // The HostRead() is the transfer the offload plan's gate is about: the
   // consumer here is a host element loop, so a kernel landed on its own
   // pays it. Measured on `convdiff -p 2 -o 3 -dg -hb -nld -npc -nls 3 -bam`
   // at 128x128 and reported in the commit; it is a link in a chain that is
   // not yet closed, not a speedup.
   Vector ru_batched;
   const bool batched_lr = (mode == MultNlMode::AtFields)
                           && CanBatchLocalResidual();
   if (batched_lr)
   {
      BuildElementDofMaps();
      Vector u_all(el_u_dofs.Size());
      u_all.UseDevice(true);
      darcy_u.GetSubVector(el_u_dofs, u_all);
      HDGMixedConductionResidualBatched(fes, m_nlfi, u_all, ru_batched);
      // Back to the host, AND it has to stop advertising itself as device
      // data. Vector's element-wise operators take
      // `use_dev = UseDevice() || v.UseDevice()`, so a device-flagged slice
      // makes `bu += *elem_flux_row` inside LocalNLOperator a DEVICE
      // operation and leaves bu device-valid -- where every reader after it
      // is host code: B.MultTranspose(), the DenseMatrix AddMults, the
      // subtraction of bu_l.
      //
      // Measured under Device("debug"), and it is not subtle: without the
      // UseDevice(false) the FLUX row of the NPC residual comes out 2.7e-2
      // and 5.0e-2 wrong at orders 1 and 2, against a row norm of 0.29 and
      // 0.34, while the potential and trace rows stay exact to 4.4e-16. A
      // ten per cent error in exactly the one block this kernel writes, and
      // completely invisible with no Device configured -- which is why
      // tests/unit/miniapps/test_debug_device.cpp carries a case for it.
      ru_batched.HostRead();
      ru_batched.UseDevice(false);
   }

   // Serial keeps the original element order exactly, so its answer is the one
   // it always was; threaded walks the colours, and within a colour no two
   // elements share a face. See BuildElementColouring().
   const bool threaded = (asm_mode == AssemblyMode::Threaded);
   if (threaded) { BuildElementColouring(); }
   const int npasses = threaded ? colour_offsets.Size() - 1 : 1;

   // The state-carrying interior-face constraint gradient, for every
   // (element, face) pair at once, instead of one AssembleHDGFaceGrad() and
   // four DenseMatrix::CopyMN() per pair. Decided ONCE here: asking per
   // element would call CanBatchNLFaceGrad(), which walks every face, inside
   // the element loop. It runs AFTER the loop because D carries the element
   // mass Jacobian the loop sets, and only in GradAtFields because that is
   // the mode whose fields survive the loop. See AssembleNLFaceGradBatched().
   const bool batch_nl_faces = (mode == MultNlMode::GradAtFields) &&
                               CanBatchNLFaceGrad();

   // And the element BLOCKS, when ConstructGrad()'s A and D work is a copy.
   // Once per call for the same reason as the line above. Only in the
   // gradient modes: the residual modes do not touch A or D.
   const bool grad_mode_pass = (mode == MultNlMode::GradAtFields ||
                                mode == MultNlMode::GradMult ||
                                mode == MultNlMode::Grad);
   const bool ad_done = grad_mode_pass && CopyLinearGradBlocks();

   // Tier 1: the nonlinear interior-face constraint's RESIDUAL for every face
   // at once, after the loop, exactly as batch_nl_faces does for the
   // gradient. Decided once here for the same reason -- asking per element
   // would walk every face inside the element loop.
   const bool batch_nl_res =
      (mode == MultNlMode::AtFields) && r_local && CanBatchNLFaceResidual();

   // **The whole element loop, when there is no nonlinearity in it.** This
   // was the offload plan's largest open item: MultNL()'s loop is the one
   // stage that had no batched counterpart, it runs once per NPC residual,
   // and it is what made host ownership of Ct, E, G, H and the local blocks
   // mandatory. See LinearResidualBatched().
   if (mode == MultNlMode::AtFields && r_local
       && LinearResidualBatched(x, bu, bp, *r_local, y))
   {
      return;
   }

   for (int pass = 0; pass < npasses; pass++)
   {
      const int i0 = threaded ? colour_offsets[pass] : 0;
      const int i1 = threaded ? colour_offsets[pass+1] : NE;

#ifdef MFEM_USE_OPENMP
      #pragma omp parallel if (threaded)
#endif
      {
         // Per THREAD, not per element: the same reuse the serial loop got
         // from declaring these once, without the sharing.
         DenseMatrix H;
         BlockVector x_l;
         Array<int> c_dofs;
         Array<int> c_offsets;
         Array<int> faces, oris;
         Vector bu_l, bp_l, u_l, p_l, y_l;
         Vector ru_int;
         Vector mi_wk;   ///< MultInv()'s one temporary, hoisted above the loop
         /// LocalResidual()'s two output rows. Declared per element until
         /// DHAT put 1,548 malloc/free pairs on the two
         /// `Vector::operator=`s that fill them -- operator= calls SetSize(),
         /// which reallocates only when GROWING, so the cost was entirely
         /// that the destination was a fresh Vector every element. Here they
         /// are warm from the second element on. Per THREAD, being inside the
         /// parallel region and above the `omp for`.
         Vector ru_l, rp_l;
         /// The trace face loop's output row, and the block integrator's
         /// argument arrays. Declared per FACE per element until DHAT put
         /// 2,880 malloc/free pairs on `elvec.SetSize(ndofs)` inside
         /// HDGDiffusionIntegrator -- an allocation the frame attributes to
         /// the integrator and that in fact belongs to this caller, because
         /// the destination was a fresh Vector every face. One serves both
         /// branches below: c_nlfi_p and c_nlfi are mutually exclusive by
         /// construction (each SetConstraintIntegrators() overload resets the
         /// others), and even were they not, the two uses are sequential.
         Vector GpHx_l;
         Array<const FiniteElement*> gp_fe_arr;
         Array<const Vector*> gp_x_arr;
         Array<Vector*> gp_y_arr;
         Array<int> u_vdofs, p_dofs;
         TransWorkspace ws;

#ifdef MFEM_USE_OPENMP
         #pragma omp for schedule(dynamic)
#endif
         for (int i = i0; i < i1; i++)
         {
            const int el = threaded ? colour_order[i] : i;
            //Load RHS

            if (mode != MultNlMode::GradMult && mode != MultNlMode::GradAtFields)
            {
               GetFDofs(el, u_vdofs);
               bu.GetSubVector(u_vdofs, bu_l);

               fes_p.GetElementVDofs(el, p_dofs);
               bp.GetSubVector(p_dofs, bp_l);
               if (bsym)
               {
                  //In the case of the symmetrized system, the sign is oppposite!
                  bp_l.Neg();
               }
            }
            else
            {
               // GradMult has no right-hand side, and GradAtFields does not read
               // one: ConstructGrad() ignores it. Both still need the dof lists,
               // GradAtFields because that is how it reaches the retained fields.
               GetFDofs(el, u_vdofs);
               fes_p.GetElementVDofs(el, p_dofs);
               bu_l.SetSize(Af_f_offsets[el+1] - Af_f_offsets[el]);
               bu_l = 0.;
               bp_l.SetSize(Df_f_offsets[el+1] - Df_f_offsets[el]);
               bp_l = 0.;
            }

            switch (dim)
            {
               case 1:
                  fes.GetMesh()->GetElementVertices(el, faces);
                  break;
               case 2:
                  fes.GetMesh()->GetElementEdges(el, faces, oris);
                  break;
               case 3:
                  fes.GetMesh()->GetElementFaces(el, faces, oris);
                  break;
            }

            c_offsets.SetSize(faces.Size()+1);
            c_offsets[0] = 0;
            for (int f = 0; f < faces.Size(); f++)
            {
               const int c_size = c_fes.GetFaceElement(faces[f])->GetDof() * c_fes.GetVDim();
               c_offsets[f+1] = c_offsets[f] + c_size;
            }

            x_l.Update(c_offsets);
            for (int f = 0; f < faces.Size(); f++)
            {
               c_fes.GetFaceVDofs(faces[f], c_dofs);
               x.GetSubVector(c_dofs, x_l.GetBlock(f));
            }

            // bu - C^T x
            for (int f = 0; f < faces.Size(); f++)
            {
               int el1, el2;
               fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
               DenseMatrix Ct;
               GetCtFaceMatrix(faces[f], el1 != el, Ct);

               const Vector &x_f = x_l.GetBlock(f);

               Ct.AddMult_a(-1., x_f, bu_l);

               //bp - E x
               if (c_bfi_p || mode == MultNlMode::GradMult)
               {
                  DenseMatrix E;
                  GetEFaceMatrix(faces[f], el1 != el, E);

                  E.AddMult_a(-1., x_f, bp_l);
               }
            }

            if (mode != MultNlMode::GradMult)
            {
               if (mode == MultNlMode::AtFields || mode == MultNlMode::GradAtFields)
               {
                  // NPC. The fields are Newton state and arrive in darcy_u/darcy_p;
                  // nothing here solves, substitutes or linearises. That is the
                  // whole difference from every other mode, and it is why NPC has no
                  // local nonlinear iteration to globalise.
                  darcy_u.GetSubVector(u_vdofs, u_l);
                  darcy_p.GetSubVector(p_dofs, p_l);

                  if (mode == MultNlMode::GradAtFields)
                  {
                     ConstructGrad(el, faces, ws, x_l, u_l, p_l,
                                   batch_nl_faces, ad_done);
                     continue;
                  }

                  // The local rows of F, at exactly these fields. bu_l already
                  // carries -C^T x from the loop above and LocalNLOperator supplies
                  // E x on the potential row, so between them the trace coupling
                  // appears once on each row.
                  if (batched_lr)
                  {
                     // MakeRef and not `ru_int = Vector(ptr, n)`: Vector has a
                     // move assignment, so a prvalue on the right ALIASES the
                     // pointer and everything done to the left-hand side is
                     // written back through it. This is a read-only view and
                     // MakeRef says so.
                     ru_int.MakeRef(ru_batched, Af_f_offsets[el],
                                    u_vdofs.Size());
                     // Memory::MakeAlias() inherits the base's USE_DEVICE
                     // flag, so this is belt as well as braces -- see the
                     // note where ru_batched is filled.
                     ru_int.UseDevice(false);
                  }
                  LocalResidual(el, faces, x_l, bu_l, bp_l, u_l, p_l, ru_l, rp_l,
                                ws, batched_lr ? &ru_int : NULL, batch_nl_res);
                  r_local->GetBlock(0).AddElementVector(u_vdofs, ru_l);
                  r_local->GetBlock(1).AddElementVector(p_dofs, rp_l);
                  // and fall through to the trace row, which is the same assembly
                  // every other mode uses.
               }
               else
               {
                  //local u
                  if (darcy_u.Size() > 0)
                  {
                     //load the initial guess from the non-reduced solution vector
                     darcy_u.GetSubVector(u_vdofs, u_l);
                  }
                  else
                  {
                     u_l.SetSize(u_vdofs.Size());
                     u_l = 0.;//initial guess?

                  }

                  //local p
                  if (darcy_p.Size() > 0)
                  {
                     //load the initial guess from the non-reduced solution vector
                     darcy_p.GetSubVector(p_dofs, p_l);
                  }
                  else
                  {
                     p_l.SetSize(p_dofs.Size());
                     p_l = 0.;//initial guess?
                  }

                  //(A^-1 - A^-1 B^T S^-1 B A^-1) (bu - C^T sol)
                  MultInvNL(el, bu_l, bp_l, x_l, u_l, p_l, ws);
               }

               if (mode == MultNlMode::Sol)
               {
                  yb.GetBlock(0).SetSubVector(u_vdofs, u_l);
                  yb.GetBlock(1).SetSubVector(p_dofs, p_l);
                  continue;
               }
               else if (mode == MultNlMode::Grad)
               {
                  ConstructGrad(el, faces, ws, x_l, u_l, p_l, false, ad_done);
                  continue;
               }
            }
            else
            {
               // (A^-1 - A^-1 B^T S^-1 B A^-1) (bu - C^T sol), with the Jacobian's
               // (0,1) block -- this is a gradient application, so d(flux
               // residual)/dp belongs in it. Passing the linear -/+B^T alone is what
               // made the matrix-free gradient disagree with the assembled one
               // whenever the flux law depended on the potential.
               MultInv(el, bu_l, bp_l, u_l, p_l, true, &mi_wk);
            }

            // C u_l
            for (int f = 0; f < faces.Size(); f++)
            {
               int el1, el2;
               fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
               DenseMatrix Ct;
               GetCtFaceMatrix(faces[f], el1 != el, Ct);

               const Vector &x_f = x_l.GetBlock(f);

               y_l.SetSize(x_f.Size());
               Ct.MultTranspose(u_l, y_l);

               //G p_l + H x_l
               if (c_bfi_p || mode == MultNlMode::GradMult)
               {
                  //linear
                  DenseMatrix G;
                  GetGFaceMatrix(faces[f], el1 != el, G);

                  G.AddMult(p_l, y_l);

                  //integrate the face contrbution only on one (first) side
                  if (el1 == el)
                  {
                     GetHFaceMatrix(faces[f], H);
                     H.AddMult(x_f, y_l);
                  }
               }
               else
               {
                  //nonlinear
                  if (c_nlfi_p)
                  {
                     int type = NonlinearFormIntegrator::HDGFaceType::CONSTR
                                | NonlinearFormIntegrator::HDGFaceType::FACE;

                     FaceElementTransformations *FTr = GetFaceTransformation(faces[f], ws);

                     if (FTr->Elem2No >= 0 && batch_nl_res)
                     {
                        // left to AssembleNLFaceResidualBatched()
                     }
                     else if (FTr->Elem2No >= 0)
                     {
                        //interior
                        if (FTr->Elem1No != el) { type |= 1; }

                        // Cleared before every call: GpHx_l is shared
                        // scratch now, so it carries the last face's size in.
                        // A fresh local arrived at size 0, and the `Size() > 0`
                        // test below -- and `y_l += GpHx_l`'s own size check --
                        // both depend on that. SetSize(0) keeps the buffer.
                        GpHx_l.SetSize(0);

                        c_nlfi_p->AssembleHDGFaceVector(type,
                                                        *c_fes.GetFaceElement(faces[f]),
                                                        *fes_p.GetFE(el),
                                                        *FTr,
                                                        x_f, p_l, GpHx_l);

                        y_l += GpHx_l;
                     }
                     else
                     {
                        //boundary
                        const int bdr_attr = fes.GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

                        for (size_t i = 0; i < boundary_constraint_pot_nonlin_integs.size(); i++)
                        {
                           if (boundary_constraint_pot_nonlin_integs_marker[i]
                               && (*boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

                           GpHx_l.SetSize(0);   // per integrator; see above

                           boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                                           *c_fes.GetFaceElement(faces[f]),
                                                                                           *fes_p.GetFE(el),
                                                                                           *FTr,
                                                                                           x_f, p_l, GpHx_l);

                           y_l += GpHx_l;
                        }
                     }
                  }

                  if (c_nlfi)
                  {
                     const FiniteElement *fe_u = fes.GetFE(el);
                     const FiniteElement *fe_p = fes_p.GetFE(el);
                     // From the per-thread block, not four locals per face;
                     // an Array<T> from an initialiser list is a new[] per
                     // construction. See the note on GpHx_l.
                     Array<const FiniteElement*> &fe_arr = gp_fe_arr;
                     Array<const Vector*> &x_arr = gp_x_arr;
                     Array<Vector*> &y_arr = gp_y_arr;
                     fe_arr.SetSize(2); fe_arr[0] = fe_u;  fe_arr[1] = fe_p;
                     x_arr.SetSize(2);  x_arr[0] = &u_l;   x_arr[1] = &p_l;
                     y_arr.SetSize(3);
                     y_arr[0] = NULL; y_arr[1] = NULL; y_arr[2] = &GpHx_l;

                     int type = BlockNonlinearFormIntegrator::HDGFaceType::CONSTR
                                | BlockNonlinearFormIntegrator::HDGFaceType::FACE;

                     FaceElementTransformations *FTr = GetFaceTransformation(faces[f], ws);

                     if (FTr->Elem2No >= 0)
                     {
                        //interior
                        if (FTr->Elem1No != el) { type |= 1; }

                        // Cleared before every call: GpHx_l is shared
                        // scratch now, so it carries the last face's size in.
                        // A fresh local arrived at size 0, and the `Size() > 0`
                        // test below -- and `y_l += GpHx_l`'s own size check --
                        // both depend on that. SetSize(0) keeps the buffer.
                        GpHx_l.SetSize(0);

                        c_nlfi->AssembleHDGFaceVector(type,
                                                      *c_fes.GetFaceElement(faces[f]),
                                                      fe_arr,
                                                      *FTr,
                                                      x_f, x_arr, y_arr);

                        if (GpHx_l.Size() > 0) { y_l += GpHx_l; }
                     }
                     else
                     {
                        //boundary
                        const int bdr_attr = fes.GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

                        for (size_t i = 0; i < boundary_constraint_nonlin_integs.size(); i++)
                        {
                           if (boundary_constraint_nonlin_integs_marker[i]
                               && (*boundary_constraint_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

                           GpHx_l.SetSize(0);   // per integrator; see above

                           boundary_constraint_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                                       *c_fes.GetFaceElement(faces[f]),
                                                                                       fe_arr,
                                                                                       *FTr,
                                                                                       x_f, x_arr, y_arr);

                           if (GpHx_l.Size() > 0) { y_l += GpHx_l; }
                        }
                     }
                  }
               }

               c_fes.GetFaceVDofs(faces[f], c_dofs);
               y.AddElementVector(c_dofs, y_l);
            }
         }
      }
   }

   if (batch_nl_faces) { AssembleNLFaceGradBatched(x); }
   // Tier 1, after the loop for the same reason the gradient's pass is: the
   // element states have to survive it. The element loop skipped the interior
   // face constraint on the strength of the same predicate.
   // VERIFY rather than a plain call: the loop above skipped the interior
   // face constraint on the strength of batch_nl_res, so a routine that
   // declined here would drop the term with no symptom but a wrong answer.
   if (batch_nl_res)
   {
      MFEM_VERIFY(AssembleNLFaceResidualBatched(x, *r_local, y, true),
                  "the batched face residual declined after the element loop "
                  "had already skipped the interior faces for it");
   }
}

void DarcyHybridization::ParMultNL(MultNlMode mode, const BlockVector &b_t,
                                   const Vector &x_t, Vector &y_t) const
{
   Vector x;
   const Operator *tr_cP;

   if (!ParallelC())
   {
      tr_cP = c_fes.GetConformingProlongation();
      if (!tr_cP)
      {
         x.MakeRef(const_cast<Vector&>(x_t), 0, x_t.Size());
      }
      else
      {
         x.SetSize(c_fes.GetVSize());
         tr_cP->Mult(x_t, x);
      }
   }
   else
   {
      x.SetSize(c_fes.GetVSize());
      c_fes.GetProlongationMatrix()->Mult(x_t, x);
   }

   Vector bu;
   const Operator *cR;

   if (!ParallelU())
   {
      if (!(cR = fes.GetConformingRestriction()))
      {
         bu.MakeRef(const_cast<Vector&>(b_t.GetBlock(0)), 0, fes.GetVSize());
      }
      else
      {
         bu.SetSize(fes.GetVSize());
         cR->MultTranspose(b_t.GetBlock(0), bu);
      }
   }
   else
   {
      bu.SetSize(fes.GetVSize());
      fes.GetRestrictionOperator()->MultTranspose(b_t.GetBlock(0), bu);
   }

   const Vector &bp = b_t.GetBlock(1);
   Vector y;

   if (mode == MultNlMode::Sol)
   {
      if (!ParallelU() && !cR)
      {
         y.MakeRef(y_t, 0, darcy_offsets.Last());
      }
      else
      {
         y.SetSize(darcy_offsets.Last());
      }
   }
   else
   {
      const Operator *tr_cR;
      if (!ParallelC() && !(tr_cR = c_fes.GetRestrictionOperator()))
      {
         y.MakeRef(y_t, 0, c_fes.GetVSize());
      }
      else
      {
         y.SetSize(c_fes.GetVSize());
      }
   }

   MultNL(mode, bu, bp, x, y);

   if (mode == MultNlMode::Sol)
   {
      if (ParallelU() || cR)
      {
         BlockVector yb(y, darcy_offsets);
         BlockVector yb_t(y_t, darcy_toffsets);

         if (!ParallelU())
         {
            cR->Mult(yb.GetBlock(0), yb_t.GetBlock(0));
         }
         else
         {
            fes.GetRestrictionOperator()->Mult(yb.GetBlock(0), yb_t.GetBlock(0));
         }

         yb_t.GetBlock(1) = yb.GetBlock(1);
      }
   }
   else if (mode != MultNlMode::Grad && mode != MultNlMode::GradAtFields)
   {
      // A gradient pass leaves nothing in y -- it writes the local blocks --
      // so there is nothing to assemble and y_t is a dummy the caller sized
      // at zero. GradAtFields is NPC's gradient pass and is the same case.
      if (!ParallelC())
      {
         if (tr_cP)
         {
            tr_cP->MultTranspose(y, y_t);
         }
      }
      else
      {
         c_fes.GetProlongationMatrix()->MultTranspose(y, y_t);
      }
   }
}

void DarcyHybridization::Finalize()
{
   if (bfin) { return; }

   // ComputeH(Linear) factors each element's A and D IN PLACE and keeps no
   // copy, which is right when the only thing ever asked of the hybridization
   // is one reduced solve and wrong for NPC, which needs the blocks
   // themselves at arbitrary states. A form with EnableNPC() is therefore
   // finalized as a fully nonlinear one -- blocks kept, no reduced H, which
   // is correct because NPCGradient() assembles its own.
   if (!NPCEnabled())
   {
#ifndef MFEM_USE_MPI
      ComputeH(ComputeHMode::Linear, H);
#else //MFEM_USE_MPI      
      ComputeParH(ComputeHMode::Linear, H, pH);
      pOp = pH;
#endif //MFEM_USE_MPI
      EliminateTraceTrueDofs(diag_policy);
   }
   else
   {
      // Each of the two specialised modes leaves one of Af_lin_data /
      // Df_lin_data empty, on the guarantee that the corresponding NONLINEAR
      // integrator supplies that block. IsNonlinear() provided that guarantee
      // and bnpc does not, so a form with no nonlinear integrator at all must
      // fall through to FullNL, which is the only branch that backs up both.
      if (IsNonlinear() && !m_nlfi_u && !m_nlfi && !c_nlfi)
      {
         lop_type = LocalOpType::PotNL;
         // backup the data for gradient construction
         Af_lin_data = Af_data;
         // The potential mass is the nonlinear one, so Df_data is about to
         // become the DESTINATION for its per-element Jacobian -- ConstructGrad()
         // zeroes it. Anything already assembled there has to survive.
         //
         // **Not only a face constraint's contribution.** This comment used
         // to add "there is no linear potential mass in this branch, or it
         // would not be nonlinear", and that is false: M_p and Mnl_p can
         // both be present and IsNonlinear() is still true, which is the
         // configuration gffp builds. The arithmetic here is additive either
         // way so the backup is right regardless, but the case the comment
         // said could not arise is a real one -- and downstream it WAS a
         // defect, because the three consumers gated on c_bfi_p. See
         // ConstructGrad(). Backing it up here is what lets a linear c_bfi_p coexist
         // with a nonlinear m_nlfi_p; the three consumers then add it as a
         // further term. Without it that contribution is silently lost after
         // the first gradient, which is what the refusal in
         // SetPotMassNonlinearIntegrator() used to be avoiding.
         if (!D_empty) { Df_lin_data = Df_data; }
         InvertA();
      }
      else if (IsNonlinear() && !m_nlfi_p && !c_nlfi_p && !c_bfi_p && !D_empty
               && !m_nlfi && !c_nlfi)
      {
         lop_type = LocalOpType::FluxNL;
         // backup the data for gradient construction
         Df_lin_data = Df_data;
         InvertD();
         // Df_data now holds the FACTORED linear potential mass, which the
         // local solve needs, so the Schur complement gets storage of its
         // own. Allocated here rather than in AllocD(), which runs before
         // lop_type is known.
         Sf_data.SetSize(Df_offsets.Last());
         Sf_data = 0.;
         Sf_ipiv.SetSize(Df_f_offsets.Last());
      }
      else
      {
         lop_type = LocalOpType::FullNL;
         if (!A_empty)
         {
            Swap(Af_data, Af_lin_data);
            Af_data.SetSize(Af_offsets.Last());
            Af_data = 0.;
         }

         if (!D_empty)
         {
            Swap(Df_data, Df_lin_data);
            if (!Df_data.Size())
            {
               Df_data.SetSize(Df_offsets.Last());
               Df_data = 0.;
            }
         }
      }

#ifdef MFEM_USE_MPI
      pOp.Reset(new ParOperator(*this));
#endif //MFEM_USE_MPI
   }

   bfin = true;
}

void DarcyHybridization::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                             const BlockVector &x, BlockVector &b)
{
   if (NPCEnabled())
   {
      MFEM_ASSERT(!ParallelU() && !ParallelP(),
                  "In parallel, use ParallelEliminateTDofsInRHS() instead!");

      //save the rhs for initial guess in the iterative local solve
      darcy_u = x.GetBlock(0);
      darcy_p = x.GetBlock(1);
   }

   MFEM_ASSERT(x.Size() == fes.GetVSize() + fes_p.GetVSize(),
               "Wrong size of the solution vector!");
   MFEM_ASSERT(b.Size() == fes.GetVSize() + fes_p.GetVSize(),
               "Wrong size of the rhs vector!");

   const int NE = fes.GetNE();

   const Vector &xu = x.GetBlock(0);
   Vector &bu = b.GetBlock(0);
   Vector &bp = b.GetBlock(1);

   // Threaded only when both field spaces are discontinuous, so each
   // element's dofs are its own; see CanThreadFieldLoop().
   const bool threaded = CanThreadFieldLoop();
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (threaded)
#endif
   {
      Vector u_e, bu_e, bp_e;
      Array<int> u_vdofs, p_dofs, edofs;
#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int el = 0; el < NE; el++)
      {
         GetEDofs(el, edofs);
         if (edofs.Size() == 0) { continue; }

         xu.GetSubVector(edofs, u_e);
         u_e.Neg();

         //bu -= A_e u_e
         const int a_size = hat_offsets[el+1] - hat_offsets[el];
         const DenseMatrix Ae(&Ae_data[Ae_offsets[el]], a_size, edofs.Size());

         bu_e.SetSize(a_size);
         Ae.Mult(u_e, bu_e);

         fes.GetElementVDofs(el, u_vdofs);
         bu.AddElementVector(u_vdofs, bu_e);

         //bp -= B_e u_e
         const int d_size = Df_f_offsets[el+1] - Df_f_offsets[el];
         const DenseMatrix Be(&Be_data[Be_offsets[el]], d_size, edofs.Size());

         bp_e.SetSize(d_size);
         Be.Mult(u_e, bp_e);
         if (bsym)
         {
            //In the case of the symmetrized system, the sign is opposite!
            bp_e.Neg();
         }

         fes_p.GetElementVDofs(el, p_dofs);
         bp.AddElementVector(p_dofs, bp_e);
      }
   }

   for (int vdof : vdofs_flux)
   {
      bu(vdof) = xu(vdof);//<--can be arbitrary as it is ignored
   }
}

void DarcyHybridization::EliminateTrueDofsInRHS(
   const Array<int> &tdofs_flux, const BlockVector &x_t, BlockVector &b_t)
{
   Vector xu, bu;

   if (!ParallelU())
   {
      const Operator *cP = fes.GetConformingProlongation();
      if (!cP)
      {
         xu.MakeRef(const_cast<Vector&>(x_t.GetBlock(0)), 0, fes.GetVSize());
      }
      else
      {
         xu.SetSize(cP->Height());
         cP->Mult(x_t.GetBlock(0), xu);
      }

      const Operator *cR = fes.GetConformingRestriction();
      if (!cR)
      {
         bu.MakeRef(b_t.GetBlock(0), 0, fes.GetVSize());
      }
      else
      {
         bu.SetSize(cR->Width());
         cR->MultTranspose(b_t.GetBlock(0), bu);
      }
   }
   else
   {
      xu.SetSize(fes.GetVSize());
      fes.GetProlongationMatrix()->Mult(x_t.GetBlock(0), xu);
      bu.SetSize(xu.Size());
      fes.GetRestrictionOperator()->MultTranspose(b_t.GetBlock(0), bu);
   }

   if (NPCEnabled())
   {
      //save the rhs for initial guess in the iterative local solve
      darcy_u = xu;
      darcy_p = x_t.GetBlock(1);
   }

   Vector &bp = b_t.GetBlock(1);

   const int NE = fes.GetNE();

   // Threaded only when both field spaces are discontinuous, so each
   // element's dofs are its own; see CanThreadFieldLoop().
   const bool threaded = CanThreadFieldLoop();
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (threaded)
#endif
   {
      Vector u_e, bu_e, bp_e;
      Array<int> u_vdofs, p_dofs, edofs;

#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int el = 0; el < NE; el++)
      {
         GetEDofs(el, edofs);
         if (edofs.Size() == 0) { continue; }

         xu.GetSubVector(edofs, u_e);
         u_e.Neg();

         //bu -= A_e u_e
         const int a_size = hat_offsets[el+1] - hat_offsets[el];
         const DenseMatrix Ae(&Ae_data[Ae_offsets[el]], a_size, edofs.Size());

         bu_e.SetSize(a_size);
         Ae.Mult(u_e, bu_e);

         fes.GetElementVDofs(el, u_vdofs);
         bu.AddElementVector(u_vdofs, bu_e);

         //bp -= B_e u_e
         const int d_size = Df_f_offsets[el+1] - Df_f_offsets[el];
         const DenseMatrix Be(&Be_data[Be_offsets[el]], d_size, edofs.Size());

         bp_e.SetSize(d_size);
         Be.Mult(u_e, bp_e);
         if (bsym)
         {
            //In the case of the symmetrized system, the sign is opposite!
            bp_e.Neg();
         }

         fes_p.GetElementVDofs(el, p_dofs);
         bp.AddElementVector(p_dofs, bp_e);
      }
   }

   if (!ParallelU())
   {
      const Operator *cP = fes.GetConformingProlongation();
      if (cP)
      {
         cP->MultTranspose(bu, b_t.GetBlock(0));
      }
   }
   else
   {
      fes.GetProlongationMatrix()->MultTranspose(bu, b_t.GetBlock(0));
   }

   for (int tdof : tdofs_flux)
   {
      b_t(tdof) = x_t(tdof);//<--can be arbitrary as it is ignored
   }
}

void DarcyHybridization::EliminateTraceTrueDofs(const Array<int> &tdofs,
                                                DiagonalPolicy dpolicy)
{
   if (NPCEnabled()) { return; } // not implemented

   if (!ParallelC())
   {
      He.reset(new SparseMatrix(H->Height()));

      if (tdofs.Size() == 0) { return; }

      for (int vdof : tdofs)
      {
         H->EliminateRowCol(vdof, *He, dpolicy);
      }

      He->Finalize();
   }
   else
   {
#ifdef MFEM_USE_MPI
      MFEM_ASSERT(pH.Type() == Operator::Hypre_ParCSR,
                  "Implemented for HypreParMatrix only!");
      pHe.Reset(pH.As<HypreParMatrix>()->EliminateRowsCols(tdofs));
#endif //MFEM_USE_MPI
   }
}

void DarcyHybridization::EliminateTraceTrueDofs(DiagonalPolicy dpolicy)
{
   EliminateTraceTrueDofs(GetEssentialTrueDofs(), dpolicy);
}

void DarcyHybridization::EliminateTraceTrueDofsInRHS(const Array<int> &tdofs_,
                                                     const Vector &x, Vector &b)
{
   if (NPCEnabled())
   {
      // Nothing to eliminate -- the reduced operator is nonlinear and there is
      // no assembled matrix to move columns out of. The essential values ride
      // in @a x and the residual is masked on those rows by Mult(), so all
      // that is needed here is a right-hand side that agrees.
      b.SetSubVector(tdofs_, 0.);
      return;
   }

   if (!ParallelC())
   {
      MFEM_VERIFY(H && He, "The hybridization matrix is not assembled!");
      He->AddMult(x, b, -1.);
      H->PartMult(tdofs_, x, b);
   }
   else
   {
#ifdef MFEM_USE_MPI
      MFEM_VERIFY(pH.Ptr() && pHe.Ptr(),
                  "The hybridization matrix is not assembled!");
      pH.As<HypreParMatrix>()->EliminateBC(*pHe.As<HypreParMatrix>(), tdofs_, x, b);
#endif //MFEM_USE_MPI
   }
}

void DarcyHybridization::EliminateTraceTrueDofsInRHS(const Vector &x, Vector &b)
{
   EliminateTraceTrueDofsInRHS(GetEssentialTrueDofs(), x, b);
}

void DarcyHybridization::MultInvNL(int el, const Vector &bu_l,
                                   const Vector &bp_l, const BlockVector &x_l,
                                   Vector &u_l, Vector &p_l,
                                   TransWorkspace &ws) const
{
   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

   MFEM_ASSERT(bu_l.Size() == a_dofs_size &&
               bp_l.Size() == d_dofs_size, "Incompatible size");

   //prepare vector of local traces

   Array<int> faces, oris;
   const int dim = fes.GetMesh()->Dimension();
   switch (dim)
   {
      case 1:
         fes.GetMesh()->GetElementVertices(el, faces);
         break;
      case 2:
         fes.GetMesh()->GetElementEdges(el, faces, oris);
         break;
      case 3:
         fes.GetMesh()->GetElementFaces(el, faces, oris);
         break;
   }

   //construct the local operator

   LocalNLOperator *lop;

   switch (lop_type)
   {
      case LocalOpType::FluxNL:
         lop = new LocalFluxNLOperator(*this, el, bp_l, x_l, faces, ws);
         break;
      case LocalOpType::PotNL:
         lop = new LocalPotNLOperator(*this, el, bu_l, x_l, faces, ws);
         break;
      case LocalOpType::FullNL:
         lop = new LocalNLOperator(*this, el, x_l, faces, ws);
         break;
   }

   //solve the local system

   IterativeSolver *lsolver;
   bool use_prec;
   switch (lsolve.type)
   {
      case LSsolveType::LBFGS:
         lsolver = new LBFGSSolver();
         use_prec = false;
         break;
      case LSsolveType::LBB:
         lsolver = new LBBSolver();
         use_prec = false;
         break;
      case LSsolveType::Newton:
         lsolver = new NewtonSolver();
         use_prec = true;
         break;
      default:
         MFEM_ABORT("Unknown local solver");
   }

   Solver *prec = NULL;
   if (use_prec)
   {
      IterativeSolver *iter_prec = NULL;
      switch (lsolve.prec.type)
      {
         case LPrecType::GMRES:
            prec = iter_prec = new GMRESSolver();
            break;
         case LPrecType::LU:
            prec = new DenseMatrixLUSolver();
            break;
         default:
            MFEM_ABORT("Unknown local preconditioner");
      }

      if (iter_prec)
      {
         iter_prec->SetMaxIter(lsolve.prec.iters);
         iter_prec->SetRelTol((lsolve.prec.rtol >= 0)?
                              (lsolve.prec.rtol):(lsolve.rtol));
         iter_prec->SetAbsTol((lsolve.prec.atol >= 0)?
                              (lsolve.prec.atol):(lsolve.atol));
      }
   }

   lsolver->SetOperator(*lop);
   if (prec) { lsolver->SetPreconditioner(*prec); }
   lsolver->SetMaxIter(lsolve.iters);
   lsolver->SetRelTol(lsolve.rtol);
   lsolver->SetAbsTol(lsolve.atol);
   lsolver->SetPrintLevel(lsolve.print_lvl);

   switch (lop_type)
   {
      case LocalOpType::FluxNL:
      {
         //solve the flux
         lsolver->SetAbsTol(std::max(bu_l.Norml2() * lsolve.rtol, lsolve.atol));
         lsolver->Mult(bu_l, u_l);

         //solve the potential
         static_cast<LocalFluxNLOperator*>(lop)->SolveP(u_l, p_l);
      }
      break;
      case LocalOpType::PotNL:
      {
         //solve the potential
         lsolver->SetAbsTol(std::max(bp_l.Norml2() * lsolve.rtol, lsolve.atol));
         lsolver->Mult(bp_l, p_l);

         //solve the flux
         static_cast<LocalPotNLOperator*>(lop)->SolveU(p_l, u_l);
      }
      break;
      case LocalOpType::FullNL:
      {
         //rhs
         BlockVector b(lop->GetOffsets());
         b.GetBlock(0) = bu_l;
         b.GetBlock(1) = bp_l;

         //x
         BlockVector x(lop->GetOffsets());
         x.GetBlock(0) = u_l;
         x.GetBlock(1) = p_l;

         //solve the flux and potential
         lsolver->SetAbsTol(std::max(b.Norml2() * lsolve.rtol, lsolve.atol));
         lsolver->Mult(b, x);

         u_l = x.GetBlock(0);
         p_l = x.GetBlock(1);
      }
      break;
   }

   // The one shared write the colouring does not cover, because it is not
   // per element: a plain += here loses counts under threading, and the count
   // is what the NPC regression references compare.
   {
      const int nit = lsolver->GetNumIterations();
#ifdef MFEM_USE_OPENMP
      #pragma omp atomic
#endif
      num_local_nl_iters += nit;
   }

   if (lsolver->GetConverged())
   {
      if (lsolve.print_lvl >= 0)
         mfem::out << "el: " << el
                   << " iters: " << lsolver->GetNumIterations()
                   << " rel. norm: " << lsolver->GetFinalRelNorm()
                   << std::endl;
   }
   else
   {
      mfem::out << "el: " << el
                << " not convered in " << lsolver->GetNumIterations() << " iters"
                << " rel. norm: " << lsolver->GetFinalRelNorm()
                << std::endl;
   }

   delete lsolver;
   delete prec;
   delete lop;
}

void DarcyHybridization::MarkEmptyTraceRows() const
{
   mf_diag_marker.SetSize(c_fes.GetVSize());
   mf_diag_marker = 1;   // 1 = no face carrying this dof contributes anything

   Array<int> c_dofs;
   const int NF = fes.GetMesh()->GetNumFaces();
   for (int f = 0; f < NF; f++)
   {
      // The trace row of a face is Ct^T u_l + G p_l + H x_f, so those three
      // are what can make it non-empty. E is not among them: it feeds the
      // local right-hand side, and if Ct, G and H all vanish the row is zero
      // whatever the local fields are.
      bool live = false;
      for (int i = Ct_offsets[f]; i < Ct_offsets[f+1] && !live; i++)
      {
         if (Ct_data[i] != 0.) { live = true; }
      }
      if (!live && G_data.Size() > 0)
      {
         for (int i = G_offsets[f]; i < G_offsets[f+1] && !live; i++)
         {
            if (G_data[i] != 0.) { live = true; }
         }
      }
      if (!live && H_data.Size() > 0)
      {
         for (int i = H_offsets[f]; i < H_offsets[f+1] && !live; i++)
         {
            if (H_data[i] != 0.) { live = true; }
         }
      }
      if (!live) { continue; }

      c_fes.GetFaceVDofs(f, c_dofs);
      for (int i = 0; i < c_dofs.Size(); i++)
      {
         mf_diag_marker[FiniteElementSpace::DecodeDof(c_dofs[i])] = 0;
      }
   }

   // That marker is in L-dofs, and the operator this is applied in works in
   // true dofs. Where the two differ, a true dof is empty only if every L-dof
   // feeding it is: sum the live flags through the prolongation and read off
   // the zeros. In parallel the assembled counterpart of this is
   // HypreParMatrix::EliminateZeroRows(), which is likewise a true-dof
   // operation.
   const Operator *P = ParallelC() ? c_fes.GetProlongationMatrix()
                       : static_cast<const Operator*>(
                          c_fes.GetConformingProlongation());
   if (!P) { return; }

   Vector live_l(mf_diag_marker.Size()), live_t(P->Width());
   for (int i = 0; i < live_l.Size(); i++)
   {
      live_l(i) = mf_diag_marker[i] ? 0.0 : 1.0;
   }
   P->MultTranspose(live_l, live_t);

   mf_diag_marker.SetSize(live_t.Size());
   for (int i = 0; i < live_t.Size(); i++)
   {
      mf_diag_marker[i] = (live_t(i) == 0.0) ? 1 : 0;
   }
}


void DarcyHybridization::EnableNPC()
{
   MFEM_VERIFY(!bfin, "EnableNPC() must be called before the hybridization is "
               "finalized; it changes what Finalize() keeps.");
   if (bnpc) { return; }
   bnpc = true;

   // Init() gates AllocH() on NPCEnabled(), so a flag set after it has run
   // leaves H_data empty and GetHFaceMatrix() over a null pointer. Init()
   // normally HAS run by now: the hybridization does not exist until
   // DarcyForm::EnableHybridization() has made it, and that is what calls
   // Init(). Ct_data.Size() is Init()'s own "already run" test, reused here.
   if (Ct_data.Size() && !H_data.Size()) { AllocH(); }
}

void DarcyHybridization::SetGradientMode(GradientMode mode)
{
   if (mode == grad_mode) { return; }

   grad_mode = mode;

   // Whatever was built belongs to the mode that built it.
   Grad.reset();
   pGrad.Clear();
}

void DarcyHybridization::SetAssemblyMode(AssemblyMode mode)
{
   if (mode == AssemblyMode::Threaded)
   {
      // Refused rather than quietly downgraded. A caller that asks for this
      // is asking a performance question, and answering it with the serial
      // loop would report a speedup nobody got.
#ifndef MFEM_USE_OPENMP
      MFEM_ABORT("AssemblyMode::Threaded needs MFEM_USE_OPENMP, and this build "
                 "has none: mfem::forall and every omp pragma in it reduce to "
                 "a serial loop.");
#endif
#ifndef MFEM_THREAD_SAFE
      MFEM_ABORT("AssemblyMode::Threaded needs MFEM_THREAD_SAFE: without it "
                 "GetElementFaces() keeps its orientation scratch in a "
                 "function-local static that every thread would share.");
#endif
   }

   asm_mode = mode;
}

/** @brief A correctly sized zero load, for the NPC gradient passes.

    ConstructGrad() does not read the right-hand side, so a gradient pass has
    no use for one -- but MultNL() and ParMultNL() both reach into the
    BlockVector they are handed before dispatching, and @a darcy_rhs is only
    filled as a side effect of ReduceRHS(), which NPC never calls. Depending on
    that side effect worked in serial only because the tests happened to call
    FormLinearSystem() first, and segfaulted in parallel the moment they did
    not. */
void DarcyHybridization::ZeroLoad(BlockVector &b, bool true_dofs) const
{
   Array<int> offs(3);
   offs[0] = 0;
   offs[1] = true_dofs ? fes.GetTrueVSize() : fes.GetVSize();
   offs[2] = true_dofs ? fes_p.GetTrueVSize() : fes_p.GetVSize();
   offs.PartialSum();
   b.Update(offs);
   b = 0.;
}

const Operator *DarcyHybridization::TraceProlongation() const
{
#ifdef MFEM_USE_MPI
   if (ParallelC()) { return c_fes.GetProlongationMatrix(); }
#endif
   // NULL for a serial DG_Interface trace space, where true dofs are L-dofs.
   return c_fes.GetConformingProlongation();
}

void DarcyHybridization::LocalResidual(int el, const Array<int> &faces,
                                       const BlockVector &x_l,
                                       const Vector &bu_l, const Vector &bp_l,
                                       const Vector &u_l, const Vector &p_l,
                                       Vector &ru_l, Vector &rp_l,
                                       TransWorkspace &ws,
                                       const Vector *elem_flux_row,
                                       bool skip_interior_faces) const
{
   // The local equations are lop(u, p) = (bu_l, bp_l) -- that is what the
   // local nonlinear solve solves in the other ordering -- so the residual is
   // one evaluation of the same operator rather than a solve with it.
   LocalNLOperator lop(*this, el, x_l, faces, ws, elem_flux_row,
                       skip_interior_faces);

   // From @a ws and not two locals: see TransWorkspace::lr_xv.
   BlockVector &xv = ws.lr_xv, &rv = ws.lr_rv;
   xv.Update(lop.GetOffsets());
   rv.Update(lop.GetOffsets());
   xv.GetBlock(0) = u_l;
   xv.GetBlock(1) = p_l;

   lop.Mult(xv, rv);

   ru_l = rv.GetBlock(0);
   ru_l -= bu_l;
   rp_l = rv.GetBlock(1);
   rp_l -= bp_l;
}

void DarcyHybridization::MultInv(int el, const Vector &bu, const Vector &bp,
                                 Vector &u, Vector &p, bool with_bnl,
                                 Vector *wk) const
{
   // ONE temporary, and @a wk is how a caller in an element loop stops it
   // being a fresh allocation per element: Vector::SetSize() does not shrink,
   // so a Vector hoisted above the loop is sized once for the mesh. Measured
   // at 1,024 malloc/free pairs on `convdiff -p 6 -nl -dg -hb -npc` at 256
   // elements -- exactly one per call, from NPCReduce() and NPCRecover().
   // A parameter and not a member, per this class's standing rule.
   //
   // The `AiBtSibp` that used to be declared beside it was never used.
   Vector local_wk;
   Vector &AiBtSiBAibu = wk ? *wk : local_wk;

   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

   MFEM_ASSERT(bu.Size() == a_dofs_size &&
               bp.Size() == d_dofs_size, "Incompatible size");

   // Load LU decomposition of A and Schur complement

   LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);
   // @a with_bnl is exactly "these are the Jacobian's blocks", and under
   // LocalOpType::FluxNL the Jacobian's Schur complement lives in Sf_data --
   // Df_data being the factored linear potential mass there. Everywhere else
   // the two coincide.
   const bool fluxnl_schur = (with_bnl && lop_type == LocalOpType::FluxNL
                              && Sf_data.Size() == Df_data.Size());
   LUFactors LU_S(fluxnl_schur ? &Sf_data[Df_offsets[el]]
                  : &Df_data[Df_offsets[el]],
                  fluxnl_schur ? &Sf_ipiv[Df_f_offsets[el]]
                  : &Df_ipiv[Df_f_offsets[el]]);

   // Load B

   const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                       d_dofs_size, a_dofs_size);

   //u = A^-1 bu
   u.SetSize(bu.Size());
   u = bu;
   LU_A.Solve(u.Size(), 1, u.GetData());

   //p = -S^-1 (B A^-1 bu - bp)
   p.SetSize(bp.Size());
   B.Mult(u, p);

   p -= bp;

   LU_S.Solve(p.Size(), 1, p.GetData());
   p.Neg();

   //u += -A^-1 B^T S^-1 (B A^-1 bu - bp)
   AiBtSiBAibu.SetSize(B.Width());
   B.MultTranspose(p, AiBtSiBAibu);

   if (with_bnl)
   {
      // The (0,1) block is s B^T + Bnl with s = -1 when symmetrized, and the
      // term below is applied with the factor -s, so Bnl enters here scaled by
      // s to come out with the factor -1 it needs.
      DenseMatrix Bnl;
      if (GetBnlMatrix(el, Bnl))
      {
         Bnl.AddMult_a((bsym)?(-1.):(1.), p, AiBtSiBAibu);
      }
   }

   LU_A.Solve(AiBtSiBAibu.Size(), 1, AiBtSiBAibu.GetData());

   if (bsym) { u += AiBtSiBAibu; }
   else { u -= AiBtSiBAibu; }
}

bool DarcyHybridization::CanBatchLocalSolve() const
{
   const int NE = fes.GetNE();

   // The condition is about STORAGE, not about which local operator is in
   // play: MultInvBatched() is a transcription of MultInv() and is valid
   // wherever MultInv() is, so what it needs is that the three arrays it
   // views as DenseTensors are present, are one block size, and are laid out
   // el*n*n. A first draft tested lop_type != FullNL instead and refused
   // every LINEAR problem, because lop_type is assigned only in the nonlinear
   // branch of Finalize() and its FullNL default therefore also means "not
   // decided". The test's own REQUIRE(can_batch_solve) is what said so.
   //
   // A zero block size is uniform but degenerate here, unlike in
   // CanBatchLocalFactor(): there is no solve to batch without a D block.
   return lfac_mode == LocalFactorMode::Batched
          && UniformBlockSize(Af_f_offsets, NE) > 0
          && UniformBlockSize(Df_f_offsets, NE) > 0
          && Af_data.Size() == Af_offsets.Last()
          && Df_data.Size() == Df_offsets.Last()
          && Bf_data.Size() == Bf_offsets.Last();
}

void DarcyHybridization::MultInvBatched(const Vector &bu, const Vector &bp,
                                        Vector &u, Vector &p,
                                        bool with_bnl) const
{
   const int NE = fes.GetNE();
   const int na = UniformBlockSize(Af_f_offsets, NE);
   const int nd = UniformBlockSize(Df_f_offsets, NE);

   MFEM_VERIFY(na > 0 && nd > 0,
               "The local blocks are not all one size; CanBatchLocalSolve() "
               "answers that before this is called.");
   MFEM_ASSERT(bu.Size() == na * NE && bp.Size() == nd * NE,
               "Incompatible size");

   // Which array holds the Schur complement -- the same question MultInv()
   // asks, and for the same reason.
   const bool fluxnl_schur = (with_bnl && lop_type == LocalOpType::FluxNL
                              && Sf_data.Size() == Df_data.Size());
   const Vector &S_data = fluxnl_schur ? Sf_data : Df_data;
   const Array<int> &S_ipiv = fluxnl_schur ? Sf_ipiv : Df_ipiv;

   // NewMemoryAndSize and not the raw-pointer constructor, for the reason
   // spelled out in InvertA(): the latter goes through Memory::Wrap(), which
   // sets VALID_HOST with no device type and pins every kernel below to the
   // host however the Device is configured.
   DenseTensor A, S, B;
   A.NewMemoryAndSize(Af_data.GetMemory(), na, na, NE, false);
   S.NewMemoryAndSize(S_data.GetMemory(), nd, nd, NE, false);
   B.NewMemoryAndSize(Bf_data.GetMemory(), nd, na, NE, false);

   u.SetSize(bu.Size());
   p.SetSize(bp.Size());

   // Every BatchedLinAlg entry point reads and writes through Read()/Write()
   // with their default on_dev = true, so the tensors go to the device
   // whatever their Memory says. The Vector element-wise operations between
   // them do NOT -- they follow UseDevice() -- so without this the route
   // ping-pongs: the batched product writes p on the device, `p -= bp` pulls
   // it back to run on the host, the next solve pushes it up again. Correct
   // either way, and three transfers per call slower.
   bu.UseDevice(true);
   bp.UseDevice(true);
   u.UseDevice(true);
   p.UseDevice(true);

   //u = A^-1 bu
   u = bu;
   BatchedLinAlg::LUSolve(A, Af_ipiv, u);

   //p = -S^-1 (B A^-1 bu - bp)
   //
   // beta = 0 and a separate subtraction, NOT beta = -1 folding bp into the
   // product: the batched kernel scales y by beta BEFORE accumulating, so
   // -bp + sum and sum - bp round differently and the route stops being
   // bit-for-bit the per-element one. The extra pass is a vector's worth of
   // work against a matrix's.
   BatchedLinAlg::AddMult(B, u, p, 1.0, 0.0);
   p -= bp;
   BatchedLinAlg::LUSolve(S, S_ipiv, p);
   p.Neg();

   //u += -A^-1 (B^T + Bnl) S^-1 (B A^-1 bu - bp)
   Vector t(na * NE);
   t.UseDevice(true);
   BatchedLinAlg::AddMult(B, p, t, 1.0, 0.0, BatchedLinAlg::Op::T);

   if (with_bnl)
   {
      // The guard GetBnlMatrix() applies per element, applied once -- neither
      // half of it depends on the element.
      if (!Bnl_empty && Bnl_data.Size() == Bf_offsets.Last())
      {
         // Bnl is stored TRANSPOSED against B -- (a_dofs, d_dofs) per
         // element, the shape GetBnlMatrix() hands out -- so this is an
         // untransposed product, and it accumulates with the sign MultInv()
         // gives it.
         DenseTensor Bnl;
         Bnl.NewMemoryAndSize(Bnl_data.GetMemory(), na, nd, NE, false);
         BatchedLinAlg::AddMult(Bnl, p, t, (bsym) ? (-1.) : (1.), 1.0);
      }
   }

   BatchedLinAlg::LUSolve(A, Af_ipiv, t);

   if (bsym) { u += t; }
   else { u -= t; }
}

/** @brief When ConstructGrad()'s A and D work is exactly `A = A_lin` and
    `D = D_lin`, do both as whole-array copies and return true.

    **On a problem with no nonlinear ELEMENT integrator that is all it is.**
    ConstructGrad() then runs `A = 0.; A += A_lin;` and `D = 0.; D += D_lin;`
    through DenseMatrix views per element -- 1.66 M host writes per gradient
    at order 2 on 64x64 -- to compute a copy. Hoisted here it is two Vector
    assignments, which are device operations when the storage is
    device-resident, and the element loop then skips the blocks entirely.

    Every condition below is a member flag rather than anything per element,
    which is what makes the whole-array form equivalent: if the branch is a
    copy for one element it is a copy for all of them. Asked once per
    MultNL(), because asking per element is how the face-constraint decision
    used to walk every face inside the element loop.

    See the body for why the copy is a HOST one. */
bool DarcyHybridization::CopyLinearGradBlocks() const
{
   // A BLOCK nonlinearity puts a real Jacobian in the blocks, and then it is
   // not a copy.
   if (m_nlfi) { return false; }

   // **A bilinear integrator on a NONLINEAR mass slot is handled here rather
   // than refused, and it has to be.** Refusing sent the gradient back to
   // evaluating the integrator per element while LinearResidualBatched() had
   // already frozen the residual at once-assembled blocks -- two different
   // operators. See EnsureResidualCache() for the measurement.
   //
   // Gated on CanBatchLinearResidual() and not on HDGIntegratorIsLinear()
   // alone, so that gradient-frozen happens exactly when residual-frozen
   // does: if the batched residual declines for any other reason the element
   // loop runs live, and this must then decline too.
   const bool nlf_slots = (m_nlfi_u != NULL) || (m_nlfi_p != NULL);
   if (nlf_slots && !CanBatchLinearResidual()) { return false; }
   // The specialised local operators keep their own block factored in place
   // and ConstructGrad() deliberately leaves it alone.
   if (lop_type == LocalOpType::PotNL || lop_type == LocalOpType::FluxNL)
   { return false; }
   // Nothing to copy FROM. `= 0.` alone is still a whole-array operation but
   // is not what this routine promises, so it is refused rather than
   // half-done.
   // With a bilinear m_nlfi_u standing in for A the linear store is empty,
   // which is the same alternative AddMultA() takes; D is additive either way.
   if (!m_nlfi_u && A_empty) { return false; }
   if (D_empty) { return false; }
   if (!m_nlfi_u && Af_lin_data.Size() != Af_data.Size()) { return false; }
   if (Df_lin_data.Size() != Df_data.Size()) { return false; }
   if (Af_data.Size() == 0 || Df_data.Size() == 0) { return false; }
   if (nlf_slots)
   {
      const int NE = fes.GetNE();
      if (!EnsureResidualCache(UniformBlockSize(Af_f_offsets, NE),
                               UniformBlockSize(Df_f_offsets, NE)))
      { return false; }
      if (m_nlfi_u && res_cache->Au_all.Size() != Af_data.Size())
      { return false; }
      if (m_nlfi_p && res_cache->Dp_all.Size() != Df_data.Size())
      { return false; }
   }

   // **A HOST copy, and it was a device one until Device("debug") segfaulted
   // on it.** A device assignment leaves these two arrays device-valid, and
   // every consumer downstream -- ConstructGrad()'s own DenseMatrix views,
   // LocalNLOperator, the per-element MultInv() -- reaches them through
   // `&Af_data[off]`, a raw host pointer that neither syncs nor invalidates.
   // So the copy has to end with the host owning them, which is this file's
   // standing rule arriving from the other direction: SyncLocalBlocksToHost()
   // exists to take exactly this ownership.
   //
   // The win is not the device then, and it does not need to be: what this
   // removes is 2*NE DenseMatrix constructions and a zero-then-add per
   // element -- 1.66 M host writes at order 2 on 64x64 -- in favour of two
   // memcpys. Making it a device copy needs the CONSUMERS batched first,
   // which is the open item, not this one.
   real_t *a_dst = Af_data.HostWrite();
   const real_t *a_src = m_nlfi_u ? res_cache->Au_all.HostRead()
                         : Af_lin_data.HostRead();
   std::memcpy(a_dst, a_src, Af_data.Size()*sizeof(real_t));
   real_t *d_dst = Df_data.HostWrite();
   const real_t *d_src = Df_lin_data.HostRead();
   std::memcpy(d_dst, d_src, Df_data.Size()*sizeof(real_t));
   // ADDITIVE for D, which mirrors AddMultDE(): the linear store holds the
   // face constraint's contribution and the mass slot's is a further term.
   if (m_nlfi_p)
   {
      const real_t *p_src = res_cache->Dp_all.HostRead();
      const int n = Df_data.Size();
      for (int i = 0; i < n; i++) { d_dst[i] += p_src[i]; }
   }
   return true;
}

/** @brief Drop what was assembled ONCE on the promise that it does not move.

    A BilinearFormIntegrator installed on a nonlinear mass form is assembled
    once and reused -- see EnsureResidualCache() -- which is right when its
    coefficients are fixed and wrong when they are a parameter the caller
    sweeps. The type system cannot tell those apart, so this is the caller's
    declaration that they moved. Cheap: the blocks are rebuilt on the next
    residual or gradient.

    It does NOT clear Af_data, Df_data or the assembled gradient, so it is not
    Reset(): a caller changing a coefficient on a form still has to
    re-Assemble() that form for the LINEAR blocks to follow, which is the
    contract those have always had. */
void DarcyHybridization::InvalidateCoefficientCache()
{
   res_cache.reset();
}

void DarcyHybridization::ConstructGrad(int el, const Array<int> &faces,
                                       TransWorkspace &ws,
                                       const BlockVector &x_l,
                                       const Vector &u_l, const Vector &p_l,
                                       bool skip_interior_faces,
                                       bool ad_done) const
{
   const FiniteElement *fe_u = fes.GetFE(el);
   const FiniteElement *fe_p = fes_p.GetFE(el);
   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   fes.GetMesh()->GetElementTransformation(el, &ws.elem);
   ElementTransformation *Tr = &ws.elem;

   DenseMatrix A(&Af_data[Af_offsets[el]], a_dofs_size, a_dofs_size);
   DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
   LUFactors LU_A(A.GetData(), &Af_ipiv[Af_f_offsets[el]]);

   if (m_nlfi)
   {
      // Block (0,1) is d(flux residual)/dp. Discarding it -- which is what
      // this did -- leaves the local Jacobian inconsistent with the local
      // residual whenever the flux law depends on the potential, and Newton
      // then converges only to first order in that dependence rather than
      // quadratically. Block (1,0) stays NULL: the divergence form is linear,
      // so B is already exact in Bf_data.
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array2D<DenseMatrix*> grad_arr(2,2);
      DenseMatrix grad_A, grad_D, grad_Aup;
      grad_arr(0,0) = &grad_A;
      grad_arr(1,0) = NULL;
      grad_arr(0,1) = &grad_Aup;
      grad_arr(1,1) = &grad_D;
      m_nlfi->AssembleElementGrad(fe_arr, *Tr, x_arr, grad_arr);
      if (grad_A.Height() != 0) { A = grad_A; }
      else { A = 0.; }
      if (grad_D.Height() != 0) { D = grad_D; }
      else { D = 0.; }

      if (grad_Aup.Height() != 0)
      {
         MFEM_VERIFY(grad_Aup.Height() == a_dofs_size &&
                     grad_Aup.Width() == d_dofs_size,
                     "The (0,1) element gradient block is "
                     << grad_Aup.Height() << "x" << grad_Aup.Width()
                     << ", expected " << a_dofs_size << "x" << d_dofs_size);
         if (Bnl_data.Size() != Bf_offsets.Last())
         {
            Bnl_data.SetSize(Bf_offsets.Last());
            Bnl_data = 0.;
         }
         DenseMatrix Bnl(&Bnl_data[Bf_offsets[el]], a_dofs_size, d_dofs_size);
         Bnl = grad_Aup;
         Bnl_empty = false;
      }
      else if (!Bnl_empty)
      {
         DenseMatrix Bnl(&Bnl_data[Bf_offsets[el]], a_dofs_size, d_dofs_size);
         Bnl = 0.;
      }
   }
   else if (!ad_done)
   {
      // if only linear data are present, A is already factored
      if (lop_type != LocalOpType::PotNL)
      {
         A = 0.;
      }
      // if only linear data are present, D is already factored
      if (lop_type != LocalOpType::FluxNL)
      {
         D = 0.;
      }
   }

   // **`!ad_done` belongs on the NONLINEAR branches too, and leaving it off
   // DOUBLE COUNTED.** CopyLinearGradBlocks() has already written A and D
   // when ad_done, so every write here is a second one. Before Tier 2 that
   // could not happen -- the copy refused outright whenever m_nlfi_u was set,
   // A_empty being true in that case -- so `if (m_nlfi_u)` needed no guard
   // and had none. Tier 2 made the copy handle a bilinear integrator on a
   // nonlinear slot, i.e. made ad_done and m_nlfi_u coexist, and the
   // unguarded `A += grad_A` then added the element flux mass on top of the
   // copy of itself.
   //
   // Measured on convdiff's own -nl shape (a plain VectorMassIntegrator on
   // GetFluxMassNonlinearForm()): the RESIDUAL was bit-identical to the
   // element loop -- 2.9217004681959e+00 to every digit -- while |S v| came
   // back 1.4826943370698e+01 against the loop's 2.5716073704950e+01. A
   // gradient that is not the derivative of the residual, again, and it cost
   // 20 of the 152 serial references: the 8 Newton and -npc cases DIVERGED
   // at a fixed ratio of 0.37 over 1000 iterations, and the 12 that merely
   // drifted were the ones whose default solver is LBFGS, which never asks
   // for a gradient. That split is what said "gradient", not "residual",
   // before anything was read.
   if (!ad_done && m_nlfi_u)
   {
      DenseMatrix grad_A;
      m_nlfi_u->AssembleElementGrad(*fe_u, *Tr, u_l, grad_A);
      A += grad_A;
   }
   else if (!ad_done && !A_empty && lop_type != LocalOpType::PotNL)
   {
      DenseMatrix A_lin(const_cast<real_t*>(&Af_lin_data[Af_offsets[el]]),
                        a_dofs_size, a_dofs_size);
      A += A_lin;
   }

   if (!ad_done && m_nlfi_p)
   {
      DenseMatrix grad_D;
      m_nlfi_p->AssembleElementGrad(*fe_p, *Tr, p_l, grad_D);
      D += grad_D;
   }
   // The linear D is an ADDITIONAL term, not an alternative, whenever a
   // nonlinear potential mass is present: Df_lin_data then holds whatever
   // was assembled linearly -- a linear face constraint, a linear M_p domain
   // mass, or both -- and nothing else supplies it. `!D_empty` is the exact
   // test for that, and is the whole condition.
   //
   // It used to read `(!m_nlfi_p || c_bfi_p) && !D_empty`, which was
   // REDUNDANT rather than wrong: Init() refused to allocate D under that
   // same condition, so !D_empty could not be true without it. Making
   // AssemblePotMassMatrix() allocate what it writes broke that coupling and
   // turned the c_bfi_p half into a live defect -- a linear M_p domain mass
   // alongside a nonlinear Mnl_p would have been assembled and then dropped
   // from both the residual and the gradient, silently, while D_empty said
   // it was there. Two other copies of this condition, in
   // LocalNLOperator's residual and gradient, were the same defect.
   if (!ad_done && !D_empty && lop_type != LocalOpType::FluxNL)
   {
      DenseMatrix D_lin(&Df_lin_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
      D += D_lin;
   }

   // D and H accumulate over every integrator that touches a face; E and G did
   // not, because they hold one block per face and side and are rewritten
   // rather than reset between passes. A face reached by MORE THAN ONE
   // constraint integrator therefore kept only the LAST one's E and G, while
   // the residual added all of them -- so the gradient stopped being the
   // derivative of the residual. A boundary face is exactly that case: the
   // interior integrators arrive already summed into c_nlfi_p / c_nlfi, while
   // the boundary ones are kept as a list and applied one at a time.
   //
   // Measured on navierstokes -p 1 -stokes, a LINEAR problem where a correct
   // Jacobian converges in one Newton step: 35 steps at a fixed residual ratio
   // of 0.517, which is a fixed-point iteration rather than Newton. What named
   // the mechanism was adding a THIRD boundary integrator that is identically
   // zero -- it cannot change the residual by a bit -- and watching the case
   // that took one step take 61.
   //
   // It was invisible until a boundary trace component was left free, because
   // an essential trace dof gets a unit row and an eliminated column and a
   // wrong boundary E and G never reach the reduced system: -bcfull converges
   // in one step on the mesh where -bcphys took 35.
   //
   // The flag is per face and spans BOTH loops below, since c_nlfi_p and
   // c_nlfi write the same block.
   Array<bool> eg_written(faces.Size());
   eg_written = false;

   if (c_nlfi_p)
   {
      //bp += E x
      for (int f = 0; f < faces.Size(); f++)
      {
         const Vector &x_f = x_l.GetBlock(f);

         FaceElementTransformations *FTr = GetFaceTransformation(faces[f], ws);

         if (FTr->Elem2No >= 0)
         {
            //interior -- left to AssembleNLFaceGradBatched() when asked
            if (skip_interior_faces) { continue; }
            AssembleHDGGrad(el, FTr, *c_nlfi_p, x_f, p_l, eg_written[f], ws);
         }
         else
         {
            //boundary
            const int bdr_attr = fes.GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

            for (size_t i = 0; i < boundary_constraint_pot_nonlin_integs.size(); i++)
            {
               if (boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               AssembleHDGGrad(el, FTr, *boundary_constraint_pot_nonlin_integs[i], x_f,
                               p_l, eg_written[f], ws);
            }
         }
      }
   }

   if (c_nlfi)
   {
      //bp += E x
      for (int f = 0; f < faces.Size(); f++)
      {
         const Vector &x_f = x_l.GetBlock(f);

         FaceElementTransformations *FTr = GetFaceTransformation(faces[f], ws);

         if (FTr->Elem2No >= 0)
         {
            //interior -- left to AssembleNLFaceGradBatched() when asked
            if (skip_interior_faces) { continue; }
            AssembleHDGGrad(el, FTr, *c_nlfi, x_f, u_l, p_l, eg_written[f]);
         }
         else
         {
            //boundary
            const int bdr_attr = fes.GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

            for (size_t i = 0; i < boundary_constraint_nonlin_integs.size(); i++)
            {
               if (boundary_constraint_nonlin_integs_marker[i]
                   && (*boundary_constraint_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               AssembleHDGGrad(el, FTr, *boundary_constraint_nonlin_integs[i], x_f,
                               u_l, p_l, eg_written[f]);
            }
         }
      }
   }

   // No factorisation here. It used to happen on the matrix-free path only,
   // duplicating ComputeH()'s and leaving out the Jacobian's (0,1) block, so
   // the two modes built different Schur complements and the matrix-free one
   // was wrong whenever the flux law depended on the potential. Both now go
   // through ComputeH(), which is called for either mode once this pass over
   // the elements is finished.
}

void DarcyHybridization::AssembleHDGGrad(
   int el, FaceElementTransformations *FTr, NonlinearFormIntegrator &nlfi,
   const Vector &x_f, const Vector &p_l, bool &eg_written,
   TransWorkspace &ws) const
{
   const int f = FTr->Face->ElementNo;
   const FiniteElement *fe_c = c_fes.GetFaceElement(f);
   const FiniteElement *fe_p = fes_p.GetFE(el);
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   const int c_dofs_size = x_f.Size();

   int type = NonlinearFormIntegrator::HDGFaceType::ELEM
              | NonlinearFormIntegrator::HDGFaceType::TRACE
              | NonlinearFormIntegrator::HDGFaceType::CONSTR
              | NonlinearFormIntegrator::HDGFaceType::FACE;

   if (FTr->Elem1No != el) { type |= 1; }

   // From @a ws and not four locals: see TransWorkspace::g_elmat. The three
   // destinations' lives do not overlap, so one block temporary serves them.
   DenseMatrix &elmat = ws.g_elmat, &blk = ws.g_blk;

   nlfi.AssembleHDGFaceGrad(type, *fe_c, *fe_p, *FTr, x_f, p_l, elmat);

   // assemble D element matrices
   DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
   blk.CopyMN(elmat, d_dofs_size, d_dofs_size, 0, 0);
   D += blk;

   // assemble E constraint -- clearing on the first writer of this face and
   // side, accumulating after it. See the note in ConstructGrad().
   const int E_off = (FTr->Elem1No == el)?(0):(c_dofs_size*d_dofs_size);
   DenseMatrix E_f(&E_data[E_offsets[f] + E_off], d_dofs_size, c_dofs_size);
   blk.CopyMN(elmat, d_dofs_size, c_dofs_size, 0, d_dofs_size);
   if (!eg_written) { E_f = blk; }
   else { E_f += blk; }

   // assemble G constraint
   const int G_off = E_off;
   DenseMatrix G_f(&G_data[G_offsets[f] + G_off], c_dofs_size, d_dofs_size);
   blk.CopyMN(elmat, c_dofs_size, d_dofs_size, d_dofs_size, 0);
   if (!eg_written) { G_f = blk; }
   else { G_f += blk; }
   eg_written = true;

   // assemble H matrix
   DenseMatrix H_f(&H_data[H_offsets[f]], c_dofs_size, c_dofs_size);
   blk.CopyMN(elmat, c_dofs_size, c_dofs_size, d_dofs_size, d_dofs_size);
   H_f += blk;
}

void DarcyHybridization::AssembleHDGGrad(
   int el, FaceElementTransformations *FTr, BlockNonlinearFormIntegrator &nlfi,
   const Vector &x_f, const Vector &u_l, const Vector &p_l,
   bool &eg_written) const
{
   const int f = FTr->Face->ElementNo;
   const FiniteElement *fe_c = c_fes.GetFaceElement(f);
   const FiniteElement *fe_u = fes.GetFE(el);
   const FiniteElement *fe_p = fes_p.GetFE(el);
   const Array<const FiniteElement*> el_arr({fe_u, fe_p});
   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   const int c_dofs_size = x_f.Size();
   const Array<const Vector*> x_arr({&u_l, &p_l});

   int type = NonlinearFormIntegrator::HDGFaceType::ELEM
              | NonlinearFormIntegrator::HDGFaceType::TRACE
              | NonlinearFormIntegrator::HDGFaceType::CONSTR
              | NonlinearFormIntegrator::HDGFaceType::FACE;

   if (FTr->Elem1No != el) { type |= 1; }

   Array2D<DenseMatrix*> elmats(3, 3);
   DenseMatrix elmat_A, elmat_D, elmat_E, elmat_G, elmat_H;
   elmats = NULL;
   elmats(0,0) = &elmat_A;
   elmats(1,1) = &elmat_D;
   elmats(1,2) = &elmat_E;
   elmats(2,1) = &elmat_G;
   elmats(2,2) = &elmat_H;

   nlfi.AssembleHDGFaceGrad(type, *fe_c, el_arr, *FTr, x_f, x_arr, elmats);

   // assemble A element matrices
   DenseMatrix A(&Af_data[Af_offsets[el]], a_dofs_size, a_dofs_size);
   if (elmat_A.Height() != 0) { A += elmat_A; }

   // assemble D element matrices
   DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
   if (elmat_D.Height() != 0) { D += elmat_D; }

   // assemble E constraint
   //
   // E and G are neither simply written nor simply accumulated, and both of
   // those were tried and are wrong. They hold one block per face and side and
   // are not reset between gradient evaluations, so accumulating
   // unconditionally made GetGradient depend on how many times it had been
   // called -- the second Newton step of a hybridized nonlinear system got a
   // doubled E and G and diverged. Overwriting, which is what replaced it,
   // kept only the LAST integrator's blocks on a face reached by several.
   // @a eg_written separates the two: clear on the first writer of this face
   // and side, accumulate after it. See ConstructGrad().
   const int E_off = (FTr->Elem1No == el)?(0):(c_dofs_size*d_dofs_size);
   DenseMatrix E_f(&E_data[E_offsets[f] + E_off], d_dofs_size, c_dofs_size);
   DenseMatrix elmat_EG;
   if (elmat_E.Height() != 0)
   {
      elmat_EG.CopyMN(elmat_E, d_dofs_size, c_dofs_size, 0, 0);
      if (!eg_written) { E_f = elmat_EG; }
      else { E_f += elmat_EG; }
   }

   // assemble G constraint
   const int G_off = E_off;
   DenseMatrix G_f(&G_data[G_offsets[f] + G_off], c_dofs_size, d_dofs_size);
   if (elmat_G.Height() != 0)
   {
      elmat_EG.CopyMN(elmat_G, c_dofs_size, d_dofs_size, 0, 0);
      if (!eg_written) { G_f = elmat_EG; }
      else { G_f += elmat_EG; }
   }

   if (elmat_E.Height() != 0 || elmat_G.Height() != 0) { eg_written = true; }

   // assemble H matrix
   DenseMatrix H_f(&H_data[H_offsets[f]], c_dofs_size, c_dofs_size);
   if (elmat_H.Height() != 0) { H_f += elmat_H; }
}

void DarcyHybridization::AddTraceRHS(Vector &b_tr, real_t a) const
{
   if (!trace_rhs) { return; }

   MFEM_VERIFY(trace_rhs->Size() == c_fes.GetVSize(),
               "the skeleton load is " << trace_rhs->Size() << " long and the "
               "constraint space has " << c_fes.GetVSize() << " L-dofs");

   const Operator *tr_P = TraceProlongation();
   if (tr_P)
   {
      // b_tr is in TRUE dofs and the load is assembled in L-dofs, so it
      // arrives the way any linear form does: through P^T.
      tr_P->AddMultTranspose(*trace_rhs, b_tr, a);
   }
   else
   {
      b_tr.Add(a, *trace_rhs);
   }
}

void DarcyHybridization::ReduceRHS(const BlockVector &b_t, Vector &b_tr) const
{
   const Operator *tr_cP = NULL;

   if (NPCEnabled())
   {
      //store RHS for Mult
      if (!darcy_offsets.Size())
      {
         darcy_offsets.SetSize(3);
         darcy_offsets[0] = 0;
         darcy_offsets[1] = fes.GetVSize();
         darcy_offsets[2] = fes_p.GetVSize();
         darcy_offsets.PartialSum();
      }
      if (!darcy_toffsets.Size())
      {
         darcy_toffsets.SetSize(3);
         darcy_toffsets[0] = 0;
         darcy_toffsets[1] = fes.GetTrueVSize();
         darcy_toffsets[2] = fes_p.GetTrueVSize();
         darcy_toffsets.PartialSum();

         darcy_rhs.Update(darcy_toffsets);
      }
      darcy_rhs = b_t;

      //initialize reduced rhs
      if (ParallelC())
      {
         const Operator *tr_P = c_fes.GetProlongationMatrix();
         if (b_tr.Size() != tr_P->Width())
         {
            b_tr.SetSize(tr_P->Width());
            b_tr = 0.;
         }
      }
      else if ((tr_cP = c_fes.GetConformingProlongation()))
      {
         if (b_tr.Size() != tr_cP->Width())
         {
            b_tr.SetSize(tr_cP->Width());
            b_tr = 0.;
         }
      }
      else
      {
         if (b_tr.Size() != c_fes.GetVSize())
         {
            b_tr.SetSize(c_fes.GetVSize());
            b_tr = 0.;
         }
      }
      return;
   }

   Vector bu;

   if (!ParallelU())
   {
      const Operator *cR = fes.GetConformingRestriction();
      if (cR)
      {
         bu.SetSize(cR->Width());
         cR->MultTranspose(b_t.GetBlock(0), bu);
      }
      else
      {
         bu.MakeRef(const_cast<Vector&>(b_t.GetBlock(0)), 0, fes.GetVSize());
      }
   }
   else
   {
      const Operator *R = fes.GetRestrictionOperator();
      bu.SetSize(R->Width());
      R->MultTranspose(b_t.GetBlock(0), bu);
   }

   const Vector &bp = b_t.GetBlock(1);

   Vector b_r;

   if (!ParallelC() && !(tr_cP = c_fes.GetConformingProlongation()))
   {
      if (b_tr.Size() != c_fes.GetVSize())
      {
         b_tr.SetSize(c_fes.GetVSize());
         b_tr = 0.;
      }
      b_r.MakeRef(b_tr, 0, b_tr.Size());
   }
   else
   {
      b_r.SetSize(c_fes.GetVSize());
      b_r = 0.;
   }

   const int NE = fes.GetNE();

   // The local solves for every element in one batch, when that is asked for.
   // The face work below then reads the answers out of these instead of
   // calling MultInv(), and is otherwise untouched -- so the two routes run
   // the same face loop in the same order over the same colouring.
   Vector u_all, p_all;
   const bool batched_solve = CanBatchLocalSolve();
   if (batched_solve)
   {
      const int na = Af_f_offsets.Last(), nd = Df_f_offsets.Last();
      Vector bu_all(na), bp_all(nd);
      // One kernel each, not a loop of per-element GetSubVector(real_t*)
      // calls: those begin with HostRead() and are host loops whatever the
      // Device is, so they pinned the blocked right-hand side to the host and
      // MultInvBatched() then had to push it back up. See el_u_dofs.
      BuildElementDofMaps();
      bu_all.UseDevice(true);
      bp_all.UseDevice(true);
      bu.GetSubVector(el_u_dofs, bu_all);
      bp.GetSubVector(el_p_dofs, bp_all);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is opposite!
         bp_all.Neg();
      }
      MultInvBatched(bu_all, bp_all, u_all, p_all);
      u_all.Neg();
      p_all.Neg();
      // THE copy back, and the only one this route needs. The face loop below
      // is host dense work reading u_l/p_l through GetData(), which does not
      // sync, so without this it would read stale host memory -- the whole
      // answer, silently, rather than an error. It is also the transfer a
      // full-device path has to remove, and naming it here is the point of
      // having it in one place.
      u_all.HostRead();
      p_all.HostRead();
   }


   // This loop scatters into the TRACE, so unlike the field loops it needs
   // the colouring -- and unlike them it is then safe whatever the flux space
   // is. Serial keeps the original element order exactly.
   const bool threaded = (asm_mode == AssemblyMode::Threaded);
   if (threaded) { BuildElementColouring(); }
   const int npasses = threaded ? colour_offsets.Size() - 1 : 1;

   for (int pass = 0; pass < npasses; pass++)
   {
      const int i0 = threaded ? colour_offsets[pass] : 0;
      const int i1 = threaded ? colour_offsets[pass+1] : NE;

#ifdef MFEM_USE_OPENMP
      #pragma omp parallel if (threaded)
#endif
      {
         Vector b_rl;
         Array<int> c_dofs;
         Array<int> faces;
         Vector bu_l, bp_l, u_l, p_l;
         Vector mi_wk;   ///< MultInv()'s one temporary, hoisted above the loop
         Array<int> u_vdofs, p_dofs;

#ifdef MFEM_USE_OPENMP
         #pragma omp for schedule(dynamic)
#endif
         for (int i = i0; i < i1; i++)
         {
            const int el = threaded ? colour_order[i] : i;

            //-A^-1 bu - A^-1 B^T S^-1 B A^-1 bu
            if (batched_solve)
            {
               u_l.MakeRef(u_all, Af_f_offsets[el],
                           Af_f_offsets[el+1] - Af_f_offsets[el]);
               p_l.MakeRef(p_all, Df_f_offsets[el],
                           Df_f_offsets[el+1] - Df_f_offsets[el]);
            }
            else
            {
               // Load RHS

               GetFDofs(el, u_vdofs);
               bu.GetSubVector(u_vdofs, bu_l);

               fes_p.GetElementVDofs(el, p_dofs);
               bp.GetSubVector(p_dofs, bp_l);
               if (bsym)
               {
                  //In the case of the symmetrized system, the sign is opposite!
                  bp_l.Neg();
               }

               MultInv(el, bu_l, bp_l, u_l, p_l, false, &mi_wk);
               u_l.Neg();
               p_l.Neg();
            }

            GetElementFaces(el, faces);

            // Mult C u + G p
            for (int f = 0; f < faces.Size(); f++)
            {
               int el1, el2;
               fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
               DenseMatrix Ct_l;
               GetCtFaceMatrix(faces[f], el1 != el, Ct_l);

               b_rl.SetSize(Ct_l.Width());
               Ct_l.MultTranspose(u_l, b_rl);

               if (c_bfi_p)
               {
                  DenseMatrix G;
                  GetGFaceMatrix(faces[f], el1 != el, G);

                  G.AddMult(p_l, b_rl);
               }

               c_fes.GetFaceVDofs(faces[f], c_dofs);
               b_r.AddElementVector(c_dofs, b_rl);
            }
         }
      }
   }


   if (!ParallelC())
   {
      if (tr_cP)
      {
         if (b_tr.Size() != tr_cP->Width())
         {
            b_tr.SetSize(tr_cP->Width());
            tr_cP->MultTranspose(b_r, b_tr);
         }
         else
         {
            tr_cP->AddMultTranspose(b_r, b_tr);
         }
      }
   }
   else
   {
      const Operator *tr_P = c_fes.GetProlongationMatrix();

      if (b_tr.Size() != tr_P->Width())
      {
         b_tr.SetSize(tr_P->Width());
         tr_P->MultTranspose(b_r, b_tr);
      }
      else
      {
         tr_P->AddMultTranspose(b_r, b_tr);
      }
   }

   // A load assembled on the SKELETON, if the caller registered one. It is a
   // right-hand side, so it ADDS here and SUBTRACTS from the NPC residual --
   // the two are the same convention read off r = A x - b. Getting this sign
   // wrong does not stop the solve converging: it converges to a different
   // answer, measured at 0.2% in the norm of the trace and 128.7 in the
   // vector, so a test that compares norms passes on it.
   AddTraceRHS(b_tr);
}

void DarcyHybridization::ProjectSolution(const BlockVector &sol,
                                         Vector &sol_r) const
{
   MFEM_VERIFY(c_fes.FEColl()->GetContType() !=
               FiniteElementCollection::CONTINUOUS,
               "Continuous trace collections are not supported in projection!");

   Mesh *mesh = c_fes.GetMesh();
   const int nfaces = mesh->GetNumFaces();

   const GridFunction p(&fes_p, const_cast<Vector&>(sol.GetBlock(1)), 0);

   DenseMatrix val_tr;
   Vector val1, val2;
   Array<int> c_vdofs;

   for (int f = 0; f < nfaces; f++)
   {
      FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(f);
      const FiniteElement *c_fe = c_fes.GetFaceElement(f);
      const IntegrationRule &nodes = c_fe->GetNodes();
      c_fes.GetFaceVDofs(f, c_vdofs);
      val_tr.SetSize(nodes.Size(), fes_p.GetVDim());
      MFEM_ASSERT(c_vdofs.Size() == nodes.Size() * fes_p.GetVDim(), "Internal error");

      for (int n = 0; n < nodes.Size(); n++)
      {
         const IntegrationPoint &ip = nodes[n];
         ftr->SetIntPoint(&ip);
         p.GetVectorValue(*ftr->Elem1, ftr->GetElement1IntPoint(), val1);
         if (ftr->Elem2No >= 0)
         {
            p.GetVectorValue(*ftr->Elem2, ftr->GetElement2IntPoint(), val2);
            val1 += val2;
            val1 *= 0.5;
         }
         val_tr.SetRow(n, val1);
      }

      sol_r.SetSubVector(c_vdofs, val_tr.GetData());
   }
}

// ----------------------------------------------------------------- NPC
// Nguyen, Peraire & Cockburn, JCP 228 (2009) 8841-8855, eqs (14)-(18). One
// Newton step on the full (q, u, lambda) system. See the doxygen on
// NPCResidual() for what these four do together and why they are not wrapped
// in an Operator.

void DarcyHybridization::NPCCheck() const
{
   MFEM_VERIFY(bfin, "DarcyHybridization must be finalized");
   // The refusal is measured, not precautionary, and what it is about is the
   // REPRESENTATION rather than the sign conventions an earlier version of
   // this guard blamed. NPC iterates on the broken state -- one flux copy per
   // element on a shared face, the trace row being what makes the copies agree
   // -- and a conforming H(div) space is exactly one dof per interior face too
   // small to hold it (192 element vdofs against a space of 144, RT order 1 on
   // a 4x4 quad mesh). Both elements then read the same value, their Ct blocks
   // carry opposite signs, and the trace row cancels identically: |C' q| came
   // out 3.2e-16 against a flux row of 9.8 at five random states, where L2
   // gives 7.9-9.0 and broken RT 7.1-8.0 on the same problem. With lambda
   // undriven NPC stalls at |F| = 0.0804, 12% off in the flux and 44% off in
   // the trace, on a problem the reduced route solves in 6 steps.
   //
   // BrokenRT_FECollection is the H(div)-shaped space that does have room, is
   // DISCONTINUOUS and so passes here already, and converges quadratically to
   // the same answer. See the note on NPCResidual().
   MFEM_VERIFY(fes.FEColl()->GetContType() ==
               FiniteElementCollection::DISCONTINUOUS,
               "NPC needs a discontinuous flux space. A conforming H(div) "
               "space cannot represent the broken state NPC iterates on, so "
               "the trace row cancels identically and lambda is never driven; "
               "use BrokenRT_FECollection for an H(div) element. See the note "
               "on NPCResidual().");
   // LocalOpType::FluxNL used to be refused here, because ComputeElementH()
   // discarded its Schur complement into a temporary and MultInv() then read
   // the factored linear potential mass in its place. ComputeElementH() now
   // writes it to Sf_data and MultInv() reads it back, so the mode is
   // supported and the guard is gone.
}

void DarcyHybridization::NPCResidual(const BlockVector &b, const BlockVector &x,
                                     const Vector &x_tr, BlockVector &r,
                                     Vector &r_tr)
{
   NPCCheck();

   const Operator *tr_P = TraceProlongation();

   // The element loops work in L-dofs; the interface is in true dofs.
   Vector x_tr_l;
   if (tr_P)
   {
      x_tr_l.SetSize(c_fes.GetVSize());
      tr_P->Mult(x_tr, x_tr_l);
   }
   else
   {
      x_tr_l.MakeRef(const_cast<Vector&>(x_tr), 0, x_tr.Size());
   }

   Vector r_tr_l;
   if (tr_P)
   {
      r_tr_l.SetSize(c_fes.GetVSize());
      if (r_tr.Size() != tr_P->Width()) { r_tr.SetSize(tr_P->Width()); }
   }
   else
   {
      if (r_tr.Size() != c_fes.GetVSize()) { r_tr.SetSize(c_fes.GetVSize()); }
      r_tr_l.MakeRef(r_tr, 0, r_tr.Size());
   }

   r = 0.;

   // The fields are state. MultNL reads them from here and does not touch
   // them. They need no mapping: NPC refuses anything but a discontinuous
   // flux space, so the local blocks are rank-local and their L-dofs are
   // their true dofs.
   darcy_u = x.GetBlock(0);
   darcy_p = x.GetBlock(1);

   MultNL(MultNlMode::AtFields, b, x_tr_l, r_tr_l, &r);

   // The trace row is the only one shared between ranks, so it is the only
   // one that has to be assembled.
   if (tr_P) { tr_P->MultTranspose(r_tr_l, r_tr); }

   // A load assembled on the SKELETON, if one is registered. It SUBTRACTS
   // here and ADDS in ReduceRHS(), r = A x - b being the same convention read
   // both ways. It goes in before the essential rows are cleared, so a load
   // sitting on an essential dof is discarded rather than fighting the datum.
   AddTraceRHS(r_tr, -1.0);

   // Essential trace dofs, carried exactly as Mult() carries them: the values
   // ride in x_tr, the residual is zero on those rows, and NPCGradient()
   // leaves a unit row to match.
   r_tr.SetSubVector(ess_tdof_list, 0.);
}

#ifdef MFEM_USE_MPI
Operator &DarcyHybridization::ParReducedGradient(MultNlMode mode,
                                                 const Vector &x_tr) const
{
   if (!Df_data.Size()) { AllocD(); }// D is resetted in ConstructGrad()
   if (!E_data.Size() || !G_data.Size()) { AllocEG(); }// E and G are rewritten
   if (!H_data.Size()) { AllocH(); }
   else if (c_nlfi_p || c_nlfi) { H_data = 0.; }

   Vector y;//dummy
   BlockVector zero_b;
   const BlockVector &b = (mode == MultNlMode::GradAtFields)
                          ? (ZeroLoad(zero_b, true), zero_b) : darcy_rhs;
   ParMultNL(mode, b, x_tr, y);

   if (grad_mode == GradientMode::Assembled)
   {
      Grad.reset();
      pGrad.SetType(pH.Type());
      ComputeParH(ComputeHMode::Gradient, Grad, pGrad);
      if (ess_tdof_list.Size() > 0)
      {
         // Rows and columns here, hypre offering no rows-with-unit-diagonal;
         // the extra column elimination is harmless for the same reason the
         // serial path can skip it.
         delete pGrad.As<HypreParMatrix>()->EliminateRowsCols(ess_tdof_list);
      }
      return *pGrad;
   }

   // Matrix-free: factor the local blocks, assemble nothing.
   Grad.reset();
   std::unique_ptr<SparseMatrix> H_unused;
   ComputeH(ComputeHMode::GradientFactorOnly, H_unused);
   pGrad.Reset(new ParGradient(*this));
   return *pGrad;
}
#endif //MFEM_USE_MPI

Operator &DarcyHybridization::NPCGradient(const BlockVector &x,
                                          const Vector &x_tr)
{
   NPCCheck();

   // The fields are state; the pass below assembles and factors the local
   // blocks at them exactly once, and solves nothing.
   darcy_u = x.GetBlock(0);
   darcy_p = x.GetBlock(1);

#ifdef MFEM_USE_MPI
   if (ParallelC())
   {
      return ParReducedGradient(MultNlMode::GradAtFields, x_tr);
   }
#endif
   return ReducedGradient(MultNlMode::GradAtFields, x_tr);
}

void DarcyHybridization::NPCReduce(const BlockVector &r, const Vector &r_tr,
                                   Vector &b_tr) const
{
   // b_tr = -( F_lambda - C' M^-1 F_local ), which is eq (18)'s right-hand
   // side. The trace row of the Jacobian is [C' G | H], so the potential
   // enters through G here and through E in NPCRecover() -- the two are
   // different blocks and swapping them is silent.
   const Operator *tr_P = TraceProlongation();

   Vector b_tr_l;
   if (tr_P)
   {
      b_tr_l.SetSize(c_fes.GetVSize());
      if (b_tr.Size() != tr_P->Width()) { b_tr.SetSize(tr_P->Width()); }
   }
   else
   {
      if (b_tr.Size() != c_fes.GetVSize()) { b_tr.SetSize(c_fes.GetVSize()); }
      b_tr_l.MakeRef(b_tr, 0, b_tr.Size());
   }
   b_tr_l = 0.;

   const int NE = fes.GetNE();
   Array<int> u_vdofs, p_dofs, faces, c_dofs;
   Vector ru_l, rp_l, du_l, dp_l, b_rl;
   Vector mi_wk;   ///< MultInv()'s one temporary, hoisted above the loop

   // Every element's M^-1 F_local in one batch, when that is asked for. This
   // is the loop that runs once per NPC Newton step, so it is where batching
   // is worth the most; the face loop below is untouched and reads the
   // answers out of the blocked vectors instead of calling MultInv().
   Vector du_all, dp_all;
   const bool batched_solve = CanBatchLocalSolve();
   if (batched_solve)
   {
      Vector ru_all(Af_f_offsets.Last()), rp_all(Df_f_offsets.Last());
      // One kernel each; see ReduceRHS() and el_u_dofs.
      BuildElementDofMaps();
      ru_all.UseDevice(true);
      rp_all.UseDevice(true);
      r.GetBlock(0).GetSubVector(el_u_dofs, ru_all);
      r.GetBlock(1).GetSubVector(el_p_dofs, rp_all);
      MultInvBatched(ru_all, rp_all, du_all, dp_all, true);
      // The face loop is host dense work through GetData(), which does not
      // sync; see ReduceRHS().
      du_all.HostRead();
      dp_all.HostRead();
   }

   for (int el = 0; el < NE; el++)
   {
      if (batched_solve)
      {
         du_l.MakeRef(du_all, Af_f_offsets[el],
                      Af_f_offsets[el+1] - Af_f_offsets[el]);
         dp_l.MakeRef(dp_all, Df_f_offsets[el],
                      Df_f_offsets[el+1] - Df_f_offsets[el]);
      }
      else
      {
         GetFDofs(el, u_vdofs);
         r.GetBlock(0).GetSubVector(u_vdofs, ru_l);
         fes_p.GetElementVDofs(el, p_dofs);
         r.GetBlock(1).GetSubVector(p_dofs, rp_l);

         // M^-1 F_local, with the JACOBIAN's (0,1) block. ReduceRHS() passes
         // the linear one, which is right for a linear system and would be a
         // different operator from the Schur complement here.
         MultInv(el, ru_l, rp_l, du_l, dp_l, true, &mi_wk);
      }

      GetElementFaces(el, faces);
      for (int f = 0; f < faces.Size(); f++)
      {
         int el1, el2;
         fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
         DenseMatrix Ct_l;
         GetCtFaceMatrix(faces[f], el1 != el, Ct_l);

         b_rl.SetSize(Ct_l.Width());
         Ct_l.MultTranspose(du_l, b_rl);

         if (G_data.Size() > 0)
         {
            DenseMatrix G_l;
            GetGFaceMatrix(faces[f], el1 != el, G_l);
            G_l.AddMult(dp_l, b_rl);
         }

         c_fes.GetFaceVDofs(faces[f], c_dofs);
         b_tr_l.AddElementVector(c_dofs, b_rl);
      }
   }

   // C' M^-1 F_local, assembled across ranks: the trace row is the only one
   // a face shares, and a face on the partition boundary is summed here.
   if (tr_P) { tr_P->MultTranspose(b_tr_l, b_tr); }

   // and r_tr is already in true dofs
   b_tr -= r_tr;
   b_tr.SetSubVector(ess_tdof_list, 0.);
}

void DarcyHybridization::NPCRecover(const BlockVector &r, const Vector &dtr,
                                    BlockVector &dx) const
{
   // dx_local = -M^-1 ( F_local + [C; E] dtr ). The flux row takes C^T dtr and
   // the potential row E dtr, which is the transpose pair of the blocks
   // NPCReduce() used.
   dx = 0.;

   const Operator *tr_P = TraceProlongation();
   Vector dtr_l;
   if (tr_P)
   {
      dtr_l.SetSize(c_fes.GetVSize());
      tr_P->Mult(dtr, dtr_l);
   }
   else
   {
      dtr_l.MakeRef(const_cast<Vector&>(dtr), 0, dtr.Size());
   }

   const int NE = fes.GetNE();
   Array<int> u_vdofs, p_dofs, faces, c_dofs;
   Vector ru_l, rp_l, du_l, dp_l, dtr_f;
   Vector mi_wk;   ///< MultInv()'s one temporary, hoisted above the loop

   // Two passes rather than one, as in ComputeSolution() and for the same
   // reason: here the face terms build the local right-hand side BEFORE the
   // solve, so every element's has to be in before any of them can be solved.
   const bool batched_solve = CanBatchLocalSolve();
   Vector ru_all, rp_all, du_all, dp_all;
   if (batched_solve)
   {
      ru_all.SetSize(Af_f_offsets.Last());
      rp_all.SetSize(Df_f_offsets.Last());
   }

   for (int el = 0; el < NE; el++)
   {
      GetFDofs(el, u_vdofs);
      r.GetBlock(0).GetSubVector(u_vdofs, ru_l);
      fes_p.GetElementVDofs(el, p_dofs);
      r.GetBlock(1).GetSubVector(p_dofs, rp_l);

      GetElementFaces(el, faces);
      for (int f = 0; f < faces.Size(); f++)
      {
         int el1, el2;
         fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
         c_fes.GetFaceVDofs(faces[f], c_dofs);
         dtr_l.GetSubVector(c_dofs, dtr_f);

         DenseMatrix Ct_l;
         GetCtFaceMatrix(faces[f], el1 != el, Ct_l);
         Ct_l.AddMult(dtr_f, ru_l);

         if (E_data.Size() > 0)
         {
            DenseMatrix E_l;
            GetEFaceMatrix(faces[f], el1 != el, E_l);
            E_l.AddMult(dtr_f, rp_l);
         }
      }

      if (batched_solve)
      {
         std::copy(ru_l.GetData(), ru_l.GetData() + ru_l.Size(),
                   ru_all.GetData() + Af_f_offsets[el]);
         std::copy(rp_l.GetData(), rp_l.GetData() + rp_l.Size(),
                   rp_all.GetData() + Df_f_offsets[el]);
         continue;
      }

      MultInv(el, ru_l, rp_l, du_l, dp_l, true, &mi_wk);
      du_l.Neg();
      dp_l.Neg();

      dx.GetBlock(0).SetSubVector(u_vdofs, du_l);
      dx.GetBlock(1).SetSubVector(p_dofs, dp_l);
   }

   if (batched_solve)
   {
      MultInvBatched(ru_all, rp_all, du_all, dp_all, true);
      du_all.Neg();
      dp_all.Neg();

      if (FieldDofsAreElementLocal())
      {
         // One kernel each and nothing comes back; see ComputeSolution().
         BuildElementDofMaps();
         dx.GetBlock(0).SetSubVector(el_u_dofs, du_all);
         dx.GetBlock(1).SetSubVector(el_p_dofs, dp_all);
      }
      else
      {
         du_all.HostRead();
         dp_all.HostRead();

         for (int el = 0; el < NE; el++)
         {
            GetFDofs(el, u_vdofs);
            dx.GetBlock(0).SetSubVector(u_vdofs,
                                        du_all.GetData() + Af_f_offsets[el]);
            fes_p.GetElementVDofs(el, p_dofs);
            dx.GetBlock(1).SetSubVector(p_dofs,
                                        dp_all.GetData() + Df_f_offsets[el]);
         }
      }
   }

   // The blocks were written, not the parent; see ComputeSolution().
   dx.SyncFromBlocks();
}

void DarcyHybridization::ComputeSolution(const BlockVector &b_t,
                                         const Vector &sol_tr, BlockVector &sol_t) const
{
   if (NPCEnabled())
   {
      ParMultNL(MultNlMode::Sol, b_t, sol_tr, sol_t);
      return;
   }

   Vector sol_r;
   if (!ParallelC())
   {
      const SparseMatrix *tr_cP = c_fes.GetConformingProlongation();
      if (!tr_cP)
      {
         sol_r.SetDataAndSize(sol_tr.GetData(), sol_tr.Size());
      }
      else
      {
         sol_r.SetSize(c_fes.GetVSize());
         tr_cP->Mult(sol_tr, sol_r);
      }
   }
   else
   {
      sol_r.SetSize(c_fes.GetVSize());
      c_fes.GetProlongationMatrix()->Mult(sol_tr, sol_r);
   }

   Vector bu, u;

   if (!ParallelU())
   {
      const Operator *cR = fes.GetConformingRestriction();
      if (!cR)
      {
         bu.MakeRef(const_cast<Vector&>(b_t.GetBlock(0)), 0, fes.GetVSize());
         u.MakeRef(sol_t.GetBlock(0), 0, fes.GetVSize());
      }
      else
      {
         bu.SetSize(fes.GetVSize());
         cR->MultTranspose(b_t.GetBlock(0), bu);
         u.SetSize(bu.Size());
         cR->MultTranspose(sol_t.GetBlock(0), u);
      }
   }
   else
   {
      bu.SetSize(fes.GetVSize());
      fes.GetRestrictionOperator()->MultTranspose(b_t.GetBlock(0), bu);
      u.SetSize(bu.Size());
      fes.GetRestrictionOperator()->MultTranspose(sol_t.GetBlock(0), u);
   }

   const Vector &bp = b_t.GetBlock(1);
   Vector &p = sol_t.GetBlock(1);

   const int NE = fes.GetNE();

   // Unlike ReduceRHS(), the face terms here modify the local right-hand side
   // BEFORE the solve, so the batched route is two passes over the elements
   // rather than one: build every element's (bu - C^T sol, bp - E sol) into
   // the blocked vectors, solve them all at once, then scatter. The face
   // arithmetic itself is the same in both routes.
   const bool batched_solve = CanBatchLocalSolve();
   Vector bu_all, bp_all, u_all, p_all;
   if (batched_solve)
   {
      bu_all.SetSize(Af_f_offsets.Last());
      bp_all.SetSize(Df_f_offsets.Last());
   }


   // Threaded only when both field spaces are discontinuous, so each
   // element's dofs are its own; see CanThreadFieldLoop(). This loop only
   // READS the trace, so it needs no colouring even then.
   const bool threaded = CanThreadFieldLoop();
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (threaded)
#endif
   {
      Vector sol_rl;
      Array<int> c_dofs;
      Array<int> faces;
      Vector bu_l, bp_l, u_l, p_l;
      Vector mi_wk;   ///< MultInv()'s one temporary, hoisted above the loop
      Array<int> u_vdofs, p_dofs;

#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int el = 0; el < NE; el++)
      {
         //Load RHS

         GetFDofs(el, u_vdofs);
         bu.GetSubVector(u_vdofs, bu_l);

         fes_p.GetElementVDofs(el, p_dofs);
         bp.GetSubVector(p_dofs, bp_l);
         if (bsym)
         {
            //In the case of the symmetrized system, the sign is opposite!
            bp_l.Neg();
         }

         GetElementFaces(el, faces);

         // bu - C^T sol
         for (int f = 0; f < faces.Size(); f++)
         {
            int el1, el2;
            fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
            DenseMatrix Ct_l;
            GetCtFaceMatrix(faces[f], el1 != el, Ct_l);

            c_fes.GetFaceVDofs(faces[f], c_dofs);
            sol_r.GetSubVector(c_dofs, sol_rl);

            Ct_l.AddMult_a(-1., sol_rl, bu_l);

            //bp - E sol
            if (c_bfi_p)
            {
               DenseMatrix E;
               GetEFaceMatrix(faces[f], el1 != el, E);

               E.AddMult_a(-1., sol_rl, bp_l);
            }
         }

         //(A^-1 - A^-1 B^T S^-1 B A^-1) (bu - C^T sol)
         if (batched_solve)
         {
            // Park this element's right-hand side and come back for the
            // answer once every element's is in.
            std::copy(bu_l.GetData(), bu_l.GetData() + bu_l.Size(),
                      bu_all.GetData() + Af_f_offsets[el]);
            std::copy(bp_l.GetData(), bp_l.GetData() + bp_l.Size(),
                      bp_all.GetData() + Df_f_offsets[el]);
            continue;
         }

         MultInv(el, bu_l, bp_l, u_l, p_l, false, &mi_wk);

         u.SetSubVector(u_vdofs, u_l);
         p.SetSubVector(p_dofs, p_l);
      }
   }


   if (batched_solve)
   {
      MultInvBatched(bu_all, bp_all, u_all, p_all);
      if (FieldDofsAreElementLocal())
      {
         // The scatter is one kernel and NOTHING comes back. This is the end
         // of the chain a full-device path has to leave device-resident, so
         // the HostRead() the other branch keeps is deliberately absent here;
         // the fields come back device-valid and the sync below is what makes
         // that legible through sol_t's blocks.
         BuildElementDofMaps();
         u.SetSubVector(el_u_dofs, u_all);
         p.SetSubVector(el_p_dofs, p_all);
      }
      else
      {
         // A shared flux dof is written by both its elements, and an
         // unordered forall would race where the serial loop's
         // last-writer-wins is element order. See FieldDofsAreElementLocal().
         u_all.HostRead();
         p_all.HostRead();

         Array<int> u_vdofs, p_dofs;
         for (int el = 0; el < NE; el++)
         {
            GetFDofs(el, u_vdofs);
            u.SetSubVector(u_vdofs, u_all.GetData() + Af_f_offsets[el]);

            fes_p.GetElementVDofs(el, p_dofs);
            p.SetSubVector(p_dofs, p_all.GetData() + Df_f_offsets[el]);
         }
      }
   }

   if (!ParallelU())
   {
      const Operator *cR = fes.GetConformingRestriction();
      if (cR)
      {
         cR->Mult(u, sol_t.GetBlock(0));
      }
      else
      {
         // u is a MakeRef of block 0, so a device write landed in ITS alias
         // buffer and the block does not know. One more level to go after it.
         u.SyncAliasMemory(sol_t.GetBlock(0));
      }
   }
   else
   {
      fes.GetRestrictionOperator()->Mult(u, sol_t.GetBlock(0));
   }

   // A BlockVector's blocks are aliases into its own storage, so a write made
   // through a block leaves the result in that alias's buffer while a second
   // view over the same range comes back marked host-valid whatever the
   // underlying state. That is step 0's caller contract in
   // doc/HDG-DEVICE-OFFLOAD.md, seen from the inside, and it is why the
   // batched route may leave the fields on the device at all: without this
   // the caller reads stale zeros, silently. Costs nothing when the write was
   // a host one -- SyncAlias only copies flags.
   sol_t.SyncFromBlocks();
}

void DarcyHybridization::ReconstructTotalFlux(
   const BlockVector &sol, const Vector &x, total_flux_fun ut_fx,
   GridFunction &ut) const
{
   // The scalar flux law adapted to the system one, so that the reconstruction
   // itself exists once. A caller who wrote against the single-field signature
   // is why this overload is here at all; giving it a system quietly, with
   // only the first field's potential reaching its callback, would be worse
   // than refusing.
   MFEM_VERIFY(ut_fx, "No total flux function was supplied");
   ReconstructTotalFlux(sol, x,
                        [&ut_fx](ElementTransformation &Tr, const Vector &u,
                                 const Vector &p, Vector &utq)
   {
      MFEM_VERIFY(p.Size() == 1, "A scalar total flux function cannot serve a "
                  "system of " << p.Size() << " fields; use the "
                  "total_flux_sys_fun overload.");
      ut_fx(Tr, u, p(0), utq);
   }, ut);
}

void DarcyHybridization::ReconstructTotalFlux(
   const BlockVector &sol, const Vector &x, total_flux_sys_fun ut_fx,
   GridFunction &ut) const
{
   const Vector &sol_u = sol.GetBlock(0);
   const Vector &sol_p = sol.GetBlock(1);

   const FiniteElementSpace &fes_ut = *ut.FESpace();

   MFEM_ASSERT(fes.GetMesh() == fes_ut.GetMesh(),
               "Different meshes are not supported!");

   MFEM_ASSERT(fes.GetMesh()->Conforming(),
               "Non-conforming meshes are not supported!");

   Mesh *mesh = fes_ut.GetMesh();
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = (c_pfes)?(c_pfes->GetParMesh()):(NULL);
   const int NE = mesh->GetNE();
   ParGridFunction pu, pp;
   if (pfes && pfes_p && pmesh)
   {
      pu.MakeRef(const_cast<ParFiniteElementSpace*>(pfes), const_cast<Vector&>(sol_u),
                 0);
      pu.ExchangeFaceNbrData();
      pp.MakeRef(const_cast<ParFiniteElementSpace*>(pfes_p),
                 const_cast<Vector&>(sol_p),
                 0);
      pp.ExchangeFaceNbrData();
   }
   else if (pfes || pfes_p || pmesh)
   {
      MFEM_ABORT("All, flux, potential and constraint parallel spaces are needed");
   }
#endif

   //element faces

   const int nfaces = mesh->GetNumFaces();
   Array<int> f_2_b = mesh->GetFaceToBdrElMap();
   Array<int> vdofs_ut, vdofs_xf, vdofs1, vdofs2, dofs1, dofs2;
   // Ct1 and Ct2 are made to *reference* the stored constraint blocks by
   // GetCtFaceMatrix(), so nothing may be assembled into them: a DenseMatrix
   // that already has the right shape keeps the pointer it was reset to, and
   // the write lands in Ct_data. Ct_own is where every path that writes a
   // constraint matrix puts it.
   DenseMatrix Ct_l, Ct1, Ct2, Ct_own, Mf;
   Vector u1, u2, p1, p2, xf, bf, bf1, bf2, ut_f;
   MassIntegrator fbfi;
   DenseMatrixInverse Mfi;

   // The number of fields. Every block below -- the constraint's rows, the
   // total flux's dofs, the potential's -- is this many copies of a scalar
   // one, equation outermost, which is the layout GetElementVDofs() and
   // GetFaceVDofs() produce locally whatever the space's Ordering is.
   const int neq = fes_ut.GetVDim();
   MFEM_VERIFY(neq == c_fes.GetVDim() && neq == fes_p.GetVDim(),
               "the total flux, the constraint and the potential must carry "
               "the same number of fields, got " << neq << ", "
               << c_fes.GetVDim() << " and " << fes_p.GetVDim());

   for (int f = 0; f < nfaces; f++)
   {
      fes_ut.GetFaceVDofs(f, vdofs_ut);
      MFEM_ASSERT(vdofs_ut.Size() == c_fes.GetFaceElement(f)->GetDof() *
                  c_fes.GetVDim(), "Incompatible constraint and total flux spaces");
      bf.SetSize(vdofs_ut.Size());
      ut_f.SetSize(vdofs_ut.Size());

      FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(f);
      const FiniteElement *fe_c = c_fes.GetFaceElement(f);

#ifdef MFEM_USE_MPI
      if (pmesh && pmesh->FaceIsTrueInterior(f) && ftr->Elem2No < 0)
      {
         // we do not store face neighbor constraint matrices so we must
         // integrate here over the face
         const FiniteElement *fe1 = fes.GetFE(ftr->Elem1No);
         const int nbr_el = -1 - ftr->Elem2No;
         const FiniteElement *fe2 = pfes->GetFaceNbrFE(nbr_el);
         ftr = pmesh->GetSharedFaceTransformationsByLocalIndex(f);
         c_bfi->AssembleFaceMatrix(*fe_c, *fe1, *fe2, *ftr, Ct_l);

         //side 1
         fes.GetElementVDofs(ftr->Elem1No, vdofs1);
         sol_u.GetSubVector(vdofs1, u1);
         Ct_own.SetSize(vdofs1.Size(), vdofs_ut.Size());
         Ct_own.CopyMN(Ct_l, vdofs1.Size(), vdofs_ut.Size(), 0, 0);
         Ct_own.MultTranspose(u1, bf);

         //side 2
         pfes->GetFaceNbrElementVDofs(nbr_el, vdofs2);
         pu.FaceNbrData().GetSubVector(vdofs2, u2);
         Ct_own.SetSize(vdofs2.Size(), vdofs_ut.Size());
         Ct_own.CopyMN(Ct_l, vdofs2.Size(), vdofs_ut.Size(), vdofs1.Size(), 0);
         // here we use the constraint integrator as well, but flip the sign
         // corresponding to the opposite normal for the total flux
         Ct_own.AddMultTranspose(u2, bf, -1.);
      }
      else
#endif
      {
         //flux constraint

         //side 1
         const DenseMatrix *Ct1_p;
         if (ftr->Elem2No >= 0)
         {
            GetCtFaceMatrix(f, 0, Ct1);
            Ct1_p = &Ct1;
         }
         else
         {
            // we do not rely on the boundary constraint integrators, which
            // might or might not be present, and apply the constraint
            // integrator at the boundaries as well
            const FiniteElement *fe1 = fes.GetFE(ftr->Elem1No);
            c_bfi->AssembleFaceMatrix(*fe_c, *fe1, *fe1, *ftr, Ct_own);
            Ct1_p = &Ct_own;
         }

         fes.GetElementVDofs(ftr->Elem1No, vdofs1);
         sol_u.GetSubVector(vdofs1, u1);
         Ct1_p->MultTranspose(u1, bf);

         //side 2
         if (ftr->Elem2No >= 0)
         {
            fes.GetElementVDofs(ftr->Elem2No, vdofs2);
            sol_u.GetSubVector(vdofs2, u2);
            GetCtFaceMatrix(f, 1, Ct2);
            // here we use the constraint integrator as well, but flip the sign
            // corresponding to the opposite normal for the total flux
            Ct2.AddMultTranspose(u2, bf, -1.);
         }
      }

      //potential constraint

      if ((c_bfi_p || c_nlfi_p) && ftr->Elem2No >= 0)
      {
         // first side
         fes_p.GetElementVDofs(ftr->Elem1No, dofs1);
         sol_p.GetSubVector(dofs1, p1);
         c_fes.GetFaceVDofs(f, vdofs_xf);
         x.GetSubVector(vdofs_xf, xf);

         const FiniteElement *fe1_p = fes_p.GetFE(ftr->Elem1No);
         const FiniteElement *face_fe = c_fes.GetFaceElement(f);

         int type = NonlinearFormIntegrator::HDGFaceType::CONSTR
                    | NonlinearFormIntegrator::HDGFaceType::FACE;

         if (c_bfi_p)
         {
            c_bfi_p->AssembleHDGFaceVector(type, *face_fe, *fe1_p, *ftr, xf, p1, bf1);
         }
         else
         {
            c_nlfi_p->AssembleHDGFaceVector(type, *face_fe, *fe1_p, *ftr, xf, p1, bf1);
         }
         bf += bf1;

         // second side
         const FiniteElement *fe2_p;
#ifdef MFEM_USE_MPI
         if (ftr->Elem2No >= NE)
         {
            const int nbr_el = ftr->Elem2No - NE;
            pfes_p->GetFaceNbrElementVDofs(nbr_el, dofs2);
            pp.FaceNbrData().GetSubVector(dofs2, p2);
            fe2_p = pfes_p->GetFaceNbrFE(nbr_el);
         }
         else
#endif
         {
            fes_p.GetElementVDofs(ftr->Elem2No, dofs2);
            sol_p.GetSubVector(dofs2, p2);
            fe2_p = fes_p.GetFE(ftr->Elem2No);
         }

         type |= 1;
         if (c_bfi_p)
         {
            c_bfi_p->AssembleHDGFaceVector(type, *face_fe, *fe2_p, *ftr, xf, p2, bf2);
         }
         else
         {
            c_nlfi_p->AssembleHDGFaceVector(type, *face_fe, *fe2_p, *ftr, xf, p2, bf2);
         }
         bf -= bf2;
      }

      // boundary potential constraint
      if (ftr->Elem2No < 0 && (!boundary_constraint_pot_integs.empty() ||
                               !boundary_constraint_pot_nonlin_integs.empty()))
      {
         constexpr int type = NonlinearFormIntegrator::HDGFaceType::CONSTR
                              | NonlinearFormIntegrator::HDGFaceType::FACE;

         const FiniteElement *fe_p = fes_p.GetFE(ftr->Elem1No);
         const FiniteElement *face_fe = c_fes.GetFaceElement(f);

         fes_p.GetElementVDofs(ftr->Elem1No, dofs1);
         sol_p.GetSubVector(dofs1, p1);
         c_fes.GetFaceVDofs(f, vdofs_xf);
         x.GetSubVector(vdofs_xf, xf);

         const int bdr_attr = mesh->GetBdrAttribute(f_2_b[f]);

         // linear
         for (size_t i = 0; i < boundary_constraint_pot_integs.size(); i++)
         {
            if (boundary_constraint_pot_integs_marker[i]
                && (*boundary_constraint_pot_integs_marker[i])[bdr_attr-1] == 0) { continue; }

            boundary_constraint_pot_integs[i]->AssembleHDGFaceVector(type, *face_fe, *fe_p,
                                                                     *ftr, xf, p1, bf1);

            bf += bf1;
         }

         // nonlinear
         for (size_t i = 0; i < boundary_constraint_pot_nonlin_integs.size(); i++)
         {
            if (boundary_constraint_pot_nonlin_integs_marker[i]
                && (*boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

            boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceVector(type, *face_fe,
                                                                            *fe_p, *ftr, xf, p1, bf1);

            bf += bf1;
         }
      }

      //face
      const FiniteElement *fe_utf = fes_ut.GetFaceElement(f);
      fbfi.AssembleElementMatrix2(*fe_utf, *fe_c, *ftr, Mf);

      // Mf is the *scalar* face mass -- MassIntegrator knows nothing of vdim
      // -- while bf and ut_f carry neq blocks. The mass is the same for every
      // field, so factor once and solve neq times against it. Before this the
      // solve was a single scalar one against an neq-times-too-long right-hand
      // side, which for one field is right and for more is a read past the
      // end of the factorisation.
      const int nd_utf = fe_utf->GetDof();
      const int nd_cf = fe_c->GetDof();
      Mfi.Factor(Mf);
      for (int e = 0; e < neq; e++)
      {
         const Vector bf_e(bf.GetData() + e * nd_cf, nd_cf);
         Vector ut_fe(ut_f.GetData() + e * nd_utf, nd_utf);
         Mfi.Mult(bf_e, ut_fe);
      }
      if (ftr->Elem2No >= 0)
      {
         // the face term should be double integrated to account for both sides
         // so divide the values by two after inversion
         ut_f *= .5;
      }

      ut.SetSubVector(vdofs_ut, ut_f);
   }

   if (fes_ut.FEColl()->GetOrder() <= 1) { return; }

   //element interior

   const int dim = mesh->Dimension();
   VectorFEMassIntegrator Mut;
   Array<int> vdofs, dofs, vdofs_ut_b, vdofs_ut_i;
   DenseMatrix Mut_z, Mut_zi;
   DenseMatrix vshape_u, vshape_ut;
   // The flux law is stated per equation throughout: the potential it is
   // handed has neq entries and the flux and total flux neq*dim, the block of
   // equation e occupying [e*dim, (e+1)*dim). One field is the case neq == 1,
   // not the only case.
   Vector shape_u, shape_ut, shape_p;
   Vector u_q(neq * dim), ut_q(neq * dim), p_q(neq);
   Vector u_z, p_z, b_z, b_zi, ut_zb, ut_zi;
   DenseMatrixInverse Muti_zi;

   for (int z = 0; z < fes.GetNE(); z++)
   {
      const FiniteElement *fe_ut = fes_ut.GetFE(z);
      const FiniteElement *fe_u = fes.GetFE(z);
      const FiniteElement *fe_p = fes_p.GetFE(z);

      ElementTransformation *Tr = mesh->GetElementTransformation(z);

      fes.GetElementVDofs(z, vdofs);
      sol_u.GetSubVector(vdofs, u_z);

      fes_p.GetElementVDofs(z, dofs);
      sol_p.GetSubVector(dofs, p_z);

      fes_ut.GetElementVDofs(z, vdofs_ut);
      const int nvdofs = vdofs_ut.Size();
      // Every shape below is the SCALAR one -- the element's, not the vdof
      // list's. A shape matrix sized to the vdof count is not merely wasteful:
      // CalcVShape() writes GetDof() rows, so the rest is uninitialised and
      // the contraction that follows mixes fields together.
      const int nd_u = fe_u->GetDof();
      const int nd_p = fe_p->GetDof();
      const int nd_ut = fe_ut->GetDof();
      MFEM_ASSERT(nvdofs == nd_ut * neq, "unexpected total flux vdof count");

      //integrate rhs

      const bool u_is_vector = (fe_u->GetRangeType() == FiniteElement::VECTOR);
      if (u_is_vector)
      {
         vshape_u.SetSize(nd_u, dim);
      }
      else
      {
         shape_u.SetSize(nd_u);
      }
      // One scalar shape per potential dof; the equations share it, which is
      // why this is GetDof() and not the vdof count that fills p_z.
      shape_p.SetSize(nd_p);
      vshape_ut.SetSize(nd_ut, dim);
      shape_ut.SetSize(nd_ut);

      b_z.SetSize(nvdofs);
      b_z = 0.;

      const int order = Tr->OrderW()
                        + std::max(fe_u->GetOrder(), fe_p->GetOrder())
                        + fe_ut->GetOrder();
      const IntegrationRule *ir = &IntRules.Get(fe_ut->GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Tr->SetIntPoint(&ip);

         if (u_is_vector)
         {
            // An H(div) flux carries the vector in the element, so vdim ==
            // neq and field e is one scalar component: its coefficients are
            // nd_u apart and share the one vector shape.
            fe_u->CalcVShape(*Tr, vshape_u);
            for (int e = 0; e < neq; e++)
            {
               const Vector u_ze(u_z.GetData() + e * nd_u, nd_u);
               Vector u_qe(u_q.GetData() + e * dim, dim);
               vshape_u.MultTranspose(u_ze, u_qe);
            }
         }
         else
         {
            // A scalar-range flux carries it in vdim, so vdim == neq*dim and
            // component e*dim+d is field e's d-th. One reshape covers every
            // field at once, and at neq == 1 it is the reshape that was here.
            fe_u->CalcPhysShape(*Tr, shape_u);
            DenseMatrix u_zm(u_z.GetData(), nd_u, neq * dim);
            u_zm.MultTranspose(shape_u, u_q);
         }

         fe_p->CalcShape(ip, shape_p);
         // p_z holds the element's potential vdofs, equation-major under
         // byNODES, so this reshape gives one value per equation.
         const DenseMatrix p_zm(p_z.GetData(), nd_p, neq);
         p_zm.MultTranspose(shape_p, p_q);

         ut_fx(*Tr, u_q, p_q, ut_q);

         fe_ut->CalcVShape(*Tr, vshape_ut);

         const real_t w = ip.weight * Tr->Weight();
         for (int e = 0; e < neq; e++)
         {
            const Vector ut_qe(ut_q.GetData() + e * dim, dim);
            vshape_ut.Mult(ut_qe, shape_ut);
            Vector b_ze(b_z.GetData() + e * nd_ut, nd_ut);
            b_ze.Add(w, shape_ut);
         }
      }

      //assemble mass matrix

      Mut.AssembleElementMatrix(*fe_ut, *Tr, Mut_z);

      //eliminate boundary rows

      // GetNumElementInteriorDofs() is a SCALAR count, so the interior dofs
      // are the tail of each FIELD's block of the vdof list, not the tail of
      // the list. Taking them as `nvdofs - nidofs` and slicing from the front
      // is right when there is one field and wrong for every other -- it puts
      // most of field 0 in the "boundary" set and reads field neq-1's block as
      // the interior. Both the elimination and the solve are therefore per
      // field, against the one scalar mass, which is the same for all of them.
      const int nidofs = fes_ut.GetNumElementInteriorDofs(z);
      const int nbdofs = nd_ut - nidofs;

      Mut_zi.CopyMN(Mut_z, nidofs, nidofs, nbdofs, nbdofs);
      Muti_zi.Factor(Mut_zi);

      for (int e = 0; e < neq; e++)
      {
         vdofs_ut_b.MakeRef(vdofs_ut.GetData() + e * nd_ut, nbdofs);
         ut.GetSubVector(vdofs_ut_b, ut_zb);

         Vector b_ze(b_z.GetData() + e * nd_ut, nd_ut);
         for (int j = 0; j < nbdofs; j++)
         {
            for (int i = 0; i < nidofs; i++)
            {
               b_ze(i+nbdofs) -= Mut_z(i+nbdofs,j) * ut_zb(j);
            }
         }

         //solve for the interior dofs

         ut_zi.SetSize(Mut_zi.Width());
         b_zi.MakeRef(b_ze, nbdofs, nidofs);
         Muti_zi.Mult(b_zi, ut_zi);

         vdofs_ut_i.MakeRef(vdofs_ut.GetData() + e * nd_ut + nbdofs, nidofs);
         ut.SetSubVector(vdofs_ut_i, ut_zi);
      }
   }
}

void DarcyHybridization::Reset()
{
   Hybridization::Reset();
   bfin = false;
   He.reset();
   pHe.Clear();
   Grad.reset();
   pGrad.Clear();

   // Connectivity only, so a Reset() that keeps the mesh and the trace space
   // keeps it valid -- but Reset() is also what a caller runs before
   // reconfiguring, and SetTraceOrders()-style changes to the trace space
   // would invalidate every offset in it. Dropped, and rebuilt lazily on the
   // next ComputeH(); it costs one host pass over the elements.
   trace_h_map.reset();
   // The face blocks it holds are re-assembled by whatever follows a Reset().
   res_cache.reset();

   A_empty = true;
   Af_data = 0.;
   Bf_data = 0.;
   if (Df_data.Size())
   {
      Df_data = 0.;
      D_empty = true;
   }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Ae_data = 0.;
   Be_data = 0.;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
}

void DarcyHybridization::Gradient::Mult(const Vector &x, Vector &y) const
{
   //note that rhs is not used, it is only a dummy
   dh.MultNL(MultNlMode::GradMult, dh.darcy_rhs, x, y);

   // The unit row the assembled mode gets from EliminateRow(DIAG_ONE). Without
   // it these rows come back as whatever the Schur complement makes of them
   // and the two modes are different operators -- and this one is singular
   // against a residual that is zero there.
   for (int i = 0; i < dh.ess_tdof_list.Size(); i++)
   {
      const int dof = dh.ess_tdof_list[i];
      y(dof) = x(dof);
   }

   // And the unit row the assembled mode gets from SetDiagIdentity(), which
   // regularises rows nothing contributed to. There is no matrix here for it
   // to act on, so it is applied by hand; see mf_diag_marker.
   if (dh.diag_policy == DIAG_ONE)
   {
      for (int i = 0; i < dh.mf_diag_marker.Size(); i++)
      {
         if (dh.mf_diag_marker[i]) { y(i) = x(i); }
      }
   }
}

#ifdef MFEM_USE_MPI
void DarcyHybridization::ParOperator::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(dh.bfin, "DarcyHybridization must be finalized");

   if (dh.pH.Ptr())
   {
      dh.pH->Mult(x, y);
      return;
   }

   dh.ParMultNL(MultNlMode::Mult, dh.darcy_rhs, x, y);

   y.SetSubVector(dh.ess_tdof_list, 0.);   // see the serial Mult()
}
Operator &DarcyHybridization::ParOperator::GetGradient(const Vector &x) const
{
   MFEM_VERIFY(dh.bfin, "DarcyHybridization must be finalized");

   if (dh.pH.Ptr()) { return *dh.pH.Ptr(); }

   if (!dh.Df_data.Size()) { dh.AllocD(); }// D is resetted in ConstructGrad()
   if (!dh.E_data.Size() || !dh.G_data.Size()) { dh.AllocEG(); }// E and G are rewritten
   if (!dh.H_data.Size()) { dh.AllocH(); }
   else if (dh.c_nlfi_p || dh.c_nlfi)
   {
      // H is resetted here for additive double side integration
      dh.H_data = 0.;
   }

   Vector y;//dummy
   dh.ParMultNL(MultNlMode::Grad, dh.darcy_rhs, x, y);

   if (dh.grad_mode == GradientMode::Assembled)
   {
      //assemble gradient matrix
      dh.Grad.reset();
      pGrad.SetType(dh.pH.Type());
      dh.ComputeParH(ComputeHMode::Gradient, dh.Grad, pGrad);
      if (dh.ess_tdof_list.Size() > 0)
      {
         // Rows and columns here, hypre offering no rows-with-unit-diagonal;
         // the extra column elimination is harmless for the same reason the
         // serial path can skip it.
         delete pGrad.As<HypreParMatrix>()->EliminateRowsCols(dh.ess_tdof_list);
      }
      return *pGrad;
   }

   // Matrix-free: factor the local blocks, assemble nothing.
   dh.Grad.reset();
   std::unique_ptr<SparseMatrix> H_unused;
   dh.ComputeH(ComputeHMode::GradientFactorOnly, H_unused);
   pGrad.Reset(new ParGradient(dh));
   return *pGrad;
}

void DarcyHybridization::ParGradient::Mult(const Vector &x, Vector &y) const
{
   // The load is a dummy -- GradMult applies the Jacobian and MultNL() zeroes
   // its local right-hand sides rather than reading this one -- but it cannot
   // be `darcy_rhs`, and that is the whole of the difference from the serial
   // Gradient::Mult() above. ParMultNL() restricts b's blocks to L-dofs BEFORE
   // it dispatches, so it dereferences the load whether the mode uses it or
   // not, and `darcy_rhs` is sized only as a side effect of ReduceRHS(), which
   // FormLinearSystem() calls. A caller that drives NPC directly -- as
   // miniapps/hdg/pnavierstokes does -- never goes through that path, so this
   // read an unsized BlockVector and segfaulted on the first matrix-free
   // gradient. Serial escapes it because MultNL() takes bu and bp by reference
   // and never touches them in this mode.
   //
   // A zero load of the right shape is the same substitution ParReducedGradient
   // already makes for MultNlMode::GradAtFields, and it cannot change any
   // answer: nothing in GradMult reads it.
   BlockVector zero_b;
   dh.ZeroLoad(zero_b, true);
   dh.ParMultNL(MultNlMode::GradMult, zero_b, x, y);

   // The unit row; see the serial Gradient::Mult(). The assembled parallel
   // path eliminates the columns as well, which this does not -- harmless for
   // the same reason it is harmless serially, the correction being zero on
   // these dofs.
   for (int i = 0; i < dh.ess_tdof_list.Size(); i++)
   {
      const int dof = dh.ess_tdof_list[i];
      y(dof) = x(dof);
   }
}
#endif // MFEM_USE_MPI

DarcyHybridization::LocalNLOperator::LocalNLOperator(
   const DarcyHybridization &dh_, int el_, const BlockVector &trps_,
   const Array<int> &faces_, TransWorkspace &ws_,
   const Vector *elem_flux_row_, bool skip_interior_faces)
   : skip_int_faces(skip_interior_faces),
     dh(dh_), el(el_), trps(trps_), faces(faces_),
     a_dofs_size(dh.Af_f_offsets[el+1] - dh.Af_f_offsets[el]),
     d_dofs_size(dh.Df_f_offsets[el+1] - dh.Df_f_offsets[el]),
     B(const_cast<real_t*>(&dh.Bf_data[dh.Bf_offsets[el]]),
       d_dofs_size, a_dofs_size),
     Bt(B), ws(ws_), elem_flux_row(elem_flux_row_),
     offsets(ws_.lop_offsets),
     Au(ws_.lop_Au), Dp(ws_.lop_Dp), DpEx(ws_.lop_DpEx),
     grad_A(ws_.lop_grad_A), grad_D(ws_.lop_grad_D),
     grad_Aup(ws_.lop_grad_Aup), grad_Bt(ws_.lop_grad_Bt)
{
   width = height = a_dofs_size + d_dofs_size;

   // @a offsets is a reference into @a ws, so it is FILLED here rather than
   // built in the initialiser list. It used to be `offsets({0, a, a+d})`, an
   // Array<int> from an initialiser list, i.e. one new[] of three ints per
   // element per evaluation. SetSize() does not shrink, so this reallocates
   // only if a later element is wider.
   ws.lop_offsets.SetSize(3);
   ws.lop_offsets[0] = 0;
   ws.lop_offsets[1] = a_dofs_size;
   ws.lop_offsets[2] = a_dofs_size + d_dofs_size;

   fe_u = dh.fes.GetFE(el);
   fe_p = dh.fes_p.GetFE(el);

   const Mesh *mesh = dh.fes.GetMesh();

   // Every transformation below belongs to @a ws and is reused across the
   // element loop; see TransWorkspace::lop_elem for what that is worth and
   // what it replaced. The vectors are sized FIRST, so the pointers taken
   // afterwards cannot be invalidated by a later growth.
   Tr = &ws.lop_elem;
   if (faces.Size() <= 0)
   {
      mesh->GetElementTransformation(el, Tr);
   }

   // face transformations
   if ((int)ws.lop_faces.size() < faces.Size())
   {
      ws.lop_faces.resize(faces.Size());
      ws.lop_nbrs.resize(faces.Size());
   }
   for (int f = 0; f < faces.Size(); f++)
   {
      FaceElementTransformations *FTr = &ws.lop_faces[f];
      // Which side of the face this element is on, and whether the face has a
      // second element, WITHOUT building any geometry. This was a third call
      // to GetFaceElementTransformations() at mask 0 -- a mask that asks for
      // no transformation at all, so its only product was Elem1No and
      // Elem2No, which Mesh::GetFaceElements() reads out of the same
      // faces_info entry. MultNL()'s own face loops already ask that way two
      // loops further down.
      //
      // Measured on `convdiff -p 1 -o 2 -dg -hb -nl -npc -nls 3 -gm 0` at
      // 128x128: 196,608 probe calls per solve, and 0.0075 s of them against
      // an instrumentation floor of about 0.008 s for that many timer scopes
      // -- so its cost is under 2% of NPCResidual and not separable from the
      // measurement. It goes because a call whose purpose is invisible is
      // worth more removed than kept, not because it was hot.
      //
      // The remaining call is NOT removable and the doc entry that asked for
      // it to be is wrong: it is made twice per interior face per residual
      // evaluation, once from each element, and it must be, because
      // LocalNLOperator holds every face of ONE element live at once with
      // Elem1/Elem2 bound to that element's own transformation objects. One
      // shared FaceElementTransformations per face needs a face-major loop,
      // which is a different operator, not a saving.
      int el1_f, el2_f;
      mesh->GetFaceElements(faces[f], &el1_f, &el2_f);
      IsoparametricTransformation *Tr1, *Tr2;
      if (el2_f >= 0)
      {
         IsoparametricTransformation *Nbr = &ws.lop_nbrs[f];
         if (el1_f == el)
         {
            Tr1 = Tr;
            Tr2 = Nbr;
         }
         else
         {
            Tr1 = Nbr;
            Tr2 = Tr;
         }

         mesh->GetFaceElementTransformations(faces[f], *FTr, *Tr1, *Tr2);
      }
#ifdef MFEM_USE_MPI
      else if (dh.ParallelC() &&
               dh.c_pfes->GetParMesh()->FaceIsTrueInterior(faces[f]))
      {
         IsoparametricTransformation *Nbr = &ws.lop_nbrs[f];
         if (el1_f == el)
         {
            Tr1 = Tr;
            Tr2 = Nbr;
         }
         else
         {
            Tr1 = Nbr;
            Tr2 = Tr;
         }

         dh.c_pfes->GetParMesh()->GetSharedFaceTransformationsByLocalIndex(faces[f],
                                                                           *FTr, *Tr1, *Tr2);
      }
#endif
      else
      {
         Tr1 = Tr2 = Tr;

         mesh->GetFaceElementTransformations(faces[f], *FTr, *Tr1, *Tr2, 21);
      }
   }
}

void DarcyHybridization::LocalNLOperator::AddMultBlock(const Vector &u_l,
                                                       const Vector &p_l, Vector &bu, Vector &bp) const
{
   // Equivalent to the two `if`s below both failing, and it is what keeps the
   // argument arrays off the no-integrator path.
   if (!dh.m_nlfi && !dh.c_nlfi) { return; }

   // From @a ws and not three locals per branch: an Array<T> built from an
   // initialiser list is a new[] per construction, and this runs per element
   // per residual evaluation. Both branches want the same fe_arr and x_arr,
   // so they are filled once; only y_arr differs, and SetSize() does not
   // shrink, so the 2-then-3 sizing below costs no reallocation.
   Array<const FiniteElement*> &fe_arr = ws.lop_fe_arr;
   Array<const Vector*> &x_arr = ws.lop_x_arr;
   Array<Vector*> &y_arr = ws.lop_y_arr;
   fe_arr.SetSize(2); fe_arr[0] = fe_u;  fe_arr[1] = fe_p;
   x_arr.SetSize(2);  x_arr[0] = &u_l;   x_arr[1] = &p_l;

   if (dh.m_nlfi)
   {
      if (elem_flux_row)
      {
         // Already computed for every element at once; see
         // DarcyHybridization::CanBatchLocalResidual(). Nothing is added to
         // bp because the gate admits only a MixedConductionNLFIntegrator,
         // whose AssembleElementVector() sets its potential row to size zero
         // -- which the per-element branch below then skips. That is a
         // property of the admitted integrator, so it is the gate's business
         // rather than something to test for here.
         MFEM_ASSERT(elem_flux_row->Size() == bu.Size(), "Incompatible size");
         bu += *elem_flux_row;
      }
      else
      {
         //element contribution
         y_arr.SetSize(2); y_arr[0] = &Au; y_arr[1] = &Dp;

         // **Cleared before every call, and this is not defensive.** @a Au and
         // @a Dp live in @a ws now, so they carry the last element's size in;
         // the `Size() != 0` tests below read "the integrator wrote this row",
         // which was true only while these were fresh locals. SetSize(0)
         // keeps the allocation -- Array/Vector shrink the size, not the
         // buffer -- so the clear is free and the hoist still holds.
         Au.SetSize(0); Dp.SetSize(0);

         dh.m_nlfi->AssembleElementVector(fe_arr, *Tr, x_arr, y_arr);
         if (Au.Size() != 0) { bu += Au; }
         if (Dp.Size() != 0) { bp += Dp; }
      }
   }

   if (dh.c_nlfi)
   {
      //face contribution
      y_arr.SetSize(3);
      y_arr[0] = &Au; y_arr[1] = &Dp; y_arr[2] = NULL;

      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = &ws.lop_faces[f];

         int type = BlockNonlinearFormIntegrator::HDGFaceType::ELEM
                    | BlockNonlinearFormIntegrator::HDGFaceType::TRACE;

         const Vector &trp_f = trps.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior
            if (FTr->Elem1No != el) { type |= 1; }

            // Per FACE, for the reason given at the element call above.
            Au.SetSize(0); Dp.SetSize(0);

            dh.c_nlfi->AssembleHDGFaceVector(type, *dh.c_fes.GetFaceElement(faces[f]),
                                             fe_arr, *FTr, trp_f, x_arr, y_arr);

            if (Au.Size() != 0) { bu += Au; }
            if (Dp.Size() != 0) { bp += Dp; }
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes.GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (size_t i = 0; i < dh.boundary_constraint_nonlin_integs.size(); i++)
            {
               if (dh.boundary_constraint_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               // Per boundary INTEGRATOR, same reason again.
               Au.SetSize(0); Dp.SetSize(0);

               dh.boundary_constraint_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                              *dh.c_fes.GetFaceElement(faces[f]),
                                                                              fe_arr,
                                                                              *FTr,
                                                                              trp_f, x_arr, y_arr);

               if (Au.Size() != 0) { bu += Au; }
               if (Dp.Size() != 0) { bp += Dp; }
            }
         }
      }
   }
}

void DarcyHybridization::LocalNLOperator::AddMultA(const Vector &u_l,
                                                   Vector &bu) const
{
   //bu += A u_l
   if (dh.m_nlfi_u)
   {
      dh.m_nlfi_u->AssembleElementVector(*fe_u, *Tr, u_l, Au);
      bu += Au;
   }
   else if (!dh.A_empty)
   {
      const DenseMatrix A(const_cast<real_t*>(&dh.Af_lin_data[dh.Af_offsets[el]]),
                          a_dofs_size, a_dofs_size);
      A.AddMult(u_l, bu);
   }
}

void DarcyHybridization::LocalNLOperator::AddMultDE(const Vector &p_l,
                                                    Vector &bp) const
{
   //bp += D p_l
   if (dh.m_nlfi_p)
   {
      dh.m_nlfi_p->AssembleElementVector(*fe_p, *Tr, p_l, Dp);
      bp += Dp;
   }
   // The linear D is an ADDITIONAL term whenever a nonlinear potential mass
   // is present, and !D_empty is the exact test for "Df_lin_data holds
   // linear content nothing else supplies". See ConstructGrad(), which
   // carries the note on why the old c_bfi_p half of this condition was a
   // defect once AssemblePotMassMatrix() began allocating what it writes.
   if (!dh.D_empty)
   {
      const DenseMatrix D(&dh.Df_lin_data[dh.Df_offsets[el]],
                          d_dofs_size, d_dofs_size);
      D.AddMult(p_l, bp);
   }

   if (dh.c_nlfi_p)
   {
      //bp += E x
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = &ws.lop_faces[f];

         int type = NonlinearFormIntegrator::HDGFaceType::ELEM
                    | NonlinearFormIntegrator::HDGFaceType::TRACE;

         const Vector &trp_f = trps.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior -- left to AssembleNLFaceResidualBatched() when asked,
            //exactly as ConstructGrad() leaves it to the gradient's kernel
            if (skip_int_faces) { continue; }
            if (FTr->Elem1No != el) { type |= 1; }

            dh.c_nlfi_p->AssembleHDGFaceVector(type, *dh.c_fes.GetFaceElement(faces[f]),
                                               *fe_p, *FTr, trp_f, p_l, DpEx);

            bp += DpEx;
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes.GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (size_t i = 0; i < dh.boundary_constraint_pot_nonlin_integs.size(); i++)
            {
               if (dh.boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               dh.boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                                  *dh.c_fes.GetFaceElement(faces[f]),
                                                                                  *fe_p, *FTr, trp_f, p_l, DpEx);

               bp += DpEx;
            }
         }
      }
   }
}

/** @brief Add the BLOCK integrator's element and face Jacobians into
    @a grad_A, @a grad_D and @a grad_Aup.

    **This used to take `DenseMatrix &gA, &gD` and both were DEAD**: each
    branch declared locals of the same names that shadowed them, and the
    accumulation went into the members instead. It worked only because the
    one caller passed those very members. Removed rather than wired up --
    the same shadowing that hid five declarations in bilininteg_hdg.cpp, and
    a parameter nothing reads is worse than no parameter. */
void DarcyHybridization::LocalNLOperator::AddGradBlock(const Vector &u_l,
                                                       const Vector &p_l) const
{
   if (!dh.m_nlfi && !dh.c_nlfi) { return; }

   // From @a ws and not fresh locals per branch; see
   // TransWorkspace::lop_gA. The integrator writes each of these and it is
   // added in immediately, so one set serves both branches and AddGradA() /
   // AddGradDE() after them.
   DenseMatrix &gA = ws.lop_gA, &gD = ws.lop_gD, &gAup = ws.lop_gAup;
   Array<const FiniteElement*> &fe_arr = ws.lop_fe_arr;
   Array<const Vector*> &x_arr = ws.lop_x_arr;
   Array2D<DenseMatrix*> &grad_arr = ws.lop_grad_arr;
   fe_arr.SetSize(2); fe_arr[0] = fe_u; fe_arr[1] = fe_p;
   x_arr.SetSize(2);  x_arr[0] = &u_l;  x_arr[1] = &p_l;

   if (dh.m_nlfi)
   {
      //element contribution
      grad_arr.SetSize(2,2);
      grad_arr = NULL;
      grad_arr(0,0) = &gA;
      grad_arr(0,1) = &gAup;
      grad_arr(1,1) = &gD;

      // **Cleared before the call, for the same reason AddMultBlock() clears
      // Au and Dp**: these live in @a ws now, so the `Height() != 0` tests
      // below would otherwise read the LAST element's size and add a stale
      // block. SetSize(0,0) keeps the allocation, DenseMatrix::data being an
      // Array that shrinks its size and not its buffer, so the clear costs
      // nothing and the hoist stands.
      gA.SetSize(0,0); gD.SetSize(0,0); gAup.SetSize(0,0);

      dh.m_nlfi->AssembleElementGrad(fe_arr, *Tr, x_arr, grad_arr);
      if (gA.Height() != 0) { grad_A += gA; }
      if (gD.Height() != 0) { grad_D += gD; }
      // d(flux residual)/dp. The element-local Newton needs it for the same
      // reason the trace gradient does: without it the local Jacobian does
      // not match the local residual and the inner solve stalls.
      if (gAup.Height() != 0)
      {
         if (grad_Aup.Height() == 0)
         {
            grad_Aup.SetSize(a_dofs_size, d_dofs_size);
            grad_Aup = 0.;
         }
         grad_Aup += gAup;
      }
   }

   if (dh.c_nlfi)
   {
      //face contribution
      grad_arr.SetSize(3,3);
      grad_arr = NULL;
      grad_arr(0,0) = &gA;
      grad_arr(1,1) = &gD;

      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = &ws.lop_faces[f];

         int type = BlockNonlinearFormIntegrator::HDGFaceType::ELEM;

         const Vector &trp_f = trps.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior
            if (FTr->Elem1No != el) { type |= 1; }

            // Per FACE; see the element call above.
            gA.SetSize(0,0); gD.SetSize(0,0);

            dh.c_nlfi->AssembleHDGFaceGrad(type, *dh.c_fes.GetFaceElement(faces[f]),
                                           fe_arr, *FTr, trp_f, x_arr, grad_arr);

            if (gA.Height() != 0) { grad_A += gA; }
            if (gD.Height() != 0) { grad_D += gD; }
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes.GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (size_t i = 0; i < dh.boundary_constraint_nonlin_integs.size(); i++)
            {
               if (dh.boundary_constraint_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               // Per boundary INTEGRATOR; see the element call above.
               gA.SetSize(0,0); gD.SetSize(0,0);

               dh.boundary_constraint_nonlin_integs[i]->AssembleHDGFaceGrad(type,
                                                                            *dh.c_fes.GetFaceElement(faces[f]),
                                                                            fe_arr,
                                                                            *FTr,
                                                                            trp_f, x_arr, grad_arr);

               if (gA.Height() != 0) { grad_A += gA; }
               if (gD.Height() != 0) { grad_D += gD; }
            }
         }
      }
   }
}

void DarcyHybridization::LocalNLOperator::AddGradA(const Vector &u_l,
                                                   DenseMatrix &grad) const
{
   //grad += A
   if (dh.m_nlfi_u)
   {
      // From @a ws; a local here shadowed the member of the same name and
      // allocated per element. See TransWorkspace::lop_gA.
      DenseMatrix &gA = ws.lop_gA;
      dh.m_nlfi_u->AssembleElementGrad(*fe_u, *Tr, u_l, gA);
      grad += gA;
   }
   else if (!dh.A_empty)
   {
      DenseMatrix A(const_cast<real_t*>(&dh.Af_lin_data[dh.Af_offsets[el]]),
                    a_dofs_size, a_dofs_size);
      grad += A;
   }
}

void DarcyHybridization::LocalNLOperator::AddGradDE(const Vector &p_l,
                                                    DenseMatrix &grad) const
{
   //grad += D
   if (dh.m_nlfi_p)
   {
      // From @a ws; see AddGradA().
      DenseMatrix &gD = ws.lop_gD;
      dh.m_nlfi_p->AssembleElementGrad(*fe_p, *Tr, p_l, gD);
      grad += gD;
   }
   // The linear D is an ADDITIONAL term whenever a nonlinear potential mass
   // is present, and !D_empty is the exact test for "Df_lin_data holds
   // linear content nothing else supplies". See ConstructGrad(), which
   // carries the note on why the old c_bfi_p half of this condition was a
   // defect once AssemblePotMassMatrix() began allocating what it writes.
   if (!dh.D_empty)
   {
      DenseMatrix D(&dh.Df_lin_data[dh.Df_offsets[el]], d_dofs_size, d_dofs_size);
      grad += D;
   }

   if (dh.c_nlfi_p)
   {
      DenseMatrix grad_Df;

      //grad += D_f
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = &ws.lop_faces[f];

         int type = NonlinearFormIntegrator::HDGFaceType::ELEM;

         const Vector &trp_f = trps.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior
            if (FTr->Elem1No != el) { type |= 1; }

            dh.c_nlfi_p->AssembleHDGFaceGrad(type, *dh.c_fes.GetFaceElement(faces[f]),
                                             *fe_p, *FTr, trp_f, p_l, grad_Df);

            grad += grad_Df;
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes.GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (size_t i = 0; i < dh.boundary_constraint_pot_nonlin_integs.size(); i++)
            {
               if (dh.boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               dh.boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceGrad(type,
                                                                                *dh.c_fes.GetFaceElement(faces[f]),
                                                                                *fe_p, *FTr, trp_f, p_l, grad_Df);

               grad += grad_Df;
            }
         }
      }
   }
}

BlockOperator &DarcyHybridization::LocalNLOperator::Grad() const
{
   // Rebuilt only when the block structure changes; see the declaration for
   // why this is lazy and why reuse cannot read a stale block pointer.
   // RowOffsets()[1] is the flux block's size, so the two tests together
   // identify the structure exactly.
   if (!ws.lop_grad || ws.lop_grad->Height() != height
       || ws.lop_grad->RowOffsets()[1] != a_dofs_size)
   {
      ws.lop_grad.reset(new BlockOperator(offsets));
   }
   return *ws.lop_grad;
}

void DarcyHybridization::LocalNLOperator::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width() && y.Size() == Height(), "Incompatible size");

   // From @a ws and not two locals: a BlockVector over caller-owned data
   // still allocates its own array of block views, which was 1,536
   // malloc/free pairs per 256-element problem here. Update() re-points them
   // at this element's offsets. See TransWorkspace::lop_xv.
   BlockVector &x_l = ws.lop_xv, &b = ws.lop_bv;
   x_l.Update(const_cast<Vector&>(x), offsets);
   const Vector &u_l = x_l.GetBlock(0);
   const Vector &p_l = x_l.GetBlock(1);
   b.Update(y, offsets);
   Vector &bu = b.GetBlock(0);
   Vector &bp = b.GetBlock(1);

   //bu = B^T p
   B.MultTranspose(p_l, bu);
   if (dh.bsym) { bu.Neg(); }

   //bu += A u
   AddMultA(u_l, bu);

   //bp = B u
   B.Mult(u_l, bp);

   //bp += D p
   AddMultDE(p_l, bp);

   //bu += A u_l - B^T p_l
   //bp += B u_l + D p_l + E x_f
   AddMultBlock(u_l, p_l, bu, bp);
}

Operator &DarcyHybridization::LocalNLOperator::GetGradient(
   const Vector &x) const
{
   MFEM_ASSERT(x.Size() == Width(), "Incompatible size");

   BlockVector &x_l = ws.lop_xv;
   x_l.Update(const_cast<Vector&>(x), offsets);
   const Vector &u_l = x_l.GetBlock(0);
   const Vector &p_l = x_l.GetBlock(1);

   BlockOperator &grad = Grad();

   grad_A.SetSize(a_dofs_size);
   grad_D.SetSize(d_dofs_size);
   grad_A = 0.;
   grad_D = 0.;
   grad_Aup.SetSize(0, 0);

   //block
   AddGradBlock(u_l, p_l);

   //A
   AddGradA(u_l, grad_A);
   grad.SetDiagonalBlock(0, &grad_A);

   //B
   grad.SetBlock(1, 0, &const_cast<DenseMatrix&>(B));

   //B^T, plus d(flux residual)/dp when the flux law supplies one
   if (grad_Aup.Height() != 0)
   {
      grad_Bt.SetSize(a_dofs_size, d_dofs_size);
      grad_Bt.Transpose(B);
      if (dh.bsym) { grad_Bt.Neg(); }
      grad_Bt += grad_Aup;
      grad.SetBlock(0, 1, &grad_Bt);
   }
   else
   {
      grad.SetBlock(0, 1, &const_cast<TransposeOperator&>(Bt),
                    (dh.bsym)?(-1.):(+1.));
   }

   //D
   AddGradDE(p_l, grad_D);
   grad.SetDiagonalBlock(1, &grad_D);

   return grad;
}

DarcyHybridization::LocalFluxNLOperator::LocalFluxNLOperator(
   const DarcyHybridization &dh_, int el_, const Vector &bp_,
   const BlockVector &trps_, const Array<int> &faces_,
   TransWorkspace &ws)
   : LocalNLOperator(dh_, el_, trps_, faces_, ws), bp(bp_),
     LU_D(&dh.Df_data[dh.Df_offsets[el]], &dh.Df_ipiv[dh.Df_f_offsets[el]]),
     p_l(ws.lop_other)
{
   MFEM_ASSERT(bp.Size() == d_dofs_size, "Incompatible size");

   width = height = a_dofs_size;
}

void DarcyHybridization::LocalFluxNLOperator::SolveP(const Vector &u_l,
                                                     Vector &p_l) const
{
   p_l = bp;

   //bp - E x - B^T p
   B.AddMult(u_l, p_l, -1.);

   //p = D^-1 rp
   LU_D.Solve(d_dofs_size, 1, p_l.GetData());
}

void DarcyHybridization::LocalFluxNLOperator::Mult(const Vector &u_l,
                                                   Vector &bu) const
{
   MFEM_ASSERT(u_l.Size() == a_dofs_size &&
               bu.Size() == a_dofs_size, "Incompatible size");

   SolveP(u_l, p_l);

   //bu = B^T p
   B.MultTranspose(p_l, bu);
   if (dh.bsym) { bu.Neg(); }

   AddMultA(u_l, bu);
}

Operator &DarcyHybridization::LocalFluxNLOperator::GetGradient(
   const Vector &u_l) const
{
   MFEM_ASSERT(u_l.Size() == a_dofs_size, "Incompatible size");

   SolveP(u_l, p_l);

   //grad = B^T D^-1 B
   DenseMatrix DiB = B;

   LU_D.Solve(d_dofs_size, a_dofs_size, DiB.GetData());
   grad_A.SetSize(a_dofs_size);
   MultAtB(B, DiB, grad_A);
   if (!dh.bsym) { grad_A.Neg(); }

   //grad += A
   AddGradA(u_l, grad_A);

   return grad_A;
}

DarcyHybridization::LocalPotNLOperator::LocalPotNLOperator(
   const DarcyHybridization &dh_, int el_, const Vector &bu_,
   const BlockVector &trps_, const Array<int> &faces_,
   TransWorkspace &ws)
   : LocalNLOperator(dh_, el_, trps_, faces_, ws), bu(bu_),
     LU_A(&dh.Af_data[dh.Af_offsets[el]], &dh.Af_ipiv[dh.Af_f_offsets[el]]),
     u_l(ws.lop_other)
{
   MFEM_ASSERT(bu.Size() == a_dofs_size, "Incompatible size");

   width = height = d_dofs_size;
}

void DarcyHybridization::LocalPotNLOperator::SolveU(const Vector &p_l,
                                                    Vector &u_l) const
{
   u_l = bu;

   //bu - C^T x + B^T p
   B.AddMultTranspose(p_l, u_l, (dh.bsym)?(+1.):(-1.));

   //u = A^-1 ru
   LU_A.Solve(a_dofs_size, 1, u_l.GetData());
}

void DarcyHybridization::LocalPotNLOperator::Mult(const Vector &p_l,
                                                  Vector &bp) const
{
   MFEM_ASSERT(p_l.Size() == d_dofs_size &&
               bp.Size() == d_dofs_size, "Incompatible size");

   SolveU(p_l, u_l);

   //bp = B u
   B.Mult(u_l, bp);

   AddMultDE(p_l, bp);
}

Operator &DarcyHybridization::LocalPotNLOperator::GetGradient(
   const Vector &p_l) const
{
   MFEM_ASSERT(p_l.Size() == d_dofs_size, "Incompatible size");

   SolveU(p_l, u_l);

   //grad = B A^-1 B^T
   DenseMatrix BAi = B;

   LU_A.RightSolve(a_dofs_size, d_dofs_size, BAi.GetData());
   grad_D.SetSize(d_dofs_size);
   MultABt(BAi, B, grad_D);
   if (!dh.bsym) { grad_D.Neg(); }

   //grad += D
   AddGradDE(p_l, grad_D);

   return grad_D;
}

// ------------------------------------------------- NPC as an MFEM Operator

DarcyNPCOperator::DarcyNPCOperator(DarcyHybridization &dh_,
                                   const Array<int> &offsets_,
                                   const BlockVector &load_)
   : Operator(offsets_.Last()), dh(&dh_), load(load_), offsets(offsets_)
{
   MFEM_VERIFY(offsets.Size() == 4,
               "offsets must be {0, flux, potential, trace}, partial-summed");
   loc_offsets.SetSize(3);
   loc_offsets[0] = 0;
   loc_offsets[1] = offsets[1] - offsets[0];
   loc_offsets[2] = offsets[2] - offsets[1];
   loc_offsets.PartialSum();
}

void DarcyNPCOperator::Mult(const Vector &x, Vector &y) const
{
   const BlockVector xb(const_cast<Vector&>(x), offsets);
   BlockVector yb(y, offsets);

   x_loc.Update(loc_offsets);
   x_loc.GetBlock(0) = xb.GetBlock(0);
   x_loc.GetBlock(1) = xb.GetBlock(1);
   r_loc.Update(loc_offsets);

   dh->NPCResidual(load, x_loc, xb.GetBlock(2), r_loc, r_tr);

   yb.GetBlock(0) = r_loc.GetBlock(0);
   yb.GetBlock(1) = r_loc.GetBlock(1);
   yb.GetBlock(2) = r_tr;
}

Operator &DarcyNPCOperator::GetGradient(const Vector &x) const
{
   const BlockVector xb(const_cast<Vector&>(x), offsets);

   x_loc.Update(loc_offsets);
   x_loc.GetBlock(0) = xb.GetBlock(0);
   x_loc.GetBlock(1) = xb.GetBlock(1);

   Operator &S = dh->NPCGradient(x_loc, xb.GetBlock(2));
   jac.reset(new Jacobian(*dh, S, offsets, loc_offsets));
   return *jac;
}

DarcyNPCSolver::DarcyNPCSolver(Solver &trace_solver_)
   : Solver(0), trace_solver(trace_solver_) { }

void DarcyNPCSolver::SetOperator(const Operator &op)
{
   jac = dynamic_cast<const DarcyNPCOperator::Jacobian*>(&op);
   MFEM_VERIFY(jac, "DarcyNPCSolver needs the handle from "
               "DarcyNPCOperator::GetGradient()");
   height = width = jac->Height();

   // The trace system is a different matrix every Newton step, so the trace
   // solver is re-pointed at it every step.
   trace_solver.SetOperator(jac->S);
}

void DarcyNPCSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(jac, "SetOperator() first");

   const BlockVector bb(const_cast<Vector&>(b), jac->offsets);
   BlockVector xb(x, jac->offsets);

   r_loc.Update(jac->loc_offsets);
   r_loc.GetBlock(0) = bb.GetBlock(0);
   r_loc.GetBlock(1) = bb.GetBlock(1);
   r_tr = bb.GetBlock(2);

   // eq (18): reduce to the trace, solve there, recover the local increments.
   jac->dh.NPCReduce(r_loc, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.;
   trace_solver.Mult(b_tr, dtr);

   dx_loc.Update(jac->loc_offsets);
   jac->dh.NPCRecover(r_loc, dtr, dx_loc);

   // NPCRecover() and NPCReduce() give the increment to ADD; NewtonSolver and
   // KINSolver both apply x_new = x - correction, so the sign flips here. The
   // one place the two conventions meet.
   xb.GetBlock(0) = dx_loc.GetBlock(0);
   xb.GetBlock(1) = dx_loc.GetBlock(1);
   xb.GetBlock(2) = dtr;
   xb.Neg();
}

}
