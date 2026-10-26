// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bras.hpp"
#include "bilinearform.hpp"
#ifdef MFEM_USE_MPI
#include "pfespace.hpp"
#endif
#include "../general/forall.hpp"
#include "../linalg/batched/batched.hpp"

namespace mfem
{

// Create the mass bilinear form for fes, with optional coefficients (at most
// one of Q and MQ is non-NULL). The integrator matches the space: vector FE
// spaces (ND, RT) use VectorFEMassIntegrator, scalar spaces use
// MassIntegrator, and scalar spaces with vdim > 1 use VectorMassIntegrator.
// The returned form is owned by the caller and is never assembled; only its
// domain integrators are used, in Setup().
static BilinearForm *MakeMassForm(const FiniteElementSpace &fes,
                                  Coefficient *Q, MatrixCoefficient *MQ)
{
   auto *a = new BilinearForm(const_cast<FiniteElementSpace*>(&fes));
   const int vdim = fes.GetVDim();
   if (fes.GetTypicalFE()->GetRangeType() == FiniteElement::VECTOR)
   {
      MFEM_VERIFY(vdim == 1, "vector-valued spaces with vdim > 1 are not "
                  "supported");
      a->AddDomainIntegrator(MQ ? new VectorFEMassIntegrator(*MQ) :
                             new VectorFEMassIntegrator(Q));
   }
   else if (vdim == 1)
   {
      MFEM_VERIFY(MQ == nullptr, "a MatrixCoefficient requires a vector-valued "
                  "space or a space with vdim > 1");
      a->AddDomainIntegrator(Q ? new MassIntegrator(*Q) : new MassIntegrator);
   }
   else
   {
      VectorMassIntegrator *integ;
      if (MQ)
      {
         MFEM_VERIFY(MQ->GetVDim() == vdim, "the MatrixCoefficient size ("
                     << MQ->GetVDim() << ") must equal the space vdim ("
                     << vdim << ")");
         integ = new VectorMassIntegrator(*MQ);
      }
      else
      {
         integ = Q ? new VectorMassIntegrator(*Q) : new VectorMassIntegrator;
         // Without a matrix coefficient, the integrator takes the number of
         // components from the space dimension unless it is set explicitly.
         integ->SetVDim(vdim);
      }
      a->AddDomainIntegrator(integ);
   }
   return a;
}

BRASPreconditioner::BRASPreconditioner(BilinearForm &a,
                                       const Array<int> &ess_tdof_list_,
                                       Weighting weighting_)
   : Solver(a.FESpace()->GetTrueVSize()), form(&a), weighting(weighting_),
     ess_tdof_list(ess_tdof_list_)
{
   Setup();
}

BRASPreconditioner::BRASPreconditioner(const FiniteElementSpace &fes_,
                                       const Array<int> &ess_tdof_list_,
                                       Weighting weighting_)
   : Solver(fes_.GetTrueVSize()),
     owned_form(MakeMassForm(fes_, nullptr, nullptr)),
     weighting(weighting_), ess_tdof_list(ess_tdof_list_)
{
   form = owned_form.get();
   Setup();
}

BRASPreconditioner::BRASPreconditioner(const FiniteElementSpace &fes_,
                                       Coefficient &Q,
                                       const Array<int> &ess_tdof_list_,
                                       Weighting weighting_)
   : Solver(fes_.GetTrueVSize()),
     owned_form(MakeMassForm(fes_, &Q, nullptr)),
     weighting(weighting_), ess_tdof_list(ess_tdof_list_)
{
   form = owned_form.get();
   Setup();
}

BRASPreconditioner::BRASPreconditioner(const FiniteElementSpace &fes_,
                                       MatrixCoefficient &Q,
                                       const Array<int> &ess_tdof_list_,
                                       Weighting weighting_)
   : Solver(fes_.GetTrueVSize()),
     owned_form(MakeMassForm(fes_, nullptr, &Q)),
     weighting(weighting_), ess_tdof_list(ess_tdof_list_)
{
   form = owned_form.get();
   Setup();
}

BRASPreconditioner::~BRASPreconditioner() = default;

// Notation. The true DOFs (T-vector) are mapped to the local DOFs (L-vector)
// by the prolongation P_L = fes->GetProlongationMatrix(), and the local DOFs
// are mapped to the element DOFs (E-vector) by the element restriction
// R_E = fes->GetElementRestriction(). The operator P of the class
// documentation is the composition R_E P_L, so that M = P^T M_br P with
// M_br = blockdiag(M_K) acting on E-vectors. On conforming meshes both R_E and
// P_L have at most one nonzero per row, equal to +1 or -1 (the sign accounts
// for the relative orientation of ND and RT DOFs), so P is a signed Boolean
// matrix.
//
// The E-vector stores the element blocks contiguously: element e owns entries
// [e*nd, (e+1)*nd), where nd is the number of DOFs per element including all
// vector components. For vdim > 1, the components of an element are stored
// one after the other (layout nd_scalar x vdim x ne), which is the same order
// as the rows of the element matrices computed by the vector integrators.
void BRASPreconditioner::Setup()
{
   fes = form->FESpace();
   Mesh &mesh = *fes->GetMesh();
   const int ne = fes->GetNE();

   // The dense element storage below assumes the same number of DOFs on all
   // elements, which rules out variable order and mixed meshes. NURBS spaces
   // do not have a broken element representation of this form.
   MFEM_VERIFY(!fes->IsVariableOrder(), "variable order spaces are not "
               "supported");
   MFEM_VERIFY(fes->GetNURBSext() == nullptr, "NURBS spaces are not "
               "supported");
   MFEM_VERIFY(mesh.GetNumGeometries(mesh.Dimension()) <= 1, "meshes with "
               "more than one element geometry are not supported");
   // On nonconforming meshes P_L interpolates the slave DOFs from the master
   // DOFs, so P is no longer Boolean and A_W is not diagonal.
   MFEM_VERIFY(fes->Conforming(), "nonconforming spaces are not supported");
#ifdef MFEM_USE_MPI
   // With shared triangular faces, the parallel prolongation of high order
   // Nedelec spaces applies a 2x2 transformation to the face DOFs (to account
   // for the face orientation on the neighboring rank), so P is not a signed
   // Boolean matrix and A_W is not diagonal.
   auto pfes = dynamic_cast<const ParFiniteElementSpace*>(fes);
   MFEM_VERIFY(!pfes || !pfes->SharedNDTriangleDofs(), "Nedelec spaces with "
               "DOFs on shared triangular faces are not supported");
#endif

   // M_br is assembled from the domain integrators only. Boundary and face
   // integrators would couple DOFs across elements, so M would no longer be
   // P^T M_br P with block diagonal M_br.
   Array<BilinearFormIntegrator*> &integs = *form->GetDBFI();
   Array<Array<int>*> &markers = *form->GetDBFI_Marker();
   MFEM_VERIFY(integs.Size() > 0, "the bilinear form has no domain "
               "integrators");
   MFEM_VERIFY(form->GetBBFI()->Size() == 0 && form->GetFBFI()->Size() == 0 &&
               form->GetBFBFI()->Size() == 0, "only domain integrators are "
               "supported");

   // The space may have changed since the last call (see Update()).
   height = width = fes->GetTrueVSize();
   // The prolongation is the identity for serial conforming spaces; storing
   // NULL in that case skips the copy through the L-vector in Mult().
   P = fes->GetProlongationMatrix();
   if (IsIdentityProlongation(P)) { P = nullptr; }
   // NATIVE ordering: the element DOFs are in the order of the element
   // matrices returned by AssembleElementMatrix(), so no permutation is
   // needed. Both R and P are owned by the space.
   R = fes->GetElementRestriction(ElementDofOrdering::NATIVE);

   const int nd = fes->GetTypicalFE()->GetDof() * fes->GetVDim();
   MFEM_VERIFY(R->Height() == ne*nd, "inconsistent element restriction");

   // Compute the element matrices M_K and replace each one by the block
   // W_K M_K^{-1} W_K, stored in WMinvW(:,:,e). The diagonal of each W_K is
   // also stored in the E-vector w, which is used below to form diag(A_W).
   WMinvW.SetSize(nd, nd, ne);
   Vector w(ne*nd);
   real_t *h_A = WMinvW.HostWrite();
   real_t *h_w = w.HostWrite();
   DenseMatrix elmat, elmat_k, inv(nd);
   Array<int> vdofs;
   DofTransformation doftrans;
   for (int e = 0; e < ne; e++)
   {
      // Sum the contributions of all domain integrators active on element e,
      // following BilinearForm::Assemble(): an integrator with an attribute
      // marker only contributes to the elements whose attribute is marked.
      const int attr = mesh.GetAttribute(e);
      elmat.SetSize(0);
      for (int k = 0; k < integs.Size(); k++)
      {
         if (markers[k] && (*markers[k])[attr-1] != 1) { continue; }
         integs[k]->AssembleElementMatrix(*fes->GetFE(e),
                                          *fes->GetElementTransformation(e),
                                          elmat_k);
         if (elmat.Size() == 0) { elmat = elmat_k; }
         else { elmat += elmat_k; }
      }
      MFEM_VERIFY(elmat.Height() == nd, "no integrator contributes to "
                  "element " << e << ", or its element matrix has the wrong "
                  "size");
      // For ND elements on tetrahedra, wedges and pyramids, the element basis
      // differs from the basis seen by the global DOFs by a transformation T
      // that depends on the face orientations, and the element restriction
      // does not apply T. BilinearForm::Assemble() adds T^T M_K T to the
      // global matrix, so the same transformation is applied here to obtain
      // the element block of M_br in the basis of the E-vector. For all other
      // spaces T is the identity and this call does nothing.
      fes->GetElementVDofs(e, vdofs, doftrans);
      doftrans.TransformDual(elmat);

      // W_K = diag(M_K) or W_K = I.
      real_t *w_e = h_w + e*nd;
      for (int i = 0; i < nd; i++)
      {
         w_e[i] = (weighting == Weighting::DIAGONAL) ? elmat(i,i) : 1.0;
      }

      // Invert M_K with a Cholesky factorization, which fails if M_K is not
      // SPD (e.g. a zero or negative coefficient on the element). The factors
      // overwrite elmat.
      CholeskyFactors chol(elmat.GetData());
      MFEM_VERIFY(chol.Factor(nd), "the matrix of element " << e
                  << " is not SPD");
      chol.GetInverseMatrix(nd, inv.GetData());

      // Store the column-major block (W_K M_K^{-1} W_K)_ij = w_i inv_ij w_j,
      // in the layout expected by BatchedLinAlg::Mult().
      real_t *A_e = h_A + e*nd*nd;
      for (int j = 0; j < nd; j++)
      {
         for (int i = 0; i < nd; i++)
         {
            A_e[i + j*nd] = w_e[i] * inv(i,j) * w_e[j];
         }
      }
   }

   // Since P is a signed Boolean matrix and W is diagonal, the columns of
   // W^{1/2} P are orthogonal and A_W = P^T W P is diagonal. Its entry for a
   // true DOF is the sum of the weights w_i over all element DOFs i that the
   // true DOF is mapped to, i.e. diag(A_W) = |P|^T w = |P_L|^T |R_E|^T w. The
   // absolute values remove the orientation signs, which square to one in
   // P^T W P. The L2 element restriction is a permutation without signs and
   // does not implement AbsMultTranspose(), so its MultTranspose() is used.
   Vector dl(fes->GetVSize());
   if (fes->IsDGSpace()) { R->MultTranspose(w, dl); }
   else { R->AbsMultTranspose(w, dl); }
   diag_inv.SetSize(height);
   if (P) { P->AbsMultTranspose(dl, diag_inv); }
   else { diag_inv = dl; }
   diag_inv.Reciprocal();

   // Re-applies the (possibly updated) essential DOFs to the new diagonal.
   SetEssentialTrueDofs(ess_tdof_list);

   // Work vectors for Mult(): T-vector, L-vector (only used when P is not
   // the identity) and the input and output E-vectors.
   xt.SetSize(height);
   xl.SetSize(P ? P->Height() : height);
   xe.SetSize(ne*nd);
   ye.SetSize(ne*nd);
}

// Let P_0 be P restricted to the columns of the free (non-essential) true
// DOFs. Weighted BRAS for the subspace of functions that vanish on the
// essential DOFs is B_0 = A_0^{-1} P_0^T W M_br^{-1} W P_0 A_0^{-1}, where
// A_0 = P_0^T W P_0 is the free block of the diagonal A_W. Extending B_0 by
// zero, B_0 = S P^T W M_br^{-1} W P S with S = diag(scale), where scale is
// diag(A_W)^{-1} with the essential entries set to zero. Mult() applies B_0
// and then sets y = x on the essential DOFs.
void BRASPreconditioner::SetEssentialTrueDofs(const Array<int> &list)
{
   if (&list != &ess_tdof_list) { ess_tdof_list = list; }
   MFEM_VERIFY(ess_tdof_list.Size() == 0 || (ess_tdof_list.Min() >= 0 &&
                                              ess_tdof_list.Max() < height),
               "essential true DOF out of range; if the space has changed, "
               "call Update(ess_tdof_list)");
   scale = diag_inv;
   scale.SetSubVector(ess_tdof_list, 0.0);
}

// Setup() reads the space from the form and recomputes everything, so it
// handles both coefficient changes and space updates.
void BRASPreconditioner::Update()
{
   Setup();
}

void BRASPreconditioner::Update(const Array<int> &ess_tdof_list_)
{
   ess_tdof_list = ess_tdof_list_;
   Setup();
}

// Apply y = S P_L^T R_E^T (W M_br^{-1} W) R_E P_L S x on the free DOFs and
// y = x on the essential DOFs, where S = diag(scale); see
// SetEssentialTrueDofs(). The only communication is in P_L and P_L^T.
void BRASPreconditioner::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == height && y.Size() == height, "invalid sizes");

   // xt = S x (T-vector).
   xt = x;
   xt *= scale;
   // xl = P_L xt (L-vector). Without a prolongation the T- and L-vectors
   // coincide, so xt is used in place of xl.
   Vector &xl_ = P ? xl : xt;
   if (P) { P->Mult(xt, xl); }
   // xe = R_E xl (E-vector), then ye = W M_br^{-1} W xe, one dense
   // matrix-vector product per element. The check skips ranks without
   // elements.
   R->Mult(xl_, xe);
   if (WMinvW.SizeK() > 0) { BatchedLinAlg::Mult(WMinvW, xe, ye); }
   // xt = P_L^T R_E^T ye: sum the element contributions of each DOF.
   R->MultTranspose(ye, xl_);
   if (P) { P->MultTranspose(xl, xt); }

   // y = S xt.
   const int n = height;
   const auto d_s = scale.Read();
   const auto d_xt = xt.Read();
   auto d_y = y.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { d_y[i] = d_s[i]*d_xt[i]; });

   // The preconditioner is the identity on the essential DOFs, where S is
   // zero and y is currently zero.
   const int n_ess = ess_tdof_list.Size();
   const auto d_ess = ess_tdof_list.Read();
   const auto d_x = x.Read();
   d_y = y.ReadWrite();
   mfem::forall(n_ess, [=] MFEM_HOST_DEVICE (int i)
   {
      const int j = d_ess[i];
      d_y[j] = d_x[j];
   });
}

} // namespace mfem
