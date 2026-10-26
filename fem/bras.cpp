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

// The sum (or the minimum, if min is true) of the local values v over the
// ranks of a parallel space; v itself for a serial space. Collective for a
// parallel space.
static real_t GlobalReduce(const FiniteElementSpace &fes, real_t v, bool min)
{
#ifdef MFEM_USE_MPI
   if (auto pfes = dynamic_cast<const ParFiniteElementSpace*>(&fes))
   {
      real_t g;
      MPI_Allreduce(&v, &g, 1, MFEM_MPI_REAL_T, min ? MPI_MIN : MPI_SUM,
                    pfes->GetComm());
      return g;
   }
#endif
   return v;
}

// Return a copy of the prolongation matrix P with every entry squared, and set
// row_nnz[i] to the number of nonzero entries in row i of P. Returns NULL if P
// is not a matrix, which is the case for the parallel prolongation of
// conforming spaces (ConformingProlongationOperator): its rows have at most
// one entry, equal to +1 or -1, so the squared matrix is |P|.
static Operator *SquaredProlongation(const Operator &P, Array<int> &row_nnz)
{
   row_nnz.SetSize(P.Height());
   row_nnz = 0;
   if (auto sP = dynamic_cast<const SparseMatrix*>(&P))
   {
      // Serial nonconforming spaces.
      auto *P2 = new SparseMatrix(*sP);
      const int *I = P2->HostReadI();
      real_t *a = P2->HostReadWriteData();
      for (int i = 0; i < P2->Height(); i++)
      {
         for (int k = I[i]; k < I[i+1]; k++)
         {
            if (a[k] != 0.0) { row_nnz[i]++; }
            a[k] *= a[k];
         }
      }
      return P2;
   }
#ifdef MFEM_USE_MPI
   if (auto hP = dynamic_cast<const HypreParMatrix*>(&P))
   {
      // Parallel nonconforming spaces, and conforming spaces with Nedelec DOFs
      // on shared triangular faces. The rows are the local DOFs; their entries
      // are split between the diagonal block (owned true DOFs) and the
      // off-diagonal block (true DOFs owned by other ranks). The data is
      // modified on the host, as in HypreParMatrix::ScaleRows().
      auto *P2 = new HypreParMatrix(*hP);
      P2->HostReadWrite();
      hypre_ParCSRMatrix *h = *P2;
      hypre_CSRMatrix *blocks[2] = { hypre_ParCSRMatrixDiag(h),
                                     hypre_ParCSRMatrixOffd(h)
                                   };
      for (hypre_CSRMatrix *csr : blocks)
      {
         const HYPRE_Int *I = hypre_CSRMatrixI(csr);
         real_t *a = hypre_CSRMatrixData(csr);
         if (!I) { continue; }
         for (int i = 0; i < hypre_CSRMatrixNumRows(csr); i++)
         {
            for (HYPRE_Int k = I[i]; k < I[i+1]; k++)
            {
               if (a[k] != 0.0) { row_nnz[i]++; }
               a[k] *= a[k];
            }
         }
      }
      P2->HypreRead();
      return P2;
   }
#endif
   return nullptr;
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
// M_br = blockdiag(M_K) acting on E-vectors. The element restriction R_E has
// one nonzero per row, equal to +1 or -1 (the sign accounts for the relative
// orientation of ND and RT DOFs). On conforming meshes P_L has the same form,
// so P is a signed Boolean matrix. On nonconforming meshes, P_L = E in the
// notation of the class documentation, and R_E = P_0.
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

   // Since R_E is a signed Boolean matrix and W is diagonal, the columns of
   // W^{1/2} R_E are orthogonal and D_W = R_E^T W R_E is diagonal. Its entry
   // for an L-DOF is the sum of the weights w_i over all element DOFs i that
   // the L-DOF is mapped to, i.e. diag(D_W) = |R_E|^T w. The absolute values
   // remove the orientation signs, which square to one in R_E^T W R_E. The
   // L2 element restriction is a permutation without signs and does not
   // implement AbsMultTranspose(), so its MultTranspose() is used.
   dl.SetSize(fes->GetVSize());
   if (fes->IsDGSpace()) { R->MultTranspose(w, dl); }
   else { R->AbsMultTranspose(w, dl); }
   SetupProlongation();

   // Work vectors for Mult(): T-vectors, L-vectors (only used when P is not
   // the identity), the input and output E-vectors, and the vectors of the
   // Chebyshev iteration (only used on nonconforming spaces).
   xt.SetSize(height);
   ut.SetSize(height);
   xl.SetSize(P ? P->Height() : height);
   zl.SetSize(xl.Size());
   xe.SetSize(ne*nd);
   ye.SetSize(ne*nd);
   if (nc)
   {
      cr.SetSize(height);
      cd.SetSize(height);
      cq.SetSize(height);
   }

   // Re-applies the (possibly updated) essential DOFs to the new diagonal.
   // This uses the work vectors for the Chebyshev setup.
   SetEssentialTrueDofs(ess_tdof_list);
}

// With D_W = diag(dl), A_W = P_L^T D_W P_L, so diag(A_W) = (P_L o P_L)^T dl,
// where P_L o P_L is P_L with squared entries. On conforming spaces this is
// |P_L|^T dl, and A_W is diagonal.
//
// On nonconforming spaces, split the rows of P_L into the J rows, with at most
// one nonzero, and the H rows, with more than one; the H rows form C. Then
// A_W = Lambda_u + C^T Lambda_h C, where Lambda_h = diag(dl) on the H rows and
// Lambda_u = E_J^T D_W E_J = (P_L o P_L)^T (jmask o dl) is diagonal, because
// each J row has a single entry. Every true DOF has a J row (the L-DOF of its
// owning rank, with a unit entry), so Lambda_u is positive.
void BRASPreconditioner::SetupProlongation()
{
   nc = false;
   diag_inv.SetSize(height);
   if (!P) { diag_inv = dl; }
   else
   {
      Array<int> row_nnz;
      std::unique_ptr<Operator> P2(SquaredProlongation(*P, row_nnz));
      if (!P2) { P->AbsMultTranspose(dl, diag_inv); }
      else
      {
         P2->MultTranspose(dl, diag_inv);
         // The prolongation has the same type on all ranks, so either all or
         // none of the ranks reach this reduction.
         const real_t nc_loc = (row_nnz.Size() > 0 && row_nnz.Max() > 1);
         nc = GlobalReduce(*fes, nc_loc, false) > 0.0;
      }
      if (nc)
      {
         const int nl = P->Height();
         jmask.SetSize(nl);
         hdl.SetSize(nl);
         const real_t *h_dl = dl.HostRead();
         real_t *h_j = jmask.HostWrite(), *h_h = hdl.HostWrite();
         for (int i = 0; i < nl; i++)
         {
            const bool j_row = row_nnz[i] <= 1;
            h_j[i] = j_row ? 1.0 : 0.0;
            h_h[i] = j_row ? 0.0 : h_dl[i];
         }
         Vector jdl(dl);
         jdl *= jmask;
         lu_inv.SetSize(height);
         P2->MultTranspose(jdl, lu_inv);
         MFEM_VERIFY(height == 0 || lu_inv.Min() > 0.0, "a true DOF has no "
                     "local DOF with a single prolongation entry");
         lu_inv.Reciprocal();
      }
   }
   diag_inv.Reciprocal();
}

// The Chebyshev polynomial in S A_W (see ApplyChebyshev()) needs an interval
// [a, b] containing the spectrum of S A_W on the free DOFs. Since
// A_W >= Lambda_u, the spectrum is bounded below by a = min(Lambda_u S). The
// upper bound b is estimated by power iteration on S A_W, using the
// generalized Rayleigh quotient (v^T A_W v)/(v^T S^{-1} v). The estimate is
// slightly below the largest eigenvalue, which gives a better polynomial than
// the (much larger) upper bounds of Gershgorin type. Since the trace of S A_W
// is the number of free DOFs, its eigenvalues average to one, so a <= 1 and
// the largest eigenvalue is at least 1.
void BRASPreconditioner::SetupChebyshev()
{
   // On the free DOFs, scale = diag_inv > 0; on the essential DOFs, scale = 0.
   const real_t *h_s = scale.HostRead();
   const real_t *h_lu = lu_inv.HostRead();
   real_t a = infinity();
   for (int i = 0; i < height; i++)
   {
      if (h_s[i] != 0.0) { a = std::min(a, h_s[i]/h_lu[i]); }
   }
   a = GlobalReduce(*fes, a, true);
   cheb_a = cheb_b = 1.0;
   // No free DOFs: the Chebyshev polynomial is never applied to a nonzero
   // vector.
   if (a == infinity()) { return; }

   // Power iteration, starting from the vector of ones on the free DOFs. The
   // number of iterations is enough for the purpose of the estimate.
   const int power_its = 20;
   Vector v(height), Av(height), Sinv(diag_inv);
   Sinv.Reciprocal();
   v = 1.0;
   v.SetSubVector(ess_tdof_list, 0.0);
   real_t b = 1.0;
   for (int it = 0; it < power_its; it++)
   {
      ApplyAW(v, Av);
      Vector vSinv(v);
      vSinv *= Sinv;
      b = GlobalReduce(*fes, v*Av, false)/GlobalReduce(*fes, v*vSinv, false);
      // v = S A_W v, normalized.
      v = Av;
      v *= scale;
      v /= std::sqrt(GlobalReduce(*fes, v*v, false));
   }
   cheb_a = a;
   // Keep the interval nondegenerate: if A_W is diagonal on the free DOFs
   // (e.g. when all the hanging DOFs depend only on essential DOFs), the
   // spectrum is {1}, a = b = 1, and any interval containing 1 is valid.
   cheb_b = std::max({b, 1.0, (1.0 + 1e-6)*a});
}

// Let P_0 be P restricted to the columns of the free (non-essential) true
// DOFs. Weighted BRAS for the subspace of functions that vanish on the
// essential DOFs is B_0 = A_0^{-1} P_0^T W M_br^{-1} W P_0 A_0^{-1}, where
// A_0 = P_0^T W P_0 is the free block of the diagonal A_W. Extending B_0 by
// zero, B_0 = S P^T W M_br^{-1} W P S with S = diag(scale), where scale is
// diag(A_W)^{-1} with the essential entries set to zero. Mult() applies B_0
// and then sets y = x on the essential DOFs.
//
// On nonconforming spaces A_0 is not diagonal, but the same holds for its
// approximations: restricting to the free columns replaces S by diag(scale),
// Lambda_u^{-1} by diag(lu_scale), and C by C restricted to the free columns,
// which is the action of C on vectors that vanish on the essential DOFs.
void BRASPreconditioner::SetEssentialTrueDofs(const Array<int> &list)
{
   if (&list != &ess_tdof_list) { ess_tdof_list = list; }
   MFEM_VERIFY(ess_tdof_list.Size() == 0 || (ess_tdof_list.Min() >= 0 &&
                                              ess_tdof_list.Max() < height),
               "essential true DOF out of range; if the space has changed, "
               "call Update(ess_tdof_list)");
   scale = diag_inv;
   scale.SetSubVector(ess_tdof_list, 0.0);
   if (nc)
   {
      lu_scale = lu_inv;
      lu_scale.SetSubVector(ess_tdof_list, 0.0);
      // The spectrum of S A_0 depends on the essential DOFs.
      if (nc_method == NCMethod::CHEBYSHEV) { SetupChebyshev(); }
   }
}

void BRASPreconditioner::SetNCMethod(NCMethod method, int degree)
{
   MFEM_VERIFY(degree >= 1, "the Chebyshev degree must be at least 1");
   nc_method = method;
   cheb_degree = degree;
   if (nc && nc_method == NCMethod::CHEBYSHEV) { SetupChebyshev(); }
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

void BRASPreconditioner::ApplyBroken(const Vector &x, Vector &y) const
{
   // xe = R_E x (E-vector), then ye = W M_br^{-1} W xe, one dense
   // matrix-vector product per element. The check skips ranks without
   // elements. Then y = R_E^T ye sums the element contributions of each DOF.
   R->Mult(x, xe);
   if (WMinvW.SizeK() > 0) { BatchedLinAlg::Mult(WMinvW, xe, ye); }
   R->MultTranspose(ye, y);
}

// Only called on nonconforming spaces, where P is not NULL. Uses zl.
void BRASPreconditioner::ApplyAW(const Vector &x, Vector &y) const
{
   P->Mult(x, zl);
   zl *= dl;
   P->MultTranspose(zl, y);
}

// The Chebyshev semi-iteration (Saad, Iterative Methods for Sparse Linear
// Systems, Algorithm 12.1) for A_0 x = b with the preconditioner S, starting
// from x = 0, where A_0 is A_W on the free DOFs and S = diag(scale). After k
// steps, x = q_k(A_0) b, where the polynomial q_k minimizes the maximum of
// |1 - t q_k(t)| over [a, b] among polynomials in S A_0 of degree k (times
// S). Uses cr (residual), cd (update) and cq (A_W cd). All the updates are
// multiplied by S, so x vanishes on the essential DOFs.
void BRASPreconditioner::ApplyChebyshev(const Vector &b, Vector &x) const
{
   const real_t theta = 0.5*(cheb_b + cheb_a);
   const real_t delta = 0.5*(cheb_b - cheb_a);
   const real_t sigma = theta/delta;
   real_t rho = 1.0/sigma;

   // r = b, d = S r / theta, x = 0.
   cr = b;
   cd = b;
   cd *= scale;
   cd /= theta;
   x = 0.0;
   for (int it = 0; ; it++)
   {
      x += cd;
      if (it == cheb_degree) { break; }
      // r -= A_W d. Since d vanishes on the essential DOFs, A_W d = A_0 d on
      // the free DOFs; the essential entries of r are multiplied by S = 0.
      ApplyAW(cd, cq);
      cr -= cq;
      // d = rho_new rho d + (2 rho_new/delta) S r.
      const real_t rho_new = 1.0/(2.0*sigma - rho);
      const real_t c1 = rho_new*rho, c2 = 2.0*rho_new/delta;
      const int n = height;
      const auto d_s = scale.Read();
      const auto d_r = cr.Read();
      auto d_d = cd.ReadWrite();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         d_d[i] = c1*d_d[i] + c2*d_s[i]*d_r[i];
      });
      rho = rho_new;
   }
}

// The left inverse method, B_X = L_X G_m L_X^T on the free DOFs, where
// G_m = R_E^T (W M_br^{-1} W) R_E acts on L-vectors (ApplyBroken()) and
// L_X approximates A_W^{-1} P_L^T. Splitting the columns of P_L^T (the rows of
// P_L) into J and H, A_W^{-1} P_L^T = [(I - X C) Lambda_u^{-1} E_J^T,
// X Lambda_h^{-1}] with X = A_W^{-1} C^T Lambda_h (from A_W^{-1} Lambda_u =
// I - X C). L_X replaces X by S C^T Lambda_h. For any X, L_X D_W P_L = I, so
// the operator R_X = L_X R_E^T W satisfies R_X P = I.
//
// In terms of the L-vector masks jmask (1 on J) and hdl (Lambda_h on H, 0 on
// J), u = L_X^T x and out = L_X g are computed as
//    c   = P_L S x,
//    z   = x - P_L^T (hdl o c),
//    u   = jmask o (P_L Lambda_u^{-1} z) + (1 - jmask) o c,
//    v   = Lambda_u^{-1} P_L^T (jmask o g),
//    out = v + S P_L^T ((1 - jmask) o g - hdl o (P_L v)),
// where o is the entrywise product: three products with P_L and three with
// P_L^T. On the free DOFs S = diag(scale) and Lambda_u^{-1} = diag(lu_scale);
// see SetEssentialTrueDofs().
void BRASPreconditioner::MultLeftInverse(const Vector &x, Vector &y) const
{
   const int nl = P->Height();
   const auto d_j = jmask.Read();
   const auto d_h = hdl.Read();

   // c = P_L S x, stored in xl.
   xt = x;
   xt *= scale;
   P->Mult(xt, xl);
   // z = x - P_L^T (hdl o c), stored in ut, then ut = Lambda_u^{-1} z.
   zl = xl;
   zl *= hdl;
   P->MultTranspose(zl, ut);
   subtract(x, ut, ut);
   ut *= lu_scale;
   // u = jmask o (P_L ut) + (1 - jmask) o c, stored in xl.
   P->Mult(ut, zl);
   {
      const auto d_z = zl.Read();
      auto d_u = xl.ReadWrite();
      mfem::forall(nl, [=] MFEM_HOST_DEVICE (int i)
      {
         d_u[i] += d_j[i]*(d_z[i] - d_u[i]);
      });
   }

   // g = G_m u, stored in xl.
   ApplyBroken(xl, xl);

   // v = Lambda_u^{-1} P_L^T (jmask o g), stored in ut.
   zl = xl;
   zl *= jmask;
   P->MultTranspose(zl, ut);
   ut *= lu_scale;
   // zl = (1 - jmask) o g - hdl o (P_L v), then xt = P_L^T zl.
   P->Mult(ut, zl);
   {
      const auto d_g = xl.Read();
      auto d_z = zl.ReadWrite();
      mfem::forall(nl, [=] MFEM_HOST_DEVICE (int i)
      {
         d_z[i] = (1.0 - d_j[i])*d_g[i] - d_h[i]*d_z[i];
      });
   }
   P->MultTranspose(zl, xt);
   // y = v + S xt.
   const int n = height;
   const auto d_s = scale.Read();
   const auto d_v = ut.Read();
   const auto d_xt = xt.Read();
   auto d_y = y.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      d_y[i] = d_v[i] + d_s[i]*d_xt[i];
   });
}

// The Chebyshev method, q_k(A_0) P_L^T G_m P_L q_k(A_0) on the free DOFs, with
// G_m as in MultLeftInverse().
void BRASPreconditioner::MultChebyshev(const Vector &x, Vector &y) const
{
   ApplyChebyshev(x, ut);
   P->Mult(ut, xl);
   ApplyBroken(xl, xl);
   P->MultTranspose(xl, xt);
   ApplyChebyshev(xt, y);
}

// On conforming spaces, and for NCMethod::DIAGONAL, apply
// y = S P_L^T R_E^T (W M_br^{-1} W) R_E P_L S x on the free DOFs, where
// S = diag(scale); see SetEssentialTrueDofs(). On conforming spaces this is
// B_W, and on nonconforming spaces it is Btilde_W. The other nonconforming
// methods are applied by MultLeftInverse() and MultChebyshev(). In all cases
// y = x on the essential DOFs. The only communication is in P_L and P_L^T.
void BRASPreconditioner::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == height && y.Size() == height, "invalid sizes");

   if (nc && nc_method == NCMethod::LEFT_INVERSE) { MultLeftInverse(x, y); }
   else if (nc && nc_method == NCMethod::CHEBYSHEV) { MultChebyshev(x, y); }
   else
   {
      // xt = S x (T-vector).
      xt = x;
      xt *= scale;
      // xl = P_L xt (L-vector), then xl = G_m xl and xt = P_L^T xl. Without a
      // prolongation the T- and L-vectors coincide, so xt is used in place of
      // xl.
      Vector &xl_ = P ? xl : xt;
      if (P) { P->Mult(xt, xl); }
      ApplyBroken(xl_, xl_);
      if (P) { P->MultTranspose(xl, xt); }

      // y = S xt.
      const int n = height;
      const auto d_s = scale.Read();
      const auto d_xt = xt.Read();
      auto d_y = y.Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         d_y[i] = d_s[i]*d_xt[i];
      });
   }

   // The preconditioner is the identity on the essential DOFs, where y is
   // currently zero.
   const int n_ess = ess_tdof_list.Size();
   const auto d_ess = ess_tdof_list.Read();
   const auto d_x = x.Read();
   auto d_y = y.ReadWrite();
   mfem::forall(n_ess, [=] MFEM_HOST_DEVICE (int i)
   {
      const int j = d_ess[i];
      d_y[j] = d_x[j];
   });
}

} // namespace mfem
