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
                                       Weighting weighting_,
                                       ElementInverse element_inverse_)
   : Solver(a.FESpace()->GetTrueVSize()), form(&a), weighting(weighting_),
     element_inverse(element_inverse_), ess_tdof_list(ess_tdof_list_)
{
   Setup();
}

BRASPreconditioner::BRASPreconditioner(const FiniteElementSpace &fes_,
                                       const Array<int> &ess_tdof_list_,
                                       Weighting weighting_,
                                       ElementInverse element_inverse_)
   : Solver(fes_.GetTrueVSize()),
     owned_form(MakeMassForm(fes_, nullptr, nullptr)),
     weighting(weighting_), element_inverse(element_inverse_),
     ess_tdof_list(ess_tdof_list_)
{
   form = owned_form.get();
   Setup();
}

BRASPreconditioner::BRASPreconditioner(const FiniteElementSpace &fes_,
                                       Coefficient &Q,
                                       const Array<int> &ess_tdof_list_,
                                       Weighting weighting_,
                                       ElementInverse element_inverse_)
   : Solver(fes_.GetTrueVSize()),
     owned_form(MakeMassForm(fes_, &Q, nullptr)),
     weighting(weighting_), element_inverse(element_inverse_),
     ess_tdof_list(ess_tdof_list_)
{
   form = owned_form.get();
   Setup();
}

BRASPreconditioner::BRASPreconditioner(const FiniteElementSpace &fes_,
                                       MatrixCoefficient &Q,
                                       const Array<int> &ess_tdof_list_,
                                       Weighting weighting_,
                                       ElementInverse element_inverse_)
   : Solver(fes_.GetTrueVSize()),
     owned_form(MakeMassForm(fes_, nullptr, &Q)),
     weighting(weighting_), element_inverse(element_inverse_),
     ess_tdof_list(ess_tdof_list_)
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
   MFEM_VERIFY(form->GetDBFI()->Size() > 0, "the bilinear form has no domain "
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
   // For ElementInverse::DENSE, NATIVE ordering: the element DOFs are in the
   // order of the element matrices returned by AssembleElementMatrix(), so
   // no permutation is needed. For ElementInverse::TENSOR, LEXICOGRAPHIC
   // ordering, in which the element mass matrices are Kronecker products.
   // Both R and P are owned by the space.
   R = fes->GetElementRestriction(element_inverse == ElementInverse::DENSE ?
                                  ElementDofOrdering::NATIVE :
                                  ElementDofOrdering::LEXICOGRAPHIC);

   const int nd = fes->GetTypicalFE()->GetDof() * fes->GetVDim();
   MFEM_VERIFY(R->Height() == ne*nd, "inconsistent element restriction");

   // The vectors used in Mult() are kept on the device, if one is
   // configured, so that their vector operations run there as well. The
   // element data is computed on the host.
   for (const Vector *v :
        {
           &diag_inv, &scale, &dl, &jmask, &hdl, &lu_inv,
           &lu_scale, &xt, &ut, &xl, &zl, &xe, &ye, &cr, &cd,
           &cq
        })
   {
      v->UseDevice(true);
   }

   // The diagonal of W, as an E-vector, used below to form diag(A_W).
   Vector w(ne*nd);
   w.UseDevice(true);
   if (element_inverse == ElementInverse::DENSE) { SetupDense(w); }
   else { SetupTensor(w); }

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

// Compute the element matrices M_K and replace each one by the block
// W_K M_K^{-1} W_K, stored in WMinvW(:,:,e). The diagonal of each W_K is
// stored in the E-vector w.
void BRASPreconditioner::SetupDense(Vector &w)
{
   Mesh &mesh = *fes->GetMesh();
   const int ne = fes->GetNE();
   const int nd = fes->GetTypicalFE()->GetDof() * fes->GetVDim();
   Array<BilinearFormIntegrator*> &integs = *form->GetDBFI();
   Array<Array<int>*> &markers = *form->GetDBFI_Marker();
   tensor_1d.Destroy();
   tensor_scale.Destroy();

   WMinvW.SetSize(nd, nd, ne);
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
}

// The 1D mass matrix M_ij = int_0^1 u_i u_j dx of n polynomials of degree
// less than n, where eval(x, u) evaluates them at x.
template <typename Eval>
static DenseMatrix Mass1D(int n, Eval eval)
{
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, 2*n);
   DenseMatrix M(n);
   M = 0.0;
   Vector u(n);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      eval(ip.x, u);
      AddMult_a_VVt(ip.weight, u, M);
   }
   return M;
}

// The basis functions of a tensor product element, in lexicographic order,
// are products u_i(x_0) v_j(x_1) w_k(x_2) of 1D basis functions (times the
// unit vector e_c of the reference element for component c of ND and RT
// elements). The 1D bases are the closed basis (GetBasis1D(), q + 1
// functions for elements of order q) and, for ND and RT elements, the open
// basis (GetOpenBasis1D(), q functions). Component c of an ND element uses
// the open basis in direction c and the closed basis in the other
// directions; RT elements do the reverse.
//
// On an element K with constant Jacobian J and constant coefficient c_K, the
// mass matrix of each component is m_K M_0 x M_1 x M_2, with the 1D mass
// matrices M_d of the bases and the scale
//    m_K = c_K |det J|           (H1 and L2, MapType VALUE),
//    m_K = c_K / |det J|         (L2, MapType INTEGRAL),
//    m_K = c_K |det J| / |J e_c|^2   (ND, covariant Piola map),
//    m_K = c_K |J e_c|^2 / |det J|   (RT, contravariant Piola map).
// For ND and RT elements this requires orthogonal columns of J; otherwise
// the components are coupled. Then W_K M_K^{-1} W_K is, for each component,
// m_K F_0 x F_1 x F_2 with F_d = diag(M_d) M_d^{-1} diag(M_d) for
// Weighting::DIAGONAL, and m_K^{-1} F_0 x F_1 x F_2 with F_d = M_d^{-1} for
// Weighting::IDENTITY.
//
// The lexicographic element restriction applies the signs of the DOF map
// (TensorBasisElement::GetDofMap()), so the E-vector holds the coefficients
// of the tensor product basis functions above.
void BRASPreconditioner::SetupTensor(Vector &w)
{
   Mesh &mesh = *fes->GetMesh();
   const int dim = mesh.Dimension();
   const int ne = fes->GetNE();
   const int vdim = fes->GetVDim();
   const FiniteElement &fe = *fes->GetTypicalFE();
   const Geometry::Type geom = fe.GetGeomType();
   const int map_type = fe.GetMapType();
   const bool vector_fe = fe.GetRangeType() == FiniteElement::VECTOR;
   const char *prefix = "ElementInverse::TENSOR: ";
   WMinvW.Clear();

   MFEM_VERIFY(Geometry::IsTensorProduct(geom) &&
               mesh.SpaceDimension() == dim, prefix << "the mesh must consist "
               "of segments, quadrilaterals or hexahedra, and have the "
               "dimension of the space");
   auto tfe = dynamic_cast<const TensorBasisElement*>(&fe);
   auto vtfe = dynamic_cast<const VectorTensorFiniteElement*>(&fe);
   MFEM_VERIFY(tfe && (!vector_fe || (vtfe && dim >= 2)), prefix << "the "
               "space must have tensor product elements, and ND and RT spaces "
               "must be two- or three-dimensional");

   // The integrators must be mass integrators matching the space, with a
   // scalar coefficient or none.
   Array<BilinearFormIntegrator*> &integs = *form->GetDBFI();
   Array<Array<int>*> &markers = *form->GetDBFI_Marker();
   Array<Coefficient*> coeffs(integs.Size());
   for (int k = 0; k < integs.Size(); k++)
   {
      const Coefficient *Q = nullptr;
      bool valid = false;
      if (vector_fe)
      {
         auto integ = dynamic_cast<VectorFEMassIntegrator*>(integs[k]);
         valid = integ && !integ->GetVectorCoefficient() &&
                 !integ->GetMatrixCoefficient();
         if (valid) { Q = integ->GetCoefficient(); }
      }
      else if (vdim == 1)
      {
         auto integ = dynamic_cast<MassIntegrator*>(integs[k]);
         valid = integ != nullptr;
         if (valid) { Q = integ->GetCoefficient(); }
      }
      else
      {
         // Without a matrix or vector coefficient, VectorMassIntegrator takes
         // the number of components from the space dimension, unless set.
         auto integ = dynamic_cast<VectorMassIntegrator*>(integs[k]);
         valid = integ && !integ->GetVectorCoefficient() &&
                 !integ->GetMatrixCoefficient() &&
                 (integ->GetVDim() == vdim ||
                  (integ->GetVDim() == -1 && mesh.SpaceDimension() == vdim));
         if (valid) { Q = integ->GetCoefficient(); }
      }
      MFEM_VERIFY(valid, prefix << "the domain integrators must be mass "
                  "integrators matching the space, with a scalar coefficient "
                  "or none");
      MFEM_VERIFY(markers[k] == nullptr && integs[k]->GetIntRule() == nullptr,
                  prefix << "attribute markers and custom integration rules "
                  "are not supported");
      coeffs[k] = const_cast<Coefficient*>(Q);
   }

   // The 1D mass matrices. The closed basis of L2 elements and the open basis
   // of ND and RT elements can be of the integrated type, whose evaluation
   // depends on a scaling flag of the (shared) basis object; it is set here as
   // in the CalcShape() and CalcVShape() methods of the elements.
   const int q = fe.GetOrder();
   auto &cbasis = const_cast<Poly_1D::Basis&>(tfe->GetBasis1D());
   DenseMatrix M_c = Mass1D(q + 1, [&](real_t x, Vector &u)
   {
      if (cbasis.IsIntegratedType())
      {
         cbasis.ScaleIntegrated(map_type == FiniteElement::VALUE);
      }
      cbasis.Eval(x, u);
   });
   DenseMatrix M_o;
   if (vector_fe)
   {
      auto &obasis = const_cast<Poly_1D::Basis&>(vtfe->GetOpenBasis1D());
      Vector uc(q + 1), dc(q + 1);
      M_o = Mass1D(q, [&](real_t x, Vector &u)
      {
         if (obasis.IsIntegratedType())
         {
            cbasis.Eval(x, uc, dc);
            obasis.ScaleIntegrated(false);
            obasis.EvalIntegrated(dc, u);
         }
         else { obasis.Eval(x, u); }
      });
   }

   // tensor_1d stores the 1 x 1 factor [1] (used in the directions beyond the
   // mesh dimension), then the factors F_d of the closed basis (b = 0) and of
   // the open basis (b = 1). The diagonals of the 1D mass matrices are kept
   // in diag1d.
   const DenseMatrix *M1d[2] = { &M_c, &M_o };
   Vector diag1d[2];
   int offset1d[2];
   tensor_1d.SetSize(1 + M_c.Height()*M_c.Height() +
                     M_o.Height()*M_o.Height());
   real_t *h_F = tensor_1d.HostWrite();
   h_F[0] = 1.0;
   for (int b = 0, o = 1; b < 2; b++)
   {
      const int n = M1d[b]->Height();
      offset1d[b] = o;
      M1d[b]->GetDiag(diag1d[b]);
      if (n == 0) { continue; }
      DenseMatrix M(*M1d[b]), inv(n);
      CholeskyFactors chol(M.GetData());
      MFEM_VERIFY(chol.Factor(n), "the 1D mass matrix is not SPD");
      chol.GetInverseMatrix(n, inv.GetData());
      for (int j = 0; j < n; j++)
      {
         for (int i = 0; i < n; i++)
         {
            h_F[o + i + j*n] = (weighting == Weighting::DIAGONAL) ?
                               diag1d[b](i) * inv(i,j) * diag1d[b](j) :
                               inv(i,j);
         }
      }
      o += n*n;
   }

   // The 1D basis (0 = closed, 1 = open) and the factor of each component in
   // each direction.
   tensor_ncomp = vector_fe ? dim : 1;
   int basis[3][3];
   int nd_s = 0;
   for (int c = 0; c < tensor_ncomp; c++)
   {
      int size = 1;
      for (int d = 0; d < 3; d++)
      {
         if (d >= dim)
         {
            basis[c][d] = -1;
            tensor_n[c][d] = 1;
            tensor_offset[c][d] = 0;
            continue;
         }
         const bool open = vector_fe &&
                           ((map_type == FiniteElement::H_CURL) == (d == c));
         basis[c][d] = open ? 1 : 0;
         tensor_n[c][d] = M1d[basis[c][d]]->Height();
         tensor_offset[c][d] = offset1d[basis[c][d]];
         size *= tensor_n[c][d];
      }
      nd_s += size;
   }
   MFEM_VERIFY(nd_s == fe.GetDof(), prefix << "unexpected number of element "
               "DOFs");

   // The scales and the diagonal of W, from the Jacobian and the coefficients
   // at the element centers. The entries of the Jacobian are polynomials of
   // degree at most T.Order() in each variable, so they are constant if they
   // are equal at the points of a tensor product Gauss rule with
   // T.Order() + 1 points per direction.
   tensor_scale.SetSize(tensor_ncomp*ne);
   real_t *h_s = tensor_scale.HostWrite();
   real_t *h_w = w.HostWrite();
   const int nd = nd_s*vdim;
   const IntegrationPoint &center = Geometries.GetCenter(geom);
   DenseMatrix J, JtJ(dim);
   for (int e = 0; e < ne; e++)
   {
      ElementTransformation &T = *fes->GetElementTransformation(e);
      T.SetIntPoint(&center);
      J = T.Jacobian();
      const real_t tol = 1e-12*J.MaxMaxNorm();
      const IntegrationRule &ir = IntRules.Get(geom, 2*T.Order());
      for (int k = 0; k < ir.GetNPoints(); k++)
      {
         T.SetIntPoint(&ir.IntPoint(k));
         const DenseMatrix &Jk = T.Jacobian();
         for (int i = 0; i < J.Height()*J.Width(); i++)
         {
            MFEM_VERIFY(std::abs(Jk.GetData()[i] - J.GetData()[i]) <= tol,
                        prefix << "element " << e << " is not affine");
         }
      }
      T.SetIntPoint(&center);
      const real_t det = std::abs(T.Weight());

      MultAtB(J, J, JtJ);
      for (int i = 0; vector_fe && i < dim; i++)
      {
         for (int j = 0; j < i; j++)
         {
            MFEM_VERIFY(std::abs(JtJ(i,j)) <=
                        1e-12*std::sqrt(JtJ(i,i)*JtJ(j,j)), prefix << "the "
                        "Jacobian of element " << e << " does not have "
                        "orthogonal columns, which ND and RT spaces require");
         }
      }

      real_t coeff = 0.0;
      for (int k = 0; k < integs.Size(); k++)
      {
         coeff += coeffs[k] ? coeffs[k]->Eval(T, center) : 1.0;
      }
      MFEM_VERIFY(coeff > 0.0, "the matrix of element " << e
                  << " is not SPD");

      real_t *w_e = h_w + e*nd;
      for (int v = 0, o = 0; v < vdim; v++)
      {
         for (int c = 0; c < tensor_ncomp; c++)
         {
            real_t m = coeff;
            switch (map_type)
            {
               case FiniteElement::H_CURL: m *= det/JtJ(c,c); break;
               case FiniteElement::H_DIV: m *= JtJ(c,c)/det; break;
               case FiniteElement::INTEGRAL: m /= det; break;
               default: m *= det; break;
            }
            h_s[c + tensor_ncomp*e] =
               (weighting == Weighting::DIAGONAL) ? m : 1.0/m;

            // The diagonal of the Kronecker product is the product of the
            // 1D diagonals.
            const int *n = tensor_n[c];
            for (int i2 = 0; i2 < n[2]; i2++)
            {
               for (int i1 = 0; i1 < n[1]; i1++)
               {
                  for (int i0 = 0; i0 < n[0]; i0++, o++)
                  {
                     if (weighting == Weighting::IDENTITY)
                     {
                        w_e[o] = 1.0;
                        continue;
                     }
                     const int idx[3] = { i0, i1, i2 };
                     w_e[o] = m;
                     for (int d = 0; d < dim; d++)
                     {
                        w_e[o] *= diag1d[basis[c][d]](idx[d]);
                     }
                  }
               }
            }
         }
      }
   }
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
// generalized Rayleigh quotient (v^T A_W v)/(v^T S^{-1} v), and multiplied by
// a safety factor, since the estimate is below the largest eigenvalue. This
// gives a better polynomial than the (several times larger) upper bounds of
// Gershgorin type. Since the trace of S A_W is the number of free DOFs, its
// eigenvalues average to one, so a <= 1 and the largest eigenvalue is at
// least 1.
//
// The start vector of the power iteration must not be (nearly) orthogonal to
// the dominant eigenvectors. Smooth vectors, such as the vector of ones, are
// invariant under the symmetries of the mesh and can be: then 20 iterations
// underestimate the largest eigenvalue by up to 35%, and the Chebyshev
// polynomial is a poor approximation of A_W^{-1}. A random vector would
// depend on the parallel partitioning. Instead, the start vector is
// v = P^T v_L, where element e adds the value ChebyshevStartValue(c_e, j) to
// its j-th local DOF (with the orientation sign), c_e being the average of
// the vertices of e. This vector is quasi-random and is the same for any
// partitioning, up to the orientation signs of the true DOFs.
static real_t ChebyshevStartValue(const Vector &c, int j)
{
   real_t t = 2.39996322972865332*j; // golden angle
   const real_t k[3] = { 12.9898, 78.233, 37.719 };
   for (int d = 0; d < c.Size(); d++) { t += k[d]*c(d); }
   return std::sin(t);
}

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

   // The start vector, see above, on the free DOFs.
   const Mesh &mesh = *fes->GetMesh();
   const int sdim = mesh.SpaceDimension();
   Vector vl(fes->GetVSize()), c(sdim);
   vl.UseDevice(true);
   vl = 0.0;
   real_t *h_vl = vl.HostReadWrite();
   Array<int> verts, vdofs;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      mesh.GetElementVertices(e, verts);
      c = 0.0;
      for (int i : verts)
      {
         const real_t *x = mesh.GetVertex(i);
         for (int d = 0; d < sdim; d++) { c(d) += x[d]; }
      }
      c *= 1.0/verts.Size();
      fes->GetElementVDofs(e, vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         const int k = vdofs[j];
         const real_t s = ChebyshevStartValue(c, j);
         if (k >= 0) { h_vl[k] += s; }
         else { h_vl[-1-k] -= s; }
      }
   }
   Vector v(height), Av(height), Sinv(diag_inv);
   v.UseDevice(true);
   Av.UseDevice(true);
   Sinv.Reciprocal();
   P->MultTranspose(vl, v);
   v.SetSubVector(ess_tdof_list, 0.0);

   // Power iteration. The number of iterations and the safety factor are
   // enough for the purpose of the estimate.
   const int power_its = 20;
   const real_t safety = 1.1;
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
   cheb_b = std::max({safety*b, 1.0, (1.0 + 1e-6)*a});
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
   ess_tdof_list.HostRead();
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
   // matrix-vector product per element, or sum factorization for
   // ElementInverse::TENSOR. The check skips ranks without elements. Then
   // y = R_E^T ye sums the element contributions of each DOF.
   R->Mult(x, xe);
   if (element_inverse == ElementInverse::TENSOR)
   {
      R->MultTranspose(ApplyTensor(), y);
      return;
   }
   if (WMinvW.SizeK() > 0) { BatchedLinAlg::Mult(WMinvW, xe, ye); }
   R->MultTranspose(ye, y);
}

// One direction d of ApplyTensor(): y = (I x F_d x I) x on each component
// block of the E-vectors x and y, multiplied by the scales if last is true.
// With the sizes n_c of component c, the block is an n_lo x n x n_hi array,
// where n = n_c[d] and n_lo is the product of the sizes in the directions
// before d. One thread computes one entry of y.
static void ApplyTensorDirection(int d, bool last, int ne, int vdim,
                                 int ncomp, const int tensor_n[3][3],
                                 const int tensor_offset[3][3],
                                 const Vector &F, const Vector &scale,
                                 const Vector &x, Vector &y)
{
   // The size n and the size n_lo of each component in direction d, the
   // offset of its factor F_d in F, and the offsets of the components in
   // the element DOFs of one vector dimension.
   int n[3], n_lo[3], off_F[3], off_c[4] = { 0, 0, 0, 0 };
   for (int c = 0; c < ncomp; c++)
   {
      const int *n_c = tensor_n[c];
      n[c] = n_c[d];
      n_lo[c] = (d > 0 ? n_c[0] : 1) * (d > 1 ? n_c[1] : 1);
      off_F[c] = tensor_offset[c][d];
      off_c[c+1] = off_c[c] + n_c[0]*n_c[1]*n_c[2];
   }
   const int nd_s = off_c[ncomp];
   const auto d_F = F.Read();
   const auto d_s = scale.Read();
   const auto d_x = x.Read();
   auto d_y = y.Write();
   mfem::forall(ne*vdim*nd_s, [=] MFEM_HOST_DEVICE (int k)
   {
      // Entry k is entry r of the vector dimension block of element e, in
      // component c, at index (l, i, h) of the n_lo x n x n_hi array.
      const int e = k/(vdim*nd_s);
      const int r = k % nd_s;
      int c = 0;
      while (r >= off_c[c+1]) { c++; }
      const int rc = r - off_c[c];
      const int l = rc % n_lo[c];
      const int i = (rc/n_lo[c]) % n[c];
      const int h = rc/(n_lo[c]*n[c]);
      // The entry (l, 0, h) of the array in x.
      const int o = k - rc + l + n_lo[c]*n[c]*h;
      const real_t *F_c = d_F + off_F[c];
      real_t sum = 0.0;
      for (int j = 0; j < n[c]; j++)
      {
         sum += F_c[i + j*n[c]] * d_x[o + n_lo[c]*j];
      }
      d_y[k] = last ? d_s[c + ncomp*e]*sum : sum;
   });
}

// For each element and vector component, apply the Kronecker product
// F_0 x F_1 x F_2 one direction at a time: the coefficients of the component
// form an n_0 x n_1 x n_2 array (direction 0 fastest), and the factor F_d
// multiplies its index in direction d. Each direction alternates between xe
// and ye, and the last one multiplies by the scales. The directions beyond
// the mesh dimension have size 1 and factor [1], and are skipped.
const Vector &BRASPreconditioner::ApplyTensor() const
{
   const int dim = fes->GetMesh()->Dimension();
   Vector *src = &xe, *dst = &ye;
   for (int d = 0; d < dim; d++)
   {
      ApplyTensorDirection(d, d == dim-1, fes->GetNE(), fes->GetVDim(),
                           tensor_ncomp, tensor_n, tensor_offset, tensor_1d,
                           tensor_scale, *src, *dst);
      std::swap(src, dst);
   }
   return *src;
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

// On conforming spaces, apply B_W, i.e.
// y = S P_L^T R_E^T (W M_br^{-1} W) R_E P_L S x on the free DOFs, where
// S = diag(scale); see SetEssentialTrueDofs(). The nonconforming methods are
// applied by MultLeftInverse() and MultChebyshev(). In all cases y = x on the
// essential DOFs. The only communication is in P_L and P_L^T.
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

// G_m = R_E^T (W M_br^{-1} W) R_E. The element VDOFs are signed in the same
// way as the element restriction, and AddSubMatrix() applies the signs of
// negative indices, so block e is added in the basis of the E-vector.
//
// For ElementInverse::TENSOR, block e is formed in the lexicographic order of
// the E-vector and then permuted to the native order of the element VDOFs
// with the DOF map: lexicographic DOF i is native DOF idx = d or -1-d, where
// d = dof_map[i], with the sign + or - (an empty map is the identity). The
// sign is the one applied by the lexicographic element restriction.
SparseMatrix *BRASPreconditioner::AssembleBroken() const
{
   auto *G = new SparseMatrix(fes->GetVSize());
   Array<int> vdofs;
   if (element_inverse == ElementInverse::DENSE)
   {
      WMinvW.HostRead();
      for (int e = 0; e < fes->GetNE(); e++)
      {
         fes->GetElementVDofs(e, vdofs);
         G->AddSubMatrix(vdofs, vdofs, WMinvW(e), 0);
      }
      G->Finalize(0);
      return G;
   }

   const FiniteElement &fe = *fes->GetTypicalFE();
   const Array<int> &dof_map =
      dynamic_cast<const TensorBasisElement&>(fe).GetDofMap();
   const int nd_s = fe.GetDof(), vdim = fes->GetVDim();
   const int dim = fes->GetMesh()->Dimension();
   Array<int> perm(nd_s);
   Vector sign(nd_s);
   for (int i = 0; i < nd_s; i++)
   {
      const int d = dof_map.Size() ? dof_map[i] : i;
      perm[i] = (d >= 0) ? d : -1 - d;
      sign(i) = (d >= 0) ? 1.0 : -1.0;
   }
   const real_t *h_F = tensor_1d.HostRead();
   const real_t *h_s = tensor_scale.HostRead();
   DenseMatrix block(nd_s*vdim);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      block = 0.0;
      for (int c = 0, o = 0; c < tensor_ncomp; c++)
      {
         // Entry (i, j) of the Kronecker product is the product of the 1D
         // factors at the multi-indices of i and j.
         const int *n = tensor_n[c];
         const int size = n[0]*n[1]*n[2];
         const real_t s = h_s[c + tensor_ncomp*e];
         for (int j = 0; j < size; j++)
         {
            for (int i = 0; i < size; i++)
            {
               real_t a = s;
               for (int d = 0, ii = i, jj = j; d < dim; d++)
               {
                  a *= h_F[tensor_offset[c][d] + ii%n[d] + n[d]*(jj%n[d])];
                  ii /= n[d];
                  jj /= n[d];
               }
               const int pi = perm[o + i], pj = perm[o + j];
               const real_t a_ij = sign(o + i) * a * sign(o + j);
               for (int v = 0; v < vdim; v++)
               {
                  block(pi + v*nd_s, pj + v*nd_s) = a_ij;
               }
            }
         }
         o += size;
      }
      fes->GetElementVDofs(e, vdofs);
      G->AddSubMatrix(vdofs, vdofs, block, 0);
   }
   G->Finalize(0);
   return G;
}

// The vector of size n equal to 1 on the entries in list and 0 elsewhere.
static Vector Indicator(int n, const Array<int> &list)
{
   Vector v(n);
   v = 0.0;
   v.SetSubVector(list, 1.0);
   v.HostRead();
   return v;
}

// The vector 1 - mask.
static Vector Complement(const Vector &mask)
{
   Vector v(mask.Size());
   v = 1.0;
   v -= mask;
   v.HostRead();
   return v;
}

// The preconditioner on the free DOFs is Q^T G_m Q, where u = Q x is the
// L-vector passed to G_m in Mult(). On conforming spaces Q = P_L S. On
// nonconforming spaces Q = L_X^T, and from the formula for u in
// MultLeftInverse(), with J = diag(jmask), H = diag(hdl) and
// U = diag(lu_scale),
//    Q = J P_L U (I - P_L^T H P_L S) + (I - J) P_L S.
// The columns of Q of the essential DOFs vanish, since S and U vanish there,
// so adding the identity on the essential DOFs gives the operator of Mult().
SparseMatrix *BRASPreconditioner::AssembleMatrix() const
{
   MFEM_VERIFY(!nc || nc_method == NCMethod::LEFT_INVERSE,
               "NCMethod::CHEBYSHEV cannot be assembled");
#ifdef MFEM_USE_MPI
   MFEM_VERIFY(!dynamic_cast<const ParFiniteElementSpace*>(fes),
               "use ParallelAssembleMatrix() for parallel spaces");
#endif
   std::unique_ptr<SparseMatrix> G(AssembleBroken());
   scale.HostRead();

   // Q = P_L S; P_L is NULL for conforming spaces.
   const SparseMatrix *sP = fes->GetConformingProlongation();
   std::unique_ptr<SparseMatrix> Q;
   if (!sP) { Q.reset(new SparseMatrix(scale)); }
   else
   {
      // P_L may have been used on the device in Mult(), so the copies are
      // made on the host.
      sP->HostReadI();
      sP->HostReadJ();
      sP->HostReadData();
      Q.reset(new SparseMatrix(*sP, true, MemoryType::HOST));
      Q->ScaleColumns(scale);
   }
   if (nc)
   {
      jmask.HostRead();
      hdl.HostRead();
      lu_scale.HostRead();
      // PJ = J P_L U and T = P_L^T H P_L S.
      SparseMatrix PJ(*sP, true, MemoryType::HOST), HPS(*Q);
      PJ.ScaleRows(jmask);
      PJ.ScaleColumns(lu_scale);
      HPS.ScaleRows(hdl);
      std::unique_ptr<SparseMatrix> T(mfem::TransposeMult(*sP, HPS));
      std::unique_ptr<SparseMatrix> PJT(mfem::Mult(PJ, *T));
      // Q = PJ - PJ T + (I - J) P_L S.
      Q->ScaleRows(Complement(jmask));
      std::unique_ptr<SparseMatrix> Q1(mfem::Add(PJ, *Q));
      Q.reset(Add(1.0, *Q1, -1.0, *PJT));
   }

   std::unique_ptr<SparseMatrix> B(mfem::RAP(*Q, *G, *Q));
   SparseMatrix I_ess(Indicator(height, ess_tdof_list));
   return mfem::Add(*B, I_ess);
}

#ifdef MFEM_USE_MPI
// A diag(s), where s is a vector on the columns of A owned by this rank.
static HypreParMatrix *ScaleColumns(const HypreParMatrix &A, const Vector &s)
{
   SparseMatrix s_diag(s);
   HypreParMatrix D(A.GetComm(), A.N(), A.GetColStarts(), &s_diag);
   return ParMult(&A, &D, true);
}

// The parallel version of AssembleMatrix(). The prolongation is the matrix
// P_L = Dof_TrueDof_Matrix(), which is the operator applied in Mult() also
// when the space is conforming.
HypreParMatrix *BRASPreconditioner::ParallelAssembleMatrix() const
{
   auto pfes = dynamic_cast<const ParFiniteElementSpace*>(fes);
   MFEM_VERIFY(pfes, "use AssembleMatrix() for serial spaces");
   MFEM_VERIFY(!nc || nc_method == NCMethod::LEFT_INVERSE,
               "NCMethod::CHEBYSHEV cannot be assembled");
   // The parallel matrix G uses the data of G_loc.
   std::unique_ptr<SparseMatrix> G_loc(AssembleBroken());
   HypreParMatrix G(pfes->GetComm(), pfes->GlobalVSize(),
                    pfes->GetDofOffsets(), G_loc.get());
   scale.HostRead();

   // Q = P_L S.
   const HypreParMatrix &hP = *pfes->Dof_TrueDof_Matrix();
   std::unique_ptr<HypreParMatrix> Q(ScaleColumns(hP, scale));
   if (nc)
   {
      jmask.HostRead();
      hdl.HostRead();
      lu_scale.HostRead();
      // PJ = J P_L U and T = P_L^T H P_L S.
      HypreParMatrix JP(hP), HPS(*Q);
      JP.ScaleRows(jmask);
      std::unique_ptr<HypreParMatrix> PJ(ScaleColumns(JP, lu_scale));
      HPS.ScaleRows(hdl);
      std::unique_ptr<HypreParMatrix> Pt(hP.Transpose());
      std::unique_ptr<HypreParMatrix> T(ParMult(Pt.get(), &HPS, true));
      std::unique_ptr<HypreParMatrix> PJT(ParMult(PJ.get(), T.get(), true));
      // Q = PJ - PJ T + (I - J) P_L S.
      Q->ScaleRows(Complement(jmask));
      std::unique_ptr<HypreParMatrix> Q1(Add(1.0, *PJ, 1.0, *Q));
      Q.reset(Add(1.0, *Q1, -1.0, *PJT));
   }

   std::unique_ptr<HypreParMatrix> B(RAP(&G, Q.get()));
   SparseMatrix I_loc(Indicator(height, ess_tdof_list));
   HypreParMatrix I_ess(B->GetComm(), B->M(), B->GetRowStarts(), &I_loc);
   return Add(1.0, *B, 1.0, I_ess);
}
#endif

} // namespace mfem
