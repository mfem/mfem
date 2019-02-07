#ifndef DGPA_OPERS
#define DGPA_OPERS

#include "dg_pa.hpp"

namespace mfem
{
namespace dg
{

template <typename D>
class BtDB : public Operator
{
   const PartialAssembly *pa;
   const FiniteElementSpace *fes;
public:
   D d;
   BtDB(const PartialAssembly *pa_) : pa(pa_), fes(pa->GetFES()) { }
   void Mult(const Vector &x, Vector &y) const
   {
      const int nc = d.NComponents();
      Array<int> vdofs;
      DenseMatrix xquad, yquad;
      DenseMatrix xel, yel;
      Vector xpt(nc), ypt(nc);
      for (int i = 0; i < fes->GetNE(); i++)
      {
         const DenseMatrix &B = pa->BasisEval(i);
         const int nquad = B.Height();
         const int ndof = B.Width();
         xquad.SetSize(nquad, nc);
         yquad.SetSize(nquad, nc);
         xel.SetSize(ndof, nc);
         yel.SetSize(ndof, nc);
         fes->GetElementVDofs(i, vdofs);
         x.GetSubVector(vdofs, xel.Data());
         mfem::Mult(B, xel, xquad);
         for (int iq = 0; iq < nquad; ++iq)
         {
            for (int ic = 0; ic < nc; ++ic)
            {
               xpt(ic) = xquad(iq,ic);
            }
            d(xpt, ypt);
            double w = pa->GetMetricTerms().Weight(i, iq);
            for (int ic = 0; ic < nc; ++ic)
            {
               yquad(iq,ic) = ypt(ic)*w;
            }
         }
         MultAtB(B, yquad, yel);
         y.AddElementVector(vdofs, yel.Data());
      }
   }
};

template <typename D>
class GtDB : public Operator
{
   const PartialAssembly *pa;
   const FiniteElementSpace *fes;
public:
   D d;
   GtDB(const PartialAssembly *pa_) : pa(pa_), fes(pa->GetFES()) { }
   void Mult(const Vector &x, Vector &y) const
   {
      const int nc = d.NComponents();
      const int dim = fes->GetFE(0)->GetDim();
      Array<int> vdofs;
      DenseMatrix xquad, xel, yel;
      DenseTensor yquad;
      Vector xpt(nc);
      DenseMatrix F(nc,dim);
      for (int i = 0; i < fes->GetNE(); i++)
      {
         const DenseMatrix &B = pa->BasisEval(i);
         const DenseTensor &G = pa->DerivEval(i);
         const int nquad = B.Height();
         const int ndof = B.Width();
         xquad.SetSize(nquad, nc);
         yquad.SetSize(nquad, nc, dim);
         xel.SetSize(ndof, nc);
         yel.SetSize(ndof, nc);
         fes->GetElementVDofs(i, vdofs);
         x.GetSubVector(vdofs, xel.Data());
         mfem::Mult(B, xel, xquad);
         yquad = 0.0;
         for (int iq = 0; iq < nquad; ++iq)
         {
            for (int ic = 0; ic < nc; ++ic)
            {
               xpt(ic) = xquad(iq, ic);
            }
            d(xpt, F.Data());
            const DenseMatrix &Jinv = pa->GetMetricTerms().InverseJacobian(i, iq);
            double w = pa->GetMetricTerms().Weight(i, iq);
            for (int ic = 0; ic < nc; ++ic)
            {
               for (int d2 = 0; d2 < dim; ++d2)
               {
                  for (int d1 = 0; d1 < dim; ++d1)
                  {
                     yquad(iq, ic, d1) += w*Jinv(d1,d2)*F(ic,d2);
                  }
               }
            }
         }
         for (int d = 0; d < dim; ++d)
         {
            MultAtB(G(d), yquad(d), yel);
            y.AddElementVector(vdofs, yel.Data());
         }
      }
   }
};

template <typename D>
class BtDB_face : public Operator
{
   const PartialAssembly *pa;
   const FiniteElementSpace *fes;
public:
   D d;
   BtDB_face(const PartialAssembly *pa_) : pa(pa_), fes(pa->GetFES()) { }
   void Mult(const Vector &x, Vector &y) const
   {
      const int nfaces = fes->GetMesh()->GetNumFaces();
      const int nc = d.NComponents();
      Array<int> vdofs1, vdofs2;
      DenseMatrix xel1, xel2, xquad1, xquad2, yel1, yel2, yquad;
      Vector xpt1(nc), xpt2(nc), Fdotn(nc);
      for (int i = 0; i < nfaces; ++i)
      {
         PartialAssembly::F2E i1, i2;
         int iel1, iel2;
         pa->GetFES()->GetMesh()->GetFaceElements(i, &iel1, &iel2);
         pa->GetF2Es(i, i1, i2);

         // 1. Evaluate DOFs from element 1 at the face
         const DenseMatrix &Bface1 = pa->FaceEval(i1);
         const int nquad = Bface1.Height();
         const int ndof1 = Bface1.Width();
         xel1.SetSize(ndof1,nc);
         yel1.SetSize(ndof1,nc);
         xquad1.SetSize(nquad,nc);
         fes->GetElementVDofs(iel1, vdofs1);
         x.GetSubVector(vdofs1, xel1.Data());
         mfem::Mult(Bface1, xel1, xquad1);
         // Is the face interior? This will check validity of the second element
         if (i2)
         {
            // 2. If the face is interior, evaluate DOFs from element 2 at the face
            const DenseMatrix &Bface2 = pa->FaceEval(i2);
            const int ndof2 = Bface2.Width();
            xel2.SetSize(ndof2, nc);
            yel2.SetSize(ndof2, nc);
            fes->GetElementVDofs(iel2, vdofs2);
            x.GetSubVector(vdofs2, xel2.Data());
            xquad2.SetSize(nquad, nc);
            mfem::Mult(Bface2, xel2, xquad2);
         }
         else
         {
            xquad2 = xquad1;
         }

         // 3. Evaluate the two-point numerical flux at each quadrature point
         yquad.SetSize(nquad, nc);
         for (int iq = 0; iq < nquad; ++iq)
         {
            for (int ic = 0; ic < nc; ++ic)
            {
               xpt1(ic) = xquad1(iq, ic);
               xpt2(ic) = xquad2(iq, ic);
            }
            // Compute face normals
            const Vector &nvec = pa->GetMetricTerms().Normal(i, iq);
            d(xpt1, xpt2, nvec.GetData(), Fdotn);
            double w = pa->GetMetricTerms().FaceWeight(i, iq);
            // Multiply by geometric factors and quadrature weights
            for (int ic = 0; ic < nc; ++ic)
            {
               yquad(iq, ic) = w*Fdotn(ic);
            }
         }

         // 4. Integrate against test functions by multiplying pointwise by
         //    quadrature weights, and then multiply by the transpose of the
         //    operators from steps 1 and 2
         // 5. Add back to residual DOFs for element 1 (and element 2
         //    if the face is interior)

         MultAtB(Bface1, yquad, yel1);
         yel1 *= -1.0;
         y.AddElementVector(vdofs1, yel1.Data());

         if (i2)
         {
            const DenseMatrix &Bface2 = pa->FaceEval(i2);
            MultAtB(Bface2, yquad, yel2);
            y.AddElementVector(vdofs2, yel2.Data());
         }
      }
   }
};

} // namespace dg
} // namespace mfem

#endif
