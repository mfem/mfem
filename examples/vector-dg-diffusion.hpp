#include "mfem.hpp"

namespace mfem
{

class VectorDGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q = nullptr;
   MatrixCoefficient *MQ = nullptr;
   real_t sigma, kappa;
   int vdim;

   // these are not thread-safe!
   Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
   VectorDGDiffusionIntegrator(real_t s, real_t k, int vd=-1)
      : sigma(s), kappa(k), vdim(vd) { }
   VectorDGDiffusionIntegrator(Coefficient &q, real_t s, real_t k, int vd=-1)
      : Q(&q), sigma(s), kappa(k), vdim(vd) { }
   VectorDGDiffusionIntegrator(MatrixCoefficient &mq, real_t s, real_t k,
                               int vd=-1)
      : MQ(&mq), sigma(s), kappa(k), vdim(vd) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &full_elmat);
};

class VectorDGDirichletLFIntegrator : public LinearFormIntegrator
{
protected:
   VectorCoefficient &uD;
   Coefficient *Q = nullptr;
   MatrixCoefficient *MQ = nullptr;
   real_t sigma, kappa;
   int vdim;

   // these are not thread-safe!
   Vector shape, dshape_dn, nor, nh, ni, uD_vec;
   DenseMatrix dshape, mq, adjJ;

public:
   VectorDGDirichletLFIntegrator(VectorCoefficient &u, real_t s, real_t k,
                                 int vd=-1)
      : uD(u), sigma(s), kappa(k), vdim(vd) { }
   VectorDGDirichletLFIntegrator(VectorCoefficient &u, Coefficient &q, real_t s,
                                 real_t k, int vd=-1)
      : uD(u), Q(&q), sigma(s), kappa(k), vdim(vd) { }
   VectorDGDirichletLFIntegrator(VectorCoefficient &u, MatrixCoefficient &mq,
                                 real_t s, real_t k, int vd=-1)
      : uD(u), MQ(&mq), sigma(s), kappa(k), vdim(vd) { }

   using LinearFormIntegrator::AssembleRHSElementVect;

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   { MFEM_ABORT("Not implemented."); }

   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override;
};

void VectorDGDiffusionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &full_elmat)
{
   int dim, ndof1, ndof2, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   real_t w, wq = 0.0;

   const int sdim = Trans.GetSpaceDim();
   if (vdim < 0) { vdim = sdim; }

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
      dshape2dn.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   ndofs = ndof1 + ndof2;
   DenseMatrix elmat;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (kappa_is_nonzero)
   {
      jmat.SetSize(ndofs);
      jmat = 0.;
   }

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      // a simple choice for the integration order
      int order;
      if (ndof2)
      {
         order = 2 * std::max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2 * el1.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: < {(Q \nabla u).n},[v] >      --> elmat
   //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2 * eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);
      w = ip.weight / Trans.Elem1->Weight();
      if (ndof2)
      {
         w /= 2;
      }
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Trans.Elem1, eip1);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Trans.Elem1, eip1);
         mq.MultTranspose(nh, ni);
      }
      CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);
      if (kappa_is_nonzero)
      {
         wq = ni * nor;
      }

      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //
      //      |nor|=measure(face)/measure(ref. face),
      //
      //      det(J1)=measure(element)/measure(ref. element),
      //
      //      and the ratios measure(ref. element)/measure(ref. face)
      //      are compatible for all element/face pairs.
      //
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      //
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      dshape1.Mult(nh, dshape1dn);
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i) * dshape1dn(j);
         }

      if (ndof2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         w = ip.weight / 2 / Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq += ni * nor;
         }

         dshape2.Mult(nh, dshape2dn);

         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
            }
      }

      if (kappa_is_nonzero)
      {
         // only assemble the lower triangular part of jmat
         wq *= kappa;
         for (int i = 0; i < ndof1; i++)
         {
            const real_t wsi = wq * shape1(i);
            for (int j = 0; j <= i; j++)
            {
               jmat(i, j) += wsi * shape1(j);
            }
         }
         if (ndof2)
         {
            for (int i = 0; i < ndof2; i++)
            {
               const int i2 = ndof1 + i;
               const real_t wsi = wq * shape2(i);
               for (int j = 0; j < ndof1; j++)
               {
                  jmat(i2, j) -= wsi * shape1(j);
               }
               for (int j = 0; j <= i; j++)
               {
                  jmat(i2, ndof1 + j) += wsi * shape2(j);
               }
            }
         }
      }
   }

   // elmat := -elmat + sigma*elmat^t + jmat
   if (kappa_is_nonzero)
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            real_t aij = elmat(i, j), aji = elmat(j, i), mij = jmat(i, j);
            elmat(i, j) = sigma * aji - aij + mij;
            elmat(j, i) = sigma * aij - aji + mij;
         }
         elmat(i, i) = (sigma - 1.) * elmat(i, i) + jmat(i, i);
      }
   }
   else
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            real_t aij = elmat(i, j), aji = elmat(j, i);
            elmat(i, j) = sigma * aji - aij;
            elmat(j, i) = sigma * aij - aji;
         }
         elmat(i, i) *= (sigma - 1.);
      }
   }

   // populate full matrix following github issue #2909
   full_elmat.SetSize(vdim*(ndof1 + ndof2));
   full_elmat = 0.0;
   for (int d=0; d<vdim; ++d)
   {
      for (int j=0; j<ndofs; ++j)
      {
         int jj = (j < ndof1) ? j + d*ndof1 : j - ndof1 + d*ndof2 + vdim*ndof1;
         for (int i=0; i<ndofs; ++i)
         {
            int ii = (i < ndof1) ? i + d*ndof1 : i - ndof1 + d*ndof2 + vdim*ndof1;
            full_elmat(ii, jj) += elmat(i, j);
         }
      }
   }
};

void VectorDGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   const int dim = el.GetDim();
   const int sdim = Tr.GetSpaceDim();

   if (vdim < 0) { vdim = sdim; }

   const int ndof = el.GetDof();

   bool kappa_is_nonzero = (kappa != 0.);
   real_t w;

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(vdim * ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      uD.Eval(uD_vec, Tr, ip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      // compute uD through the face transformation
      w = ip.weight / Tr.Elem1->Weight();
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Tr.Elem1, eip);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Tr.Elem1, eip);
         mq.MultTranspose(nh, ni);
      }
      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);

      dshape.Mult(nh, dshape_dn);

      for (int vd = 0; vd < vdim; ++vd)
      {
         for (int i = 0; i < ndof; ++i)
         {
            elvect[i + vd*ndof] += sigma * uD_vec[vd] * dshape_dn[i];
         }
      }
      if (kappa_is_nonzero)
      {
         for (int vd = 0; vd < vdim; ++vd)
         {
            for (int i = 0; i < ndof; ++i)
            {
               elvect[i + vd*ndof] += kappa*(ni*nor) * uD_vec[vd] * shape[i];
            }
         }
      }
   }
}

} // namespace mfem
