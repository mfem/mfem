#include "mfem.hpp"
#include "bilininteg.hpp"

namespace mfem
{

void BoundaryProjectionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector shape1;
   Vector w;
   Vector a;
#endif
   MFEM_ASSERT(Trans.Elem2No < 0,
               "support for interior faces is not implemented");

   const int dim = Trans.GetSpaceDim();
   const int ndofs1 = el1.GetDof();
   const int nvdofs = dim * ndofs1;

   elmat.SetSize(nvdofs);
   elmat = 0.0;

   shape1.SetSize(ndofs1);
   w.SetSize(dim);
   a.SetSize(nvdofs);

   real_t val;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 2 * el1.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   Vector n(dim);

   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      el1.CalcPhysShape(*Trans.Elem1, shape1);

      val = ip.weight * Trans.Weight() * Q.Eval(Trans, ip);

      if (!W)
      {
         Trans.SetIntPoint(&Geometries.GetCenter(Trans.GetGeometryType()));
         CalcOrtho(Trans.Jacobian(), n);
         n /= n.Norml2();
         w = n;
      }
      else { W->Eval(w, Trans, ip); }

      for (int m = 0, i = 0; m < dim; ++m)
      {
         const real_t wm = w(m);
         for (int dof = 0; dof < ndofs1; ++dof, ++i)
         {
            a(i) = shape1(dof) * wm;
         }
      }
      AddMult_a_VVt(val, a, elmat);
   }
}

}
