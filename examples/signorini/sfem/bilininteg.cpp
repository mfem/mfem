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
#endif
   MFEM_ASSERT(Trans.Elem2No < 0,
               "support for interior faces is not implemented");

   const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   const int nvdofs = dim * ndofs1;

   elmat.SetSize(nvdofs);
   elmat = 0.0;

   shape1.SetSize(ndofs1);
   w.SetSize(dim);

   real_t val;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 2 * el1.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   Vector n(dim);
   Trans.SetIntPoint(&Geometries.GetCenter(Trans.GetGeometryType()));
   CalcOrtho(Trans.Jacobian(), n);
   n /= n.Norml2();

   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      el1.CalcPhysShape(*Trans.Elem1, shape1);

      val = ip.weight * Trans.Weight() * Q.Eval(Trans, ip);

      if (!W) { w = n; }
      else { W->Eval(w, Trans, ip); }

      for (int jm = 0, j = 0; jm < dim; ++jm)
      {
         for (int jdof = 0; jdof < ndofs1; ++jdof, ++j)
         {
            const real_t sj = val * shape1(jdof) * w(jm);
            for (int im = 0, i = 0; im < dim; ++im)
            {
               for (int idof = 0; idof < ndofs1; ++idof, ++i)
               {
                  elmat(i, j) += shape1(idof) * sj * w(im);
               }
            }
         }
      }
   }
}

}
