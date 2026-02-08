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
#endif

   const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   const int nvdofs = dim * ndofs1;

   elmat.SetSize(nvdofs);
   elmat = 0.0;

   shape1.SetSize(ndofs1);

   real_t val;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 2 * el1.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      el1.CalcShape(eip1, shape1);

      val = ip.weight / Trans.Elem1->Weight() * Q.Eval(*Trans.Elem1, eip1);

      for (int jm = 0, j = 0; jm < dim; ++jm)
      {
         for (int jdof = 0; jdof < ndofs1; ++jdof, ++j)
         {
            const real_t sj = val * shape1(jdof) * W(jm);
            for (int im = 0, i = 0; im < dim; ++im)
            {
               for (int idof = 0; idof < ndofs1; ++idof, ++i)
               {
                  elmat(i, j) += shape1(idof) * sj * W(im);
               }
            }
         }
      }
   }
}

}
