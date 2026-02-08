#include "mfem.hpp"
#include "lininteg.hpp"

namespace mfem
{

void BoundaryProjectionLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("BoundaryProjectionLFIntegrator::AssembleRHSElementVect");
}

void BoundaryProjectionLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int spaceDim = Tr.GetSpaceDim();
   int vdim = std::max(spaceDim, el.GetRangeDim());
   int dof  = el.GetDof();

   real_t val, cf;

   shape.SetSize(dof);
   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;    // <------ user control
      ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      val = Tr.Face->Weight() * Q.Eval(*Tr.Face, ip);

      el.CalcShape(eip, shape);

      for (int k = 0; k < vdim; k++)
      {
         cf = val * W(k);

         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += ip.weight * cf * shape(s);
         }
      }
   }
}

void VectorDomainLFStrainIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   const int dim = el.GetDim();
   const int dof = el.GetDof();
   const int vdim = Q.GetVDim();
   const int sdim = Tr.GetSpaceDim();

   dshape.SetSize(dof, sdim);
   elvect.SetSize(dof * (vdim / sdim));
   elvect = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   Vector pelvect(dof);
   Vector part_x(dim);

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);

      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);

      Q.Eval(Qvec, Tr, ip);
      Qvec *= 0.5 * ip.weight * Tr.Weight();

      for (int k = 0; k < vdim / sdim; k++)
      {
         pelvect = 0.0;

         for (int d = 0; d < sdim; ++d) { part_x(d) = Qvec(k*sdim+d) + Qvec(d*sdim+k); }
         dshape.Mult(part_x, pelvect);
         for (int s = 0; s < dof; s++) { elvect(k*dof+s) += pelvect(s); }
      }
   }
}

} // namespace mfem
