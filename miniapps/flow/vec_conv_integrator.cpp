#include "vec_conv_integrator.hpp"

namespace mfem
{
void VectorConvectionNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   Vector &elvect)
{
   int nd = el.GetDof();
   int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   elvect.SetSize(nd * dim);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);
   ELV.UseExternalData(elvect.GetData(), nd, dim);

   double w;
   Vector vec1(dim), vec2(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   // Same as elvect = 0.0;
   ELV = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcPhysDShape(trans, dshape);

      w = ip.weight * trans.Weight();

      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }

      MultAtB(EF, dshape, gradEF);
      EF.MultTranspose(shape, vec1);
      gradEF.Mult(vec1, vec2);
      vec2 *= w;

      AddMultVWt(shape, vec2, ELV);
   }
}

void VectorConvectionNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   double w;
   Vector vec1(dim), vec2(dim), vec3(nd);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      Mult(dshape, trans.InverseJacobian(), dshapex);

      w = ip.weight;

      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }

      MultAtB(EF, dshapex, gradEF);
      EF.MultTranspose(shape, vec1);

      trans.AdjugateJacobian().Mult(vec1, vec2);

      vec2 *= w;
      dshape.Mult(vec2, vec3);
      MultVWt(shape, vec3, elmat_comp);

      for (int i = 0; i < dim; i++)
      {
         elmat.AddMatrix(elmat_comp, i * nd, i * nd);
      }

      MultVVt(shape, elmat_comp);
      w = ip.weight * trans.Weight();
      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            elmat.AddMatrix(w * gradEF(i, j), elmat_comp, i * nd, j * nd);
         }
      }
   }
}
} // namespace mfem
