// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"
#include <cmath>
#include "intrules.hpp"

namespace mfem
{
void LinearFormIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                          const Array<int> &markers,
                                          Vector &b)
{
   MFEM_ABORT("Not supported.");
}

void LinearFormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("LinearFormIntegrator::AssembleRHSElementVect(...)");
}

void LinearFormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("LinearFormIntegrator::AssembleRHSElementVect(...)");
}

DomainLFIntegrator::DomainLFIntegrator(Coefficient &QF, int a, int b)
   : DeltaLFIntegrator(QF), Q(QF), oa(a), ob(b)
{
   static Kernels kernels;
}

DomainLFIntegrator::DomainLFIntegrator(Coefficient &QF,
                                       const IntegrationRule *ir)
   : DeltaLFIntegrator(QF, ir), Q(QF), oa(1), ob(1)
{
   static Kernels kernels;
}

void DomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                ElementTransformation &Tr,
                                                Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);

   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      real_t val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcPhysShape(Tr, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void DomainLFIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

void DomainLFGradIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();

   dshape.SetSize(dof, spaceDim);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);

      Q.Eval(Qvec, Tr, ip);
      Qvec *= ip.weight * Tr.Weight();

      dshape.AddMult(Qvec, elvect);
   }
}

void DomainLFGradIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(vec_delta != NULL,"coefficient must be VectorDeltaCoefficient");
   int dof = fe.GetDof();
   int spaceDim = Trans.GetSpaceDim();

   dshape.SetSize(dof, spaceDim);
   fe.CalcPhysDShape(Trans, dshape);

   vec_delta->EvalDelta(Qvec, Trans, Trans.GetIntPoint());

   elvect.SetSize(dof);
   dshape.Mult(Qvec, elvect);
}

void BoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);        // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      real_t val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void BoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);        // vector of size dof
   elvect.SetSize(dof);
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

      real_t val = Tr.Face->Weight() * ip.weight * Q.Eval(*Tr.Face, ip);

      el.CalcShape(eip, shape);

      add(elvect, val, shape, elvect);
   }
}

void BoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector nor(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      if (dim > 1)
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      else
      {
         nor[0] = 1.0;
      }
      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      elvect.Add(ip.weight*(Qvec*nor), shape);
   }
}

void BoundaryTangentialLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector tangent(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   if (dim != 2)
   {
      mfem_error("These methods make sense only in 2D problems.");
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      const DenseMatrix &Jac = Tr.Jacobian();
      tangent(0) =  Jac(0,0);
      tangent(1) = Jac(1,0);

      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight*(Qvec*tangent), shape, elvect);
   }
}

VectorDomainLFIntegrator::VectorDomainLFIntegrator(VectorCoefficient &QF,
                                                   const IntegrationRule *ir)
   : DeltaLFIntegrator(QF, ir), Q(QF)
{
   static DomainLFIntegrator::Kernels kernels;
}

void VectorDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   real_t val,cf;

   shape.SetSize(dof);       // vector of size dof

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      val = Tr.Weight();

      el.CalcPhysShape(Tr, shape);
      Q.Eval (Qvec, Tr, ip);

      for (int k = 0; k < vdim; k++)
      {
         cf = val * Qvec(k);

         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += ip.weight * cf * shape(s);
         }
      }
   }
}

void VectorDomainLFIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(vec_delta != NULL, "coefficient must be VectorDeltaCoefficient");
   int vdim = Q.GetVDim();
   int dof  = fe.GetDof();

   shape.SetSize(dof);
   fe.CalcPhysShape(Trans, shape);

   vec_delta->EvalDelta(Qvec, Trans, Trans.GetIntPoint());

   elvect.SetSize(dof*vdim);
   DenseMatrix elvec_as_mat(elvect.GetData(), dof, vdim);
   MultVWt(shape, Qvec, elvec_as_mat);
}

void VectorDomainLFGradIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   const int dim = el.GetDim();
   const int dof = el.GetDof();
   const int vdim = Q.GetVDim();
   const int sdim = Tr.GetSpaceDim();

   dshape.SetSize(dof,sdim);

   elvect.SetSize(dof*(vdim/sdim));
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
      Qvec *= ip.weight * Tr.Weight();

      for (int k = 0; k < vdim/sdim; k++)
      {
         for (int d=0; d < sdim; ++d) { part_x(d) = Qvec(k*sdim+d); }
         dshape.Mult(part_x, pelvect);
         for (int s = 0; s < dof; ++s) { elvect(s+k*dof) += pelvect(s); }
      }
   }
}

void VectorDomainLFGradIntegrator::AssembleDeltaElementVect(
   const FiniteElement&, ElementTransformation&, Vector&)
{
   MFEM_ABORT("Not implemented!");
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      Q.Eval(vec, Tr, ip);
      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);
      for (int k = 0; k < vdim; k++)
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
   }
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      // Use Tr transformation in case Q depends on boundary attribute
      Q.Eval(vec, Tr, ip);
      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(eip, shape);
      for (int k = 0; k < vdim; k++)
      {
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
      }
   }
}

void VectorFEDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();
   int vdim = std::max(spaceDim, el.GetRangeDim());

   vshape.SetSize(dof,vdim);
   vec.SetSize(vdim);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   if (ir == NULL)
   {
      // int intorder = 2*el.GetOrder() - 1; // ok for O(h^{k+1}) conv. in L2
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      el.CalcVShape(Tr, vshape);

      QF.Eval (vec, Tr, ip);
      vec *= ip.weight * Tr.Weight();
      vshape.AddMult (vec, elvect);
   }
}

void VectorFEDomainLFIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(vec_delta != NULL, "coefficient must be VectorDeltaCoefficient");
   int dof = fe.GetDof();
   int spaceDim = Trans.GetSpaceDim();

   vshape.SetSize(dof, spaceDim);
   fe.CalcPhysVShape(Trans, vshape);

   vec_delta->EvalDelta(vec, Trans, Trans.GetIntPoint());

   elvect.SetSize(dof);
   vshape.Mult(vec, elvect);
}

void VectorFEDomainLFCurlIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();
   int n=(spaceDim == 3)? spaceDim : 1;
   curlshape.SetSize(dof,n);
   vec.SetSize(n);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      el.CalcPhysCurlShape(Tr, curlshape);
      QF->Eval(vec, Tr, ip);

      vec *= ip.weight * Tr.Weight();
      curlshape.AddMult (vec, elvect);
   }
}

void VectorFEDomainLFCurlIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   int spaceDim = Trans.GetSpaceDim();
   MFEM_ASSERT(vec_delta != NULL,
               "coefficient must be VectorDeltaCoefficient");
   int dof = fe.GetDof();
   int n=(spaceDim == 3)? spaceDim : 1;
   vec.SetSize(n);
   curlshape.SetSize(dof, n);
   elvect.SetSize(dof);
   fe.CalcPhysCurlShape(Trans, curlshape);

   vec_delta->EvalDelta(vec, Trans, Trans.GetIntPoint());
   curlshape.Mult(vec, elvect);
}

void VectorFEDomainLFDivIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   divshape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      real_t val = Tr.Weight() * Q.Eval(Tr, ip);
      el.CalcPhysDivShape(Tr, divshape);

      add(elvect, ip.weight * val, divshape, elvect);
   }
}

void VectorFEDomainLFDivIntegrator::AssembleDeltaElementVect(
   const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysDivShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

void VectorBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();

   shape.SetSize (dof);
   nor.SetSize (dim);
   elvect.SetSize (dim*dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() + 1);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint (&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      el.CalcShape (ip, shape);
      nor *= Sign * ip.weight * F -> Eval (Tr, ip);
      for (int j = 0; j < dof; j++)
         for (int k = 0; k < dim; k++)
         {
            elvect(dof*k+j) += nor(k) * shape(j);
         }
   }
}


void VectorFEBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);

      real_t val = ip.weight;
      if (F)
      {
         Tr.SetIntPoint (&ip);
         val *= F->Eval(Tr, ip);
      }

      elvect.Add(val, shape);
   }
}

void VectorFEBoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector nor(dim), Fvec(dim);

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder() + Tr.OrderW();  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      F.Eval(Fvec, Tr, ip);
      real_t val = ip.weight * (Fvec*nor) / Tr.Weight();

      el.CalcShape(ip, shape);

      elvect.Add(val, shape);
   }
}

void VectorFEBoundaryTangentLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int dim = el.GetDim();
   int vdim = el.GetRangeDim();
   DenseMatrix vshape(dof, vdim);
   Vector f_loc(3);
   Vector f_hat(2);

   MFEM_VERIFY(vdim == 2, "VectorFEBoundaryTangentLFIntegrator "
               "must be called with vector basis functions of dimension 2.");

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      el.CalcVShape(ip, vshape);

      Tr.SetIntPoint(&ip);
      f.Eval(f_loc, Tr, ip);

      if (dim == 2)
      {
         Tr.Jacobian().MultTranspose(f_loc, f_hat);
      }
      else if (dim == 1)
      {
         const DenseMatrix & J = Tr.Jacobian();
         f_hat(0) = J(0,0) * f_loc(0) + J(1,0) * f_loc(1);
         f_hat(1) = f_loc(2);
      }
      else
      {
         f_hat(0) = f_loc(1);
         f_hat(1) = f_loc(2);
      }

      Swap<real_t>(f_hat(0), f_hat(1));
      f_hat(0) = -f_hat(0);
      f_hat *= ip.weight;

      vshape.AddMult(f_hat, elvect);
   }
}

void BoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("BoundaryFlowIntegrator::AssembleRHSElementVect\n"
              "  is not implemented as boundary integrator!\n"
              "  Use LinearForm::AddBdrFaceIntegrator instead of\n"
              "  LinearForm::AddBoundaryIntegrator.");
}

void BoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof, order;
   real_t un, w, vu_data[3], nor_data[3];

   dim  = el.GetDim();
   ndof = el.GetDof();
   Vector vu(vu_data, dim), nor(nor_data, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Assuming order(u)==order(mesh)
      order = Tr.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   shape.SetSize(ndof);
   elvect.SetSize(ndof);
   elvect = 0.0;

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
      el.CalcShape(eip, shape);

      // Use Tr.Elem1 transformation for u so that it matches the coefficient
      // used with the ConvectionIntegrator and/or the DGTraceIntegrator.
      u->Eval(vu, *Tr.Elem1, eip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      un = vu * nor;
      w = 0.5*alpha*un - beta*fabs(un);
      w *= ip.weight*f->Eval(Tr, ip);
      elvect.Add(w, shape);
   }
}

void DGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
}

void DGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof;
   bool kappa_is_nonzero = (kappa != 0.);
   real_t w;

   dim = el.GetDim();
   ndof = el.GetDof();

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

   elvect.SetSize(ndof);
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
      w = ip.weight * uD->Eval(Tr, ip) / Tr.Elem1->Weight();
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
      elvect.Add(sigma, dshape_dn);

      if (kappa_is_nonzero)
      {
         elvect.Add(kappa*(ni*nor), shape);
      }
   }
}

void DGElasticityDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGElasticityDirichletLFIntegrator::AssembleRHSElementVect");
}

void DGElasticityDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   MFEM_ASSERT(Tr.Elem2No < 0, "interior boundary is not supported");

#ifdef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix adjJ;
   DenseMatrix dshape_ps;
   Vector nor;
   Vector dshape_dn;
   Vector dshape_du;
   Vector u_dir;
#endif

   const int dim = el.GetDim();
   const int ndofs = el.GetDof();
   const int nvdofs = dim*ndofs;

   elvect.SetSize(nvdofs);
   elvect = 0.0;

   adjJ.SetSize(dim);
   shape.SetSize(ndofs);
   dshape.SetSize(ndofs, dim);
   dshape_ps.SetSize(ndofs, dim);
   nor.SetSize(dim);
   dshape_dn.SetSize(ndofs);
   dshape_du.SetSize(ndofs);
   u_dir.SetSize(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*el.GetOrder(); // <-----
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int pi = 0; pi < ir->GetNPoints(); ++pi)
   {
      const IntegrationPoint &ip = ir->IntPoint(pi);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      // Evaluate the Dirichlet b.c. using the face transformation.
      uD.Eval(u_dir, Tr, ip);

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      Mult(dshape, adjJ, dshape_ps);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      real_t wL, wM, jcoef;
      {
         const real_t w = ip.weight / Tr.Elem1->Weight();
         wL = w * lambda->Eval(*Tr.Elem1, eip);
         wM = w * mu->Eval(*Tr.Elem1, eip);
         jcoef = kappa * (wL + 2.0*wM) * (nor*nor);
         dshape_ps.Mult(nor, dshape_dn);
         dshape_ps.Mult(u_dir, dshape_du);
      }

      // alpha < uD, (lambda div(v) I + mu (grad(v) + grad(v)^T)) . n > +
      //   + kappa < h^{-1} (lambda + 2 mu) uD, v >

      // i = idof + ndofs * im
      // v_phi(i,d) = delta(im,d) phi(idof)
      // div(v_phi(i)) = dphi(idof,im)
      // (grad(v_phi(i)))(k,l) = delta(im,k) dphi(idof,l)
      //
      // term 1:
      //   alpha < uD, lambda div(v_phi(i)) n >
      //   alpha lambda div(v_phi(i)) (uD.n) =
      //   alpha lambda dphi(idof,im) (uD.n) --> quadrature -->
      //   ip.weight/det(J1) alpha lambda (uD.nor) dshape_ps(idof,im) =
      //   alpha * wL * (u_dir*nor) * dshape_ps(idof,im)
      // term 2:
      //   < alpha uD, mu grad(v_phi(i)).n > =
      //   alpha mu uD^T grad(v_phi(i)) n =
      //   alpha mu uD(k) delta(im,k) dphi(idof,l) n(l) =
      //   alpha mu uD(im) dphi(idof,l) n(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu uD(im) dshape_ps(idof,l) nor(l) =
      //   alpha * wM * u_dir(im) * dshape_dn(idof)
      // term 3:
      //   < alpha uD, mu (grad(v_phi(i)))^T n > =
      //   alpha mu n^T grad(v_phi(i)) uD =
      //   alpha mu n(k) delta(im,k) dphi(idof,l) uD(l) =
      //   alpha mu n(im) dphi(idof,l) uD(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu nor(im) dshape_ps(idof,l) uD(l) =
      //   alpha * wM * nor(im) * dshape_du(idof)
      // term j:
      //   < kappa h^{-1} (lambda + 2 mu) uD, v_phi(i) > =
      //   kappa/h (lambda + 2 mu) uD(k) v_phi(i,k) =
      //   kappa/h (lambda + 2 mu) uD(k) delta(im,k) phi(idof) =
      //   kappa/h (lambda + 2 mu) uD(im) phi(idof) --> quadrature -->
      //      [ 1/h = |nor|/det(J1) ]
      //   ip.weight/det(J1) |nor|^2 kappa (lambda + 2 mu) uD(im) phi(idof) =
      //   jcoef * u_dir(im) * shape(idof)

      wM *= alpha;
      const real_t t1 = alpha * wL * (u_dir*nor);
      for (int im = 0, i = 0; im < dim; ++im)
      {
         const real_t t2 = wM * u_dir(im);
         const real_t t3 = wM * nor(im);
         const real_t tj = jcoef * u_dir(im);
         for (int idof = 0; idof < ndofs; ++idof, ++i)
         {
            elvect(i) += (t1*dshape_ps(idof,im) + t2*dshape_dn(idof) +
                          t3*dshape_du(idof) + tj*shape(idof));
         }
      }
   }
}



void WhiteGaussianNoiseDomainLFIntegrator::AssembleRHSElementVect
(const FiniteElement &el,
 ElementTransformation &Tr,
 Vector &elvect)
{
   int n = el.GetDof();
   elvect.SetSize(n);
   for (int i = 0; i < n; i++)
   {
      elvect(i) = dist(generator);
   }

   int iel = Tr.ElementNo;

   if (!save_factors || !L[iel])
   {
      DenseMatrix *M, m;
      if (save_factors)
      {
         L[iel]=new DenseMatrix;
         M = L[iel];
      }
      else
      {
         M = &m;
      }
      massinteg.AssembleElementMatrix(el, Tr, *M);
      CholeskyFactors chol(M->Data());
      chol.Factor(M->Height());
      chol.LMult(n,1,elvect.GetData());
   }
   else
   {
      CholeskyFactors chol(L[iel]->Data());
      chol.LMult(n,1,elvect.GetData());
   }
}


void VectorQuadratureLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &fe, ElementTransformation &Tr, Vector &elvect)
{
   const IntegrationRule *ir =
      &vqfc.GetQuadFunction().GetSpace()->GetIntRule(Tr.ElementNo);

   const int nqp = ir->GetNPoints();
   const int vdim = vqfc.GetVDim();
   const int ndofs = fe.GetDof();
   Vector shape(ndofs);
   Vector temp(vdim);
   elvect.SetSize(vdim * ndofs);
   elvect = 0.0;
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      const real_t w = Tr.Weight() * ip.weight;
      vqfc.Eval(temp, Tr, ip);
      fe.CalcShape(ip, shape);
      for (int ind = 0; ind < vdim; ind++)
      {
         for (int nd = 0; nd < ndofs; nd++)
         {
            elvect(nd + ind * ndofs) += w * shape(nd) * temp(ind);
         }
      }
   }
}


void QuadratureLFIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                                    ElementTransformation &Tr,
                                                    Vector &elvect)
{
   const IntegrationRule *ir =
      &qfc.GetQuadFunction().GetSpace()->GetIntRule(Tr.ElementNo);

   const int nqp = ir->GetNPoints();
   const int ndofs = fe.GetDof();
   Vector shape(ndofs);
   elvect.SetSize(ndofs);
   elvect = 0.0;
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint (&ip);
      const real_t w = Tr.Weight() * ip.weight;
      real_t temp = qfc.Eval(Tr, ip);
      fe.CalcShape(ip, shape);
      shape *= (w * temp);
      elvect += shape;
   }
}

}
