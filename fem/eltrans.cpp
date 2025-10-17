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

#include "../mesh/mesh_headers.hpp"
#include "fem.hpp"
#include "eltrans/eltrans_basis.hpp"

#include <cmath>

namespace mfem
{

ElementTransformation::ElementTransformation()
   : IntPoint(static_cast<IntegrationPoint *>(NULL)),
     EvalState(0),
     geom(Geometry::INVALID),
     Attribute(-1),
     ElementNo(-1),
     mesh(nullptr)
{ }

real_t ElementTransformation::EvalWeight()
{
   MFEM_ASSERT((EvalState & WEIGHT_MASK) == 0, "");
   Jacobian();
   EvalState |= WEIGHT_MASK;
   return (Wght = (dFdx.Width() == 0) ? 1.0 : dFdx.Weight());
}

const DenseMatrix &ElementTransformation::EvalAdjugateJ()
{
   MFEM_ASSERT((EvalState & ADJUGATE_MASK) == 0, "");
   Jacobian();
   adjJ.SetSize(dFdx.Width(), dFdx.Height());
   if (dFdx.Width() > 0) { CalcAdjugate(dFdx, adjJ); }
   EvalState |= ADJUGATE_MASK;
   return adjJ;
}

const DenseMatrix &ElementTransformation::EvalTransAdjugateJ()
{
   MFEM_ASSERT((EvalState & TRANS_ADJUGATE_MASK) == 0, "");
   Jacobian();
   adjJT.SetSize(dFdx.Height(), dFdx.Width());
   if (dFdx.Width() == dFdx.Height()) { CalcAdjugateTranspose(dFdx, adjJT); }
   else { AdjugateJacobian(); adjJT.Transpose(adjJ); }
   EvalState |= TRANS_ADJUGATE_MASK;
   return adjJT;
}

const DenseMatrix &ElementTransformation::EvalInverseJ()
{
   // TODO: compute as invJ = / adjJ/Weight,    if J is square,
   //                         \ adjJ/Weight^2,  otherwise.
   MFEM_ASSERT((EvalState & INVERSE_MASK) == 0, "");
   Jacobian();
   invJ.SetSize(dFdx.Width(), dFdx.Height());
   if (dFdx.Width() > 0) { CalcInverse(dFdx, invJ); }
   EvalState |= INVERSE_MASK;
   return invJ;
}

int InverseElementTransformation::FindClosestPhysPoint(
   const Vector& pt, const IntegrationRule &ir)
{
   MFEM_VERIFY(T != NULL, "invalid ElementTransformation");
   MFEM_VERIFY(pt.Size() == T->GetSpaceDim(), "invalid point");

   DenseMatrix physPts;
   T->Transform(ir, physPts);

   // Initialize distance and index of closest point
   int minIndex = -1;
   real_t minDist = std::numeric_limits<real_t>::max();

   // Check all integration points in ir
   const int npts = ir.GetNPoints();
   for (int i = 0; i < npts; ++i)
   {
      real_t dist = pt.DistanceTo(physPts.GetColumn(i));
      if (dist < minDist)
      {
         minDist = dist;
         minIndex = i;
      }
   }
   return minIndex;
}

int InverseElementTransformation::FindClosestRefPoint(
   const Vector& pt, const IntegrationRule &ir)
{
   MFEM_VERIFY(T != NULL, "invalid ElementTransformation");
   MFEM_VERIFY(pt.Size() == T->GetSpaceDim(), "invalid point");

   // Initialize distance and index of closest point
   int minIndex = -1;
   real_t minDist = std::numeric_limits<real_t>::max();

   // Check all integration points in ir using the local metric at each point
   // induced by the transformation.
   Vector dp(T->GetSpaceDim()), dr(T->GetDimension());
   const int npts = ir.GetNPoints();
   for (int i = 0; i < npts; ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      T->Transform(ip, dp);
      dp -= pt;
      T->SetIntPoint(&ip);
      T->InverseJacobian().Mult(dp, dr);
      real_t dist = dr.Norml2();
      // double dist = dr.Normlinf();
      if (dist < minDist)
      {
         minDist = dist;
         minIndex = i;
      }
   }
   return minIndex;
}

void InverseElementTransformation::NewtonPrint(int mode, real_t val)
{
   std::ostream &os = mfem::out;

   // separator:
   switch (mode%3)
   {
      case 0: os << ", "; break;
      case 1: os << "Newton: "; break;
      case 2: os << "                   "; break;
         //          "Newton: iter = xx, "
   }
   switch ((mode/3)%4)
   {
      case 0: os << "iter = " << std::setw(2) << int(val); break;
      case 1: os << "delta_ref = " << std::setw(11) << val; break;
      case 2: os << " err_phys = " << std::setw(11) << val; break;
      case 3: break;
   }
   // ending:
   switch ((mode/12)%4)
   {
      case 0: break;
      case 1: os << '\n'; break;
      case 2: os << " (converged)\n"; break;
      case 3: os << " (actual)\n"; break;
   }
}

void InverseElementTransformation::NewtonPrintPoint(const char *prefix,
                                                    const Vector &pt,
                                                    const char *suffix)
{
   std::ostream &os = mfem::out;

   os << prefix << " = (";
   for (int j = 0; j < pt.Size(); j++)
   {
      os << (j > 0 ? ", " : "") << pt(j);
   }
   os << ')' << suffix;
}

int InverseElementTransformation::NewtonSolve(const Vector &pt,
                                              IntegrationPoint &ip)
{
   MFEM_ASSERT(pt.Size() == T->GetSpaceDim(), "invalid point");

   const real_t phys_tol = phys_rtol*pt.Normlinf();

   const int geom = T->GetGeometryType();
   const int dim = T->GetDimension();
   const int sdim = T->GetSpaceDim();
   IntegrationPoint xip, prev_xip;
   real_t xd[3], yd[3], dxd[3], dxpd[3], dx_norm = -1.0, err_phys,
                                         real_dx_norm = -1.0;
   Vector x(xd, dim), y(yd, sdim), dx(dxd, dim), dx_prev(dxpd, dim);
   bool hit_bdr = false, prev_hit_bdr = false;

   // Use ip0 as initial guess:
   xip = ip0;
   xip.Get(xd, dim); // xip -> x
   if (print_level >= 3)
   {
      NewtonPrint(1, 0.); // iter 0
      NewtonPrintPoint(",    ref_pt", x, "\n");
   }

   for (int it = 0; true; )
   {
      // Remarks:
      // If f(x) := 1/2 |pt-F(x)|^2, then grad(f)(x) = -J^t(x) [pt-F(x)].
      // Linearize F(y) at y=x: F(y) ~ L[x](y) := F(x) + J(x) [y-x].
      // Newton iteration for F(y)=b is given by L[x_old](x_new) = b, i.e.
      // F(x_old) + J(x_old) [x_new-x_old] = b.
      //
      // To minimize: 1/2 |F(y)-b|^2, subject to: l(y) >= 0, we may consider the
      // iteration: minimize: |L[x_old](x_new)-b|^2, subject to l(x_new) >= 0,
      // i.e. minimize: |F(x_old) + J(x_old) [x_new-x_old] - b|^2.

      // This method uses:
      // Newton iteration:    x := x + J(x)^{-1} [pt-F(x)]
      // or when dim != sdim: x := x + [J^t.J]^{-1}.J^t [pt-F(x)]

      // Compute the physical coordinates of the current point:
      T->Transform(xip, y);
      if (print_level >= 3)
      {
         NewtonPrint(11, 0.); // continuation line
         NewtonPrintPoint("approx_pt", y, ", ");
         NewtonPrintPoint("exact_pt", pt, "\n");
      }
      subtract(pt, y, y); // y = pt-y

      // Check for convergence in physical coordinates:
      err_phys = y.Normlinf();
      if (err_phys < phys_tol)
      {
         if (print_level >= 1)
         {
            NewtonPrint(1, (real_t)it);
            NewtonPrint(3, dx_norm);
            NewtonPrint(30, err_phys);
         }
         ip = xip;
         if (solver_type != Newton) { return Inside; }
         return Geometry::CheckPoint(geom, ip, ip_tol) ? Inside : Outside;
      }
      if (print_level >= 1)
      {
         if (it == 0 || print_level >= 2)
         {
            NewtonPrint(1, (real_t)it);
            NewtonPrint(3, dx_norm);
            NewtonPrint(18, err_phys);
         }
      }

      if (hit_bdr)
      {
         xip.Get(xd, dim); // xip -> x
         if (prev_hit_bdr || it == max_iter || print_level >= 2)
         {
            prev_xip.Get(dxd, dim); // prev_xip -> dx
            subtract(x, dx, dx);    // dx = xip - prev_xip
            real_dx_norm = dx.Normlinf();
            if (print_level >= 2)
            {
               NewtonPrint(41, real_dx_norm);
            }
            if (prev_hit_bdr && real_dx_norm < ref_tol)
            {
               if (print_level >= 0)
               {
                  if (print_level <= 1)
                  {
                     NewtonPrint(1, (real_t)it);
                     NewtonPrint(3, dx_norm);
                     NewtonPrint(18, err_phys);
                     NewtonPrint(41, real_dx_norm);
                  }
                  mfem::out << "Newton: *** stuck on boundary!\n";
               }
               return Outside;
            }
         }
      }

      if (it == max_iter) { break; }

      // Perform a Newton step:
      T->SetIntPoint(&xip);
      T->InverseJacobian().Mult(y, dx);
      x += dx;
      it++;
      if (solver_type != Newton)
      {
         prev_xip = xip;
         prev_hit_bdr = hit_bdr;
      }
      xip.Set(xd, dim); // x -> xip

      // Perform projection based on solver_type:
      switch (solver_type)
      {
         case Newton: break;
         case NewtonSegmentProject:
            hit_bdr = !Geometry::ProjectPoint(geom, prev_xip, xip); break;
         case NewtonElementProject:
            hit_bdr = !Geometry::ProjectPoint(geom, xip); break;
         default: MFEM_ABORT("invalid solver type");
      }
      if (print_level >= 3)
      {
         NewtonPrint(1, real_t(it));
         xip.Get(xd, dim); // xip -> x
         NewtonPrintPoint(",    ref_pt", x, "\n");
      }

      // Check for convergence in reference coordinates:
      dx_norm = dx.Normlinf();
      if (dx_norm < ref_tol)
      {
         if (print_level >= 1)
         {
            NewtonPrint(1, (real_t)it);
            NewtonPrint(27, dx_norm);
         }
         ip = xip;
         if (solver_type != Newton) { return Inside; }
         return Geometry::CheckPoint(geom, ip, ip_tol) ? Inside : Outside;
      }
   }
   if (print_level >= 0)
   {
      if (print_level <= 1)
      {
         NewtonPrint(1, (real_t)max_iter);
         NewtonPrint(3, dx_norm);
         NewtonPrint(18, err_phys);
         if (hit_bdr) { NewtonPrint(41, real_dx_norm); }
      }
      mfem::out << "Newton: *** iteration did not converge!\n";
   }
   ip = xip;
   return Unknown;
}

int InverseElementTransformation::Transform(const Vector &pt,
                                            IntegrationPoint &ip)
{
   MFEM_VERIFY(T != NULL, "invalid ElementTransformation");

   // Select initial guess ...
   switch (init_guess_type)
   {
      case Center:
         ip0 = Geometries.GetCenter(T->GetGeometryType());
         break;

      case ClosestPhysNode:
      case ClosestRefNode:
      {
         const int order = qpts_order >= 0
                           ? qpts_order
                           : std::max(T->Order() + rel_qpts_order, 0);
         if (order == 0)
         {
            ip0 = Geometries.GetCenter(T->GetGeometryType());
         }
         else
         {
            RefinedGeometry &RefG = *refiner.Refine(T->GetGeometryType(), order);
            int closest_idx = (init_guess_type == ClosestPhysNode) ?
                              FindClosestPhysPoint(pt, RefG.RefPts) :
                              FindClosestRefPoint(pt, RefG.RefPts);
            ip0 = RefG.RefPts.IntPoint(closest_idx);
         }
         break;
      }
      case EdgeScan:
      {
         const int order = qpts_order >= 0
                           ? qpts_order
                           : std::max(T->Order() + rel_qpts_order, 0);
         if (order == 0)
         {
            ip0 = Geometries.GetCenter(T->GetGeometryType());
         }
         else
         {
            auto &ir = *refiner.EdgeScan(T->GetGeometryType(), order + 1);
            int res = Outside;
            int npts = ir.GetNPoints();
            // will return Inside if any test point reports Inside, Outside if
            // all points report Outside, else Unknown
            for (int i = 0; i < npts; ++i)
            {
               ip0 = ir.IntPoint(i);
               int tmp_res = NewtonSolve(pt, ip);
               switch (tmp_res)
               {
                  case Inside:
                     return Inside;
                  case Outside:
                     break;
                  case Unknown:
                     res = Unknown;
                     break;
               }
            }
            return res;
         }
         break;
      }
      case GivenPoint:
         break;

      default:
         MFEM_ABORT("invalid initial guess type");
   }

   // Call the solver ...
   return NewtonSolve(pt, ip);
}


void IsoparametricTransformation::SetIdentityTransformation(
   Geometry::Type GeomType)
{
   switch (GeomType)
   {
      case Geometry::POINT :       FElem = &PointFE; break;
      case Geometry::SEGMENT :     FElem = &SegmentFE; break;
      case Geometry::TRIANGLE :    FElem = &TriangleFE; break;
      case Geometry::SQUARE :      FElem = &QuadrilateralFE; break;
      case Geometry::TETRAHEDRON : FElem = &TetrahedronFE; break;
      case Geometry::CUBE :        FElem = &HexahedronFE; break;
      case Geometry::PRISM :       FElem = &WedgeFE; break;
      case Geometry::PYRAMID :     FElem = &PyramidFE; break;
      default:
         MFEM_ABORT("unknown Geometry::Type!");
   }
   int dim = FElem->GetDim();
   int dof = FElem->GetDof();
   const IntegrationRule &nodes = FElem->GetNodes();
   PointMat.SetSize(dim, dof);
   for (int j = 0; j < dof; j++)
   {
      nodes.IntPoint(j).Get(&PointMat(0,j), dim);
   }
   geom = GeomType;
}

const DenseMatrix &IsoparametricTransformation::EvalJacobian()
{
   MFEM_ASSERT((EvalState & JACOBIAN_MASK) == 0, "");

   dshape.SetSize(FElem->GetDof(), FElem->GetDim());
   dFdx.SetSize(PointMat.Height(), dshape.Width());
   if (dshape.Width() > 0)
   {
      FElem->CalcDShape(*IntPoint, dshape);
      Mult(PointMat, dshape, dFdx);
   }
   EvalState |= JACOBIAN_MASK;

   return dFdx;
}

const DenseMatrix &IsoparametricTransformation::EvalHessian()
{
   MFEM_ASSERT((EvalState & HESSIAN_MASK) == 0, "");

   int Dim = FElem->GetDim();
   d2shape.SetSize(FElem->GetDof(), (Dim*(Dim+1))/2);
   d2Fdx2.SetSize(PointMat.Height(), d2shape.Width());
   if (d2shape.Width() > 0)
   {
      FElem->CalcHessian(*IntPoint, d2shape);
      Mult(PointMat, d2shape, d2Fdx2);
   }
   EvalState |= HESSIAN_MASK;

   return d2Fdx2;
}

int IsoparametricTransformation::OrderJ() const
{
   switch (FElem->Space())
   {
      case FunctionSpace::Pk:
         return (FElem->GetOrder()-1);
      case FunctionSpace::Qk:
         return (FElem->GetOrder());
      case FunctionSpace::Uk:
         return (FElem->GetOrder());
      default:
         MFEM_ABORT("unsupported finite element");
   }
   return 0;
}

int IsoparametricTransformation::OrderW() const
{
   switch (FElem->Space())
   {
      case FunctionSpace::Pk:
         return (FElem->GetOrder() - 1) * FElem->GetDim();
      case FunctionSpace::Qk:
         return (FElem->GetOrder() * FElem->GetDim() - 1);
      case FunctionSpace::Uk:
         return (FElem->GetOrder() * FElem->GetDim() - 1);
      default:
         MFEM_ABORT("unsupported finite element");
   }
   return 0;
}

int IsoparametricTransformation::OrderGrad(const FiniteElement *fe) const
{
   if (FElem->Space() == fe->Space())
   {
      int k = FElem->GetOrder();
      int d = FElem->GetDim();
      int l = fe->GetOrder();
      switch (fe->Space())
      {
         case FunctionSpace::Pk:
            return ((k-1)*(d-1)+(l-1));
         case FunctionSpace::Qk:
            return (k*(d-1)+(l-1));
         case FunctionSpace::Uk:
            return (k*(d-1)+(l-1));
         default:
            MFEM_ABORT("unsupported finite element");
      }
   }
   MFEM_ABORT("incompatible finite elements");
   return 0;
}

void IsoparametricTransformation::Transform (const IntegrationPoint &ip,
                                             Vector &trans)
{
   MFEM_ASSERT(FElem != nullptr, "Must provide a valid FiniteElement object!");
   shape.SetSize(FElem->GetDof());
   trans.SetSize(PointMat.Height());

   FElem -> CalcShape(ip, shape);
   PointMat.Mult(shape, trans);
}

void IsoparametricTransformation::Transform (const IntegrationRule &ir,
                                             DenseMatrix &tr)
{
   int dof, n, dim, i, j, k;

   dim = PointMat.Height();
   dof = FElem->GetDof();
   n = ir.GetNPoints();

   shape.SetSize(dof);
   tr.SetSize(dim, n);

   for (j = 0; j < n; j++)
   {
      FElem -> CalcShape (ir.IntPoint(j), shape);
      for (i = 0; i < dim; i++)
      {
         tr(i, j) = 0.0;
         for (k = 0; k < dof; k++)
         {
            tr(i, j) += PointMat(i, k) * shape(k);
         }
      }
   }
}

void IsoparametricTransformation::Transform (const DenseMatrix &matrix,
                                             DenseMatrix &result)
{
   MFEM_ASSERT(matrix.Height() == GetDimension(), "invalid input");
   result.SetSize(PointMat.Height(), matrix.Width());

   IntegrationPoint ip;
   Vector col;

   for (int j = 0; j < matrix.Width(); j++)
   {
      ip.Set(matrix.GetColumn(j), matrix.Height());

      result.GetColumnReference(j, col);
      Transform(ip, col);
   }
}

void IntegrationPointTransformation::Transform (const IntegrationPoint &ip1,
                                                IntegrationPoint &ip2)
{
   real_t vec[3];
   Vector v (vec, Transf.GetPointMat().Height());

   Transf.Transform (ip1, v);
   ip2.Set(vec, v.Size());
}

void IntegrationPointTransformation::Transform (const IntegrationRule &ir1,
                                                IntegrationRule &ir2)
{
   int i, n;

   n = ir1.GetNPoints();
   for (i = 0; i < n; i++)
   {
      Transform (ir1.IntPoint(i), ir2.IntPoint(i));
   }
}

void FaceElementTransformations::SetIntPoint(const IntegrationPoint *face_ip)
{
   IsoparametricTransformation::SetIntPoint(face_ip);

   if (mask & 4)
   {
      Loc1.Transform(*face_ip, eip1);
      if (Elem1)
      {
         Elem1->SetIntPoint(&eip1);
      }
   }
   if (mask & 8)
   {
      Loc2.Transform(*face_ip, eip2);
      if (Elem2)
      {
         Elem2->SetIntPoint(&eip2);
      }
   }
}

ElementTransformation &
FaceElementTransformations::GetElement1Transformation()
{
   MFEM_VERIFY(mask & HAVE_ELEM1 && Elem1 != NULL, "The ElementTransformation "
               "for the element has not been configured for side 1.");
   return *Elem1;
}

ElementTransformation &
FaceElementTransformations::GetElement2Transformation()
{
   MFEM_VERIFY(mask & HAVE_ELEM2 && Elem2 != NULL, "The ElementTransformation "
               "for the element has not been configured for side 2.");
   return *Elem2;
}

IntegrationPointTransformation &
FaceElementTransformations::GetIntPoint1Transformation()
{
   MFEM_VERIFY(mask & HAVE_LOC1, "The IntegrationPointTransformation "
               "for the element has not been configured for side 1.");
   return Loc1;
}

IntegrationPointTransformation &
FaceElementTransformations::GetIntPoint2Transformation()
{
   MFEM_VERIFY(mask & HAVE_LOC2, "The IntegrationPointTransformation "
               "for the element has not been configured for side 2.");
   return Loc2;
}

void FaceElementTransformations::Transform(const IntegrationPoint &ip,
                                           Vector &trans)
{
   MFEM_VERIFY(mask & HAVE_FACE, "The ElementTransformation "
               "for the face has not been configured.");
   IsoparametricTransformation::Transform(ip, trans);
}

void FaceElementTransformations::Transform(const IntegrationRule &ir,
                                           DenseMatrix &tr)
{
   MFEM_VERIFY(mask & HAVE_FACE, "The ElementTransformation "
               "for the face has not been configured.");
   IsoparametricTransformation::Transform(ir, tr);
}

void FaceElementTransformations::Transform(const DenseMatrix &matrix,
                                           DenseMatrix &result)
{
   MFEM_VERIFY(mask & HAVE_FACE, "The ElementTransformation "
               "for the face has not been configured.");
   IsoparametricTransformation::Transform(matrix, result);
}

real_t FaceElementTransformations::CheckConsistency(int print_level,
                                                    std::ostream &os)
{
   // Check that the face vertices are mapped to the same physical location
   // when using the following three transformations:
   // - the face transformation, *this
   // - Loc1 + Elem1
   // - Loc2 + Elem2, if present.

   const bool have_face = (mask & 16);
   const bool have_el1 = (mask & 1) && (mask & 4);
   const bool have_el2 = (mask & 2) && (mask & 8) && (Elem2No >= 0);
   if (int(have_face) + int(have_el1) + int(have_el2) < 2)
   {
      // need at least two different transformations to perform a check
      return 0.0;
   }

   const IntegrationRule &v_ir = *Geometries.GetVertices(GetGeometryType());

   real_t max_dist = 0.0;
   Vector dist(v_ir.GetNPoints());
   DenseMatrix coords_base, coords_el;
   IntegrationRule v_eir(v_ir.GetNPoints());
   if (have_face)
   {
      Transform(v_ir, coords_base);
      if (print_level > 0)
      {
         os << "\nface vertex coordinates (from face transform):\n"
            << "----------------------------------------------\n";
         coords_base.PrintT(os, coords_base.Height());
      }
   }
   if (have_el1)
   {
      Loc1.Transform(v_ir, v_eir);
      Elem1->Transform(v_eir, coords_el);
      if (print_level > 0)
      {
         os << "\nface vertex coordinates (from element 1 transform):\n"
            << "---------------------------------------------------\n";
         coords_el.PrintT(os, coords_el.Height());
      }
      if (have_face)
      {
         coords_el -= coords_base;
         coords_el.Norm2(dist);
         max_dist = std::max(max_dist, dist.Normlinf());
      }
      else
      {
         coords_base = coords_el;
      }
   }
   if (have_el2)
   {
      Loc2.Transform(v_ir, v_eir);
      Elem2->Transform(v_eir, coords_el);
      if (print_level > 0)
      {
         os << "\nface vertex coordinates (from element 2 transform):\n"
            << "---------------------------------------------------\n";
         coords_el.PrintT(os, coords_el.Height());
      }
      coords_el -= coords_base;
      coords_el.Norm2(dist);
      max_dist = std::max(max_dist, dist.Normlinf());
   }

   return max_dist;
}
}
