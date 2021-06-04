// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include <cmath>

namespace mfem
{

ElementTransformation::ElementTransformation()
   : IntPoint(static_cast<IntegrationPoint *>(NULL)),
     EvalState(0),
     Attribute(-1),
     ElementNo(-1)
{ }

double ElementTransformation::EvalWeight()
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
   double minDist = std::numeric_limits<double>::max();

   // Check all integration points in ir
   const int npts = ir.GetNPoints();
   for (int i = 0; i < npts; ++i)
   {
      double dist = pt.DistanceTo(physPts.GetColumn(i));
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
   double minDist = std::numeric_limits<double>::max();

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
      double dist = dr.Norml2();
      // double dist = dr.Normlinf();
      if (dist < minDist)
      {
         minDist = dist;
         minIndex = i;
      }
   }
   return minIndex;
}

void InverseElementTransformation::NewtonPrint(int mode, double val)
{
   std::ostream &out = mfem::out;

   // separator:
   switch (mode%3)
   {
      case 0: out << ", "; break;
      case 1: out << "Newton: "; break;
      case 2: out << "                   "; break;
         //          "Newton: iter = xx, "
   }
   switch ((mode/3)%4)
   {
      case 0: out << "iter = " << std::setw(2) << int(val); break;
      case 1: out << "delta_ref = " << std::setw(11) << val; break;
      case 2: out << " err_phys = " << std::setw(11) << val; break;
      case 3: break;
   }
   // ending:
   switch ((mode/12)%4)
   {
      case 0: break;
      case 1: out << '\n'; break;
      case 2: out << " (converged)\n"; break;
      case 3: out << " (actual)\n"; break;
   }
}

void InverseElementTransformation::NewtonPrintPoint(const char *prefix,
                                                    const Vector &pt,
                                                    const char *suffix)
{
   std::ostream &out = mfem::out;

   out << prefix << " = (";
   for (int j = 0; j < pt.Size(); j++)
   {
      out << (j > 0 ? ", " : "") << pt(j);
   }
   out << ')' << suffix;
}

int InverseElementTransformation::NewtonSolve(const Vector &pt,
                                              IntegrationPoint &ip)
{
   MFEM_ASSERT(pt.Size() == T->GetSpaceDim(), "invalid point");

   const double phys_tol = phys_rtol*pt.Normlinf();

   const int geom = T->GetGeometryType();
   const int dim = T->GetDimension();
   const int sdim = T->GetSpaceDim();
   IntegrationPoint xip, prev_xip;
   double xd[3], yd[3], dxd[3], dx_norm = -1.0, err_phys, real_dx_norm = -1.0;
   Vector x(xd, dim), y(yd, sdim), dx(dxd, dim);
   bool hit_bdr = false, prev_hit_bdr = false;

   // Use ip0 as initial guess:
   xip = *ip0;
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
            NewtonPrint(1, (double)it);
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
            NewtonPrint(1, (double)it);
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
                     NewtonPrint(1, (double)it);
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
         NewtonPrint(1, double(it));
         xip.Get(xd, dim); // xip -> x
         NewtonPrintPoint(",    ref_pt", x, "\n");
      }

      // Check for convergence in reference coordinates:
      dx_norm = dx.Normlinf();
      if (dx_norm < ref_tol)
      {
         if (print_level >= 1)
         {
            NewtonPrint(1, (double)it);
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
         NewtonPrint(1, (double)max_iter);
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
         ip0 = &Geometries.GetCenter(T->GetGeometryType());
         break;

      case ClosestPhysNode:
      case ClosestRefNode:
      {
         const int order = std::max(T->Order()+rel_qpts_order, 0);
         if (order == 0)
         {
            ip0 = &Geometries.GetCenter(T->GetGeometryType());
         }
         else
         {
            const int old_type = GlobGeometryRefiner.GetType();
            GlobGeometryRefiner.SetType(qpts_type);
            RefinedGeometry &RefG =
               *GlobGeometryRefiner.Refine(T->GetGeometryType(), order);
            int closest_idx = (init_guess_type == ClosestPhysNode) ?
                              FindClosestPhysPoint(pt, RefG.RefPts) :
                              FindClosestRefPoint(pt, RefG.RefPts);
            ip0 = &RefG.RefPts.IntPoint(closest_idx);
            GlobGeometryRefiner.SetType(old_type);
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
   space_dim = dim;
}

const DenseMatrix &IsoparametricTransformation::EvalJacobian()
{
   MFEM_ASSERT(space_dim == PointMat.Height(),
               "the IsoparametricTransformation has not been finalized;"
               " call FinilizeTransformation() after setup");
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
   MFEM_ASSERT(space_dim == PointMat.Height(),
               "the IsoparametricTransformation has not been finalized;"
               " call FinilizeTransformation() after setup");
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

int IsoparametricTransformation::OrderJ()
{
   switch (FElem->Space())
   {
      case FunctionSpace::Pk:
         return (FElem->GetOrder()-1);
      case FunctionSpace::Qk:
         return (FElem->GetOrder());
      default:
         mfem_error("IsoparametricTransformation::OrderJ()");
   }
   return 0;
}

int IsoparametricTransformation::OrderW()
{
   switch (FElem->Space())
   {
      case FunctionSpace::Pk:
         return (FElem->GetOrder() - 1) * FElem->GetDim();
      case FunctionSpace::Qk:
         return (FElem->GetOrder() * FElem->GetDim() - 1);
      default:
         mfem_error("IsoparametricTransformation::OrderW()");
   }
   return 0;
}

int IsoparametricTransformation::OrderGrad(const FiniteElement *fe)
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
      }
   }
   mfem_error("IsoparametricTransformation::OrderGrad(...)");
   return 0;
}

void IsoparametricTransformation::Transform (const IntegrationPoint &ip,
                                             Vector &trans)
{
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
   double vec[3];
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


////////////////////////////// ADDED /////////////////////////////////
int IsoparametricTransformation::TransformBack(const Vector &pt,
                                               IntegrationPoint &ip,
                                               IntegrationPoint &xip)
{
   const int    max_iter = 32;
   const double  ref_tol = 1e-12;
   const double phys_tol = 1e-12*pt.Normlinf();

   const int dim = FElem->GetDim();
   const int sdim = PointMat.Height();
   const int geom = FElem->GetGeomType();
  // IntegrationPoint xip, prev_xip;
   IntegrationPoint prev_xip;
   double xd[3], yd[3], dxd[3], Jid[9];
   Vector x(xd, dim), y(yd, sdim), dx(dxd, dim);
   DenseMatrix Jinv(Jid, dim, sdim);
   bool hit_bdr = false, prev_hit_bdr;

   // Use the center of the element as initial guess
 //  xip = Geometries.GetCenter(geom);
 //  xip.Get(xd, dim); // xip -> x

   for (int it = 0; it < max_iter; it++)
   {
      // Newton iteration:    x := x + J(x)^{-1} [pt-F(x)]
      // or when dim != sdim: x := x + [J^t.J]^{-1}.J^t [pt-F(x)]
      Transform(xip, y);
      subtract(pt, y, y); // y = pt-y
      if (y.Normlinf() < phys_tol) { ip = xip; return 0; }
      SetIntPoint(&xip);
      CalcInverse(Jacobian(), Jinv);
      Jinv.Mult(y, dx);
      x += dx;
      prev_xip = xip;
      prev_hit_bdr = hit_bdr;
      xip.Set(xd, dim); // x -> xip
      // If xip is ouside project it on the boundary on the line segment
      // between prev_xip and xip
      hit_bdr = !Geometry::ProjectPoint(geom, prev_xip, xip);
      if (dx.Normlinf() < ref_tol) { ip = xip; return 0; }
      if (hit_bdr)
      {
         xip.Get(xd, dim); // xip -> x
         if (prev_hit_bdr)
         {
            prev_xip.Get(dxd, dim); // prev_xip -> dx
            subtract(x, dx, dx);    // dx = xip - prev_xip
            if (dx.Normlinf() < ref_tol) { return 1; }
         }
      }
   }
   ip = xip;
   return 2;
}
////////////////////////////// ADDED /////////////////////////////////

}
