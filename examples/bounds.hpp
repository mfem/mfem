// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// MFEM Mesh Optimizer Miniapp - Serial/Parallel Shared Code

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

// Type = 0 = Mesh nodes
//        1 = GLL points
//        2 = Closed Uniform
//        3 = Gauss-Legendre
double GetMeshMinDet(Mesh *mesh, int order, int type = 0)
{
   GridFunction *nodes = mesh->GetNodes();
   FiniteElementSpace *fespace = nodes->FESpace();
   double mindet = std::numeric_limits<double>::infinity();
   IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
   IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);
   IntegrationRules IntRulesGL(0, Quadrature1D::GaussLegendre);
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const FiniteElement *fe = fespace->GetFE(e);
      ElementTransformation *transf = mesh->GetElementTransformation(e);
      const int geom = fe->GetGeomType();
      const int dim = fe->GetDim();
      Vector loc(dim);
      DenseMatrix Jac(fe->GetDim());
      const IntegrationRule ir = type == 0 ? fe->GetNodes() :
                                 type == 1 ? IntRulesGLL.Get(geom, order) :
                                 type == 2 ? IntRulesCU.Get(geom, order) :
                                 IntRulesGL.Get(geom, order);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         IntegrationPoint ip = ir.IntPoint(q);

         transf->SetIntPoint(&ip);
         transf->Transform(ip, loc);
         Jac = transf->Jacobian();
         mindet = std::min(mindet, Jac.Det());
      }
   }

#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh *>(mesh);
   MPI_Allreduce(MPI_IN_PLACE, &mindet, 1,  MPI_DOUBLE, MPI_MIN,
                 pmesh->GetComm());
#endif
   return mindet;
}

static IntegrationRule PermuteIR(const IntegrationRule *irule,
                                 const Array<int> &perm)
{
   const int np = irule->GetNPoints();
   MFEM_VERIFY(np == perm.Size(), "Invalid permutation size");
   IntegrationRule ir(np);
   ir.SetOrder(irule->GetOrder());

   for (int i = 0; i < np; i++)
   {
      IntegrationPoint &ip_new = ir.IntPoint(i);
      const IntegrationPoint &ip_old = irule->IntPoint(perm[i]);
      ip_new.Set(ip_old.x, ip_old.y, ip_old.z, ip_old.weight);
   }

   return ir;
}

void ScaleNodes(const Vector &in, double a, double b, Vector &out)
{
   double maxv = in.Max();
   double minv = in.Min();
   out.SetSize(in.Size());
   for (int i = 0; i < in.Size(); i++)
   {
      out(i) = a + (b-a)*(in(i)-minv)/(maxv-minv);
   }
}

// Get N Chebyshev nodes in the interval [-1,1]
Vector GetChebyshevNodes(int n)
{
   MFEM_VERIFY(n > 2, "Invalid number of nodes");
   Vector nodes(n);
   nodes(0) = -1.0;
   nodes(n-1) = 1.0;
   for (int i = 2; i < n; i++)
   {
      nodes(i-1) = -std::cos(M_PI*((i-1)*1.0/(n-1)));
   }
   return nodes;
}

// Functions to mimic
IntegrationRule GetTensorProductRule(const Vector &ipx,
                                     const Vector &ipy)
{
   int nx = ipx.Size();
   int ny = ipy.Size();
   IntegrationRule irule(nx*ny);
   int i, j;

   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {
         IntegrationPoint &ip  = irule.IntPoint(j*nx+i);
         ip.x = ipx(i);
         ip.y = ipy(j);
         ip.weight = 0.0;
      }
   }
   return irule;
}

void GetTensorProductVector(const Vector &ipx,
                            const Vector &ipy,
                            Vector &ipxx,
                            Vector &ipyy)
{
   int nx = ipx.Size();
   int ny = ipy.Size();
   ipxx.SetSize(nx*ny);
   ipyy.SetSize(nx*ny);
   int i, j;

   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {
         ipxx(j*nx+i) = ipx(i);
         ipyy(j*nx+i) = ipy(j);
      }
   }
}

void GetValuesAtIntegrationPoints(Poly_1D::Basis &basis,
                                  const Vector &coeff, const Vector &ip,
                                  Vector &values)
{
   int n1D = ip.Size();
   values.SetSize(n1D);
   Vector basisvals(n1D);
   for (int i = 0; i < n1D; i++)
   {
      basis.Eval(ip(i), basisvals);
      values(i) = coeff*basisvals;
   }
}

// Set scaled to true if gllX is in [0,1] and false if it is in [-1,1]
// intX is automatically scaled based on gllX.min and gllX.max.
void Get1DBounds(const Vector &gllX,
                 const Vector &intX,
                 const Vector &gllW,
                 const DenseMatrix &lbound, const DenseMatrix &ubound,
                 Vector &coeff,  // nodal coeff of solution
                 Vector &intmin, Vector &intmax,
                 bool scaled=true)
{
   int nr = gllX.Size();
   int mr = intX.Size();
   MFEM_VERIFY(lbound.Height() == nr && lbound.Width() == mr,
               "Invalid bounds matrix");
   MFEM_VERIFY(ubound.Height() == nr && ubound.Width() == mr,
               "Invalid bounds matrix");

   Vector intScaled = intX;
   ScaleNodes(intX, gllX(0), gllX(nr-1), intScaled);

   intmin.SetSize(mr);
   intmax.SetSize(mr);
   Vector coeffm(nr);
   coeffm = 0.0;

   double a0 = 0.0;
   double a1 = 0.0;

   for (int i = 0; i < nr; i++)
   {
      // gll point from [0,1] to [-1, 1]
      double x = scaled ? 2.0*gllX(i)-1 : gllX(i); // x-coordinate
      double w = scaled ? 2.0*gllW(i) : gllW(i); // weight
      a0 += 0.5*coeff(i)*w;
      a1 += 1.5*coeff(i)*w*x;
   }

   // TODO:
   // a0 + a1x is in [-1, 1]
   // a0 + a1(2*x-1) where x is in 0,1
   // Set a0 = a0 -a1, a1 = 2*a1, if we want to avoid change in variables

   for (int i = 0; i < nr; i++)
   {
      double x = scaled ? 2.0*gllX(i)-1 : gllX(i);
      coeffm(i) = coeff(i) - a0 - a1*x;
   }

   for (int j = 0; j < mr; j++)
   {
      double hx = scaled ? 2.0*intScaled(j)-1 : intScaled(j); // x-coordinate
      intmin(j) = a0 + a1*hx;
      intmax(j) = a0 + a1*hx;
   }

   for (int i = 0; i < nr; i++)
   {
      double c = coeffm(i);
      for (int j = 0; j < mr; j++)
      {
         intmin(j) += std::min(lbound(i,j)*c, ubound(i,j)*c);
         intmax(j) += std::max(lbound(i,j)*c, ubound(i,j)*c);
      }
   }
}

// Recurively iterate around interval points that return a -ve minbound and
// +ve maxbound
void GetRecursiveExtrema1D(int currentdepth,
                           const int maxdepth,
                           const Vector &solcoeff,
                           const Vector &ref_nodes_gll,
                           const Vector &ref_nodes_wts,
                           const Vector &int_nodes,
                           const DenseMatrix &lbound,
                           const DenseMatrix &ubound,
                           Poly_1D::Basis &basis,
                           double rx0, double rx1,
                           Array<double> &intpts,
                           Array<double> &intmin,
                           Array<double> &intmax,
                           Array<int> &intdepth)
{
   const int nr = ref_nodes_gll.Size();
   const int mr = int_nodes.Size();
   int det_order = nr-1;
   int nIntervals = mr-1;

   // Get Nodes to get solution
   Vector ref_nodes_x = ref_nodes_gll;
   Vector solint; // interpolated solution
   ScaleNodes(ref_nodes_gll, rx0, rx1, ref_nodes_x);
   GetValuesAtIntegrationPoints(basis, solcoeff, ref_nodes_x, solint);

   Vector int_nodes_scaled_r = int_nodes;
   ScaleNodes(int_nodes, ref_nodes_gll(0), ref_nodes_gll(nr-1),
              int_nodes_scaled_r);

   bool scaling = ref_nodes_gll.Min() == 0 ? true : false;
   Vector qpmin, qpmax;
   Get1DBounds(ref_nodes_gll, int_nodes, ref_nodes_wts, lbound, ubound, solint,
               qpmin, qpmax, scaling);

   // Scale interval points for output
   Vector int_nodes_scaled_x = int_nodes_scaled_r;
   ScaleNodes(int_nodes_scaled_r, rx0, rx1, int_nodes_scaled_x);

   double minminval = qpmin.Min();
   double minmaxval = qpmax.Min();

   if (minminval > 0 || minmaxval < 0 || currentdepth == maxdepth)
      // if definitely positive or negative or we have reached our recursion limit
   {
      for (int i = 0; i < nIntervals; i++)
      {
         intpts.Append(int_nodes_scaled_x(i));
         intpts.Append(int_nodes_scaled_x(i+1));
         intmin.Append(qpmin(i));
         intmin.Append(qpmin(i+1));
         intmax.Append(qpmax(i));
         intmax.Append(qpmax(i+1));
         intdepth.Append(currentdepth);
         intdepth.Append(currentdepth);
      }
      return;
   }
   for (int i = 0; i < nIntervals; i++)
   {
      double rxstart = int_nodes_scaled_x(i);
      double rxend   = int_nodes_scaled_x(i+1);
      double leftmin = qpmin(i);
      double rightmin = qpmin(i+1);
      double leftmax = qpmax(i);
      double rightmax = qpmax(i+1);
      intpts.Append(rxstart);
      intpts.Append(rxend);
      intmin.Append(leftmin);
      intmin.Append(rightmin);
      intmax.Append(leftmax);
      intmax.Append(rightmax);
      if ((leftmin <= 0 && leftmax >= 0) ||
          (rightmin <= 0 && rightmax >= 0))
      {
         intdepth.Append(-currentdepth); //negative depth to indicate recursion
         intdepth.Append(-currentdepth);
         GetRecursiveExtrema1D(currentdepth+1, maxdepth, solcoeff,
                               ref_nodes_gll, ref_nodes_wts, int_nodes_scaled_r, lbound, ubound, basis,
                               rxstart, rxend,
                               intpts, intmin, intmax, intdepth);
      }
      else
      {
         intdepth.Append(currentdepth);
         intdepth.Append(currentdepth);
      }
   }
}

void Get2DBounds(const Vector &gllX,
                 const Vector &intX,
                 const Vector &gllW,
                 const DenseMatrix &lbound,
                 const DenseMatrix &ubound,
                 Vector &coeff, // nodal solution - lexicographic ordering
                 Vector &intmin, Vector &intmax,
                 bool scaled=true)
{
   int nr = gllX.Size();
   int mr = intX.Size();
   MFEM_VERIFY(lbound.Height() == nr && lbound.Width() == mr,
               "Invalid bounds matrix");
   MFEM_VERIFY(ubound.Height() == nr && ubound.Width() == mr,
               "Invalid bounds matrix");

   Vector intScaled = intX;
   ScaleNodes(intX, gllX(0), gllX(nr-1), intScaled);

   intmin.SetSize(mr*mr);
   intmax.SetSize(mr*mr);
   Vector intminT(mr*nr);
   Vector intmaxT(mr*nr);

   // Get bounds for each row of the solution
   for (int i = 0; i < nr; i++)
   {
      Vector solcoeff(coeff.GetData()+i*nr, nr);
      Vector intminrow(intminT.GetData()+i*mr, mr);
      Vector intmaxrow(intmaxT.GetData()+i*mr, mr);
      Get1DBounds(gllX, intScaled, gllW, lbound, ubound, solcoeff, intminrow,
                  intmaxrow, scaled);
   }

   // Compute a0 and a1 for each column of nodes
   Vector a0V(mr), a1V(mr);
   a0V = 0.0;
   a1V = 0.0;
   for (int j = 0; j < nr; j++) // row of nodes
   {
      double x = scaled ? 2.0*gllX(j)-1 : gllX(j); // x-coordinate
      double w = scaled ? 2.0*gllW(j) : gllW(j); // weight
      for (int i = 0; i < mr; i++) // column of interval points
      {
         double t = 0.5*(intminT(j*mr+i)+intmaxT(j*mr+i));
         a0V(i) += 0.5*t*w;
         a1V(i) += 1.5*t*w*x;
      }
   }

   // Initialize bounds using a0 and a1 values
   for (int j = 0; j < mr; j++) // row j
   {
      for (int i = 0; i < mr; i++) // column i
      {
         double hx = scaled ? 2.0*intScaled(j)-1 : intScaled(j);
         intmin(j*mr+i) = a0V(i) + a1V(i)*hx;
         intmax(j*mr+i) = a0V(i) + a1V(i)*hx;
      }
   }

   // Compute bounds
   double id1 = 0, id2 = 0;
   Vector vals(4);
   for (int j = 0; j < nr; j++)
   {
      double hx = scaled ? 2.0*gllX(j)-1 : gllX(j);
      for (int i = 0; i < mr; i++) // ith column
      {
         double t  = a0V(i) + a1V(i)*hx;
         double w0 = intminT(id1++) - t;
         double w1 = intmaxT(id2++) - t;
         for (int k = 0; k < mr; k++) // kth row
         {
            vals(0) = w0*lbound(j,k);
            vals(1) = w0*ubound(j,k);
            vals(2) = w1*lbound(j,k);
            vals(3) = w1*ubound(j,k);
            intmin(k*mr+i) += vals.Min();
            intmax(k*mr+i) += vals.Max();
         }
      }
   }
}


static int done2Drecursion = false;
void GetRecursiveExtrema2D(int currentdepth,
                           const int maxdepth,
                           const int elem,
                           const GridFunction &detgf,
                           const Vector &gllX,
                           const Vector &intX,
                           const Vector &gllW,
                           const DenseMatrix &lbound,
                           const DenseMatrix &ubound,
                           double rx0, double rx1,
                           double ry0, double ry1,
                           Array<double> &intptsx,
                           Array<double> &intptsy,
                           Array<double> &intmin,
                           Array<double> &intmax,
                           Array<double> &intdepth)
{
   const int dim = 2;
   const int nr = gllX.Size();
   const int mr = intX.Size();
   int det_order = detgf.FESpace()->GetElementOrder(elem);
   int nIntervals = mr-1;
   int parentdepth = currentdepth;

   // Get IntegrationRule
   Vector ref_nodes_x = gllX;
   ScaleNodes(gllX, rx0, rx1, ref_nodes_x);
   Vector ref_nodes_y = gllX;
   ScaleNodes(gllX, ry0, ry1, ref_nodes_y);

   IntegrationRule irulexy = GetTensorProductRule(ref_nodes_x, ref_nodes_y);

   // Get GridFunction Values for the IntegrationRule and store in lexicographic
   // ordering
   Vector solint; // interpolated solution
   detgf.GetValues(elem, irulexy, solint);

   Vector qpmin, qpmax;
   Get2DBounds(gllX, intX, gllW, lbound, ubound, solint, qpmin, qpmax);

   Vector intX_scaled_x = intX;
   Vector intX_scaled_y = intX;
   double minminval = qpmin.Min();
   double minmaxval = qpmax.Min();
   ScaleNodes(intX, rx0, rx1, intX_scaled_x);
   ScaleNodes(intX, ry0, ry1, intX_scaled_y);
   Vector qpxx, qpyy;
   GetTensorProductVector(intX_scaled_x, intX_scaled_y, qpxx, qpyy);

   if (minminval > 0 || minmaxval < 0 || currentdepth == maxdepth)
      // if definitely positive or negative or we have reached our recursion limit
   {
      for (int j = 0; j < intX_scaled_y.Size()-1; j++)
      {
         for (int i = 0; i < intX_scaled_x.Size()-1; i++)
         {
            int lr = j*mr+i;
            int ur = lr + mr;
            intptsx.Append(qpxx(lr));
            intptsy.Append(qpyy(lr));
            intmin.Append(qpmin(lr));
            intmax.Append(qpmax(lr));
            intdepth.Append(currentdepth);

            intptsx.Append(qpxx(lr+1));
            intptsy.Append(qpyy(lr+1));
            intmin.Append(qpmin(lr+1));
            intmax.Append(qpmax(lr+1));
            intdepth.Append(currentdepth);

            intptsx.Append(qpxx(ur));
            intptsy.Append(qpyy(ur));
            intmin.Append(qpmin(ur));
            intmax.Append(qpmax(ur));
            intdepth.Append(currentdepth);

            intptsx.Append(qpxx(ur+1));
            intptsy.Append(qpyy(ur+1));
            intmin.Append(qpmin(ur+1));
            intmax.Append(qpmax(ur+1));
            intdepth.Append(currentdepth);
         }
      }
      done2Drecursion = minmaxval < 0;
      return;
   }
   for (int j = 0; j < nIntervals; j++)
   {
      double rystart = intX_scaled_y(j);
      double ryend   = intX_scaled_y(j+1);
      for (int i = 0; i < nIntervals; i++)
      {
         double rxstart = intX_scaled_x(i);
         double rxend   = intX_scaled_x(i+1);

         int bi = j*mr+i;

         int ti = bi+mr;

         // four corners of the interval in lexicographic ordering
         double p0min = qpmin(bi);
         double p0max = qpmax(bi);
         double p1min = qpmin(bi+1);
         double p1max = qpmax(bi+1);
         double p2min = qpmin(ti);
         double p2max = qpmax(ti);
         double p3min = qpmin(ti+1);
         double p3max = qpmax(ti+1);

         intptsx.Append(rxstart);
         intptsy.Append(rystart);
         intmin.Append(p0min);
         intmax.Append(p0max);

         intptsx.Append(rxend);
         intptsy.Append(rystart);
         intmin.Append(p1min);
         intmax.Append(p1max);

         intptsx.Append(rxstart);
         intptsy.Append(ryend);
         intmin.Append(p2min);
         intmax.Append(p2max);

         intptsx.Append(rxend);
         intptsy.Append(ryend);
         intmin.Append(p3min);
         intmax.Append(p3max);

         if (((p0min <= 0 && p0max >= 0) ||
             (p1min <= 0 && p1max >= 0) ||
             (p2min <= 0 && p2max >= 0) ||
             (p3min <= 0 && p3max >= 0)) && !done2Drecursion)
         {
            intdepth.Append(-currentdepth);
            intdepth.Append(-currentdepth);
            intdepth.Append(-currentdepth);
            intdepth.Append(-currentdepth);
            GetRecursiveExtrema2D(parentdepth+1, maxdepth, elem, detgf,
                                  gllX, intX, gllW, lbound, ubound,
                                  rxstart, rxend, rystart, ryend,
                                  intptsx, intptsy, intmin, intmax, intdepth);
         }
         else
         {
            intdepth.Append(currentdepth);
            intdepth.Append(currentdepth);
            intdepth.Append(currentdepth);
            intdepth.Append(currentdepth);
         }
      }
   }
}

void GetRecursiveMinMaxBound2D(int currentdepth,
                               const int maxdepth,
                               const int elem,
                               const GridFunction &detgf,
                               const Vector &gllX,
                               const Vector &intX,
                               const Vector &gllW,
                               const DenseMatrix &lbound,
                               const DenseMatrix &ubound,
                               double rx0, double rx1,
                               double ry0, double ry1,
                               double &minminvalg,
                               double &minmaxvalg,
                               int &converged)
{
   const int dim = 2;
   const int nr = gllX.Size();
   const int mr = intX.Size();
   int det_order = detgf.FESpace()->GetElementOrder(elem);
   int nIntervals = mr-1;

   // Get IntegrationRule
   Vector ref_nodes_x = gllX;
   ScaleNodes(gllX, rx0, rx1, ref_nodes_x);
   Vector ref_nodes_y = gllX;
   ScaleNodes(gllX, ry0, ry1, ref_nodes_y);

   IntegrationRule irulexy = GetTensorProductRule(ref_nodes_x, ref_nodes_y);

   // Get GridFunction Values for the IntegrationRule and store in lexicographic
   // ordering
   Vector solint; // interpolated solution
   detgf.GetValues(elem, irulexy, solint);

   Vector qpmin, qpmax;
   Get2DBounds(gllX, intX, gllW, lbound, ubound, solint, qpmin, qpmax);

   Vector intX_scaled_x = intX;
   Vector intX_scaled_y = intX;
   double minminval = qpmin.Min();
   double minmaxval = qpmax.Min();
   ScaleNodes(intX, rx0, rx1, intX_scaled_x);
   ScaleNodes(intX, ry0, ry1, intX_scaled_y);
   Vector qpxx, qpyy;
   GetTensorProductVector(intX_scaled_x, intX_scaled_y, qpxx, qpyy);

   if (minminval > 0 || minmaxval < 0 || currentdepth == maxdepth)
      // if definitely positive or negative or we have reached our recursion limit
   {
      if (minmaxval < minmaxvalg)
      {
         minmaxvalg = minmaxval;
         minminvalg = minminval;
      }
      if (currentdepth == maxdepth)
      {
         converged = 0;
      }
      return;
   }
   for (int j = 0; j < nIntervals; j++)
   {
      double rystart = intX_scaled_y(j);
      double ryend   = intX_scaled_y(j+1);
      for (int i = 0; i < nIntervals; i++)
      {
         double rxstart = intX_scaled_x(i);
         double rxend   = intX_scaled_x(i+1);

         int bi = j*mr+i;
         int ti = bi+mr;

         // four corners of the interval in lexicographic ordering
         double p0min = qpmin(bi);
         double p0max = qpmax(bi);
         double p1min = qpmin(bi+1);
         double p1max = qpmax(bi+1);
         double p2min = qpmin(ti);
         double p2max = qpmax(ti);
         double p3min = qpmin(ti+1);
         double p3max = qpmax(ti+1);

         if ((p0min <= 0 && p0max >= 0) ||
             (p1min <= 0 && p1max >= 0) ||
             (p2min <= 0 && p2max >= 0) ||
             (p3min <= 0 && p3max >= 0))
         {
            GetRecursiveMinMaxBound2D(currentdepth+1, maxdepth, elem, detgf,
                                      gllX, intX, gllW, lbound, ubound,
                                      rxstart, rxend, rystart, ryend,
                                      minminvalg, minmaxvalg, converged);
         }
      }
   }
}

void ReadCustomBounds(Vector &gll, Vector &interval, DenseMatrix &lbound,
                      DenseMatrix &ubound, std::string filename)
{
   std::ifstream file(filename);
   if (!file.is_open())
   {
      MFEM_ABORT("File did not open\n");
   }

   int n;
   double val;

   file >> n;
   // std::cout << "Number of GLL points in file: " << n << std::endl;
   gll.SetSize(n);

   for (int i = 0; i < n; ++i)
   {
      file >> val;
      gll(i) = val;
   }

   int m;
   file >> m;
   // std::cout << "Number of interval points in file: " << m << std::endl;
   interval.SetSize(m);
   for (int i = 0; i < m; ++i)
   {
      file >> val;
      interval(i) = val;
   }

   lbound.SetSize(n, m);
   ubound.SetSize(n, m);

   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < m; j++)
      {
         file >> val;
         lbound(i,j) = val;
      }
      for (int j = 0; j < m; j++)
      {
         file >> val;
         ubound(i,j) = val;
      }
   }
}

#ifdef MFEM_USE_GSLIB
// Get Bounds on the nr basis functions at mr points in [0, 1]
void GetGSLIBBasisBounds(int nr, int mr,
                         DenseMatrix &lbound,                         DenseMatrix &ubound)
{
   lbound.SetSize(nr, mr);
   ubound.SetSize(nr, mr);

   Vector data_r;
   data_r.SetSize(lob_bnd_size(nr, mr));
   lob_bnd_setup(data_r.GetData(), nr,mr);

   double *bndsp = data_r.GetData() + 3*nr + mr;
   for (int i = 0; i < nr; i++)
   {
      for (int j = 0; j < mr; j++)
      {
         lbound(i, j) = bndsp[i*mr*2+2*j+0];
         ubound(i, j) = bndsp[i*mr*2+2*j+1];
      }
   }
}

void GetGSLIBBasisBoundsMe(int nr, int mr,
                           DenseMatrix &lbound,                         DenseMatrix &ubound)
{
   lbound.SetSize(nr, mr);
   ubound.SetSize(nr, mr);

   // Define lambda function for GetChebyshevNodes
   auto GetChebyshevNodes2 = [](int n) -> Vector
   {
      MFEM_VERIFY(n > 2, "Invalid number of nodes");
      Vector nodes(n);
      nodes(0) = -1.0;
      nodes(n - 1) = 1.0;
      for (int i = 2; i < n; ++i)
      {
         nodes(i - 1) = -std::cos(M_PI * ((i - 1.0)*1.0 / (n - 1)));
      }
      return nodes;
   };

   Vector cheb = GetChebyshevNodes2(mr);
   ScaleNodes(cheb, 0.0, 1.0, cheb);
   TensorBasisElement tbe = TensorBasisElement(1, nr-1, BasisType::GaussLobatto,
                                               TensorBasisElement::DofMapType::H1_DOF_MAP);
   const Poly_1D::Basis basis1d = tbe.GetBasis1D();

   // initialize
   lbound = 0.0;
   ubound = 0.0;

   for (int i = 0; i < nr; i++)
   {
      for (int j = 0; j < mr; j++)
      {
         if ( (j == 0 && i == 0) || (j == mr-1 && i == nr-1) )
         {
            lbound(i, j) = 1.0;
            ubound(i, j) = 1.0;
         }
      }
   }

   Vector bmv(nr), bpv(nr), bv(nr); // basis values
   Vector bdmv(nr), bdpv(nr), bdv(nr); // basis derivative values
   Vector vals(3);

   for (int j = 1; j < mr-1; j++)
   {
      double x = cheb(j);
      double xm = 0.5*(cheb(j-1)+cheb(j));
      double xp = 0.5*(cheb(j)+cheb(j+1));
      basis1d.Eval(xm, bmv, bdmv);
      basis1d.Eval(xp, bpv, bdpv);
      basis1d.Eval(x, bv);
      double dm = x-xm;
      double dp = x-xp;
      for (int i = 0; i < nr; i++)
      {
         vals(0)  = bv(i);
         vals(1) = bmv(i) +  dm*bdmv(i);
         vals(2) = bpv(i) +  dp*bdpv(i);
         lbound(i, j) = vals.Min();
         ubound(i, j) = vals.Max();
      }
   }
}

class GSLIBBound
{
private:
   int nr = 0;
   int mr = 0;
   Vector data_r;
   Vector cheb;
   Vector cheb_mfem;

public:
   GSLIBBound(int nr_, int mr_)
   {
      SetupWorkArrays(nr_, mr_);
   }

   ~GSLIBBound() {};

   void SetupWorkArrays(int nrnew, int mrnew);


   void GetGridFunctionBounds(GridFunction &gf,
                              Vector &qpmin, Vector &qpmax,
                              Vector &elminmax, Vector &elminmin,
                              Vector &elmin, Vector &elmax);

   double GetDetJBounds(Mesh *mesh,
                        Vector &qpmin, Vector &qpmax,
                        Vector &elminmax, Vector &elminmin,
                        Vector &elmin, Vector &elmax);

   void GetBounds(Vector &vect, Vector &qpmin, Vector &qpmax, int dim=1);

   // write setters for nr, mrfac and getters as well
   void Setnr(int nr_) { nr = nr_; }
   void Setmrfac(int mr_) { mr = mr_; }
   int Getnr() { return nr; }
   int Getmr() { return mr; }
   const Vector &GetChebMFEM() { return cheb_mfem; }
};

void GSLIBBound::SetupWorkArrays(int nrnew, int mrnew)
{
   if (nrnew != nr || mrnew != mr)
   {
      nr = nrnew;
      mr = mrnew;
      data_r.SetSize(lob_bnd_size(nr, mr));
      lob_bnd_setup(data_r.GetData(), nr,mr);
      cheb.SetSize(mr);
      cheb = data_r.GetData()+3*nr;
      cheb_mfem = cheb;
      cheb_mfem += 1;
      cheb_mfem *= 0.5;
   }
}

void GSLIBBound::GetBounds(Vector &vect, Vector &qpmin, Vector &qpmax, int dim)
{
   int nqpts = std::pow(mr, dim);

   qpmin.SetSize(nqpts);
   qpmax.SetSize(nqpts);
   qpmin = 0.0;
   qpmax = 0.0;
   MFEM_VERIFY(std::pow(nr,dim) == vect.Size(), "Incompatible size.");

   int wrksize = 2*mr;
   if (dim == 2)
   {
      wrksize = 2*mr*(nr+mr+1);
   }
   else if (dim == 3)
   {
      wrksize = 2*mr*mr*(nr+mr+1);
   }

   Vector work(wrksize);
   struct dbl_range bound;
   if (dim == 1)
   {
      bound = lob_bnd_1(data_r.GetData(),nr,mr, vect.GetData(),
                        work.GetData());
   }
   else if (dim == 2)
   {
      bound = lob_bnd_2(data_r.GetData(),nr,mr,
                        data_r.GetData(),nr,mr,
                        vect.GetData(),
                        work.GetData()); // compute bounds on u2
   }
   else
   {
      bound = lob_bnd_3(data_r.GetData(),nr,mr,
                        data_r.GetData(),nr,mr,
                        data_r.GetData(),nr,mr,
                        vect.GetData(),
                        work.GetData()); // compute bounds on u2
   }
   if (dim == 1)
   {
      for (int i = 0; i < nqpts; i++)
      {
         qpmin(i) = work[2*i];
         qpmax(i) = work[2*i+1];
      }
   }
   else if (dim == 2)
   {
      // place this lexicographically
      for (int j = 0; j < mr; j++)
      {
         for (int i = 0; i < mr; i++)
         {
            qpmin(j*mr+i) = work[2*(i*mr+j)+0];
            qpmax(j*mr+i) = work[2*(i*mr+j)+1];
         }
      }
   }
   else if (dim == 3)
   {
      DenseTensor Jtrl(mr,mr,mr), Jtru(mr,mr,mr);
      DenseTensor Jtrls(Jtrl.Data(), mr*mr*mr,1,1),
                  Jtrus(Jtru.Data(), mr*mr*mr,1,1);
      int count = 0;
      for (int k = 0; k < mr; k++)
      {
         for (int j = 0; j < mr; j++)
         {
            for (int i = 0; i < mr; i++)
            {
               Jtrl(k,j,i) = work(2*count+0);
               Jtru(k,j,i) = work(2*count+1);
               count++;
            }
         }
      }
      for (int j = 0; j < mr*mr*mr; j++)
      {
         qpmin(j) = Jtrls(j,0,0);
         qpmax(j) = Jtrus(j,0,0);
      }
   }
}

void GSLIBBound::GetGridFunctionBounds(GridFunction &gf,
                                       Vector &qpmin, Vector &qpmax,
                                       Vector &elminmax, Vector &elmaxmin,
                                       Vector &elmin, Vector &elmax)
{
   const FiniteElementSpace *fespace = gf.FESpace();
   const Mesh *mesh = fespace->GetMesh();
   const int dim = mesh->Dimension();
   Array<int> dofs;
   Vector vect;
   int nelem = mesh->GetNE();
   int maxorder = fespace->GetMaxElementOrder();
   int nqpel = dim == 1 ? mr : (dim == 2 ? mr*mr : mr*mr*mr);
   int nqpts = nelem*nqpel;

   elmin.SetSize(nelem);
   elmax.SetSize(nelem);
   elminmax.SetSize(nelem);
   elmaxmin.SetSize(nelem);
   qpmin.SetSize(nqpts);
   qpmax.SetSize(nqpts);
   elmin = 0.0;
   elmax = 0.0;
   elmaxmin = 0.0;
   elminmax = 0.0;
   qpmin = 0.0;
   qpmax = 0.0;

   int n = 0;
   for (int e = 0; e < nelem; e++)
   {
      fespace->GetElementDofs(e, dofs);
      gf.GetSubVector(dofs, vect);
      int order = fespace->GetOrder(e);
      MFEM_VERIFY(nr == order+1,"Incompatible nr set for GSLIBBound.");
      int wrksize = 2*mr;
      if (dim == 2)
      {
         wrksize = 2*mr*(nr+mr+1);
      }
      else if (dim == 3)
      {
         wrksize = 2*mr*mr*(nr+mr+1);
      }

      Vector work(wrksize);
      struct dbl_range bound;
      if (dim == 1)
      {
         bound = lob_bnd_1(data_r.GetData(),nr,mr, vect.GetData(),
                           work.GetData());
      }
      else if (dim == 2)
      {
         bound = lob_bnd_2(data_r.GetData(),nr,mr,
                           data_r.GetData(),nr,mr,
                           vect.GetData(),
                           work.GetData()); // compute bounds on u2
      }
      else
      {
         bound = lob_bnd_3(data_r.GetData(),nr,mr,
                           data_r.GetData(),nr,mr,
                           data_r.GetData(),nr,mr,
                           vect.GetData(),
                           work.GetData()); // compute bounds on u2
      }
      elmin(e) = bound.min;
      elmax(e) = bound.max;
      double min_max_bound = std::numeric_limits<double>::infinity();
      double max_min_bound = -std::numeric_limits<double>::infinity();
      for (int i = 0; i < nqpel; i++)
      {
         qpmin(n) = work[2*i];
         qpmax(n) = work[2*i+1];
         min_max_bound = std::min(min_max_bound, work[2*i+1]);
         max_min_bound = std::max(min_max_bound, work[2*i+0]);
         n++;
      }
      elminmax(e) = min_max_bound;
      elmaxmin(e) = max_min_bound;
   }
   qpmin.SetSize(n);
   qpmax.SetSize(n);
}

double GSLIBBound::GetDetJBounds(Mesh *mesh,
                                 Vector &qpmin, Vector &qpmax,
                                 Vector &elminmax, Vector &elmaxmin,
                                 Vector &elmin, Vector &elmax)
{
   int mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();
   int det_order = 2*mesh_order;
   int dim = mesh->Dimension();
   L2_FECollection fec(det_order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(mesh, &fec);
   GridFunction detgf(&fespace);
   Array<int> dofs;

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const FiniteElement *fe = fespace.GetFE(e);
      const IntegrationRule ir = fe->GetNodes();
      ElementTransformation *transf = mesh->GetElementTransformation(e);
      DenseMatrix Jac(fe->GetDim());
      const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>
                                      (fe);
      const Array<int> &irordering = nfe->GetLexicographicOrdering();
      IntegrationRule ir2 = irordering.Size() ?
                            PermuteIR(&ir, irordering) :
                            ir;

      Vector detvals(ir2.GetNPoints());
      Vector loc(dim);
      for (int q = 0; q < ir2.GetNPoints(); q++)
      {
         IntegrationPoint ip = ir2.IntPoint(q);
         transf->SetIntPoint(&ip);
         transf->Transform(ip, loc);
         Jac = transf->Jacobian();
         detvals(q) = Jac.Det();
      }

      fespace.GetElementDofs(e, dofs);
      if (irordering.Size())
      {
         for (int i = 0; i < dofs.Size(); i++)
         {
            detgf(dofs[i]) = detvals(irordering[i]);
         }
      }
      else
      {
         detgf.SetSubVector(dofs, detvals);
      }
   }

   GetGridFunctionBounds(detgf, qpmin, qpmax, elminmax, elmaxmin, elmin, elmax);
   return elminmax.Min();
}

#endif
