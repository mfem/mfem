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

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/text.hpp"

#include <fstream>
#include <algorithm>
#if defined(_MSC_VER) && (_MSC_VER < 1800)
#include <float.h>
#define copysign _copysign
#endif

namespace mfem
{

using namespace std;

const int KnotVector::MaxOrder = 10;


void GetShiftedGridPoints2D(int m, int n, int i, int j, int signedShift,
                            int& sm, int& sn, int& si, int& sj);

void GetInverseShiftedDimensions2D(int signedShift, int sm, int sn, int &m,
                                   int &n);

KnotVector::KnotVector(istream &input)
{
   input >> Order >> NumOfControlPoints;

   knot.Load(input, NumOfControlPoints + Order + 1);
   GetElements();
}

KnotVector::KnotVector(int order, int NCP)
{
   Order = order;
   NumOfControlPoints = NCP;
   knot.SetSize(NumOfControlPoints + Order + 1);
   NumOfElements = 0;

   knot = -1.;
}

KnotVector &KnotVector::operator=(const KnotVector &kv)
{
   Order = kv.Order;
   NumOfControlPoints = kv.NumOfControlPoints;
   NumOfElements = kv.NumOfElements;
   knot = kv.knot;
   coarse = kv.coarse;
   if (kv.spacing) { spacing = kv.spacing->Clone(); }

   // alternatively, re-compute NumOfElements
   // GetElements();

   return *this;
}

KnotVector *KnotVector::DegreeElevate(int t) const
{
   if (t < 0)
   {
      mfem_error("KnotVector::DegreeElevate :\n"
                 " Parent KnotVector order higher than child");
   }

   const int nOrder = Order + t;
   KnotVector *newkv = new KnotVector(nOrder, GetNCP() + t);

   for (int i = 0; i <= nOrder; i++)
   {
      (*newkv)[i] = knot(0);
   }
   for (int i = nOrder + 1; i < newkv->GetNCP(); i++)
   {
      (*newkv)[i] = knot(i - t);
   }
   for (int i = 0; i <= nOrder; i++)
   {
      (*newkv)[newkv->GetNCP() + i] = knot(knot.Size()-1);
   }

   newkv->GetElements();

   return newkv;
}

void KnotVector::UniformRefinement(Vector &newknots, int rf) const
{
   MFEM_VERIFY(rf > 1, "Refinement factor must be at least 2.");

   const real_t h = 1.0 / ((real_t) rf);

   newknots.SetSize(NumOfElements * (rf - 1));
   int j = 0;
   for (int i = 0; i < knot.Size()-1; i++)
   {
      if (knot(i) != knot(i+1))
      {
         for (int m = 1; m < rf; ++m)
         {
            newknots(j) = m * h * (knot(i) + knot(i+1));
            j++;
         }
      }
   }
}

int KnotVector::GetCoarseningFactor() const
{
   if (spacing)
   {
      if (spacing->Nested())
      {
         return 1;
      }
      else
      {
         return spacing->Size();   // Coarsen only if non-nested
      }
   }
   else
   {
      return 1;
   }
}

Vector KnotVector::GetFineKnots(const int cf) const
{
   Vector fine;
   if (cf < 2) { return fine; }

   const int cne = NumOfElements / cf;  // Coarse number of elements
   MFEM_VERIFY(cne > 0 && cne * cf == NumOfElements, "Invalid coarsening factor");

   fine.SetSize(cne * (cf - 1));

   int fcnt = 0;
   int i = Order;
   real_t kprev = knot(Order);
   for (int c=0; c<cne; ++c)  // Loop over coarse elements
   {
      int cnt = 0;
      while (cnt < cf)
      {
         i++;
         if (knot(i) != kprev)
         {
            kprev = knot(i);
            cnt++;
            if (cnt < cf)
            {
               fine[fcnt] = knot(i);
               fcnt++;
            }
         }
      }
   }

   MFEM_VERIFY(fcnt == fine.Size(), "");

   return fine;
}

void KnotVector::Refinement(Vector &newknots, int rf) const
{
   MFEM_VERIFY(rf > 1, "Refinement factor must be at least 2.");

   if (spacing)
   {
      spacing->ScaleParameters(1.0 / ((real_t) rf));
      spacing->SetSize(rf * NumOfElements);
      Vector s;
      spacing->EvalAll(s);

      newknots.SetSize((rf - 1) * NumOfElements);

      const real_t k0 = knot(0);
      const real_t k1 = knot(knot.Size()-1);

      Array<int> span0(NumOfElements + 1);
      span0[0] = 0;

      int j = 1;
      for (int i = 0; i < knot.Size()-1; i++)
      {
         if (knot(i) != knot(i+1))
         {
            span0[j] = i+1;
            j++;
         }
      }

      MFEM_VERIFY(j == NumOfElements + 1, "");

      real_t s0 = 0.0;

      for (int i=0; i<NumOfElements; ++i)
      {
         // Note that existing coarse knots are not modified here according to
         // the spacing formula, because modifying them will not produce a
         // correctly spaced mesh without also updating control points. Here, we
         // only define new knots according to the spacing formula. Non-nested
         // refinement should be done by using a single element per patch and
         // a sufficiently large refinement factor to produce the desired mesh
         // with only one refinement.

         s0 += s[rf*i];

         for (j=0; j<rf-1; ++j)
         {
            // Define a new knot between the modified coarse knots
            newknots(((rf - 1) * i) + j) = ((1.0 - s0) * k0) + (s0 * k1);

            s0 += s[(rf*i) + j + 1];
         }
      }
   }
   else
   {
      UniformRefinement(newknots, rf);
   }
}

void KnotVector::GetElements()
{
   NumOfElements = 0;
   for (int i = Order; i < NumOfControlPoints; i++)
   {
      if (knot(i) != knot(i+1))
      {
         NumOfElements++;
      }
   }
}

void KnotVector::Flip()
{
   real_t apb = knot(0) + knot(knot.Size()-1);

   int ns = (NumOfControlPoints - Order)/2;
   for (int i = 1; i <= ns; i++)
   {
      real_t tmp = apb - knot(Order + i);
      knot(Order + i) = apb - knot(NumOfControlPoints - i);
      knot(NumOfControlPoints - i) = tmp;
   }
}

void KnotVector::Print(std::ostream &os) const
{
   os << Order << ' ' << NumOfControlPoints << ' ';
   knot.Print(os, knot.Size());
}

void KnotVector::PrintFunctions(std::ostream &os, int samples) const
{
   MFEM_VERIFY(GetNE(), "Elements not counted. Use GetElements().");

   Vector shape(Order+1);

   real_t x, dx = 1.0/real_t (samples - 1);

   /* @a cnt is a counter including elements between repeated knots if
      present. This is required for usage of CalcShape. */
   int cnt = 0;

   for (int e = 0; e < GetNE(); e++, cnt++)
   {
      // Avoid printing shapes between repeated knots
      if (!isElement(cnt)) { e--; continue; }

      for (int j = 0; j <samples; j++)
      {
         x = j*dx;
         os << x + e;

         CalcShape(shape, cnt, x);
         for (int d = 0; d < Order+1; d++) { os<<"\t"<<shape[d]; }

         CalcDShape(shape, cnt, x);
         for (int d = 0; d < Order+1; d++) { os<<"\t"<<shape[d]; }

         CalcD2Shape(shape, cnt, x);
         for (int d = 0; d < Order+1; d++) { os<<"\t"<<shape[d]; }
         os << endl;
      }
   }
}

// Routine from "The NURBS book" - 2nd ed - Piegl and Tiller
// Algorithm A2.2 p. 70
void KnotVector::CalcShape(Vector &shape, int i, real_t xi) const
{
   MFEM_ASSERT(Order <= MaxOrder, "Order > MaxOrder!");

   int    p = Order;
   int    ip = (i >= 0) ? (i + p) : (-1 - i + p);
   real_t u = getKnotLocation((i >= 0) ? xi : 1. - xi, ip), saved, tmp;
   real_t left[MaxOrder+1], right[MaxOrder+1];

   shape(0) = 1.;
   for (int j = 1; j <= p; ++j)
   {
      left[j]  = u - knot(ip+1-j);
      right[j] = knot(ip+j) - u;
      saved    = 0.;
      for (int r = 0; r < j; ++r)
      {
         tmp      = shape(r)/(right[r+1] + left[j-r]);
         shape(r) = saved + right[r+1]*tmp;
         saved    = left[j-r]*tmp;
      }
      shape(j) = saved;
   }
}

// Routine from "The NURBS book" - 2nd ed - Piegl and Tiller
// Algorithm A2.3 p. 72
void KnotVector::CalcDShape(Vector &grad, int i, real_t xi) const
{
   int    p = Order, rk, pk;
   int    ip = (i >= 0) ? (i + p) : (-1 - i + p);
   real_t u = getKnotLocation((i >= 0) ? xi : 1. - xi, ip), temp, saved, d;
   real_t ndu[MaxOrder+1][MaxOrder+1], left[MaxOrder+1], right[MaxOrder+1];

#ifdef MFEM_DEBUG
   if (p > MaxOrder)
   {
      mfem_error("KnotVector::CalcDShape : Order > MaxOrder!");
   }
#endif

   ndu[0][0] = 1.0;
   for (int j = 1; j <= p; j++)
   {
      left[j] = u - knot(ip-j+1);
      right[j] = knot(ip+j) - u;
      saved = 0.0;
      for (int r = 0; r < j; r++)
      {
         ndu[j][r] = right[r+1] + left[j-r];
         temp = ndu[r][j-1]/ndu[j][r];
         ndu[r][j] = saved + right[r+1]*temp;
         saved = left[j-r]*temp;
      }
      ndu[j][j] = saved;
   }

   for (int r = 0; r <= p; ++r)
   {
      d = 0.0;
      rk = r-1;
      pk = p-1;
      if (r >= 1)
      {
         d = ndu[rk][pk]/ndu[p][rk];
      }
      if (r <= pk)
      {
         d -= ndu[r][pk]/ndu[p][r];
      }
      grad(r) = d;
   }

   if (i >= 0)
   {
      grad *= p*(knot(ip+1) - knot(ip));
   }
   else
   {
      grad *= p*(knot(ip) - knot(ip+1));
   }
}

// Routine from "The NURBS book" - 2nd ed - Piegl and Tiller
// Algorithm A2.3 p. 72
void KnotVector::CalcDnShape(Vector &gradn, int n, int i, real_t xi) const
{
   int    p = Order, rk, pk, j1, j2,r,j,k;
   int    ip = (i >= 0) ? (i + p) : (-1 - i + p);
   real_t u = getKnotLocation((i >= 0) ? xi : 1. - xi, ip);
   real_t temp, saved, d;
   real_t a[2][MaxOrder+1],ndu[MaxOrder+1][MaxOrder+1], left[MaxOrder+1],
          right[MaxOrder+1];

#ifdef MFEM_DEBUG
   if (p > MaxOrder)
   {
      mfem_error("KnotVector::CalcDnShape : Order > MaxOrder!");
   }
#endif

   ndu[0][0] = 1.0;
   for (j = 1; j <= p; j++)
   {
      left[j] = u - knot(ip-j+1);
      right[j] = knot(ip+j)- u;

      saved = 0.0;
      for (r = 0; r < j; r++)
      {
         ndu[j][r] = right[r+1] + left[j-r];
         temp = ndu[r][j-1]/ndu[j][r];
         ndu[r][j] = saved + right[r+1]*temp;
         saved = left[j-r]*temp;
      }
      ndu[j][j] = saved;
   }

   for (r = 0; r <= p; r++)
   {
      int s1 = 0;
      int s2 = 1;
      a[0][0] = 1.0;
      for (k = 1; k <= n; k++)
      {
         d = 0.0;
         rk = r-k;
         pk = p-k;
         if (r >= k)
         {
            a[s2][0] = a[s1][0]/ndu[pk+1][rk];
            d = a[s2][0]*ndu[rk][pk];
         }

         if (rk >= -1)
         {
            j1 = 1;
         }
         else
         {
            j1 = -rk;
         }

         if (r-1<= pk)
         {
            j2 = k-1;
         }
         else
         {
            j2 = p-r;
         }

         for (j = j1; j <= j2; j++)
         {
            a[s2][j] = (a[s1][j] - a[s1][j-1])/ndu[pk+1][rk+j];
            d += a[s2][j]*ndu[rk+j][pk];
         }

         if (r <= pk)
         {
            a[s2][k] = - a[s1][k-1]/ndu[pk+1][r];
            d += a[s2][j]*ndu[rk+j][pk];
         }
         gradn[r] = d;
         j = s1;
         s1 = s2;
         s2 = j;
      }
   }

   if (i >= 0)
   {
      u = (knot(ip+1) - knot(ip));
   }
   else
   {
      u = (knot(ip) - knot(ip+1));
   }

   temp = p*u;
   for (k = 1; k <= n-1; k++) { temp *= (p-k)*u; }

   for (j = 0; j <= p; j++) { gradn[j] *= temp; }

}

void KnotVector::FindMaxima(Array<int> &ks, Vector &xi, Vector &u) const
{
   Vector shape(Order+1);
   Vector maxima(GetNCP());
   real_t arg1, arg2, arg, max1, max2, max;

   xi.SetSize(GetNCP());
   u.SetSize(GetNCP());
   ks.SetSize(GetNCP());
   for (int j = 0; j <GetNCP(); j++)
   {
      maxima[j] = 0;
      for (int d = 0; d < Order+1; d++)
      {
         int i = j - d;
         if (isElement(i))
         {
            arg1 = 1e-16;
            CalcShape(shape, i, arg1);
            max1 = shape[d];

            arg2 = 1-(1e-16);
            CalcShape(shape, i, arg2);
            max2 = shape[d];

            arg = (arg1 + arg2)/2;
            CalcShape(shape, i, arg);
            max = shape[d];

            while ( ( max > max1 ) || (max > max2) )
            {
               if (max1 < max2)
               {
                  max1 = max;
                  arg1 = arg;
               }
               else
               {
                  max2 = max;
                  arg2 = arg;
               }

               arg = (arg1 + arg2)/2;
               CalcShape(shape, i, arg);
               max = shape[d];
            }

            if (max > maxima[j])
            {
               maxima[j] = max;
               ks[j] = i;
               xi[j] = arg;
               u[j]  = getKnotLocation(arg, i+Order);
            }
         }
      }
   }
}

// Routine from "The NURBS book" - 2nd ed - Piegl and Tiller
// Algorithm A9.1 p. 369
void KnotVector::FindInterpolant(Array<Vector*> &x)
{
   int order = GetOrder();
   int ncp = GetNCP();

   // Find interpolation points
   Vector xi_args, u_args;
   Array<int> i_args;
   FindMaxima(i_args,xi_args, u_args);

   // Assemble collocation matrix
   Vector shape(order+1);
   DenseMatrix A(ncp,ncp);
   A = 0.0;
   for (int i = 0; i < ncp; i++)
   {
      CalcShape(shape, i_args[i], xi_args[i]);
      for (int p = 0; p < order+1; p++)
      {
         A(i,i_args[i] + p) =  shape[p];
      }
   }

   // Solve problems
   A.Invert();
   Vector tmp;
   for (int i= 0; i < x.Size(); i++)
   {
      tmp = *x[i];
      A.Mult(tmp,*x[i]);
   }
}

int KnotVector::findKnotSpan(real_t u) const
{
   int low, mid, high;

   if (u == knot(NumOfControlPoints+Order))
   {
      mid = NumOfControlPoints;
   }
   else
   {
      low = Order;
      high = NumOfControlPoints + 1;
      mid = (low + high)/2;
      while ( (u < knot(mid-1)) || (u > knot(mid)) )
      {
         if (u < knot(mid-1))
         {
            high = mid;
         }
         else
         {
            low = mid;
         }
         mid = (low + high)/2;
      }
   }
   return mid;
}

void KnotVector::Difference(const KnotVector &kv, Vector &diff) const
{
   if (Order != kv.GetOrder())
   {
      mfem_error("KnotVector::Difference :\n"
                 " Can not compare knot vectors with different orders!");
   }

   int s = kv.Size() - Size();
   if (s < 0)
   {
      kv.Difference(*this, diff);
      return;
   }

   diff.SetSize(s);

   if (s == 0) { return; }

   s = 0;
   int i = 0;
   for (int j = 0; j < kv.Size(); j++)
   {
      if (abs(knot(i) - kv[j]) < 2 * std::numeric_limits<real_t>::epsilon())
      {
         i++;
      }
      else
      {
         diff(s) = kv[j];
         s++;
      }
   }
}

void NURBSPatch::init(int dim)
{
   Dim = dim;
   sd = nd = -1;

   if (kv.Size() == 1)
   {
      ni = kv[0]->GetNCP();
      nj = -1;
      nk = -1;

      data = new real_t[ni*Dim];

#ifdef MFEM_DEBUG
      for (int i = 0; i < ni*Dim; i++)
      {
         data[i] = -999.99;
      }
#endif
   }
   else if (kv.Size() == 2)
   {
      ni = kv[0]->GetNCP();
      nj = kv[1]->GetNCP();
      nk = -1;

      data = new real_t[ni*nj*Dim];

#ifdef MFEM_DEBUG
      for (int i = 0; i < ni*nj*Dim; i++)
      {
         data[i] = -999.99;
      }
#endif
   }
   else if (kv.Size() == 3)
   {
      ni = kv[0]->GetNCP();
      nj = kv[1]->GetNCP();
      nk = kv[2]->GetNCP();

      data = new real_t[ni*nj*nk*Dim];

#ifdef MFEM_DEBUG
      for (int i = 0; i < ni*nj*nk*Dim; i++)
      {
         data[i] = -999.99;
      }
#endif
   }
   else
   {
      mfem_error("NURBSPatch::init : Wrong dimension of knotvectors!");
   }
}

NURBSPatch::NURBSPatch(const NURBSPatch &orig)
   : ni(orig.ni), nj(orig.nj), nk(orig.nk), Dim(orig.Dim),
     data(NULL), kv(orig.kv.Size()), nd(orig.nd), ls(orig.ls), sd(orig.sd)
{
   // Allocate and copy data:
   const int data_size = Dim*ni*nj*((kv.Size() == 2) ? 1 : nk);
   data = new real_t[data_size];
   std::memcpy(data, orig.data, data_size*sizeof(real_t));

   // Copy the knot vectors:
   for (int i = 0; i < kv.Size(); i++)
   {
      kv[i] = new KnotVector(*orig.kv[i]);
   }
}

NURBSPatch::NURBSPatch(std::istream &input)
{
   int pdim, dim, size = 1;
   string ident;

   input >> ws >> ident >> pdim; // knotvectors
   kv.SetSize(pdim);
   for (int i = 0; i < pdim; i++)
   {
      kv[i] = new KnotVector(input);
      size *= kv[i]->GetNCP();
   }

   input >> ws >> ident >> dim; // dimension
   init(dim + 1);

   input >> ws >> ident; // controlpoints (homogeneous coordinates)
   if (ident == "controlpoints" || ident == "controlpoints_homogeneous")
   {
      for (int j = 0, i = 0; i < size; i++)
         for (int d = 0; d <= dim; d++, j++)
         {
            input >> data[j];
         }
   }
   else // "controlpoints_cartesian" (Cartesian coordinates with weight)
   {
      for (int j = 0, i = 0; i < size; i++)
      {
         for (int d = 0; d <= dim; d++)
         {
            input >> data[j+d];
         }
         for (int d = 0; d < dim; d++)
         {
            data[j+d] *= data[j+dim];
         }
         j += (dim+1);
      }
   }
}

NURBSPatch::NURBSPatch(const KnotVector *kv0, const KnotVector *kv1, int dim)
{
   kv.SetSize(2);
   kv[0] = new KnotVector(*kv0);
   kv[1] = new KnotVector(*kv1);
   init(dim);
}

NURBSPatch::NURBSPatch(const KnotVector *kv0, const KnotVector *kv1,
                       const KnotVector *kv2, int dim)
{
   kv.SetSize(3);
   kv[0] = new KnotVector(*kv0);
   kv[1] = new KnotVector(*kv1);
   kv[2] = new KnotVector(*kv2);
   init(dim);
}

NURBSPatch::NURBSPatch(Array<const KnotVector *> &kvs, int dim)
{
   kv.SetSize(kvs.Size());
   for (int i = 0; i < kv.Size(); i++)
   {
      kv[i] = new KnotVector(*kvs[i]);
   }
   init(dim);
}

NURBSPatch::NURBSPatch(NURBSPatch *parent, int dir, int Order, int NCP)
{
   kv.SetSize(parent->kv.Size());
   for (int i = 0; i < kv.Size(); i++)
      if (i != dir)
      {
         kv[i] = new KnotVector(*parent->kv[i]);
      }
      else
      {
         kv[i] = new KnotVector(Order, NCP);
      }
   init(parent->Dim);
}

void NURBSPatch::swap(NURBSPatch *np)
{
   if (data != NULL)
   {
      delete [] data;
   }

   for (int i = 0; i < kv.Size(); i++)
   {
      if (kv[i]) { delete kv[i]; }
   }

   data = np->data;
   np->kv.Copy(kv);

   ni  = np->ni;
   nj  = np->nj;
   nk  = np->nk;
   Dim = np->Dim;

   np->data = NULL;
   np->kv.SetSize(0);

   delete np;
}

NURBSPatch::~NURBSPatch()
{
   if (data != NULL)
   {
      delete [] data;
   }

   for (int i = 0; i < kv.Size(); i++)
   {
      if (kv[i]) { delete kv[i]; }
   }
}

void NURBSPatch::Print(std::ostream &os) const
{
   int size = 1;

   os << "knotvectors\n" << kv.Size() << '\n';
   for (int i = 0; i < kv.Size(); i++)
   {
      kv[i]->Print(os);
      size *= kv[i]->GetNCP();
   }

   os << "\ndimension\n" << Dim - 1
      << "\n\ncontrolpoints\n";
   for (int j = 0, i = 0; i < size; i++)
   {
      os << data[j++];
      for (int d = 1; d < Dim; d++)
      {
         os << ' ' << data[j++];
      }
      os << '\n';
   }
}

int NURBSPatch::SetLoopDirection(int dir)
{
   if (nj == -1)  // 1D case
   {
      if (dir == 0)
      {
         sd = Dim;
         nd = ni;
         ls = Dim;
         return ls;
      }
      else
      {
         mfem::err << "NURBSPatch::SetLoopDirection :\n"
                   " Direction error in 1D patch, dir = " << dir << '\n';
         mfem_error();
      }
   }
   else if (nk == -1)  // 2D case
   {
      if (dir == 0)
      {
         sd = Dim;
         nd = ni;
         ls = nj*Dim;
         return ls;
      }
      else if (dir == 1)
      {
         sd = ni*Dim;
         nd = nj;
         ls = ni*Dim;
         return ls;
      }
      else
      {
         mfem::err << "NURBSPatch::SetLoopDirection :\n"
                   " Direction error in 2D patch, dir = " << dir << '\n';
         mfem_error();
      }
   }
   else  // 3D case
   {
      if (dir == 0)
      {
         sd = Dim;
         nd = ni;
         ls = nj*nk*Dim;
         return ls;
      }
      else if (dir == 1)
      {
         sd = ni*Dim;
         nd = nj;
         ls = ni*nk*Dim;
         return ls;
      }
      else if (dir == 2)
      {
         sd = ni*nj*Dim;
         nd = nk;
         ls = ni*nj*Dim;
         return ls;
      }
      else
      {
         mfem::err << "NURBSPatch::SetLoopDirection :\n"
                   " Direction error in 3D patch, dir = " << dir << '\n';
         mfem_error();
      }
   }

   return -1;
}

void NURBSPatch::UniformRefinement(Array<int> const& rf)
{
   Vector newknots;
   for (int dir = 0; dir < kv.Size(); dir++)
   {
      if (rf[dir] != 1)
      {
         kv[dir]->Refinement(newknots, rf[dir]);
         KnotInsert(dir, newknots);
      }
   }
}

void NURBSPatch::UniformRefinement(int rf)
{
   Array<int> rf_array(kv.Size());
   rf_array = rf;
   UniformRefinement(rf_array);
}

void NURBSPatch::Coarsen(Array<int> const& cf, real_t tol)
{
   for (int dir = 0; dir < kv.Size(); dir++)
   {
      if (!kv[dir]->coarse)
      {
         const int ne_fine = kv[dir]->GetNE();
         KnotRemove(dir, kv[dir]->GetFineKnots(cf[dir]), tol);
         kv[dir]->coarse = true;
         kv[dir]->GetElements();

         const int ne_coarse = kv[dir]->GetNE();
         MFEM_VERIFY(ne_fine == cf[dir] * ne_coarse, "");
         if (kv[dir]->spacing)
         {
            kv[dir]->spacing->SetSize(ne_coarse);
            kv[dir]->spacing->ScaleParameters((real_t) cf[dir]);
         }
      }
   }
}

void NURBSPatch::Coarsen(int cf, real_t tol)
{
   Array<int> cf_array(kv.Size());
   cf_array = cf;
   Coarsen(cf_array, tol);
}

void NURBSPatch::GetCoarseningFactors(Array<int> & f) const
{
   f.SetSize(kv.Size());
   for (int dir = 0; dir < kv.Size(); dir++)
   {
      f[dir] = kv[dir]->GetCoarseningFactor();
   }
}

void NURBSPatch::KnotInsert(Array<KnotVector *> &newkv)
{
   MFEM_ASSERT(newkv.Size() == kv.Size(), "Invalid input to KnotInsert");
   for (int dir = 0; dir < kv.Size(); dir++)
   {
      KnotInsert(dir, *newkv[dir]);
   }
}

void NURBSPatch::KnotInsert(int dir, const KnotVector &newkv)
{
   if (dir >= kv.Size() || dir < 0)
   {
      mfem_error("NURBSPatch::KnotInsert : Incorrect direction!");
   }

   int t = newkv.GetOrder() - kv[dir]->GetOrder();

   if (t > 0)
   {
      DegreeElevate(dir, t);
   }
   else if (t < 0)
   {
      mfem_error("NURBSPatch::KnotInsert : Incorrect order!");
   }

   Vector diff;
   GetKV(dir)->Difference(newkv, diff);
   if (diff.Size() > 0)
   {
      KnotInsert(dir, diff);
   }
}

void NURBSPatch::KnotInsert(Array<Vector *> &newkv)
{
   MFEM_ASSERT(newkv.Size() == kv.Size(), "Invalid input to KnotInsert");
   for (int dir = 0; dir < kv.Size(); dir++)
   {
      KnotInsert(dir, *newkv[dir]);
   }
}

void NURBSPatch::KnotRemove(Array<Vector *> &rmkv, real_t tol)
{
   for (int dir = 0; dir < kv.Size(); dir++)
   {
      KnotRemove(dir, *rmkv[dir], tol);
   }
}

void NURBSPatch::KnotRemove(int dir, const Vector &knot, real_t tol)
{
   // TODO: implement an efficient version of this!
   for (auto k : knot)
   {
      KnotRemove(dir, k, 1, tol);
   }
}

// Algorithm A5.5 from "The NURBS Book", 2nd ed, Piegl and Tiller, chapter 5.
void NURBSPatch::KnotInsert(int dir, const Vector &knot)
{
   if (knot.Size() == 0 ) { return; }

   if (dir >= kv.Size() || dir < 0)
   {
      mfem_error("NURBSPatch::KnotInsert : Invalid direction!");
   }

   NURBSPatch &oldp  = *this;
   KnotVector &oldkv = *kv[dir];

   NURBSPatch *newpatch = new NURBSPatch(this, dir, oldkv.GetOrder(),
                                         oldkv.GetNCP() + knot.Size());
   NURBSPatch &newp  = *newpatch;
   KnotVector &newkv = *newp.GetKV(dir);

   newkv.spacing = oldkv.spacing;

   int size = oldp.SetLoopDirection(dir);
   if (size != newp.SetLoopDirection(dir))
   {
      mfem_error("NURBSPatch::KnotInsert : Size mismatch!");
   }

   int rr = knot.Size() - 1;
   int a  = oldkv.findKnotSpan(knot(0))  - 1;
   int b  = oldkv.findKnotSpan(knot(rr)) - 1;
   int pl = oldkv.GetOrder();
   int ml = oldkv.GetNCP();

   for (int j = 0; j <= a; j++)
   {
      newkv[j] = oldkv[j];
   }
   for (int j = b+pl; j <= ml+pl; j++)
   {
      newkv[j+rr+1] = oldkv[j];
   }
   for (int k = 0; k <= (a-pl); k++)
   {
      for (int ll = 0; ll < size; ll++)
      {
         newp.slice(k,ll) = oldp.slice(k,ll);
      }
   }
   for (int k = (b-1); k < ml; k++)
   {
      for (int ll = 0; ll < size; ll++)
      {
         newp.slice(k+rr+1,ll) = oldp.slice(k,ll);
      }
   }

   int i = b+pl-1;
   int k = b+pl+rr;

   for (int j = rr; j >= 0; j--)
   {
      while ( (knot(j) <= oldkv[i]) && (i > a) )
      {
         newkv[k] = oldkv[i];
         for (int ll = 0; ll < size; ll++)
         {
            newp.slice(k-pl-1,ll) = oldp.slice(i-pl-1,ll);
         }

         k--;
         i--;
      }

      for (int ll = 0; ll < size; ll++)
      {
         newp.slice(k-pl-1,ll) = newp.slice(k-pl,ll);
      }

      for (int l = 1; l <= pl; l++)
      {
         int ind = k-pl+l;
         real_t alfa = newkv[k+l] - knot(j);
         if (fabs(alfa) == 0.0)
         {
            for (int ll = 0; ll < size; ll++)
            {
               newp.slice(ind-1,ll) = newp.slice(ind,ll);
            }
         }
         else
         {
            alfa = alfa/(newkv[k+l] - oldkv[i-pl+l]);
            for (int ll = 0; ll < size; ll++)
            {
               newp.slice(ind-1,ll) = alfa*newp.slice(ind-1,ll) +
                                      (1.0-alfa)*newp.slice(ind,ll);
            }
         }
      }

      newkv[k] = knot(j);
      k--;
   }

   newkv.GetElements();

   swap(newpatch);
}

// Algorithm A5.8 from "The NURBS Book", 2nd ed, Piegl and Tiller, chapter 5.
int NURBSPatch::KnotRemove(int dir, real_t knot, int ntimes, real_t tol)
{
   if (dir >= kv.Size() || dir < 0)
   {
      mfem_error("NURBSPatch::KnotRemove : Invalid direction!");
   }

   NURBSPatch &oldp  = *this;
   KnotVector &oldkv = *kv[dir];

   // Find the index of the last occurrence of the knot.
   int id = -1;
   int multiplicity = 0;
   for (int i=0; i<oldkv.Size(); ++i)
   {
      if (oldkv[i] == knot)
      {
         id = i;
         multiplicity++;
      }
   }

   MFEM_VERIFY(0 < id && id < oldkv.Size() - 1 && ntimes <= multiplicity,
               "Only interior knots of sufficient multiplicity may be removed.");

   const int p = oldkv.GetOrder();

   NURBSPatch tmpp(this, dir, p, oldkv.GetNCP());

   const int size = oldp.SetLoopDirection(dir);
   if (size != tmpp.SetLoopDirection(dir))
   {
      mfem_error("NURBSPatch::KnotRemove : Size mismatch!");
   }

   // Copy old data
   for (int k = 0; k < oldp.nd; ++k)
   {
      for (int ll = 0; ll < size; ll++)
      {
         tmpp.slice(k,ll) = oldp.slice(k,ll);
      }
   }

   const int r = id;
   const int s = multiplicity;

   int last = r - s;
   int first = r - p;

   int i = first;
   int j = last;

   Array2D<real_t> temp(last + ntimes + 1, size);

   for (int t=0; t<ntimes; ++t)
   {
      int off = first - 1;  // Difference in index between temp and P.

      for (int ll = 0; ll < size; ll++)
      {
         temp(0, ll) = oldp.slice(off, ll);
         temp(last + 1 - off, ll) = oldp.slice(last + 1, ll);
      }

      int ii = 1;
      int jj = last - off;

      while (j - i > t)
      {
         // Compute new control points for one removal step
         const real_t a_i = (knot - oldkv[i]) / (oldkv[i+p+1] - oldkv[i]);
         const real_t a_j = (knot - oldkv[j]) / (oldkv[j+p+1] - oldkv[j]);

         for (int ll = 0; ll < size; ll++)
         {
            temp(ii,ll) = (1.0 / a_i) * oldp.slice(i,ll) -
                          ((1.0/a_i) - 1.0) * temp(ii - 1, ll);

            temp(jj,ll) = (1.0 / (1.0 - a_j)) * (oldp.slice(j,ll) -
                                                 (a_j * temp(jj + 1, ll)));
         }

         i++;  ii++;
         j--;  jj--;
      }

      // Check whether knot is removable
      Vector diff(size);
      if (j - i < t)
      {
         for (int ll = 0; ll < size; ll++)
         {
            diff[ll] = temp(ii-1, ll) - temp(jj+1, ll);
         }
      }
      else
      {
         const real_t a_i = (knot - oldkv[i]) / (oldkv[i+p+1] - oldkv[i]);
         for (int ll = 0; ll < size; ll++)
            diff[ll] = oldp.slice(i,ll) - (a_i * temp(ii+1, ll))
                       - ((1.0 - a_i) * temp(ii-1, ll));
      }

      const real_t dist = diff.Norml2();
      if (dist >= tol)
      {
         // Removal failed. Return the number of successful removals.
         mfem::out << "Knot removal failed after " << t
                   << " successful removals" << endl;
         return t;
      }

      // Note that the new weights may not be positive.

      // Save new control points
      i = first;
      j = last;

      while (j - i > t)
      {
         for (int ll = 0; ll < size; ll++)
         {
            tmpp.slice(i,ll) = temp(i - off,ll);
            tmpp.slice(j,ll) = temp(j - off,ll);
         }
         i++;
         j--;
      }

      first--;
      last++;
   }  // End of loop (t) over ntimes.

   const int fout = ((2*r) - s - p) / 2;  // First control point out
   j = fout;
   i = j;

   for (int k=1; k<ntimes; ++k)
   {
      if (k % 2 == 1)
      {
         i++;
      }
      else
      {
         j--;
      }
   }

   NURBSPatch *newpatch = new NURBSPatch(this, dir, p,
                                         oldkv.GetNCP() - ntimes);
   NURBSPatch &newp = *newpatch;
   if (size != newp.SetLoopDirection(dir))
   {
      mfem_error("NURBSPatch::KnotRemove : Size mismatch!");
   }

   for (int k = 0; k < fout; ++k)
   {
      for (int ll = 0; ll < size; ll++)
      {
         newp.slice(k,ll) = oldp.slice(k,ll);  // Copy old data
      }
   }

   for (int k = i+1; k < oldp.nd; ++k)
   {
      for (int ll = 0; ll < size; ll++)
      {
         newp.slice(j,ll) = tmpp.slice(k,ll);  // Shift
      }

      j++;
   }

   KnotVector &newkv = *newp.GetKV(dir);
   MFEM_VERIFY(newkv.Size() == oldkv.Size() - ntimes, "");

   newkv.spacing = oldkv.spacing;
   newkv.coarse = oldkv.coarse;

   for (int k = 0; k < id - ntimes + 1; k++)
   {
      newkv[k] = oldkv[k];
   }
   for (int k = id + 1; k < oldkv.Size(); k++)
   {
      newkv[k - ntimes] = oldkv[k];
   }

   newkv.GetElements();

   swap(newpatch);

   return ntimes;
}

void NURBSPatch::DegreeElevate(int t)
{
   for (int dir = 0; dir < kv.Size(); dir++)
   {
      DegreeElevate(dir, t);
   }
}

// Routine from "The NURBS book" - 2nd ed - Piegl and Tiller
void NURBSPatch::DegreeElevate(int dir, int t)
{
   if (dir >= kv.Size() || dir < 0)
   {
      mfem_error("NURBSPatch::DegreeElevate : Incorrect direction!");
   }

   MFEM_ASSERT(t >= 0, "DegreeElevate cannot decrease the degree.");

   int i, j, k, kj, mpi, mul, mh, kind, cind, first, last;
   int r, a, b, oldr, save, s, tr, lbz, rbz, l;
   real_t inv, ua, ub, numer, alf, den, bet, gam;

   NURBSPatch &oldp  = *this;
   KnotVector &oldkv = *kv[dir];
   oldkv.GetElements();

   NURBSPatch *newpatch = new NURBSPatch(this, dir, oldkv.GetOrder() + t,
                                         oldkv.GetNCP() + oldkv.GetNE()*t);
   NURBSPatch &newp  = *newpatch;
   KnotVector &newkv = *newp.GetKV(dir);

   if (oldkv.spacing) { newkv.spacing = oldkv.spacing; }

   int size = oldp.SetLoopDirection(dir);
   if (size != newp.SetLoopDirection(dir))
   {
      mfem_error("NURBSPatch::DegreeElevate : Size mismatch!");
   }

   int p = oldkv.GetOrder();
   int n = oldkv.GetNCP()-1;

   DenseMatrix bezalfs (p+t+1, p+1);
   DenseMatrix bpts    (p+1,   size);
   DenseMatrix ebpts   (p+t+1, size);
   DenseMatrix nextbpts(p-1,   size);
   Vector      alphas  (p-1);

   int m = n + p + 1;
   int ph = p + t;
   int ph2 = ph/2;

   {
      Array2D<int> binom(ph+1, ph+1);
      for (i = 0; i <= ph; i++)
      {
         binom(i,0) = binom(i,i) = 1;
         for (j = 1; j < i; j++)
         {
            binom(i,j) = binom(i-1,j) + binom(i-1,j-1);
         }
      }

      bezalfs(0,0)  = 1.0;
      bezalfs(ph,p) = 1.0;

      for (i = 1; i <= ph2; i++)
      {
         inv = 1.0/binom(ph,i);
         mpi = min(p,i);
         for (j = max(0,i-t); j <= mpi; j++)
         {
            bezalfs(i,j) = inv*binom(p,j)*binom(t,i-j);
         }
      }
   }

   for (i = ph2+1; i < ph; i++)
   {
      mpi = min(p,i);
      for (j = max(0,i-t); j <= mpi; j++)
      {
         bezalfs(i,j) = bezalfs(ph-i,p-j);
      }
   }

   mh = ph;
   kind = ph + 1;
   r = -1;
   a = p;
   b = p + 1;
   cind = 1;
   ua = oldkv[0];
   for (l = 0; l < size; l++)
   {
      newp.slice(0,l) = oldp.slice(0,l);
   }
   for (i = 0; i <= ph; i++)
   {
      newkv[i] = ua;
   }

   for (i = 0; i <= p; i++)
   {
      for (l = 0; l < size; l++)
      {
         bpts(i,l) = oldp.slice(i,l);
      }
   }

   while (b < m)
   {
      i = b;
      while (b < m && oldkv[b] == oldkv[b+1]) { b++; }

      mul = b-i+1;

      mh = mh + mul + t;
      ub = oldkv[b];
      oldr = r;
      r = p-mul;
      if (oldr > 0) { lbz = (oldr+2)/2; }
      else { lbz = 1; }

      if (r > 0) { rbz = ph-(r+1)/2; }
      else { rbz = ph; }

      if (r > 0)
      {
         numer = ub - ua;
         for (k = p ; k > mul; k--)
         {
            alphas[k-mul-1] = numer/(oldkv[a+k]-ua);
         }

         for (j = 1; j <= r; j++)
         {
            save = r-j;
            s = mul+j;
            for (k = p; k >= s; k--)
            {
               for (l = 0; l < size; l++)
                  bpts(k,l) = (alphas[k-s]*bpts(k,l) +
                               (1.0-alphas[k-s])*bpts(k-1,l));
            }
            for (l = 0; l < size; l++)
            {
               nextbpts(save,l) = bpts(p,l);
            }
         }
      }

      for (i = lbz; i <= ph; i++)
      {
         for (l = 0; l < size; l++)
         {
            ebpts(i,l) = 0.0;
         }
         mpi = min(p,i);
         for (j = max(0,i-t); j <= mpi; j++)
         {
            for (l = 0; l < size; l++)
            {
               ebpts(i,l) += bezalfs(i,j)*bpts(j,l);
            }
         }
      }

      if (oldr > 1)
      {
         first = kind-2;
         last = kind;
         den = ub-ua;
         bet = (ub-newkv[kind-1])/den;

         for (tr = 1; tr < oldr; tr++)
         {
            i = first;
            j = last;
            kj = j-kind+1;
            while (j-i > tr)
            {
               if (i < cind)
               {
                  alf = (ub-newkv[i])/(ua-newkv[i]);
                  for (l = 0; l < size; l++)
                  {
                     newp.slice(i,l) = alf*newp.slice(i,l)-(1.0-alf)*newp.slice(i-1,l);
                  }
               }
               if (j >= lbz)
               {
                  if ((j-tr) <= (kind-ph+oldr))
                  {
                     gam = (ub-newkv[j-tr])/den;
                     for (l = 0; l < size; l++)
                     {
                        ebpts(kj,l) = gam*ebpts(kj,l) + (1.0-gam)*ebpts(kj+1,l);
                     }
                  }
                  else
                  {
                     for (l = 0; l < size; l++)
                     {
                        ebpts(kj,l) = bet*ebpts(kj,l) + (1.0-bet)*ebpts(kj+1,l);
                     }
                  }
               }
               i = i+1;
               j = j-1;
               kj = kj-1;
            }
            first--;
            last++;
         }
      }

      if (a != p)
      {
         for (i = 0; i < (ph-oldr); i++)
         {
            newkv[kind] = ua;
            kind = kind+1;
         }
      }
      for (j = lbz; j <= rbz; j++)
      {
         for (l = 0; l < size; l++)
         {
            newp.slice(cind,l) =  ebpts(j,l);
         }
         cind = cind +1;
      }

      if (b < m)
      {
         for (j = 0; j <r; j++)
            for (l = 0; l < size; l++)
            {
               bpts(j,l) = nextbpts(j,l);
            }

         for (j = r; j <= p; j++)
            for (l = 0; l < size; l++)
            {
               bpts(j,l) = oldp.slice(b-p+j,l);
            }

         a = b;
         b = b+1;
         ua = ub;
      }
      else
      {
         for (i = 0; i <= ph; i++)
         {
            newkv[kind+i] = ub;
         }
      }
   }
   newkv.GetElements();

   swap(newpatch);
}

void NURBSPatch::FlipDirection(int dir)
{
   int size = SetLoopDirection(dir);

   for (int id = 0; id < nd/2; id++)
      for (int i = 0; i < size; i++)
      {
         Swap<real_t>((*this).slice(id,i), (*this).slice(nd-1-id,i));
      }
   kv[dir]->Flip();
}

void NURBSPatch::SwapDirections(int dir1, int dir2)
{
   if (abs(dir1-dir2) == 2)
   {
      mfem_error("NURBSPatch::SwapDirections :"
                 " directions 0 and 2 are not supported!");
   }

   Array<const KnotVector *> nkv(kv);

   Swap<const KnotVector *>(nkv[dir1], nkv[dir2]);
   NURBSPatch *newpatch = new NURBSPatch(nkv, Dim);

   int size = SetLoopDirection(dir1);
   newpatch->SetLoopDirection(dir2);

   for (int id = 0; id < nd; id++)
      for (int i = 0; i < size; i++)
      {
         (*newpatch).slice(id,i) = (*this).slice(id,i);
      }

   swap(newpatch);
}

void NURBSPatch::Rotate(real_t angle, real_t n[])
{
   if (Dim == 3)
   {
      Rotate2D(angle);
   }
   else
   {
      if (n == NULL)
      {
         mfem_error("NURBSPatch::Rotate : Specify an angle for a 3D rotation.");
      }

      Rotate3D(n, angle);
   }
}

void NURBSPatch::Get2DRotationMatrix(real_t angle, DenseMatrix &T)
{
   real_t s = sin(angle);
   real_t c = cos(angle);

   T.SetSize(2);
   T(0,0) = c;
   T(0,1) = -s;
   T(1,0) = s;
   T(1,1) = c;
}

void NURBSPatch::Rotate2D(real_t angle)
{
   if (Dim != 3)
   {
      mfem_error("NURBSPatch::Rotate2D : not a NURBSPatch in 2D!");
   }

   DenseMatrix T(2);
   Vector x(2), y(NULL, 2);

   Get2DRotationMatrix(angle, T);

   int size = 1;
   for (int i = 0; i < kv.Size(); i++)
   {
      size *= kv[i]->GetNCP();
   }

   for (int i = 0; i < size; i++)
   {
      y.SetData(data + i*Dim);
      x = y;
      T.Mult(x, y);
   }
}

void NURBSPatch::Get3DRotationMatrix(real_t n[], real_t angle, real_t r,
                                     DenseMatrix &T)
{
   real_t c, s, c1;
   const real_t l2 = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];
   const real_t l = sqrt(l2);

   MFEM_ASSERT(l2 > 0.0, "3D rotation axis is undefined");

   if (fabs(angle) == (real_t)(M_PI_2))
   {
      s = r*copysign(1., angle);
      c = 0.;
      c1 = -1.;
   }
   else if (fabs(angle) == (real_t)(M_PI))
   {
      s = 0.;
      c = -r;
      c1 = c - 1.;
   }
   else
   {
      s = r*sin(angle);
      c = r*cos(angle);
      c1 = c - 1.;
   }

   T.SetSize(3);

   T(0,0) =  (n[0]*n[0] + (n[1]*n[1] + n[2]*n[2])*c)/l2;
   T(0,1) = -(n[0]*n[1]*c1)/l2 - (n[2]*s)/l;
   T(0,2) = -(n[0]*n[2]*c1)/l2 + (n[1]*s)/l;
   T(1,0) = -(n[0]*n[1]*c1)/l2 + (n[2]*s)/l;
   T(1,1) =  (n[1]*n[1] + (n[0]*n[0] + n[2]*n[2])*c)/l2;
   T(1,2) = -(n[1]*n[2]*c1)/l2 - (n[0]*s)/l;
   T(2,0) = -(n[0]*n[2]*c1)/l2 - (n[1]*s)/l;
   T(2,1) = -(n[1]*n[2]*c1)/l2 + (n[0]*s)/l;
   T(2,2) =  (n[2]*n[2] + (n[0]*n[0] + n[1]*n[1])*c)/l2;
}

void NURBSPatch::Rotate3D(real_t n[], real_t angle)
{
   if (Dim != 4)
   {
      mfem_error("NURBSPatch::Rotate3D : not a NURBSPatch in 3D!");
   }

   DenseMatrix T(3);
   Vector x(3), y(NULL, 3);

   Get3DRotationMatrix(n, angle, 1., T);

   int size = 1;
   for (int i = 0; i < kv.Size(); i++)
   {
      size *= kv[i]->GetNCP();
   }

   for (int i = 0; i < size; i++)
   {
      y.SetData(data + i*Dim);
      x = y;
      T.Mult(x, y);
   }
}

int NURBSPatch::MakeUniformDegree(int degree)
{
   int maxd = degree;

   if (maxd == -1)
   {
      for (int dir = 0; dir < kv.Size(); dir++)
      {
         maxd = std::max(maxd, kv[dir]->GetOrder());
      }
   }

   for (int dir = 0; dir < kv.Size(); dir++)
   {
      if (maxd > kv[dir]->GetOrder())
      {
         DegreeElevate(dir, maxd - kv[dir]->GetOrder());
      }
   }

   return maxd;
}

NURBSPatch *Interpolate(NURBSPatch &p1, NURBSPatch &p2)
{
   if (p1.kv.Size() != p2.kv.Size() || p1.Dim != p2.Dim)
   {
      mfem_error("Interpolate(NURBSPatch &, NURBSPatch &)");
   }

   int size = 1, dim = p1.Dim;
   Array<const KnotVector *> kv(p1.kv.Size() + 1);

   for (int i = 0; i < p1.kv.Size(); i++)
   {
      if (p1.kv[i]->GetOrder() < p2.kv[i]->GetOrder())
      {
         p1.KnotInsert(i, *p2.kv[i]);
         p2.KnotInsert(i, *p1.kv[i]);
      }
      else
      {
         p2.KnotInsert(i, *p1.kv[i]);
         p1.KnotInsert(i, *p2.kv[i]);
      }
      kv[i] = p1.kv[i];
      size *= kv[i]->GetNCP();
   }

   KnotVector &nkv = *(new KnotVector(1, 2));
   nkv[0] = nkv[1] = 0.0;
   nkv[2] = nkv[3] = 1.0;
   nkv.GetElements();
   kv.Last() = &nkv;

   NURBSPatch *patch = new NURBSPatch(kv, dim);
   delete kv.Last();

   for (int i = 0; i < size; i++)
   {
      for (int d = 0; d < dim; d++)
      {
         patch->data[i*dim+d] = p1.data[i*dim+d];
         patch->data[(i+size)*dim+d] = p2.data[i*dim+d];
      }
   }

   return patch;
}

NURBSPatch *Revolve3D(NURBSPatch &patch, real_t n[], real_t ang, int times)
{
   if (patch.Dim != 4)
   {
      mfem_error("Revolve3D(NURBSPatch &, real_t [], real_t)");
   }

   int size = 1, ns;
   Array<const KnotVector *> nkv(patch.kv.Size() + 1);

   for (int i = 0; i < patch.kv.Size(); i++)
   {
      nkv[i] = patch.kv[i];
      size *= nkv[i]->GetNCP();
   }
   ns = 2*times + 1;
   KnotVector &lkv = *(new KnotVector(2, ns));
   nkv.Last() = &lkv;
   lkv[0] = lkv[1] = lkv[2] = 0.0;
   for (int i = 1; i < times; i++)
   {
      lkv[2*i+1] = lkv[2*i+2] = i;
   }
   lkv[ns] = lkv[ns+1] = lkv[ns+2] = times;
   lkv.GetElements();
   NURBSPatch *newpatch = new NURBSPatch(nkv, 4);
   delete nkv.Last();

   DenseMatrix T(3), T2(3);
   Vector u(NULL, 3), v(NULL, 3);

   NURBSPatch::Get3DRotationMatrix(n, ang, 1., T);
   real_t c = cos(ang/2);
   NURBSPatch::Get3DRotationMatrix(n, ang/2, 1./c, T2);
   T2 *= c;

   real_t *op = patch.data, *np;
   for (int i = 0; i < size; i++)
   {
      np = newpatch->data + 4*i;
      for (int j = 0; j < 4; j++)
      {
         np[j] = op[j];
      }
      for (int j = 0; j < times; j++)
      {
         u.SetData(np);
         v.SetData(np += 4*size);
         T2.Mult(u, v);
         v[3] = c*u[3];
         v.SetData(np += 4*size);
         T.Mult(u, v);
         v[3] = u[3];
      }
      op += 4;
   }

   return newpatch;
}

void NURBSPatch::SetKnotVectorsCoarse(bool c)
{
   for (int i=0; i<kv.Size(); ++i)
   {
      kv[i]->coarse = c;
   }
}

NURBSExtension::NURBSExtension(const NURBSExtension &orig)
   : mOrder(orig.mOrder), mOrders(orig.mOrders),
     NumOfKnotVectors(orig.NumOfKnotVectors),
     NumOfVertices(orig.NumOfVertices),
     NumOfElements(orig.NumOfElements),
     NumOfBdrElements(orig.NumOfBdrElements),
     NumOfDofs(orig.NumOfDofs),
     NumOfActiveVertices(orig.NumOfActiveVertices),
     NumOfActiveElems(orig.NumOfActiveElems),
     NumOfActiveBdrElems(orig.NumOfActiveBdrElems),
     NumOfActiveDofs(orig.NumOfActiveDofs),
     activeVert(orig.activeVert),
     activeElem(orig.activeElem),
     activeBdrElem(orig.activeBdrElem),
     activeDof(orig.activeDof),
     patchTopo(new Mesh(*orig.patchTopo)),
     own_topo(true),
     edge_to_knot(orig.edge_to_knot),
     knotVectors(orig.knotVectors.Size()), // knotVectors are copied in the body
     knotVectorsCompr(orig.knotVectorsCompr.Size()),
     weights(orig.weights),
     d_to_d(orig.d_to_d),
     master(orig.master),
     slave(orig.slave),
     v_meshOffsets(orig.v_meshOffsets),
     e_meshOffsets(orig.e_meshOffsets),
     f_meshOffsets(orig.f_meshOffsets),
     p_meshOffsets(orig.p_meshOffsets),
     aux_e_meshOffsets(orig.aux_e_meshOffsets),
     aux_f_meshOffsets(orig.aux_f_meshOffsets),
     v_spaceOffsets(orig.v_spaceOffsets),
     e_spaceOffsets(orig.e_spaceOffsets),
     f_spaceOffsets(orig.f_spaceOffsets),
     p_spaceOffsets(orig.p_spaceOffsets),
     aux_e_spaceOffsets(orig.aux_e_spaceOffsets),
     aux_f_spaceOffsets(orig.aux_f_spaceOffsets),
     auxEdges(orig.auxEdges),
     auxFaces(orig.auxFaces),
     el_dof(orig.el_dof ? new Table(*orig.el_dof) : NULL),
     bel_dof(orig.bel_dof ? new Table(*orig.bel_dof) : NULL),
     el_to_patch(orig.el_to_patch),
     bel_to_patch(orig.bel_to_patch),
     el_to_IJK(orig.el_to_IJK),
     bel_to_IJK(orig.bel_to_IJK),
     patches(orig.patches.Size()) // patches are copied in the body
{
   // Copy the knot vectors:
   for (int i = 0; i < knotVectors.Size(); i++)
   {
      knotVectors[i] = new KnotVector(*orig.knotVectors[i]);
   }
   CreateComprehensiveKV();

   // Copy the patches:
   for (int p = 0; p < patches.Size(); p++)
   {
      patches[p] = new NURBSPatch(*orig.patches[p]);
   }
}

NURBSExtension::NURBSExtension(std::istream &input, bool spacing, bool nc)
{
   // Read topology
   patchTopo = new Mesh;
   if (nc)
   {
      patchTopo->LoadNonconformingPatchTopo(input, edge_to_knot);
      nonconforming = true;
   }
   else
   {
      patchTopo->LoadPatchTopo(input, edge_to_knot);
   }

   own_topo = true;

   CheckPatches();
   // CheckBdrPatches();

   skip_comment_lines(input, '#');

   // Read knotvectors or patches
   string ident;
   input >> ws >> ident; // 'knotvectors' or 'patches'
   if (ident == "knotvectors")
   {
      input >> NumOfKnotVectors;
      knotVectors.SetSize(NumOfKnotVectors);
      for (int i = 0; i < NumOfKnotVectors; i++)
      {
         knotVectors[i] = new KnotVector(input);
      }

      if (spacing)  // Read spacing formulas for knotvectors
      {
         input >> ws >> ident; // 'spacing'
         MFEM_VERIFY(ident == "spacing",
                     "Spacing formula section missing from NURBS mesh file");
         int numSpacing = 0;
         input >> numSpacing;
         for (int j = 0; j < numSpacing; j++)
         {
            int ki, spacingType, numIntParam, numRealParam;
            input >> ki >> spacingType >> numIntParam >> numRealParam;

            MFEM_VERIFY(0 <= ki && ki < NumOfKnotVectors,
                        "Invalid knotvector index");
            MFEM_VERIFY(numIntParam >= 0 && numRealParam >= 0,
                        "Invalid number of parameters in KnotVector");

            Array<int> ipar(numIntParam);
            Vector dpar(numRealParam);

            for (int i=0; i<numIntParam; ++i)
            {
               input >> ipar[i];
            }

            for (int i=0; i<numRealParam; ++i)
            {
               input >> dpar[i];
            }

            const SpacingType s = (SpacingType) spacingType;
            knotVectors[ki]->spacing = GetSpacingFunction(s, ipar, dpar);
         }
      }
   }
   else if (ident == "patches")
   {
      patches.SetSize(GetNP());
      for (int p = 0; p < patches.Size(); p++)
      {
         skip_comment_lines(input, '#');
         patches[p] = new NURBSPatch(input);
      }

      NumOfKnotVectors = 0;
      for (int i = 0; i < patchTopo->GetNEdges(); i++)
         if (NumOfKnotVectors < KnotInd(i))
         {
            NumOfKnotVectors = KnotInd(i);
         }
      NumOfKnotVectors++;
      knotVectors.SetSize(NumOfKnotVectors);
      knotVectors = NULL;

      Array<int> edges, oedge;
      for (int p = 0; p < patches.Size(); p++)
      {
         if (Dimension() == 1)
         {
            if (knotVectors[KnotInd(p)] == NULL)
            {
               knotVectors[KnotInd(p)] =
                  new KnotVector(*patches[p]->GetKV(0));
            }
         }
         else if (Dimension() == 2)
         {
            patchTopo->GetElementEdges(p, edges, oedge);
            if (knotVectors[KnotInd(edges[0])] == NULL)
            {
               knotVectors[KnotInd(edges[0])] =
                  new KnotVector(*patches[p]->GetKV(0));
            }
            if (knotVectors[KnotInd(edges[1])] == NULL)
            {
               knotVectors[KnotInd(edges[1])] =
                  new KnotVector(*patches[p]->GetKV(1));
            }
         }
         else if (Dimension() == 3)
         {
            patchTopo->GetElementEdges(p, edges, oedge);
            if (knotVectors[KnotInd(edges[0])] == NULL)
            {
               knotVectors[KnotInd(edges[0])] =
                  new KnotVector(*patches[p]->GetKV(0));
            }
            if (knotVectors[KnotInd(edges[3])] == NULL)
            {
               knotVectors[KnotInd(edges[3])] =
                  new KnotVector(*patches[p]->GetKV(1));
            }
            if (knotVectors[KnotInd(edges[8])] == NULL)
            {
               knotVectors[KnotInd(edges[8])] =
                  new KnotVector(*patches[p]->GetKV(2));
            }
         }
      }
   }
   else
   {
      MFEM_ABORT("invalid section: " << ident);
   }

   CreateComprehensiveKV();

   SetOrdersFromKnotVectors();

   GenerateOffsets();
   CountElements();
   CountBdrElements();
   // NumOfVertices, NumOfElements, NumOfBdrElements, NumOfDofs

   skip_comment_lines(input, '#');

   // Check for a list of mesh elements
   if (patches.Size() == 0)
   {
      input >> ws >> ident;
   }
   if (patches.Size() == 0 && ident == "mesh_elements")
   {
      input >> NumOfActiveElems;
      activeElem.SetSize(GetGNE());
      activeElem = false;
      int glob_elem;
      for (int i = 0; i < NumOfActiveElems; i++)
      {
         input >> glob_elem;
         activeElem[glob_elem] = true;
      }

      skip_comment_lines(input, '#');
      input >> ws >> ident;
   }
   else
   {
      NumOfActiveElems = NumOfElements;
      activeElem.SetSize(NumOfElements);
      activeElem = true;
   }

   if (nc && patchTopo->ncmesh &&
       patchTopo->ncmesh->GetVertexToKnot().NumRows() > 0)
   {
      // Set map from patchTopo edges to patchTopo->ncmesh edges
      Array<int> vert;
      for (int i=0; i<patchTopo->GetNEdges(); ++i)
      {
         patchTopo->GetEdgeVertices(i, vert);
         const std::pair<int, int> vpair(vert[0], vert[1]);
         const int ncedge = v2e[vpair];

         e2nce[i] = ncedge;

         // TODO: if this is always true, we don't need e2nce or v2e.
         // Of course, this is only for the vertex_to_knot case, not the vertex_parents
         // case, assuming those 2 cases are mutually exclusive.
         MFEM_VERIFY(i == ncedge, "");
      }
   }

   GenerateActiveVertices();
   InitDofMap();
   GenerateElementDofTable();
   GenerateActiveBdrElems();
   GenerateBdrElementDofTable();

   // periodic
   if (ident == "periodic")
   {
      master.Load(input);
      slave.Load(input);

      skip_comment_lines(input, '#');
      input >> ws >> ident;
   }

   if (patches.Size() == 0)
   {
      // weights
      if (ident == "weights")
      {
         weights.Load(input, GetNDof());
      }
      else // e.g. ident = "unitweights" or "autoweights"
      {
         weights.SetSize(GetNDof());
         weights = 1.0;
      }
   }

   // periodic
   ConnectBoundaries();
}

NURBSExtension::NURBSExtension(NURBSExtension *parent, int newOrder)
{
   patchTopo = parent->patchTopo;
   own_topo = false;

   parent->edge_to_knot.Copy(edge_to_knot);

   NumOfKnotVectors = parent->GetNKV();
   knotVectors.SetSize(NumOfKnotVectors);
   knotVectorsCompr.SetSize(parent->GetNP()*parent->Dimension());
   const Array<int> &pOrders = parent->GetOrders();
   for (int i = 0; i < NumOfKnotVectors; i++)
   {
      if (newOrder > pOrders[i])
      {
         knotVectors[i] =
            parent->GetKnotVector(i)->DegreeElevate(newOrder - pOrders[i]);
      }
      else
      {
         knotVectors[i] = new KnotVector(*parent->GetKnotVector(i));
      }
   }
   CreateComprehensiveKV();

   // copy some data from parent
   NumOfElements    = parent->NumOfElements;
   NumOfBdrElements = parent->NumOfBdrElements;

   SetOrdersFromKnotVectors();

   GenerateOffsets(); // dof offsets will be different from parent

   NumOfActiveVertices = parent->NumOfActiveVertices;
   NumOfActiveElems    = parent->NumOfActiveElems;
   NumOfActiveBdrElems = parent->NumOfActiveBdrElems;
   parent->activeVert.Copy(activeVert);
   InitDofMap();
   parent->activeElem.Copy(activeElem);
   parent->activeBdrElem.Copy(activeBdrElem);

   GenerateElementDofTable();
   GenerateBdrElementDofTable();

   weights.SetSize(GetNDof());
   weights = 1.0;

   // periodic
   parent->master.Copy(master);
   parent->slave.Copy(slave);
   ConnectBoundaries();
}

NURBSExtension::NURBSExtension(NURBSExtension *parent,
                               const Array<int> &newOrders, Mode mode)
   : mode(mode)
{
   newOrders.Copy(mOrders);
   SetOrderFromOrders();

   patchTopo = parent->patchTopo;
   own_topo = false;

   parent->edge_to_knot.Copy(edge_to_knot);

   NumOfKnotVectors = parent->GetNKV();
   MFEM_VERIFY(mOrders.Size() == NumOfKnotVectors, "invalid newOrders array");
   knotVectors.SetSize(NumOfKnotVectors);
   const Array<int> &pOrders = parent->GetOrders();

   for (int i = 0; i < NumOfKnotVectors; i++)
   {
      if (mOrders[i] > pOrders[i])
      {
         knotVectors[i] =
            parent->GetKnotVector(i)->DegreeElevate(mOrders[i] - pOrders[i]);
      }
      else
      {
         knotVectors[i] = new KnotVector(*parent->GetKnotVector(i));
      }
   }
   CreateComprehensiveKV();

   // copy some data from parent
   NumOfElements    = parent->NumOfElements;
   NumOfBdrElements = parent->NumOfBdrElements;

   GenerateOffsets(); // dof offsets will be different from parent

   NumOfActiveVertices = parent->NumOfActiveVertices;
   NumOfActiveElems    = parent->NumOfActiveElems;
   NumOfActiveBdrElems = parent->NumOfActiveBdrElems;
   parent->activeVert.Copy(activeVert);
   InitDofMap();
   parent->activeElem.Copy(activeElem);
   parent->activeBdrElem.Copy(activeBdrElem);

   GenerateElementDofTable();
   GenerateBdrElementDofTable();

   weights.SetSize(GetNDof());
   weights = 1.0;

   parent->master.Copy(master);
   parent->slave.Copy(slave);
   ConnectBoundaries();
}

NURBSExtension::NURBSExtension(Mesh *mesh_array[], int num_pieces)
{
   NURBSExtension *parent = mesh_array[0]->NURBSext;

   if (!parent->own_topo)
   {
      mfem_error("NURBSExtension::NURBSExtension :\n"
                 "  parent does not own the patch topology!");
   }
   patchTopo = parent->patchTopo;
   own_topo = true;
   parent->own_topo = false;

   parent->edge_to_knot.Copy(edge_to_knot);

   parent->GetOrders().Copy(mOrders);
   mOrder = parent->GetOrder();

   NumOfKnotVectors = parent->GetNKV();
   knotVectors.SetSize(NumOfKnotVectors);
   for (int i = 0; i < NumOfKnotVectors; i++)
   {
      knotVectors[i] = new KnotVector(*parent->GetKnotVector(i));
   }
   CreateComprehensiveKV();

   GenerateOffsets();
   CountElements();
   CountBdrElements();

   // assuming the meshes define a partitioning of all the elements
   NumOfActiveElems = NumOfElements;
   activeElem.SetSize(NumOfElements);
   activeElem = true;

   GenerateActiveVertices();
   InitDofMap();
   GenerateElementDofTable();
   GenerateActiveBdrElems();
   GenerateBdrElementDofTable();

   weights.SetSize(GetNDof());
   MergeWeights(mesh_array, num_pieces);
}

NURBSExtension::~NURBSExtension()
{
   if (patches.Size() == 0)
   {
      delete bel_dof;
      delete el_dof;
   }

   for (int i = 0; i < knotVectors.Size(); i++)
   {
      delete knotVectors[i];
   }

   for (int i = 0; i < knotVectorsCompr.Size(); i++)
   {
      delete knotVectorsCompr[i];
   }

   for (int i = 0; i < patches.Size(); i++)
   {
      delete patches[i];
   }

   if (own_topo)
   {
      delete patchTopo;
   }
}

void NURBSExtension::Print(std::ostream &os, const std::string &comments) const
{
   Array<int> kvSpacing;
   if (patches.Size() == 0)
   {
      for (int i = 0; i < NumOfKnotVectors; i++)
      {
         if (knotVectors[i]->spacing) { kvSpacing.Append(i); }
      }
   }

   const int version = kvSpacing.Size() > 0 ? 11 : 10;  // v1.0 or v1.1
   if (patchTopo->ncmesh)
   {
      // TODO: include version?
      patchTopo->ncmesh->Print(os, comments, true);
      patchTopo->PrintTopoEdges(os, edge_to_knot, true);
   }
   else
   {
      patchTopo->PrintTopo(os, edge_to_knot, version, comments);
   }

   if (patches.Size() == 0)
   {
      os << "\nknotvectors\n" << NumOfKnotVectors << '\n';
      for (int i = 0; i < NumOfKnotVectors; i++)
      {
         knotVectors[i]->Print(os);
      }

      if (kvSpacing.Size() > 0)
      {
         os << "\nspacing\n" << kvSpacing.Size() << '\n';
         for (auto kv : kvSpacing)
         {
            os << kv << " ";
            knotVectors[kv]->spacing->Print(os);
         }
      }

      if (NumOfActiveElems < NumOfElements)
      {
         os << "\nmesh_elements\n" << NumOfActiveElems << '\n';
         for (int i = 0; i < NumOfElements; i++)
            if (activeElem[i])
            {
               os << i << '\n';
            }
      }

      os << "\nweights\n";
      weights.Print(os, 1);
   }
   else
   {
      os << "\npatches\n";
      for (int p = 0; p < patches.Size(); p++)
      {
         os << "\n# patch " << p << "\n\n";
         patches[p]->Print(os);
      }
   }
}

void NURBSExtension::PrintCharacteristics(std::ostream &os) const
{
   os <<
      "NURBS Mesh entity sizes:\n"
      "Dimension           = " << Dimension() << "\n"
      "Unique Orders       = ";
   Array<int> unique_orders(mOrders);
   unique_orders.Sort();
   unique_orders.Unique();
   unique_orders.Print(os, unique_orders.Size());
   os <<
      "NumOfKnotVectors    = " << GetNKV() << "\n"
      "NumOfPatches        = " << GetNP() << "\n"
      "NumOfBdrPatches     = " << GetNBP() << "\n"
      "NumOfVertices       = " << GetGNV() << "\n"
      "NumOfElements       = " << GetGNE() << "\n"
      "NumOfBdrElements    = " << GetGNBE() << "\n"
      "NumOfDofs           = " << GetNTotalDof() << "\n"
      "NumOfActiveVertices = " << GetNV() << "\n"
      "NumOfActiveElems    = " << GetNE() << "\n"
      "NumOfActiveBdrElems = " << GetNBE() << "\n"
      "NumOfActiveDofs     = " << GetNDof() << '\n';
   for (int i = 0; i < NumOfKnotVectors; i++)
   {
      os << ' ' << i + 1 << ") ";
      knotVectors[i]->Print(os);
   }
   os << endl;
}

void NURBSExtension::PrintFunctions(const char *basename, int samples) const
{
   std::ofstream os;
   for (int i = 0; i < NumOfKnotVectors; i++)
   {
      std::ostringstream filename;
      filename << basename << "_" << i << ".dat";
      os.open(filename.str().c_str());
      knotVectors[i]->PrintFunctions(os,samples);
      os.close();
   }
}

void NURBSExtension::InitDofMap()
{
   master.SetSize(0);
   slave.SetSize(0);
   d_to_d.SetSize(0);
}

void NURBSExtension::ConnectBoundaries(Array<int> &bnds0, Array<int> &bnds1)
{
   bnds0.Copy(master);
   bnds1.Copy(slave);
   ConnectBoundaries();
}

void NURBSExtension::ConnectBoundaries()
{
   if (master.Size() != slave.Size())
   {
      mfem_error("NURBSExtension::ConnectBoundaries() boundary lists not of equal size");
   }
   if (master.Size() == 0 ) { return; }

   // Initialize d_to_d
   d_to_d.SetSize(NumOfDofs);
   for (int i = 0; i < NumOfDofs; i++) { d_to_d[i] = i; }

   // Connect
   for (int i = 0; i < master.Size(); i++)
   {
      int bnd0 = -1, bnd1 = -1;
      for (int b = 0; b < GetNBP(); b++)
      {
         if (master[i] == patchTopo->GetBdrAttribute(b)) { bnd0 = b; }
         if (slave[i]== patchTopo->GetBdrAttribute(b)) { bnd1  = b; }
      }
      MFEM_VERIFY(bnd0  != -1,"Bdr 0 not found");
      MFEM_VERIFY(bnd1  != -1,"Bdr 1 not found");

      if (Dimension() == 1)
      {
         ConnectBoundaries1D(bnd0, bnd1);
      }
      else if (Dimension() == 2)
      {
         ConnectBoundaries2D(bnd0, bnd1);
      }
      else
      {
         ConnectBoundaries3D(bnd0, bnd1);
      }
   }

   // Clean d_to_d
   Array<int> tmp(d_to_d.Size()+1);
   tmp = 0;

   for (int i = 0; i < d_to_d.Size(); i++)
   {
      tmp[d_to_d[i]] = 1;
   }

   int cnt = 0;
   for (int i = 0; i < tmp.Size(); i++)
   {
      if (tmp[i] == 1) { tmp[i] = cnt++; }
   }
   NumOfDofs = cnt;

   for (int i = 0; i < d_to_d.Size(); i++)
   {
      d_to_d[i] = tmp[d_to_d[i]];
   }

   // Finalize
   if (el_dof) { delete el_dof; }
   if (bel_dof) { delete bel_dof; }
   GenerateElementDofTable();
   GenerateBdrElementDofTable();
}

void NURBSExtension::ConnectBoundaries1D(int bnd0, int bnd1)
{
   NURBSPatchMap p2g0(this);
   NURBSPatchMap p2g1(this);

   int okv0[1],okv1[1];
   const KnotVector *kv0[1],*kv1[1];

   p2g0.SetBdrPatchDofMap(bnd0, kv0, okv0);
   p2g1.SetBdrPatchDofMap(bnd1, kv1, okv1);

   d_to_d[p2g0(0)] = d_to_d[p2g1(0)];
}

void NURBSExtension::ConnectBoundaries2D(int bnd0, int bnd1)
{
   NURBSPatchMap p2g0(this);
   NURBSPatchMap p2g1(this);

   int okv0[1],okv1[1];
   const KnotVector *kv0[1],*kv1[1];

   p2g0.SetBdrPatchDofMap(bnd0, kv0, okv0);
   p2g1.SetBdrPatchDofMap(bnd1, kv1, okv1);

   int nx = p2g0.nx();
   int nks0 = kv0[0]->GetNKS();

#ifdef MFEM_DEBUG
   bool compatible = true;
   if (p2g0.nx() != p2g1.nx()) { compatible = false; }
   if (kv0[0]->GetNKS() != kv1[0]->GetNKS()) { compatible = false; }
   if (kv0[0]->GetOrder() != kv1[0]->GetOrder()) { compatible = false; }

   if (!compatible)
   {
      mfem::out<<p2g0.nx()<<" "<<p2g1.nx()<<endl;
      mfem::out<<kv0[0]->GetNKS()<<" "<<kv1[0]->GetNKS()<<endl;
      mfem::out<<kv0[0]->GetOrder()<<" "<<kv1[0]->GetOrder()<<endl;
      mfem_error("NURBS boundaries not compatible");
   }
#endif

   for (int i = 0; i < nks0; i++)
   {
      if (kv0[0]->isElement(i))
      {
         if (!kv1[0]->isElement(i)) { mfem_error("isElement does not match"); }
         for (int ii = 0; ii <= kv0[0]->GetOrder(); ii++)
         {
            int ii0 = (okv0[0] >= 0) ? (i+ii) : (nx-i-ii);
            int ii1 = (okv1[0] >= 0) ? (i+ii) : (nx-i-ii);

            d_to_d[p2g0(ii0)] = d_to_d[p2g1(ii1)];
         }

      }
   }
}

void NURBSExtension::ConnectBoundaries3D(int bnd0, int bnd1)
{
   NURBSPatchMap p2g0(this);
   NURBSPatchMap p2g1(this);

   int okv0[2],okv1[2];
   const KnotVector *kv0[2],*kv1[2];

   p2g0.SetBdrPatchDofMap(bnd0, kv0, okv0);
   p2g1.SetBdrPatchDofMap(bnd1, kv1, okv1);

   int nx = p2g0.nx();
   int ny = p2g0.ny();

   int nks0 = kv0[0]->GetNKS();
   int nks1 = kv0[1]->GetNKS();

#ifdef MFEM_DEBUG
   bool compatible = true;
   if (p2g0.nx() != p2g1.nx()) { compatible = false; }
   if (p2g0.ny() != p2g1.ny()) { compatible = false; }

   if (kv0[0]->GetNKS() != kv1[0]->GetNKS()) { compatible = false; }
   if (kv0[1]->GetNKS() != kv1[1]->GetNKS()) { compatible = false; }

   if (kv0[0]->GetOrder() != kv1[0]->GetOrder()) { compatible = false; }
   if (kv0[1]->GetOrder() != kv1[1]->GetOrder()) { compatible = false; }

   if (!compatible)
   {
      mfem::out<<p2g0.nx()<<" "<<p2g1.nx()<<endl;
      mfem::out<<p2g0.ny()<<" "<<p2g1.ny()<<endl;

      mfem::out<<kv0[0]->GetNKS()<<" "<<kv1[0]->GetNKS()<<endl;
      mfem::out<<kv0[1]->GetNKS()<<" "<<kv1[1]->GetNKS()<<endl;

      mfem::out<<kv0[0]->GetOrder()<<" "<<kv1[0]->GetOrder()<<endl;
      mfem::out<<kv0[1]->GetOrder()<<" "<<kv1[1]->GetOrder()<<endl;
      mfem_error("NURBS boundaries not compatible");
   }
#endif

   for (int j = 0; j < nks1; j++)
   {
      if (kv0[1]->isElement(j))
      {
         if (!kv1[1]->isElement(j)) { mfem_error("isElement does not match #1"); }
         for (int i = 0; i < nks0; i++)
         {
            if (kv0[0]->isElement(i))
            {
               if (!kv1[0]->isElement(i)) { mfem_error("isElement does not match #0"); }
               for (int jj = 0; jj <= kv0[1]->GetOrder(); jj++)
               {
                  int jj0 = (okv0[1] >= 0) ? (j+jj) : (ny-j-jj);
                  int jj1 = (okv1[1] >= 0) ? (j+jj) : (ny-j-jj);

                  for (int ii = 0; ii <= kv0[0]->GetOrder(); ii++)
                  {
                     int ii0 = (okv0[0] >= 0) ? (i+ii) : (nx-i-ii);
                     int ii1 = (okv1[0] >= 0) ? (i+ii) : (nx-i-ii);

                     d_to_d[p2g0(ii0,jj0)] = d_to_d[p2g1(ii1,jj1)];
                  }
               }
            }
         }
      }
   }
}

void NURBSExtension::GenerateActiveVertices()
{
   int vert[8], nv, g_el, nx, ny, nz, dim = Dimension();

   NURBSPatchMap p2g(this);
   const KnotVector *kv[3];

   g_el = 0;
   activeVert.SetSize(GetGNV());
   activeVert = -1;
   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchVertexMap(p, kv);

      nx = p2g.nx();
      ny = (dim >= 2) ? p2g.ny() : 1;
      nz = (dim == 3) ? p2g.nz() : 1;

      for (int k = 0; k < nz; k++)
      {
         for (int j = 0; j < ny; j++)
         {
            for (int i = 0; i < nx; i++)
            {
               if (activeElem[g_el])
               {
                  if (dim == 1)
                  {
                     vert[0] = p2g(i  );
                     vert[1] = p2g(i+1);
                     nv = 2;
                  }
                  else if (dim == 2)
                  {
                     vert[0] = p2g(i,  j  );
                     vert[1] = p2g(i+1,j  );
                     vert[2] = p2g(i+1,j+1);
                     vert[3] = p2g(i,  j+1);
                     nv = 4;
                  }
                  else
                  {
                     vert[0] = p2g(i,  j,  k);
                     vert[1] = p2g(i+1,j,  k);
                     vert[2] = p2g(i+1,j+1,k);
                     vert[3] = p2g(i,  j+1,k);

                     vert[4] = p2g(i,  j,  k+1);
                     vert[5] = p2g(i+1,j,  k+1);
                     vert[6] = p2g(i+1,j+1,k+1);
                     vert[7] = p2g(i,  j+1,k+1);
                     nv = 8;
                  }

                  for (int v = 0; v < nv; v++)
                  {
                     activeVert[vert[v]] = 1;
                  }
               }
               g_el++;
            }
         }
      }
   }

   NumOfActiveVertices = 0;
   for (int i = 0; i < GetGNV(); i++)
      if (activeVert[i] == 1)
      {
         activeVert[i] = NumOfActiveVertices++;
      }
}

void NURBSExtension::GenerateActiveBdrElems()
{
   int dim = Dimension();
   Array<KnotVector *> kv(dim);

   activeBdrElem.SetSize(GetGNBE());
   if (GetGNE() == GetNE())
   {
      activeBdrElem = true;
      NumOfActiveBdrElems = GetGNBE();
      return;
   }
   activeBdrElem = false;
   NumOfActiveBdrElems = 0;
   // the mesh will generate the actual boundary including boundary
   // elements that are not on boundary patches. we use this for
   // visualization of processor boundaries

   // TODO: generate actual boundary?
}


void NURBSExtension::MergeWeights(Mesh *mesh_array[], int num_pieces)
{
   Array<int> lelem_elem;

   for (int i = 0; i < num_pieces; i++)
   {
      NURBSExtension *lext = mesh_array[i]->NURBSext;

      lext->GetElementLocalToGlobal(lelem_elem);

      for (int lel = 0; lel < lext->GetNE(); lel++)
      {
         int gel = lelem_elem[lel];

         int nd = el_dof->RowSize(gel);
         int *gdofs = el_dof->GetRow(gel);
         int *ldofs = lext->el_dof->GetRow(lel);
         for (int j = 0; j < nd; j++)
         {
            weights(gdofs[j]) = lext->weights(ldofs[j]);
         }
      }
   }
}

void NURBSExtension::MergeGridFunctions(
   GridFunction *gf_array[], int num_pieces, GridFunction &merged)
{
   FiniteElementSpace *gfes = merged.FESpace();
   Array<int> lelem_elem, dofs;
   Vector lvec;

   for (int i = 0; i < num_pieces; i++)
   {
      FiniteElementSpace *lfes = gf_array[i]->FESpace();
      NURBSExtension *lext = lfes->GetMesh()->NURBSext;

      lext->GetElementLocalToGlobal(lelem_elem);

      for (int lel = 0; lel < lext->GetNE(); lel++)
      {
         lfes->GetElementVDofs(lel, dofs);
         gf_array[i]->GetSubVector(dofs, lvec);

         gfes->GetElementVDofs(lelem_elem[lel], dofs);
         merged.SetSubVector(dofs, lvec);
      }
   }
}

void NURBSExtension::CheckPatches()
{
   if (Dimension() == 1 ) { return; }

   Array<int> edges, oedge;

   for (int p = 0; p < GetNP(); p++)
   {
      patchTopo->GetElementEdges(p, edges, oedge);

      for (int i = 0; i < edges.Size(); i++)
      {
         edges[i] = edge_to_knot[edges[i]];
         if (oedge[i] < 0)
         {
            edges[i] = -1 - edges[i];
         }
      }

      if ((Dimension() == 2 &&
           (edges[0] != -1 - edges[2] || edges[1] != -1 - edges[3])) ||

          (Dimension() == 3 &&
           (edges[0] != edges[2] || edges[0] != edges[4] ||
            edges[0] != edges[6] || edges[1] != edges[3] ||
            edges[1] != edges[5] || edges[1] != edges[7] ||
            edges[8] != edges[9] || edges[8] != edges[10] ||
            edges[8] != edges[11])))
      {
         mfem::err << "NURBSExtension::CheckPatch (patch = " << p
                   << ")\n  Inconsistent edge-to-knot mapping!\n";
         mfem_error();
      }
   }
}

void NURBSExtension::CheckBdrPatches()
{
   Array<int> edges;
   Array<int> oedge;

   for (int p = 0; p < GetNBP(); p++)
   {
      patchTopo->GetBdrElementEdges(p, edges, oedge);

      for (int i = 0; i < edges.Size(); i++)
      {
         edges[i] = edge_to_knot[edges[i]];
         if (oedge[i] < 0)
         {
            edges[i] = -1 - edges[i];
         }
      }

      if ((Dimension() == 2 && (edges[0] < 0)) ||
          (Dimension() == 3 && (edges[0] < 0 || edges[1] < 0)))
      {
         mfem::err << "NURBSExtension::CheckBdrPatch (boundary patch = "
                   << p << ") : Bad orientation!\n";
         mfem_error();
      }
   }
}

void NURBSExtension::CheckKVDirection(int p, Array <int> &kvdir)
{
   // patchTopo->GetElementEdges is not yet implemented for 1D
   MFEM_VERIFY(Dimension()>1, "1D not yet implemented.");

   kvdir.SetSize(Dimension());
   kvdir = 0;

   Array<int> patchvert, edges, orient, edgevert;

   patchTopo->GetElementVertices(p, patchvert);

   patchTopo->GetElementEdges(p, edges, orient);

   // Compare the vertices of the patches with the vertices of the knotvectors of knot2dge
   // Based on the match the orientation will be a 1 or a -1
   // -1: direction is flipped
   //  1: direction is not flipped

   for (int i = 0; i < edges.Size(); i++)
   {
      // First side
      patchTopo->GetEdgeVertices(edges[i], edgevert);
      if (edgevert[0] == patchvert[0]  && edgevert[1] == patchvert[1])
      {
         kvdir[0] = 1;
      }

      if (edgevert[0] == patchvert[1]  && edgevert[1] == patchvert[0])
      {
         kvdir[0] = -1;
      }

      // Second side
      if (edgevert[0] == patchvert[1]  && edgevert[1] == patchvert[2])
      {
         kvdir[1] = 1;
      }

      if (edgevert[0] == patchvert[2]  && edgevert[1] == patchvert[1])
      {
         kvdir[1] = -1;
      }
   }

   if (Dimension() == 3)
   {
      // Third side
      for (int i = 0; i < edges.Size(); i++)
      {
         patchTopo->GetEdgeVertices(edges[i], edgevert);

         if (edgevert[0] == patchvert[0]  && edgevert[1] == patchvert[4])
         {
            kvdir[2] = 1;
         }

         if (edgevert[0] == patchvert[4]  && edgevert[1] == patchvert[0])
         {
            kvdir[2] = -1;
         }
      }
   }

   MFEM_VERIFY(kvdir.Find(0) == -1, "Could not find direction of knotvector.");
}

void NURBSExtension::CreateComprehensiveKV()
{
   Array<int> edges, orient, kvdir;
   Array<int> e(Dimension());

   // 1D: comprehensive and unique KV are the same
   if (Dimension() == 1)
   {
      knotVectorsCompr.SetSize(GetNKV());
      for (int i = 0; i < GetNKV(); i++)
      {
         knotVectorsCompr[i] = new KnotVector(*(KnotVec(i)));
      }
      return;
   }
   else if (Dimension() == 2)
   {
      knotVectorsCompr.SetSize(GetNP()*Dimension());
      e[0] = 0;
      e[1] = 1;
   }
   else if (Dimension() == 3)
   {
      knotVectorsCompr.SetSize(GetNP()*Dimension());
      e[0] = 0;
      e[1] = 3;
      e[2] = 8;
   }

   for (int p = 0; p < GetNP(); p++)
   {
      CheckKVDirection(p, kvdir);

      patchTopo->GetElementEdges(p, edges, orient);

      for (int d = 0; d < Dimension(); d++)
      {
         // Indices in unique and comprehensive sets of the KnotVector
         int iun = edges[e[d]];
         int icomp = Dimension()*p+d;

         knotVectorsCompr[icomp] = new KnotVector(*(KnotVec(iun)));

         if (kvdir[d] == -1) {knotVectorsCompr[icomp]->Flip();}
      }
   }

   MFEM_VERIFY(ConsistentKVSets(), "Mismatch in KnotVectors");
}

void NURBSExtension::UpdateUniqueKV()
{
   Array<int> e(Dimension());

   // 1D: comprehensive and unique KV are the same
   if (Dimension() == 1)
   {
      for (int i = 0; i < GetNKV(); i++)
      {
         *(KnotVec(i)) = *(knotVectorsCompr[i]);
      }
      return;
   }
   else if (Dimension() == 2)
   {
      e[0] = 0;
      e[1] = 1;
   }
   else if (Dimension() == 3)
   {
      e[0] = 0;
      e[1] = 3;
      e[2] = 8;
   }

   for (int p = 0; p < GetNP(); p++)
   {
      Array<int> edges, orient, kvdir;

      patchTopo->GetElementEdges(p, edges, orient);
      CheckKVDirection(p, kvdir);

      for ( int d = 0; d < Dimension(); d++)
      {
         bool flip = false;
         if (kvdir[d] == -1) {flip = true;}

         // Indices in unique and comprehensive sets of the KnotVector
         int iun = edges[e[d]];
         int icomp = Dimension()*p+d;

         // Check if difference in order
         int o1 = KnotVec(iun)->GetOrder();
         int o2 = knotVectorsCompr[icomp]->GetOrder();
         int diffo = abs(o1 - o2);

         if (diffo)
         {
            // Update reduced set of knotvectors
            *(KnotVec(iun)) = *(knotVectorsCompr[icomp]);

            // Give correct direction to unique knotvector.
            if (flip) { KnotVec(iun)->Flip(); }
         }

         // Check if difference between knots
         Vector diffknot;

         if (flip) { knotVectorsCompr[icomp]->Flip(); }

         KnotVec(iun)->Difference(*(knotVectorsCompr[icomp]), diffknot);

         if (flip) { knotVectorsCompr[icomp]->Flip(); }

         if (diffknot.Size() > 0)
         {
            // Update reduced set of knotvectors
            *(KnotVec(iun)) = *(knotVectorsCompr[icomp]);

            // Give correct direction to unique knotvector.
            if (flip) {KnotVec(iun)->Flip();}
         }
      }
   }

   MFEM_VERIFY(ConsistentKVSets(), "Mismatch in KnotVectors");
}

bool NURBSExtension::ConsistentKVSets()
{
   // patchTopo->GetElementEdges is not yet implemented for 1D
   MFEM_VERIFY(Dimension() > 1, "1D not yet implemented.");

   Array<int> edges, orient, kvdir;
   Vector diff;

   Array<int> e(Dimension());

   e[0] = 0;

   if (Dimension() == 2)
   {
      e[1] = 1;
   }
   else if (Dimension() == 3)
   {
      e[1] = 3;
      e[2] = 8;
   }

   for (int p = 0; p < GetNP(); p++)
   {
      patchTopo->GetElementEdges(p, edges, orient);

      CheckKVDirection(p, kvdir);

      for (int d = 0; d < Dimension(); d++)
      {
         bool flip = false;
         if (kvdir[d] == -1) {flip = true;}

         // Indices in unique and comprehensive sets of the KnotVector
         int iun = edges[e[d]];
         int icomp = Dimension()*p+d;

         // Check if KnotVectors are of equal order
         int o1 = KnotVec(iun)->GetOrder();
         int o2 = knotVectorsCompr[icomp]->GetOrder();
         int diffo = abs(o1 - o2);

         if (diffo)
         {
            mfem::out << "\norder of knotVectorsCompr " << d << " of patch " << p;
            mfem::out << " does not agree with knotVectors " << KnotInd(iun) << "\n";
            return false;
         }

         // Check if Knotvectors have the same knots
         if (flip) {knotVectorsCompr[icomp]->Flip();}

         KnotVec(iun)->Difference(*(knotVectorsCompr[icomp]), diff);

         if (flip) {knotVectorsCompr[icomp]->Flip();}

         if (diff.Size() > 0)
         {
            mfem::out << "\nknotVectorsCompr " << d << " of patch " << p;
            mfem::out << " does not agree with knotVectors " << KnotInd(iun) << "\n";
            return false;
         }
      }
   }
   return true;
}

void NURBSExtension::GetPatchKnotVectors(int p, Array<KnotVector *> &kv)
{
   Array<int> edges, orient;

   kv.SetSize(Dimension());

   if (Dimension() == 1)
   {
      kv[0] = knotVectorsCompr[Dimension()*p];
   }
   else if (Dimension() == 2)
   {
      kv[0] = knotVectorsCompr[Dimension()*p];
      kv[1] = knotVectorsCompr[Dimension()*p + 1];
   }
   else
   {
      kv[0] = knotVectorsCompr[Dimension()*p];
      kv[1] = knotVectorsCompr[Dimension()*p + 1];
      kv[2] = knotVectorsCompr[Dimension()*p + 2];
   }
}

void NURBSExtension::GetPatchKnotVectors(int p, Array<const KnotVector *> &kv)
const
{
   kv.SetSize(Dimension());

   if (Dimension() == 1)
   {
      kv[0] = knotVectorsCompr[Dimension()*p];
   }
   else if (Dimension() == 2)
   {
      kv[0] = knotVectorsCompr[Dimension()*p];
      kv[1] = knotVectorsCompr[Dimension()*p + 1];
   }
   else
   {
      kv[0] = knotVectorsCompr[Dimension()*p];
      kv[1] = knotVectorsCompr[Dimension()*p + 1];
      kv[2] = knotVectorsCompr[Dimension()*p + 2];
   }
}

void NURBSExtension::GetBdrPatchKnotVectors(int bp, Array<KnotVector *> &kv)
{
   Array<int> edges;
   Array<int> orient;

   kv.SetSize(Dimension() - 1);

   if (Dimension() == 2)
   {
      patchTopo->GetBdrElementEdges(bp, edges, orient);
      kv[0] = KnotVec(edges[0]);
   }
   else if (Dimension() == 3)
   {
      patchTopo->GetBdrElementEdges(bp, edges, orient);
      kv[0] = KnotVec(edges[0]);
      kv[1] = KnotVec(edges[1]);
   }
}

void NURBSExtension::GetBdrPatchKnotVectors(
   int bp, Array<const KnotVector *> &kv) const
{
   Array<int> edges;
   Array<int> orient;

   kv.SetSize(Dimension() - 1);

   if (Dimension() == 2)
   {
      patchTopo->GetBdrElementEdges(bp, edges, orient);
      kv[0] = KnotVec(edges[0]);
   }
   else if (Dimension() == 3)
   {
      patchTopo->GetBdrElementEdges(bp, edges, orient);
      kv[0] = KnotVec(edges[0]);
      kv[1] = KnotVec(edges[1]);
   }
}

void NURBSExtension::SetOrderFromOrders()
{
   MFEM_VERIFY(mOrders.Size() > 0, "");
   mOrder = mOrders[0];
   for (int i = 1; i < mOrders.Size(); i++)
   {
      if (mOrders[i] != mOrder)
      {
         mOrder = NURBSFECollection::VariableOrder;
         return;
      }
   }
}

void NURBSExtension::SetOrdersFromKnotVectors()
{
   mOrders.SetSize(NumOfKnotVectors);
   for (int i = 0; i < NumOfKnotVectors; i++)
   {
      mOrders[i] = knotVectors[i]->GetOrder();
   }
   SetOrderFromOrders();
}

int GetFaceOrientation(const Mesh *mesh, const int face,
                       const Array<int> & verts);

bool Reorder2D(int n1, int n2, int ori, std::vector<int> & s0);

int GetNCPperEdge(const KnotVector *kv)
{
   const int ne = kv->GetNE();

   // Total number of CP on edge, excluding vertex CP.
   const int totalEdgeCP = kv->GetNCP() - 2 - ne + 1;
   const int perEdgeCP = totalEdgeCP / ne;

   MFEM_VERIFY(perEdgeCP * ne == totalEdgeCP, "");

   return perEdgeCP;
}

void NURBSExtension::FindAdditionalSlaveAndAuxiliaryFaces(
   std::map<std::pair<int, int>, int> & v2f,
   std::set<int> & addParentFaces,
   std::vector<FacePairInfo> & facePairs)
{
   const int ipair[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

   for (int f=0; f<patchTopo->GetNFaces(); ++f)
   {
      if (masterFaces.find(f) != masterFaces.end())
      {
         continue;   // Already a master face
      }

      Array<int> edges, ori, verts;
      patchTopo->GetFaceEdges(f, edges, ori);
      patchTopo->GetFaceVertices(f, verts);

      MFEM_VERIFY(edges.Size() == 4 && verts.Size() == 4, "");

      const int fn1 = KnotVec(edges[0])->GetNE();
      const int fn2 = KnotVec(edges[1])->GetNE();

      // Loop over the 2 pairs of opposite sides
      for (int p=0; p<2; ++p)  // Pair p
      {
         Array<int> oppEdges(2);
         const int sideEdge0 = edges[1 - p];
         bool eitherMaster = false;
         bool bothMaster = true;
         for (int s=0; s<2; ++s)
         {
            oppEdges[s] = edges[p + 2*s];

            bool isTrueMasterEdge = false;
            if (masterEdges.count(oppEdges[s]) > 0)
            {
               const int mid = masterEdgeToId.at(oppEdges[s]);
               if (masterEdgeSlaves[mid].size() > 0)
               {
                  isTrueMasterEdge = true;
               }
            }

            if (isTrueMasterEdge)
            {
               eitherMaster = true;
            }
            else
            {
               bothMaster = false;
            }
         }

         if (bothMaster)
         {
            // Possibly define auxiliary and/or slave faces on this face

            // Check for auxiliary and slave edges
            std::vector<Array<int>> sideAuxEdges(2);
            std::vector<Array<int>> sideSlaveEdges(2);
            for (int s=0; s<2; ++s)
            {
               const int mid = masterEdgeToId.at(oppEdges[s]);
               for (auto edge : masterEdgeSlaves[mid])
               {
                  if (edge < 0)
                  {
                     sideAuxEdges[s].Append(-1 - edge);
                  }
                  else
                  {
                     sideSlaveEdges[s].Append(edge);
                  }
               }
            }

            const bool hasAux = sideAuxEdges[0].Size() > 0;
            const bool hasSlave = sideSlaveEdges[0].Size() > 0;

            // Find patchTopo vertices in the interior of each side

            if (hasAux || hasSlave)
            {
               std::vector<Array<int>> edgeV(2);
               std::vector<Array<int>> edgeE(2);
               std::vector<Array<int>> edgeVki(2);

               Array2D<int> epv(2,2);

               for (int s=0; s<2; ++s)
               {
                  const int mid = masterEdgeToId.at(oppEdges[s]);
                  const int nes = masterEdgeSlaves[mid].size();
                  MFEM_VERIFY(masterEdgeVerts[mid].size() + 1 == nes, "");

                  // Vertices in masterEdgeVerts[mid] are ordered starting
                  // from the master edge endpoint with lower vertex index.

                  Array<int> everts;
                  patchTopo->GetEdgeVertices(oppEdges[s], everts);

                  for (int i=0; i<2; ++i)
                  {
                     const int v = everts[i];
                     // Find the index of vertex v in verts.
                     const int vid = verts.Find(v);
                     MFEM_ASSERT(vid >= 0, "");
                     epv(s,i) = vid;
                  }

                  edgeV[s].Append(everts[0]);
                  edgeVki[s].Append(0);

                  MFEM_VERIFY(masterEdgeVerts[mid].size() == masterEdgeKI[mid].size(), "");
                  for (int i=0; i<masterEdgeVerts[mid].size(); ++i)
                  {
                     edgeV[s].Append(masterEdgeVerts[mid][i]);
                     edgeVki[s].Append(masterEdgeKI[mid][i]);
                  }

                  const int nelem = KnotVec(oppEdges[s])->GetNE();

                  edgeV[s].Append(everts[1]);
                  edgeVki[s].Append(nelem);

                  for (int i=0; i<nes; ++i)
                  {
                     const int edge = slaveEdges[masterEdgeSlaves[mid][i]];
                     edgeE[s].Append(edge);

                     Array<int> sverts(2);
                     if (edge >= 0)  // If a slave edge
                     {
                        patchTopo->GetEdgeVertices(edge, sverts);
                     }
                     else
                     {
                        const int auxEdge = -1 - edge;
                        GetAuxEdgeVertices(auxEdge, sverts);
                     }

                     MFEM_VERIFY((sverts[0] == edgeV[s][i] && sverts[1] == edgeV[s][i+1]) ||
                                 (sverts[1] == edgeV[s][i] && sverts[0] == edgeV[s][i+1]), "");
                  }
               }

               // Check whether the number and types of edges on opposite
               // sides match. If not, skip this pair of sides.
               if (edgeE[0].Size() != edgeE[1].Size())
               {
                  continue;
               }

               const int nes = edgeE[0].Size();

               {
                  bool matching = true;
                  for (int i=0; i<nes; ++i)
                  {
                     if ((edgeE[0][i] >= 0) != (edgeE[1][i] >= 0))
                     {
                        matching = false;
                        break;
                     }
                  }

                  if (!matching)
                  {
                     continue;
                  }
               }

               // Check whether edgeV[s] are in the same order or reversed, for s=0,1.
               bool rev = true;
               {
                  Array<int> sideVerts0;
                  patchTopo->GetEdgeVertices(sideEdge0, sideVerts0);
                  sideVerts0.Sort();

                  Array<bool> found(2);
                  found = false;

                  Array<int> ep(2);
                  for (int e=0; e<2; ++e) // Loop over ends
                  {
                     for (int s=0; s<2; ++s) // Loop over sides
                     {
                        ep[s] = edgeV[s][e * (edgeV[s].Size() - 1)];

                        for (int i=0; i<2; ++i)
                        {
                           if (ep[s] == sideVerts0[i])
                           {
                              found[i] = true;
                           }
                        }
                     }

                     ep.Sort();
                     if (ep == sideVerts0)
                     {
                        rev = false;
                     }
                  }

                  MFEM_VERIFY(found[0] && found[1], "");
               }

               // Find auxiliary or slave subfaces of face f.
               // Note that there may be no master faces in patchTopo->ncmesh.

               for (int i=0; i<2; ++i)
               {
                  MFEM_VERIFY(edgeV[i].Size() == nes + 1, "");
               }

               for (int e=0; e<nes; ++e) // Loop over edges
               {
                  const int e1 = rev ? nes - 1 - e : e;

                  Array<int> fverts(4);
                  fverts[0] = edgeV[0][e];
                  fverts[1] = edgeV[0][e + 1];

                  fverts[2] = edgeV[1][rev ? nes - e - 1 : e + 1];
                  fverts[3] = edgeV[1][rev ? nes - e : e];

                  const int eki = edgeVki[0][e];  // Knot index w.r.t. the edge.
                  const int eki1 = edgeVki[0][e + 1];  // Knot index w.r.t. the edge.

                  const int e1ki = edgeVki[1][rev ? nes - e : e];  // Knot index w.r.t. the edge.
                  const int e1ki1 = edgeVki[1][rev ? nes - e - 1 : e +
                                               1];  // Knot index w.r.t. the edge.

                  // ori_f is the signed shift such that fverts[abs1(ori_f)] is
                  // closest to vertex 0, verts[0], of parent face f, and the
                  // relative direction of the ordering is encoded in the sign.
                  // TODO: this is not used, since we put fverts into auxFaces with reordering to match
                  // the orientation of the master face, effectively making ori_f = 0.
                  int ori_f = 0;

                  // Set the 2D knot indices of the 4 vertices in fverts, with respect to the face f.
                  Array2D<int> fki(4,2);
                  {
                     // Use epv and eki to get the 2D face knot index.
                     const int eNE = edgeVki[0][edgeVki[0].Size() -
                                                1];  // Number of elements on the edge.
                     if (p == 0)
                     {
                        MFEM_ASSERT(edgeV[0][0] == verts[0] ||
                                    edgeV[0][edgeV[0].Size() - 1] == verts[0], "");
                        MFEM_ASSERT(edgeV[1][0] == verts[3] ||
                                    edgeV[1][edgeV[0].Size() - 1] == verts[3], "");
                        MFEM_ASSERT(eNE == fn1, "");
                        MFEM_ASSERT(edgeVki[1][edgeVki[1].Size() - 1] == fn1, "");

                        const bool rev0 = edgeV[0][0] != verts[0];
                        fki(0,0) = rev0 ? eNE - eki : eki;
                        fki(0,1) = 0;

                        fki(1,0) = rev0 ? eNE - eki1 : eki1;
                        fki(1,1) = 0;

                        if (rev0)
                        {
                           ori_f = -2;
                        }

                        // Other side
                        const bool rev1 = edgeV[1][0] != verts[3];

                        fki(2,0) = rev1 ? eNE - e1ki1 : e1ki1;
                        fki(2,1) = fn2;

                        fki(3,0) = rev1 ? eNE - e1ki : e1ki;
                        fki(3,1) = fn2;

                        MFEM_ASSERT(fki(0,0) == fki(3,0) && fki(1,0) == fki(2,0), "");
                     }
                     else
                     {
                        MFEM_ASSERT(edgeV[0][0] == verts[1] ||
                                    edgeV[0][edgeV[0].Size() - 1] == verts[1], "");
                        MFEM_ASSERT(edgeV[1][0] == verts[0] ||
                                    edgeV[1][edgeV[0].Size() - 1] == verts[0], "");
                        MFEM_ASSERT(eNE == fn2, "");
                        MFEM_ASSERT(edgeVki[1][edgeVki[1].Size() - 1] == fn2, "");

                        const bool rev0 = edgeV[0][0] != verts[1];
                        fki(0,0) = fn1;
                        fki(0,1) = rev0 ? eNE - eki : eki;

                        fki(1,0) = fn1;
                        fki(1,1) = rev0 ? eNE - eki1 : eki1;

                        if (rev0)
                        {
                           ori_f = -3;
                        }
                        else
                        {
                           ori_f = 3;
                        }

                        // Other side
                        const bool rev1 = edgeV[1][0] != verts[0];

                        fki(2,0) = 0;
                        fki(2,1) = rev1 ? fn2 - e1ki1 : e1ki1;

                        fki(3,0) = 0;
                        fki(3,1) = rev1 ? fn2 - e1ki : e1ki;

                        MFEM_ASSERT(fki(0,1) == fki(3,1) && fki(1,1) == fki(2,1), "");
                     }
                  }

                  const int vmin = fverts.Min();
                  const int idmin = fverts.Find(vmin);

                  if (edgeE[0][e] >= 0)
                  {
                     std::pair<int, int> vpair(fverts[idmin], fverts[(idmin + 2) % 4]);
                     const bool vPairTopo = v2f.count(vpair) > 0;
                     MFEM_VERIFY(vPairTopo, "");

                     const int sface = v2f.at(std::pair<int, int> (fverts[idmin],
                                                                   fverts[(idmin + 2) % 4]));
                     addParentFaces.insert(f);

                     // Set facePairs

                     // TODO: refactor this repeated code to find vMinID

                     // Find the vertex with minimum knot indices.
                     int vMinID = -1;
                     {
                        Array<int> kiMin(2);
                        for (int j=0; j<2; ++j)
                        {
                           kiMin[j] = fki(0,j);
                           for (int i=1; i<4; ++i)
                           {
                              if (fki(i,j) < kiMin[j]) { kiMin[j] = fki(i,j); }
                           }
                        }

                        for (int i=0; i<4; ++i)
                        {
                           if (fki(i,0) == kiMin[0] && fki(i,1) == kiMin[1])
                           {
                              MFEM_ASSERT(vMinID == -1, "");
                              vMinID = i;
                           }
                        }
                     }

                     MFEM_ASSERT(vMinID >= 0, "");

                     Array<int> fvertsMasterOrdering(4);
                     for (int i=0; i<4; ++i)
                     {
                        if (ori_f >= 0)
                        {
                           fvertsMasterOrdering[i] = fverts[(vMinID + i) % 4];
                        }
                        else
                        {
                           fvertsMasterOrdering[i] = fverts[(vMinID + 4 - i) % 4];
                        }
                     }

                     const int ori_sface = GetFaceOrientation(patchTopo, sface,
                                                              fvertsMasterOrdering);

                     facePairs.emplace_back(FacePairInfo{fverts[vMinID], sface,
                                                         f, ori_sface,
                                                         fki(vMinID,0), fki(vMinID,1),
                                                         fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                                                         fki((vMinID + 2) % 4,1) - fki(vMinID,1)});
                  }
                  else  // Auxiliary face
                  {
                     // TODO: use a fast, efficient look-up, using something like auxv2f.
                     const int nauxf = auxFaces.size();
                     Array<int> afverts(4);

                     Array<int> fvertsSorted(fverts);
                     fvertsSorted.Sort();

                     int afid = -1;
                     for (int af = 0; af < nauxf; ++af)
                     {
                        for (int i=0; i<4; ++i)
                        {
                           afverts[i] = auxFaces[af].v[i];
                        }

                        afverts.Sort();

                        if (afverts == fvertsSorted)
                        {
                           MFEM_VERIFY(afid == -1, "");
                           afid = af;
                        }
                     }

                     addParentFaces.insert(f);

                     // Find the vertex with minimum knot indices.
                     int vMinID = -1;
                     {
                        Array<int> kiMin(2);
                        for (int j=0; j<2; ++j)
                        {
                           kiMin[j] = fki(0,j);
                           for (int i=1; i<4; ++i)
                           {
                              if (fki(i,j) < kiMin[j]) { kiMin[j] = fki(i,j); }
                           }
                        }

                        for (int i=0; i<4; ++i)
                        {
                           if (fki(i,0) == kiMin[0] && fki(i,1) == kiMin[1])
                           {
                              MFEM_ASSERT(vMinID == -1, "");
                              vMinID = i;
                           }
                        }
                     }

                     MFEM_ASSERT(vMinID >= 0, "");

                     if (afid >= 0)
                     {
                        const int auxFace = -1 - afid;

                        // Find orientation of ordered vertices for this face, in fvertsSorted, w.r.t. the auxFaces ordering.

                        for (int i=0; i<4; ++i)
                        {
                           afverts[i] = auxFaces[afid].v[i];
                        }

                        int ori_f2 = -1;  // TODO: better name?
                        for (int i=0; i<4; ++i)
                        {
                           if (ori_f >= 0)
                           {
                              fvertsSorted[i] = fverts[(vMinID + i) % 4];
                           }
                           else
                           {
                              fvertsSorted[i] = fverts[(vMinID + 4 - i) % 4];
                           }

                           if (fvertsSorted[i] == afverts[0])
                           {
                              ori_f2 = i;
                           }
                        }

                        MFEM_ASSERT(ori_f2 >= 0, "");

                        if (fvertsSorted[(ori_f2 + 1) % 4] != afverts[1])
                        {
                           for (int j=0; j<4; ++j)
                           {
                              MFEM_ASSERT(fvertsSorted[(ori_f2 + 4 - j) % 4] == afverts[j], "");
                           }

                           ori_f2 = -1 - ori_f2;
                        }
                        else
                        {
                           for (int j=0; j<4; ++j)
                           {
                              MFEM_ASSERT(fvertsSorted[(ori_f2 + j) % 4] == afverts[j], "");
                           }
                        }

                        facePairs.emplace_back(FacePairInfo{fverts[vMinID], -1 - afid,
                                                            f, ori_f2,
                                                            fki(vMinID,0), fki(vMinID,1),
                                                            fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                                                            fki((vMinID + 2) % 4,1) - fki(vMinID,1)});
                     }
                     else
                     {
                        // Create a new auxiliary face.
                        const int auxFaceId = auxFaces.size();
                        // Find the knot indices of the vertices in fverts,
                        // with respect to the parent face.
                        AuxiliaryFace auxFace;
                        for (int i=0; i<4; ++i)
                        {
                           if (ori_f >= 0)
                           {
                              auxFace.v[i] = fverts[(vMinID + i) % 4];
                           }
                           else
                           {
                              auxFace.v[i] = fverts[(vMinID + 4 - i) % 4];
                           }
                        }

                        // Orientation is 0 for a new auxiliary face, by construction.
                        ori_f = 0;

                        auxFace.parent = f;
                        auxFace.ori = ori_f;
                        for (int i=0; i<2; ++i)
                        {
                           auxFace.ki0[i] = fki(vMinID,i);
                           auxFace.ki1[i] = fki((vMinID + 2) % 4,i);
                        }

                        auxFaces.push_back(auxFace);

                        facePairs.emplace_back(FacePairInfo{fverts[vMinID], -1 - auxFaceId,
                                                            f, ori_f,
                                                            fki(vMinID,0), fki(vMinID,1),
                                                            fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                                                            fki((vMinID + 2) % 4,1) - fki(vMinID,1)});
                     }
                  }
               }
            }
         }  // if (bothMaster)
      }  // Pair (p) loop
   }  // f
}

void NURBSExtension::ProcessFacePairs(int start, int midStart,
                                      const std::vector<int> & parentN1,
                                      const std::vector<int> & parentN2,
                                      std::vector<int> & parentVerts,
                                      const std::vector<FacePairInfo> & facePairs)
{
   const int nfpairs = facePairs.size();
   const bool is3D = Dimension() == 3;

   MFEM_VERIFY(nfpairs > 0 || !is3D, "");
   int midPrev = -1;
   int pfcnt = 0;
   int orientation = 0;
   bool parentFaceChanged = false;
   for (int q=start; q<nfpairs; ++q)
   {
      // We assume that j is the fast index in (i,j).
      // Note that facePairs is set by ProcessVertexToKnot3D.
      const int i = facePairs[q].ki[0];
      const int j = facePairs[q].ki[1];
      const int nfe1 = facePairs[q].ne[0];  // Number of elements, direction 1
      const int nfe2 = facePairs[q].ne[1];  // Number of elements, direction 2
      const int v0 = facePairs[q].v0;  // Bottom-left corner vertex of child face
      const int childFace = facePairs[q].child;
      const int parentFace = facePairs[q].parent;
      const int cpori =
         facePairs[q].ori;  // Orientation for childFace w.r.t. parentFace
      const int mid = masterFaceToId.at(parentFace);

      if (mid < midStart)
      {
         // Ignore data about master faces already processed.
         // TODO: instead of just ignoring, should we verify that the new data matches the old?
         continue;
      }

      MFEM_VERIFY(0 <= i && i < parentN1[mid], "");
      MFEM_VERIFY(0 <= j && j < parentN2[mid], "");
      if (mid != midPrev)  // Next parent face
      {
         Array<int> pv(parentVerts.data() + (4*mid), 4);
         const int ori = GetFaceOrientation(patchTopo, parentFace, pv);
         // Ori is the signed shift such that pv[abs1(ori)] is vertex 0 of parentFace,
         // and the relative direction of the ordering is encoded in the sign.

         if (q > start && midPrev >= 0)
         {
            // For the previous parentFace, use previous orientation
            // to reorder masterFaceSlaves, masterFaceSlaveCorners,
            // masterFaceSizes, masterFaceVerts.
            std::vector<int> s0(2);
            const bool rev = Reorder2D(parentN1[midPrev], parentN2[midPrev],
                                       orientation, s0);
            masterFaceS0[midPrev] = s0;
            masterFaceRev[midPrev] = rev;
            Reorder2D(parentN1[midPrev], parentN2[midPrev],
                      orientation, s0);
         }

         orientation = ori;

         midPrev = mid;

         pfcnt++;
      }  // next parent face

      slaveFaces.push_back(childFace);
      slaveFaceI.push_back(i);
      slaveFaceJ.push_back(j);
      slaveFaceNE1.push_back(nfe1);
      slaveFaceNE2.push_back(nfe2);
      slaveFaceOri.push_back(cpori);

      const int si = slaveFaces.size() - 1;
      masterFaceSlaves[mid].push_back(si);
      masterFaceSlaveCorners[mid].push_back(v0);

      masterFaceSizes[mid][0] = parentN1[mid];
      masterFaceSizes[mid][1] = parentN2[mid];

      if (i < parentN1[mid] - 1 && j < parentN2[mid] - 1)
      {
         // Find the interior vertex associated with this child face.

         // For left-most faces, use right side of face, else left side.
         const int vi0 = (i == 0) ? 1 : 0;

         // For bottom-most faces, use top side of face, else bottom side.
         const int vi1 = (j == 0) ? 1 : 0;

         // Get the face vertex at position (vi0, vi1) of the quadrilateral child face.

         int qid[2][2] = {{0, 3}, {1, 2}};

         const int vid = qid[vi0][vi1];

         // Find the index of vertex v0, which is at the bottom-left corner.
         Array<int> vert;
         if (childFace >= 0)
         {
            patchTopo->GetFaceVertices(childFace, vert);
         }
         else
         {
            // Auxiliary Face
            GetAuxFaceVertices(-1 - childFace, vert);
         }

         MFEM_VERIFY(vert.Size() == 4, "TODO: remove this obvious check");
         const int v0id = vert.Find(v0);
         MFEM_VERIFY(v0id >= 0, "");

         // Set the interior vertex associated with this child face.
         const int vint = vert[(vid - v0id + 4) % 4];
         masterFaceVerts[mid].push_back(vint);  // TODO: not used?
      }
   }  // Loop (q) over facePairs

   const int numParentFaces = is3D ? masterFaces.size() : 0;
   MFEM_VERIFY(numParentFaces == 0 || pfcnt <= numParentFaces, "");

   // TODO: restructure the above loop (q) over nfpairs to avoid this copying of code for Reorder2D.
   if (midPrev >= 0)
   {
      std::vector<int> s0(2);
      const bool rev = Reorder2D(parentN1[midPrev], parentN2[midPrev],
                                 orientation, s0);

      masterFaceS0[midPrev] = s0;
      masterFaceRev[midPrev] = rev;
      Reorder2D(parentN1[midPrev], parentN2[midPrev],
                orientation, s0);
   }
}

void NURBSExtension::GenerateOffsets()
{
   int nv = patchTopo->GetNV();
   int ne = patchTopo->GetNEdges();
   int nf = patchTopo->GetNFaces();
   int np = patchTopo->GetNE();
   int meshCounter, spaceCounter, dim = Dimension();

   std::set<int> reversedParents;
   if (patchTopo->ncmesh)
   {
      // Note that master or slave entities exist only for a mesh with
      // vertex_parents, not for the vertex_to_knot case. Currently, a mesh is
      // not allowed to have both cases, see the MFEM_VERIFY below.

      // TODO: for simplicity, should we only support vertex_to_knot in NC-NURBS,
      // not vertex_parents? Or should we allow either-or, but not both in the same mesh?

      const NCMesh::NCList& nce = patchTopo->ncmesh->GetNCList(1);
      const NCMesh::NCList& ncf = patchTopo->ncmesh->GetNCList(2);

      masterEdges.clear();
      masterFaces.clear();
      slaveEdges.clear();
      slaveFaces.clear();
      slaveFaceNE1.clear();
      slaveFaceNE2.clear();
      masterEdgeToId.clear();
      masterFaceToId.clear();

      MFEM_VERIFY(nce.masters.Size() > 0 ||
                  patchTopo->ncmesh->GetVertexToKnot().NumRows() > 0, "");
      MFEM_VERIFY(!(nce.masters.Size() > 0 &&
                    patchTopo->ncmesh->GetVertexToKnot().NumRows() > 0), "");

      std::vector<EdgePairInfo> edgePairs;
      std::vector<FacePairInfo> facePairs;
      std::vector<int> parentN1, parentN2, parentFaces, parentVerts;

      const bool is3D = dim == 3;

      std::map<std::pair<int, int>, int> v2f;

      if (patchTopo->ncmesh->GetVertexToKnot().NumRows() > 0)
      {
         // Intersections of master edges may not be edges in patchTopo->ncmesh,
         // so we represent them in auxEdges, to account for their vertices and
         // DOFs.
         {
            int vert_index[2];
            const NCMesh::NCList& EL = patchTopo->ncmesh->GetEdgeList();
            for (auto edgeID : EL.conforming)
            {
               patchTopo->ncmesh->GetEdgeVertices(edgeID, vert_index);
               v2e[std::pair<int, int> (vert_index[0], vert_index[1])] = edgeID.index;
            }
         }

         if (is3D)
         {
            Array<int> vert;
            for (int i=0; i<patchTopo->GetNumFaces(); ++i)
            {
               // TODO: should this be a loop over conforming faces as for v2e?
               patchTopo->GetFaceVertices(i, vert);
               MFEM_VERIFY(vert.Size() == 4, "TODO: remove this obvious check");
               const int vmin = vert.Min();
               const int idmin = vert.Find(vmin);
               v2f[std::pair<int, int> (vert[idmin], vert[(idmin + 2) % 4])] = i;
            }
         }

         Array2D<int> const& v2k = patchTopo->ncmesh->GetVertexToKnot();

         if (is3D)
            ProcessVertexToKnot3D(v2k, v2f, parentN1, parentN2, edgePairs,
                                  facePairs, parentFaces, parentVerts);
         else
         {
            ProcessVertexToKnot2D(v2k, reversedParents, edgePairs);
         }
      } // if using vertex_to_knot

      const int numMasters = is3D ? ncf.masters.Size() : nce.masters.Size();

      const int numParentFaces = is3D ? masterFaces.size() : 0;

      if (is3D)
      {
         for (auto masterFace : ncf.masters)
         {
            masterFaces.insert(masterFace.index);
         }
      }

      for (auto masterEdge : nce.masters)
      {
         masterEdges.insert(masterEdge.index);
      }

      Array<int> masterEdgeIndex(masterEdges.size());
      int cnt = 0;
      for (auto medge : masterEdges)
      {
         masterEdgeIndex[cnt] = medge;
         masterEdgeToId[medge] = cnt;
         cnt++;
      }
      MFEM_VERIFY(cnt == masterEdgeIndex.Size(), "");

      Array<int> masterFaceIndex(parentFaces.size());

      // Note that masterFaces is a subset of parentFaces.
      MFEM_VERIFY(masterFaces.size() <= parentFaces.size(), "");

      cnt = 0;
      for (auto mface : parentFaces)
      {
         masterFaceIndex[cnt] = mface;
         masterFaceToId[mface] =
            cnt;  // TODO: is this actually parentFaceToId, which includes master faces?
         cnt++;
      }

      MFEM_VERIFY(cnt == masterFaceIndex.Size(), "");

      masterEdgeSlaves.clear();
      masterEdgeVerts.clear();
      masterEdgeKI.clear();

      masterEdgeSlaves.resize(masterEdgeIndex.Size());
      masterEdgeVerts.resize(masterEdgeIndex.Size());
      masterEdgeKI.resize(masterEdgeIndex.Size());

      masterFaceSlaves.clear();
      masterFaceSlaveCorners.clear();
      masterFaceSizes.clear();
      masterFaceVerts.clear();

      masterFaceSlaves.resize(masterFaceIndex.Size());
      masterFaceSlaveCorners.resize(masterFaceIndex.Size());
      masterFaceSizes.resize(masterFaceIndex.Size());
      masterFaceVerts.resize(masterFaceIndex.Size());

      masterFaceS0.resize(masterFaceIndex.Size());
      masterFaceRev.resize(masterFaceIndex.Size());

      for (unsigned int i=0; i<masterFaceSizes.size(); ++i)
      {
         masterFaceSizes[i].resize(2);
         masterFaceS0[i].resize(2);
      }

      if (patchTopo->ncmesh->GetVertexToKnot().NumRows() > 0)
      {
         // Note that this is used in 2D and 3D.
         const int npairs = edgePairs.size();

         for (int i=0; i<npairs; ++i)
         {
            if (!edgePairs[i].isSet)
            {
               continue;
            }

            const int v = edgePairs[i].v;
            const int s = edgePairs[i].child;
            const int m = edgePairs[i].parent;
            const int ki = edgePairs[i].ki;

            slaveEdges.push_back(s);

            const int mid = masterEdgeToId[m];
            const int si = slaveEdges.size() - 1;
            masterEdgeSlaves[mid].push_back(si);
            if (v >= 0)
            {
               masterEdgeVerts[mid].push_back(v);
               masterEdgeKI[mid].push_back(ki);
            }
         }

         ProcessFacePairs(0, 0, parentN1, parentN2, parentVerts, facePairs);
      }

      for (int i=0; i<nce.slaves.Size(); ++i)
      {
         const NCMesh::Slave& slaveEdge = nce.slaves[i];
         int vert_index[2];
         patchTopo->ncmesh->GetEdgeVertices(slaveEdge, vert_index);
         slaveEdges.push_back(slaveEdge.index);

         const int mid = masterEdgeToId[slaveEdge.master];
         masterEdgeSlaves[mid].push_back(i);
      }

      if (!is3D)
      {
         for (int m=0; m<numMasters; ++m)
         {
            // Order the slaves of each master edge, from the first to second
            // vertex of the master edge.
            const int numSlaves = masterEdgeSlaves[m].size();
            MFEM_VERIFY(numSlaves > 0, "");
            int mvert[2];
            int svert[2];
            patchTopo->ncmesh->GetEdgeVertices(nce.masters[m], mvert);

            std::vector<int> orderedSlaves(numSlaves);

            int vi = mvert[0];
            for (int s=0; s<numSlaves; ++s)
            {
               // Find the slave edge containing vertex vi

               // TODO: This has quadratic complexity. Can it be improved? Does it
               // matter, since the number of slave edges should always be small?
               orderedSlaves[s] = -1;
               for (int t=0; t<numSlaves; ++t)
               {
                  const int sid = masterEdgeSlaves[m][t];
                  patchTopo->ncmesh->GetEdgeVertices(nce.slaves[sid], svert);
                  if (svert[0] == vi || svert[1] == vi)
                  {
                     orderedSlaves[s] = sid;
                     break;
                  }
               }

               MFEM_VERIFY(orderedSlaves[s] >= 0, "");

               // Update vi to the next vertex
               vi = (svert[0] == vi) ? svert[1] : svert[0];

               if (s < numSlaves - 1)
               {
                  masterEdgeVerts[m].push_back(vi);
                  masterEdgeKI[m].push_back(-1);  // TODO: this is not set and not used in 2D.
               }
            }

            masterEdgeSlaves[m] = orderedSlaves;

            MFEM_VERIFY(masterEdgeSlaves[m].size() == masterEdgeVerts[m].size() + 1, "");
            MFEM_VERIFY(masterEdgeVerts[m].size() > 0, "");
         } // m
      }

      if (is3D)
      {
         // Remove edges from masterEdges if they do not have any slave edges.
         std::vector<int> falseMasterEdges;
         for (auto me : masterEdges)
         {
            const int mid = masterEdgeToId.at(me);
            if (masterEdgeSlaves[mid].size() == 0)
            {
               falseMasterEdges.push_back(me);
            }
         }

         for (auto me : falseMasterEdges)
         {
            masterEdges.erase(me);
         }

         // Find slave and auxiliary faces not yet defined.
         const int nfp0 = facePairs.size();
         std::set<int> addParentFaces;
         FindAdditionalSlaveAndAuxiliaryFaces(v2f, addParentFaces, facePairs);

         cnt = parentFaces.size();
         const int npf0 = cnt;

         for (auto pf : addParentFaces)
         {
            if (masterFaces.count(pf) == 0)
            {
               masterFaces.insert(pf);
               masterFaceIndex.Append(pf);

               masterFaceToId[pf] = cnt;
               cnt++;

               {
                  Array<int> edges, ori, verts;
                  patchTopo->GetFaceEdges(pf, edges, ori);
                  patchTopo->GetFaceVertices(pf, verts);

                  MFEM_VERIFY(edges.Size() == 4 && verts.Size() == 4, "");

                  parentN1.push_back(KnotVec(edges[0])->GetNE());
                  parentN2.push_back(KnotVec(edges[1])->GetNE());

                  masterFaceSizes.push_back({KnotVec(edges[0])->GetNE(),
                                             KnotVec(edges[1])->GetNE()});

                  masterFaceS0.push_back({-1, -1});
                  masterFaceRev.push_back(false);

                  for (int i=0; i<4; ++i)
                  {
                     parentVerts.push_back(verts[i]);
                  }
               }
            }
         }

         MFEM_VERIFY(cnt == masterFaceIndex.Size(), "");

         masterFaceSlaves.resize(masterFaceIndex.Size());
         masterFaceSlaveCorners.resize(masterFaceIndex.Size());
         masterFaceVerts.resize(masterFaceIndex.Size());

         ProcessFacePairs(nfp0, npf0, parentN1, parentN2, parentVerts, facePairs);
      }
   }

   for (auto rp : reversedParents)
   {
      const int mid = masterEdgeToId[rp];
      std::reverse(masterEdgeSlaves[mid].begin(), masterEdgeSlaves[mid].end());
      std::reverse(masterEdgeVerts[mid].begin(), masterEdgeVerts[mid].end());
      std::reverse(masterEdgeKI[mid].begin(), masterEdgeKI[mid].end());
   }

   Array<int> edges;
   Array<int> orient;

   v_meshOffsets.SetSize(nv);
   e_meshOffsets.SetSize(ne);
   f_meshOffsets.SetSize(nf);
   p_meshOffsets.SetSize(np);

   v_spaceOffsets.SetSize(nv);
   e_spaceOffsets.SetSize(ne);
   f_spaceOffsets.SetSize(nf);
   p_spaceOffsets.SetSize(np);

   // Get vertex offsets
   for (meshCounter = 0; meshCounter < nv; meshCounter++)
   {
      v_meshOffsets[meshCounter]  = meshCounter;
      v_spaceOffsets[meshCounter] = meshCounter;
   }
   spaceCounter = meshCounter;

   // Get edge offsets
   int nint = 0;
   for (int e = 0; e < ne; e++)
   {
      e_meshOffsets[e]  = meshCounter;
      e_spaceOffsets[e] = spaceCounter;

      if (masterEdges.count(e) == 0)  // If not a master edge
      {
         meshCounter  += KnotVec(e)->GetNE() - 1;
         spaceCounter += KnotVec(e)->GetNCP() - 2;
         if (KnotVec(e)->GetNE() - 1 > 0) { nint += KnotVec(e)->GetNE() - 1; }
      }
   }

   const int nauxe = auxEdges.size();
   aux_e_meshOffsets.SetSize(nauxe + 1);
   aux_e_spaceOffsets.SetSize(nauxe + 1);
   for (int e = 0; e < nauxe; e++)
   {
      aux_e_meshOffsets[e] = meshCounter;
      aux_e_spaceOffsets[e] = spaceCounter;

      // Find the number of elements and CP in this auxiliary edge, which is
      // defined only on part of the master edge knotvector.

      const int signedParentEdge = auxEdges[e].parent;
      const int ki0 = auxEdges[e].ki[0];
      const int ki1raw = auxEdges[e].ki[1];

      const bool rev = signedParentEdge < 0;
      const int parentEdge = rev ? -1 - signedParentEdge : signedParentEdge;

      const int masterNE = KnotVec(parentEdge)->GetNE();

      const int ki1 = ki1raw == -1 ? masterNE : ki1raw;

      const int perEdgeCP = GetNCPperEdge(KnotVec(e));

      const int auxne = ki1 - ki0;
      MFEM_ASSERT(auxne > 0, "");
      meshCounter += auxne - 1;
      spaceCounter += (auxne * perEdgeCP) + auxne - 1;
   }

   aux_e_meshOffsets[nauxe] = meshCounter;
   aux_e_spaceOffsets[nauxe] = spaceCounter;

   // Get face offsets
   for (int f = 0; f < nf; f++)
   {
      f_meshOffsets[f]  = meshCounter;
      f_spaceOffsets[f] = spaceCounter;

      const bool trueMaster = masterFaces.count(f) > 0;
      /*
      if (trueMaster)
      {
        const int mid = masterFaceToId.at(f);
        if (masterFaceSizes[mid][0] == 0 && masterFaceSizes[mid][1] == 0)
          trueMaster = false;
      }
           */

      if (!trueMaster)  // If not a master face
      {
         patchTopo->GetFaceEdges(f, edges, orient);

         meshCounter +=
            (KnotVec(edges[0])->GetNE() - 1) *
            (KnotVec(edges[1])->GetNE() - 1);
         spaceCounter +=
            (KnotVec(edges[0])->GetNCP() - 2) *
            (KnotVec(edges[1])->GetNCP() - 2);
      }
   }

   const int nauxf = auxFaces.size();
   aux_f_meshOffsets.SetSize(nauxf + 1);
   aux_f_spaceOffsets.SetSize(nauxf + 1);
   for (int f = 0; f < nauxf; f++)
   {
      aux_f_meshOffsets[f] = meshCounter;
      aux_f_spaceOffsets[f] = spaceCounter;

      const int parentFace = auxFaces[f].parent;
      patchTopo->GetFaceEdges(parentFace, edges, orient);

      // Number of control points per edge, in first and second directions.
      const int perEdgeCP0 = GetNCPperEdge(KnotVec(edges[0]));
      const int perEdgeCP1 = GetNCPperEdge(KnotVec(edges[1]));

      const int auxne0 = auxFaces[f].ki1[0] - auxFaces[f].ki0[0];
      const int auxne1 = auxFaces[f].ki1[1] - auxFaces[f].ki0[1];
      meshCounter += (auxne0 - 1) * (auxne1 - 1);
      spaceCounter += ((auxne0 * perEdgeCP0) + auxne0 - 1) *
                      ((auxne1 * perEdgeCP1) + auxne1 - 1);
   }

   aux_f_meshOffsets[nauxf] = meshCounter;
   aux_f_spaceOffsets[nauxf] = spaceCounter;

   // Get patch offsets
   for (int p = 0; p < np; p++)
   {
      p_meshOffsets[p]  = meshCounter;
      p_spaceOffsets[p] = spaceCounter;

      if (dim == 1)
      {
         meshCounter  += KnotVec(0)->GetNE() - 1;
         spaceCounter += KnotVec(0)->GetNCP() - 2;
      }
      else if (dim == 2)
      {
         patchTopo->GetElementEdges(p, edges, orient);
         meshCounter +=
            (KnotVec(edges[0])->GetNE() - 1) *
            (KnotVec(edges[1])->GetNE() - 1);
         spaceCounter +=
            (KnotVec(edges[0])->GetNCP() - 2) *
            (KnotVec(edges[1])->GetNCP() - 2);
      }
      else
      {
         patchTopo->GetElementEdges(p, edges, orient);
         meshCounter +=
            (KnotVec(edges[0])->GetNE() - 1) *
            (KnotVec(edges[3])->GetNE() - 1) *
            (KnotVec(edges[8])->GetNE() - 1);
         spaceCounter +=
            (KnotVec(edges[0])->GetNCP() - 2) *
            (KnotVec(edges[3])->GetNCP() - 2) *
            (KnotVec(edges[8])->GetNCP() - 2);
      }
   }
   NumOfVertices = meshCounter;
   NumOfDofs     = spaceCounter;
}

NURBSExtension* NURBSExtension::GetDivExtension(int component)
{
   // Smarter routine
   if (GetNP() > 1)
   {
      mfem_error("NURBSExtension::GetDivExtension currently "
                 "only works for single patch NURBS meshes ");
   }

   Array<int> newOrders  = GetOrders();
   newOrders[component] += 1;

   return new NURBSExtension(this, newOrders, Mode::H_DIV);
}

NURBSExtension* NURBSExtension::GetCurlExtension(int component)
{
   // Smarter routine
   if (GetNP() > 1)
   {
      mfem_error("NURBSExtension::GetCurlExtension currently "
                 "only works for single patch NURBS meshes ");
   }

   Array<int> newOrders  = GetOrders();
   for (int c = 0; c < newOrders.Size(); c++) { newOrders[c]++; }
   newOrders[component] -= 1;

   return new NURBSExtension(this, newOrders, Mode::H_CURL);
}

void NURBSExtension::GetAuxEdgeVertices(int auxEdge, Array<int> &verts) const
{
   verts.SetSize(2);
   for (int i=0; i<2; ++i)
   {
      verts[i] = auxEdges[auxEdge].v[i];
   }
}

void NURBSExtension::GetAuxFaceVertices(int auxFace, Array<int> & verts) const
{
   verts.SetSize(4);
   for (int i=0; i<4; ++i)
   {
      verts[i] = auxFaces[auxFace].v[i];
   }
}

// Note that v2e is a map only for conforming patchTopo->ncmesh edges.
// Auxiliary edges are not included.
void NURBSExtension::GetAuxFaceEdges(int auxFace,
                                     Array<int> & edges) const
{
   edges.SetSize(4);
   Array<int> verts(2);
   for (int i=0; i<4; ++i)
   {
      for (int j=0; j<2; ++j)
      {
         verts[j] = auxFaces[auxFace].v[(i + j) % 4];
      }

      verts.Sort();

      const std::pair<int, int> edge_v(verts[0], verts[1]);
      if (v2e.count(edge_v) > 0)
      {
         edges[i] = v2e.at(edge_v);  // patchTopo edge
      }
      else
      {
         // Auxiliary edge

         // TODO: use a map like auxv2e in ProcessVertexToKnot3D for
         // efficient look-up of auxiliary edges.
         int auxEdge = -1;
         const int nauxe = auxEdges.size();
         Array<int> aeverts(2);
         for (int ae = 0; ae < nauxe; ++ae)
         {
            for (int j=0; j<2; ++j)
            {
               aeverts[j] = auxEdges[ae].v[j];
            }

            aeverts.Sort();

            if (aeverts == verts)
            {
               MFEM_ASSERT(auxEdge == -1, "");
               auxEdge = ae;
            }
         }

         MFEM_ASSERT(auxEdge >= 0, "");
         edges[i] = -1 - auxEdge;
      }
   }
}

void NURBSExtension::CountElements()
{
   int dim = Dimension();
   Array<const KnotVector *> kv(dim);

   NumOfElements = 0;
   for (int p = 0; p < GetNP(); p++)
   {
      GetPatchKnotVectors(p, kv);

      int ne = kv[0]->GetNE();
      for (int d = 1; d < dim; d++)
      {
         ne *= kv[d]->GetNE();
      }

      NumOfElements += ne;
   }
}

void NURBSExtension::CountBdrElements()
{
   int dim = Dimension() - 1;
   Array<KnotVector *> kv(dim);

   NumOfBdrElements = 0;
   for (int p = 0; p < GetNBP(); p++)
   {
      GetBdrPatchKnotVectors(p, kv);

      int ne = 1;
      for (int d = 0; d < dim; d++)
      {
         ne *= kv[d]->GetNE();
      }

      NumOfBdrElements += ne;
   }
}

void NURBSExtension::GetElementTopo(Array<Element *> &elements) const
{
   elements.SetSize(GetNE());

   if (Dimension() == 1)
   {
      Get1DElementTopo(elements);
   }
   else if (Dimension() == 2)
   {
      Get2DElementTopo(elements);
   }
   else
   {
      Get3DElementTopo(elements);
   }
}

void NURBSExtension::Get1DElementTopo(Array<Element *> &elements) const
{
   int el = 0;
   int eg = 0;
   int ind[2];
   NURBSPatchMap p2g(this);
   const KnotVector *kv[1];

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchVertexMap(p, kv);
      int nx = p2g.nx();

      int patch_attr = patchTopo->GetAttribute(p);

      for (int i = 0; i < nx; i++)
      {
         if (activeElem[eg])
         {
            ind[0] = activeVert[p2g(i)];
            ind[1] = activeVert[p2g(i+1)];

            elements[el] = new Segment(ind, patch_attr);
            el++;
         }
         eg++;
      }
   }
}

void NURBSExtension::Get2DElementTopo(Array<Element *> &elements) const
{
   int el = 0;
   int eg = 0;
   int ind[4];
   NURBSPatchMap p2g(this);
   const KnotVector *kv[2];

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchVertexMap(p, kv);
      int nx = p2g.nx();
      int ny = p2g.ny();

      int patch_attr = patchTopo->GetAttribute(p);

      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            if (activeElem[eg])
            {
               ind[0] = activeVert[p2g(i,  j  )];
               ind[1] = activeVert[p2g(i+1,j  )];
               ind[2] = activeVert[p2g(i+1,j+1)];
               ind[3] = activeVert[p2g(i,  j+1)];

               elements[el] = new Quadrilateral(ind, patch_attr);
               el++;
            }
            eg++;
         }
      }
   }
}

void NURBSExtension::Get3DElementTopo(Array<Element *> &elements) const
{
   int el = 0;
   int eg = 0;
   int ind[8];
   NURBSPatchMap p2g(this);
   const KnotVector *kv[3];

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchVertexMap(p, kv);
      int nx = p2g.nx();
      int ny = p2g.ny();
      int nz = p2g.nz();

      int patch_attr = patchTopo->GetAttribute(p);

      for (int k = 0; k < nz; k++)
      {
         for (int j = 0; j < ny; j++)
         {
            for (int i = 0; i < nx; i++)
            {
               if (activeElem[eg])
               {
                  ind[0] = activeVert[p2g(i,  j,  k)];
                  ind[1] = activeVert[p2g(i+1,j,  k)];
                  ind[2] = activeVert[p2g(i+1,j+1,k)];
                  ind[3] = activeVert[p2g(i,  j+1,k)];

                  ind[4] = activeVert[p2g(i,  j,  k+1)];
                  ind[5] = activeVert[p2g(i+1,j,  k+1)];
                  ind[6] = activeVert[p2g(i+1,j+1,k+1)];
                  ind[7] = activeVert[p2g(i,  j+1,k+1)];

                  elements[el] = new Hexahedron(ind, patch_attr);
                  el++;
               }
               eg++;
            }
         }
      }
   }
}

void NURBSExtension::GetBdrElementTopo(Array<Element *> &boundary) const
{
   boundary.SetSize(GetNBE());

   if (Dimension() == 1)
   {
      Get1DBdrElementTopo(boundary);
   }
   else if (Dimension() == 2)
   {
      Get2DBdrElementTopo(boundary);
   }
   else
   {
      Get3DBdrElementTopo(boundary);
   }
}

void NURBSExtension::Get1DBdrElementTopo(Array<Element *> &boundary) const
{
   int g_be, l_be;
   int ind[2], okv[1];
   NURBSPatchMap p2g(this);
   const KnotVector *kv[1];

   g_be = l_be = 0;
   for (int b = 0; b < GetNBP(); b++)
   {
      p2g.SetBdrPatchVertexMap(b, kv, okv);
      int bdr_patch_attr = patchTopo->GetBdrAttribute(b);

      if (activeBdrElem[g_be])
      {
         ind[0] = activeVert[p2g[0]];
         boundary[l_be] = new Point(ind, bdr_patch_attr);
         l_be++;
      }
      g_be++;
   }
}

void NURBSExtension::Get2DBdrElementTopo(Array<Element *> &boundary) const
{
   int g_be, l_be;
   int ind[2], okv[1];
   NURBSPatchMap p2g(this);
   const KnotVector *kv[1];

   g_be = l_be = 0;
   for (int b = 0; b < GetNBP(); b++)
   {
      p2g.SetBdrPatchVertexMap(b, kv, okv);
      int nx = p2g.nx();

      int bdr_patch_attr = patchTopo->GetBdrAttribute(b);

      for (int i = 0; i < nx; i++)
      {
         if (activeBdrElem[g_be])
         {
            int i_ = (okv[0] >= 0) ? i : (nx - 1 - i);
            ind[0] = activeVert[p2g[i_  ]];
            ind[1] = activeVert[p2g[i_+1]];

            boundary[l_be] = new Segment(ind, bdr_patch_attr);
            l_be++;
         }
         g_be++;
      }
   }
}

void NURBSExtension::Get3DBdrElementTopo(Array<Element *> &boundary) const
{
   int g_be, l_be;
   int ind[4], okv[2];
   NURBSPatchMap p2g(this);
   const KnotVector *kv[2];

   g_be = l_be = 0;
   for (int b = 0; b < GetNBP(); b++)
   {
      p2g.SetBdrPatchVertexMap(b, kv, okv);
      int nx = p2g.nx();
      int ny = p2g.ny();

      int bdr_patch_attr = patchTopo->GetBdrAttribute(b);

      for (int j = 0; j < ny; j++)
      {
         int j_ = (okv[1] >= 0) ? j : (ny - 1 - j);
         for (int i = 0; i < nx; i++)
         {
            if (activeBdrElem[g_be])
            {
               int i_ = (okv[0] >= 0) ? i : (nx - 1 - i);
               ind[0] = activeVert[p2g(i_,  j_  )];
               ind[1] = activeVert[p2g(i_+1,j_  )];
               ind[2] = activeVert[p2g(i_+1,j_+1)];
               ind[3] = activeVert[p2g(i_,  j_+1)];

               boundary[l_be] = new Quadrilateral(ind, bdr_patch_attr);
               l_be++;
            }
            g_be++;
         }
      }
   }
}

void NURBSExtension::GenerateElementDofTable()
{
   activeDof.SetSize(GetNTotalDof());
   activeDof = 0;

   if (Dimension() == 1)
   {
      Generate1DElementDofTable();
   }
   else if (Dimension() == 2)
   {
      Generate2DElementDofTable();
   }
   else
   {
      Generate3DElementDofTable();
   }

   SetPatchToElements();

   NumOfActiveDofs = 0;
   for (int d = 0; d < GetNTotalDof(); d++)
      if (activeDof[d])
      {
         NumOfActiveDofs++;
         activeDof[d] = NumOfActiveDofs;
      }

   int *dof = el_dof->GetJ();
   int ndof = el_dof->Size_of_connections();
   for (int i = 0; i < ndof; i++)
   {
      dof[i] = activeDof[dof[i]] - 1;
   }
}

void NURBSExtension::Generate1DElementDofTable()
{
   int el = 0;
   int eg = 0;
   const KnotVector *kv[2];
   NURBSPatchMap p2g(this);

   Array<Connection> el_dof_list;
   el_to_patch.SetSize(NumOfActiveElems);
   el_to_IJK.SetSize(NumOfActiveElems, 2);

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);

      // Load dofs
      const int ord0 = kv[0]->GetOrder();
      for (int i = 0; i < kv[0]->GetNKS(); i++)
      {
         if (kv[0]->isElement(i))
         {
            if (activeElem[eg])
            {
               Connection conn(el,0);
               for (int ii = 0; ii <= ord0; ii++)
               {
                  conn.to = DofMap(p2g(i+ii));
                  activeDof[conn.to] = 1;
                  el_dof_list.Append(conn);
               }
               el_to_patch[el] = p;
               el_to_IJK(el,0) = i;

               el++;
            }
            eg++;
         }
      }
   }
   // We must NOT sort el_dof_list in this case.
   el_dof = new Table(NumOfActiveElems, el_dof_list);
}

void NURBSExtension::Generate2DElementDofTable()
{
   int el = 0;
   int eg = 0;
   const KnotVector *kv[2];
   NURBSPatchMap p2g(this);

   Array<Connection> el_dof_list;
   el_to_patch.SetSize(NumOfActiveElems);
   el_to_IJK.SetSize(NumOfActiveElems, 2);

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);

      // Load dofs
      const int ord0 = kv[0]->GetOrder();
      const int ord1 = kv[1]->GetOrder();
      for (int j = 0; j < kv[1]->GetNKS(); j++)
      {
         if (kv[1]->isElement(j))
         {
            for (int i = 0; i < kv[0]->GetNKS(); i++)
            {
               if (kv[0]->isElement(i))
               {
                  if (activeElem[eg])
                  {
                     Connection conn(el,0);
                     for (int jj = 0; jj <= ord1; jj++)
                     {
                        for (int ii = 0; ii <= ord0; ii++)
                        {
                           conn.to = DofMap(p2g(i+ii,j+jj));
                           activeDof[conn.to] = 1;
                           el_dof_list.Append(conn);
                        }
                     }
                     el_to_patch[el] = p;
                     el_to_IJK(el,0) = i;
                     el_to_IJK(el,1) = j;

                     el++;
                  }
                  eg++;
               }
            }
         }
      }
   }
   // We must NOT sort el_dof_list in this case.
   el_dof = new Table(NumOfActiveElems, el_dof_list);
}

void NURBSExtension::Generate3DElementDofTable()
{
   int el = 0;
   int eg = 0;
   const KnotVector *kv[3];
   NURBSPatchMap p2g(this);

   Array<Connection> el_dof_list;
   el_to_patch.SetSize(NumOfActiveElems);
   el_to_IJK.SetSize(NumOfActiveElems, 3);

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);

      // Load dofs
      const int ord0 = kv[0]->GetOrder();
      const int ord1 = kv[1]->GetOrder();
      const int ord2 = kv[2]->GetOrder();
      for (int k = 0; k < kv[2]->GetNKS(); k++)
      {
         if (kv[2]->isElement(k))
         {
            for (int j = 0; j < kv[1]->GetNKS(); j++)
            {
               if (kv[1]->isElement(j))
               {
                  for (int i = 0; i < kv[0]->GetNKS(); i++)
                  {
                     if (kv[0]->isElement(i))
                     {
                        if (activeElem[eg])
                        {
                           Connection conn(el,0);
                           for (int kk = 0; kk <= ord2; kk++)
                           {
                              for (int jj = 0; jj <= ord1; jj++)
                              {
                                 for (int ii = 0; ii <= ord0; ii++)
                                 {
                                    conn.to = DofMap(p2g(i+ii, j+jj, k+kk));
                                    activeDof[conn.to] = 1;
                                    el_dof_list.Append(conn);
                                 }
                              }
                           }

                           el_to_patch[el] = p;
                           el_to_IJK(el,0) = i;
                           el_to_IJK(el,1) = j;
                           el_to_IJK(el,2) = k;

                           el++;
                        }
                        eg++;
                     }
                  }
               }
            }
         }
      }
   }
   // We must NOT sort el_dof_list in this case.
   el_dof = new Table(NumOfActiveElems, el_dof_list);
}

void NURBSExtension::GetPatchDofs(const int patch, Array<int> &dofs)
{
   const KnotVector *kv[3];
   NURBSPatchMap p2g(this);

   p2g.SetPatchDofMap(patch, kv);

   if (Dimension() == 1)
   {
      const int nx = kv[0]->GetNCP();
      dofs.SetSize(nx);

      for (int i=0; i<nx; ++i)
      {
         dofs[i] = DofMap(p2g(i));
      }
   }
   else if (Dimension() == 2)
   {
      const int nx = kv[0]->GetNCP();
      const int ny = kv[1]->GetNCP();
      dofs.SetSize(nx * ny);

      for (int j=0; j<ny; ++j)
         for (int i=0; i<nx; ++i)
         {
            dofs[i + (nx * j)] = DofMap(p2g(i, j));
         }
   }
   else if (Dimension() == 3)
   {
      const int nx = kv[0]->GetNCP();
      const int ny = kv[1]->GetNCP();
      const int nz = kv[2]->GetNCP();
      dofs.SetSize(nx * ny * nz);

      for (int k=0; k<nz; ++k)
         for (int j=0; j<ny; ++j)
            for (int i=0; i<nx; ++i)
            {
               dofs[i + (nx * (j + (k * ny)))] = DofMap(p2g(i, j, k));
            }
   }
   else
   {
      MFEM_ABORT("Only 1D/2D/3D supported currently in NURBSExtension::GetPatchDofs");
   }
}

void NURBSExtension::GenerateBdrElementDofTable()
{
   if (Dimension() == 1)
   {
      Generate1DBdrElementDofTable();
   }
   else if (Dimension() == 2)
   {
      Generate2DBdrElementDofTable();
   }
   else
   {
      Generate3DBdrElementDofTable();
   }

   SetPatchToBdrElements();

   int *dof = bel_dof->GetJ();
   int ndof = bel_dof->Size_of_connections();
   for (int i = 0; i < ndof; i++)
   {
      int idx = dof[i];
      if (idx < 0)
      {
         dof[i] = -1 - (activeDof[-1-idx] - 1);
         dof[i] = -activeDof[-1-idx];
      }
      else
      {
         dof[i] = activeDof[idx] - 1;
      }
   }
}

void NURBSExtension::Generate1DBdrElementDofTable()
{
   int gbe = 0;
   int lbe = 0, okv[1];
   const KnotVector *kv[1];
   NURBSPatchMap p2g(this);

   Array<Connection> bel_dof_list;
   bel_to_patch.SetSize(NumOfActiveBdrElems);
   bel_to_IJK.SetSize(NumOfActiveBdrElems, 1);

   for (int b = 0; b < GetNBP(); b++)
   {
      p2g.SetBdrPatchDofMap(b, kv, okv);
      // Load dofs
      if (activeBdrElem[gbe])
      {
         Connection conn(lbe,0);
         conn.to = DofMap(p2g[0]);
         bel_dof_list.Append(conn);
         bel_to_patch[lbe] = b;
         bel_to_IJK(lbe,0) = 0;
         lbe++;
      }
      gbe++;
   }
   // We must NOT sort bel_dof_list in this case.
   bel_dof = new Table(NumOfActiveBdrElems, bel_dof_list);
}

void NURBSExtension::Generate2DBdrElementDofTable()
{
   int gbe = 0;
   int lbe = 0, okv[1];
   const KnotVector *kv[1];
   NURBSPatchMap p2g(this);

   Array<Connection> bel_dof_list;
   bel_to_patch.SetSize(NumOfActiveBdrElems);
   bel_to_IJK.SetSize(NumOfActiveBdrElems, 1);

   for (int b = 0; b < GetNBP(); b++)
   {
      p2g.SetBdrPatchDofMap(b, kv, okv);
      const int nx = p2g.nx(); // NCP-1
      // Load dofs
      const int nks0 = kv[0]->GetNKS();
      const int ord0 = kv[0]->GetOrder();

      bool add_dofs = true;
      int  s = 1;

      if (mode == Mode::H_DIV)
      {
         int fn = patchTopo->GetBdrElementFaceIndex(b);
         if (ord0 == mOrders.Max()) { add_dofs = false; }
         if (fn == 0) { s = -1; }
         if (fn == 2) { s = -1; }
      }
      else if (mode == Mode::H_CURL)
      {
         if (ord0 == mOrders.Max()) { add_dofs = false; }
      }

      for (int i = 0; i < nks0; i++)
      {
         if (kv[0]->isElement(i))
         {
            if (activeBdrElem[gbe])
            {
               Connection conn(lbe,0);
               if (add_dofs)
               {
                  for (int ii = 0; ii <= ord0; ii++)
                  {
                     conn.to = DofMap(p2g[(okv[0] >= 0) ? (i+ii) : (nx-i-ii)]);
                     if (s == -1) { conn.to = -1 -conn.to; }
                     bel_dof_list.Append(conn);
                  }
               }
               bel_to_patch[lbe] = b;
               bel_to_IJK(lbe,0) = (okv[0] >= 0) ? i : (-1-i);
               lbe++;
            }
            gbe++;
         }
      }
   }
   // We must NOT sort bel_dof_list in this case.
   bel_dof = new Table(NumOfActiveBdrElems, bel_dof_list);
}


void NURBSExtension::Generate3DBdrElementDofTable()
{
   int gbe = 0;
   int lbe = 0, okv[2];
   const KnotVector *kv[2];
   NURBSPatchMap p2g(this);

   Array<Connection> bel_dof_list;
   bel_to_patch.SetSize(NumOfActiveBdrElems);
   bel_to_IJK.SetSize(NumOfActiveBdrElems, 2);

   for (int b = 0; b < GetNBP(); b++)
   {
      p2g.SetBdrPatchDofMap(b, kv, okv);
      const int nx = p2g.nx(); // NCP0-1
      const int ny = p2g.ny(); // NCP1-1

      // Load dofs
      const int nks0 = kv[0]->GetNKS();
      const int ord0 = kv[0]->GetOrder();
      const int nks1 = kv[1]->GetNKS();
      const int ord1 = kv[1]->GetOrder();

      // Check if dofs are actually defined on boundary
      bool add_dofs = true;
      int  s = 1;

      if (mode == Mode::H_DIV)
      {
         int fn = patchTopo->GetBdrElementFaceIndex(b);
         if (ord0 != ord1) { add_dofs = false; }
         if (fn == 4) { s = -1; }
         if (fn == 1) { s = -1; }
         if (fn == 0) { s = -1; }
      }
      else if (mode == Mode::H_CURL)
      {
         if (ord0 == ord1) { add_dofs = false; }
      }


      for (int j = 0; j < nks1; j++)
      {
         if (kv[1]->isElement(j))
         {
            for (int i = 0; i < nks0; i++)
            {
               if (kv[0]->isElement(i))
               {
                  if (activeBdrElem[gbe])
                  {
                     Connection conn(lbe,0);
                     if (add_dofs)
                     {
                        for (int jj = 0; jj <= ord1; jj++)
                        {
                           const int jj_ = (okv[1] >= 0) ? (j+jj) : (ny-j-jj);
                           for (int ii = 0; ii <= ord0; ii++)
                           {
                              const int ii_ = (okv[0] >= 0) ? (i+ii) : (nx-i-ii);
                              conn.to = DofMap(p2g(ii_, jj_));
                              if (s == -1) { conn.to = -1 -conn.to; }
                              bel_dof_list.Append(conn);
                           }
                        }
                     }
                     bel_to_patch[lbe] = b;
                     bel_to_IJK(lbe,0) = (okv[0] >= 0) ? i : (-1-i);
                     bel_to_IJK(lbe,1) = (okv[1] >= 0) ? j : (-1-j);
                     lbe++;
                  }
                  gbe++;
               }
            }
         }
      }
   }
   // We must NOT sort bel_dof_list in this case.
   bel_dof = new Table(NumOfActiveBdrElems, bel_dof_list);
}

void NURBSExtension::GetVertexLocalToGlobal(Array<int> &lvert_vert)
{
   lvert_vert.SetSize(GetNV());
   for (int gv = 0; gv < GetGNV(); gv++)
      if (activeVert[gv] >= 0)
      {
         lvert_vert[activeVert[gv]] = gv;
      }
}

void NURBSExtension::GetElementLocalToGlobal(Array<int> &lelem_elem)
{
   lelem_elem.SetSize(GetNE());
   for (int le = 0, ge = 0; ge < GetGNE(); ge++)
      if (activeElem[ge])
      {
         lelem_elem[le++] = ge;
      }
}

void NURBSExtension::LoadFE(int i, const FiniteElement *FE) const
{
   const NURBSFiniteElement *NURBSFE =
      dynamic_cast<const NURBSFiniteElement *>(FE);

   if (NURBSFE->GetElement() != i)
   {
      Array<int> dofs;
      NURBSFE->SetIJK(el_to_IJK.GetRow(i));
      if (el_to_patch[i] != NURBSFE->GetPatch())
      {
         GetPatchKnotVectors(el_to_patch[i], NURBSFE->KnotVectors());
         NURBSFE->SetPatch(el_to_patch[i]);
         NURBSFE->SetOrder();
      }
      el_dof->GetRow(i, dofs);
      weights.GetSubVector(dofs, NURBSFE->Weights());
      NURBSFE->SetElement(i);
   }
}

void NURBSExtension::LoadBE(int i, const FiniteElement *BE) const
{
   if (Dimension() == 1) { return; }

   const NURBSFiniteElement *NURBSFE =
      dynamic_cast<const NURBSFiniteElement *>(BE);

   if (NURBSFE->GetElement() != i)
   {
      Array<int> dofs;
      NURBSFE->SetIJK(bel_to_IJK.GetRow(i));
      if (bel_to_patch[i] != NURBSFE->GetPatch())
      {
         GetBdrPatchKnotVectors(bel_to_patch[i], NURBSFE->KnotVectors());
         NURBSFE->SetPatch(bel_to_patch[i]);
         NURBSFE->SetOrder();
      }
      bel_dof->GetRow(i, dofs);
      weights.GetSubVector(dofs, NURBSFE->Weights());
      NURBSFE->SetElement(i);
   }
}

void NURBSExtension::ConvertToPatches(const Vector &Nodes)
{
   delete el_dof;
   delete bel_dof;

   if (patches.Size() == 0)
   {
      GetPatchNets(Nodes, Dimension());
   }
}

void NURBSExtension::SetCoordsFromPatches(Vector &Nodes)
{
   if (patches.Size() == 0) { return; }

   SetSolutionVector(Nodes, Dimension());
   patches.SetSize(0);
}

void NURBSExtension::SetKnotsFromPatches()
{
   if (patches.Size() == 0)
   {
      mfem_error("NURBSExtension::SetKnotsFromPatches :"
                 " No patches available!");
   }

   Array<KnotVector *> kv;

   for (int p = 0; p < patches.Size(); p++)
   {
      GetPatchKnotVectors(p, kv);

      for (int i = 0; i < kv.Size(); i++)
      {
         *kv[i] = *patches[p]->GetKV(i);
      }
   }

   UpdateUniqueKV();
   SetOrdersFromKnotVectors();

   GenerateOffsets();
   CountElements();
   CountBdrElements();

   // all elements must be active
   NumOfActiveElems = NumOfElements;
   activeElem.SetSize(NumOfElements);
   activeElem = true;

   GenerateActiveVertices();
   InitDofMap();
   GenerateElementDofTable();
   GenerateActiveBdrElems();
   GenerateBdrElementDofTable();

   ConnectBoundaries();
}

void NURBSExtension::LoadSolution(std::istream &input, GridFunction &sol) const
{
   const FiniteElementSpace *fes = sol.FESpace();
   MFEM_VERIFY(fes->GetNURBSext() == this, "");

   sol.SetSize(fes->GetVSize());

   Array<const KnotVector *> kv(Dimension());
   NURBSPatchMap p2g(this);
   const int vdim = fes->GetVDim();

   for (int p = 0; p < GetNP(); p++)
   {
      skip_comment_lines(input, '#');

      p2g.SetPatchDofMap(p, kv);
      const int nx = kv[0]->GetNCP();
      const int ny = kv[1]->GetNCP();
      const int nz = (kv.Size() == 2) ? 1 : kv[2]->GetNCP();
      for (int k = 0; k < nz; k++)
      {
         for (int j = 0; j < ny; j++)
         {
            for (int i = 0; i < nx; i++)
            {
               const int ll = (kv.Size() == 2) ? p2g(i,j) : p2g(i,j,k);
               const int l  = DofMap(ll);
               for (int vd = 0; vd < vdim; vd++)
               {
                  input >> sol(fes->DofToVDof(l,vd));
               }
            }
         }
      }
   }
}

void NURBSExtension::PrintSolution(const GridFunction &sol, std::ostream &os)
const
{
   const FiniteElementSpace *fes = sol.FESpace();
   MFEM_VERIFY(fes->GetNURBSext() == this, "");

   Array<const KnotVector *> kv(Dimension());
   NURBSPatchMap p2g(this);
   const int vdim = fes->GetVDim();

   for (int p = 0; p < GetNP(); p++)
   {
      os << "\n# patch " << p << "\n\n";

      p2g.SetPatchDofMap(p, kv);
      const int nx = kv[0]->GetNCP();
      const int ny = kv[1]->GetNCP();
      const int nz = (kv.Size() == 2) ? 1 : kv[2]->GetNCP();
      for (int k = 0; k < nz; k++)
      {
         for (int j = 0; j < ny; j++)
         {
            for (int i = 0; i < nx; i++)
            {
               const int ll = (kv.Size() == 2) ? p2g(i,j) : p2g(i,j,k);
               const int l  = DofMap(ll);
               os << sol(fes->DofToVDof(l,0));
               for (int vd = 1; vd < vdim; vd++)
               {
                  os << ' ' << sol(fes->DofToVDof(l,vd));
               }
               os << '\n';
            }
         }
      }
   }
}

void NURBSExtension::DegreeElevate(int rel_degree, int degree)
{
   for (int p = 0; p < patches.Size(); p++)
   {
      for (int dir = 0; dir < patches[p]->GetNKV(); dir++)
      {
         int oldd = patches[p]->GetKV(dir)->GetOrder();
         int newd = std::min(oldd + rel_degree, degree);
         if (newd > oldd)
         {
            patches[p]->DegreeElevate(dir, newd - oldd);
         }
      }
   }
}

void NURBSExtension::UniformRefinement(Array<int> const& rf)
{
   for (int p = 0; p < patches.Size(); p++)
   {
      patches[p]->UniformRefinement(rf);
   }
}

void NURBSExtension::UniformRefinement(int rf)
{
   Array<int> rf_array(Dimension());
   rf_array = rf;
   UniformRefinement(rf_array);
}

void NURBSExtension::Coarsen(Array<int> const& cf, real_t tol)
{
   // First, mark all knot vectors on all patches as not coarse. This prevents
   // coarsening the same knot vector twice.
   for (int p = 0; p < patches.Size(); p++)
   {
      patches[p]->SetKnotVectorsCoarse(false);
   }

   for (int p = 0; p < patches.Size(); p++)
   {
      patches[p]->Coarsen(cf, tol);
   }
}

void NURBSExtension::Coarsen(int cf, real_t tol)
{
   Array<int> cf_array(Dimension());
   cf_array = cf;
   Coarsen(cf_array, tol);
}

void NURBSExtension::GetCoarseningFactors(Array<int> & f) const
{
   f.SetSize(0);
   for (auto patch : patches)
   {
      Array<int> pf;
      patch->GetCoarseningFactors(pf);
      if (f.Size() == 0)
      {
         f = pf; // Initialize
      }
      else
      {
         MFEM_VERIFY(f.Size() == pf.Size(), "");
         for (int i=0; i<f.Size(); ++i)
         {
            MFEM_VERIFY(f[i] == pf[i] || f[i] == 1 || pf[i] == 1,
                        "Inconsistent patch coarsening factors");
            if (f[i] == 1 && pf[i] != 1)
            {
               f[i] = pf[i];
            }
         }
      }
   }
}

void NURBSExtension::KnotInsert(Array<KnotVector *> &kv)
{
   Array<int> edges;
   Array<int> orient;
   Array<int> kvdir;

   Array<KnotVector *> pkv(Dimension());

   for (int p = 0; p < patches.Size(); p++)
   {
      if (Dimension()==1)
      {
         pkv[0] = kv[KnotInd(p)];
      }
      else if (Dimension()==2)
      {
         patchTopo->GetElementEdges(p, edges, orient);
         pkv[0] = kv[KnotInd(edges[0])];
         pkv[1] = kv[KnotInd(edges[1])];
      }
      else if (Dimension()==3)
      {
         patchTopo->GetElementEdges(p, edges, orient);
         pkv[0] = kv[KnotInd(edges[0])];
         pkv[1] = kv[KnotInd(edges[3])];
         pkv[2] = kv[KnotInd(edges[8])];
      }


      // Check whether inserted knots should be flipped before inserting.
      // Knotvectors are stored in a different array pkvc such that the original
      // knots which are inserted are not changed.
      // We need those knots for multiple patches so they have to remain original
      CheckKVDirection(p, kvdir);

      Array<KnotVector *> pkvc(Dimension());
      for (int d = 0; d < Dimension(); d++)
      {
         pkvc[d] = new KnotVector(*(pkv[d]));

         if (kvdir[d] == -1)
         {
            pkvc[d]->Flip();
         }
      }

      patches[p]->KnotInsert(pkvc);
      for (int d = 0; d < Dimension(); d++) { delete pkvc[d]; }
   }
}

void NURBSExtension::KnotInsert(Array<Vector *> &kv)
{
   Array<int> edges;
   Array<int> orient;
   Array<int> kvdir;

   Array<Vector *> pkv(Dimension());

   for (int p = 0; p < patches.Size(); p++)
   {
      if (Dimension()==1)
      {
         pkv[0] = kv[KnotInd(p)];
      }
      else if (Dimension()==2)
      {
         patchTopo->GetElementEdges(p, edges, orient);
         pkv[0] = kv[KnotInd(edges[0])];
         pkv[1] = kv[KnotInd(edges[1])];
      }
      else if (Dimension()==3)
      {
         patchTopo->GetElementEdges(p, edges, orient);
         pkv[0] = kv[KnotInd(edges[0])];
         pkv[1] = kv[KnotInd(edges[3])];
         pkv[2] = kv[KnotInd(edges[8])];
      }

      // Check whether inserted knots should be flipped before inserting.
      // Knotvectors are stored in a different array pkvc such that the original
      // knots which are inserted are not changed.
      CheckKVDirection(p, kvdir);

      Array<Vector *> pkvc(Dimension());
      for (int d = 0; d < Dimension(); d++)
      {
         pkvc[d] = new Vector(*(pkv[d]));

         if (kvdir[d] == -1)
         {
            // Find flip point, for knotvectors that do not have the domain [0:1]
            KnotVector *kva = knotVectorsCompr[Dimension()*p+d];
            double apb = (*kva)[0] + (*kva)[kva->Size()-1];

            // Flip vector
            int size = pkvc[d]->Size();
            int ns = ceil(size/2.0);
            for (int j = 0; j < ns; j++)
            {
               double tmp = apb - pkvc[d]->Elem(j);
               pkvc[d]->Elem(j) = apb - pkvc[d]->Elem(size-1-j);
               pkvc[d]->Elem(size-1-j) = tmp;
            }
         }
      }

      patches[p]->KnotInsert(pkvc);

      for (int i = 0; i < Dimension(); i++) { delete pkvc[i]; }
   }
}

void NURBSExtension::KnotRemove(Array<Vector *> &kv, real_t tol)
{
   Array<int> edges;
   Array<int> orient;
   Array<int> kvdir;

   Array<Vector *> pkv(Dimension());

   for (int p = 0; p < patches.Size(); p++)
   {
      if (Dimension()==1)
      {
         pkv[0] = kv[KnotInd(p)];
      }
      else if (Dimension()==2)
      {
         patchTopo->GetElementEdges(p, edges, orient);
         pkv[0] = kv[KnotInd(edges[0])];
         pkv[1] = kv[KnotInd(edges[1])];
      }
      else if (Dimension()==3)
      {
         patchTopo->GetElementEdges(p, edges, orient);
         pkv[0] = kv[KnotInd(edges[0])];
         pkv[1] = kv[KnotInd(edges[3])];
         pkv[2] = kv[KnotInd(edges[8])];
      }

      // Check whether knots should be flipped before removing.
      CheckKVDirection(p, kvdir);

      Array<Vector *> pkvc(Dimension());
      for (int d = 0; d < Dimension(); d++)
      {
         pkvc[d] = new Vector(*(pkv[d]));

         if (kvdir[d] == -1)
         {
            // Find flip point, for knotvectors that do not have the domain [0:1]
            KnotVector *kva = knotVectorsCompr[Dimension()*p+d];
            real_t apb = (*kva)[0] + (*kva)[kva->Size()-1];

            // Flip vector
            int size = pkvc[d]->Size();
            int ns = ceil(size/2.0);
            for (int j = 0; j < ns; j++)
            {
               real_t tmp = apb - pkvc[d]->Elem(j);
               pkvc[d]->Elem(j) = apb - pkvc[d]->Elem(size-1-j);
               pkvc[d]->Elem(size-1-j) = tmp;
            }
         }
      }

      patches[p]->KnotRemove(pkvc, tol);

      for (int i = 0; i < Dimension(); i++) { delete pkvc[i]; }
   }
}

void NURBSExtension::GetPatchNets(const Vector &coords, int vdim)
{
   if (Dimension() == 1)
   {
      Get1DPatchNets(coords, vdim);
   }
   else if (Dimension() == 2)
   {
      Get2DPatchNets(coords, vdim);
   }
   else
   {
      Get3DPatchNets(coords, vdim);
   }
}

void NURBSExtension::Get1DPatchNets(const Vector &coords, int vdim)
{
   Array<const KnotVector *> kv(1);
   NURBSPatchMap p2g(this);

   patches.SetSize(GetNP());
   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);
      patches[p] = new NURBSPatch(kv, vdim+1);
      NURBSPatch &Patch = *patches[p];

      for (int i = 0; i < kv[0]->GetNCP(); i++)
      {
         const int l = DofMap(p2g(i));
         for (int d = 0; d < vdim; d++)
         {
            Patch(i,d) = coords(l*vdim + d)*weights(l);
         }
         Patch(i,vdim) = weights(l);
      }
   }

}

void NURBSExtension::Get2DPatchNets(const Vector &coords, int vdim)
{
   Array<const KnotVector *> kv(2);
   NURBSPatchMap p2g(this);

   patches.SetSize(GetNP());
   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);
      patches[p] = new NURBSPatch(kv, vdim+1);
      NURBSPatch &Patch = *patches[p];

      for (int j = 0; j < kv[1]->GetNCP(); j++)
      {
         for (int i = 0; i < kv[0]->GetNCP(); i++)
         {
            const int l = DofMap(p2g(i,j));
            for (int d = 0; d < vdim; d++)
            {
               Patch(i,j,d) = coords(l*vdim + d)*weights(l);
            }
            Patch(i,j,vdim) = weights(l);
         }
      }
   }
}

void NURBSExtension::Get3DPatchNets(const Vector &coords, int vdim)
{
   Array<const KnotVector *> kv(3);
   NURBSPatchMap p2g(this);

   patches.SetSize(GetNP());
   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);
      patches[p] = new NURBSPatch(kv, vdim+1);
      NURBSPatch &Patch = *patches[p];

      for (int k = 0; k < kv[2]->GetNCP(); k++)
      {
         for (int j = 0; j < kv[1]->GetNCP(); j++)
         {
            for (int i = 0; i < kv[0]->GetNCP(); i++)
            {
               const int l = DofMap(p2g(i,j,k));
               for (int d = 0; d < vdim; d++)
               {
                  Patch(i,j,k,d) = coords(l*vdim + d)*weights(l);
               }
               Patch(i,j,k,vdim) = weights(l);
            }
         }
      }
   }
}

void NURBSExtension::SetSolutionVector(Vector &coords, int vdim)
{
   if (Dimension() == 1)
   {
      Set1DSolutionVector(coords, vdim);
   }
   else if (Dimension() == 2)
   {
      Set2DSolutionVector(coords, vdim);
   }
   else
   {
      Set3DSolutionVector(coords, vdim);
   }
}

void NURBSExtension::Set1DSolutionVector(Vector &coords, int vdim)
{
   Array<const KnotVector *> kv(1);
   NURBSPatchMap p2g(this);

   weights.SetSize(GetNDof());
   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);
      NURBSPatch &patch = *patches[p];
      MFEM_ASSERT(vdim+1 == patch.GetNC(), "");

      for (int i = 0; i < kv[0]->GetNCP(); i++)
      {
         const int l = p2g(i);
         for (int d = 0; d < vdim; d++)
         {
            coords(l*vdim + d) = patch(i,d)/patch(i,vdim);
         }
         weights(l) = patch(i,vdim);
      }

      delete patches[p];
   }
}


void NURBSExtension::Set2DSolutionVector(Vector &coords, int vdim)
{
   Array<const KnotVector *> kv(2);
   NURBSPatchMap p2g(this);

   weights.SetSize(GetNDof());
   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);
      NURBSPatch &patch = *patches[p];
      MFEM_ASSERT(vdim+1 == patch.GetNC(), "");

      for (int j = 0; j < kv[1]->GetNCP(); j++)
      {
         for (int i = 0; i < kv[0]->GetNCP(); i++)
         {
            const int l = p2g(i,j);
            for (int d = 0; d < vdim; d++)
            {
               coords(l*vdim + d) = patch(i,j,d)/patch(i,j,vdim);
            }
            weights(l) = patch(i,j,vdim);
         }
      }
      delete patches[p];
   }
}

void NURBSExtension::Set3DSolutionVector(Vector &coords, int vdim)
{
   Array<const KnotVector *> kv(3);
   NURBSPatchMap p2g(this);

   weights.SetSize(GetNDof());
   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);
      NURBSPatch &patch = *patches[p];
      MFEM_ASSERT(vdim+1 == patch.GetNC(), "");

      for (int k = 0; k < kv[2]->GetNCP(); k++)
      {
         for (int j = 0; j < kv[1]->GetNCP(); j++)
         {
            for (int i = 0; i < kv[0]->GetNCP(); i++)
            {
               const int l = p2g(i,j,k);
               for (int d = 0; d < vdim; d++)
               {
                  coords(l*vdim + d) = patch(i,j,k,d)/patch(i,j,k,vdim);
               }
               weights(l) = patch(i,j,k,vdim);
            }
         }
      }
      delete patches[p];
   }
}

void NURBSExtension::GetElementIJK(int elem, Array<int> & ijk)
{
   MFEM_VERIFY(ijk.Size() == el_to_IJK.NumCols(), "");
   el_to_IJK.GetRow(elem, ijk);
}

void NURBSExtension::SetPatchToElements()
{
   const int np = GetNP();
   patch_to_el.resize(np);

   for (int e=0; e<el_to_patch.Size(); ++e)
   {
      patch_to_el[el_to_patch[e]].Append(e);
   }
}

void NURBSExtension::SetPatchToBdrElements()
{
   const int nbp = GetNBP();
   patch_to_bel.resize(nbp);

   for (int e=0; e<bel_to_patch.Size(); ++e)
   {
      patch_to_bel[bel_to_patch[e]].Append(e);
   }
}

const Array<int>& NURBSExtension::GetPatchElements(int patch)
{
   MFEM_ASSERT(patch_to_el.size() > 0, "patch_to_el not set");

   return patch_to_el[patch];
}

const Array<int>& NURBSExtension::GetPatchBdrElements(int patch)
{
   MFEM_ASSERT(patch_to_bel.size() > 0, "patch_to_el not set");

   return patch_to_bel[patch];
}

void NURBSExtension::GetVertexDofs(int index, Array<int> &dofs) const
{
   MFEM_ASSERT(index < v_spaceOffsets.Size(), "");

   const int os = v_spaceOffsets[index];
   const int os1 = index + 1 == v_spaceOffsets.Size() ? e_spaceOffsets[0] :
                   v_spaceOffsets[index + 1];

   dofs.SetSize(0);
   dofs.Reserve(os1 - os);

   for (int i=os; i<os1; ++i)
   {
      dofs.Append(i);
   }
}

int NURBSExtension::GetEdgeDofs(int index, Array<int> &dofs) const
{
   MFEM_ASSERT(index < e_spaceOffsets.Size(), "");

   const int os = e_spaceOffsets[index];
   const int os_upper = f_spaceOffsets.Size() > 0 ? f_spaceOffsets[0] :
                        p_spaceOffsets[0];
   const int os1 = index + 1 == e_spaceOffsets.Size() ? os_upper :
                   v_spaceOffsets[index + 1];

   dofs.SetSize(0);
   // Reserve 2 for the two vertices and os1 - os for the interior edge DOFs.
   dofs.Reserve(2 + os1 - os);

   // First get the DOFs for the vertices of the edge.

   Array<int> vert;
   patchTopo->GetEdgeVertices(index, vert);
   MFEM_ASSERT(vert.Size() == 2, "TODO: remove this");

   for (auto v : vert)
   {
      Array<int> vdofs;
      GetVertexDofs(v, vdofs);
      dofs.Append(vdofs);
   }

   // Now get the interior edge DOFs.

   for (int i=os; i<os1; ++i)
   {
      dofs.Append(i);
   }

   return GetOrder();
}

int NURBSExtension::GetEntityDofs(int entity, int index, Array<int> &dofs) const
{
   switch (entity)
   {
      case 0:
         GetVertexDofs(index, dofs);
         return 0;
      case 1:
         return GetEdgeDofs(index, dofs);
      default:
         MFEM_ABORT("TODO: entity type not yet supported in GetEntityDofs");
         return 0;
   }
}

#ifdef MFEM_USE_MPI
ParNURBSExtension::ParNURBSExtension(const ParNURBSExtension &orig)
   : NURBSExtension(orig),
     partitioning(orig.partitioning),
     gtopo(orig.gtopo),
     ldof_group(orig.ldof_group)
{
}

ParNURBSExtension::ParNURBSExtension(MPI_Comm comm, NURBSExtension *parent,
                                     const int *partitioning_,
                                     const Array<bool> &active_bel)
   : gtopo(comm)
{
   if (parent->NumOfActiveElems < parent->NumOfElements)
   {
      // SetActive (BuildGroups?) and the way the weights are copied
      // do not support this case
      mfem_error("ParNURBSExtension::ParNURBSExtension :\n"
                 " all elements in the parent must be active!");
   }

   patchTopo = parent->patchTopo;
   // steal ownership of patchTopo from the 'parent' NURBS extension
   if (!parent->own_topo)
   {
      mfem_error("ParNURBSExtension::ParNURBSExtension :\n"
                 "  parent does not own the patch topology!");
   }
   own_topo = true;
   parent->own_topo = false;

   parent->edge_to_knot.Copy(edge_to_knot);

   parent->GetOrders().Copy(mOrders);
   mOrder = parent->GetOrder();

   NumOfKnotVectors = parent->GetNKV();
   knotVectors.SetSize(NumOfKnotVectors);
   for (int i = 0; i < NumOfKnotVectors; i++)
   {
      knotVectors[i] = new KnotVector(*parent->GetKnotVector(i));
   }
   CreateComprehensiveKV();

   GenerateOffsets();
   CountElements();
   CountBdrElements();

   // copy 'partitioning_' to 'partitioning'
   partitioning.SetSize(GetGNE());
   for (int i = 0; i < GetGNE(); i++)
   {
      partitioning[i] = partitioning_[i];
   }
   SetActive(partitioning, active_bel);

   GenerateActiveVertices();
   GenerateElementDofTable();
   // GenerateActiveBdrElems(); // done by SetActive for now
   GenerateBdrElementDofTable();

   Table *serial_elem_dof = parent->GetElementDofTable();
   BuildGroups(partitioning, *serial_elem_dof);

   weights.SetSize(GetNDof());
   // copy weights from parent
   for (int gel = 0, lel = 0; gel < GetGNE(); gel++)
   {
      if (activeElem[gel])
      {
         int  ndofs = el_dof->RowSize(lel);
         int *ldofs = el_dof->GetRow(lel);
         int *gdofs = serial_elem_dof->GetRow(gel);
         for (int i = 0; i < ndofs; i++)
         {
            weights(ldofs[i]) = parent->weights(gdofs[i]);
         }
         lel++;
      }
   }
}

ParNURBSExtension::ParNURBSExtension(NURBSExtension *parent,
                                     const ParNURBSExtension *par_parent)
   : gtopo(par_parent->gtopo.GetComm())
{
   // steal all data from parent
   mOrder = parent->mOrder;
   Swap(mOrders, parent->mOrders);

   patchTopo = parent->patchTopo;
   own_topo = parent->own_topo;
   parent->own_topo = 0;

   Swap(edge_to_knot, parent->edge_to_knot);

   NumOfKnotVectors = parent->NumOfKnotVectors;
   Swap(knotVectors, parent->knotVectors);
   Swap(knotVectorsCompr, parent->knotVectorsCompr);

   NumOfVertices    = parent->NumOfVertices;
   NumOfElements    = parent->NumOfElements;
   NumOfBdrElements = parent->NumOfBdrElements;
   NumOfDofs        = parent->NumOfDofs;

   Swap(v_meshOffsets, parent->v_meshOffsets);
   Swap(e_meshOffsets, parent->e_meshOffsets);
   Swap(f_meshOffsets, parent->f_meshOffsets);
   Swap(p_meshOffsets, parent->p_meshOffsets);

   Swap(v_spaceOffsets, parent->v_spaceOffsets);
   Swap(e_spaceOffsets, parent->e_spaceOffsets);
   Swap(f_spaceOffsets, parent->f_spaceOffsets);
   Swap(p_spaceOffsets, parent->p_spaceOffsets);

   Swap(aux_e_meshOffsets, parent->aux_e_meshOffsets);
   Swap(aux_e_spaceOffsets, parent->aux_e_spaceOffsets);
   Swap(auxEdges, parent->auxEdges);

   Swap(d_to_d, parent->d_to_d);
   Swap(master, parent->master);
   Swap(slave,  parent->slave);

   NumOfActiveVertices = parent->NumOfActiveVertices;
   NumOfActiveElems    = parent->NumOfActiveElems;
   NumOfActiveBdrElems = parent->NumOfActiveBdrElems;
   NumOfActiveDofs     = parent->NumOfActiveDofs;

   Swap(activeVert, parent->activeVert);
   Swap(activeElem, parent->activeElem);
   Swap(activeBdrElem, parent->activeBdrElem);
   Swap(activeDof, parent->activeDof);

   el_dof  = parent->el_dof;
   bel_dof = parent->bel_dof;
   parent->el_dof = parent->bel_dof = NULL;

   Swap(el_to_patch, parent->el_to_patch);
   Swap(bel_to_patch, parent->bel_to_patch);
   Swap(el_to_IJK, parent->el_to_IJK);
   Swap(bel_to_IJK, parent->bel_to_IJK);

   Swap(weights, parent->weights);
   MFEM_VERIFY(!parent->HavePatches(), "");

   delete parent;

   partitioning = NULL;

   MFEM_VERIFY(par_parent->partitioning,
               "parent ParNURBSExtension has no partitioning!");

   // Support for the case when 'parent' is not a local NURBSExtension, i.e.
   // NumOfActiveElems is not the same as in 'par_parent'. In that case, we
   // assume 'parent' is a global NURBSExtension, i.e. all elements are active.
   bool extract_weights = false;
   if (NumOfActiveElems != par_parent->NumOfActiveElems)
   {
      MFEM_ASSERT(NumOfActiveElems == NumOfElements, "internal error");

      SetActive(par_parent->partitioning, par_parent->activeBdrElem);
      GenerateActiveVertices();
      delete el_dof;
      el_to_patch.DeleteAll();
      el_to_IJK.DeleteAll();
      GenerateElementDofTable();
      // GenerateActiveBdrElems(); // done by SetActive for now
      delete bel_dof;
      bel_to_patch.DeleteAll();
      bel_to_IJK.DeleteAll();
      GenerateBdrElementDofTable();
      extract_weights = true;
   }

   Table *glob_elem_dof = GetGlobalElementDofTable();
   BuildGroups(par_parent->partitioning, *glob_elem_dof);
   if (extract_weights)
   {
      Vector glob_weights;
      Swap(weights, glob_weights);
      weights.SetSize(GetNDof());
      // Copy the local 'weights' from the 'glob_weights'.
      // Assumption: the local element ids follow the global ordering.
      for (int gel = 0, lel = 0; gel < GetGNE(); gel++)
      {
         if (activeElem[gel])
         {
            int  ndofs = el_dof->RowSize(lel);
            int *ldofs = el_dof->GetRow(lel);
            int *gdofs = glob_elem_dof->GetRow(gel);
            for (int i = 0; i < ndofs; i++)
            {
               weights(ldofs[i]) = glob_weights(gdofs[i]);
            }
            lel++;
         }
      }
   }
   delete glob_elem_dof;
}

Table *ParNURBSExtension::GetGlobalElementDofTable()
{
   if (Dimension() == 1)
   {
      return Get1DGlobalElementDofTable();
   }
   else if (Dimension() == 2)
   {
      return Get2DGlobalElementDofTable();
   }
   else
   {
      return Get3DGlobalElementDofTable();
   }
}

Table *ParNURBSExtension::Get1DGlobalElementDofTable()
{
   int el = 0;
   const KnotVector *kv[1];
   NURBSPatchMap p2g(this);
   Array<Connection> gel_dof_list;

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);

      // Load dofs
      const int ord0 = kv[0]->GetOrder();

      for (int i = 0; i < kv[0]->GetNKS(); i++)
      {
         if (kv[0]->isElement(i))
         {
            Connection conn(el,0);
            for (int ii = 0; ii <= ord0; ii++)
            {
               conn.to = DofMap(p2g(i+ii));
               gel_dof_list.Append(conn);
            }
            el++;
         }
      }
   }
   // We must NOT sort gel_dof_list in this case.
   return (new Table(GetGNE(), gel_dof_list));
}

Table *ParNURBSExtension::Get2DGlobalElementDofTable()
{
   int el = 0;
   const KnotVector *kv[2];
   NURBSPatchMap p2g(this);
   Array<Connection> gel_dof_list;

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);

      // Load dofs
      const int ord0 = kv[0]->GetOrder();
      const int ord1 = kv[1]->GetOrder();
      for (int j = 0; j < kv[1]->GetNKS(); j++)
      {
         if (kv[1]->isElement(j))
         {
            for (int i = 0; i < kv[0]->GetNKS(); i++)
            {
               if (kv[0]->isElement(i))
               {
                  Connection conn(el,0);
                  for (int jj = 0; jj <= ord1; jj++)
                  {
                     for (int ii = 0; ii <= ord0; ii++)
                     {
                        conn.to = DofMap(p2g(i+ii,j+jj));
                        gel_dof_list.Append(conn);
                     }
                  }
                  el++;
               }
            }
         }
      }
   }
   // We must NOT sort gel_dof_list in this case.
   return (new Table(GetGNE(), gel_dof_list));
}

Table *ParNURBSExtension::Get3DGlobalElementDofTable()
{
   int el = 0;
   const KnotVector *kv[3];
   NURBSPatchMap p2g(this);
   Array<Connection> gel_dof_list;

   for (int p = 0; p < GetNP(); p++)
   {
      p2g.SetPatchDofMap(p, kv);

      // Load dofs
      const int ord0 = kv[0]->GetOrder();
      const int ord1 = kv[1]->GetOrder();
      const int ord2 = kv[2]->GetOrder();
      for (int k = 0; k < kv[2]->GetNKS(); k++)
      {
         if (kv[2]->isElement(k))
         {
            for (int j = 0; j < kv[1]->GetNKS(); j++)
            {
               if (kv[1]->isElement(j))
               {
                  for (int i = 0; i < kv[0]->GetNKS(); i++)
                  {
                     if (kv[0]->isElement(i))
                     {
                        Connection conn(el,0);
                        for (int kk = 0; kk <= ord2; kk++)
                        {
                           for (int jj = 0; jj <= ord1; jj++)
                           {
                              for (int ii = 0; ii <= ord0; ii++)
                              {
                                 conn.to = DofMap(p2g(i+ii,j+jj,k+kk));
                                 gel_dof_list.Append(conn);
                              }
                           }
                        }
                        el++;
                     }
                  }
               }
            }
         }
      }
   }
   // We must NOT sort gel_dof_list in this case.
   return (new Table(GetGNE(), gel_dof_list));
}

void ParNURBSExtension::SetActive(const int *partition,
                                  const Array<bool> &active_bel)
{
   activeElem.SetSize(GetGNE());
   activeElem = false;
   NumOfActiveElems = 0;
   const int MyRank = gtopo.MyRank();
   for (int i = 0; i < GetGNE(); i++)
      if (partition[i] == MyRank)
      {
         activeElem[i] = true;
         NumOfActiveElems++;
      }

   active_bel.Copy(activeBdrElem);
   NumOfActiveBdrElems = 0;
   for (int i = 0; i < GetGNBE(); i++)
      if (activeBdrElem[i])
      {
         NumOfActiveBdrElems++;
      }
}

void ParNURBSExtension::BuildGroups(const int *partition,
                                    const Table &elem_dof)
{
   Table dof_proc;

   ListOfIntegerSets  groups;
   IntegerSet         group;

   Transpose(elem_dof, dof_proc); // dof_proc is dof_elem

   // convert elements to processors
   for (int i = 0; i < dof_proc.Size_of_connections(); i++)
   {
      dof_proc.GetJ()[i] = partition[dof_proc.GetJ()[i]];
   }

   // the first group is the local one
   int MyRank = gtopo.MyRank();
   group.Recreate(1, &MyRank);
   groups.Insert(group);

   int dof = 0;
   ldof_group.SetSize(GetNDof());
   for (int d = 0; d < GetNTotalDof(); d++)
      if (activeDof[d])
      {
         group.Recreate(dof_proc.RowSize(d), dof_proc.GetRow(d));
         ldof_group[dof] = groups.Insert(group);

         dof++;
      }

   gtopo.Create(groups, 1822);
}
#endif // MFEM_USE_MPI


void NURBSPatchMap::GetPatchKnotVectors(int p, const KnotVector *kv[])
{
   Ext->patchTopo->GetElementVertices(p, verts);

   if (Ext->Dimension() == 1)
   {
      kv[0] = Ext->knotVectorsCompr[Ext->Dimension()*p];
   }
   else if (Ext->Dimension() == 2)
   {
      Ext->patchTopo->GetElementEdges(p, edges, oedge);

      kv[0] = Ext->knotVectorsCompr[Ext->Dimension()*p];
      kv[1] = Ext->knotVectorsCompr[Ext->Dimension()*p + 1];
   }
   else if (Ext->Dimension() == 3)
   {
      Ext->patchTopo->GetElementEdges(p, edges, oedge);
      Ext->patchTopo->GetElementFaces(p, faces, oface);

      kv[0] = Ext->knotVectorsCompr[Ext->Dimension()*p];
      kv[1] = Ext->knotVectorsCompr[Ext->Dimension()*p + 1];
      kv[2] = Ext->knotVectorsCompr[Ext->Dimension()*p + 2];
   }
   opatch = 0;
}

void NURBSPatchMap::GetBdrPatchKnotVectors(int p, const KnotVector *kv[],
                                           int *okv)
{
   Ext->patchTopo->GetBdrElementVertices(p, verts);

   if (Ext->Dimension() == 2)
   {
      Ext->patchTopo->GetBdrElementEdges(p, edges, oedge);
      kv[0] = Ext->KnotVec(edges[0], oedge[0], &okv[0]);
      opatch = oedge[0];
   }
   else if (Ext->Dimension() == 3)
   {
      faces.SetSize(1);
      Ext->patchTopo->GetBdrElementEdges(p, edges, oedge);
      Ext->patchTopo->GetBdrElementFace(p, &faces[0], &opatch);

      kv[0] = Ext->KnotVec(edges[0], oedge[0], &okv[0]);
      kv[1] = Ext->KnotVec(edges[1], oedge[1], &okv[1]);
   }
}

void NURBSPatchMap::GetMasterEdgeDofs(int edge,
                                      const Array<int>& v_offsets,
                                      const Array<int>& e_offsets,
                                      const Array<int>& aux_e_offsets,
                                      Array<int> & dofs)
{
   {
      const bool isMaster = Ext->masterEdges.count(edge) > 0;
      MFEM_VERIFY(isMaster, "");
   }

   const int mid = Ext->masterEdgeToId.at(edge);
   MFEM_ASSERT(mid >= 0, "Master edge index not found");

   MFEM_VERIFY(Ext->masterEdgeVerts[mid].size() ==
               Ext->masterEdgeSlaves[mid].size() - 1, "");

   for (unsigned int s=0; s<Ext->masterEdgeSlaves[mid].size(); ++s)
   {
      const int slaveId = Ext->slaveEdges[Ext->masterEdgeSlaves[mid][s]];

      Array<int> svert;
      if (slaveId >= 0)
      {
         Ext->patchTopo->GetEdgeVertices(slaveId, svert);
      }
      else
      {
         // Auxiliary edge
         Ext->GetAuxEdgeVertices(-1 - slaveId, svert);
      }

      const int mev = Ext->masterEdgeVerts[mid][std::max((int) s - 1,0)];
      MFEM_VERIFY(mev == svert[0] || mev == svert[1], "");
      bool reverse = false;
      if (s == 0)
      {
         // In this case, mev is the second vertex of the edge.
         if (svert[0] == mev) { reverse = true; }
      }
      else
      {
         // In this case, mev is the first vertex of the edge.
         if (svert[1] == mev) { reverse = true; }
      }

      const int eos = slaveId >= 0 ? e_offsets[slaveId] : aux_e_offsets[-1 - slaveId];

      // TODO: in 3D, the next offset would be f_offsets[0], not
      // p_offsets[0]. This needs to be generalized in an elegant way.
      // How about appending the next offset to the end of e_offsets?
      // Would increasing the size of e_offsets by 1 break something?

      const int eos1 = slaveId >= 0 ? (slaveId + 1 < e_offsets.Size() ?
                                       e_offsets[slaveId + 1] : aux_e_offsets[0]) :
                       aux_e_offsets[-slaveId];

      const int nvs = eos1 - eos;

      MFEM_VERIFY(nvs >= 0, "");

      // Add all slave edge vertices/DOFs

      Array<int> sdofs(nvs);
      for (int j=0; j<nvs; ++j)
      {
         sdofs[j] = reverse ? eos1 - 1 - j : eos + j;
      }

      dofs.Append(sdofs);

      if (s < Ext->masterEdgeSlaves[mid].size() - 1)
      {
         // Add interior vertex DOF
         dofs.Append(v_offsets[Ext->masterEdgeVerts[mid][s]]);
      }
   }
}

// This sets masterDofs but does not change e_meshOffsets.
void NURBSPatchMap::SetMasterEdges(bool dof, const KnotVector *kv[])
{
   const Array<int>& v_offsets = dof ? Ext->v_spaceOffsets : Ext->v_meshOffsets;
   const Array<int>& e_offsets = dof ? Ext->e_spaceOffsets : Ext->e_meshOffsets;

   const Array<int>& aux_e_offsets = dof ? Ext->aux_e_spaceOffsets :
                                     Ext->aux_e_meshOffsets;

   edgeMaster.SetSize(edges.Size());
   edgeMasterOffset.SetSize(edges.Size());
   masterDofs.SetSize(0);

   const int hexEdgeDir[12] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2};
   const int quadEdgeDir[4] = {0, 1, 0, 1};
   const int dim = Ext->Dimension();

   int mos = 0;
   for (int i=0; i<edges.Size(); ++i)
   {
      edgeMaster[i] = Ext->masterEdges.count(edges[i]) > 0;
      edgeMasterOffset[i] = mos;

      if (edgeMaster[i])
      {
         const int mid = Ext->masterEdgeToId.at(edges[i]);
         MFEM_ASSERT(mid >= 0, "Master edge index not found");
         MFEM_ASSERT(Ext->masterEdgeSlaves[mid].size() > 0,
                     "False master edges should have been removed");

         Array<int> mdof;
         GetMasterEdgeDofs(edges[i], v_offsets, e_offsets, aux_e_offsets, mdof);

         for (auto dof : mdof)
         {
            masterDofs.Append(dof);
         }

         mos += mdof.Size();
      }
   }
}

// The input is assumed to be such that the face of patchTopo has {n1,n2}
// interior entities in master face directions {1,2}; v0 is the bottom-left
// vertex with respect to the master face directions; edges {e1,e2} are local
// edges of the face on the bottom and right side (master face directions). We
// find the permutation perm of face interior entities such that entity perm[i]
// of the face should be entity i in the master face ordering.
// Note that, in the above comments, it is irrelevant whether entities are interior.
void NURBSPatchMap::GetFaceOrdering(int face, int n1, int n2, int v0,
                                    int e1, int e2, Array<int> & perm)
{
   perm.SetSize(n1 * n2);

   // The ordering of entities in the face is based on the vertices.

   Array<int> faceEdges, ori, evert, e2vert, vert;
   Ext->patchTopo->GetFaceEdges(face, faceEdges, ori);
   Ext->patchTopo->GetFaceVertices(face, vert);

   MFEM_VERIFY(vert.Size() == 4, "");
   int v0id = -1;
   for (int i=0; i<4; ++i)
   {
      if (vert[i] == v0)
      {
         v0id = i;
      }
   }

   MFEM_VERIFY(v0id >= 0, "");

   Ext->patchTopo->GetEdgeVertices(faceEdges[e1], evert);
   MFEM_VERIFY(evert[0] == v0 || evert[1] == v0, "");

   bool d[2];
   d[0] = (evert[0] == v0);

   const int v10 = d[0] ? evert[1] : evert[0];

   // The face has {fn1,fn2} interior entities, with ordering based on `vert`.
   // Now, we find these sizes, by first finding the edge with vertices [v0, v10].
   int e0 = -1;
   for (int i=0; i<4; ++i)
   {
      Ext->patchTopo->GetEdgeVertices(faceEdges[i], evert);
      if ((evert[0] == v0 && evert[1] == v10) ||
          (evert[1] == v0 && evert[0] == v10))
      {
         e0 = i;
      }
   }

   MFEM_VERIFY(e0 >= 0, "");

   const bool tr = e0 % 2 == 1;  // True means (fn1,fn2) == (n2,n1)

   Ext->patchTopo->GetEdgeVertices(faceEdges[e2], evert);
   MFEM_VERIFY(evert[0] == v10 || evert[1] == v10, "");
   d[1] = (evert[0] == v10);

   const int v11 = d[1] ? evert[1] : evert[0];

   int v01 = -1;
   for (int i=0; i<4; ++i)
   {
      if (vert[i] != v0 && vert[i] != v10 && vert[i] != v11)
      {
         v01 = vert[i];
      }
   }

   MFEM_VERIFY(v01 >= 0 && v01 == vert.Sum() - v0 - v10 - v11, "");

   // Translate indices [v0, v10, v11, v01] to pairs of indices in {0,1}.
   const int ipair[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
   int f00[2];

   int allv[4] = {v0, v10, v11, v01};
   int locv[4];
   for (int i=0; i<4; ++i)
   {
      locv[i] = -1;
      for (int j=0; j<4; ++j)
      {
         if (vert[j] == allv[i])
         {
            locv[i] = j;
         }
      }

      MFEM_VERIFY(locv[i] >= 0, "");
   }

   for (int i=0; i<2; ++i)
   {
      f00[i] = ipair[locv[0]][i];
   }

   const int i0 = f00[0];
   const int j0 = f00[1];

   for (int i=0; i<n1; ++i)
      for (int j=0; j<n2; ++j)
      {
         // Entity perm[i] of the face should be entity i in the master face ordering.
         // The master face ordering varies faster in the direction from v0 to v10,
         // and slower in the direction from v10 to v11, or equivalently, from v0 to v01.
         if (tr)
         {
            const int fi = i0 == 0 ? j : n2 - 1 - j;
            const int fj = j0 == 0 ? i : n1 - 1 - i;
            const int p = fi + (fj * n2);  // Index in face ordering
            const int m = i + (j * n1);  // Index in the master face ordering
            perm[m] = p;
         }
         else
         {
            const int fi = i0 == 0 ? i : n1 - 1 - i;
            const int fj = j0 == 0 ? j : n2 - 1 - j;
            const int p = fi + (fj * n1);  // Index in face ordering
            const int m = i + (j * n1);  // Index in the master face ordering
            perm[m] = p;
         }
      }
}

bool ConsistentlySetEntry(int v, int & e)
{
   const bool consistent = e == -1 || e == v;
   e = v;
   return consistent;
}

void ReorderArray2D(bool rev, int i0, int j0, const Array2D<int> & a,
                    Array2D<int> & b)
{
   const int ma = a.NumRows();
   const int na = a.NumCols();

   const int mb = rev ? na : ma;
   const int nb = rev ? ma : na;

   b.SetSize(mb, nb);

   if (rev)
   {
      MFEM_ABORT("TODO");
   }
   else
   {
      const int s0 = i0 == 0 ? 1 : -1;
      const int s1 = j0 == 0 ? 1 : -1;
      for (int i=0; i<mb; ++i)
      {
         const int ia = (i0 * (ma - 1)) + (s0 * i);
         for (int j=0; j<nb; ++j)
         {
            const int ja = (j0 * (na - 1)) + (s1 * j);
            b(i, j) = a(ia, ja);
         }
      }
   }
}

void GetVertexOrdering(int ori, Array<int> & perm)
{
   perm.SetSize(4);
   const int oriAbs = ori < 0 ? -1 - ori : ori;

   for (int i=0; i<4; ++i)
   {
      if (ori < 0)
      {
         perm[i] = (oriAbs - i + 4) % 4;
      }
      else
      {
         perm[i] = (ori + i) % 4;
      }
   }
}

// This sets masterDofs but does not change f_meshOffsets.
void NURBSPatchMap::SetMasterFaces(bool dof)
{
   const Array<int>& v_offsets = dof ? Ext->v_spaceOffsets : Ext->v_meshOffsets;
   const Array<int>& e_offsets = dof ? Ext->e_spaceOffsets : Ext->e_meshOffsets;
   const Array<int>& f_offsets = dof ? Ext->f_spaceOffsets : Ext->f_meshOffsets;

   const Array<int>& aux_e_offsets = dof ? Ext->aux_e_spaceOffsets :
                                     Ext->aux_e_meshOffsets;
   const Array<int>& aux_f_offsets = dof ? Ext->aux_f_spaceOffsets :
                                     Ext->aux_f_meshOffsets;

   faceMaster.SetSize(faces.Size());
   faceMasterOffset.SetSize(faces.Size());

   // The loop over master edges is already done by SetMasterEdges, and now we
   // append face DOFs to masterDofs.

   int mos = masterDofs.Size();
   for (int i=0; i<faces.Size(); ++i)
   {
      faceMaster[i] = Ext->masterFaces.count(faces[i]) > 0;
      faceMasterOffset[i] = mos;

      if (!faceMaster[i]) { continue; }

      const int mid = Ext->masterFaceToId.at(faces[i]);
      MFEM_ASSERT(mid >= 0, "Master face index not found");

      const bool rev = Ext->masterFaceRev[mid];
      const int s0i = Ext->masterFaceS0[mid][0];
      const int s0j = Ext->masterFaceS0[mid][1];

      const int n1orig = Ext->masterFaceSizes[mid][0];
      const int n2orig = Ext->masterFaceSizes[mid][1];

      // TODO: use a better way to mark master faces without slave faces, instead
      // of just failing to set Ext->masterFaceSizes[mid].
      // Skip master faces with no slave faces (only having slave edges).
      if (n1orig == 0 && n2orig == 0)
      {
         faceMaster[i] = false;
         continue;
      }

      const int n1 = rev ? n2orig : n1orig;
      const int n2 = rev ? n1orig : n2orig;

      MFEM_VERIFY(n1 > 1 || n2 > 1, "");
      MFEM_VERIFY(n1 * n2 >= (int) Ext->masterFaceSlaves[mid].size(),
                  "Inconsistent number of faces");

      MFEM_VERIFY((n1 - 1) * (n2 - 1) >= (int) Ext->masterFaceVerts[mid].size(),
                  "Inconsistent number of vertices");

      // TODO: define masterFaceSlaveCorners at master face corners omitted from V2K?

      int fcnt = 0;
      for (auto slaveId : Ext->masterFaceSlaves[mid])
      {
         if (slaveId < 0)
         {
            // Auxiliary face
            MFEM_ABORT("TODO: aux face implementation is not done");
         }
         else
         {
            fcnt += Ext->slaveFaceNE1[slaveId] * Ext->slaveFaceNE2[slaveId];
         }
      }

      MFEM_VERIFY(fcnt == n1 * n2, "");

      MFEM_VERIFY(Ext->masterFaceSlaveCorners[mid].size() <= n1*n2, "");

      // Set an array of vertices or DOFs for the interior of this master face, `faces[i]`.
      // Set master face entity dimensions.
      int mnf1, mnf2;
      Array<int> medges;
      {
         Array<int> mori;
         Ext->patchTopo->GetFaceEdges(faces[i], medges, mori);
      }

      MFEM_ASSERT(medges.Size() == 4, "");

      if (dof)
      {
         mnf1 = Ext->KnotVec(medges[0])->GetNCP() - 2;
         mnf2 = Ext->KnotVec(medges[1])->GetNCP() - 2;
      }
      else
      {
         mnf1 = Ext->KnotVec(medges[0])->GetNE() - 1;
         mnf2 = Ext->KnotVec(medges[1])->GetNE() - 1;
      }

      // Set dimensions for a single mesh edge.
      int sne1 = (mnf1 - n1 + 1) / n1;
      int sne2 = (mnf2 - n2 + 1) / n2;

      MFEM_ASSERT(sne1 * n1 == mnf1 - n1 + 1, "");
      MFEM_ASSERT(sne2 * n2 == mnf2 - n2 + 1, "");

      Array2D<int> mdof(mnf1, mnf2);
      mdof = -1;

      bool consistent = true;

      for (int s=0; s<Ext->masterFaceSlaves[mid].size(); ++s)
      {
         const int sId = Ext->masterFaceSlaves[mid][s];
         const int slaveId = Ext->slaveFaces[sId];
         const int v0 = Ext->masterFaceSlaveCorners[mid][s];
         const int ori = Ext->slaveFaceOri[sId];
         // ori gives the orientation and index of cv matching the first
         // vertex of childFace, where cv is the array of slave face
         // vertices in CCW w.r.t. to the master face.

         const int sI = Ext->slaveFaceI[sId];
         const int sJ = Ext->slaveFaceJ[sId];
         const int ne1 = Ext->slaveFaceNE1[sId];
         const int ne2 = Ext->slaveFaceNE2[sId];

         const int fos = slaveId >= 0 ? f_offsets[slaveId] : aux_f_offsets[-1 - slaveId];
         const int fos1 = slaveId >= 0 ? (slaveId + 1 < f_offsets.Size() ?
                                          f_offsets[slaveId + 1] : aux_f_offsets[0]) :
                          aux_f_offsets[-slaveId];

         const int nvs = fos1 - fos;

         if (slaveId < 0)
         {
            // Auxiliary face
            const int auxFace = -1 - slaveId;

            // If ori >= 0, then vertex ori of the aux face is closest to vertex 0 of
            // the parent face. If ori < 0, then the orientations of the aux face and parent face
            // are reversed.
            const int oriAbs = ori >= 0 ? ori : -1 - ori;
            const bool revDim = ori % 2 == 1;  // Correct for ori >= 0 and ori < 0.

            // TODO: eliminate auxFaceOri? Isn't it always equal to Ext->slaveFaceOri[sId]?
            // Actually, I think auxFaces has ori defined as 0 for its original parent face,
            // and ori here is Ext->slaveFaceOri, which comes from facePairs, which can be
            // w.r.t. a different parent face. This means auxFaceOri is always 0 and is not
            // useful, so that only ori should be used.
            //MFEM_VERIFY(auxFaceOri == ori, "");

            // TODO: refactor the duplicated code in this case and the other case, slaveId >= 0.

            // Set slave face entity dimensions.
            int nf1, nf2;
            if (dof)
            {
               nf1 = (sne1 * ne1) + ne1 - 1;
               nf2 = (sne2 * ne2) + ne2 - 1;
            }
            else
            {
               nf1 = Ext->auxFaces[auxFace].ki1[0] - Ext->auxFaces[auxFace].ki0[0] - 1;
               nf2 = Ext->auxFaces[auxFace].ki1[1] - Ext->auxFaces[auxFace].ki0[1] - 1;
            }

            MFEM_ASSERT(sne1 * ne1 == nf1 - ne1 + 1, "");
            MFEM_ASSERT(sne2 * ne2 == nf2 - ne2 + 1, "");

            MFEM_VERIFY(nvs == nf1 * nf2, "");

            /*
               NOTE: When an aux face is first defined, it is constructed
            with orientation 0 w.r.t. its parent face. However, it can be
            part of another parent face. In this case, the original aux
            face index is used, but it is paired with a different parent
            face, with possibly different orientation w.r.t. that parent
            face. In NURBSPatchMap::SetMasterFaces, the DOFs are set on
            the parent face in its knot indices, with an aux face of
            possibly nonzero orientation. When the orientation is nonzero,
            the aux face DOFs must be reordered for the parent face, based
            on orientation.
             */

            // These offsets are for the interior entities of the face. To get the lower edge, subtract 1.
            const int os1 = (sI * sne1) + sI;
            const int os2 = (sJ * sne2) + sJ;

            // TODO: is there a function to permute 2D indices based on orientation? GetShiftedGridPoints2D
            // TODO: the problem with using GetShiftedGridPoints2D is that we have (nf1, nf2) for the shifted (new)
            // face, not for the original (old) face.
            int onf1, onf2;
            GetInverseShiftedDimensions2D(ori, nf1, nf2, onf1, onf2);
            for (int k=0; k<onf2; ++k)
               for (int j=0; j<onf1; ++j)
               {
                  int sm, sn, sj, sk;
                  GetShiftedGridPoints2D(onf1, onf2, j, k, ori, sm, sn, sj, sk);
                  MFEM_VERIFY(sm == nf1 && sn == nf2, "TODO: remove this?");

                  const int q = j + (k * onf1);
                  consistent = ConsistentlySetEntry(fos + q, mdof(os1 + sj, os2 + sk));
               }

            // Set entries on edges of this face, if they are interior to the master face.
            std::vector<bool> edgeBdry(4);

            // Horizontal edges
            edgeBdry[0] = sJ == 0;
            edgeBdry[2] = sJ + ne2 == n2;

            // Vertical edges
            edgeBdry[1] = sI + ne1 == n1;
            edgeBdry[3] = sI == 0;

            Array<int> faceEdges;
            Ext->GetAuxFaceEdges(auxFace, faceEdges);

            Array<int> orderedVertices(4);
            Array<int> perm(4);
            GetVertexOrdering(ori, perm);

            std::set<int> vbdry;

            int vstart = v0;
            for (int eidx=0; eidx<4; ++eidx)
            {
               orderedVertices[eidx] = vstart;
               // TODO: eliminate orderedVertices if it is redundant.
               MFEM_VERIFY(orderedVertices[eidx] == Ext->auxFaces[auxFace].v[perm[eidx]], "");

               const int eperm = ori < 0 ? (perm[eidx] - 1 + 4) % 4 : perm[eidx];
               const int edge = faceEdges[eperm];

               Array<int> evert;

               if (edge >= 0)
               {
                  Ext->patchTopo->GetEdgeVertices(edge, evert);
               }
               else
               {
                  const int auxEdge = -1 - edge;
                  Ext->GetAuxEdgeVertices(auxEdge, evert);
               }

               MFEM_VERIFY(evert[0] == vstart || evert[1] == vstart, "");

               const bool reverse = (vstart == evert[1]);
               const int vend = evert.Sum() - vstart;
               vstart = vend;

               // Skip edges on the boundary of the master face.
               if (edgeBdry[eidx])
               {
                  for (auto v : evert)
                  {
                     vbdry.insert(v);
                  }

                  continue;
               }
               const bool horizontal = (eidx % 2 == 0);
               const int nf_e = horizontal ? nf1 : nf2;

               // Edge entities
               const int eos = edge >= 0 ? e_offsets[edge] : aux_e_offsets[-1 - edge];
               const int eos1 = edge >= 0 ? ((edge + 1 < e_offsets.Size()) ?
                                             e_offsets[edge + 1] : aux_e_offsets[0]) : aux_e_offsets[-edge];

               const bool edgeIsMaster = Ext->masterEdges.count(edge) > 0;

               Array<int> edofs;
               if (edgeIsMaster)
               {
                  // This edge is a slave edge and a master edge.
                  // Instead of getting DOFs from e_offsets, take them
                  // from the slave edges of this edge.
                  GetMasterEdgeDofs(edge, v_offsets, e_offsets,
                                    aux_e_offsets, edofs);
               }
               else
               {
                  MFEM_VERIFY(eos1 - eos == nf_e, "");

                  edofs.SetSize(nf_e);
                  for (int j=0; j<nf_e; ++j)
                  {
                     edofs[j] = eos + j;
                  }
               }

               MFEM_VERIFY(edofs.Size() == nf_e, "");

               for (int j=0; j<nf_e; ++j)
               {
                  int m1, m2;
                  if (eidx == 0)
                  {
                     m1 = os1 + j;
                     m2 = os2 - 1;
                  }
                  else if (eidx == 1)
                  {
                     m1 = os1 + nf1;
                     m2 = os2 + j;
                  }
                  else if (eidx == 2)
                  {
                     m1 = os1 + nf1 - 1 - j;
                     m2 = os2 + nf2;
                  }
                  else
                  {
                     m1 = os1 - 1;
                     m2 = os2 + nf2 - 1 - j;
                  }

                  consistent = ConsistentlySetEntry(reverse ? edofs[nf_e - 1 - j] : edofs[j],
                                                    mdof(m1, m2));
               }
            }  // eidx

            // Set entries at vertices of this face, if they are interior to the master face.
            for (int vidx=0; vidx<4; ++vidx)
            {
               const int v = orderedVertices[vidx];
               if (vbdry.count(v) == 0)  // If not on the boundary of the master face.
               {
                  int m1, m2;
                  if (vidx == 0)
                  {
                     m1 = os1 - 1;
                     m2 = os2 - 1;
                  }
                  else if (vidx == 1)
                  {
                     m1 = os1 + nf1;
                     m2 = os2 - 1;
                  }
                  else if (vidx == 2)
                  {
                     m1 = os1 + nf1;
                     m2 = os2 + nf2;
                  }
                  else
                  {
                     m1 = os1 - 1;
                     m2 = os2 + nf2;
                  }

                  consistent = ConsistentlySetEntry(v_offsets[v],
                                                    mdof(m1, m2));
               }
            }  // vidx
         }
         else
         {
            // Determine which slave face edges are in the first and second
            // dimensions of the master face, by using ori.
            int e1 = -1, e2 = -1;
            {
               const int aori = ori < 0 ? -1 - ori : ori;
               if (aori % 2 == 0)
               {
                  e1 = 0;
                  e2 = 1;
               }
               else
               {
                  e1 = 1;
                  e2 = 0;
               }

               if (ori < 0)
               {
                  // Swap e1, e2
                  const int sw = e1;
                  e1 = e2;
                  e2 = sw;
               }
            }

            // Now, e1 is one of the two horizontal edges in the master face directions.
            // If it does not touch v0, then take the other horizontal edge.
            // Do the same for e2.

            Array<int> sedges;
            {
               Array<int> sori;
               Ext->patchTopo->GetFaceEdges(slaveId, sedges, sori);
               MFEM_VERIFY(sedges.Size() == 4, "TODO: remove this obvious check");
            }

            int v1 = -1;
            {
               Array<int> evert;
               Ext->patchTopo->GetEdgeVertices(sedges[e1], evert);
               if (evert.Find(v0) == -1)
               {
                  e1 += 2;
                  Ext->patchTopo->GetEdgeVertices(sedges[e1], evert);
               }

               const int idv0 = evert.Find(v0);
               MFEM_ASSERT(idv0 >= 0, "");
               v1 = evert[1 - idv0];

               Ext->patchTopo->GetEdgeVertices(sedges[e2], evert);
               if (evert.Find(v1) == -1)
               {
                  e2 += 2;
                  Ext->patchTopo->GetEdgeVertices(sedges[e2], evert);
               }

               MFEM_ASSERT(evert.Find(v1) >= 0, "");
            }

            // Set slave face entity dimensions.
            int nf1, nf2;
            if (dof)
            {
               nf1 = Ext->KnotVec(sedges[e1])->GetNCP() - 2;
               nf2 = Ext->KnotVec(sedges[e2])->GetNCP() - 2;
            }
            else
            {
               nf1 = Ext->KnotVec(sedges[e1])->GetNE() - 1;
               nf2 = Ext->KnotVec(sedges[e2])->GetNE() - 1;
            }

            MFEM_ASSERT(sne1 * ne1 == nf1 - ne1 + 1, "");
            MFEM_ASSERT(sne2 * ne2 == nf2 - ne2 + 1, "");

            MFEM_VERIFY(nvs == nf1 * nf2, "");

            // Find the DOFs of the slave face ordered for the master
            // face. We know that e1 and e2 are the local indices of
            // the slave face edges on the bottom and right side, with
            // respect to the master face directions.
            Array<int> perm;
            GetFaceOrdering(slaveId, nf1, nf2, v0, e1, e2, perm);

            // These offsets are for the interior entities of the face.
            // To get the lower edge, subtract 1.
            const int os1 = (sI * sne1) + sI;
            const int os2 = (sJ * sne2) + sJ;

            for (int k=0; k<nf2; ++k)
               for (int j=0; j<nf1; ++j)
               {
                  const int q = j + (k * nf1);
                  consistent = ConsistentlySetEntry(fos + perm[q], mdof(os1 + j, os2 + k));
               }

            // Set entries on edges of this face, if they are interior to the master face.
            std::vector<int> edgeOrder{e1, e2, (e1 + 2) % 4, (e2 + 2) % 4};
            std::vector<bool> edgeBdry(4);

            // Horizontal edges
            edgeBdry[0] = sJ == 0;
            edgeBdry[2] = sJ + ne2 == n2;

            // Vertical edges
            edgeBdry[1] = sI + ne1 == n1;
            edgeBdry[3] = sI == 0;

            Array<int> orderedVertices(4);

            std::set<int> vbdry;

            int vstart = v0;
            for (int eidx=0; eidx<4; ++eidx)
            {
               orderedVertices[eidx] = vstart;

               const int edge = sedges[edgeOrder[eidx]];

               Array<int> evert;
               Ext->patchTopo->GetEdgeVertices(edge, evert);

               const bool reverse = (vstart == evert[1]);

               const int vend = evert.Sum() - vstart;
               vstart = vend;

               // Skip edges on the boundary of the master face.
               if (edgeBdry[eidx])
               {
                  for (auto v : evert)
                  {
                     vbdry.insert(v);
                  }

                  continue;
               }
               const bool horizontal = (eidx % 2 == 0);
               const int nf_e = horizontal ? nf1 : nf2;

               // Edge entities
               const int eos = e_offsets[edge];
               const int eos1 = (edge + 1 < e_offsets.Size()) ?
                                e_offsets[edge + 1] : aux_e_offsets[0];

               const bool edgeIsMaster = Ext->masterEdges.count(edge) > 0;

               Array<int> edofs;
               if (edgeIsMaster)
               {
                  // TODO: shouldn't the DOFs from this master edge be obtainable from SetMasterEdges?
                  // I guess SetMasterEdges is only for the patch edges, not master edges on the interior
                  // of a patch face.
                  // However, can we reuse/refactor code from SetMasterEdges?
                  GetMasterEdgeDofs(edge, v_offsets, e_offsets, aux_e_offsets, edofs);
               }
               else
               {
                  MFEM_VERIFY(eos1 - eos == nf_e, "");

                  edofs.SetSize(nf_e);
                  for (int j=0; j<nf_e; ++j)
                  {
                     edofs[j] = eos + j;
                  }
               }

               MFEM_VERIFY(edofs.Size() == nf_e, "");

               for (int j=0; j<nf_e; ++j)
               {
                  int m1, m2;
                  if (eidx == 0)
                  {
                     m1 = os1 + j;
                     m2 = os2 - 1;
                  }
                  else if (eidx == 1)
                  {
                     m1 = os1 + nf1;
                     m2 = os2 + j;
                  }
                  else if (eidx == 2)
                  {
                     m1 = os1 + nf1 - 1 - j;
                     m2 = os2 + nf2;
                  }
                  else
                  {
                     m1 = os1 - 1;
                     m2 = os2 + nf2 - 1 - j;
                  }

                  consistent = ConsistentlySetEntry(reverse ? edofs[nf_e - 1 - j] : edofs[j],
                                                    mdof(m1, m2));
               }
            }  // eidx

            // Set entries at vertices of this face, if they are interior to the master face.
            for (int vidx=0; vidx<4; ++vidx)
            {
               const int v = orderedVertices[vidx];
               if (vbdry.count(v) == 0)  // If not on the boundary of the master face.
               {
                  int m1, m2;
                  if (vidx == 0)
                  {
                     m1 = os1 - 1;
                     m2 = os2 - 1;
                  }
                  else if (vidx == 1)
                  {
                     m1 = os1 + nf1;
                     m2 = os2 - 1;
                  }
                  else if (vidx == 2)
                  {
                     m1 = os1 + nf1;
                     m2 = os2 + nf2;
                  }
                  else
                  {
                     m1 = os1 - 1;
                     m2 = os2 + nf2;
                  }

                  consistent = ConsistentlySetEntry(v_offsets[v],
                                                    mdof(m1, m2));
               }
            }  // vidx
         }
      }  // Loop (s) over slave faces.

      // Let `ori` be the signed shift such that pv[abs1(ori)] is vertex 0 of parentFace,
      // and the relative direction of the ordering is encoded in the sign.
      // Here, pv means parent vertices, as ordered in the mesh file.
      // Then Reorder2D takes `ori` and computes (s0i, s0j) as the
      // corresponding integer coordinates in {0,1}x{0,1}. Thus reference
      // vertex (s0i, s0j) of pv (parent vertices in mesh file) is vertex
      // (0, 0) of parentFace.
      // Currently, mdof is in the ordering of pv, and the entries are now reordered,
      // according to parentFace vertex ordering, for appending to masterDofs.
      // That means the first entry appended to masterDofs should be the entry of
      // mdof corresponding to (s0i, s0j).

      Array2D<int> mdof_reordered;
      ReorderArray2D(rev, s0i, s0j, mdof, mdof_reordered);

      bool allset = true;
      for (int j=0; j<mdof_reordered.NumCols(); ++j)
         for (int k=0; k<mdof_reordered.NumRows(); ++k)
         {
            if (mdof_reordered(k,j) < 0)
            {
               allset = false;
            }

            masterDofs.Append(mdof_reordered(k,j));
         }

      MFEM_ASSERT(allset, "");
      MFEM_ASSERT(consistent, "");

      mos += mnf1 * mnf2;
   }  // loop (i) over faces
}

int NURBSPatchMap::GetMasterEdgeDof(const int e, const int i) const
{
   const int os = edgeMasterOffset[e];
   return masterDofs[os + i];
}

int NURBSPatchMap::GetMasterFaceDof(const int f, const int i) const
{
   const int os = faceMasterOffset[f];
   return masterDofs[os + i];
}

void NURBSPatchMap::SetPatchVertexMap(int p, const KnotVector *kv[])
{
   GetPatchKnotVectors(p, kv);

   I = kv[0]->GetNE() - 1;

   for (int i = 0; i < verts.Size(); i++)
   {
      verts[i] = Ext->v_meshOffsets[verts[i]];
   }

   if (Ext->Dimension() >= 2)
   {
      J = kv[1]->GetNE() - 1;
      SetMasterEdges(false, kv);
      for (int i = 0; i < edges.Size(); i++)
      {
         edges[i] = Ext->e_meshOffsets[edges[i]];
      }
   }
   if (Ext->Dimension() == 3)
   {
      K = kv[2]->GetNE() - 1;
      SetMasterFaces(false);
      for (int i = 0; i < faces.Size(); i++)
      {
         faces[i] = Ext->f_meshOffsets[faces[i]];
      }
   }

   pOffset = Ext->p_meshOffsets[p];
}

void NURBSPatchMap::SetPatchDofMap(int p, const KnotVector *kv[])
{
   GetPatchKnotVectors(p, kv);

   I = kv[0]->GetNCP() - 2;

   for (int i = 0; i < verts.Size(); i++)
   {
      verts[i] = Ext->v_spaceOffsets[verts[i]];
   }
   if (Ext->Dimension() >= 2)
   {
      J = kv[1]->GetNCP() - 2;
      SetMasterEdges(true);

      if (Ext->nonconforming && Ext->patchTopo->ncmesh
          && Ext->patchTopo->ncmesh->GetVertexToKnot().NumRows() > 0)
      {
         // Use e2nce to map from patchTopo edges to patchTopo->ncmesh edges.
         for (int i = 0; i < edges.Size(); i++)
         {
            int nce = -1;
            auto s = Ext->e2nce.find(edges[i]);
            if (s != Ext->e2nce.end())
            {
               nce = s->second;
            }
            else
            {
               MFEM_ABORT("TODO");
            }

            edges[i] = Ext->e_spaceOffsets[nce];
         }
      }
      else
      {
         for (int i = 0; i < edges.Size(); i++)
         {
            edges[i] = Ext->e_spaceOffsets[edges[i]];
         }
      }
   }
   if (Ext->Dimension() == 3)
   {
      K = kv[2]->GetNCP() - 2;
      SetMasterFaces(true);
      for (int i = 0; i < faces.Size(); i++)
      {
         faces[i] = Ext->f_spaceOffsets[faces[i]];
      }
   }

   pOffset = Ext->p_spaceOffsets[p];
}

void NURBSPatchMap::SetBdrPatchVertexMap(int p, const KnotVector *kv[],
                                         int *okv)
{
   GetBdrPatchKnotVectors(p, kv, okv);

   for (int i = 0; i < verts.Size(); i++)
   {
      verts[i] = Ext->v_meshOffsets[verts[i]];
   }

   if (Ext->Dimension() == 1)
   {
      I = 0;
   }
   else if (Ext->Dimension() == 2)
   {
      I = kv[0]->GetNE() - 1;
      pOffset = Ext->e_meshOffsets[edges[0]];
   }
   else if (Ext->Dimension() == 3)
   {
      I = kv[0]->GetNE() - 1;
      J = kv[1]->GetNE() - 1;

      SetMasterEdges(false);
      SetMasterFaces(false);
      for (int i = 0; i < edges.Size(); i++)
      {
         edges[i] = Ext->e_meshOffsets[edges[i]];
      }

      pOffset = Ext->f_meshOffsets[faces[0]];
   }
}

void NURBSPatchMap::SetBdrPatchDofMap(int p, const KnotVector *kv[],  int *okv)
{
   GetBdrPatchKnotVectors(p, kv, okv);

   for (int i = 0; i < verts.Size(); i++)
   {
      verts[i] = Ext->v_spaceOffsets[verts[i]];
   }

   if (Ext->Dimension() == 1)
   {
      I = 0;
   }
   else if (Ext->Dimension() == 2)
   {
      I = kv[0]->GetNCP() - 2;
      pOffset = Ext->e_spaceOffsets[edges[0]];
   }
   else if (Ext->Dimension() == 3)
   {
      I = kv[0]->GetNCP() - 2;
      J = kv[1]->GetNCP() - 2;

      SetMasterEdges(true);

      for (int i = 0; i < edges.Size(); i++)
      {
         edges[i] = Ext->e_spaceOffsets[edges[i]];
      }

      pOffset = Ext->f_spaceOffsets[faces[0]];
   }
}

void NURBSExtension::ProcessVertexToKnot2D(const Array2D<int> & v2k,
                                           std::set<int> & reversedParents,
                                           std::vector<EdgePairInfo> & edgePairs)
{
   auxEdges.clear();

   std::map<std::pair<int, int>, int> auxv2e;

   const int nv2k = v2k.NumRows();
   MFEM_VERIFY(4 == v2k.NumCols(), "");

   int prevParent = -1;
   int prevV = -1;
   int prevKI = -1;
   for (int i=0; i<nv2k; ++i)
   {
      const int tv = v2k(i,0);
      const int knotIndex = v2k(i,1);
      const int pv0 = v2k(i,2);
      const int pv1 = v2k(i,3);

      // Given that the parent Mesh is not yet constructed, and all we have at
      // this point is patchTopo->ncmesh, we should only define master/slave
      // edges by indices in patchTopo->ncmesh, as done in the case of nonempty
      // nce.masters. Now find the edge in patchTopo->ncmesh with vertices
      // (pv0, pv1), and define it as a master edge.

      const std::pair<int, int> parentPair(pv0 < pv1 ? pv0 : pv1,
                                           pv0 < pv1 ? pv1 : pv0);

      MFEM_VERIFY(v2e.count(parentPair) > 0, "Vertex pair not found");

      const int parentEdge = v2e[parentPair];
      masterEdges.insert(parentEdge);

      if (pv1 < pv0)
      {
         reversedParents.insert(parentEdge);
      }

      // Note that the logic here assumes that the "vertex_to_knot" data in the
      // mesh file has vertices in order of ascending knotIndex.

      const bool newParentEdge = (prevParent != parentEdge);
      const int v0 = newParentEdge ? pv0 : prevV;

      if (knotIndex == 1)
      {
         MFEM_VERIFY(newParentEdge, "");
      }

      // Find the edge in patchTopo->ncmesh with vertices (v0, tv), and define
      // it as a slave edge.

      const std::pair<int, int> childPair(v0 < tv ? v0 : tv, v0 < tv ? tv : v0);
      const bool childPairTopo = v2e.count(childPair) > 0;
      if (!childPairTopo)
      {
         // Check whether childPair is in auxEdges.
         if (auxv2e.count(childPair) == 0)
         {
            // Create a new auxiliary edge
            auxv2e[childPair] = auxEdges.size();
            auxEdges.emplace_back(AuxiliaryEdge{pv0 < pv1 ? parentEdge : -1 - parentEdge,
                                                childPair.first, childPair.second,
                                                newParentEdge ? 0 : prevKI, knotIndex});
         }
      }

      const int childEdge = childPairTopo ? v2e[childPair] : -1 - auxv2e[childPair];

      // Check whether this is the final vertex in this parent edge.
      // TODO: this logic for comparing (pv0,pv1) to the next parents assumes
      // the ordering won't change. If the next v2k entry has (pv1,pv0), this
      // would cause a bug. An improvement in the implementation should avoid
      // this issue. Or is it not possible to change the order, since the knot
      // index is assumed to increase from pv0 to pv1?
      const bool finalVertex = (i == nv2k-1) || (v2k(i+1,2) != pv0) ||
                               (v2k(i+1,3) != pv1);

      edgePairs.emplace_back(tv, knotIndex, childEdge, parentEdge);

      if (finalVertex)
      {
         // Also find the edge with vertices (tv, pv1), and define it as a slave
         // edge.
         const std::pair<int, int> finalChildPair(tv < pv1 ? tv : pv1,
                                                  tv < pv1 ? pv1 : tv);
         const bool finalChildPairTopo = v2e.count(finalChildPair) > 0;
         if (!finalChildPairTopo)
         {
            // Check whether finalChildPair is in auxEdges.
            if (auxv2e.count(finalChildPair) == 0)
            {
               // Create a new auxiliary edge
               auxv2e[finalChildPair] = auxEdges.size();

               // -1 denotes `ne` at endpoint
               auxEdges.emplace_back(AuxiliaryEdge{pv0 < pv1 ? -1 - parentEdge : parentEdge,
                                                   finalChildPair.first, finalChildPair.second,
                                                   knotIndex, -1});
            }
         }

         const int finalChildEdge = finalChildPairTopo ? v2e[finalChildPair]: -1 -
                                    auxv2e[finalChildPair];
         edgePairs.emplace_back(-1, -1, finalChildEdge, parentEdge);
      }

      prevV = tv;
      prevKI = knotIndex;
      prevParent = parentEdge;
   }  // loop over vertices in vertex_to_knot
}

// TODO: better code design.
void NURBSExtension::ProcessVertexToKnot3D(Array2D<int> const& v2k,
                                           const std::map<std::pair<int, int>, int> & v2f,
                                           std::vector<int> & parentN1,
                                           std::vector<int> & parentN2,
                                           std::vector<EdgePairInfo> & edgePairs,
                                           std::vector<FacePairInfo> & facePairs,
                                           std::vector<int> & parentFaces,
                                           std::vector<int> & parentVerts)
{
   auxEdges.clear();
   auxFaces.clear();

   std::map<std::pair<int, int>, int> auxv2e, auxv2f;

   // Each entry of v2k has the following 7 entries: tv, ki1, ki2, p0, p1, p2, p3
   constexpr int np = 7;  // Number of integers for each entry in v2k.

   const int nv2k = v2k.NumRows();
   MFEM_VERIFY(np == v2k.NumCols(), "");

   // Note that the logic here assumes that the "vertex_to_knot" data
   // in the mesh file has vertices in order of ascending (k1,k2), with k2
   // being the fast variable, and with corners skipped.
   // TODO: is this assumption still required and valid?

   // Find parentOffset, which stores the indices in v2k at which parent faces start.
   int prevParent = -1;
   std::vector<int> parentOffset;
   std::vector<bool> parentV2Kedge;
   int n1 = 0;
   int n2 = 0;
   int n1min = 0;
   int n2min = 0;
   for (int i = 0; i < nv2k; ++i)
   {
      const int ki1 = v2k(i,1);
      const int ki2 = v2k(i,2);

      std::vector<int> pv(4);
      for (int j=0; j<4; ++j)
      {
         pv[j] = v2k(i, 3 + j);
      }

      // The face with vertices (pv0, pv1, pv2, pv3) is defined as a parent face.
      const auto pvmin = std::min_element(pv.begin(), pv.end());
      const int idmin = std::distance(pv.begin(), pvmin);
      const int c0 = pv[idmin];  // First corner
      const int c1 = pv[(idmin + 2) % 4];  // Opposite corner

      const std::pair<int, int> parentPair(c0, c1);

      const int parentFace = v2f.at(parentPair);
      const bool newParentFace = (prevParent != parentFace);
      if (newParentFace)
      {
         parentOffset.push_back(i);
         parentFaces.push_back(parentFace);

         if (i > 0)
         {
            // In the case of only 1 element in the 1-direction, it is assumed that
            // the 2-direction has more than 1 element, so there are knots (0, ki2)
            // and (1, ki2) for 0 < ki2 < n2. This will result in n1 = 0, which
            // should be 1. Also, n2 will be 1 less than it should be.
            // Similarly for the situation with directions reversed.
            // TODO: fix/test this in the 1-element case.

            const int n1range = n1 - n1min;
            const int n2range = n2 - n2min;
            parentV2Kedge.push_back(n1range == 0 || n2range == 0);
         }

         // TODO: refactor!
         {
            Array<int> ev(2);
            for (int j=0; j<2; ++j)
            {
               ev[j] = pv[j];
            }

            ev.Sort();
            const std::pair<int, int> edge_v(ev[0], ev[1]);
            const int edge = v2e.at(edge_v);

            n1 = KnotVecNE(edge);
            parentN1.push_back(n1);
         }

         {
            Array<int> ev(2);
            for (int j=0; j<2; ++j)
            {
               ev[j] = pv[j+1];
            }

            ev.Sort();
            const std::pair<int, int> edge_v(ev[0], ev[1]);
            const int edge = v2e.at(edge_v);

            n2 = KnotVecNE(edge);
            parentN2.push_back(n2);
         }

         n1 = ki1;  // Finding max of ki1
         n2 = ki2;  // Finding max of ki2

         n1min = n1;
         n2min = n2;
      }
      else
      {
         n1 = std::max(n1, ki1);  // Finding max of ki1
         n2 = std::max(n2, ki2);  // Finding max of ki2

         n1min = std::min(n1min, ki1);
         n2min = std::min(n2min, ki2);
      }

      prevParent = parentFace;
   }

   {
      const int n1range = n1 - n1min;
      const int n2range = n2 - n2min;
      parentV2Kedge.push_back(n1range == 0 || n2range == 0);
   }

   const int numParents = parentOffset.size();
   parentOffset.push_back(nv2k);

   std::set<int> visitedParentEdges;
   std::map<int, int> edgePairOS;
   bool consistent = true;

   for (int parent = 0; parent < numParents; ++parent)
   {
      const int parentFace = parentFaces[parent];

      int parentEdges[4];
      bool parentEdgeRev[4];

      // Set all 4 edges of the parent face as master edges.
      {
         Array<int> ev(2);
         for (int i=0; i<4; ++i)
         {
            for (int j=0; j<2; ++j)
            {
               ev[j] = v2k(parentOffset[parent],3 + ((i + j) % 4));
            }

            const bool reverse = (ev[1] < ev[0]);
            parentEdgeRev[i] = reverse;

            ev.Sort();

            const std::pair<int, int> edge_i(ev[0], ev[1]);

            const int parentEdge = v2e.at(edge_i);
            masterEdges.insert(parentEdge);
            parentEdges[i] = parentEdge;
         }
      }

      n1 = parentN1[parent];
      n2 = parentN2[parent];
      Array2D<int> gridVertex(n1 + 1, n2 + 1);

      for (int i=0; i<=n1; ++i)
         for (int j=0; j<=n2; ++j)
         {
            gridVertex(i,j) = -1;
         }

      gridVertex(0,0) = v2k(parentOffset[parent],3);
      gridVertex(n1,0) = v2k(parentOffset[parent],4);
      gridVertex(n1,n2) = v2k(parentOffset[parent],5);
      gridVertex(0,n2) = v2k(parentOffset[parent],6);

      for (int i=0; i<4; ++i)
      {
         parentVerts.push_back(v2k(parentOffset[parent],3 + i));
      }

      int r1min = -1;
      int r1max = -1;
      int r2min = -1;
      int r2max = -1;

      for (int i = parentOffset[parent]; i < parentOffset[parent + 1]; ++i)
      {
         const int tv = v2k(i,0);
         const int ki1 = v2k(i,1);
         const int ki2 = v2k(i,2);

         gridVertex(ki1, ki2) = tv;

         if (i == parentOffset[parent])
         {
            // Initialize min/max
            r1min = ki1;
            r1max = ki1;

            r2min = ki2;
            r2max = ki2;
         }
         else
         {
            r1min = std::min(r1min, ki1);
            r1max = std::max(r1max, ki1);

            r2min = std::min(r2min, ki2);
            r2max = std::max(r2max, ki2);
         }
      } // loop over vertices in v2k

      const int n1set = r1max - r1min + 1;
      const int n2set = r2max - r2min + 1;

      MFEM_VERIFY(n1set * n2set >= parentOffset[parent + 1] - parentOffset[parent],
                  "");

      bool allset = true;
      bool hasSlaveFaces = false;
      bool hasAuxFace = false;
      for (int i=0; i<=n1; ++i)
         for (int j=0; j<=n2; ++j)
         {
            if (gridVertex(i,j) < 0)
            {
               allset = false;
            }
            else if (0 < i && i < n1 && 0 < j && j < n2)
            {
               hasSlaveFaces = true;
            }
         }

      // Loop over child faces and set facePairs, as well as auxiliary faces as needed.
      // TODO: just loop over (r1min, r1max) and (r2min, r2max)?
      for (int i=0; i<n1; ++i)
         for (int j=0; j<n2; ++j)
         {
            std::vector<int> cv(4);
            cv[0] = gridVertex(i,j);
            cv[1] = gridVertex(i+1,j);
            cv[2] = gridVertex(i+1,j+1);
            cv[3] = gridVertex(i,j+1);

            const auto cvmin = std::min_element(cv.begin(), cv.end());
            const int idmin = std::distance(cv.begin(), cvmin);
            const int c0 = cv[idmin];  // First corner
            const int c1 = cv[(idmin + 2) % 4];  // Opposite corner

            if (c0 < 0)  // This may occur, if gridVertex is not set everywhere.
            {
               continue;
            }

            const std::pair<int, int> childPair(c0 < c1 ? c0 : c1, c0 < c1 ? c1 : c0);
            const bool childPairTopo = v2f.count(childPair) > 0;
            if (childPairTopo)
            {
               const int childFace = v2f.at(childPair);
               Array<int> acv(cv.data(), 4);
               const int ori = GetFaceOrientation(patchTopo, childFace, acv);
               // ori gives the orientation and index of cv matching the first vertex of childFace.

               facePairs.emplace_back(FacePairInfo{cv[0], childFace, parentFace, ori,
                                                   i, j, 1, 1});
            }
            else
            {
               hasAuxFace = true;
               // Check whether childPair is in auxFaces.
               if (auxv2f.count(childPair) == 0)
               {
                  MFEM_ABORT("TODO: set facePairs?");

                  // Create a new auxiliary face
                  auxv2f[childPair] = auxFaces.size();

                  AuxiliaryFace auxFace;
                  for (int k=0; k<4; ++k)
                  {
                     auxFace.v[k] = cv[k];
                  }

                  auxFace.parent = parentFace;
                  // Orientation is 0 for a new auxiliary face, by construction.
                  auxFace.ori = 0;

                  auxFace.ki0[0] = i;
                  auxFace.ki0[1] = j;
                  auxFace.ki1[0] = i + 1;
                  auxFace.ki1[1] = j + 1;

                  auxFaces.push_back(auxFace);
               }
            }
         }

      // Loop over child boundary edges and set edgePairs.
      for (int dir=1; dir<=2; ++dir)
      {
         const int ne = dir == 1 ? n1 : n2;
         for (int s=0; s<2; ++s)  // Loop over 2 sides for this direction.
         {
            const int parentEdge = parentEdges[dir == 1 ? 2*s : (2*s) + 1];
            const bool reverse_p = parentEdgeRev[dir == 1 ? 2*s : (2*s) + 1];
            const bool reverse = s == 0 ? reverse_p :
                                 !reverse_p;  // Sides with s=1 are reversed in defining parentEdgeRev.

            const bool parentVisited = visitedParentEdges.count(parentEdge) > 0;

            if (!parentVisited)
            {
               edgePairOS[parentEdge] = edgePairs.size();
               const int os0 = edgePairs.size();
               edgePairs.resize(edgePairs.size() + ne);
            }

            int tvprev = -1;  // TODO: change name
            int kiprev = -1;  // TODO: change name
            bool lagTV = false;

            int firstEdge = -1;

            for (int e_i=0; e_i<ne; ++e_i)  // edges in direction `dir`
            {
               // For both directions, side s=0 has increasing indices and
               // s=1 has decreasing indices.

               const int i0 = e_i;
               const int i1 = e_i + 1;

               const int e_idx = reverse ? ne - e_i - 1 :
                                 e_i;  // Edge index with respect to the master edge.

               Array<int> cv(2);
               if (dir == 1)
               {
                  cv[0] = gridVertex(i0,s*n2);
                  cv[1] = gridVertex(i1,s*n2);
               }
               else
               {
                  cv[0] = gridVertex((1-s)*n1, i0);
                  cv[1] = gridVertex((1-s)*n1, i1);
               }

               const int cv0 = cv[0];

               int tv_int = -1;  // Top-vertex interior to the master edge
               int ki = -1;  // Knot index of tv_int, with respect to the master edge

               if (lagTV)
               {
                  tv_int = tvprev;
                  ki = kiprev;
               }

               if (tvprev == -1)
               {
                  tvprev = (i0 == 0 ||
                            i0 == ne) ? cv[1] : cv[0];  // Top-vertex interior to the master edge
                  kiprev = (i0 == 0 || i0 == ne) ? i1 : i0;
               }
               else if (e_i < ne - 1)  // Don't set to the endpoint
               {
                  kiprev = (tvprev == cv[0]) ? i1 : i0;
                  tvprev = (tvprev == cv[0]) ? cv[1] :
                           cv[0];  // Next interior vertex along the master edge
               }

               if (!lagTV)
               {
                  tv_int = tvprev;
                  ki = kiprev;
               }

               cv.Sort();

               if (cv[0] < 0)  // This may occur if gridVertex is not set everywhere.
               {
                  continue;
               }
               else if (firstEdge == -1)
               {
                  firstEdge = e_i;
                  if (e_i > 0)
                  {
                     tv_int = cv0;
                     ki = i0;
                     tvprev = cv.Sum() - cv0;  // cv1
                     kiprev = i1;
                     lagTV = true;
                  }
               }

               const int tv = (e_idx == ne - 1) ? -1 : tv_int;
               const int tvki = (e_idx == ne - 1) ? -1 : (reverse ? ne - ki : ki);

               if (tv == -1)
               {
                  lagTV = true;
               }

               const std::pair<int, int> edge_i(cv[0], cv[1]);

               const int childEdge = v2e.at(edge_i);

               if (!parentVisited)
               {
                  // edgePairs is ordered starting from the vertex of lower index.
                  edgePairs[edgePairOS[parentEdge] + e_idx].Set(tv, tvki, childEdge, parentEdge);
               }
               else
               {
                  // Consistency check
                  const int os = edgePairOS[parentEdge];
                  if (edgePairs[os + e_idx].child != childEdge ||
                      edgePairs[os + e_idx].parent != parentEdge)
                  {
                     consistent = false;
                  }
               }
            }

            visitedParentEdges.insert(parentEdge);
         }
      } // dir

      // Set auxiliary and patch-slave faces outside the set gridVertex.
      // Here, patch-slave refers to a slave face that is a face of a
      // neighboring patch and may contain multiple mesh faces.
      // In general, there can be at most 8 = 3^2 - 1 such faces.

      Array<int> gv1(4);
      Array<int> gv2(4);

      gv1[0] = 0;
      gv1[1] = r1min;
      gv1[2] = r1max;
      gv1[3] = n1;

      gv2[0] = 0;
      gv2[1] = r2min;
      gv2[2] = r2max;
      gv2[3] = n2;

      if (hasSlaveFaces && !allset)
      {
         for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
            {
               if (i == 1 && j == 1)  // Skip the middle, which is covered by gridVertex.
               {
                  continue;
               }

               // Define an auxiliary face (gv1[i], gv1[i+1]) x (gv2[j], gv2[j+1])

               // Skip degenerate faces
               if (gv1[i] == gv1[i+1] || gv2[j] == gv2[j+1])
               {
                  continue;
               }

               // TODO: the following code is copied and modified from above. Refactor into a function?
               std::vector<int> cv(4);
               std::vector<int> cv0(4);  // TODO: remove?
               std::vector<int> cv1(4);  // TODO: remove?

               cv[0] = gridVertex(gv1[i],gv2[j]);
               cv[1] = gridVertex(gv1[i+1],gv2[j]);
               cv[2] = gridVertex(gv1[i+1],gv2[j+1]);
               cv[3] = gridVertex(gv1[i],gv2[j+1]);

               cv0[0] = gv1[i];
               cv0[1] = gv1[i+1];
               cv0[2] = gv1[i+1];
               cv0[3] = gv1[i];

               cv1[0] = gv2[j];
               cv1[1] = gv2[j];
               cv1[2] = gv2[j+1];
               cv1[3] = gv2[j+1];

               const auto cvmin = std::min_element(cv.begin(), cv.end());
               const int idmin = std::distance(cv.begin(), cvmin);
               const int c0 = cv[idmin];  // First corner
               const int c1 = cv[(idmin + 2) % 4];  // Opposite corner

               MFEM_ASSERT(c0 >= 0, "");

               const std::pair<int, int> childPair(c0, c1);
               const bool childPairTopo = v2f.count(childPair) > 0;
               if (childPairTopo)
               {
                  const int childFace = v2f.at(childPair);
                  Array<int> acv(cv.data(), 4);

                  // ori gives the orientation and index of cv matching the
                  // first vertex of childFace.
                  const int ori = GetFaceOrientation(patchTopo, childFace, acv);
                  facePairs.emplace_back(FacePairInfo{cv[0], childFace, parentFace, ori,
                                                      gv1[i], gv2[j], gv1[i+1] - gv1[i], gv2[j+1] - gv2[j]});
               }
               else
               {
                  hasAuxFace = true;

                  // Check whether childPair is in auxFaces.
                  if (auxv2f.count(childPair) == 0)
                  {
                     // Create a new auxiliary face
                     auxv2f[childPair] = auxFaces.size();
                     AuxiliaryFace auxFace;
                     for (int k=0; k<4; ++k)
                     {
                        auxFace.v[k] = cv[k];
                     }

                     auxFace.parent = parentFace;
                     // Orientation is 0 for a new auxiliary face, by construction.
                     auxFace.ori = 0;

                     auxFace.ki0[0] = gv1[i];
                     auxFace.ki0[1] = gv2[j];
                     auxFace.ki1[0] = gv1[i+1];
                     auxFace.ki1[1] = gv2[j+1];

                     auxFaces.push_back(auxFace);

                     // Orientation is 0 for a new auxiliary face, by construction.
                     facePairs.emplace_back(FacePairInfo{cv[0], -1 - auxv2f[childPair], parentFace, 0,
                                                         gv1[i], gv2[j], gv1[i+1] - gv1[i], gv2[j+1] - gv2[j]});

                     // TODO: is it redundant to put parentFace in both
                     // facePairs and auxFaces?
                     // TODO: is it redundant to put gv1[i+{0,1}], gv2[j+{0,1}] in both
                     // facePairs and auxFaces?
                     // TODO: can facePairs and auxFaces be combined into
                     // one array of a struct?
                  }
               }
            }
      }

      // Set auxiliary edges outside the gridVertex, on the boundary of the parent face.
      // Note that auxiliary edges cannot simply be found as edges of auxiliary faces, because
      // the faces above are defined only on parent faces listed in V2K data. Other auxiliary
      // faces will be defined in FindAdditionalSlaveAndAuxiliaryFaces by using auxiliary edges found below.

      // Auxiliary edges in first and second directions
      for (int d=0; d<2; ++d)
      {
         // TODO: interchange i- and s- loops?
         for (int i=0; i<3; ++i)
         {
            if (i == 1)  // Skip the middle, which is covered by gridVertex.
            {
               continue;
            }

            for (int s=0; s<2; ++s)  // Loop over 2 sides in this direction
            {
               // Set gv1 (d == 0) or gv2 (d == 1) for this edge.
               // Set range of set knot indices for this edge.
               int rmin = -1;
               int rmax = -1;
               const int n_d = d == 0 ? n1 : n2;
               for (int j=1; j<n_d; ++j)
               {
                  const int tv = d == 0 ? gridVertex(j,s*n2) : gridVertex((1-s)*n1,j);
                  if (tv >= 0)
                  {
                     if (rmin == -1)
                     {
                        // Initialize range
                        rmin = j;
                        rmax = j;
                     }
                     else
                     {
                        rmin = std::min(rmin, j);
                        rmax = std::max(rmax, j);
                     }
                  }
               }

               if (rmax == -1)  // No vertices set in gridVertex on the interior of this edge.
               {
                  continue;
               }

               if (d == 0)
               {
                  gv1[1] = rmin;
                  gv1[2] = rmax;
               }
               else
               {
                  gv2[1] = rmin;
                  gv2[2] = rmax;
               }

               const int pid = d == 0 ? 2*s : (2*s) + 1; // Parent index

               const bool reverse_p = parentEdgeRev[pid];
               // Sides with s=1 are reversed in defining parentEdgeRev.
               const bool reverse = s == 0 ? reverse_p : !reverse_p;

               // Define an auxiliary edge (gv1[i], gv1[i+1])
               Array<int> cv(2);
               Array<int> ki(2);

               if (d == 0)
               {
                  cv[0] = gridVertex(gv1[i],s*n2);
                  cv[1] = gridVertex(gv1[i+1],s*n2);

                  ki[0] = gv1[i];
                  ki[1] = gv1[i+1];
               }
               else
               {
                  cv[0] = gridVertex((1-s)*n1,gv2[i]);
                  cv[1] = gridVertex((1-s)*n1,gv2[i+1]);

                  ki[0] = gv2[i];
                  ki[1] = gv2[i+1];
               }

               if (cv[0] == cv[1])
               {
                  continue;
               }

               const int tv = i == 0 ? cv[1] :
                              cv[0];  // Top-vertex interior to the master edge.
               const int tvki_f = i == 0 ? ki[1] : ki[0];  // Knot index w.r.t. face
               const int tvki = reverse ? n_d - tvki_f : tvki_f;  // Knot index w.r.t. edge

               cv.Sort();

               MFEM_ASSERT(cv[0] >= 0, "");

               const std::pair<int, int> childPair(cv[0], cv[1]);
               const bool childPairTopo = v2e.count(childPair) > 0;
               if (!childPairTopo)
               {
                  const int pv0 = d == 0 ? gridVertex(0,s*n2) : gridVertex((1-s)*n1,0);
                  const int pv1 = d == 0 ? gridVertex(n1,s*n2) : gridVertex((1-s)*n1,n2);

                  const std::pair<int, int> parentPair(pv0 < pv1 ? pv0 : pv1,
                                                       pv0 < pv1 ? pv1 : pv0);

                  const int parentEdge = v2e.at(parentPair);

                  MFEM_VERIFY(parentEdges[pid] == parentEdge, "");

                  // Check whether childPair is in auxEdges.
                  if (auxv2e.count(childPair) == 0)
                  {
                     const int knotIndex0 = (d == 0) ? gv1[i] : gv2[i];
                     const int knotIndex1 = (d == 0) ? gv1[i+1] : gv2[i+1];

                     // Create a new auxiliary edge
                     auxv2e[childPair] = auxEdges.size();
                     auxEdges.emplace_back(AuxiliaryEdge{pv0 < pv1 ? parentEdge : -1 - parentEdge,
                                                         childPair.first, childPair.second,
                                                         knotIndex0, knotIndex1});
                  }

                  // TODO: ne is identical to n_d.
                  const int ne = d == 0 ? n1 : n2;
                  const int e_idx_i = i == 0 ? 0 : ne - 1;
                  const int e_idx = reverse ? ne - 1 - e_idx_i : e_idx_i;

                  const EdgePairInfo ep_e((e_idx == ne - 1) ? -1 : tv,
                                          (e_idx == ne - 1) ? -1 : tvki,
                                          -1 - auxv2e[childPair], parentEdge);

                  const bool unset = !edgePairs[edgePairOS[parentEdge] + e_idx].isSet;
                  if (unset)
                  {
                     edgePairs[edgePairOS[parentEdge] + e_idx] = ep_e;
                  }
                  else
                  {
                     // Verify matching
                     MFEM_VERIFY(edgePairs[edgePairOS[parentEdge] + e_idx] == ep_e, "");
                  }
               }
               else  // childPairTopo == true, so this edge is a slave edge.
               {
                  const int childEdge = v2e.at(childPair);

                  const int pv0 = d == 0 ? gridVertex(0,s*n2) : gridVertex((1-s)*n1,0);
                  const int pv1 = d == 0 ? gridVertex(n1,s*n2) : gridVertex((1-s)*n1,n2);

                  const std::pair<int, int> parentPair(pv0 < pv1 ? pv0 : pv1,
                                                       pv0 < pv1 ? pv1 : pv0);

                  const int parentEdge = v2e.at(parentPair);

                  MFEM_VERIFY(parentEdges[pid] == parentEdge, "");

                  // TODO: ne is identical to n_d.
                  const int ne = d == 0 ? n1 : n2;

                  const int e_idx_i = i == 0 ? 0 : ne - 1;
                  const int e_idx = reverse ? ne - 1 - e_idx_i : e_idx_i;

                  const int tv_e = (e_idx == ne - 1) ? -1 : tv;
                  const int tv_ki = (e_idx == ne - 1) ? -1 : tvki;  // TODO: better naming

                  const EdgePairInfo ep_e(tv_e, tv_ki, childEdge, parentEdge);

                  const bool unset = !edgePairs[edgePairOS[parentEdge] + e_idx].isSet;
                  const bool matching = edgePairs[edgePairOS[parentEdge] + e_idx] == ep_e;

                  edgePairs[edgePairOS[parentEdge] + e_idx] = ep_e;

                  MFEM_ASSERT(unset || matching, "");
               }
            }
         }
      }

      if (hasSlaveFaces || hasAuxFace)
      {
         masterFaces.insert(parentFace);
      }
   } // loop over parents

   MFEM_VERIFY(consistent, "");
}

int GetFaceOrientation(const Mesh *mesh, const int face,
                       const Array<int> & verts)
{
   Array<int> fverts;
   mesh->GetFaceVertices(face, fverts);

   MFEM_VERIFY(verts.Size() == 4 && fverts.Size() == 4, "");

   // Verify that verts and fvert have the same entries as sets, by deep-copying and sorting.
   {
      Array<int> s1(verts);
      Array<int> s2(fverts);

      s1.Sort(); s2.Sort();
      MFEM_VERIFY(s1 == s2, "");
   }

   // Find the shift of the first vertex.
   int s = -1;
   for (int i=0; i<4; ++i)
   {
      if (verts[i] == fverts[0]) { s = i; }
   }

   // Check whether ordering is reversed.
   const bool rev = verts[(s + 1) % 4] != fverts[1];

   if (rev) { s = -1 - s; }  // Reversed order is encoded by the sign.

   // Sanity check (TODO: remove this)
   for (int i=0; i<4; ++i)
   {
      const int j = s < 0 ? (-1 - s) - i : i + s;
      MFEM_VERIFY(verts[(j + 4) % 4] == fverts[i], "");
   }

   return s;
}

int GetShiftedQuadIndex(int i, int ori)
{
   // Map index i with ori determined by GetFaceOrientation.

   const int s = ori < 0 ? -1 - ori : ori;
   const bool rev = (ori < 0);

   return rev ? (s - i + 4) % 4 : (i + s) % 4;
}

// The 2D array `a` is of size n1*n2, with index
// j + n2*i corresponding to (i,j) with the fast index j,
// for 0 <= i < n1 and 0 <= j < n2.
// We assume that j is the fast index in (i,j).
// The orientation is encoded by ori, defining a shift and relative
// direction, such that a quad face F1, on which the ordering of `a` is based,
// has vertex with index `shift` matching vertex 0 of the new quad face F2,
// on which the new ordering of `a` should be based.
// For more details, see GetFaceOrientation.
bool Reorder2D(int n1, int n2, int ori, std::vector<int> & s0)
{
   const bool noReorder = false;
   if (noReorder)
   {
      s0[0] = 0;
      s0[1] = 0;
      return false;
   }

   const int shift = ori < 0 ? -1 - ori : ori;

   // Shift is an F1 index in the counter-clockwise ordering of 4 quad vertices.
   // Now find the (i,j) indices of this index, with i,j in {0,1}.
   const int s0i = (shift == 0 || shift == 3) ? 0 : 1;
   const int s0j = (shift < 2) ? 0 : 1;

   s0[0] = s0i;
   s0[1] = s0j;

   // Determine whether the dimensions of F1 and F2 are reversed.
   // Do this by finding the (i,j) indices of s1, which is the next vertex on F1.
   const int shift1 = ori < 0 ? shift - 1: shift + 1;

   const int s1 = (shift1 + 4) % 4;

   const int s1i = (s1 == 0 || s1 == 3) ? 0 : 1;
   const bool dimReverse = s0i == s1i;

   return dimReverse;
}

void GetInverseShiftedDimensions2D(int signedShift, int sm, int sn, int &m,
                                   int &n)
{
   const bool rev = (signedShift < 0);
   const int shift = rev ? -1 - signedShift : signedShift;
   MFEM_VERIFY(0 <= shift && shift < 4, "");
   // TODO: condense this somehow?

   // We consider 8 cases for the possible values of rev and shift.
   if (rev)
   {
      if (shift == 0)
      {
         // New: 3 2  Old: 1 2
         //      0 1       0 3
         n = sm;
         m = sn;
      }
      else if (shift == 1)
      {
         // New: 3 2  Old: 2 3
         //      0 1       1 0
         m = sm;
         n = sn;
      }
      else if (shift == 2)
      {
         // New: 3 2  Old: 3 0
         //      0 1       2 1
         n = sm;
         m = sn;
      }
      else // shift == 3
      {
         // New: 3 2  Old: 0 1
         //      0 1       3 2
         m = sm;
         n = sn;
      }
   }
   else
   {
      if (shift == 0)
      {
         // New: 3 2  Old: 3 2
         //      0 1       0 1
         m = sm;
         n = sn;
      }
      else if (shift == 1)
      {
         // New: 3 2  Old: 0 3
         //      0 1       1 2
         n = sm;
         m = sn;
      }
      else if (shift == 2)
      {
         // New: 3 2  Old: 1 0
         //      0 1       2 3
         m = sm;
         n = sn;
      }
      else // shift == 3
      {
         // New: 3 2  Old: 2 1
         //      0 1       3 0
         n = sm;
         m = sn;
      }
   }
}

// TODO: reuse this in other places, to reduce code duplication.
void GetShiftedGridPoints2D(int m, int n, int i, int j, int signedShift,
                            int& sm, int& sn, int& si, int& sj)
{
   const bool rev = (signedShift < 0);
   const int shift = rev ? -1 - signedShift : signedShift;
   MFEM_VERIFY(0 <= shift && shift < 4, "");

   // (0,0) <= (i,j) < (m,n) are old indices, and old vertex [shift] maps
   // to new vertex 0 in counter-clockwise quad ordering.

   // TODO: condense this somehow?

   // We consider 8 cases for the possible values of rev and shift.
   if (rev)
   {
      if (shift == 0)
      {
         // New: 3 2  Old: 1 2
         //      0 1       0 3
         sm = n;
         sn = m;

         si = j;
         sj = i;
      }
      else if (shift == 1)
      {
         // New: 3 2  Old: 2 3
         //      0 1       1 0
         sm = m;
         sn = n;

         si = m - 1 - i;
         sj = j;
      }
      else if (shift == 2)
      {
         // New: 3 2  Old: 3 0
         //      0 1       2 1
         sm = n;
         sn = m;

         si = n - 1 - j;
         sj = m - 1 - i;
      }
      else // shift == 3
      {
         // New: 3 2  Old: 0 1
         //      0 1       3 2
         sm = m;
         sn = n;

         si = i;
         sj = n - 1 - j;
      }
   }
   else
   {
      if (shift == 0)
      {
         // New: 3 2  Old: 3 2
         //      0 1       0 1
         sm = m;
         sn = n;

         si = i;
         sj = j;
      }
      else if (shift == 1)
      {
         // New: 3 2  Old: 0 3
         //      0 1       1 2
         sm = n;
         sn = m;

         si = j;
         sj = m - 1 - i;
      }
      else if (shift == 2)
      {
         // New: 3 2  Old: 1 0
         //      0 1       2 3
         sm = m;
         sn = n;

         si = m - 1 - i;
         sj = n - 1 - j;
      }
      else // shift == 3
      {
         // New: 3 2  Old: 2 1
         //      0 1       3 0
         sm = n;
         sn = m;

         si = n - 1 - j;
         sj = i;
      }
   }
}

}
