#include "pbounds.hpp"

ParBounds::ParBounds(ParFiniteElementSpace *pfes_, ParFiniteElementSpace *pfesH1_)
   : Bounds(pfes_, pfesH1_), pfes(pfes_), pfesH1(pfesH1_), px_min(pfesH1_),
   px_max(pfesH1_) { }

void ParBounds::ComputeBounds(const Vector &x)
{
   GroupCommunicator &gcomm = pfesH1->GroupComm();

   px_min =  std::numeric_limits<double>::infinity();
   px_max = -std::numeric_limits<double>::infinity();

   for (int e = 0; e < ne; e++)
   {
      pfesH1->GetElementDofs(e, eldofs);
      ComputeElementBounds(0, e, x);
   }

   Array<double> minvals(px_min.GetData(), px_min.Size()),
   maxvals(px_max.GetData(), px_max.Size());

   gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
   gcomm.Bcast(minvals);
   gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
   gcomm.Bcast(maxvals);

   for (int e = 0; e < ne; e++)
   {
      pfesH1->GetElementDofs(e, eldofs);
      for (int j = 0; j < nd; j++)
      {
         xi_min(e*nd + j) = px_min(eldofs[DofMapH1[j]]);
         xi_max(e*nd + j) = px_max(eldofs[DofMapH1[j]]);
      }
   }

   for (int n = 1; n < NumEq; n++)
   {
      px_min =  std::numeric_limits<double>::infinity();
      px_max = -std::numeric_limits<double>::infinity();

      for (int e = 0; e < ne; e++)
      {
         pfesH1->GetElementDofs(e, eldofs);
         ComputeSequentialBounds(n, e, x);
      }

      Array<double> minvals2(px_min.GetData(), px_min.Size()),
      maxvals2(px_max.GetData(), px_max.Size());

      gcomm.Reduce<double>(minvals2, GroupCommunicator::Min);
      gcomm.Bcast(minvals2);
      gcomm.Reduce<double>(maxvals2, GroupCommunicator::Max);
      gcomm.Bcast(maxvals2);

      for (int e = 0; e < ne; e++)
      {
         pfesH1->GetElementDofs(e, eldofs);
         for (int j = 0; j < nd; j++)
         {
            xi_min(n*ne*nd + e*nd + j) = px_min(eldofs[DofMapH1[j]]);
            xi_max(n*ne*nd + e*nd + j) = px_max(eldofs[DofMapH1[j]]);
         }
      }
   }
}


ParTightBounds::ParTightBounds(ParFiniteElementSpace *pfes_, ParFiniteElementSpace *pfesH1_)
   : ParBounds(pfes_, pfesH1_)
{
   FillClosestNbrs(pfes->GetFE(0), ClosestNbrs);
}

void ParTightBounds::ComputeElementBounds(int n, int e, const Vector &x)
{
   for (int i = 0; i < nd; i++)
   {
      const int I = eldofs[DofMapH1[i]];
      px_min(I) = min(px_min(I), xi_min(e*nd + i));
      px_max(I) = max(px_max(I), xi_max(e*nd + i));

      // for (int j = 0; j < ClosestNbrs.Width(); j++)
      // {
      //    if (ClosestNbrs(i,j) == -1) { break; }

      //    const int J = n*ne*nd + e*nd + ClosestNbrs(i,j);
      //    px_min(I) = min(px_min(I), x(J));
      //    px_max(I) = max(px_max(I), x(J));
      // }
   }
}

void ParTightBounds::ComputeSequentialBounds(int n, int e, const Vector &x)
{
   for (int i = 0; i < nd; i++)
   {
      const int I = eldofs[DofMapH1[i]];
      // double quotient = x(n*ne*nd + e*nd + i) / x(e*nd + i);
      // px_min(I) = min(quotient, xi_min(n*ne*nd + e*nd + i));
      // px_max(I) = max(quotient, xi_max(n*ne*nd + e*nd + i));

      for (int j = 0; j < ClosestNbrs.Width(); j++)
      {
         if (ClosestNbrs(i,j) == -1) { break; }

         const int J = n*ne*nd + e*nd + ClosestNbrs(i,j);
         px_min(I) = min(px_min(I), xi_min(J));
         px_max(I) = max(px_max(I), xi_max(J));
      }
   }
}

ParLooseBounds::ParLooseBounds(ParFiniteElementSpace *pfes_, ParFiniteElementSpace *pfesH1_)
   : ParBounds(pfes_, pfesH1_) { }

void ParLooseBounds::ComputeElementBounds(int n, int e, const Vector &x)
{
   double xe_min =  std::numeric_limits<double>::infinity();
   double xe_max = -std::numeric_limits<double>::infinity();

   for (int j = 0; j < nd; j++)
   {
      xe_min = min(xe_min, x(n*ne*nd + e*nd+j));
      xe_max = max(xe_max, x(n*ne*nd + e*nd+j));
   }

   for (int j = 0; j < nd; j++)
   {
      int I = eldofs[DofMapH1[j]];
      px_min(I) = min(px_min(I), xe_min);
      px_max(I) = max(px_max(I), xe_max);
   }
}

void ParLooseBounds::ComputeSequentialBounds(int n, int e, const Vector &x)
{
   // TODO
}
