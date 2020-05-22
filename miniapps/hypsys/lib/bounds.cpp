#include "bounds.hpp"

Bounds::Bounds(FiniteElementSpace *fes_, FiniteElementSpace *fesH1_)
      : fes(fes_), fesH1(fesH1_), x_min(fesH1_), x_max(fesH1_)
{
   Mesh *mesh = fes->GetMesh();
   const FiniteElement *el = fes->GetFE(0);

   nd = el->GetDof();
   ne = mesh->GetNE();
   NumEq = fes->GetVSize() / (ne*nd);

   xi_min.SetSize(NumEq*ne*nd);
   xi_max.SetSize(NumEq*ne*nd);

   FillDofMap();
}

void Bounds::FillDofMap()
{
   const FiniteElement *el = fes->GetFE(0);

   if (el->GetGeomType() != Geometry::TRIANGLE)
   {
      const TensorBasisElement *TensorElem =
      dynamic_cast<const TensorBasisElement *>(fesH1->GetFE(0));
      DofMapH1 = TensorElem->GetDofMap();
      return;
   }

   const int p = el->GetOrder();
   const int nd = (p+1)*(p+2) / 2;
   DofMapH1.SetSize(nd);

   // Corners
   DofMapH1[0] = 0;
   DofMapH1[p] = 1;
   DofMapH1[nd-1] = 2;

   int ctr1 = 2*p;
   int ctr2 = nd-3;

   // Element edges
   for (int i = 1; i < p; i++)
   {
      DofMapH1[i] = 2+i;
      DofMapH1[ctr1] = 2+i+p-1;
      DofMapH1[ctr2] = 2+i+2*(p-1);
      ctr1 += (p-i);
      ctr2 -= 2+i;
   }

   ctr1 = p+2;
   ctr2 = 3*p;

   // Element interior
   for (int j = 1; j < p-1; j++)
   {
      for (int i = 1; i < p-j; i++)
      {
         DofMapH1[ctr1] = ctr2;
         ctr1++;
         ctr2++;
      }
      ctr1 += 2;
   }
}

void Bounds::ComputeBounds(const Vector &x)
{
   x_min =  std::numeric_limits<double>::infinity();
   x_max = -std::numeric_limits<double>::infinity();

   for (int e = 0; e < ne; e++)
   {
      fesH1->GetElementDofs(e, eldofs);
      ComputeElementBounds(0, e, x);
   }

   for (int e = 0; e < ne; e++)
   {
      fesH1->GetElementDofs(e, eldofs);
      for (int j = 0; j < nd; j++)
      {
         xi_min(e*nd + j) = x_min(eldofs[DofMapH1[j]]);
         xi_max(e*nd + j) = x_max(eldofs[DofMapH1[j]]);
         // xi_min(e*nd + j) = min(xi_min(e*nd + j), x_min(eldofs[DofMapH1[j]]));
         // xi_max(e*nd + j) = max(xi_max(e*nd + j), x_max(eldofs[DofMapH1[j]]));
      }
   }

   for (int n = 1; n < NumEq; n++)
   {
      x_min =  std::numeric_limits<double>::infinity();
      x_max = -std::numeric_limits<double>::infinity();

      for (int e = 0; e < ne; e++)
      {
         fesH1->GetElementDofs(e, eldofs);
         ComputeSequentialBounds(n, e, x);
      }

      for (int e = 0; e < ne; e++)
      {
         fesH1->GetElementDofs(e, eldofs);
         for (int j = 0; j < nd; j++)
         {
            xi_min(n*ne*nd + e*nd + j) = x_min(eldofs[DofMapH1[j]]);
            xi_max(n*ne*nd + e*nd + j) = x_max(eldofs[DofMapH1[j]]);
         }
      }
   }
}


TightBounds::TightBounds(FiniteElementSpace *fes_, FiniteElementSpace *fesH1_)
   : Bounds(fes_, fesH1_)
{
   FillClosestNbrs(fes->GetFE(0), ClosestNbrs);
}

void TightBounds::ComputeElementBounds(int n, int e, const Vector &x)
{
   for (int i = 0; i < nd; i++)
   {
      const int I = eldofs[DofMapH1[i]];
      x_min(I) = min(x_min(I), xi_min(e*nd + i));
      x_max(I) = max(x_max(I), xi_max(e*nd + i));

      // for (int j = 0; j < ClosestNbrs.Width(); j++)
      // {
      //    if (ClosestNbrs(i,j) == -1) { break; }

      //    const int J = n*ne*nd + e*nd + ClosestNbrs(i,j);
      //    x_min(I) = min(x_min(I), x(J));
      //    x_max(I) = max(x_max(I), x(J));
      // }
   }
}

void TightBounds::ComputeSequentialBounds(int n, int e, const Vector &x)
{
   for (int i = 0; i < nd; i++)
   {
      const int I = eldofs[DofMapH1[i]];
      // // double quotient = x(n*ne*nd + e*nd + i) / x(e*nd + i);
      // x_min(I) = min( x_min(I), /* min(quotient, */ xi_min(n*ne*nd + e*nd + i)) /* ) */;
      // x_max(I) = max( x_max(I), /* max(quotient, */ xi_max(n*ne*nd + e*nd + i)) /* ) */;

      for (int j = 0; j < ClosestNbrs.Width(); j++)
      {
         if (ClosestNbrs(i,j) == -1) { break; }

         const int J = n*ne*nd + e*nd + ClosestNbrs(i,j);
         x_min(I) = min(x_min(I), xi_min(J));
         x_max(I) = max(x_max(I), xi_max(J));
      }
   }
}

LooseBounds::LooseBounds(FiniteElementSpace *fes_, FiniteElementSpace *fesH1_)
   : Bounds(fes_, fesH1_) { }

void LooseBounds::ComputeElementBounds(int n, int e, const Vector &x)
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
      x_min(I) = min(x_min(I), xe_min);
      x_max(I) = max(x_max(I), xe_max);
   }
}

void LooseBounds::ComputeSequentialBounds(int n, int e, const Vector &x)
{
   // TODO
}


void FillClosestNbrs(const FiniteElement *el, DenseMatrix &ClosestNbrs)
{
   const int nd = el->GetDof();
   const int p = el->GetOrder();
   Geometry::Type gtype = el->GetGeomType();

   switch (gtype)
   {
      case Geometry::SEGMENT:
      {
         ClosestNbrs.SetSize(nd, p==1 ? 2 : 3);
         ClosestNbrs = -1;

         ClosestNbrs(0,0) = 0;
         ClosestNbrs(0,1) = 1;

         for (int i = 1; i < p; i++)
         {
            ClosestNbrs(i,0) = i-1;
            ClosestNbrs(i,1) = i;
            ClosestNbrs(i,2) = i+1;
         }

         ClosestNbrs(p,0) = p-1;
         ClosestNbrs(p,1) = p;
         break;
      }
      case Geometry::TRIANGLE:
      {
         ClosestNbrs.SetSize(nd, p==1 ? 3 : (p==2 ? 5 : 7));
         ClosestNbrs = -1;

         ClosestNbrs(0,0) = 0;
         ClosestNbrs(0,1) = 1;
         ClosestNbrs(0,2) = p+1;

         for (int i = 1; i < p; i++)
         {
            ClosestNbrs(i,0) = i-1;
            ClosestNbrs(i,1) = i;
            ClosestNbrs(i,2) = i+1;
            ClosestNbrs(i,3) = p+i;
            ClosestNbrs(i,4) = p+i+1;
         }

         ClosestNbrs(p,0) = p-1;
         ClosestNbrs(p,1) = p;
         ClosestNbrs(p,2) = 2*p;

         int ctr = p+1;
         for (int j = 1; j < p; j++)
         {
            int lower = (j-1)*(p+2) - (j-1)*j/2;
            int upper = lower + 2*(p-j) + 3;

            ClosestNbrs(ctr,0) = lower;
            ClosestNbrs(ctr,1) = lower+1;
            ClosestNbrs(ctr,2) = ctr;
            ClosestNbrs(ctr,3) = ctr+1;
            ClosestNbrs(ctr,4) = upper;
            ctr++;

            for (int i = 1; i < p-j; i++)
            {
               ClosestNbrs(ctr,0) = lower+i;
               ClosestNbrs(ctr,1) = lower+i+1;
               ClosestNbrs(ctr,2) = ctr-1;
               ClosestNbrs(ctr,3) = ctr;
               ClosestNbrs(ctr,4) = ctr+1;
               ClosestNbrs(ctr,5) = upper+i-1;
               ClosestNbrs(ctr,6) = upper+i;
               ctr++;
            }

            ClosestNbrs(ctr,0) = lower + p-j;
            ClosestNbrs(ctr,1) = lower + p-j+1;
            ClosestNbrs(ctr,2) = ctr-1;
            ClosestNbrs(ctr,3) = ctr;
            ClosestNbrs(ctr,4) = upper+p-j-1;
            ctr++;
         }

         ClosestNbrs(nd-1,0) = nd-3;
         ClosestNbrs(nd-1,1) = nd-2;
         ClosestNbrs(nd-1,2) = nd-1;

         break;
      }
      case Geometry::SQUARE:
      {
         ClosestNbrs.SetSize(nd, p == 1 ? 4: 9);
         ClosestNbrs = -1;

         for (int i = 0; i < nd; i++)
         {
            int ctr = 0;

            // lower neighbors
            if (i > p)
            {
               if (i % (p+1) != 0) { ClosestNbrs(i,ctr) = i-p-2; ctr++; }
               ClosestNbrs(i,ctr) = i-p-1; ctr++;
               if ((i+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = i-p; ctr++; }
            }

            // horizontal neighbors
            if (i % (p+1) != 0) { ClosestNbrs(i,ctr) = i-1; ctr++; }
            ClosestNbrs(i,ctr) = i; ctr++;
            if ((i+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = i+1; ctr++; }

            // upper neighbors
            if (i < p*(p+1))
            {
               if (i % (p+1) != 0) { ClosestNbrs(i,ctr) = i+p; ctr++; }
               ClosestNbrs(i,ctr) = i+p+1; ctr++;
               if ((i+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = i+p+2; ctr++; }
            }
         }

         break;
      }
      case Geometry::CUBE:
      {
         ClosestNbrs.SetSize(nd, p==1 ? 8 : 27);
         ClosestNbrs = -1;

         for (int i = 0; i < nd; i++)
         {
            int ctr = 0;

            if (i >= (p+1)*(p+1)) // There is a lower plane
            {
               int k = i - (p+1)*(p+1); // lower neighbor in z direction
               int j = k % ((p+1)*(p+1));

               // lower neighbors in y direction
               if (j > p)
               {
                  if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = k-p-2; ctr++; }
                  ClosestNbrs(i,ctr) = k-p-1; ctr++;
                  if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = k-p; ctr++; }
               }

               // horizontal neighbors
               if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = k-1; ctr++; }
               ClosestNbrs(i,ctr) = k; ctr++;
               if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = k+1; ctr++; }

               // upper neighbors
               if (j < p*(p+1))
               {
                  if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = k+p; ctr++; }
                  ClosestNbrs(i,ctr) = k+p+1; ctr++;
                  if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = k+p+2; ctr++; }
               }
            }

            int j = i % ((p+1)*(p+1));

            // lower neighbors
            if (j > p)
            {
               if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = i-p-2; ctr++; }
               ClosestNbrs(i,ctr) = i-p-1; ctr++;
               if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = i-p; ctr++; }
            }

            // horizontal neighbors
            if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = i-1; ctr++; }
            ClosestNbrs(i,ctr) = i; ctr++;
            if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = i+1; ctr++; }

            // upper neighbors
            if (j < p*(p+1))
            {
               if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = i+p; ctr++; }
               ClosestNbrs(i,ctr) = i+p+1; ctr++;
               if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = i+p+2; ctr++; }
            }

            if (i < p*(p+1)*(p+1)) // There is an upper plane
            {
               int k = i + (p+1)*(p+1); // upper neighbor in z direction
               int j = k % ((p+1)*(p+1));

               // lower neighbors in y direction
               if (j > p)
               {
                  if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = k-p-2; ctr++; }
                  ClosestNbrs(i,ctr) = k-p-1; ctr++;
                  if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = k-p; ctr++; }
               }

               // horizontal neighbors
               if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = k-1; ctr++; }
               ClosestNbrs(i,ctr) = k; ctr++;
               if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = k+1; ctr++; }

               // upper neighbors
               if (j < p*(p+1))
               {
                  if (j % (p+1) != 0) { ClosestNbrs(i,ctr) = k+p; ctr++; }
                  ClosestNbrs(i,ctr) = k+p+1; ctr++;
                  if ((j+1) % (p+1) != 0) { ClosestNbrs(i,ctr) = k+p+2; ctr++; }
               }
            }

         }
      }
   }
}
