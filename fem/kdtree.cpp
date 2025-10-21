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

#include "kdtree.hpp"

namespace mfem
{

template<>
void KDTreeNodalProjection<2>::Project(const Vector& coords,const Vector& src,
                                       int ordering, real_t lerr)
{
   const int dim=dest->FESpace()->GetMesh()->SpaceDimension();
   const int vd=dest->VectorDim(); // dimension of the vector field
   const int np=src.Size()/vd; // number of points
   int ind;
   real_t dist;
   bool pt_inside_bbox;
   KDTree2D::PointND pnd;
   for (int i=0; i<np; i++)
   {
      pnd.xx[0]=coords(i*dim+0);
      pnd.xx[1]=coords(i*dim+1);

      pt_inside_bbox=true;
      for (int j=0; j<dim; j++)
      {
         if (pnd.xx[j]>(maxbb[j]+lerr)) {pt_inside_bbox=false; break;}
         if (pnd.xx[j]<(minbb[j]-lerr)) {pt_inside_bbox=false; break;}
      }

      if (pt_inside_bbox)
      {
         kdt->FindClosestPoint(pnd,ind,dist);
         if (dist<lerr)
         {
            if (dest->FESpace()->GetOrdering()==Ordering::byNODES)
            {
               if (ordering==Ordering::byNODES)
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di*np+ind]=src[di*np+i];
                  }
               }
               else
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di*np+ind]=src[di+i*vd];
                  }
               }
            }
            else
            {
               if (ordering==Ordering::byNODES)
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di+ind*vd]=src[di*np+i];
                  }
               }
               else
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di+ind*vd]=src[di+i*vd];
                  }
               }
            }
         }
      }
   }
}

template<>
void KDTreeNodalProjection<3>::Project(const Vector& coords,const Vector& src,
                                       int ordering, real_t lerr)
{
   const int dim=dest->FESpace()->GetMesh()->SpaceDimension();
   const int vd=dest->VectorDim(); // dimension of the vector field
   const int np=src.Size()/vd; // number of points
   int ind;
   real_t dist;
   bool pt_inside_bbox;
   KDTree3D::PointND pnd;
   for (int i=0; i<np; i++)
   {
      pnd.xx[0]=coords(i*dim+0);
      pnd.xx[1]=coords(i*dim+1);
      pnd.xx[2]=coords(i*dim+2);

      pt_inside_bbox=true;
      for (int j=0; j<dim; j++)
      {
         if (pnd.xx[j]>(maxbb[j]+lerr)) {pt_inside_bbox=false; break;}
         if (pnd.xx[j]<(minbb[j]-lerr)) {pt_inside_bbox=false; break;}
      }

      if (pt_inside_bbox)
      {
         kdt->FindClosestPoint(pnd,ind,dist);
         if (dist<lerr)
         {
            if (dest->FESpace()->GetOrdering()==Ordering::byNODES)
            {
               if (ordering==Ordering::byNODES)
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di*np+ind]=src[di*np+i];
                  }
               }
               else
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di*np+ind]=src[di+i*vd];
                  }
               }
            }
            else
            {
               if (ordering==Ordering::byNODES)
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di+ind*vd]=src[di*np+i];
                  }
               }
               else
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di+ind*vd]=src[di+i*vd];
                  }
               }
            }
         }
      }
   }
}

template<>
void KDTreeNodalProjection<2>::Project(const GridFunction& gf, real_t lerr)
{
   int ordering = gf.FESpace()->GetOrdering();
   Vector coo;
   int np=gf.FESpace()->GetVSize()/gf.FESpace()->GetVDim();
   coo.SetSize(np*2);
   int vd=dest->VectorDim();
   int ind;
   real_t dist;

   Vector maxbb_src(2);
   Vector minbb_src(2);

   // extract the nodal coordinates from gf
   {
      ElementTransformation *trans;
      const IntegrationRule* ir=nullptr;
      Array<int> vdofs;
      DenseMatrix elco;
      int isca=1;
      if (gf.FESpace()->GetOrdering()==Ordering::byVDIM)
      {
         isca=gf.FESpace()->GetVDim();
      }

      // initialize bbmax and bbmin
      const FiniteElement* el=gf.FESpace()->GetFE(0);
      trans = gf.FESpace()->GetElementTransformation(0);
      ir=&(el->GetNodes());
      gf.FESpace()->GetElementVDofs(0,vdofs);
      elco.SetSize(2,ir->GetNPoints());
      trans->Transform(*ir,elco);
      for (int d=0; d<2; d++)
      {
         maxbb_src(d)=elco(d,0);
         minbb_src(d)=elco(d,0);
      }

      for (int i=0; i<gf.FESpace()->GetNE(); i++)
      {
         el=gf.FESpace()->GetFE(i);
         //get the element transformation
         trans = gf.FESpace()->GetElementTransformation(i);
         ir=&(el->GetNodes());
         gf.FESpace()->GetElementVDofs(i,vdofs);
         elco.SetSize(2,ir->GetNPoints());
         trans->Transform(*ir,elco);
         for (int p=0; p<ir->GetNPoints(); p++)
         {
            for (int d=0; d<2; d++)
            {
               coo[vdofs[p]*2/isca+d]=elco(d,p);

               if (maxbb_src(d)<elco(d,p)) {maxbb_src(d)=elco(d,p);}
               if (minbb_src(d)>elco(d,p)) {minbb_src(d)=elco(d,p);}
            }
         }
      }
   }

   maxbb_src+=lerr;
   minbb_src-=lerr;

   // check for intersection
   bool flag;
   {
      flag=true;
      for (int i=0; i<2; i++)
      {
         if (minbb_src(i)>maxbb(i)) {flag=false;}
         if (maxbb_src(i)<minbb(i)) {flag=false;}
      }
      if (flag==false) {return;}
   }

   {
      KDTree2D::PointND pnd;
      for (int i=0; i<np; i++)
      {
         pnd.xx[0]=coo(i*2+0);
         pnd.xx[1]=coo(i*2+1);

         kdt->FindClosestPoint(pnd,ind,dist);
         if (dist<lerr)
         {
            if (dest->FESpace()->GetOrdering()==Ordering::byNODES)
            {
               if (ordering==Ordering::byNODES)
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di*np+ind]=gf[di*np+i];
                  }
               }
               else
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di*np+ind]=gf[di+i*vd];
                  }
               }
            }
            else
            {
               if (ordering==Ordering::byNODES)
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di+ind*vd]=gf[di*np+i];
                  }
               }
               else
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di+ind*vd]=gf[di+i*vd];
                  }
               }
            }
         }
      }
   }
}

template<>
void KDTreeNodalProjection<3>::Project(const GridFunction& gf, real_t lerr)
{
   int ordering = gf.FESpace()->GetOrdering();
   int dim=dest->FESpace()->GetMesh()->SpaceDimension();
   Vector coo;
   int np=gf.FESpace()->GetVSize()/gf.FESpace()->GetVDim();
   coo.SetSize(np*dim);
   int vd=dest->VectorDim();
   int ind;
   real_t dist;

   Vector maxbb_src(dim);
   Vector minbb_src(dim);

   // extract the nodal coordinates from gf
   {
      ElementTransformation *trans;
      const IntegrationRule* ir=nullptr;
      Array<int> vdofs;
      DenseMatrix elco;
      int isca=1;
      if (gf.FESpace()->GetOrdering()==Ordering::byVDIM)
      {
         isca=gf.FESpace()->GetVDim();
      }

      // initialize bbmax and bbmin
      const FiniteElement* el=gf.FESpace()->GetFE(0);
      trans = gf.FESpace()->GetElementTransformation(0);
      ir=&(el->GetNodes());
      gf.FESpace()->GetElementVDofs(0,vdofs);
      elco.SetSize(dim,ir->GetNPoints());
      trans->Transform(*ir,elco);
      for (int d=0; d<dim; d++)
      {
         maxbb_src(d)=elco(d,0);
         minbb_src(d)=elco(d,0);
      }

      for (int i=0; i<gf.FESpace()->GetNE(); i++)
      {
         el=gf.FESpace()->GetFE(i);
         // get the element transformation
         trans = gf.FESpace()->GetElementTransformation(i);
         ir=&(el->GetNodes());
         gf.FESpace()->GetElementVDofs(i,vdofs);
         elco.SetSize(dim,ir->GetNPoints());
         trans->Transform(*ir,elco);
         for (int p=0; p<ir->GetNPoints(); p++)
         {
            for (int d=0; d<dim; d++)
            {
               coo[vdofs[p]*dim/isca+d]=elco(d,p);

               if (maxbb_src(d)<elco(d,p)) {maxbb_src(d)=elco(d,p);}
               if (minbb_src(d)>elco(d,p)) {minbb_src(d)=elco(d,p);}
            }
         }
      }
   }

   maxbb_src+=lerr;
   minbb_src-=lerr;

   // check for intersection
   bool flag;
   {
      flag=true;
      for (int i=0; i<dim; i++)
      {
         if (minbb_src(i)>maxbb(i)) {flag=false;}
         if (maxbb_src(i)<minbb(i)) {flag=false;}
      }
      if (flag==false) {return;}
   }

   {
      KDTree3D::PointND pnd;
      for (int i=0; i<np; i++)
      {
         pnd.xx[0]=coo(i*dim+0);
         pnd.xx[1]=coo(i*dim+1);
         pnd.xx[2]=coo(i*dim+2);

         kdt->FindClosestPoint(pnd,ind,dist);
         if (dist<lerr)
         {
            if (dest->FESpace()->GetOrdering()==Ordering::byNODES)
            {
               if (ordering==Ordering::byNODES)
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di*np+ind]=gf[di*np+i];
                  }
               }
               else
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di*np+ind]=gf[di+i*vd];
                  }
               }
            }
            else
            {
               if (ordering==Ordering::byNODES)
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di+ind*vd]=gf[di*np+i];
                  }
               }
               else
               {
                  for (int di=0; di<vd; di++)
                  {
                     (*dest)[di+ind*vd]=gf[di+i*vd];
                  }
               }
            }
         }
      }
   }
}

} // namespace mfem
