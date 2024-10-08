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

#ifndef MFEM_KDTREE_PROJECTION
#define MFEM_KDTREE_PROJECTION

#include "../general/kdtree.hpp"
#include "gridfunc.hpp"

namespace mfem
{

/// Base class for KDTreeNodalProjection.
class BaseKDTreeNodalProjection
{
public:
   virtual ~BaseKDTreeNodalProjection()
   {}

   /// The projection method can be called as many time as necessary with
   /// different sets of coordinates and corresponding values. For vector
   /// grid function, users have to specify the data ordering and for all
   /// cases the user can modify the error tolerance err to smaller or
   /// bigger value. A node in the target grid function is matching
   /// a point with coordinates specified in the vector coords if the
   /// distance between them is smaller than lerr.
   virtual
   void Project(const Vector& coords,const Vector& src,
                int ordering, real_t lerr) = 0;

   /// The project method can be called as many times as necessary with
   /// different grid functions gf. A node in the target grid function is
   /// matching a node from the source grid function if the distance
   /// between them is smaller than lerr.
   virtual
   void Project(const GridFunction& gf, real_t lerr) = 0;
};

/// The class provides methods for projecting function values evaluated on a
/// set of points to a grid function. The values are directly copied to the
/// nodal values of the target grid function if any of the points is matching
/// a node of the grid function. For example, if a parallel grid function is
/// saved in parallel, every saved chunk can be read on every other process
/// and mapped to a local grid function that does not have the same structure
/// as the original one. The functionality is based on a kd-tree search in a
/// cloud of points.
template<int kdim=3>
class KDTreeNodalProjection : public BaseKDTreeNodalProjection
{
private:
   /// Pointer to the KDTree
   std::unique_ptr<KDTree<int,real_t,kdim>> kdt;

   /// Pointer to the target grid function
   GridFunction* dest;

   /// Upper corner of the bounding box
   Vector maxbb;

   /// Lower corner of the bounding box
   Vector minbb;

public:
   /// The constructor takes as input an L2 or H1 grid function (it can be
   /// a vector grid function). The Project method copies a set of values
   /// to the grid function.
   KDTreeNodalProjection(GridFunction& dest_)
   {
      dest=&dest_;
      FiniteElementSpace* space=dest->FESpace();

      MFEM_VERIFY(
         dynamic_cast<const H1_FECollection*>(space->FEColl()) != nullptr ||
         dynamic_cast<const L2_FECollection*>(space->FEColl()) != nullptr,
         "Error!");

      Mesh* mesh=space->GetMesh();

      const int dim=mesh->SpaceDimension();
      MFEM_VERIFY(kdim==dim, "GridFunction dimension does not match!");

      kdt=std::unique_ptr<KDTree<int,real_t,kdim>>(
             new KDTree<int,real_t,kdim>());

      std::vector<bool> indt;
      indt.resize(space->GetVSize()/space->GetVDim(), true);

      minbb.SetSize(dim);
      maxbb.SetSize(dim);

      //set the loocal coordinates
      {
         ElementTransformation *trans;
         const IntegrationRule* ir=nullptr;
         Array<int> vdofs;
         DenseMatrix elco;
         int isca=1;
         if (space->GetOrdering()==Ordering::byVDIM)
         {
            isca=space->GetVDim();
         }

         // intialize the bounding box
         const FiniteElement* el=space->GetFE(0);
         trans = space->GetElementTransformation(0);
         ir=&(el->GetNodes());
         space->GetElementVDofs(0,vdofs);
         elco.SetSize(dim,ir->GetNPoints());
         trans->Transform(*ir,elco);
         for (int d=0; d<dim; d++)
         {
            minbb[d]=elco(d,0);
            maxbb[d]=elco(d,0);
         }

         for (int i=0; i<space->GetNE(); i++)
         {
            el=space->GetFE(i);
            // get the element transformation
            trans = space->GetElementTransformation(i);
            ir=&(el->GetNodes());
            space->GetElementVDofs(i,vdofs);
            elco.SetSize(dim,ir->GetNPoints());
            trans->Transform(*ir,elco);

            for (int p=0; p<ir->GetNPoints(); p++)
            {
               int bind=vdofs[p]/isca;
               if (indt[bind]==true)
               {
                  kdt->AddPoint(elco.GetColumn(p),bind);
                  indt[bind]=false;

                  for (int d=0; d<kdim; d++)
                  {
                     if (minbb[d]>elco(d,p)) {minbb[d]=elco(d,p);}
                     if (maxbb[d]<elco(d,p)) {maxbb[d]=elco(d,p);}
                  }
               }
            }
         }
      }

      // build the KDTree
      kdt->Sort();
   }

   /// The projection method can be called as many time as necessary with
   /// different sets of coordinates and corresponding values. For vector
   /// grid function, users have to specify the data ordering and for all
   /// cases the user can modify the error tolerance err to smaller or
   /// bigger value. A node in the target grid function is matching
   /// a point with coordinates specified in the vector coords if the
   /// distance between them is smaller than lerr.
   void Project(const Vector& coords,const Vector& src,
                int ordering=Ordering::byNODES, real_t lerr=1e-8) override;

   /// The project method can be called as many times as necessary with
   /// different grid functions gf. A node in the target grid function is
   /// matching a node from the source grid function if the distance
   /// between them is smaller than lerr.
   void Project(const GridFunction& gf, real_t lerr=1e-8) override;
};

} // namespace mfem

#endif // MFEM_KDTREE_PROJECTION
