// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// This file contains operator-based bilinear form integrators used
// with BilinearFormOperator.

#ifndef MFEM_TENSORFUNC_CPP
#define MFEM_TENSORFUNC_CPP

#include "tensor.hpp"
#include "../../fem/fespace.hpp"
#include "tensorialfunctions.hpp"
#include "../../fem/gridfunc.hpp"
// #include "fem.hpp"
// #include "dgpabilininteg.hpp"

namespace mfem
{

namespace pa
{

void ScatterDofs(const mfem::FiniteElementSpace* mfes, const Table& eldof,
                 const mfem::Array<int>& dof_map,
                 const GridFunction* nodes, const int dofs, const int dim, const int e,
                 Tensor<2>& LexPointMat)
{
   if (dof_map.Size()==0)
   {
      if (mfes->GetOrdering()==Ordering::byVDIM)
      {
         for (int i = 0; i < dofs; ++i)
         {
            for (int j = 0; j < dim; ++j)
            {
               LexPointMat(j,i) = (*nodes)( ( e*dofs + i )*dim + j );
            }
         }
      }
      else
      {
         for (int i = 0; i < dofs; ++i)
         {
            for (int j = 0; j < dim; ++j)
            {
               LexPointMat(j,i) = (*nodes)( ( e*dofs + i ) + j*mfes->GetNDofs() );
            }
         }
      }
   }
   else
   {
      if (mfes->GetOrdering()==Ordering::byVDIM)
      {
         for (int i = 0; i < dofs; ++i)
         {
            const int pivot = dof_map[i];
            for (int j = 0; j < dim; ++j)
            {
               LexPointMat(j,i) = (*nodes)(eldof.GetJ()[ e*dofs + pivot ]*dim + j);
            }
         }
      }
      else
      {
         for (int i = 0; i < dofs; ++i)
         {
            const int pivot = dof_map[i];
            for (int j = 0; j < dim; ++j)
            {
               LexPointMat(j,i) = (*nodes)(eldof.GetJ()[ e*dofs + pivot ] +
                                           j*mfes->GetNDofs());
            }
         }
      }
   }
}

static void EvalJacobians1D(const mfem::FiniteElementSpace* fes,
                            const int order, Tensor<1>& J)
{
   const int dim = 1;

   const Mesh* mesh = fes->GetMesh();
   const mfem::FiniteElementSpace* mfes = mesh->GetNodalFESpace();
   const Table& eldof = mfes->GetElementToDofTable();
   const FiniteElement* fe = mfes->GetFE(0);
   const TensorBasisElement* tfe = dynamic_cast<const TensorBasisElement*>(fe);
   const mfem::Array<int>& dof_map = tfe->GetDofMap();
   const GridFunction* nodes = mesh->GetNodes();
   Tensor<2> shape1d(GetNDofs1d(mfes),GetNQuads1d(order)),
          dshape1d(GetNDofs1d(mfes),GetNQuads1d(order));
   ComputeBasis1d( fe, order, shape1d, dshape1d );

   const int NE = fes->GetNE();

   const int quads1d = shape1d.Width();
   const int dofs1d = shape1d.Height();
   const int dofs = dofs1d;

   Tensor<2> Jac(J.getData(),quads1d,NE);
   Jac.zero();
   Tensor<2> LexPointMat(dim,dofs);

   Tensor<1> T0(LexPointMat.getData(),dofs1d);
   for (int e = 0; e < NE; ++e)
   {
      ScatterDofs(mfes, eldof, dof_map, nodes, dofs, dim, e, LexPointMat);
      // Computing the Jacobian with the tensor product structure
      for (int j1 = 0; j1 < quads1d; ++j1)
      {
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            for (int d = 0; d < dim; ++d)
            {
               Jac(j1,e) += T0(i1) * dshape1d(i1,j1);
            }
         }
      }
   }
}

static void EvalJacobians2D(const mfem::FiniteElementSpace* fes,
                            const int order, Tensor<1>& J)
{
   const int dim = 2;

   const Mesh* mesh = fes->GetMesh();
   const mfem::FiniteElementSpace* mfes = mesh->GetNodalFESpace();
   const Table& eldof = mfes->GetElementToDofTable();
   const FiniteElement* fe = mfes->GetFE(0);
   const TensorBasisElement* tfe = dynamic_cast<const TensorBasisElement*>(fe);
   const mfem::Array<int>& dof_map = tfe->GetDofMap();
   const GridFunction* nodes = mesh->GetNodes();
   Tensor<2> shape1d(GetNDofs1d(mfes),GetNQuads1d(order)),
          dshape1d(GetNDofs1d(mfes),GetNQuads1d(order));
   ComputeBasis1d( fe, order, shape1d, dshape1d );

   const int NE = mfes->GetNE();

   const int quads1d = shape1d.Width();
   const int dofs1d  = shape1d.Height();
   const int dofs = dofs1d * dofs1d;


   Tensor<5> Jac(J.getData(),dim,dim,quads1d,quads1d,NE);
   Jac.zero();
   Tensor<2> LexPointMat(dim,dofs);

   Tensor<3> T0(LexPointMat.getData(),dim,dofs1d,dofs1d);
   Tensor<2> T1b(dim,quads1d), T1d(dim,quads1d);
   for (int e = 0; e < NE; ++e)
   {
      ScatterDofs(mfes, eldof, dof_map, nodes, dofs, dim, e, LexPointMat);
      // Computing the Jacobian with the tensor product structure
      for (int i2 = 0; i2 < dofs1d; ++i2)
      {
         T1b.zero();
         T1d.zero();
         for (int j1 = 0; j1 < quads1d; ++j1)
         {
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               for (int d = 0; d < dim; ++d)
               {
                  T1b(d,j1) += T0(d,i1,i2) *  shape1d(i1,j1);
                  T1d(d,j1) += T0(d,i1,i2) * dshape1d(i1,j1);
               }
            }
         }
         for (int j2 = 0; j2 < quads1d; ++j2)
         {
            for (int j1 = 0; j1 < quads1d; ++j1)
            {
               for (int d = 0; d < dim; ++d)
               {
                  Jac(d,0,j1,j2,e) += T1d(d,j1) *  shape1d(i2,j2);
                  Jac(d,1,j1,j2,e) += T1b(d,j1) * dshape1d(i2,j2);
               }
            }
         }
      }
   }
}

static void EvalJacobians3D(const mfem::FiniteElementSpace* fes,
                            const int order, Tensor<1>& J)
{
   const int dim = 3;

   const Mesh* mesh = fes->GetMesh();
   const mfem::FiniteElementSpace* mfes = mesh->GetNodalFESpace();
   const Table& eldof = mfes->GetElementToDofTable();
   const FiniteElement* fe = mfes->GetFE(0);
   const TensorBasisElement* tfe = dynamic_cast<const TensorBasisElement*>(fe);
   const mfem::Array<int>& dof_map = tfe->GetDofMap();
   const GridFunction* nodes = mesh->GetNodes();
   Tensor<2> shape1d(GetNDofs1d(mfes),GetNQuads1d(order)),
          dshape1d(GetNDofs1d(mfes),GetNQuads1d(order));
   ComputeBasis1d( fe, order, shape1d, dshape1d );

   const int NE = fes->GetNE();

   const int quads1d = shape1d.Width();
   const int dofs1d = shape1d.Height();
   const int dofs = dofs1d * dofs1d * dofs1d;

   Tensor<6> Jac(J.getData(),dim,dim,quads1d,quads1d,quads1d,NE);
   Jac.zero();
   Tensor<2> LexPointMat(dim,dofs);

   Tensor<4> T0(LexPointMat.getData(),dim,dofs1d,dofs1d,dofs1d);
   Tensor<2> T1b(dim,quads1d), T1d(dim,quads1d);
   Tensor<3> T2bb(dim,quads1d,quads1d), T2db(dim,quads1d,quads1d), T2bd(dim,
                                                                        quads1d,quads1d);
   for (int e = 0; e < NE; ++e)
   {
      // Modifies T0 in the same time...
      ScatterDofs(mfes, eldof, dof_map, nodes, dofs, dim, e, LexPointMat);
      // Computing the Jacobian with the tensor product structure
      for (int i3 = 0; i3 < dofs1d; ++i3)
      {
         T2bb.zero();
         T2db.zero();
         T2bd.zero();
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T1b.zero();
            T1d.zero();
            for (int j1 = 0; j1 < quads1d; ++j1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  for (int d = 0; d < dim; ++d)
                  {
                     T1b(d,j1) += T0(d,i1,i2,i3) *  shape1d(i1,j1);
                     T1d(d,j1) += T0(d,i1,i2,i3) * dshape1d(i1,j1);
                  }
               }
            }
            for (int j2 = 0; j2 < quads1d; ++j2)
            {
               for (int j1 = 0; j1 < quads1d; ++j1)
               {
                  for (int d = 0; d < dim; ++d)
                  {
                     T2bb(d,j1,j2) += T1b(d,j1) *  shape1d(i2,j2);
                     T2bd(d,j1,j2) += T1b(d,j1) * dshape1d(i2,j2);
                     T2db(d,j1,j2) += T1d(d,j1) *  shape1d(i2,j2);
                  }
               }
            }
         }
         for (int j3 = 0; j3 < quads1d; ++j3)
         {
            for (int j2 = 0; j2 < quads1d; ++j2)
            {
               for (int j1 = 0; j1 < quads1d; ++j1)
               {
                  for (int d = 0; d < dim; ++d)
                  {
                     Jac(d,0,j1,j2,j3,e) += T2db(d,j1,j2) *  shape1d(i3,j3);
                     Jac(d,1,j1,j2,j3,e) += T2bd(d,j1,j2) *  shape1d(i3,j3);
                     Jac(d,2,j1,j2,j3,e) += T2bb(d,j1,j2) * dshape1d(i3,j3);
                  }
               }
            }
         }
      }
   }
}

void EvalJacobians( const int dim, const mfem::FiniteElementSpace* fes,
                    const int order,
                    Tensor<1>& J )
{
   switch (dim)
   {
      case 1:
         EvalJacobians1D( fes, order, J );
         break;
      case 2:
         EvalJacobians2D( fes, order, J );
         break;
      case 3:
         EvalJacobians3D( fes, order, J );
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
   }
}

}

}

#endif