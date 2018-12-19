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

#ifndef MFEM4_MESH_BATCHGEOM_HPP
#define MFEM4_MESH_BATCHGEOM_HPP

#include "mfem4/general/tensor.hpp"

namespace mfem4
{

using namespace mfem;


/** Stores element transformation data for a batch of elements, evaluated at
 *  integration points.
 */
class BatchGeometry
{
public:
   BatchGeometry(const Mesh &mesh, const Array<int> &batch);

   const Tensor<4>& GetNodes(const IntegrationRule &ir) const;
   const Tensor<4>& GetJacobians(const IntegrationRule &ir) const;
   const Tensor<4>& GetInvJacobians(const IntegrationRule &ir) const;
   const Tensor<2>& GetDetJacobians(const IntegrationRule &ir) const;

   const Array<int> GetElements() const { return batch; }

   int GetNE() const { batch.Size(); }

   //const Mesh &GetMesh() const {} // ?

protected:
   Array<int> batch;

   /*Tensor<4> nodes;
   Tensor<4> J, invJ, detJ;*/
};







#endif // MFEM4_MESH_BATCHGEOM_HPP
