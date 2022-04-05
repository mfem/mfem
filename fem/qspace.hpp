// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_QSPACE
#define MFEM_QSPACE

#include "../config/config.hpp"
#include "fespace.hpp"

namespace mfem
{

class QuadratureSpaceBase
{
protected:
   friend class QuadratureFunctionBase; // Uses the offsets.

   Mesh &mesh;
   int order;
   int size;
   Array<int> offsets;

   const IntegrationRule *int_rule[Geometry::NumGeom];

   QuadratureSpaceBase(Mesh &mesh_, int order_ = 0)
      : mesh(mesh_), order(order_) { }

public:
   /// Return the total number of quadrature points.
   int GetSize() const { return size; }

   /// Return the order of the quadrature rule(s) used by all elements.
   int GetOrder() const { return order; }

   /// Returns the mesh
   inline Mesh *GetMesh() const { return &mesh; }

   virtual ~QuadratureSpaceBase() { }
};

/// Class representing the storage layout of a QuadratureFunction.
/** Multiple QuadratureFunction%s can share the same QuadratureSpace. */
class QuadratureSpace : public QuadratureSpaceBase
{
protected:
   // Assuming mesh and order are set, construct the members: int_rule,
   // element_offsets, and size.
   void Construct();
public:
   /// Create a QuadratureSpace based on the global rules from #IntRules.
   QuadratureSpace(Mesh *mesh_, int order_)
      : QuadratureSpaceBase(*mesh_, order_) { Construct(); }

   /// Read a QuadratureSpace from the stream @a in.
   QuadratureSpace(Mesh *mesh_, std::istream &in);

   /// Returns number of elements in the mesh.
   inline int GetNE() const { return mesh.GetNE(); }

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetElementIntRule(int idx) const
   { return *int_rule[mesh.GetElementBaseGeometry(idx)]; }

   /// Write the QuadratureSpace to the stream @a out.
   void Save(std::ostream &out) const;
};

/// Class representing the storage layout of a FaceQuadratureFunction.
class FaceQuadratureSpace : public QuadratureSpaceBase
{
   // Assuming mesh and order are set, construct the members: int_rule,
   // element_offsets, and size.
   void Construct();

public:
   /// Create a FaceQuadratureSpace based on the global rules from #IntRules.
   FaceQuadratureSpace(Mesh &mesh_, int order_)
      : QuadratureSpaceBase(mesh_, order_) { Construct(); }

   /// Returns number of faces in the mesh.
   inline int GetNumFaces() const { return mesh.GetNumFaces(); }

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetFaceIntRule(int idx) const
   { return *int_rule[mesh.GetFaceGeometry(idx)]; }
};

}

#endif
