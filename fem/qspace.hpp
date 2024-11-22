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

#ifndef MFEM_QSPACE
#define MFEM_QSPACE

#include "../config/config.hpp"
#include "fespace.hpp"
#include <unordered_map>

namespace mfem
{

/// Abstract base class for QuadratureSpace and FaceQuadratureSpace.
/** This class represents the storage layout for QuadratureFunction%s, that may
    be defined either on mesh elements or mesh faces. */
class QuadratureSpaceBase
{
protected:
   friend class QuadratureFunction; // Uses the offsets.

   Mesh &mesh; ///< The underlying mesh.
   int order; ///< The order of integration rule.
   int size; ///< Total number of quadrature points.
   int ne; ///< Actual number of entities
   mutable Vector weights; ///< Integration weights.
   mutable long nodes_sequence = 0; ///< Nodes counter for cache invalidation.

   /// @brief Entity quadrature point offset array. Supports a constant
   /// compression scheme for meshes which have a single geometry type. When
   /// compressed, will have a single value. The true offset can be computed as
   /// i * offsets[0], where i is the entity index. Otherwise has size
   /// num_entities + 1.
   ///
   Array<int> offsets;
   /// The quadrature rules used for each geometry type.
   const IntegrationRule *int_rule[Geometry::NumGeom];

   /// Protected constructor. Used by derived classes.
   QuadratureSpaceBase(Mesh &mesh_, int order_ = 0)
      : mesh(mesh_), order(order_) { }

   /// Protected constructor. Used by derived classes.
   QuadratureSpaceBase(Mesh &mesh_, Geometry::Type geom,
                       const IntegrationRule &ir);

   /// Fill the @ref int_rule array for each geometry type using @ref order.
   void ConstructIntRules(int dim);

   /// Compute the det(J) (volume or faces, depending on the type).
   virtual const Vector &GetGeometricFactorWeights() const = 0;

   /// Compute the integration weights.
   void ConstructWeights() const;

   /// Gets the offset for a given entity @a idx.
   ///
   /// The quadrature point values for entity i are stored in the indices
   /// between Offset(i) and Offset(i+1)
   int Offset(int idx) const
   {
      return (offsets.Size() == 1) ? (idx * offsets[0]) : offsets[idx];
   }

public:
   /// Return the total number of quadrature points.
   int GetSize() const { return size; }

   /// Return the order of the quadrature rule(s) used by all elements.
   int GetOrder() const { return order; }

   /// Return the number of entities.
   int GetNE() const { return offsets.Size() - 1; }

   /// Returns the mesh.
   inline Mesh *GetMesh() const { return &mesh; }

   /// Get the (element or face) transformation of entity @a idx.
   virtual ElementTransformation *GetTransformation(int idx) = 0;

   /// Return the geometry type of entity (element or face) @a idx.
   virtual Geometry::Type GetGeometry(int idx) const = 0;

   /// Return the IntegrationRule associated with entity @a idx.
   const IntegrationRule &GetIntRule(int idx) const
   { return *int_rule[GetGeometry(idx)]; }

   /// @brief Returns the permuted index of the @a iq quadrature point in entity
   /// @a idx.
   ///
   /// For tensor-product faces, returns the lexicographic index of the
   /// quadrature point, oriented relative to "element 1". For QuadratureSpace%s
   /// defined on elements (not faces), the permutation is trivial, and this
   /// returns @a iq.
   virtual int GetPermutedIndex(int idx, int iq) const = 0;

   /// @brief Returns the index in the quadrature space of the entity associated
   /// with the transformation @a T.
   ///
   /// For a QuadratureSpace defined on elements, this just returns the element
   /// index. For FaceQuadratureSpace, the returned index depends on the chosen
   /// FaceType. If the entity is not found (for example, if @a T represents an
   /// interior face, and the space has FaceType::Boundary) then -1 is returned.
   virtual int GetEntityIndex(const ElementTransformation &T) const = 0;

   /// Write the QuadratureSpace to the stream @a out.
   virtual void Save(std::ostream &out) const = 0;

   /// Return the integration weights (including geometric factors).
   const Vector &GetWeights() const;

   /// Return the integral of the scalar Coefficient @a coeff.
   real_t Integrate(Coefficient &coeff) const;

   /// Return the integral of the VectorCoefficient @a coeff in @a integrals.
   void Integrate(VectorCoefficient &coeff, Vector &integrals) const;

   virtual ~QuadratureSpaceBase() { }
};

/// Class representing the storage layout of a QuadratureFunction.
/** Multiple QuadratureFunction%s can share the same QuadratureSpace. */
class QuadratureSpace : public QuadratureSpaceBase
{
protected:
   const Vector &GetGeometricFactorWeights() const override;
   void ConstructOffsets();
   void Construct();
public:
   /// Create a QuadratureSpace based on the global rules from #IntRules.
   QuadratureSpace(Mesh *mesh_, int order_)
      : QuadratureSpaceBase(*mesh_, order_) { Construct(); }

   /// @brief Create a QuadratureSpace with an IntegrationRule, valid only when
   /// the mesh has one element type.
   QuadratureSpace(Mesh &mesh_, const IntegrationRule &ir);

   /// Read a QuadratureSpace from the stream @a in.
   QuadratureSpace(Mesh *mesh_, std::istream &in);

   /// Returns number of elements in the mesh.
   inline int GetNE() const { return mesh.GetNE(); }

   /// Returns the element transformation of element @a idx.
   ElementTransformation *GetTransformation(int idx) override
   { return mesh.GetElementTransformation(idx); }

   /// Returns the geometry type of element @a idx.
   Geometry::Type GetGeometry(int idx) const override
   { return mesh.GetElementGeometry(idx); }

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetElementIntRule(int idx) const
   { return *int_rule[mesh.GetElementBaseGeometry(idx)]; }

   /// @brief Returns the permuted index of the @a iq quadrature point in entity
   /// @a idx.
   ///
   /// The member function QuadratureSpace::GetPermutedIndex always returns @a
   /// iq, the permutation is only nontrivial for FaceQuadratureSpace.
   int GetPermutedIndex(int idx, int iq) const override { return iq; }

   /// Returns the element index of @a T.
   int GetEntityIndex(const ElementTransformation &T) const override { return T.ElementNo; }

   /// Write the QuadratureSpace to the stream @a out.
   void Save(std::ostream &out) const override;
};

/// Class representing the storage layout of a FaceQuadratureFunction.
/** FaceQuadratureSpace is defined on either the interior or boundary faces
    of a mesh, depending on the provided FaceType. */
class FaceQuadratureSpace : public QuadratureSpaceBase
{
   FaceType face_type; ///< Is the space defined on interior or boundary faces?
   const int num_faces; ///< Number of faces.

   /// Map from boundary or interior face indices to mesh face indices.
   Array<int> face_indices;

   /// Inverse of the map @a face_indices.
   std::unordered_map<int,int> face_indices_inv;

   const Vector &GetGeometricFactorWeights() const override;
   void ConstructOffsets();
   void Construct();

public:
   /// Create a FaceQuadratureSpace based on the global rules from #IntRules.
   FaceQuadratureSpace(Mesh &mesh_, int order_, FaceType face_type_);

   /// @brief Create a FaceQuadratureSpace with an IntegrationRule, valid only
   /// when the mesh has one type of face geometry.
   FaceQuadratureSpace(Mesh &mesh_, const IntegrationRule &ir,
                       FaceType face_type_);

   /// Returns number of faces in the mesh.
   inline int GetNumFaces() const { return num_faces; }

   /// Returns the face type (boundary or interior).
   FaceType GetFaceType() const { return face_type; }

   /// Returns the face transformation of face @a idx.
   ElementTransformation *GetTransformation(int idx) override
   { return mesh.GetFaceTransformation(face_indices[idx]); }

   /// Returns the geometry type of face @a idx.
   Geometry::Type GetGeometry(int idx) const override
   { return mesh.GetFaceGeometry(face_indices[idx]); }

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetFaceIntRule(int idx) const
   { return *int_rule[GetGeometry(idx)]; }

   /// @brief Returns the permuted index of the @a iq quadrature point in entity
   /// @a idx.
   ///
   /// For tensor-product faces, returns the lexicographic index of the
   /// quadrature point, oriented relative to "element 1".
   int GetPermutedIndex(int idx, int iq) const override;

   /// @brief Get the face index (in the standard Mesh numbering) associated
   /// with face @a idx in the FaceQuadratureSpace.
   int GetMeshFaceIndex(int idx) const { return face_indices[idx]; }

   /// @brief Returns the index associated with the face described by @a T.
   ///
   /// The index may differ from the mesh face or boundary element index
   /// depending on the FaceType used to construct the FaceQuadratureSpace.
   int GetEntityIndex(const ElementTransformation &T) const override;

   /// Write the FaceQuadratureSpace to the stream @a out.
   void Save(std::ostream &out) const override;
};

}

#endif
