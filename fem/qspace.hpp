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

#ifndef MFEM_QSPACE
#define MFEM_QSPACE

#include "../config/config.hpp"
#include "fespace.hpp"
#include "pointer_utils.hpp"
#include <unordered_map>
#include <memory>
#include <array>

namespace mfem
{

/// Abstract base class for QuadratureSpace and FaceQuadratureSpace.
/** This class represents the storage layout for QuadratureFunction%s, that may
    be defined either on mesh elements or mesh faces. */
class QuadratureSpaceBase : public std::enable_shared_from_this<QuadratureSpaceBase>
{
protected:
   friend class QuadratureFunction; // Uses the offsets.

   std::shared_ptr<Mesh> mesh; ///< The underlying mesh.
   int order; ///< The order of integration rule.
   int size; ///< Total number of quadrature points.
   mutable Vector weights; ///< Integration weights.
   mutable long nodes_sequence = 0; ///< Nodes counter for cache invalidation.

   /// @brief Entity quadrature point offset array, of size num_entities + 1.
   ///
   /// The quadrature point values for entity i are stored in the indices between
   /// offsets[i] and offsets[i+1].
   Array<int> offsets;
   
   /// The quadrature rules used for each geometry type.
   std::array<const IntegrationRule*, Geometry::NumGeom> int_rule = {};

   /// Protected constructor. Used by derived classes.
   QuadratureSpaceBase(std::shared_ptr<Mesh> mesh_, int order_ = 0)
      : mesh(std::move(mesh_)), order(order_) { }

   /// Protected constructor with raw pointer (deprecated)
   [[deprecated("Use constructor with std::shared_ptr<Mesh> instead")]]
   QuadratureSpaceBase(Mesh* mesh_, int order_ = 0)
      : mesh(ptr_utils::borrow_ptr(mesh_)), order(order_) { }

   /// Protected constructor. Used by derived classes.
   QuadratureSpaceBase(std::shared_ptr<Mesh> mesh_, Geometry::Type geom,
                       const IntegrationRule &ir);

   /// Protected constructor with raw pointer (deprecated)
   [[deprecated("Use constructor with std::shared_ptr<Mesh> instead")]]
   QuadratureSpaceBase(Mesh* mesh_, Geometry::Type geom, const IntegrationRule &ir);

   /// Fill the @ref int_rule array for each geometry type using @ref order.
   void ConstructIntRules(int dim);

   /// Compute the det(J) (volume or faces, depending on the type).
   virtual const Vector &GetGeometricFactorWeights() const = 0;

   /// Compute the integration weights.
   void ConstructWeights() const;

public:
   /// Return the total number of quadrature points.
   [[nodiscard]] int GetSize() const { return size; }

   /// Return the order of the quadrature rule(s) used by all elements.
   [[nodiscard]] int GetOrder() const { return order; }

   /// Return the number of entities.
   [[nodiscard]] int GetNE() const { return offsets.Size() - 1; }

   /// Returns the mesh as shared_ptr (modern).
   [[nodiscard]] std::shared_ptr<Mesh> GetMeshShared() const { return mesh; }

   /// Returns the mesh as raw pointer (deprecated).
   [[deprecated("Use GetMeshShared() instead")]]
   [[nodiscard]] Mesh* GetMesh() const { return mesh.get(); }

   /// Get the (element or face) transformation of entity @a idx.
   [[nodiscard]] virtual ElementTransformation *GetTransformation(int idx) = 0;

   /// Return the geometry type of entity (element or face) @a idx.
   [[nodiscard]] virtual Geometry::Type GetGeometry(int idx) const = 0;

   /// Return the IntegrationRule associated with entity @a idx.
   [[nodiscard]] const IntegrationRule &GetIntRule(int idx) const
   { return *int_rule[GetGeometry(idx)]; }

   /// @brief Returns the permuted index of the @a iq quadrature point in entity
   /// @a idx.
   ///
   /// For tensor-product faces, returns the lexicographic index of the
   /// quadrature point, oriented relative to "element 1". For QuadratureSpace%s
   /// defined on elements (not faces), the permutation is trivial, and this
   /// returns @a iq.
   [[nodiscard]] virtual int GetPermutedIndex(int idx, int iq) const = 0;

   /// @brief Returns the index in the quadrature space of the entity associated
   /// with the transformation @a T.
   ///
   /// For a QuadratureSpace defined on elements, this just returns the element
   /// index. For FaceQuadratureSpace, the returned index depends on the chosen
   /// FaceType. If the entity is not found (for example, if @a T represents an
   /// interior face, and the space has FaceType::Boundary) then -1 is returned.
   [[nodiscard]] virtual int GetEntityIndex(const ElementTransformation &T) const = 0;

   /// Write the QuadratureSpace to the stream @a out.
   virtual void Save(std::ostream &out) const = 0;

   /// Return the integration weights (including geometric factors).
   [[nodiscard]] const Vector &GetWeights() const;

   /// Return the integral of the scalar Coefficient @a coeff.
   [[nodiscard]] real_t Integrate(Coefficient &coeff) const;

   /// Return the integral of the VectorCoefficient @a coeff in @a integrals.
   void Integrate(VectorCoefficient &coeff, Vector &integrals) const;

   // Factory methods for creating smart pointers
   
   /// Create a shared_ptr from raw pointer without taking ownership (deprecated)
   [[deprecated("Use make_shared or shared_from_this instead")]]
   static std::shared_ptr<QuadratureSpaceBase> CreateShared(QuadratureSpaceBase* qspace) {
      return ptr_utils::borrow_ptr(qspace);
   }

   virtual ~QuadratureSpaceBase() = default;
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
   QuadratureSpace(std::shared_ptr<Mesh> mesh_, int order_)
      : QuadratureSpaceBase(std::move(mesh_), order_) { Construct(); }

   /// Create a QuadratureSpace based on the global rules from #IntRules (deprecated raw pointer version).
   [[deprecated("Use constructor with std::shared_ptr<Mesh> instead")]]
   QuadratureSpace(Mesh* mesh_, int order_)
      : QuadratureSpaceBase(ptr_utils::borrow_ptr(mesh_), order_) { Construct(); }

   /// @brief Create a QuadratureSpace with an IntegrationRule, valid only when
   /// the mesh has one element type.
   QuadratureSpace(std::shared_ptr<Mesh> mesh_, const IntegrationRule &ir);

   /// @brief Create a QuadratureSpace with an IntegrationRule (deprecated raw pointer version).
   [[deprecated("Use constructor with std::shared_ptr<Mesh> instead")]]
   QuadratureSpace(Mesh* mesh_, const IntegrationRule &ir)
      : QuadratureSpace(ptr_utils::borrow_ptr(mesh_), ir) { }

   /// Read a QuadratureSpace from the stream @a in.
   QuadratureSpace(std::shared_ptr<Mesh> mesh_, std::istream &in);

   /// Read a QuadratureSpace from the stream @a in (deprecated raw pointer version).
   [[deprecated("Use constructor with std::shared_ptr<Mesh> instead")]]
   QuadratureSpace(Mesh* mesh_, std::istream &in)
      : QuadratureSpace(ptr_utils::borrow_ptr(mesh_), in) { }

   /// Returns number of elements in the mesh.
   [[nodiscard]] inline int GetNE() const { return mesh->GetNE(); }

   /// Returns the element transformation of element @a idx.
   [[nodiscard]] ElementTransformation *GetTransformation(int idx) override
   { return mesh->GetElementTransformation(idx); }

   /// Returns the geometry type of element @a idx.
   [[nodiscard]] Geometry::Type GetGeometry(int idx) const override
   { return mesh->GetElementGeometry(idx); }

   /// Get the IntegrationRule associated with mesh element @a idx.
   [[nodiscard]] const IntegrationRule &GetElementIntRule(int idx) const
   { return *int_rule[mesh->GetElementBaseGeometry(idx)]; }

   /// @brief Returns the permuted index of the @a iq quadrature point in entity
   /// @a idx.
   ///
   /// The member function QuadratureSpace::GetPermutedIndex always returns @a
   /// iq, the permutation is only nontrivial for FaceQuadratureSpace.
   [[nodiscard]] int GetPermutedIndex(int idx, int iq) const override { return iq; }

   /// Returns the element index of @a T.
   [[nodiscard]] int GetEntityIndex(const ElementTransformation &T) const override { return T.ElementNo; }

   /// Write the QuadratureSpace to the stream @a out.
   void Save(std::ostream &out) const override;

   // Factory methods for creating QuadratureSpace instances

   /// Create a shared_ptr QuadratureSpace from a Mesh shared_ptr
   static std::shared_ptr<QuadratureSpace> Create(std::shared_ptr<Mesh> mesh, int order) {
      return std::make_shared<QuadratureSpace>(std::move(mesh), order);
   }

   /// Create a shared_ptr QuadratureSpace from a raw Mesh pointer (deprecated)
   [[deprecated("Use Create() with std::shared_ptr<Mesh> instead")]]
   static std::shared_ptr<QuadratureSpace> Create(Mesh* mesh, int order) {
      return std::make_shared<QuadratureSpace>(ptr_utils::borrow_ptr(mesh), order);
   }
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
   FaceQuadratureSpace(std::shared_ptr<Mesh> mesh_, int order_, FaceType face_type_);

   /// Create a FaceQuadratureSpace based on the global rules from #IntRules (deprecated raw pointer version).
   [[deprecated("Use constructor with std::shared_ptr<Mesh> instead")]]
   FaceQuadratureSpace(Mesh* mesh_, int order_, FaceType face_type_)
      : FaceQuadratureSpace(ptr_utils::borrow_ptr(mesh_), order_, face_type_) { }

   /// @brief Create a FaceQuadratureSpace with an IntegrationRule, valid only
   /// when the mesh has one type of face geometry.
   FaceQuadratureSpace(std::shared_ptr<Mesh> mesh_, const IntegrationRule &ir,
                       FaceType face_type_);

   /// @brief Create a FaceQuadratureSpace with an IntegrationRule (deprecated raw pointer version).
   [[deprecated("Use constructor with std::shared_ptr<Mesh> instead")]]
   FaceQuadratureSpace(Mesh* mesh_, const IntegrationRule &ir, FaceType face_type_)
      : FaceQuadratureSpace(ptr_utils::borrow_ptr(mesh_), ir, face_type_) { }

   /// Returns number of faces in the mesh.
   [[nodiscard]] inline int GetNumFaces() const { return num_faces; }

   /// Returns the face type (boundary or interior).
   [[nodiscard]] FaceType GetFaceType() const { return face_type; }

   /// Returns the face transformation of face @a idx.
   [[nodiscard]] ElementTransformation *GetTransformation(int idx) override
   { return mesh->GetFaceTransformation(face_indices[idx]); }

   /// Returns the geometry type of face @a idx.
   [[nodiscard]] Geometry::Type GetGeometry(int idx) const override
   { return mesh->GetFaceGeometry(face_indices[idx]); }

   /// Get the IntegrationRule associated with mesh element @a idx.
   [[nodiscard]] const IntegrationRule &GetFaceIntRule(int idx) const
   { return *int_rule[GetGeometry(idx)]; }

   /// @brief Returns the permuted index of the @a iq quadrature point in entity
   /// @a idx.
   ///
   /// For tensor-product faces, returns the lexicographic index of the
   /// quadrature point, oriented relative to "element 1".
   [[nodiscard]] int GetPermutedIndex(int idx, int iq) const override;

   /// @brief Get the face index (in the standard Mesh numbering) associated
   /// with face @a idx in the FaceQuadratureSpace.
   [[nodiscard]] int GetMeshFaceIndex(int idx) const { return face_indices[idx]; }

   /// @brief Returns the index associated with the face described by @a T.
   ///
   /// The index may differ from the mesh face or boundary element index
   /// depending on the FaceType used to construct the FaceQuadratureSpace.
   [[nodiscard]] int GetEntityIndex(const ElementTransformation &T) const override;

   /// Write the FaceQuadratureSpace to the stream @a out.
   void Save(std::ostream &out) const override;

   // Factory methods for creating FaceQuadratureSpace instances

   /// Create a shared_ptr FaceQuadratureSpace from a Mesh shared_ptr
   static std::shared_ptr<FaceQuadratureSpace> Create(std::shared_ptr<Mesh> mesh, 
                                                     int order, 
                                                     FaceType face_type) {
      return std::make_shared<FaceQuadratureSpace>(std::move(mesh), order, face_type);
   }

   /// Create a shared_ptr FaceQuadratureSpace from a raw Mesh pointer (deprecated)
   [[deprecated("Use Create() with std::shared_ptr<Mesh> instead")]]
   static std::shared_ptr<FaceQuadratureSpace> Create(Mesh* mesh, int order, FaceType face_type) {
      return std::make_shared<FaceQuadratureSpace>(ptr_utils::borrow_ptr(mesh), order, face_type);
   }
};

}

#endif