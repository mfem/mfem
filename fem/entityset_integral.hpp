// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ENTITYSET_INTEGRAL
#define MFEM_ENTITYSET_INTEGRAL

#include "gridfunc.hpp"
#include "../mesh/entsets.hpp"
#include "../mesh/orientededgeset.hpp"

namespace mfem
{

/** @brief Common state for quantities evaluated on an EntitySets entry.

    The field and entity set are not owned. Derived classes define the
    entity-specific evaluation, while this class verifies that the set belongs
    to the field's mesh and provides an ownership-correct global sum for
    parallel meshes. */
class EntitySetFunctional
{
protected:
   EntitySetFunctional(const GridFunction &field, const EntitySets &sets,
                       EntitySets::EntityType type, int set_index);

   const GridFunction &Field() const { return field_; }
   const EntitySets &Sets() const { return sets_; }
   int SetIndex() const { return set_index_; }
   const std::set<int> &Entities() const { return sets_(type_, set_index_); }

   /// Sum locally evaluated contributions over all ranks, when applicable.
   real_t GlobalSum(real_t local_value) const;

   /// Verify that the field is current and is defined on the entity-set mesh.
   void VerifyCompatibleField() const;

private:
   const GridFunction &field_;
   const EntitySets &sets_;
   EntitySets::EntityType type_;
   int set_index_;
};
/** @brief Evaluate a scalar H1 GridFunction at the vertices in an entity set.

    Eval() evaluates the field directly on every call. The selected entities
    are topological mesh vertices, not all high-order nodal points or degrees
    of freedom. The entity set and GridFunction are not owned and must remain
    valid for the lifetime of this object.

    In serial, vertex_ids are local mesh vertex indices. In parallel, they are
    global vertex IDs and the results contain each selected physical vertex
    once on every rank, sorted by vertex_ids. This makes the result independent
    of which local element is used to evaluate a shared vertex. Parallel calls
    to Eval() are collective over the mesh communicator. */
class H1VertexSetMeasurement : public EntitySetFunctional
{
public:
   H1VertexSetMeasurement(const GridFunction &field, const EntitySets &sets,
                          int set_index);

   H1VertexSetMeasurement(const GridFunction &field, const EntitySets &sets,
                          const std::string &set_name);

   /** @brief Evaluate the field and return the corresponding vertex IDs.

       The output arrays have matching sizes. See the class description for
       the serial and parallel meanings of @a vertex_ids. */
   void Eval(Array<long long> &vertex_ids, Vector &values) const;

   /** @brief Return the values from Eval(), ordered by their vertex IDs. */
   Vector Eval() const;

private:
   void VerifyCompatibleField() const;
};

/** @brief Integrate an H(curl) GridFunction along an OrientedEdgeSet.

    Eval() directly evaluates the field using a local element incident to each
    edge and integrates $\mathbf{u}\cdot d\mathbf{x}$. In parallel, each
    rank supplies the oriented edges it owns and Eval() sums the local
    contributions over the mesh communicator. The path and GridFunction are
    not owned and must remain valid for the lifetime of this object.

    The default quadrature order is $2p+3$, where $p$ is the order of
    the incident element finite element. An explicit non-negative order
    overrides that default. */
class HCurlOrientedEdgeSetIntegral
{
public:
   HCurlOrientedEdgeSetIntegral(const GridFunction &field,
                                const OrientedEdgeSet &edges,
                                int integration_order = -1);

   /// Return the oriented line integral over the path.
   real_t Eval() const;

   int GetIntegrationOrder() const { return integration_order_; }

private:
   void VerifyCompatibleField() const;
   real_t GlobalSum(real_t local_value) const;

   const GridFunction &field_;
   const OrientedEdgeSet &edges_;
   int integration_order_;
};

/** @brief Integrate the normal flux of an H(div) GridFunction over an
    OrientedFaceSet.

    Eval() directly evaluates the field in a local element adjacent to every
    face and integrates $\mathbf{u}\cdot\mathbf{n}$, including each face's
    stored normal sign. In parallel, each rank supplies its owned faces and
    Eval() sums the local contributions over the mesh communicator. The face
    set and GridFunction are not owned and must remain valid for the lifetime
    of this object.

    The default quadrature order is $2p+3$, where $p$ is the order of
    the adjacent element finite element. An explicit non-negative order
    overrides that default. */
class HDivOrientedFaceSetIntegral
{
public:
   HDivOrientedFaceSetIntegral(const GridFunction &field,
                               const OrientedFaceSet &faces,
                               int integration_order = -1);

   /// Return the oriented normal-flux integral over the face set.
   real_t Eval() const;

   int GetIntegrationOrder() const { return integration_order_; }

private:
   void VerifyCompatibleField() const;
   real_t GlobalSum(real_t local_value) const;

   const GridFunction &field_;
   const OrientedFaceSet &faces_;
   int integration_order_;
};

/** @brief Integrate a scalar L2 field over the elements in an entity set.

    The result is evaluated directly on every call to Eval(). In parallel, the
    locally owned element contributions are summed and the global result is
    returned on every rank. The entity set and GridFunction are not owned and
    must remain valid for the lifetime of this object.

    The default quadrature order is $2p+3$, where $p$ is the order of
    each element's finite element. An explicit non-negative order overrides
    that default. */
class L2ElementSetIntegral : public EntitySetFunctional
{
public:
   L2ElementSetIntegral(const GridFunction &field, const EntitySets &sets,
                        int set_index, int integration_order = -1);

   L2ElementSetIntegral(const GridFunction &field, const EntitySets &sets,
                        const std::string &set_name,
                        int integration_order = -1);

   /// Return the integral of the field over the selected elements.
   real_t Eval() const;

   /// Return -1 when the default, element-dependent quadrature order is used.
   int GetIntegrationOrder() const { return integration_order_; }

private:
   int integration_order_;
};

} // namespace mfem

#endif // MFEM_ENTITYSET_INTEGRAL
