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

/** @brief Integrate a scalar L2 field over the elements in an entity set.

    The result is evaluated directly on every call to Eval(). In parallel, the
    locally owned element contributions are summed and the global result is
    returned on every rank. The entity set and GridFunction are not owned and
    must remain valid for the lifetime of this object.

    The default quadrature order is @f$2p+3@f$, where @f$p@f$ is the order of
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
