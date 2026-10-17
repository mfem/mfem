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

#include "entityset_integral.hpp"

#ifdef MFEM_USE_MPI
#include "../mesh/pmesh.hpp"
#endif

namespace mfem
{

EntitySetFunctional::EntitySetFunctional(const GridFunction &field,
                                         const EntitySets &sets,
                                         EntitySets::EntityType type,
                                         int set_index)
   : field_(field), sets_(sets), type_(type), set_index_(set_index)
{
   MFEM_VERIFY(set_index >= 0 && sets_.SetExists(type_, set_index),
               "Entity set index out of range.");
   VerifyCompatibleField();
}

void EntitySetFunctional::VerifyCompatibleField() const
{
   const FiniteElementSpace *fes = field_.FESpace();
   MFEM_VERIFY(fes != NULL, "The GridFunction has no finite element space.");
   MFEM_VERIFY(field_.GetSequence() == fes->GetSequence(),
               "The GridFunction must be updated with its finite element space.");
   MFEM_VERIFY(sets_.GetMesh() == fes->GetMesh(),
               "The entity set and GridFunction must use the same mesh.");
}

real_t EntitySetFunctional::GlobalSum(real_t local_value) const
{
#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh *>
                          (Field().FESpace()->GetMesh());
   if (pmesh)
   {
      real_t global_value;
      MPI_Allreduce(&local_value, &global_value, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, pmesh->GetComm());
      return global_value;
   }
#endif
   return local_value;
}

L2ElementSetIntegral::L2ElementSetIntegral(const GridFunction &field,
                                           const EntitySets &sets,
                                           int set_index,
                                           int integration_order)
   : EntitySetFunctional(field, sets, EntitySets::ELEMENT, set_index),
     integration_order_(integration_order)
{
   MFEM_VERIFY(integration_order_ >= -1, "Invalid integration order.");

   const FiniteElementSpace *fes = Field().FESpace();
   MFEM_VERIFY(fes->IsDGSpace(),
               "L2ElementSetIntegral requires a discontinuous L2 finite "
               "element space.");
   MFEM_VERIFY(fes->GetVDim() == 1,
               "L2ElementSetIntegral currently requires a scalar field.");
   MFEM_VERIFY(fes->GetTypicalFE()->GetRangeType() == FiniteElement::SCALAR,
               "L2ElementSetIntegral requires scalar finite elements.");
}

L2ElementSetIntegral::L2ElementSetIntegral(const GridFunction &field,
                                           const EntitySets &sets,
                                           const std::string &set_name,
                                           int integration_order)
   : L2ElementSetIntegral(field, sets,
                          sets.GetSetIndex(EntitySets::ELEMENT, set_name),
                          integration_order)
{
}

real_t L2ElementSetIntegral::Eval() const
{
   VerifyCompatibleField();

   const FiniteElementSpace *fes = Field().FESpace();
   real_t local_value = 0.0;

   for (const int element : Entities())
   {
      MFEM_VERIFY(element >= 0 && element < fes->GetNE(),
                  "Entity set contains an invalid element index.");

      const FiniteElement *fe = fes->GetFE(element);
      const int order = (integration_order_ >= 0) ? integration_order_
                        : 2 * fe->GetOrder() + 3;
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), order);
      ElementTransformation *trans = fes->GetElementTransformation(element);

      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         trans->SetIntPoint(&ip);
         local_value += ip.weight * trans->Weight() * Field().GetValue(element, ip);
      }
   }

   return GlobalSum(local_value);
}

} // namespace mfem
