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

#ifndef MFEM_PAR_ENTITY_SETS
#define MFEM_PAR_ENTITY_SETS

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "entsets.hpp"
#include "../general/communication.hpp"

namespace mfem
{

class ParMesh;
class ParNCMesh;

class ParEntitySets : public EntitySets
{
   friend class ParMesh;

public:
   ParEntitySets(const ParEntitySets & ent_sets);
   ParEntitySets & operator=(const ParEntitySets & ent_sets) = delete;
   ParEntitySets(ParMesh & _mesh, const EntitySets & ent_sets, int * part,
                 const Array<int> & vert_global_local);
   ParEntitySets(ParMesh & mesh, ParNCMesh &ncmesh);

   virtual ~ParEntitySets();

   virtual void PrintSetInfo(std::ostream &output) const;

   inline ParMesh *GetParMesh() const { return pmesh_; }

private:

   void PrintEntitySetInfo(std::ostream & output, EntityType t,
                           const std::string & ent_name) const;

   void BuildEntitySets(ParNCMesh &pncmesh, EntityType t);

   ParMesh  * pmesh_;
   int        NRanks_;
   int        MyRank_;
};

class ParNCEntitySets : public NCEntitySets
{
public:
   ParNCEntitySets(MPI_Comm comm, const NCMesh &ncmesh);
   ParNCEntitySets & operator=(const ParNCEntitySets & pncent_sets) = delete;

private:
   MPI_Comm MyComm_;
   int      NRanks_;
   int      MyRank_;
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PAR_ENTITY_SETS
