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
   // ParNCEntitySets(MPI_Comm comm, EntitySets &ent_sets, NCMesh &ncmesh);
   ParNCEntitySets(MPI_Comm comm, const NCMesh &ncmesh);
   // ParNCEntitySets(const ParMesh & pmesh, const ParNCMesh &pncmesh);
   // ParNCEntitySets(const ParNCEntitySets & pncent_sets);

private:
   MPI_Comm MyComm_;
   int      NRanks_;
   int      MyRank_;
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PAR_ENTITY_SETS
