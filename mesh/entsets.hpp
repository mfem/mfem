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

#ifndef MFEM_ENTITY_SETS
#define MFEM_ENTITY_SETS

#include "../config/config.hpp"
#include "../general/table.hpp"
#include "../general/stable3d.hpp"
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace mfem
{

class Mesh;
class NCMesh;
class NCEntitySets;

class EntitySets
{
   friend class Mesh;
   friend class NCMesh;
   friend class NCEntitySets;

public:
   enum EntityType {INVALID = -1, VERTEX = 0, EDGE = 1, FACE = 2, ELEMENT = 3};

   static std::map<EntityType,std::string> EntityTypeNames;

   EntitySets(Mesh & mesh);
   EntitySets(const EntitySets & ent_sets);
   EntitySets(Mesh & mesh, NCMesh &ncmesh);

   virtual ~EntitySets();

   static const std::string & GetTypeName(EntityType t);

   bool SetExists(EntityType t, unsigned int s) const;
   bool SetExists(EntityType t, const std::string & s) const;

   void Load(std::istream &input);
   void Print(std::ostream &output) const;
   virtual void PrintSetInfo(std::ostream &output) const;

   inline Mesh *GetMesh() const { return mesh_; }

   unsigned int GetNumSets(EntityType t) const;

   const std::string & GetSetName(EntityType t, unsigned int s) const;
   unsigned int        GetNumEntities(EntityType t, unsigned int s) const;

   int          GetSetIndex(EntityType t, const std::string & s) const;
   unsigned int GetNumEntities(EntityType t, const std::string & s) const;

   inline std::set<int> & operator()(EntityType t, unsigned int s)
   { return sets_[t][s]; }
   inline const std::set<int> & operator()(EntityType t, unsigned int s) const
   { return sets_[t][s]; }

   const Table * GetEdgeVertexTable() const { return edge_vertex_; }
   const Table * GetFaceVertexTable() const { return face_vertex_; }
   const Table * GetFaceEdgeTable()   const { return face_edge_; }

   // void Prune(int nelems);

protected:

   void SetNumSets(EntityType t, unsigned int n)
   { sets_[t].resize(n); set_names_[t].resize(n); }
   void SetSetName(EntityType t, int s, const std::string & name)
   { set_names_[t][s] = name; set_index_by_name_[t][name] = s; }

   /// Make local copies of edge_vertex, face_vertex, and face_edge tables.
   void CopyMeshTables();

   /// Refine quadrilateral mesh.
   virtual void QuadUniformRefinement();

   /// Refine hexahedral mesh.
   virtual void HexUniformRefinement();

   /// Refine 2D mesh.
   virtual void UniformRefinement2D();

   /// Refine 3D mesh.
   virtual void UniformRefinement3D();

private:

   static void skip_comment_lines(std::istream &is, const char comment_char)
   {
      while (1)
      {
         is >> std::ws;
         if (is.peek() != comment_char) { break; }
         is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
   }
   // Check for, and remove, a trailing '\r'.
   static void filter_dos(std::string &line)
   {
      if (!line.empty() && *line.rbegin() == '\r')
      { line.resize(line.size()-1); }
   }

   static std::map<EntityType,std::string> init_type_names();

   void LoadEntitySets(std::istream &input, EntityType t,
                       const std::string & header);

   void PrintEntitySets(std::ostream &output, EntityType t,
                        const std::string & header) const;

   void PrintEdgeSets(std::ostream &output) const;

   void PrintFaceSets(std::ostream &output) const;

   void PrintEntitySetInfo(std::ostream & output, EntityType t,
                           const std::string & ent_name) const;

   void CopyEntitySets(const EntitySets & ent_sets, EntityType t);
   void BuildEntitySets(NCMesh &ncmesh, EntityType t);

protected:

   Mesh     * mesh_;
   Table    * edge_vertex_;
   Table    * face_vertex_;
   Table    * face_edge_;

   int NumOfVertices_;
   int NumOfEdges_;
   int NumOfElements_;

   /** The node/edge/face/element indices needed by the finite element
       space to look up DoFs. */
   std::vector<std::vector<std::set<int> > > sets_;

   /// Names of each entity set
   std::vector<std::vector<std::string> >    set_names_;

   /// Indices of each entity set indexed by set name
   std::vector<std::map<std::string, int> >  set_index_by_name_;
};

class NCEntitySets
{
   friend class EntitySets;

public:
   NCEntitySets(const EntitySets & ent_sets, NCMesh &ncmesh);
   NCEntitySets(const NCEntitySets & ncent_sets);

   bool SetExists(EntitySets::EntityType t, unsigned int s) const;
   bool SetExists(EntitySets::EntityType t, const std::string & s) const;

   unsigned int GetNumSets(EntitySets::EntityType t) const;

   static int GetEntitySize(EntitySets::EntityType t);

   const std::string & GetSetName(EntitySets::EntityType t, int s) const;
   unsigned int        GetNumEntities(EntitySets::EntityType t, int s) const;
   void                GetEntityIndex(EntitySets::EntityType t, int s,
                                      int i, Array<int> & inds) const;

   int          GetSetIndex(EntitySets::EntityType t,
                            const std::string & s) const;
   unsigned int GetNumEntities(EntitySets::EntityType t,
                               const std::string & s) const;
   void         GetEntityIndex(EntitySets::EntityType t,
                               const std::string & s, int i,
                               Array<int> & inds) const;

   inline std::vector<int> & operator()(EntitySets::EntityType t, int s)
   { return sets_[t][s]; }
   inline const std::vector<int> & operator()(EntitySets::EntityType t,
                                              int s) const
   { return sets_[t][s]; }
   inline int & operator()(EntitySets::EntityType t, int s, int i)
   { return sets_[t][s][i]; }
   inline int operator()(EntitySets::EntityType t, int s, int i) const
   { return sets_[t][s][i]; }

private:
   void CopyNCEntitySets(const NCEntitySets & ncent_sets,
                         EntitySets::EntityType t);

protected:

   NCMesh * ncmesh_;

   /// The nodes defining the node/edge/face/element sets
   std::vector<std::vector<std::vector<int> > > sets_;

   /// Names of each entity set
   std::vector<std::vector<std::string> >      set_names_;

   /// Indices of each entity set indexed by set name
   std::vector<std::map<std::string, int> >    set_index_by_name_;

   /// Number of indices per entity
   static const int entity_size_[4];
};

} // namespace mfem

#endif // MFEM_ENTITY_SETS
