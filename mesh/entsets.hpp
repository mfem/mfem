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
#include <string>
#include <vector>

namespace mfem
{

class Mesh;
class NCMesh;

class EntitySets
{
   friend class Mesh;
   friend class NCMesh;

public:
   enum EntityType {INVALID = -1, VERTEX = 0, EDGE = 1, FACE = 2, ELEMENT = 3};

   EntitySets(Mesh & mesh);
   EntitySets(const EntitySets & ent_sets);

   virtual ~EntitySets();

   void Load(std::istream &input);
   void Print(std::ostream &output) const;
   virtual void PrintSetInfo(std::ostream &output) const;

   inline Mesh *GetMesh() const { return mesh_; }

   unsigned int GetNumSets(EntityType t) const;

   const std::string & GetSetName(EntityType t, int s) const;
   unsigned int        GetNumEntities(EntityType t, int s) const;
   int                 GetEntityIndex(EntityType t, int s, int i) const;

   int          GetSetIndex(EntityType t, const std::string & s) const;
   unsigned int GetNumEntities(EntityType t, const std::string & s) const;
   int          GetEntityIndex(EntityType t,
                               const std::string & s, int i) const;

   inline std::vector<int> & operator()(EntityType t, int s)
   { return sets_[t][s]; }
   inline const std::vector<int> & operator()(EntityType t, int s) const
   { return sets_[t][s]; }
   inline int & operator()(EntityType t, int s, int i)
   { return sets_[t][s][i]; }
   inline int operator()(EntityType t, int s, int i) const
   { return sets_[t][s][i]; }

   inline std::vector<int> & coarse(EntityType t, int s)
   { return set_coarse_[t][s]; }
   inline const std::vector<int> & coarse(EntityType t, int s) const
   { return set_coarse_[t][s]; }
   inline int & coarse(EntityType t, int s, int i)
   { return set_coarse_[t][s][i]; }
   inline int coarse(EntityType t, int s, int i) const
   { return set_coarse_[t][s][i]; }

   const Table * GetEdgeVertexTable() const { return edge_vertex_; }
   const Table * GetFaceVertexTable() const { return face_vertex_; }
   const Table * GetFaceEdgeTable()   const { return face_edge_; }

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

   void LoadEntitySets(std::istream &input, EntityType t,
                       const std::string & header);

   void PrintEntitySets(std::ostream &output, EntityType t,
                        const std::string & header) const;

   void PrintEdgeSets(std::ostream &output) const;

   void PrintFaceSets(std::ostream &output) const;

   void PrintEntitySetInfo(std::ostream & output, EntityType t,
                           const std::string & ent_name) const;

   void CopyEntitySets(const EntitySets & ent_sets, EntityType t);

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
   std::vector<std::vector<std::vector<int> > > sets_;

   /// Names of each entity set
   std::vector<std::vector<std::string> >       set_names_;

   /// Indices of each entity set indexed by set name
   std::vector<std::map<std::string, int> >     set_index_by_name_;

   /// The node IDs and element IDs from the coarse mesh that define each set
   std::vector<std::vector<std::vector<int> > > set_coarse_;
};

} // namespace mfem

#endif // MFEM_ENTITY_SETS
