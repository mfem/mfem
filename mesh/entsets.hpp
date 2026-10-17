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

#ifndef MFEM_ENTITY_SETS
#define MFEM_ENTITY_SETS

#include "../config/config.hpp"
#include "../general/table.hpp"
#include "../general/stable3d.hpp"
#include <functional>
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
class Coefficient;
class Vector;

class EntitySets
{
   friend class Mesh;
   friend class NCMesh;
   friend class NCEntitySets;

public:
   enum EntityType {INVALID = -1, VERTEX = 0, EDGE = 1, FACE = 2, ELEMENT = 3};

   /// Rules for selecting elements using a physical-space point predicate.
   enum class ElementSetPointRule { CENTER, ALL_VERTICES };

   /// Rules for selecting edges using a physical-space point predicate.
   enum class EdgeSetPointRule { CENTER, ALL_VERTICES };

   /// Rules for selecting edges incident to an element set.
   enum class ElementSetEdgeRule { ALL, INTERIOR, FRONTIER };

   /// A function that determines whether a physical-space point is selected.
   using PointPredicate = std::function<bool(const Vector &)>;

   static std::map<EntityType,std::string> EntityTypeNames;

   EntitySets(Mesh & mesh);
   EntitySets(const EntitySets & ent_sets);
   EntitySets & operator=(const EntitySets & ent_sets) = delete;
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

   /// Create a named set and return its index within the specified entity type.
   int AddSet(EntityType t, const std::string & name);

   /// Add an entity identified by its local mesh index to a set.
   void AddEntity(EntityType t, int set_index, int entity_index);
   void AddEntity(EntityType t, const std::string & set_name, int entity_index);

   /// Add an edge set member using its two endpoint vertex indices.
   void AddEdge(int set_index, int v0, int v1);
   void AddEdge(const std::string & set_name, int v0, int v1);

   /// Add a triangular face set member using its vertex indices.
   void AddTriFace(int set_index, int v0, int v1, int v2);
   void AddTriFace(const std::string & set_name, int v0, int v1, int v2);

   /// Add a quadrilateral face set member using its vertex indices.
   void AddQuadFace(int set_index, int v0, int v1, int v2, int v3);
   void AddQuadFace(const std::string & set_name, int v0, int v1, int v2,
                    int v3);

   /** @brief Create an element set from local element indices.

       Repeated indices are ignored. */
   int AddElementSet(const std::string & name, const Array<int> & elements);

   /** @brief Create an element set containing elements with @a attribute. */
   int AddElementSetByAttribute(const std::string & name, int attribute);

   /** @brief Create an element set from a local element marker array.

       The marker must have one entry per local element; nonzero entries select
       their corresponding elements. */
   int AddElementSetByMarker(const std::string & name,
                             const Array<int> & marker);

   /** @brief Create an edge set from edges incident to an element set.

       ALL selects every edge of a selected element. INTERIOR selects edges
       whose incident elements are all selected. FRONTIER selects edges with
       both selected and unselected incident elements. */
   int AddEdgeSetFromElementSet(const std::string & name, int element_set,
                                ElementSetEdgeRule rule =
                                   ElementSetEdgeRule::ALL);

   /** @brief Create an edge set from boundary elements with @a attribute. */
   int AddEdgeSetByBoundaryAttribute(const std::string & name, int attribute);

   /** @brief Create an edge set by applying a predicate to physical points.

       With EdgeSetPointRule::CENTER, the predicate is evaluated at each edge
       midpoint. With EdgeSetPointRule::ALL_VERTICES, both edge endpoints must
       satisfy the predicate. */
   int AddEdgeSetByPredicate(const std::string & name,
                             const PointPredicate & predicate,
                             EdgeSetPointRule rule =
                                EdgeSetPointRule::CENTER);

   /** @brief Create an element set by applying a predicate to physical points.

       With ElementSetPointRule::CENTER, the predicate is evaluated at the
       image of each reference-element center. With
       ElementSetPointRule::ALL_VERTICES, every physical element vertex must
       satisfy the predicate. The selected elements are captured when this
       method is called; subsequent refinement follows the usual entity-set
       descendant rules. */
   int AddElementSetByPredicate(const std::string & name,
                                const PointPredicate & predicate,
                                ElementSetPointRule rule =
                                   ElementSetPointRule::CENTER);

   /** @brief Create an element set from the sublevel set @a level_set <=
       @a tolerance.

       The coefficient is evaluated at the same physical points as in
       AddElementSetByPredicate(). */
   int AddElementSetByLevelSet(const std::string & name,
                               Coefficient & level_set,
                               real_t tolerance = 0.0,
                               ElementSetPointRule rule =
                                  ElementSetPointRule::CENTER);

   inline std::set<int> & operator()(EntityType t, unsigned int s)
   { return sets_[t][s]; }
   inline const std::set<int> & operator()(EntityType t, unsigned int s) const
   { return sets_[t][s]; }

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
   void VerifyEntityType(EntityType t) const;
   void VerifySetIndex(EntityType t, int set_index) const;
   void VerifyEntityIndex(EntityType t, int entity_index) const;
   void VerifyVertexIndex(int vertex_index) const;
   void VerifyCanModify() const;

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
   NCEntitySets & operator=(const NCEntitySets & ncent_sets) = delete;

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
