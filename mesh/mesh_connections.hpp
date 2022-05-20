// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MESH_CONNECTIONS
#define MFEM_MESH_CONNECTIONS

#include "../general/array.hpp"
#include <vector>

namespace mfem
{

class Table;
class Mesh;


/** @a Boundary indices are indexing the subset of boundary entities in a mesh.  MFEM meshes 
   currently only have one kind of Boundary entity indexed this way which is the mesh_dim-1 boundary.
   @a All indices are indexing all of the entities of a given dimension in a mesh (including
   those on the boundary).*/
enum EntityIndexType : bool 
{ 
  BdrIdx=true,    ///Index space contains only the entities on the boundary
  AllIdx=false    ///Index space contains all entities including those on the boundary
};

/** @brief Struct representing the index number of a geometrical mesh entity.  In
   MFEM meshes there is currently support for up to 5 entity kinds that all have their
   own index numberings: (0D, 1D, 2D, 1D Boundary) in 2D and (0D, 1D, 2D, 3D, 2D Boundary)
   in 3D.  These kinds are represented by the @a dim and @a boundary members.  For
   instance a 1D boundary member on a 2D mesh will have dim=1 and index_type=BdrIdx.  It
   should be noted that edges on the boundary of a 2D mesh have 2 numbers associated with them,
   their 1D edge numering and their 1D boundary element numbering.  This is of course also
   the case with 3D meshes and the 2D faces and 2D oundary elements.**/
struct EntityIndex
{
  unsigned short dim;
  EntityIndexType index_type;
  int index;
};

/** @brief Struct representing the plural index numbers of a geometrical mesh entities.  In
   MFEM meshes there is currently support for up to 5 entity kinds that all have their
   own index numberings: (0D-All, 1D=All, 2D-All, 1D-Bdr) in 2D and 
   (0D-All, 1D-All, 2D-All, 3D-All, 2D-Bdr) in 3D.  These kinds are represented by the 
   @a dim and @a index_type members.  For instance a 1D boundary member on a 2D mesh will have 
   dim=1 and index_type=BdrIdx.  It should be noted that edges on the boundary of a 2D mesh have 
   2 numbers associated with them, their 1D all edge numering and their 1D boundary element numbering.  This 
   is of course also the case with 3D meshes and the 2D faces and 2D boundary elements.**/
struct EntityIndices
{
  unsigned short dim;
  EntityIndexType index_type;
  Array<int> indices;
};


/** @brief Class that exposes connections between Mesh entities such as
    vertices, edges, faces, and volumes.  This is done without assuming
    the elements have a particular dimension or type and will support
    dimensions > 3 in the future.  EntityIndex numbers are described in the
    \ref EntityIndex and \ref EntityIndices structs.  The connections are defined
    on the Mesh object given at the time of construction, and Table objects are built
    lazily as they are required so the first query of a paticular kind may be a lot slower
    than the following queries.  It is worth noting that our naming convention of child
    and parent connections actually describes ancestor/decendent relationship eg. a node can
    be a "child" of edges, faces, and volumes.  It should also be noted that the ConnectionsOf
    methods provide all of the funcitonality of the ChildrenOf/ParentsOf methods and vice-versa.
    Both approaches are provided as a convenience to folks that are used to thiking of meshes 
    in either context.**/
class MeshConnections
{
   friend class Mesh;

private:
   /// The mesh object that the connections are defined on
   const Mesh &mesh;

   /** @brief 2D Array of tables describing connections between entity indices in the Mesh.
       The 2D array represents tables mapping from row indices to column indices of the 4-5
       different kinds of entities (0D/All, 1D/All, 2D/All, 3D/All, 2D/Bdr) respectively.  For example
       the Table mapping Vertices to Boundary Elements/Faces on a 3D mesh would be at T[0][3]*/
   mutable std::vector<std::vector<Table*>> T;

public:


   /** @brief Sets the Mesh object connections are defined on. */
   /** @note When this method is called, no connection tables
       are generated so they will all be null.  They will be generated
       lazily as access is needed.**/
   MeshConnections(const Mesh &m);

   /** @brief General method for getting the @a connected entity indices of the given 
       @a entity.  This method can represent both parent/child and child/parent relationships
       by setting the dimensions of @a entity and @a connected properly and will return the 
       same answers as the associated Children/Parents methods.  There are no self connections
       and using this method with the @a entity.dim = @a connected.dim will always return an 
       empty set of connections.  If a parent/child relationship is being used the 
       @a connected.indices will be listed in the natural order derived from the parent's
       element definition.  Otherwise the ordering of the connected.indices  not well 
       defined. The @a connected struct operates as an  in/out parameter.  The kind of the 
       connected entities is set in the @a connected.dim and @a connected.index_type members.  The 
       connected entity index numbers are placed in @a connected.indices member.  For the
       example mesh with vertex, edge, face/element, and boundary element
       Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                            entity      connected
                          dim,bdry,id  dim,bdry,ids
      ConnectionsOfEntity({2,AllIdx,0}, {0,AllIdx,{}}) ->  {0,AllIdx,{0,1,4,3}}, ChildrenOF relationship, nodes in element order 
      ConnectionsOfEntity({2,AllIdx,1}, {1,BdrIdx,{}}) ->  {1,BdrIdx,{1,5}}, ChildrenOf relationship
      ConnectionsOfEntity({1,AllIdx,7}, {2,AllIdx,{}}) ->  {2,AllIdx,{0,1}}, ParentsOf relationship
      ConnectionsOfEntity({1,BdrIdx,5}, {2,AllIdx,{}}) ->  {2,AllIdx,{1}}, ParentsOf relationship
      ConnectionsOfEntity({2,AllIdx,0}, {2,AllIdx,{}}) ->  {2,AllIdx,{}}, No connections defined of the same dimension**/
   void ConnectionsOfEntity(const EntityIndex entity, EntityIndices connected);

   /** @brief General method for getting the @a connected entity indices of the given set of
       @a entities.  This method can represent both parent/child and child/parent relationships
       by setting the dimensions of @a entities and @a connected properly and will return the 
       same answers as the associated Children/Parents methods.  There are no self connections
       and using this method with the @a entity.dim = @a connected.dim will always return an 
       empty set of connections.  The @a covered variable sets the behavior of which entities count
       in ParentsOf connections.  If @a covered is false then all connected parents will be returned.  If
       @a covered is true, the only the parents that are fully covered by the set of provided children 
       will be returned.  The @a connected struct operates as an in/out parameter.  The kind of the 
       connected entities is set in the @a connected.dim and @a connected.index_type members.  The 
       connected entity index numbers are placed in @a connected.indices member.  For the
       example mesh with vertex, edge, face/element, and boundary element
       Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                              entities          connected
                            dim,bdry,id       dim,bdry,ids
      ConnectionsOfEntities({1,AllIdx,{2,3}}, {0,AllIdx,{}}, false)      ->  {0,AllIdx,{3,4,5}}, not in natural element order
      ConnectionsOfEntities({2,AllIdx,{0,2}}, {1,BdrIdx,{}}, false)      ->  {1,BdrIdx,{0,2,4,6}} 
      ConnectionsOfEntities({0,AllIdx,{0,8}}, {2,AllIdx,{}}, false)      ->  {2,AllIdx,{0,3}}
      ConnectionsOfEntities({1,AllIdx,{0,1,2,6,7}}, {2,AllIdx,{}}, true) ->  {2,AllIdx,{0}}, only elem 0 is covered                     
    */
   void ConnectionsOfEntities(const EntityIndices entities, EntityIndices connected, bool covered);

   /** @brief Returns true if the @a parent entity is indeed a
       a parent of the @a child entity.  For the example mesh
       with vertex, edge, face/element, and boundary element Indices
       we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                 parent        child
               dim,bdry,id  dim,bdry,id
      IsChild({2,AllIdx,0}, {0,AllIdx,4})  -> true, any decendent counts as a child
      IsChild({2,AllIdx,0}, {0,AllIdx,8})  -> false
      IsChild({2,AllIdx,3}, {1,AllIdx,10}) -> true
      IsChild({2,AllIdx,0}, {2,AllIdx,0})  -> false, no self children
      IsChild({2,AllIdx,0}, {1,BdrIdx,4})  -> true
      IsChild({1,BdrIdx,3}, {0,AllIdx,8})  -> true
      IsChild({2,BdrIdx,3}, {0,AllIdx,8})  -> Error, no dim 2, boundary objects**/
   bool IsChild(const EntityIndex &parent, const EntityIndex &child) const;

   /** @brief Returns the child entity indices of the given @a parent
       entity.  If the children have a natural order in the given element
       this will return them in that order. The @a children struct operates as an 
       in/out parameter.  The kind of the child entities is set in the
       @a children.dim and @a children.index_type members.  The child
       entity index numbers are placed in @a children.indices member.  For the
       example mesh with vertex, edge, face/element, and boundary element
       Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                         parent       children
                       dim,bdry,id  dim,bdry,ids
      ChildrenOfEntity({2,AllIdx,0}, {0,AllIdx,{}}) ->  {0,AllIdx,{0,1,4,3}}, nodes in element ordering
      ChildrenOfEntity({2,AllIdx,1}, {1,BdrIdx,{}}) ->  {1,BdrIdx,{1,5}}
      ChildrenOfEntity({1,AllIdx,3}, {0,AllIdx,{}}) ->  {1,BdrIdx,{1,5}}**/
   void ChildrenOfEntity(const EntityIndex &parent, EntityIndices &children) const;

   /** @brief Returns the child entity indices of the given @a parents
       entities.  Since this is collective call that does not duplicate entries, 
       you cannot expect the children to be listed in their natural element
       orders. The @a children struct operates as an 
       in/out parameter.  The kind of the child entities is set in the
       @a children.dim and @a children.index_type members.  The child
       entity index numbers are placed in @a children.indices member.  For the
       example mesh with vertex, edge, face/element, and boundary element
       Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                            parent          children
                          dim,bdry,id     dim,bdry,ids
      ChildrenOfEntities({1,AllIdx,{2,3}}, {0,AllIdx,{}}) ->  {0,AllIdx,{3,4,5}}, not in natural element order
      ChildrenOfEntities({2,AllIdx,{0,2}}, {1,BdrIdx,{}}) ->  {1,BdrIdx,{0,2,4,6}}
      ChildrenOfEntities({1,BdrIdx,{4,7}}, {0,AllIdx,{}}) ->  {0,AllIdx,{0,3,5,8}}**/
   void ChildrenOfEntities(const EntityIndices &parents,EntityIndices &children) const;

   /** @brief Returns the parent entity indices of the given @a child
       entity.  The @a parents struct operates as an in/out
       parameter.  The kind of the parent entities is set in the
       @a parents.dim and @a parents.index_type members.  The parent
       entity index numbers are placed in @a parents.indices member.  For the
       example mesh with vertex, edge, face/element, and boundary element
       Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                        child         parents
                      dim,bdry,id   dim,bdry,ids
      ParentsOfEntity({1,AllIdx,7}, {2,AllIdx,{}}) ->  {2,AllIdx,{0,1}}
      ParentsOfEntity({1,BdrIdx,5}, {2,AllIdx,{}}) ->  {2,AllIdx,{1}}**/
   void ParentsOfEntity(const EntityIndex &child, EntityIndices &parents) const;

   /** @brief Returns the parent entity indices of any of the given @a children
       entities.  The @a parents struct operates as an in/out
       parameter.  The kind of the parent entities is set in the
       @a parents.dim and @a parents.index_type members.  The parent
       entity index numbers are placed in @a parents.indices member.  For the
       example mesh with vertex, edge, face/element, and boundary element
       Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                             children          parents
                           dim,bdry,ids      dim,bdry,ids
      ParentsOfAnyEntities({0,AllIdx,{0,8}}, {2,AllIdx,{}}) ->  {2,AllIdx,{0,3}}
      ParentsOfAnyEntities({1,BdrIdx,{4,6}}, {2,AllIdx,{}}) ->  {2,AllIdx,{0,2}}**/
   void ParentsOfAnyEntities(const EntityIndices &children,
                             EntityIndices &parents) const;

   /** @brief Returns the parent entity indices that have all their child entities
       covered by the @a children entities.  The @a parents struct operates as
       an in/out parameter.  The kind of the parent entities is set in the
       @a parents.dim and @a parents.index_type members.  The parent
       entity index numbers are placed in @a parents.indices member.  For the
       example mesh with vertex, edge, face/element, and boundary element
       Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                                  children                parents
                                dim,bdry,ids            dim,bdry,ids
      ParentsCoveredByEntities({1,AllIdx,{0,1,2,6,7}}, {2,AllIdx,{}}) ->  {2,AllIdx,{0}}, element 1 is not covered
      ParentsCoveredByEntities({0,AllIdx,{0,1,2,3,4}}, {2,AllIdx,{}}) ->  {2,AllIdx,{0}}
      ParentsCoveredByEntities({0,AllIdx,{0,1,2,3,6}}, {1,BdrIdx,{}}) ->  {1,BdrIdx,{0,1,4,6}}**/
   void ParentsCoveredByEntities(const EntityIndices &children,
                                 EntityIndices &parents) const;

   /** @brief Returns the neighbor entity indices of the given @a entity.  The @a neighbors
       struct operates as an in/out parameter.  The kind of the neighbor entities is set in the
       @a neighbors.dim and @a neighbors.index_type members.  The neighbor
       entity index numbers are placed in @a neighbors.ids member.  The @a shared_dim parameter
       sets the dimension of the entities that the neighbors are sharing across.  Note:  The
       entity.dim and neighbors.dim must be the same.  For example if the entity/neighbor dims are 0
       and the shared_dim is 0 this will return the vertex that are neighbors across the edges connected
       to the vertex entity.  For the example mesh with vertex, edge, face/element, and boundary element
       Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                            entity                           neighbors
                         dim,bdry,ids                      dim,bdry,ids
      NeighborsOfEntity({0,AllIdx,1},1,{0,AllIdx,{}}) ->  {0,AllIdx,{0,2,4}}
      NeighborsOfEntity({0,AllIdx,1},2,{0,AllIdx,{}}) ->  {0,AllIdx,{0,2,3,4,5}}
      NeighborsOfEntity({1,BdrIdx,5},0,{1,AllIdx,{}}) ->  {0,AllIdx,{1,11}},   edge numbering
      NeighborsOfEntity({1,BdrIdx,5},0,{1,BdrIdx,{}}) ->  {0,BdrIdx,{1,7}},    boundary element numbering
      NeighborsOfEntity({0,AllIdx,1},2,{1,AllIdx,{}}) ->  error, entity.dim != neighbors.dim**/
   void NeighborsOfEntity(const EntityIndex &entity, int shared_dim,
                          EntityIndices &neighbors) const;

   /** @brief Returns the neighbor entity indices of the given @a entities.  The @a neighbors
       struct operates as an in/out parameter.  The kind of the neighbor entities is set in the
       @a neighbors.dim and @a neighbors.index_type members.  The neighbor
       entity index numbers are placed in @a neighbors.ids member.  The @a shared_dim parameter
       sets the dimension of the entities that the neighbors are sharing across.  For example if the
       entity/neighbor dims are 0 and the shared_dim is 0 this will return the vertex that are
       neighbors across the edges connected to the vertex entity.  Note:  The entities.dim and
       neighbors.dim must be the same.  Also, none of the index numbers from the @a entities set
       will be returned in the @a neighbors set, even if they are neighbors of others in the
       @a entities set.  For the example mesh with vertex, edge, face/element, and
       boundary element Indices we have:

        0,All          1,All          2,All          1,Bdr
      6---7---8      +-4-+-5-+      +---+---+      +-2-+-3-+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+

                               entity                            neighbors
                            dim,bdry,ids                        dim,bdry,ids
      NeighborsOfEntities({0,AllIdx,{1,2}},1,{0,AllIdx,{}}) ->  {0,AllIdx,{0,4,5}},  2 is not included
      NeighborsOfEntities({0,AllIdx,{0,1}},2,{0,AllIdx,{}}) ->  {0,AllIdx,{2,3,4,5}}
      NeighborsOfEntities({1,BdrIdx,{1,5}},0,{1,AllIdx,{}}) ->  {1,AllIdx,{0,11}},   edge numbering
      NeighborsOfEntities({1,BdrIdx,{1,5}},0,{1,BdrIdx,{}}) ->  {1,BdrIdx,{0,7}},    boundary element numbering
      NeighborsOfEntities({0,AllIdx,{1,2}},2,{1,AllIdx,{}}) ->  error, entity.dim != neighbors.dim**/
   void NeighborsOfEntities(const EntityIndices &entities, int shared_dim,
                            EntityIndices &neighbors) const;

   EntityIndex& GetAllIdxFronBdrIdx(const EntityIndex &entity);
   EntityIndices& GetAllIdxFronBdrIdx(const EntityIndices &entity);

   EntityIndex& GetBdrIdxFromAllIdx(const EntityIndex &entity);
   EntityIndices& GetBdrIdxFromAllIdx(const EntityIndices &entity);

    private:
   /// Raw table access
   Table* GetTable(int row_dim, bool row_ind_type, int col_dim, bool col_ind_type) const;
};

}

#endif
