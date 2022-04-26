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

/** @brief Class that exposes connections between Mesh entities such as
    vertices, edges, faces, and volumes.  This is done without assuming 
    the elements have a particular dimension or type and will support 
    dimensions > 3 in the future.  EntityIndex numbers are described in the 
    \ref EntityIndex and \ref EntityIndices sturcts.  The connections are defined 
    on the Mesh object given at the time of construction, and Table objects are built 
    lazily as they are required so the first query of a paticular kind may be a lot slower
    than the following queries.  It is worth noting that the our naming convention of child
    and parent connections actually describes ancestor/decendent relationship eg. a node can 
    e a "child" of edges, faces, and volumes.**/
class MeshConnections
{
   friend class Mesh;

   private:
   /// The mesh object that the connections are defined on
   const Mesh &mesh;

   /** @brief 2D Array of tables describing connections between entity indices in the Mesh.
       The 2D array represents tables mapping from row indices to column indices of the 3-4
       different kinds of entities (0D, 1D, 2D, 3D, 2D Boundary) respectively.  For example 
       the Table mapping Vertices to Boundary Elements/Faces on a 3D mesh would be at T[0][3]*/
   std::vector<std::vector<Table*>> T;



   public:
   /** @brief Struct representing the index number of a geometrical mesh entity.  In 
       MFEM meshes there is currently support for 5 entity kinds that all have their
       own index numberings:  vertices, edges, faces, elements, and boundary elements. 
       These kinds are represented by the @a dim and @a boundary members.  For
       instance a mesh edge on the boundary will have dim=1 and boundary=true.  
       This struct is used to describe the index numbers recieved and returned by the 
       \ref MeshConnections class.**/
   struct EntityIndex
   {
      int dim;
      bool boundary;
      int index;
   };

   /** @brief Struct representing plural index numbers of geometrical mesh entities of the 
       same kind (dimension and boundary).  In MFEM meshes there is currently support for 5 
       (4 in 2D) entity kinds that all have their own index numberings:  vertices, edges, faces, 
       elements, and boundary elements.  These types are represented by the @a dim and @a boundary 
       members.  For instance faces in a 3D mesh interior will have dim=2 and boundary=false.
       This struct is used to describe the index numbers recieved and returned by the 
       \ref MeshConnections class.**/
   struct EntityIndices
   {
      int dim;
      bool boundary;
      Array<int> indices;      
   };

   /** @brief Sets the Mesh object connections are defined on. */
   /** @note When this method is called, no connection tables
       are generated so they will all be null.  They will be generated
       lazily as access is needed.**/
   MeshConnections(const Mesh &m);
   
   /** @brief Returns true if the @a parent entity is indeed a 
       a parent of the @a child entity.  For the example mesh
       with vertex, edge, face/element, and boundary element Indices
       we have:

       dim = 0        dim = 1        dim = 2     Boundary Elems 
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      7---8---9      +-4-+-5-+      +---+---+      +-2-+-3-+ 

                parent        child
              dim,bdry,id  dim,bdry,id
      IsChild({2,false,0}, {0,false,4})  -> true, any decendent counts as a child
      IsChild({2,false,0}, {0,false,8})  -> false
      IsChild({2,false,3}, {1,false,10}) -> true
      IsChild({2,false,0}, {2,false,0})  -> false, no self children
      IsChild({2,false,0}, {1,true, 6})  -> true
      IsChild({1,true, 3}, {0,false,8})  -> true
      IsChild({2,true, 3}, {0,false,8})  -> Error, no dim 2, boundary objects**/
   bool IsChild(EntityIndex &parent, EntityIndex &child);

   /** @brief Returns the child entity indices of the given @a parent
       entity.  The @a children struct operates as an in/out 
       parameter.  The kind of the child entities is set in the 
       @a children.dim and @a children.boundary members.  The child 
       entity index numbers are placed in @a children.indices member.  For the 
       example mesh with vertex, edge, face/element, and boundary element 
       Indices we have:

       dim = 0        dim = 1        dim = 2     Boundary Elems 
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      7---8---9      +-4-+-5-+      +---+---+      +-2-+-3-+ 

                         parent       children
                       dim,bdry,id  dim,bdry,ids
      ChildrenOfEntity({2,false,0}, {0,false,{}}) ->  {0,false,{0,1,3,4}}
      ChildrenOfEntity({2,false,1}, {1,true ,{}}) ->  {1,true ,{1,5}}**/
   void ChildrenOfEntity(const EntityIndex &parent, EntityIndices &children);

   /** @brief Returns the child entity indices of the given @a parents
       entities.  The @a children struct operates as an in/out 
       parameter.  The kind of the child entities is set in the 
       @a children.dim and @a children.boundary members.  The child 
       entity index numbers are placed in @a children.indices member.  For the 
       example mesh with vertex, edge, face/element, and boundary element 
       Indices we have:

       dim = 0        dim = 1        dim = 2     Boundary Elems 
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      7---8---9      +-4-+-5-+      +---+---+      +-2-+-3-+ 

                            parent          children
                          dim,bdry,id     dim,bdry,ids
      ChildrenOfEntities({1,false,{2,3}}, {0,false,{}}) ->  {0,false,{3,4,5}}
      ChildrenOfEntities({2,false,{0,2}}, {1,true ,{}}) ->  {1,true ,{4,6}}
      ChildrenOfEntities({1,true ,{4,7}}, {0,false,{}}) ->  {0,false,{0,3,5,9}}**/
   void ChildrenOfEntities(const EntityIndices &parents, EntityIndices &children);

   /** @brief Returns the parent entity indices of the given @a child
       entity.  The @a parents struct operates as an in/out 
       parameter.  The kind of the parent entities is set in the 
       @a parents.dim and @a parents.boundary members.  The parent 
       entity index numbers are placed in @a parents.indices member.  For the 
       example mesh with vertex, edge, face/element, and boundary element 
       Indices we have:

       dim = 0        dim = 1        dim = 2     Boundary Elems 
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      7---8---9      +-4-+-5-+      +---+---+      +-2-+-3-+ 

                        child        parents
                      dim,bdry,id  dim,bdry,ids
      ParentsOfEntity({1,false,7}, {2,false,{}}) ->  {2,false,{0,1}}
      ParentsOfEntity({1,true ,5}, {2,false,{}}) ->  {2,false,{1}}**/
   void ParentsOfEntity(const EntityIndex &child, EntityIndices &parents);

   /** @brief Returns the parent entity indices of any of the given @a children
       entities.  The @a parents struct operates as an in/out 
       parameter.  The kind of the parent entities is set in the 
       @a parents.dim and @a parents.boundary members.  The parent 
       entity index numbers are placed in @a parents.indices member.  For the 
       example mesh with vertex, edge, face/element, and boundary element 
       Indices we have:

       dim = 0        dim = 1        dim = 2     Boundary Elems 
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      7---8---9      +-4-+-5-+      +---+---+      +-2-+-3-+ 

                             children          parents
                           dim,bdry,ids     dim,bdry,ids
      ParentsOfAnyEntities({0,false,{0,9}}, {2,false,{}}) ->  {2,false,{0,3}}
      ParentsOfAnyEntities({1,true ,{4,6}}, {2,false,{}}) ->  {2,false,{0,2}}**/
   void ParentsOfAnyEntities(const EntityIndices &children, EntityIndices &parents);

   /** @brief Returns the parent entity indices that have all their child entities
       covered by the @a children entities.  The @a parents struct operates as 
       an in/out parameter.  The kind of the parent entities is set in the 
       @a parents.dim and @a parents.boundary members.  The parent 
       entity index numbers are placed in @a parents.indices member.  For the 
       example mesh with vertex, edge, face/element, and boundary element 
       Indices we have:

       dim = 0        dim = 1        dim = 2     Boundary Elems 
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      7---8---9      +-4-+-5-+      +---+---+      +-2-+-3-+ 

                                  children              parents
                                dim,bdry,ids          dim,bdry,ids
      ParentsCoveredByEntities({1,false,{0,1,2,6,7}}, {2,false,{}}) ->  {2,false,{0}}, element 1 is not covered
      ParentsCoveredByEntities({0,false,{0,1,2,3,4}}, {2,false,{}}) ->  {2,false,{0}}
      ParentsCoveredByEntities({0,false,{0,1,2,3,7}}, {1,true ,{}}) ->  {1,true ,{0,1,4,6}}**/   
   void ParentsCoveredByEntities(const EntityIndices &children, EntityIndices &parents);

   /** @brief Returns the neighbor entity indices of the given @a entity.  The @a neighbors 
       struct operates as an in/out parameter.  The kind of the neighbor entities is set in the 
       @a neighbors.dim and @a neighbors.boundary members.  The neighbor 
       entity index numbers are placed in @a neighbors.ids member.  The @a shared_dim parameter
       sets the dimension of the entities that the neighbors are sharing across.  Note:  The 
       entity.dim and neighbors.dim must be the same.  For example if the entity/neighbor dims are 0
       and the shared_dim is 0 this will return the vertex that are neighbors across the edges connected
       to the vertex entity.  For the example mesh with vertex, edge, face/element, and boundary element 
       Indices we have:

       dim = 0        dim = 1        dim = 2     Boundary Elems 
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      7---8---9      +-4-+-5-+      +---+---+      +-2-+-3-+ 

                            entity                         neighbors
                         dim,bdry,ids                    dim,bdry,ids
      NeighborsOfEntity({0,false,1},1,{0,false,{}}) ->  {0,false,{0,2,4}}
      NeighborsOfEntity({0,false,1},2,{0,false,{}}) ->  {0,false,{0,1,3,4}}
      NeighborsOfEntity({1,true ,5},0,{1,false,{}}) ->  {0,false,{1,11}},   edge numbering
      NeighborsOfEntity({1,true ,5},0,{1,true ,{}}) ->  {0,true,{1,7}},     boundary element numbering
      NeighborsOfEntity({0,false,1},2,{1,false,{}}) ->  error, entity.dim != neighbors.dim**/  
   void NeighborsOfEntity(const EntityIndex &entity, int shared_dim, EntityIndices &neighbors);

   /** @brief Returns the neighbor entity indices of the given @a entities.  The @a neighbors 
       struct operates as an in/out parameter.  The kind of the neighbor entities is set in the 
       @a neighbors.dim and @a neighbors.boundary members.  The neighbor 
       entity index numbers are placed in @a neighbors.ids member.  The @a shared_dim parameter
       sets the dimension of the entities that the neighbors are sharing across.  For example if the 
       entity/neighbor dims are 0 and the shared_dim is 0 this will return the vertex that are 
       neighbors across the edges connected to the vertex entity.  Note:  The entities.dim and 
       neighbors.dim must be the same.  Also, none of the index numbers from the @a entities set
       will be returned in the @a neighbors set, even if they are neighbors of others in the 
       @entities set.  For the example mesh with vertex, edge, face/element, and 
       boundary element Indices we have:

       dim = 0        dim = 1        dim = 2     Boundary Elems 
      0---1---2      +-0-+-1-+      +---+---+      +-0-+-1-+
      |   |   |      6   7   8      | 0 | 1 |      4   |   5
      3---4---5      +-2-+-3-+      +---+---+      +---+---+
      |   |   |      9  10  11      | 2 | 3 |      6   |   7
      7---8---9      +-4-+-5-+      +---+---+      +-2-+-3-+ 

                               entity                           neighbors
                            dim,bdry,ids                       dim,bdry,ids
      NeighborsOfEntities({0,false,{1,2}},1,{0,false,{}}) ->  {0,false,{0,4,5}}, 2 is not included
      NeighborsOfEntities({0,false,{0,1}},2,{0,false,{}}) ->  {0,false,{2,3,4,5}}
      NeighborsOfEntities({1,true ,{1,5}},0,{1,false,{}}) ->  {1,false,{0,3,7,11}},   edge numbering
      NeighborsOfEntities({1,true ,{1,5}},0,{1,true ,{}}) ->  {1,true,{0,7}},     boundary element numbering
      NeighborsOfEntities({0,false,{1,2}},2,{1,false,{}}) ->  error, entity.dim != neighbors.dim**/  
   void NeighborsOfEntities(const EntityIndices &entities, int shared_dim, EntityIndices &neighbors);   

   private:
   /// Raw table access
   Table* GetTable(int row_dim, bool row_bndry, int col_dim, bool col_bndry);
};

}

#endif