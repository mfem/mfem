// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mesquite.hpp"

#ifdef MFEM_USE_MESQUITE

#include "../fem/fem.hpp"
#include <iostream>

namespace mfem
{

using namespace std;

MesquiteMesh::MeshTags::TagData::~TagData()
{
   if (elementData)
   {
      free(elementData);
   }
   if (vertexData)
   {
      free(vertexData);
   }
   if (defaultValue)
   {
      free(defaultValue);
   }
}


void MesquiteMesh::MeshTags::clear()
{
   for (std::vector<TagData*>::iterator iter = tagList.begin();
        iter != tagList.end(); ++iter)
      if (*iter)
      {
         delete *iter;
      }

   tagList.clear();
}

size_t MesquiteMesh::MeshTags::size_from_tag_type( Mesh::TagType type )
{
   switch ( type )
   {
      case Mesh::BYTE:   return 1;
      case Mesh::BOOL:   return sizeof(bool);
      case Mesh::DOUBLE: return sizeof(double);
      case Mesh::INT:    return sizeof(int);
      case Mesh::HANDLE: return sizeof(void*);
      case Mesh::LONG_LONG: return sizeof(long long);
      default: assert(0); return 0;
   }
}

size_t MesquiteMesh::MeshTags::create( const std::string& name,
                                       Mesh::TagType type,
                                       unsigned length,
                                       const void* defval,
                                       MsqError& err )
{
   size_t h = handle( name, err );
   if (h)
   {
      MSQ_SETERR(err)(name, MsqError::TAG_ALREADY_EXISTS);
      return 0;
   }

   if (length == 0 || size_from_tag_type(type) == 0)
   {
      MSQ_SETERR(err)(MsqError::INVALID_ARG);
      return 0;
   }

   TagData* tag = new TagData( name, type, length );
   h = tagList.size();
   tagList.push_back(tag);

   if (defval)
   {
      tag->defaultValue = malloc( tag->desc.size );
      memcpy( tag->defaultValue, defval, tag->desc.size );
   }

   return h+1;
}

size_t MesquiteMesh::MeshTags::create( const MfemTagDescription& desc,
                                       const void* defval,
                                       MsqError& err )
{
   size_t h = handle( desc.name.c_str(), err );
   if (h)
   {
      MSQ_SETERR(err)(desc.name.c_str(), MsqError::TAG_ALREADY_EXISTS);
      return 0;
   }

   err.clear();
   if (desc.size == 0 || (desc.size % size_from_tag_type(desc.type)) != 0)
   {
      MSQ_SETERR(err)(MsqError::INVALID_ARG);
      return 0;
   }

   TagData* tag = new TagData( desc );
   h = tagList.size();
   tagList.push_back(tag);

   if (defval)
   {
      tag->defaultValue = malloc( tag->desc.size );
      memcpy( tag->defaultValue, defval, tag->desc.size );
   }

   return h+1;
}

void MesquiteMesh::MeshTags::destroy( size_t tag_index, MsqError& err )
{
   --tag_index;
   if (tag_index >= tagList.size() || 0 == tagList[tag_index])
   {
      MSQ_SETERR(err)(MsqError::TAG_NOT_FOUND);
      return ;
   }

   delete tagList[tag_index];
   tagList[tag_index] = 0;
}

size_t MesquiteMesh::MeshTags::handle( const std::string& name,
                                       MsqError& err ) const
{
   for (size_t i = 0; i < tagList.size(); ++i)
      if (tagList[i] && tagList[i]->desc.name == name)
      {
         return i+1;
      }

   return 0;
}

const MesquiteMesh::MfemTagDescription& MesquiteMesh::MeshTags::properties(
   size_t tag_index, MsqError& err ) const
{
   static MfemTagDescription dummy_desc;
   --tag_index;

   if (tag_index >= tagList.size() || !tagList[tag_index])
   {
      MSQ_SETERR(err)("Invalid tag handle", MsqError::INVALID_ARG);
      return dummy_desc;
   }

   return tagList[tag_index]->desc;
}


void MesquiteMesh::MeshTags::set_element_data( size_t tag_index,
                                               size_t num_indices,
                                               const size_t* index_array,
                                               const void* values,
                                               MsqError& err )
{
   size_t i;
   char* data;
   --tag_index;
   if (tag_index >= tagList.size() || !tagList[tag_index])
   {
      MSQ_SETERR(err)("Invalid tag handle", MsqError::INVALID_ARG);
      return;
   }

   TagData* tag = tagList[tag_index];

   // Get highest element index
   size_t total = tag->elementCount;
   for (i = 0; i < num_indices; ++i)
      if (index_array[i] >= total)
      {
         total = index_array[i] + 1;
      }

   // If need more space
   if (total > tag->elementCount)
   {
      // allocate more space
      tag->elementData = realloc( tag->elementData, tag->desc.size * total );
      // if a default value, initialize new space with it
      if (tag->defaultValue)
      {
         data = ((char*)tag->elementData) + tag->elementCount * tag->desc.size;
         for (i = tag->elementCount; i < total; ++i)
         {
            memcpy( data, tag->defaultValue, tag->desc.size );
            data += tag->desc.size;
         }
      }
      else
      {
         memset( (char*)tag->elementData + tag->elementCount * tag->desc.size, 0,
                 (total - tag->elementCount) * tag->desc.size );
      }
      tag->elementCount = total;
   }

   // Store passed tag values
   data = (char*)tag->elementData;
   const char* iter = (const char*)values;
   for (i = 0; i < num_indices; ++i)
   {
      memcpy( data + index_array[i]*tag->desc.size, iter, tag->desc.size );
      iter += tag->desc.size;
   }
}

void MesquiteMesh::MeshTags::get_element_data( size_t tag_index,
                                               size_t num_indices,
                                               const size_t* index_array,
                                               void* values,
                                               MsqError& err ) const
{
   --tag_index;
   if (tag_index >= tagList.size() || !tagList[tag_index])
   {
      MSQ_SETERR(err)("Invalid tag handle", MsqError::INVALID_ARG);
      return;
   }

   TagData* tag = tagList[tag_index];

   char* iter = (char*)values;
   const char* data = (const char*)tag->elementData;

   for (size_t i = 0; i < num_indices; ++i)
   {
      const void* ptr;
      size_t index = index_array[i];
      if (index >= tag->elementCount)
      {
         ptr = tag->defaultValue;
         if (!ptr)
         {
            MSQ_SETERR(err)(MsqError::TAG_NOT_FOUND);
            return;
         }
      }
      else
      {
         ptr = data + index * tag->desc.size;
      }

      memcpy( iter, ptr, tag->desc.size );
      iter += tag->desc.size;
   }
}

void MesquiteMesh::MeshTags::set_vertex_data( size_t tag_index,
                                              size_t num_indices,
                                              const size_t* index_array,
                                              const void* values,
                                              MsqError& err )
{
   size_t i;
   char* data;
   --tag_index;
   if (tag_index >= tagList.size() || !tagList[tag_index])
   {
      MSQ_SETERR(err)("Invalid tag handle", MsqError::INVALID_ARG);
      return;
   }

   TagData* tag = tagList[tag_index];

   // Get highest element index
   size_t total = tag->vertexCount;
   for (i = 0; i < num_indices; ++i)
      if (index_array[i] >= total)
      {
         total = index_array[i] + 1;
      }

   // If need more space
   if (total > tag->vertexCount)
   {
      // allocate more space
      tag->vertexData = realloc( tag->vertexData, tag->desc.size * total );
      // if a default value, initialize new space with it
      if (tag->defaultValue)
      {
         data = ((char*)tag->vertexData) + tag->vertexCount * tag->desc.size;
         for (i = tag->vertexCount; i < total; ++i)
         {
            memcpy( data, tag->defaultValue, tag->desc.size );
            data += tag->desc.size;
         }
      }
      else
      {
         memset( (char*)tag->vertexData + tag->vertexCount * tag->desc.size, 0,
                 (total - tag->vertexCount) * tag->desc.size );
      }
      tag->vertexCount = total;
   }

   // Store passed tag values
   data = (char*)tag->vertexData;
   const char* iter = (const char*)values;
   for (i = 0; i < num_indices; ++i)
   {
      memcpy( data + index_array[i]*tag->desc.size, iter, tag->desc.size );
      iter += tag->desc.size;
   }
}

void MesquiteMesh::MeshTags::get_vertex_data( size_t tag_index,
                                              size_t num_indices,
                                              const size_t* index_array,
                                              void* values,
                                              MsqError& err ) const
{
   --tag_index;
   if (tag_index >= tagList.size() || !tagList[tag_index])
   {
      MSQ_SETERR(err)("Invalid tag handle", MsqError::INVALID_ARG);
      return;
   }

   TagData* tag = tagList[tag_index];

   char* iter = (char*)values;
   const char* data = (const char*)tag->vertexData;

   for (size_t i = 0; i < num_indices; ++i)
   {
      const void* ptr;
      size_t index = index_array[i];
      if (index >= tag->vertexCount)
      {
         ptr = tag->defaultValue;
         if (!ptr)
         {
            MSQ_SETERR(err)(MsqError::TAG_NOT_FOUND);
            return;
         }
      }
      else
      {
         ptr = data + index * tag->desc.size;
      }

      memcpy( iter, ptr, tag->desc.size );
      iter += tag->desc.size;
   }
}

bool MesquiteMesh::MeshTags::tag_has_vertex_data( size_t tag_index,
                                                  MsqError& err )
{
   --tag_index;
   if (tag_index >= tagList.size() || !tagList[tag_index])
   {
      MSQ_SETERR(err)("Invalid tag handle", MsqError::INVALID_ARG);
      return false;
   }

   TagData* tag = tagList[tag_index];
   return 0 != tag->vertexData || tag->defaultValue;
}

bool MesquiteMesh::MeshTags::tag_has_element_data( size_t tag_index,
                                                   MsqError& err )
{
   --tag_index;
   if (tag_index >= tagList.size() || !tagList[tag_index])
   {
      MSQ_SETERR(err)("Invalid tag handle", MsqError::INVALID_ARG);
      return false;
   }

   TagData* tag = tagList[tag_index];
   return 0 != tag->elementData || tag->defaultValue;
}

MesquiteMesh::MeshTags::TagIterator MesquiteMesh::MeshTags::tag_begin()
{
   size_t index = 0;
   while (index < tagList.size() && tagList[index] == NULL)
   {
      ++index;
   }
   return TagIterator( this, index );
}

MesquiteMesh::MeshTags::TagIterator
MesquiteMesh::MeshTags::TagIterator::operator++()
{
   ++index;
   while (index < tags->tagList.size() && NULL == tags->tagList[index])
   {
      ++index;
   }
   return TagIterator( tags, index );
}

MesquiteMesh::MeshTags::TagIterator
MesquiteMesh::MeshTags::TagIterator::operator--()
{
   --index;
   while (index < tags->tagList.size() && NULL == tags->tagList[index])
   {
      --index;
   }
   return TagIterator( tags, index );
}

MesquiteMesh::MeshTags::TagIterator
MesquiteMesh::MeshTags::TagIterator::operator++(int)
{
   size_t old = index;
   ++index;
   while (index < tags->tagList.size() && NULL == tags->tagList[index])
   {
      ++index;
   }
   return TagIterator( tags, old );
}

MesquiteMesh::MeshTags::TagIterator
MesquiteMesh::MeshTags::TagIterator::operator--(int)
{
   size_t old = index;
   --index;
   while (index < tags->tagList.size() && NULL == tags->tagList[index])
   {
      --index;
   }
   return TagIterator( tags, old );
}

//
// MesquiteMesh implementation follows
//
MesquiteMesh::MesquiteMesh(mfem::Mesh *mfem_mesh)
   :   myTags( new MeshTags )
{
   mesh = mfem_mesh;
   nelems = mesh->GetNE();
   nodes = mesh->GetNodes();

   if (nodes)
   {
      fes = nodes->FESpace();
      ndofs = fes->GetNDofs();
      fes->BuildElementToDofTable();
      const Table *elem_dof = &fes->GetElementToDofTable();
      dof_elem = new Table;
      Transpose(*elem_dof, *dof_elem, ndofs);
   }
   else
   {
      ndofs = mesh->GetNV();
      dof_elem = mesh->GetVertexToElementTable();
   }

   mByte    = vector<char>(ndofs);
   mFixed   = vector<bool>(ndofs, false);

   // By default, flag all boundary nodes as fixed
   Array<int> bdofs;
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      if (nodes)
      {
         fes->GetBdrElementDofs(i, bdofs);
      }
      else
      {
         mesh->GetBdrElementVertices(i, bdofs);
      }
      for (int j = 0; j < bdofs.Size(); j++)
      {
         mFixed[bdofs[j]] = true;
      }
   }

}

int MesquiteMesh::get_geometric_dimension(Mesquite::MsqError &err)
{
   return mesh->Dimension();
}

void MesquiteMesh::get_all_elements(std::vector<ElementHandle>& elements,
                                    Mesquite::MsqError& err)
{
   elements.resize(nelems);
   for (int i = 0; i < nelems; i++)
   {
      elements[i] = (ElementHandle) i;
   }
}

void MesquiteMesh::get_all_vertices(std::vector<VertexHandle>& vertices,
                                    Mesquite::MsqError& err)
{
   vertices.resize(ndofs);
   for (int i = 0; i < ndofs; i++)
   {
      vertices[i] = (VertexHandle) i;
   }
}

void MesquiteMesh::vertices_get_coordinates(const VertexHandle vert_array[],
                                            MsqVertex* coordinates,
                                            size_t num_vtx,
                                            MsqError &err)
{

   const size_t *indices = (const size_t*) vert_array;
   double coords[3];
   for (int i = 0; i < num_vtx; i++)
   {
      mesh->GetNode(indices[i], coords);
      coordinates[i].x(coords[0]);
      coordinates[i].y(coords[1]);
      if (mesh->Dimension() == 3 )
      {
         coordinates[i].z(coords[2]);
      }
      else
      {
         coordinates[i].z(0.0);
      }
   }
}

void MesquiteMesh::vertex_set_coordinates(VertexHandle vertex,
                                          const Vector3D &coordinates,
                                          MsqError &err)
{
   double coords[3];
   coords[0] = coordinates.x();
   coords[1] = coordinates.y();
   coords[2] = coordinates.z();
   mesh->SetNode((size_t) vertex, coords);
}

void MesquiteMesh::vertex_set_byte(VertexHandle vertex,
                                   unsigned char byte,
                                   MsqError &err)
{
   size_t index = (size_t) vertex;
   mByte[index] = byte;
}

void MesquiteMesh::vertices_set_byte(const VertexHandle *vert_array,
                                     const unsigned char *byte_array,
                                     size_t array_size,
                                     MsqError &err)
{
   const size_t* indices = (const size_t*) vert_array;
   for (int i = 0; i < array_size; i++)
   {
      mByte[indices[i]] = byte_array[i];
   }
}

void MesquiteMesh::vertex_get_byte(const VertexHandle vertex,
                                   unsigned char *byte,
                                   MsqError &err )
{
   *byte = mByte[(const size_t) vertex];
}

void MesquiteMesh::vertices_get_byte(const VertexHandle *vertex,
                                     unsigned char *byte_array,
                                     size_t array_size,
                                     MsqError &err )
{
   const size_t* indices = (const size_t*) vertex;
   for (int i = 0; i < array_size; i++)
   {
      byte_array[i] = mByte[indices[i]];
   }
}

void MesquiteMesh::vertices_get_fixed_flag(const VertexHandle vert_array[],
                                           std::vector<bool>& fixed_flag_array,
                                           size_t num_vtx,
                                           MsqError &err )
{
   fixed_flag_array.resize(num_vtx + 1);
   const size_t* indices = (const size_t*) vert_array;
   for (int i = 0; i < num_vtx; i++)
   {
      fixed_flag_array[i] = mFixed[indices[i]];
   }
}

void MesquiteMesh::vertices_set_fixed_flag(const VertexHandle vert_array[],
                                           const std::vector< bool > &fixed_flag_array,
                                           size_t num_vtx,
                                           MsqError &err )
{
   const size_t* indices = (const size_t*) vert_array;
   for (int i = 0; i < num_vtx; i++)
   {
      mFixed[indices[i]] = fixed_flag_array[i];
   }
}

void MesquiteMesh::elements_get_attached_vertices(const ElementHandle
                                                  *elem_handles,
                                                  size_t num_elems,
                                                  std::vector<VertexHandle>& vert_handles,
                                                  std::vector<size_t>& offsets,
                                                  MsqError &err)
{
   const size_t* indices = (const size_t*) elem_handles;
   vert_handles.clear();
   offsets.resize(num_elems + 1);

   Array<int> elem_dofs;

   for (int i = 0; i < num_elems; i++)
   {
      offsets[i] = vert_handles.size();
      elem = mesh->GetElement(indices[i]);
      if (nodes)
      {
         fes->GetElementDofs(indices[i],elem_dofs);
      }
      else
      {
         elem->GetVertices(elem_dofs);
      }

      for (int j = 0; j < elem_dofs.Size(); j++)
      {
         // Ordering of this matters!!!
         // We are good for triangles, quads and hexes. What about tets?
         vert_handles.push_back( (VertexHandle) elem_dofs[j] );
      }
   }
   offsets[num_elems] = vert_handles.size();
}

void MesquiteMesh::vertices_get_attached_elements(const VertexHandle*
                                                  vertex_array,
                                                  size_t num_vertex,
                                                  std::vector<ElementHandle>& elements,
                                                  std::vector<size_t>& offsets,
                                                  MsqError& err)
{
   const size_t* indices = (const size_t*) vertex_array;
   elements.clear();
   offsets.resize(num_vertex + 1);

   for (int i = 0; i < num_vertex; i++)
   {
      offsets[i] = elements.size();
      int* vertex_elems = dof_elem->GetRow(indices[i]);
      for (int j = 0; j < dof_elem->RowSize(indices[i]); j++)
      {
         elements.push_back( (ElementHandle) vertex_elems[j] );
      }
   }
   offsets[num_vertex] = elements.size();
}

void MesquiteMesh::elements_get_topologies(const ElementHandle
                                           *element_handle_array,
                                           EntityTopology *element_topologies,
                                           size_t num_elements,
                                           MsqError &err)
{
   // In MESQUITE:
   // TRIANGLE      = 8
   // QUADRILATERAL = 9
   // TETRAHEDRON   = 11
   // HEXAHEDRON    = 12

   // In MFEM:
   // POINT         = 0
   // SEGMENT       = 1
   // TRIANGLE      = 2
   // QUADRILATERAL = 3
   // TETRAHEDRON   = 4
   // HEXAHEDRON    = 5
   // BISECTED      = 6
   // QUADRISECTED  = 7
   // OCTASECTED    = 8

   int mfem_to_mesquite[9] = {0,0,8,9,11,12,0,0,0};
   const size_t* indices = (const size_t*) element_handle_array;
   for (int i = 0; i < num_elements; i++)
   {
      element_topologies[i] = (EntityTopology) mfem_to_mesquite[mesh->GetElementType(
                                                                   indices[i])];
   }
}

MesquiteMesh::~MesquiteMesh()
{
   MsqPrintError err(mfem::err);
   delete myTags;

   delete dof_elem;
}

void
MesquiteMesh::tag_attributes()
{
   MsqError err;

   // create a tag for a single integer value
   TagHandle attributeTagHandle = tag_create( "material", Mesh::INT, 1, 0, err );

   int *materialValues = new int[nelems];
   for (int i=0; i<nelems; i++)
   {
      materialValues[i] = mesh->GetAttribute(i);
   }

   //
   // now put these values into an element tag
   //
   std::vector<ElementHandle> elements;
   get_all_elements(elements, err );

   tag_set_element_data( attributeTagHandle, nelems, arrptr(elements),
                         (const void*)(materialValues), err );

   delete[] materialValues;
}

TagHandle MesquiteMesh::tag_create( const std::string& name,
                                    TagType type,
                                    unsigned length,
                                    const void* defval,
                                    MsqError& err )
{

   size_t size = MeshTags::size_from_tag_type( type );
   MfemTagDescription desc( name, type, length*size );
   size_t index = myTags->create( desc, defval, err ); MSQ_ERRZERO(err);
   return (TagHandle)index;
}

void MesquiteMesh::tag_delete( TagHandle handle, MsqError& err )
{
   myTags->destroy( (size_t)handle, err ); MSQ_CHKERR(err);
}

TagHandle MesquiteMesh::tag_get( const std::string& name, MsqError& err )
{
   size_t index = myTags->handle( name, err ); MSQ_ERRZERO(err);
   if (!index)
   {
      MSQ_SETERR(err)( MsqError::TAG_NOT_FOUND, "could not find tag \"%s\"",
                       name.c_str() );
   }
   return (TagHandle)index;
}

void MesquiteMesh::tag_properties( TagHandle handle,
                                   std::string& name,
                                   TagType& type,
                                   unsigned& length,
                                   MsqError& err )
{
   const MfemTagDescription& desc
      = myTags->properties( (size_t)handle, err ); MSQ_ERRRTN(err);

   name = desc.name;
   type = desc.type;
   length = (unsigned)(desc.size / MeshTags::size_from_tag_type( desc.type ));
}


void MesquiteMesh::tag_set_element_data( TagHandle handle,
                                         size_t num_elems,
                                         const ElementHandle* elem_array,
                                         const void* values,
                                         MsqError& err )
{
   myTags->set_element_data( (size_t)handle,
                             num_elems,
                             (const size_t*)elem_array,
                             values,
                             err );  MSQ_CHKERR(err);
}

void MesquiteMesh::tag_get_element_data( TagHandle handle,
                                         size_t num_elems,
                                         const ElementHandle* elem_array,
                                         void* values,
                                         MsqError& err )
{
   myTags->get_element_data( (size_t)handle,
                             num_elems,
                             (const size_t*)elem_array,
                             values,
                             err );  MSQ_CHKERR(err);
}

void MesquiteMesh::tag_set_vertex_data(  TagHandle handle,
                                         size_t num_elems,
                                         const VertexHandle* elem_array,
                                         const void* values,
                                         MsqError& err )
{
   myTags->set_vertex_data( (size_t)handle,
                            num_elems,
                            (const size_t*)elem_array,
                            values,
                            err );  MSQ_CHKERR(err);
}

void MesquiteMesh::tag_get_vertex_data(  TagHandle handle,
                                         size_t num_elems,
                                         const VertexHandle* elem_array,
                                         void* values,
                                         MsqError& err )
{
   myTags->get_vertex_data( (size_t)handle,
                            num_elems,
                            (const size_t*)elem_array,
                            values,
                            err );  MSQ_CHKERR(err);
}

static void BoundaryPreservingOptimization(mfem::MesquiteMesh &mesh)
{

   MsqDebug::enable(1);
   MsqPrintError err(mfem::err);

   int pOrder = 2;
   int mNumInterfaceSmoothIters = 5;
   bool mFixBndryInInterfaceSmooth = true; //this fixes exterior surface nodes
   bool project_gradient= false;
   double cos_crease_angle=0.2;

   // get all vertices
   std::vector<Mesquite::Mesh::VertexHandle> vertices;
   mesh.get_all_vertices(vertices, err);
   if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}
   int num_vertices = vertices.size();

   // get application fixed vertices
   std::vector<bool> app_fixed(num_vertices);

   mesh.vertices_get_fixed_flag(&(vertices[0]), app_fixed, num_vertices, err);
   if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}
   int num_app_fixed = 0;

   for (int i = 0; i < num_vertices; i++)
   {
      if (app_fixed[i]) { num_app_fixed++; }
   }

   mfem::out << "mesh has " << num_vertices << " vertices and " << num_app_fixed <<
             " are app fixed. ";

   // create planar domain for interior and assessor queues
   Mesquite::PlanarDomain geom( PlanarDomain::XY );

   // tag the underlying mesh material attributes.
   mesh.tag_attributes();

   // create boundary domain and find mesh boundary
   Mesquite::MeshBoundaryDomain2D* mesh_domain = new
   Mesquite::MeshBoundaryDomain2D( MeshBoundaryDomain2D::XY, 0.0, project_gradient,
                                   Mesquite::MeshBoundaryDomain2D::QUADRATIC);
   mesh_domain->skin_area_mesh(&mesh,cos_crease_angle,"material");

   std::vector<Mesquite::Mesh::VertexHandle> theBoundaryVertices;
   mesh_domain->get_boundary_vertices( theBoundaryVertices );

   int num_boundary_vertices = theBoundaryVertices.size();

   std::vector<Mesquite::Mesh::VertexHandle> theBoundaryEdges;
   mesh_domain->get_boundary_edges( theBoundaryEdges );

   //     int num_boundary_edges = theBoundaryEdges.size();
   std::vector<bool> fixed_flags_boundary(num_boundary_vertices);

   // get application fixed boundary vertices
   std::vector<bool> app_fixed_boundary(num_boundary_vertices);
   mesh.vertices_get_fixed_flag(&(theBoundaryVertices[0]),app_fixed_boundary,
                                num_boundary_vertices, err);

   int num_app_fixed_boundary = 0;
   for (int i = 0; i < num_boundary_vertices; i++)
   {
      if (app_fixed_boundary[i]) { num_app_fixed_boundary++; }
   }

   mfem::out << "mesh has " << num_boundary_vertices << " boundary vertices and "
             << num_app_fixed_boundary << " are app fixed" << std::endl;


   // only fix  boundary vertices along corners
   int num_fixed_boundary_flags = 0;

   std::vector<Mesquite::Mesh::VertexHandle> theCornerVertices;
   mesh_domain->get_corner_vertices( theCornerVertices );


   // fix only vertices that are classified as corners
   for (int i = 0; i < num_boundary_vertices; i++)
   {
      if (!mFixBndryInInterfaceSmooth)
      {
         fixed_flags_boundary[i] = false;
      }
      else
      {
         fixed_flags_boundary[i] = app_fixed_boundary[i];
      }

      for (int j = 0; j < theCornerVertices.size(); j++)
      {
         // printf("theCornerVertices[%d]=%lu\n",j, (size_t)(theCornerVertices[j]));
         if (theCornerVertices[j] == theBoundaryVertices[i])
         {
            fixed_flags_boundary[i] = true;
            num_fixed_boundary_flags++;
            break;
         }
      }
   }
   printf("fixed %d of %d boundary vertices (those classified corner)\n",
          num_fixed_boundary_flags, num_boundary_vertices);


   // creates three intruction queues
   Mesquite::InstructionQueue boundary_queue;
   Mesquite::InstructionQueue interior_queue;


   boundary_queue.set_slaved_ho_node_mode(Settings::SLAVE_ALL);
   interior_queue.set_slaved_ho_node_mode(Settings::SLAVE_ALL);

   TShapeB1 targetMetric;

   IdealShapeTarget tc;

   TQualityMetric metric( &tc, &targetMetric );

   Mesquite::LPtoPTemplate* obj_func = new Mesquite::LPtoPTemplate(&metric, pOrder,
                                                                   err);
   if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}

   Mesquite::QuasiNewton* boundary_alg = new Mesquite::QuasiNewton( obj_func);
   boundary_alg->use_element_on_vertex_patch();

   Mesquite::QuasiNewton* interior_alg = new Mesquite::QuasiNewton( obj_func );
   interior_alg->use_global_patch();

   // **************Set stopping criterion**************
   double grad_norm = 1e-5;
   double successiveEps = 1e-5;
   int boundary_outer = 5;
   int boundary_inner = 3;

   // for boundary
   Mesquite::TerminationCriterion* boundaryTermInner = new
   Mesquite::TerminationCriterion();
   Mesquite::TerminationCriterion* boundaryTermOuter = new
   Mesquite::TerminationCriterion();
   // boundaryTermInner->add_absolute_gradient_L2_norm(grad_norm);
   // boundaryTermInner->add_relative_successive_improvement(successiveEps);
   boundaryTermOuter->add_relative_successive_improvement(successiveEps);
   boundaryTermOuter->add_iteration_limit(boundary_outer);
   boundaryTermInner->add_iteration_limit(boundary_inner);

   ostringstream bndryStream;
   bndryStream<<"boundary_"<<targetMetric.get_name()<<"_p"<<pOrder;
   boundaryTermOuter->write_mesh_steps(bndryStream.str().c_str());
   boundary_alg->set_outer_termination_criterion(boundaryTermOuter);
   boundary_alg->set_inner_termination_criterion(boundaryTermInner);

   // for interior
   Mesquite::TerminationCriterion* interiorTermInner = new
   Mesquite::TerminationCriterion();
   Mesquite::TerminationCriterion* interiorTermOuter = new
   Mesquite::TerminationCriterion();
   interiorTermInner->add_absolute_gradient_L2_norm(grad_norm);
   interiorTermInner->add_relative_successive_improvement(successiveEps);
   // interiorTermInner->add_iteration_limit(3); // for element_on_vertex_patch mode
   interiorTermInner->add_iteration_limit(100); // for global_patch mode

   ostringstream interiorStream;
   interiorStream<<"interior_"<<targetMetric.get_name()<<"_p"<<pOrder;
   interiorTermOuter->write_mesh_steps(interiorStream.str().c_str());
   interiorTermOuter->add_iteration_limit(1);
   interior_alg->set_outer_termination_criterion(interiorTermOuter);
   interior_alg->set_inner_termination_criterion(interiorTermInner);

   // ConditionNumberQualityMetric qm_metric;
   // QualityAssessor boundary_assessor,interior_assessor;

   // boundary_assessor.add_quality_assessment( &metric, 10 );
   // boundary_assessor.add_quality_assessment( &qm_metric );

   // interior_assessor.add_quality_assessment( &metric, 10 );
   // interior_assessor.add_quality_assessment( &qm_metric );


   // set the boundary instruction queue
   // boundary_queue.add_quality_assessor( &boundary_assessor, err );
   boundary_queue.set_master_quality_improver(boundary_alg, err);
   // boundary_queue.add_quality_assessor( &boundary_assessor, err );
   if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}

   // set the interior instruction queue
   // interior_queue.add_quality_assessor( &interior_assessor, err );
   interior_queue.set_master_quality_improver(interior_alg, err);
   // interior_queue.add_quality_assessor( &interior_assessor, err );
   if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}

   err.clear();

   std::vector<bool> fixed_flags(num_vertices);

   for (int j=0; j<mNumInterfaceSmoothIters; j++)
   {

      mfem::out<<" Boundary + Interior smoothing pass "<< j<<"....."<<endl;

      // smooth boundary only
      for (int i = 0; i < num_vertices; i++) { fixed_flags[i] = true; }
      mesh.vertices_set_fixed_flag(&(vertices[0]),fixed_flags,num_vertices, err);
      if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}

      mesh.vertices_set_fixed_flag(&(theBoundaryVertices[0]),fixed_flags_boundary,
                                   num_boundary_vertices, err);
      if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}

      // -----------------------------------------------------
      // debug
      //
      mesh.vertices_get_fixed_flag(&(vertices[0]),fixed_flags,num_vertices, err);
      if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}
      int num_fixed = 0;
      for (int i = 0; i < num_vertices; i++)
      {
         if (fixed_flags[i]) { num_fixed++; }
      }

      mfem::out << "   For Boundary smooth, mesh has " << num_vertices <<
                " vertices and " << num_fixed << " are fixed. "<<endl;
      //
      // debug
      // -----------------------------------------------------


      boundary_queue.run_instructions(&mesh, mesh_domain, err);
      if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}
      mfem::out<<" boundary smooth completed in "<<boundaryTermOuter->get_iteration_count()
               <<" outer and "<<boundaryTermInner->get_iteration_count()
               <<" inner iterations."<<endl;


      // smooth interior only

      if (!mFixBndryInInterfaceSmooth)
      {
         //
         // let all interior vertices float during the interior smooth.
         //
         for (int i = 0; i < num_vertices; i++) { fixed_flags[i] = false; }
         mesh.vertices_set_fixed_flag(&(vertices[0]),fixed_flags,num_vertices, err);
      }
      else
      {
         //
         // use the app_fixed settings for the fixed state of boundary vertices.
         //
         mesh.vertices_set_fixed_flag(&(vertices[0]),app_fixed,num_vertices, err);
      }
      if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}
      for (int i = 0; i < num_boundary_vertices; i++) { fixed_flags[i] = true; }

      mesh.vertices_set_fixed_flag(&(theBoundaryVertices[0]),fixed_flags,
                                   num_boundary_vertices, err);
      if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}

      // -----------------------------------------------------
      // debug
      //
      mesh.vertices_get_fixed_flag(&(vertices[0]),fixed_flags,num_vertices, err);
      if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}
      num_fixed = 0;
      for (int i = 0; i < num_vertices; i++)
      {
         if (fixed_flags[i]) { num_fixed++; }
      }

      mfem::out << "    For Interior smooth, mesh has " << num_vertices <<
                " vertices and " << num_fixed << " are fixed. "<<endl;
      //
      // debug
      // -----------------------------------------------------

      interior_queue.run_instructions(&mesh, &geom, err);
      if (MSQ_CHKERR(err)) {mfem::out << err << std::endl; exit(EXIT_FAILURE);}
      mfem::out<<" interior smooth completed in "<<interiorTermOuter->get_iteration_count()
               <<" outer and "<<interiorTermInner->get_iteration_count()
               <<" inner iterations."<<endl;

   }


   delete mesh_domain;
   delete interiorTermOuter;
   delete interiorTermInner;
   delete boundaryTermOuter;
   delete boundaryTermInner;
   delete interior_alg;
   delete boundary_alg;
   delete obj_func;

}


// Implementation of Mesh::MesquiteSmooth method

void mfem::Mesh::MesquiteSmooth(const int mesquite_option)
{
   mfem::MesquiteMesh msq_mesh(this);
   MsqDebug::enable(1);
   MsqPrintError err(mfem::err);

   Wrapper *method;

   const double vert_move_tol = 1e-3;

   switch (mesquite_option)
   {
      case 0:  method = new LaplaceWrapper(); break;
      case 1:  method = new UntangleWrapper(); break;
      case 2:  method = new ShapeImprover(); break;
      case 3:  method = new PaverMinEdgeLengthWrapper(vert_move_tol); break;
   }

   if ( mesquite_option < 4 )
   {
      // Specify SLAVE_NONE for high order node positions
      method->set_slaved_ho_node_mode(Settings::SLAVE_NONE);

      if ( this->Dimension() == 3 )
      {
         method->run_instructions(&msq_mesh, err);
      }
      else
      {
         Vector3D normal(0,0,1);
         Vector3D  point(0,0,0);
         PlanarDomain mesh_plane(normal, point);

         method->run_instructions(&msq_mesh, &mesh_plane, err);
      }
   }
   else
   {
      // boundary perserving smoothing doesn't have a wrapper yet.
      BoundaryPreservingOptimization( msq_mesh );
   }
}

}

#endif
