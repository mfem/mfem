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

#ifndef MFEM_MESQUITE
#define MFEM_MESQUITE

#include "../config/config.hpp"

#ifdef MFEM_USE_MESQUITE

#include "mesh.hpp"

#include "Mesquite_all_headers.hpp"

namespace mfem
{

using namespace Mesquite;

class MesquiteMesh : public Mesquite::Mesh
{
   // tagging interface definition
private:

   struct MfemTagDescription
   {

      std::string name;       //!< Tag name
      Mesh::TagType type;     //!< Tag data type
      size_t size;            //!< Size of tag data (sizeof(type)*array_length)

      inline MfemTagDescription( std::string n,
                                 Mesh::TagType t,
                                 size_t s)
         : name(n), type(t), size(s) {}

      inline MfemTagDescription( )
         : type(Mesh::BYTE), size(0) {}

      inline bool operator==(const MfemTagDescription& o) const
      { return name == o.name && type == o.type && size == o.size; }
      inline bool operator!=(const MfemTagDescription& o) const
      { return name != o.name || type != o.type || size != o.size; }
   };

   /**\class MeshTags
    *
    * Store tags and tag data for MFEM's mesh representation.
    * Stores for each tag: properties, element data, and vertex data.
    * The tag element and vertex data sets are maps between some element
    * or vertex index and a tag value.
    */
   class MeshTags
   {
   public:

      ~MeshTags() { clear(); }

      /** \class TagData
       * Store data for a single tag
       */
      struct TagData
      {

         //! tag meta data
         const MfemTagDescription desc;

         //! per-element data, or NULL if none has been set.
         void* elementData;

         //! number of entries in elementData
         size_t elementCount;

         //! per-vertex data, or NULL if none has been set.
         void* vertexData;

         //! number of entries in vertexData
         size_t vertexCount;

         //! Default value for tag
         void* defaultValue;

         /** \brief Construct tag
          *\param name Tag name
          *\param type Tag data type
          *\param length Tag array length (1 for scalar/non-array)
          *\param default_val Default value for tag
          */
         inline TagData( const std::string& name,
                         Mesh::TagType type, unsigned length,
                         void* default_val = 0)
            : desc(name, type, length*size_from_tag_type(type)),
              elementData(0), elementCount(0),
              vertexData(0), vertexCount(0),
              defaultValue(default_val) {}

         /** \brief Construct tag
          *\param descr Tag description object
          */
         inline TagData( const MfemTagDescription& descr )
            : desc(descr), elementData(0), elementCount(0),
              vertexData(0), vertexCount(0),
              defaultValue(0) {}

         ~TagData();
      };

      /** \brief Get the size of the passed data type */
      static size_t size_from_tag_type( Mesh::TagType type );

      /** \brief Clear all data */
      void clear();

      /** \brief Get tag index from name */
      size_t handle( const std::string& name, MsqError& err ) const;

      /** \brief Get tag properties */
      const MfemTagDescription& properties( size_t tag_handle, MsqError& err ) const;

      /** \brief Create a new tag
       *
       * Create a new tag with the passed properties
       *\param name Tag name (must be unique)
       *\param type Tag data type
       *\param length Number of values in tag (array length, 1 for scalar)
       *\param defval Optional default value for tag
       */
      size_t create( const std::string& name,
                     Mesh::TagType type,
                     unsigned length,
                     const void* defval,
                     MsqError& err );

      /** \brief Create a new tag
       *
       * Create a new tag with the passed properties
       */
      size_t create( const MfemTagDescription& desc,
                     const void* defval,
                     MsqError& err );

      /**\brief Remove a tag */
      void destroy( size_t tag_index, MsqError& err );

      /**\brief Set tag data on elements */
      void set_element_data( size_t tag_handle,
                             size_t num_indices,
                             const size_t* elem_indices,
                             const void* tag_data,
                             MsqError& err );

      /**\brief Set tag data on vertices */
      void set_vertex_data( size_t tag_handle,
                            size_t num_indices,
                            const size_t* elem_indices,
                            const void* tag_data,
                            MsqError& err );

      /**\brief Get tag data on elements */
      void get_element_data( size_t tag_handle,
                             size_t num_indices,
                             const size_t* elem_indices,
                             void* tag_data,
                             MsqError& err ) const;

      /**\brief Get tag data on vertices */
      void get_vertex_data( size_t tag_handle,
                            size_t num_indices,
                            const size_t* elem_indices,
                            void* tag_data,
                            MsqError& err ) const;

      /**\class TagIterator
       *
       * Iterate over list of valid tag handles
       */
      class TagIterator
      {
      public:
         TagIterator() : tags(0), index(0) {}
         TagIterator( MeshTags* d, size_t i ) : tags(d), index(i) {}
         size_t operator*() const  { return index+1; }
         TagIterator operator++();
         TagIterator operator--();
         TagIterator operator++(int);
         TagIterator operator--(int);
         bool operator==(TagIterator other) const { return index == other.index; }
         bool operator!=(TagIterator other) const { return index != other.index; }
      private:
         MeshTags* tags;
         size_t index;
      };
      TagIterator tag_begin();
      TagIterator tag_end()   { return TagIterator(this,tagList.size()); }

      /**\brief Check if any vertices have tag */
      bool tag_has_vertex_data( size_t index, MsqError& err ) ;
      /**\brief Check if any elements have tag */
      bool tag_has_element_data( size_t index, MsqError& err ) ;

   private:

      friend class MeshTags::TagIterator;

      std::vector<TagData*> tagList;
   }; // class MeshTags

   // data members
private:
   int ndofs;                      // number of nodes (or vertices) in mesh
   int nelems;                     // number of elements in mesh
   mfem::Mesh *mesh;               // pointer to mfem mesh object
   mfem::Element *elem;            // pointer to mfem element object
   mfem::GridFunction *nodes;      // pointer to mfem grid function object
   // for nodes
   mfem::FiniteElementSpace *fes;  // pointer to mfem finite element
   // space object
   mfem::Table *dof_elem;          // dof to element table
   std::vector<char> mByte;        // length = ndofs
   std::vector<bool> mFixed;       // length = ndofs

   MeshTags* myTags;

public:

   // The constructor
   MesquiteMesh(mfem::Mesh *mfem_mesh);

   // The mesh dimension
   int get_geometric_dimension(MsqError &err);

   // The handles are just pointers to the indexes of the nodes/elements
   void get_all_elements(std::vector<ElementHandle>& elements,
                         MsqError& err );
   void get_all_vertices(std::vector<VertexHandle>& vertices,
                         MsqError& err );

   // Get/set vertex coordinates
   void vertices_get_coordinates(const VertexHandle vert_array[],
                                 MsqVertex* coordinates,
                                 size_t num_vtx,
                                 MsqError &err );
   void vertex_set_coordinates(VertexHandle vertex,
                               const Vector3D &coordinates,
                               MsqError &err );

   // These are internal markers for Mesquite that we should allocate for them
   void vertex_set_byte(VertexHandle vertex,
                        unsigned char byte,
                        MsqError &err);
   void vertices_set_byte(const VertexHandle *vert_array,
                          const unsigned char *byte_array,
                          size_t array_size,
                          MsqError &err );
   void vertices_get_fixed_flag(const VertexHandle vert_array[],
                                std::vector<bool>& fixed_flag_array,
                                size_t num_vtx,
                                MsqError &err );
   void vertices_set_fixed_flag(const VertexHandle vert_array[],
                                const std::vector< bool > &fixed_flag_array,
                                size_t num_vtx,
                                MsqError &err );
   void vertex_get_byte( const VertexHandle vertex,
                         unsigned char *byte,
                         MsqError &err );
   void vertices_get_byte( const VertexHandle *vertex,
                           unsigned char *byte_array,
                           size_t array_size,
                           MsqError &err );
   // The dof_elem table
   void vertices_get_attached_elements(const VertexHandle* vertex_array,
                                       size_t num_vertex,
                                       std::vector<ElementHandle>& elements,
                                       std::vector<size_t>& offsets,
                                       MsqError& err );
   // The elem_dof table
   void elements_get_attached_vertices(const ElementHandle *elem_handles,
                                       size_t num_elems,
                                       std::vector<VertexHandle>& vert_handles,
                                       std::vector<size_t>& offsets,
                                       MsqError &err);

   // The topology of the elements: tri/tet/quad/hex...
   void elements_get_topologies(const ElementHandle *element_handle_array,
                                EntityTopology *element_topologies,
                                size_t num_elements,
                                MsqError &err);
   // The destructor
   ~MesquiteMesh();

   // tag the attributes on the elements from the underlying mfem mesh
   void tag_attributes();

   // Clean these up .....
   void vertices_get_slaved_flag( const VertexHandle vert_array[],
                                  std::vector<bool>& slaved_flag_array,
                                  size_t num_vtx,
                                  MsqError &err ) {};

   TagHandle tag_create( const std::string& tag_name,
                         TagType type, unsigned length,
                         const void* default_value,
                         MsqError &err);

   void tag_delete( TagHandle handle, MsqError& err );

   TagHandle tag_get( const std::string& name,
                      MsqError& err );

   void tag_properties( TagHandle handle,
                        std::string& name_out,
                        TagType& type_out,
                        unsigned& length_out,
                        MsqError& err );

   void tag_set_element_data( TagHandle handle,
                              size_t num_elems,
                              const ElementHandle* elem_array,
                              const void* tag_data,
                              MsqError& err );

   void tag_set_vertex_data ( TagHandle handle,
                              size_t num_elems,
                              const VertexHandle* node_array,
                              const void* tag_data,
                              MsqError& err );

   void tag_get_element_data( TagHandle handle,
                              size_t num_elems,
                              const ElementHandle* elem_array,
                              void* tag_data,
                              MsqError& err );

   void tag_get_vertex_data ( TagHandle handle,
                              size_t num_elems,
                              const VertexHandle* node_array,
                              void* tag_data,
                              MsqError& err );

   void release_entity_handles(const EntityHandle *handle_array,
                               size_t num_handles,
                               MsqError &err) {};
   void release() {};
};

}

#endif

#endif
