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


#include "../config/config.hpp"

#ifdef MFEM_USE_CONDUIT

#include "fem.hpp"
#include "../general/text.hpp"
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <string>
#include <sstream>

using namespace conduit;

namespace mfem
{
   
//---------------------------------------------------------------------------//
// class ConduitDataCollection implementation
//---------------------------------------------------------------------------//
   

//------------------------------
// begin public methods
//------------------------------

//---------------------------------------------------------------------------//
ConduitDataCollection::ConduitDataCollection(const std::string& coll_name,
                                             Mesh *mesh)
   : DataCollection(coll_name, mesh),
     relay_protocol("hdf5")
{
   appendRankToFileName = true; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names
}


//---------------------------------------------------------------------------//
ConduitDataCollection::~ConduitDataCollection()
{
   // empty
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::Save()
{
   create_directory(MeshDirectoryName(), mesh, myid);
   
   Node n_mesh;
   //mesh->toConduitBlueprint(n_mesh);
   MeshToBlueprintMesh(mesh,n_mesh);
   
   Node verify_info;
   if(!blueprint::mesh::verify(n_mesh,verify_info))
   {
      MFEM_ABORT("Conduit Mesh Blueprint Verify Failed:\n"
                  << verify_info.to_json());
   }
   
   FieldMapConstIterator itr;
   for( itr = field_map.begin(); itr != field_map.end(); itr++)
   {
      GridFunction *gf = itr->second;
      //gf->toConduitBlueprint(n_mesh["fields"][it->first]);
      GridFunctionToBlueprintField(gf,
                                   n_mesh["fields"][itr->first]);
   }

   // save mesh data
   SaveMeshAndFields(myid,
                     n_mesh,
                     relay_protocol);
   
   if(myid == 0)
   {
      // save root file
      SaveRootFile(num_procs,
                   n_mesh,
                   relay_protocol);
   }
}



//---------------------------------------------------------------------------//
void
ConduitDataCollection::Load(int cycle)
{
   DeleteAll();
   this->cycle = cycle;
   
   // Note: We aren't currently using any info from the root file ...
   // with cycle, we can use implicit mfem conduit file layout
   
   Node n_root;
   LoadRootFile(n_root);
   relay_protocol = n_root["protocol/name"].as_string();
   
   // load the mesh and fields
   LoadMeshAndFields(myid,relay_protocol);

   // TODO: am I properly wielding this?
   own_data = true;

}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::SetProtocol(const std::string &protocol)
{
   relay_protocol = protocol;
}

//------------------------------
// end public methods
//------------------------------

//------------------------------
// begin protected methods
//------------------------------

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::RootFileName()
{
    std::string res = prefix_path + name + "_" +
                      to_padded_string(cycle, pad_digits_cycle) +
                      ".root";
    return res;
}

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::MeshFileName(int domain_id,
                                    const std::string &relay_protocol)
{
   std::string res = prefix_path + 
                     name  + 
                     "_" +
                     to_padded_string(cycle, pad_digits_cycle) +
                     "/domain_" +
                     to_padded_string(domain_id, pad_digits_rank) +
                     "." +
                     relay_protocol;

   return res;
}

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::MeshDirectoryName()
{
   std::string res = prefix_path +
                     name +
                     "_" +
                     to_padded_string(cycle, pad_digits_cycle);
   return res;
}

//---------------------------------------------------------------------------//
std::string
ConduitDataCollection::MeshFilePattern(const std::string &relay_protocol)
{
   std::ostringstream oss;
   oss << prefix_path 
       << name 
       << "_"
       << to_padded_string(cycle, pad_digits_cycle)
       << "/domain_%0"
       << pad_digits_rank
       << "d."
       << relay_protocol;

   return oss.str();
}


//---------------------------------------------------------------------------//
void
ConduitDataCollection::SaveRootFile(int num_domains,
                                    const Node &n_mesh, 
                                    const std::string &relay_protocol)
{
   // default to json root file, except for hdf5 case
   std::string root_proto = "json";
   
   if(relay_protocol == "hdf5")
   {
      root_proto = relay_protocol;
   }

   Node n_root;
   // create blueprint index
   Node &n_bp_idx = n_root["blueprint_index"];

   blueprint::mesh::generate_index(n_mesh,
                                   "",
                                   num_domains,
                                   n_bp_idx["mesh"]);
   
   // add extra header info
   n_root["protocol/name"]    =  relay_protocol;
   n_root["protocol/version"] = "0.3.1";

   
   // we will save one file per domain, so trees == files
   n_root["number_of_files"]  = num_domains;
   n_root["number_of_trees"]  = num_domains;
   n_root["file_pattern"]     = MeshFilePattern(relay_protocol);
   n_root["tree_pattern"]     = "/";

   relay::io::save(n_root, RootFileName(), root_proto);
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::SaveMeshAndFields(int domain_id,
                                         const Node &n_mesh,
                                         const std::string &relay_protocol)
{
   relay::io::save(n_mesh, MeshFileName(domain_id, relay_protocol));
}



//---------------------------------------------------------------------------//
void
ConduitDataCollection::LoadRootFile(Node &root_out)
{
   relay::io::load(RootFileName(), relay_protocol, root_out);
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::LoadMeshAndFields(int domain_id, 
                                         const std::string &relay_protocol)
{ 
   // Note this path doesn't use any info from the root file
   // it uses the implicit mfem layout
   
   Node n_mesh;
   relay::io::load( MeshFileName(domain_id, relay_protocol), n_mesh);

   Node verify_info;
   if(!blueprint::mesh::verify(n_mesh,verify_info))
   {
      MFEM_ABORT("Conduit Mesh Blueprint Verify Failed:\n"
                  << verify_info.to_json());
   }

   mesh = BlueprintMeshToMesh(n_mesh);

   field_map.clear();
   
   NodeConstIterator itr = n_mesh["fields"].children();
   
   // todo: do we need to filter mesh_nodes, material_att, etc?
   while(itr.has_next())
   {
      const Node &n_field = itr.next();
      std::string field_name = itr.name();

      GridFunction *gf = BlueprintFieldToGridFunction(mesh, n_field);
      field_map[field_name] = gf;
   }
}


//------------------------------
// end protected methods
//------------------------------


//------------------------------
// begin static private methods
//------------------------------


//---------------------------------------------------------------------------//
std::string 
ConduitDataCollection::ElementTypeToShapeName(Element::Type element_type)
{
   // Note -- the mapping from Element::Type to string is based on
   //   enum Element::Type { POINT, SEGMENT, TRIANGLE, QUADRILATERAL,
   //                        TETRAHEDRON, HEXAHEDRON };
   // Note: -- the string names are from conduit's blueprint

   switch (element_type)
   {
      case Element::POINT:          return "point";
      case Element::SEGMENT:        return "line";
      case Element::TRIANGLE:       return "tri";
      case Element::QUADRILATERAL:  return "quad";
      case Element::TETRAHEDRON:    return "tet";
      case Element::HEXAHEDRON:     return "hex";
   }

   return "unknown";
}


//---------------------------------------------------------------------------//
mfem::Geometry::Type
ConduitDataCollection::ShapeNameToGeomType(const std::string &shape_name)
{
    // init to something to avoid invalid memory access 
    // in the mfem mesh constructor 
    mfem::Geometry::Type res = mfem::Geometry::POINT;
    
    if(shape_name == "point")
        res = mfem::Geometry::POINT;
    else if(shape_name == "line")
        res =  mfem::Geometry::SEGMENT;
    else if(shape_name == "tri")
        res =  mfem::Geometry::TRIANGLE;
    else if(shape_name == "quad")
        res =  mfem::Geometry::SQUARE;
    else if(shape_name == "tet")
        res =  mfem::Geometry::TETRAHEDRON;
    else if(shape_name == "hex")
        res =  mfem::Geometry::CUBE;
    else
    {
       MFEM_ABORT("Unsupported Element Shape: " << shape_name);
    }

    return res;
}

//---------------------------------------------------------------------------//
mfem::Mesh *
ConduitDataCollection::BlueprintMeshToMesh(const Node &n_mesh)
{

   MFEM_ASSERT(n_mesh.has_path("coordsets/coords"),
               "Expected topology named \"coords\" "
               "(node is missing path \"coordsets/coords\")");
   
   const Node n_coordset = n_mesh["coordsets/coords"];

   // get the number of dims of the coordset    
   int ndims = n_coordset["values"].number_of_children();

   // mfem mesh constructor needs coords with interleaved (aos) type ordering
   Node coords_values;
   blueprint::mcarray::to_interleaved(n_coordset["values"],coords_values);
  
   int num_verts         = coords_values[0].dtype().number_of_elements();
   double *verts_indices = coords_values[0].value();

   MFEM_ASSERT(n_mesh.has_path("topologies/main"),
               "Expected topology named \"main\" "
               "(node is missing path \"topologies/main\")");

   const Node &n_mesh_topo    = n_mesh["topologies/main"];
   std::string mesh_ele_shape = n_mesh_topo["elements/shape"].as_string();
   
   mfem::Geometry::Type mesh_geo = ShapeNameToGeomType(mesh_ele_shape);
   int num_idxs_per_ele = Geometry::NumVerts[mesh_geo];

   const int *elem_indices = n_mesh_topo["elements/connectivity"].value();
   int num_mesh_ele        = n_mesh_topo["elements/connectivity"].dtype().number_of_elements();
   num_mesh_ele            = num_mesh_ele / num_idxs_per_ele;


   const int *bndry_indices = NULL;
   int num_bndry_ele        = 0;
   // init to something b/c the mesh constructor will use this for a 
   // table lookup, even if we don't have boundary info.
   mfem::Geometry::Type bndry_geo = mfem::Geometry::POINT;

   if( n_mesh.has_path("topologies/boundary") )
   {
      const Node &n_bndry_topo    = n_mesh["topologies/boundary"];
      std::string bndry_ele_shape = n_bndry_topo["elements/shape"].as_string();
      
      bndry_geo = ShapeNameToGeomType(bndry_ele_shape);
      int num_idxs_per_bndry_ele = Geometry::NumVerts[mesh_geo];
      
      
      bndry_indices = n_bndry_topo["elements/connectivity"].value();
      num_bndry_ele = n_bndry_topo["elements/connectivity"].dtype().number_of_elements();
      num_bndry_ele = num_bndry_ele / num_idxs_per_bndry_ele;
   }
   else
   {
      // Skipping Boundary Element Data
   }

   const int *mesh_atts  = NULL;
   const int *bndry_atts = NULL;

   int num_mesh_atts_entires = 0;
   int num_bndry_atts_entires = 0;

   if( n_mesh.has_child("fields/main_attribute") )
   {
      const Node &n_mesh_atts_vals = n_mesh["fields/mesh_attribute/values"];
      mesh_atts  = n_mesh_atts_vals.value();
      num_mesh_atts_entires = n_mesh_atts_vals.dtype().number_of_elements();
   }
   else
   {
      // Skipping Mesh Attribute Data
   }

   if( n_mesh.has_child("fields/boundary_attribute") )
   {
      // BP_PLUGIN_INFO("Getting Boundary Attribute Data");
      bndry_atts = n_mesh["fields/boundary_attribute/values"].value();
      const Node &n_bndry_atts_vals = n_mesh["fields/boundary_attribute/values"];
      bndry_atts  = n_bndry_atts_vals.value();
      num_bndry_atts_entires = n_bndry_atts_vals.dtype().number_of_elements();
  
   }
   else
   {
      // Skipping Boundary Attribute Data
   }

   // BP_PLUGIN_INFO("Number of Vertices: " << num_verts  << endl
   //                << "Number of Mesh Elements: "    << num_mesh_ele   << endl
   //                << "Number of Boundary Elements: " << num_bndry_ele  << endl
   //                << "Number of Mesh Attribute Entries: "
   //                << num_mesh_atts_entires << endl
   //                << "Number of Boundary Attribute Entries: "
   //                << num_bndry_atts_entires << endl);
   //

   // Construct MFEM Mesh Object with externally owned data
   Mesh *mesh = new Mesh(// from coordset
                         verts_indices,
                         num_verts,
                         // from topology
                         const_cast<int*>(elem_indices),
                         mesh_geo,
                         // from mesh_attribute field
                         const_cast<int*>(mesh_atts),
                         num_mesh_ele,
                         // from boundary topology
                         const_cast<int*>(bndry_indices),
                         bndry_geo,
                         // from boundary_attribute field
                         const_cast<int*>(bndry_atts),
                         num_bndry_ele,
                         ndims,
                         1); // we need this flag

   // Attach Nodes Grid Function
   mfem::GridFunction *nodes = BlueprintFieldToGridFunction(mesh,
                                                            n_mesh["grid_function"]);

   mesh->NewNodes(*nodes,true);

   // the mesh above contains references to external data, to get a
   // copy independent of the conduit data, we use:

   Mesh *res = new Mesh(*mesh,true);

   delete mesh;

   return res;
}


//---------------------------------------------------------------------------//
mfem::GridFunction *
ConduitDataCollection::BlueprintFieldToGridFunction(Mesh *mesh,
                                                    const Node &n_field)
{
   // we need basis name to create the proper mfem fec
   std::string fec_name = n_field["basis"].as_string();

   mfem::FiniteElementCollection *fec = FiniteElementCollection::New(fec_name.c_str());


   Node n_field_vals;

   // TODO: USE ORDERING IN FiniteElementSpace CONSTRUCTOR TO AVOID REPACKING?
   int vdim = 1;
   if(n_field["values"].dtype().is_object())
   {
      // for mcarray case, the mfem gf constructor we need to use 
      // requires a contiguous (soa) ordering
      vdim = n_field["values"].number_of_children();
      blueprint::mcarray::to_contiguous(n_field["values"],
                                        n_field_vals);
   }
   else
   {
      n_field["values"].compact_to(n_field_vals);
   }

   mfem::FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec, vdim);


   double *vals_ptr = NULL;
   if(n_field["values"].dtype().is_object())
   {
      //the vals are contiguous, we fetch the pointer
      // to the first component in the mcarray
      vals_ptr = n_field_vals.child_ptr(0)->value();
   }
   else
   {
      vals_ptr = n_field_vals.value();
   }

   // zero copy case:
   // mfem::GridFunction *res = new GridFunction(fes,vals_ptr);
   
   // copy case
   mfem::GridFunction *res = new GridFunction(fes,NULL);
   res->NewDataAndSize(vals_ptr,fes->GetVSize());

   // TODO: I believe the GF already has ownership of fes, so this
   // should be all we need to do to avoid leaking objs created
   // here?
   res->MakeOwner(fec);
   
   return res;
}


//---------------------------------------------------------------------------//
void
ConduitDataCollection::MeshToBlueprintMesh(Mesh *mesh,
                                           Node &n_mesh)
{
   int dim = mesh->SpaceDimension();
   MFEM_ASSERT(dim >= 1 && dim <= 3, "invalid mesh dimension");

   ////////////////////////////////////////////
   // Setup main coordset "coords"
   ////////////////////////////////////////////

   // Assumes  mfem::Vertex has the layout of a double array.
   const int num_coords = sizeof(mfem::Vertex) / sizeof(double);
   const int num_vertices = mesh->GetNV();

   n_mesh["coordsets/coords/type"] =  "explicit";

   double *coords_ptr = mesh->GetVertex(0);

   n_mesh["coordsets/coords/values/x"].set_external(coords_ptr,
                                                    num_vertices,
                                                    0,
                                                    sizeof(double) * num_coords);
   
   if (dim >= 2)
   {
      n_mesh["coordsets/coords/values/y"].set_external(coords_ptr,
                                                       num_vertices,
                                                       sizeof(double),
                                                       sizeof(double) * num_coords);
   }
   if (dim >= 3)
   {
      n_mesh["coordsets/coords/values/z"].set_external(coords_ptr,
                                                       num_vertices,
                                                       sizeof(double) * 2,
                                                       sizeof(double) * num_coords);
   }


   ////////////////////////////////////////////
   // Setup main topo "main"
   ////////////////////////////////////////////

   Node &n_topo = n_mesh["topologies/main"];

   n_topo["type"]  = "unstructured";
   n_topo["coordset"] = "coords";

   Element::Type ele_type = static_cast<Element::Type>(mesh->GetElement(0)->GetType());

   std::string ele_shape = ElementTypeToShapeName(ele_type);

   n_topo["elements/shape"] = ele_shape;

   GridFunction *gf_mesh_nodes = mesh->GetNodes();

   if (gf_mesh_nodes != NULL)
   {
       n_topo["grid_function"] =  "mesh_nodes"; 
   }

   // connectivity 
   // TODO: generic case, i don't think we can zero-copy (there is an 
   // alloc per element) so we alloc our own contig array and copy out
   int num_ele = mesh->GetNE();
   int geom = mesh->GetElementBaseGeometry(0);
   int idxs_per_ele = Geometry::NumVerts[geom];
   std::cout << "idxs_per_ele" <<  idxs_per_ele << std::endl;
   int num_conn_idxs =  num_ele * idxs_per_ele;
   
   n_topo["elements/connectivity"].set(DataType::c_int(num_conn_idxs));
   
   int *conn_ptr = n_topo["elements/connectivity"].value();
      
   for(int i=0; i < num_ele; i++)
   {
      const Element *ele = mesh->GetElement(i);
      const int *ele_verts = ele->GetVertices();
      
      memcpy(conn_ptr, ele_verts, idxs_per_ele * sizeof(int));
      
      conn_ptr += idxs_per_ele;
   }

   if(gf_mesh_nodes != NULL)
   {
      GridFunctionToBlueprintField(gf_mesh_nodes,
                                   n_mesh["fields/mesh_nodes"]);
   }

   ////////////////////////////////////////////
   // Setup mesh attribute
   ////////////////////////////////////////////


   Node &n_mesh_att = n_mesh["fields/mesh_attribute"];
   
   n_mesh_att["association"] = "element";
   n_mesh_att["topology"] = "main";
   n_mesh_att["values"].set(DataType::c_int(num_ele));
   
   int_array att_vals = n_mesh_att["values"].value();
   for(int i = 0; i < num_ele; i++)
   {
      att_vals[i] = mesh->GetAttribute(i);
   }

   ////////////////////////////////////////////
   // Setup bndry topo "boundary"
   ////////////////////////////////////////////
   
   // guard vs if we have boundary elements
   if(mesh->GetNBE() > 0)
   {
      Node &n_bndry_topo = n_mesh["topologies/boundary"];
   
      n_bndry_topo["type"]     = "unstructured";
      n_bndry_topo["coordset"] = "coords";

      Element::Type bndry_ele_type = static_cast<Element::Type>(mesh->GetBdrElement(0)->GetType());

      std::string bndry_ele_shape = ElementTypeToShapeName(bndry_ele_type);

      n_bndry_topo["elements/shape"] = bndry_ele_shape;


      int num_bndry_ele = mesh->GetNBE();
      int bndry_geom    = mesh->GetBdrElementBaseGeometry(0);
      int bndry_idxs_per_ele  = Geometry::NumVerts[geom];
      int num_bndry_conn_idxs =  num_bndry_ele * bndry_idxs_per_ele;
   
      n_bndry_topo["elements/connectivity"].set(DataType::c_int(num_bndry_conn_idxs));
   
      int *bndry_conn_ptr = n_bndry_topo["elements/connectivity"].value();
      
      for(int i=0; i < num_bndry_ele; i++)
      {
         const Element *ele = mesh->GetBdrElement(i);
         const int *ele_verts = ele->GetVertices();

         memcpy(bndry_conn_ptr, ele_verts, bndry_idxs_per_ele  * sizeof(int));
      
         bndry_conn_ptr += bndry_idxs_per_ele;
      }
      
      ////////////////////////////////////////////
      // Setup bndry mesh attribute
      ////////////////////////////////////////////

      Node &n_bndry_mesh_att = n_mesh["fields/boundary_attribute"];
   
      n_bndry_mesh_att["association"] = "element";
      n_bndry_mesh_att["topology"] = "boundary";
      n_bndry_mesh_att["values"].set(DataType::c_int(num_ele));
   
      int_array bndry_att_vals = n_bndry_mesh_att["values"].value();
      for(int i = 0; i < num_ele; i++)
      {
         bndry_att_vals[i] = mesh->GetBdrAttribute(i);
      }
   }



}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::GridFunctionToBlueprintField(mfem::GridFunction *gf,
                                                    Node &n_field)
{
   
   n_field["basis"] = gf->FESpace()->FEColl()->Name();
   n_field["topology"] = "main";

   int vdim  = gf->FESpace()->GetVDim();
   int ndofs = gf->FESpace()->GetNDofs();

   if (vdim == 1) // scalar case
   {
      n_field["values"].set_external(gf->GetData(),
                                     ndofs);
   }
   else // vector case
   {
      // deal with striding of all components
      
      Ordering::Type ordering = gf->FESpace()->GetOrdering();
      
      int entry_stride = (ordering == Ordering::byNODES ? 1 : vdim);
      int vdim_stride  = (ordering == Ordering::byNODES ? ndofs : 1);
      
      index_t offset = 0;
      index_t stride = sizeof(double) * entry_stride;
      
      for (int d = 0;  d < vdim; d++)
      {
         std::ostringstream oss;
         oss << "v" << d;
         std::string comp_name = oss.str();
         n_field["values"][comp_name].set_external(gf->GetData(),
                                                   ndofs,
                                                   offset,
                                                   stride);
         offset +=  sizeof(double) * vdim_stride;
      }
   }
}


//------------------------------
// end static private methods
//------------------------------




} // end namespace mfem

#endif
