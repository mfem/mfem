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

#ifdef MFEM_USE_MPI
//---------------------------------------------------------------------------//
ConduitDataCollection::ConduitDataCollection(MPI_Comm comm,
                                             const std::string& coll_name,
                                             Mesh *mesh)
   : DataCollection(coll_name, mesh),
     relay_protocol("hdf5")
{
   m_comm = comm;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
   appendRankToFileName = true; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names
}
#endif

//---------------------------------------------------------------------------//
ConduitDataCollection::~ConduitDataCollection()
{
   // empty
}

//---------------------------------------------------------------------------//
void ConduitDataCollection::Save()
{
   std::string dir_name = MeshDirectoryName();
   int err = create_directory(dir_name, mesh, myid);
   if (err)
   {
      MFEM_ABORT("Error creating directory: " << dir_name);
   }

   Node n_mesh;
   // future? If moved into Mesh class
   // mesh->toConduitBlueprint(n_mesh);
   MeshToBlueprintMesh(mesh,n_mesh);

   Node verify_info;
   if (!blueprint::mesh::verify(n_mesh,verify_info))
   {
      MFEM_ABORT("Conduit Mesh Blueprint Verify Failed:\n"
                 << verify_info.to_json());
   }

   FieldMapConstIterator itr;
   for ( itr = field_map.begin(); itr != field_map.end(); itr++)
   {
      std::string name = itr->first;
      GridFunction *gf = itr->second;
      // don't save mesh nodes twice ...
      if (  gf != mesh->GetNodes())
      {
         // future? If moved into GridFunction class
         //gf->toConduitBlueprint(n_mesh["fields"][it->first]);
         GridFunctionToBlueprintField(gf,
                                      n_mesh["fields"][name]);
      }
   }

   // save mesh data
   SaveMeshAndFields(myid,
                     n_mesh,
                     relay_protocol);

   if (myid == 0)
   {
      // save root file
      SaveRootFile(num_procs,
                   n_mesh,
                   relay_protocol);
   }
}

//---------------------------------------------------------------------------//
void ConduitDataCollection::Load(int cycle)
{
   DeleteAll();
   this->cycle = cycle;

   // Note: We aren't currently using much info from the root file ...
   // with cycle, we can use implicit mfem conduit file layout

   Node n_root;
   LoadRootFile(n_root);
   relay_protocol = n_root["protocol/name"].as_string();

   // for MPI case, we assume that we have # of mpi tasks
   //  == number of domains

   int num_domains = n_root["number_of_trees"].to_int();

   if (num_procs != num_domains)
   {
      error = READ_ERROR;
      MFEM_WARNING("num_procs must equal num_domains");
      return;
   }

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
// begin static public methods
//------------------------------

//---------------------------------------------------------------------------//
mfem::Mesh *
ConduitDataCollection::BlueprintMeshToMesh(const Node &n_mesh,
                                           const std::string &main_toplogy_name,
                                           bool zero_copy)
{
   // n_conv holds converted data (when necessary for mfem api)
   // if n_conv is used ( !n_conv.dtype().empty() ) we
   // now that some data allocation was necessary, so we
   // can't return a mesh that zero copies the conduit data
   Node n_conv;

   //
   // we need to find the topology and its coordset.
   //

   std::string topo_name = main_toplogy_name;
   // if topo name is not set, look for first topology
   if (topo_name == "")
   {
      topo_name = n_mesh["topologies"].schema().child_name(0);
   }

   MFEM_ASSERT(n_mesh.has_path("topologies/" + topo_name),
               "Expected topology named \"" + topo_name + "\" "
               "(node is missing path \"topologies/" + topo_name + "\")");

   // find the coord set
   std::string coords_name =
      n_mesh["topologies"][topo_name]["coordset"].as_string();


   MFEM_ASSERT(n_mesh.has_path("coordsets/" + coords_name),
               "Expected topology named \"" + coords_name + "\" "
               "(node is missing path \"coordsets/" + coords_name + "\")");

   const Node &n_coordset = n_mesh["coordsets"][coords_name];
   const Node &n_coordset_vals = n_coordset["values"];

   // get the number of dims of the coordset
   int ndims = n_coordset_vals.number_of_children();

   // get the number of points
   int num_verts = n_coordset_vals[0].dtype().number_of_elements();
   // get vals for points
   const double *verts_ptr = NULL;

   // the mfem mesh constructor needs coords with interleaved (aos) type
   // ordering, even for 1d + 2d we always need 3 doubles b/c it uses
   // Array<Vertex> and Vertex is a pod of 3 doubles. we check for this
   // case, if we don't have it we convert the data

   if (ndims == 3 &&
       n_coordset_vals[0].dtype().is_double() &&
       blueprint::mcarray::is_interleaved(n_coordset_vals) )
   {
      // already interleaved mcarray of 3 doubles,
      // return ptr to beginning
      verts_ptr = n_coordset_vals[0].value();
   }
   else
   {
      Node n_tmp;
      // check all vals, if we don't have doubles convert
      // to doubles
      NodeConstIterator itr = n_coordset_vals.children();
      while (itr.has_next())
      {
         const Node &c_vals = itr.next();
         std::string c_name = itr.name();

         if ( c_vals.dtype().is_double() )
         {
            // zero copy current coords
            n_tmp[c_name].set_external(c_vals);

         }
         else
         {
            // convert
            c_vals.to_double_array(n_tmp[c_name]);
         }
      }

      // check if we need to add extra dims to get
      // proper interleaved array
      if (ndims < 3)
      {
         // add dummy z
         n_tmp["z"].set(DataType::c_double(num_verts));
      }

      if (ndims < 2)
      {
         // add dummy y
         n_tmp["y"].set(DataType::c_double(num_verts));
      }

      Node &n_conv_coords_vals = n_conv["coordsets"][coords_name]["values"];
      blueprint::mcarray::to_interleaved(n_tmp,
                                         n_conv_coords_vals);
      verts_ptr = n_conv_coords_vals[0].value();
   }



   const Node &n_mesh_topo    = n_mesh["topologies"][topo_name];
   std::string mesh_ele_shape = n_mesh_topo["elements/shape"].as_string();

   mfem::Geometry::Type mesh_geo = ShapeNameToGeomType(mesh_ele_shape);
   int num_idxs_per_ele = Geometry::NumVerts[mesh_geo];

   const Node &n_mesh_conn = n_mesh_topo["elements/connectivity"];

   const int *elem_indices = NULL;
   // mfem requires ints, we could have int64s, etc convert if necessary
   if (n_mesh_conn.dtype().is_int() &&
       n_mesh_conn.is_compact() )
   {
      elem_indices = n_mesh_topo["elements/connectivity"].value();
   }
   else
   {
      Node &n_mesh_conn_conv=
         n_conv["topologies"][topo_name]["elements/connectivity"];
      n_mesh_conn.to_int_array(n_mesh_conn_conv);
      elem_indices = n_mesh_conn_conv.value();
   }

   int num_mesh_ele        =
      n_mesh_topo["elements/connectivity"].dtype().number_of_elements();
   num_mesh_ele            = num_mesh_ele / num_idxs_per_ele;


   const int *bndry_indices = NULL;
   int num_bndry_ele        = 0;
   // init to something b/c the mesh constructor will use this for a
   // table lookup, even if we don't have boundary info.
   mfem::Geometry::Type bndry_geo = mfem::Geometry::POINT;

   if ( n_mesh_topo.has_child("boundary_topology") )
   {
      std::string bndry_topo_name = n_mesh_topo["boundary_topology"].as_string();

      // In VisIt, we encountered a case were a mesh specified a boundary
      // topology, but the boundary topology was omitted from the blueprint
      // index, so it's data could not be obtained.
      //
      // This guard prevents an error in that case, allowing the mesh to be
      // created without boundary info

      if (n_mesh["topologies"].has_child(bndry_topo_name))
      {
         const Node &n_bndry_topo    = n_mesh["topologies"][bndry_topo_name];
         std::string bndry_ele_shape = n_bndry_topo["elements/shape"].as_string();

         bndry_geo = ShapeNameToGeomType(bndry_ele_shape);
         int num_idxs_per_bndry_ele = Geometry::NumVerts[mesh_geo];

         const Node &n_bndry_conn = n_bndry_topo["elements/connectivity"];

         // mfem requires ints, we could have int64s, etc convert if necessary
         if ( n_bndry_conn.dtype().is_int() &&
              n_bndry_conn.is_compact())
         {
            bndry_indices = n_bndry_conn.value();
         }
         else
         {
            Node &(n_bndry_conn_conv) =
               n_conv["topologies"][bndry_topo_name]["elements/connectivity"];
            n_bndry_conn.to_int_array(n_bndry_conn_conv);
            bndry_indices = (n_bndry_conn_conv).value();

         }

         num_bndry_ele =
            n_bndry_topo["elements/connectivity"].dtype().number_of_elements();
         num_bndry_ele = num_bndry_ele / num_idxs_per_bndry_ele;
      }
   }
   else
   {
      // Skipping Boundary Element Data
   }

   const int *mesh_atts  = NULL;
   const int *bndry_atts = NULL;

   // These variables are used in debug code below.
   // int num_mesh_atts_entires = 0;
   // int num_bndry_atts_entires = 0;

   // the attribute fields could have several names
   // for the element attributes check for first occurrence of field with
   // name containing "_attribute", that doesn't contain "boundary"
   std::string main_att_name = "";

   const Node &n_fields = n_mesh["fields"];
   NodeConstIterator itr = n_fields.children();

   while ( itr.has_next() && main_att_name == "" )
   {
      itr.next();
      std::string fld_name = itr.name();
      if ( fld_name.find("boundary")   == std::string::npos &&
           fld_name.find("_attribute") != std::string::npos )
      {
         main_att_name = fld_name;
      }
   }

   if ( main_att_name != "" )
   {
      const Node &n_mesh_atts_vals = n_fields[main_att_name]["values"];

      // mfem requires ints, we could have int64s, etc convert if necessary
      if (n_mesh_atts_vals.dtype().is_int() &&
          n_mesh_atts_vals.is_compact() )
      {
         mesh_atts = n_mesh_atts_vals.value();
      }
      else
      {
         Node &n_mesh_atts_vals_conv = n_conv["fields"][main_att_name]["values"];
         n_mesh_atts_vals.to_int_array(n_mesh_atts_vals_conv);
         mesh_atts = n_mesh_atts_vals_conv.value();
      }

      // num_mesh_atts_entires = n_mesh_atts_vals.dtype().number_of_elements();
   }
   else
   {
      // Skipping Mesh Attribute Data
   }

   // for the boundary attributes check for first occurrence of field with
   // name containing "_attribute", that also contains "boundary"
   std::string bnd_att_name = "";
   itr = n_fields.children();

   while ( itr.has_next() && bnd_att_name == "" )
   {
      itr.next();
      std::string fld_name = itr.name();
      if ( fld_name.find("boundary")   != std::string::npos &&
           fld_name.find("_attribute") != std::string::npos )
      {
         bnd_att_name = fld_name;
      }
   }

   if ( bnd_att_name != "" )
   {
      // Info: "Getting Boundary Attribute Data"
      const Node &n_bndry_atts_vals =n_fields[bnd_att_name]["values"];

      // mfem requires ints, we could have int64s, etc convert if necessary
      if ( n_bndry_atts_vals.dtype().is_int() &&
           n_bndry_atts_vals.is_compact())
      {
         bndry_atts = n_bndry_atts_vals.value();
      }
      else
      {
         Node &n_bndry_atts_vals_conv = n_conv["fields"][bnd_att_name]["values"];
         n_bndry_atts_vals.to_int_array(n_bndry_atts_vals_conv);
         bndry_atts = n_bndry_atts_vals_conv.value();
      }

      // num_bndry_atts_entires = n_bndry_atts_vals.dtype().number_of_elements();

   }
   else
   {
      // Skipping Boundary Attribute Data
   }

   // Info: "Number of Vertices: " << num_verts  << endl
   //         << "Number of Mesh Elements: "    << num_mesh_ele   << endl
   //         << "Number of Boundary Elements: " << num_bndry_ele  << endl
   //         << "Number of Mesh Attribute Entries: "
   //         << num_mesh_atts_entires << endl
   //         << "Number of Boundary Attribute Entries: "
   //         << num_bndry_atts_entires << endl);

   // Construct MFEM Mesh Object with externally owned data
   // Note: if we don't have a gf, we need to provide the proper space dim
   //       if nodes gf is attached later, it resets the space dim based
   //       on the gf's fes.
   Mesh *mesh = new Mesh(// from coordset
      const_cast<double*>(verts_ptr),
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
      ndims, // dim
      ndims); // space dim

   // Attach Nodes Grid Function, if it exists
   if (n_mesh_topo.has_child("grid_function"))
   {
      std::string nodes_gf_name = n_mesh_topo["grid_function"].as_string();

      // fetch blueprint field for the nodes gf
      const Node &n_mesh_gf = n_mesh["fields"][nodes_gf_name];
      // create gf
      mfem::GridFunction *nodes = BlueprintFieldToGridFunction(mesh,
                                                               n_mesh_gf);
      // attach to mesh
      mesh->NewNodes(*nodes,true);
   }


   if (zero_copy && !n_conv.dtype().is_empty())
   {
      //Info: "Cannot zero-copy since data conversions were necessary"
      zero_copy = false;
   }

   Mesh *res = NULL;

   if (zero_copy)
   {
      res = mesh;
   }
   else
   {
      // the mesh above contains references to external data, to get a
      // copy independent of the conduit data, we use:
      res = new Mesh(*mesh,true);
      delete mesh;
   }

   return res;
}

//---------------------------------------------------------------------------//
mfem::GridFunction *
ConduitDataCollection::BlueprintFieldToGridFunction(Mesh *mesh,
                                                    const Node &n_field,
                                                    bool zero_copy)
{
   // n_conv holds converted data (when necessary for mfem api)
   // if n_conv is used ( !n_conv.dtype().empty() ) we
   // know that some data allocation was necessary, so we
   // can't return a gf that zero copies the conduit data
   Node n_conv;

   const double *vals_ptr = NULL;

   int vdim = 1;

   Ordering::Type ordering = Ordering::byNODES;

   if (n_field["values"].dtype().is_object())
   {
      vdim = n_field["values"].number_of_children();

      // need to check that we have doubles and
      // cover supported layouts

      if ( n_field["values"][0].dtype().is_double() )
      {
         // check for contig
         if (n_field["values"].is_contiguous())
         {
            // conduit mcarray contig  == mfem byNODES
            vals_ptr = n_field["values"].child(0).value();
         }
         // check for interleaved
         else if (blueprint::mcarray::is_interleaved(n_field["values"]))
         {
            // conduit mcarray interleaved == mfem byVDIM
            ordering = Ordering::byVDIM;
            vals_ptr = n_field["values"].child(0).value();
         }
         else
         {
            // for mcarray generic case --  default to byNODES
            // and provide values w/ contiguous (soa) ordering
            blueprint::mcarray::to_contiguous(n_field["values"],
                                              n_conv["values"]);
            vals_ptr = n_conv["values"].child(0).value();
         }
      }
      else // convert to doubles and use contig
      {
         Node n_tmp;
         // check all vals, if we don't have doubles convert
         // to doubles
         NodeConstIterator itr = n_field["values"].children();
         while (itr.has_next())
         {
            const Node &c_vals = itr.next();
            std::string c_name = itr.name();

            if ( c_vals.dtype().is_double() )
            {
               // zero copy current coords
               n_tmp[c_name].set_external(c_vals);

            }
            else
            {
               // convert
               c_vals.to_double_array(n_tmp[c_name]);
            }
         }

         // for mcarray generic case --  default to byNODES
         // and provide values w/ contiguous (soa) ordering
         blueprint::mcarray::to_contiguous(n_tmp,
                                           n_conv["values"]);
         vals_ptr = n_conv["values"].child(0).value();
      }
   }
   else
   {
      if (n_field["values"].dtype().is_double() &&
          n_field["values"].is_compact())
      {
         vals_ptr = n_field["values"].value();
      }
      else
      {
         n_field["values"].to_double_array(n_conv["values"]);
         vals_ptr = n_conv["values"].value();
      }
   }

   if (zero_copy && !n_conv.dtype().is_empty())
   {
      //Info: "Cannot zero-copy since data conversions were necessary"
      zero_copy = false;
   }

   // we need basis name to create the proper mfem fec
   std::string fec_name = n_field["basis"].as_string();

   GridFunction *res = NULL;
   mfem::FiniteElementCollection *fec = FiniteElementCollection::New(
                                           fec_name.c_str());
   mfem::FiniteElementSpace *fes = new FiniteElementSpace(mesh,
                                                          fec,
                                                          vdim,
                                                          ordering);

   if (zero_copy)
   {
      res = new GridFunction(fes,const_cast<double*>(vals_ptr));
   }
   else
   {
      // copy case, this constructor will alloc the space for the GF data
      res = new GridFunction(fes);
      // create an mfem vector that wraps the conduit data
      Vector vals_vec(const_cast<double*>(vals_ptr),fes->GetVSize());
      // copy values into the result
      (*res) = vals_vec;
   }

   // TODO: I believe the GF already has ownership of fes, so this should be all
   // we need to do to avoid leaking objs created here?
   res->MakeOwner(fec);

   return res;
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::MeshToBlueprintMesh(Mesh *mesh,
                                           Node &n_mesh,
                                           const std::string &coordset_name,
                                           const std::string &main_topology_name,
                                           const std::string &boundary_topology_name)
{
   int dim = mesh->SpaceDimension();

   MFEM_ASSERT(dim >= 1 && dim <= 3, "invalid mesh dimension");

   ////////////////////////////////////////////
   // Setup main coordset
   ////////////////////////////////////////////

   // Assumes  mfem::Vertex has the layout of a double array.

   // this logic assumes an mfem vertex is always 3 doubles wide
   int stride = sizeof(mfem::Vertex);
   int num_vertices = mesh->GetNV();

   MFEM_ASSERT( ( stride == 3 * sizeof(double) ),
                "Unexpected stride for Vertex");

   Node &n_mesh_coords = n_mesh["coordsets"][coordset_name];
   n_mesh_coords["type"] =  "explicit";


   double *coords_ptr = mesh->GetVertex(0);

   n_mesh_coords["values/x"].set_external(coords_ptr,
                                          num_vertices,
                                          0,
                                          stride);

   if (dim >= 2)
   {
      n_mesh_coords["values/y"].set_external(coords_ptr,
                                             num_vertices,
                                             sizeof(double),
                                             stride);
   }
   if (dim >= 3)
   {
      n_mesh_coords["values/z"].set_external(coords_ptr,
                                             num_vertices,
                                             sizeof(double) * 2,
                                             stride);
   }

   ////////////////////////////////////////////
   // Setup main topo
   ////////////////////////////////////////////

   Node &n_topo = n_mesh["topologies"][main_topology_name];

   n_topo["type"]  = "unstructured";
   n_topo["coordset"] = coordset_name;

   Element::Type ele_type = mesh->GetElementType(0);

   std::string ele_shape = ElementTypeToShapeName(ele_type);

   n_topo["elements/shape"] = ele_shape;

   GridFunction *gf_mesh_nodes = mesh->GetNodes();

   if (gf_mesh_nodes != NULL)
   {
      n_topo["grid_function"] =  "mesh_nodes";
   }

   // connectivity
   // TODO: generic case, i don't think we can zero-copy (mfem allocs
   // an array per element) so we alloc our own contig array and
   // copy out. Some other cases (sidre) may actually have contig
   // allocation but I am  not sure how to detect this case from mfem
   int num_ele = mesh->GetNE();
   int geom = mesh->GetElementBaseGeometry(0);
   int idxs_per_ele = Geometry::NumVerts[geom];
   int num_conn_idxs =  num_ele * idxs_per_ele;

   n_topo["elements/connectivity"].set(DataType::c_int(num_conn_idxs));

   int *conn_ptr = n_topo["elements/connectivity"].value();

   for (int i=0; i < num_ele; i++)
   {
      const Element *ele = mesh->GetElement(i);
      const int *ele_verts = ele->GetVertices();

      memcpy(conn_ptr, ele_verts, idxs_per_ele * sizeof(int));

      conn_ptr += idxs_per_ele;
   }

   if (gf_mesh_nodes != NULL)
   {
      GridFunctionToBlueprintField(gf_mesh_nodes,
                                   n_mesh["fields/mesh_nodes"],
                                   main_topology_name);
   }

   ////////////////////////////////////////////
   // Setup mesh attribute
   ////////////////////////////////////////////

   Node &n_mesh_att = n_mesh["fields/element_attribute"];

   n_mesh_att["association"] = "element";
   n_mesh_att["topology"] = main_topology_name;
   n_mesh_att["values"].set(DataType::c_int(num_ele));

   int_array att_vals = n_mesh_att["values"].value();
   for (int i = 0; i < num_ele; i++)
   {
      att_vals[i] = mesh->GetAttribute(i);
   }

   ////////////////////////////////////////////
   // Setup bndry topo "boundary"
   ////////////////////////////////////////////

   // guard vs if we have boundary elements
   if (mesh->GetNBE() > 0)
   {
      n_topo["boundary_topology"] = boundary_topology_name;

      Node &n_bndry_topo = n_mesh["topologies"][boundary_topology_name];

      n_bndry_topo["type"]     = "unstructured";
      n_bndry_topo["coordset"] = coordset_name;

      Element::Type bndry_ele_type = mesh->GetBdrElementType(0);

      std::string bndry_ele_shape = ElementTypeToShapeName(bndry_ele_type);

      n_bndry_topo["elements/shape"] = bndry_ele_shape;


      int num_bndry_ele = mesh->GetNBE();
      int bndry_geom    = mesh->GetBdrElementBaseGeometry(0);
      int bndry_idxs_per_ele  = Geometry::NumVerts[bndry_geom];
      int num_bndry_conn_idxs =  num_bndry_ele * bndry_idxs_per_ele;

      n_bndry_topo["elements/connectivity"].set(DataType::c_int(num_bndry_conn_idxs));

      int *bndry_conn_ptr = n_bndry_topo["elements/connectivity"].value();

      for (int i=0; i < num_bndry_ele; i++)
      {
         const Element *bndry_ele = mesh->GetBdrElement(i);
         const int *bndry_ele_verts = bndry_ele->GetVertices();

         memcpy(bndry_conn_ptr, bndry_ele_verts, bndry_idxs_per_ele  * sizeof(int));

         bndry_conn_ptr += bndry_idxs_per_ele;
      }

      ////////////////////////////////////////////
      // Setup bndry mesh attribute
      ////////////////////////////////////////////

      Node &n_bndry_mesh_att = n_mesh["fields/boundary_attribute"];

      n_bndry_mesh_att["association"] = "element";
      n_bndry_mesh_att["topology"] = boundary_topology_name;
      n_bndry_mesh_att["values"].set(DataType::c_int(num_bndry_ele));

      int_array bndry_att_vals = n_bndry_mesh_att["values"].value();
      for (int i = 0; i < num_bndry_ele; i++)
      {
         bndry_att_vals[i] = mesh->GetBdrAttribute(i);
      }
   }
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::GridFunctionToBlueprintField(mfem::GridFunction *gf,
                                                    Node &n_field,
                                                    const std::string &main_topology_name)
{
   n_field["basis"] = gf->FESpace()->FEColl()->Name();
   n_field["topology"] = main_topology_name;

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
// end static public methods
//------------------------------

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

   if (relay_protocol == "hdf5")
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

   // there are cases where the data backing the gf fields doesn't
   // accurately represent the number of components in physical space,
   // so we loop over all gfs and fix those that are incorrect

   FieldMapConstIterator itr;
   for ( itr = field_map.begin(); itr != field_map.end(); itr++)
   {
      std::string gf_name = itr->first;
      GridFunction *gf = itr->second;

      Node &idx_gf_ncomps = n_bp_idx["mesh/fields"][gf_name]["number_of_components"];
      // check that the number_of_components in the index matches what we expect
      // correct if necessary
      if ( idx_gf_ncomps.to_int() != gf->VectorDim() )
      {
         idx_gf_ncomps = gf->VectorDim();
      }
   }
   // add extra header info
   n_root["protocol/name"]    =  relay_protocol;
   n_root["protocol/version"] = "0.3.1";


   // we will save one file per domain, so trees == files
   n_root["number_of_files"]  = num_domains;
   n_root["number_of_trees"]  = num_domains;
   n_root["file_pattern"]     = MeshFilePattern(relay_protocol);
   n_root["tree_pattern"]     = "";

   // Add the time, time step, and cycle
   n_root["blueprint_index/mesh/state/time"] = time;
   n_root["blueprint_index/mesh/state/time_step"] = time_step;
   n_root["blueprint_index/mesh/state/cycle"] = cycle;

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
   if (myid == 0)
   {
      // assume root file is json, unless hdf5 is specified
      std::string root_protocol = "json";

      if ( relay_protocol.find("hdf5") != std::string::npos )
      {
         root_protocol = "hdf5";
      }


      relay::io::load(RootFileName(), root_protocol, root_out);
#ifdef MFEM_USE_MPI
      // broadcast contents of root file other ranks
      // (conduit relay mpi  would simplify, but we would need to link another
      // lib for mpi case)

      // create json string
      std::string root_json = root_out.to_json();
      // string size +1 for null term
      int json_str_size = root_json.size() + 1;

      // broadcast json string buffer size
      int mpi_status = MPI_Bcast((void*)&json_str_size, // ptr
                                 1, // size
                                 MPI_INT, // type
                                 0, // root
                                 m_comm); // comm

      if (mpi_status != MPI_SUCCESS)
      {
         MFEM_ABORT("Broadcast of root file json string size failed");
      }

      // broadcast json string
      mpi_status = MPI_Bcast((void*)root_json.c_str(), // ptr
                             json_str_size, // size
                             MPI_CHAR, // type
                             0, // root
                             m_comm); // comm

      if (mpi_status != MPI_SUCCESS)
      {
         MFEM_ABORT("Broadcast of root file json string failed");
      }

#endif
   }

#ifdef MFEM_USE_MPI
   else
   {
      // recv json string buffer size via broadcast
      int json_str_size = -1;
      int mpi_status = MPI_Bcast(&json_str_size, // ptr
                                 1, // size
                                 MPI_INT, // type
                                 0, // root
                                 m_comm); // comm

      if (mpi_status != MPI_SUCCESS)
      {
         MFEM_ABORT("Broadcast of root file json string size failed");
      }

      // recv json string buffer via broadcast
      char *json_buff = new char[json_str_size];
      mpi_status = MPI_Bcast(json_buff,  // ptr
                             json_str_size, // size
                             MPI_CHAR, // type
                             0, // root
                             m_comm); // comm

      if (mpi_status != MPI_SUCCESS)
      {
         MFEM_ABORT("Broadcast of root file json string failed");
      }

      // reconstruct root file contents
      Generator g(std::string(json_buff),"json");
      g.walk(root_out);
      // cleanup temp buffer
      delete [] json_buff;
   }
#endif
}

//---------------------------------------------------------------------------//
void
ConduitDataCollection::LoadMeshAndFields(int domain_id,
                                         const std::string &relay_protocol)
{
   // Note: This path doesn't use any info from the root file
   // it uses the implicit mfem ConduitDataCollection layout

   Node n_mesh;
   relay::io::load( MeshFileName(domain_id, relay_protocol), n_mesh);


   Node verify_info;
   if (!blueprint::mesh::verify(n_mesh,verify_info))
   {
      MFEM_ABORT("Conduit Mesh Blueprint Verify Failed:\n"
                 << verify_info.to_json());
   }

   mesh = BlueprintMeshToMesh(n_mesh);

   field_map.clear();

   NodeConstIterator itr = n_mesh["fields"].children();

   std::string nodes_gf_name = "";

   const Node &n_topo = n_mesh["topologies/main"];
   if (n_topo.has_child("grid_function"))
   {
      nodes_gf_name = n_topo["grid_function"].as_string();
   }

   while (itr.has_next())
   {
      const Node &n_field = itr.next();
      std::string field_name = itr.name();

      // skip mesh nodes gf since they are already processed
      // skip attribute fields, they aren't grid functions
      if ( field_name != nodes_gf_name &&
           field_name.find("_attribute") == std::string::npos
         )
      {
         GridFunction *gf = BlueprintFieldToGridFunction(mesh, n_field);
         field_map.Register(field_name, gf, true);
      }
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
   // Adapted from SidreDataCollection

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
      case Element::WEDGE:
      default: ;
   }

   return "unknown";
}

//---------------------------------------------------------------------------//
mfem::Geometry::Type
ConduitDataCollection::ShapeNameToGeomType(const std::string &shape_name)
{
   // Note: must init to something to avoid invalid memory access
   // in the mfem mesh constructor
   mfem::Geometry::Type res = mfem::Geometry::POINT;

   if (shape_name == "point")
   {
      res = mfem::Geometry::POINT;
   }
   else if (shape_name == "line")
   {
      res =  mfem::Geometry::SEGMENT;
   }
   else if (shape_name == "tri")
   {
      res =  mfem::Geometry::TRIANGLE;
   }
   else if (shape_name == "quad")
   {
      res =  mfem::Geometry::SQUARE;
   }
   else if (shape_name == "tet")
   {
      res =  mfem::Geometry::TETRAHEDRON;
   }
   else if (shape_name == "hex")
   {
      res =  mfem::Geometry::CUBE;
   }
   else
   {
      MFEM_ABORT("Unsupported Element Shape: " << shape_name);
   }

   return res;
}

//------------------------------
// end static private methods
//------------------------------

} // end namespace mfem

#endif
