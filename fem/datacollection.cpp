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

#include "fem.hpp"
#include "../mesh/nurbs.hpp"
#include "../general/binaryio.hpp"
#include "../general/text.hpp"
#include "picojson.h"

#include <cerrno>      // errno
#include <sstream>

#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

namespace mfem
{

// static method
int DataCollection::create_directory(const std::string &dir_name,
                                     const Mesh *mesh, int myid)
{
   // create directories recursively
   const char path_delim = '/';
   std::string::size_type pos = 0;
   int err;
#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh*>(mesh);
#endif

   do
   {
      pos = dir_name.find(path_delim, pos+1);
      std::string subdir = dir_name.substr(0, pos);

#ifndef MFEM_USE_MPI
      err = mkdir(subdir.c_str(), 0777);
      err = (err && (errno != EEXIST)) ? 1 : 0;
#else
      if (myid == 0 || pmesh == NULL)
      {
         err = mkdir(subdir.c_str(), 0777);
         err = (err && (errno != EEXIST)) ? 1 : 0;
      }
#endif
   }
   while ( pos != std::string::npos );

#ifdef MFEM_USE_MPI
   if (pmesh)
   {
      MPI_Bcast(&err, 1, MPI_INT, 0, pmesh->GetComm());
   }
#endif

   return err;
}

// class DataCollection implementation

DataCollection::DataCollection(const std::string& collection_name, Mesh *mesh_)
{
   std::string::size_type pos = collection_name.find_last_of('/');
   if (pos == std::string::npos)
   {
      name = collection_name;
      // leave prefix_path empty
   }
   else
   {
      prefix_path = collection_name.substr(0, pos+1);
      name = collection_name.substr(pos+1);
   }
   mesh = mesh_;
   myid = 0;
   num_procs = 1;
   serial = true;
   appendRankToFileName = false;

#ifdef MFEM_USE_MPI
   m_comm = MPI_COMM_NULL;
   ParMesh *par_mesh = dynamic_cast<ParMesh*>(mesh);
   if (par_mesh)
   {
      myid = par_mesh->GetMyRank();
      num_procs = par_mesh->GetNRanks();
      m_comm = par_mesh->GetComm();
      serial = false;
      appendRankToFileName = true;
   }
#endif
   own_data = false;
   cycle = -1;
   time = 0.0;
   time_step = 0.0;
   precision = precision_default;
   pad_digits_cycle = pad_digits_rank = pad_digits_default;
   format = SERIAL_FORMAT; // use serial mesh format
   compression = false;
   error = NO_ERROR;
}

void DataCollection::SetMesh(Mesh *new_mesh)
{
   if (own_data && new_mesh != mesh) { delete mesh; }
   mesh = new_mesh;
   myid = 0;
   num_procs = 1;
   serial = true;
   appendRankToFileName = false;

#ifdef MFEM_USE_MPI
   m_comm = MPI_COMM_NULL;
   ParMesh *par_mesh = dynamic_cast<ParMesh*>(mesh);
   if (par_mesh)
   {
      myid = par_mesh->GetMyRank();
      num_procs = par_mesh->GetNRanks();
      m_comm = par_mesh->GetComm();
      serial = false;
      appendRankToFileName = true;
   }
#endif
}

#ifdef MFEM_USE_MPI
void DataCollection::SetMesh(MPI_Comm comm, Mesh *new_mesh)
{
   // This seems to be the cleanest way to accomplish this
   // and avoid duplicating fine grained details:

   SetMesh(new_mesh);

   m_comm = comm;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
}
#endif

void DataCollection::SetFormat(int fmt)
{
   switch (fmt)
   {
      case SERIAL_FORMAT: break;
#ifdef MFEM_USE_MPI
      case PARALLEL_FORMAT: break;
#endif
      default: MFEM_ABORT("unknown format: " << fmt);
   }
   format = fmt;
}

void DataCollection::SetCompression(bool comp)
{
   compression = comp;
#ifndef MFEM_USE_ZLIB
   MFEM_VERIFY(!compression, "ZLib not enabled in MFEM build.");
#endif
}

void DataCollection::SetPrefixPath(const std::string& prefix)
{
   if (!prefix.empty())
   {
      prefix_path = prefix;
      if (!prefix_path.empty() && prefix_path[prefix_path.size()-1] != '/')
      {
         prefix_path += '/';
      }
   }
   else
   {
      prefix_path.clear();
   }
}

void DataCollection::Load(int cycle)
{
   MFEM_ABORT("this method is not implemented");
}

void DataCollection::Save()
{
   SaveMesh();

   if (error) { return; }

   for (FieldMapIterator it = field_map.begin(); it != field_map.end(); ++it)
   {
      SaveOneField(it);
      // Even if there is an error, try saving the other fields
   }

   for (QFieldMapIterator it = q_field_map.begin(); it != q_field_map.end();
        ++it)
   {
      SaveOneQField(it);
   }
}

void DataCollection::SaveMesh()
{
   int err;

   std::string dir_name = prefix_path + name;
   if (cycle != -1)
   {
      dir_name += "_" + to_padded_string(cycle, pad_digits_cycle);
   }
   err = create_directory(dir_name, mesh, myid);
   if (err)
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error creating directory: " << dir_name);
      return; // do not even try to write the mesh
   }

   std::string mesh_name = GetMeshFileName();
   mfem::ofgzstream mesh_file(mesh_name, compression);
   mesh_file.precision(precision);
#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh*>(mesh);
   if (pmesh && format == PARALLEL_FORMAT)
   {
      pmesh->ParPrint(mesh_file);
   }
   else
#endif
   {
      mesh->Print(mesh_file);
   }
   if (!mesh_file)
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing mesh to file: " << mesh_name);
   }
}

std::string DataCollection::GetMeshShortFileName() const
{
   return (serial || format == SERIAL_FORMAT) ? "mesh" : "pmesh";
}

std::string DataCollection::GetMeshFileName() const
{
   return GetFieldFileName(GetMeshShortFileName());
}

std::string DataCollection::GetFieldFileName(const std::string &field_name)
const
{
   std::string dir_name = prefix_path + name;
   if (cycle != -1)
   {
      dir_name += "_" + to_padded_string(cycle, pad_digits_cycle);
   }
   std::string file_name = dir_name + "/" + field_name;
   if (appendRankToFileName)
   {
      file_name += "." + to_padded_string(myid, pad_digits_rank);
   }
   return file_name;
}

void DataCollection::SaveOneField(const FieldMapIterator &it)
{
   mfem::ofgzstream field_file(GetFieldFileName(it->first), compression);

   field_file.precision(precision);
   (it->second)->Save(field_file);
   if (!field_file)
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing field to file: " << it->first);
   }
}

void DataCollection::SaveOneQField(const QFieldMapIterator &it)
{
   mfem::ofgzstream q_field_file(GetFieldFileName(it->first), compression);

   q_field_file.precision(precision);
   (it->second)->Save(q_field_file);
   if (!q_field_file)
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing q-field to file: " << it->first);
   }
}

void DataCollection::SaveField(const std::string &field_name)
{
   FieldMapIterator it = field_map.find(field_name);
   if (it != field_map.end())
   {
      SaveOneField(it);
   }
}

void DataCollection::SaveQField(const std::string &q_field_name)
{
   QFieldMapIterator it = q_field_map.find(q_field_name);
   if (it != q_field_map.end())
   {
      SaveOneQField(it);
   }
}

void DataCollection::DeleteData()
{
   if (own_data) { delete mesh; }
   mesh = NULL;

   field_map.DeleteData(own_data);
   q_field_map.DeleteData(own_data);
   own_data = false;
}

void DataCollection::DeleteAll()
{
   DeleteData();
   field_map.clear();
   q_field_map.clear();
}

DataCollection::~DataCollection()
{
   DeleteData();
}


// class VisItDataCollection implementation

void VisItDataCollection::UpdateMeshInfo()
{
   if (mesh)
   {
      spatial_dim = mesh->SpaceDimension();
      topo_dim = mesh->Dimension();
      if (mesh->NURBSext)
      {
         visit_levels_of_detail =
            std::max(visit_levels_of_detail, mesh->NURBSext->GetOrder());
      }
   }
   else
   {
      spatial_dim = 0;
      topo_dim = 0;
   }
}

VisItDataCollection::VisItDataCollection(const std::string& collection_name,
                                         Mesh *mesh)
   : DataCollection(collection_name, mesh)
{
   appendRankToFileName = true; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names

   visit_levels_of_detail = 1;
   visit_max_levels_of_detail = 32;

   UpdateMeshInfo();
}

#ifdef MFEM_USE_MPI
VisItDataCollection::VisItDataCollection(MPI_Comm comm,
                                         const std::string& collection_name,
                                         Mesh *mesh)
   : DataCollection(collection_name, mesh)
{
   m_comm = comm;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
   appendRankToFileName = true; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names

   visit_levels_of_detail = 1;
   visit_max_levels_of_detail = 32;

   UpdateMeshInfo();
}
#endif

void VisItDataCollection::SetMesh(Mesh *new_mesh)
{
   DataCollection::SetMesh(new_mesh);
   appendRankToFileName = true;
   UpdateMeshInfo();
}

#ifdef MFEM_USE_MPI
void VisItDataCollection::SetMesh(MPI_Comm comm, Mesh *new_mesh)
{
   // use VisItDataCollection's custom SetMesh, then set MPI info
   SetMesh(new_mesh);
   m_comm = comm;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
}
#endif

void VisItDataCollection::RegisterField(const std::string& name,
                                        GridFunction *gf)
{
   int LOD = 1;
   if (gf->FESpace()->GetNURBSext())
   {
      LOD = gf->FESpace()->GetNURBSext()->GetOrder();
   }
   else
   {
      for (int e=0; e<gf->FESpace()->GetNE(); e++)
      {
         LOD = std::max(LOD,gf->FESpace()->GetFE(e)->GetOrder());
      }
   }

   DataCollection::RegisterField(name, gf);
   field_info_map[name] = VisItFieldInfo("nodes", gf->VectorDim(), LOD);
   visit_levels_of_detail = std::max(visit_levels_of_detail, LOD);
}

void VisItDataCollection::RegisterQField(const std::string& name,
                                         QuadratureFunction *qf)
{
   int LOD = -1;
   Mesh *mesh = qf->GetSpace()->GetMesh();
   for (int e=0; e<qf->GetSpace()->GetNE(); e++)
   {
      int locLOD = GlobGeometryRefiner.GetRefinementLevelFromElems(
                      mesh->GetElementBaseGeometry(e),
                      qf->GetElementIntRule(e).GetNPoints());

      LOD = std::max(LOD,locLOD);
   }

   DataCollection::RegisterQField(name, qf);
   field_info_map[name] = VisItFieldInfo("elements", 1, LOD);
   visit_levels_of_detail = std::max(visit_levels_of_detail, LOD);
}

void VisItDataCollection::SetLevelsOfDetail(int levels_of_detail)
{
   visit_levels_of_detail = levels_of_detail;
}

void VisItDataCollection::SetMaxLevelsOfDetail(int max_levels_of_detail)
{
   visit_max_levels_of_detail = max_levels_of_detail;
}

void VisItDataCollection::DeleteAll()
{
   field_info_map.clear();
   DataCollection::DeleteAll();
}

void VisItDataCollection::Save()
{
   DataCollection::Save();
   SaveRootFile();
}

void VisItDataCollection::SaveRootFile()
{
   if (myid != 0) { return; }

   std::string root_name = prefix_path + name + "_" +
                           to_padded_string(cycle, pad_digits_cycle) +
                           ".mfem_root";
   std::ofstream root_file(root_name.c_str());
   root_file << GetVisItRootString();
   if (!root_file)
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing VisIt root file: " << root_name);
   }
}

void VisItDataCollection::Load(int cycle_)
{
   DeleteAll();
   time_step = 0.0;
   error = NO_ERROR;
   cycle = cycle_;
   std::string root_name = prefix_path + name + "_" +
                           to_padded_string(cycle, pad_digits_cycle) +
                           ".mfem_root";
   LoadVisItRootFile(root_name);
   if (format != SERIAL_FORMAT || num_procs > 1)
   {
#ifndef MFEM_USE_MPI
      MFEM_WARNING("Cannot load parallel VisIt root file in serial.");
      error = READ_ERROR;
#else
      if (m_comm == MPI_COMM_NULL)
      {
         MFEM_WARNING("Cannot load parallel VisIt root file without MPI"
                      " communicator");
         error = READ_ERROR;
      }
      else
      {
         // num_procs was read from the root file, check for consistency with
         // the associated MPI_Comm, m_comm:
         int comm_size;
         MPI_Comm_size(m_comm, &comm_size);
         if (comm_size != num_procs)
         {
            MFEM_WARNING("Processor number mismatch: VisIt root file: "
                         << num_procs << ", MPI_comm: " << comm_size);
            error = READ_ERROR;
         }
         else
         {
            // myid was set when setting m_comm
         }
      }
#endif
   }
   if (!error)
   {
      LoadMesh(); // sets own_data to true, when there is no error
   }
   if (!error)
   {
      LoadFields();
   }
   if (error)
   {
      DeleteAll();
   }
}

void VisItDataCollection::LoadVisItRootFile(const std::string& root_name)
{
   std::ifstream root_file(root_name.c_str());
   std::stringstream buffer;
   buffer << root_file.rdbuf();
   if (!buffer)
   {
      error = READ_ERROR;
      MFEM_WARNING("Error reading the VisIt root file: " << root_name);
   }
   else
   {
      ParseVisItRootString(buffer.str());
   }
}

void VisItDataCollection::LoadMesh()
{
   // GetMeshFileName() uses 'serial', so we need to set it in advance.
   serial = (format == SERIAL_FORMAT);
   std::string mesh_fname = GetMeshFileName();
   named_ifgzstream file(mesh_fname);
   // TODO: in parallel, check for errors on all processors
   if (!file)
   {
      error = READ_ERROR;
      MFEM_WARNING("Unable to open mesh file: " << mesh_fname);
      return;
   }
   // TODO: 1) load parallel mesh on one processor
   if (format == SERIAL_FORMAT)
   {
      mesh = new Mesh(file, 1, 0, false);
      serial = true;
   }
   else
   {
#ifdef MFEM_USE_MPI
      mesh = new ParMesh(m_comm, file);
      serial = false;
#else
      error = READ_ERROR;
      MFEM_WARNING("Reading parallel format in serial is not supported");
      return;
#endif
   }
   spatial_dim = mesh->SpaceDimension();
   topo_dim = mesh->Dimension();
   own_data = true;
}

void VisItDataCollection::LoadFields()
{
   std::string path_left = prefix_path + name + "_" +
                           to_padded_string(cycle, pad_digits_cycle) + "/";
   std::string path_right = "." + to_padded_string(myid, pad_digits_rank);

   field_map.clear();
   for (FieldInfoMapIterator it = field_info_map.begin();
        it != field_info_map.end(); ++it)
   {
      std::string fname = path_left + it->first + path_right;
      mfem::ifgzstream file(fname);
      // TODO: in parallel, check for errors on all processors
      if (!file)
      {
         error = READ_ERROR;
         MFEM_WARNING("Unable to open field file: " << fname);
         return;
      }
      // TODO: 1) load parallel GridFunction on one processor
      if (serial)
      {
         if ((it->second).association == "nodes")
         {
            field_map.Register(it->first, new GridFunction(mesh, file), own_data);
         }
         else if ((it->second).association == "elements")
         {
            q_field_map.Register(it->first, new QuadratureFunction(mesh, file), own_data);
         }
      }
      else
      {
#ifdef MFEM_USE_MPI
         if ((it->second).association == "nodes")
         {
            field_map.Register(
               it->first,
               new ParGridFunction(dynamic_cast<ParMesh*>(mesh), file), own_data);
         }
         else if ((it->second).association == "elements")
         {
            q_field_map.Register(it->first, new QuadratureFunction(mesh, file), own_data);
         }
#else
         error = READ_ERROR;
         MFEM_WARNING("Reading parallel format in serial is not supported");
         return;
#endif
      }
   }
}

std::string VisItDataCollection::GetVisItRootString()
{
   // Get the path string (relative to where the root file is, i.e. no prefix).
   std::string path_str =
      name + "_" + to_padded_string(cycle, pad_digits_cycle) + "/";

   // We have to build the json tree inside out to get all the values in there
   picojson::object top, dsets, main, mesh, fields, field, mtags, ftags;

   // Build the mesh data
   std::string file_ext_format = ".%0" + to_string(pad_digits_rank) + "d";
   mtags["spatial_dim"] = picojson::value(to_string(spatial_dim));
   mtags["topo_dim"] = picojson::value(to_string(topo_dim));
   mtags["max_lods"] = picojson::value(to_string(visit_max_levels_of_detail));
   mesh["path"] = picojson::value(path_str + GetMeshShortFileName() +
                                  file_ext_format);
   mesh["tags"] = picojson::value(mtags);
   mesh["format"] = picojson::value(to_string(format));

   // Build the fields data entries
   for (FieldInfoMapIterator it = field_info_map.begin();
        it != field_info_map.end(); ++it)
   {
      ftags["assoc"] = picojson::value((it->second).association);
      ftags["comps"] = picojson::value(to_string((it->second).num_components));
      ftags["lod"] = picojson::value(to_string((it->second).lod));
      field["path"] = picojson::value(path_str + it->first + file_ext_format);
      field["tags"] = picojson::value(ftags);
      fields[it->first] = picojson::value(field);
   }

   main["cycle"] = picojson::value(double(cycle));
   main["time"] = picojson::value(time);
   main["time_step"] = picojson::value(time_step);
   main["domains"] = picojson::value(double(num_procs));
   main["mesh"] = picojson::value(mesh);
   if (!field_info_map.empty())
   {
      main["fields"] = picojson::value(fields);
   }

   dsets["main"] = picojson::value(main);
   top["dsets"] = picojson::value(dsets);

   return picojson::value(top).serialize(true);
}

void VisItDataCollection::ParseVisItRootString(const std::string& json)
{
   picojson::value top, dsets, main, mesh, fields;
   std::string parse_err = picojson::parse(top, json);
   if (!parse_err.empty())
   {
      error = READ_ERROR;
      MFEM_WARNING("Unable to parse VisIt root data.");
      return;
   }

   // Process "main"
   dsets = top.get("dsets");
   main = dsets.get("main");
   cycle = int(main.get("cycle").get<double>());
   time = main.get("time").get<double>();
   if (main.contains("time_step"))
   {
      time_step = main.get("time_step").get<double>();
   }
   num_procs = int(main.get("domains").get<double>());
   mesh = main.get("mesh");
   fields = main.get("fields");

   // ... Process "mesh"

   // Set the DataCollection::name using the mesh path
   std::string path = mesh.get("path").get<std::string>();
   size_t right_sep = path.find('_');
   if (right_sep == std::string::npos)
   {
      error = READ_ERROR;
      MFEM_WARNING("Unable to parse VisIt root data.");
      return;
   }
   name = path.substr(0, right_sep);

   if (mesh.contains("format"))
   {
      format = to_int(mesh.get("format").get<std::string>());
   }
   spatial_dim = to_int(mesh.get("tags").get("spatial_dim").get<std::string>());
   topo_dim = to_int(mesh.get("tags").get("topo_dim").get<std::string>());
   visit_max_levels_of_detail =
      to_int(mesh.get("tags").get("max_lods").get<std::string>());

   // ... Process "fields"
   field_info_map.clear();
   if (fields.is<picojson::object>())
   {
      picojson::object fields_obj = fields.get<picojson::object>();
      for (picojson::object::iterator it = fields_obj.begin();
           it != fields_obj.end(); ++it)
      {
         picojson::value tags = it->second.get("tags");
         field_info_map[it->first] =
            VisItFieldInfo(tags.get("assoc").get<std::string>(),
                           to_int(tags.get("comps").get<std::string>()));
      }
   }
}

ParaViewDataCollection::ParaViewDataCollection(const std::string&
                                               collection_name,
                                               Mesh *mesh_)
   : DataCollection(collection_name, mesh_),
     levels_of_detail(1),
     pv_data_format(VTKFormat::BINARY),
     high_order_output(false)
{
#ifdef MFEM_USE_ZLIB
   compression = -1; // default zlib compression level, equivalent to 6
#else
   compression = 0;
#endif
}

void ParaViewDataCollection::SetLevelsOfDetail(int levels_of_detail_)
{
   levels_of_detail = levels_of_detail_;
}

void ParaViewDataCollection::Load(int )
{
   MFEM_WARNING("ParaViewDataCollection::Load() is not implemented!");
}

std::string ParaViewDataCollection::GenerateCollectionPath()
{
   std::string out = "";
   out = prefix_path + DataCollection::GetCollectionName();
   return out;
}

std::string ParaViewDataCollection::GeneratePVTUPath()
{
   std::string out = "Cycle" + to_padded_string(cycle,pad_digits_cycle);
   return out;
}

std::string ParaViewDataCollection::GenerateVTUPath()
{
   std::string out = GeneratePVTUPath();
   return out;
}

std::string ParaViewDataCollection::GeneratePVDFileName()
{
   std::string out = GetCollectionName()+".pvd";
   return out;
}

std::string ParaViewDataCollection::GeneratePVTUFileName()
{
   std::string out = "data.pvtu";
   return out;
}

std::string ParaViewDataCollection::GenerateVTUFileName()
{
   std::string out = "proc" + to_padded_string(myid,pad_digits_rank)+".vtu";
   return out;
}
std::string ParaViewDataCollection::GenerateVTUFileName(int crank)
{
   std::string out = "proc" + to_padded_string(crank,pad_digits_rank)+".vtu";
   return out;
}

void ParaViewDataCollection::Save()
{
   // add a new collection to the PDV file

   // check if the directories are created
   {
      std::string path = GenerateCollectionPath()+"/"+GenerateVTUPath();
      int err = create_directory(path, mesh, myid);
      if (err)
      {
         error = WRITE_ERROR;
         MFEM_WARNING("Error creating directory: " << path);
         return; // do not even try to write the mesh
      }
   }
   // the directory is created

   // create pvd file if needed
   if (myid == 0 && !pvd_stream.is_open())
   {
      std::string dpath=GenerateCollectionPath();
      std::string pvdname=dpath+"/"+GeneratePVDFileName();
      pvd_stream.open(pvdname.c_str(),std::ios::out);
      // initialize the file
      pvd_stream << "<?xml version=\"1.0\"?>\n";
      pvd_stream << "<VTKFile type=\"Collection\" version=\"0.1\"";
      pvd_stream << " byte_order=\"" << VTKByteOrder() << "\">\n";
      pvd_stream << "<Collection>" << std::endl;
   }

   // define the vtu file
   {
      std::string fname = GenerateCollectionPath()+"/"+GenerateVTUPath()+"/"
                          +GenerateVTUFileName();
      std::fstream out(fname.c_str(), std::ios::out);
      out.precision(precision);
      SaveDataVTU(out,levels_of_detail);
      out.close();
   }

   // define the pvtu file only on process 0
   if (myid==0)
   {
      std::string fname = GenerateCollectionPath()+"/"+GeneratePVTUPath()+"/"
                          +GeneratePVTUFileName();
      std::fstream out(fname.c_str(), std::ios::out);

      out << "<?xml version=\"1.0\"?>\n";
      out << "<VTKFile type=\"PUnstructuredGrid\"";
      out << " version =\"0.1\" byte_order=\"" << VTKByteOrder() << "\">\n";
      out << "<PUnstructuredGrid GhostLevel=\"0\">\n";

      out << "<PPoints>\n";
      out << "\t<PDataArray type=\"" << GetDataTypeString() << "\" ";
      out << " Name=\"Points\" NumberOfComponents=\"3\""
          << " format=\"" << GetDataFormatString() << "\"/>\n";
      out << "</PPoints>\n";

      out << "<PCells>\n";
      out << "\t<PDataArray type=\"Int32\" ";
      out << " Name=\"connectivity\" NumberOfComponents=\"1\""
          << " format=\"" << GetDataFormatString() << "\"/>\n";
      out << "\t<PDataArray type=\"Int32\" ";
      out << " Name=\"offsets\"      NumberOfComponents=\"1\""
          << " format=\"" << GetDataFormatString() << "\"/>\n";
      out << "\t<PDataArray type=\"UInt8\" ";
      out << " Name=\"types\"        NumberOfComponents=\"1\""
          << " format=\"" << GetDataFormatString() << "\"/>\n";
      out << "</PCells>\n";

      out << "<PPointData>\n";
      for (FieldMapIterator it=field_map.begin(); it!=field_map.end(); ++it)
      {
         int vec_dim=it->second->VectorDim();
         out << "<PDataArray type=\"" << GetDataTypeString()
             << "\" Name=\"" << it->first
             << "\" NumberOfComponents=\"" << vec_dim << "\" "
             << "format=\"" << GetDataFormatString() << "\" />\n";
      }
      out << "</PPointData>\n";

      // CELL DATA
      out << "<PCellData>\n";
      out << "\t<PDataArray type=\"Int32\" Name=\"" << "material"
          << "\" NumberOfComponents=\"1\""
          << " format=\"" << GetDataFormatString() << "\"/>\n";
      out << "</PCellData>\n";

      for (int ii=0; ii<num_procs; ii++)
      {
         // this one is generated without the path
         std::string nfname=GenerateVTUFileName(ii);
         out << "<Piece Source=\"" << nfname << "\"/>\n";
      }
      out << "</PUnstructuredGrid>\n";
      out << "</VTKFile>\n";
      out.close();

      fname = GeneratePVTUPath()+"/"+GeneratePVTUFileName();
      // add the pvtu file to the pvd_stream
      pvd_stream << "<DataSet timestep=\"" << GetTime();  // GetCycle();
      pvd_stream << "\" group=\"\" part=\"" << 0 << "\" file=\"";
      pvd_stream << fname << "\"/>\n";
      std::fstream::pos_type pos = pvd_stream.tellp();
      pvd_stream << "</Collection>\n";
      pvd_stream << "</VTKFile>" << std::endl;
      pvd_stream.seekp(pos);
   }
}

void ParaViewDataCollection::SaveDataVTU(std::ostream &out, int ref)
{
   out << "<VTKFile type=\"UnstructuredGrid\"";
   if (compression != 0)
   {
      out << " compressor=\"vtkZLibDataCompressor\"";
   }
   out << " version=\"0.1\" byte_order=\"" << VTKByteOrder() << "\">\n";
   out << "<UnstructuredGrid>\n";
   mesh->PrintVTU(out,ref,pv_data_format,high_order_output,compression);

   // dump out the grid functions as point data
   out << "<PointData >\n";
   // save the grid functions
   // iterate over all grid functions
   for (FieldMapIterator it=field_map.begin(); it!=field_map.end(); ++it)
   {
      SaveGFieldVTU(out,ref,it);
   }
   // iterate over all quadrature functions
   // if the Quadrature functions are dumped as cell data
   // the cycle should be moved before the grid functions
   // and the PrintVTU CellData section should be open in the mesh dump
   for (QFieldMapIterator it=q_field_map.begin(); it!=q_field_map.end(); ++it)
   {
      // save the quadrature functions
      // this one is not implemented yet
      SaveQFieldVTU(out,ref,it);
   }
   out << "</PointData>\n";
   // close the mesh
   out << "</Piece>\n"; // close the piece open in the PrintVTU method
   out << "</UnstructuredGrid>\n";
   out << "</VTKFile>" << std::endl;
}

void ParaViewDataCollection::SaveQFieldVTU(std::ostream &out, int ref,
                                           const QFieldMapIterator& it )
{
   MFEM_WARNING("SaveQFieldVTU is not currently implemented - field name:"<<it->second);
}

void ParaViewDataCollection::SaveGFieldVTU(std::ostream &out, int ref_,
                                           const FieldMapIterator& it)
{
   RefinedGeometry *RefG;
   Vector val;
   DenseMatrix vval, pmat;
   std::vector<char> buf;
   int vec_dim = it->second->VectorDim();
   if (vec_dim == 1)
   {
      // scalar data
      out << "<DataArray type=\"" << GetDataTypeString()
          << "\" Name=\"" << it->first;
      out << "\" NumberOfComponents=\"1\" format=\""
          << GetDataFormatString() << "\" >\n";
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref_, 1);
         it->second->GetValues(i, RefG->RefPts, val, pmat);
         for (int j = 0; j < val.Size(); j++)
         {
            if (pv_data_format == VTKFormat::ASCII)
            {
               out << val(j) << '\n';
            }
            else if (pv_data_format == VTKFormat::BINARY)
            {
               bin_io::AppendBytes(buf, val(j));
            }
            else
            {
               bin_io::AppendBytes<float>(buf, float(val(j)));
            }
         }
      }
   }
   else
   {
      // vector data
      out << "<DataArray type=\"" << GetDataTypeString()
          << "\" Name=\"" << it->first;
      out << "\" NumberOfComponents=\"" << vec_dim << "\""
          << " format=\"" << GetDataFormatString() << "\" >" << '\n';
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref_, 1);

         it->second->GetVectorValues(i, RefG->RefPts, vval, pmat);

         for (int jj = 0; jj < vval.Width(); jj++)
         {
            for (int ii = 0; ii < vval.Height(); ii++)
            {
               if (pv_data_format == VTKFormat::ASCII)
               {
                  out << vval(ii,jj) << ' ';
               }
               else if (pv_data_format == VTKFormat::BINARY)
               {
                  bin_io::AppendBytes(buf, vval(ii,jj));
               }
               else
               {
                  bin_io::AppendBytes<float>(buf, float(vval(ii,jj)));
               }
            }
            if (pv_data_format == VTKFormat::ASCII) { out << '\n'; }
         }
      }
   }

   if (IsBinaryFormat())
   {
      WriteVTKEncodedCompressed(out,buf.data(),buf.size(),compression);
      out << '\n';
   }
   out << "</DataArray>" << std::endl;
}

void ParaViewDataCollection::SetDataFormat(VTKFormat fmt)
{
   pv_data_format = fmt;
}

bool ParaViewDataCollection::IsBinaryFormat() const
{
   return pv_data_format != VTKFormat::ASCII;
}

void ParaViewDataCollection::SetHighOrderOutput(bool high_order_output_)
{
   high_order_output = high_order_output_;
}

void ParaViewDataCollection::SetCompressionLevel(int compression_level_)
{
   MFEM_ASSERT(compression_level_ >= -1 && compression_level_ <= 9,
               "Compression level must be between -1 and 9 (inclusive).");
   compression = compression_level_;
}

void ParaViewDataCollection::SetCompression(bool compression_)
{
   // If we are enabling compression, and it was disabled previously, use the
   // default compression level. Otherwise, leave the compression level
   // unchanged.
   if (compression_ && compression == 0)
   {
      SetCompressionLevel(-1);
   }
}

const char *ParaViewDataCollection::GetDataFormatString() const
{
   if (pv_data_format == VTKFormat::ASCII)
   {
      return "ascii";
   }
   else
   {
      return "binary";
   }
}

const char *ParaViewDataCollection::GetDataTypeString() const
{
   if (pv_data_format==VTKFormat::ASCII || pv_data_format==VTKFormat::BINARY)
   {
      return "Float64";
   }
   else
   {
      return "Float32";
   }
}

}  // end namespace MFEM
