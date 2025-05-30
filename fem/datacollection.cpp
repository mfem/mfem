// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include "../mesh/vtk.hpp"
#include "../mesh/vtkhdf.hpp"
#include "../general/binaryio.hpp"
#include "../general/text.hpp"
#include "picojson.h"

#include <cerrno>      // errno
#include <sstream>
#include <regex>

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
   int err_flag;
#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh*>(mesh);
#endif

   do
   {
      pos = dir_name.find(path_delim, pos+1);
      std::string subdir = dir_name.substr(0, pos);

#ifndef MFEM_USE_MPI
      err_flag = mkdir(subdir.c_str(), 0777);
      err_flag = (err_flag && (errno != EEXIST)) ? 1 : 0;
#else
      if (myid == 0 || pmesh == NULL)
      {
         err_flag = mkdir(subdir.c_str(), 0777);
         err_flag = (err_flag && (errno != EEXIST)) ? 1 : 0;
      }
#endif
   }
   while ( pos != std::string::npos );

#ifdef MFEM_USE_MPI
   if (pmesh)
   {
      MPI_Bcast(&err_flag, 1, MPI_INT, 0, pmesh->GetComm());
   }
#endif

   return err_flag;
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
   compression = 0;
   error = No_Error;
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

void DataCollection::Load(int cycle_)
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
   std::string dir_name = prefix_path + name;
   if (cycle != -1)
   {
      dir_name += "_" + to_padded_string(cycle, pad_digits_cycle);
   }
   int error_code = create_directory(dir_name, mesh, myid);
   if (error_code)
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
                      qf->GetIntRule(e).GetNPoints());

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
   std::ofstream root_file(root_name);
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
   error = No_Error;
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
   std::ifstream root_file(root_name);
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
   size_t right_sep = path.rfind('_');
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

ParaViewDataCollectionBase::ParaViewDataCollectionBase(
   const std::string &name, Mesh *mesh) : DataCollection(name, mesh)
{
   cycle = 0;
#ifdef MFEM_USE_ZLIB
   // If we have zlib, enable compression. Otherwise, compression is disabled in
   // the DataCollection base class constructor.
   compression = true;
#endif
}

void ParaViewDataCollectionBase::SetLevelsOfDetail(int levels_of_detail_)
{
   levels_of_detail = levels_of_detail_;
}

void ParaViewDataCollectionBase::SetHighOrderOutput(bool high_order_output_)
{
   high_order_output = high_order_output_;
}

void ParaViewDataCollectionBase::SetCompressionLevel(int compression_level_)
{
   MFEM_ASSERT(compression_level_ >= -1 && compression_level_ <= 9,
               "Compression level must be between -1 and 9 (inclusive).");
   if (compression_level_ != 0) { SetCompression(true);}
   compression_level = compression_level_;
}

int ParaViewDataCollectionBase::GetCompressionLevel() const
{
   return compression ? compression_level : 0;
}

void ParaViewDataCollectionBase::SetDataFormat(VTKFormat fmt)
{
   pv_data_format = fmt;
}

bool ParaViewDataCollectionBase::IsBinaryFormat() const
{
   return pv_data_format != VTKFormat::ASCII;
}

void ParaViewDataCollectionBase::UseRestartMode(bool restart_mode_)
{
   restart_mode = restart_mode_;
}

ParaViewDataCollection::ParaViewDataCollection(
   const std::string& collection_name, Mesh *mesh_)
   : ParaViewDataCollectionBase(collection_name, mesh_) { }

std::string ParaViewDataCollection::GenerateCollectionPath()
{
   return prefix_path + DataCollection::GetCollectionName();
}

std::string ParaViewDataCollection::GeneratePVTUPath()
{
   return "Cycle" + to_padded_string(cycle,pad_digits_cycle);
}

std::string ParaViewDataCollection::GenerateVTUPath()
{
   return GeneratePVTUPath();
}

std::string ParaViewDataCollection::GeneratePVDFileName()
{
   return GetCollectionName() + ".pvd";
}

std::string ParaViewDataCollection::GeneratePVTUFileName(
   const std::string &prefix)
{
   return prefix + ".pvtu";
}

std::string ParaViewDataCollection::GenerateVTUFileName(
   const std::string &prefix, int rank)
{
   return prefix + to_padded_string(rank, pad_digits_rank) + ".vtu";
}

void ParaViewDataCollection::Save()
{
   // add a new collection to the PDV file

   std::string col_path = GenerateCollectionPath();
   // check if the directories are created
   {
      std::string path = col_path + "/" + GenerateVTUPath();
      int error_code = create_directory(path, mesh, myid);
      if (error_code)
      {
         error = WRITE_ERROR;
         MFEM_WARNING("Error creating directory: " << path);
         return; // do not even try to write the mesh
      }
   }
   // the directory is created

   // create pvd file if needed. If we are not in restart mode, a new pvd file
   // is always created. In restart mode, we keep any previously defined
   // timestep values as long as they are less than the currently defined time.

   if (myid == 0 && !pvd_stream.is_open())
   {
      std::string pvdname = col_path + "/" + GeneratePVDFileName();

      bool write_header = true;
      std::ifstream pvd_in;
      if (restart_mode && (pvd_in.open(pvdname,std::ios::binary),pvd_in.good()))
      {
         // PVD file exists and restart mode enabled: preserve existing time
         // steps less than the current time.
         std::fstream::pos_type pos_begin = pvd_in.tellg();
         std::fstream::pos_type pos_end = pos_begin;

         std::regex regexp("timestep=\"([^[:space:]]+)\".*file=\"Cycle(\\d+)");
         std::smatch match;

         std::string line;
         while (getline(pvd_in,line))
         {
            if (regex_search(line,match,regexp))
            {
               MFEM_ASSERT(match.size() == 3, "Unable to parse DataSet");
               double tvalue = std::stod(match[1]);
               if (tvalue >= GetTime()) { break; }
               int cvalue = std::stoi(match[2]);
               MFEM_VERIFY(cvalue < GetCycle(), "Cycle " << GetCycle() <<
                           " is too small for restart mode: trying to overwrite"
                           " existing data.");
               pos_end = pvd_in.tellg();
            }
         }
         // Since pvd_in is opened in binary mode, count will store the number
         // of bytes from the beginning of the file until the desired insertion
         // point (in text mode on Windows this is not the case).
         size_t count = pos_end - pos_begin;
         if (count != 0)
         {
            write_header = false;
            std::vector<char> buf(count);
            // Read the contents of the PVD file, from the beginning to the
            // insertion point.
            pvd_in.clear();
            pvd_in.seekg(pos_begin);
            pvd_in.read(buf.data(), count);
            pvd_in.close();
            // Open the PVD file in truncate mode to delete the previous
            // contents. Open in binary mode to write the data buffer without
            // converting \r\n to \r\r\n on Windows.
            pvd_stream.open(pvdname,std::ios::out|std::ios::trunc|std::ios::binary);
            pvd_stream.write(buf.data(), count);
            // Close and reopen the file in text mode, appending to the end.
            pvd_stream.close();
            pvd_stream.open(pvdname,std::ios::in|std::ios::out|std::ios::ate);
         }
      }
      if (write_header)
      {
         // Initialize new pvd file.
         pvd_stream.open(pvdname,std::ios::out|std::ios::trunc);
         pvd_stream << "<?xml version=\"1.0\"?>\n";
         pvd_stream << "<VTKFile type=\"Collection\" version=\"2.2\"";
         pvd_stream << " byte_order=\"" << VTKByteOrder() << "\">\n";
         pvd_stream << "<Collection>" << std::endl;
      }
   }

   std::string vtu_prefix = col_path + "/" + GenerateVTUPath() + "/";

   // Save the local part of the mesh and grid functions fields to the local
   // VTU file
   {
      std::ofstream os(vtu_prefix + GenerateVTUFileName("proc", myid));
      os.precision(precision);
      SaveDataVTU(os, levels_of_detail);
   }

   // Save the local part of the quadrature function fields
   for (const auto &qfield : q_field_map)
   {
      const std::string &field_name = qfield.first;
      std::ofstream os(vtu_prefix + GenerateVTUFileName(field_name, myid));
      qfield.second->SaveVTU(os, pv_data_format, GetCompressionLevel(), field_name);
   }

   // MPI rank 0 also creates a "PVTU" file that points to all of the separately
   // written VTU files.
   // This file path is then appended to the PVD file.
   if (myid == 0)
   {
      // Create the main PVTU file
      {
         std::ofstream pvtu_out(vtu_prefix + GeneratePVTUFileName("data"));
         WritePVTUHeader(pvtu_out);

         // Grid function fields
         pvtu_out << "<PPointData>\n";
         for (auto &field_it : field_map)
         {
            int vec_dim = field_it.second->VectorDim();
            pvtu_out << "<PDataArray type=\"" << GetDataTypeString()
                     << "\" Name=\"" << field_it.first
                     << "\" NumberOfComponents=\"" << vec_dim << "\" "
                     << VTKComponentLabels(vec_dim) << " "
                     << "format=\"" << GetDataFormatString() << "\" />\n";
         }
         pvtu_out << "</PPointData>\n";
         // Element attributes
         pvtu_out << "<PCellData>\n";
         pvtu_out << "\t<PDataArray type=\"Int32\" Name=\"" << "attribute"
                  << "\" NumberOfComponents=\"1\""
                  << " format=\"" << GetDataFormatString() << "\"/>\n";
         pvtu_out << "</PCellData>\n";

         WritePVTUFooter(pvtu_out, "proc");
      }

      // Add the latest PVTU to the PVD
      pvd_stream << "<DataSet timestep=\"" << GetTime()
                 << "\" group=\"\" part=\"" << 0 << "\" file=\""
                 << GeneratePVTUPath() + "/" + GeneratePVTUFileName("data")
                 << "\" name=\"mesh\"/>\n";

      // Create PVTU files for each quadrature field and add them to the PVD
      // file
      for (auto &q_field : q_field_map)
      {
         const std::string &q_field_name = q_field.first;
         std::string q_fname = GeneratePVTUPath() + "/"
                               + GeneratePVTUFileName(q_field_name);

         std::ofstream pvtu_out(col_path + "/" + q_fname);
         WritePVTUHeader(pvtu_out);
         int vec_dim = q_field.second->GetVDim();
         pvtu_out << "<PPointData>\n";
         pvtu_out << "<PDataArray type=\"" << GetDataTypeString()
                  << "\" Name=\"" << q_field_name
                  << "\" NumberOfComponents=\"" << vec_dim << "\" "
                  << VTKComponentLabels(vec_dim) << " "
                  << "format=\"" << GetDataFormatString() << "\" />\n";
         pvtu_out << "</PPointData>\n";
         WritePVTUFooter(pvtu_out, q_field_name);

         pvd_stream << "<DataSet timestep=\"" << GetTime()
                    << "\" group=\"\" part=\"" << 0 << "\" file=\""
                    << q_fname << "\" name=\"" << q_field_name << "\"/>\n";
      }
      pvd_stream.flush();
      // Move the insertion point before the closing collection tag, so that
      // the PVD file is valid even when writing incrementally.
      std::fstream::pos_type pos = pvd_stream.tellp();
      pvd_stream << "</Collection>\n";
      pvd_stream << "</VTKFile>" << std::endl;
      pvd_stream.seekp(pos);
   }
}

void ParaViewDataCollection::WritePVTUHeader(std::ostream &os)
{
   os << "<?xml version=\"1.0\"?>\n";
   os << "<VTKFile type=\"PUnstructuredGrid\"";
   os << " version =\"2.2\" byte_order=\"" << VTKByteOrder() << "\">\n";
   os << "<PUnstructuredGrid GhostLevel=\"0\">\n";

   os << "<PPoints>\n";
   os << "\t<PDataArray type=\"" << GetDataTypeString() << "\" ";
   os << " Name=\"Points\" NumberOfComponents=\"3\""
      << " format=\"" << GetDataFormatString() << "\"/>\n";
   os << "</PPoints>\n";

   os << "<PCells>\n";
   os << "\t<PDataArray type=\"Int32\" ";
   os << " Name=\"connectivity\" NumberOfComponents=\"1\""
      << " format=\"" << GetDataFormatString() << "\"/>\n";
   os << "\t<PDataArray type=\"Int32\" ";
   os << " Name=\"offsets\"      NumberOfComponents=\"1\""
      << " format=\"" << GetDataFormatString() << "\"/>\n";
   os << "\t<PDataArray type=\"UInt8\" ";
   os << " Name=\"types\"        NumberOfComponents=\"1\""
      << " format=\"" << GetDataFormatString() << "\"/>\n";
   os << "</PCells>\n";
}

void ParaViewDataCollection::WritePVTUFooter(std::ostream &os,
                                             const std::string &vtu_prefix)
{
   for (int ii=0; ii<num_procs; ii++)
   {
      std::string vtu_filename = GenerateVTUFileName(vtu_prefix, ii);
      os << "<Piece Source=\"" << vtu_filename << "\"/>\n";
   }
   os << "</PUnstructuredGrid>\n";
   os << "</VTKFile>\n";
}

void ParaViewDataCollection::SaveDataVTU(std::ostream &os, int ref)
{
   os << "<VTKFile type=\"UnstructuredGrid\"";
   if (GetCompressionLevel() != 0)
   {
      os << " compressor=\"vtkZLibDataCompressor\"";
   }
   os << " version=\"2.2\" byte_order=\"" << VTKByteOrder() << "\">\n";
   os << "<UnstructuredGrid>\n";
   mesh->PrintVTU(os,ref,pv_data_format,high_order_output,GetCompressionLevel());

   // dump out the grid functions as point data
   os << "<PointData >\n";
   // save the grid functions
   // iterate over all grid functions
   for (FieldMapIterator it=field_map.begin(); it!=field_map.end(); ++it)
   {
      SaveGFieldVTU(os,ref,it);
   }
   os << "</PointData>\n";
   // close the mesh
   os << "</Piece>\n"; // close the piece open in the PrintVTU method
   os << "</UnstructuredGrid>\n";
   os << "</VTKFile>" << std::endl;
}

void ParaViewDataCollection::SaveGFieldVTU(std::ostream &os, int ref_,
                                           const FieldMapIterator &it)
{
   RefinedGeometry *RefG;
   Vector val;
   DenseMatrix vval, pmat;
   std::vector<char> buf;
   int vec_dim = it->second->VectorDim();
   os << "<DataArray type=\"" << GetDataTypeString()
      << "\" Name=\"" << it->first
      << "\" NumberOfComponents=\"" << vec_dim << "\" "
      << VTKComponentLabels(vec_dim) << " "
      << "format=\"" << GetDataFormatString() << "\" >" << '\n';
   if (vec_dim == 1)
   {
      // scalar data
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref_, 1);
         it->second->GetValues(i, RefG->RefPts, val, pmat);
         for (int j = 0; j < val.Size(); j++)
         {
            WriteBinaryOrASCII(os, buf, val(j), "\n", pv_data_format);
         }
      }
   }
   else
   {
      // vector data
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref_, 1);
         it->second->GetVectorValues(i, RefG->RefPts, vval, pmat);
         for (int jj = 0; jj < vval.Width(); jj++)
         {
            for (int ii = 0; ii < vval.Height(); ii++)
            {
               WriteBinaryOrASCII(os, buf, vval(ii,jj), " ", pv_data_format);
            }
            if (pv_data_format == VTKFormat::ASCII) { os << '\n'; }
         }
      }
   }

   if (IsBinaryFormat())
   {
      WriteVTKEncodedCompressed(os,buf.data(),buf.size(),GetCompressionLevel());
      os << '\n';
   }
   os << "</DataArray>" << std::endl;
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

#ifdef MFEM_USE_HDF5

ParaViewHDFDataCollection::ParaViewHDFDataCollection(
   const std::string &collection_name, Mesh *mesh)
   : ParaViewDataCollectionBase(collection_name, mesh)
{
   compression = true;
}

void ParaViewHDFDataCollection::SetCompression(bool compression_)
{
   compression = compression_;
}

void ParaViewHDFDataCollection::EnsureVTKHDF()
{
   if (!vtkhdf)
   {
      if (!prefix_path.empty())
      {
         const int error_code = create_directory(prefix_path, mesh, myid);
         MFEM_VERIFY(error_code == 0, "Error creating directory " << prefix_path);
      }

      std::string fname = prefix_path + name + ".vtkhdf";
      bool use_mpi = false;
#ifdef MFEM_USE_MPI
      if (ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh))
      {
         use_mpi = true;
#ifdef MFEM_PARALLEL_HDF5
         vtkhdf.reset(new VTKHDF(fname, pmesh->GetComm(), {restart_mode, time}));
#else
         MFEM_ABORT("Requires HDF5 library with parallel support enabled");
#endif
      }
#endif
      if (!use_mpi)
      {
         vtkhdf.reset(new VTKHDF(fname, {restart_mode, time}));
      }
   }
}

template <typename FP_T>
void ParaViewHDFDataCollection::TSave()
{
   EnsureVTKHDF();

   if (compression)
   {
      vtkhdf->EnableCompression(compression_level >= 0 ? compression_level : 6);
   }
   else
   {
      vtkhdf->DisableCompression();
   }

   vtkhdf->SaveMesh<FP_T>(*mesh, high_order_output, levels_of_detail);
   for (const auto &field : field_map)
   {
      vtkhdf->SaveGridFunction<FP_T>(*field.second, field.first);
   }
   vtkhdf->UpdateSteps(time);
   vtkhdf->Flush();
}

void ParaViewHDFDataCollection::Save()
{
   switch (pv_data_format)
   {
      case VTKFormat::BINARY32: TSave<float>(); break;
      case VTKFormat::BINARY: TSave<double>(); break;
      default: MFEM_ABORT("Unsupported VTK format.");
   }
}

ParaViewHDFDataCollection::~ParaViewHDFDataCollection() = default;

#endif

}  // end namespace MFEM
