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

#include "fem.hpp"
#include "../general/text.hpp"
#include "picojson.h"

#include <fstream>
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
   name = collection_name;
   // leave prefix_path empty
   mesh = mesh_;
   myid = 0;
   num_procs = 1;
   serial = true;
   appendRankToFileName = false;

#ifdef MFEM_USE_MPI
   ParMesh *par_mesh = dynamic_cast<ParMesh*>(mesh);
   if (par_mesh)
   {
      myid = par_mesh->GetMyRank();
      num_procs = par_mesh->GetNRanks();
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
   format = 0; // use older serial mesh format
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
   ParMesh *par_mesh = dynamic_cast<ParMesh*>(mesh);
   if (par_mesh)
   {
      myid = par_mesh->GetMyRank();
      num_procs = par_mesh->GetNRanks();
      serial = false;
      appendRankToFileName = true;
   }
#endif
}

void DataCollection::RegisterField(const std::string& name, GridFunction *gf)
{
   GridFunction *&ref = field_map[name];
   if (own_data)
   {
      delete ref; // if newly allocated -> ref is null -> OK
   }
   ref = gf;
}

void DataCollection::DeregisterField(const std::string& name)
{
   FieldMapIterator it = field_map.find(name);
   if (it != field_map.end())
   {
      if (own_data)
      {
         delete it->second;
      }
      field_map.erase(it);
   }
}

void DataCollection::RegisterQField(const std::string& q_field_name,
                                    QuadratureFunction *qf)
{
   QuadratureFunction *&ref = q_field_map[q_field_name];
   if (own_data)
   {
      delete ref; // if newly allocated -> ref is null -> OK
   }
   ref = qf;
}

void DataCollection::DeregisterQField(const std::string& name)
{
   QFieldMapIterator it = q_field_map.find(name);
   if (it != q_field_map.end())
   {
      if (own_data)
      {
         delete it->second;
      }
      q_field_map.erase(it);
   }
}

GridFunction *DataCollection::GetField(const std::string& field_name)
{
   FieldMapConstIterator it = field_map.find(field_name);

   return (it != field_map.end()) ? it->second : NULL;
}

QuadratureFunction *DataCollection::GetQField(const std::string& q_field_name)
{
   QFieldMapConstIterator it = q_field_map.find(q_field_name);

   return (it != q_field_map.end()) ? it->second : NULL;
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

void DataCollection::SaveMesh(std::string *_dir_name, std::stringstream *_strstrm)
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
   std::string mesh_name = dir_name +
                           ((serial || format == 0 )? "/mesh" : "/pmesh");
   if (appendRankToFileName)
   {
      mesh_name += "." + to_padded_string(myid, pad_digits_rank);
   }

   std::ostream *mesh_file = _strstrm;
   if (!mesh_file)
   {
       mesh_file = new std::ofstream(mesh_name.c_str());
       if (!mesh_file || !mesh_file->good())
       {
          error = WRITE_ERROR;
          MFEM_WARNING("Error writing mesh to file: " << mesh_name);
          return;
       }
   }
   mesh_file->precision(precision);
#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh*>(mesh);
   if (pmesh && format == 1 )
   {
      pmesh->ParPrint(*mesh_file);
   }
   else
#endif
   {
      mesh->Print(*mesh_file);
   }
   if (!mesh_file->good())
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing mesh to file: " << mesh_name);
   }
   if (_dir_name)
       *_dir_name = dir_name;
   if (!_strstrm)
      delete mesh_file;
}

std::string DataCollection::GetFieldFileName(const std::string &field_name)
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

void DataCollection::SaveOneField(const FieldMapIterator &it, std::stringstream *_strstrm)
{
   std::ostream *field_file = _strstrm;
   if (!field_file)
       field_file = new std::ofstream(GetFieldFileName(it->first).c_str());
   if (!field_file || !field_file->good())
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing field to file: " << it->first);
      return;
   }
   field_file->precision(precision);
   (it->second)->Save(*field_file);
   if (!field_file->good())
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing field to file: " << it->first);
   }
   if (!_strstrm)
      delete field_file;
}

void DataCollection::SaveOneQField(const QFieldMapIterator &it, std::stringstream *_strstrm)
{
   std::ostream *q_field_file = _strstrm;
   if (!q_field_file)
      q_field_file = new std::ofstream(GetFieldFileName(it->first).c_str());
   if (!q_field_file || !q_field_file->good())
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing q-field to file: " << it->first);
      return;
   }
   q_field_file->precision(precision);
   (it->second)->Save(*q_field_file);
   if (!q_field_file->good())
   {
      error = WRITE_ERROR;
      MFEM_WARNING("Error writing q-field to file: " << it->first);
   }
   if (!_strstrm)
      delete q_field_file;
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

   for (FieldMapIterator it = field_map.begin(); it != field_map.end(); ++it)
   {
      if (own_data) { delete it->second; }
      it->second = NULL;
   }
   for (QFieldMapIterator it = q_field_map.begin();
        it != q_field_map.end(); ++it)
   {
      if (own_data) { delete it->second; }
      it->second = NULL;
   }
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

VisItDataCollection::VisItDataCollection(const std::string& collection_name,
                                         Mesh *mesh)
   : DataCollection(collection_name, mesh)
{
   appendRankToFileName = true; // always include rank in file names
   cycle = 0;                   // always include cycle in directory names

   if (mesh)
   {
      spatial_dim = mesh->SpaceDimension();
      topo_dim = mesh->Dimension();
   }
   else
   {
      spatial_dim = 0;
      topo_dim = 0;
   }
   visit_max_levels_of_detail = 32;
}

void VisItDataCollection::SetMesh(Mesh *new_mesh)
{
   DataCollection::SetMesh(new_mesh);
   appendRankToFileName = true;
   spatial_dim = mesh->SpaceDimension();
   topo_dim = mesh->Dimension();
}

void VisItDataCollection::RegisterField(const std::string& name,
                                        GridFunction *gf)
{
   DataCollection::RegisterField(name, gf);
   field_info_map[name] = VisItFieldInfo("nodes", gf->VectorDim());
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
      MFEM_WARNING("Error writting VisIt Root file: " << root_name);
   }
}

void VisItDataCollection::Load(int cycle_)
{
   DeleteAll();
   cycle = cycle_;
   std::string root_name = prefix_path + name + "_" +
                           to_padded_string(cycle, pad_digits_cycle) +
                           ".mfem_root";
   LoadVisItRootFile(root_name);
   if (!error)
   {
      LoadMesh();
   }
   if (!error)
   {
      LoadFields();
   }
   if (!error)
   {
      own_data = true;
   }
   else
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
      MFEM_WARNING("Error reading the VisIt Root file: " << root_name);
   }
   else
   {
      ParseVisItRootString(buffer.str());
   }
}

void VisItDataCollection::LoadMesh()
{
   std::string mesh_fname = prefix_path + name + "_" +
                            to_padded_string(cycle, pad_digits_cycle) +
                            "/mesh." + to_padded_string(myid, pad_digits_rank);
   named_ifgzstream file(mesh_fname.c_str());
   if (!file)
   {
      error = READ_ERROR;
      MFEM_WARNING("Unable to open mesh file: " << mesh_fname);
      return;
   }
   // TODO: 1) load parallel mesh on one processor
   //       2) load parallel mesh on the same number of processors
   mesh = new Mesh(file, 1, 1);
   spatial_dim = mesh->SpaceDimension();
   topo_dim = mesh->Dimension();
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
      std::ifstream file(fname.c_str());
      if (!file)
      {
         error = READ_ERROR;
         MFEM_WARNING("Unable to open field file: " << fname);
         return;
      }
      // TODO: 1) load parallel GridFunction on one processor
      //       2) load parallel GridFunction on the same number of processors
      field_map[it->first] = new GridFunction(mesh, file);
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
   mesh["path"] = picojson::value(path_str + ((format==0)?"":"p") + "mesh" +
                                  file_ext_format);
   mesh["tags"] = picojson::value(mtags);

   // Build the fields data entries
   for (FieldInfoMapIterator it = field_info_map.begin();
        it != field_info_map.end(); ++it)
   {
      ftags["assoc"] = picojson::value((it->second).association);
      ftags["comps"] = picojson::value(to_string((it->second).num_components));
      field["path"] = picojson::value(path_str + it->first + file_ext_format);
      field["tags"] = picojson::value(ftags);
      fields[it->first] = picojson::value(field);
   }

   main["cycle"] = picojson::value(double(cycle));
   main["time"] = picojson::value(time);
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
      MFEM_WARNING("Unable to parse visit root data.");
      return;
   }

   // Process "main"
   dsets = top.get("dsets");
   main = dsets.get("main");
   cycle = int(main.get("cycle").get<double>());
   time = main.get("time").get<double>();
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
      MFEM_WARNING("Unable to parse visit root data.");
      return;
   }
   name = path.substr(0, right_sep);

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

#ifdef MFEM_USE_HDF5

#include <hdf5.h>
#include <libgen.h> // for basename, may relplace with find_last_of('/')

#include <vector>

#include "H5Zzfp_plugin.h"

// useful macro for comparing HDF5 versions
#define HDF5_VERSION_GE(Maj,Min,Rel)  \
        (((H5_VERS_MAJOR==Maj) && (H5_VERS_MINOR==Min) && (H5_VERS_RELEASE>=Rel)) || \
         ((H5_VERS_MAJOR==Maj) && (H5_VERS_MINOR>Min)) || \
         (H5_VERS_MAJOR>Maj))

static int const elsiz[6] = {1,2,3,4,4,8};

Hdf5ZfpDataCollection::Hdf5ZfpDataCollection(const std::string &name, Mesh *mesh,
    zfp_config_t const *_zfpconfig)
   : DataCollection(name, mesh)
{
    if (_zfpconfig)
        zfpconfig = *((zfp_config_t*)_zfpconfig);
}

// Read either fixed or variable length lines of data from stringstream
// and populate a vector of appropriate type
template <class T> static std::vector<T>
GetLinesToVec(std::stringstream &strm, int nlines, int nfixed, int sizer,
    int &d2size, int &etag, int &etyp)
{
    int const elsiz[6] = {1,2,3,4,4,8};
    std::vector<T> veca; // all data
    std::vector<T> vecb; // nfixed data stripped
    d2size = etag = etyp = -1;
    for (int l = 0; l < nlines; l++)
    {
        int size = 0;
        for (int f = 0; f < nfixed; f++)
        {
            T val;
            strm >> val;
            if ((f == 0 && etag != -1 && etag != val) ||
                (f == 1 && etyp != -1 && etyp != val))
                etag = etyp = -1;
            veca.push_back(val);
            if (sizer >= 0)
            {
                if (f == sizer)
                    size = elsiz[(int)val];
            }
            else
            {
                vecb.push_back(val);
            }
        }
        for (int v = 0; v < size; v++)
        {
            T val;
            strm >> val;
            veca.push_back(val);
            vecb.push_back(val);
        }
        if (l == 0)
        {
            d2size = size;
            etag = veca[0];
            etyp = veca[1];
        }
        else
        {
            if (d2size != size)
                d2size = etag = etyp = -1;
        }
    }
    if (sizer >= 0)
    {
        if (d2size == -1 || etag == -1 || etyp == -1)
            return veca; // C++-11 and RVO in most compilers ==> !copy
        else
            return vecb; // C++-11 and RVO in most compilers ==> !copy
    }
    if (etag == -1 || etyp == -1)
        return veca; // C++-11 and RVO in most compilers ==> !copy
    else
        return vecb; // C++-11 and RVO in most compilers ==> !copy
}

// Convenience functions for templetizing HDF5 type constants
inline hid_t HDF5Type(const char &) { return H5T_NATIVE_CHAR; }
inline hid_t HDF5Type(const int &) { return H5T_NATIVE_INT; }
inline hid_t HDF5Type(const float &) { return H5T_NATIVE_FLOAT; }
inline hid_t HDF5Type(const double &) { return H5T_NATIVE_DOUBLE; }
template <typename T>
inline hid_t HDF5Type() { return HDF5Type(T()); }

static void
WriteIntAttrToHDF5Dataset(hid_t dsid, char const *name, int attr)
{
    hsize_t const dims = 1;
    hid_t aspid = H5Screate_simple(1, &dims, 0);
    hid_t aid = H5Acreate(dsid, name, H5T_NATIVE_INT, aspid, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(aid, H5T_NATIVE_INT, &attr);
    H5Sclose(aspid);
    H5Aclose(aid);
}

template <class T> static std::vector<T>
ReadVecFromHDF5(hid_t fid, char const *name)
{
    std::vector<T> retval;

#if HDF5_VERSION_GE(1,8,0)
    hid_t dsid = H5Dopen(fid, name, H5P_DEFAULT);
#else
    hid_t dsid = H5Dopen(fid, name);
#endif

    hid_t spid = H5Dget_space(dsid);
    hssize_t npts = H5Sget_simple_extent_npoints(spid);
    retval.resize((size_t) npts);

    H5Dread(dsid, HDF5Type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &retval[0]);

    H5Sclose(spid);
    H5Dclose(dsid);

    return retval; // C++-11 and RVO in most compilers ==> !copy
}

// Write a vector to HDF5 as either 1D or 2D depending on d2size
template <class T> static void
WriteVecToHDF5(hid_t fid, char const *name, const T *vec,
    int size, int d2size, int attrA=-1, int attrB=-1,
    Hdf5ZfpDataCollection::zfp_config_t const *zfpconfig = 0)
{
    hsize_t d2s = d2size>1?d2size:1;
    hsize_t siz2d[2] = {(hsize_t) size / d2s, d2s};
    hid_t spid;
    unsigned int cd_values[10];
    int cd_nelmts = 10;

    hid_t cpid = H5Pcreate(H5P_DATASET_CREATE);

    if (zfpconfig && zfpconfig->zfpmode > 0)
    {
        if (zfpconfig->chunk[5])
        {
            hsize_t chunk[6] = {(hsize_t)zfpconfig->chunk[0],(hsize_t)zfpconfig->chunk[1],
                                (hsize_t)zfpconfig->chunk[2],(hsize_t)zfpconfig->chunk[3],
                                (hsize_t)zfpconfig->chunk[4],(hsize_t)zfpconfig->chunk[5]};
            H5Pset_chunk(cpid, 5, chunk);
        }
        else if (zfpconfig->chunk[4])
        {
            hsize_t chunk[5] = {(hsize_t)zfpconfig->chunk[0],(hsize_t)zfpconfig->chunk[1],
                                (hsize_t)zfpconfig->chunk[2],(hsize_t)zfpconfig->chunk[3],
                                (hsize_t)zfpconfig->chunk[4]};
            H5Pset_chunk(cpid, 5, chunk);
        }
        else if (zfpconfig->chunk[3])
        {
            hsize_t chunk[4] = {(hsize_t)zfpconfig->chunk[0],(hsize_t)zfpconfig->chunk[1],
                                (hsize_t)zfpconfig->chunk[2],(hsize_t)zfpconfig->chunk[3]};
            H5Pset_chunk(cpid, 4, chunk);
        }
        else if (zfpconfig->chunk[2])
        {
            hsize_t chunk[3] = {(hsize_t)zfpconfig->chunk[0],(hsize_t)zfpconfig->chunk[1],
                                (hsize_t)zfpconfig->chunk[2]};
            H5Pset_chunk(cpid, 3, chunk);
        }
        else if (zfpconfig->chunk[1])
        {
            hsize_t chunk[2] = {(hsize_t)zfpconfig->chunk[0],(hsize_t)zfpconfig->chunk[1]};
            H5Pset_chunk(cpid, 2, chunk);
        }
        else
        {
            hsize_t chunk[1] = {(hsize_t)zfpconfig->chunk[0]};
            H5Pset_chunk(cpid, 1, chunk);
        }

        // setup zfp filter via generic (cd_values) interface
        if (zfpconfig->zfpmode == H5Z_ZFP_MODE_RATE)
            H5Pset_zfp_rate_cdata(zfpconfig->rate, cd_nelmts, cd_values);
        else if (zfpconfig->zfpmode == H5Z_ZFP_MODE_PRECISION)
            H5Pset_zfp_precision_cdata(zfpconfig->prec, cd_nelmts, cd_values);
        else if (zfpconfig->zfpmode == H5Z_ZFP_MODE_ACCURACY)
            H5Pset_zfp_accuracy_cdata(zfpconfig->acc, cd_nelmts, cd_values);
        else if (zfpconfig->zfpmode == H5Z_ZFP_MODE_EXPERT)
            H5Pset_zfp_expert_cdata(zfpconfig->minbits, zfpconfig->maxbits,
                zfpconfig->maxprec, zfpconfig->minexp, cd_nelmts, cd_values);
        else
            cd_nelmts = 0; // cause default zfp library behavior

        if (d2size && sqrt((double)d2size) == (int) sqrt((double)d2size))
        {
            hsize_t siza = (hsize_t) sqrt((double)d2size);
            hsize_t siz3d[4] = {size/d2size, siza, siza, 2};
            spid = H5Screate_simple(4, siz3d, 0);
        }
        else if (d2size && cbrt((double)d2size) == (int) cbrt((double)d2size))
        {
            hsize_t siza = (hsize_t) cbrt((double)d2size);
            hsize_t siz4d[5] = {size/d2size, siza, siza, siza, 2};
            spid = H5Screate_simple(5, siz4d, 0);
        }
        else
        {
            hsize_t siz2d[3] = {size/d2size, d2size, 2};
            spid = H5Screate_simple(3, siz2d, 0);
        }

        // Add filter to the pipeline via generic interface
        H5Pset_filter(cpid, H5Z_FILTER_ZFP, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values);

    }
    else
    {
        spid = H5Screate_simple(d2size>1?2:1, siz2d, 0);
    }

#if HDF5_VERSION_GE(1,8,0)
    hid_t dsid = H5Dcreate(fid, name, HDF5Type<T>(), spid, H5P_DEFAULT, cpid, H5P_DEFAULT);
#else
    hid_t dsid = H5Dcreate(fid, name, HDF5Type<T>(), spid, cpid);
#endif
    if (attrA != -1)
        WriteIntAttrToHDF5Dataset(dsid, "elem_tag", attrA);
    if (attrB != -1)
        WriteIntAttrToHDF5Dataset(dsid, "elem_typ", attrB);
    H5Dwrite(dsid, HDF5Type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, vec);
    H5Dclose(dsid);
    H5Sclose(spid);
}

// Like a normal string::find() but limit search to maxlen chars
static size_t shortfind(std::string const &str, std::string const &needle,
    size_t start, size_t maxlen=100)
{
    std::string const &shortstr = str.substr(start, maxlen);
    size_t n = shortstr.find(needle);
    if (n == shortstr.npos)
        return str.npos;
    return start + n;
}

void
Hdf5ZfpDataCollection::SaveMesh(hid_t fid, std::string const &name, std::stringstream &strstrm)
{
    std::string const &str = strstrm.str();
    int ndims = 0, nels = 0, nbnd = 0, nbnd2 = 0, nverts = 0, vertsize = 0;
    int femh = 0, femd = 0, femp = 0;
    size_t n;

    // read dimension
    n = shortfind(str, "\ndimension\n", 0, 300);
    if (n == str.npos)
    {
        error = WRITE_ERROR;
        MFEM_WARNING("Error finding dimension.");
        return;
    }
    strstrm.clear();
    strstrm.seekg(n+11);
    strstrm >> ndims;

    // read element count
    n = shortfind(str, "\nelements\n", n);
    if (n == str.npos)
    {
        error = WRITE_ERROR;
        MFEM_WARNING("Error element count.");
        return;
    }
    strstrm.clear();
    strstrm.seekg(n+10);
    strstrm >> nels;

    // Get nels lines of main topology data and write to HDF5
    int d2size, mtag, mtyp;
    std::vector<int> topodata =
        GetLinesToVec<int>(strstrm, nels, 2, 1, d2size, mtag, mtyp);
    WriteVecToHDF5<int>(fid, (name+"_topo").c_str(), &topodata[0],
        (int) topodata.size(), d2size, mtag, mtyp);
    topodata = std::vector<int>(); // deallocate space

    // read boundary info
    strstrm.clear();
    n = strstrm.tellg();
    n = shortfind(str, "\nboundary\n", n);
    if (n == str.npos)
    {
        error = WRITE_ERROR;
        MFEM_WARNING("Error finding boundary section.");
        return;
    }
    strstrm.seekg(n+10);
    strstrm >> nbnd;

    // Get nbnd lines of boundary topology data and write to HDF5
    int btag, btyp;
    std::vector<int> bnd_topodata =
        GetLinesToVec<int>(strstrm, nbnd, 2, 1, d2size, btag, btyp);
    WriteVecToHDF5<int>(fid, ("bnd_"+name+"_topo").c_str(),
        &bnd_topodata[0], (int) bnd_topodata.size(), d2size, btag, btyp);
    bnd_topodata = std::vector<int>(); // deallocate space;
    strstrm.clear();
    n = strstrm.tellg();

    // read vertices info but look ahead for nnodes first
    size_t n1 = shortfind(str, "\nnodes\n", n);
    n = shortfind(str, "\nvertices\n", n);
    if (n == str.npos)
    {
        error = WRITE_ERROR;
        MFEM_WARNING("Error finding vertex section.");
        return;
    }
    strstrm.seekg(n+10);
    strstrm >> nverts;
    if (n1 == str.npos)
        strstrm >> vertsize;
    else
    {
        n = shortfind(str, "\nFiniteElementCollection: ", n);
        if (n == str.npos)
        {
            error = WRITE_ERROR;
            MFEM_WARNING("Error finding nodes FiniteElementCollection: line.");
            return;
        }
        std::string fem_coll_str; 
        strstrm.seekg(n+26);
        strstrm >> fem_coll_str; // e.g. H1_2D_P5
        if (sscanf(fem_coll_str.c_str(), "H%1d_%1dD_P%d", &femh, &femd, &femp) != 3)
        {
            error = WRITE_ERROR;
            MFEM_WARNING("Unable to parse FiniteElementCollection string.");
            return;
        }
        n = shortfind(str, "\nVDim: ", n);
        if (n == str.npos)
        {
            error = WRITE_ERROR;
            MFEM_WARNING("Error finding nodes VDim: line.");
            return;
        }
        strstrm.seekg(n+6);
        strstrm >> vertsize;
        strstrm.seekg(n+20);

        // Count lines remaining in strstrm (trick only works for 2D)
        size_t n2 = n+14;
        int nrem = 0;
        while ((n2 = str.find('\n',n2+1)) != str.npos)
            nrem++;
        nbnd2 = (nrem - 2 - nverts - nels*(femp-1)*(femp-1))/(femp-1);
        strstrm.clear();
        strstrm.seekg(n+20);
    }

    // Query stream's precision and float format to determine if
    // intention is float or double precision 
    bool isDouble = false;
    std::ios_base::fmtflags ff = strstrm.flags();
    int ndigits = (int) strstrm.precision();
    if (!(ff & std::ios_base::fixed) || ndigits > 7)
        isDouble = true;

    if (femh && femd && femp)
    { 
        // We will output to multiple HDF5 datasets, one each for
        // verts, edges, faces and volumes
        std::vector<double> vertdata =
            GetLinesToVec<double>(strstrm, nverts, vertsize, -1, d2size, btag, btyp);
std::cout << "vertices.size() = " << vertdata.size() << std::endl;
        WriteVecToHDF5<double>(fid, "vertices", &vertdata[0], nverts*vertsize, vertsize);

        vertdata =
            GetLinesToVec<double>(strstrm, nbnd2, (femp-1)*vertsize, -1, d2size, btag, btyp);
std::cout << "boundaries.size() = " << vertdata.size() << std::endl;
        WriteVecToHDF5<double>(fid, "boundaries", &vertdata[0], nbnd2*(femp-1)*vertsize, femp-1);

        // Apply ZFP compression only to internal dof data for now
        vertdata =
            GetLinesToVec<double>(strstrm, nels, (femp-1)*(femp-1)*vertsize, -1, d2size, btag, btyp);
std::cout << "elements.size() = " << vertdata.size() << std::endl;
        WriteVecToHDF5<double>(fid, "elements", &vertdata[0], nels*(femp-1)*(femp-1),(femp-1)*(femp-1),-1,-1,&zfpconfig);

        // Ok, lets produce a normal MFEM ascii file from the compressed data
        // as a diagnostic tool
        {
            std::ofstream mesh_file("zfpmesh");
            mesh_file << strstrm.str().substr(0,n+20) << std::endl << std::endl;
            mesh_file << std::setprecision(14);
            char const *names[] = {"vertices","boundaries","elements"};
            for (int n = 0; n < sizeof(names)/sizeof(names[0]); n++)
            {
                std::vector<double> dofs = ReadVecFromHDF5<double>(fid, names[n]);
                std::cout << "dofs.size() = " << dofs.size() << std::endl;
                for (size_t l = 0; l < dofs.size() / vertsize; l++)
                {
                    mesh_file << dofs[vertsize*l+0];
                    for (int c = 1; c < vertsize; c++)
                        mesh_file << " " << dofs[vertsize*l+c];
                    mesh_file << std::endl;
                }
            }
        }
    }
    else if (isDouble)
    {
        // Get nverts lines of vertex data and write to HDF5 as double
        std::vector<double> vertdata =
            GetLinesToVec<double>(strstrm, nverts, vertsize, -1, d2size, btag, btyp);
        WriteVecToHDF5<double>(fid, "vertices", &vertdata[0], nverts, vertsize);
    }
    else
    {
        // Get nverts lines of vertex data and write to HDF5 as float
        std::vector<float> vertdata =
            GetLinesToVec<float>(strstrm, nverts, vertsize, -1, d2size, btag, btyp);
        WriteVecToHDF5<float>(fid, "vertices", &vertdata[0], nverts, vertsize);
    }
}

void
Hdf5ZfpDataCollection::SaveMfemStringStreamToHDF5(
    hid_t fid, std::string const &name, std::stringstream &strstrm)
{
    std::string const &str = strstrm.str();
    int ndims = 0, nels = 0, nbnd = 0, nverts = 0, vertsize = 0;
    int eltyp;
    char bname[512];

    strncpy(bname, name.c_str(), sizeof(bname));
    bname[511] = '\0';

    if (name == "mesh")
        SaveMesh(fid, "mesh", strstrm);

    // Fallback writes as just character data. Will be removed in a future update.
    // WriteVecToHDF5<char>(fid, basename(bname), &str[0], str.size(), 1);
}

void Hdf5ZfpDataCollection::Save()
{
   // Save mesh like we ordinarily would except to an stringstream.
   // Would be best to use a compressed stream here. But, gzstream is for
   // *file* streams and won't work on string streams alone.
   std::string dir_name;
   std::stringstream strstrm;
   DataCollection::SaveMesh(&dir_name, &strstrm);
   if (error) { return; }

   // Create HDF5 File
   std::string file_name = dir_name +
                           ((serial || format == 0 )? "/mfem" : "/pmfem");
   if (appendRankToFileName)
   {
      file_name += "." + to_padded_string(myid, pad_digits_rank) + ".h5";
   }

   hid_t fid = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   SaveMfemStringStreamToHDF5(fid, "mesh", strstrm);
   if (error) { return; }
   strstrm.clear();
   strstrm.str("");

   for (FieldMapIterator it = field_map.begin(); it != field_map.end(); ++it)
   {
      SaveOneField(it, &strstrm);
      SaveMfemStringStreamToHDF5(fid, GetFieldFileName(it->first), strstrm);
      strstrm.clear();
      strstrm.str("");
      // Even if there is an error, try saving the other fields
   }

   for (QFieldMapIterator it = q_field_map.begin(); it != q_field_map.end();
        ++it)
   {
      SaveOneQField(it, &strstrm);
      SaveMfemStringStreamToHDF5(fid, GetFieldFileName(it->first), strstrm);
      strstrm.clear();
      strstrm.str("");
   }

   H5Fclose(fid);
}

void Hdf5ZfpDataCollection::Load(int cycle)
{
}
#endif // MFEM_USE_HDF5

}  // end namespace MFEM
