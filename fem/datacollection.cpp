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

// useful macro for comparing HDF5 versions
#define HDF5_VERSION_GE(Maj,Min,Rel)  \
        (((H5_VERS_MAJOR==Maj) && (H5_VERS_MINOR==Min) && (H5_VERS_RELEASE>=Rel)) || \
         ((H5_VERS_MAJOR==Maj) && (H5_VERS_MINOR>Min)) || \
         (H5_VERS_MAJOR>Maj))

Hdf5ZfpDataCollection::Hdf5ZfpDataCollection(const std::string &name, Mesh *mesh)
   : DataCollection(name, mesh)
{
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
        if (d2size == -1)
            return veca; // C++-11 and RVO in most compilers ==> !copy
        else
            return vecb; // C++-11 and RVO in most compilers ==> !copy
    }
    return vecb; // C++-11 and RVO in most compilers ==> !copy
}

// Convenience functions for templetizing HDF5 type constants
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

// Write a vector to HDF5 as either 1D or 2D depending on d2size
template <class T> static void
WriteVecToHDF5(hid_t fid, char const *name, std::vector<T> const &vec,
    int d2size, int attrA, int attrB)
{
    hsize_t d2s = d2size>1?d2size:1;
    hsize_t siz2d[2] = {(hsize_t) vec.size() / d2s, d2s};
    hid_t spid = H5Screate_simple(d2size>1?2:1, siz2d, 0);
#if HDF5_VERSION_GE(1,8,0)
    hid_t dsid = H5Dcreate(fid, name, HDF5Type<T>(), spid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#else
    hid_t dsid = H5Dcreate(fid, name, HDF5Type<T>(), spid, H5P_DEFAULT);
#endif
    if (attrA != -1)
        WriteIntAttrToHDF5Dataset(dsid, "elem_tag", attrA);
    if (attrB != -1)
        WriteIntAttrToHDF5Dataset(dsid, "elem_typ", attrB);
    H5Dwrite(dsid, HDF5Type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &vec[0]);
    H5Dclose(dsid);
    H5Sclose(spid);
}

void
Hdf5ZfpDataCollection::SaveMfemStringStreamToHDF5(
    hid_t fid, std::string const &name, std::stringstream &strstrm)
{
    std::string const &str = strstrm.str();
    hsize_t len = (hsize_t) str.size();
    hid_t spid = H5Screate_simple(1, &len, 0);
    int ndims = 0, nels = 0, nbnd = 0, nverts = 0, vertsize = 0;
    int eltyp;
    char bname[512];

    strncpy(bname, name.c_str(), sizeof(bname));
    bname[511] = '\0';

    if (name == "mesh")
    {
        size_t n;

        // read dimension
        n = str.find("\ndimension\n");
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
        n = str.find("\nelements\n", n);
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
        int d2size, etag, etyp;
        std::vector<int> topodata =
            GetLinesToVec<int>(strstrm, nels, 2, 1, d2size, etag, etyp);
        WriteVecToHDF5<int>(fid, (name+"_topo").c_str(), topodata, d2size, etag, etyp);
        topodata.clear(); // get rid of bulk data

#if 0
        bool nozfp = false;
        if (elmap[2] || elmap[4]) // no tris or tets
            nozfp = true;
        int emsum = 0;
        for (int i = 0; i < (int) sizeof(elmap)/sizeof(elmap[0]); emsum+=elmap[i], i++);
        if (emsum > 1) // only one elem type
            nozfp = true;
        if (nozfp) 
            MFEM_WARNING("ZFP compression not possible on this mesh.");
#endif

        // read boundary info
        strstrm.clear();
        n = strstrm.tellg();
        n = str.find("\nboundary\n", n);
        if (n == str.npos)
        {
            error = WRITE_ERROR;
            MFEM_WARNING("Error finding boundary section.");
            return;
        }
        strstrm.seekg(n+10);
        strstrm >> nbnd;

        // Get nbnd lines of boundary topology data and write to HDF5
        std::vector<int> bnd_topodata =
            GetLinesToVec<int>(strstrm, nbnd, 2, 1, d2size, etag, etyp);
        WriteVecToHDF5<int>(fid, ("bnd_"+name+"_topo").c_str(), bnd_topodata, d2size, etag, etyp);
        bnd_topodata.clear(); // get rid of bulk data
        strstrm.clear();
        n = strstrm.tellg();

        // read vertices info
        n = str.find("\nvertices\n", n);
        if (n == str.npos)
        {
            error = WRITE_ERROR;
            MFEM_WARNING("Error finding vertex section.");
            return;
        }
        strstrm.seekg(n+10);
        strstrm >> nverts;
        strstrm >> vertsize;

        // Query stream's precision and float format to determine if
        // intention is float or double precision 
        bool isDouble = false;
        std::ios_base::fmtflags ff = strstrm.flags();
        int ndigits = (int) strstrm.precision();
        if (!(ff & std::ios_base::fixed) && ndigits > 7)
            isDouble = true;
        if (isDouble)
        {
            // Get nverts lines of vertex data and write to HDF5 as double
            std::vector<double> vertdata =
                GetLinesToVec<double>(strstrm, nverts, vertsize, -1, d2size, etag, etyp);
            WriteVecToHDF5<double>(fid, "vertices", vertdata, vertsize, etag, etyp);
            vertdata.clear(); // get rid of bulk data
        }
        else
        {
            // Get nverts lines of vertex data and write to HDF5 as float
            std::vector<float> vertdata =
                GetLinesToVec<float>(strstrm, nverts, vertsize, -1, d2size, etag, etyp);
            WriteVecToHDF5<float>(fid, "vertices", vertdata, vertsize, etag, etyp);
            vertdata.clear(); // get rid of bulk data
        }
    }

#if HDF5_VERSION_GE(1,8,0)
    hid_t dsid = H5Dcreate(fid, basename(bname), H5T_NATIVE_CHAR, spid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#else
    hid_t dsid = H5Dcreate(fid, basename(bname), H5T_NATIVE_CHAR, spid, H5P_DEFAULT);
#endif
    H5Dwrite(dsid, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, &str[0]);
    H5Dclose(dsid);
    H5Sclose(spid);
}

void Hdf5ZfpDataCollection::Save()
{
#if 0
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

   std::string file_name = dir_name +
                           ((serial || format == 0 )? "/mfem" : "/pmfem");
   if (appendRankToFileName)
   {
      file_name += "." + to_padded_string(myid, pad_digits_rank) + ".h5";
   }

   // Here is where decide to create file in memory if 
   // wanna gather many outputs to single writer
   fid = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

   SaveMesh(fid);
   if (error) return;

   SaveFields(fid);
#else

   // Save mesh like we ordinarily would except to an ostring stream.
   // Would be best to use a compressed stream here. But, gzstream is for
   // *file* streams and won't work on string streams alone.
   std::string dir_name;
   std::stringstream strstrm;
   SaveMesh(&dir_name, &strstrm);
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
   strstrm.str("");

   for (FieldMapIterator it = field_map.begin(); it != field_map.end(); ++it)
   {
      SaveOneField(it, &strstrm);
      SaveMfemStringStreamToHDF5(fid, GetFieldFileName(it->first), strstrm);
      strstrm.str("");
      // Even if there is an error, try saving the other fields
   }

   for (QFieldMapIterator it = q_field_map.begin(); it != q_field_map.end();
        ++it)
   {
      SaveOneQField(it, &strstrm);
      SaveMfemStringStreamToHDF5(fid, GetFieldFileName(it->first), strstrm);
      strstrm.str("");
   }

   H5Fclose(fid);

#endif
}

void Hdf5ZfpDataCollection::Load(int cycle)
{
}
#endif // MFEM_USE_HDF5

}  // end namespace MFEM
