// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mesh.hpp"

#include <iomanip>
#include <cerrno>
#include <cstdlib>
#include <cstdint>

#ifdef MFEM_USE_HDF5
#include <hdf5.h>

namespace mfem
{

namespace
{

constexpr char nurbs_hdf5_group[] = "/MFEM/NURBS";
constexpr int nurbs_hdf5_schema_version[] = {1, 0};

/// Values stored in the NURBS HDF5 `token_types` dataset.
enum class NURBSHDF5TokenType : uint8_t
{
   String  = 0,
   Integer = 1,
   Real    = 2,
   Newline = 3
};

class HDF5Handle
{
private:
   hid_t id;
   herr_t (*close_fn)(hid_t);

public:
   HDF5Handle(hid_t id_, herr_t (*close_fn_)(hid_t))
      : id(id_), close_fn(close_fn_) { }
   ~HDF5Handle() { if (id >= 0) { close_fn(id); } }
   HDF5Handle(const HDF5Handle &) = delete;
   HDF5Handle &operator=(const HDF5Handle &) = delete;
   operator hid_t() const { return id; }
};

void CheckHDF5(herr_t status, const char *operation)
{
   MFEM_VERIFY(status >= 0, "HDF5 operation failed: " << operation);
}

template <typename T>
void WriteNURBSHDF5Vector(hid_t group, const char *name,
                          const std::vector<T> &values, hid_t file_type,
                          hid_t memory_type, int compression_level)
{
   const hsize_t size = values.size();
   HDF5Handle space(size ? H5Screate_simple(1, &size, NULL) :
                    H5Screate(H5S_NULL), H5Sclose);
   MFEM_VERIFY(hid_t(space) >= 0,
               "Unable to create HDF5 dataspace for " << name);

   HDF5Handle properties(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
   MFEM_VERIFY(hid_t(properties) >= 0,
               "Unable to create HDF5 properties for " << name);
   if (size && compression_level >= 0 &&
       H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0)
   {
      const hsize_t chunk = std::min<hsize_t>(size, 131072);
      CheckHDF5(H5Pset_chunk(properties, 1, &chunk), "set chunk size");
      if (H5Tget_size(file_type) > 1)
      {
         CheckHDF5(H5Pset_shuffle(properties), "enable shuffle filter");
      }
      CheckHDF5(H5Pset_deflate(properties, compression_level),
                "enable deflate filter");
   }

   HDF5Handle dataset(H5Dcreate2(group, name, file_type, space, H5P_DEFAULT,
                                 properties, H5P_DEFAULT), H5Dclose);
   MFEM_VERIFY(hid_t(dataset) >= 0, "Unable to create HDF5 dataset " << name);
   if (size)
   {
      CheckHDF5(H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                         values.data()), "write dataset");
   }
}

template <typename T>
std::vector<T> ReadNURBSHDF5Vector(hid_t group, const char *name,
                                   hid_t memory_type)
{
   HDF5Handle dataset(H5Dopen2(group, name, H5P_DEFAULT), H5Dclose);
   MFEM_VERIFY(hid_t(dataset) >= 0, "Missing HDF5 dataset " << name);
   HDF5Handle space(H5Dget_space(dataset), H5Sclose);
   MFEM_VERIFY(hid_t(space) >= 0, "Unable to inspect HDF5 dataset " << name);

   const int rank = H5Sget_simple_extent_ndims(space);
   MFEM_VERIFY(rank == 1 || rank == 0,
               "Invalid rank for HDF5 dataset " << name);
   hsize_t size = 0;
   if (rank == 1) { H5Sget_simple_extent_dims(space, &size, NULL); }
   MFEM_VERIFY(size <= std::numeric_limits<size_t>::max(),
               "HDF5 dataset is too large: " << name);
   std::vector<T> values(static_cast<size_t>(size));
   if (size)
   {
      CheckHDF5(H5Dread(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        values.data()), "read dataset");
   }
   return values;
}

} // namespace

bool IsNativeNURBSHDF5(const std::string &filename)
{
   htri_t is_hdf5 = -1;
   H5E_BEGIN_TRY
   {
      is_hdf5 = H5Fis_hdf5(filename.c_str());
   }
   H5E_END_TRY;
   if (is_hdf5 <= 0) { return false; }

   hid_t file_id = -1;
   hid_t group_id = -1;
   htri_t has_schema = 0;
   H5E_BEGIN_TRY
   {
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      if (file_id >= 0)
      {
         group_id = H5Gopen2(file_id, nurbs_hdf5_group, H5P_DEFAULT);
      }
      if (group_id >= 0)
      {
         has_schema = H5Aexists(group_id, "schema_version");
      }
   }
   H5E_END_TRY;
   if (group_id >= 0) { H5Gclose(group_id); }
   if (file_id >= 0) { H5Fclose(file_id); }
   return has_schema > 0;
}

std::string ReadNativeNURBSHDF5(const std::string &filename)
{
   HDF5Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                   H5Fclose);
   MFEM_VERIFY(hid_t(file) >= 0,
               "Unable to open NURBS HDF5 file: " << filename);
   HDF5Handle group(H5Gopen2(file, nurbs_hdf5_group, H5P_DEFAULT), H5Gclose);
   MFEM_VERIFY(hid_t(group) >= 0,
               "Missing " << nurbs_hdf5_group << " in " << filename);

   int schema_version[2] = { 0, 0 };
   HDF5Handle attribute(H5Aopen(group, "schema_version", H5P_DEFAULT),
                        H5Aclose);
   MFEM_VERIFY(hid_t(attribute) >= 0,
               "Missing NURBS HDF5 schema version in " << filename);
   HDF5Handle attribute_space(H5Aget_space(attribute), H5Sclose);
   MFEM_VERIFY(hid_t(attribute_space) >= 0 &&
               H5Sget_simple_extent_npoints(attribute_space) == 2,
               "Invalid NURBS HDF5 schema version in " << filename);
   CheckHDF5(H5Aread(attribute, H5T_NATIVE_INT, schema_version),
             "read schema version");
   MFEM_VERIFY(schema_version[0] == nurbs_hdf5_schema_version[0],
               "Unsupported MFEM NURBS HDF5 schema version "
               << schema_version[0] << '.' << schema_version[1]);

   const std::vector<uint8_t> format = ReadNURBSHDF5Vector<uint8_t>(
                                          group, "format", H5T_NATIVE_UCHAR);
   const std::string header(format.begin(), format.end());
   MFEM_VERIFY(header == "MFEM NURBS mesh v1.0" ||
               header == "MFEM NURBS mesh v1.1" ||
               header == "MFEM NURBS NC-patch mesh v1.0",
               "Invalid NURBS mesh format in " << filename);

   const std::vector<NURBSHDF5TokenType> types =
      ReadNURBSHDF5Vector<NURBSHDF5TokenType>(
         group, "token_types", H5T_NATIVE_UCHAR);
   const std::vector<int64_t> integers = ReadNURBSHDF5Vector<int64_t>(
                                            group, "integers", H5T_NATIVE_INT64);
   const std::vector<double> reals = ReadNURBSHDF5Vector<double>(
                                        group, "reals", H5T_NATIVE_DOUBLE);
   const std::vector<uint64_t> offsets = ReadNURBSHDF5Vector<uint64_t>(
                                            group, "string_offsets", H5T_NATIVE_UINT64);
   const std::vector<uint8_t> strings = ReadNURBSHDF5Vector<uint8_t>(
                                           group, "strings", H5T_NATIVE_UCHAR);

   MFEM_VERIFY(!offsets.empty() && offsets[0] == 0 &&
               offsets.back() == strings.size(),
               "Invalid string offsets in " << filename);

   size_t ni = 0, nr = 0, ns = 0;
   bool line_start = true;
   std::ostringstream text;
   text << header << '\n' << std::setprecision(17);
   for (NURBSHDF5TokenType type : types)
   {
      switch (type)
      {
         case NURBSHDF5TokenType::Newline:
            text << '\n';
            line_start = true;
            continue;
         case NURBSHDF5TokenType::String:
            if (!line_start) { text << ' '; }
            line_start = false;
            MFEM_VERIFY(ns + 1 < offsets.size() &&
                        offsets[ns] <= offsets[ns+1] &&
                        offsets[ns+1] <= strings.size(),
                        "Invalid string table in " << filename);
            text.write(
               reinterpret_cast<const char*>(strings.data() + offsets[ns]),
               offsets[ns+1] - offsets[ns]);
            ns++;
            break;
         case NURBSHDF5TokenType::Integer:
            if (!line_start) { text << ' '; }
            line_start = false;
            MFEM_VERIFY(ni < integers.size(),
                        "Invalid integer token stream in " << filename);
            text << integers[ni++];
            break;
         case NURBSHDF5TokenType::Real:
            if (!line_start) { text << ' '; }
            line_start = false;
            MFEM_VERIFY(nr < reals.size(),
                        "Invalid real token stream in " << filename);
            text << reals[nr++];
            break;
         default:
            MFEM_ABORT("Invalid token type in " << filename);
      }
   }
   if (!line_start) { text << '\n'; }
   MFEM_VERIFY(ni == integers.size() && nr == reals.size() &&
               ns + 1 == offsets.size(),
               "Unused data in NURBS HDF5 token tables in " << filename);
   return text.str();
}

void Mesh::SaveNURBSHDF5(const std::string &fname,
                         int compression_level) const
{
   MFEM_VERIFY(NURBSext, "SaveNURBSHDF5 requires a NURBS mesh");
   MFEM_VERIFY(compression_level >= -1 && compression_level <= 9,
               "HDF5 compression level must be between -1 and 9");

   // Serialize through the native NURBS printer, then split the lexical stream
   // into portable typed arrays. Reusing the native parser on input ensures
   // that this format supports every NURBS section, including future optional
   // sections whose tokens follow the existing text grammar.
   std::ostringstream serialized;
   serialized << std::setprecision(std::numeric_limits<real_t>::max_digits10);
   Print(serialized);

   std::istringstream lines(serialized.str());
   std::string header;
   std::getline(lines, header);
   filter_dos(header);
   MFEM_VERIFY(header == "MFEM NURBS mesh v1.0" ||
               header == "MFEM NURBS mesh v1.1" ||
               header == "MFEM NURBS NC-patch mesh v1.0",
               "Unsupported native NURBS mesh format: " << header);

   std::vector<NURBSHDF5TokenType> types;
   std::vector<int64_t> integers;
   std::vector<double> reals;
   std::vector<uint64_t> string_offsets(1, 0);
   std::vector<uint8_t> strings;
   std::string line, token;
   while (std::getline(lines, line))
   {
      const size_t first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos || line[first] == '#') { continue; }
      std::istringstream tokens(line);
      while (tokens >> token)
      {
         char *end = NULL;
         errno = 0;
         const long long integer = std::strtoll(token.c_str(), &end, 10);
         if (errno == 0 && end == token.c_str() + token.size())
         {
            types.push_back(NURBSHDF5TokenType::Integer);
            integers.push_back(static_cast<int64_t>(integer));
            continue;
         }

         errno = 0;
         const double real = std::strtod(token.c_str(), &end);
         if (errno == 0 && end == token.c_str() + token.size())
         {
            types.push_back(NURBSHDF5TokenType::Real);
            reals.push_back(real);
            continue;
         }

         types.push_back(NURBSHDF5TokenType::String);
         strings.insert(strings.end(), token.begin(), token.end());
         string_offsets.push_back(strings.size());
      }
      // Preserve line-sensitive GridFunction metadata.
      types.push_back(NURBSHDF5TokenType::Newline);
   }

   HDF5Handle file(H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                             H5P_DEFAULT), H5Fclose);
   MFEM_VERIFY(hid_t(file) >= 0,
               "Unable to create NURBS HDF5 file: " << fname);
   HDF5Handle mfem_group(H5Gcreate2(file, "MFEM", H5P_DEFAULT, H5P_DEFAULT,
                                    H5P_DEFAULT), H5Gclose);
   MFEM_VERIFY(hid_t(mfem_group) >= 0,
               "Unable to create /MFEM in " << fname);
   HDF5Handle nurbs_group(H5Gcreate2(mfem_group, "NURBS", H5P_DEFAULT,
                                     H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
   MFEM_VERIFY(hid_t(nurbs_group) >= 0,
               "Unable to create " << nurbs_hdf5_group << " in " << fname);

   const hsize_t version_size = 2;
   HDF5Handle version_space(H5Screate_simple(1, &version_size, NULL), H5Sclose);
   HDF5Handle version_attr(H5Acreate2(nurbs_group, "schema_version",
                                      H5T_STD_I32LE, version_space,
                                      H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
   MFEM_VERIFY(hid_t(version_attr) >= 0,
               "Unable to create NURBS HDF5 schema version");
   CheckHDF5(H5Awrite(version_attr, H5T_NATIVE_INT,
                      nurbs_hdf5_schema_version), "write schema version");

   const std::vector<uint8_t> format(header.begin(), header.end());
   WriteNURBSHDF5Vector(nurbs_group, "format", format, H5T_STD_U8LE,
                        H5T_NATIVE_UCHAR, compression_level);
   WriteNURBSHDF5Vector(nurbs_group, "token_types", types, H5T_STD_U8LE,
                        H5T_NATIVE_UCHAR, compression_level);
   WriteNURBSHDF5Vector(nurbs_group, "integers", integers, H5T_STD_I64LE,
                        H5T_NATIVE_INT64, compression_level);
   WriteNURBSHDF5Vector(nurbs_group, "reals", reals, H5T_IEEE_F64LE,
                        H5T_NATIVE_DOUBLE, compression_level);
   WriteNURBSHDF5Vector(nurbs_group, "string_offsets", string_offsets,
                        H5T_STD_U64LE, H5T_NATIVE_UINT64, compression_level);
   WriteNURBSHDF5Vector(nurbs_group, "strings", strings, H5T_STD_U8LE,
                        H5T_NATIVE_UCHAR, compression_level);
}

} // namespace mfem

#endif // MFEM_USE_HDF5
