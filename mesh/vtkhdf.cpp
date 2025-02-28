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

#include "vtkhdf.hpp"

#ifdef MFEM_USE_HDF5

#include "../general/binaryio.hpp"

#include <hdf5_hl.h>
#include <cstdint>

namespace mfem
{

namespace vtkhdf
{

template <typename T>
std::vector<T> GetPointsBuffer(const Mesh &mesh)
{
   const int ne = mesh.GetNE();
   int np = 0;
   for (int i = 0; i < ne; i++)
   {
      const Geometry::Type geom = mesh.GetElementGeometry(i);
      np += Geometries.GetVertices(geom)->GetNPoints();
   }

   std::vector<T> points;
   points.reserve(np * 3);

   IsoparametricTransformation Tr;
   DenseMatrix pmat;
   for (int i = 0; i < ne; ++i)
   {
      const Geometry::Type geom = mesh.GetElementGeometry(i);
      RefinedGeometry &ref_geom = *GlobGeometryRefiner.Refine(geom, 1, 1);
      mesh.GetElementTransformation(i, &Tr);
      Tr.Transform(ref_geom.RefPts, pmat);

      for (int j = 0; j < pmat.Width(); j++)
      {
         points.push_back(pmat(0,j));
         if (pmat.Height() > 1) { points.push_back(pmat(1,j)); }
         else { points.push_back(0.0); }
         if (pmat.Height() > 2) { points.push_back(pmat(2,j)); }
         else { points.push_back(0.0); }
      }
   }

   return points;
}

} // namespace vtkhdf

template <> struct VTKHDF::TypeID<float> { static hid_t Get() { return H5T_NATIVE_FLOAT; } };
template <> struct VTKHDF::TypeID<double> { static hid_t Get() { return H5T_NATIVE_DOUBLE; } };
template <> struct VTKHDF::TypeID<int32_t> { static hid_t Get() { return H5T_NATIVE_INT32; } };
template <> struct VTKHDF::TypeID<uint32_t> { static hid_t Get() { return H5T_NATIVE_UINT32; } };
template <> struct VTKHDF::TypeID<int64_t> { static hid_t Get() { return H5T_NATIVE_INT64; } };
template <> struct VTKHDF::TypeID<uint64_t> { static hid_t Get() { return H5T_NATIVE_UINT64; } };
template <> struct VTKHDF::TypeID<size_t> { static hid_t Get() { return H5T_NATIVE_HSIZE; } };
template <> struct VTKHDF::TypeID<unsigned char> { static hid_t Get() { return H5T_NATIVE_UCHAR; } };

template <typename T>
hid_t VTKHDF::GetTypeID() { return TypeID<typename std::decay<T>::type>::Get(); }

void VTKHDF::SetupVTKHDF()
{
   vtk = H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

   // Set attributes: version and type
   const long version_buf[2] = {2, 2}; // VTKHDF version 2.2
   H5LTset_attribute_long(vtk, ".", "Version", version_buf, 2);

   // Note: we don't use the high-level API here since it will write out the
   // null terminator, which confuses the VTKHDF reader in ParaView.
   const std::string type_str = "UnstructuredGrid";
   const hid_t type_id = H5Tcopy(H5T_C_S1);
   H5Tset_size(type_id, type_str.size());
   H5Tset_strpad(type_id, H5T_STR_NULLPAD);

   const hid_t data_space = H5Screate(H5S_SCALAR);
   const hid_t type_attr = H5Acreate2(vtk, "Type", type_id, data_space,
                                      H5P_DEFAULT, H5P_DEFAULT);
   H5Awrite(type_attr, type_id, type_str.data());

   H5Aclose(type_attr);
   H5Sclose(data_space);
   H5Tclose(type_id);
}

void VTKHDF::EnsureSteps()
{
   // If the Steps group has already been created, return early.
   if (steps != H5I_INVALID_HID) { return; }

   // Otherwise, create the group and its datasets.
   steps = H5Gcreate2(vtk, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

   const hsize_t dim = 0;
   const hsize_t maxdim = H5S_UNLIMITED;
   // const hsize_t maxdim = 1;
   const hid_t fspace = H5Screate_simple(1, &dim, &maxdim);

   const hsize_t chunk = 1024;
   const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
   H5Pset_chunk(dcpl, 1, &chunk);

   auto create_offset_dataset = [&](const char *name, const hid_t type_id)
   {
      const hid_t d = H5Dcreate2(steps, name, type_id, fspace, H5P_DEFAULT,
                                 dcpl, H5P_DEFAULT);
      H5Dclose(d);
   };

   create_offset_dataset("PartOffsets", H5T_NATIVE_HSIZE);
   create_offset_dataset("Values", GetTypeID<real_t>());
   create_offset_dataset("CellOffsets", H5T_NATIVE_HSIZE);
   create_offset_dataset("PointOffsets", H5T_NATIVE_HSIZE);
   create_offset_dataset("ConnectivityIdOffsets", H5T_NATIVE_HSIZE);

   H5Pclose(dcpl);
   H5Sclose(fspace);
}

hid_t VTKHDF::EnsureDataset(hid_t f, const std::string &name, hid_t type,
                            int ndims)
{
   const char *name_c = name.c_str();
   const herr_t status = H5LTfind_dataset(f, name_c);

   if (status == 0)
   {
      // Dataset does not exist, create it.
      Dims dims(ndims);
      Dims maxdims(ndims, H5S_UNLIMITED);
      const hid_t fspace = H5Screate_simple(ndims, dims, maxdims);

      Dims chunk(ndims);
      chunk[0] = 1024;
      for (int i = 1; i < ndims; ++i) { chunk[i] = 16; }
      const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      H5Pset_chunk(dcpl, ndims, chunk);

      const hid_t d = H5Dcreate2(f, name_c, type, fspace, H5P_DEFAULT,
                                 dcpl, H5P_DEFAULT);
      H5Pclose(dcpl);
      return d;
   }
   else if (status > 0)
   {
      // Dataset exists, open it.
      return H5Dopen2(f, name_c, H5P_DEFAULT);
   }
   else
   {
      // Error occurred in H5LTfind_dataset.
      MFEM_ABORT("Error finding HDF5 dataset " << name);
   }
}

template <typename T>
void VTKHDF::AppendParData(hid_t f, const std::string &name, hsize_t locsize,
                           hsize_t offset, Dims globsize, T *data, hid_t dxpl)
{
   const int ndims = globsize.ndims;
   const hid_t d = EnsureDataset(f, name, GetTypeID<T>(), ndims);

   // Resize the dataset, set dims to its new size.
   hsize_t old_size;
   Dims dims(ndims);
   {
      const hid_t dspace = H5Dget_space(d);
      const int ndims_dset = H5Sget_simple_extent_ndims(dspace);
      MFEM_VERIFY(ndims_dset == ndims, "");
      H5Sget_simple_extent_dims(dspace, dims, NULL);
      H5Sclose(dspace);
      old_size = dims[0];
      dims[0] += locsize;
      for (int i = 1; i < ndims; ++i) { dims[i] = globsize[i]; }
      H5Dset_extent(d, dims);
   }

   // Write the new entry.
   const hid_t dspace = H5Dget_space(d);
   Dims start(ndims);
   start[0] = old_size + offset;
   Dims count(ndims);
   count[0] = locsize;
   for (int i = 1; i < ndims; ++i) { count[i] = globsize[i]; }

   H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, NULL, count, NULL);
   const hid_t memspace = H5Screate_simple(ndims, count, count);

   H5Dwrite(d, GetTypeID<T>(), memspace, dspace, dxpl, data);

   H5Sclose(memspace);
   H5Sclose(dspace);

   H5Dclose(d);
}

template <typename T>
void VTKHDF::WriteParData(hid_t f, const std::string &name, hsize_t locsize,
                          hsize_t offset, hsize_t globsize, T *data, hid_t dxpl)
{
   const hid_t fspace = H5Screate_simple(1, &globsize, NULL);
   const hid_t memspace = H5Screate_simple(1, &locsize, NULL);
   H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &offset, NULL, &locsize, NULL);
   const auto type_id = GetTypeID<T>();
   const hid_t d = H5Dcreate2(f, name.c_str(), type_id, fspace, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(d, type_id, memspace, fspace, dxpl, data);
   H5Dclose(d);
   H5Sclose(memspace);
   H5Sclose(fspace);
}

template <typename T>
std::vector<T> VTKHDF::AllGather(const T loc) const
{
   std::vector<T> all(mpi_size);
   if (UsingMpi())
   {
#ifdef MFEM_USE_MPI
      const MPI_Datatype type = MPITypeMap<T>::mpi_type;
      MPI_Allgather(&loc, 1, type, all.data(), 1, type, comm);
#endif
   }
   else
   {
      all[0] = loc;
   }
   return all;
}

template <typename T>
std::pair<T, T> VTKHDF::GetOffsetAndTotal(const T loc) const
{
   const auto all = AllGather(loc);

   T offset = 0;
   for (int i = 0; i < mpi_rank; ++i)
   {
      offset += all[i];
   }
   T total = offset;
   for (int i = mpi_rank; i < mpi_size; ++i)
   {
      total += all[i];
   }

   return std::make_pair(offset, total);
}

template <typename T>
void VTKHDF::AppendParVector(hid_t f, const std::string &name,
                             const std::vector<T> &data, hid_t dxpl, Dims dims)
{
   const uint64_t locsize = data.size();
   const auto offset_total = GetOffsetAndTotal(locsize);
   const auto offset = offset_total.first;
   const auto total = offset_total.second;

   size_t m = 1;
   for (int i = 1; i < dims.ndims; ++i) { m *= dims[i]; }
   dims[0] = total/m;

   AppendParData(f, name, locsize/m, offset/m, dims, data.data(), dxpl);
}

bool VTKHDF::UsingMpi() const
{
#ifndef MFEM_USE_MPI
   return false;
#else
   return comm != MPI_COMM_NULL;
#endif
}

VTKHDF::VTKHDF(const std::string &filename)
   : mpi_size(1),
     mpi_rank(0)
{
   fapl = H5P_DEFAULT;
   file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
   SetupVTKHDF();
   dxpl = H5Pcreate(H5P_DATASET_XFER);
}

static int MpiCommSize(MPI_Comm comm)
{
   int comm_size;
   MPI_Comm_size(comm, &comm_size);
   return comm_size;
}

static int MpiCommRank(MPI_Comm comm)
{
   int rank;
   MPI_Comm_rank(comm, &rank);
   return rank;
}

VTKHDF::VTKHDF(const std::string &filename, MPI_Comm comm_)
   : comm(comm_),
     mpi_size(MpiCommSize(comm)),
     mpi_rank(MpiCommRank(comm))
{
   // Create parallel access property list
   fapl = H5Pcreate(H5P_FILE_ACCESS);
   const MPI_Info info = MPI_INFO_NULL;
   H5Pset_fapl_mpio(fapl, comm, info);
   // Create the file (MPI collective since passing 'fapl' here)
   file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
   SetupVTKHDF();

   dxpl = H5Pcreate(H5P_DATASET_XFER);
   // H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
   H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);
}

template <typename T>
void VTKHDF::AppendValue(const hid_t loc_id, const std::string &name, T value)
{
   const hid_t d = H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT);

   // Resize the dataset, set dim to its new size.
   hsize_t dim;
   {
      const hid_t dspace = H5Dget_space(d);
      const int ndims = H5Sget_simple_extent_ndims(dspace);
      MFEM_VERIFY(ndims == 1, "");
      H5Sget_simple_extent_dims(dspace, &dim, NULL);
      H5Sclose(dspace);
      dim += 1;
      H5Dset_extent(d, &dim);
   }

   // Write the new entry.
   const hid_t dspace = H5Dget_space(d);
   const hsize_t coord = dim - 1;
   H5Sselect_elements(dspace, H5S_SELECT_SET, 1, &coord);
   const hsize_t one = 1;
   const hid_t memspace = H5Screate_simple(1, &one, &one);

   H5Dwrite(d, GetTypeID<T>(), memspace, dspace, dxpl, &value);

   H5Sclose(memspace);
   H5Sclose(dspace);
   H5Dclose(d);
}

void VTKHDF::UpdateSteps(real_t t)
{
   EnsureSteps();

   // Set the NSteps attribute
   ++nsteps;
   H5LTset_attribute_int(steps, ".", "NSteps", &nsteps, 1);

   AppendValue(steps, "Values", t);
   AppendValue(steps, "PartOffsets", (nsteps - 1)*mpi_size);
   AppendValue(steps, "PointOffsets", point_offsets.current);
   AppendValue(steps, "CellOffsets", cell_offsets.current);
   AppendValue(steps, "ConnectivityIdOffsets", connectivity_offsets.current);
}

void VTKHDF::SaveMesh(const Mesh &mesh)
{
   // If the mesh hasn't changed, we can return early.
   if (!mesh_id.HasChanged(mesh))
   {
      return;
   }
   // Set the cached MeshId.
   mesh_id.Set(mesh);

   // Get and count the points
   using T = real_t;
   std::vector<T> points = vtkhdf::GetPointsBuffer<T>(mesh);
   const int64_t np = points.size() / 3;
   const int64_t ne = mesh.GetNE();

   WriteParData(vtk, "NumberOfPoints", 1, mpi_rank, mpi_size, &np, dxpl);
   WriteParData(vtk, "NumberOfCells", 1, mpi_rank, mpi_size, &ne, dxpl);

   // const auto p_offset_total = GetOffsetAndTotal(np);
   // const auto p_offset = p_offset_total.first;
   // const auto np_total = p_offset_total.second;

   // Write out 2D data for points
   AppendParVector(vtk, "Points", points, dxpl, Dims({0, 3}));
   {
      // const hsize_t globsize[2] = { hsize_t(np_total), 3 };
      // const hsize_t locsize[2] = { hsize_t(np), 3 };
      // const hsize_t offset[2] = { hsize_t(p_offset), 0 };
      // const hid_t fspace = H5Screate_simple(2, globsize, NULL);
      // const hid_t memspace = H5Screate_simple(2, locsize, NULL);
      // H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, locsize, NULL);
      // const auto type_id = GetTypeID<T>();
      // const hid_t d = H5Dcreate2(vtk, "Points", type_id, fspace, H5P_DEFAULT,
      //                            H5P_DEFAULT, H5P_DEFAULT);
      // H5Dwrite(d, type_id, memspace, fspace, dxpl, points.data());
      // H5Dclose(d);
      // H5Sclose(memspace);
      // H5Sclose(fspace);
   }

   // Cell data
   {
      const auto e_offset_total = GetOffsetAndTotal(ne);
      const auto e_offset = e_offset_total.first;
      const auto ne_total = e_offset_total.second;

      // Offsets and connectivity
      {
         std::vector<int> offsets(ne + 1);
         std::vector<int> connectivity;
         int coff = 0;
         for (int i = 0; i < ne; i++)
         {
            offsets[i] = coff;
            Geometry::Type geom = mesh.GetElementGeometry(i);
            const int nv = Geometries.GetVertices(geom)->GetNPoints();
            RefinedGeometry &ref_geom = *GlobGeometryRefiner.Refine(geom, 1, 1);
            Array<int> &rg = ref_geom.RefGeoms;
            const int *p = VTKGeometry::VertexPermutation[geom];
            for (int k = 0; k < nv; k++)
            {
               connectivity.push_back(coff + rg[p ? p[k] : k]);
            }
            coff += nv;
         }
         offsets.back() = coff;

         const size_t n = connectivity.size();
         WriteParData(vtk, "NumberOfConnectivityIds", 1, mpi_rank, mpi_size, &n, dxpl);

         AppendParVector(vtk, "Connectivity", connectivity, dxpl);
         WriteParData(vtk, "Offsets", ne + 1, e_offset + mpi_rank,
                      ne_total + mpi_size, offsets.data(), dxpl);
      }

      // Cell types
      {
         std::vector<unsigned char> cell_types(ne);
         const int *vtk_geom_map = VTKGeometry::Map;
         for (int i = 0; i < ne; i++)
         {
            cell_types[i] = vtk_geom_map[mesh.GetElementGeometry(i)];
         }
         WriteParData(vtk, "Types", ne, e_offset, ne_total, cell_types.data(), dxpl);
      }

      // Attributes
      {
         std::vector<int> attributes(ne);
         for (int i = 0; i < ne; i++)
         {
            attributes[i] = mesh.GetAttribute(i);
         }

         const hid_t cell_data = H5Gcreate2(vtk, "CellData", H5P_DEFAULT,
                                            H5P_DEFAULT, H5P_DEFAULT);
         WriteParData(cell_data, "attribute", ne, e_offset, ne_total,
                      attributes.data(), dxpl);
         H5Gclose(cell_data);
      }
   }
}

void VTKHDF::SaveGridFunction(const GridFunction &gf, const std::string &name)
{
   // Create the point data group if needed
   if (point_data == H5I_INVALID_HID)
   {
      point_data = H5Gcreate2(vtk, "PointData", H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT);
   }

   const Mesh &mesh = *gf.FESpace()->GetMesh();

   std::vector<real_t> point_values;

   const int ref = 1;
   // scalar data
   Vector val;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      RefinedGeometry *ref_geom = GlobGeometryRefiner.Refine(
                                     mesh.GetElementBaseGeometry(i), ref, 1);
      gf.GetValues(i, ref_geom->RefPts, val);
      std::copy(val.begin(), val.end(), std::back_inserter(point_values));
   }

   AppendParVector(point_data, name, point_values, dxpl);

   point_data_offsets[name].Update(point_values.size());
}

void VTKHDF::Flush()
{
   H5Fflush(file, H5F_SCOPE_GLOBAL);
}

VTKHDF::~VTKHDF()
{
   if (steps != H5I_INVALID_HID) { H5Gclose(steps); }
   if (point_data != H5I_INVALID_HID) { H5Gclose(point_data); }
   H5Pclose(dxpl);
   H5Gclose(vtk);
   H5Fclose(file);
   H5Pclose(fapl);
}

} // namespace mfem

#endif // MFEM_USE_HDF5
