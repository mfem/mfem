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

#include <algorithm>
#include <numeric>
#include <hdf5_hl.h>

namespace mfem
{

namespace
{

// Template class for HDF5 type IDs (specialized for each type T).
template <typename T> struct TypeID { };

template <> struct TypeID<float> { static hid_t Get() { return H5T_NATIVE_FLOAT; } };
template <> struct TypeID<double> { static hid_t Get() { return H5T_NATIVE_DOUBLE; } };
template <> struct TypeID<int32_t> { static hid_t Get() { return H5T_NATIVE_INT32; } };
template <> struct TypeID<uint64_t> { static hid_t Get() { return H5T_NATIVE_UINT64; } };
template <> struct TypeID<unsigned char> { static hid_t Get() { return H5T_NATIVE_UCHAR; } };

}

hsize_t VTKHDF::Dims::TotalSize() const
{
   return std::accumulate(data.begin(), data.begin() + ndims, 1,
                          std::multiplies<hsize_t>());
}

template <typename T>
hid_t VTKHDF::GetTypeID() { return TypeID<typename std::decay<T>::type>::Get(); }

void VTKHDF::SetupVTKHDF()
{
   vtk = H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

   // Set attributes: version and type
   const long version_buf[2] = {2, 2}; // VTKHDF version 2.2
   H5LTset_attribute_long(vtk, ".", "Version", version_buf, 2);

   // Note: we don't use the high-level API here since it will write out the
   // null terminator, which confuses the VTKHDF reader in ParaView. Fixed in
   // VTK MR !12044, https://gitlab.kitware.com/vtk/vtk/-/merge_requests/12044.
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
   EnsureGroup("Steps", steps);
   hid_t pd_offsets = H5I_INVALID_HID;
   EnsureGroup("Steps/PointDataOffsets", pd_offsets);
   H5Gclose(pd_offsets);
}

hid_t VTKHDF::EnsureDataset(hid_t f, const std::string &name, hid_t type,
                            Dims &dims)
{
   const char *name_c = name.c_str();

   const herr_t status = H5LTfind_dataset(f, name_c);
   Barrier();

   if (status == 0)
   {
      // Dataset does not exist, create it.
      const int ndims = dims.ndims;
      // The dataset is allowed to grow in the first dimension, but is fixed
      // in size in all other dimesions; the maximum dataset size is same as
      // dims, but unlimited in first dimension.
      Dims max_dims = dims;
      max_dims[0] = H5S_UNLIMITED;
      const hid_t fspace = H5Screate_simple(ndims, dims, max_dims);

      Dims chunk(ndims);
      size_t chunk_size_bytes = 1024 * 1024 / 2; // 0.5 MB
      const size_t t_bytes = H5Tget_size(type);
      for (int i = 1; i < ndims; ++i)
      {
         chunk[i] = dims[i];
         chunk_size_bytes /= dims[i];
      }
      chunk[0] = chunk_size_bytes / t_bytes;
      const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      H5Pset_chunk(dcpl, ndims, chunk);
      if (compression_level >= 0)
      {
         H5Pset_shuffle(dcpl);
         H5Pset_deflate(dcpl, compression_level);
      }

      const hid_t d = H5Dcreate2(f, name_c, type, fspace, H5P_DEFAULT,
                                 dcpl, H5P_DEFAULT);
      H5Pclose(dcpl);
      return d;
   }
   else if (status > 0)
   {
      // Dataset exists, open it.
      const hid_t d = H5Dopen2(f, name_c, H5P_DEFAULT);

      // Resize the dataset, set dims to its new size.
      Dims old_dims(dims.ndims);
      const hid_t dspace = H5Dget_space(d);
      const int ndims_dset = H5Sget_simple_extent_ndims(dspace);
      MFEM_VERIFY(ndims_dset == dims.ndims, "");
      H5Sget_simple_extent_dims(dspace, old_dims, NULL);
      H5Sclose(dspace);
      dims[0] += old_dims[0];
      H5Dset_extent(d, dims);

      return d;
   }
   else
   {
      // Error occurred in H5LTfind_dataset.
      MFEM_ABORT("Error finding HDF5 dataset " << name);
   }
}

void VTKHDF::EnsureGroup(const std::string &name, hid_t &group)
{
   if (group != H5I_INVALID_HID) { return; }

   const char *cname = name.c_str();
   const htri_t found = H5Lexists(vtk, cname, H5P_DEFAULT);
   Barrier();

   if (found > 0)
   {
      group = H5Gopen(vtk, cname, H5P_DEFAULT);
   }
   else if (found == 0)
   {
      group = H5Gcreate2(vtk, cname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   }
   else
   {
      MFEM_ABORT("Error finding HDF5 group " << name);
   }
}

template <typename T>
void VTKHDF::AppendParData(hid_t f, const std::string &name, hsize_t locsize,
                           hsize_t offset, Dims globsize, T *data)
{
   const int ndims = globsize.ndims;
   Dims dims = globsize;
   const hid_t d = EnsureDataset(f, name, GetTypeID<T>(), dims);

   // Write the new entry.
   const hid_t dspace = H5Dget_space(d);
   Dims start(ndims);
   start[0] = dims[0] - globsize[0] + offset;
   Dims count(ndims);
   count[0] = locsize;
   for (int i = 1; i < ndims; ++i) { count[i] = globsize[i]; }

   H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, NULL, count, NULL);
   H5Dwrite(d, GetTypeID<T>(), H5S_BLOCK, dspace, dxpl, data);
   H5Sclose(dspace);

   H5Dclose(d);
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

VTKHDF::OffsetTotal VTKHDF::GetOffsetAndTotal(const size_t loc) const
{
   const auto all = AllGather(uint64_t(loc));

   size_t offset = 0;
   for (int i = 0; i < mpi_rank; ++i)
   {
      offset += all[i];
   }
   size_t total = offset;
   for (int i = mpi_rank; i < mpi_size; ++i)
   {
      total += all[i];
   }

   return {offset, total};
}

template <typename T>
VTKHDF::OffsetTotal VTKHDF::AppendParVector(
   hid_t f, const std::string &name, const std::vector<T> &data, Dims dims)
{
   const size_t locsize = data.size();
   const auto offset_total = GetOffsetAndTotal(locsize);
   const auto offset = offset_total.offset;
   const auto total = offset_total.total;

   hsize_t m = 1;
   for (int i = 1; i < dims.ndims; ++i) { m *= dims[i]; }
   dims[0] = total/m;

   AppendParData(f, name, locsize/m, offset/m, dims, data.data());

   return {offset/m, total/m};
}

bool VTKHDF::UsingMpi() const
{
#ifdef MFEM_USE_MPI
   return comm != MPI_COMM_NULL;
#else
   return false;
#endif
}

void VTKHDF::Barrier() const
{
#ifdef MFEM_USE_MPI
   if (UsingMpi()) { MPI_Barrier(comm); }
#endif
}

template <typename T>
std::vector<T> VTKHDF::ReadDataset(const std::string &name) const
{
   const char *cname = name.c_str();
   int ndims;
   H5LTget_dataset_ndims(vtk, cname, &ndims);
   Dims dims(ndims);
   H5LTget_dataset_info(vtk, cname, dims, nullptr, nullptr);
   std::vector<T> vals(dims.TotalSize());
   H5LTread_dataset(vtk, cname, GetTypeID<T>(), vals.data());
   return vals;
}

template <typename T>
T VTKHDF::ReadValue(const std::string &name, hsize_t index) const
{
   const char *cname = name.c_str();

   int ndims;
   H5LTget_dataset_ndims(vtk, cname, &ndims);

   const hid_t d = H5Dopen(vtk, cname, H5P_DEFAULT);

   // Write the new entry.
   const hid_t dspace = H5Dget_space(d);
   Dims start(ndims);
   start[0] = index;
   Dims count(ndims);
   for (int i = 0; i < ndims; ++i) { count[i] = 1; }

   H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, NULL, count, NULL);
   const hid_t memspace = H5Screate_simple(ndims, count, count);

   T value;
   H5Dread(d, GetTypeID<T>(), memspace, dspace, dxpl, &value);

   H5Sclose(memspace);
   H5Sclose(dspace);
   H5Dclose(d);

   return value;
}

void VTKHDF::TruncateDataset(const std::string &name, hsize_t size)
{
   const hid_t d = H5Dopen2(vtk, name.c_str(), H5P_DEFAULT);
   const hid_t dspace = H5Dget_space(d);
   const int ndims = H5Sget_simple_extent_ndims(dspace);
   Dims dims(ndims);
   H5Sget_simple_extent_dims(dspace, dims, NULL);
   H5Sclose(dspace);
   dims[0] = size;
   H5Dset_extent(d, dims);
   H5Dclose(d);
}

void VTKHDF::Truncate(const real_t t)
{
   // Find the first time step 'i' at least as large as 't'. Truncate all
   // datasets at the corresponding offsets.
   const std::vector<real_t> tvals = ReadDataset<real_t>("Steps/Values");
   auto it = std::find_if(tvals.begin(), tvals.end(), [t](real_t t2) { return t2 >= t; });

   // Sanity check: we can only use restart mode with the same number of MPI
   // ranks (mesh partitions) as the originally save file.
   {
      Dims dims(1);
      H5LTget_dataset_info(vtk, "NumberOfCells", dims, nullptr, nullptr);
      MFEM_VERIFY(dims[0] == tvals.size() * mpi_size, "Incompatible VTKHDF sizes.");
   }

   // Index of found time index (may be 'one-past-the-end' if not found)
   const ptrdiff_t i = std::distance(tvals.begin(), it);

   // Only truncate if needed
   const bool truncate = it != tvals.end();

   // Number of steps we are keeping
   nsteps = i;
   H5LTset_attribute_ulong(vtk, "Steps", "NSteps", &nsteps, 1);

   // We want to continue writing immediately after step 'i - 1'. If i = 0,
   // then this is at the beginning of the file, and the offsets do not need
   // to be updated.
   hsize_t npoints = 0;
   if (i > 0)
   {
      point_offsets.next = ReadValue<hsize_t>("Steps/PointOffsets", i - 1);
      cell_offsets.next = ReadValue<hsize_t>("Steps/CellOffsets", i - 1);
      connectivity_offsets.next =
         ReadValue<hsize_t>("Steps/ConnectivityIdOffsets", i - 1);

      for (int part = 0; part < mpi_size; ++part)
      {
         const hsize_t p_i = ReadValue<hsize_t>("Steps/PartOffsets", i - 1 + part);
         npoints += ReadValue<hsize_t>("NumberOfPoints", p_i);
         cell_offsets.next += ReadValue<hsize_t>("NumberOfCells", p_i);
         connectivity_offsets.next +=
            ReadValue<hsize_t>("NumberOfConnectivityIds", p_i);
      }
      point_offsets.next += npoints;
   }

   // Find the offsets associated with all saved grid functions.
   const hid_t g = H5Gopen2(vtk, "Steps/PointDataOffsets", H5P_DEFAULT);
   if (g != H5I_INVALID_HID)
   {
      std::vector<std::string> names;
      auto itfn = [](hid_t, const char *name, const H5L_info2_t*, void *data)
      {
         auto names_ptr = static_cast<std::vector<std::string>*>(data);
         names_ptr->emplace_back(name);
         return herr_t(0);
      };
      H5Literate2(g, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, itfn, &names);
      H5Gclose(g);

      for (auto name : names)
      {
         const std::string dset_name = "Steps/PointDataOffsets/" + name;
         hsize_t offset = 0;
         if (i > 0)
         {
            offset = ReadValue<hsize_t>(dset_name, i - 1) + npoints;
         }
         point_data_offsets[name].next = offset;
         if (truncate)
         {
            TruncateDataset(dset_name, nsteps);
            TruncateDataset("PointData/" + name, offset);
         }
      }
   }

   if (truncate)
   {
      TruncateDataset("Steps/Values", nsteps);
      TruncateDataset("Steps/PartOffsets", nsteps);
      TruncateDataset("Steps/PointOffsets", nsteps);
      TruncateDataset("Steps/CellOffsets", nsteps);
      TruncateDataset("Steps/ConnectivityIdOffsets", nsteps);

      TruncateDataset("NumberOfCells", nsteps * mpi_size);
      TruncateDataset("NumberOfConnectivityIds", nsteps * mpi_size);
      TruncateDataset("NumberOfPoints", nsteps * mpi_size);

      TruncateDataset("CellData/attribute", cell_offsets.next);
      TruncateDataset("Types", cell_offsets.next);
      TruncateDataset("Points", point_offsets.next);
      TruncateDataset("Connectivity", connectivity_offsets.next);
      TruncateDataset("Offsets", cell_offsets.next + nsteps * mpi_size);
   }
}

void VTKHDF::CreateFile(const std::string &filename, Restart restart)
{
   if (restart.enabled)
   {
      bool file_exists = mpi_rank == 0 && [&filename]()
      {
         std::ifstream f(filename);
         return f.good();
      }();

#ifdef MFEM_USE_MPI
      if (UsingMpi())
      {
         MPI_Allreduce(MPI_IN_PLACE, &file_exists, 1, MPI_CXX_BOOL, MPI_LOR, comm);
      }
#endif

      if (file_exists)
      {
         // Disable file locking, allowing modification to files that may be
         // open in ParaView (otherwise writes will fail).
         H5Pset_file_locking(fapl, false, true);

         file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl);
         vtk = H5Gopen(file, "VTKHDF", H5P_DEFAULT);
         Truncate(restart.time);
         return;
      }
   }

   // At this point, either restart is disabled, or file doesn't exist

   // Delete the file if it exists
   std::remove(filename.c_str());
   // Create the new file
   file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
   // Setup 'VTKHDF' group
   SetupVTKHDF();
}

VTKHDF::VTKHDF(const std::string &filename, Restart restart)
{
   fapl = H5Pcreate(H5P_FILE_ACCESS);
   CreateFile(filename, restart);
}

#ifdef MFEM_PARALLEL_HDF5

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

VTKHDF::VTKHDF(const std::string &filename, MPI_Comm comm_, Restart restart)
   : comm(comm_),
     mpi_size(MpiCommSize(comm)),
     mpi_rank(MpiCommRank(comm))
{
   // Create file access property list, needed for parallel I/O
   fapl = H5Pcreate(H5P_FILE_ACCESS);
   const MPI_Info info = MPI_INFO_NULL;
   H5Pset_fapl_mpio(fapl, comm, info);
   // Create parallel data transfer property list
   dxpl = H5Pcreate(H5P_DATASET_XFER);
   H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

   CreateFile(filename, restart);
}

#endif

template <typename T>
void VTKHDF::AppendValue(const hid_t f, const std::string &name, T value)
{
   const hsize_t locsize = (mpi_rank == 0) ? 1 : 0;
   AppendParData(f, name, locsize, 0, Dims({1}), &value);
}

void VTKHDF::UpdateSteps(real_t t)
{
   EnsureSteps();

   // Set the NSteps attribute
   ++nsteps;
   H5LTset_attribute_ulong(steps, ".", "NSteps", &nsteps, 1);

   AppendValue(steps, "Values", t);
   AppendValue(steps, "PartOffsets", part_offset);
   AppendValue(steps, "PointOffsets", point_offsets.current);
   AppendValue(steps, "CellOffsets", cell_offsets.current);
   AppendValue(steps, "ConnectivityIdOffsets", connectivity_offsets.current);

   if (!point_data_offsets.empty())
   {
      const hid_t g = H5Gopen2(steps, "PointDataOffsets", H5P_DEFAULT);
      for (const auto &pd : point_data_offsets)
      {
         const char *name = pd.first.c_str();
         AppendValue(g, name, pd.second.current);
      }
      H5Gclose(g);
   }
}

template <typename FP_T>
void VTKHDF::SaveMesh(const Mesh &mesh, bool high_order, int ref)
{
   // If refinement level not set, set to default value
   if (ref <= 0)
   {
      ref = 1;
      if (high_order)
      {
         if (auto *nodal_space = mesh.GetNodalFESpace())
         {
            ref = nodal_space->GetMaxElementOrder();
         }
      }
   }

   const Dims mpi_dims({mpi_size});

   // If the mesh hasn't changed, we can return early.
   if (!mesh_id.HasChanged(mesh, high_order, ref))
   {
      // The HDF5 format assumes that the "NumberOf" datasets will have size
      // given by the number of parts (number of MPI ranks) times the number of
      // time steps (see
      // https://gitlab.kitware.com/vtk/vtk/-/issues/18981#note_1366124).
      //
      // If the mesh doesn't change, we don't increment the value in
      // 'PartOffsets', and so these values in the "NumberOf" datasets will
      // never be read, so we just fill them with a dummy value.
      const hsize_t zero = 0;
      AppendParData(vtk, "NumberOfPoints", 1, mpi_rank, mpi_dims, &zero);
      AppendParData(vtk, "NumberOfCells", 1, mpi_rank, mpi_dims, &zero);
      AppendParData(vtk, "NumberOfConnectivityIds", 1, mpi_rank, mpi_dims, &zero);
      const int zero_int = 0;
      AppendParData(vtk, "Offsets", 1, mpi_rank, mpi_dims, &zero_int);
      return;
   }

   // Set the cached MeshId
   mesh_id.Set(mesh, high_order, ref);

   // Update the part offsets
   part_offset = nsteps * mpi_size;

   // Number of times to refine each element
   const int ref_0 = high_order ? 1 : ref;
   // Return the RefinementGeometry object for element 'e'
   auto get_ref_geom = [&](int e, int r) -> RefinedGeometry&
   {
      const Geometry::Type geom = mesh.GetElementGeometry(e);
      return *GlobGeometryRefiner.Refine(geom, r, 1);
   };
   // Return the number of vertices in element 'e'
   auto get_nv = [&](int e)
   {
      return Geometries.NumVerts[mesh.GetElementGeometry(e)];
   };
   // Return the number of refined elements for element 'e'
   auto get_ne_ref = [&](int e, int r)
   {
      return get_ref_geom(e, r).RefGeoms.Size() / get_nv(e);
   };

   // Count the points (and number of refined elements, needed if high_order is
   // false).
   std::vector<FP_T> points;
   hsize_t ne_ref = 0;
   hsize_t np = 0;
   {
      const int ne = mesh.GetNE();
      for (int e = 0; e < ne; e++)
      {
         RefinedGeometry &ref_geom = get_ref_geom(e, ref);
         np += ref_geom.RefPts.GetNPoints();
         ne_ref += ref_geom.RefGeoms.Size() / get_nv(e);
      }

      points.reserve(np * 3);

      IsoparametricTransformation Tr;
      DenseMatrix pmat;
      for (int e = 0; e < ne; ++e)
      {
         RefinedGeometry &ref_geom = get_ref_geom(e, ref);
         mesh.GetElementTransformation(e, &Tr);
         Tr.Transform(ref_geom.RefPts, pmat);

         for (int i = 0; i < pmat.Width(); i++)
         {
            points.push_back(FP_T(pmat(0,i)));
            if (pmat.Height() > 1) { points.push_back(FP_T(pmat(1,i))); }
            else { points.push_back(0.0); }
            if (pmat.Height() > 2) { points.push_back(FP_T(pmat(2,i))); }
            else { points.push_back(0.0); }
         }
      }
   }

   const int ne_0 = mesh.GetNE();
   const hsize_t ne = high_order ? ne_0 : ne_ref;

   AppendParData(vtk, "NumberOfPoints", 1, mpi_rank, mpi_dims, &np);
   AppendParData(vtk, "NumberOfCells", 1, mpi_rank, mpi_dims, &ne);

   // Save the number of points written
   last_np = np;

   // Write out 2D data for points
   auto point_offset_total = AppendParVector(vtk, "Points", points, Dims({0, 3}));
   point_offsets.Update(point_offset_total.total);

   // Cell data
   {
      const auto e_offset_total = GetOffsetAndTotal(ne);
      const auto e_offset = e_offset_total.offset;
      const auto ne_total = e_offset_total.total;

      cell_offsets.Update(ne_total);

      // Offsets and connectivity
      {
         std::vector<int> offsets(ne + 1);
         std::vector<int> connectivity;

         int off = 0;
         if (high_order)
         {
            Array<int> local_connectivity;
            for (int e = 0; e < int(ne); ++e)
            {
               offsets[e] = off;
               const Geometry::Type geom = mesh.GetElementGeometry(e);
               CreateVTKElementConnectivity(local_connectivity, geom, ref);
               const int nnodes = local_connectivity.Size();
               for (int i = 0; i < nnodes; ++i)
               {
                  connectivity.push_back(off + local_connectivity[i]);
               }
               off += nnodes;
            }
            offsets.back() = off;
         }
         else
         {
            int off_0 = 0;
            int e_ref = 0;
            for (int e = 0; e < ne_0; ++e)
            {
               const Geometry::Type geom = mesh.GetElementGeometry(e);
               const int nv = get_nv(e);
               RefinedGeometry &ref_geom = get_ref_geom(e, ref_0);
               Array<int> &rg = ref_geom.RefGeoms;
               for (int r = 0; r < rg.Size(); ++e_ref)
               {
                  offsets[e_ref] = off;
                  off += nv;
                  const int *p = VTKGeometry::VertexPermutation[geom];
                  for (int k = 0; k < nv; ++k, ++r)
                  {
                     connectivity.push_back(off_0 + rg[p ? (r - k + p[k]) : r]);
                  }
               }
               off_0 += ref_geom.RefPts.Size();
            }
            offsets.back() = off;
         }

         const hsize_t n = connectivity.size();
         AppendParData(vtk, "NumberOfConnectivityIds", 1, mpi_rank, mpi_dims, &n);

         auto connectivity_offset_total
            = AppendParVector(vtk, "Connectivity", connectivity);
         connectivity_offsets.Update(connectivity_offset_total.total);

         AppendParData(vtk, "Offsets", ne + 1, e_offset + mpi_rank,
                       Dims({ne_total + mpi_size}), offsets.data());

      }

      // Cell types
      {
         std::vector<unsigned char> cell_types(ne);
         const int *vtk_geom_map =
            high_order ? VTKGeometry::HighOrderMap : VTKGeometry::Map;
         int e_ref = 0;
         for (int e = 0; e < ne_0; ++e)
         {
            const int ne_ref_e = get_ne_ref(e, ref_0);
            for (int i = 0; i < ne_ref_e; ++i, ++e_ref)
            {
               cell_types[e_ref] = static_cast<unsigned char>(
                                      vtk_geom_map[mesh.GetElementGeometry(e)]);
            }
         }
         AppendParData(vtk, "Types", ne, e_offset, Dims({ne_total}),
                       cell_types.data());
      }

      // Attributes
      {
         // Ensure cell data group exists
         EnsureGroup("CellData", cell_data);
         std::vector<int> attributes(ne);
         hsize_t e_ref = 0;
         for (int e = 0; e < ne_0; ++e)
         {
            const int attr = mesh.GetAttribute(e);
            const int ne_ref_e = get_ne_ref(e, ref_0);
            for (int i = 0; i < ne_ref_e; ++i, ++e_ref)
            {
               attributes[e_ref] = attr;
            }
         }
         AppendParData(cell_data, "attribute", ne, e_offset, Dims({ne_total}),
                       attributes.data());
      }
   }
}

template <typename FP_T>
void VTKHDF::SaveGridFunction(const GridFunction &gf, const std::string &name)
{
   // Create the point data group if needed
   EnsureGroup("PointData", point_data);

   const Mesh &mesh = *gf.FESpace()->GetMesh();

   MFEM_VERIFY(!mesh_id.HasChanged(mesh), "Mesh must be saved first");
   const int ref = mesh_id.GetRefinementLevel();

   const int vdim = gf.VectorDim();

   std::vector<FP_T> point_values(vdim * last_np);
   DenseMatrix vec_val, pmat;
   int off = 0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      RefinedGeometry &ref_geom = *GlobGeometryRefiner.Refine(
                                     mesh.GetElementBaseGeometry(e), ref, 1);
      gf.GetVectorValues(e, ref_geom.RefPts, vec_val, pmat);
      for (int i = 0; i < vec_val.Width(); ++i)
      {
         for (int vd = 0; vd < vdim; ++vd)
         {
            point_values[off] = FP_T(vec_val(vd, i));
            ++off;
         }
      }
   }

   Dims dims(vdim == 1 ? 1 : 2);
   if (vdim > 1) { dims[1] = vdim; }
   auto offset_total = AppendParVector(point_data, name, point_values, dims);
   point_data_offsets[name].Update(offset_total.total);
}

void VTKHDF::Flush()
{
   H5Fflush(file, H5F_SCOPE_GLOBAL);
}

VTKHDF::~VTKHDF()
{
   if (steps != H5I_INVALID_HID) { H5Gclose(steps); }
   if (cell_data != H5I_INVALID_HID) { H5Gclose(cell_data); }
   if (point_data != H5I_INVALID_HID) { H5Gclose(point_data); }
   if (dxpl != H5P_DEFAULT) { H5Pclose(dxpl); }
   if (vtk != H5I_INVALID_HID) { H5Gclose(vtk); }
   if (fapl != H5I_INVALID_HID) { H5Pclose(fapl); }
   if (file != H5I_INVALID_HID) { H5Fclose(file); }
}

template void VTKHDF::SaveMesh<float>(const Mesh&, bool, int);
template void VTKHDF::SaveMesh<double>(const Mesh&, bool, int);
template void VTKHDF::SaveGridFunction<float>(const GridFunction&,
                                              const std::string&);
template void VTKHDF::SaveGridFunction<double>(const GridFunction&,
                                               const std::string&);

} // namespace mfem

#endif // MFEM_USE_HDF5
