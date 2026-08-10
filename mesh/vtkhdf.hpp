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

#ifndef MFEM_VTKHDF
#define MFEM_VTKHDF

#include "../config/config.hpp"

#ifdef MFEM_USE_HDF5

#include "../fem/gridfunc.hpp"

#include <hdf5.h>
#include <cstdint>
#include <unordered_map>

#if defined(MFEM_USE_MPI) && defined(H5_HAVE_PARALLEL)
#define MFEM_PARALLEL_HDF5
#endif

namespace mfem
{

/// @brief Low-level class for writing %VTKHDF data (for use in ParaView).
///
/// Users should typically use ParaViewHDFDataCollection, Mesh::SaveVTKHDF, or
/// GridFunction::SaveVTKHDF instead.
class VTKHDF
{
public:
   /// Helper used in VTKHDF::VTKHDF for enabling/disabling restart mode.
   struct Restart
   {
      bool enabled = false;
      real_t time = 0.0;
      Restart() = default;
      Restart(real_t time_) : enabled(true), time(time_) { }
      Restart(bool enabled_, real_t time_) : enabled(enabled_), time(time_) { }
      static Restart Disabled() { return Restart(); }
   };

private:
#ifdef MFEM_USE_MPI
   /// MPI communicator (only available if MPI is enabled).
   MPI_Comm comm = MPI_COMM_NULL;
#endif
   /// Size of the MPI communicator (1 if MPI is not enabled).
   const int mpi_size = 1;
   /// Rank within MPI communicator (0 if MPI is not enabled).
   const int mpi_rank = 0;

   /// File access property list (needed for MPI I/O).
   hid_t fapl = H5I_INVALID_HID;
   /// HDF5 file handle.
   hid_t file = H5I_INVALID_HID;
   /// The 'VTKHDF' root group within the file.
   hid_t vtk = H5I_INVALID_HID;
   /// Data transfer property list.
   hid_t dxpl = H5P_DEFAULT;
   /// The group to cell data (element attributes).
   hid_t cell_data = H5I_INVALID_HID;
   /// The group to store point data (e.g. grid functions).
   hid_t point_data = H5I_INVALID_HID;

   /// Compression level (-1 means disabled, 0 through 9 enabled). Default is 6.
   int compression_level = 6;

   /// Wrapper for storing dataset dimensions (max ndims is 2D in VTKHDF).
   struct Dims
   {
      static constexpr size_t MAX_NDIMS = 2;
      std::array<hsize_t, MAX_NDIMS> data = { }; // Zero initialized
      int ndims = 0;
      Dims() = default;
      Dims(int ndims_) : ndims(ndims_) { MFEM_ASSERT(ndims <= MAX_NDIMS, ""); }
      Dims(int ndims_, hsize_t val) : Dims(ndims_) { data.fill(val); }
      template <typename T>
      Dims(std::initializer_list<T> data_) : Dims(int(data_.size()))
      { std::copy(data_.begin(), data_.end(), data.begin()); }
      operator hsize_t*() { return data.data(); }
      hsize_t &operator[](int i) { return data[i]; }
      hsize_t TotalSize() const;
   };

   /// @name Needed for time-dependent data sets.
   ///@{

   /// The group to store time step information.
   hid_t steps = H5I_INVALID_HID;

   /// Number of time steps saved.
   unsigned long nsteps = 0;

   /// Keep track of the offsets into the data arrays at each time step.
   struct Offsets
   {
      hsize_t current = 0;
      hsize_t next = 0;
      void Update(hsize_t offset)
      {
         current = next;
         next += offset;
      }
   };

   hsize_t part_offset; ///< Offset into the "NumberOf" arrays.
   Offsets point_offsets; ///< Offsets into the point-sized arrays.
   Offsets cell_offsets; ///< Offsets into the cell-sized arrays.
   Offsets connectivity_offsets; ///< Offsets into the connectivity arrays.

   /// Offsets into named point data arrays (for saved GridFunctions).
   std::unordered_map<std::string,Offsets> point_data_offsets;

   /// Track when the mesh has changed (enable reusing the previous saved mesh).
   class MeshId
   {
      const Mesh *mesh_ptr = nullptr;
      long sequence = -1;
      long nodes_sequence = -1;
      bool high_order = true;
      int ref = -1;
   public:
      MeshId() = default;
      /// @brief Is the given @a mesh different from the cached MeshId?
      ///
      /// The mesh is the same if the pointer is the same and sequence and nodes
      /// sequence are both the same.
      ///
      /// If HasChanged() returns true, then new mesh data should be saved. If
      /// it returns false, then the mesh doesn't need to be saved again.
      bool HasChanged(const Mesh &mesh) const
      {
         if (mesh_ptr == &mesh)
         {
            return sequence != mesh.GetSequence() ||
                   nodes_sequence != mesh.GetNodesSequence();
         }
         return true;
      }
      /// @brief Is the given @a mesh different from the cached MeshId, taking
      /// into account the order and refinement level of the previously saved
      /// mesh.
      bool HasChanged(const Mesh &mesh, bool high_order_, int ref_) const
      {
         return HasChanged(mesh) || high_order != high_order_ || ref != ref_;
      }
      /// Set the cached MeshId given @a mesh.
      void Set(const Mesh &mesh, bool high_order_, int ref_)
      {
         mesh_ptr = &mesh;
         sequence = mesh.GetSequence();
         nodes_sequence = mesh.GetNodesSequence();
         high_order = high_order_;
         ref = ref_;
      }
      /// Return the refinement level of the previously saved mesh.
      int GetRefinementLevel() const { return ref; }
   };

   /// The most recently saved MeshId.
   MeshId mesh_id;

   /// Number of points of most recently saved mesh.
   hsize_t last_np = 0;

   ///@}

   /// Hold rank-offset and total size for parallel I/O.
   struct OffsetTotal { size_t offset; size_t total; };

   /// Ensure that the 'Steps' group and 'PointDataOffsets' subgroup exist.
   void EnsureSteps();

   /// @brief Ensure that the dataset named @a name in @a f exists.
   ///
   /// If the dataset does not exist, create it and return the ID. If the
   /// dataset exists, open it and return the ID.
   ///
   /// The rank (number of dimensions) of the dataset is given by @a ndims and
   /// its data type is given by @a type.
   ///
   /// If the dataset does not exist, it will initially have size @a dims.
   /// Otherwise, it will be resized to append data of size @a dims, and @a dims
   /// will be set to the new total size.
   hid_t EnsureDataset(hid_t f, const std::string &name, hid_t type, Dims &dims);

   /// @brief Ensure the named group is open, creating it if needed. Set @a
   /// group to the ID.
   ///
   /// If @a group has already been opened, do nothing.
   void EnsureGroup(const std::string &name, hid_t &group);

   /// Appends data in parallel to the dataset named @a name in @a f.
   ///
   /// Data is appended along the zeroth dimension. Data of length @a locsize
   /// will be written at offset @a offset. The sum over all MPI ranks of @a
   /// locsize is equal to the zeroth dimension of @a globsize. The data is
   /// written in row-major order.
   template <typename T>
   void AppendParData(hid_t f, const std::string &name, hsize_t locsize,
                      hsize_t offset, Dims globsize, T *data);

   /// Appends data in parallel to the dataset named @a name in @a f.
   ///
   /// The input parameter @a dims is used to determine the rank of the data to
   /// be written, and the extent of all but the first dimension. For example,
   /// if dims = (0), then the written dimensions will be (N), where N is the
   /// sum of the size of @a data across all MPI ranks. If dims = (0, m), then
   /// the written dimensions will be (N/m, m). The data will be written in
   /// row-major ordering.
   ///
   /// The row offset and total are returned.
   template <typename T>
   OffsetTotal AppendParVector(hid_t f, const std::string &name,
                               const std::vector<T> &data, Dims dims = Dims(1));

   /// @brief Append a single value to the dataset named @a name in @a f.
   template <typename T>
   void AppendValue(const hid_t f, const std::string &name, T value);

   /// Gather the value 'loc' from all MPI ranks and return the resulting array.
   template <typename T>
   std::vector<T> AllGather(const T loc) const;

   /// @brief Returns the pair (offset, total) across MPI ranks given local data
   /// 'loc'.
   ///
   /// If loc_i represents the value of loc on MPI rank i, then the returned
   /// offset is the sum from i = 0 to R, where R is the current MPI rank. The
   /// returned total is the sum over all MPI ranks.
   ///
   /// Requires performing 'gather all' operation.
   OffsetTotal GetOffsetAndTotal(const size_t loc) const;

   /// @brief Return true if the VTKHDF file is using MPI, false otherwise.
   ///
   /// The object is using MPI if the MPI communicator was passed to the
   /// constructor. This is only possible if MFEM_USE_MPI is enabled. Even with
   /// MPI enabled, a non-MPI VTKHDF object may be created.
   bool UsingMpi() const;

   /// Calls MPI_Barrier if in parallel, no-op otherwise.
   void Barrier() const;

   /// Return the HDF5 type ID corresponding to type @a T.
   template <typename T> static hid_t GetTypeID();

   /// Common setup (VTK group creation, etc.) for serial and parallel.
   void SetupVTKHDF();

   /// Create a new file (deleting existing file if needed).
   void CreateFile(const std::string &filename, Restart restart);

   /// Read the entire named dataset.
   template <typename T>
   std::vector<T> ReadDataset(const std::string &name) const;

   /// Read a single value from the named dataset.
   template <typename T>
   T ReadValue(const std::string &name, hsize_t index) const;

   /// Truncate the named dataset after position size in the first dimension.
   void TruncateDataset(const std::string &name, hsize_t size);

   /// Truncate all datasets on and after time @a t.
   void Truncate(const real_t t);

public:

   /// @brief Create a new %VTKHDF file for serial I/O.
   ///
   /// If @a restart is enabled, then the file (if it exists) will be opened,
   /// and time steps before the given time will be preserved. If @a restart
   /// is not enabled, the file (if it exists) will be deleted, and a new file
   /// will be created.
   VTKHDF(const std::string &filename, Restart restart = Restart::Disabled());

#ifdef MFEM_PARALLEL_HDF5
   /// @brief Create a new %VTKHDF file for parallel I/O.
   ///
   /// If @a restart is enabled, then the file (if it exists) will be opened,
   /// and time steps before the given time will be preserved. If @a restart
   /// is not enabled, the file (if it exists) will be deleted, and a new file
   /// will be created.
   ///
   /// If using restart mode, the file must have previously been saved with the
   /// same number of MPI ranks.
   VTKHDF(const std::string &filename, MPI_Comm comm_,
          Restart restart = Restart::Disabled());
#endif

   /// @name Not copyable or movable.
   ///@{
   VTKHDF(const VTKHDF &) = delete;
   VTKHDF(VTKHDF &&) = delete;
   VTKHDF &operator=(const VTKHDF &) = delete;
   VTKHDF &operator=(VTKHDF &&) = delete;
   ///@}

   /// Update the time step data after saving the mesh and grid functions.
   void UpdateSteps(real_t t);

   /// Disable zlib compression.
   void DisableCompression() { compression_level = -1; }

   /// @brief Enable zlib compression at the specified level.
   ///
   /// @a level must be between 0 and 9, in increasing order of compression.
   void EnableCompression(int level = 6) { compression_level = level; }

   /// @brief Save the mesh, appending as a new time step.
   ///
   /// If @a high_order is true, @a ref determines the polynomial degree of the
   /// mesh elements (-1 indicates the same as the mesh nodes). If @a high_order
   /// is false, the elements are uniformly subdivided according to @a ref.
   template <typename FP_T = real_t>
   void SaveMesh(const Mesh &mesh, bool high_order = true, int ref = -1);

   /// Save the grid function with the given name, appending as a new time step.
   template <typename FP_T = real_t>
   void SaveGridFunction(const GridFunction &gf, const std::string &name);

   /// Flush the file.
   void Flush();

   ///< Destructor. Close the file.
   ~VTKHDF();
};

} // namespace mfem

#endif // MFEM_USE_HDF5
#endif // MFEM_VTKHDF
