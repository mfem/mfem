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
#pragma once

#include "partial_qspace.hpp"

#include "../config/config.hpp"
#include "qspace.hpp"
#include "qfunction.hpp"

#include <vector>

namespace mfem
{

/// Class for representing quadrature functions on a subset of mesh elements.
/** PartialQuadratureFunction extends MFEM's QuadratureFunction to efficiently store and
    manipulate quadrature point data for only a subset of mesh elements. This is essential
    for multi-material simulations where different constitutive models and state variables
    apply to different regions of the mesh.
    
    The class maintains compatibility with MFEM's QuadratureFunction interface while
    providing optimized memory usage and performance for partial element sets. It handles
    the mapping between partial and full quadrature spaces automatically and provides
    default values for elements not in the partial set. */
class PartialQuadratureFunction : public QuadratureFunction
{
private:
   /// Reference to the specialized PartialQuadratureSpace.
   PartialQuadratureSpace *part_quad_space;
   
   /// Default value for elements not in the partial set.
   real_t default_value;

public:
   /// Create a PartialQuadratureFunction.
   /** @param[in] qspace_ Pointer to the PartialQuadratureSpace defining the element subset.
       @param[in] vdim_ Vector dimension (number of components per quadrature point).
       @param[in] default_val Default value for elements not in the partial set. */
   PartialQuadratureFunction(PartialQuadratureSpace *qspace_, int vdim_ = 1,
                             real_t default_val = -1.0)
      : QuadratureFunction(qspace_, vdim_), part_quad_space(qspace_),
        default_value(default_val) { }

   /// Create a PartialQuadratureFunction based on the given PartialQuadratureSpace.
   /** The PartialQuadratureFunction does not assume ownership of the
       PartialQuadratureSpace or the external data.
       @warning @a qspace_ may not be NULL.
       @note @a qf_data must be a valid **host** pointer (see the constructor
       Vector::Vector(real_t *, int)). */
   PartialQuadratureFunction(PartialQuadratureSpace *qspace_, real_t *qf_data,
                             int vdim_ = 1, real_t default_val = -1.0)
      : QuadratureFunction(qspace_, qf_data, vdim_), part_quad_space(qspace_),
        default_value(default_val) { }

   /// Get the specialized PartialQuadratureSpace.
   PartialQuadratureSpace *GetPartialSpace() const { return part_quad_space; }

   /// Set this equal to a constant value.
   PartialQuadratureFunction &operator=(real_t value) override
   {
      QuadratureFunction::operator=(value);
      return *this;
   }

   /// Copy the data from a Vector.
   PartialQuadratureFunction &operator=(const Vector &vec) override
   {
      MFEM_ASSERT(part_quad_space && vec.Size() == this->Size(), "");
      QuadratureFunction::operator=(vec);
      return *this;
   }

   /// Copy the data from another QuadratureFunction.
   /** This operator intelligently copies data from a QuadratureFunction, handling
       both cases where the source function has the same size (direct copy) or
       different size (element-by-element mapping). */
   PartialQuadratureFunction &operator=(const QuadratureFunction &qf);

   /// Fill a global QuadratureFunction with data from this partial function.
   /** @param[out] qf Reference to the global QuadratureFunction to fill.
       @param[in] fill Whether to initialize non-partial elements with default value. */
   void FillQuadratureFunction(QuadratureFunction &qf, const bool fill = false);

   /// Override ProjectGridFunction to project only onto the partial space.
   void ProjectGridFunction(const GridFunction &gf) override
   {
      MFEM_ABORT("Unsupported case.");
   }

   /// Return all values associated with mesh element as a reference Vector.
   /** @param[in] idx Global element index.
       @param[out] values Output vector that will reference the internal data or be filled
                   with defaults. */
   inline void GetValues(int idx, Vector &values) override;

   /// Return all values associated with mesh element as a copy Vector.
   /** @param[in] idx Global element index.
       @param[out] values Output vector to store the copied values. */
   inline void GetValues(int idx, Vector &values) const override;

   /// Return quadrature function values at a specific integration point as reference.
   /** @param[in] idx Global element index.
       @param[in] ip_num Quadrature point number within the element.
       @param[out] values Output vector that will reference the internal data or be filled
                   with defaults. */
   inline void GetValues(int idx, const int ip_num, Vector &values) override;

   /// Return quadrature function values at a specific integration point as copy.
   /** @param[in] idx Global element index.
       @param[in] ip_num Quadrature point number within the element.
       @param[out] values Output vector to store the copied values. */
   inline void GetValues(int idx, const int ip_num, Vector &values) const override;

   /// Return all values associated with mesh element as a reference DenseMatrix.
   /** @param[in] idx Global element index.
       @param[out] values Output matrix that will reference the internal data or be filled
                   with defaults. */
   inline void GetValues(int idx, DenseMatrix &values) override;

   /// Return all values associated with mesh element as a copy DenseMatrix.
   /** @param[in] idx Global element index.
       @param[out] values Output matrix to store the copied values. */
   inline void GetValues(int idx, DenseMatrix &values) const override;

   /// Get the IntegrationRule associated with entity (element or face).
   using QuadratureFunction::GetIntRule;

   /// Write the PartialQuadratureFunction to a stream.
   virtual void Save(std::ostream &os) const override
   {
      if (part_quad_space->global_offsets.Size() == 1)
      {
         QuadratureFunction::Save(os);
         return;
      }
      MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
   }

   /// Write the PartialQuadratureFunction to an output stream in VTU format.
   virtual void SaveVTU(std::ostream &os, VTKFormat format = VTKFormat::ASCII,
                        int compression_level = 0,
                        const std::string &field_name = "u") const override
   {
      if (part_quad_space->global_offsets.Size() == 1)
      {
         QuadratureFunction::SaveVTU(os, format, compression_level, field_name);
         return;
      }
      MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
   }

   /// Save the PartialQuadratureFunction to a VTU (ParaView) file.
   virtual void SaveVTU(const std::string &filename,
                        VTKFormat format = VTKFormat::ASCII,
                        int compression_level = 0,
                        const std::string &field_name = "u") const override
   {
      if (part_quad_space->global_offsets.Size() == 1)
      {
         QuadratureFunction::SaveVTU(filename, format, compression_level, field_name);
         return;
      }
      MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
   }

   /// Return the integral of the quadrature function (vdim = 1 only).
   virtual real_t Integrate() const override
   {
      if (part_quad_space->global_offsets.Size() == 1)
      {
         return QuadratureFunction::Integrate();
      }
      MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
      return default_value;
   }

   /// Integrate the vector-valued quadrature function.
   virtual void Integrate(Vector &integrals) const override
   {
      if (part_quad_space->global_offsets.Size() == 1)
      {
         QuadratureFunction::Integrate(integrals);
         return;
      }
      MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
   }
};

/// Factory class for creating PartialQuadratureFunction objects from a memory pool.
/** PartialQuadratureFunctionFactory uses a bump allocator pattern to create
    PartialQuadratureFunction objects from a pre-allocated memory pool. This is useful
    for reducing memory allocation overhead when creating many quadrature functions,
    particularly in performance-critical contexts.
    
    The factory does not own the memory pool - it is managed by an external entity.
    The memory pool must be large enough to accommodate all PartialQuadratureFunction
    data that will be created. If the memory pool is exhausted, undefined behavior will
    occur.
    
    @warning The provided memory pool must be sufficiently large to hold all
    PartialQuadratureFunction data. Each PartialQuadratureFunction requires
    vdim * qspace->GetSize() * sizeof(real_t) bytes. Insufficient memory will
    result in undefined behavior (buffer overflow, data corruption, crashes).
    
    @warning This class does NOT own the memory pool. The caller is responsible for
    managing the lifetime of the memory pool and ensuring it remains valid for the
    lifetime of all created PartialQuadratureFunction objects. */
class PartialQuadratureFunctionFactory
{
private:
   /// Memory pool pointer (not owned by this class).
   real_t *memory_pool;
   
   /// Current offset into the memory pool.
   int memory_offset;

public:
   /// Create a factory with the given memory pool.
   /** @param[in] pool Pointer to pre-allocated memory pool. The factory does NOT take
                 ownership of this memory. The caller must ensure the pool remains
                 valid for the lifetime of all created PartialQuadratureFunction objects.
       @warning The memory pool must be large enough to accommodate all
                PartialQuadratureFunction objects that will be created. */
   inline PartialQuadratureFunctionFactory(real_t *pool) :
   memory_pool(pool), memory_offset(0)
   {
      MFEM_VERIFY(pool != nullptr, "Memory pool cannot be NULL");
   }

   /// Create a PartialQuadratureFunction using the memory pool.
   /** This method creates a PartialQuadratureFunction that uses memory from the
       internal memory pool, advancing the pool offset by the required amount.
       
       @param[in] qspace Pointer to the PartialQuadratureSpace (must not be NULL).
       @param[in] vdim Vector dimension (must be >= 1).
       @param[in] default_val Default value for elements not in the partial set.
       @return PartialQuadratureFunction using memory from the pool.
       
       @warning The memory pool must have sufficient remaining space:
                Required bytes = vdim * qspace->GetSize() * sizeof(real_t).
                Insufficient space results in undefined behavior. */
   inline PartialQuadratureFunction Create(PartialQuadratureSpace *qspace,
                                           int vdim = 1,
                                           real_t default_val = -1.0);

   /// Get the current memory offset into the pool.
   /** @return Number of real_t elements currently allocated from the pool. */
   int GetMemoryOffset() const { return memory_offset; }

   /// Reset the memory pool offset to zero.
   /** This allows reusing the memory pool from the beginning. Use with caution -
       any previously created PartialQuadratureFunction objects will have their
       data invalidated if new objects overwrite the memory. */
   void Reset() { memory_offset = 0; }

   /// Get the memory pool pointer.
   /** @return Pointer to the memory pool (not owned by this class). */
   real_t *GetMemoryPool() const { return memory_pool; }

   /// Calculate the total memory pool size required for multiple quadrature functions.
   /** This helper function computes the total number of real_t elements needed to
       store multiple PartialQuadratureFunction objects with potentially different
       vector dimensions.
       
       @param[in] qspaces Vector of PartialQuadratureSpace pointers.
       @param[in] vdims Vector of vector dimensions (must be same size as qspaces).
       @return Total number of real_t elements required.
       
       @note Each element in vdims corresponds to the vdim for the corresponding
             qspace at the same index. */
   static inline int CalculateMemorySize(
      const std::vector<PartialQuadratureSpace*> &qspaces,
      const std::vector<int> &vdims);

   /// Calculate the total memory pool size required for multiple quadrature functions.
   /** This helper function computes the total number of real_t elements needed to
       store multiple PartialQuadratureFunction objects, all with the same vector
       dimension.
       
       @param[in] qspaces Vector of PartialQuadratureSpace pointers.
       @param[in] vdim Vector dimension to use for all spaces.
       @return Total number of real_t elements required. */
   static inline int CalculateMemorySize(
      const std::vector<PartialQuadratureSpace*> &qspaces,
      int vdim);
};

// Inline methods

inline void PartialQuadratureFunction::GetValues(int idx, Vector &values)
{
   const int local_index = part_quad_space->GlobalToLocal(idx);
   // If global_offsets.Size() == 1 then we'll always go down this path
   if (local_index > -1)
   {
      const int s_offset = part_quad_space->offsets[local_index];
      const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
      values.MakeRef(*this, vdim * s_offset, vdim * sl_size);
   }
   else
   {
      const int s_offset = part_quad_space->global_offsets[idx];
      const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
      values.Destroy();
      values.SetSize(vdim * sl_size);
      values.HostWrite();
      for (int i = 0; i < values.Size(); i++)
      {
         values(i) = default_value;
      }
   }
}

inline void PartialQuadratureFunction::GetValues(int idx, Vector &values) const
{
   const int local_index = part_quad_space->GlobalToLocal(idx);
   // If global_offsets.Size() == 1 then we'll always go down this path
   if (local_index > -1)
   {
      const int s_offset = part_quad_space->offsets[local_index];
      const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
      values.SetSize(vdim * sl_size);
      values.HostWrite();
      const real_t *q = HostRead() + vdim * s_offset;
      for (int i = 0; i < values.Size(); i++)
      {
         values(i) = *(q++);
      }
   }
   else
   {
      const int s_offset = part_quad_space->global_offsets[idx];
      const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
      values.SetSize(vdim * sl_size);
      values.HostWrite();
      for (int i = 0; i < values.Size(); i++)
      {
         values(i) = default_value;
      }
   }
}

inline void PartialQuadratureFunction::GetValues(int idx, const int ip_num,
                                                  Vector &values)
{
   const int local_index = part_quad_space->GlobalToLocal(idx);
   // If global_offsets.Size() == 1 then we'll always go down this path
   if (local_index > -1)
   {
      const int s_offset = (part_quad_space->offsets[local_index] + ip_num) * vdim;
      values.MakeRef(*this, s_offset, vdim);
   }
   else
   {
      values.Destroy();
      values.SetSize(vdim);
      values.HostWrite();
      for (int i = 0; i < values.Size(); i++)
      {
         values(i) = default_value;
      }
   }
}

inline void PartialQuadratureFunction::GetValues(int idx, const int ip_num,
                                                  Vector &values) const
{
   const int local_index = part_quad_space->GlobalToLocal(idx);
   // If global_offsets.Size() == 1 then we'll always go down this path
   if (local_index > -1)
   {
      const int s_offset = (part_quad_space->offsets[local_index] + ip_num) * vdim;
      const real_t *q = HostRead() + s_offset;
      values.SetSize(vdim);
      values.HostWrite();
      for (int i = 0; i < values.Size(); i++)
      {
         values(i) = *(q++);
      }
   }
   else
   {
      values.Destroy();
      values.SetSize(vdim);
      values.HostWrite();
      for (int i = 0; i < values.Size(); i++)
      {
         values(i) = default_value;
      }
   }
}

inline void PartialQuadratureFunction::GetValues(int idx, DenseMatrix &values)
{
   const int local_index = part_quad_space->GlobalToLocal(idx);
   // If global_offsets.Size() == 1 then we'll always go down this path
   if (local_index > -1)
   {
      const int s_offset = part_quad_space->offsets[local_index];
      const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
      // Make the values matrix memory an alias of the quadrature function memory
      Memory<real_t> &values_mem = values.GetMemory();
      values_mem.Delete();
      values_mem.MakeAlias(GetMemory(), vdim * s_offset, vdim * sl_size);
      values.SetSize(vdim, sl_size);
   }
   else
   {
      const int s_offset = part_quad_space->global_offsets[idx];
      const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
      values.Clear();
      values.SetSize(vdim, sl_size);
      values.HostWrite();
      for (int j = 0; j < sl_size; j++)
      {
         for (int i = 0; i < vdim; i++)
         {
            values(i, j) = default_value;
         }
      }
   }
}

inline void PartialQuadratureFunction::GetValues(int idx,
                                                  DenseMatrix &values) const
{
   const int local_index = part_quad_space->GlobalToLocal(idx);
   // If global_offsets.Size() == 1 then we'll always go down this path
   if (local_index > -1)
   {
      const int s_offset = part_quad_space->offsets[local_index];
      const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
      values.SetSize(vdim, sl_size);
      values.HostWrite();
      const real_t *q = HostRead() + vdim * s_offset;
      for (int j = 0; j < sl_size; j++)
      {
         for (int i = 0; i < vdim; i++)
         {
            values(i, j) = *(q++);
         }
      }
   }
   else
   {
      const int s_offset = part_quad_space->global_offsets[idx];
      const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
      values.Clear();
      values.SetSize(vdim, sl_size);
      values.HostWrite();
      for (int j = 0; j < sl_size; j++)
      {
         for (int i = 0; i < vdim; i++)
         {
            values(i, j) = default_value;
         }
      }
   }
}

inline PartialQuadratureFunction
PartialQuadratureFunctionFactory::Create(PartialQuadratureSpace *qspace,
                                         int vdim,
                                         real_t default_val)
{
   MFEM_VERIFY(qspace != nullptr, "PartialQuadratureSpace cannot be NULL");
   MFEM_VERIFY(vdim >= 1, "Vector dimension must be >= 1");

   // Calculate the size needed for this PartialQuadratureFunction
   const int size = vdim * qspace->GetSize();
   
   // Get pointer to the current position in the memory pool
   real_t *qf_data = memory_pool + memory_offset;
   
   // Advance the memory pool offset
   memory_offset += size;
   
   // Create and return the PartialQuadratureFunction using external data
   return PartialQuadratureFunction(qspace, qf_data, vdim, default_val);
}

inline int PartialQuadratureFunctionFactory::CalculateMemorySize(
   const std::vector<PartialQuadratureSpace*> &qspaces,
   const std::vector<int> &vdims)
{
   MFEM_VERIFY(qspaces.size() == vdims.size(),
               "qspaces and vdims must have the same size");
   
   int total_size = 0;
   for (size_t i = 0; i < qspaces.size(); i++)
   {
      MFEM_VERIFY(qspaces[i] != nullptr,
                  "PartialQuadratureSpace at index " << i << " cannot be NULL");
      MFEM_VERIFY(vdims[i] >= 1,
                  "Vector dimension at index " << i << " must be >= 1");
      total_size += vdims[i] * qspaces[i]->GetSize();
   }
   
   return total_size;
}

inline int PartialQuadratureFunctionFactory::CalculateMemorySize(
   const std::vector<PartialQuadratureSpace*> &qspaces,
   int vdim)
{
   MFEM_VERIFY(vdim >= 1, "Vector dimension must be >= 1");
   
   int total_size = 0;
   for (size_t i = 0; i < qspaces.size(); i++)
   {
      MFEM_VERIFY(qspaces[i] != nullptr,
                  "PartialQuadratureSpace at index " << i << " cannot be NULL");
      total_size += vdim * qspaces[i]->GetSize();
   }
   
   return total_size;
}

} // namespace mfem