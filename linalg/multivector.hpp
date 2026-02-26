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

#ifndef MFEM_MULTIVECTOR_HPP
#define MFEM_MULTIVECTOR_HPP

#include "../general/array.hpp"
#include "vector.hpp"
#include <vector>

namespace mfem
{

/// Class representing an array of Vectors with generally different sizes.
/** This class is similar to BlockVector with the following two main
    differences:
    - the data for the individual Vector blocks does not need to be part of one
      big contiguous memory allocation;
    - this class does not inherit from class Vector (as a consequence of the
      first bullet). */
class MultiVector
{
private:
   std::vector<Vector> blocks;

public:
   /// Create an empty MultiVector with zero blocks.
   MultiVector() = default;

   /** @brief Create a MultiVector with @a num_blocks blocks. The individual
       Vector blocks are default initialized, i.e. they all have size zero. */
   MultiVector(int num_blocks)
      : blocks(num_blocks) { }

   /** @brief Construct a MultiVector with number of blocks and individual block
       Vector sizes given by @a vector_sizes.

       @note The memory of the individual Vector blocks is NOT initialized. */
   MultiVector(const Array<int> &vector_sizes);

   /** @brief Construct a MultiVector with number of blocks and individual block
       Vector sizes given by @a vector_sizes. All Vector blocks use the
       MemoryType @a mt.

       @note The memory of the individual Vector blocks is NOT initialized. */
   MultiVector(const Array<int> &vector_sizes, MemoryType mt);

   /** @brief Construct a MultiVector referencing data within a given monolithic
       Vector @a base. */
   MultiVector(Vector &base, const Array<int> &vector_sizes);

   /** @brief Construct a MultiVector referencing the data of multiple Vectors
       given as arguments.

       The VectorTypes reference arguments are expected to be static_cast-able
       to (Vector &) which is the case if the types are derived from Vector,
       e.g. HypreParVector, GridFunction, etc. */
   template <typename... VectorTypes>
   MultiVector(VectorTypes &...vs) { MakeRef(vs...); }

   /// Read-write access to the i-th Vector.
   Vector &operator[](int i) { return blocks[i]; }

   /// Read-only access to the i-th Vector.
   const Vector &operator[](int i) const { return blocks[i]; }

   /** @brief Update the MultiVector according to the given @a vector_sizes.

       This method can be used to add or remove blocks. The individual Vector
       sizes are updated using the method Vector::SetSize(int). */
   void SetSizes(const Array<int> &vector_sizes);

   /** @brief Update the MultiVector according to the given @a vector_sizes and
       MemoryType @a mt.

       This method can be used to add or remove blocks. The individual Vector
       sizes and MemoryType are updated using the method
       Vector::SetSize(int, MemoryType). */
   void SetSizes(const Array<int> &vector_sizes, MemoryType mt);

   /** @brief Update the MultiVector to reference data within a given monolithic
       Vector @a base. */
   void MakeRef(Vector &base, const Array<int> &vector_sizes);

   /** @brief Update the MultiVector to reference the data of multiple Vectors
       given as arguments.

       The VectorTypes reference arguments are expected to be static_cast-able
       to (Vector &) which is the case if the types are derived from Vector,
       e.g. HypreParVector, GridFunction, etc. */
   template <typename... VectorTypes>
   void MakeRef(VectorTypes &...vs)
   {
      blocks.resize(sizeof...(vs));
      if constexpr (sizeof...(vs) > 0)
      {
         const std::array vs_p{&static_cast<Vector&>(vs)...};
         for (std::size_t i = 0; i < sizeof...(vs); i++)
         {
            blocks[i].MakeRef(*vs_p[i], 0, vs_p[i]->Size());
         }
      }
   }
};

} // namespace mfem

#endif // MFEM_MULTIVECTOR_HPP
