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
#include <array>
#include <variant>

namespace mfem
{

/// Class representing an array of Vectors with generally different sizes.
/** This class is similar to BlockVector with the following two main
    differences:
    - the data for the individual Vector blocks does not need to be part of one
      big contiguous memory allocation;
    - this class does not inherit from class Vector (as a consequence of the
      first bullet).

    Internally, each Vector block is represented as either:
    - (default) a Vector object constructed and owned by this class; this
      object, in turn, as any Vector object, can own its Memory allocation or
      refer to a sub-Memory of another Memory object; or
    - a pointer to an externally allocated Vector or classes derived from
      Vector. */
class MultiVector
{
private:
   std::vector<std::variant<Vector,Vector*>> blocks;

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
       Vector @a base.

       With this constructor, the Memory flags of @a base and of the individual
       Vector blocks may need to be explicitly synchronized when data is moved
       between host and device. */
   MultiVector(Vector &base, const Array<int> &vector_sizes);

   /** @brief Construct a MultiVector referencing multiple Vectors given as
       arguments.

       The VectorTypes reference arguments are expected to be static_cast-able
       to (Vector &) which is the case if the types are derived from Vector,
       e.g. HypreParVector, GridFunction, etc.

       With this constructor, operations on individual Vector blocks are
       performed directly on the objects @a vs. In particular, there is no need
       to synchronize the Memory flags of @a vs and the ones of the individual
       Vector blocks when data is moved between host and device. */
   template <typename... VectorTypes,
             std::enable_if_t<
                std::conjunction_v<
                   std::is_convertible<VectorTypes&,Vector&>...>, bool> = true>
   MultiVector(VectorTypes &...vs) { MakeRef(vs...); }

   /// Return the number of Vectors in the MultiVector.
   int NumBlocks() const { return blocks.size(); }

   /** @brief Set the number of Vectors in the MultiVector. Existing Vector
       blocks will remain unmodified. New Vector blocks will be default
       initialized, i.e. they all have size zero. */
   void SetNumBlocks(int num_blocks) { blocks.resize(num_blocks); }

   /// Read-write access to the i-th Vector.
   inline Vector &operator[](int i);

   /// Read-only access to the i-th Vector.
   inline const Vector &operator[](int i) const;

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
       Vector @a base.

       After calling this method, the Memory flags of @a base and of the
       individual Vector blocks may need to be explicitly synchronized when data
       is moved between host and device.*/
   void MakeRef(Vector &base, const Array<int> &vector_sizes);

   /** @brief Update the @a i-th MultiVector block to reference data within the
       given monolithic Vector @a base at the given @a offset and with the given
       @a size.

       After calling this method, the Memory flags of @a base and of the @a i-th
       Vector block may need to be explicitly synchronized when data is moved
       between host and device.*/
   inline void MakeRef(int i, Vector &base, int offset, int size)
   {
      blocks[i].emplace<0>(base, offset, size);
   }

   /** @brief Update the MultiVector to reference multiple Vectors given as
       arguments.

       The VectorTypes reference arguments are expected to be static_cast-able
       to (Vector &) which is the case if the types are derived from Vector,
       e.g. HypreParVector, GridFunction, etc.

       After calling this method, operations on individual Vector blocks are
       performed directly on the objects @a vs. In particular, there is no need
       to synchronize the Memory flags of @a vs and the ones of the individual
       Vector blocks when data is moved between host and device. */
   template <typename... VectorTypes,
             std::enable_if_t<
                std::conjunction_v<
                   std::is_convertible<VectorTypes&,Vector&>...>, bool> = true>
   inline void MakeRef(VectorTypes &...vs);

   /** @brief Update the @a i-th MultiVector block to reference the given
       Vector @a v.

       After calling this method, operations on the @a i-th Vector block are
       performed directly on the Vector @a v. In particular, there is no need
       to synchronize the Memory flags of @a v and the ones of the @a i-th
       Vector blocks when data is moved between host and device. */
   inline void MakeRef(int i, Vector &v) { blocks[i] = &v; }
};

// Inline and template methods

inline Vector &MultiVector::operator[](int i)
{
   auto &bi = blocks[i];
   return (bi.index() == 0) ? std::get<0>(bi) : *std::get<1>(bi);
}

inline const Vector &MultiVector::operator[](int i) const
{
   auto &bi = blocks[i];
   return (bi.index() == 0) ? std::get<0>(bi) : *std::get<1>(bi);
}

template <typename... VectorTypes,
          std::enable_if_t<
             std::conjunction_v<
                std::is_convertible<VectorTypes&,Vector&>...>, bool>>
inline void MultiVector::MakeRef(VectorTypes &...vs)
{
   blocks.resize(sizeof...(vs));
   if constexpr (sizeof...(vs) > 0)
   {
      const std::array vs_p{&static_cast<Vector&>(vs)...};
      for (std::size_t i = 0; i < sizeof...(vs); i++)
      {
         blocks[i] = vs_p[i];
      }
   }
}

} // namespace mfem

#endif // MFEM_MULTIVECTOR_HPP
