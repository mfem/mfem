#ifndef MFEM_LOOP_LENGTH_HPP
#define MFEM_LOOP_LENGTH_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../fem/pfespace.hpp"
#include <unordered_map>

namespace mfem
{

/// @brief Compute the length of the boundary loop portion owned by the current processor
///
/// This function calculates the sum of the lengths of all boundary edges owned by
/// the current processor that are part of the specified boundary attributes.
///
/// @param[in] pmesh Parallel mesh
/// @param[in] dof_to_edge Map from DoFs to edge indices
/// @param[in] bdr_attr_marker Array marking which boundary attributes to include
/// @return The length of the boundary loop portion owned by this processor
double ComputeBoundaryLoopLength(ParMesh* pmesh, const std::unordered_map<int, int>& dof_to_edge);

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LOOP_LENGTH_HPP