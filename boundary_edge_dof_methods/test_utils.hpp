// Test utility functions for boundary edge DOF testing
#pragma once

#include "mfem.hpp"
#include <vector>

using namespace mfem;

/**
 * @brief Create a test mesh with two tetrahedra sharing a triangular face
 * 
 * @param orientation Face orientation (1, 3, or 5)
 * @param add_extbdr Whether to add external boundary elements
 * @return Mesh Test mesh with oriented triangular face
 */
Mesh OrientedTriFaceMesh(int orientation, bool add_extbdr);

/**
 * @brief Create a refined mesh with specific tetrahedral pair marked as boundary
 * 
 * @param orientation Face orientation (1, 3, or 5)
 * @param pair_index Index of tetrahedral pair to mark (0-119)
 * @return Mesh* Pointer to refined mesh with marked boundary
 */
Mesh* RefinedTetPairMesh(int orientation, int pair_index);

/**
 * @brief Generate all possible partitionings of elements among processors
 * 
 * @param n_elements Number of mesh elements
 * @param num_procs Number of processors
 * @param all_partitionings Output vector of all possible partitionings
 */
void GeneratePartitionings(int n_elements, int num_procs, 
                          std::vector<std::vector<int>>& all_partitionings);