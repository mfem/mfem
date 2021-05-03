#ifndef MFEM_L2P_MESH_UTILS_HPP
#define MFEM_L2P_MESH_UTILS_HPP

#include "../fem/fem.hpp"

namespace mfem {
Element *NewElem(const int type, const int *cells_data, const int attr);
void Finalize(Mesh &mesh, const bool generate_edges);

void MaxCol(const DenseMatrix &mat, double *vec, bool include_vec_elements);
void MinCol(const DenseMatrix &mat, double *vec, bool include_vec_elements);

int MaxVertsXFace(const int type);
double Sum(const DenseMatrix &mat);
} // namespace mfem

#endif // MFEM_L2P_MESH_UTILS_HPP
