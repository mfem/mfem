#ifndef MFEM_L2P_MESH_UTILS_HPP
#define MFEM_L2P_MESH_UTILS_HPP 

#include "../fem/fem.hpp"

namespace mfem {
	Element * NewElem(const int type, const int *cells_data, const int attr);
	void Finalize(Mesh &mesh, const bool generate_edges);
}

#endif //MFEM_L2P_MESH_UTILS_HPP
