#ifndef PALACE_MESH_IO_HPP
#define PALACE_MESH_IO_HPP

#include <iostream>
#include <string>

namespace palace::mesh
{

	//
	// Functions for mesh format conversion to Gmsh format, which is supported by MFEM. In both
	// cases, the user should configure the buffer for the desired floating point
	// format/precision for writing node coordinates.
	//

	// Convert a binary or ASCII COMSOL (.mphbin/.mphtxt) mesh to Gmsh v2.2.
	void ConvertMeshComsol(const std::string& filename, std::ostream& buffer);

	// Convert an ASCII NASTRAN (.nas/.bdf) mesh to Gmsh v2.2.
	void ConvertMeshNastran(const std::string& filename, std::ostream& buffer);

}  // namespace palace::mesh

#endif  // PALACE_MESH_IO_HPP
