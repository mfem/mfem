#ifndef MFEM_DOF_INFO
#define MFEM_DOF_INFO

#include "mfem.hpp"

using namespace mfem;

class DofInfo
{
private:
   Mesh *mesh;
   FiniteElementSpace *fes;

	void ExtractBdrDofs(int p, Geometry::Type gtype, DenseMatrix &dofs);
	
	// NOTE: The mesh is assumed to consist of segments, quads or hexes.
   // NOTE: This approach will not work for meshes with hanging nodes.
   void FillNeighborDofs();

   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   void FillSubcell2CellDof();

public:
   GridFunction x_min, x_max;

   Vector xi_min, xi_max; // min/max values for each dof
   Vector xe_min, xe_max; // min/max values for each element

   DenseMatrix BdrDofs, Sub2Ind;
   DenseTensor NbrDofs;

   int dim, NumBdrs, NumFaceDofs, numSubcells, numDofsSubcell;

   DofInfo(FiniteElementSpace *fes_sltn, FiniteElementSpace *fes_bounds);
   ~DofInfo() { }
   
   // Computes the admissible interval of values for each DG dof from the values
   // of all elements that feature the dof at its physical location.
   // Assumes that xe_min and xe_max are already computed.
   void ComputeBounds();
};

#endif
