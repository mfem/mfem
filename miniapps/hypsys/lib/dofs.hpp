#ifndef HYPSYS_DOFS
#define HYPSYS_DOFS

#include "../../../mfem.hpp"

using namespace std;
using namespace mfem;

// NOTE: The mesh is assumed to consist of segments, triangles quads or hexes.
class DofInfo
{
public:
   Mesh *mesh;
   FiniteElementSpace *fes;

   GridFunction x_min, x_max;

   Vector xi_min, xi_max; // min/max values for each dof

   DenseMatrix BdrDofs, Sub2Ind, SubcellCross, Loc2Multiindex, ClosestNbrs;
   DenseTensor NbrDofs; // Negative values correspond to the boundary attributes.

   int dim, NumBdrs, NumFaceDofs, numSubcells, numDofsSubcell, NumEq;
   Array<int> DofMapH1;

   DofInfo(FiniteElementSpace *fes_sltn, FiniteElementSpace *fes_bounds, int NumEq_);

   ~DofInfo() { }

   // Computes the admissible interval of values for each DG dof from the
   // values of all elements that feature the dof at its physical location.
   // Assumes that xe_min and xe_max are already computed.
   void ComputeBounds(const Vector &x);

   // NOTE: This approach will not work for meshes with hanging nodes.
   void FillNeighborDofs();

   void ExtractBdrDofs(); // TODO rename
   void FillSubcell2CellDof();
   void FillClosestNbrs();
   void FillSubcellCross();

   // The following three routines work only for TRIANGLES.
   void FillTriangleDofMap(int p);
   void FillLoc2Multiindex(int p);
   int GetLocFromMultiindex(int p, const Vector &a) const;

   // Auxiliary routines.
   int GetLocalFaceDofIndex3D(int loc_face_id, int face_orient,
                              int face_dof_id, int face_dof1D_cnt);

   int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                            int face_dof_id, int face_dof1D_cnt);
};

#endif
