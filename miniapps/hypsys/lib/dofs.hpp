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

   DenseMatrix BdrDofs, Sub2Ind, SubcellCross, Loc2Multiindex;
   DenseTensor NbrDofs; // Negative values correspond to the boundary attributes.

   int dim, NumBdrs, NumFaceDofs, numSubcells, numDofsSubcell;
   Array<int> DofMapH1;

   DofInfo(FiniteElementSpace *fes_);

   virtual ~DofInfo() { }

   // NOTE: This approach will not work for meshes with hanging h- or p-nodes.
   void FillNeighborDofs();

   void FillBdrDofs();
   void FillSubcell2CellDof();
   void FillSubcellCross();

   // The following two routines work only for TRIANGLES.
   void FillLoc2Multiindex();
   int GetLocFromMultiindex(int p, const Vector &a) const;

   // Auxiliary routines.
   int GetLocalFaceDofIndex3D(int loc_face_id, int face_orient,
                              int face_dof_id, int face_dof1D_cnt);

   int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                            int face_dof_id, int face_dof1D_cnt);
};

#endif
