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
   DenseTensor NbrDof;

   int dim, numBdrs, numFaceDofs, numSubcells, numDofsSubcell;

   DofInfo(FiniteElementSpace *fes_sltn, FiniteElementSpace *fes_bounds)
      : mesh(fes_sltn->GetMesh()), fes(fes_sltn),
        x_min(fes_bounds), x_max(fes_bounds)
   {
      dim = mesh->Dimension();

      int n = fes->GetVSize();
      int ne = mesh->GetNE();

      xi_min.SetSize(n);
      xi_max.SetSize(n);
      xe_min.SetSize(ne);
      xe_max.SetSize(ne);

      ExtractBdrDofs(fes->GetFE(0)->GetOrder(),
                     fes->GetFE(0)->GetGeomType(), BdrDofs);
      numFaceDofs = BdrDofs.Height();
      numBdrs = BdrDofs.Width();

      FillNeighborDofs();    // Fill NbrDof.
      FillSubcell2CellDof(); // Fill Sub2Ind.
   }

   // Computes the admissible interval of values for each DG dof from the values
   // of all elements that feature the dof at its physical location.
   // Assumes that xe_min and xe_max are already computed.
   void ComputeBounds()
   {
      FiniteElementSpace *fesCG = x_min.FESpace();
#ifdef MFEM_USE_MPI
      GroupCommunicator &gcomm = fesCG->GroupComm();
#endif
      Array<int> dofsCG;

      // Form min/max at each CG dof, considering element overlaps.
      x_min =   std::numeric_limits<double>::infinity();
      x_max = - std::numeric_limits<double>::infinity();
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         x_min.FESpace()->GetElementDofs(i, dofsCG);
         for (int j = 0; j < dofsCG.Size(); j++)
         {
            x_min(dofsCG[j]) = std::min(x_min(dofsCG[j]), xe_min(i));
            x_max(dofsCG[j]) = std::max(x_max(dofsCG[j]), xe_max(i));
         }
      }
      Array<double> minvals(x_min.GetData(), x_min.Size()),
            maxvals(x_max.GetData(), x_max.Size());

#ifdef MFEM_USE_MPI
      gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
      gcomm.Bcast(minvals);
      gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
      gcomm.Bcast(maxvals);
#endif

      // Use (x_min, x_max) to fill (xi_min, xi_max) for each DG dof.
      const TensorBasisElement *fe_cg =
         dynamic_cast<const TensorBasisElement *>(fesCG->GetFE(0));
      const Array<int> &dof_map = fe_cg->GetDofMap();
      const int ndofs = dof_map.Size();
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         x_min.FESpace()->GetElementDofs(i, dofsCG);
         for (int j = 0; j < dofsCG.Size(); j++)
         {
            xi_min(i*ndofs + j) = x_min(dofsCG[dof_map[j]]);
            xi_max(i*ndofs + j) = x_max(dofsCG[dof_map[j]]);
         }
      }
   }

   ~DofInfo() { }
};

#endif
