// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MARKING_HPP
#define MFEM_MARKING_HPP

#include "mfem.hpp"

namespace mfem
{

// Marking operations for elements, faces, dofs, etc, related to shifted
// boundary and interface methods.
class ShiftedFaceMarker
{
protected:
   ParMesh &pmesh;                    // Mesh whose elements have to be marked.
   ParFiniteElementSpace *pfes_sltn;  // FESpace associated with the solution.

   // Indicates whether cut-cells will be included in assembly.
   const bool include_cut_cell;
   // Indicates whether all the elements have been marked at-least once.
   bool initial_marking_done;

   // Marking of face dofs by using an averaged continuous GridFunction.
   const bool func_dof_marking = true;
   // Alternative implementation of ListShiftedFaceDofs().
   void ListShiftedFaceDofs2(const Array<int> &elem_marker,
                             Array<int> &sface_dof_list) const;

private:
   int level_set_index;

public:
   /// Element type related to shifted boundaries (not interfaces).
   /// For more than 1 level-set, we set the marker to CUT+level_set_index
   /// to discern between different level-sets.
   enum SBElementType {INSIDE = 0, OUTSIDE = 1, CUT = 2};

   ShiftedFaceMarker(ParMesh &pm, ParFiniteElementSpace &pfes,
                     bool include_cut_cell_)
      : pmesh(pm), pfes_sltn(&pfes),
        include_cut_cell(include_cut_cell_), initial_marking_done(false),
        level_set_index(0) { }

   /// Mark all the elements in the mesh using the @a SBElementType.
   /// A point is considered inside when the level set function is positive.
   /// Assumes the ExchangeFaceNbrData() has been called for pmesh, ls_func.
   void MarkElements(const ParGridFunction &ls_func, Array<int> &elem_marker);

   /// List dofs associated with the surrogate boundary.
   /// If @a include_cut_cell = false, the surrogate boundary includes faces
   /// between elements cut by the true boundary and the elements that are
   /// located inside the true domain.
   /// If @a include_cut_cell = true, the surrogate boundary is the faces
   /// between elements outside the true domain and the elements cut by the true
   /// boundary.
   void ListShiftedFaceDofs(const Array<int> &elem_marker,
                            Array<int> &sface_dof_list) const;

   /// List the dofs that will be inactive for the computation on the surrogate
   /// domain. This includes dofs for the elements located outside the true
   /// domain (and optionally, for the elements cut by the true boundary, if
   /// @a include_cut_cell = false) minus the dofs that are located on the
   /// surrogate boundary.
   void ListEssentialTDofs(const Array<int> &elem_marker,
                           const Array<int> &sface_dof_list,
                           Array<int> &ess_tdof_list,
                           Array<int> &ess_shift_bdr) const;
};

} // namespace mfem

#endif
