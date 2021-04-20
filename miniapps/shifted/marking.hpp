// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
   ParMesh &pmesh;
   ParGridFunction &ls_func;
   ParFiniteElementSpace &pfes_sltn;
   bool include_cut_cell;

   // Marking of face dofs by using an averaged continuous GridFunction.
   const bool func_dof_marking = false;

   // Alternative implementation of ListShiftedFaceDofs().
   void ListShiftedFaceDofs2(const Array<int> &elem_marker,
                             Array<int> &sface_dof_list) const;

public:
   // Element type related to shifted boundaries (not interfaces).
   enum SBElementType {INSIDE, OUTSIDE, CUT};

   ShiftedFaceMarker(ParMesh &pm, ParGridFunction &ls,
                     ParFiniteElementSpace &space_sltn,
                     bool include_cut_cell_)
      : pmesh(pm), ls_func(ls), pfes_sltn(space_sltn),
        include_cut_cell(include_cut_cell_) { }

   void MarkElements(Array<int> &elem_marker) const;

   void ListShiftedFaceDofs(const Array<int> &elem_marker,
                            Array<int> &sface_dof_list) const;

   // Related to shifted boundary methods, where inactive dofs must be
   // eliminated from the system, based on the element markers.
   void ListEssentialTDofs(const Array<int> &elem_marker,
                           const Array<int> &sface_dof_list,
                           Array<int> &ess_tdof_list,
                           Array<int> &ess_shift_bdr) const;
};

} // namespace mfem

#endif
