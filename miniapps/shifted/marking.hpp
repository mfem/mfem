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

public:
   // Element type related to shifted boundaries (not interfaces).
   enum SBElementType {INSIDE, OUTSIDE, CUT};

   ShiftedFaceMarker(ParMesh &pm, ParGridFunction &ls)
      : pmesh(pm), ls_func(ls) { }

   void MarkElements(Array<int> &elem_marker) const;
};

} // namespace mfem

#endif
