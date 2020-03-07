// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MGBILINEARFORM
#define MFEM_MGBILINEARFORM

#include "../linalg/multigrid.hpp"
#include "bilinearform.hpp"
#include "spacehierarchy.hpp"

namespace mfem
{

class MultigridBilinearForm : public MultigridOperator
{
protected:
   Array<BilinearForm*> bfs;
   Array<Array<int>*> essentialTrueDofs;

public:
   /// Empty constructor
   MultigridBilinearForm();

   /// Constructor for a multigrid bilinear form for a given SpaceHierarchy and
   /// bilinear form. Uses Chebyshev accelerated smoothing. Only supports
   /// partial assembly bilinear forms.
   /// At the moment, only the DomainIntegrators of \p bf are copied.
   MultigridBilinearForm(SpaceHierarchy& spaceHierarchy, BilinearForm& bf,
                         Array<int>& ess_bdr, int chebyshevOrder = 2);

   virtual ~MultigridBilinearForm();
};

} // namespace mfem
#endif
