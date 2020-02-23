// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

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
