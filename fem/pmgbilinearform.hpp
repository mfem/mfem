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

#ifndef MFEM_PMGBILINEARFORM
#define MFEM_PMGBILINEARFORM

#include "../linalg/multigrid.hpp"
#include "spacehierarchy.hpp"

#include "mgbilinearform.hpp"
#include "pbilinearform.hpp"

namespace mfem
{

class ParMultigridBilinearForm : public MultigridBilinearForm
{
private:
   ParMesh* pmesh_lor;
   H1_FECollection* fec_lor;
   ParFiniteElementSpace* fespace_lor;
   ParBilinearForm* a_lor;

public:
   /// Constructor for a multigrid bilinear form for a given SpaceHierarchy and
   /// bilinear form. Uses Chebyshev accelerated smoothing.
   /// At the moment, only the DomainIntegrators of \p bf are copied.
   ParMultigridBilinearForm(ParSpaceHierarchy& spaceHierarchy,
                            ParBilinearForm& bf, Array<int>& ess_bdr,
                            int chebyshevOrder = 2);

   virtual ~ParMultigridBilinearForm();
};

} // namespace mfem
#endif
