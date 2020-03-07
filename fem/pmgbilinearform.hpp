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
