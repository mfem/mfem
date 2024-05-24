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

#ifndef MFEM_HYBRIDIZATION_EXT
#define MFEM_HYBRIDIZATION_EXT

#include "../config/config.hpp"
#include "../general/array.hpp"

namespace mfem
{

class HybridizationExtension
{
protected:
   class Hybridization &h; ///< The associated Hybridization object.=
   /// Construct the constraint matrix.
   void ConstructC();
public:
   /// Constructor.
   HybridizationExtension(class Hybridization &hybridization_);
   /// Prepare for assembly; form the constraint matrix.
   void Init(const Array<int> &ess_tdof_list);
};

}

#endif
