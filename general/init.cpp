// Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"
#include "init.hpp"

#ifdef MFEM_USE_OCCA
#  include <occa.hpp>
#endif

namespace mfem
{
   void Init()
   {
     InitOcca();
   }

   void InitOcca()
   {
#ifdef MFEM_USE_OCCA
     occa::io::addLibraryPath(
       "mfem",
       MFEM_INSTALL_DIR "/kernels"
     );
#endif
   }
}
