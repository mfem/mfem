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

__asm__(
   ".data\n"
   ".globl xfl_mfem_hpp,_xfl_mfem_hpp\n"
   ".globl xfl_mfem_hpp_size,_xfl_mfem_hpp_size\n"
   ".align 4\n"
   "xfl_mfem_hpp:\n_xfl_mfem_hpp:\n"
   ".incbin \"${CMAKE_CURRENT_SOURCE_DIR}/xfl_mfem.hpp\"\n"
   ".align 4\n"
   "xfl_mfem_hpp_end:\n_xfl_mfem_hpp_end:\n"
   ".align 4\n"
   "xfl_mfem_hpp_size:\n_xfl_mfem_hpp_size:\n"
   ".long xfl_mfem_hpp_end - xfl_mfem_hpp\n"
   ".align 4\n"
   ".text\n"
);

