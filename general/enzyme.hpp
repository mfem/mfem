// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#ifndef MFEM_ENZYME_HPP
#define MFEM_ENZYME_HPP

#ifdef MFEM_USE_ENZYME
/*
 * Variables prefixed with enzyme_* or function types prefixed with __enzyme_*,
 * are variables which will get preprocessed in the LLVM intermediate
 * representation when the Enzyme LLVM plugin is loaded. See the Enzyme
 * documentation (https://enzyme.mit.edu) for more information.
 */

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

template <typename return_type, typename... Args>
return_type __enzyme_autodiff(Args...);

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);
#endif
#endif
