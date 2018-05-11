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
#ifndef MFEM_BACKENDS_RAJA_LINALG_HPP
#define MFEM_BACKENDS_RAJA_LINALG_HPP

namespace mfem
{

namespace raja
{

namespace linalg
{

  // ***************************************************************************
  double dot(memory vec1, memory vec2);

  // ***************************************************************************
  template <typename T>
  inline void operator_eq(raja::memory vec, T value){
    const std::size_t sz = vec.size();
    vector_op_eq(sz/sizeof(T),value,(double*)vec.ptr());
  } 
  
} // namespace mfem::raja::linalg

} // namespace mfem::raja

} // namespace mfem

#endif // MFEM_BACKENDS_RAJA_LINALG_HPP
