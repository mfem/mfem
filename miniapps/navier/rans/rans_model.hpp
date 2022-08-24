// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NAVIER_RANS_MODEL_HPP
#define MFEM_NAVIER_RANS_MODEL_HPP

#include "mfem.hpp"

namespace mfem
{
namespace navier
{

class RANSModel : public TimeDependentOperator {};

} // namespace navier

} // namespace mfem

#endif