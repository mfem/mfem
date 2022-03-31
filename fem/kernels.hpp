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

#ifndef MFEM_FEM_KERNELS_HPP
#define MFEM_FEM_KERNELS_HPP

#include "../config/config.hpp"
#include "../linalg/dtensor.hpp"
#include "../general/device.hpp"

// Experimental helper functions for MFEM_FORALL FEM kernels
// For the 2D functions, NBZ should be tied to '1' for now

#include "kernels/eval_2D.hpp"
#include "kernels/eval_3D.hpp"
#include "kernels/eval_fast_2D.hpp"
#include "kernels/eval_fast_3D.hpp"

#include "kernels/grad_2D.hpp"
#include "kernels/grad_3D.hpp"
#include "kernels/grad_fast_2D.hpp"
#include "kernels/grad_fast_3D.hpp"

#include "kernels/load.hpp"
#include "kernels/pool.hpp"
#include "kernels/pull.hpp"
#include "kernels/push.hpp"

#endif // MFEM_FEM_KERNELS_HPP
