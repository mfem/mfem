// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

/** @file form.hpp
    Generic MMA Apply form layer (pointwise QFns on future::tensor).

    Integrator-agnostic: fields, plans, simplex/tensor apply. Physics QFns live under
    fem/integ/ only (never under mma/).

    Author convention:
      - trial:  const eval_t &  /  const grad_t<Dim> &
      - test:   eval_t &        /  grad_t<Dim> &
      - coeff:  real_t or tensor (point-local; no q/e)
      - y = d * u;   or   y = A * u;

    Layout:
      fields.hpp | plan.hpp | simplex.hpp — simplex dense Apply / ApplyLF
      tensors.hpp                          — tensor-product ApplyTensor + engines

    Authoring guide: form/README.md
    Design: docs/design/mma-declarative-kernels.md
*/

#include "fields.hpp" // IWYU pragma: export
#include "plan.hpp" // IWYU pragma: export
#include "simplex.hpp" // IWYU pragma: export
#include "tensors.hpp" // IWYU pragma: export
