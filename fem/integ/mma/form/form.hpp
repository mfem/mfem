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

    Integrator-agnostic: fields, plans, pipelines. Physics QFns live under
    fem/integ/ only (never under mma/).

    Author convention:
      - trial:  const eval_t &  /  const grad_t<Dim> &
      - test:   eval_t &        /  grad_t<Dim> &
      - coeff:  real_t or tensor (point-local; no q/e)
      - y = d * u;   or   y = A * u;

    Layout:
      fields.hpp | plan.hpp | pipeline.hpp   — simplex dense Apply / ApplyLF
      apply_tensor.hpp                       — tensor-product ApplyTensor
      tensor_eval / tensor_grad / tensor_metric — sum-fact engines

    Authoring guide: form/README.md
    Design: docs/design/mma-declarative-kernels.md
*/

#include "fields.hpp" // IWYU pragma: export
#include "plan.hpp" // IWYU pragma: export
#include "pipeline.hpp" // IWYU pragma: export
#include "apply_tensor.hpp" // IWYU pragma: export
