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

#ifndef MFEM_DARCY_FUNCTIONALS_HDG
#define MFEM_DARCY_FUNCTIONALS_HDG

#include "../../config/config.hpp"
#include "../gridfunc.hpp"

namespace mfem
{

/** @brief Functionals of a hybridized Darcy solution evaluated from the
    numerical trace.

    The quantity of interest in a Darcy-like problem is often not a field but a
    number: the flow through a surface. Evaluating it by integrating the
    element flux `q_h` is an after-the-fact diagnostic -- `q_h` is not normally
    continuous, so the answer depends on which side of the surface it is read
    from, and for a problem whose answer *is* a small flux that difference can
    be the whole result.

    The total flux is the object to integrate instead. It is what
    DarcyForm::ReconstructTotalFlux() produces, in a Raviart-Thomas space whose
    face degrees of freedom are those of the trace, so its normal component is
    single valued by construction and the integral below is a pairing against
    the trace space rather than a pointwise evaluation.

    **That distinction is not pedantry.** With a solution-dependent
    stabilization the trace equation does not force `q_h + tau(u_h - lambda) n`
    to be single valued pointwise at all -- it forces only the L2 projection of
    its normal component into the trace space to be, which is enough for local
    conservation and is what the method actually delivers (Nguyen, Peraire &
    Cockburn, JCP 228 (2009), immediately after eq (5)). Integrating the
    reconstructed total flux respects that; integrating `q_h + tau(u_h-lambda)n`
    pointwise does not, and the conservation identity below then fails to hold
    to round-off. */

/** @brief Net outward flux of @a ut through the boundary of the element
    subdomain marked by @a elem_marker.

    @a ut must be normally continuous -- the grid function
    DarcyForm::ReconstructTotalFlux() fills. @a elem_marker is indexed by
    element and is nonzero on the subdomain; the surface integrated over is the
    set of faces with the subdomain on exactly one side, oriented outward from
    it, and the mesh boundary counts as outside.

    Taking the subdomain rather than the surface as the argument is what makes
    the orientation unambiguous, and it is also the form the conservation
    identity wants: for a steady problem the value returned equals the integral
    of the source over the same subdomain, to round-off. That identity is the
    sharpest available test of the whole assembly, because every part of it --
    the local solves, the trace solve and the flux reconstruction -- has to be
    right for the two numbers to agree. */
real_t ComputeOutwardFlux(const GridFunction &ut, const Array<int> &elem_marker,
                          int ir_order = -1);

/** @brief Flux of @a ut through the mesh boundary faces whose boundary
    attribute is marked in @a bdr_attr_marker, with the normal pointing out of
    the domain.

    The common case of the above, when the surface of interest is part of the
    domain boundary and no subdomain needs naming. */
real_t ComputeBoundaryFlux(const GridFunction &ut,
                           const Array<int> &bdr_attr_marker,
                           int ir_order = -1);

}

#endif
