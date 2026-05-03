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

#include <cassert>
#include <cstddef>
#include <utility>

#include "fem/dfem/integrator_ctx.hpp"

// #include "../qf_local_devices.hpp"

#include "../../../util.hpp"

// #include "fem/kernels3d.hpp"
// namespace ker = mfem::kernels::internal;
// namespace low = mfem::kernels::internal::low;
// #include "fem/kernel_dispatch.hpp"

namespace mfem::future
{

///////////////////////////////////////////////////////////////////////////////
namespace LocalQFDevicesPolyImpl
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   std::size_t ninputs = std::tuple_size_v<inputs_t>,
   std::size_t noutputs = std::tuple_size_v<outputs_t>>
class Action
{
   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::array<size_t, ninputs> input_to_infd;
   std::array<size_t, noutputs> output_to_outfd;

   std::array<FieldBasis, ninputs> input_bases;
   std::array<FieldBasis, noutputs> output_bases;

   // std::array<const DofToQuad*, ninputs> input_dtq_maps;
   // std::array<const DofToQuad*, noutputs> output_dtq_maps;

   // using local_restriction_callback_t =
   //    std::function<void(std::vector<Vector> &,
   //                       const std::vector<Vector> &,
   //                       std::vector<Vector> &)>;

   // local_restriction_callback_t &restriction_cb;
   // const DofToQuadMap input_dtq_maps;
   // const int num_entities;
   // const ThreadBlocks thread_blocks;
   // const Array<int> &attributes;
   // const Array<int> *elem_attributes;
   // // refs
   // std::vector<Vector> &fields_e;
   // Vector &residual_e;
   // std::function<void(Vector &, Vector &)> &output_restriction_transpose;

public:

   ////////////////////////////////////////////////////////
   Action() = delete;

   Action(IntegratorContext ctx,
          qfunc_t qfunc,
          inputs_t inputs,
          outputs_t outputs) :
      ctx(ctx),
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs)
      // restriction_cb(*ctx.local.local_restriction_callback),
      // input_dtq_maps(ctx.local.input_dtq_maps),
      // num_entities(ctx.local.num_entities),
      // thread_blocks(ctx.local.thread_blocks),
      // attributes(*ctx.local.attributes),
      // elem_attributes(ctx.local.elem_attributes),
      // fields_e(*ctx.local.local_fields_e),
      // residual_e(*ctx.local.local_residual_e),
      // output_restriction_transpose(*ctx.local.output_restriction_transpose)
   {
      NVTX_MARK_FUNCTION;
      create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);

      check_consistency(inputs, input_to_infd, ctx.infds);
      check_consistency(outputs, output_to_outfd, ctx.outfds);

      create_fieldbases(inputs, input_to_infd, ctx.infds, ctx.ir, input_bases);
      create_fieldbases(outputs, output_to_outfd, ctx.outfds, ctx.ir, output_bases);

      if (!ctx.local.use_kernel_specializations) { return; }
#ifdef MFEM_ADD_SPECIALIZATIONS
      NewActionCallbackKernels::template Specialization<3>::Add(); // 1
      NewActionCallbackKernels::template Specialization<4>::Add(); // 2
      NewActionCallbackKernels::template Specialization<5>::Add(); // 3
      NewActionCallbackKernels::template Specialization<6>::Add(); // 4
      NewActionCallbackKernels::template Specialization<7>::Add(); // 5
      NewActionCallbackKernels::template Specialization<8>::Add(); // 6
#endif
   }

   void operator()(const std::vector<Vector *> &,
                   std::vector<Vector *> &) const
   {
      NVTX_MARK_FUNCTION;
      assert(false && "Not implemented 🔥🔥🔥");
   }
};

} // namespace LocalQFDevicesPolyImpl

} // namespace mfem::future
