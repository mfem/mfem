#pragma once

#include "../../util.hpp"
#include "../../../integrator_ctx.hpp"

#include <utility>

namespace mfem::future
{

namespace LocalQFDevicesImpl
{

template<typename qfunc_t,
         typename inputs_t,
         typename outputs_t,
         size_t ninputs = std::tuple_size_v<inputs_t>,
         size_t noutputs = std::tuple_size_v<outputs_t>>
struct Action
{
   Action(IntegratorContext ctx,
          qfunc_t qfunc,
          inputs_t inputs,
          outputs_t outputs) :
      ctx(ctx),
      qfunc(std::move(qfunc)),
      inputs(inputs),
      outputs(outputs)
   {
      create_fop_to_fd(inputs, ctx.infds, input_to_infd);
      create_fop_to_fd(outputs, ctx.outfds, output_to_outfd);

      check_consistency(inputs, input_to_infd, ctx.infds);
      check_consistency(outputs, output_to_outfd, ctx.outfds);

      // const int nqp = ctx.ir.GetNPoints();

      // Initialize DofToQuad maps for inputs
      for_constexpr<ninputs>([&](auto i)
      {
         const auto &fd = ctx.infds[input_to_infd[i]];
         std::visit([&](auto* space_ptr)
         {
            using T = std::decay_t<decltype(*space_ptr)>;
            if constexpr (std::is_same_v<T, FiniteElementSpace> ||
                          std::is_same_v<T, ParFiniteElementSpace>)
            {
               const auto *fe = space_ptr->GetTypicalFE();
               input_dtq_maps[i] = &fe->GetDofToQuad(ctx.ir, DofToQuad::TENSOR);
            }
         }, fd.data);
      });

      // Initialize DofToQuad maps for outputs
      for_constexpr<noutputs>([&](auto i)
      {
         const auto &fd = ctx.outfds[output_to_outfd[i]];
         std::visit([&](auto* space_ptr)
         {
            using T = std::decay_t<decltype(*space_ptr)>;
            if constexpr (std::is_same_v<T, FiniteElementSpace> ||
                          std::is_same_v<T, ParFiniteElementSpace>)
            {
               const auto *fe = space_ptr->GetTypicalFE();
               output_dtq_maps[i] = &fe->GetDofToQuad(ctx.ir, DofToQuad::TENSOR);
            }
         }, fd.data);
      });
   }

   void operator()(const std::vector<Vector *> &xe,
                   std::vector<Vector *> &ye) const
   {
      if (ctx.attr.Size() == 0) { return; }

      // input_dtq_maps

      // const auto B = (const real_t*)input_dtq_maps[0/*i*/].B;
      // const auto G = (const real_t*)input_dtq_maps[0/*i*/].G;

      //    dfem::forall<T_Q1D*T_Q1D*T_Q1D>([=] MFEM_HOST_DEVICE (int e, void *)
      //    {
      //       if (has_attr && !d_attr[d_elem_attr[e] - 1]) { return; }

      //       constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;

      //       MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
      //       MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];

      //       low::regs3d_t<DIM, MQ1> reg;
      //       const real_t *rd = dx_ptr;

      //       MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
      //       {
      //          low::LoadMatrix(d1d, q1d, B, sB);
      //          low::LoadMatrix(d1d, q1d, G, sG);
      //          {
      //             low::LoadDofs3d(e, d1d, XE, sm0);
      //             low::Grad3d(d1d, q1d, sB, sG, sm0, sm1, reg);
      //          }
      //       }
      //       // else if constexpr (is_identity_fop<field_operator_t>::value)   // Identity
      //       {
      //          // db1("Identity");
      //          // rd = fields_e_ptr[input_to_field[i]];
      //          // rd = dx_ptr;
      //       }
      //    }

      //    MFEM_FOREACH_THREAD_DIRECT(qz,z,q1d)
      //    {
      //       MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
      //       {
      //          MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
      //          {

      //             auto args = decay_tuple<qf_param_ts> {};
      //             get<0>(args) = as_tensor<real_t, 3>(&reg[qz][qy][qx][0]);
      //             if constexpr (T_Q1D > 0)
      //             {
      //                get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*(qx*T_Q1D*T_Q1D + qy*T_Q1D + qz));
      //             }
      //             else
      //             {
      //                get<1>(args) = as_tensor<real_t, 3, 3>(rd + 9*(qx*q1d*q1d + qy*q1d + qz));
      //             }
      //             auto r = get<0>(apply(qfunc, args));
      //             if constexpr (decltype(r)::ndim == 1)
      //             {
      //                as_tensor<real_t, 3>(&reg[qz][qy][qx][0]) = r;
      //             }
      //             else { static_assert(false); }
      //          }
      //       }
      //    }
      //    MFEM_SYNC_THREAD;
      //    // Integrate
      //    // if constexpr (is_gradient_fop<std::decay_t<output_fop_t>>::value) // Gradient
      //    {
      //       // const auto sB = reinterpret_cast<const real_t (*)[MQ1]>(Bo);
      //       // const auto sG = reinterpret_cast<const real_t (*)[MQ1]>(Go);
      //       low::GradTranspose3d(d1d, q1d, sB, sG, reg, sm1, sm0);
      //       low::WriteDofs3d(d1d, 0, e, reg, YE);
      //    }
      // },
      // num_entities, thread_blocks, 0, nullptr);
   }


   IntegratorContext ctx;
   qfunc_t qfunc;
   inputs_t inputs;
   outputs_t outputs;

   std::array<size_t, ninputs> input_to_infd;
   std::array<size_t, noutputs> output_to_outfd;

   std::array<const DofToQuad*, ninputs> input_dtq_maps;
   std::array<const DofToQuad*, noutputs> output_dtq_maps;
};

} // namespace LocalQFDevicesImpl

} // namespace mfem::future
