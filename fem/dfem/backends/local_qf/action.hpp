#pragma once

#include "../util.hpp"
#include "../../integrator_ctx.hpp"
#include "../fem/kernels3d.hpp"

#include <utility>

namespace mfem::future
{

namespace LocalQFImpl
{

template<
   typename qfunc_t,
   typename inputs_t,
   typename outputs_t,
   size_t ninputs = tuple_size<inputs_t>::value,
   size_t noutputs = tuple_size<outputs_t>::value>
struct Action
{
   Action(
      IntegratorContext ctx,
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

      const int nqp = ctx.ir.GetNPoints();

      for_constexpr<ninputs>([&](auto i)
      {
         const auto &fd = ctx.infds[input_to_infd[i]];
         idtq[i] = GetDofToQuad<Entity::Element>(fd, ctx.ir,
                                                 DofToQuad::TENSOR);
      });

      for_constexpr<noutputs>([&](auto i)
      {
         const auto &fd = ctx.outfds[output_to_outfd[i]];
         odtq[i] = GetDofToQuad<Entity::Element>(fd, ctx.ir,
                                                 DofToQuad::TENSOR);
      });
   }

   void operator()(
      const std::vector<Vector *> &xe,
      std::vector<Vector *> &ye) const
   {
      constexpr int T_Q1D = 0;
      constexpr int DIM = 3;

      if (ctx.attr.Size() == 0) { return; }

      // read all maps into GPU memory
      for_constexpr<ninputs>([&](auto i)
      {
         idtqb[i] = idtq[i]->B.Read();
         idtqg[i] = idtq[i]->G.Read();
      });

      for_constexpr<noutputs>([&](auto i)
      {
         odtqb[i] = odtq[i]->B.Read();
         odtqg[i] = odtq[i]->G.Read();
      });

      const auto XE = Reshape(fields_e[0].Read(), d1d, d1d, d1d, VDIM, NE);
      const real_t *dx_ptr = fields_e[1].Read();

      mfem::future::forall([=] MFEM_HOST_DEVICE (int e, void *)
      {
         constexpr int MQ1 = T_Q1D > 0 ? T_Q1D : 8;

         MFEM_SHARED real_t sm0[MQ1][MQ1][MQ1][3];
         MFEM_SHARED real_t sm1[MQ1][MQ1][MQ1][3];

         mfem::kernels::internal::low::regs3d_t<DIM, MQ1> reg;
         const real_t *rd = dx_ptr;

      });

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

   std::array<const DofToQuad*, ninputs> idtq;
   std::array<const DofToQuad*, noutputs> odtq;

   std::array<const real_t*, ninputs> idtqb;
   std::array<const real_t*, ninputs> idtqg;

   std::array<const real_t*, ninputs> odtqb;
   std::array<const real_t*, ninputs> odtqg;

};

}
}
