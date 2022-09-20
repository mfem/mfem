#include "../../hooke/kernels/kernel_helpers.hpp"

namespace mfem
{
namespace navier
{

using mfem::internal::tensor;
using mfem::internal::make_tensor;

template <int d1d, int q1d> static inline
void StressEvaluatorApply2D(
   const int ne,
   const Array<double> &B_,
   const Array<double> &G_,
   const Array<double> &W_,
   const Vector &Jacobian_,
   const Vector &detJ_,
   const Vector &X_, Vector &Y_,
   const Vector &dkv_
)
{
   constexpr int dim = 2;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   const tensor<double, q1d, d1d> &B =
   make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

   const tensor<double, q1d, d1d> &G =
   make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, dim, dim, ne);
   const auto U = Reshape(X_.Read(), d1d, d1d, dim, ne);
   auto fq = Reshape(Y_.ReadWrite(), q1d, q1d, dim, ne);
   auto dkv = Reshape(dkv_.Read(), q1d, q1d, dim, ne);

   MFEM_FORALL_2D(e, ne, q1d, q1d, 1,
                  // for (int e = 0; e < ne; e++)
   {
      MFEM_SHARED tensor<double, 2, 3, q1d, q1d> smem;
      MFEM_SHARED tensor<double, q1d, q1d, dim> gradkvS;
      MFEM_SHARED tensor<double, q1d, q1d, dim, dim> dudxi;

      const auto U_el = Reshape(&U(0, 0, 0, e), d1d, d1d, dim);
      KernelHelpers::CalcGrad(B, G, smem, U_el, dudxi);

      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            auto invJqp = inv(make_tensor<dim, dim>(
            [&](int i, int j) { return J(qx, qy, i, j, e); }));

            auto gradkv = make_tensor<dim>(
            [&](int i) { return dkv(qx, qy, i, e); });

            const auto dudx = dudxi(qy, qx) * invJqp;
            gradkvS(qx, qy) = dot(gradkv, dudx + transpose(dudx));

            for (int d = 0; d < dim; d++)
            {
               fq(qx, qy, d, e) = gradkvS(qx, qy, d);
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

} // namespace navier
} // namespace mfem

