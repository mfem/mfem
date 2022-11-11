#include "../../hooke/kernels/kernel_helpers.hpp"

namespace mfem
{
namespace navier
{

using mfem::internal::tensor;
using mfem::internal::make_tensor;

static inline
double sqrtk_f(const double k)
{
   return sqrt(0.5*(k+abs(k)));
}

static inline
double tls_f(const double k, const double wd)
{
   const double L = 0.03,
                tau = 1.0,
                d = 0.41*wd*sqrt(wd/L);

   double tls = std::min(d, sqrt(2.0)*sqrtk_f(k)*tau);
   if (tls == 0.0)
   {
      tls = 1e-8;
   }
   // double tls = wd;
   return tls;
}

class EddyViscosityCoefficient : public Coefficient
{
public:
   EddyViscosityCoefficient(const ParGridFunction &k,
                            const ParGridFunction &wall_distance,
                            const double mu) :
      k(k),
      wall_distance(wall_distance),
      mu(mu) {}

   double  Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      const double k_q = k.GetValue(T, ip);
      const double wd_q = wall_distance.GetValue(T, ip);

      return mu * sqrtk_f(k_q) * tls_f(k_q, wd_q);
   }

private:
   const ParGridFunction &k;
   const ParGridFunction &wall_distance;
   const double mu;
};

template <int d1d, int q1d> static inline
void PrandtlKolmogorovApply2D(
   TimeDependentOperator::EvalMode eval_mode,
   const int ne,
   const Array<double> &B_,
   const Array<double> &G_,
   const Array<double> &W_,
   const Vector &Jacobian_,
   const Vector &detJ_,
   const Vector &X_, Vector &Y_,
   const Vector &u_q_,
   const Vector &kv_q_,
   const Vector &f_q_,
   const Vector &wd_q_,
   const double mu_calibration_const)
{
   constexpr int dim = 2;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   const tensor<double, q1d, d1d> &B =
   make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

   const tensor<double, q1d, d1d> &G =
   make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

   const auto qweights = Reshape(W_.Read(), q1d, q1d);
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, ne);
   const auto k = Reshape(X_.Read(), d1d, d1d, ne);
   auto f = Reshape(Y_.ReadWrite(), d1d, d1d, ne);
   const auto u_q = Reshape(u_q_.Read(), q1d, q1d, dim, ne);
   const auto kv_q = Reshape(kv_q_.Read(), q1d, q1d, ne);
   const auto f_q__ = Reshape(f_q_.Read(), q1d, q1d, ne);
   const auto wd_q = Reshape(wd_q_.Read(), q1d, q1d, ne);

   if (eval_mode == TimeDependentOperator::EvalMode::NORMAL ||
       eval_mode == TimeDependentOperator::EvalMode::ADDITIVE_TERM_1)
   {
      // Explicit contributions
      for (int e = 0; e < ne; e++)
      {
         auto F = Reshape(&f(0, 0, e), d1d, d1d);

         MFEM_SHARED tensor<double, 2, 3, q1d, q1d> smem;

         auto k_el = make_tensor<d1d, d1d>([&](int i, int j)
         { return k(i, j, e); });

         const auto k_q = KernelHelpers::EvaluateAtQuadraturePoints(k_el, B);

         MFEM_SHARED tensor<double, q1d, q1d, dim, dim> dudxi;

         const auto u_el = Reshape(&u_q(0, 0, 0, e), d1d, d1d, dim);
         KernelHelpers::CalcGrad(B, G, smem, u_el, dudxi);

         const tensor<double, q1d, q1d> &f_q =
         make_tensor<q1d, q1d>([&](int i, int j) { return f_q__(i, j, e); });

         MFEM_SHARED tensor<double, q1d, q1d, dim> dkdxi;
         const auto k_el1 = Reshape(&k(0, 0, e), d1d, d1d);
         KernelHelpers::CalcGrad(B, G, smem, k_el1, dkdxi);

         for (int qx = 0; qx < q1d; qx++)
         {
            for (int qy = 0; qy < q1d; qy++)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, i, j, e); }));

               auto vel = make_tensor<dim>(
               [&](int i) { return u_q(qx, qy, i, e);});

               const auto dkdx = dkdxi(qy, qx) * invJqp;
               const auto dudx = dudxi(qy, qx) * invJqp;

               const double sqrt_k = sqrtk_f(k_q(qx,qy));
               const double tls = tls_f(k_q(qx,qy), wd_q(qx,qy,e));

               // grad(u) + grad(u)^T
               const auto S = (dudx + transpose(dudx));

               // nu_t |grad(u) + grad(u)^T|^2 = nu_t |S|^2 = nu_t S:S
               // const auto nu_t_abs_S_squared = mu_calibration_const *
               //                                 tls * sqrt_k * ddot(S,S);
               const auto nu_t_abs_S_squared = mu_calibration_const *
                                               ddot(S,S);

               const double JxW = detJ(qx, qy, e) * qweights(qx, qy);
               const auto dphidx = KernelHelpers::GradAllShapeFunctions(qx, qy, B, G, invJqp);

               for (int dx = 0; dx < d1d; dx++)
               {
                  for (int dy = 0; dy < d1d; dy++)
                  {
                     F(dx, dy) +=
                        // advection
                        // \nabla phi \cdot (v * k)
                        + dot(dphidx(dx, dy), vel * k_q(qx, qy)) * JxW
                        // v \cdot \nabla k * phi @TODO
                        // - dot(vel, dkdx) * dphidx(dx, dy, 0) * JxW // ??????
                        // custom forcing (e.g. MMS forcing)
                        // f * phi
                        + B(qx,dx) * f_q(qx,qy) * B(qy,dy) * JxW
                        // k * sqrt(k)/l * phi
                        - B(qx,dx) * k_q(qx, qy) * (sqrt_k/tls) * B(qy,dy) * JxW
                        // nu_t |grad(u) + grad(u)^T|^2 * phi
                        + B(qx,dx) * nu_t_abs_S_squared * B(qy,dy) * JxW;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }

   // Implicit contributions
   if (eval_mode == TimeDependentOperator::EvalMode::NORMAL ||
       eval_mode == TimeDependentOperator::EvalMode::ADDITIVE_TERM_2)
   {
      for (int e = 0; e < ne; e++)
      {
         auto F = Reshape(&f(0, 0, e), d1d, d1d);

         MFEM_SHARED tensor<double, 2, 3, q1d, q1d> smem;
         MFEM_SHARED tensor<double, q1d, q1d, dim> dkdxi;

         const auto k_el1 = Reshape(&k(0, 0, e), d1d, d1d);
         KernelHelpers::CalcGrad(B, G, smem, k_el1, dkdxi);

         auto k_el = make_tensor<d1d, d1d>([&](int i, int j)
         { return k(i, j, e); });
         const auto k_q = KernelHelpers::EvaluateAtQuadraturePoints(k_el, B);

         for (int qx = 0; qx < q1d; qx++)
         {
            for (int qy = 0; qy < q1d; qy++)
            {
               auto invJqp = inv(make_tensor<dim, dim>(
               [&](int i, int j) { return J(qx, qy, i, j, e); }));

               const double JxW = detJ(qx, qy, e) * qweights(qx, qy);
               const auto dphidx = KernelHelpers::GradAllShapeFunctions(qx, qy, B, G, invJqp);
               auto dkdx = dkdxi(qy, qx) * invJqp;

               const double sqrt_k = sqrtk_f(k_q(qx,qy));
               const double tls = tls_f(k_q(qx,qy), wd_q(qx,qy,e));

               const double kv_star =
                  kv_q(qx, qy, e) + mu_calibration_const * tls * sqrt_k;

               for (int dx = 0; dx < d1d; dx++)
               {
                  for (int dy = 0; dy < d1d; dy++)
                  {
                     F(dx, dy) +=
                        // diffusion
                        // \nabla kv_star * k \cdot \nabla phi
                        - dot(kv_star * dkdx, dphidx(dx, dy)) * JxW;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
}

template <int d1d, int q1d> static inline
void PrandtlKolmogorovAssembleJacobian2D(
   const int ne,
   const Array<double> &B_,
   const Array<double> &G_,
   const Array<double> &W_,
   const Vector &Jacobian_,
   const Vector &detJ_,
   const Vector &X_,
   Vector &Y_,
   const Vector &kv_q_,
   const Vector &wd_q_,
   const double mu_calibration_const,
   const double gamma)
{
   constexpr int dim = 2;
   KernelHelpers::CheckMemoryRestriction(d1d, q1d);

   const tensor<double, q1d, d1d> &B =
   make_tensor<q1d, d1d>([&](int i, int j) { return B_[i + q1d*j]; });

   const tensor<double, q1d, d1d> &G =
   make_tensor<q1d, d1d>([&](int i, int j) { return G_[i + q1d*j]; });

   const auto qweights = Reshape(W_.Read(), q1d, q1d);
   const auto J = Reshape(Jacobian_.Read(), q1d, q1d, dim, dim, ne);
   const auto detJ = Reshape(detJ_.Read(), q1d, q1d, ne);
   const auto k = Reshape(X_.Read(), d1d, d1d, ne);
   auto dRdk = Reshape(Y_.ReadWrite(), d1d * d1d, d1d * d1d, ne);
   const auto kv_q = Reshape(kv_q_.Read(), q1d, q1d, ne);
   const auto wd_q = Reshape(wd_q_.Read(), q1d, q1d, ne);

   for (int e = 0; e < ne; e++)
   {
      // d1d^2 x d1d^2
      auto dRdk_e = Reshape(&dRdk(0, 0, e), d1d * d1d, d1d * d1d);

      MFEM_SHARED tensor<double, 2, 3, q1d, q1d> smem;
      MFEM_SHARED tensor<double, q1d, q1d, dim> dkdxi;

      const auto k_el1 = Reshape(&k(0, 0, e), d1d, d1d);
      KernelHelpers::CalcGrad(B, G, smem, k_el1, dkdxi);

      auto k_el = make_tensor<d1d, d1d>([&](int i, int j)
      { return k(i, j, e); });
      const auto k_q = KernelHelpers::EvaluateAtQuadraturePoints(k_el, B);

      for (int qx = 0; qx < q1d; qx++)
      {
         for (int qy = 0; qy < q1d; qy++)
         {
            auto invJqp = inv(make_tensor<dim, dim>(
            [&](int i, int j) { return J(qx, qy, i, j, e); }));

            const double JxW = detJ(qx, qy, e) * qweights(qx, qy);
            const auto phi = KernelHelpers::AllShapeFunctions(qx, qy, B);
            const auto dphidx = KernelHelpers::GradAllShapeFunctions(qx, qy, B, G, invJqp);
            auto dkdx = dkdxi(qy, qx) * invJqp;
            const double sqrt_k = sqrtk_f(k_q(qx,qy));
            const double tls = tls_f(k_q(qx,qy), wd_q(qx,qy,e));
            const double kv_star =
               kv_q(qx, qy, e) + mu_calibration_const * tls * sqrt_k;

            double term1;
            if (sqrt_k > 0.0)
            {
               term1 = ((mu_calibration_const * tls) / (2.0 * sqrt_k));
            }
            else
            {
               term1 = 0.0;
            }

            for (int dx_i = 0; dx_i < d1d; dx_i++)
            {
               for (int dy_i = 0; dy_i < d1d; dy_i++)
               {
                  for (int dx_j = 0; dx_j < d1d; dx_j++)
                  {
                     for (int dy_j = 0; dy_j < d1d; dy_j++)
                     {
                        // PLEASE BE CORRECT -> THUMBSUP
                        const int row = dx_i + dy_i * d1d;
                        const int col = dx_j + dy_j * d1d;

                        // printf("0: %.1E ", dRdk_e(row, col));

                        dRdk_e(row, col) +=
                           + phi(dx_i, dy_i) * phi(dx_j, dy_j) * JxW;

                        // b := dot product sum index
                        for (int b = 0; b < dim; b++)
                        {
                           dRdk_e(row, col) +=
                              + dphidx(dx_i, dy_i, b) * (
                                 + term1 * dkdx(b) * phi(dx_j, dy_j)
                                 + kv_star * dphidx(dx_j, dy_j, b)
                              ) * JxW * gamma;
                        }
                     }
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

} // namespace navier
} // namespace mfem
