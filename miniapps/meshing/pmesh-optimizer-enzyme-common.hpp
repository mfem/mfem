#ifndef MFEM_PMESH_OPTIMIZER_ENZYME_COMMON_HPP
#define MFEM_PMESH_OPTIMIZER_ENZYME_COMMON_HPP

#include "mfem.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_ENZYME)

#include "../../fem/dfem/doperator.hpp"
#include "../../fem/dfem/backends/local_qf/prelude.hpp"
#include "../../fem/dfem/backends/local_qf/revdiff_transformer.hpp"
#include "mesh-fitting.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace mfem;

namespace future = mfem::future;

using future::DerivativeOperator;
using future::DifferentiableOperator;
using future::FieldDescriptor;
using future::FunctionalValue;
using future::Identity;
using future::LocalQFBackend;
using future::tensor;
using future::Value;
using future::Weight;

namespace pmesh_optimizer_enzyme
{
// Shared 2D layout: primary=size; secondary=aspect (target 6) or orientation
// (target 8).
static constexpr int TARGET_POSITION = 0;
static constexpr int TARGET_PRIMARY_VALUE = 2;
static constexpr int TARGET_SECONDARY_VALUE = 3;
static constexpr int TARGET_PRIMARY_GRAD = 4;
static constexpr int TARGET_SECONDARY_GRAD = 6;
static constexpr int TARGET_PRIMARY_HESS = 8;
static constexpr int TARGET_SECONDARY_HESS = 11;

// Target 5 stores a 3D scalar field and its full gradient and Hessian in the
// shared discrete-target stride.
static constexpr int TARGET5_VALUE = 0;
static constexpr int TARGET5_GRAD = 1;
static constexpr int TARGET5_HESS = 4;
static constexpr int TARGET5_MIN_SIZE = 13;
static constexpr int TARGET5_DATA_SIZE = 14;

template <int dim>
struct SurfaceFitDataLayout
{
   static constexpr int COEFFICIENT = 0;
   static constexpr int VALUE = 1;
   static constexpr int GRADIENT = VALUE + 1;
   static constexpr int HESSIAN = GRADIENT + dim;
   static constexpr int SIZE = HESSIAN + dim * dim;
};

}

namespace
{
using namespace pmesh_optimizer_enzyme;

static constexpr int X = 0;
static constexpr int Q = 1;
static constexpr int TARGET_W = 2;
static constexpr int REFERENCE_X = 3;
static constexpr int LIMIT_COEFF = 4;
static constexpr int TARGET_DATA = 5;
static constexpr int SURFACE_FIT_DATA = 6;
static constexpr int TARGET6_DATA_SIZE = 16;
static constexpr int TARGET8_DATA_SIZE = 14;

template <int target_id>
constexpr int TargetDataSize()
{
   static_assert(target_id == 5 || target_id == 6 || target_id == 8,
                 "Target data is available only for target ids 5, 6, and 8.");
   if constexpr (target_id == 5) { return TARGET5_DATA_SIZE; }
   if constexpr (target_id == 6) { return TARGET6_DATA_SIZE; }
   return TARGET8_DATA_SIZE;
}

// VectorQuadratureSpace requires a positive vdim when target data is unused.
inline int TargetDataVDim(int target_id)
{
   switch (target_id)
   {
      case 5: return TARGET5_DATA_SIZE;
      case 6: return TARGET6_DATA_SIZE;
      case 8: return TARGET8_DATA_SIZE;
      default: return 1;
   }
}

inline int SurfaceFitDataSize(int dim)
{
   return 2 + dim + dim * dim;
}

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

struct Target6Parameters
{
   real_t size_ratio = 40.0;
   real_t aspect_ratio = 20.0;
   real_t metric_shape_weight = 0.5;
};

enum FieldDerivativeBackend
{
   CLASSIC_HOST_DERIVATIVES = 0,
   TENSOR_KERNEL_DERIVATIVES = 1
};

struct EnzymeOptimizerResult
{
   int status = 0;
   bool converged = true;
   real_t initial_energy = 0.0;
   real_t final_energy = 0.0;
   real_t final_grad_norm = 0.0;
   real_t final_surface_fit_coefficient = 0.0;
};

real_t GlobalVectorNorm(MPI_Comm comm, const Vector &x)
{
   const real_t local_norm2 = x * x;
   real_t global_norm2 = 0.0;
   MPI_Allreduce(&local_norm2, &global_norm2, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, comm);
   return std::sqrt(global_norm2);
}

enum Target6Parameter
{
   TARGET6_SIZE_RATIO,
   TARGET6_ASPECT_RATIO,
   TARGET6_SHAPE_WEIGHT,
   TARGET6_NUM_PARAMETERS
};

} // namespace

namespace pmesh_optimizer_enzyme
{

// Custom Enzyme rules for discrete targets 6 and 8.

/// Return the primary sampled field while keeping x active for Enzyme.
MFEM_HOST_DEVICE
void get_target_primary(const real_t *x, const real_t *data, real_t *value)
{
   volatile real_t active_x = x[0];
   const real_t zero = active_x - active_x;
   *value = data[TARGET_PRIMARY_VALUE] + zero;
}

/// Return the secondary sampled field while keeping x active for Enzyme.
MFEM_HOST_DEVICE
void get_target_secondary(const real_t *x, const real_t *data, real_t *value)
{
   volatile real_t active_x = x[0];
   const real_t zero = active_x - active_x;
   *value = data[TARGET_SECONDARY_VALUE] + zero;
}

// Primal identity treated as constant by Enzyme. Thus x-Stop(x) is zero in the
// primal but has derivative dx, allowing registered rules to inject sampled
// gradients and Hessians. The attributes keep Clang from removing this delta.
MFEM_HOST_DEVICE MFEM_ENZYME_INACTIVE
__attribute__((noinline, optnone))
real_t StopDiscreteTargetGradient(real_t value)
{
   return value;
}

// Return the sampled value; the zero-valued Taylor term exposes its supplied
// gradient during nested differentiation.
template <int value_offset, int grad_offset>
MFEM_HOST_DEVICE
void *get_target_aug_impl(const real_t *x, real_t *dx,
                          const real_t *data, real_t *ddata,
                          real_t *value, real_t *dvalue)
{
   MFEM_CONTRACT_VAR(dx);
   MFEM_CONTRACT_VAR(ddata);
   MFEM_CONTRACT_VAR(dvalue);
   const real_t delta0 = x[0] - StopDiscreteTargetGradient(x[0]);
   const real_t delta1 = x[1] - StopDiscreteTargetGradient(x[1]);
   *value = data[value_offset] + data[grad_offset] * delta0 +
            data[grad_offset + 1] * delta1;
   return nullptr;
}

// Accumulate the supplied gradient; the zero-valued Hessian correction becomes
// active when Enzyme differentiates this rule again.
template <int grad_offset, int hess_offset>
MFEM_HOST_DEVICE
void get_target_rev_impl(const real_t *x, real_t *dx,
                         const real_t *data, real_t *ddata,
                         const real_t *value, const real_t *dvalue,
                         void *tape)
{
   MFEM_CONTRACT_VAR(ddata);
   MFEM_CONTRACT_VAR(value);
   MFEM_CONTRACT_VAR(tape);
   const real_t delta0 = x[0] - StopDiscreteTargetGradient(x[0]);
   const real_t delta1 = x[1] - StopDiscreteTargetGradient(x[1]);
   const real_t grad0 = data[grad_offset] +
                        data[hess_offset] * delta0 +
                        data[hess_offset + 1] * delta1;
   const real_t grad1 = data[grad_offset + 1] +
                        data[hess_offset + 1] * delta0 +
                        data[hess_offset + 2] * delta1;
   dx[0] += *dvalue * grad0;
   dx[1] += *dvalue * grad1;
}

MFEM_HOST_DEVICE
void *get_target_primary_aug(const real_t *x, real_t *dx,
                             const real_t *data, real_t *ddata,
                             real_t *value, real_t *dvalue)
{
   return get_target_aug_impl<TARGET_PRIMARY_VALUE, TARGET_PRIMARY_GRAD>(
             x, dx, data, ddata, value, dvalue);
}

MFEM_HOST_DEVICE
void get_target_primary_rev(const real_t *x, real_t *dx,
                            const real_t *data, real_t *ddata,
                            const real_t *value, const real_t *dvalue,
                            void *tape)
{
   get_target_rev_impl<TARGET_PRIMARY_GRAD, TARGET_PRIMARY_HESS>(
      x, dx, data, ddata, value, dvalue, tape);
}

MFEM_HOST_DEVICE
void *get_target_secondary_aug(const real_t *x, real_t *dx,
                               const real_t *data, real_t *ddata,
                               real_t *value, real_t *dvalue)
{
   return get_target_aug_impl<TARGET_SECONDARY_VALUE, TARGET_SECONDARY_GRAD>(
             x, dx, data, ddata, value, dvalue);
}

MFEM_HOST_DEVICE
void get_target_secondary_rev(const real_t *x, real_t *dx,
                              const real_t *data, real_t *ddata,
                              const real_t *value, const real_t *dvalue,
                              void *tape)
{
   get_target_rev_impl<TARGET_SECONDARY_GRAD, TARGET_SECONDARY_HESS>(
      x, dx, data, ddata, value, dvalue, tape);
}

void *__enzyme_register_gradient_get_target_primary[3] =
{
   (void *)&get_target_primary,
   (void *)&get_target_primary_aug,
   (void *)&get_target_primary_rev
};

void *__enzyme_register_gradient_get_target_secondary[3] =
{
   (void *)&get_target_secondary,
   (void *)&get_target_secondary_aug,
   (void *)&get_target_secondary_rev
};

// Custom Enzyme rules for target 5.

/// Return the sampled target-5 size while keeping x active for Enzyme.
MFEM_HOST_DEVICE
void get_target5_size(const real_t *x, const real_t *data, real_t *value)
{
   volatile real_t active_x = x[0];
   const real_t zero = active_x - active_x;
   *value = data[TARGET5_VALUE] + zero;
}

MFEM_HOST_DEVICE
void *get_target5_size_aug(const real_t *x, real_t *dx,
                           const real_t *data, real_t *ddata,
                           real_t *value, real_t *dvalue)
{
   MFEM_CONTRACT_VAR(dx);
   MFEM_CONTRACT_VAR(ddata);
   MFEM_CONTRACT_VAR(dvalue);
   *value = data[TARGET5_VALUE];
   for (int d = 0; d < 3; d++)
   {
      const real_t delta = x[d] - StopDiscreteTargetGradient(x[d]);
      *value += data[TARGET5_GRAD + d] * delta;
   }
   return nullptr;
}

MFEM_HOST_DEVICE
void get_target5_size_rev(const real_t *x, real_t *dx,
                          const real_t *data, real_t *ddata,
                          const real_t *value, const real_t *dvalue,
                          void *tape)
{
   MFEM_CONTRACT_VAR(ddata);
   MFEM_CONTRACT_VAR(value);
   MFEM_CONTRACT_VAR(tape);
   real_t delta[3];
   for (int d = 0; d < 3; d++)
   {
      delta[d] = x[d] - StopDiscreteTargetGradient(x[d]);
   }
   for (int i = 0; i < 3; i++)
   {
      real_t grad = data[TARGET5_GRAD + i];
      for (int j = 0; j < 3; j++)
      {
         grad += data[TARGET5_HESS + i * 3 + j] * delta[j];
      }
      dx[i] += *dvalue * grad;
   }
}

void *__enzyme_register_gradient_get_target5_size[3] =
{
   (void *)&get_target5_size,
   (void *)&get_target5_size_aug,
   (void *)&get_target5_size_rev
};

// Custom Enzyme rules for level-set surface fitting.

template <int dim>
MFEM_HOST_DEVICE inline
void get_sigma_impl(const real_t *x, const real_t *data, real_t *sigma)
{
   // Retain active x while returning the sampled primal value.
   volatile real_t active_x = x[0];
   const real_t zero = active_x - active_x;
   *sigma = data[SurfaceFitDataLayout<dim>::VALUE] + zero;
}

/// Primal identity with zero Enzyme derivative.
// Keep the call visible so Clang cannot fold x-Stop(x) before differentiation.
MFEM_HOST_DEVICE MFEM_ENZYME_INACTIVE
__attribute__((noinline, optnone))
real_t StopSurfaceFittingGradient(real_t value)
{
   return value;
}

template <int dim>
MFEM_HOST_DEVICE inline
void *get_sigma_aug_impl(const real_t *x, real_t *dx,
                         const real_t *data, real_t *ddata,
                         real_t *sigma, real_t *dsigma)
{
   MFEM_CONTRACT_VAR(dx);
   MFEM_CONTRACT_VAR(ddata);
   MFEM_CONTRACT_VAR(dsigma);
   *sigma = data[SurfaceFitDataLayout<dim>::VALUE];
   for (int d = 0; d < dim; d++)
   {
      // Zero in the primal; active under nested differentiation.
      const real_t delta =
         x[d] - StopSurfaceFittingGradient(x[d]);
      *sigma += data[SurfaceFitDataLayout<dim>::GRADIENT + d] * delta;
   }
   return nullptr;
}

template <int dim>
MFEM_HOST_DEVICE inline
void get_sigma_rev_impl(const real_t *x, real_t *dx,
                        const real_t *data, real_t *ddata,
                        const real_t *sigma, const real_t *dsigma,
                        void *tape)
{
   MFEM_CONTRACT_VAR(ddata);
   MFEM_CONTRACT_VAR(sigma);
   MFEM_CONTRACT_VAR(tape);
   real_t delta[dim];
   for (int d = 0; d < dim; d++)
   {
      delta[d] = x[d] - StopSurfaceFittingGradient(x[d]);
   }
   for (int i = 0; i < dim; i++)
   {
      real_t grad = data[SurfaceFitDataLayout<dim>::GRADIENT + i];
      for (int j = 0; j < dim; j++)
      {
         // Match classic TMOP by mirroring the stored upper triangle.
         const int row = i < j ? i : j;
         const int col = i < j ? j : i;
         grad += data[SurfaceFitDataLayout<dim>::HESSIAN +
                      row * dim + col] * delta[j];
      }
      dx[i] += *dsigma * grad;
   }
}

MFEM_HOST_DEVICE __attribute__((noinline))
void get_sigma_2d(const real_t *x, const real_t *data, real_t *sigma)
{
   get_sigma_impl<2>(x, data, sigma);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void *get_sigma_2d_aug(const real_t *x, real_t *dx,
                       const real_t *data, real_t *ddata,
                       real_t *sigma, real_t *dsigma)
{
   return get_sigma_aug_impl<2>(x, dx, data, ddata, sigma, dsigma);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void get_sigma_2d_rev(const real_t *x, real_t *dx,
                      const real_t *data, real_t *ddata,
                      const real_t *sigma, const real_t *dsigma,
                      void *tape)
{
   get_sigma_rev_impl<2>(x, dx, data, ddata, sigma, dsigma, tape);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void get_sigma_3d(const real_t *x, const real_t *data, real_t *sigma)
{
   get_sigma_impl<3>(x, data, sigma);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void *get_sigma_3d_aug(const real_t *x, real_t *dx,
                       const real_t *data, real_t *ddata,
                       real_t *sigma, real_t *dsigma)
{
   return get_sigma_aug_impl<3>(x, dx, data, ddata, sigma, dsigma);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void get_sigma_3d_rev(const real_t *x, real_t *dx,
                      const real_t *data, real_t *ddata,
                      const real_t *sigma, const real_t *dsigma,
                      void *tape)
{
   get_sigma_rev_impl<3>(x, dx, data, ddata, sigma, dsigma, tape);
}

void *__enzyme_register_gradient_get_sigma_2d[3] =
{
   (void *)&get_sigma_2d,
   (void *)&get_sigma_2d_aug,
   (void *)&get_sigma_2d_rev
};

void *__enzyme_register_gradient_get_sigma_3d[3] =
{
   (void *)&get_sigma_3d,
   (void *)&get_sigma_3d_aug,
   (void *)&get_sigma_3d_rev
};

} // namespace pmesh_optimizer_enzyme

namespace
{

template <typename scalar_t, int dim, int target_id, int metric_id,
          int target_data_size = 1>
MFEM_HOST_DEVICE inline
tensor<scalar_t, dim, dim>
TargetMatrix(const tensor<scalar_t, dim> &x,
             const tensor<real_t, dim, dim> &constant_W,
             scalar_t &shape_weight,
             const tensor<scalar_t, target_data_size> *target_data = nullptr)
{
   tensor<scalar_t, dim, dim> W {};
   shape_weight = 0.5_r;

   if constexpr (target_id == 1)
   {
      MFEM_CONTRACT_VAR(x);
      MFEM_CONTRACT_VAR(target_data);
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            W(i,j) = constant_W(i,j);
         }
      }
   }
   else if constexpr (target_id == 4)
   {
      static_assert(dim == 2, "Analytic target id 4 is implemented only in 2D.");
      MFEM_CONTRACT_VAR(constant_W);
      MFEM_CONTRACT_VAR(target_data);

      if constexpr (metric_id == 14)
      {
         const auto xc = x(0);
         const auto yc = x(1);
         const auto theta = M_PI * yc * (1.0_r - yc) *
                            cos(2.0_r * M_PI * xc);
         const auto alpha_bar = 0.1_r;

         W(0,0) =  alpha_bar * cos(theta);
         W(1,0) =  alpha_bar * sin(theta);
         W(0,1) = -alpha_bar * sin(theta);
         W(1,1) =  alpha_bar * cos(theta);
      }
      else if constexpr (metric_id == 85)
      {
         auto xc = x(0) - 0.5_r;
         auto yc = x(1) - 0.5_r;
         const auto th = 22.5_r * M_PI / 180.0_r;
         const auto xn =  cos(th) * xc + sin(th) * yc;
         const auto yn = -sin(th) * xc + cos(th) * yc;
         xc = xn;
         yc = yn;

         const auto tfac = 20.0_r;
         const auto s1 = 3.0_r;
         const auto s2 = 2.0_r;
         auto wgt = tanh((tfac * yc + s2 * sin(s1 * M_PI * xc)) + 1.0_r)
                    - tanh((tfac * yc + s2 * sin(s1 * M_PI * xc)) - 1.0_r);
         if (wgt > 1.0_r) { wgt = 1.0_r; }
         if (wgt < 0.0_r) { wgt = 0.0_r; }

         xc = x(0);
         yc = x(1);
         const auto theta = M_PI * yc * (1.0_r - yc) *
                            cos(2.0_r * M_PI * xc);
         const auto c = cos(theta);
         const auto s = sin(theta);
         const auto asp_ratio_tar = 0.1_r +
                                    (1.0_r - wgt) * (1.0_r - wgt);
         const auto inv_sqrt_asp = 1.0_r / sqrt(asp_ratio_tar);
         const auto sqrt_asp = sqrt(asp_ratio_tar);

         W(0,0) =  c * inv_sqrt_asp;
         W(1,0) =  s * inv_sqrt_asp;
         W(0,1) = -s * sqrt_asp;
         W(1,1) =  c * sqrt_asp;
      }
      else
      {
         const auto xc = x(0) - 0.5_r;
         const auto yc = x(1) - 0.5_r;
         const auto r2 = xc * xc + yc * yc;
         const auto r = (r2 > 0.0_r) ? sqrt(r2) : 0.0_r;
         const auto tan1 = tanh(30.0_r * (r - 0.15_r));
         const auto tan2 = tanh(30.0_r * (r - 0.35_r));

         W(0,0) = 0.5_r + tan1 - tan2;
         W(0,1) = 0.0_r;
         W(1,0) = 0.0_r;
         W(1,1) = 1.0_r;
      }
   }
   else if constexpr (target_id == 5)
   {
      static_assert(target_data_size == TargetDataSize<5>(),
                    "Unexpected target-5 data size.");
      const auto &data = *target_data;

      scalar_t size_raw;
      get_target5_size(&x[0], &data[0], &size_raw);
      const auto size = (size_raw > data(TARGET5_MIN_SIZE))
                        ? size_raw : data(TARGET5_MIN_SIZE);
      const auto scale = pow(size, 1.0_r / dim);
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            W(i,j) = scale * constant_W(i,j);
         }
      }
   }
   else if constexpr (target_id == 6)
   {
      static_assert(dim == 2, "Target id 6 is implemented only in 2D.");
      static_assert(target_data_size == TargetDataSize<6>(),
                    "Unexpected target-6 data size.");
      const auto &data = *target_data;

      scalar_t size_raw;
      scalar_t aspect;
      get_target_primary(&x[0], &data[0], &size_raw);
      get_target_secondary(&x[0], &data[0], &aspect);
      const auto size = (size_raw > data(14)) ? size_raw : data(14);
      const auto z = pow(size, 0.5_r);
      const auto rho0 = 1.0_r / pow(aspect, 0.5_r);
      const auto rho1 = pow(aspect, 0.5_r);

      W(0,0) = rho0 * z * constant_W(0,0);
      W(0,1) = rho0 * z * constant_W(0,1);
      W(1,0) = rho1 * z * constant_W(1,0);
      W(1,1) = rho1 * z * constant_W(1,1);
      shape_weight = data(15);
   }
   else if constexpr (target_id == 8)
   {
      static_assert(dim == 2, "Target id 8 is implemented only in 2D.");
      static_assert(metric_id == 36, "Target id 8 requires metric 36.");
      static_assert(target_data_size == TargetDataSize<8>(),
                    "Unexpected target-8 data size.");
      MFEM_CONTRACT_VAR(constant_W);
      const auto &data = *target_data;

      scalar_t size_raw;
      scalar_t ori;
      get_target_primary(&x[0], &data[0], &size_raw);
      get_target_secondary(&x[0], &data[0], &ori);
      const auto size = (size_raw > 1.0e-4_r) ? size_raw : 1.0e-4_r;
      const auto alpha = sqrt(size);
      const auto c = cos(ori);
      const auto s = sin(ori);

      W(0,0) =  alpha * c;
      W(1,0) =  alpha * s;
      W(0,1) = -alpha * s;
      W(1,1) =  alpha * c;
   }
   else if constexpr (target_id == 9)
   {
      static_assert(dim == 3, "Analytic target id 9 is implemented only in 3D.");
      MFEM_CONTRACT_VAR(constant_W);
      MFEM_CONTRACT_VAR(target_data);

      const auto xc = x(0) - 0.5_r;
      const auto yc = x(1) - 0.5_r;
      const auto zc = x(2) - 0.5_r;
      const auto r = sqrt(xc * xc + yc * yc + zc * zc);
      constexpr auto inner_radius = 0.15_r;
      constexpr auto outer_radius = 0.35_r;
      constexpr auto transition_scale = 10.0_r;
      const auto tan1 = tanh(transition_scale * (r - inner_radius));
      const auto tan2 = tanh(transition_scale * (r - outer_radius));
      const auto normalization =
         tanh(0.5_r * transition_scale * (outer_radius - inner_radius));
      const auto ind = 0.5_r * (tan1 - tan2) / normalization;

      const auto size = ind * 0.005_r + (1.0_r - ind) * 0.1_r;
      // Here size is det(W), so an isotropic 3D target uses its cube root.
      const auto scale = pow(size, 1.0_r / 3.0_r);
      W(0,0) = scale;
      W(1,1) = scale;
      W(2,2) = scale;
   }
   else
   {
      static_assert(target_id == 1 || target_id == 4 || target_id == 5 ||
                    target_id == 6 || target_id == 8 || target_id == 9,
                    "Unsupported target_id");
   }

   return W;
}

// Compile-time TMOP metric dispatch: 2D {2,14,36,58,80,85};
// 3D {301,302,303,321}.
// T = A W^{-1}; shape_weight applies only to composite metrics.
template <typename scalar_t, int dim, int metric_id>
MFEM_HOST_DEVICE inline
scalar_t EvaluateTMOPMetric(const tensor<scalar_t, dim, dim> &T,
                            scalar_t shape_weight = 0.5_r)
{
   const auto tau = det(T);
   const auto norm2 = sqnorm(T);

   if constexpr (dim == 2 && metric_id == 2)
   {
      // mu_2 = 0.5 |T|^2 / det(T) - 1
      return 0.5_r * norm2 / tau - 1.0_r;
   }
   else if constexpr (dim == 2 && metric_id == 14)
   {
      const auto TminusI_00 = T(0,0) - 1.0_r;
      const auto TminusI_01 = T(0,1);
      const auto TminusI_10 = T(1,0);
      const auto TminusI_11 = T(1,1) - 1.0_r;
      return TminusI_00 * TminusI_00 + TminusI_01 * TminusI_01 +
             TminusI_10 * TminusI_10 + TminusI_11 * TminusI_11;
   }
   else if constexpr (dim == 2 && metric_id == 58)
   {
      const auto i1b = norm2 / tau;
      return i1b * (i1b - 2.0_r);
   }
   else if constexpr (dim == 2 && metric_id == 36)
   {
      // For identity W, mu_36 = |T-I|^2/det(T); nonidentity W is handled by
      // DiscreteTarget8.
      const auto TminusI_00 = T(0,0) - 1.0_r;
      const auto TminusI_01 = T(0,1);
      const auto TminusI_10 = T(1,0);
      const auto TminusI_11 = T(1,1) - 1.0_r;
      const auto fnorm2 = TminusI_00 * TminusI_00 + TminusI_01 * TminusI_01 +
                          TminusI_10 * TminusI_10 + TminusI_11 * TminusI_11;
      return fnorm2 / tau;
   }
   else if constexpr (dim == 2 && metric_id == 80)
   {
      // mu_80 = w*mu_2 + (1-w)*mu_77, where
      // mu_77 = 0.5*(tau^2 + 1/tau^2) - 1.
      const auto mu2 = 0.5_r * norm2 / tau - 1.0_r;
      const auto tau2 = tau * tau;
      const auto mu77 = 0.5_r * (tau2 + 1.0_r / tau2) - 1.0_r;
      return shape_weight * mu2 + (1.0_r - shape_weight) * mu77;
   }
   else if constexpr (dim == 2 && metric_id == 85)
   {
      const auto alpha = sqrt(0.5_r * norm2);
      const auto TminusTp_00 = T(0,0) - alpha;
      const auto TminusTp_01 = T(0,1);
      const auto TminusTp_10 = T(1,0);
      const auto TminusTp_11 = T(1,1) - alpha;
      return TminusTp_00 * TminusTp_00 + TminusTp_01 * TminusTp_01 +
             TminusTp_10 * TminusTp_10 + TminusTp_11 * TminusTp_11;
   }
   else if constexpr (dim == 3 && (metric_id == 301 ||
                                   metric_id == 302 ||
                                   metric_id == 321))
   {
      const auto C00 = T(1,1) * T(2,2) - T(1,2) * T(2,1);
      const auto C01 = T(1,2) * T(2,0) - T(1,0) * T(2,2);
      const auto C02 = T(1,0) * T(2,1) - T(1,1) * T(2,0);
      const auto C10 = T(0,2) * T(2,1) - T(0,1) * T(2,2);
      const auto C11 = T(0,0) * T(2,2) - T(0,2) * T(2,0);
      const auto C12 = T(0,1) * T(2,0) - T(0,0) * T(2,1);
      const auto C20 = T(0,1) * T(1,2) - T(0,2) * T(1,1);
      const auto C21 = T(0,2) * T(1,0) - T(0,0) * T(1,2);
      const auto C22 = T(0,0) * T(1,1) - T(0,1) * T(1,0);
      const auto cofactor_norm2 =
         C00 * C00 + C01 * C01 + C02 * C02 +
         C10 * C10 + C11 * C11 + C12 * C12 +
         C20 * C20 + C21 * C21 + C22 * C22;

      if constexpr (metric_id == 301)
      {
         return sqrt(norm2 * cofactor_norm2) / (3.0_r * tau) - 1.0_r;
      }
      else if constexpr (metric_id == 302)
      {
         return norm2 * cofactor_norm2 / (9.0_r * tau * tau) - 1.0_r;
      }
      else
      {
         return norm2 + cofactor_norm2 / (tau * tau) - 6.0_r;
      }
   }
   else if constexpr (dim == 3 && metric_id == 303)
   {
      return norm2 / (3.0_r * pow(tau, 2.0_r / 3.0_r)) - 1.0_r;
   }
   else
   {
      static_assert((dim == 2 &&
                     (metric_id == 2 || metric_id == 14 ||
                      metric_id == 36 || metric_id == 58 ||
                      metric_id == 80 || metric_id == 85)) ||
                    (dim == 3 &&
                     (metric_id == 301 || metric_id == 302 ||
                      metric_id == 303 || metric_id == 321)),
                    "Unsupported metric/dimension combination");
      return 0.0_r;
   }
}

// TMOP energy for analytic targets: construct W(x) directly from x.
template <typename scalar_t, int dim, int target_id, int metric_id>
struct AnalyticTargetTMOPEnergy
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<scalar_t, dim> &x,
                   const tensor<scalar_t, dim, dim> &dxdr,
                   const tensor<real_t, dim, dim> &constant_W,
                   const real_t &w_q,
                   real_t &f) const
   {
      scalar_t shape_weight = 0.5_r;
      const auto W = TargetMatrix<scalar_t, dim, target_id, metric_id>(
                        x, constant_W, shape_weight);
      const auto T = dxdr * inv(W);
      const auto weight = det(W) * w_q;
      const auto val = EvaluateTMOPMetric<scalar_t, dim, metric_id>(
                          T, shape_weight);
      f = val * weight;
   }
};

// TMOP energy for discrete targets: construct W(x) from sampled target data.
template <typename scalar_t, int dim, int target_id, int metric_id>
struct DiscreteTargetTMOPEnergy
{
   static constexpr int target_data_size = TargetDataSize<target_id>();

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<scalar_t, dim> &x,
                   const tensor<scalar_t, dim, dim> &dxdr,
                   const tensor<real_t, dim, dim> &constant_W,
                   const tensor<scalar_t, target_data_size> &target_data,
                   const real_t &w_q,
                   real_t &f) const
   {
      scalar_t shape_weight = 0.5_r;
      const auto W = TargetMatrix<scalar_t, dim, target_id, metric_id>(
                        x, constant_W, shape_weight, &target_data);
      const auto T = dxdr * inv(W);
      const auto weight = det(W) * w_q;
      const auto val = EvaluateTMOPMetric<scalar_t, dim, metric_id>(
                          T, shape_weight);
      f = val * weight;
   }
};

// TMOP energy for precomputed W, used by constant targets and frozen
// linearizations; x enters only through A = grad(x).
template <typename scalar_t, int dim, int metric_id>
struct PrecomputedTargetTMOPEnergy
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<scalar_t, dim, dim> &dxdr,
                   const tensor<real_t, dim, dim> &W,
                   const real_t &w_q,
                   real_t &f) const
   {
      const auto T = dxdr * inv(W);
      const auto weight = det(W) * w_q;
      auto val = EvaluateTMOPMetric<scalar_t, dim, metric_id>(T);
      f = val * weight;
   }
};

template <typename scalar_t, int dim>
MFEM_HOST_DEVICE inline
scalar_t NodeLimitingValue(const tensor<scalar_t, dim> &x,
                           const tensor<real_t, dim> &x0,
                           const real_t &limit_coeff)
{
   scalar_t dist2 = 0.0_r;
   for (int d = 0; d < dim; d++)
   {
      const auto diff = x(d) - x0(d);
      dist2 += diff * diff;
   }
   return 0.5_r * limit_coeff * dist2;
}

// Node-limiting energy weighted by an analytic target W(x).
template <typename scalar_t, int dim, int target_id, int metric_id>
struct AnalyticTargetNodeLimitingEnergy
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<scalar_t, dim> &x,
                   const tensor<real_t, dim> &x0,
                   const tensor<real_t, dim, dim> &constant_W,
                   const real_t &limit_coeff,
                   const real_t &w_q,
                   real_t &f) const
   {
      scalar_t shape_weight = 0.5_r;
      const auto W = TargetMatrix<scalar_t, dim, target_id, metric_id>(
                        x, constant_W, shape_weight);
      f = NodeLimitingValue<scalar_t, dim>(x, x0, limit_coeff) *
          det(W) * w_q;
   }
};

// Node-limiting energy weighted by a discrete target W(x).
template <typename scalar_t, int dim, int target_id, int metric_id>
struct DiscreteTargetNodeLimitingEnergy
{
   static constexpr int target_data_size = TargetDataSize<target_id>();

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<scalar_t, dim> &x,
                   const tensor<real_t, dim> &x0,
                   const tensor<real_t, dim, dim> &constant_W,
                   const tensor<scalar_t, target_data_size> &target_data,
                   const real_t &limit_coeff,
                   const real_t &w_q,
                   real_t &f) const
   {
      scalar_t shape_weight = 0.5_r;
      const auto W = TargetMatrix<scalar_t, dim, target_id, metric_id>(
                        x, constant_W, shape_weight, &target_data);
      f = NodeLimitingValue<scalar_t, dim>(x, x0, limit_coeff) *
          det(W) * w_q;
   }
};

// Node-limiting energy weighted by a precomputed W.
template <typename scalar_t, int dim>
struct PrecomputedTargetNodeLimitingEnergy
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<scalar_t, dim> &x,
                   const tensor<real_t, dim> &x0,
                   const tensor<real_t, dim, dim> &W,
                   const real_t &limit_coeff,
                   const real_t &w_q,
                   real_t &f) const
   {
      f = NodeLimitingValue<scalar_t, dim>(x, x0, limit_coeff) *
          det(W) * w_q;
   }
};

struct SurfaceFittingOptions
{
   enum LevelSetSource
   {
      ANALYTIC = 0,
      DISCRETE = 1
   };

   enum AnalyticLevelSet
   {
      CIRCLE = 1,
      SQUIRCLE = 3,
      SPHERE = 7,
      QUADRATIC_INTERFACE = 5,
      CUBIC_INTERFACE = 6
   };

   enum DiscreteDerivativeMode
   {
      INTERPOLATED_SOURCE = 1,
      ELEMENT_LOCAL = 2
   };

   bool enabled = false;
   LevelSetSource source = ANALYTIC;
   AnalyticLevelSet analytic_level_set = CIRCLE;
   const Vector *interface_parameters = nullptr;
   const ParGridFunction *discrete_level_set = nullptr;
   const Array<bool> *marker = nullptr;
   real_t coefficient = 0.0;
   DiscreteDerivativeMode discrete_derivative_mode = INTERPOLATED_SOURCE;
   bool discrete_from_background = false;
};

MFEM_HOST_DEVICE inline
void EvalAnalyticLevelSet(int dim,
                          int analytic_level_set,
                          const real_t *x,
                          const real_t *parameters,
                          int num_parameters,
                          real_t &sigma,
                          real_t *gradient,
                          real_t *hessian)
{
   sigma = 0.0;
   for (int d = 0; d < dim; d++) { gradient[d] = 0.0; }
   for (int i = 0; i < dim * dim; i++) { hessian[i] = 0.0; }

   const real_t xc = x[0] - 0.5;
   const real_t yc = x[1] - 0.5;
   if (analytic_level_set == SurfaceFittingOptions::QUADRATIC_INTERFACE ||
       analytic_level_set == SurfaceFittingOptions::CUBIC_INTERFACE)
   {
      const real_t y = x[1];
      const real_t z = y - 0.5;
      const real_t bow = y * (1.0 - y);
      real_t h = 0.5;
      real_t hy = 0.0;
      real_t hyy = 0.0;

      if (analytic_level_set == SurfaceFittingOptions::QUADRATIC_INTERFACE &&
          num_parameters == 2)
      {
         h += parameters[0] * z + parameters[1] * bow;
         hy += parameters[0] + parameters[1] * (1.0 - 2.0 * y);
         hyy -= 2.0 * parameters[1];
      }
      else
      {
         h += parameters[0] + parameters[1] * z + parameters[2] * bow;
         hy += parameters[1] + parameters[2] * (1.0 - 2.0 * y);
         hyy -= 2.0 * parameters[2];
         if (analytic_level_set == SurfaceFittingOptions::CUBIC_INTERFACE)
         {
            const real_t cubic = 4.0 * z * bow;
            const real_t cubic_y =
               4.0 * (bow + z * (1.0 - 2.0 * y));
            const real_t cubic_yy = 4.0 * (3.0 - 6.0 * y);
            h += parameters[3] * cubic;
            hy += parameters[3] * cubic_y;
            hyy += parameters[3] * cubic_yy;
         }
      }
      sigma = x[0] - h;
      gradient[0] = 1.0;
      gradient[1] = -hy;
      hessian[1 * dim + 1] = -hyy;
   }
   else if (analytic_level_set == SurfaceFittingOptions::SQUIRCLE)
   {
      const real_t radius = 0.24;
      const real_t xc2 = xc * xc;
      const real_t yc2 = yc * yc;
      sigma = xc2 * xc2 + yc2 * yc2 -
              radius * radius * radius * radius;
      gradient[0] = 4.0 * xc * xc2;
      gradient[1] = 4.0 * yc * yc2;
      hessian[0] = 12.0 * xc2;
      hessian[1 * dim + 1] = 12.0 * yc2;
   }
   else
   {
      const bool sphere =
         analytic_level_set == SurfaceFittingOptions::SPHERE;
      const real_t zc = sphere ? x[2] - 0.5 : 0.0;
      const real_t r2 = xc * xc + yc * yc + zc * zc;
      const real_t r = sqrt(r2);
      sigma = r - 0.25;
      if (r > 1.0e-14)
      {
         const real_t centered[3] = {xc, yc, zc};
         for (int i = 0; i < dim; i++)
         {
            gradient[i] = centered[i] / r;
            for (int j = 0; j < dim; j++)
            {
               hessian[i * dim + j] =
                  ((i == j ? 1.0 : 0.0) -
                   gradient[i] * gradient[j]) / r;
            }
         }
      }
   }
}

template <typename scalar_t, int dim>
struct SurfaceFittingLevelSetEnergy
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<scalar_t, dim> &x,
                   const tensor<scalar_t,
                                SurfaceFitDataLayout<dim>::SIZE> &data,
                   real_t &f) const
   {
      scalar_t sigma;
      if constexpr (dim == 2)
      {
         pmesh_optimizer_enzyme::get_sigma_2d(&x[0], &data[0], &sigma);
      }
      else
      {
         pmesh_optimizer_enzyme::get_sigma_3d(&x[0], &data[0], &sigma);
      }
      f = data(SurfaceFitDataLayout<dim>::COEFFICIENT) * sigma * sigma;
   }
};

inline IntegrationRule MakeTensorNodalIntegrationRule(const FiniteElement &fe)
{
   const IntegrationRule &nodes = fe.GetNodes();
   IntegrationRule lex_nodes(nodes.GetNPoints());
   lex_nodes.SetOrder(nodes.GetOrder());

   const auto *nfe = dynamic_cast<const NodalFiniteElement *>(&fe);
   const Array<int> *lex =
      (nfe && nfe->GetLexicographicOrdering().Size() > 0)
      ? &nfe->GetLexicographicOrdering() : nullptr;

   for (int i = 0; i < nodes.GetNPoints(); i++)
   {
      lex_nodes.IntPoint(i) = nodes.IntPoint(lex ? (*lex)[i] : i);
   }
   return lex_nodes;
}

// Evaluate an H1 field with tensor kernels. Output layout:
// q + nq*(component + vdim*element).
void EvaluateFieldValuesOnDevice(const ParGridFunction &field,
                                 const IntegrationRule &ir,
                                 Vector &values)
{
   const ParFiniteElementSpace &fes = *field.ParFESpace();
   const int ne = fes.GetNE();
   const int nq = ir.GetNPoints();
   const int vdim = fes.GetVDim();
   const MemoryType mt = Device::GetDeviceMemoryType();

   MFEM_VERIFY(UsesTensorBasis(fes),
               "Device field evaluation requires a tensor-product basis.");
   const Operator *restriction =
      fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(restriction, "Element restriction is required.");
   Vector element_values;
   element_values.SetSize(restriction->Height(), mt);
   restriction->Mult(field, element_values);

   values.SetSize(ne * nq * vdim, mt);
   Vector empty;
   const QuadratureInterpolator *qi = fes.GetQuadratureInterpolator(ir);
   MFEM_VERIFY(qi, "Quadrature interpolation is not supported for this field.");
   qi->SetOutputLayout(QVectorLayout::byNODES);
   qi->EnableTensorProducts();
   qi->Mult(element_values, QuadratureInterpolator::VALUES,
            values, empty, empty);
}

/** Build a byNODES nodal-position list without the host-only
    FiniteElementSpace::GetNodePositions path. */
void BuildNodalPointListOnDevice(const ParGridFunction &nodes,
                                 const ParFiniteElementSpace &target_fes,
                                 const IntegrationRule &nodal_ir,
                                 Vector &positions)
{
   const int dim = nodes.ParFESpace()->GetVDim();
   const int ne = target_fes.GetNE();
   const int nq = nodal_ir.GetNPoints();
   const int points = ne * nq;
   const MemoryType mt = Device::GetDeviceMemoryType();
   MFEM_VERIFY(dim == target_fes.GetMesh()->SpaceDimension() &&
               nodes.ParFESpace()->GetNE() == ne &&
               target_fes.GetTypicalFE()->GetDof() == nq,
               "Incompatible mesh-coordinate and target nodal spaces.");

   Vector element_positions;
   EvaluateFieldValuesOnDevice(nodes, nodal_ir, element_positions);
   const real_t *element_data = element_positions.Read();
   positions.UseDevice(true);
   positions.SetSize(points * dim, mt);
   real_t *position_data = positions.Write();
   mfem::forall(points * dim, [=] MFEM_HOST_DEVICE (int k)
   {
      const int point = k % points;
      const int d = k / points;
      const int q = point % nq;
      const int e = point / nq;
      position_data[k] = element_data[q + nq * (d + dim * e)];
   });
}

/** Assemble element-node samples into a continuous H1 field. GSLIB evaluates
    shared copies at the same point, so no MPI reduction is needed. */
void ScatterNodalValuesOnDevice(const Vector &element_values,
                                ParGridFunction &field)
{
   ParFiniteElementSpace &fes = *field.ParFESpace();
   const MemoryType mt = Device::GetDeviceMemoryType();
   MFEM_VERIFY(fes.GetVDim() == 1,
               "Nodal device scatter currently requires a scalar field.");
   const Operator *restriction =
      fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(restriction && restriction->Height() == element_values.Size(),
               "Unexpected element-node value layout.");

   field.UseDevice(true);
   restriction->MultTranspose(element_values, field);
   Vector element_ones(restriction->Height(), mt), overlap(fes.GetVSize(), mt);
   element_ones.UseDevice(true);
   overlap.UseDevice(true);
   element_ones = 1.0;
   restriction->AbsMultTranspose(element_ones, overlap);

   real_t *field_data = field.ReadWrite();
   const real_t *overlap_data = overlap.Read();
   const int size = field.Size();
   mfem::forall(size, [=] MFEM_HOST_DEVICE (int i)
   {
      field_data[i] /= overlap_data[i];
   });
}

// Evaluate values and element-local physical derivatives with tensor kernels.
// This avoids shared-node averaging and MPI reduction, so derivatives may jump
// across elements.
void EvaluateScalarFieldElementDerivativesOnDevice(
   const ParGridFunction &field,
   const IntegrationRule &ir,
   const IntegrationRule &nodal_ir,
   Vector &values,
   Vector *gradient = nullptr,
   Vector *hessian = nullptr)
{
   const ParFiniteElementSpace &fes = *field.ParFESpace();
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int dim = fes.GetMesh()->Dimension();
   const int sdim = fes.GetMesh()->SpaceDimension();
   const int ne = fes.GetNE();
   const int nd = fe.GetDof();
   const int nq = ir.GetNPoints();
   const MemoryType mt = Device::GetDeviceMemoryType();

   MFEM_VERIFY(fes.GetVDim() == 1,
               "Device scalar-field evaluation requires VDIM = 1.");
   MFEM_VERIFY(dim == sdim && (dim == 2 || dim == 3),
               "Device scalar-field derivatives require a 2D or 3D volume mesh.");
   MFEM_VERIFY(UsesTensorBasis(fes),
               "Device scalar-field derivatives require a tensor-product basis.");
   MFEM_VERIFY(nodal_ir.GetNPoints() == nd,
               "Collocated derivative rule must contain one point per DOF.");

   const Operator *restriction =
      fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(restriction, "Element restriction is required.");
   Vector element_values(restriction->Height(), mt);
   restriction->Mult(field, element_values);

   values.SetSize(ne * nq, mt);
   if (gradient) { gradient->SetSize(ne * nq * dim, mt); }
   Vector empty;
   const QuadratureInterpolator *qi = fes.GetQuadratureInterpolator(ir);
   MFEM_VERIFY(qi, "Quadrature interpolation is not supported for this field.");
   qi->SetOutputLayout(QVectorLayout::byNODES);
   qi->EnableTensorProducts();
   const unsigned flags = QuadratureInterpolator::VALUES |
                          (gradient ?
                           QuadratureInterpolator::PHYSICAL_DERIVATIVES : 0);
   qi->Mult(element_values, flags, values,
            gradient ? *gradient : empty, empty);

   if (!hessian) { return; }

   const DofToQuad &node_maps =
      fe.GetDofToQuad(nodal_ir, DofToQuad::TENSOR);
   const int node_d1d = node_maps.ndof;
   const int node_q1d = node_maps.nqpt;
   const int node_ndof = node_d1d * node_d1d *
                         ((dim == 3) ? node_d1d : 1);
   MFEM_VERIFY(node_d1d == node_q1d && node_ndof == nd,
               "Unexpected collocated derivative map dimensions.");
   const GeometricFactors *node_geom = fes.GetMesh()->GetGeometricFactors(
                                          nodal_ir,
                                          GeometricFactors::JACOBIANS,
                                          mt);
   Vector nodal_gradient(ne * nd * dim, mt);
   QuadratureInterpolator::CollocatedGradKernels::Run(
      dim, QVectorLayout::byNODES, true, 1, node_d1d,
      ne, node_maps.G.Read(), node_geom->J.Read(), element_values.Read(),
      nodal_gradient.Write(), sdim, 1, node_d1d);

   const DofToQuad &quad_maps = fe.GetDofToQuad(ir, DofToQuad::TENSOR);
   const int quad_d1d = quad_maps.ndof;
   const int quad_q1d = quad_maps.nqpt;
   const int quad_ndof = quad_d1d * quad_d1d *
                         ((dim == 3) ? quad_d1d : 1);
   const int quad_nqpt = quad_q1d * quad_q1d *
                         ((dim == 3) ? quad_q1d : 1);
   MFEM_VERIFY(quad_ndof == nd && quad_nqpt == nq,
               "Unexpected quadrature derivative map dimensions.");
   const GeometricFactors *quad_geom = fes.GetMesh()->GetGeometricFactors(
                                          ir,
                                          GeometricFactors::JACOBIANS,
                                          mt);
   hessian->SetSize(ne * nq * dim * dim, mt);
   QuadratureInterpolator::GradKernels::Run(
      dim, QVectorLayout::byNODES, true, dim, quad_d1d, quad_q1d,
      ne, quad_maps.B.Read(), quad_maps.G.Read(), quad_geom->J.Read(),
      nodal_gradient.Read(), hessian->Write(), sdim, dim,
      quad_d1d, quad_q1d);
}

void ProjectPhysicalGradientOnDevice(const ParGridFunction &field,
                                     ParGridFunction &gradient,
                                     const IntegrationRule &nodal_ir);

// Evaluate a scalar H1 field and projected physical derivatives with tensor
// kernels. Collocated nodal derivatives are shared-node averaged, matching
// GetDerivative, before quadrature sampling. nodal_ir must be collocated;
// derivative outputs are optional.
void EvaluateScalarFieldWithTensorKernels(
   const ParGridFunction &field,
   const IntegrationRule &ir,
   const IntegrationRule &nodal_ir,
   Vector &values,
   Vector *gradient = nullptr,
   Vector *hessian = nullptr)
{
   const ParFiniteElementSpace &fes = *field.ParFESpace();
   const int dim = fes.GetMesh()->Dimension();
   const int sdim = fes.GetMesh()->SpaceDimension();
   const int ne = fes.GetNE();
   const int nq = ir.GetNPoints();
   const MemoryType mt = Device::GetDeviceMemoryType();

   MFEM_VERIFY(fes.GetVDim() == 1,
               "Device scalar-field evaluation requires VDIM = 1.");
   MFEM_VERIFY(dim == sdim && (dim == 2 || dim == 3),
               "Device scalar-field derivatives require a 2D or 3D volume mesh.");
   MFEM_VERIFY(UsesTensorBasis(fes),
               "Device scalar-field derivatives require a tensor-product basis.");
   MFEM_VERIFY(nodal_ir.GetNPoints() == fes.GetTypicalFE()->GetDof(),
               "Collocated derivative rule must contain one point per DOF.");

   EvaluateFieldValuesOnDevice(field, ir, values);
   if (!gradient && !hessian) { return; }

   ParFiniteElementSpace gradient_fes(fes.GetParMesh(), fes.FEColl(), dim,
                                      Ordering::byNODES);
   ParGridFunction projected_gradient(&gradient_fes);
   projected_gradient.UseDevice(true);
   ProjectPhysicalGradientOnDevice(field, projected_gradient, nodal_ir);
   if (gradient)
   {
      EvaluateFieldValuesOnDevice(projected_gradient, ir, *gradient);
   }

   if (!hessian) { return; }

   ParFiniteElementSpace hessian_fes(fes.GetParMesh(), fes.FEColl(), dim * dim,
                                     Ordering::byNODES);
   ParGridFunction projected_hessian(&hessian_fes);
   projected_hessian.UseDevice(true);
   ProjectPhysicalGradientOnDevice(projected_gradient, projected_hessian,
                                   nodal_ir);

   // Convert row-major projected components to first-Hessian-index-fastest.
   Vector component_major_hessian;
   EvaluateFieldValuesOnDevice(projected_hessian, ir,
                               component_major_hessian);
   hessian->SetSize(ne * nq * dim * dim, mt);
   const real_t *source = component_major_hessian.Read();
   real_t *destination = hessian->Write();
   mfem::forall(ne * nq * dim * dim, [=] MFEM_HOST_DEVICE (int k)
   {
      const int q = k % nq;
      const int first = (k / nq) % dim;
      const int second = (k / (nq * dim)) % dim;
      const int e = k / (nq * dim * dim);
      destination[k] =
         source[q + nq * (first * dim + second + dim * dim * e)];
   });
}

class ScopedHostGridFunctionAccess
{
public:
   ScopedHostGridFunctionAccess(GridFunction &field_, bool overwrite = false)
      : field(field_), used_device(field_.UseDevice())
   {
      if (overwrite) { field.HostWrite(); }
      else { field.HostRead(); }
      field.UseDevice(false);
   }

   ~ScopedHostGridFunctionAccess()
   {
      field.UseDevice(used_device);
   }

private:
   GridFunction &field;
   bool used_device;
};

/** Evaluate a scalar H1 field through host GetDerivative/GetValues, using the
    same layouts as EvaluateScalarFieldWithTensorKernels. */
void EvaluateScalarFieldOnHost(const ParGridFunction &field,
                               const IntegrationRule &ir,
                               Vector &values,
                               Vector *gradient = nullptr,
                               Vector *hessian = nullptr)
{
   ParFiniteElementSpace &fes = *field.ParFESpace();
   const int dim = fes.GetMesh()->Dimension();
   const int ne = fes.GetNE();
   const int nq = ir.GetNPoints();
   MFEM_VERIFY(fes.GetVDim() == 1 && (dim == 2 || dim == 3),
               "Classic scalar-field derivatives require a scalar 2D or 3D "
               "H1 field.");

   ScopedHostGridFunctionAccess field_access(
      const_cast<ParGridFunction &>(field));
   GridFunction *mesh_nodes = fes.GetMesh()->GetNodes();
   std::unique_ptr<ScopedHostGridFunctionAccess> node_access;
   if (mesh_nodes)
   {
      node_access = std::make_unique<ScopedHostGridFunctionAccess>(
                       *mesh_nodes);
   }

   std::vector<std::unique_ptr<ParGridFunction>> gradients;
   std::vector<std::unique_ptr<ParGridFunction>> hessians;
   if (gradient || hessian)
   {
      gradients.reserve(dim);
      for (int d = 0; d < dim; d++)
      {
         gradients.emplace_back(std::make_unique<ParGridFunction>(&fes));
         gradients.back()->UseDevice(false);
         field.GetDerivative(1, d, *gradients.back());
      }
   }
   if (hessian)
   {
      hessians.reserve(dim * dim);
      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < dim; j++)
         {
            hessians.emplace_back(std::make_unique<ParGridFunction>(&fes));
            hessians.back()->UseDevice(false);
            gradients[d]->GetDerivative(1, j, *hessians.back());
         }
      }
   }

   const MemoryType mt = Device::GetDeviceMemoryType();
   values.SetSize(ne * nq, mt);
   real_t *value_data = values.HostWrite();
   real_t *gradient_data = nullptr;
   real_t *hessian_data = nullptr;
   if (gradient)
   {
      gradient->SetSize(ne * nq * dim, mt);
      gradient_data = gradient->HostWrite();
   }
   if (hessian)
   {
      hessian->SetSize(ne * nq * dim * dim, mt);
      hessian_data = hessian->HostWrite();
   }

   Vector element_values;
   std::vector<Vector> element_gradients(dim);
   std::vector<Vector> element_hessians(dim * dim);
   for (int e = 0; e < ne; e++)
   {
      field.GetValues(e, ir, element_values);
      const real_t *element_value_data = element_values.HostRead();
      for (int q = 0; q < nq; q++)
      {
         value_data[q + nq * e] = element_value_data[q];
      }

      if (gradient || hessian)
      {
         for (int d = 0; d < dim; d++)
         {
            gradients[d]->GetValues(e, ir, element_gradients[d]);
            if (gradient)
            {
               const real_t *element_gradient_data =
                  element_gradients[d].HostRead();
               for (int q = 0; q < nq; q++)
               {
                  gradient_data[q + nq * (d + dim * e)] =
                     element_gradient_data[q];
               }
            }
         }
      }
      if (hessian)
      {
         for (int d = 0; d < dim; d++)
         {
            for (int j = 0; j < dim; j++)
            {
               const int component = d * dim + j;
               hessians[component]->GetValues(
                  e, ir, element_hessians[component]);
               const real_t *element_hessian_data =
                  element_hessians[component].HostRead();
               for (int q = 0; q < nq; q++)
               {
                  hessian_data[q + nq * (d + dim * (j + dim * e))] =
                     element_hessian_data[q];
               }
            }
         }
      }
   }

   // Permit later device packing; this is free with -d cpu.
   values.UseDevice(true);
   if (gradient) { gradient->UseDevice(true); }
   if (hessian) { hessian->UseDevice(true); }
}

/** Compute nodal gradient and Hessian fields with host GetDerivative. */
void ProjectPhysicalDerivativesOnHost(const ParGridFunction &field,
                                      ParGridFunction &gradient,
                                      ParGridFunction &hessian)
{
   ParFiniteElementSpace &scalar_fes = *field.ParFESpace();
   const int dim = scalar_fes.GetMesh()->Dimension();
   const int scalar_size = field.Size();
   MFEM_VERIFY(gradient.Size() == dim * scalar_size &&
               hessian.Size() == dim * dim * scalar_size,
               "Incompatible classic derivative fields.");

   ScopedHostGridFunctionAccess field_access(
      const_cast<ParGridFunction &>(field));
   GridFunction *mesh_nodes = scalar_fes.GetMesh()->GetNodes();
   std::unique_ptr<ScopedHostGridFunctionAccess> node_access;
   if (mesh_nodes)
   {
      node_access = std::make_unique<ScopedHostGridFunctionAccess>(
                       *mesh_nodes);
   }
   ScopedHostGridFunctionAccess gradient_access(gradient, true);
   ScopedHostGridFunctionAccess hessian_access(hessian, true);

   real_t *gradient_data = gradient.GetData();
   for (int d = 0; d < dim; d++)
   {
      ParGridFunction grad_component(&scalar_fes,
                                     gradient_data + d * scalar_size);
      grad_component.UseDevice(false);
      field.GetDerivative(1, d, grad_component);
   }

   real_t *hessian_data = hessian.GetData();
   for (int d = 0; d < dim; d++)
   {
      ParGridFunction grad_component(&scalar_fes,
                                     gradient_data + d * scalar_size);
      grad_component.UseDevice(false);
      for (int j = 0; j < dim; j++)
      {
         ParGridFunction hess_component(
            &scalar_fes, hessian_data + (d * dim + j) * scalar_size);
         hess_component.UseDevice(false);
         grad_component.GetDerivative(1, j, hess_component);
      }
   }
}

/** Project collocated physical derivatives into H1. Kernels run on device;
    shared-DOF reduction is host-staged for GroupCommunicator. */
void ProjectPhysicalGradientOnDevice(const ParGridFunction &field,
                                     ParGridFunction &gradient,
                                     const IntegrationRule &nodal_ir)
{
   const ParFiniteElementSpace &in_fes = *field.ParFESpace();
   ParFiniteElementSpace &out_fes = *gradient.ParFESpace();
   const FiniteElement &fe = *in_fes.GetTypicalFE();
   const int dim = in_fes.GetMesh()->Dimension();
   const int sdim = in_fes.GetMesh()->SpaceDimension();
   const int ne = in_fes.GetNE();
   const int nd = fe.GetDof();
   const int in_vdim = in_fes.GetVDim();
   const MemoryType mt = Device::GetDeviceMemoryType();

   MFEM_VERIFY(in_fes.GetMesh() == out_fes.GetMesh() &&
               out_fes.GetVDim() == in_vdim * dim &&
               out_fes.GetOrdering() == Ordering::byNODES,
               "Incompatible output space for projected device gradient.");
   MFEM_VERIFY(UsesTensorBasis(in_fes) && nodal_ir.GetNPoints() == nd,
               "Projected device gradients require collocated tensor elements.");

   const Operator *in_restriction =
      in_fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   const Operator *out_restriction =
      out_fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(in_restriction && out_restriction,
               "Element restrictions are required.");

   Vector element_values;
   element_values.SetSize(in_restriction->Height(), mt);
   in_restriction->Mult(field, element_values);

   const DofToQuad &maps = fe.GetDofToQuad(nodal_ir, DofToQuad::TENSOR);
   const int d1d = maps.ndof;
   const int q1d = maps.nqpt;
   const int tensor_ndof = d1d * d1d * ((dim == 3) ? d1d : 1);
   MFEM_VERIFY(d1d == q1d && tensor_ndof == nd,
               "Unexpected collocated derivative map dimensions.");
   const GeometricFactors *geom = in_fes.GetMesh()->GetGeometricFactors(
                                     nodal_ir,
                                     GeometricFactors::JACOBIANS,
                                     mt);
   Vector raw_gradient, element_gradient;
   raw_gradient.SetSize(out_restriction->Height(), mt);
   QuadratureInterpolator::CollocatedGradKernels::Run(
      dim, QVectorLayout::byNODES, true, in_vdim, d1d,
      ne, maps.G.Read(), geom->J.Read(), element_values.Read(),
      raw_gradient.Write(), sdim, in_vdim, d1d);

   // Convert (input, direction) kernel ordering to row-major components.
   element_gradient.SetSize(out_restriction->Height(), mt);
   const real_t *raw_data = raw_gradient.Read();
   real_t *element_gradient_data = element_gradient.Write();
   mfem::forall(ne * nd * in_vdim * dim,
                [=] MFEM_HOST_DEVICE (int k)
   {
      const int node = k % nd;
      const int component = (k / nd) % in_vdim;
      const int direction = (k / (nd * in_vdim)) % dim;
      const int element = k / (nd * in_vdim * dim);
      const int raw_component = component + in_vdim * direction;
      const int output_component = component * dim + direction;
      element_gradient_data[node + nd *
                            (output_component + in_vdim * dim * element)] =
         raw_data[node + nd *
                  (raw_component + in_vdim * dim * element)];
   });
   out_restriction->MultTranspose(element_gradient, gradient);

   Vector element_ones, overlap;
   element_ones.SetSize(out_restriction->Height(), mt);
   element_ones = 1.0;
   overlap.SetSize(out_fes.GetVSize(), mt);
   out_restriction->AbsMultTranspose(element_ones, overlap);

   int nranks = 1;
   MPI_Comm_size(out_fes.GetComm(), &nranks);
   if (nranks > 1)
   {
      GroupCommunicator &gcomm = out_fes.GroupComm();
      real_t *gradient_data = gradient.HostReadWrite();
      real_t *overlap_data = overlap.HostReadWrite();
      gcomm.Reduce<real_t>(gradient_data, GroupCommunicator::Sum);
      gcomm.Bcast<real_t>(gradient_data);
      gcomm.Reduce<real_t>(overlap_data, GroupCommunicator::Sum);
      gcomm.Bcast<real_t>(overlap_data);
   }

   real_t *gradient_data = gradient.ReadWrite();
   const real_t *overlap_data = overlap.Read();
   const int size = gradient.Size();
   mfem::forall(size, [=] MFEM_HOST_DEVICE (int i)
   {
      gradient_data[i] /= overlap_data[i];
   });
}

// Target 5 size remapping, matching classic 3D -ae 1.
class Target5Remap
{
public:
   explicit Target5Remap(
      ParMesh &pmesh,
      int derivative_backend_ = TENSOR_KERNEL_DERIVATIVES)
      : mesh(pmesh),
        source_mesh(pmesh, true),
        fec(1, pmesh.Dimension()),
        source_fes(&source_mesh, &fec),
        current_fes(&pmesh, &fec),
        initial_size(&source_fes),
        current_size(&current_fes),
        nodal_ir(MakeTensorNodalIntegrationRule(*current_fes.GetTypicalFE())),
        derivative_backend(derivative_backend_)
   {
      MFEM_VERIFY(pmesh.Dimension() == 3,
                  "Target id 5 is implemented only for 3D meshes.");
      initial_size.UseDevice(true);
      current_size.UseDevice(true);
      ConstructSizeGF(initial_size);
      current_size = initial_size;

      const real_t local_min_size = initial_size.Min();
      MPI_Allreduce(&local_min_size, &min_size, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    pmesh.GetComm());

#ifdef MFEM_USE_GSLIB
      finder = std::make_unique<FindPointsGSLIB>(source_mesh.GetComm());
      finder->Setup(source_mesh, 0.1, 1.0e-12, 256);
#else
      MFEM_ABORT("Target id 5 in pmesh-optimizer-enzyme requires GSLIB "
                 "for reinterpolation.");
#endif
   }

   real_t MinSize() const { return min_size; }
   const ParGridFunction &Size() const { return initial_size; }
   const ParGridFunction &CurrentSize() const { return current_size; }

   void Remap(const ParGridFunction &nodes) const
   {
#ifdef MFEM_USE_GSLIB
      Vector point_positions, element_size;
      BuildNodalPointListOnDevice(nodes, current_fes, nodal_ir,
                                  point_positions);
      finder->FindPoints(point_positions, Ordering::byNODES);
      finder->Interpolate(initial_size, element_size, Ordering::byNODES);
      ScatterNodalValuesOnDevice(element_size, current_size);
#else
      MFEM_CONTRACT_VAR(nodes);
      MFEM_ABORT("Target id 5 in pmesh-optimizer-enzyme requires GSLIB "
                 "for reinterpolation.");
#endif
   }

   void FillQuadratureData(const ParGridFunction &nodes,
                           const QuadratureSpace &qspace,
                           QuadratureFunction &qdata,
                           bool include_derivatives) const
   {
      Remap(nodes);

      // Derivatives of the remapped field use the current mesh geometry.
      mesh.SetNodalGridFunction(const_cast<ParGridFunction *>(&nodes));
      mesh.NodesUpdated();

      const IntegrationRule &ir = qspace.GetIntRule(0);
      const int ne = qspace.GetNE();
      const int nq = ir.GetNPoints();
      MFEM_VERIFY(qdata.Size() == ne * nq * TARGET5_DATA_SIZE,
                  "Unexpected target-5 quadrature layout.");
      Vector size_values, gradients, hessians;
      if (include_derivatives &&
          derivative_backend == CLASSIC_HOST_DERIVATIVES)
      {
         EvaluateScalarFieldOnHost(current_size, ir, size_values,
                                   &gradients, &hessians);
      }
      else
      {
         EvaluateScalarFieldWithTensorKernels(
            current_size, ir, nodal_ir, size_values,
            include_derivatives ? &gradients : nullptr,
            include_derivatives ? &hessians : nullptr);
      }

      const real_t *size_data = size_values.Read();
      const real_t *grad_data = include_derivatives ? gradients.Read() : nullptr;
      const real_t *hess_data = include_derivatives ? hessians.Read() : nullptr;
      real_t *target_data = qdata.Write();
      const real_t minimum_size = min_size;
      mfem::forall(ne * nq, [=] MFEM_HOST_DEVICE (int k)
      {
         const int q = k % nq;
         const int e = k / nq;
         real_t *data = target_data + k * TARGET5_DATA_SIZE;
         for (int i = 0; i < TARGET5_DATA_SIZE; i++) { data[i] = 0.0; }
         data[TARGET5_VALUE] = size_data[k];
         data[TARGET5_MIN_SIZE] = minimum_size;
         if (include_derivatives)
         {
            for (int d = 0; d < 3; d++)
            {
               data[TARGET5_GRAD + d] = grad_data[q + nq * (d + 3 * e)];
               for (int j = 0; j < 3; j++)
               {
                  data[TARGET5_HESS + d * 3 + j] =
                     hess_data[q + nq * (d + 3 * (j + 3 * e))];
               }
            }
         }
      });
   }

private:
   ParMesh &mesh;
   ParMesh source_mesh;
   H1_FECollection fec;
   ParFiniteElementSpace source_fes;
   ParFiniteElementSpace current_fes;
   ParGridFunction initial_size;
   mutable ParGridFunction current_size;
   IntegrationRule nodal_ir;
   int derivative_backend;
   real_t min_size = 0.0;
#ifdef MFEM_USE_GSLIB
   std::unique_ptr<FindPointsGSLIB> finder;
#endif
};

class Target6Remap
{
public:
   enum Field
   {
      SIZE,
      ASPECT,
      NUM_SOURCE_FIELDS
   };

   explicit Target6Remap(ParMesh &pmesh,
                         int derivative_backend_ =
                            TENSOR_KERNEL_DERIVATIVES)
      : mesh(pmesh),
        source_mesh(pmesh, true),
        fec(1, 2),
        fes(&source_mesh, &fec),
        current_fes(&pmesh, &fec),
        current_size(&current_fes),
        current_aspect(&current_fes),
        nodal_ir(MakeTensorNodalIntegrationRule(*current_fes.GetTypicalFE())),
        derivative_backend(derivative_backend_)
   {
      MFEM_VERIFY(source_mesh.Dimension() == 2,
                  "Target id 6 is implemented only for 2D meshes.");
      MFEM_VERIFY(parameters.size_ratio > 1.0,
                  "Target id 6 size_ratio must be greater than 1.");
      MFEM_VERIFY(parameters.aspect_ratio >= 1.0,
                  "Target id 6 aspect_ratio must be at least 1.");
      MFEM_VERIFY(parameters.metric_shape_weight >= 0.0 &&
                  parameters.metric_shape_weight <= 1.0,
                  "Target id 6 metric_shape_weight must be in [0, 1].");
      fields.reserve(NUM_SOURCE_FIELDS);
      for (int i = 0; i < NUM_SOURCE_FIELDS; i++)
      {
         fields.emplace_back(std::make_unique<ParGridFunction>(&fes));
         fields.back()->UseDevice(true);
      }
      current_size.UseDevice(true);
      current_aspect.UseDevice(true);
      BuildSizeAndAspectFields();
      BuildFinder();
   }

   real_t MinSize() const { return min_size; }

   const ParGridFunction &Size() const { return *fields[SIZE]; }
   const ParGridFunction &Aspect() const { return *fields[ASPECT]; }
   const ParGridFunction &CurrentSize() const { return current_size; }
   const ParGridFunction &CurrentAspect() const { return current_aspect; }

   void Remap(const ParGridFunction &nodes) const
   {
      UpdateCurrentTargetFields(nodes, false);
   }

   void FillQuadratureData(const ParGridFunction &nodes,
                           const QuadratureSpace &qspace,
                           QuadratureFunction &qdata,
                           bool include_derivatives) const
   {
#ifdef MFEM_USE_GSLIB
      mesh.SetNodalGridFunction(const_cast<ParGridFunction *>(&nodes));
      mesh.NodesUpdated();

      UpdateCurrentTargetFields(nodes, include_derivatives);
      const IntegrationRule &ir = qspace.GetIntRule(0);
      const int ne = qspace.GetNE();
      const int nq = ir.GetNPoints();
      MFEM_VERIFY(qdata.Size() == ne * nq * TARGET6_DATA_SIZE,
                  "Unexpected target-6 quadrature layout.");

      Vector size_values, size_gradients, size_hessians;
      Vector aspect_values, aspect_gradients, aspect_hessians;
      if (include_derivatives &&
          derivative_backend == CLASSIC_HOST_DERIVATIVES)
      {
         EvaluateScalarFieldOnHost(current_size, ir, size_values,
                                   &size_gradients, &size_hessians);
         EvaluateScalarFieldOnHost(current_aspect, ir, aspect_values,
                                   &aspect_gradients, &aspect_hessians);
      }
      else
      {
         EvaluateScalarFieldWithTensorKernels(
            current_size, ir, nodal_ir, size_values,
            include_derivatives ? &size_gradients : nullptr,
            include_derivatives ? &size_hessians : nullptr);
         EvaluateScalarFieldWithTensorKernels(
            current_aspect, ir, nodal_ir, aspect_values,
            include_derivatives ? &aspect_gradients : nullptr,
            include_derivatives ? &aspect_hessians : nullptr);
      }

      const GeometricFactors *geom = mesh.GetGeometricFactors(
                                        ir,
                                        GeometricFactors::COORDINATES |
                                        GeometricFactors::JACOBIANS,
                                        Device::GetDeviceMemoryType());
      const real_t *positions = geom->X.Read();
      const real_t *size_data = size_values.Read();
      const real_t *aspect_data = aspect_values.Read();
      const real_t *size_grad = include_derivatives ?
                                size_gradients.Read() : nullptr;
      const real_t *aspect_grad = include_derivatives ?
                                  aspect_gradients.Read() : nullptr;
      const real_t *size_hess = include_derivatives ?
                                size_hessians.Read() : nullptr;
      const real_t *aspect_hess = include_derivatives ?
                                  aspect_hessians.Read() : nullptr;
      real_t *target_data = qdata.Write();
      const real_t minimum_size = min_size;
      const real_t shape_weight = parameters.metric_shape_weight;
      mfem::forall(ne * nq, [=] MFEM_HOST_DEVICE (int k)
      {
         const int q = k % nq;
         const int e = k / nq;
         real_t *data = target_data + k * TARGET6_DATA_SIZE;
         for (int i = 0; i < TARGET6_DATA_SIZE; i++) { data[i] = 0.0; }
         data[TARGET_POSITION] = positions[q + nq * (0 + 2 * e)];
         data[TARGET_POSITION + 1] = positions[q + nq * (1 + 2 * e)];
         data[TARGET_PRIMARY_VALUE] = size_data[k];
         data[TARGET_SECONDARY_VALUE] = aspect_data[k];
         if (include_derivatives)
         {
            data[TARGET_PRIMARY_GRAD] = size_grad[q + nq * (0 + 2 * e)];
            data[TARGET_PRIMARY_GRAD + 1] =
               size_grad[q + nq * (1 + 2 * e)];
            data[TARGET_SECONDARY_GRAD] =
               aspect_grad[q + nq * (0 + 2 * e)];
            data[TARGET_SECONDARY_GRAD + 1] =
               aspect_grad[q + nq * (1 + 2 * e)];
            data[TARGET_PRIMARY_HESS] =
               size_hess[q + nq * (0 + 2 * (0 + 2 * e))];
            data[TARGET_PRIMARY_HESS + 1] =
               size_hess[q + nq * (0 + 2 * (1 + 2 * e))];
            data[TARGET_PRIMARY_HESS + 2] =
               size_hess[q + nq * (1 + 2 * (1 + 2 * e))];
            data[TARGET_SECONDARY_HESS] =
               aspect_hess[q + nq * (0 + 2 * (0 + 2 * e))];
            data[TARGET_SECONDARY_HESS + 1] =
               aspect_hess[q + nq * (0 + 2 * (1 + 2 * e))];
            data[TARGET_SECONDARY_HESS + 2] =
               aspect_hess[q + nq * (1 + 2 * (1 + 2 * e))];
         }
         data[14] = minimum_size;
         data[15] = shape_weight;
      });
#else
      MFEM_CONTRACT_VAR(nodes);
      MFEM_CONTRACT_VAR(qspace);
      MFEM_CONTRACT_VAR(qdata);
      MFEM_ABORT("Target id 6 in pmesh-optimizer-enzyme requires GSLIB "
                 "for reinterpolation.");
#endif
   }

private:
   void BuildSizeAndAspectFields()
   {
      ParGridFunction disc(&fes);
      disc.UseDevice(true);
      FunctionCoefficient mat_coeff(material_indicator_2d);
      disc.ProjectCoefficient(mat_coeff);
      DiffuseField(disc, 2);

      ParFiniteElementSpace gradient_fes(&source_mesh, &fec, 2,
                                         Ordering::byNODES);
      ParGridFunction disc_gradient(&gradient_fes);
      disc_gradient.UseDevice(true);
      if (derivative_backend == CLASSIC_HOST_DERIVATIVES)
      {
         ParGridFunction derivative_x(&fes), derivative_y(&fes);
         derivative_x.UseDevice(false);
         derivative_y.UseDevice(false);
         ScopedHostGridFunctionAccess disc_access(disc);
         GridFunction *mesh_nodes = source_mesh.GetNodes();
         std::unique_ptr<ScopedHostGridFunctionAccess> node_access;
         if (mesh_nodes)
         {
            node_access = std::make_unique<ScopedHostGridFunctionAccess>(
                             *mesh_nodes);
         }
         disc.GetDerivative(1, 0, derivative_x);
         disc.GetDerivative(1, 1, derivative_y);
         const real_t *dx = derivative_x.HostRead();
         const real_t *dy = derivative_y.HostRead();
         real_t *gradient_data = disc_gradient.HostWrite();
         const int scalar_size = fes.GetVSize();
         for (int i = 0; i < scalar_size; i++)
         {
            gradient_data[i] = dx[i];
            gradient_data[i + scalar_size] = dy[i];
         }
      }
      else
      {
         ProjectPhysicalGradientOnDevice(disc, disc_gradient, nodal_ir);
      }

      ParGridFunction &size = *fields[SIZE];
      ParGridFunction &aspr = *fields[ASPECT];

      const int ndofs = size.Size();
      const real_t *gradient_data = disc_gradient.Read();
      real_t *size_data = size.Write();
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t dx = gradient_data[i];
         const real_t dy = gradient_data[i + ndofs];
         size_data[i] = dx * dx + dy * dy;
      });

      const real_t max_local = size.Max();
      real_t max_all = 0.0;
      MPI_Allreduce(&max_local, &max_all, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, source_mesh.GetComm());

      const real_t aspr_ratio = parameters.aspect_ratio;
      const real_t size_ratio = parameters.size_ratio;

      size_data = size.ReadWrite();
      real_t *aspect_data = aspr.Write();
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
      {
         size_data[i] /= max_all;
         const real_t raw_aspect =
            0.1 + 0.9 * (1.0 - size_data[i]) * (1.0 - size_data[i]);
         aspect_data[i] = raw_aspect > aspr_ratio ? aspr_ratio :
                          raw_aspect < 1.0 / aspr_ratio ?
                          1.0 / aspr_ratio : raw_aspect;
      });

      Vector vals;
      real_t volume = 0.0, volume_ind = 0.0;
      for (int i = 0; i < source_mesh.GetNE(); i++)
      {
         ElementTransformation *Tr = source_mesh.GetElementTransformation(i);
         const IntegrationRule &ir =
            IntRules.Get(source_mesh.GetElementBaseGeometry(i), Tr->OrderJ());
         size.GetValues(i, ir, vals);
         const real_t *vals_data = vals.HostRead();
         for (int j = 0; j < ir.GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            Tr->SetIntPoint(&ip);
            volume     += ip.weight * Tr->Weight();
            volume_ind += vals_data[j] * ip.weight * Tr->Weight();
         }
      }

      real_t volume_all = 0.0, volume_ind_all = 0.0;
      MPI_Allreduce(&volume, &volume_all, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, source_mesh.GetComm());
      MPI_Allreduce(&volume_ind, &volume_ind_all, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    source_mesh.GetComm());

      const real_t avg_zone_size = volume_all / source_mesh.GetGlobalNE();
      const real_t small_avg_ratio =
         (volume_ind_all + (volume_all - volume_ind_all) / size_ratio)
         / volume_all;
      const real_t small_zone_size = small_avg_ratio * avg_zone_size;
      const real_t big_zone_size = size_ratio * small_zone_size;

      size_data = size.ReadWrite();
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t denom = 1.0 + (size_ratio - 1.0) * size_data[i];
         size_data[i] = big_zone_size / denom;
      });

      DiffuseField(size, 2);
      DiffuseField(aspr, 2);
      const real_t min_size_local = size.Min();
      MPI_Allreduce(&min_size_local, &min_size, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    source_mesh.GetComm());
   }

   void UpdateCurrentTargetFields(const ParGridFunction &nodes,
                                  bool include_derivatives) const
   {
#ifdef MFEM_USE_GSLIB
      Vector point_positions, element_size, element_aspect;
      BuildNodalPointListOnDevice(nodes, current_fes, nodal_ir,
                                  point_positions);
      finder->FindPoints(point_positions, Ordering::byNODES);
      finder->Interpolate(*fields[SIZE], element_size, Ordering::byNODES);
      finder->Interpolate(*fields[ASPECT], element_aspect, Ordering::byNODES);
      ScatterNodalValuesOnDevice(element_size, current_size);
      ScatterNodalValuesOnDevice(element_aspect, current_aspect);
      MFEM_CONTRACT_VAR(include_derivatives);
#else
      MFEM_CONTRACT_VAR(nodes);
      MFEM_ABORT("Target id 6 in pmesh-optimizer-enzyme requires GSLIB "
                 "for reinterpolation.");
#endif
   }

   void BuildFinder()
   {
#ifdef MFEM_USE_GSLIB
      const real_t rel_bbox_el = 0.1;
      const real_t newton_tol  = 1.0e-12;
      const int npts_at_once   = 256;
      finder = std::make_unique<FindPointsGSLIB>(source_mesh.GetComm());
      finder->Setup(source_mesh, rel_bbox_el, newton_tol, npts_at_once);
#else
      MFEM_ABORT("Target id 6 in pmesh-optimizer-enzyme requires GSLIB "
                 "for reinterpolation.");
#endif
   }

   ParMesh &mesh;
   ParMesh source_mesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   ParFiniteElementSpace current_fes;
   std::vector<std::unique_ptr<ParGridFunction>> fields;
   mutable ParGridFunction current_size;
   mutable ParGridFunction current_aspect;
   IntegrationRule nodal_ir;
   Target6Parameters parameters;
   int derivative_backend;
   real_t min_size = 0.0;
#ifdef MFEM_USE_GSLIB
   std::unique_ptr<FindPointsGSLIB> finder;
#endif
};

// Target 8 discrete size and orientation data (2D only).
class Target8Remap
{
public:
   enum Field { SIZE, ORI, NUM_SOURCE_FIELDS };

   explicit Target8Remap(
      ParMesh &pmesh,
      int derivative_backend_ = TENSOR_KERNEL_DERIVATIVES)
      : mesh(pmesh),
        source_mesh(pmesh, true),
        fec(1, 2),
        fes(&source_mesh, &fec),
        current_fes(&pmesh, &fec),
        current_size(&current_fes),
        current_ori(&current_fes),
        nodal_ir(MakeTensorNodalIntegrationRule(*current_fes.GetTypicalFE())),
        derivative_backend(derivative_backend_)
   {
      MFEM_VERIFY(source_mesh.Dimension() == 2,
                  "Target id 8 is implemented only for 2D meshes.");
      fields.reserve(NUM_SOURCE_FIELDS);
      for (int i = 0; i < NUM_SOURCE_FIELDS; i++)
      {
         fields.emplace_back(std::make_unique<ParGridFunction>(&fes));
         fields.back()->UseDevice(true);
      }
      current_size.UseDevice(true);
      current_ori.UseDevice(true);
      BuildSizeAndOriFields();
      BuildFinder();
   }

   real_t MinSize() const { return min_size; }

   const ParGridFunction &Size() const { return *fields[SIZE]; }
   const ParGridFunction &Ori() const { return *fields[ORI]; }
   const ParGridFunction &CurrentSize() const { return current_size; }
   const ParGridFunction &CurrentOri() const { return current_ori; }

   void Remap(const ParGridFunction &nodes) const
   {
      UpdateCurrentTargetFields(nodes, false);
   }

   void FillQuadratureData(const ParGridFunction &nodes,
                           const QuadratureSpace &qspace,
                           QuadratureFunction &qdata,
                           bool include_derivatives) const
   {
#ifdef MFEM_USE_GSLIB
      mesh.SetNodalGridFunction(const_cast<ParGridFunction *>(&nodes));
      mesh.NodesUpdated();

      UpdateCurrentTargetFields(nodes, include_derivatives);
      const IntegrationRule &ir = qspace.GetIntRule(0);
      const int ne = qspace.GetNE();
      const int nq = ir.GetNPoints();
      MFEM_VERIFY(qdata.Size() == ne * nq * TARGET8_DATA_SIZE,
                  "Unexpected target-8 quadrature layout.");

      Vector size_values, size_gradients, size_hessians;
      Vector ori_values, ori_gradients, ori_hessians;
      if (include_derivatives &&
          derivative_backend == CLASSIC_HOST_DERIVATIVES)
      {
         EvaluateScalarFieldOnHost(current_size, ir, size_values,
                                   &size_gradients, &size_hessians);
         EvaluateScalarFieldOnHost(current_ori, ir, ori_values,
                                   &ori_gradients, &ori_hessians);
      }
      else
      {
         EvaluateScalarFieldWithTensorKernels(
            current_size, ir, nodal_ir, size_values,
            include_derivatives ? &size_gradients : nullptr,
            include_derivatives ? &size_hessians : nullptr);
         EvaluateScalarFieldWithTensorKernels(
            current_ori, ir, nodal_ir, ori_values,
            include_derivatives ? &ori_gradients : nullptr,
            include_derivatives ? &ori_hessians : nullptr);
      }

      const GeometricFactors *geom = mesh.GetGeometricFactors(
                                        ir,
                                        GeometricFactors::COORDINATES |
                                        GeometricFactors::JACOBIANS,
                                        Device::GetDeviceMemoryType());
      const real_t *positions = geom->X.Read();
      const real_t *size_data = size_values.Read();
      const real_t *ori_data = ori_values.Read();
      const real_t *size_grad = include_derivatives ?
                                size_gradients.Read() : nullptr;
      const real_t *ori_grad = include_derivatives ?
                               ori_gradients.Read() : nullptr;
      const real_t *size_hess = include_derivatives ?
                                size_hessians.Read() : nullptr;
      const real_t *ori_hess = include_derivatives ?
                               ori_hessians.Read() : nullptr;
      real_t *target_data = qdata.Write();
      mfem::forall(ne * nq, [=] MFEM_HOST_DEVICE (int k)
      {
         const int q = k % nq;
         const int e = k / nq;
         real_t *data = target_data + k * TARGET8_DATA_SIZE;
         for (int i = 0; i < TARGET8_DATA_SIZE; i++) { data[i] = 0.0; }
         data[TARGET_POSITION] = positions[q + nq * (0 + 2 * e)];
         data[TARGET_POSITION + 1] = positions[q + nq * (1 + 2 * e)];
         data[TARGET_PRIMARY_VALUE] = size_data[k];
         data[TARGET_SECONDARY_VALUE] = ori_data[k];
         if (include_derivatives)
         {
            data[TARGET_PRIMARY_GRAD] = size_grad[q + nq * (0 + 2 * e)];
            data[TARGET_PRIMARY_GRAD + 1] =
               size_grad[q + nq * (1 + 2 * e)];
            data[TARGET_SECONDARY_GRAD] = ori_grad[q + nq * (0 + 2 * e)];
            data[TARGET_SECONDARY_GRAD + 1] =
               ori_grad[q + nq * (1 + 2 * e)];
            data[TARGET_PRIMARY_HESS] =
               size_hess[q + nq * (0 + 2 * (0 + 2 * e))];
            data[TARGET_PRIMARY_HESS + 1] =
               size_hess[q + nq * (0 + 2 * (1 + 2 * e))];
            data[TARGET_PRIMARY_HESS + 2] =
               size_hess[q + nq * (1 + 2 * (1 + 2 * e))];
            data[TARGET_SECONDARY_HESS] =
               ori_hess[q + nq * (0 + 2 * (0 + 2 * e))];
            data[TARGET_SECONDARY_HESS + 1] =
               ori_hess[q + nq * (0 + 2 * (1 + 2 * e))];
            data[TARGET_SECONDARY_HESS + 2] =
               ori_hess[q + nq * (1 + 2 * (1 + 2 * e))];
         }
      });
#else
      MFEM_CONTRACT_VAR(nodes);
      MFEM_CONTRACT_VAR(qspace);
      MFEM_CONTRACT_VAR(qdata);
      MFEM_CONTRACT_VAR(include_derivatives);
      MFEM_ABORT("Target id 8 in pmesh-optimizer-enzyme requires GSLIB "
                 "for reinterpolation.");
#endif
   }

private:
   void BuildSizeAndOriFields()
   {
      ParGridFunction &size = *fields[SIZE];
      ParGridFunction &ori = *fields[ORI];

      // Constant small size and analytic orientation from mesh-optimizer.hpp.
      ConstantCoefficient size_coeff(0.1 * 0.1);
      size.ProjectCoefficient(size_coeff);

      auto ori_func = [](const Vector &x_vec)
      {
         return M_PI * x_vec(1) * (1.0 - x_vec(1)) * std::cos(2 * M_PI * x_vec(0));
      };
      FunctionCoefficient ori_coeff(ori_func);
      ori.ProjectCoefficient(ori_coeff);

      const real_t min_size_local = size.Min();
      MPI_Allreduce(&min_size_local, &min_size, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    source_mesh.GetComm());
   }

   void UpdateCurrentTargetFields(const ParGridFunction &nodes,
                                  bool include_derivatives) const
   {
#ifdef MFEM_USE_GSLIB
      Vector point_positions, element_size, element_ori;
      BuildNodalPointListOnDevice(nodes, current_fes, nodal_ir,
                                  point_positions);
      finder->FindPoints(point_positions, Ordering::byNODES);
      finder->Interpolate(*fields[SIZE], element_size, Ordering::byNODES);
      finder->Interpolate(*fields[ORI], element_ori, Ordering::byNODES);
      ScatterNodalValuesOnDevice(element_size, current_size);
      ScatterNodalValuesOnDevice(element_ori, current_ori);
      MFEM_CONTRACT_VAR(include_derivatives);
#else
      MFEM_CONTRACT_VAR(nodes);
      MFEM_CONTRACT_VAR(include_derivatives);
      MFEM_ABORT("Target id 8 requires GSLIB.");
#endif
   }

   void BuildFinder()
   {
#ifdef MFEM_USE_GSLIB
      const real_t rel_bbox_el = 0.1;
      const real_t newton_tol  = 1.0e-12;
      const int npts_at_once   = 256;
      finder = std::make_unique<FindPointsGSLIB>(source_mesh.GetComm());
      finder->Setup(source_mesh, rel_bbox_el, newton_tol, npts_at_once);
#else
      MFEM_ABORT("Target id 8 requires GSLIB.");
#endif
   }

   ParMesh &mesh;
   ParMesh source_mesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   ParFiniteElementSpace current_fes;
   std::vector<std::unique_ptr<ParGridFunction>> fields;
   mutable ParGridFunction current_size;
   mutable ParGridFunction current_ori;
   IntegrationRule nodal_ir;
   int derivative_backend;
   real_t min_size = 0.0;
#ifdef MFEM_USE_GSLIB
   std::unique_ptr<FindPointsGSLIB> finder;
#endif
};

class SurfaceFittingData
{
public:
   SurfaceFittingData(ParMesh &pmesh,
                      ParFiniteElementSpace &mesh_fes,
                      const SurfaceFittingOptions &options,
                      int derivative_backend_ =
                         TENSOR_KERNEL_DERIVATIVES)
      : mesh(pmesh),
        dim(pmesh.Dimension()),
        order(GetH1Order(mesh_fes)),
        basis(GetH1BasisType(mesh_fes)),
        current_fec(order, dim, basis),
        current_fes(&pmesh, &current_fec),
        current_sigma(&current_fes),
        nodal_ir(MakeTensorNodalIntegrationRule(*current_fes.GetTypicalFE())),
        marker(*options.marker),
        coefficient(options.coefficient),
        source(options.source),
        analytic_level_set(options.analytic_level_set),
        discrete_derivative_mode(options.discrete_derivative_mode),
        discrete_from_background(options.discrete_from_background),
        derivative_backend(derivative_backend_),
        interface_parameters(options.interface_parameters ?
                             *options.interface_parameters :
                             Vector())
   {
      MFEM_VERIFY(dim == 2 || dim == 3,
                  "Surface fitting requires a 2D or 3D mesh.");
      MFEM_VERIFY(options.enabled && coefficient > 0.0,
                  "Surface fitting requires a positive coefficient.");
      MFEM_VERIFY(options.marker != nullptr,
                  "Surface fitting requires a DOF marker.");
      current_sigma.UseDevice(true);
      current_node_pos.UseDevice(true);
      sigma_samples.UseDevice(true);
      grad_samples.UseDevice(true);
      hess_samples.UseDevice(true);
      interface_parameters.UseDevice(true);
      MFEM_VERIFY(marker.Size() == current_fes.GetVSize(),
                  "Surface fitting marker size does not match scalar node space.");
      MFEM_VERIFY(analytic_level_set !=
                  SurfaceFittingOptions::QUADRATIC_INTERFACE ||
                  (dim == 2 && (interface_parameters.Size() == 2 ||
                                interface_parameters.Size() == 3)),
                  "Quadratic-interface surface fitting requires two or three "
                  "parameters.");
      MFEM_VERIFY(analytic_level_set !=
                  SurfaceFittingOptions::CUBIC_INTERFACE ||
                  (dim == 2 && interface_parameters.Size() == 4),
                  "Cubic-interface surface fitting requires four parameters.");

      ParGridFunction counter(&current_fes);
      counter.CountElementsPerVDof(dof_count);

      const int ndofs = current_fes.GetVSize();
      marked_dof_indices.Reserve(ndofs);
      const bool *marker_data = marker.HostRead();
      for (int i = 0; i < ndofs; i++)
      {
         if (marker_data[i]) { marked_dof_indices.Append(i); }
      }

      if (source == SurfaceFittingOptions::DISCRETE)
      {
         MFEM_VERIFY(options.discrete_level_set != nullptr,
                     "Discrete surface fitting requires an initial level set.");
         MFEM_VERIFY(discrete_derivative_mode ==
                     SurfaceFittingOptions::INTERPOLATED_SOURCE ||
                     discrete_derivative_mode ==
                     SurfaceFittingOptions::ELEMENT_LOCAL,
                     "Discrete derivative mode must be 1 or 2.");
         SetupDiscreteLevelSet(*options.discrete_level_set);
      }
   }

   void FillQuadratureData(const ParGridFunction &nodes,
                           const QuadratureSpace &node_qspace,
                           QuadratureFunction &qdata) const
   {
      UpdateCurrentNodes(nodes);
      if (source == SurfaceFittingOptions::DISCRETE)
      {
         UpdateDiscreteSamples();
      }
      else
      {
         UpdateAnalyticSamples();
      }

      if (derivative_backend == CLASSIC_HOST_DERIVATIVES)
      {
         FillQuadratureDataOnHost(node_qspace, qdata);
         return;
      }

      const IntegrationRule &ir = node_qspace.GetIntRule(0);
      const int ne = node_qspace.GetNE();
      const int nq = ir.GetNPoints();
      const int ndofs = current_fes.GetVSize();
      const int stride = SurfaceFitDataSize(dim);
      MFEM_VERIFY(nq == current_fes.GetTypicalFE()->GetDof() &&
                  qdata.Size() == ne * nq * stride,
                  "Unexpected surface-fitting nodal quadrature layout.");

      const bool use_element_derivatives =
         source == SurfaceFittingOptions::DISCRETE &&
         !discrete_from_background &&
         discrete_derivative_mode == SurfaceFittingOptions::ELEMENT_LOCAL;
      Vector element_values, element_gradients, element_hessians;
      if (use_element_derivatives)
      {
         EvaluateScalarFieldElementDerivativesOnDevice(
            current_sigma, ir, nodal_ir, element_values, &element_gradients,
            &element_hessians);
      }

      const auto *restriction = dynamic_cast<const ElementRestriction *>(
         current_fes.GetElementRestriction(
            ElementDofOrdering::LEXICOGRAPHIC));
      MFEM_VERIFY(restriction,
                  "Surface fitting requires an H1 element restriction.");
      const int *gather_map = restriction->GatherMap().Read();
      const bool *marker_data = marker.Read();
      const int *count_data = dof_count.Read();
      const real_t *sigma_data = use_element_derivatives ?
                                 element_values.Read() : sigma_samples.Read();
      const real_t *gradient_data = use_element_derivatives ?
                                    element_gradients.Read() :
                                    grad_samples.Read();
      const real_t *hessian_data = use_element_derivatives ?
                                   element_hessians.Read() :
                                   hess_samples.Read();
      real_t *qdata_ptr = qdata.Write();
      const int dimension = dim;
      const real_t fit_coefficient = coefficient;
      mfem::forall(ne * nq, [=] MFEM_HOST_DEVICE (int k)
      {
         const int q = k % nq;
         const int e = k / nq;
         const int signed_dof = gather_map[q + nq * e];
         const int dof = signed_dof >= 0 ? signed_dof : -1 - signed_dof;
         real_t *data = qdata_ptr + k * stride;
         for (int i = 0; i < stride; i++) { data[i] = 0.0; }
         data[SurfaceFitDataLayout<2>::COEFFICIENT] =
            marker_data[dof] ? fit_coefficient / count_data[dof] : 0.0;
         data[SurfaceFitDataLayout<2>::VALUE] =
            use_element_derivatives ? sigma_data[k] : sigma_data[dof];
         for (int d = 0; d < dimension; d++)
         {
            data[2 + d] = use_element_derivatives ?
               gradient_data[q + nq * (d + dimension * e)] :
               gradient_data[dof + d * ndofs];
            for (int j = 0; j < dimension; j++)
            {
               data[2 + dimension + d * dimension + j] =
                  use_element_derivatives ?
                  hessian_data[q + nq *
                               (d + dimension * (j + dimension * e))] :
                  hessian_data[dof + (d * dimension + j) * ndofs];
            }
         }
      });
   }

   void UpdateCurrentNodes(const ParGridFunction &nodes) const
   {
      MFEM_VERIFY(nodes.Size() == dim * current_fes.GetVSize(),
                  "Mesh and level-set nodal spaces must have the same order.");

      // Only element-local derivatives require current mesh transformations.
      if (source == SurfaceFittingOptions::DISCRETE &&
          !discrete_from_background &&
          discrete_derivative_mode == SurfaceFittingOptions::ELEMENT_LOCAL)
      {
         GridFunction *mesh_nodes = mesh.GetNodes();
         MFEM_VERIFY(mesh_nodes && mesh_nodes->Size() == nodes.Size(),
                     "Current mesh nodes are incompatible with the fitting "
                     "space.");
         *mesh_nodes = nodes;
         mesh.NodesUpdated();
         mesh.ExchangeFaceNbrData();
      }

      current_node_pos = nodes;
      if (nodes.ParFESpace()->GetOrdering() != Ordering::byNODES)
      {
         const int ndofs = current_fes.GetVSize();
         const int dimension = dim;
         const real_t *node_data = nodes.Read();
         real_t *position_data = current_node_pos.Write();
         mfem::forall(ndofs * dimension, [=] MFEM_HOST_DEVICE (int k)
         {
            const int i = k % ndofs;
            const int d = k / ndofs;
            position_data[k] = node_data[d + dimension * i];
         });
      }
   }

   void GetErrors(real_t &err_avg, real_t &err_max) const
   {
      err_avg = 0.0;
      err_max = 0.0;
      int count = 0;
      const bool *marker_data = marker.HostRead();
      const real_t *sigma_data = sigma_samples.HostRead();
      for (int i = 0; i < marker.Size(); i++)
      {
         if (!marker_data[i]) { continue; }
         if (current_fes.GetLocalTDofNumber(i) < 0) { continue; }
         const real_t err = std::abs(sigma_data[i]);
         err_avg += err;
         err_max = std::max(err_max, err);
         count++;
      }

      MPI_Allreduce(MPI_IN_PLACE, &err_avg, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, mesh.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &err_max, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, mesh.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, mesh.GetComm());
      if (count > 0) { err_avg /= count; }
   }

   void ScaleCoefficient(real_t factor)
   {
      MFEM_VERIFY(factor >= 1.0,
                  "Surface fitting coefficient scale must be at least one.");
      coefficient *= factor;
   }

   real_t GetCoefficient() const { return coefficient; }

private:
   static int GetH1Order(const ParFiniteElementSpace &fes)
   {
      const auto *fec = dynamic_cast<const H1_FECollection *>(fes.FEColl());
      MFEM_VERIFY(fec, "Surface fitting requires an H1 mesh space.");
      return fec->GetOrder();
   }

   static int GetH1BasisType(const ParFiniteElementSpace &fes)
   {
      const auto *fec = dynamic_cast<const H1_FECollection *>(fes.FEColl());
      MFEM_VERIFY(fec, "Surface fitting requires an H1 mesh space.");
      return fec->GetBasisType();
   }

   void ComputeElementDerivativesOnHost(int element,
                                        DenseMatrix &gradient,
                                        DenseMatrix &hessian) const
   {
      const FiniteElement &fe = *current_fes.GetFE(element);
      ElementTransformation &trans =
         *current_fes.GetElementTransformation(element);
      const int dof = fe.GetDof();

      Array<int> dofs;
      Vector sigma_element;
      current_fes.GetElementDofs(element, dofs);
      current_sigma.GetSubVector(dofs, sigma_element);

      DenseMatrix projected_gradient;
      fe.ProjectGrad(fe, trans, projected_gradient);
      gradient.SetSize(dof, dim);
      Vector gradient_data(gradient.GetData(), dof * dim);
      projected_gradient.Mult(sigma_element, gradient_data);

      // Match the classic element-local second ProjectGrad application.
      hessian.SetSize(dof * dim, dim);
      Mult(projected_gradient, gradient, hessian);
      hessian.SetSize(dof, dim * dim);
   }

   void FillQuadratureDataOnHost(const QuadratureSpace &node_qspace,
                                 QuadratureFunction &qdata) const
   {
      const int stride = SurfaceFitDataSize(dim);
      const int ndofs = current_fes.GetVSize();
      const bool element_local =
         source == SurfaceFittingOptions::DISCRETE &&
         !discrete_from_background &&
         discrete_derivative_mode == SurfaceFittingOptions::ELEMENT_LOCAL;
      std::unique_ptr<ScopedHostGridFunctionAccess> sigma_access;
      std::unique_ptr<ScopedHostGridFunctionAccess> node_access;
      if (element_local)
      {
         sigma_access = std::make_unique<ScopedHostGridFunctionAccess>(
                           current_sigma);
         GridFunction *mesh_nodes = mesh.GetNodes();
         if (mesh_nodes)
         {
            node_access = std::make_unique<ScopedHostGridFunctionAccess>(
                             *mesh_nodes);
         }
      }
      const bool *marker_data = marker.HostRead();
      const int *count_data = dof_count.HostRead();
      const real_t *sigma_data = sigma_samples.HostRead();
      const real_t *gradient_data = element_local ? nullptr :
                                    grad_samples.HostRead();
      const real_t *hessian_data = element_local ? nullptr :
                                   hess_samples.HostRead();
      real_t *quadrature_data = qdata.HostWrite();

      Array<int> dofs;
      for (int e = 0; e < node_qspace.GetNE(); e++)
      {
         const IntegrationRule &ir = node_qspace.GetIntRule(e);
         current_fes.GetElementDofs(e, dofs);
         MFEM_VERIFY(dofs.Size() == ir.GetNPoints(),
                     "Nodal quadrature must match scalar mesh DOFs.");
         const int offset = node_qspace.Offset(e);

         const FiniteElement *fe = current_fes.GetFE(e);
         const auto *nfe = dynamic_cast<const NodalFiniteElement *>(fe);
         const Array<int> *lex_to_native =
            nfe && nfe->GetLexicographicOrdering().Size() > 0 ?
            &nfe->GetLexicographicOrdering() : nullptr;

         bool compute_element_derivatives = element_local;
         if (compute_element_derivatives)
         {
            bool has_marked_dof = false;
            for (int i = 0; i < dofs.Size(); i++)
            {
               has_marked_dof = has_marked_dof || marker_data[dofs[i]];
            }
            compute_element_derivatives = has_marked_dof;
         }

         DenseMatrix element_gradient, element_hessian;
         if (compute_element_derivatives)
         {
            ComputeElementDerivativesOnHost(e, element_gradient,
                                            element_hessian);
         }

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const int local_dof = lex_to_native ? (*lex_to_native)[q] : q;
            const int dof = dofs[local_dof];
            real_t *data = quadrature_data + (offset + q) * stride;
            for (int i = 0; i < stride; i++) { data[i] = 0.0; }
            data[0] = marker_data[dof] ? coefficient / count_data[dof] : 0.0;
            data[1] = sigma_data[dof];

            if (compute_element_derivatives)
            {
               for (int d = 0; d < dim; d++)
               {
                  data[2 + d] = element_gradient(local_dof, d);
                  for (int j = 0; j < dim; j++)
                  {
                     data[2 + dim + d * dim + j] =
                        element_hessian(local_dof, d * dim + j);
                  }
               }
            }
            else if (!element_local)
            {
               for (int d = 0; d < dim; d++)
               {
                  data[2 + d] = gradient_data[dof + d * ndofs];
                  for (int j = 0; j < dim; j++)
                  {
                     data[2 + dim + d * dim + j] =
                        hessian_data[dof + (d * dim + j) * ndofs];
                  }
               }
            }
         }
      }
   }

   void SetupDiscreteLevelSet(const ParGridFunction &level_set)
   {
#ifdef MFEM_USE_GSLIB
      const ParFiniteElementSpace *level_fes = level_set.ParFESpace();
      MFEM_VERIFY(level_fes,
                  "The discrete level set must use a parallel FE space.");
      const auto *level_fec =
         dynamic_cast<const H1_FECollection *>(level_fes->FEColl());
      MFEM_VERIFY(level_fec,
                  "The discrete level set must use an H1 space.");

      source_mesh = std::make_unique<ParMesh>(*level_fes->GetParMesh(), true);
      source_fec = std::make_unique<H1_FECollection>(
                      level_fec->GetOrder(), dim, level_fec->GetBasisType());
      source_fes = std::make_unique<ParFiniteElementSpace>(
                      source_mesh.get(), source_fec.get());
      MFEM_VERIFY(level_set.Size() == source_fes->GetVSize(),
                  "The copied source space is incompatible with the discrete "
                  "level set.");
      source_sigma = std::make_unique<ParGridFunction>(source_fes.get());
      source_sigma->UseDevice(true);
      *source_sigma = level_set;
      source_nodal_ir =
         MakeTensorNodalIntegrationRule(*source_fes->GetTypicalFE());

      if (discrete_from_background ||
          discrete_derivative_mode ==
          SurfaceFittingOptions::INTERPOLATED_SOURCE)
      {
         source_grad_fes = std::make_unique<ParFiniteElementSpace>(
                              source_mesh.get(), source_fec.get(), dim,
                              Ordering::byNODES);
         source_hess_fes = std::make_unique<ParFiniteElementSpace>(
                              source_mesh.get(), source_fec.get(), dim * dim,
                              Ordering::byNODES);
         source_grad = std::make_unique<ParGridFunction>(source_grad_fes.get());
         source_hess = std::make_unique<ParGridFunction>(source_hess_fes.get());
         source_grad->UseDevice(true);
         source_hess->UseDevice(true);
         if (derivative_backend == CLASSIC_HOST_DERIVATIVES)
         {
            ProjectPhysicalDerivativesOnHost(*source_sigma, *source_grad,
                                             *source_hess);
         }
         else
         {
            ProjectPhysicalGradientOnDevice(*source_sigma, *source_grad,
                                            source_nodal_ir);
            ProjectPhysicalGradientOnDevice(*source_grad, *source_hess,
                                            source_nodal_ir);
         }
      }

      finder = std::make_unique<FindPointsGSLIB>(source_mesh->GetComm());
      finder->Setup(*source_mesh, 0.1, 1.0e-12, 256);
#else
      MFEM_CONTRACT_VAR(level_set);
      MFEM_ABORT("Discrete surface fitting requires GSLIB.");
#endif
   }

   void UpdateDiscreteSamples() const
   {
#ifdef MFEM_USE_GSLIB
      const int ndofs = current_fes.GetVSize();
      const int marked_count = marked_dof_indices.Size();

      if (!discrete_from_background &&
          discrete_derivative_mode == SurfaceFittingOptions::ELEMENT_LOCAL)
      {
         // Collocated derivatives need sigma at all element nodes, including
         // unmarked neighbors of fitting DOFs.
         finder->FindPoints(current_node_pos, Ordering::byNODES);
         finder->Interpolate(*source_sigma, sigma_samples,
                             Ordering::byNODES);
         current_sigma = sigma_samples;
         return;
      }

      sigma_samples.SetSize(ndofs);
      sigma_samples = 0.0;
      grad_samples.SetSize(ndofs * dim);
      grad_samples = 0.0;
      hess_samples.SetSize(ndofs * dim * dim);
      hess_samples = 0.0;
      if (marked_count == 0)
      {
         return;
      }

      Vector marked_positions(marked_count * dim);
      marked_positions.UseDevice(true);
      const real_t *position_data = current_node_pos.Read();
      const int *marked_dofs = marked_dof_indices.Read();
      real_t *marked_position_data = marked_positions.Write();
      const int dimension = dim;
      mfem::forall(marked_count * dim, [=] MFEM_HOST_DEVICE (int k)
      {
         const int i = k % marked_count;
         const int d = k / marked_count;
         marked_position_data[k] =
            position_data[marked_dofs[i] + d * ndofs];
      });

      Vector marked_sigma, marked_grad, marked_hess;
      finder->FindPoints(marked_positions, Ordering::byNODES);
      finder->Interpolate(*source_sigma, marked_sigma, Ordering::byNODES);
      finder->Interpolate(*source_grad, marked_grad, Ordering::byNODES);
      finder->Interpolate(*source_hess, marked_hess, Ordering::byNODES);

      const real_t *marked_sigma_data = marked_sigma.Read();
      const real_t *marked_grad_data = marked_grad.Read();
      const real_t *marked_hess_data = marked_hess.Read();
      real_t *sigma_data = sigma_samples.ReadWrite();
      real_t *grad_data = grad_samples.ReadWrite();
      real_t *hess_data = hess_samples.ReadWrite();
      mfem::forall(marked_count, [=] MFEM_HOST_DEVICE (int i)
      {
         const int dof = marked_dofs[i];
         sigma_data[dof] = marked_sigma_data[i];
         for (int d = 0; d < dimension; d++)
         {
            grad_data[dof + d * ndofs] =
               marked_grad_data[i + d * marked_count];
         }
         for (int h = 0; h < dimension * dimension; h++)
         {
            hess_data[dof + h * ndofs] =
               marked_hess_data[i + h * marked_count];
         }
      });
#else
      MFEM_ABORT("Discrete surface fitting requires GSLIB.");
#endif
   }

   void UpdateAnalyticSamples() const
   {
      const int ndofs = current_fes.GetVSize();
      sigma_samples.SetSize(ndofs);
      grad_samples.SetSize(dim * ndofs);
      hess_samples.SetSize(dim * dim * ndofs);

      const int num_parameters = interface_parameters.Size();
      const int dimension = dim;
      const int level_set = analytic_level_set;
      if (derivative_backend == CLASSIC_HOST_DERIVATIVES)
      {
         const real_t *position_data = current_node_pos.HostRead();
         real_t *sigma_data = sigma_samples.HostWrite();
         real_t *grad_data = grad_samples.HostWrite();
         real_t *hess_data = hess_samples.HostWrite();
         const real_t *parameters = interface_parameters.HostRead();
         for (int i = 0; i < ndofs; i++)
         {
            real_t x[3] = {0.0, 0.0, 0.0};
            real_t gradient[3] = {0.0, 0.0, 0.0};
            real_t hessian[9] = {0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0};
            for (int d = 0; d < dimension; d++)
            {
               x[d] = position_data[i + d * ndofs];
            }
            real_t sigma = 0.0;
            EvalAnalyticLevelSet(dimension, level_set, x, parameters,
                                 num_parameters, sigma, gradient, hessian);
            sigma_data[i] = sigma;
            for (int d = 0; d < dimension; d++)
            {
               grad_data[i + d * ndofs] = gradient[d];
            }
            for (int r = 0; r < dimension; r++)
            {
               for (int c = 0; c < dimension; c++)
               {
                  hess_data[i + (r * dimension + c) * ndofs] =
                     hessian[r * dimension + c];
               }
            }
         }
         return;
      }

      const real_t *position_data = current_node_pos.Read();
      real_t *sigma_data = sigma_samples.Write();
      real_t *grad_data = grad_samples.Write();
      real_t *hess_data = hess_samples.Write();
      const real_t *parameters = interface_parameters.Read();
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
      {
         real_t x[3] = {0.0, 0.0, 0.0};
         real_t gradient[3] = {0.0, 0.0, 0.0};
         real_t hessian[9] = {0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0};
         for (int d = 0; d < dimension; d++)
         {
            x[d] = position_data[i + d * ndofs];
         }
         real_t sigma = 0.0;
         EvalAnalyticLevelSet(dimension, level_set, x, parameters,
                              num_parameters, sigma, gradient, hessian);
         sigma_data[i] = sigma;
         for (int d = 0; d < dimension; d++)
         {
            grad_data[i + d * ndofs] = gradient[d];
         }
         for (int r = 0; r < dimension; r++)
         {
            for (int c = 0; c < dimension; c++)
            {
               hess_data[i + (r * dimension + c) * ndofs] =
                  hessian[r * dimension + c];
            }
         }
      });
   }

   ParMesh &mesh;
   int dim;
   int order;
   int basis;
   H1_FECollection current_fec;
   ParFiniteElementSpace current_fes;
   mutable ParGridFunction current_sigma;
   IntegrationRule nodal_ir;
   Array<bool> marker;
   Array<int> dof_count;
   Array<int> marked_dof_indices;
   real_t coefficient;
   SurfaceFittingOptions::LevelSetSource source;
   SurfaceFittingOptions::AnalyticLevelSet analytic_level_set;
   SurfaceFittingOptions::DiscreteDerivativeMode discrete_derivative_mode;
   bool discrete_from_background;
   int derivative_backend;
   Vector interface_parameters;
   mutable Vector current_node_pos;
   mutable Vector sigma_samples;
   mutable Vector grad_samples;
   mutable Vector hess_samples;

   std::unique_ptr<ParMesh> source_mesh;
   std::unique_ptr<H1_FECollection> source_fec;
   std::unique_ptr<ParFiniteElementSpace> source_fes;
   std::unique_ptr<ParFiniteElementSpace> source_grad_fes;
   std::unique_ptr<ParFiniteElementSpace> source_hess_fes;
   IntegrationRule source_nodal_ir;
   std::unique_ptr<ParGridFunction> source_sigma;
   std::unique_ptr<ParGridFunction> source_grad;
   std::unique_ptr<ParGridFunction> source_hess;
#ifdef MFEM_USE_GSLIB
   std::unique_ptr<FindPointsGSLIB> finder;
#endif
};

/** Enzyme-differentiable TMOP metric with optional limiting and surface fitting.
    Supports targets {1,4,5,6,8,9}, the corresponding 2D/3D metrics, exact or
    frozen target linearization, and host or tensor derivative backends. */
template <int dim>
class EnzymeTMOPFunctional
{
public:
   EnzymeTMOPFunctional(ParFiniteElementSpace &fes,
                        ParMesh &pmesh,
                        const IntegrationRule &ir,
                        int target_id_,
                        int metric_id_,
                        bool exact_action_,
                        bool freeze_target_linearization_,
                        const Vector &reference_nodes_,
                        real_t limit_coeff_,
                        const SurfaceFittingOptions *surface_fit_options_ =
                           nullptr,
                        int derivative_backend_ =
                           TENSOR_KERNEL_DERIVATIVES)
      : comm(fes.GetComm()),
        mesh(pmesh),
        fes(fes),
        qspace(pmesh, ir),
        surface_node_ir(MakeTensorNodalIntegrationRule(*fes.GetTypicalFE())),
        surface_node_qspace(pmesh, surface_node_ir),
        target_qspace_vec(qspace, dim * dim),
        target_w(target_qspace_vec),
        frozen_target_w(target_qspace_vec),
        target_data_qspace_vec(qspace, TargetDataVDim(target_id_)),
        target_qdata(target_data_qspace_vec),
        qspace_vec(qspace, 1),
        limit_qdata(qspace_vec),
        q(qspace_vec),
        surface_fit_qspace_vec(surface_node_qspace,
                               SurfaceFitDataLayout<dim>::SIZE),
        surface_fit_qdata(surface_fit_qspace_vec),
        surface_qspace_vec(surface_node_qspace, 1),
        surface_q(surface_qspace_vec),
        current_nodes(&fes),
        reference_nodes(reference_nodes_),
        target_id(target_id_),
        metric_id(metric_id_),
        exact_action(exact_action_),
        freeze_target_linearization(freeze_target_linearization_),
        has_node_limiting(limit_coeff_ != 0.0_r),
        derivative_backend(derivative_backend_)
   {
      MFEM_VERIFY(derivative_backend == CLASSIC_HOST_DERIVATIVES ||
                  derivative_backend == TENSOR_KERNEL_DERIVATIVES,
                  "Unknown field derivative backend.");
      SetTargetData();
      limit_qdata = limit_coeff_;
      if (target_id_ == 5)
      {
         target5_data = std::make_unique<Target5Remap>(
                           pmesh, derivative_backend);
      }
      else if (target_id_ == 6)
      {
         target6_data = std::make_unique<Target6Remap>(
                           pmesh, derivative_backend);
      }
      else if (target_id_ == 8)
      {
         target8_data = std::make_unique<Target8Remap>(
                           pmesh, derivative_backend);
      }

      Array<int> all_domain_attr;
      if (pmesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(pmesh.attributes.Max());
         all_domain_attr = 1;
      }

      SetupTMOPOperatorsDispatch(ir, all_domain_attr, target_id_, metric_id_);
      if (surface_fit_options_ && surface_fit_options_->enabled)
      {
         surface_fit_data = std::make_unique<SurfaceFittingData>(
                               pmesh, fes, *surface_fit_options_,
                               derivative_backend);
         SetupSurfaceFittingOperators(all_domain_attr);
      }
      SetupGradientOperators();
   }

   /// Refresh mutable target and fitting caches for a nonlinear state.
   void UpdateAfterMeshPositionChange(const Vector &x) const
   {
      current_nodes.SetFromTrueDofs(x);

      // Exact action also needs target derivatives.
      if (target_id == 5)
      {
         UpdateTarget5Data(exact_action);
      }
      else if (target_id == 6)
      {
         UpdateTarget6Data(exact_action);
      }
      else if (target_id == 8)
      {
         UpdateTarget8Data(exact_action);
      }

      if (UseFrozenTargetLinearization())
      {
         MFEM_VERIFY(frozen_target_updater,
                     "Frozen target updater is not initialized.");
         frozen_target_updater(*this);
      }

      if (surface_fit_data)
      {
         UpdateSurfaceFittingData();
      }
   }

   real_t MetricEnergy(const Vector &x) const
   {
      MultiVector Qmv{q};
      q = 0.0;

      if (target_id == 5 || target_id == 6 || target_id == 8)
      {
         MultiVector Xmv{x, target_w, target_qdata};
         energy_dop->Mult(Xmv, Qmv);
      }
      else
      {
         MultiVector Xmv{x, target_w};
         energy_dop->Mult(Xmv, Qmv);
      }

      const real_t local_energy = q.Sum();
      real_t global_energy = 0.0;
      MPI_Allreduce(&local_energy, &global_energy, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global_energy;
   }

   real_t LimitingEnergy(const Vector &x) const
   {
      if (!node_limiting_energy_dop) { return 0.0; }

      q = 0.0;
      MultiVector Qmv{q};
      if (target_id == 5)
      {
         MultiVector Xmv{x, reference_nodes, target_w, target_qdata,
                         limit_qdata};
         node_limiting_energy_dop->Mult(Xmv, Qmv);
      }
      else if (target_id == 6)
      {
         MultiVector Xmv{x, reference_nodes, target_w, target_qdata,
                         limit_qdata};
         node_limiting_energy_dop->Mult(Xmv, Qmv);
      }
      else if (target_id == 8)
      {
         MultiVector Xmv{x, reference_nodes, target_w, target_qdata,
                         limit_qdata};
         node_limiting_energy_dop->Mult(Xmv, Qmv);
      }
      else
      {
         MultiVector Xmv{x, reference_nodes, target_w, limit_qdata};
         node_limiting_energy_dop->Mult(Xmv, Qmv);
      }

      const real_t local_energy = q.Sum();
      real_t global_energy = 0.0;
      MPI_Allreduce(&local_energy, &global_energy, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global_energy;
   }

   real_t SurfaceFittingEnergy(const Vector &x) const
   {
      if (!surface_fit_data) { return 0.0; }

      surface_q = 0.0;
      MultiVector Xmv{x, surface_fit_qdata};
      MultiVector SQmv{surface_q};
      surface_energy_dop->Mult(Xmv, SQmv);

      const real_t local_energy = surface_q.Sum();
      real_t global_energy = 0.0;
      MPI_Allreduce(&local_energy, &global_energy, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global_energy;
   }

   real_t Energy(const Vector &x) const
   {
      return MetricEnergy(x) + LimitingEnergy(x) + SurfaceFittingEnergy(x);
   }

   void Gradient(const Vector &x, Vector &g) const
   {
      g = 0.0;
      MultiVector Gmv{g};
      if (!exact_action)
      {
         MFEM_VERIFY(frozen_target_energy_dop,
                     "Frozen target energy operator is not initialized.");
         MFEM_VERIFY(frozen_target_updater,
                     "Frozen target updater is not initialized.");
         MultiVector Xmv{x, frozen_target_w};
         metric_gradient_dop->Mult(Xmv, Gmv);
      }
      else if (target_id == 5)
      {
         MultiVector Xmv{x, target_w, target_qdata};
         metric_gradient_dop->Mult(Xmv, Gmv);
      }
      else if (target_id == 6)
      {
         MultiVector Xmv{x, target_w, target_qdata};
         metric_gradient_dop->Mult(Xmv, Gmv);
      }
      else if (target_id == 8)
      {
         MultiVector Xmv{x, target_w, target_qdata};
         metric_gradient_dop->Mult(Xmv, Gmv);
      }
      else
      {
         MultiVector Xmv{x, target_w};
         metric_gradient_dop->Mult(Xmv, Gmv);
      }

      DifferentiableOperator *limit_dop = !exact_action
                                          ? frozen_node_limiting_energy_dop.get()
                                          : node_limiting_energy_dop.get();
      if (limit_dop)
      {
         node_limiting_gradient.SetSize(g.Size());
         node_limiting_gradient = 0.0;
         MultiVector LGmv{node_limiting_gradient};
         if (!exact_action)
         {
            MultiVector Xmv{x, reference_nodes, frozen_target_w, limit_qdata};
            node_limiting_gradient_dop->Mult(Xmv, LGmv);
         }
         else if (target_id == 5 || target_id == 6 || target_id == 8)
         {
            MultiVector Xmv{x, reference_nodes, target_w, target_qdata,
                            limit_qdata};
            node_limiting_gradient_dop->Mult(Xmv, LGmv);
         }
         else
         {
            MultiVector Xmv{x, reference_nodes, target_w, limit_qdata};
            node_limiting_gradient_dop->Mult(Xmv, LGmv);
         }
         g += node_limiting_gradient;
      }

      if (surface_fit_data)
      {
         surface_gradient.SetSize(g.Size());
         surface_gradient = 0.0;
         MultiVector Xmv{x, surface_fit_qdata};
         MultiVector SGmv{surface_gradient};
         surface_gradient_dop->Mult(Xmv, SGmv);
         g += surface_gradient;
      }
   }

   std::unique_ptr<Operator> HessianOperator(const Vector &x) const;

   bool HasSurfaceFitting() const { return surface_fit_data != nullptr; }

   real_t GetSurfaceFittingCoefficient() const
   {
      return surface_fit_data ? surface_fit_data->GetCoefficient() : 0.0;
   }

   void ScaleSurfaceFittingCoefficient(real_t factor) const
   {
      MFEM_VERIFY(surface_fit_data,
                  "Surface fitting is not enabled for this functional.");
      surface_fit_data->ScaleCoefficient(factor);
      UpdateSurfaceFittingData();
   }

   void GetSurfaceFittingErrors(real_t &err_avg,
                                real_t &err_max) const
   {
      if (!surface_fit_data)
      {
         err_avg = 0.0;
         err_max = 0.0;
         return;
      }
      surface_fit_data->GetErrors(err_avg, err_max);
   }

private:
   void SetupTMOPOperatorsDispatch(const IntegrationRule &ir,
                                   const Array<int> &all_domain_attr,
                                   int target_id_val,
                                   int metric_id_val)
   {
#ifdef MFEM_ENZYME_TMOP_DIGITAL_TWIN_ONLY
      MFEM_VERIFY(target_id_val == 1 && metric_id_val == 2,
                  "The digital-twin driver only instantiates target id 1 "
                  "with metric id 2.");
      return SetupTMOPOperators<1, 2>(ir, all_domain_attr);
#else
      if constexpr (dim == 2)
      {
         switch (target_id_val)
         {
            case 1:
               switch (metric_id_val)
               {
                  case 2:  return SetupTMOPOperators<1, 2>(ir, all_domain_attr);
                  case 58: return SetupTMOPOperators<1, 58>(ir, all_domain_attr);
                  case 80: return SetupTMOPOperators<1, 80>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Target id 1 supports metric ids 2, 58, and 80 "
                                "in 2D.");
               }
            case 4:
               switch (metric_id_val)
               {
                  case 2:  return SetupTMOPOperators<4, 2>(ir, all_domain_attr);
                  case 14: return SetupTMOPOperators<4, 14>(ir, all_domain_attr);
                  case 80: return SetupTMOPOperators<4, 80>(ir, all_domain_attr);
                  case 85: return SetupTMOPOperators<4, 85>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Target id 4 supports metric ids 2, 14, 80, and 85.");
               }
            case 6:
               switch (metric_id_val)
               {
                  case 80: return SetupTMOPOperators<6, 80>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Target id 6 supports only metric id 80.");
               }
            case 8:
               switch (metric_id_val)
               {
                  case 36: return SetupTMOPOperators<8, 36>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Target id 8 supports only metric id 36.");
               }
            default:
               MFEM_ABORT("Unsupported target id: " << target_id_val);
         }
      }
      else if constexpr (dim == 3)
      {
         switch (target_id_val)
         {
            case 1:
               switch (metric_id_val)
               {
                  case 301:
                     return SetupTMOPOperators<1, 301>(ir, all_domain_attr);
                  case 302:
                     return SetupTMOPOperators<1, 302>(ir, all_domain_attr);
                  case 303:
                     return SetupTMOPOperators<1, 303>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Target id 1 supports metric ids 301, 302, and "
                                "303 in 3D.");
               }
            case 5:
               switch (metric_id_val)
               {
                  case 321:
                     return SetupTMOPOperators<5, 321>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Target id 5 supports only metric id 321 in 3D.");
               }
            case 9:
               switch (metric_id_val)
               {
                  case 321:
                     return SetupTMOPOperators<9, 321>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Target id 9 supports only metric id 321.");
               }
            default:
               MFEM_ABORT("Unsupported 3D target id: " << target_id_val);
         }
      }
#endif
   }

   template <int target_id_val, int metric_id_val>
   void SetupTMOPOperators(const IntegrationRule &ir,
                           const Array<int> &all_domain_attr)
   {
      if constexpr (target_id_val == 5 || target_id_val == 6 ||
                    target_id_val == 8)
      {
         const std::vector in
         {
            FieldDescriptor{X, &fes},
            FieldDescriptor{TARGET_W, &target_qspace_vec},
            FieldDescriptor{TARGET_DATA, &target_data_qspace_vec}
         };
         const std::vector out { FieldDescriptor{Q, &qspace_vec} };

         energy_dop =
            std::make_unique<DifferentiableOperator>(in, out, mesh);
         DiscreteTargetTMOPEnergy<real_t, dim, target_id_val,
                                  metric_id_val> energy;
         auto derivatives = std::integer_sequence<size_t, X> {};
         energy_dop->AddDomainIntegrator<LocalQFBackend>(
            energy,
            future::tuple{Value<X>{}, future::Gradient<X>{},
                          Identity<TARGET_W>{}, Identity<TARGET_DATA>{},
                          Weight{}},
            future::tuple{FunctionalValue<Q>{}},
            ir, all_domain_attr, derivatives);
      }
      else if constexpr (target_id_val == 1)
      {
         const std::vector in
         {
            FieldDescriptor{X, &fes},
            FieldDescriptor{TARGET_W, &target_qspace_vec}
         };
         const std::vector out { FieldDescriptor{Q, &qspace_vec} };

         energy_dop =
            std::make_unique<DifferentiableOperator>(in, out, mesh);
         auto derivatives = std::integer_sequence<size_t, X> {};
         PrecomputedTargetTMOPEnergy<real_t, dim, metric_id_val> energy;
         energy_dop->AddDomainIntegrator<LocalQFBackend>(
            energy,
            future::tuple{future::Gradient<X>{}, Identity<TARGET_W>{},
                          Weight{}},
            future::tuple{FunctionalValue<Q>{}},
            ir, all_domain_attr, derivatives);
      }
      else
      {
         const std::vector in
         {
            FieldDescriptor{X, &fes},
            FieldDescriptor{TARGET_W, &target_qspace_vec}
         };
         const std::vector out { FieldDescriptor{Q, &qspace_vec} };

         energy_dop =
            std::make_unique<DifferentiableOperator>(in, out, mesh);
         auto derivatives = std::integer_sequence<size_t, X> {};
         AnalyticTargetTMOPEnergy<real_t, dim, target_id_val,
                                  metric_id_val> energy;
         energy_dop->AddDomainIntegrator<LocalQFBackend>(
            energy,
            future::tuple{Value<X>{}, future::Gradient<X>{},
                          Identity<TARGET_W>{}, Weight{}},
            future::tuple{FunctionalValue<Q>{}},
            ir, all_domain_attr, derivatives);
      }

      if (has_node_limiting)
      {
         if constexpr (target_id_val == 5 || target_id_val == 6 ||
                       target_id_val == 8)
         {
            SetupDiscreteTargetNodeLimitingFunctional<target_id_val,
                                                       metric_id_val>(
               ir, all_domain_attr);
         }
         else if constexpr (target_id_val == 1)
         {
            SetupNodeLimitingFunctional(
               ir, all_domain_attr,
               PrecomputedTargetNodeLimitingEnergy<real_t, dim> {});
         }
         else
         {
            SetupNodeLimitingFunctional(
               ir, all_domain_attr,
               AnalyticTargetNodeLimitingEnergy<real_t, dim, target_id_val,
                                                 metric_id_val> {});
         }
      }

      if (UseFrozenTargetLinearization())
      {
         SetupFrozenTargetEnergy<metric_id_val>(ir, all_domain_attr);
         frozen_target_updater =
            &CallUpdateFrozenTargetData<target_id_val, metric_id_val>;
      }
   }

   template <int target_id_val, int metric_id_val>
   void SetupDiscreteTargetNodeLimitingFunctional(
      const IntegrationRule &ir,
      const Array<int> &all_domain_attr)
   {
      const std::vector in
      {
         FieldDescriptor{X, &fes},
         FieldDescriptor{REFERENCE_X, &fes},
         FieldDescriptor{TARGET_W, &target_qspace_vec},
         FieldDescriptor{TARGET_DATA, &target_data_qspace_vec},
         FieldDescriptor{LIMIT_COEFF, &qspace_vec}
      };
      const std::vector out { FieldDescriptor{Q, &qspace_vec} };

      node_limiting_energy_dop =
         std::make_unique<DifferentiableOperator>(in, out, mesh);
      auto derivatives = std::integer_sequence<size_t, X> {};
      DiscreteTargetNodeLimitingEnergy<real_t, dim, target_id_val,
                                       metric_id_val> energy;
      node_limiting_energy_dop->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{Value<X>{}, Value<REFERENCE_X>{},
                       Identity<TARGET_W>{}, Identity<TARGET_DATA>{},
                       Identity<LIMIT_COEFF>{}, Weight{}},
         future::tuple{FunctionalValue<Q>{}},
         ir, all_domain_attr, derivatives);
   }

   template <typename Energy>
   void SetupNodeLimitingFunctional(const IntegrationRule &ir,
                                    const Array<int> &all_domain_attr,
                                    Energy energy)
   {
      const std::vector in
      {
         FieldDescriptor{X, &fes},
         FieldDescriptor{REFERENCE_X, &fes},
         FieldDescriptor{TARGET_W, &target_qspace_vec},
         FieldDescriptor{LIMIT_COEFF, &qspace_vec}
      };
      const std::vector out { FieldDescriptor{Q, &qspace_vec} };

      node_limiting_energy_dop =
         std::make_unique<DifferentiableOperator>(in, out, mesh);
      auto derivatives = std::integer_sequence<size_t, X> {};
      node_limiting_energy_dop->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{Value<X>{}, Value<REFERENCE_X>{},
                       Identity<TARGET_W>{}, Identity<LIMIT_COEFF>{}, Weight{}},
         future::tuple{FunctionalValue<Q>{}},
         ir, all_domain_attr, derivatives);
   }

   template <int metric_id_>
   void SetupFrozenTargetEnergy(const IntegrationRule &ir,
                                const Array<int> &all_domain_attr)
   {
      const std::vector in
      {
         FieldDescriptor{X, &fes},
         FieldDescriptor{TARGET_W, &target_qspace_vec}
      };
      const std::vector out { FieldDescriptor{Q, &qspace_vec} };

      frozen_target_energy_dop =
         std::make_unique<DifferentiableOperator>(in, out, mesh);
      auto derivatives = std::integer_sequence<size_t, X> {};
      PrecomputedTargetTMOPEnergy<real_t, dim, metric_id_> energy;
      frozen_target_energy_dop->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{future::Gradient<X>{}, Identity<TARGET_W>{}, Weight{}},
         future::tuple{FunctionalValue<Q>{}},
         ir, all_domain_attr, derivatives);

      if (has_node_limiting)
      {
         const std::vector limit_in
         {
            FieldDescriptor{X, &fes},
            FieldDescriptor{REFERENCE_X, &fes},
            FieldDescriptor{TARGET_W, &target_qspace_vec},
            FieldDescriptor{LIMIT_COEFF, &qspace_vec}
         };
         frozen_node_limiting_energy_dop =
            std::make_unique<DifferentiableOperator>(limit_in, out, mesh);
         PrecomputedTargetNodeLimitingEnergy<real_t, dim> limit_energy;
         frozen_node_limiting_energy_dop->AddDomainIntegrator<LocalQFBackend>(
            limit_energy,
            future::tuple{Value<X>{}, Value<REFERENCE_X>{},
                          Identity<TARGET_W>{}, Identity<LIMIT_COEFF>{},
                          Weight{}},
            future::tuple{FunctionalValue<Q>{}},
            ir, all_domain_attr, derivatives);
      }
   }

   void SetupSurfaceFittingOperators(const Array<int> &all_domain_attr)
   {
      const IntegrationRule &node_ir = surface_node_qspace.GetIntRule(0);
      {
         const std::vector in
         {
            FieldDescriptor{X, &fes},
            FieldDescriptor{SURFACE_FIT_DATA, &surface_fit_qspace_vec}
         };
         const std::vector out { FieldDescriptor{Q, &surface_qspace_vec} };

         surface_energy_dop =
            std::make_unique<DifferentiableOperator>(in, out, mesh);
         SurfaceFittingLevelSetEnergy<real_t, dim> energy;
         auto derivatives = std::integer_sequence<size_t, X> {};
         surface_energy_dop->AddDomainIntegrator<LocalQFBackend>(
            energy,
            future::tuple{Value<X>{}, Identity<SURFACE_FIT_DATA>{}},
            future::tuple{FunctionalValue<Q>{}},
            node_ir, all_domain_attr, derivatives);
      }
   }

   void SetupGradientOperators()
   {
      DifferentiableOperator *metric_dop =
         exact_action ? energy_dop.get() : frozen_target_energy_dop.get();
      MFEM_VERIFY(metric_dop,
                  "TMOP gradient operator is not initialized.");
      metric_gradient_dop = metric_dop->GetDerivative(X);

      DifferentiableOperator *limit_dop =
         exact_action ? node_limiting_energy_dop.get() :
         frozen_node_limiting_energy_dop.get();
      if (limit_dop)
      {
         node_limiting_gradient_dop = limit_dop->GetDerivative(X);
      }
      if (surface_energy_dop)
      {
         surface_gradient_dop = surface_energy_dop->GetDerivative(X);
      }
   }

   void SetTargetData()
   {
      const int vdim = dim * dim;
      const DenseMatrix &W =
         Geometries.GetGeomToPerfGeomJac(dim == 2 ? Geometry::SQUARE :
                                                    Geometry::CUBE);
      MFEM_VERIFY(W.Height() == dim && W.Width() == dim,
                  "Unexpected target matrix dimension.");
      real_t constant_W[9] {};
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            constant_W[i * dim + j] = W(i, j);
         }
      }
      const real_t W00 = constant_W[0];
      const real_t W01 = constant_W[1];
      const real_t W02 = constant_W[2];
      const real_t W10 = constant_W[dim];
      const real_t W11 = constant_W[dim + 1];
      const real_t W12 = (dim == 3) ? constant_W[dim + 2] : 0.0;
      const real_t W20 = (dim == 3) ? constant_W[2 * dim] : 0.0;
      const real_t W21 = (dim == 3) ? constant_W[2 * dim + 1] : 0.0;
      const real_t W22 = (dim == 3) ? constant_W[2 * dim + 2] : 0.0;
      real_t *data = target_w.Write();
      const int npoints = target_w.Size() / vdim;
      mfem::forall(npoints, [=] MFEM_HOST_DEVICE (int k)
      {
         real_t *Wq = data + k * vdim;
         Wq[0] = W00;
         Wq[1] = W01;
         Wq[dim] = W10;
         Wq[dim + 1] = W11;
         if (dim == 3)
         {
            Wq[2] = W02;
            Wq[5] = W12;
            Wq[6] = W20;
            Wq[7] = W21;
            Wq[8] = W22;
         }
      });
   }

   void UpdateTarget6Data(bool include_derivatives) const
   {
      MFEM_VERIFY(target6_data, "Target id 6 data has not been initialized.");
      target6_data->FillQuadratureData(current_nodes, qspace, target_qdata,
                                       include_derivatives);
   }

   void UpdateTarget5Data(bool include_derivatives) const
   {
      MFEM_VERIFY(target5_data, "Target id 5 data has not been initialized.");
      target5_data->FillQuadratureData(current_nodes, qspace, target_qdata,
                                       include_derivatives);
   }

   void UpdateTarget8Data(bool include_derivatives) const
   {
      MFEM_VERIFY(target8_data, "Target id 8 data has not been initialized.");
      target8_data->FillQuadratureData(current_nodes, qspace, target_qdata,
                                       include_derivatives);
   }

   void UpdateSurfaceFittingData() const
   {
      MFEM_VERIFY(surface_fit_data,
                  "Surface fitting data has not been initialized.");
      surface_fit_data->FillQuadratureData(current_nodes, surface_node_qspace,
                                           surface_fit_qdata);
   }

   template <int target_id_val, int metric_id_val>
   void UpdateFrozenTargetData() const
   {
      const IntegrationRule &ir = qspace.GetIntRule(0);
      const int ne = qspace.GetNE();
      const int nq = ir.GetNPoints();
      Vector positions;
      EvaluateFieldValuesOnDevice(current_nodes, ir, positions);

      const real_t *position_data = positions.Read();
      const real_t *const_target = target_w.Read();
      const real_t *target_data = nullptr;
      if constexpr (target_id_val == 5 || target_id_val == 6 ||
                    target_id_val == 8)
      {
         target_data = target_qdata.Read();
      }
      real_t *frozen_target = frozen_target_w.Write();
      const int vdim = dim * dim;
      mfem::forall(ne * nq, [=] MFEM_HOST_DEVICE (int k)
      {
         const int q = k % nq;
         const int e = k / nq;
         tensor<real_t, dim> xq {};
         tensor<real_t, dim, dim> constant_W {};
         for (int d = 0; d < dim; d++)
         {
            xq(d) = position_data[q + nq * (d + dim * e)];
         }
         const real_t *base = const_target + k * vdim;
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               constant_W(i,j) = base[i * dim + j];
            }
         }

         real_t shape_weight = 0.5;
         tensor<real_t, dim, dim> W {};
         if constexpr (target_id_val == 5 || target_id_val == 6 ||
                       target_id_val == 8)
         {
            constexpr int target_data_size = TargetDataSize<target_id_val>();
            tensor<real_t, target_data_size> qdata {};
            const real_t *data = target_data + k * target_data_size;
            for (int i = 0; i < target_data_size; i++)
            {
               qdata(i) = data[i];
            }
            W = TargetMatrix<real_t, dim, target_id_val, metric_id_val>(
                   xq, constant_W, shape_weight, &qdata);
         }
         else
         {
            W = TargetMatrix<real_t, dim, target_id_val, metric_id_val>(
                   xq, constant_W, shape_weight);
         }

         real_t *target = frozen_target + k * vdim;
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               target[i * dim + j] = W(i,j);
            }
         }
      });
   }

   bool UseFrozenTargetLinearization() const
   {
      return !exact_action || freeze_target_linearization;
   }

   using FrozenTargetUpdater =
      void (*)(const EnzymeTMOPFunctional<dim> &);

   template <int target_id_val, int metric_id_val>
   static void CallUpdateFrozenTargetData(
      const EnzymeTMOPFunctional<dim> &self)
   {
      self.template UpdateFrozenTargetData<target_id_val, metric_id_val>();
   }

   MPI_Comm comm;
   ParMesh &mesh;
   ParFiniteElementSpace &fes;
   QuadratureSpace qspace;
   IntegrationRule surface_node_ir;
   QuadratureSpace surface_node_qspace;
   VectorQuadratureSpace target_qspace_vec;
   QuadratureFunction target_w;
   mutable QuadratureFunction frozen_target_w;
   VectorQuadratureSpace target_data_qspace_vec;
   mutable QuadratureFunction target_qdata;
   VectorQuadratureSpace qspace_vec;
   QuadratureFunction limit_qdata;
   mutable QuadratureFunction q;
   VectorQuadratureSpace surface_fit_qspace_vec;
   mutable QuadratureFunction surface_fit_qdata;
   VectorQuadratureSpace surface_qspace_vec;
   mutable QuadratureFunction surface_q;
   mutable ParGridFunction current_nodes;
   Vector reference_nodes;
   int target_id;
   int metric_id;
   bool exact_action;
   bool freeze_target_linearization;
   bool has_node_limiting;
   int derivative_backend;
   std::unique_ptr<Target5Remap> target5_data;
   std::unique_ptr<Target6Remap> target6_data;
   std::unique_ptr<Target8Remap> target8_data;
   std::unique_ptr<SurfaceFittingData> surface_fit_data;
   std::unique_ptr<DifferentiableOperator> energy_dop;
   std::unique_ptr<DifferentiableOperator> node_limiting_energy_dop;
   std::unique_ptr<DifferentiableOperator> frozen_target_energy_dop;
   std::unique_ptr<DifferentiableOperator> frozen_node_limiting_energy_dop;
   std::unique_ptr<DifferentiableOperator> surface_energy_dop;
   std::shared_ptr<DerivativeOperator> metric_gradient_dop;
   std::shared_ptr<DerivativeOperator> node_limiting_gradient_dop;
   std::shared_ptr<DerivativeOperator> surface_gradient_dop;
   mutable Vector node_limiting_gradient;
   mutable Vector surface_gradient;
   FrozenTargetUpdater frozen_target_updater = nullptr;
};

class SingleOutputDerivativeOperator : public Operator
{
public:
   SingleOutputDerivativeOperator(std::shared_ptr<DerivativeOperator> op,
                                  const ParFiniteElementSpace &fes)
      : Operator(fes.GetTrueVSize()),
        derivative(std::move(op))
   { }

   MemoryClass GetMemoryClass() const override
   {
      return Device::GetDeviceMemoryClass();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MultiVector Ymv{y};
      derivative->Mult(x, Ymv);
   }

   void AssembleDiagonal(Vector &diag) const override
   {
      derivative->AssembleDiagonal(diag);
   }

private:
   std::shared_ptr<DerivativeOperator> derivative;
};

class SumWithDiagonalOperator : public Operator
{
public:
   SumWithDiagonalOperator(std::unique_ptr<Operator> a_,
                           std::unique_ptr<Operator> b_)
      : Operator(a_->Height(), a_->Width()),
        a(std::move(a_)),
        b(std::move(b_))
   {
      MFEM_VERIFY(a->Height() == b->Height() && a->Width() == b->Width(),
                  "Cannot sum incompatible operators.");
      work.UseDevice(true);
   }

   MemoryClass GetMemoryClass() const override
   {
      return Device::GetDeviceMemoryClass();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      work.SetSize(Height());
      a->Mult(x, work);
      b->Mult(x, y);
      y += work;
   }

   void AssembleDiagonal(Vector &diag) const override
   {
      a->AssembleDiagonal(diag);
      work.SetSize(diag.Size());
      b->AssembleDiagonal(work);
      diag += work;
   }

private:
   std::unique_ptr<Operator> a;
   std::unique_ptr<Operator> b;
   mutable Vector work;
};

template <int dim>
std::unique_ptr<Operator>
EnzymeTMOPFunctional<dim>::HessianOperator(const Vector &x) const
{
   // Reuse quadrature Hessian data throughout the Krylov solve.
   std::unique_ptr<Operator> tmop_hessian;
   if (UseFrozenTargetLinearization())
   {
      MFEM_VERIFY(frozen_target_energy_dop,
                  "Frozen target energy operator is not initialized.");
      MFEM_VERIFY(frozen_target_updater,
                  "Frozen target updater is not initialized.");
      MultiVector Xmv{x, frozen_target_w};
      tmop_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                        frozen_target_energy_dop->GetSecondDerivative(
                           X, Xmv, true),
                        fes);
   }
   else if (target_id == 5 || target_id == 6 || target_id == 8)
   {
      MultiVector Xmv{x, target_w, target_qdata};
      tmop_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                        energy_dop->GetSecondDerivative(X, Xmv, true), fes);
   }
   else
   {
      MultiVector Xmv{x, target_w};
      tmop_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                        energy_dop->GetSecondDerivative(X, Xmv, true), fes);
   }

   DifferentiableOperator *limit_dop = UseFrozenTargetLinearization()
                                       ? frozen_node_limiting_energy_dop.get()
                                       : node_limiting_energy_dop.get();
   if (limit_dop)
   {
      std::unique_ptr<Operator> limit_hessian;
      if (UseFrozenTargetLinearization())
      {
         MultiVector Xmv{x, reference_nodes, frozen_target_w, limit_qdata};
         limit_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                            limit_dop->GetSecondDerivative(X, Xmv, true), fes);
      }
      else if (target_id == 6 || target_id == 8)
      {
         MultiVector Xmv{x, reference_nodes, target_w, target_qdata,
                         limit_qdata};
         limit_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                            limit_dop->GetSecondDerivative(X, Xmv, true), fes);
      }
      else
      {
         MultiVector Xmv{x, reference_nodes, target_w, limit_qdata};
         limit_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                            limit_dop->GetSecondDerivative(X, Xmv, true), fes);
      }
      tmop_hessian = std::make_unique<SumWithDiagonalOperator>(
                        std::move(tmop_hessian), std::move(limit_hessian));
   }

   if (!surface_fit_data) { return tmop_hessian; }

   MultiVector Xmv{x, surface_fit_qdata};
   auto surface_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                             surface_energy_dop->GetSecondDerivative(
                                X, Xmv, true),
                             fes);
   return std::make_unique<SumWithDiagonalOperator>(
             std::move(tmop_hessian), std::move(surface_hessian));
}

template <int dim>
class EnzymeTMOPNonlinearForm : public ParNonlinearForm
{
public:
   EnzymeTMOPNonlinearForm(ParFiniteElementSpace &fes,
                           const EnzymeTMOPFunctional<dim> &functional)
      : ParNonlinearForm(&fes),
        functional(functional),
        x_abs(fes.GetTrueVSize())
   {
      reference_true.SetSize(fes.GetTrueVSize());
      reference_true.UseDevice(true);
      x_abs.UseDevice(true);
   }

   MemoryClass GetMemoryClass() const override
   {
      return Device::GetDeviceMemoryClass();
   }

   void SetReference(const Vector &x0)
   {
      reference_true = x0;
   }

   const Vector &ComputeAbsoluteState(const Vector &dx) const
   {
      add(reference_true, dx, x_abs);
      return x_abs;
   }

   void UpdateAfterMeshPositionChange(const Vector &x) const
   {
      functional.UpdateAfterMeshPositionChange(x);
   }

   void GetSurfaceFittingErrors(real_t &err_avg, real_t &err_max) const
   {
      functional.GetSurfaceFittingErrors(err_avg, err_max);
   }

   real_t GetSurfaceFittingCoefficient() const
   {
      return functional.GetSurfaceFittingCoefficient();
   }

   void ScaleSurfaceFittingCoefficient(real_t factor) const
   {
      functional.ScaleSurfaceFittingCoefficient(factor);
   }

   real_t GetEnergy(const Vector &dx) const override
   {
      add(reference_true, dx, x_abs);
      return functional.Energy(x_abs);
   }

   void Mult(const Vector &dx, Vector &y) const override
   {
      add(reference_true, dx, x_abs);
      functional.Gradient(x_abs, y);
      const Array<int> &ess_tdofs = GetEssentialTrueDofs();
      if (ess_tdofs.Size() > 0) { y.SetSubVector(ess_tdofs, 0.0); }
   }

   Operator &GetGradient(const Vector &dx) const override
   {
      add(reference_true, dx, x_abs);
      hessian = functional.HessianOperator(x_abs);
      const Array<int> &ess_tdofs = GetEssentialTrueDofs();
      constrained_hessian =
         std::make_unique<ConstrainedOperator>(hessian.get(), ess_tdofs, false);
      return *constrained_hessian;
   }

private:
   const EnzymeTMOPFunctional<dim> &functional;
   Vector reference_true;
   mutable Vector x_abs;
   mutable std::unique_ptr<Operator> hessian;
   mutable std::unique_ptr<ConstrainedOperator> constrained_hessian;
};

template <int dim>
class EnzymeTMOPNewtonSolver : public TMOPNewtonSolver
{
public:
   EnzymeTMOPNewtonSolver(MPI_Comm comm,
                          const IntegrationRule &ir,
                          EnzymeTMOPNonlinearForm<dim> &nlf)
      : TMOPNewtonSolver(comm, ir, 0), enzyme_nlf(nlf) { }

   void ConfigureAdaptiveSurfaceFitting(real_t scale_factor,
                                        real_t max_error,
                                        real_t weight_limit,
                                        bool converge_by_error)
   {
      MFEM_VERIFY(scale_factor > 1.0,
                  "Adaptive surface fitting scale must be greater than one.");
      MFEM_VERIFY(weight_limit > 0.0,
                  "Surface fitting weight limit must be positive.");
      MFEM_VERIFY(!converge_by_error || max_error >= 0.0,
                  "Error-based surface fitting requires a nonnegative "
                  "error threshold.");
      surf_fit_scale_factor = scale_factor;
      surf_fit_max_err_limit = max_error;
      surf_fit_weight_limit = weight_limit;
      surf_fit_converge_error = converge_by_error;
   }

   void ResetAdaptiveSurfaceFittingState() const
   {
      previous_surf_fit_avg_error = 10000.0;
      update_surface_fit_coefficient = false;
      surf_fit_adapt_count = 0;
   }

   real_t ComputeScalingFactor(const Vector &dx,
                               const Vector &b) const override
   {
      if (surf_fit_scale_factor > 0.0)
      {
         real_t avg_error = 0.0, max_error = 0.0;
         enzyme_nlf.GetSurfaceFittingErrors(avg_error, max_error);
         if ((surf_fit_converge_error &&
              max_error <= surf_fit_max_err_limit) ||
             surf_fit_adapt_count >= surf_fit_adapt_count_limit)
         {
            return 0.0;
         }
      }

      const real_t scale = TMOPNewtonSolver::ComputeScalingFactor(dx, b);
      if (scale > 0.0 && surf_fit_scale_factor > 0.0)
      {
         update_surface_fit_coefficient = true;
      }
      return scale;
   }

   void ProcessNewState(const Vector &dx) const override
   {
      const Vector &x = enzyme_nlf.ComputeAbsoluteState(dx);
      enzyme_nlf.UpdateAfterMeshPositionChange(x);

      if (!update_surface_fit_coefficient) { return; }

      real_t avg_error = 0.0, max_error = 0.0;
      enzyme_nlf.GetSurfaceFittingErrors(avg_error, max_error);
      const real_t coefficient = enzyme_nlf.GetSurfaceFittingCoefficient();
      const real_t relative_change =
         (previous_surf_fit_avg_error - avg_error) /
         previous_surf_fit_avg_error;
      const bool below_weight_limit = coefficient < surf_fit_weight_limit;
      const bool should_increase =
         relative_change < surf_fit_err_rel_change_limit &&
         below_weight_limit &&
         (surf_fit_converge_error || max_error > surf_fit_max_err_limit);

      if (should_increase)
      {
         const real_t factor =
            std::min(surf_fit_scale_factor,
                     surf_fit_weight_limit / coefficient);
         enzyme_nlf.ScaleSurfaceFittingCoefficient(factor);
         surf_fit_adapt_count++;
         if (print_options.iterations && Mpi::Root())
         {
            std::cout << "Adaptive surface fitting: fit_avg=" << avg_error
                      << ", fit_max=" << max_error
                      << ", coefficient="
                      << enzyme_nlf.GetSurfaceFittingCoefficient() << '\n';
         }
      }
      else
      {
         surf_fit_adapt_count = 0;
      }
      previous_surf_fit_avg_error = avg_error;
      update_surface_fit_coefficient = false;
   }

private:
   EnzymeTMOPNonlinearForm<dim> &enzyme_nlf;
   real_t surf_fit_scale_factor = 0.0;
   real_t surf_fit_max_err_limit = -1.0;
   real_t surf_fit_err_rel_change_limit = 0.001;
   real_t surf_fit_weight_limit = 1.0e20;
   bool surf_fit_converge_error = false;
   mutable real_t previous_surf_fit_avg_error = 10000.0;
   mutable bool update_surface_fit_coefficient = false;
   mutable int surf_fit_adapt_count = 0;
   int surf_fit_adapt_count_limit = 10;
};

real_t MinimumDetJ(ParMesh &pmesh,
                   const ParFiniteElementSpace &pfespace,
                   IntegrationRules &irules,
                   int quad_order)
{
   real_t min_detJ = infinity();
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      const IntegrationRule &ir =
         irules.Get(pfespace.GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *trans = pmesh.GetElementTransformation(i);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         trans->SetIntPoint(&ir.IntPoint(q));
         min_detJ = std::min(min_detJ, trans->Jacobian().Det());
      }
   }

   real_t global_min_detJ = 0.0;
   MPI_Allreduce(&min_detJ, &global_min_detJ, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh.GetComm());
   return global_min_detJ;
}

void SaveMesh(ParMesh &pmesh, const char *filename)
{
   std::ofstream mesh_ofs(filename);
   mesh_ofs.precision(8);
   pmesh.PrintAsOne(mesh_ofs);
}

struct TMOPVisualizationData
{
   std::unique_ptr<TMOP_QualityMetric> metric;
   std::unique_ptr<TargetConstructor> target;
   std::unique_ptr<TMOPMatrixCoefficient> analytic_target_coeff;
   std::unique_ptr<Target5Remap> target5_data;
   std::unique_ptr<Target6Remap> target6_data;
   std::unique_ptr<Target8Remap> target8_data;
};

TMOPVisualizationData MakeVisualizationData(int dim,
                                            int metric_id,
                                            int target_id,
                                            ParGridFunction &nodes)
{
   TMOPVisualizationData data;

   if (dim == 2 && metric_id == 36)
   {
      data.metric = std::make_unique<TMOP_AMetric_036>();
   }
   else if (dim == 2 && metric_id == 14)
   {
      data.metric = std::make_unique<TMOP_Metric_014>();
   }
   else if (dim == 2 && metric_id == 58)
   {
      data.metric = std::make_unique<TMOP_Metric_058>();
   }
   else if (dim == 2 && metric_id == 80)
   {
      data.metric = std::make_unique<TMOP_Metric_080>(0.5);
   }
   else if (dim == 2 && metric_id == 85)
   {
      data.metric = std::make_unique<TMOP_Metric_085>();
   }
   else if (dim == 2)
   {
      data.metric = std::make_unique<TMOP_Metric_002>();
   }
   else if (dim == 3 && metric_id == 301)
   {
      data.metric = std::make_unique<TMOP_Metric_301>();
   }
   else if (dim == 3 && metric_id == 302)
   {
      data.metric = std::make_unique<TMOP_Metric_302>();
   }
   else if (dim == 3 && metric_id == 303)
   {
      data.metric = std::make_unique<TMOP_Metric_303>();
   }
   else if (dim == 3 && metric_id == 321)
   {
      data.metric = std::make_unique<TMOP_Metric_321>();
   }
   else
   {
      MFEM_ABORT("Unsupported visualization metric id: " << metric_id);
   }

   if (target_id == 1)
   {
      data.target = std::make_unique<TargetConstructor>(
                       TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                       nodes.ParFESpace()->GetComm());
   }
   else if (target_id == 4)
   {
      auto target = std::make_unique<AnalyticAdaptTC>(
                       TargetConstructor::GIVEN_FULL);
      data.analytic_target_coeff =
         std::make_unique<HessianCoefficient>(dim, metric_id);
      target->SetAnalyticTargetSpec(NULL, NULL,
                                    data.analytic_target_coeff.get());
      data.target = std::move(target);
   }
   else if (target_id == 9)
   {
      auto target = std::make_unique<AnalyticAdaptTC>(
                       TargetConstructor::GIVEN_FULL);
      data.analytic_target_coeff =
         std::make_unique<HRHessianCoefficient>(dim, 0);
      target->SetAnalyticTargetSpec(nullptr, nullptr,
                                    data.analytic_target_coeff.get());
      data.target = std::move(target);
   }
   else if (target_id == 5)
   {
      data.target5_data = std::make_unique<Target5Remap>(
                             *nodes.ParFESpace()->GetParMesh());
      auto target = std::make_unique<DiscreteAdaptTC>(
                       TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE);
#ifdef MFEM_USE_GSLIB
      target->SetAdaptivityEvaluator(new InterpolatorFP);
#else
      MFEM_ABORT("Target id 5 visualization requires GSLIB.");
#endif
      target->SetParDiscreteTargetSize(data.target5_data->Size());
      target->SetMinSizeForTargets(data.target5_data->MinSize());
      data.target = std::move(target);
   }
   else if (target_id == 6)
   {
#ifdef MFEM_USE_GSLIB
      data.target6_data = std::make_unique<Target6Remap>(
                             *nodes.ParFESpace()->GetParMesh());
      auto target = std::make_unique<DiscreteAdaptTC>(
                       TargetConstructor::GIVEN_SHAPE_AND_SIZE);
      target->SetAdaptivityEvaluator(new InterpolatorFP);
      target->SetParDiscreteTargetSize(data.target6_data->Size());
      target->SetMinSizeForTargets(data.target6_data->MinSize());
      target->SetParDiscreteTargetAspectRatio(data.target6_data->Aspect());
      data.target = std::move(target);
#else
      MFEM_ABORT("Target id 6 visualization requires GSLIB.");
#endif
   }
   else if (target_id == 8)
   {
#ifdef MFEM_USE_GSLIB
      data.target8_data = std::make_unique<Target8Remap>(
                             *nodes.ParFESpace()->GetParMesh());
      auto target = std::make_unique<DiscreteAdaptTC>(
                       TargetConstructor::GIVEN_SHAPE_AND_SIZE);
      target->SetAdaptivityEvaluator(new InterpolatorFP);
      target->SetParDiscreteTargetSize(data.target8_data->Size());
      target->SetMinSizeForTargets(data.target8_data->MinSize());
      target->SetParDiscreteTargetOrientation(data.target8_data->Ori());
      data.target = std::move(target);
#else
      MFEM_ABORT("Target id 8 visualization requires GSLIB.");
#endif
   }
   else
   {
      MFEM_ABORT("Unsupported visualization target id: " << target_id);
   }

   data.target->SetNodes(nodes);
   return data;
}

void VisualizeMetricValues(int mesh_poly_deg,
                           TMOPVisualizationData &vis_data,
                           ParMesh &pmesh,
                           ParGridFunction &nodes,
                           const char *title,
                           int position)
{
   if (auto *discrete =
          dynamic_cast<DiscreteAdaptTC *>(vis_data.target.get()))
   {
      if (vis_data.target5_data)
      {
         vis_data.target5_data->Remap(nodes);
         discrete->SetParDiscreteTargetSize(
            vis_data.target5_data->CurrentSize());
      }
      else if (vis_data.target6_data)
      {
         vis_data.target6_data->Remap(nodes);
         discrete->SetParDiscreteTargetSize(
            vis_data.target6_data->CurrentSize());
         discrete->SetParDiscreteTargetAspectRatio(
            vis_data.target6_data->CurrentAspect());
      }
      else if (vis_data.target8_data)
      {
         vis_data.target8_data->Remap(nodes);
         discrete->SetParDiscreteTargetSize(
            vis_data.target8_data->CurrentSize());
         discrete->SetParDiscreteTargetOrientation(
            vis_data.target8_data->CurrentOri());
      }
      else
      {
         discrete->ResetUpdateFlags();
         discrete->UpdateTargetSpecification(
            nodes, false, nodes.ParFESpace()->GetOrdering());
      }
   }
   vis_data.target->SetNodes(nodes);
   vis_tmop_metric_p(mesh_poly_deg, *vis_data.metric, *vis_data.target,
                     pmesh, const_cast<char *>(title), position);
}

void VisualizeField(ParMesh &pmesh,
                    ParGridFunction &field,
                    const char *title,
                    int x,
                    int y,
                    int w = 600,
                    int h = 600)
{
   socketstream sock;
   if (Mpi::Root())
   {
      sock.open("localhost", 19916);
      sock.precision(8);
      sock << "solution\n";
   }

   pmesh.PrintAsOne(sock);
   field.SaveAsOne(sock);

   if (Mpi::Root())
   {
      sock << "window_title '" << title << "'\n"
           << "window_geometry " << x << " " << y << " " << w << " " << h
           << "\n"
           << "keys jRmclA\n" << std::flush;
   }
}

void GetMeshOptimizerEssentialTrueDofs(const ParFiniteElementSpace &pfespace,
                                       bool move_bnd,
                                       Array<int> &ess_tdofs)
{
   ess_tdofs.DeleteAll();
   const ParMesh *pmesh = pfespace.GetParMesh();
   if (pmesh->bdr_attributes.Size() == 0) { return; }

   if (!move_bnd)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      pfespace.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
      return;
   }

   const int dim = pmesh->Dimension();
   int n = 0;
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      const int nd = pfespace.GetBE(i)->GetDof();
      const int attr = pmesh->GetBdrElement(i)->GetAttribute();
      MFEM_VERIFY(!(dim == 2 && attr == 3),
                  "Boundary attribute 3 must be used only for 3D meshes. "
                  "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                  "components, rest for free nodes), or use -fix-bnd.");
      if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
      if (attr == 4) { n += nd * dim; }
   }

   Array<int> vdofs, ess_vdofs(n);
   n = 0;
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      const int nd = pfespace.GetBE(i)->GetDof();
      const int attr = pmesh->GetBdrElement(i)->GetAttribute();
      pfespace.GetBdrElementVDofs(i, vdofs);
      if (attr == 1)
      {
         for (int j = 0; j < nd; j++) { ess_vdofs[n++] = vdofs[j]; }
      }
      else if (attr == 2)
      {
         for (int j = 0; j < nd; j++) { ess_vdofs[n++] = vdofs[j + nd]; }
      }
      else if (attr == 3)
      {
         for (int j = 0; j < nd; j++) { ess_vdofs[n++] = vdofs[j + 2 * nd]; }
      }
      else if (attr == 4)
      {
         for (int j = 0; j < vdofs.Size(); j++) { ess_vdofs[n++] = vdofs[j]; }
      }
   }

   Array<int> ess_vdof_marker, ess_tdof_marker;
   FiniteElementSpace::ListToMarker(ess_vdofs, pfespace.GetVSize(),
                                    ess_vdof_marker);
   ess_tdof_marker.SetSize(pfespace.GetTrueVSize());
   pfespace.Dof_TrueDof_Matrix()->BooleanMultTranspose(
      1, ess_vdof_marker, 0, ess_tdof_marker);
   FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdofs);
}

void GetFittingEssentialTrueDofs(const ParFiniteElementSpace &pfespace,
                                 bool move_bnd,
                                 int marking_type,
                                 Array<int> &ess_tdofs)
{
   if (move_bnd || marking_type == 0)
   {
      GetMeshOptimizerEssentialTrueDofs(pfespace, move_bnd, ess_tdofs);
      return;
   }

   const ParMesh *pmesh = pfespace.GetParMesh();
   if (pmesh->bdr_attributes.Size() == 0)
   {
      ess_tdofs.DeleteAll();
      return;
   }

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   MFEM_VERIFY(marking_type <= ess_bdr.Size(),
               "Surface marking boundary attribute " << marking_type
               << " is not present in the mesh.");
   ess_bdr[marking_type - 1] = 0;
   pfespace.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
}

IntegrationRules &SelectIntegrationRules(int quad_type)
{
   switch (quad_type)
   {
      case 1: return IntRulesLo;
      case 2: return IntRules;
      case 3: return IntRulesCU;
      default: MFEM_ABORT("Unknown quadrature rule type: " << quad_type);
   }
}

void MarkSurfaceFittingDofs(ParMesh &pmesh,
                            ParGridFunction &level_set,
                            int marking_type,
                            Array<bool> &marker,
                            ParGridFunction *marker_vis = nullptr)
{
   ParFiniteElementSpace *sfes = level_set.ParFESpace();
   marker.SetSize(level_set.Size());
   marker = false;

   ParGridFunction local_marker(sfes);
   local_marker = 0.0;
   real_t *local_marker_data = local_marker.HostReadWrite();

   Array<int> dofs;
   if (marking_type == 0)
   {
      L2_FECollection mat_coll(0, pmesh.Dimension());
      ParFiniteElementSpace mat_fes(&pmesh, &mat_coll);
      ParGridFunction mat(&mat_fes);
      real_t *mat_data = mat.HostWrite();
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         mat_data[e] = material_id(e, level_set);
      }

      mat.ExchangeFaceNbrData();
      const Vector &face_nbr_data = mat.FaceNbrData();
      const real_t *face_nbr = face_nbr_data.HostRead();
      const real_t *local_material = mat.HostRead();

      for (int i = 0; i < pmesh.GetNumFaces(); i++)
      {
         auto tr = pmesh.GetInteriorFaceTransformations(i);
         if (!tr) { continue; }
         if (local_material[tr->Elem1No] != local_material[tr->Elem2No])
         {
            sfes->GetFaceDofs(i, dofs);
            const int *face_dofs = dofs.HostRead();
            for (int j = 0; j < dofs.Size(); j++)
            {
               local_marker_data[face_dofs[j]] = 1.0;
            }
         }
      }

      for (int i = 0; i < pmesh.GetNSharedFaces(); i++)
      {
         auto tr = pmesh.GetSharedFaceTransformations(i);
         if (!tr) { continue; }
         const int face = pmesh.GetSharedFace(i);
         const real_t mat1 = local_material[tr->Elem1No];
         const real_t mat2 = face_nbr[tr->Elem2No - pmesh.GetNE()];
         if (mat1 != mat2)
         {
            sfes->GetFaceDofs(face, dofs);
            const int *face_dofs = dofs.HostRead();
            for (int j = 0; j < dofs.Size(); j++)
            {
               local_marker_data[face_dofs[j]] = 1.0;
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < pmesh.GetNBE(); i++)
      {
         if (pmesh.GetBdrElement(i)->GetAttribute() != marking_type)
         {
            continue;
         }
         sfes->GetBdrElementVDofs(i, dofs);
         const int *bdr_dofs = dofs.HostRead();
         for (int j = 0; j < dofs.Size(); j++)
         {
            local_marker_data[bdr_dofs[j]] = 1.0;
         }
      }
   }

   local_marker.ExchangeFaceNbrData();
   {
      GroupCommunicator &gcomm = sfes->GroupComm();
      local_marker_data = local_marker.HostReadWrite();
      Array<real_t> marker_array(local_marker_data, local_marker.Size());
      gcomm.Reduce<real_t>(marker_array, GroupCommunicator::Max);
      gcomm.Bcast(marker_array);
   }
   local_marker.ExchangeFaceNbrData();

   const real_t *shared_marker_data = local_marker.HostRead();
   bool *marker_data = marker.HostWrite();
   for (int i = 0; i < local_marker.Size(); i++)
   {
      marker_data[i] = (shared_marker_data[i] == 1.0);
   }
   if (marker_vis) { *marker_vis = local_marker; }
}

/** Run Enzyme TMOP optimization and update the mesh in place.
    Returns 0 on convergence and 2 on nonconvergence.
    exactaction differentiates W;
    freeze_target_linearization freezes it throughout the Newton state.
    Positive surface_fit_adapt scales stalled fitting penalties; positive
    surface_fit_tol enables fitting-error stopping. */
template <int dim>
int RunOptimizer(ParMesh &pmesh,
                 ParFiniteElementSpace &pfespace,
                 ParGridFunction &x,
                 IntegrationRules &irules,
                 int quad_order,
                 const Array<int> &ess_tdofs,
                 real_t min_detJ,
                 int solver_iter,
                 real_t solver_rtol,
                 real_t solver_atol,
                 int lin_solver,
                 int solver_art_type,
                 int max_lin_iter,
                 int target_id,
                 int metric_id,
                 bool exactaction,
                 bool freeze_target_linearization,
                 real_t lim_const,
                 int verbosity_level,
                 const SurfaceFittingOptions *surface_fit_options = nullptr,
                 real_t surface_fit_tol = -1.0,
                 EnzymeOptimizerResult *run_result = nullptr,
                 real_t surface_fit_adapt = 0.0,
                 real_t surface_fit_weight_limit = 1.0e20,
                 bool surface_fit_converge_error = false,
                 bool surface_fit_require_stationarity = false,
                 int derivative_backend = TENSOR_KERNEL_DERIVATIVES)
{
   Vector Xtrue(pfespace.GetTrueVSize());
   x.GetTrueDofs(Xtrue);

   const IntegrationRule &ir =
      irules.Get(pmesh.GetTypicalElementGeometry(), quad_order);
   EnzymeTMOPFunctional<dim> functional(pfespace, pmesh, ir, target_id,
                                        metric_id, exactaction,
                                        freeze_target_linearization,
                                        Xtrue, lim_const,
                                        surface_fit_options,
                                        derivative_backend);
   functional.UpdateAfterMeshPositionChange(Xtrue);
   auto constrained_tmop_grad_norm = [&](const Vector &x) -> real_t
   {
      Vector grad(x.Size());
      functional.Gradient(x, grad);
      if (ess_tdofs.Size() > 0) { grad.SetSubVector(ess_tdofs, 0.0); }
      return GlobalVectorNorm(pfespace.GetComm(), grad);
   };
   const real_t init_metric_energy = functional.MetricEnergy(Xtrue);
   const real_t init_energy =
      init_metric_energy + functional.LimitingEnergy(Xtrue) +
      functional.SurfaceFittingEnergy(Xtrue);
   real_t init_fit_avg = 0.0, init_fit_max = 0.0;
   if (functional.HasSurfaceFitting() && surface_fit_tol > 0.0)
   {
      functional.GetSurfaceFittingErrors(init_fit_avg, init_fit_max);
      if (init_fit_max <= surface_fit_tol)
      {
         if (run_result)
         {
            run_result->converged = true;
            run_result->status = 0;
            run_result->initial_energy = init_energy;
            run_result->final_energy = init_energy;
            run_result->final_grad_norm = constrained_tmop_grad_norm(Xtrue);
            run_result->final_surface_fit_coefficient =
               functional.GetSurfaceFittingCoefficient();
         }
         if (Mpi::Root() && verbosity_level > 0)
         {
            std::cout << std::scientific << std::setprecision(4)
                      << "Surface fitting tolerance reached before Newton: "
                      << "fit_avg=" << init_fit_avg
                      << ", fit_max=" << init_fit_max << '\n';
         }
         x.SetFromTrueDofs(Xtrue);
         pmesh.SetNodalGridFunction(&x);
         pmesh.NodesUpdated();
         pmesh.ExchangeFaceNbrData();
         return 0;
      }
   }

   MFEM_VERIFY(lin_solver == 2 || lin_solver == 3,
               "Only -ls 2 and -ls 3 are supported for now.");
   EnzymeTMOPNonlinearForm<dim> oper(pfespace, functional);
   oper.SetEssentialTrueDofs(ess_tdofs);
   oper.SetReference(Xtrue);

#ifdef MFEM_USE_SINGLE
   const real_t linsol_rtol = 1e-5;
#else
   const real_t linsol_rtol = 1e-12;
#endif
   IterativeSolver::PrintLevel linear_print;
   if (verbosity_level > 1)
   {
      linear_print.Errors().Warnings().FirstAndLast();
   }
   if (verbosity_level > 2)
   {
      linear_print.Errors().Warnings().Iterations();
   }
   MINRESSolver linear_solver(pfespace.GetComm());
   linear_solver.SetMaxIter(max_lin_iter);
   linear_solver.SetRelTol(linsol_rtol);
   linear_solver.SetAbsTol(0.0);
   linear_solver.SetPrintLevel(linear_print);
   OperatorJacobiSmoother jacobi;
   if (lin_solver == 3)
   {
      jacobi.SetPositiveDiagonal(true);
      linear_solver.SetPreconditioner(jacobi);
   }

   EnzymeTMOPNewtonSolver<dim> solver(pfespace.GetComm(), ir, oper);
   solver.SetIntegrationRules(irules, quad_order);
   solver.SetMinDetPtr(&min_detJ);
   if (functional.HasSurfaceFitting())
   {
      solver.SetMinimumDeterminantThreshold(0.001*min_detJ);
   }
   solver.SetOperator(oper);
   solver.SetPreconditioner(linear_solver);
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(solver_atol);
   if (solver_art_type > 0 && surface_fit_adapt <= 0.0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   else if (solver_art_type > 0 && Mpi::Root() && verbosity_level > 0)
   {
      std::cout << "Disabling adaptive linear tolerance while the surface "
                << "fitting coefficient is adaptive.\n";
   }
   if (functional.HasSurfaceFitting() && surface_fit_adapt > 0.0)
   {
      solver.ConfigureAdaptiveSurfaceFitting(
         surface_fit_adapt, surface_fit_tol, surface_fit_weight_limit,
         surface_fit_converge_error &&
         !surface_fit_require_stationarity);
   }
   IterativeSolver::PrintLevel newton_print;
   if (verbosity_level > 0)
   {
      newton_print.Errors().Warnings().Iterations();
   }
   solver.SetPrintLevel(newton_print);

   Vector zero;
   solver.Mult(zero, Xtrue);

   // Increase the penalty if residual convergence precedes the fitting target.
   if (functional.HasSurfaceFitting() && surface_fit_adapt > 0.0 &&
       surface_fit_converge_error && surface_fit_tol >= 0.0)
   {
      for (int stage = 0; stage < 10; stage++)
      {
         functional.UpdateAfterMeshPositionChange(Xtrue);
         real_t stage_fit_avg = 0.0, stage_fit_max = 0.0;
         functional.GetSurfaceFittingErrors(stage_fit_avg, stage_fit_max);
         if (stage_fit_max <= surface_fit_tol)
         {
            if (!surface_fit_require_stationarity || solver.GetConverged())
            {
               break;
            }

            // Finish stationarity at the accepted fitting coefficient.
            oper.SetReference(Xtrue);
            solver.ResetAdaptiveSurfaceFittingState();
            solver.Mult(zero, Xtrue);
            continue;
         }
         if (functional.GetSurfaceFittingCoefficient() >=
             surface_fit_weight_limit)
         {
            break;
         }

         const real_t factor = std::min(
            surface_fit_adapt,
            surface_fit_weight_limit /
            functional.GetSurfaceFittingCoefficient());
         functional.ScaleSurfaceFittingCoefficient(factor);
         if (Mpi::Root() && verbosity_level > 0)
         {
            std::cout << "Restarting surface fit: fit_avg=" << stage_fit_avg
                      << ", fit_max=" << stage_fit_max
                      << ", coefficient="
                      << functional.GetSurfaceFittingCoefficient() << '\n';
         }
         oper.SetReference(Xtrue);
         solver.ResetAdaptiveSurfaceFittingState();
         solver.Mult(zero, Xtrue);
      }
   }

   x.SetFromTrueDofs(Xtrue);
   pmesh.SetNodalGridFunction(&x);
   pmesh.NodesUpdated();
   pmesh.ExchangeFaceNbrData();
   functional.UpdateAfterMeshPositionChange(Xtrue);

   const real_t final_metric_energy = functional.MetricEnergy(Xtrue);
   const real_t final_energy =
      final_metric_energy + functional.LimitingEnergy(Xtrue) +
      functional.SurfaceFittingEnergy(Xtrue);
   const real_t final_grad_norm = constrained_tmop_grad_norm(Xtrue);
   real_t final_fit_avg = 0.0, final_fit_max = 0.0;
   if (functional.HasSurfaceFitting())
   {
      functional.GetSurfaceFittingErrors(final_fit_avg, final_fit_max);
   }
   const bool fit_converged = !functional.HasSurfaceFitting() ||
                              surface_fit_tol < 0.0 ||
                              final_fit_max <= surface_fit_tol;
   bool converged = solver.GetConverged();
   if (surface_fit_converge_error)
   {
      converged = fit_converged &&
                  (!surface_fit_require_stationarity || converged);
   }
   if (functional.HasSurfaceFitting() && surface_fit_tol > 0.0 &&
       final_fit_max <= surface_fit_tol)
   {
      if (Mpi::Root() && verbosity_level > 0)
      {
         std::cout << std::scientific << std::setprecision(4)
                   << "Surface fitting tolerance reached after Newton: "
                   << "fit_avg=" << final_fit_avg
                   << ", fit_max=" << final_fit_max << '\n';
      }
   }
   if (run_result)
   {
      run_result->converged = converged;
      run_result->status = converged ? 0 : 2;
      run_result->initial_energy = init_energy;
      run_result->final_energy = final_energy;
      run_result->final_grad_norm = final_grad_norm;
      run_result->final_surface_fit_coefficient =
         functional.GetSurfaceFittingCoefficient();
   }
   x.SetFromTrueDofs(Xtrue);
   pmesh.SetNodalGridFunction(&x);
   pmesh.NodesUpdated();
   pmesh.ExchangeFaceNbrData();
   if (Mpi::Root() && verbosity_level > 0)
   {
      std::cout << std::scientific << std::setprecision(4);
      std::cout << "Initial strain energy: " << init_energy
                << " = metrics: " << init_metric_energy
                << " + extra terms: " << init_energy - init_metric_energy
                << '\n';
      std::cout << "  Final strain energy: " << final_energy
                << " = metrics: " << final_metric_energy
                << " + extra terms: " << final_energy - final_metric_energy
                << '\n';
      std::cout << "Final TMOP gradient norm: " << final_grad_norm << '\n';
      if (init_energy != 0.0)
      {
         std::cout << "The strain energy decreased by: "
                   << (init_energy - final_energy) * 100.0 / init_energy
                   << " %.\n";
      }
      if (functional.HasSurfaceFitting())
      {
         std::cout << "Avg fitting error: " << final_fit_avg << '\n'
                   << "Max fitting error: " << final_fit_max << '\n';
      }
   }

   return converged ? 0 : 2;
}

} // namespace

#endif // MFEM_USE_MPI && MFEM_USE_ENZYME

#endif // MFEM_PMESH_OPTIMIZER_ENZYME_COMMON_HPP
