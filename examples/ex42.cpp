#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "mfem.hpp"
#include "tensor.hpp"

using namespace std;
using namespace mfem;
using namespace serac;

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... Args>
return_type __enzyme_autodiff(Args...);

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);

template <int dim> struct LinearElasticMaterial {
  static constexpr auto I = Identity<dim>();

  tensor<double, dim, dim> stress(const tensor<double, dim, dim> &dudx) const {
    auto epsilon = sym(dudx);
    return lambda * tr(epsilon) * I + 2.0 * mu * epsilon;
  }

  tensor<double, dim, dim>
  action_of_gradient(const tensor<double, dim, dim> & /* dudx */,
                     const tensor<double, dim, dim> &ddudxi) const {
    return stress(ddudxi);
  }

  double mu = 50;
  double lambda = 100;
};

template <int dim> struct NeoHookeanMaterial {
  static_assert(dim == 3, "NeoHookean model only defined in 3D");

  static constexpr auto I = Identity<dim>();

  template <typename T>
  tensor<T, dim, dim>
  stress(const tensor<T, dim, dim> &__restrict__ du_dx) const {
    T J = det(I + du_dx);
    T p = -2.0 * D1 * J * (J - 1);
    auto devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
    return -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
  }

  static void stress_wrapper(NeoHookeanMaterial<dim> *self,
                             tensor<double, dim, dim> &du_dx,
                             tensor<double, dim, dim> &sigma) {
    sigma = self->stress(du_dx);
  }

  tensor<double, dim, dim>
  action_of_gradient(const tensor<double, dim, dim> &dudx,
                     const tensor<double, dim, dim> &ddudxi) const {
    // return action_of_gradient_enzyme_fwd(dudx, ddudxi);
    // return action_of_gradient_enzyme_rev(dudx, ddudxi);
    // return action_of_gradient_fd(dudx, ddudxi);
    return action_of_gradient_symbolic(dudx, ddudxi);
    // return action_of_gradient_dual(dudx, ddudxi);
  }

  tensor<double, dim, dim>
  action_of_gradient_dual(const tensor<double, dim, dim> &dudx,
                          const tensor<double, dim, dim> &ddudxi) const {
    tensor<dual<double>, dim, dim> dudx_and_ddudxi;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        dudx_and_ddudxi[i][j].value = dudx[i][j];
        dudx_and_ddudxi[i][j].gradient = ddudxi[i][j];
      }
    }
    return get_gradient<>(stress(dudx_and_ddudxi));
  }

  tensor<double, dim, dim>
  action_of_gradient_enzyme_fwd(const tensor<double, dim, dim> &dudx,
                                const tensor<double, dim, dim> &ddudxi) const {
    tensor<double, dim, dim> sigma{};
    tensor<double, dim, dim> dsigma{};

    __enzyme_fwddiff<void>(stress_wrapper, enzyme_const, this, enzyme_dup,
                           &dudx, &ddudxi, enzyme_dupnoneed, &sigma, &dsigma);
    return dsigma;
  }

  tensor<double, dim, dim>
  action_of_gradient_enzyme_rev(const tensor<double, dim, dim> &dudx,
                                const tensor<double, dim, dim> &ddudxi) const {
    tensor<double, dim, dim, dim, dim> gradient{};
    tensor<double, dim, dim> sigma{};
    tensor<double, dim, dim> dir{};

    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        dir[i][j] = 1;
        __enzyme_autodiff<void>(stress_wrapper, enzyme_const, this, enzyme_dup,
                                &dudx, &gradient[i][j], enzyme_dupnoneed,
                                &sigma, &dir);
        dir[i][j] = 0;
      }
    }
    return ddot(gradient, ddudxi);
  }

  tensor<double, dim, dim>
  action_of_gradient_fd(const tensor<double, dim, dim> &dudx,
                        const tensor<double, dim, dim> &ddudxi) const {
    return (stress(dudx + 1.0e-8 * ddudxi) - stress(dudx - 1.0e-8 * ddudxi)) /
           2.0e-8;
  }

  // d(stress)_{ij} := (d(stress)_ij / d(du_dx)_{kl}) * d(du_dx)_{kl}
  // Only works with 3D stress
  tensor<double, dim, dim>
  action_of_gradient_symbolic(const tensor<double, dim, dim> &du_dx,
                              const tensor<double, dim, dim> &ddu_dx) const {
    tensor<double, dim, dim> F = I + du_dx;
    tensor<double, dim, dim> invFT = inv(transpose(F));
    tensor<double, dim, dim> devB =
        dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
    double J = det(F);
    double coef = (C1 / pow(J, 5.0 / 3.0));
    double a1 = ddot(invFT, ddu_dx);
    double a2 = ddot(F, ddu_dx);

    return (2.0 * D1 * J * a1 - (4.0 / 3.0) * coef * a2) * I -
           ((10.0 / 3.0) * coef * a1) * devB +
           (2 * coef) * (dot(ddu_dx, transpose(F)) + dot(F, transpose(ddu_dx)));
  }

  double C1 = 50.0;
  double D1 = 100.0;
};

struct QuadratureData {
  QuadratureData(const int dim, const int NE, const int NQ)
      : detJw_(NE * NQ),
        elasticity_tensor_data_(dim * dim * dim * dim * NE * NQ) {
    detJw_.UseDevice(true);
    elasticity_tensor_data_.UseDevice(true);
  }

  Vector detJw_;
  Vector elasticity_tensor_data_;
};

class ElasticityGradientOperator;

class ElasticityOperator : public Operator {
public:
  ElasticityOperator(const int dim, const int size,
                     ParFiniteElementSpace &h1_fes,
                     const Array<int> &ess_tdof_list);

  ~ElasticityOperator() {
    delete gradient;
    delete qdata_;
  }

  virtual void Mult(const Vector &X, Vector &Y) const override;

  Operator &GetGradient(const Vector &x) const override;

  void SetState(const Vector &x);

  void GradientMult(const Vector &X, Vector &Y) const;

public:
  const int size_;
  const int DIM_;
  const int VDIM_;
  Array<int> ess_tdof_list_;
  ParFiniteElementSpace &h1_fes_;
  const int NE_;
  IntegrationRule *ir_ = nullptr;
  int Ndofs1d_;
  int Nq1d_;
  const Operator *h1_element_restriction_;
  const Operator *h1_element_prolongation_;
  ElasticityGradientOperator *gradient;
  // Data on the quadrature points
  mutable QuadratureData *qdata_;
  const GeometricFactors *geometric_factors_;
  const DofToQuad *maps;
  // State E-vector
  mutable Vector X_evec, Y_evec, current_state, cstate_evec;
  mutable bool qdata_is_current = false;

  std::function<void(const int, const Array<double> &, const Array<double> &,
                     const Array<double> &, const Vector &, const Vector &,
                     const Vector &, Vector &)>
      element_kernel_wrapper;

  std::function<void(const int, const Array<double> &, const Array<double> &,
                     const Array<double> &, const Vector &, const Vector &,
                     const Vector &, Vector &, const Vector &)>
      element_apply_gradient_kernel_wrapper;

  template <typename material_type>
  void SetMaterial(const material_type &material) {
    element_kernel_wrapper =
        [=](const int NE, const Array<double> &B_, const Array<double> &G_,
            const Array<double> &W_, const Vector &Jacobian_,
            const Vector &detJ_, const Vector &X_, Vector &Y_) {
          const int id = (Ndofs1d_ << 4) | Nq1d_;
          if (DIM_ == 3) {
            switch (id) {
            case 0x22:
              Apply3D<2, 2, material_type>(NE, B_, G_, W_, Jacobian_, detJ_, X_,
                                           Y_, material);
              break;
            default:
              MFEM_ABORT("not implemented");
            }
          } else {
            MFEM_ABORT("dim != 3 not implemented");
          }
        };

    element_apply_gradient_kernel_wrapper =
        [=](const int NE, const Array<double> &B_, const Array<double> &G_,
            const Array<double> &W_, const Vector &Jacobian_,
            const Vector &detJ_, const Vector &dU_, Vector &dF_,
            const Vector &U_) {
          const int id = (Ndofs1d_ << 4) | Nq1d_;
          if (DIM_ == 3) {
            switch (id) {
            case 0x22:
              ApplyGradient3D<2, 2, material_type>(
                  NE, B_, G_, W_, Jacobian_, detJ_, dU_, dF_, U_, material);
              break;
            default:
              MFEM_ABORT("not implemented");
            }
          } else {
            MFEM_ABORT("dim != 3 not implemented");
          }
        };
  }

  template <int DIM, int Q1D>
  MFEM_HOST_DEVICE static inline void
  UpdateQuadratureDataKernel(const int NE, const Array<double> &weights,
                             const Vector &Jacobians, Vector &detJw_) {
    // const auto w = Reshape(weights.Read(), Q1D, Q1D);
    // const auto J = Reshape(Jacobians.Read(), Q1D, Q1D, 2, 2, NE);
    // auto detJw = Reshape(detJw_.Write(), Q1D, Q1D, NE);
  }

  void UpdateQuadratureData(const Vector &X) const {
    // TODO: Write dispatcher
  }

  template <int D1D, int Q1D, typename material_type>
  static inline void
  ApplyGradient3D(const int NE, const Array<double> &B_,
                  const Array<double> &G_, const Array<double> &W_,
                  const Vector &Jacobian_, const Vector &detJ_,
                  const Vector &dU_, Vector &dF_, const Vector &U_,
                  const material_type &material) {
    constexpr int DIM = 3;

    MFEM_VERIFY(D1D <= MAX_D1D, "Maximum D1D reached");
    MFEM_VERIFY(Q1D <= MAX_Q1D, "Maximum Q1D reached");
    // 1D Basis functions B_
    // column-major layout nq1d x ndofs1d
    const auto B = Reshape(B_.Read(), Q1D, D1D);
    // Gradients of 1D basis functions evaluated at quadrature points G_
    // column-major layout nq1d x ndofs1d
    const auto G = Reshape(G_.Read(), Q1D, D1D);
    const auto qweights = Reshape(W_.Read(), Q1D, Q1D, Q1D);
    // Jacobians of the element transformations at all quadrature points.
    // This array uses a column-major layout with dimensions (nq1d x nq1d x SDIM
    // x DIM x NE)
    const auto J = Reshape(Jacobian_.Read(), Q1D, Q1D, Q1D, DIM, DIM, NE);
    const auto detJ = Reshape(detJ_.Read(), Q1D, Q1D, Q1D, NE);
    // Input vector dU_
    // ndofs1d x ndofs1d x VDIM x NE
    const auto dU = Reshape(dU_.Read(), D1D, D1D, D1D, DIM, NE);
    // Output vector Y_
    // ndofs1d x ndofs1d x VDIM x NE
    auto force = Reshape(dF_.ReadWrite(), D1D, D1D, D1D, DIM, NE);
    // Input vector U_
    // ndofs1d x ndofs1d x VDIM x NE
    const auto U = Reshape(U_.Read(), D1D, D1D, D1D, DIM, NE);

    // cauchy stress
    tensor<double, Q1D, Q1D, Q1D, DIM, DIM> invJ_dsigma_detJw;

    for (int e = 0; e < NE; e++) {
      // du/dxi
      tensor<double, Q1D, Q1D, Q1D, DIM, DIM> dudxi{};
      for (int c = 0; c < DIM; ++c) {
        for (int dz = 0; dz < D1D; ++dz) {
          tensor<double, Q1D, Q1D, DIM> gradXY{};
          for (int dy = 0; dy < D1D; ++dy) {
            tensor<double, Q1D, 2> gradX;
            for (int qx = 0; qx < Q1D; ++qx) {
              gradX[qx][0] = 0.0;
              gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx) {
              const double s = U(dx, dy, dz, c, e);
              for (int qx = 0; qx < Q1D; ++qx) {
                gradX[qx][0] += s * B(qx, dx);
                gradX[qx][1] += s * G(qx, dx);
              }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
              const double wy = B(qy, dy);
              const double wDy = G(qy, dy);
              for (int qx = 0; qx < Q1D; ++qx) {
                const double wx = gradX[qx][0];
                const double wDx = gradX[qx][1];
                gradXY[qy][qx][0] += wDx * wy;
                gradXY[qy][qx][1] += wx * wDy;
                gradXY[qy][qx][2] += wx * wy;
              }
            }
          }
          for (int qz = 0; qz < Q1D; ++qz) {
            const double wz = B(qz, dz);
            const double wDz = G(qz, dz);
            for (int qy = 0; qy < Q1D; ++qy) {
              for (int qx = 0; qx < Q1D; ++qx) {
                dudxi[qz][qy][qx][c][0] += gradXY[qy][qx][0] * wz;
                dudxi[qz][qy][qx][c][1] += gradXY[qy][qx][1] * wz;
                dudxi[qz][qy][qx][c][2] += gradXY[qy][qx][2] * wDz;
              }
            }
          }
        }
      }

      // \partial du / \partial xi
      tensor<double, Q1D, Q1D, Q1D, DIM, DIM> ddudxi{};
      for (int c = 0; c < DIM; ++c) {
        for (int dz = 0; dz < D1D; ++dz) {
          tensor<double, Q1D, Q1D, DIM> ddudxiXY{};
          for (int dy = 0; dy < D1D; ++dy) {
            tensor<double, Q1D, 2> ddudxiX;
            for (int qx = 0; qx < Q1D; ++qx) {
              ddudxiX[qx][0] = 0.0;
              ddudxiX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx) {
              const double s = dU(dx, dy, dz, c, e);
              for (int qx = 0; qx < Q1D; ++qx) {
                ddudxiX[qx][0] += s * B(qx, dx);
                ddudxiX[qx][1] += s * G(qx, dx);
              }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
              const double wy = B(qy, dy);
              const double wDy = G(qy, dy);
              for (int qx = 0; qx < Q1D; ++qx) {
                const double wx = ddudxiX[qx][0];
                const double wDx = ddudxiX[qx][1];
                ddudxiXY[qy][qx][0] += wDx * wy;
                ddudxiXY[qy][qx][1] += wx * wDy;
                ddudxiXY[qy][qx][2] += wx * wy;
              }
            }
          }
          for (int qz = 0; qz < Q1D; ++qz) {
            const double wz = B(qz, dz);
            const double wDz = G(qz, dz);
            for (int qy = 0; qy < Q1D; ++qy) {
              for (int qx = 0; qx < Q1D; ++qx) {
                ddudxi[qz][qy][qx][c][0] += ddudxiXY[qy][qx][0] * wz;
                ddudxi[qz][qy][qx][c][1] += ddudxiXY[qy][qx][1] * wz;
                ddudxi[qz][qy][qx][c][2] += ddudxiXY[qy][qx][2] * wDz;
              }
            }
          }
        }
      }

      for (int qx = 0; qx < Q1D; qx++) {
        for (int qy = 0; qy < Q1D; qy++) {
          for (int qz = 0; qz < Q1D; qz++) {
            auto invJqp = inv(make_tensor<DIM, DIM>(
                [&](int i, int j) { return J(qx, qy, qz, i, j, e); }));

            // du/dx = du/dxi * dxi/dx = du/dxi * inv(dx/dxi)
            auto dudx = dudxi(qz, qy, qx) * invJqp;

            auto dsigma = material.action_of_gradient(dudx, ddudxi(qz, qy, qx));

            invJ_dsigma_detJw(qx, qy, qz) =
                invJqp * dsigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
          }
        }
      }

      for (int c = 0; c < DIM; ++c) {
        for (int qz = 0; qz < Q1D; ++qz) {
          tensor<double, Q1D, Q1D, DIM> gradXY{};
          for (int qy = 0; qy < Q1D; ++qy) {
            tensor<double, D1D, DIM> gradX{};
            for (int qx = 0; qx < Q1D; ++qx) {
              const double gX = invJ_dsigma_detJw[qx][qy][qz][0][c];
              const double gY = invJ_dsigma_detJw[qx][qy][qz][1][c];
              const double gZ = invJ_dsigma_detJw[qx][qy][qz][2][c];
              for (int dx = 0; dx < D1D; ++dx) {
                const double wx = B(qx, dx);
                const double wDx = G(qx, dx);
                gradX[dx][0] += gX * wDx;
                gradX[dx][1] += gY * wx;
                gradX[dx][2] += gZ * wx;
              }
            }
            for (int dy = 0; dy < D1D; ++dy) {
              const double wy = B(qy, dy);
              const double wDy = G(qy, dy);
              for (int dx = 0; dx < D1D; ++dx) {
                gradXY[dy][dx][0] += gradX[dx][0] * wy;
                gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                gradXY[dy][dx][2] += gradX[dx][2] * wy;
              }
            }
          }
          for (int dz = 0; dz < D1D; ++dz) {
            const double wz = B(qz, dz);
            const double wDz = G(qz, dz);
            for (int dy = 0; dy < D1D; ++dy) {
              for (int dx = 0; dx < D1D; ++dx) {
                force(dx, dy, dz, c, e) +=
                    ((gradXY[dy][dx][0] * wz) + (gradXY[dy][dx][1] * wz) +
                     (gradXY[dy][dx][2] * wDz));
              }
            }
          }
        }
      }
    } // for each element
  }

  template <int D1D, int Q1D, typename material_type>
  static inline void
  Apply3D(const int NE, const Array<double> &B_, const Array<double> &G_,
          const Array<double> &W_, const Vector &Jacobian_, const Vector &detJ_,
          const Vector &X_, Vector &Y_, const material_type &material) {
    constexpr int DIM = 3;

    MFEM_VERIFY(D1D <= MAX_D1D, "Maximum D1D reached");
    MFEM_VERIFY(Q1D <= MAX_Q1D, "Maximum Q1D reached");
    // 1D Basis functions B_
    // column-major layout nq1d x ndofs1d
    const auto B = Reshape(B_.Read(), Q1D, D1D);
    // Gradients of 1D basis functions evaluated at quadrature points G_
    // column-major layout nq1d x ndofs1d
    const auto G = Reshape(G_.Read(), Q1D, D1D);
    const auto qweights = Reshape(W_.Read(), Q1D, Q1D, Q1D);
    // Jacobians of the element transformations at all quadrature points.
    // This array uses a column-major layout with dimensions (nq1d x nq1d x SDIM
    // x DIM x NE)
    const auto J = Reshape(Jacobian_.Read(), Q1D, Q1D, Q1D, DIM, DIM, NE);
    const auto detJ = Reshape(detJ_.Read(), Q1D, Q1D, Q1D, NE);
    // Input vector X_
    // ndofs1d x ndofs1d x VDIM x NE
    const auto U = Reshape(X_.Read(), D1D, D1D, D1D, DIM, NE);
    // Output vector Y_
    // ndofs1d x ndofs1d x VDIM x NE
    auto force = Reshape(Y_.ReadWrite(), D1D, D1D, D1D, DIM, NE);

    // cauchy stress
    tensor<double, Q1D, Q1D, Q1D, DIM, DIM> invJ_sigma_detJw;

    for (int e = 0; e < NE; e++) {
      // du/dxi
      tensor<double, Q1D, Q1D, Q1D, DIM, DIM> dudxi{};
      for (int c = 0; c < DIM; ++c) {
        for (int dz = 0; dz < D1D; ++dz) {
          tensor<double, Q1D, Q1D, DIM> gradXY{};
          for (int dy = 0; dy < D1D; ++dy) {
            tensor<double, Q1D, 2> gradX;
            for (int qx = 0; qx < Q1D; ++qx) {
              gradX[qx][0] = 0.0;
              gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx) {
              const double s = U(dx, dy, dz, c, e);
              for (int qx = 0; qx < Q1D; ++qx) {
                gradX[qx][0] += s * B(qx, dx);
                gradX[qx][1] += s * G(qx, dx);
              }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
              const double wy = B(qy, dy);
              const double wDy = G(qy, dy);
              for (int qx = 0; qx < Q1D; ++qx) {
                const double wx = gradX[qx][0];
                const double wDx = gradX[qx][1];
                gradXY[qy][qx][0] += wDx * wy;
                gradXY[qy][qx][1] += wx * wDy;
                gradXY[qy][qx][2] += wx * wy;
              }
            }
          }
          for (int qz = 0; qz < Q1D; ++qz) {
            const double wz = B(qz, dz);
            const double wDz = G(qz, dz);
            for (int qy = 0; qy < Q1D; ++qy) {
              for (int qx = 0; qx < Q1D; ++qx) {
                dudxi[qz][qy][qx][c][0] += gradXY[qy][qx][0] * wz;
                dudxi[qz][qy][qx][c][1] += gradXY[qy][qx][1] * wz;
                dudxi[qz][qy][qx][c][2] += gradXY[qy][qx][2] * wDz;
              }
            }
          }
        }
      }

      for (int qx = 0; qx < Q1D; qx++) {
        for (int qy = 0; qy < Q1D; qy++) {
          for (int qz = 0; qz < Q1D; qz++) {
            auto invJqp = inv(make_tensor<DIM, DIM>(
                [&](int i, int j) { return J(qx, qy, qz, i, j, e); }));

            auto dudx = dudxi(qz, qy, qx) * invJqp;

            auto sigma = material.stress(dudx);

            invJ_sigma_detJw(qx, qy, qz) =
                invJqp * sigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
          }
        }
      }

      for (int c = 0; c < DIM; ++c) {
        for (int qz = 0; qz < Q1D; ++qz) {
          tensor<double, Q1D, Q1D, DIM> gradXY{};
          for (int qy = 0; qy < Q1D; ++qy) {
            tensor<double, D1D, DIM> gradX{};
            for (int qx = 0; qx < Q1D; ++qx) {
              const double gX = invJ_sigma_detJw[qx][qy][qz][0][c];
              const double gY = invJ_sigma_detJw[qx][qy][qz][1][c];
              const double gZ = invJ_sigma_detJw[qx][qy][qz][2][c];
              for (int dx = 0; dx < D1D; ++dx) {
                const double wx = B(qx, dx);
                const double wDx = G(qx, dx);
                gradX[dx][0] += gX * wDx;
                gradX[dx][1] += gY * wx;
                gradX[dx][2] += gZ * wx;
              }
            }
            for (int dy = 0; dy < D1D; ++dy) {
              const double wy = B(qy, dy);
              const double wDy = G(qy, dy);
              for (int dx = 0; dx < D1D; ++dx) {
                gradXY[dy][dx][0] += gradX[dx][0] * wy;
                gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                gradXY[dy][dx][2] += gradX[dx][2] * wy;
              }
            }
          }
          for (int dz = 0; dz < D1D; ++dz) {
            const double wz = B(qz, dz);
            const double wDz = G(qz, dz);
            for (int dy = 0; dy < D1D; ++dy) {
              for (int dx = 0; dx < D1D; ++dx) {
                force(dx, dy, dz, c, e) +=
                    ((gradXY[dy][dx][0] * wz) + (gradXY[dy][dx][1] * wz) +
                     (gradXY[dy][dx][2] * wDz));
              }
            }
          }
        }
      }
    } // for each element
  }
};

class ElasticityGradientOperator : public Operator {
public:
  ElasticityGradientOperator(ElasticityOperator *op);

  void Mult(const Vector &x, Vector &y) const override;

private:
  ElasticityOperator *op_;
};

int main(int argc, char *argv[]) {
  MPI_Session mpi;
  int myid = mpi.WorldRank();

  const char *mesh_file = "../data/beam-hex.mesh";
  int order = 1;
  const char *device_config = "cpu";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  Device device(device_config);
  if (myid == 0) {
    device.Print();
  }

  // Mesh mesh(mesh_file, 1, 1);
  auto mesh =
      Mesh::MakeCartesian3D(8, 1, 1, Element::HEXAHEDRON, 8.0, 1.0, 1.0);
  mesh.EnsureNodes();

  // for (int l = 0; l < 2; l++)
  // {
  //   mesh.UniformRefinement();
  // }



  int dim = mesh.Dimension();

  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();

  FiniteElementCollection *fec;
  fec = new H1_FECollection(order, dim);
  ParFiniteElementSpace fespace(&pmesh, fec, dim, Ordering::byVDIM);
  HYPRE_BigInt size = fespace.GlobalTrueVSize();
  if (myid == 0) {
    cout << "dim = " << dim << endl;
    cout << "Number of finite element unknowns: " << size << endl;
  }

  Array<int> ess_tdof_list;
  if (pmesh.bdr_attributes.Size()) {
    Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[4] = 1;
    ess_bdr[2] = 1;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  Array<int> displaced_tdof_list;
  if (pmesh.bdr_attributes.Size()) {
    Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[2] = 1;
    fespace.GetEssentialTrueDofs(ess_bdr, displaced_tdof_list);
  }

  ParGridFunction U(&fespace);
  U = 0.0;

  ElasticityOperator elasticity_op(pmesh.SpaceDimension(), U.Size(), fespace,
                                   ess_tdof_list);

  const NeoHookeanMaterial<3> material{};

  // tensor<double, 3, 3> du_dx = {{{0.2, 0.4, -0.1}, {0.2, 0.1, 0.3}, {0.01, -0.2, 0.3}}};
  // tensor<double, 3, 3> perturbation = {{{1.0, 0.2, 0.8}, {2.0, 0.1, 0.3}, {0.4, 0.2, 0.7}}};

  // auto grad_fd = (material.stress(du_dx + 1.0e-8 * perturbation) - material.stress(du_dx - 1.0e-8 * perturbation)) / 2.0e-8;
  // auto grad_symbolic = material.action_of_gradient(du_dx, perturbation);

  // std::cout << "grad_fd = " << grad_fd << std::endl;
  // std::cout << "grad_sy = " << grad_symbolic << std::endl;

  // exit(0);

  elasticity_op.SetMaterial(material);

  // Assign load
  U.SetSubVector(displaced_tdof_list, 1.0e-1);

  GMRESSolver gmres(MPI_COMM_WORLD);
  gmres.SetRelTol(1e-8);
  gmres.SetMaxIter(1000);
  gmres.SetKDim(200);
  gmres.SetPrintLevel(2);

  NewtonSolver newton(MPI_COMM_WORLD);
  newton.SetOperator(elasticity_op);
  newton.SetSolver(gmres);
  newton.SetRelTol(1e-8);
  newton.SetMaxIter(10);
  newton.SetPrintLevel(1);

  Vector zero;
  newton.Mult(zero, U);

  ParaViewDataCollection *pd = NULL;
  pd = new ParaViewDataCollection("elast", &pmesh);
  pd->RegisterField("solution", &U);
  pd->SetLevelsOfDetail(order);
  pd->SetDataFormat(VTKFormat::BINARY);
  pd->SetHighOrderOutput(true);
  pd->SetCycle(0);
  pd->SetTime(0.0);
  pd->Save();

  delete pd;
  delete fec;

  return 0;
}

ElasticityGradientOperator::ElasticityGradientOperator(ElasticityOperator *op)
    : Operator(op->Height()), op_(op) {}

void ElasticityGradientOperator::Mult(const Vector &x, Vector &y) const {
  op_->GradientMult(x, y);
}

ElasticityOperator::ElasticityOperator(const int dim, const int size,
                                       ParFiniteElementSpace &h1_fes,
                                       const Array<int> &ess_tdof_list)
    : Operator(h1_fes.GetTrueVSize()), DIM_(dim), size_(size), h1_fes_(h1_fes),
      VDIM_(h1_fes.GetVDim()), NE_(h1_fes.GetParMesh()->GetNE()),
      ess_tdof_list_(ess_tdof_list),
      h1_element_restriction_(
          h1_fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      h1_element_prolongation_(h1_fes.GetProlongationMatrix()) {
  const MemoryType mt = MemoryType::HOST_DEBUG;
  ir_ = const_cast<IntegrationRule *>(
      &IntRules.Get(mfem::Element::HEXAHEDRON, 2 * h1_fes_.GetOrder(0)));

  geometric_factors_ = h1_fes_.GetParMesh()->GetGeometricFactors(
      *ir_, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS, mt);
  qdata_ = new QuadratureData(DIM_, NE_, ir_->GetNPoints());
  maps = &h1_fes_.GetFE(0)->GetDofToQuad(*ir_, DofToQuad::TENSOR);
  Ndofs1d_ = maps->ndof;
  Nq1d_ = maps->nqpt;
  X_evec.SetSize(h1_element_restriction_->Height(),
                 Device::GetDeviceMemoryType());
  Y_evec.SetSize(h1_element_restriction_->Height(),
                 Device::GetDeviceMemoryType());
  cstate_evec.SetSize(h1_element_restriction_->Height(),
                      Device::GetDeviceMemoryType());
  Y_evec.UseDevice(true);

  gradient = new ElasticityGradientOperator(this);
}

void ElasticityOperator::Mult(const Vector &X, Vector &Y) const {
  Vector *X_p = const_cast<Vector *>(&X);
  ParGridFunction x, y;
  x.MakeRef(&h1_fes_, *X_p);
  y.MakeRef(&h1_fes_, Y);

  // L-vector to E-vector
  h1_element_restriction_->Mult(x, X_evec);

  // Reset output vector
  Y_evec = 0.0;

  // Apply operator
  element_kernel_wrapper(NE_, maps->B, maps->G, ir_->GetWeights(),
                         geometric_factors_->J, geometric_factors_->detJ,
                         X_evec, Y_evec);

  // E-vector to L-vector
  h1_element_restriction_->MultTranspose(Y_evec, Y);

  y.SetSubVector(ess_tdof_list_, 0.0);
}

Operator &ElasticityOperator::GetGradient(const Vector &x) const {
  current_state = x;
  return *gradient;
}

void ElasticityOperator::GradientMult(const Vector &dX, Vector &Y) const {
  Vector *dX_p = const_cast<Vector *>(&dX);
  // x is the data for the gridfunction we want to linearize on
  ParGridFunction dx, y, cstate;
  dx.MakeRef(&h1_fes_, *dX_p);
  y.MakeRef(&h1_fes_, Y);
  cstate.MakeRef(&h1_fes_, current_state);

  // L-vector to E-vector
  h1_element_restriction_->Mult(dx, X_evec);
  h1_element_restriction_->Mult(cstate, cstate_evec);

  // Reset output vector
  Y_evec = 0.0;

  // Apply operator
  element_apply_gradient_kernel_wrapper(
      NE_, maps->B, maps->G, ir_->GetWeights(), geometric_factors_->J,
      geometric_factors_->detJ, X_evec, Y_evec, cstate_evec);

  // E-vector to L-vector
  h1_element_restriction_->MultTranspose(Y_evec, Y);

  y.SetSubVector(ess_tdof_list_, 0.0);
}
