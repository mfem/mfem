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

MFEM_HOST_DEVICE static constexpr auto I = Identity<3>();

template <int dim>
struct LinearElasticMaterial
{
  tensor<double, dim, dim> MFEM_HOST_DEVICE stress(const tensor<double, dim, dim> &dudx) const
  {
    auto epsilon = sym(dudx);
    return lambda * tr(epsilon) * I + 2.0 * mu * epsilon;
  }

  tensor<double, dim, dim> MFEM_HOST_DEVICE
  action_of_gradient(const tensor<double, dim, dim> & /* dudx */,
                     const tensor<double, dim, dim> &ddudx) const
  {
    return stress(ddudx);
  }

  tensor<double, dim, dim, dim, dim> MFEM_HOST_DEVICE
  gradient(tensor<double, dim, dim> du_dx) const
  {
    return make_tensor<dim, dim, dim, dim>([&](auto i, auto j, auto k, auto l)
                                           { return lambda * (i == j) * (k == l) + mu * ((i == l) * (j == k) + (i == k) * (j == l)); });
  }

  double mu = 50;
  double lambda = 100;
};

template <int dim>
struct NeoHookeanMaterial
{
  static_assert(dim == 3, "NeoHookean model only defined in 3D");

  template <typename T>
  MFEM_HOST_DEVICE
      tensor<T, dim, dim>
      stress(const tensor<T, dim, dim> &__restrict__ du_dx) const
  {
    T J = det(I + du_dx);
    T p = -2.0 * D1 * J * (J - 1);
    auto devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));

    auto sigma = -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;

    // auto sigma = -(p / J) * I + 2 * (C1 / pow(std::cbrt(J), 5.0)) * devB;

    return sigma;
  }

  MFEM_HOST_DEVICE static void stress_wrapper(NeoHookeanMaterial<dim> *self,
                                              tensor<double, dim, dim> &du_dx,
                                              tensor<double, dim, dim> &sigma)
  {
    sigma = self->stress(du_dx);
  }

  MFEM_HOST_DEVICE tensor<double, dim, dim, dim, dim>
  gradient(tensor<double, dim, dim> du_dx) const
  {
    tensor<double, dim, dim> F = I + du_dx;
    tensor<double, dim, dim> invF = inv(F);
    tensor<double, dim, dim> devB = dev(du_dx + transpose(du_dx) + dot(du_dx, transpose(du_dx)));
    double J = det(F);
    double coef = (C1 / pow(J, 5.0 / 3.0));
    // clang-format off
    return make_tensor<3, 3, 3, 3>([&](auto i, auto j, auto k, auto l) {
      return 2.0 * (D1 * J * (i == j) - (5.0 / 3.0) * coef * devB[i][j]) * invF[l][k] +
             2.0 * coef * ((i == k) * F[j][l] + F[i][l] * (j == k) - (2.0 / 3.0) * ((i == j) * F[k][l]));
    });
    // clang-format on
  }

  MFEM_HOST_DEVICE tensor<double, dim, dim>
  action_of_gradient(const tensor<double, dim, dim> &dudx,
                     const tensor<double, dim, dim> &ddudx) const
  {
    // return action_of_gradient_enzyme_fwd(dudx, ddudx);
    // return action_of_gradient_enzyme_rev(dudx, ddudx);
    // return action_of_gradient_fd(dudx, ddudx);
    return action_of_gradient_symbolic(dudx, ddudx);
    // return action_of_gradient_dual(dudx, ddudx);
  }

  MFEM_HOST_DEVICE tensor<double, dim, dim>
  action_of_gradient_dual(const tensor<double, dim, dim> &dudx,
                          const tensor<double, dim, dim> &ddudx) const
  {
    tensor<dual<double>, dim, dim> dudx_and_ddudx;
    for (int i = 0; i < dim; i++)
    {
      for (int j = 0; j < dim; j++)
      {
        dudx_and_ddudx[i][j].value = dudx[i][j];
        dudx_and_ddudx[i][j].gradient = ddudx[i][j];
      }
    }
    return get_gradient(stress(dudx_and_ddudx));
  }

  MFEM_HOST_DEVICE tensor<double, dim, dim>
  action_of_gradient_enzyme_fwd(const tensor<double, dim, dim> &dudx,
                                const tensor<double, dim, dim> &ddudx) const
  {
    tensor<double, dim, dim> sigma{};
    tensor<double, dim, dim> dsigma{};

    __enzyme_fwddiff<void>(stress_wrapper, enzyme_const, this, enzyme_dup,
                           &dudx, &ddudx, enzyme_dupnoneed, &sigma, &dsigma);
    return dsigma;
  }

  MFEM_HOST_DEVICE tensor<double, dim, dim>
  action_of_gradient_enzyme_rev(const tensor<double, dim, dim> &dudx,
                                const tensor<double, dim, dim> &ddudx) const
  {
    tensor<double, dim, dim, dim, dim> gradient{};
    tensor<double, dim, dim> sigma{};
    tensor<double, dim, dim> dir{};

    for (int i = 0; i < dim; i++)
    {
      for (int j = 0; j < dim; j++)
      {
        dir[i][j] = 1;
        __enzyme_autodiff<void>(stress_wrapper, enzyme_const, this, enzyme_dup,
                                &dudx, &gradient[i][j], enzyme_dupnoneed,
                                &sigma, &dir);
        dir[i][j] = 0;
      }
    }
    return ddot(gradient, ddudx);
  }

  MFEM_HOST_DEVICE tensor<double, dim, dim>
  action_of_gradient_fd(const tensor<double, dim, dim> &dudx,
                        const tensor<double, dim, dim> &ddudx) const
  {
    return (stress(dudx + 1.0e-8 * ddudx) - stress(dudx - 1.0e-8 * ddudx)) /
           2.0e-8;
  }

  // d(stress)_{ij} := (d(stress)_ij / d(du_dx)_{kl}) * d(du_dx)_{kl}
  // Only works with 3D stress
  MFEM_HOST_DEVICE tensor<double, dim, dim>
  action_of_gradient_symbolic(const tensor<double, dim, dim> &du_dx,
                              const tensor<double, dim, dim> &ddu_dx) const
  {
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

class ElasticityGradientOperator;

class ElasticityOperator : public Operator
{
public:
  ElasticityOperator(ParMesh &mesh, const int order);

  virtual void Mult(const Vector &X, Vector &Y) const override;

  Operator &GetGradient(const Vector &x) const override;

  void SetState(const Vector &x);

  void GradientMult(const Vector &X, Vector &Y) const;

  void AssembleGradientDiagonal(Vector &Ke_diag, Vector &K_diag_local, Vector &K_diag) const;

  ~ElasticityOperator();

  ParMesh &mesh_;
  const int order_;
  const int DIM_;
  const int VDIM_;
  const int NE_;
  H1_FECollection h1_fec_;
  ParFiniteElementSpace h1_fes_;
  IntegrationRule *ir_ = nullptr;
  int Ndofs1d_;
  int Nq1d_;
  const Operator *h1_element_restriction_;
  const Operator *h1_prolongation_;
  Array<int> ess_tdof_list_;
  Array<int> displaced_tdof_list_;
  ElasticityGradientOperator *gradient;
  const GeometricFactors *geometric_factors_;
  const DofToQuad *maps;
  // State E-vector
  mutable Vector dX_ess, X_local, X_el, Y_local, Y_el, current_state, cstate_local,
      cstate_el;

  std::function<void(const int, const Array<double> &, const Array<double> &,
                     const Array<double> &, const Vector &, const Vector &,
                     const Vector &, Vector &)>
      element_kernel_wrapper;

  std::function<void(const int, const Array<double> &, const Array<double> &,
                     const Array<double> &, const Vector &, const Vector &,
                     const Vector &, Vector &, const Vector &)>
      element_apply_gradient_kernel_wrapper;

  std::function<void(const int, const Array<double> &, const Array<double> &,
                     const Array<double> &, const Vector &, const Vector &,
                     const Vector &, Vector &)>
      element_kernel_assemble_diagonal_wrapper;

  template <typename material_type>
  void SetMaterial(const material_type &material)
  {
    if (DIM_ != 3)
    {
      MFEM_ABORT("dim != 3 not implemented");
    }

    element_kernel_wrapper =
        [=](const int NE, const Array<double> &B_, const Array<double> &G_,
            const Array<double> &W_, const Vector &Jacobian_,
            const Vector &detJ_, const Vector &X_, Vector &Y_)
    {
      const int id = (Ndofs1d_ << 4) | Nq1d_;
      switch (id)
      {
      case 0x22:
        Apply3D<2, 2, material_type>(NE, B_, G_, W_, Jacobian_, detJ_, X_,
                                     Y_, material);
        break;
      case 0x33:
        Apply3D<3, 3, material_type>(NE, B_, G_, W_, Jacobian_, detJ_, X_,
                                     Y_, material);
        break;
      default:
        MFEM_ABORT("not implemented");
      }
    };

    element_apply_gradient_kernel_wrapper =
        [=](const int NE, const Array<double> &B_, const Array<double> &G_,
            const Array<double> &W_, const Vector &Jacobian_,
            const Vector &detJ_, const Vector &dU_, Vector &dF_,
            const Vector &U_)
    {
      const int id = (Ndofs1d_ << 4) | Nq1d_;
      switch (id)
      {
      case 0x22:
        ApplyGradient3D<2, 2, material_type>(NE, B_, G_, W_, Jacobian_,
                                             detJ_, dU_, dF_, U_, material);
        break;
      case 0x33:
        ApplyGradient3D<3, 3, material_type>(NE, B_, G_, W_, Jacobian_,
                                             detJ_, dU_, dF_, U_, material);
        break;
      default:
        MFEM_ABORT("not implemented");
      }
    };

    element_kernel_assemble_diagonal_wrapper =
        [=](const int NE, const Array<double> &B_, const Array<double> &G_,
            const Array<double> &W_, const Vector &Jacobian_,
            const Vector &detJ_, const Vector &X_, Vector &Y_)
    {
      const int id = (Ndofs1d_ << 4) | Nq1d_;
      switch (id)
      {
      case 0x22:
        AssembleGradientDiagonal3D<2, 2, material_type>(
            NE, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
        break;
      case 0x33:
        AssembleGradientDiagonal3D<3, 3, material_type>(
            NE, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
        break;
      default:
        MFEM_ABORT("not implemented");
      }
    };
  }

  void SetEssentialAttributes(const Array<int> attr)
  {
    h1_fes_.GetEssentialTrueDofs(attr, ess_tdof_list_);
  }

  void SetDisplacedAttributes(const Array<int> attr)
  {
    h1_fes_.GetEssentialTrueDofs(attr, displaced_tdof_list_);
  }

  const Array<int> GetDisplacedTDofs() { return displaced_tdof_list_; };

  // DeviceTensor<2> means RANK=2
  // Multi-component gradient evaluation from DOFs to quadrature points in
  // reference coordinates.
  template <int DIM, int D1D, int Q1D>
  static inline void MFEM_HOST_DEVICE
  CalcGrad(const DeviceTensor<2, const double> &B, // Q1D x D1D
           const DeviceTensor<2, const double> &G, // Q1D x D1D
           const DeviceTensor<4, const double> &U, // D1D x D1D x D1D x DIM
           tensor<double, Q1D, Q1D, Q1D, DIM, DIM> &dUdxi)
  {
    for (int c = 0; c < DIM; ++c)
    {
      for (int dz = 0; dz < D1D; ++dz)
      {
        tensor<double, Q1D, Q1D, DIM> gradXY{};
        for (int dy = 0; dy < D1D; ++dy)
        {
          tensor<double, Q1D, 2> gradX{};
          for (int dx = 0; dx < D1D; ++dx)
          {
            const double s = U(dx, dy, dz, c);
            for (int qx = 0; qx < Q1D; ++qx)
            {
              gradX[qx][0] += s * B(qx, dx);
              gradX[qx][1] += s * G(qx, dx);
            }
          }
          for (int qy = 0; qy < Q1D; ++qy)
          {
            const double wy = B(qy, dy);
            const double wDy = G(qy, dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
              const double wx = gradX[qx][0];
              const double wDx = gradX[qx][1];
              gradXY[qy][qx][0] += wDx * wy;
              gradXY[qy][qx][1] += wx * wDy;
              gradXY[qy][qx][2] += wx * wy;
            }
          }
        }
        for (int qz = 0; qz < Q1D; ++qz)
        {
          const double wz = B(qz, dz);
          const double wDz = G(qz, dz);
          for (int qy = 0; qy < Q1D; ++qy)
          {
            for (int qx = 0; qx < Q1D; ++qx)
            {
              dUdxi[qz][qy][qx][c][0] += gradXY[qy][qx][0] * wz;
              dUdxi[qz][qy][qx][c][1] += gradXY[qy][qx][1] * wz;
              dUdxi[qz][qy][qx][c][2] += gradXY[qy][qx][2] * wDz;
            }
          }
        }
      }
    }
  }

  // DeviceTensor<2> means RANK=2
  // Multi-component transpose gradient evaluation from DOFs to quadrature
  // points in reference coordinates with contraction of the D vector.
  template <int DIM, int D1D, int Q1D>
  MFEM_HOST_DEVICE static inline void CalcGradTSum(
      const DeviceTensor<2, const double> &B,           // Q1D x D1D
      const DeviceTensor<2, const double> &G,           // Q1D x D1D
      const tensor<double, Q1D, Q1D, Q1D, DIM, DIM> &U, // Q1D x Q1D x Q1D x DIM
      DeviceTensor<4, double> &F)                       // D1D x D1D x D1D x DIM
  {
    for (int c = 0; c < DIM; ++c)
    {
      for (int qz = 0; qz < Q1D; ++qz)
      {
        tensor<double, D1D, D1D, DIM> gradXY{};
        for (int qy = 0; qy < Q1D; ++qy)
        {
          tensor<double, D1D, DIM> gradX{};
          for (int qx = 0; qx < Q1D; ++qx)
          {
            const double gX = U[qx][qy][qz][0][c];
            const double gY = U[qx][qy][qz][1][c];
            const double gZ = U[qx][qy][qz][2][c];
            for (int dx = 0; dx < D1D; ++dx)
            {
              const double wx = B(qx, dx);
              const double wDx = G(qx, dx);
              gradX[dx][0] += gX * wDx;
              gradX[dx][1] += gY * wx;
              gradX[dx][2] += gZ * wx;
            }
          }
          for (int dy = 0; dy < D1D; ++dy)
          {
            const double wy = B(qy, dy);
            const double wDy = G(qy, dy);
            for (int dx = 0; dx < D1D; ++dx)
            {
              gradXY[dy][dx][0] += gradX[dx][0] * wy;
              gradXY[dy][dx][1] += gradX[dx][1] * wDy;
              gradXY[dy][dx][2] += gradX[dx][2] * wy;
            }
          }
        }
        for (int dz = 0; dz < D1D; ++dz)
        {
          const double wz = B(qz, dz);
          const double wDz = G(qz, dz);
          for (int dy = 0; dy < D1D; ++dy)
          {
            for (int dx = 0; dx < D1D; ++dx)
            {
              F(dx, dy, dz, c) +=
                  ((gradXY[dy][dx][0] * wz) + (gradXY[dy][dx][1] * wz) +
                   (gradXY[dy][dx][2] * wDz));
            }
          }
        }
      }
    }
  }

  template <int DIM, int D1D, int Q1D>
  MFEM_HOST_DEVICE static inline tensor<double, D1D, D1D, D1D, DIM>
  gradient_of_all_shape_functions(int qx, int qy, int qz, const DeviceTensor<2, const double> &B,
                                  const DeviceTensor<2, const double> &G, const tensor<double, DIM, DIM> &invJ)
  {
    MFEM_SHARED tensor<double, D1D, D1D, D1D, DIM> dphi_dx;
    // G (x) B (x) B
    // B (x) G (x) B
    // B (x) B (x) G
    MFEM_FOREACH_THREAD(dx, x, D1D)
    {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
        MFEM_FOREACH_THREAD(dz, z, D1D)
        {

          dphi_dx[dx][dy][dz] = transpose(invJ) * tensor<double, DIM>{
                                                      G(qx, dx) * B(qy, dy) * B(qz, dz),
                                                      B(qx, dx) * G(qy, dy) * B(qz, dz),
                                                      B(qx, dx) * B(qy, dy) * G(qz, dz)};
        }
      }
    }
    MFEM_SYNC_THREAD;
    return dphi_dx;
  }

  template <int D1D, int Q1D, typename material_type>
  static inline void
  ApplyGradient3D(const int NE, const Array<double> &B_,
                  const Array<double> &G_, const Array<double> &W_,
                  const Vector &Jacobian_, const Vector &detJ_,
                  const Vector &dU_, Vector &dF_, const Vector &U_,
                  const material_type &material)
  {
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

    MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D, {
      // cauchy stress
      MFEM_SHARED tensor<double, Q1D, Q1D, Q1D, DIM, DIM> invJ_dsigma_detJw;

      // du/dxi
      tensor<double, Q1D, Q1D, Q1D, DIM, DIM> dudxi{};
      const auto U_el = Reshape(&U(0, 0, 0, 0, e), D1D, D1D, D1D, DIM);
      CalcGrad<DIM, D1D, Q1D>(B, G, U_el, dudxi);

      // ddu/dxi
      tensor<double, Q1D, Q1D, Q1D, DIM, DIM> ddudxi{};
      const auto dU_el = Reshape(&dU(0, 0, 0, 0, e), D1D, D1D, D1D, DIM);
      CalcGrad<DIM, D1D, Q1D>(B, G, dU_el, ddudxi);

      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
        MFEM_FOREACH_THREAD(qy, y, Q1D)
        {
          MFEM_FOREACH_THREAD(qz, z, Q1D)
          {
            auto invJqp = inv(make_tensor<DIM, DIM>(
                [&](int i, int j)
                { return J(qx, qy, qz, i, j, e); }));

            auto dudx = dudxi(qz, qy, qx) * invJqp;
            auto ddudx = ddudxi(qz, qy, qx) * invJqp;

            auto dsigma = material.action_of_gradient(dudx, ddudx);

            invJ_dsigma_detJw(qx, qy, qz) =
                invJqp * dsigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
          }
        }
      }
      MFEM_SYNC_THREAD;
      auto F = Reshape(&force(0, 0, 0, 0, e), D1D, D1D, D1D, DIM);
      CalcGradTSum<DIM, D1D, Q1D>(B, G, invJ_dsigma_detJw, F);
    }); // for each element
  }

  template <int D1D, int Q1D, typename material_type>
  static inline void
  Apply3D(const int NE, const Array<double> &B_, const Array<double> &G_,
          const Array<double> &W_, const Vector &Jacobian_, const Vector &detJ_,
          const Vector &X_, Vector &Y_, const material_type &material)
  {
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

    MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D, {
      // cauchy stress
      MFEM_SHARED tensor<double, Q1D, Q1D, Q1D, DIM, DIM> invJ_sigma_detJw;

      // du/dxi
      tensor<double, Q1D, Q1D, Q1D, DIM, DIM> dudxi{};
      const auto U_el = Reshape(&U(0, 0, 0, 0, e), D1D, D1D, D1D, DIM);
      CalcGrad<DIM, D1D, Q1D>(B, G, U_el, dudxi);

      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
        MFEM_FOREACH_THREAD(qy, y, Q1D)
        {
          MFEM_FOREACH_THREAD(qz, z, Q1D)
          {
            auto invJqp = inv(make_tensor<DIM, DIM>(
                [&](int i, int j)
                { return J(qx, qy, qz, i, j, e); }));

            auto dudx = dudxi(qz, qy, qx) * invJqp;

            auto sigma = material.stress(dudx);

            invJ_sigma_detJw(qx, qy, qz) =
                invJqp * sigma * detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
          }
        }
      }
      MFEM_SYNC_THREAD;
      auto F = Reshape(&force(0, 0, 0, 0, e), D1D, D1D, D1D, DIM);
      CalcGradTSum<DIM, D1D, Q1D>(B, G, invJ_sigma_detJw, F);
    }); // for each element
  }

  template <int D1D, int Q1D, typename material_type>
  static inline void AssembleGradientDiagonal3D(
      const int NE, const Array<double> &B_, const Array<double> &G_,
      const Array<double> &W_, const Vector &Jacobian_, const Vector &detJ_,
      const Vector &X_, Vector &Ke_diag_memory, const material_type &material)
  {
    constexpr int dim = 3;

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
    const auto J = Reshape(Jacobian_.Read(), Q1D, Q1D, Q1D, dim, dim, NE);
    const auto detJ = Reshape(detJ_.Read(), Q1D, Q1D, Q1D, NE);
    // Input vector X_
    // ndofs1d x ndofs1d x VDIM x NE
    const auto U = Reshape(X_.Read(), D1D, D1D, D1D, dim, NE);
    // Output vector Y_
    // ndofs1d x ndofs1d x VDIM x NE
    auto Ke_diag_m = Reshape(Ke_diag_memory.ReadWrite(), D1D, D1D, D1D, dim, NE, dim);

    MFEM_FORALL(e, NE, {
      tensor<double, D1D, D1D, D1D, dim, dim> Ke_diag{};

      // du/dxi
      tensor<double, Q1D, Q1D, Q1D, dim, dim> dudxi{};
      const auto U_el = Reshape(&U(0, 0, 0, 0, e), D1D, D1D, D1D, dim);
      CalcGrad<dim, D1D, Q1D>(B, G, U_el, dudxi);

      for (int qx = 0; qx < Q1D; qx++)
      {
        for (int qy = 0; qy < Q1D; qy++)
        {
          for (int qz = 0; qz < Q1D; qz++)
          {
            auto invJqp = inv(make_tensor<dim, dim>(
                [&](int i, int j)
                { return J(qx, qy, qz, i, j, e); }));

            auto dudx = dudxi(qz, qy, qx) * invJqp;

            auto dsigma_ddudx = material.gradient(dudx);

            double JxW = detJ(qx, qy, qz, e) * qweights(qx, qy, qz);
            auto dNdx = gradient_of_all_shape_functions<dim, D1D, Q1D>(qx, qy, qz, B, G, invJqp);

            for (int dx = 0; dx < D1D; dx++)
            {
              for (int dy = 0; dy < D1D; dy++)
              {
                for (int dz = 0; dz < D1D; dz++)
                {
                  // phi_i * f(...) * phi_i
                  // dNdx_i dsigma_ddudx_ijkl dNdx_l
                  Ke_diag[dx][dy][dz] += (dNdx[dx][dy][dz] * dsigma_ddudx * dNdx[dx][dy][dz]) * JxW;
                }
              }
            }
          }
        }
      }
      for (int i = 0; i < D1D; i++)
      {
        for (int j = 0; j < D1D; j++)
        {
          for (int k = 0; k < D1D; k++)
          {
            for (int l = 0; l < dim; l++)
            {
              for (int m = 0; m < dim; m++)
              {
                Ke_diag_m(i, j, k, l, e, m) = Ke_diag[i][j][k][l][m];
              }
            }
          }
        }
      }
    }); // for each element
  }
};

class ElasticityGradientOperator : public Operator
{
public:
  ElasticityGradientOperator(ElasticityOperator &op);

  void AssembleGradientDiagonal(Vector &Ke_diag, Vector &K_diag_local, Vector &K_diag) const
  {
    elasticity_op_.AssembleGradientDiagonal(Ke_diag, K_diag_local, K_diag);
  }

  void Mult(const Vector &x, Vector &y) const override;

  ElasticityOperator &elasticity_op_;
};

class ElasticityDiagonalPreconditioner : public Solver
{
public:
  static constexpr int dim = 3;

  enum Type
  {
    Diagonal,
    BlockDiagonal
  };

  ElasticityDiagonalPreconditioner(Type type = Type::Diagonal) : Solver(), type_(type) {}

  void SetOperator(const Operator &op) override
  {
    gradient_operator_ = dynamic_cast<const ElasticityGradientOperator *>(&op);
    MFEM_ASSERT(gradient_operator_ != nullptr, "Operator is not ElasticityGradientOperator");

    width = height = op.Height();

    gradient_operator_->AssembleGradientDiagonal(Ke_diag, K_diag_local, K_diag);

    submat_height = gradient_operator_->elasticity_op_.h1_fes_.GetVDim();
    num_submats = gradient_operator_->elasticity_op_.h1_fes_.GetTrueVSize() /
                  gradient_operator_->elasticity_op_.h1_fes_.GetVDim();
  }

  void Mult(const Vector &x, Vector &y) const override
  {
    if (type_ == Type::Diagonal)
    {
      auto K_diag_submats = Reshape(K_diag.Read(), num_submats, submat_height, submat_height);
      // TODO: This could be MFEM_FORALL
      // Assuming Y and X are ordered byNODES. K_diag is ordered byVDIM.
      for (int s = 0; s < num_submats; s++)
      {
        for (int i = 0; i < submat_height; i++)
        {
          int idx = s + i * num_submats;
          y(idx) = x(idx) / K_diag_submats(s, i, i);
        }
      }
    }
    else if (type_ == Type::BlockDiagonal)
    {
      auto K_diag_submats = Reshape(K_diag.Read(), num_submats, submat_height, submat_height);

      for (int s = 0; s < num_submats; s++)
      {
        auto submat_inv = inv(make_tensor<dim, dim>([&](auto i, auto j)
                                                    { return K_diag_submats(s, i, j); }));

        auto x_block = make_tensor<dim>([&](auto i)
                                        { return x(s + i * num_submats); });

        tensor<double, dim> y_block;

        y_block = submat_inv * x_block;

        for (int i = 0; i < dim; i++)
        {
          int idx = s + i * num_submats;
          y(idx) = y_block(i);
        }
      }
    }
    else
    {
      MFEM_ABORT("Unknwon ElasticityDiagonalPreconditioner::Type");
    }
  }

private:
  const ElasticityGradientOperator *gradient_operator_;
  int num_submats, submat_height;
  Vector Ke_diag, K_diag_local, K_diag;
  Type type_;
};

int main(int argc, char *argv[])
{
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
  if (!args.Good())
  {
    if (myid == 0)
    {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (myid == 0)
  {
    args.PrintOptions(cout);
  }

  Device device(device_config);
  if (myid == 0)
  {
    device.Print();
  }

  auto mesh =
      Mesh::MakeCartesian3D(8, 1, 1, Element::HEXAHEDRON, 8.0, 1.0, 1.0);
  // auto mesh = Mesh()
  mesh.EnsureNodes();

  for (int l = 0; l < 3; l++)
  {
    mesh.UniformRefinement();
  }

  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();

  ElasticityOperator elasticity_op(pmesh, order);

  const NeoHookeanMaterial<3> material{};
  elasticity_op.SetMaterial(material);

  if (pmesh.bdr_attributes.Size())
  {
    Array<int> ess_attr(pmesh.bdr_attributes.Max());
    ess_attr = 0;
    ess_attr[4] = 1;
    ess_attr[2] = 1;
    elasticity_op.SetEssentialAttributes(ess_attr);
  }

  if (pmesh.bdr_attributes.Size())
  {
    Array<int> displaced_attr(pmesh.bdr_attributes.Max());
    displaced_attr = 0;
    displaced_attr[2] = 1;
    elasticity_op.SetDisplacedAttributes(displaced_attr);
  }

  ParGridFunction U_gf(&elasticity_op.h1_fes_);
  U_gf = 0.0;

  Vector U;
  U_gf.GetTrueDofs(U);

  // Assign load
  U.SetSubVector(elasticity_op.GetDisplacedTDofs(), 1.0e-2);

  ElasticityDiagonalPreconditioner diagonal_pc(ElasticityDiagonalPreconditioner::Type::Diagonal);

  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(1e-1);
  cg.SetMaxIter(10000);
  cg.SetPrintLevel(2);
  // cg.SetPreconditioner(diagonal_pc);

  NewtonSolver newton(MPI_COMM_WORLD);
  newton.SetSolver(cg);
  newton.SetOperator(elasticity_op);
  newton.SetRelTol(1e-6);
  newton.SetMaxIter(10);
  newton.SetPrintLevel(1);

  Vector zero;
  newton.Mult(zero, U);

  U_gf.Distribute(U);

  ParaViewDataCollection *pd = NULL;
  pd = new ParaViewDataCollection("elast", &pmesh);
  pd->RegisterField("solution", &U_gf);
  pd->SetLevelsOfDetail(order);
  pd->SetDataFormat(VTKFormat::BINARY);
  pd->SetHighOrderOutput(true);
  pd->SetCycle(0);
  pd->SetTime(0.0);
  pd->Save();

  delete pd;

  return 0;
}

ElasticityGradientOperator::ElasticityGradientOperator(ElasticityOperator &op)
    : Operator(op.Height()), elasticity_op_(op) {}

void ElasticityGradientOperator::Mult(const Vector &x, Vector &y) const
{
  elasticity_op_.GradientMult(x, y);
}

ElasticityOperator::ElasticityOperator(ParMesh &mesh, const int order)
    : Operator(), mesh_(mesh), order_(order), DIM_(mesh_.SpaceDimension()), VDIM_(mesh_.SpaceDimension()),
      NE_(mesh_.GetNE()), h1_fec_(order_, DIM_), h1_fes_(&mesh_, &h1_fec_, VDIM_, Ordering::byNODES)
{
  this->height = h1_fes_.GetTrueVSize();
  this->width = this->height;

  int global_tdof_size = h1_fes_.GlobalTrueVSize();
  if (mesh.GetMyRank() == 0)
  {
    cout << "#dofs: " << global_tdof_size << endl;
  }

  h1_element_restriction_ = h1_fes_.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
  h1_prolongation_ = h1_fes_.GetProlongationMatrix();

  ir_ = const_cast<IntegrationRule *>(
      &IntRules.Get(mfem::Element::HEXAHEDRON, 2 * h1_fes_.GetOrder(0) + 1));

  geometric_factors_ = h1_fes_.GetParMesh()->GetGeometricFactors(
      *ir_, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
  maps = &h1_fes_.GetFE(0)->GetDofToQuad(*ir_, DofToQuad::TENSOR);
  Ndofs1d_ = maps->ndof;
  Nq1d_ = maps->nqpt;

  dX_ess.UseDevice(true);
  dX_ess.SetSize(h1_fes_.GetTrueVSize());

  X_el.UseDevice(true);
  X_el.SetSize(h1_element_restriction_->Height());

  Y_el.UseDevice(true);
  Y_el.SetSize(h1_element_restriction_->Height());

  cstate_el.UseDevice(true);
  cstate_el.SetSize(h1_element_restriction_->Height());

  X_local.UseDevice(true);
  X_local.SetSize(h1_prolongation_->Height());

  Y_local.UseDevice(true);
  Y_local.SetSize(h1_prolongation_->Height());

  cstate_local.UseDevice(true);
  cstate_local.SetSize(h1_prolongation_->Height());

  gradient = new ElasticityGradientOperator(*this);
}

void ElasticityOperator::Mult(const Vector &X, Vector &Y) const
{
  ess_tdof_list_.Read();

  // T-vector to L-vector
  h1_prolongation_->Mult(X, X_local);
  // L-vector to E-vector
  h1_element_restriction_->Mult(X_local, X_el);

  // Reset output vector
  Y_el = 0.0;

  // Apply operator
  element_kernel_wrapper(NE_, maps->B, maps->G, ir_->GetWeights(),
                         geometric_factors_->J, geometric_factors_->detJ, X_el,
                         Y_el);

  // E-vector to L-vector
  h1_element_restriction_->MultTranspose(Y_el, Y_local);
  // L-vector to T-vector
  h1_prolongation_->MultTranspose(Y_local, Y);

  // Set the residual at Dirichlet dofs on the T-vector to zero
  Y.SetSubVector(ess_tdof_list_, 0.0);
}

Operator &ElasticityOperator::GetGradient(const Vector &x) const
{
  h1_prolongation_->Mult(x, cstate_local);
  h1_element_restriction_->Mult(cstate_local, cstate_el);
  return *gradient;
}

void ElasticityOperator::GradientMult(const Vector &dX, Vector &Y) const
{
  ess_tdof_list_.Read();

  // Column elimination for essential dofs
  dX_ess = dX;
  dX_ess.SetSubVector(ess_tdof_list_, 0.0);

  // T-vector to L-vector
  h1_prolongation_->Mult(dX_ess, X_local);
  // L-vector to E-vector
  h1_element_restriction_->Mult(X_local, X_el);

  // Reset output vector
  Y_el = 0.0;

  // Apply operator
  element_apply_gradient_kernel_wrapper(
      NE_, maps->B, maps->G, ir_->GetWeights(), geometric_factors_->J,
      geometric_factors_->detJ, X_el, Y_el, cstate_el);

  // E-vector to L-vector
  h1_element_restriction_->MultTranspose(Y_el, Y_local);
  // L-vector to T-vector
  h1_prolongation_->MultTranspose(Y_local, Y);

  {
    const auto d_dX = dX.Read();
    auto d_Y = Y.ReadWrite();
    const auto d_ess_tdof_list = ess_tdof_list_.Read();
    MFEM_FORALL(i, ess_tdof_list_.Size(),
                d_Y[d_ess_tdof_list[i]] = d_dX[d_ess_tdof_list[i]];);
  }
}

void ElasticityOperator::AssembleGradientDiagonal(Vector &Ke_diag, Vector &K_diag_local, Vector &K_diag) const
{
  Ke_diag.SetSize(Ndofs1d_ * Ndofs1d_ * Ndofs1d_ * DIM_ * NE_ * DIM_);
  K_diag_local.SetSize(h1_element_restriction_->Width() * DIM_);
  K_diag.SetSize(h1_prolongation_->Width() * DIM_);

  element_kernel_assemble_diagonal_wrapper(
      NE_, maps->B, maps->G, ir_->GetWeights(), geometric_factors_->J,
      geometric_factors_->detJ, cstate_el, Ke_diag);

  for (int i = 0; i < DIM_; i++)
  {
    // Scalar component E-size
    int sce_sz = Ndofs1d_ * Ndofs1d_ * Ndofs1d_ * DIM_ * NE_;
    // Scalar component L-size
    int scl_sz = h1_element_restriction_->Width();

    Vector vin_local, vout_local;
    vin_local.MakeRef(Ke_diag, i * sce_sz, sce_sz);
    vout_local.MakeRef(K_diag_local, i * scl_sz, scl_sz);
    h1_element_restriction_->MultTranspose(vin_local, vout_local);

    // Scalar component T-size
    int sct_sz = h1_prolongation_->Width();
    Vector vout;
    vout.MakeRef(K_diag, i * sct_sz, sct_sz);
    h1_prolongation_->MultTranspose(vout_local, vout);
  }

  int num_submats = h1_fes_.GetTrueVSize() / h1_fes_.GetVDim();
  auto K_diag_submats = Reshape(K_diag.Write(), num_submats, DIM_, DIM_);
  for (int i = 0; i < ess_tdof_list_.Size(); i++)
  {
    int ess_idx = ess_tdof_list_[i];
    int submat = ess_idx % num_submats;
    int row = ess_idx / num_submats;
    for (int j = 0; j < DIM_; j++)
    {
      if (row == j)
      {
        K_diag_submats(submat, row, j) = 1.0;
      }
      else
      {
        K_diag_submats(submat, row, j) = 0.0;
        K_diag_submats(submat, j, row) = 0.0;
      }
    }
  }
}

ElasticityOperator::~ElasticityOperator() { delete gradient; }