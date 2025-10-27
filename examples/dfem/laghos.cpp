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
//
// implicit 3point:
//   ./laghos -p 3 -pt 1 -s 12 -tf 5.0 -rs 1 -cfl 16 -nmi 50 -dump-jacobians 0 -nretry 50000 -kmi 500 -av -av-type 7 -ov 2 -oe 1
//

#include <mfem.hpp>

// TODO: Do we want this to be included from mfem.hpp automatically now?
#include "fem/dfem/doperator.hpp"
#include "linalg/tensor.hpp"

#include <limits>
#include <memory>
#include <string>

using namespace mfem;
using mfem::internal::tensor;

constexpr int VELOCITY = 0;
constexpr int DENSITY0 = 1;
constexpr int COORDINATES0 = 2;
constexpr int COORDINATES = 3;
constexpr int MATERIAL = 4;
constexpr int SPECIFIC_INTERNAL_ENERGY = 5;
constexpr int DT_EST = 8;
constexpr int STRESS_TENSOR = 9;

constexpr int DIMENSION = 2;

enum EXT_DATA_IDX
{
   CFL = 0,
   ORDER_VEL,
   VISCOSITY_FLAG,
   VISCOSITY_TYPE,
   H0,
   DT_ESTIMATE,
   COUNT
};

int problem = 0;

enum PRECONDITIONER_TYPE
{
   SUPERLU,
   BLOCK_DIAGONAL_AMG,
};

void threshold(Vector &v)
{
   for (int i = 0; i < v.Size(); i++)
   {
      if (abs(v(i)) <= 1e-12)
      {
         v(i) = 0.0;
      }
   }
}

MFEM_HOST_DEVICE inline
real_t taylor_source(const Vector &x)
{
   return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                               cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
};

MFEM_HOST_DEVICE inline
real_t smoothmin(real_t a, real_t b, real_t k = 1e6)
{
   return -1.0 / k * log(exp(-k * a) + exp(-k * b));
}

MFEM_HOST_DEVICE inline
real_t smoothmax(real_t a, real_t b, real_t k = 1e-6)
{
   return 0.5 * (a + b + sqrt((a-b)*(a-b) + k*k));
}

MFEM_HOST_DEVICE inline
real_t smoothabs(real_t x, real_t k = 1e-6)
{
   return sqrt(x * x + k);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
tensor<T, n> shift(const tensor<T, n> &v, T s)
{
   tensor<T, n> sv;
   for (int i = 0; i < n; i++)
   {
      sv(i) = v(i) - s;
   }
   return sv;
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
T min(const tensor<T, n> &v)
{
   T min = std::numeric_limits<T>::min();
   for (int i = 0; i < n; i++)
   {
      if (v(i) < min)
      {
         min = v(i);
      }
   }
   return min;
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
std::tuple<T, tensor<T, n>> sinvpm(const tensor<T, n, n> &A, int maxit, T tol)
{
   auto shift = [](const tensor<T, n, n> &A, const T& mu)
   {
      const auto I = mfem::internal::Identity<n>();
      tensor<T, n, n> B = A;
      B -= mu * I;
      return B;
   };

   auto As = shift(A, tol);
   auto mu = min(std::get<0>(eig(As)));
   tensor<T, n> x = {};
   x(0) = 1.0;
   x = x / norm(x);
   const auto Binv = inv(shift(A, mu));
   auto y = Binv * x;
   auto la = dot(y, x);

   for (int i = 0; i < maxit; i++)
   {
      // const auto err = norm(y - la * x) / norm(y);
      // if (err <= tol)
      // {
      //    break;
      // }
      x = y / norm(y);
      y = Binv * x;
      la = dot(y, x);
   }

   return {mu + 1.0 / la, x};
}

// Smooth transition between 0 and 1 for x in [-eps, eps].
MFEM_HOST_DEVICE inline
real_t smooth_step_01(real_t x, real_t eps)
{
   const real_t y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

MFEM_HOST_DEVICE inline
void ComputeMaterialProperties(const real_t &gamma, const real_t &rho,
                               const real_t &E, real_t &p, real_t &cs)
{
   p = (gamma - 1.0) * rho * E;
   cs = sqrt(gamma * (gamma - 1.0) * E);
}

using vecd = tensor<real_t, DIMENSION>;
using matd = tensor<real_t, DIMENSION, DIMENSION>;

struct TaylorSourceQFunction
{
   TaylorSourceQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const vecd &x,
      const matd &J,
      const real_t &w) const
   {
      auto f = 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                    cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );

      return mfem::tuple{f * det(J) * w};
   }
};


static inline MFEM_HOST_DEVICE
std::tuple<tensor<real_t, 2>, tensor<real_t, 2, 2>> eig2(
                                                    tensor<real_t, 2, 2> &A)
{
   return eig(A);
}

static inline MFEM_HOST_DEVICE
std::tuple<tensor<real_t, 2>, tensor<real_t, 2, 2>> grad_eig2(
                                                    tensor<real_t, 2, 2> &A,
                                                    tensor<real_t, 2, 2> &dA)
{
   return grad_eig(A, dA);
}

void* __enzyme_register_derivative_eig[] =
{
   (std::tuple<tensor<real_t, 2>, tensor<real_t, 2, 2>>*)eig2,
   (std::tuple<tensor<real_t, 2>, tensor<real_t, 2, 2>>*)grad_eig2,
};

MFEM_HOST_DEVICE inline
matd qdata_setup(
   const matd &dvdxi,
   const real_t &rho0,
   const matd &J0,
   const matd &J,
   const real_t &gamma,
   const real_t &E,
   const real_t &w,
   const real_t &h0,
   const real_t &order_v,
   const real_t &cfl,
   const bool &use_viscosity,
   const int &viscosity_type,
   real_t &dt_est)
{
   constexpr real_t eps = 1e-12;
   constexpr real_t vorticity_coeff = 1.0;
   real_t p, cs;
   const real_t detJ = det(J);
   const matd invJ = inv(J);
   matd stress{{{0.0}}};
   const real_t rho = rho0 * det(J0) / detJ;
   const real_t Ez = smoothmax(0.0, E);
   real_t dt_visc_coeff = 0.0;

   ComputeMaterialProperties(gamma, rho, Ez, p, cs);

   for (int d = 0; d < DIMENSION; d++)
   {
      stress(d, d) = -p;
   }

   // out << "qpinfo: " << rho << " " << Ez << " " << p << " " << cs << "\n";
   // out << "Ez: " << E << "\n";

   if (use_viscosity)
   {
      auto softstep = [](const real_t &eps, const real_t x)
      {
         auto arg = std::max(std::min(x / (eps + 1e-6), 40.0),
                             -40.0); // [-40,40] covers most practical values
         return 1.0 / (1.0 + std::exp(-arg));
      };

      auto softabs = [=](const real_t &eps, const real_t x)
      {
         // Clamp eps to avoid division/overflow, and to promote smoothing for AD
         auto e = std::max(eps, 1e-6_r);
         return sqrt(x * x + e * e);
      };

      if (viscosity_type == 2)
      {
         auto symdvdx = sym(dvdxi * invJ);
         auto [eigvals, eigvecs] = eig(symdvdx);

         const real_t mu = eigvals(0);
         vecd compr_dir = get_col(eigvecs, 0);
         auto ph_dir = (J * inv(J0)) * compr_dir;
         const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
         // Measure of maximal compression.
         auto visc_coeff = 2.0 * rho * h * h * fabs(mu);
         visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
                       (1.0 - softstep(eps, mu - 2.0 * eps));
         stress += visc_coeff * symdvdx;
         dt_visc_coeff = visc_coeff;
      }
      else if (viscosity_type == 21)
      {
         auto symdvdx = sym(dvdxi * invJ);
         auto [mu, compr_dir] = sinvpm(symdvdx, 10, 1e-12);

         auto ph_dir = (J * inv(J0)) * compr_dir;
         const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
         // Measure of maximal compression.
         auto visc_coeff = 2.0 * rho * h * h * softabs(1e-6, mu);
         visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
                       (1.0 - softstep(eps, mu - 2.0 * eps));
         stress += visc_coeff * symdvdx;
         dt_visc_coeff = visc_coeff;
      }
      else if (viscosity_type == 22)
      {
         const auto delta = 0.2 * cs;

         auto symdvdx = sym(dvdxi * invJ);
         auto [eigvals, eigvecs] = eig2(symdvdx);

         const real_t mu = eigvals(0);
         vecd compr_dir = get_col(eigvecs, 0);
         for (int i = 0; i < compr_dir.first_dim; i++)
         {
            compr_dir(i) = softstep(delta, compr_dir(i));
         }
         auto ph_dir = (J * inv(J0)) * compr_dir;
         const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
         // Measure of maximal compression.
         auto visc_coeff = 2.0 * rho * h * h * softabs(delta, mu);
         visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
                       (1.0 - softstep(eps, mu - 2.0 * eps));
         if (!std::isfinite(visc_coeff))
         {
            out << "err\n";
            exit(1);
         }
         stress += visc_coeff * symdvdx;
         dt_visc_coeff = visc_coeff;
      }
      else if (viscosity_type == 4)
      {
         auto symdvdx = sym(dvdxi * invJ);
         auto [lam, s] = eig(symdvdx);
         const auto delta = 0.2 * cs;

         for (int k = 0; k < DIMENSION; k++)
         {
            const auto ph_dir = (J * inv(J0)) * s(k);
            const real_t h = h0 * norm(ph_dir) / norm(s(k));
            auto visc_coeff = 2.0 * rho * h * h * softabs(delta, lam(k));
            visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
                          (1.0 - softstep(delta, lam(k) - 2.0 * eps));
            stress += visc_coeff * symdvdx;
            dt_visc_coeff += visc_coeff;
         }
      }
      else if (viscosity_type == 7)
      {
         const auto delta = 0.2 * cs;
         const auto dvdx = dvdxi * inv(J);
         const auto h = h0 * pow(det(J), 1.0 / DIMENSION);
         const auto delta_v = h * tr(dvdx);
         const auto psi1 = softstep(delta, -delta_v);
         const auto q1 = 5.0;
         const auto q2 = 5.0;
         const auto mu = 3.0 / 4.0 * rho * h * psi1 * (q2 * softabs(delta,
                                                                    delta_v) + q1 * cs);
         stress += 2.0 * mu * sym(dvdx);
         dt_visc_coeff = 2.0 * mu;
      }
      else
      {
         exit(1);
      }
   }

   if (rho < 0.0)
   {
      MFEM_ABORT("negative density on quadrature point \n" "detJ = " << detJ);
      exit(1);
   }

   if (detJ < 0.0)
   {
      // MFEM_ABORT("inverted element detected in qdata_setup");
      // This will force repetition of the step with smaller dt.
      dt_est = 0.0;
   }
   else
   {
      const real_t sv = calcsv(J, DIMENSION-1);
      // out << sv << ", ";
      const real_t hmin = sv / static_cast<real_t>(order_v);
      const real_t ihmin = 1.0 / hmin;
      const real_t irhoihminsq = ihmin * ihmin / rho;
      const real_t idt = cs / hmin + 2.5 * dt_visc_coeff * irhoihminsq;

      if (idt > 0.0)
      {
         dt_est = fmin(dt_est, cfl / idt);
      }
   }

   matd stressJiT = stress * transpose(invJ) * detJ * w;
   return stressJiT;
}

struct TimeStepEstimateQFunction
{
   TimeStepEstimateQFunction(real_t *external_data) :
      external_data(external_data) {}

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &w) const
   {
      // real_t dt_est = mfem::get<1>(
      //                    qdata_setup(
      //                       dvdxi, rho0, J0, J, gamma, E, w,
      //                       external_data[EXT_DATA_IDX::H0],
      //                       external_data[EXT_DATA_IDX::ORDER_VEL],
      //                       external_data[EXT_DATA_IDX::CFL],
      //                       static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
      //                       dt_est));
      out << " >>>> PANIC \n";
      exit(0);
      return mfem::tuple{external_data[EXT_DATA_IDX::DT_ESTIMATE]};
   }

   real_t *external_data;
};

struct UpdateQuadratureDataQFunction
{
   UpdateQuadratureDataQFunction(real_t *external_data) :
      external_data(external_data) {}

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &w) const
   {
      // out << "qdata update on qp\n";
      auto stressJiT =
         qdata_setup(
            dvdxi, rho0, J0, J, gamma, E, w,
            external_data[EXT_DATA_IDX::H0],
            external_data[EXT_DATA_IDX::ORDER_VEL],
            external_data[EXT_DATA_IDX::CFL],
            static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
            external_data[EXT_DATA_IDX::VISCOSITY_TYPE],
            external_data[EXT_DATA_IDX::DT_ESTIMATE]);
      return mfem::tuple{stressJiT};
   }

   real_t *external_data;
};

class MomentumQFunction
{
public:
   MomentumQFunction(real_t *external_data) :
      external_data(external_data) {}

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &w) const
   {
      auto stressJiT =
         qdata_setup(
            dvdxi, rho0, J0, J, gamma, E, w,
            external_data[EXT_DATA_IDX::H0],
            external_data[EXT_DATA_IDX::ORDER_VEL],
            external_data[EXT_DATA_IDX::CFL],
            static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
            external_data[EXT_DATA_IDX::VISCOSITY_TYPE],
            dt_est_dummy);
      return mfem::tuple{stressJiT};
   }

   mutable real_t dt_est_dummy = std::numeric_limits<real_t>::infinity();
   real_t *external_data;
};

class MomentumPAQFunction
{
public:
   MomentumPAQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &stressJiT) const
   {
      return mfem::tuple{stressJiT};
   }
};

class EnergyConservationQFunction
{
public:
   EnergyConservationQFunction(const real_t *external_data) :
      external_data(external_data) {}

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &w) const
   {
      auto stressJiT =
         qdata_setup(
            dvdxi, rho0, J0, J, gamma, E, w,
            external_data[EXT_DATA_IDX::H0],
            external_data[EXT_DATA_IDX::ORDER_VEL],
            external_data[EXT_DATA_IDX::CFL],
            static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
            external_data[EXT_DATA_IDX::VISCOSITY_TYPE],
            dt_est);
      return mfem::tuple{ddot(stressJiT, dvdxi)};
   }

   mutable real_t dt_est = std::numeric_limits<real_t>::infinity();
   const real_t *external_data;
};

class EnergyConservationPAQFunction
{
public:
   EnergyConservationPAQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const matd &stressJiT) const
   {
      return mfem::tuple{ddot(stressJiT, dvdxi)};
   }
};

class TotalInternalEnergyQFunction
{
public:
   TotalInternalEnergyQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator() (
      const real_t &E,
      const real_t &rho0,
      const matd &J0,
      const real_t &w) const
   {
      return mfem::tuple{rho0 * E * det(J0) * w};
   }
};

class TotalKineticEnergyQFunction
{
public:
   TotalKineticEnergyQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator() (
      const vecd &v,
      const real_t &rho0,
      const matd &J0,
      const real_t &w) const
   {
      return mfem::tuple{rho0 * 0.5 * v * v * det(J0) * w};
   }
};

class DensityQFunction
{
public:
   DensityQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator() (
      const real_t &rho0,
      const matd &J0,
      const real_t &w) const
   {
      return mfem::tuple{rho0 * det(J0) * w};
   }
};


struct QuadratureData
{
   static constexpr int aux_dim = 1;
   QuadratureData(const ParMesh &mesh, const IntegrationRule &ir) :
      StressSpace(mesh.Dimension(), mesh.Dimension()*mesh.Dimension(),
                  ir.GetNPoints(),
                  mesh.Dimension()*mesh.Dimension()*ir.GetNPoints()*mesh.GetNE()),
      stressp(StressSpace),
      R(mesh.Dimension(),
        aux_dim,
        ir.GetNPoints(),
        aux_dim*ir.GetNPoints()*mesh.GetNE()),
      dt_est(R)
   {
      dt_est.UseDevice(true);
      stressp.UseDevice(true);
   }

   ParametricSpace StressSpace;
   ParametricFunction stressp;

   ParametricSpace R;
   ParametricFunction dt_est;
};

class MassPAOperator : public Operator
{
public:
   MassPAOperator(ParFiniteElementSpace &pfes,
                  const IntegrationRule &ir,
                  Coefficient &Q) :
      Operator(pfes.GetTrueVSize()),
      comm(pfes.GetParMesh()->GetComm()),
      dim(pfes.GetMesh()->Dimension()),
      NE(pfes.GetMesh()->GetNE()),
      vsize(pfes.GetVSize()),
      pabf(&pfes),
      ess_tdofs_count(0),
      ess_tdofs(0)
   {
      if (dim > 1)
      {
         pabf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      pabf.AddDomainIntegrator(new mfem::MassIntegrator(Q, &ir));
      pabf.Assemble();
      pabf.FormSystemMatrix(mfem::Array<int>(), mass);
   }

   void SetEssentialTrueDofs(Array<int> &dofs)
   {
      ess_tdofs_count = dofs.Size();
      if (ess_tdofs.Size() == 0)
      {
         int ess_tdofs_sz;
         MPI_Allreduce(&ess_tdofs_count,&ess_tdofs_sz, 1, MPI_INT, MPI_SUM, comm);
         MFEM_ASSERT(ess_tdofs_sz > 0, "ess_tdofs_sz should be positive!");
         ess_tdofs.SetSize(ess_tdofs_sz);
      }
      if (ess_tdofs_count == 0) { return; }
      ess_tdofs = dofs;
   }

   void EliminateRHS(Vector &b) const
   {
      if (ess_tdofs_count > 0) { b.SetSubVector(ess_tdofs, 0.0); }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      mass->Mult(x, y);
      if (ess_tdofs_count > 0) { y.SetSubVector(ess_tdofs, 0.0); }
   }

   void FullAddMult(const Vector &x, Vector &y) const
   {
      mass->AddMult(x, y);
   }

   const ParBilinearForm &GetBF() const { return pabf; }

   const MPI_Comm comm;
   const int dim, NE, vsize;
   ParBilinearForm pabf;
   int ess_tdofs_count;
   Array<int> ess_tdofs;
   OperatorPtr mass;
};

class LagrangianHydroOperator : public TimeDependentOperator
{
public:

   class LineSearchNewtonSolver : public NewtonSolver
   {
   public:
      LineSearchNewtonSolver(MPI_Comm comm, const LagrangianHydroOperator &hydro)
         : NewtonSolver(comm),
           hydro(hydro),
           beta(0.5),    // Backtracking reduction factor
           alpha(1e-2),  // Sufficient decrease constant (Armijo condition)
           max_line_iter(5) {}

      void SetBacktrackingFactor(real_t b) { beta = b; }
      void SetArmijoConstant(real_t a) { alpha = a; }
      void SetMaxLineSearchIter(int n) { max_line_iter = n; }

   protected:
      void ProcessNewState(const Vector &x) const override
      {
         s_new.SetSize(x.Size());
         s_new = x;
         s_new *= hydro.residual->dt;
         s_new += hydro.residual->x;
         real_t min_detJ = hydro.ComputeMinDet(s_new);
         if (min_detJ <= 0.0)
         {
            out << "ProcessNewState negative detJ = " << min_detJ << "\n";
            illegal_state = true;
         }
         illegal_state = false;
      }

      // Override the scaling factor computation to implement line search
      real_t ComputeScalingFactor(const Vector &x, const Vector &b) const override
      {
         const bool have_b = (b.Size() == Height());
         real_t lambda = 1.0;  // Initial step length
         x_new.SetSize(x.Size());
         r_new.SetSize(x.Size());
         s_new.SetSize(x.Size());

         const real_t initial_norm = Norm(r);  // Current residual norm

         grad->Mult(c, r_new);
         const real_t grad_norm = Norm(r_new);     // Gradient norm

         for (int i = 0; i < max_line_iter; i++)
         {
            // Try step: x_new = x - lambda * c
            add(x, -lambda, c, x_new);

            // Before calling Mult on the operator we have to check mesh
            // validity by computing the minimum determinant of the mesh
            // Jacobian.
            s_new = x_new;
            s_new *= hydro.residual->dt;
            s_new += hydro.residual->x;
            real_t min_detJ = hydro.ComputeMinDet(s_new);
            if (min_detJ <= 0.0)
            {
               out << "linesearch found negative detJ = " << min_detJ << "\n";
               lambda *= beta;
               continue;
            }

            // Evaluate residual at new point
            oper->Mult(x_new, r_new);
            if (have_b)
            {
               subtract(r_new, b, r_new);
            }

            const real_t new_norm = Norm(r_new);

            if (new_norm <= initial_norm - alpha * lambda * grad_norm)
            {
               // Found acceptable step
               out << "linesearch lambda: " << lambda << "\n";
               return lambda;
            }

            // Backtrack
            lambda *= beta;
         }

         out << ">>> linesearch didn't converge\n";
         return 0.0;
      }

   private:
      real_t beta;        // Backtracking factor (how much to reduce step)
      real_t alpha;       // Sufficient decrease parameter
      int max_line_iter;  // Maximum line search iterations
      mutable Vector x_new, r_new, s_new;
      const LagrangianHydroOperator &hydro;
   };


   class LagrangianHydroJacobianOperator : public Operator
   {
   public:
      LagrangianHydroJacobianOperator(
         LagrangianHydroOperator &hydro,
         std::shared_ptr<DerivativeOperator> dRvdx,
         std::shared_ptr<DerivativeOperator> dRvdv,
         std::shared_ptr<DerivativeOperator> dRvde,
         std::shared_ptr<DerivativeOperator> dRedx,
         std::shared_ptr<DerivativeOperator> dRedv,
         std::shared_ptr<DerivativeOperator> dRede,
         std::shared_ptr<DerivativeOperator> dTaylorSourcedx,
         real_t h) :
         Operator(2*hydro.H1.GetTrueVSize() + hydro.L2.GetTrueVSize()),
         h(h),
         H1tsize(hydro.H1.GetTrueVSize()),
         L2tsize(hydro.L2.GetTrueVSize()),
         w(height),
         z(height),
         hydro(hydro),
         dRvdx(dRvdx),
         dRvdv(dRvdv),
         dRvde(dRvde),
         dRedx(dRedx),
         dRedv(dRedv),
         dRede(dRede),
         dTaylorSourcedx(dTaylorSourcedx)
      {}

      void Mult(const Vector &u, Vector &y) const override
      {
         w = u;
         Vector wx, wv, we;
         wx.MakeRef(w, 0, H1tsize);
         wv.MakeRef(w, H1tsize, H1tsize);
         we.MakeRef(w, 2*H1tsize, L2tsize);

         Vector zx, zv, ze;
         zx.MakeRef(z, 0, H1tsize);
         zv.MakeRef(z, H1tsize, H1tsize);
         ze.MakeRef(z, 2*H1tsize, L2tsize);

         Vector yx, yv, ye;
         yx.MakeRef(y, 0, H1tsize);
         yv.MakeRef(y, H1tsize, H1tsize);
         ye.MakeRef(y, 2*H1tsize, L2tsize);

         // position
         yx = wv;
         yx *= -h;
         yx += wx;

         // velocity
         // wv.SetSubVector(hydro.ess_tdof, 0.0);
         dRvdx->Mult(wx, zv);
         if (zv.CheckFinite() != 0)
         {
            pretty_print(zv);
         }
         MFEM_VERIFY(zv.CheckFinite() == 0, "err");
         zv *= h;
         yv = zv;
         dRvdv->Mult(wv, zv);
         MFEM_VERIFY(zv.CheckFinite() == 0, "err");
         zv *= h;
         yv += zv;
         // hydro.Mv.TrueAddMult(wv, yv);
         Vector wvc, yvc;
         for (int c = 0; c < hydro.H1.GetMesh()->Dimension(); c++)
         {
            wvc.MakeRef(wv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
            yvc.MakeRef(yv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
            hydro.Mv->FullAddMult(wvc, yvc);
            yvc.SyncAliasMemory(yv);
         }
         yv.SyncAliasMemory(y);

         dRvde->Mult(we, zv);
         MFEM_VERIFY(zv.CheckFinite() == 0, "err");
         zv *= h;
         yv += zv;
         yv.SetSubVector(hydro.ess_tdof, 0.0);

         // for (int i = 0; i < hydro.ess_tdof.Size(); i++)
         // {
         //    // yv(hydro.ess_tdof[i]) = uv(hydro.ess_tdof[i]);
         //    yv(hydro.ess_tdof[i]) = 0.0;
         // }
         // yv = 0.0;

         // energy
         //                          [ wx ]
         // [ dRe/dx dRe/dv dRe/de ] [ wv ]
         //                          [ we ]
         //

         dRedx->Mult(wx, ze);
         MFEM_VERIFY(ze.CheckFinite() == 0, "err");
         if (problem == 0)
         {
            // dTaylorSourcedx->AddMult(wx, ze);
            MFEM_VERIFY(ze.CheckFinite() == 0, "err");
         }
         ze *= -h;
         ye = ze;

         dRedv->Mult(wv, ze);
         MFEM_VERIFY(ze.CheckFinite() == 0, "err");
         ze *= -h;
         ye += ze;

         dRede->Mult(we, ze);
         MFEM_VERIFY(ze.CheckFinite() == 0, "err");
         ze *= -h;
         // OLD hydro.Me.TrueAddMult(we, ze);
         hydro.Me->FullAddMult(we, ze);
         MFEM_VERIFY(ze.CheckFinite() == 0, "err");

         ye += ze;

         yx.SyncAliasMemory(y);
         yv.SyncAliasMemory(y);
         ye.SyncAliasMemory(y);
      }

      virtual MemoryClass GetMemoryClass() const override
      {
         return Device::GetDeviceMemoryClass();
      }

      real_t h;
      std::function<void(const Vector &, Vector &)> jvp, assembled_jvp;
      const int H1tsize;
      const int L2tsize;
      mutable Vector w, z;

      LagrangianHydroOperator &hydro;
      std::shared_ptr<DerivativeOperator> dRvdx;
      std::shared_ptr<DerivativeOperator> dRvdv;
      std::shared_ptr<DerivativeOperator> dRvde;
      std::shared_ptr<DerivativeOperator> dRedx;
      std::shared_ptr<DerivativeOperator> dRedv;
      std::shared_ptr<DerivativeOperator> dRede;
      std::shared_ptr<DerivativeOperator> dTaylorSourcedx;
   };

   class Preconditioner : public Solver
   {
   public:
      Preconditioner(LagrangianHydroOperator &hydro) :
         Solver(hydro.Height()),
         hydro(hydro)
      {};

      void SetRebuildFlag(bool flag)
      {
         rebuild = flag;
      }

      void SetOperator(const Operator &op) override
      {
         if (!rebuild)
         {
            return;
         }

         jacobian = dynamic_cast<const LagrangianHydroJacobianOperator*>(&op);
         MFEM_ASSERT(jacobian != nullptr,
                     "Preconditioner can only be set with LagrangianHydroJacobianOperator");

         auto comm = hydro.H1.GetComm();
         const real_t h = jacobian->h;

         HYPRE_BigInt *tdof_offsets = hydro.H1.GetTrueDofOffsets();

         // First row
         // Rx = x - h * v
         // yx = I * wx - h I * wv
         SparseMatrix dRxdx_diag(hydro.H1.GetTrueVSize());
         for (int i = 0; i < dRxdx_diag.Height(); i++)
         {
            dRxdx_diag.Set(i, i, 1.0);
         }
         dRxdx_diag.Finalize();
         HypreParMatrix dRxdx_mat(comm, hydro.H1.GlobalTrueVSize(),
                                  tdof_offsets, &dRxdx_diag);

         // dRvdv = (Mv + h dF/dv)
         HypreParMatrix dRvdv_mat;
         jacobian->dRvdv->Assemble(dRvdv_mat);
         HypreParMatrix *Mv_hdRvdv_mat = Add(1.0, hydro.Mv_mat, h, dRvdv_mat);
         auto tmp2 = Mv_hdRvdv_mat->EliminateRowsCols(hydro.ess_tdof);
         delete tmp2;

         // dRede = Me - h * dF^T/de
         HypreParMatrix dRede_mat;
         jacobian->dRede->Assemble(dRede_mat);
         HypreParMatrix *Me_hdRede_mat = Add(1.0, hydro.Me_mat, -h, dRede_mat);

         if (hydro.preconditioner_type == PRECONDITIONER_TYPE::BLOCK_DIAGONAL_AMG)
         {
            out << "building pc\n";
            vv_mat.reset(Mv_hdRvdv_mat);
            amg_v.reset(new HypreBoomerAMG(*vv_mat));
            amg_v->SetPrintLevel(0);

            ee_mat.reset(Me_hdRede_mat);
            amg_e.reset(new HypreBoomerAMG(*ee_mat));
            amg_e->SetPrintLevel(0);
         }

         else if (hydro.preconditioner_type == PRECONDITIONER_TYPE::SUPERLU)
         {
#ifdef MFEM_USE_SUPERLU
            // dRxdv = -h * I
            SparseMatrix dRxdv_diag(hydro.H1.GetTrueVSize());
            for (int i = 0; i < dRxdv_diag.Height(); i++)
            {
               dRxdv_diag.Set(i, i, -h);
            }

            for (int i = 0; i < hydro.ess_tdof.Size(); i++)
            {
               dRxdv_diag.Set(hydro.ess_tdof[i], hydro.ess_tdof[i], 0.0);
            }
            dRxdv_diag.Finalize();

            HypreParMatrix dRxdv_mat(comm, hydro.H1.GlobalTrueVSize(),
                                     tdof_offsets, &dRxdv_diag);

            // Second row
            // Rv = Mv * v + F * I
            // yv = (Mv + h * dF/dv) * wv + h * dF/dx * wx + h * dF/de * we

            // dRvdx = h * dF/dx
            HypreParMatrix dRvdx_mat;
            jacobian->dRvdx->Assemble(dRvdx_mat);
            dRvdx_mat.EliminateRows(hydro.ess_tdof);
            dRvdx_mat *= h;

            // dRvde = h * dF/de
            HypreParMatrix dRvde_mat;
            jacobian->dRvde->Assemble(dRvde_mat);
            dRvde_mat.EliminateRows(hydro.ess_tdof);
            dRvde_mat *= h;

            // Third row
            // Re = Me * e - F^T
            // ye = (Me - h * dF/de) * we - h * dF/dx * wx - h * dF/dv * wv

            // dRedx = -h * dF^T/dx
            HypreParMatrix dRedx_mat;
            jacobian->dRedx->Assemble(dRedx_mat);

            if (problem == 0)
            {
               HypreParMatrix dTaylorSourcedx_mat;
               jacobian->dTaylorSourcedx->Assemble(dTaylorSourcedx_mat);
               dRedx_mat.Add(1.0, dTaylorSourcedx_mat);
            }

            dRedx_mat *= -h;

            // dRedv = -h * dF^T/dv
            HypreParMatrix dRedv_mat;
            jacobian->dRedv->Assemble(dRedv_mat);
            auto tmp1 = dRedv_mat.EliminateCols(hydro.ess_tdof);
            delete tmp1;
            dRedv_mat *= -h;

            Array2D<const HypreParMatrix*> blocks(3, 3);
            blocks = nullptr;
            blocks(0, 0) = &dRxdx_mat;
            blocks(0, 1) = &dRxdv_mat;
            blocks(1, 0) = &dRvdx_mat;
            blocks(1, 1) = Mv_hdRvdv_mat;
            blocks(1, 2) = &dRvde_mat;
            blocks(2, 0) = &dRedx_mat;
            blocks(2, 1) = &dRedv_mat;
            blocks(2, 2) = Me_hdRede_mat;

            superlu_solver.reset(new SuperLUSolver(MPI_COMM_WORLD, 1));
            block_hypre.reset(HypreParMatrixFromBlocks(blocks, nullptr));
            superlu_mat.reset(new SuperLURowLocMatrix(*block_hypre));
            superlu_solver->SetSymmetricPattern(false);
            superlu_solver->SetOperator(*superlu_mat);
            superlu_solver->SetPrintStatistics(false);
#else
            MFEM_ABORT("MFEM is not built with SuperLU");
#endif
         }

         rebuild = false;
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         if (hydro.preconditioner_type == PRECONDITIONER_TYPE::BLOCK_DIAGONAL_AMG)
         {
            w = x;
            Vector wx, wv, we;
            wx.MakeRef(w, 0, hydro.H1.GetTrueVSize());
            wv.MakeRef(w, hydro.H1.GetTrueVSize(), hydro.H1.GetTrueVSize());
            we.MakeRef(w, 2*hydro.H1.GetTrueVSize(), hydro.L2.GetTrueVSize());

            Vector yx, yv, ye;
            yx.MakeRef(y, 0, hydro.H1.GetTrueVSize());
            yv.MakeRef(y, hydro.H1.GetTrueVSize(), hydro.H1.GetTrueVSize());
            ye.MakeRef(y, 2*hydro.H1.GetTrueVSize(), hydro.L2.GetTrueVSize());

            yx = wx;
            amg_v->Mult(wv, yv);
            amg_e->Mult(we, ye);
         }
         else if (hydro.preconditioner_type == PRECONDITIONER_TYPE::SUPERLU)
         {
#ifdef MFEM_USE_SUPERLU
            superlu_solver->Mult(x, y);
#else
            MFEM_ABORT("MFEM is not built with SuperLU");
#endif
         }
      }

      mutable Vector w;
      LagrangianHydroOperator &hydro;
      bool rebuild = true;
      const LagrangianHydroJacobianOperator *jacobian = nullptr;
      std::shared_ptr<HypreParMatrix> block_hypre;
      std::shared_ptr<HypreParMatrix> vv_mat;
      std::shared_ptr<HypreBoomerAMG> amg_v;
      std::shared_ptr<HypreParMatrix> ee_mat;
      std::shared_ptr<HypreBoomerAMG> amg_e;
#ifdef MFEM_USE_SUPERLU
      std::shared_ptr<SuperLURowLocMatrix> superlu_mat;
      std::shared_ptr<SuperLUSolver> superlu_solver;
#endif
   };

   class LagrangianHydroResidualOperator : public Operator
   {
   public:
      LagrangianHydroResidualOperator(LagrangianHydroOperator &hydro,
                                      const Vector &x, bool fd_gradient, int dump_jacobians) :
         Operator(2*hydro.H1.GetTrueVSize()+hydro.L2.GetTrueVSize()),
         hydro(hydro),
         H1tsize(hydro.H1.GetTrueVSize()),
         H1vsize(hydro.H1.GetVSize()),
         L2tsize(hydro.L2.GetTrueVSize()),
         L2vsize(hydro.L2.GetVSize()),
         x(x),
         u(x.Size()),
         u_l(2*H1vsize + L2vsize),
         e_source_t(hydro.L2.GetTrueVSize()),
         fd_gradient(fd_gradient),
         dump_jacobians(dump_jacobians) {}

      void SetTimeStep(const real_t &time_step) { this->dt = time_step; }

      void Mult(const Vector &k, Vector &R) const override
      {
         u = k;
         u *= dt;
         u += x;

         auto kptr = const_cast<Vector*>(&k);
         Vector kx, kv, ke;
         kx.MakeRef(*kptr, 0, H1tsize);
         kv.MakeRef(*kptr, H1tsize, H1tsize);
         ke.MakeRef(*kptr, 2*H1tsize, L2tsize);

         Vector ux, uv, ue;
         ux.MakeRef(u, 0, H1tsize);
         uv.MakeRef(u, H1tsize, H1tsize);
         ue.MakeRef(u, 2*H1tsize, L2tsize);

         Vector Rx, Rv, Re;
         Rx.MakeRef(R, 0, H1tsize);
         Rv.MakeRef(R, H1tsize, H1tsize);
         Re.MakeRef(R, 2*H1tsize, L2tsize);

         Vector ux_l, uv_l, ue_l;
         ux_l.MakeRef(u_l, 0, H1vsize);
         uv_l.MakeRef(u_l, H1vsize, H1vsize);
         ue_l.MakeRef(u_l, 2*H1vsize, L2vsize);

         hydro.H1.GetProlongationMatrix()->Mult(ux, ux_l);
         hydro.H1.GetProlongationMatrix()->Mult(uv, uv_l);
         hydro.L2.GetProlongationMatrix()->Mult(ue, ue_l);

         hydro.UpdateMesh(ux_l);
         hydro.mesh_nodes.SyncMemory(ux_l);

         Rx = kx;
         Rx -= uv;

         // out << "in qupdate\n";
         // pretty_print(ux_l);
         hydro.qdata_is_current = false;
         hydro.UpdateQuadratureData(u_l);

         hydro.momentum_pa->SetParameters({&hydro.qdata->stressp});
         hydro.momentum_pa->Mult(uv, Rv);

         // hydro.Mv.TrueAddMult(kv, Rv);
         Vector kvc, Rvc;
         for (int c = 0; c < hydro.H1.GetMesh()->Dimension(); c++)
         {
            kvc.MakeRef(kv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
            Rvc.MakeRef(Rv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
            hydro.Mv->FullAddMult(kvc, Rvc);
            Rvc.SyncAliasMemory(Rv);
         }
         Rv.SyncAliasMemory(R);

         Rv.SetSubVector(hydro.ess_tdof, 0.0);
         // Rv = 0.0;

         hydro.energy_conservation_pa->SetParameters({&uv_l, &hydro.qdata->stressp});
         hydro.energy_conservation_pa->Mult(ue, Re);

         Re.Neg();

         if (problem == 0)
         {
            hydro.taylor_source_mf->SetParameters({&ux_l});
            hydro.taylor_source_mf->Mult(e_source_t, e_source_t);
            Re -= e_source_t;
         }

         // hydro.Me.TrueAddMult(ke, Re);
         hydro.Me->FullAddMult(ke, Re);

         Rx.SyncAliasMemory(R);
         Rv.SyncAliasMemory(R);
         Re.SyncAliasMemory(R);
      }

      Operator& GetGradient(const Vector &k) const override
      {
         u = k;
         u *= dt;
         u += x;

         auto kptr = const_cast<Vector*>(&k);
         Vector kx, kv, ke;
         kx.MakeRef(*kptr, 0, H1tsize);
         kv.MakeRef(*kptr, H1tsize, H1tsize);
         ke.MakeRef(*kptr, 2*H1tsize, L2tsize);

         Vector ux, uv, ue;
         ux.MakeRef(u, 0, H1tsize);
         uv.MakeRef(u, H1tsize, H1tsize);
         ue.MakeRef(u, 2*H1tsize, L2tsize);

         Vector ux_l, uv_l, ue_l;
         ux_l.MakeRef(u_l, 0, H1vsize);
         uv_l.MakeRef(u_l, H1vsize, H1vsize);
         ue_l.MakeRef(u_l, 2*H1vsize, L2vsize);

         hydro.H1.GetProlongationMatrix()->Mult(ux, ux_l);
         hydro.H1.GetProlongationMatrix()->Mult(uv, uv_l);
         hydro.L2.GetProlongationMatrix()->Mult(ue, ue_l);

         if (fd_gradient)
         {
            fd_jacobian = std::make_shared<FDJacobian>(*this, k, 1e-8);
            return *fd_jacobian;
         }
         else
         {
            auto dRvdx = hydro.momentum_mf->GetDerivative(COORDINATES, {&uv_l},
            {&hydro.rho0, &hydro.x0, &ux_l, &hydro.material, &ue_l});

            auto dRvdv = hydro.momentum_mf->GetDerivative(VELOCITY, {&uv_l},
            {&hydro.rho0, &hydro.x0, &ux_l, &hydro.material, &ue_l});

            auto dRvde = hydro.momentum_mf->GetDerivative(SPECIFIC_INTERNAL_ENERGY, {&uv_l},
            {&hydro.rho0, &hydro.x0, &ux_l, &hydro.material, &ue_l});

            auto dRedx = hydro.energy_conservation_mf->GetDerivative(COORDINATES, {&ue_l},
            {&uv_l, &hydro.rho0, &hydro.x0, &ux_l, &hydro.material});

            auto dRedv = hydro.energy_conservation_mf->GetDerivative(VELOCITY, {&ue_l},
            {&uv_l, &hydro.rho0, &hydro.x0, &ux_l, &hydro.material});

            auto dRede = hydro.energy_conservation_mf->GetDerivative(
            SPECIFIC_INTERNAL_ENERGY, {&ue_l},
            {&uv_l, &hydro.rho0, &hydro.x0, &ux_l, &hydro.material});

            auto dTaylorSourcedx = hydro.taylor_source_mf->GetDerivative(COORDINATES, {&ue_l}, {&ux_l});

            jacobian = std::make_shared<LagrangianHydroJacobianOperator>(
                          hydro, dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede, dTaylorSourcedx, dt);

            if (dump_jacobians > 0)
            {
               auto ess_tdof_backup(hydro.ess_tdof);
               hydro.ess_tdof.SetSize(0);

               out << "\ndumping jacobians\n";
               std::ofstream jvpmat("jvpmat.m");
               jacobian->PrintMatlab(jvpmat);
               jvpmat.close();

               fd_jacobian = std::make_shared<FDJacobian>(*this, k, 1e-8);
               std::ofstream fdjacmat("fdjacmat.m");
               fd_jacobian->PrintMatlab(fdjacmat);
               fdjacmat.close();

               // hydro.preconditioner->SetOperator(*jacobian);
               // std::ofstream jprecmat("jprecmat.m");
               // hydro.preconditioner->block_hypre->PrintMatlab(jprecmat);
               // jprecmat.close();

               if (dump_jacobians == 1)
               {
                  exit(0);
               }

               hydro.ess_tdof = ess_tdof_backup;
            }

            return *jacobian;
         }
      }

      LagrangianHydroOperator &hydro;
      real_t dt;
      const int H1tsize;
      const int H1vsize;
      const int L2tsize;
      const int L2vsize;
      const Vector &x;
      mutable Vector u, u_l, e_source_t;
      mutable std::shared_ptr<FDJacobian> fd_jacobian;
      mutable std::shared_ptr<LagrangianHydroJacobianOperator> jacobian;
      bool fd_gradient;
      int dump_jacobians;
   };

   LagrangianHydroOperator(
      ParFiniteElementSpace &H1,
      ParFiniteElementSpace &L2,
      Array<int> &ess_tdof,
      const IntegrationRule &ir,
      FunctionCoefficient &rho0_coeff,
      ParGridFunction &x0_gf,
      ParGridFunction &rho0_gf,
      ParGridFunction &material_gf,
      std::shared_ptr<DifferentiableOperator> update_qdata,
      std::shared_ptr<DifferentiableOperator> dtest_mf,
      std::shared_ptr<DifferentiableOperator> momentum_mf,
      std::shared_ptr<DifferentiableOperator> momentum_pa,
      std::shared_ptr<DifferentiableOperator> energy_conservation_mf,
      std::shared_ptr<DifferentiableOperator> energy_conservation_pa,
      std::shared_ptr<DifferentiableOperator> total_internal_energy_mf,
      std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf,
      std::shared_ptr<DifferentiableOperator> density_mf,
      std::shared_ptr<DifferentiableOperator> taylor_source_mf,
      std::shared_ptr<QuadratureData> qdata,
      const bool &fd_gradient,
      const int &dump_jacobians,
      const int &nonlinear_maximum_iterations,
      const real_t &nonlinear_relative_tolerance,
      const int &krylov_maximum_iterations,
      const int &preconditioner_lag,
      const PRECONDITIONER_TYPE &preconditioner_type,
      Vector& external_data) :
      TimeDependentOperator(2*H1.GetVSize()+L2.GetVSize()),
      H1(H1),
      L2(L2),
      H1c(H1.GetParMesh(), H1.FEColl(), 1),
      ess_tdof(ess_tdof),
      ir(ir),
      x0(x0_gf),
      rho0(rho0_gf),
      material(material_gf),
      update_qdata(update_qdata),
      dtest_mf(dtest_mf),
      momentum_mf(momentum_mf),
      momentum_pa(momentum_pa),
      energy_conservation_mf(energy_conservation_mf),
      energy_conservation_pa(energy_conservation_pa),
      total_internal_energy_mf(total_internal_energy_mf),
      total_kinetic_energy_mf(total_kinetic_energy_mf),
      density_mf(density_mf),
      taylor_source_mf(taylor_source_mf),
      qdata(qdata),
      mesh_nodes(&H1),
      rhsvc(&H1c),
      dvc(&H1c),
      Mv_blf(&H1),
      Me_blf(&L2),
      rho0_coeff(rho0_coeff),
      RHSv(H1.GetTrueVSize()),
      rhsv(H1.GetVSize()),
      X(2*H1.GetTrueVSize()+L2.GetTrueVSize()),
      Xv(H1.GetTrueVSize()),
      Xvc(H1c.GetTrueVSize()),
      Xe(L2.GetTrueVSize()),
      K(2*H1.GetTrueVSize()+L2.GetTrueVSize()),
      B(H1c.GetTrueVSize()),
      RHSe(L2.GetTrueVSize()),
      rhse(L2.GetVSize()),
      nl2dofs(L2.GetFE(0)->GetDof()),
      fd_gradient(fd_gradient),
      dump_jacobians(dump_jacobians),
      nonlinear_maximum_iterations(nonlinear_maximum_iterations),
      nonlinear_relative_tolerance(nonlinear_relative_tolerance),
      krylov_maximum_iterations(krylov_maximum_iterations),
      preconditioner_lag(preconditioner_lag),
      preconditioner_type(preconditioner_type),
      external_data(external_data)
   {
      Mv = new MassPAOperator(H1c, ir, rho0_coeff);
      Array<int> empty_tdofs;
      Mv_Jprec = new OperatorJacobiSmoother(Mv->GetBF(), empty_tdofs);

      Me = new MassPAOperator(L2, ir, rho0_coeff);

      // Inside the above constructors for mass, there is reordering of the mesh
      // nodes which is performed on the host. Since the mesh nodes are a
      // subvector, so we need to sync with the rest of the base vector (which
      // is assumed to be in the memory space used by the mfem::Device).
      H1.GetParMesh()->GetNodes()->ReadWrite();
      // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
      // we must enforce v_x/y/z = 0 for the velocity components.
      const int bdr_attr_max = H1.GetMesh()->bdr_attributes.Max();
      Array<int> ess_bdr(bdr_attr_max);
      for (int c = 0; c < H1.GetMesh()->Dimension(); c++)
      {
         ess_bdr = 0;
         ess_bdr[c] = 1;
         H1c.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
         c_tdofs[c].Read();
      }

      Mv_blf.AddDomainIntegrator(new VectorMassIntegrator(rho0_coeff, &ir));
      Mv_blf.Assemble();
      Mv_blf.FormSystemMatrix(mfem::Array<int>(), Mv_mat);

      Me_blf.AddDomainIntegrator(new MassIntegrator(rho0_coeff, &ir));
      Me_blf.Assemble();
      Me_blf.FormSystemMatrix(mfem::Array<int>(), Me_mat);

      residual.reset(new LagrangianHydroResidualOperator(*this, X, fd_gradient,
                                                         dump_jacobians));
      preconditioner.reset(new Preconditioner(*this));

      auto gmres = new GMRESSolver(MPI_COMM_WORLD);
      // gmres->SetMaxIter(500);
      gmres->SetKDim(500);
      gmres->SetMaxIter(krylov_maximum_iterations);
      gmres->SetRelTol(1e-12);
      gmres->SetAbsTol(0.0);
      gmres->SetPrintLevel(IterativeSolver::PrintLevel().None());
      gmres->SetPreconditioner(*preconditioner);
      krylov.reset(gmres);

      newton.reset(new LineSearchNewtonSolver(MPI_COMM_WORLD, *this));
      newton->SetPrintLevel(IterativeSolver::PrintLevel().Summary());
      newton->SetOperator(*residual);
      newton->SetSolver(*krylov);
      newton->SetMaxIter(nonlinear_maximum_iterations);
      newton->SetRelTol(nonlinear_relative_tolerance);
      newton->SetAbsTol(0.0);
      // newton->SetAdaptiveLinRtol();
   }

   void Mult(const Vector &S, Vector &dSdt) const override
   {
      UpdateMesh(S);
      UpdateQuadratureData(S);

      auto sptr = const_cast<Vector*>(&S);
      const int H1vsize = H1.GetVSize();

      ParGridFunction x, v, e;
      x.MakeRef(&H1, *sptr, 0);
      v.MakeRef(&H1, *sptr, H1vsize);
      e.MakeRef(&L2, *sptr, 2*H1vsize);

      ParGridFunction dx, dv, de;
      dx.MakeRef(&H1, dSdt, 0);
      dv.MakeRef(&H1, dSdt, H1vsize);
      de.MakeRef(&L2, dSdt, 2*H1vsize);

      // solve position
      dx = v;

      // out << ">>> dx\n";
      // pretty_print(dx);

      // solve velocity
      {
         dv = 0.0;

         H1.GetRestrictionMatrix()->Mult(v, Xv);
         // momentum_mf->SetParameters({&rho0, &x0, &x, &material, &e});
         // momentum_mf->Mult(Xv, RHSv);
         momentum_pa->SetParameters({&qdata->stressp});
         momentum_pa->Mult(Xv, RHSv);
         RHSv.Neg();
         H1.GetRestrictionMatrix()->MultTranspose(RHSv, rhsv);

         // out << ">>> rhsv\n";
         // pretty_print(rhsv);

         // solve for each velocity component
         const int size = H1c.GetVSize();
         const Operator *Pconf = H1c.GetProlongationMatrix();
         for (int c = 0; c < H1.GetMesh()->Dimension(); c++)
         {
            dvc.MakeRef(&H1c, dSdt, H1vsize + c*size);
            rhsvc.MakeRef(&H1c, rhsv, c*size);
            if (Pconf)
            {
               Pconf->MultTranspose(rhsvc, B);
            }
            else
            {
               B = rhsvc;
            }

            CGSolver cg(H1c.GetParMesh()->GetComm());
            cg.SetPreconditioner(*Mv_Jprec);
            cg.SetOperator(*Mv);
            cg.SetRelTol(1e-8);
            cg.SetAbsTol(0.0);
            cg.SetMaxIter(300);
            cg.SetPrintLevel(-1);

            H1c.GetRestrictionMatrix()->Mult(dvc, Xvc);
            Mv->SetEssentialTrueDofs(c_tdofs[c]);
            Mv->EliminateRHS(B);
            cg.Mult(B, Xvc);
            if (Pconf)
            {
               Pconf->Mult(Xvc, dvc);
            }
            else
            {
               dvc = Xvc;
            }
            dvc.GetMemory().SyncAlias(dSdt.GetMemory(), dvc.Size());
         }
      }
      // out << ">>> dv\n";
      // pretty_print(dv);

      // solve energy
      {
         de = 0.0;

         L2.GetRestrictionMatrix()->Mult(e, Xe);
         // energy_conservation_mf->SetParameters({&v, &rho0, &x0, &x, &material});
         // energy_conservation_mf->Mult(Xe, RHSe);
         energy_conservation_pa->SetParameters({&v, &qdata->stressp});
         energy_conservation_pa->Mult(Xe, RHSe);
         L2.GetRestrictionMatrix()->MultTranspose(RHSe, rhse);

         if (problem == 0)
         {
            LinearForm e_source(&L2);
            L2.GetMesh()->DeleteGeometricFactors();
            FunctionCoefficient coeff(taylor_source);
            DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &ir);
            e_source.AddDomainIntegrator(d);
            e_source.UseFastAssembly(true);
            e_source.Assemble();
            rhse += e_source;
         }

         CGSolver cg(L2.GetParMesh()->GetComm());
         cg.SetOperator(*Me);
         cg.iterative_mode = false;
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(300);
         cg.SetPrintLevel(-1);
         cg.Mult(rhse, de);
         de.GetMemory().SyncAlias(dSdt.GetMemory(), de.Size());

         // out << ">>> de\n";
         // pretty_print(de);
      }

      qdata_is_current = false;

      // out << ">>> dSdt\n";
      // pretty_print(dSdt);
   }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override
   {
      auto xptr = const_cast<Vector*>(&x);

      Vector xx, xv, xe;
      xx.MakeRef(*xptr, 0, H1.GetVSize());
      xv.MakeRef(*xptr, H1.GetVSize(), H1.GetVSize());
      xe.MakeRef(*xptr, 2*H1.GetVSize(), L2.GetVSize());

      Xx.MakeRef(X, 0, H1.GetTrueVSize());
      Xv.MakeRef(X, H1.GetTrueVSize(), H1.GetTrueVSize());
      Xe.MakeRef(X, 2*H1.GetTrueVSize(), L2.GetTrueVSize());

      H1.GetRestrictionMatrix()->Mult(xx, Xx);
      H1.GetRestrictionMatrix()->Mult(xv, Xv);
      L2.GetRestrictionMatrix()->Mult(xe, Xe);

      Xx.SyncAliasMemory(X);
      Xv.SyncAliasMemory(X);
      Xe.SyncAliasMemory(X);

      residual->SetTimeStep(dt);

      if (current_dt != dt || lag >= preconditioner_lag)
      {
         lag = 0;
         current_dt = dt;
         preconditioner->SetRebuildFlag(true);
      }
      lag++;

      Vector zero;
      K = X;
      newton->Mult(zero, K);

      newton_converged = newton->GetConverged();

      Kx.MakeRef(K, 0, H1.GetTrueVSize());
      Kv.MakeRef(K, H1.GetTrueVSize(), H1.GetTrueVSize());
      Ke.MakeRef(K, 2*H1.GetTrueVSize(), L2.GetTrueVSize());

      Vector kx, kv, ke;
      kx.MakeRef(k, 0, H1.GetVSize());
      kv.MakeRef(k, H1.GetVSize(), H1.GetVSize());
      ke.MakeRef(k, 2*H1.GetVSize(), L2.GetVSize());

      H1.GetProlongationMatrix()->Mult(Kx, kx);
      H1.GetProlongationMatrix()->Mult(Kv, kv);
      L2.GetProlongationMatrix()->Mult(Ke, ke);
   }

   void UpdateMesh(const Vector &S) const
   {
      Vector* sptr = const_cast<Vector*>(&S);
      mesh_nodes.MakeRef(&H1, *sptr, 0);
      H1.GetParMesh()->NewNodes(mesh_nodes, false);
   }

   void ResetTimeStepEstimate()
   {
      external_data[EXT_DATA_IDX::DT_ESTIMATE] =
         std::numeric_limits<double>::infinity();
   }

   void ResetQuadratureData() const
   {
      qdata_is_current = false;
   }

   real_t GetTimeStepEstimate(const Vector &S)
   {
      UpdateMesh(S);
      UpdateQuadratureData(S);

      // auto sptr = const_cast<Vector*>(&S);
      // const int H1vsize = H1.GetVSize();
      // ParGridFunction x, v, e;
      // x.MakeRef(&H1, *sptr, 0);
      // v.MakeRef(&H1, *sptr, H1vsize);
      // e.MakeRef(&L2, *sptr, 2*H1vsize);
      // dtest_mf->SetParameters({&v, &rho0, &x0, &x, &material, &e});
      // auto &dt_est = qdata->dt_est;
      // dtest_mf->Mult(dt_est, dt_est);

      // out << ">>> dt_est\n";
      // pretty_print(dt_est);

      // real_t dt_est_local = std::numeric_limits<real_t>::infinity();
      // for (int i = 0; i < dt_est.Size(); i++)
      // {
      //    if (dt_est(i) == 0.0)
      //    {
      //       return 0.0;
      //    }
      //    dt_est_local = fmin(dt_est_local, dt_est(i));
      // }

      // update_qdata

      real_t dt_est_local = external_data[EXT_DATA_IDX::DT_ESTIMATE];

      real_t dt_est_global;
      MPI_Allreduce(&dt_est_local, &dt_est_global, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN,
                    L2.GetComm());

      return dt_est_global;
   }

   real_t InternalEnergy(ParGridFunction &e)
   {
      const auto mt = Device::GetDeviceMemoryType();
      Vector E(L2.GetTrueVSize(), mt), Y(L2.GetTrueVSize(), mt);
      total_internal_energy_mf->SetParameters({&rho0, &x0});
      L2.GetRestrictionMatrix()->Mult(e, E);
      total_internal_energy_mf->Mult(E, Y);
      const real_t ie_local = Y.Sum();
      real_t ie_global = 0.0;
      MPI_Allreduce(&ie_local, &ie_global, 1, MPI_DOUBLE, MPI_SUM,
                    L2.GetParMesh()->GetComm());
      return ie_global;
   }

   real_t KineticEnergy(ParGridFunction &v)
   {
      const auto mt = Device::GetDeviceMemoryType();
      Vector V(H1.GetTrueVSize(), mt), Y(L2.GetTrueVSize(), mt);
      total_kinetic_energy_mf->SetParameters({&rho0, &x0});
      H1.GetRestrictionMatrix()->Mult(v, V);
      total_kinetic_energy_mf->Mult(V, Y);
      const real_t ke_local = Y.Sum();
      real_t ke_global = 0.0;
      MPI_Allreduce(&ke_local, &ke_global, 1, MPI_DOUBLE, MPI_SUM,
                    H1.GetParMesh()->GetComm());
      return ke_global;
   }

   void ComputeDensity(ParGridFunction &rho)
   {
      rho.SetSpace(&L2);

      ParGridFunction rhs_l(&L2);

      Vector rho0_t(L2.GetTrueVSize()),
             rho_t(L2.GetTrueVSize()),
             rhs(L2.GetTrueVSize());

      const int l2dofs_cnt = L2.GetFE(0)->GetDof();
      DenseMatrix Mrho(l2dofs_cnt);
      DenseMatrixInverse inv(&Mrho);
      Vector rhs_e(l2dofs_cnt), rho_z(l2dofs_cnt);
      Array<int> dofs(l2dofs_cnt);
      MassIntegrator mi(&ir);

      density_mf->SetParameters({&x0});
      L2.GetProlongationMatrix()->MultTranspose(rho0, rho0_t);
      density_mf->Mult(rho0_t, rhs);
      L2.GetProlongationMatrix()->Mult(rhs, rhs_l);

      for (int e = 0; e < L2.GetParMesh()->GetNE(); e++)
      {
         const FiniteElement &fe = *L2.GetFE(e);
         ElementTransformation &eltr = *L2.GetElementTransformation(e);
         L2.GetElementDofs(e, dofs);
         mi.AssembleElementMatrix(fe, eltr, Mrho);
         inv.Factor();
         rhs_l.GetElementDofValues(e, rhs_e);
         inv.Mult(rhs_e, rho_z);
         rho.SetSubVector(dofs, rho_z);
      }
   }

   real_t ComputeMinDet(const Vector &S) const
   {
      auto sptr = const_cast<Vector*>(&S);
      Vector x, x_loc;
      x.MakeRef(*sptr, 0, H1.GetTrueVSize());
      H1.GetProlongationMatrix()->Mult(x, x_loc);

      // out << "in ComputeMinDet\n";
      // pretty_print(x_loc);

      ParGridFunction xgf;
      xgf.MakeRef(&H1, x_loc, 0);

      auto g = GeometricFactors(xgf, ir,
                                GeometricFactors::JACOBIANS |
                                GeometricFactors::DETERMINANTS);

      return g.detJ.Min();
   }

   void UpdateQuadratureData(const Vector &S) const
   {
      if (qdata_is_current) { return; }
      // out << "updating qdata\n";
      qdata_is_current = true;

      auto sptr = const_cast<Vector*>(&S);
      const int H1vsize = H1.GetVSize();
      ParGridFunction x, v, e;
      x.MakeRef(&H1, *sptr, 0);
      v.MakeRef(&H1, *sptr, H1vsize);
      e.MakeRef(&L2, *sptr, 2*H1vsize);
      update_qdata->SetParameters({&v, &rho0, &x0, &x, &material, &e});
      update_qdata->Mult(qdata->stressp, qdata->stressp);
   }

   virtual MemoryClass GetMemoryClass() const override
   {
      return Device::GetDeviceMemoryClass();
   }

   ~LagrangianHydroOperator()
   {
      delete Mv;
      delete Mv_Jprec;
      delete Me;
   }

   ParFiniteElementSpace &H1;
   ParFiniteElementSpace &L2;
   mutable ParFiniteElementSpace H1c;
   Array<int> &ess_tdof;
   mutable Array<int> c_tdofs[3];
   const IntegrationRule &ir;
   ParGridFunction &x0;
   ParGridFunction &rho0;
   ParGridFunction &material;
   std::shared_ptr<DifferentiableOperator> update_qdata;
   std::shared_ptr<DifferentiableOperator> dtest_mf;
   std::shared_ptr<DifferentiableOperator> momentum_mf;
   std::shared_ptr<DifferentiableOperator> momentum_pa;
   std::shared_ptr<DifferentiableOperator> energy_conservation_mf;
   std::shared_ptr<DifferentiableOperator> energy_conservation_pa;
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf;
   std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf;
   std::shared_ptr<DifferentiableOperator> density_mf;
   std::shared_ptr<DifferentiableOperator> taylor_source_mf;
   std::shared_ptr<QuadratureData> qdata;
   mutable ParGridFunction mesh_nodes, rhsvc, dvc;
   mutable MassPAOperator *Mv = nullptr, *Me = nullptr;
   ParBilinearForm Mv_blf, Me_blf;
   HypreParMatrix Mv_mat, Me_mat;

   std::shared_ptr<LagrangianHydroResidualOperator> residual;
   std::shared_ptr<Preconditioner> preconditioner;
   std::shared_ptr<Solver> krylov;
   std::shared_ptr<NewtonSolver> newton;
   real_t current_dt = 0.0;
   int lag = 0;

   mutable FunctionCoefficient rho0_coeff;
   OperatorJacobiSmoother *Mv_Jprec = nullptr;
   mutable Vector RHSv, rhsv, X, Xx, Xv, Xvc, Xe, K, Kx, Kv, Ke, B, RHSe, rhse;
   const int nl2dofs;
   bool fd_gradient;
   int dump_jacobians;
   const int nonlinear_maximum_iterations;
   const real_t nonlinear_relative_tolerance;
   const int krylov_maximum_iterations;
   const int preconditioner_lag;
   const PRECONDITIONER_TYPE preconditioner_type;
   bool newton_converged = true;
   Vector &external_data;
   mutable bool qdata_is_current = false;
};

static auto CreateLagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   Vector &external_data,
   const IntegrationRule &ir,
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const PRECONDITIONER_TYPE &preconditioner_type)
{
   ParMesh &mesh = *H1.GetParMesh();

   auto qdata = std::make_shared<QuadratureData>(mesh, ir);

   int ne_loc = mesh.GetNE(), ne_global = 0;
   real_t vol_loc = 0.0, vol_global = 0.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      vol_loc += mesh.GetElementVolume(e);
   }
   MPI_Allreduce(&vol_loc, &vol_global, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   MPI_Allreduce(&ne_loc, &ne_global, 1, MPI_INT, MPI_SUM, mesh.GetComm());

   real_t &h0 = external_data[EXT_DATA_IDX::H0];
   switch (mesh.GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT: h0 = vol_global / ne_global; break;
      case Geometry::SQUARE: h0 = sqrt(vol_global / ne_global); break;
      case Geometry::TRIANGLE: h0 = sqrt(2.0 * vol_global / ne_global); break;
      case Geometry::CUBE: h0 = pow(vol_global / ne_global, 1./3.); break;
      case Geometry::TETRAHEDRON: h0 = pow(6.0 * vol_global / ne_global,
                                              1./3.); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   h0 /= (double) H1.GetOrder(0);

   // const real_t h0 = sqrt(vol_global / ne_global) /
   //                   static_cast<real_t>(H1.GetOrder(0));

   // qdata->order_v = order_v;
   qdata->dt_est = std::numeric_limits<real_t>::infinity();

   auto d_external_data = external_data.ReadWrite();

   Array<int> all_domain_attr(mesh.attributes.Max());
   all_domain_attr = 1;

   std::shared_ptr<DifferentiableOperator> dt_est_mf;
   {
      mfem::tuple dt_est_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };

      mfem::tuple dt_est_kernel_oo = {None<DT_EST>{}};

      std::vector dt_est_solutions =
      {
         FieldDescriptor{DT_EST, &qdata->R}
      };

      std::vector dt_est_parameters =
      {
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{COORDINATES, &H1},
         FieldDescriptor{MATERIAL, material_gf.ParFESpace()},
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2},
      };

      dt_est_mf = std::make_shared<DifferentiableOperator>(
                     dt_est_solutions, dt_est_parameters, mesh);
      TimeStepEstimateQFunction dt_est_qf(d_external_data);
      dt_est_mf->AddDomainIntegrator(dt_est_qf, dt_est_kernel_ao,
                                     dt_est_kernel_oo,
                                     ir,
                                     all_domain_attr);
   }

   std::shared_ptr<DifferentiableOperator> update_qdata;
   {
      mfem::tuple update_qdata_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };

      mfem::tuple update_qdata_kernel_oo = {None<STRESS_TENSOR>{}};

      std::vector<FieldDescriptor> update_qdata_solutions =
      {
         {STRESS_TENSOR, &qdata->StressSpace}
      };

      std::vector<FieldDescriptor> update_qdata_parameters =
      {
         {VELOCITY, &H1},
         {DENSITY0, &L2},
         {COORDINATES0, &H1},
         {COORDINATES, &H1},
         {MATERIAL, material_gf.ParFESpace()},
         {SPECIFIC_INTERNAL_ENERGY, &L2},
      };

      update_qdata = std::make_shared<DifferentiableOperator>(
                        update_qdata_solutions, update_qdata_parameters, mesh);
      UpdateQuadratureDataQFunction update_qdata_qf(d_external_data);
      update_qdata->AddDomainIntegrator(update_qdata_qf, update_qdata_kernel_ao,
                                        update_qdata_kernel_oo,
                                        ir,
                                        all_domain_attr);
   }

   // Create momentum operator
   std::shared_ptr<DifferentiableOperator> momentum_mf;
   {
      mfem::tuple momentum_mf_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };

      mfem::tuple momentum_mf_kernel_oo = {Gradient<VELOCITY>{}};

      // <sigma, grad(w) * J^-T> * det(J) * weights
      // <sigma(J^-T det(J) weights), grad(w)>

      std::vector momentum_mf_solutions =
      {
         FieldDescriptor{VELOCITY, &H1}
      };

      std::vector momentum_mf_parameters =
      {
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{COORDINATES, &H1},
         FieldDescriptor{MATERIAL, material_gf.ParFESpace()},
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2},
      };

      momentum_mf = std::make_shared<DifferentiableOperator>(
                       momentum_mf_solutions, momentum_mf_parameters, mesh);

      MomentumQFunction momentum_qf(d_external_data);
      auto derivatives =
         std::integer_sequence<size_t, VELOCITY, COORDINATES, SPECIFIC_INTERNAL_ENERGY> {};
      momentum_mf->AddDomainIntegrator(momentum_qf, momentum_mf_kernel_ao,
                                       momentum_mf_kernel_oo, ir, all_domain_attr, derivatives);
   }

   std::shared_ptr<DifferentiableOperator> momentum_pa;
   {
      mfem::tuple momentum_pa_kernel_ao = {None<STRESS_TENSOR>{}};
      mfem::tuple momentum_pa_kernel_oo = {Gradient<VELOCITY>{}};

      std::vector<FieldDescriptor> momentum_pa_solutions = {{VELOCITY, &H1}};
      std::vector<FieldDescriptor> momentum_pa_parameters = {{STRESS_TENSOR, &qdata->StressSpace}};

      momentum_pa = std::make_shared<DifferentiableOperator>(
                       momentum_pa_solutions, momentum_pa_parameters, mesh);

      MomentumPAQFunction momentum_pa_qf;
      momentum_pa->AddDomainIntegrator(momentum_pa_qf, momentum_pa_kernel_ao,
                                       momentum_pa_kernel_oo, ir, all_domain_attr);
   }

   // Create energy conservation operator
   std::shared_ptr<DifferentiableOperator> energy_conservation_mf;
   {
      mfem::tuple energy_conservation_mf_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };

      mfem::tuple energy_conservation_mf_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      // <sigma, grad(v) * inv(J) * phi> * det(J) * w
      // <sigma(J^-T det(J) w), grad(v) * inv(J)>

      std::vector energy_conservation_mf_solutions =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      std::vector energy_conservation_mf_parameters =
      {
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{COORDINATES, &H1},
         FieldDescriptor{MATERIAL, material_gf.ParFESpace()},
      };

      energy_conservation_mf =
         std::make_shared<DifferentiableOperator>(
            energy_conservation_mf_solutions, energy_conservation_mf_parameters, mesh);

      EnergyConservationQFunction energy_conservation_qf(d_external_data);
      auto derivatives =
         std::integer_sequence<size_t, VELOCITY, COORDINATES, SPECIFIC_INTERNAL_ENERGY> {};
      energy_conservation_mf->AddDomainIntegrator(
         energy_conservation_qf, energy_conservation_mf_kernel_ao,
         energy_conservation_mf_kernel_oo, ir, all_domain_attr, derivatives);
   }

   std::shared_ptr<DifferentiableOperator> energy_conservation_pa;
   {
      mfem::tuple energy_conservation_pa_kernel_ao = {Gradient<VELOCITY>{}, None<STRESS_TENSOR>{}};
      mfem::tuple energy_conservation_pa_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      std::vector<FieldDescriptor> energy_conservation_pa_solutions =
      {
         {SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      std::vector<FieldDescriptor> energy_conservation_pa_parameters =
      {
         {VELOCITY, &H1},
         {STRESS_TENSOR, &qdata->StressSpace}
      };

      energy_conservation_pa =
         std::make_shared<DifferentiableOperator>(
            energy_conservation_pa_solutions, energy_conservation_pa_parameters, mesh);

      EnergyConservationPAQFunction energy_conservation_qf;
      energy_conservation_pa->AddDomainIntegrator(
         energy_conservation_qf, energy_conservation_pa_kernel_ao,
         energy_conservation_pa_kernel_oo, ir, all_domain_attr);
   }

   // Create total internal energy operator
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf;
   {
      mfem::tuple total_internal_energy_kernel_ao =
      {
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };

      mfem::tuple total_internal_energy_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      std::vector total_internal_energy_solutions =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      std::vector total_internal_energy_parameters =
      {
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };

      total_internal_energy_mf =
         std::make_shared<DifferentiableOperator>(
            total_internal_energy_solutions,
            total_internal_energy_parameters,
            mesh);

      TotalInternalEnergyQFunction total_internal_energy_qf;
      total_internal_energy_mf->AddDomainIntegrator(
         total_internal_energy_qf, total_internal_energy_kernel_ao,
         total_internal_energy_kernel_oo, ir, all_domain_attr);
   }

   // Create total kinetic energy operator
   std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf;
   {
      mfem::tuple total_kinetic_energy_kernel_ao =
      {
         Value<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };

      mfem::tuple total_kinetic_energy_kernel_oo = {Value<DENSITY0>{}};

      std::vector total_kinetic_energy_solutions =
      {
         FieldDescriptor{VELOCITY, &H1}
      };

      std::vector total_kinetic_energy_parameters =
      {
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };

      total_kinetic_energy_mf =
         std::make_shared<DifferentiableOperator>(
            total_kinetic_energy_solutions,
            total_kinetic_energy_parameters, mesh);
      TotalKineticEnergyQFunction total_kinetic_energy_qf;
      total_kinetic_energy_mf->AddDomainIntegrator(
         total_kinetic_energy_qf, total_kinetic_energy_kernel_ao,
         total_kinetic_energy_kernel_oo, ir, all_domain_attr);
   }

   // Create density operator
   std::shared_ptr<DifferentiableOperator> density_mf;
   {
      mfem::tuple density_kernel_ao =
      {
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };

      mfem::tuple density_kernel_oo = {Value<DENSITY0>{}};

      std::vector density_solutions =
      {
         FieldDescriptor{DENSITY0, &L2}
      };

      std::vector density_parameters =
      {
         FieldDescriptor{COORDINATES0, &H1}
      };

      density_mf = std::make_shared<DifferentiableOperator>(
                      density_solutions, density_parameters, mesh);

      DensityQFunction density_qf;
      density_mf->AddDomainIntegrator(density_qf, density_kernel_ao,
                                      density_kernel_oo, ir, all_domain_attr);
   }

   // Create taylor source oeprator
   std::shared_ptr<DifferentiableOperator> taylor_source_mf;
   {
      mfem::tuple taylor_source_kernel_ao =
      {
         Value<COORDINATES>{},
         Gradient<COORDINATES>{},
         Weight{}
      };

      mfem::tuple taylor_source_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      std::vector taylor_source_solutions =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      std::vector taylor_source_parameters =
      {
         FieldDescriptor{COORDINATES, &H1}
      };

      taylor_source_mf = std::make_shared<DifferentiableOperator>(
                            taylor_source_solutions, taylor_source_parameters, mesh);

      TaylorSourceQFunction taylor_source_qf;
      auto derivatives = std::integer_sequence<size_t, COORDINATES> {};
      taylor_source_mf->AddDomainIntegrator(taylor_source_qf, taylor_source_kernel_ao,
                                            taylor_source_kernel_oo, ir, all_domain_attr,
                                            derivatives);
   }

   return new LagrangianHydroOperator(
             H1,
             L2,
             ess_tdof,
             ir,
             rho0_coeff,
             x0_gf,
             rho0_gf,
             material_gf,
             update_qdata,
             dt_est_mf,
             momentum_mf,
             momentum_pa,
             energy_conservation_mf,
             energy_conservation_pa,
             total_internal_energy_mf,
             total_kinetic_energy_mf,
             density_mf,
             taylor_source_mf,
             qdata,
             fd_gradient,
             dump_jacobians,
             nonlinear_maximum_iterations,
             nonlinear_relative_tolerance,
             krylov_maximum_iterations,
             preconditioner_lag,
             preconditioner_type,
             external_data);
}


void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec = false)
{
   gf.HostRead();
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         const char* keys = (gf.FESpace()->GetMesh()->Dimension() == 2)
         ? "mAcRjl" : "mmaaAcl";

         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys " << keys;
         if ( vec ) { sock << "vvv"; }
         sock << std::endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   out << std::setprecision(15);

   const char *device_config = "cpu";

   const char *mesh_file = "./rectangle01_quad.mesh";

   int refinements = 0;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   real_t t_final = 0.0;
   real_t blast_energy = 0.25;
   real_t blast_position[] = {0.0, 0.0, 0.0};
   int ode_solver_type = 4;
   bool fd_gradient = false;
   bool use_viscosity = false;
   real_t cfl = 0.5;
   real_t nonlinear_relative_tolerance = 1e-5;
   int nonlinear_maximum_iterations = 10;
   int krylov_maximum_iterations = 10;
   int preconditioner_lag = 0;
   int vis_steps = 1;
   bool glvis = false;
   int viscosity_type = 2;
   int preconditioner_type =
      PRECONDITIONER_TYPE::BLOCK_DIAGONAL_AMG;
   int dump_jacobians = 0;
   int nretry = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&refinements, "-rs", "--ref", "");
   args.AddOption(&order_v, "-ov", "--ov", "");
   args.AddOption(&order_e, "-oe", "--oe", "");
   args.AddOption(&order_q, "-oq", "--oq", "");
   args.AddOption(&t_final, "-tf", "--tf", "");
   args.AddOption(&problem, "-p", "--p", "");
   args.AddOption(&cfl, "-cfl", "--cfl", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_viscosity, "-av", "--av", "-no-av", "--no-av", "");
   args.AddOption(&fd_gradient, "-fd", "--fd", "-no-fd", "--no-fd", "");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg."
                  "            11 - Backward Euler"
                  "            12 - Implicit Midpoint"
                  "            13 - SDIRK33Solver");
   args.AddOption(&nonlinear_maximum_iterations, "-nmi", "--nmi",
                  "Maximum number of nonlinear iterations.");
   args.AddOption(&nonlinear_relative_tolerance, "-nrt", "--nrt",
                  "Nonlinear relative tolerance.");
   args.AddOption(&krylov_maximum_iterations, "-kmi", "--kmi",
                  "Maximum number of Krylov iterations.");
   args.AddOption(&preconditioner_lag, "-pl", "--pl",
                  "Number of nonlinear solves to wait before updating the preconditioner.");
   args.AddOption(&preconditioner_type, "-pt", "--pt",
                  "Preconditioner type: 0 - SuperLU_DIST, 1 - Block Diagonal AMG");
   args.AddOption(&vis_steps, "-vis-steps", "--vis-steps",
                  "Number of visualization steps.");
   args.AddOption(&glvis, "-glvis", "--glvis", "-no-glvis", "--no-glvis", "");
   args.AddOption(&viscosity_type, "-av-type", "--av-type", "");
   args.AddOption(&dump_jacobians, "-dump-jacobians", "--dump-jacobians", "");
   args.AddOption(&nretry, "-nretry", "--nretry", "");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh serial_mesh = Mesh(mesh_file, true, true);

   if (problem == 0 || problem == 1)
   {
      serial_mesh = Mesh(Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                               true));
      const int NBE = serial_mesh.GetNBE();
      for (int b = 0; b < NBE; b++)
      {
         Element *bel = serial_mesh.GetBdrElement(b);
         const int attr = (b < NBE/2) ? 2 : 1;
         bel->SetAttribute(attr);
      }
   }

   if (problem == 2)
   {
      serial_mesh = Mesh(Mesh::MakeCartesian1D(1));
      serial_mesh.GetBdrElement(0)->SetAttribute(1);
      serial_mesh.GetBdrElement(1)->SetAttribute(1);
   }

   for (int i = 0; i < refinements; i++)
   {
      serial_mesh.UniformRefinement();
   }

   // serial_mesh.EnsureNCMesh();
   // serial_mesh.RandomRefinement(0.1);

   ParMesh mesh = ParMesh(MPI_COMM_WORLD, serial_mesh);
   const int dim = mesh.Dimension();

   MFEM_ASSERT(dim == DIMENSION, "mesh dimension inconsistency");

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace H1FESpace(&mesh, &H1FEC, dim);
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   ParFiniteElementSpace L2FESpace(&mesh, &L2FEC);

   const auto global_ne = mesh.GetGlobalNE();
   const auto global_h1tsize = H1FESpace.GlobalTrueVSize();
   const auto global_l2tsize = L2FESpace.GlobalTrueVSize();

   if (Mpi::Root())
   {
      out << "num el: " << global_ne << "\n";
      out << "num kinematic dofs: " << global_h1tsize << "\n";
      out << "num thermodynamic dofs: " << global_l2tsize << "\n";
   }

   Array<int> ess_tdof, ess_vdofs;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max()), dofs_marker, dofs_list;
      for (int d = 0; d < mesh.Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries,
         // i.e., we must enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, d);
         ess_tdof.Append(dofs_list);
         H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, d);
         FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
         ess_vdofs.Append(dofs_list);
      }
   }

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy
   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();
   Array<int> offset(4);
   offset[0] = 0;
   offset[1] = offset[0] + Vsize_h1;
   offset[2] = offset[1] + Vsize_h1;
   offset[3] = offset[2] + Vsize_l2;
   BlockVector S(offset, Device::GetDeviceMemoryType());

   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, offset[0]);
   v_gf.MakeRef(&H1FESpace, S, offset[1]);
   e_gf.MakeRef(&L2FESpace, S, offset[2]);

   mesh.SetNodalGridFunction(&x_gf);
   x_gf.SyncAliasMemory(S);

   ParGridFunction x0_gf = x_gf;

   auto v0 = [](const Vector &x, Vector &v)
   {
      switch (problem)
      {
         case 0:
            v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
            v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
            if (x.Size() == 3)
            {
               v(0) *= cos(M_PI*x(2));
               v(1) *= cos(M_PI*x(2));
               v(2) = 0.0;
            }
            break;
         case 1: v = 0.0; break;
         case 2: v = 0.0; break;
         case 3: v = 0.0; break;
         default: MFEM_ABORT("error");
      }
   };

   VectorFunctionCoefficient v_coeff(dim, v0);
   v_gf.ProjectCoefficient(v_coeff);
   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      v_gf(ess_vdofs[i]) = 0.0;
   }
   v_gf.SyncAliasMemory(S);

   auto rho0 = [&dim](const Vector &x)
   {
      switch (problem)
      {
         case 0: return 1.0;
         case 1: return 1.0;
         case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
         case 3: return (dim == 2) ? (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0
                           : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
                                            (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;
         default: MFEM_ABORT("error");
      }
   };

   ParGridFunction rho0_gf(&L2FESpace);
   FunctionCoefficient rho0_coeff(rho0);
   L2_FECollection l2_fec(order_e, mesh.Dimension());
   ParFiniteElementSpace l2_fes(&mesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes);
   l2_rho0_gf.ProjectCoefficient(rho0_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);

   auto gamma_func = [](const Vector &x)
   {
      switch (problem)
      {
         case 0: return 5.0 / 3.0;
         case 1: return 1.4;
         case 2: return 1.4;
         case 3: return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5;
         default: MFEM_ABORT("error");
      }
   };

   auto e0 = [&rho0, &gamma_func](const Vector &x)
   {
      switch (problem)
      {
         case 0:
         {
            const real_t denom = 2.0 / 3.0;  // (5/3 - 1) * density.
            real_t val;
            if (x.Size() == 2)
            {
               val = 1.0 + (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) / 4.0;
            }
            else
            {
               val = 100.0 + ((cos(2*M_PI*x(2)) + 2) *
                              (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) - 2) / 16.0;
            }
            return val/denom;
         }
         case 1: return 0.0; // This case in initialized in main().
         case 2: return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma_func(x) - 1.0)
                           : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
         case 3: return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma_func(x) - 1.0)
                           : 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         default: MFEM_ABORT("error");
      }
   };

   if (problem == 1)
   {
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                               blast_position[2], blast_energy);
      l2_e.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }

   e_gf.ProjectGridFunction(l2_e);
   e_gf.SyncAliasMemory(S);

   L2_FECollection material_fec(0, dim);
   ParFiniteElementSpace L2CFESpace(&mesh, &material_fec);
   ParGridFunction material_gf(&L2CFESpace);
   FunctionCoefficient material_coeff(gamma_func);
   material_gf.ProjectCoefficient(material_coeff);

   ParGridFunction rho_gf(&L2FESpace);

   IntegrationRule ir = IntRules.Get(mesh.GetElementBaseGeometry(0),
                                     3 * H1FESpace.GetOrder(0) + L2FESpace.GetOrder(0) - 1);

   if (Mpi::Root())
   {
      out << "num qp: " << ir.GetNPoints() << "\n";
   }

   // Create external data vector
   // Layout is [cfl, order_velocity, use_viscosity, h0]
   Vector external_data(EXT_DATA_IDX::COUNT);
   external_data[EXT_DATA_IDX::CFL] = cfl;
   external_data[EXT_DATA_IDX::ORDER_VEL] = order_v;
   external_data[EXT_DATA_IDX::VISCOSITY_FLAG] = use_viscosity;
   external_data[EXT_DATA_IDX::VISCOSITY_TYPE] = viscosity_type;
   external_data[EXT_DATA_IDX::DT_ESTIMATE] =
      std::numeric_limits<real_t>::infinity();

   auto hydro = CreateLagrangianHydroOperator(H1FESpace,
                                              L2FESpace,
                                              ess_tdof,
                                              rho0_coeff,
                                              x0_gf,
                                              rho0_gf,
                                              material_gf,
                                              external_data,
                                              ir,
                                              fd_gradient,
                                              dump_jacobians,
                                              nonlinear_maximum_iterations,
                                              nonlinear_relative_tolerance,
                                              krylov_maximum_iterations,
                                              preconditioner_lag,
                                              (PRECONDITIONER_TYPE)preconditioner_type);

   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new ImplicitMidpointSolver; break;
      case 13: ode_solver = new SDIRK33Solver; break;
      case 14: ode_solver = new SDIRK34Solver; break;
      default:
         out << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return -1;
   }
   ode_solver->Init(*hydro);

   hydro->ComputeDensity(rho_gf);
   const real_t energy_init = hydro->InternalEnergy(e_gf) +
                              hydro->KineticEnergy(v_gf);

   socketstream vis_rho, vis_v, vis_e;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (glvis)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(mesh.GetComm());
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets
      if (problem != 0 && problem != 4)
      {
         VisualizeField(vis_rho, vishost, visport, rho_gf,
                        "Density", Wx, Wy, Ww, Wh);
      }
      Wx += offx;
      VisualizeField(vis_v, vishost, visport, v_gf,
                     "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(vis_e, vishost, visport, e_gf,
                     "Specific Internal Energy", Wx, Wy, Ww, Wh);
   }

   if (Mpi::Root())
   {
      out << "IE: " << energy_init << "\n";
   }

   // out << "IE " << hydro.InternalEnergy(e_gf) << "\n"
   //     << "KE "<< hydro.KineticEnergy(v_gf) << "\n";

   external_data[EXT_DATA_IDX::CFL] = 0.5;
   hydro->ResetTimeStepEstimate();
   real_t t = 0.0;
   real_t dt = hydro->GetTimeStepEstimate(S);
   external_data[EXT_DATA_IDX::CFL] = cfl;

   if (Mpi::Root())
   {
      out << "time step estimate: " << dt << "\n";
   }

   real_t t_old;
   bool last_step = false;
   [[maybe_unused]] int steps = 0;
   BlockVector S_old(S);

   ParGridFunction verr_gf(v_gf);
   verr_gf.ProjectCoefficient(v_coeff);
   v_gf.SyncAliasMemory(S);
   v_gf.HostRead();
   verr_gf.HostReadWrite();
   for (int i = 0; i < verr_gf.Size(); i++)
   {
      verr_gf(i) = abs(verr_gf(i) - std::as_const(v_gf)(i));
   }

   QuadratureSpace qs(mesh, ir);

   QuadratureFunction vqf(&qs, v_gf.VectorDim());
   vqf.ProjectGridFunction(v_gf);

   QuadratureFunction rqf(&qs, rho_gf.VectorDim());
   rqf.ProjectGridFunction(rho_gf);

   QuadratureFunction eqf(&qs, e_gf.VectorDim());
   eqf.ProjectGridFunction(e_gf);

   ParaViewDataCollection dc("dfem", &mesh);
   dc.SetLevelsOfDetail(order_v);
   dc.SetDataFormat(VTKFormat::BINARY);
   dc.SetHighOrderOutput(true);
   dc.SetCycle(0);
   dc.SetTime(0.0);
   dc.RegisterField("velocity", &v_gf);
   dc.RegisterField("density", &rho_gf);
   dc.RegisterField("specific_internal_energy", &e_gf);
   dc.RegisterField("material", &material_gf);
   dc.RegisterQField("velocity_qf", &rqf);
   dc.RegisterQField("density_qf", &rqf);
   dc.RegisterQField("specific_internal_energy_qf", &eqf);

   dc.SetCycle(0);
   dc.SetTime(0);
   dc.Save();

   auto vizcb = [&](const int &ti, const real_t &t)
   {
      hydro->ComputeDensity(rho_gf);
      vqf.ProjectGridFunction(v_gf);
      rqf.ProjectGridFunction(rho_gf);
      dc.SetCycle(ti);
      dc.SetTime(t);
      dc.Save();
   };

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }

      if (last_step)
      {
         // hydro->residual->dump_jacobians = 1;
      }

      S_old = S;
      t_old = t;
      hydro->ResetTimeStepEstimate();

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);
      steps++;

      // out << ">>> S:\n";
      // pretty_print(S);

      // Adaptive time step control.
      const real_t dt_est = hydro->GetTimeStepEstimate(S);
      // out << "dt_est: " << dt_est << "\n";

      if (ode_solver_type > 10)
      {
         if (dt_est < dt || !hydro->newton_converged)
         {
            if (!hydro->newton_converged)
            {
               if (Mpi::Root())
               {
                  out << "writing viz for non converged newton step " << ti << " at time=" << t <<
                      "\n";
               }
               // vizcb(ti+100, t+100.0);
            }

            dt *= 0.85;
            t = t_old;
            S = S_old;
            hydro->ResetQuadratureData();
            last_step = false;
            if (Mpi::Root())
            {
               out << "Repeating step " << ti << " with dt: " << dt << "." << std::endl;
               out << "Reason: ";
               if (!hydro->newton_converged)
               {
                  out << "Newton did not converge in " << hydro->newton->GetNumIterations() <<
                      " iterations. ";
               }
               else
               {
                  out << "Estimated time step lower than taken time step.";
               }
               out << std::endl;
            }

            if (!hydro->newton_converged)
            {
               if (nretry == 1)
               {
                  hydro->residual->dump_jacobians = 1;
               }
               else if (nretry == 0)
               {
                  if (Mpi::Root())
                  {
                     out << "no retry planned, exit\n";
                  }
                  exit(1);
               }
               nretry--;
            }

            ti--; continue;
         }
         else if (dt_est > 1.25 * dt) { dt *= 1.15; }
      }
      else if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < std::numeric_limits<real_t>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         hydro->ResetQuadratureData();
         last_step = false;
         if (Mpi::Root()) { out << "Repeating step " << ti << std::endl; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      mesh.NewNodes(x_gf, false);

      // out << "x_gf outer loop\n";
      // print_vector(x_gf);

      if (Mpi::Root())
      {
         out << "step " << std::setw(5) << ti
             << ",\tt = " << std::setw(5) << t
             << ",\tdt = " << std::setw(5) << dt;
         out << std::endl;
      }

      // verr_gf.ProjectCoefficient(v_coeff);
      // for (int i = 0; i < verr_gf.Size(); i++)
      // {
      //    verr_gf(i) = abs(verr_gf(i) - v_gf(i));
      // }

      if (ti % vis_steps == 0 || last_step)
      {
         vizcb(ti, t);

         if (glvis)
         {
            hydro->ComputeDensity(rho_gf);
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets
            if (problem != 0 && problem != 4)
            {
               VisualizeField(vis_rho, vishost, visport, rho_gf,
                              "Density", Wx, Wy, Ww, Wh);
            }
            Wx += offx;
            VisualizeField(vis_v, vishost, visport,
                           v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            VisualizeField(vis_e, vishost, visport, e_gf,
                           "Specific Internal Energy", Wx, Wy, Ww,Wh);
            Wx += offx;
         }
      }
   }

   const real_t energy_final = hydro->InternalEnergy(e_gf)
                               + hydro->KineticEnergy(v_gf);

   if (Mpi::Root())
   {
      out << std::scientific << std::setprecision(2)
          << "Energy diff: " << fabs(energy_init - energy_final) << std::endl;
   }

   if (problem == 0)
   {
      const real_t v_err_max = v_gf.ComputeMaxError(v_coeff);
      const real_t v_err_l1 = v_gf.ComputeL1Error(v_coeff);
      const real_t v_err_l2 = v_gf.ComputeL2Error(v_coeff);
      if (Mpi::Root())
      {
         out << "L_inf  error: " << v_err_max << std::endl
             << "L_1    error: " << v_err_l1 << std::endl
             << "L_2    error: " << v_err_l2 << std::endl;
      }
   }

   delete hydro;

   return 0;
}
