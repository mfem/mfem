//                  MFEM Example 42 - Serial/Parallel Shared Code
//                      (Implementation of Time-dependent DG Operator)
//
// This code provide example problems for the Euler equations and implements
// the time-dependent DG operator given by the equation:
//
//            (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u, n), [[v]]>_F = 0.
//
// This operator is designed for explicit time stepping methods. Specifically,
// the function DGHyperbolicConservationLaws::Mult implements the following
// transformation:
//
//                             u ↦ M⁻¹(-DF(u) + NF(u))
//
// where M is the mass matrix, DF is the weak divergence of flux, and NF is the
// interface flux.
//

#include "mfem.hpp"

#include "fem/kernel_dispatch.hpp"
#include "general/forall.hpp"

#include <functional>

namespace mfem
{

class EulerInterior : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes = nullptr;
   real_t gas_gamma;

public:
   using BilinearFormIntegrator::AssembleMF;
   void AssembleMF(const FiniteElementSpace &fes) override { this->fes = &fes; }

   void AddMultMF(const Vector &x, Vector &y) const override;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         const ElementTransformation &Trans)
   {
      int order =
         Trans.OrderGrad(&trial_fe) + Trans.Order() + test_fe.GetOrder();

      return IntRules.Get(trial_fe.GetGeomType(), order);
   }

   static const IntegrationRule &GetRule(const FiniteElement &el,
                                         const ElementTransformation &Trans)
   {
      return GetRule(el, el, Trans);
   }

   /// args: ne, W, J, B, Bt, G, Gt, x, y, dofs1D, quad1D
   using ApplyMFKernelType = void (*)(const int, const Array<real_t> &,
                                      const Vector &, const Array<real_t> &,
                                      const Array<real_t> &,
                                      const Array<real_t> &,
                                      const Array<real_t> &, const Vector &,
                                      Vector &, const int, const int);

   // args: dims, nq1d, ndof1d
   MFEM_REGISTER_KERNELS(ApplyMFKernels, ApplyMFKernelType, (int, int, int));

   struct Kernels
   {
      Kernels();
   };

   template <int DIM, int D1D, int Q1D> static void AddSpecialization()
   {
      ApplyMFKernels::Specialization<DIM, D1D, Q1D>::Add();
   }

   EulerInterior(real_t gas_gamma, const IntegrationRule *ir = nullptr)
      : BilinearFormIntegrator(ir), gas_gamma(gas_gamma)
   {
      static Kernels kernels;
   }

protected:
   const IntegrationRule *
   GetDefaultIntegrationRule(const FiniteElement &trial_fe,
                             const FiniteElement &test_fe,
                             const ElementTransformation &trans) const override
   {
      return &GetRule(trial_fe, test_fe, trans);
   }
};

class EulerRusanov : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes = nullptr;
   real_t gas_gamma;
   FaceType type;

public:
   void AssembleMFInteriorFaces(const FiniteElementSpace &fes) override
   {
      this->fes = &fes;
      type = FaceType::Interior;
   }

   void AssembleMFBoundaryFaces(const FiniteElementSpace &fes) override
   {
      this->fes = &fes;
      type = FaceType::Boundary;
   }

   void AddMultMF(const Vector &x, Vector &y) const override;

   static const IntegrationRule &GetRule(Geometry::Type geom, int order,
                                         const FaceElementTransformations &T)
   {
      return GetRule(geom, order, *T.Elem1);
   }

   static const IntegrationRule &GetRule(Geometry::Type geom, int order,
                                         const ElementTransformation &T)
   {
      return IntRules.Get(geom, T.OrderW() + 2 * order);
   }

   /// args: nf, W, detJ, normal, B, Bt, x, y, dofs1D, quad1D
   using ApplyMFKernelType = void (*)(const int, const Array<real_t> &,
                                      const Vector &, const Vector &,
                                      const Array<real_t> &,
                                      const Array<real_t> &, const Vector &,
                                      Vector &, const int, const int);

   // args: dims, nq1d, ndof1d
   MFEM_REGISTER_KERNELS(ApplyMFKernels, ApplyMFKernelType, (int, int, int));

   struct Kernels
   {
      Kernels();
   };

   template <int DIM, int D1D, int Q1D> static void AddSpecialization()
   {
      ApplyMFKernels::Specialization<DIM, D1D, Q1D>::Add();
   }

   EulerRusanov(real_t gas_gamma, const IntegrationRule *ir = nullptr)
      : BilinearFormIntegrator(ir), gas_gamma(gas_gamma)
   {
      static Kernels kernels;
   }
};

namespace internal
{}

void EulerInterior::AddMultMF(const Vector &x, Vector &y) const
{
   const MemoryType mt =
      (pa_mt == MemoryType::DEFAULT) ? Device::GetDeviceMemoryType() : pa_mt;
   Mesh *mesh = fes->GetMesh();
   const FiniteElement &el = *fes->GetTypicalFE();
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Trans);
   auto geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   auto nq = ir->GetNPoints();
   auto dim = mesh->Dimension();
   auto ne = fes->GetNE();
   auto maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   auto dofs1D = maps->ndof;
   auto quad1D = maps->nqpt;
   const Array<real_t> &B = maps->B;
   const Array<real_t> &Bt = maps->Bt;
   const Array<real_t> &G = maps->G;
   const Array<real_t> &Gt = maps->Gt;
   const Array<real_t> &W = ir->GetWeights();
   const Vector &J = geom->J;
   ApplyMFKernels::Run(dim, dofs1D, quad1D, ne, W, J, B, Bt, G, Gt, x, y,
                       dofs1D, quad1D);
}

void EulerRusanov::AddMultMF(const Vector &x, Vector &y) const
{
   const MemoryType mt =
      (pa_mt == MemoryType::DEFAULT) ? Device::GetDeviceMemoryType() : pa_mt;
   Mesh *mesh = fes->GetMesh();
   const FiniteElement &el = *fes->GetTypicalFE();
   const IntegrationRule *ir =
      IntRule ? IntRule
      : &GetRule(el.GetGeomType(), el.GetOrder(),
                 *mesh->GetTypicalElementTransformation());
   auto nf = mesh->GetNFbyType(type);
   auto nq = ir->GetNPoints();
   auto geom = mesh->GetFaceGeometricFactors(
                  *ir, FaceGeometricFactors::DETERMINANTS | FaceGeometricFactors::NORMALS,
                  type, mt);
   auto dim = mesh->Dimension();
   auto ne = fes->GetNE();
   auto maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   auto dofs1D = maps->ndof;
   auto quad1D = maps->nqpt;
   const Array<real_t> &B = maps->B;
   const Array<real_t> &Bt = maps->Bt;
   const Array<real_t> &W = ir->GetWeights();
   const Vector &detJ = geom->detJ;
   const Vector &normal = geom->normal;
   ApplyMFKernels::Run(dim, dofs1D, quad1D, nf, W, detJ, normal, B, Bt, x, y,
                       dofs1D, quad1D);
}

template <int DIM, int T_D1D, int T_Q1D>
EulerInterior::ApplyMFKernelType EulerInterior::ApplyMFKernels::Kernel()
{
   if constexpr (DIM == 1)
   {
      // TODO
   }
   else if constexpr (DIM == 2)
   {
      // TODO
   }
   else if constexpr (DIM == 3)
   {
      // TODO
   }
   MFEM_ABORT("");
}

inline EulerInterior::ApplyMFKernelType
EulerInterior::ApplyMFKernels::Fallback(int DIM, int, int)
{
   switch (DIM)
   {
      case 1:
      // TODO
      case 2:
      // TODO
      case 3:
      // TODO
      default:
         MFEM_ABORT("");
   }
}

template <int DIM, int T_D1D, int T_Q1D>
EulerRusanov::ApplyMFKernelType EulerRusanov::ApplyMFKernels::Kernel()
{
   if constexpr (DIM == 1)
   {
      // TODO
   }
   else if constexpr (DIM == 2)
   {
      // TODO
   }
   else if constexpr (DIM == 3)
   {
      // TODO
   }
   MFEM_ABORT("");
}

inline EulerRusanov::ApplyMFKernelType
EulerRusanov::ApplyMFKernels::Fallback(int DIM, int, int)
{
   switch (DIM)
   {
      case 1:
      // TODO
      case 2:
      // TODO
      case 3:
      // TODO
      default:
         MFEM_ABORT("");
   }
}

std::function<void(const Vector &, Vector &)>
GetMovingVortexInit(const real_t radius, const real_t Minf, const real_t beta,
                    const real_t gas_constant, const real_t specific_heat_ratio)
{
   return [specific_heat_ratio, gas_constant, Minf, radius,
                                beta](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2, "");

      const real_t xc = 0.0, yc = 0.0;

      // Nice units
      const real_t vel_inf = 1.;
      const real_t den_inf = 1.;

      // Derive remainder of background state from this and Minf
      const real_t pres_inf =
         (den_inf / specific_heat_ratio) * (vel_inf / Minf) * (vel_inf / Minf);
      const real_t temp_inf = pres_inf / (den_inf * gas_constant);

      real_t r2rad = 0.0;
      r2rad += (x(0) - xc) * (x(0) - xc);
      r2rad += (x(1) - yc) * (x(1) - yc);
      r2rad /= (radius * radius);

      const real_t shrinv1 = 1.0 / (specific_heat_ratio - 1.);

      const real_t velX =
         vel_inf * (1 - beta * (x(1) - yc) / radius * std::exp(-0.5 * r2rad));
      const real_t velY =
         vel_inf * beta * (x(0) - xc) / radius * std::exp(-0.5 * r2rad);
      const real_t vel2 = velX * velX + velY * velY;

      const real_t specific_heat = gas_constant * specific_heat_ratio * shrinv1;
      const real_t temp = temp_inf - 0.5 * (vel_inf * beta) * (vel_inf * beta) /
                          specific_heat * std::exp(-r2rad);

      const real_t den = den_inf * std::pow(temp / temp_inf, shrinv1);
      const real_t pres = den * gas_constant * temp;
      const real_t energy = shrinv1 * pres / den + 0.5 * vel2;

      y(0) = den;
      y(1) = den * velX;
      y(2) = den * velY;
      y(3) = den * energy;
   };
}

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
      case 2:
      case 3:
         return Mesh("../data/periodic-square.mesh");
         break;
      case 4:
         return Mesh("../data/periodic-segment.mesh");
         break;
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

// Initial condition
VectorFunctionCoefficient
EulerInitialCondition(const int problem, const real_t specific_heat_ratio,
                      const real_t gas_constant)
{
   switch (problem)
   {
      case 1: // fast moving vortex
         return VectorFunctionCoefficient(
                   4, GetMovingVortexInit(0.2, 0.5, 1. / 5., gas_constant,
                                          specific_heat_ratio));
      case 2: // slow moving vortex
         return VectorFunctionCoefficient(
                   4, GetMovingVortexInit(0.2, 0.05, 1. / 50., gas_constant,
                                          specific_heat_ratio));
      case 3: // moving sine wave
         return VectorFunctionCoefficient(4, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            const real_t density = 1.0 + 0.2 * std::sin(M_PI * (x(0) + x(1)));
            const real_t velocity_x = 0.7;
            const real_t velocity_y = 0.3;
            const real_t pressure = 1.0;
            const real_t energy =
               pressure / (1.4 - 1.0) +
               density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = energy;
         });
      case 4:
         return VectorFunctionCoefficient(3, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 1, "");
            const real_t density = 1.0 + 0.2 * std::sin(M_PI * 2 * x(0));
            const real_t velocity_x = 1.0;
            const real_t pressure = 1.0;
            const real_t energy =
               pressure / (1.4 - 1.0) + density * 0.5 * (velocity_x * velocity_x);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = energy;
         });
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

EulerInterior::Kernels::Kernels()
{
   // 2D
   EulerInterior::AddSpecialization<2, 1, 1>();
   EulerInterior::AddSpecialization<2, 2, 2>();
   EulerInterior::AddSpecialization<2, 3, 3>();
   EulerInterior::AddSpecialization<2, 4, 4>();

   EulerInterior::AddSpecialization<2, 1, 2>();
   EulerInterior::AddSpecialization<2, 2, 3>();
   EulerInterior::AddSpecialization<2, 3, 4>();
   EulerInterior::AddSpecialization<2, 4, 5>();
   // 3D
   EulerInterior::AddSpecialization<3, 1, 1>();
   EulerInterior::AddSpecialization<3, 2, 2>();
   EulerInterior::AddSpecialization<3, 3, 3>();
   EulerInterior::AddSpecialization<3, 4, 4>();

   EulerInterior::AddSpecialization<3, 1, 2>();
   EulerInterior::AddSpecialization<3, 2, 3>();
   EulerInterior::AddSpecialization<3, 3, 4>();
   EulerInterior::AddSpecialization<3, 4, 5>();
}

EulerRusanov::Kernels::Kernels()
{
   // 2D
   EulerRusanov::AddSpecialization<2, 1, 1>();
   EulerRusanov::AddSpecialization<2, 2, 2>();
   EulerRusanov::AddSpecialization<2, 3, 3>();
   EulerRusanov::AddSpecialization<2, 4, 4>();

   EulerRusanov::AddSpecialization<2, 1, 2>();
   EulerRusanov::AddSpecialization<2, 2, 3>();
   EulerRusanov::AddSpecialization<2, 3, 4>();
   EulerRusanov::AddSpecialization<2, 4, 5>();
   // 3D
   EulerRusanov::AddSpecialization<3, 1, 1>();
   EulerRusanov::AddSpecialization<3, 2, 2>();
   EulerRusanov::AddSpecialization<3, 3, 3>();
   EulerRusanov::AddSpecialization<3, 4, 4>();

   EulerRusanov::AddSpecialization<3, 1, 2>();
   EulerRusanov::AddSpecialization<3, 2, 3>();
   EulerRusanov::AddSpecialization<3, 3, 4>();
   EulerRusanov::AddSpecialization<3, 4, 5>();
}

} // namespace mfem
