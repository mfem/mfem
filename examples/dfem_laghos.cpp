#include "dfem/dfem_refactor.hpp"
#include "examples/dfem/dfem_util.hpp"
#include "fem/bilininteg.hpp"
#include "fem/pbilinearform.hpp"
#include "fem/pgridfunc.hpp"
#include "linalg/handle.hpp"
#include "linalg/solvers.hpp"
#include <limits>
#include <memory>

using namespace mfem;
using mfem::internal::tensor;

constexpr int VELOCITY = 0;
constexpr int DENSITY0 = 1;
constexpr int COORDINATES0 = 2;
constexpr int COORDINATES = 3;
constexpr int MATERIAL = 4;
constexpr int SPECIFIC_INTERNAL_ENERGY = 5;
constexpr int ELEMENT_SIZE0 = 6;
constexpr int ORDER_VEL = 7;
constexpr int DT_EST = 8;
constexpr int STRESS_TENSOR = 9;

constexpr int DIMENSION = 1;

int problem = 0;
real_t cfl = 0.5;
bool use_viscosity = false;

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

template <bool compute_dtest = false>
MFEM_HOST_DEVICE inline
mfem::tuple<matd, real_t> qdata_setup(
   const matd &dvdxi,
   const real_t &rho0,
   const matd &J0,
   const matd &J,
   const real_t &gamma,
   const real_t &E,
   const real_t &h0,
   const real_t &order_v,
   const real_t &w)
{
   constexpr real_t eps = 1e-12;
   constexpr real_t vorticity_coeff = 1.0;
   real_t p, cs;
   real_t detJ = det(J);
   matd invJ = inv(J);
   matd stress{0.0};
   const real_t rho = rho0 * det(J0) / detJ;
   const real_t Ez = fmax(0.0, E);
   real_t visc_coeff = 0.0;
   real_t dt_est = std::numeric_limits<real_t>::infinity();

   ComputeMaterialProperties(gamma, rho, Ez, p, cs);

   for (int d = 0; d < DIMENSION; d++)
   {
      stress(d, d) = -p;
   }

   if (use_viscosity)
   {
      auto symdvdx = sym(dvdxi * invJ);
      auto [eigvals, eigvecs] = eig(symdvdx);
      vecd compr_dir = get_col(eigvecs, 0);
      auto ph_dir = (J * inv(J0)) * compr_dir;
      const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
      // Measure of maximal compression.
      const real_t mu = eigvals(0);
      visc_coeff = 2.0 * rho * h * h * fabs(mu);
      visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
                    (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
      stress += visc_coeff * symdvdx;
   }

   if constexpr (compute_dtest)
   {
      if (detJ < 0.0)
      {
         // This will force repetition of the step with smaller dt.
         dt_est = 0.0;
      }
      else
      {
         const real_t h_min = calcsv(J, DIMENSION-1) / static_cast<real_t>(order_v);
         const real_t idt = cs / h_min + 2.5 * visc_coeff / rho / h_min / h_min;

         if (idt > 0.0)
         {
            dt_est = cfl / idt;
         }
         else
         {
            dt_est = std::numeric_limits<real_t>::infinity();
         }
      }
   }

   matd stressJiT = stress * transpose(invJ) * detJ * w;
   return mfem::tuple{stressJiT, dt_est};
}

struct TimeStepEstimateQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &h0,
      const real_t &order_v,
      const real_t &w) const
   {
      real_t dt_est = mfem::get<1>(
                         qdata_setup<true>(dvdxi, rho0, J0, J, gamma, E, h0, order_v, w));
      return mfem::tuple{dt_est};
   }
};

struct UpdateQuadratureDataQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &h0,
      const real_t &order_v,
      const real_t &w) const
   {
      matd stressJiT = mfem::get<0>(
                          qdata_setup<false>(dvdxi, rho0, J0, J, gamma, E, h0, order_v, w));
      return mfem::tuple{stressJiT};
   }
};

class MomentumQFunction
{
public:
   MomentumQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &h0,
      const real_t &order_v,
      const real_t &w) const
   {
      auto stressJiT = mfem::get<0>(
                          qdata_setup(dvdxi, rho0, J0, J, gamma, E, h0, order_v, w));

      // out << gamma << " " << rho << " " << Ez << " " << p << " " << cs << "\n";
      // out << stressJiT << "\n";
      // TODO-bug: investigate transpose of matrices in return types
      // return mfem::tuple{transpose(stressJiT)};
      return mfem::tuple{stressJiT};
   }
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
   EnergyConservationQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &h0,
      const real_t &order_v,
      const real_t &w) const
   {
      auto stressJiT = mfem::get<0>(
                          qdata_setup(dvdxi, rho0, J0, J, gamma, E, h0, order_v, w));
      return mfem::tuple{ddot(stressJiT, dvdxi)};
   }
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
      h0(R),
      order_v(R),
      dt_est(R)
   {
      h0.UseDevice(true);
      order_v.UseDevice(true);
      dt_est.UseDevice(true);
      stressp.UseDevice(true);
   }

   ParametricSpace StressSpace;
   ParametricFunction stressp;

   ParametricSpace R;
   ParametricFunction h0, order_v, dt_est;
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

class LagrangianHydroJacobianOperator : public Operator
{
public:
   LagrangianHydroJacobianOperator(real_t h, int H1tsize, int L2tsize) :
      Operator(2*H1tsize + L2tsize), h(h), H1tsize(H1tsize), L2tsize(L2tsize)  {}

   void Mult(const Vector &k, Vector &y) const override
   {
      jvp(k, y);
   }

   template <typename hydro_t>
   void Setup(hydro_t &hydro,
              std::shared_ptr<DerivativeOperator> dRvdx,
              std::shared_ptr<DerivativeOperator> dRvdv,
              std::shared_ptr<DerivativeOperator> dRvde,
              std::shared_ptr<DerivativeOperator> dRedx,
              std::shared_ptr<DerivativeOperator> dRedv,
              std::shared_ptr<DerivativeOperator> dRede)
   {
      w.SetSize(this->height);
      z.SetSize(this->height);

      jvp = [dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede, this, &hydro]
            (const Vector &u, Vector &y)
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
         zv *= h;
         yv = zv;
         dRvdv->Mult(wv, zv);
         zv *= h;
         yv += zv;
         // hydro.Mv.TrueAddMult(wv, yv);
         Vector wvc, yvc;
         for (int c = 0; c < hydro.H1.GetMesh()->Dimension(); c++)
         {
            wvc.MakeRef(wv, c*hydro.H1c.GetVSize(), hydro.H1c.GetVSize());
            yvc.MakeRef(yv, c*hydro.H1c.GetVSize(), hydro.H1c.GetVSize());
            hydro.Mv->FullAddMult(wvc, yvc);
            yvc.SyncAliasMemory(yv);
         }
         yv.SyncAliasMemory(y);

         dRvde->Mult(we, zv);
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
         ze *= -h;
         ye = ze;

         dRedv->Mult(wv, ze);
         ze *= -h;
         ye += ze;

         dRede->Mult(we, ze);
         ze *= -h;
         // hydro.Me.TrueAddMult(we, ze);
         hydro.Me->FullAddMult(we, ze);

         ye += ze;

         yx.SyncAliasMemory(y);
         yv.SyncAliasMemory(y);
         ye.SyncAliasMemory(y);
      };
   }

   virtual MemoryClass GetMemoryClass() const override
   {
      return Device::GetDeviceMemoryClass();
   }

   real_t h;
   std::function<void(const Vector &, Vector &)> jvp;
   const int H1tsize;
   const int L2tsize;
   Vector w, z;
};

template <typename hydro_t>
class LagrangianHydroResidualOperator : public Operator
{
public:
   LagrangianHydroResidualOperator(hydro_t &hydro, const real_t dt,
                                   const Vector &x, bool fd_gradient) :
      Operator(2*hydro.H1.GetTrueVSize()+hydro.L2.GetTrueVSize()),
      hydro(hydro),
      dt(dt),
      x(x),
      u(x.Size()),
      H1tsize(hydro.H1.GetTrueVSize()),
      L2tsize(hydro.L2.GetTrueVSize()),
      fd_gradient(fd_gradient) {}

   void Mult(const Vector &k, Vector &R) const override
   {
      hydro.UpdateMesh(u);

      u = k;
      u *= dt;
      u += x;

      hydro.mesh_nodes.SyncMemory(u);

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

      Rx = kx;
      Rx -= uv;

      hydro.momentum_mf->SetParameters({&hydro.rho0, &hydro.x0, &ux, &hydro.material, &ue, &hydro.qdata->h0, &hydro.qdata->order_v});
      hydro.momentum_mf->Mult(uv, Rv);

      // hydro.Mv.TrueAddMult(kv, Rv);
      Vector kvc, Rvc;
      for (int c = 0; c < hydro.H1.GetMesh()->Dimension(); c++)
      {
         kvc.MakeRef(kv, c*hydro.H1c.GetVSize(), hydro.H1c.GetVSize());
         Rvc.MakeRef(Rv, c*hydro.H1c.GetVSize(), hydro.H1c.GetVSize());
         hydro.Mv->FullAddMult(kvc, Rvc);
         Rvc.SyncAliasMemory(Rv);
      }
      Rv.SyncAliasMemory(R);

      Rv.SetSubVector(hydro.ess_tdof, 0.0);
      // Rv = 0.0;

      hydro.energy_conservation_mf->SetParameters({&uv, &hydro.rho0, &hydro.x0, &ux, &hydro.material, &hydro.qdata->h0, &hydro.qdata->order_v});
      hydro.energy_conservation_mf->Mult(ue, Re);

      Re.Neg();

      if (problem == 0)
      {
         LinearForm e_source(&hydro.L2);
         hydro.L2.GetMesh()->DeleteGeometricFactors();
         FunctionCoefficient coeff(taylor_source);
         DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &hydro.ir);
         e_source.AddDomainIntegrator(d);
         e_source.UseFastAssembly(true);
         e_source.Assemble();

         Re -= e_source;
      }

      // hydro.Me.TrueAddMult(ke, Re);
      hydro.Me->FullAddMult(ke, Re);

      Rx.SyncAliasMemory(R);
      Rv.SyncAliasMemory(R);
      Re.SyncAliasMemory(R);
   }

   Operator& GetGradient(const Vector &k) const override
   {
      jacobian.reset(new LagrangianHydroJacobianOperator(dt, H1tsize, L2tsize));

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

      if (fd_gradient)
      {
         fd_jacobian.reset(new FDJacobian(*this, k));
         return *fd_jacobian;
      }
      else
      {
         auto dRvdx = hydro.momentum_mf->GetDerivative(COORDINATES, {&uv},
         {&hydro.rho0, &hydro.x0, &ux, &hydro.material, &ue, &hydro.qdata->h0, &hydro.qdata->order_v});

         auto dRvdv = hydro.momentum_mf->GetDerivative(VELOCITY, {&uv},
         {&hydro.rho0, &hydro.x0, &ux, &hydro.material, &ue, &hydro.qdata->h0, &hydro.qdata->order_v});

         auto dRvde = hydro.momentum_mf->GetDerivative(SPECIFIC_INTERNAL_ENERGY, {&uv},
         {&hydro.rho0, &hydro.x0, &ux, &hydro.material, &ue, &hydro.qdata->h0, &hydro.qdata->order_v});

         auto dRedx = hydro.energy_conservation_mf->GetDerivative(COORDINATES, {&ue},
         {&uv, &hydro.rho0, &hydro.x0, &ux, &hydro.material, &hydro.qdata->h0, &hydro.qdata->order_v});

         auto dRedv = hydro.energy_conservation_mf->GetDerivative(VELOCITY, {&ue},
         {&uv, &hydro.rho0, &hydro.x0, &ux, &hydro.material, &hydro.qdata->h0, &hydro.qdata->order_v});

         auto dRede = hydro.energy_conservation_mf->GetDerivative(
         SPECIFIC_INTERNAL_ENERGY, {&ue},
         {&uv, &hydro.rho0, &hydro.x0, &ux, &hydro.material, &hydro.qdata->h0, &hydro.qdata->order_v});

         jacobian->Setup(hydro, dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede);
         return *jacobian;
      }
   }

   hydro_t &hydro;
   const real_t dt;
   const Vector &x;
   mutable Vector u;
   const int H1tsize;
   const int L2tsize;
   mutable std::shared_ptr<FDJacobian> fd_jacobian;
   mutable std::shared_ptr<LagrangianHydroJacobianOperator> jacobian;
   bool fd_gradient;
};

class LagrangianHydroOperator : public TimeDependentOperator
{
public:
   LagrangianHydroOperator(
      ParFiniteElementSpace &H1,
      ParFiniteElementSpace &L2,
      const Array<int> &ess_tdof,
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
      std::shared_ptr<QuadratureData> qdata,
      bool fd_gradient) :
      TimeDependentOperator(2*H1.GetVSize()+L2.GetVSize()),
      H1(H1),
      H1c(H1.GetParMesh(), H1.FEColl(), 1),
      L2(L2),
      ess_tdof(ess_tdof),
      ir(ir),
      rho0_coeff(rho0_coeff),
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
      qdata(qdata),
      mesh_nodes(&H1),
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
      rhsvc(&H1c),
      dvc(&H1c),
      nl2dofs(L2.GetFE(0)->GetDof()),
      fd_gradient(fd_gradient)
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

      // solve velocity
      {
         dv = 0.0;

         // momentum_mf->SetParameters({&rho0, &x0, &x, &material, &e, &qdata->h0, &qdata->order_v});
         momentum_pa->SetParameters({&qdata->stressp});
         H1.GetRestrictionMatrix()->Mult(v, Xv);
         // momentum_mf->Mult(Xv, RHSv);
         momentum_pa->Mult(Xv, RHSv);
         RHSv.Neg();
         H1.GetRestrictionMatrix()->MultTranspose(RHSv, rhsv);

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

      // solve energy
      {
         de = 0.0;

         // energy_conservation_mf->SetParameters({&v, &rho0, &x0, &x, &material, &qdata->h0, &qdata->order_v});
         energy_conservation_pa->SetParameters({&v, &qdata->stressp});
         L2.GetRestrictionMatrix()->Mult(e, Xe);
         // energy_conservation_mf->Mult(Xe, RHSe);
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
      }
   }

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override
   {
      auto xptr = const_cast<Vector*>(&x);

      Vector xx, xv, xe;
      xx.MakeRef(*xptr, 0, H1.GetVSize());
      xv.MakeRef(*xptr, H1.GetVSize(), H1.GetVSize());
      xe.MakeRef(*xptr, 2*H1.GetVSize(), L2.GetVSize());

      Vector Xx, Xv, Xe;
      Xx.MakeRef(X, 0, H1.GetTrueVSize());
      Xv.MakeRef(X, H1.GetTrueVSize(), H1.GetTrueVSize());
      Xe.MakeRef(X, 2*H1.GetTrueVSize(), L2.GetTrueVSize());

      H1.GetRestrictionMatrix()->Mult(xx, Xx);
      H1.GetRestrictionMatrix()->Mult(xv, Xv);
      L2.GetRestrictionMatrix()->Mult(xe, Xe);

      Xx.SyncAliasMemory(X);
      Xv.SyncAliasMemory(X);
      Xe.SyncAliasMemory(X);

      auto residual = LagrangianHydroResidualOperator(*this, dt, X, fd_gradient);

      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetMaxIter(500);
      gmres.SetKDim(500);
      gmres.SetRelTol(1e-8);
      gmres.SetAbsTol(1e-12);
      gmres.SetPrintLevel(IterativeSolver::PrintLevel().None());

      NewtonSolver newton(MPI_COMM_WORLD);
      newton.SetPrintLevel(IterativeSolver::PrintLevel().None());
      newton.SetOperator(residual);
      newton.SetSolver(gmres);
      newton.SetAdaptiveLinRtol();
      newton.SetMaxIter(10);
      newton.SetRelTol(1e-5);
      newton.SetAbsTol(1e-12);

      Vector zero;
      K = X;
      newton.Mult(zero, K);

      Vector Kx, Kv, Ke;
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
      // kx.SyncAliasMemory(k);
      // kv.SyncAliasMemory(k);
      // ke.SyncAliasMemory(k);
   }

   void UpdateMesh(const Vector &S) const
   {
      Vector* sptr = const_cast<Vector*>(&S);
      mesh_nodes.MakeRef(&H1, *sptr, 0);
      H1.GetParMesh()->NewNodes(mesh_nodes, false);
   }

   real_t GetTimeStepEstimate(const Vector &S)
   {
      UpdateMesh(S);

      auto sptr = const_cast<Vector*>(&S);
      const int H1vsize = H1.GetVSize();
      ParGridFunction x, v, e;
      x.MakeRef(&H1, *sptr, 0);
      v.MakeRef(&H1, *sptr, H1vsize);
      e.MakeRef(&L2, *sptr, 2*H1vsize);
      dtest_mf->SetParameters({&v, &rho0, &x0, &x, &material, &e, &qdata->h0, &qdata->order_v});
      auto &dt_est = qdata->dt_est;
      dtest_mf->Mult(dt_est, dt_est);

      real_t dt_est_local = std::numeric_limits<real_t>::infinity();
      for (int i = 0; i < dt_est.Size(); i++)
      {
         if (dt_est(i) == 0.0)
         {
            return 0.0;
         }
         dt_est_local = fmin(dt_est_local, dt_est(i));
      }

      real_t dt_est_global;
      MPI_Allreduce(&dt_est_local, &dt_est_global, 1, MPI_DOUBLE, MPI_MIN,
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

   void UpdateQuadratureData(const Vector &S) const
   {
      auto sptr = const_cast<Vector*>(&S);
      const int H1vsize = H1.GetVSize();
      ParGridFunction x, v, e;
      x.MakeRef(&H1, *sptr, 0);
      v.MakeRef(&H1, *sptr, H1vsize);
      e.MakeRef(&L2, *sptr, 2*H1vsize);
      update_qdata->SetParameters({&v, &rho0, &x0, &x, &material, &e, &qdata->h0, &qdata->order_v});
      update_qdata->Mult(qdata->stressp, qdata->stressp);
   }

   virtual MemoryClass GetMemoryClass() const override
   {
      return Device::GetDeviceMemoryClass();
   }

   ParFiniteElementSpace &H1;
   ParFiniteElementSpace &L2;
   mutable ParFiniteElementSpace H1c;
   const Array<int> &ess_tdof;
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
   std::shared_ptr<QuadratureData> qdata;
   mutable ParGridFunction mesh_nodes, rhsvc, dvc;
   mutable MassPAOperator *Mv = nullptr, *Me = nullptr;
   mutable FunctionCoefficient rho0_coeff;
   OperatorJacobiSmoother *Mv_Jprec = nullptr;
   mutable Vector RHSv, rhsv, X, Xx, Xv, Xvc, Xe, K, Kx, Kv, Ke, B, RHSe, rhse;
   const int nl2dofs;
   bool fd_gradient;
};

static auto CreateLagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   const IntegrationRule &ir,
   bool fd_gradient)
{
   const int order_v = H1.GetOrder(0);
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

   switch (mesh.GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT: qdata->h0 = vol_global / ne_global; break;
      case Geometry::SQUARE: qdata->h0 = sqrt(vol_global / ne_global); break;
      case Geometry::TRIANGLE: qdata->h0 = sqrt(2.0 * vol_global / ne_global); break;
      case Geometry::CUBE: qdata->h0 = pow(vol_global / ne_global, 1./3.); break;
      case Geometry::TETRAHEDRON: qdata->h0 = pow(6.0 * vol_global / ne_global,
                                                     1./3.); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   qdata->h0 /= (double) H1.GetOrder(0);

   // const real_t h0 = sqrt(vol_global / ne_global) /
   //                   static_cast<real_t>(H1.GetOrder(0));

   qdata->order_v = order_v;
   qdata->dt_est = std::numeric_limits<real_t>::infinity();

   std::shared_ptr<DifferentiableOperator> dt_est;
   {
      mfem::tuple dt_est_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         None<ELEMENT_SIZE0>{},
         None<ORDER_VEL>{},
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
         FieldDescriptor{ELEMENT_SIZE0, &qdata->R},
         FieldDescriptor{ORDER_VEL, &qdata->R}
      };

      dt_est = std::make_shared<DifferentiableOperator>(
                  dt_est_solutions, dt_est_parameters, mesh);
      TimeStepEstimateQFunction dt_est_qf;
      dt_est->AddDomainIntegrator(dt_est_qf, dt_est_kernel_ao,
                                  dt_est_kernel_oo,
                                  ir);
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
         None<ELEMENT_SIZE0>{},
         None<ORDER_VEL>{},
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
         {ELEMENT_SIZE0, &qdata->R},
         {ORDER_VEL, &qdata->R}
      };

      update_qdata = std::make_shared<DifferentiableOperator>(
                        update_qdata_solutions, update_qdata_parameters, mesh);
      UpdateQuadratureDataQFunction update_qdata_qf;
      update_qdata->AddDomainIntegrator(update_qdata_qf, update_qdata_kernel_ao,
                                        update_qdata_kernel_oo,
                                        ir);
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
         None<ELEMENT_SIZE0>{},
         None<ORDER_VEL>{},
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
         FieldDescriptor{ELEMENT_SIZE0, &qdata->R},
         FieldDescriptor{ORDER_VEL, &qdata->R}
      };

      momentum_mf = std::make_shared<DifferentiableOperator>(
                       momentum_mf_solutions, momentum_mf_parameters, mesh);

      MomentumQFunction momentum_qf;
      auto derivatives =
         std::integer_sequence<size_t, VELOCITY, COORDINATES, SPECIFIC_INTERNAL_ENERGY> {};
      momentum_mf->AddDomainIntegrator(momentum_qf, momentum_mf_kernel_ao,
                                       momentum_mf_kernel_oo, ir, derivatives);
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
                                       momentum_pa_kernel_oo, ir);
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
         None<ELEMENT_SIZE0>{},
         None<ORDER_VEL>{},
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
         FieldDescriptor{ELEMENT_SIZE0, &qdata->R},
         FieldDescriptor{ORDER_VEL, &qdata->R}
      };

      energy_conservation_mf =
         std::make_shared<DifferentiableOperator>(
            energy_conservation_mf_solutions, energy_conservation_mf_parameters, mesh);

      EnergyConservationQFunction energy_conservation_qf;
      auto derivatives =
         std::integer_sequence<size_t, VELOCITY, COORDINATES, SPECIFIC_INTERNAL_ENERGY> {};
      energy_conservation_mf->AddDomainIntegrator(
         energy_conservation_qf, energy_conservation_mf_kernel_ao,
         energy_conservation_mf_kernel_oo, ir, derivatives);
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
         energy_conservation_pa_kernel_oo, ir);
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
         total_internal_energy_kernel_oo, ir);
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
         total_kinetic_energy_kernel_oo, ir);
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
                                      density_kernel_oo, ir);
   }

   return LagrangianHydroOperator(
             H1,
             L2,
             ess_tdof,
             ir,
             rho0_coeff,
             x0_gf,
             rho0_gf,
             material_gf,
             update_qdata,
             dt_est,
             momentum_mf,
             momentum_pa,
             energy_conservation_mf,
             energy_conservation_pa,
             total_internal_energy_mf,
             total_kinetic_energy_mf,
             density_mf,
             qdata,
             fd_gradient);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";

   const char *mesh_file =
      "/Users/andrej1/repos/Laghos/data/rectangle01_quad.mesh";

   int refinements = 0;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   real_t t_final = 0.0;
   real_t blast_energy = 0.25;
   real_t blast_position[] = {0.0, 0.0, 0.0};
   int ode_solver_type = 4;
   bool fd_gradient = false;

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
      serial_mesh = Mesh(Mesh::MakeCartesian1D(2));
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

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace H1FESpace(&mesh, &H1FEC, dim);
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   ParFiniteElementSpace L2FESpace(&mesh, &L2FEC);

   const int global_ne = mesh.GetGlobalNE();
   const int global_h1tsize = H1FESpace.GlobalTrueVSize();
   const int global_l2tsize = L2FESpace.GlobalTrueVSize();

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

   auto hydro = CreateLagrangianHydroOperator(H1FESpace,
                                              L2FESpace,
                                              ess_tdof,
                                              rho0_coeff,
                                              x0_gf,
                                              rho0_gf,
                                              material_gf,
                                              ir,
                                              fd_gradient);

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
      default:
         out << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return -1;
   }
   ode_solver->Init(hydro);

   hydro.ComputeDensity(rho_gf);
   const real_t energy_init = hydro.InternalEnergy(e_gf) +
                              hydro.KineticEnergy(v_gf);

   if (Mpi::Root())
   {
      out << "energy initial: " << energy_init << "\n";
   }

   out << "IE " << hydro.InternalEnergy(e_gf) << "\n"
       << "KE "<< hydro.KineticEnergy(v_gf) << "\n";

   real_t t = 0.0;
   real_t dt = hydro.GetTimeStepEstimate(S);
   out << "time step estimate: " << dt << "\n";
   real_t t_old;
   bool last_step = false;
   int steps = 0;
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

   ParaViewDataCollection paraview_dc("dfem", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order_v);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("velocity", &v_gf);
   paraview_dc.RegisterField("density", &rho_gf);
   paraview_dc.RegisterField("specific_internal_energy", &e_gf);
   paraview_dc.RegisterField("material", &material_gf);
   // paraview_dc.RegisterField("velocity_error", &verr_gf);

   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0);
   paraview_dc.Save();

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      S_old = S;
      t_old = t;

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);
      steps++;

      // Adaptive time step control.
      const real_t dt_est = hydro.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < std::numeric_limits<real_t>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
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
             << ",\tt = " << std::setw(5) << std::setprecision(4) << t
             << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt;
         out << std::endl;
      }

      // verr_gf.ProjectCoefficient(v_coeff);
      // for (int i = 0; i < verr_gf.Size(); i++)
      // {
      //    verr_gf(i) = abs(verr_gf(i) - v_gf(i));
      // }

      hydro.ComputeDensity(rho_gf);

      paraview_dc.SetCycle(ti);
      paraview_dc.SetTime(t);
      paraview_dc.Save();
   }

   const real_t energy_final = hydro.InternalEnergy(e_gf)
                               + hydro.KineticEnergy(v_gf);
   const real_t v_err_max = v_gf.ComputeMaxError(v_coeff);
   const real_t v_err_l1 = v_gf.ComputeL1Error(v_coeff);
   const real_t v_err_l2 = v_gf.ComputeL2Error(v_coeff);

   if (Mpi::Root())
   {
      out << std::scientific << std::setprecision(2)
          << "Energy diff: " << fabs(energy_init - energy_final) << std::endl
          << "L_inf  error: " << v_err_max << std::endl
          << "L_1    error: " << v_err_l1 << std::endl
          << "L_2    error: " << v_err_l2 << std::endl;
   }

   return 0;
}
