#include <limits>
#include <memory>
#include <mfem.hpp>
#include "dfem/dfem.hpp"
#include "examples/dfem/dfem_parametricspace.hpp"
#include "fem/bilininteg.hpp"
#include "fem/coefficient.hpp"
#include "fem/intrules.hpp"
#include "fem/lor/lor.hpp"
#include "fem/pfespace.hpp"
#include "fem/pgridfunc.hpp"
#include "linalg/auxiliary.hpp"
#include "linalg/hypre.hpp"
#include "linalg/ode.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers.hpp"
#include "linalg/tensor.hpp"
#include "mpi.h"

using namespace mfem;
using mfem::internal::tensor;

int problem = 0;
double cfl = 0.5;
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
double taylor_source(const Vector &x)
{
   return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                               cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
};

// Smooth transition between 0 and 1 for x in [-eps, eps].
MFEM_HOST_DEVICE inline
double smooth_step_01(double x, double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

MFEM_HOST_DEVICE inline
void ComputeMaterialProperties(const double &gamma, const double &rho,
                               const double &E, double &p, double &cs)
{
   p = (gamma - 1.0) * rho * E;
   cs = sqrt(gamma * (gamma - 1.0) * E);
}

using vecd = tensor<double, 2>;
using vecaux = tensor<double, 3>;
using matd = tensor<double, 2, 2>;
using real = double;

template <bool compute_dtest = false>
std::tuple<matd, double> qdata_setup(
   const matd &dvdxi,
   const real &rho0,
   const matd &J0,
   const matd &J,
   const real &gamma,
   const real &E,
   const real &h0,
   const real &order_v,
   const real &w)
{
   constexpr int dim = 2;
   constexpr real eps = 1e-12;
   constexpr real vorticity_coeff = 1.0;
   real p, cs;
   real detJ = det(J);
   matd invJ = inv(J);
   matd stress{0.0};
   const real rho = rho0 * det(J0) / detJ;
   const real Ez = fmax(0.0, E);
   real visc_coeff = 0.0;
   real dt_est = std::numeric_limits<real>::infinity();

   ComputeMaterialProperties(gamma, rho, Ez, p, cs);

   for (int d = 0; d < dim; d++)
   {
      stress(d, d) = -p;
   }

   if (use_viscosity)
   {
      auto symdvdx = sym(dvdxi * invJ);
      auto [eigvals, eigvecs] = eig(symdvdx);
      vecd compr_dir = get_col(eigvecs, 0);
      auto ph_dir = (J * inv(J0)) * compr_dir;
      const double h = h0 * norm(ph_dir) / norm(compr_dir);
      // Measure of maximal compression.
      const double mu = eigvals(0);
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
         const real h_min = calcsv(J, dim-1) / static_cast<real>(order_v);
         const real idt = cs / h_min + 2.5 * visc_coeff / rho / h_min / h_min;
         if (idt > 0.0)
         {
            dt_est = cfl / idt;
         }
         else
         {
            dt_est = std::numeric_limits<real>::infinity();
         }
      }
   }

   matd stressJiT = stress * transpose(invJ) * detJ * w;
   return std::tuple{stressJiT, dt_est};
}

struct QuadratureData
{
   static constexpr int aux_dim = 1;
   QuadratureData(const ParMesh &mesh, const IntegrationRule &ir) :
      StressSpace(mesh.Dimension()*mesh.Dimension(),
                  ir.GetNPoints(),
                  mesh.Dimension()*mesh.Dimension()*ir.GetNPoints()*mesh.GetNE()),
      stressp(StressSpace),
      R(aux_dim,
        ir.GetNPoints(),
        aux_dim*ir.GetNPoints()*mesh.GetNE()),
      h0(R),
      order_v(R),
      dt_est(R)
   {}

   ParametricSpace StressSpace;
   ParametricFunction stressp;

   ParametricSpace R;
   ParametricFunction h0, order_v, dt_est;
};

class FDJacobian : public Operator
{
public:
   FDJacobian(const Operator &op, const Vector &x) :
      Operator(op.Height()),
      op(op),
      x(x)
   {
      f.SetSize(Height());
      xpev.SetSize(Height());
      op.Mult(x, f);
      xnorm = x.Norml2();
   }

   void Mult(const Vector &v, Vector &y) const
   {
      // See [1] for choice of eps.
      //
      // [1] Woodward, C.S., Gardner, D.J. and Evans, K.J., 2015. On the use of
      // finite difference matrix-vector products in Newton-Krylov solvers for
      // implicit climate dynamics with spectral elements. Procedia Computer
      // Science, 51, pp.2036-2045.
      double eps = lambda * (lambda + xnorm / v.Norml2());

      for (int i = 0; i < x.Size(); i++)
      {
         xpev(i) = x(i) + eps * v(i);
      }

      // y = f(x + eps * v)
      op.Mult(xpev, y);

      // y = (f(x + eps * v) - f(x)) / eps
      for (int i = 0; i < x.Size(); i++)
      {
         y(i) = (y(i) - f(i)) / eps;
      }
   }

private:
   const Operator &op;
   Vector x, f;
   mutable Vector xpev;
   double lambda = 1.0e-6;
   double xnorm;
};

class LagrangianHydroJacobianOperator : public Operator
{
public:
   LagrangianHydroJacobianOperator(double h, int H1tsize, int L2tsize) :
      Operator(2*H1tsize + L2tsize), h(h), H1tsize(H1tsize), L2tsize(L2tsize)  {}

   void Mult(const Vector &k, Vector &y) const override
   {
      jvp(k, y);
   }

   template <
      typename hydro_t,
      typename dRvdx_t,
      typename dRvdv_t,
      typename dRvde_t,
      typename dRedx_t,
      typename dRedv_t,
      typename dRede_t>
   void Setup(hydro_t &hydro,
              std::shared_ptr<dRvdx_t> dRvdx,
              std::shared_ptr<dRvdv_t> dRvdv,
              std::shared_ptr<dRvde_t> dRvde,
              std::shared_ptr<dRedx_t> dRedx,
              std::shared_ptr<dRedv_t> dRedv,
              std::shared_ptr<dRede_t> dRede)
   {
      w.SetSize(this->height);
      z.SetSize(this->height);

      jvp = [dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede, this, &hydro]
            (const Vector &u, Vector &y)
      {
         auto uptr = const_cast<Vector*>(&u);
         Vector uv;
         uv.MakeRef(*uptr, H1tsize, H1tsize);

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
         hydro.Mv.TrueAddMult(wv, yv);
         dRvde->Mult(we, zv);
         zv *= h;
         yv += zv;
         for (int i = 0; i < hydro.ess_tdof.Size(); i++)
         {
            // yv(hydro.ess_tdof[i]) = uv(hydro.ess_tdof[i]);
            yv(hydro.ess_tdof[i]) = 0.0;
         }
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
         hydro.Me.TrueAddMult(we, ze);
         ye += ze;
      };
   }

   double h;
   std::function<void(const Vector &, Vector &)> jvp;
   const int H1tsize;
   const int L2tsize;
   Vector w, z;
};

template <typename hydro_t>
class LagrangianHydroResidualOperator : public Operator
{
public:
   LagrangianHydroResidualOperator(hydro_t &hydro, const double dt,
                                   const Vector &x) :
      Operator(2*hydro.H1.GetTrueVSize()+hydro.L2.GetTrueVSize()),
      hydro(hydro),
      dt(dt),
      x(x),
      u(x.Size()),
      H1tsize(hydro.H1.GetTrueVSize()),
      L2tsize(hydro.L2.GetTrueVSize()) {}

   void Mult(const Vector &k, Vector &R) const override
   {
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

      u = k;
      u *= dt;
      u += x;

      hydro.UpdateMesh(u);

      Rx = kx;
      Rx -= uv;

      hydro.momentum_mf->SetParameters({&hydro.rho0, &hydro.x0, &ux, &hydro.material, &ue, &hydro.qdata->h0, &hydro.qdata->order_v});
      hydro.momentum_mf->Mult(uv, Rv);
      hydro.Mv.TrueAddMult(kv, Rv);
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
         e_source.Assemble();
         Re -= e_source;
      }

      hydro.Me.TrueAddMult(ke, Re);

      hydro.qdata_is_current = false;
   }

   Operator& GetGradient(const Vector &k) const override
   {
      jacobian.reset(new LagrangianHydroJacobianOperator(dt, H1tsize, L2tsize));

      auto kptr = const_cast<Vector*>(&k);
      Vector kx, kv, ke;
      kx.MakeRef(*kptr, 0, H1tsize);
      kv.MakeRef(*kptr, H1tsize, H1tsize);
      ke.MakeRef(*kptr, 2*H1tsize, L2tsize);

      Vector ux, uv, ue;
      ux.MakeRef(u, 0, H1tsize);
      uv.MakeRef(u, H1tsize, H1tsize);
      ue.MakeRef(u, 2*H1tsize, L2tsize);

      u = k;
      u *= dt;
      u += x;

      auto dRvdx = hydro.momentum_mf->template GetDerivativeWrt<3>( { &uv }, { &hydro.rho0, &hydro.x0, &ux, &hydro.material, &ue, &hydro.qdata->h0, &hydro.qdata->order_v });
      auto dRvdv = hydro.momentum_mf->template GetDerivativeWrt<0>( { &uv }, { &hydro.rho0, &hydro.x0, &ux, &hydro.material, &ue, &hydro.qdata->h0, &hydro.qdata->order_v });
      auto dRvde = hydro.momentum_mf->template GetDerivativeWrt<5>( { &uv }, { &hydro.rho0, &hydro.x0, &ux, &hydro.material, &ue, &hydro.qdata->h0, &hydro.qdata->order_v });
      auto dRedx = hydro.energy_conservation_mf->template GetDerivativeWrt<4>( { &ue }, { &uv, &hydro.rho0, &hydro.x0, &ux, &hydro.material, &hydro.qdata->h0, &hydro.qdata->order_v });
      auto dRedv = hydro.energy_conservation_mf->template GetDerivativeWrt<1>( { &ue }, { &uv, &hydro.rho0, &hydro.x0, &ux, &hydro.material, &hydro.qdata->h0, &hydro.qdata->order_v });
      auto dRede = hydro.energy_conservation_mf->template GetDerivativeWrt<0>( { &ue }, { &uv, &hydro.rho0, &hydro.x0, &ux, &hydro.material, &hydro.qdata->h0, &hydro.qdata->order_v });


      jacobian->Setup(hydro, dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede);
      return *jacobian;

      // fd_jacobian.reset(new FDJacobian(*this, k));
      // return *fd_jacobian;
   }

   hydro_t &hydro;
   const double dt;
   const Vector &x;
   mutable Vector u;
   const int H1tsize;
   const int L2tsize;
   mutable std::shared_ptr<FDJacobian> fd_jacobian;
   mutable std::shared_ptr<LagrangianHydroJacobianOperator> jacobian;
};

template <
   typename dtest_mf_t,
   typename momentum_mf_t,
   typename energy_conservation_mf_t,
   typename total_internal_energy_mf_t,
   typename total_kinetic_energy_mf_t,
   typename density_mf_t>
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
      std::shared_ptr<dtest_mf_t> dtest_mf,
      std::shared_ptr<momentum_mf_t> momentum_mf,
      std::shared_ptr<energy_conservation_mf_t> energy_conservation_mf,
      std::shared_ptr<total_internal_energy_mf_t> total_internal_energy_mf,
      std::shared_ptr<total_kinetic_energy_mf_t> total_kinetic_energy_mf,
      std::shared_ptr<density_mf_t>(density_mf),
      std::shared_ptr<QuadratureData> qdata) :
      TimeDependentOperator(2*H1.GetVSize()+L2.GetVSize()),
      H1(H1),
      L2(L2),
      ess_tdof(ess_tdof),
      ir(ir),
      rho0_coeff(rho0_coeff),
      x0(x0_gf),
      rho0(rho0_gf),
      material(material_gf),
      dtest_mf(dtest_mf),
      momentum_mf(momentum_mf),
      energy_conservation_mf(energy_conservation_mf),
      total_internal_energy_mf(total_internal_energy_mf),
      total_kinetic_energy_mf(total_kinetic_energy_mf),
      density_mf(density_mf),
      qdata(qdata),
      mesh_nodes(&H1),
      Mx(&H1),
      Mv(&H1),
      Me(&L2),
      RHSv(H1.GetTrueVSize()),
      rhsv(H1.GetVSize()),
      X(2*H1.GetTrueVSize()+L2.GetTrueVSize()),
      Xv(H1.GetTrueVSize()),
      Xe(L2.GetTrueVSize()),
      K(2*H1.GetTrueVSize()+L2.GetTrueVSize()),
      B(H1.GetTrueVSize()),
      dE(L2.GetTrueVSize()),
      RHSe(L2.GetTrueVSize()),
      rhse(L2.GetVSize()),
      nl2dofs(L2.GetFE(0)->GetDof())
   {
      VectorMassIntegrator *vmi = new VectorMassIntegrator;
      vmi->SetIntRule(&ir);
      vmi->SetVDim(2);
      Mx.AddDomainIntegrator(vmi);
      Mx.Assemble();

      vmi = new VectorMassIntegrator(rho0_coeff, &ir);
      Mv.AddDomainIntegrator(vmi);
      Mv.Assemble();

      MassIntegrator *mi = new MassIntegrator(rho0_coeff, &ir);
      Me.AddDomainIntegrator(mi);
      Me.Assemble();
   }

   void Mult(const Vector &S, Vector &dSdt) const override
   {
      UpdateMesh(S);

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

         momentum_mf->SetParameters({&rho0, &x0, &x, &material, &e, &qdata->h0, &qdata->order_v});
         H1.GetRestrictionMatrix()->Mult(v, Xv);
         momentum_mf->Mult(Xv, RHSv);
         RHSv.Neg();
         H1.GetRestrictionMatrix()->MultTranspose(RHSv, rhsv);

         HypreParMatrix A;
         Mv.FormLinearSystem(ess_tdof, dv, rhsv, A, Xv, B);

         CGSolver cg(H1.GetParMesh()->GetComm());
         HypreSmoother prec;
         prec.SetType(HypreSmoother::Jacobi, 1);
         cg.SetPreconditioner(prec);
         cg.SetOperator(A);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(300);
         cg.SetPrintLevel(-1);
         cg.Mult(B, Xv);
         Mv.RecoverFEMSolution(Xv, rhsv, dv);
      }

      // solve energy
      {
         de = 0.0;

         energy_conservation_mf->SetParameters({&v, &rho0, &x0, &x, &material, &qdata->h0, &qdata->order_v});
         L2.GetRestrictionMatrix()->Mult(e, Xe);
         energy_conservation_mf->Mult(Xe, RHSe);
         L2.GetRestrictionMatrix()->MultTranspose(RHSe, rhse);

         if (problem == 0)
         {
            LinearForm e_source(&L2);
            L2.GetMesh()->DeleteGeometricFactors();
            FunctionCoefficient coeff(taylor_source);
            DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &ir);
            e_source.AddDomainIntegrator(d);
            e_source.Assemble();
            rhse += e_source;
         }

         HypreParMatrix A;
         Array<int> empty;
         Me.FormSystemMatrix(empty, A);

         CGSolver cg(H1.GetParMesh()->GetComm());
         HypreSmoother prec;
         prec.SetType(HypreSmoother::Jacobi, 1);
         cg.SetPreconditioner(prec);
         cg.SetOperator(A);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(300);
         cg.SetPrintLevel(-1);
         cg.Mult(rhse, Xe);
         L2.GetProlongationMatrix()->Mult(Xe, de);
      }
   }

   void ImplicitSolve(const double dt, const Vector &x, Vector &k) override
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

      auto residual = LagrangianHydroResidualOperator(*this, dt, X);

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
   }

   void UpdateMesh(const Vector &S) const
   {
      Vector* sptr = const_cast<Vector*>(&S);
      mesh_nodes.MakeRef(&H1, *sptr, 0);
      H1.GetParMesh()->NewNodes(mesh_nodes, false);
   }

   double GetTimeStepEstimate(const Vector &S)
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

      double dt_est_local = std::numeric_limits<double>::infinity();
      for (int i = 0; i < dt_est.Size(); i++)
      {
         if (dt_est(i) == 0.0)
         {
            return 0.0;
         }
         dt_est_local = fmin(dt_est_local, dt_est(i));
      }

      double dt_est_global;
      MPI_Allreduce(&dt_est_local, &dt_est_global, 1, MPI_DOUBLE, MPI_MIN,
                    L2.GetComm());

      return dt_est_global;
   }

   void ResetQuadratureData() { qdata_is_current = false; }

   double InternalEnergy(ParGridFunction &e)
   {
      total_internal_energy_mf->SetParameters({&rho0, &x0});
      Vector E(L2.GetTrueVSize()), Y(L2.GetTrueVSize());
      L2.GetRestrictionMatrix()->Mult(e, E);
      total_internal_energy_mf->Mult(E, Y);
      const double ie_local = Y.Sum();
      double ie_global = 0.0;
      MPI_Allreduce(&ie_local, &ie_global, 1, MPI_DOUBLE, MPI_SUM,
                    L2.GetParMesh()->GetComm());
      return ie_global;
   }

   double KineticEnergy(ParGridFunction &v)
   {
      total_kinetic_energy_mf->SetParameters({&rho0, &x0});
      Vector V(H1.GetTrueVSize()), Y(L2.GetTrueVSize()), y(L2.GetVSize());
      H1.GetRestrictionMatrix()->Mult(v, V);
      total_kinetic_energy_mf->Mult(V, Y);
      const double ke_local = Y.Sum();
      double ke_global = 0.0;
      MPI_Allreduce(&ke_local, &ke_global, 1, MPI_DOUBLE, MPI_SUM,
                    H1.GetParMesh()->GetComm());
      return ke_global;
   }

   void ComputeDensity(ParGridFunction &rho) { }

   ParFiniteElementSpace &H1;
   ParFiniteElementSpace &L2;
   const Array<int> &ess_tdof;
   const IntegrationRule &ir;
   ParGridFunction &x0;
   ParGridFunction &rho0;
   ParGridFunction &material;
   std::shared_ptr<dtest_mf_t> dtest_mf;
   std::shared_ptr<momentum_mf_t> momentum_mf;
   std::shared_ptr<energy_conservation_mf_t> energy_conservation_mf;
   std::shared_ptr<total_internal_energy_mf_t> total_internal_energy_mf;
   std::shared_ptr<total_kinetic_energy_mf_t> total_kinetic_energy_mf;
   std::shared_ptr<density_mf_t> density_mf;
   std::shared_ptr<QuadratureData> qdata;
   mutable bool qdata_is_current = false;
   mutable ParGridFunction mesh_nodes;
   mutable ParBilinearForm Mx, Mv, Me;
   mutable FunctionCoefficient rho0_coeff;
   mutable Vector RHSv, rhsv, X, Xx, Xv, Xe, K, Kx, Kv, Ke, B, RHSe, rhse, dE;
   const int nl2dofs;
};

static auto CreateLagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   const IntegrationRule &ir)
{
   const int order_v = H1.GetOrder(0);
   ParMesh &mesh = *H1.GetParMesh();

   auto qdata = std::make_shared<QuadratureData>(mesh, ir);

   int ne_loc = mesh.GetNE(), ne_global = 0;
   double vol_loc = 0.0, vol_global = 0.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      vol_loc += mesh.GetElementVolume(e);
   }
   MPI_Allreduce(&vol_loc, &vol_global, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   MPI_Allreduce(&ne_loc, &ne_global, 1, MPI_INT, MPI_SUM, mesh.GetComm());
   const double h0 = sqrt(vol_global / ne_global) /
                     static_cast<double>(H1.GetOrder(0));

   qdata->h0 = h0;
   qdata->order_v = order_v;
   qdata->dt_est = std::numeric_limits<double>::infinity();

   auto dt_est_kernel =
      MFEM_HOST_DEVICE [](
         const matd &dvdxi,
         const real &rho0,
         const matd &J0,
         const matd &J,
         const real &gamma,
         const real &E,
         const real &h0,
         const real &order_v,
         const real &w)
   {
      real dt_est = std::get<1>(
                       qdata_setup<true>(dvdxi, rho0, J0, J, gamma, E, h0, order_v, w));
      return std::tuple{dt_est};
   };

   std::tuple dt_est_kernel_ao =
   {
      Gradient{"velocity"},
      Value{"density0"},
      Gradient{"coordinates0"},
      Gradient{"coordinates"},
      Value{"material"},
      Value{"specific_internal_energy"},
      None{"element_size0"},
      None{"order_v"},
      Weight{}
   };

   std::tuple dt_est_kernel_oo = {None{"dt_est"}};

   ElementOperator dt_est_eop{dt_est_kernel, dt_est_kernel_ao, dt_est_kernel_oo};
   auto dt_est_ops = std::tuple{dt_est_eop};

   std::array dt_est_solutions =
   {
      FieldDescriptor{&qdata->R, "dt_est"}
   };

   std::array dt_est_parameters =
   {
      FieldDescriptor{&H1, "velocity"},
      FieldDescriptor{&L2, "density0"},
      FieldDescriptor{&H1, "coordinates0"},
      FieldDescriptor{&H1, "coordinates"},
      FieldDescriptor{material_gf.ParFESpace(), "material"},
      FieldDescriptor{&L2, "specific_internal_energy"},
      FieldDescriptor{&qdata->R, "element_size0"},
      FieldDescriptor{&qdata->R, "order_v"}
   };

   auto dt_est = new DifferentiableOperator(dt_est_solutions,
                                            dt_est_parameters,
                                            dt_est_ops,
                                            mesh, ir);
   using dt_est_t = typename std::remove_pointer<decltype(dt_est)>::type;

   auto momentum_mf_kernel =
      [](
         const matd &dvdxi,
         const real &rho0,
         const matd &J0,
         const matd &J,
         const real &gamma,
         const real &E,
         const real &h0,
         const real &order_v,
         const real &w)
   {
      auto stressJiT = std::get<0>(
                          qdata_setup(dvdxi, rho0, J0, J, gamma, E, h0, order_v, w));
      // TODO-bug: investigate transpose of matrices in return types
      return std::tuple{transpose(stressJiT)};
   };

   std::tuple momentum_mf_kernel_ao =
   {
      Gradient{"velocity"},
      Value{"density0"},
      Gradient{"coordinates0"},
      Gradient{"coordinates"},
      Value{"material"},
      Value{"specific_internal_energy"},
      None{"element_size0"},
      None{"order_v"},
      Weight{}
   };

   std::tuple momentum_mf_kernel_oo = {Gradient{"velocity"}};

   // <sigma, grad(w) * J^-T> * det(J) * weights
   // <sigma(J^-T det(J) weights), grad(w)>
   ElementOperator momentum_mf_eop{momentum_mf_kernel, momentum_mf_kernel_ao, momentum_mf_kernel_oo};
   auto momentum_mf_ops = std::tuple{momentum_mf_eop};

   std::array momentum_mf_solutions =
   {
      FieldDescriptor{&H1, "velocity"}
   };

   std::array momentum_mf_parameters =
   {
      FieldDescriptor{&L2, "density0"},
      FieldDescriptor{&H1, "coordinates0"},
      FieldDescriptor{&H1, "coordinates"},
      FieldDescriptor{material_gf.ParFESpace(), "material"},
      FieldDescriptor{&L2, "specific_internal_energy"},
      FieldDescriptor{&qdata->R, "element_size0"},
      FieldDescriptor{&qdata->R, "order_v"}
   };

   auto momentum_mf = new DifferentiableOperator(momentum_mf_solutions,
                                                 momentum_mf_parameters,
                                                 momentum_mf_ops,
                                                 mesh, ir);
   using momentum_mf_t = typename
                         std::remove_pointer<decltype(momentum_mf)>::type;

   auto energy_conservation_mf_kernel =
      [](
         const matd &dvdxi,
         const real &rho0,
         const matd &J0,
         const matd &J,
         const real &gamma,
         const real &E,
         const real &h0,
         const real &order_v,
         const real &w)
   {
      auto stressJiT = std::get<0>(
                          qdata_setup(dvdxi, rho0, J0, J, gamma, E, h0, order_v, w));
      // TODO-bug: investigate transpose of matrices in return types
      return std::tuple{ddot(stressJiT, dvdxi)};
   };

   std::tuple energy_conservation_mf_kernel_ao =
   {
      Gradient{"velocity"},
      Value{"density0"},
      Gradient{"coordinates0"},
      Gradient{"coordinates"},
      Value{"material"},
      Value{"specific_internal_energy"},
      None{"element_size0"},
      None{"order_v"},
      Weight{}
   };

   std::tuple energy_conservation_mf_kernel_oo = {Value{"specific_internal_energy"}};

   // <sigma, grad(v) * inv(J) * phi> * det(J) * w
   // <sigma(J^-T det(J) w), grad(v) * inv(J)>
   ElementOperator energy_conservation_mf_eop{energy_conservation_mf_kernel, energy_conservation_mf_kernel_ao, energy_conservation_mf_kernel_oo};
   auto energy_conservation_mf_ops = std::tuple{energy_conservation_mf_eop};

   std::array energy_conservation_mf_solutions =
   {
      FieldDescriptor{&L2, "specific_internal_energy"}
   };

   std::array energy_conservation_mf_parameters =
   {
      FieldDescriptor{&H1, "velocity"},
      FieldDescriptor{&L2, "density0"},
      FieldDescriptor{&H1, "coordinates0"},
      FieldDescriptor{&H1, "coordinates"},
      FieldDescriptor{material_gf.ParFESpace(), "material"},
      FieldDescriptor{&qdata->R, "element_size0"},
      FieldDescriptor{&qdata->R, "order_v"}
   };

   auto energy_conservation_mf = new DifferentiableOperator(
      energy_conservation_mf_solutions,
      energy_conservation_mf_parameters,
      energy_conservation_mf_ops,
      mesh, ir);
   using energy_conservation_mf_t = typename
                                    std::remove_pointer<decltype(energy_conservation_mf)>::type;

   auto total_internal_energy_kernel =
      [](
         const real &E,
         const real &rho0,
         const matd &J0,
         const real &w)
   {
      return std::tuple{rho0 * E * det(J0) * w};
   };

   std::tuple total_internal_energy_kernel_ao =
   {
      Value{"specific_internal_energy"},
      Value{"density0"},
      Gradient{"coordinates0"},
      Weight{}
   };

   std::tuple total_internal_energy_kernel_oo = {Value{"specific_internal_energy"}};

   ElementOperator total_internal_energy_eop{total_internal_energy_kernel, total_internal_energy_kernel_ao, total_internal_energy_kernel_oo};
   auto total_internal_energy_ops = std::tuple{total_internal_energy_eop};

   std::array total_internal_energy_solutions =
   {
      FieldDescriptor{&L2, "specific_internal_energy"}
   };

   std::array total_internal_energy_parameters =
   {
      FieldDescriptor{&L2, "density0"},
      FieldDescriptor{&H1, "coordinates0"}
   };

   auto total_internal_energy_mf = new DifferentiableOperator(
      total_internal_energy_solutions,
      total_internal_energy_parameters,
      total_internal_energy_ops,
      mesh, ir);

   using total_internal_energy_mf_t = typename
                                      std::remove_pointer<decltype(total_internal_energy_mf)>::type;

   auto total_kinetic_energy_kernel =
      [](
         const vecd &v,
         const real &rho0,
         const matd &J0,
         const real &w)
   {
      return std::tuple{rho0 * 0.5 * v * v * det(J0) * w};
   };

   std::tuple total_kinetic_energy_kernel_ao =
   {
      Value{"velocity"},
      Value{"density0"},
      Gradient{"coordinates0"},
      Weight{}
   };

   std::tuple total_kinetic_energy_kernel_oo = {Value{"density0"}};

   ElementOperator total_kinetic_energy_eop{total_kinetic_energy_kernel, total_kinetic_energy_kernel_ao, total_kinetic_energy_kernel_oo};
   auto total_kinetic_energy_ops = std::tuple{total_kinetic_energy_eop};

   std::array total_kinetic_energy_solutions =
   {
      FieldDescriptor{&H1, "velocity"}
   };

   std::array total_kinetic_energy_parameters =
   {
      FieldDescriptor{&L2, "density0"},
      FieldDescriptor{&H1, "coordinates0"}
   };

   auto total_kinetic_energy_mf = new DifferentiableOperator(
      total_kinetic_energy_solutions,
      total_kinetic_energy_parameters,
      total_kinetic_energy_ops,
      mesh, ir);

   using total_kinetic_energy_mf_t = typename
                                     std::remove_pointer<decltype(total_kinetic_energy_mf)>::type;

   auto density_kernel =
      [](
         const real &rho0,
         const matd &J0,
         const real &w)
   {
      return std::tuple{rho0 * det(J0) * w};
   };

   std::tuple density_kernel_ao =
   {
      Value{"density0"},
      Gradient{"coordinates0"},
      Weight{}
   };

   std::tuple density_kernel_oo = {Value{"density0"}};

   ElementOperator density_eop{density_kernel, density_kernel_ao, density_kernel_oo};
   auto density_ops = std::tuple{density_eop};

   std::array density_solutions =
   {
      FieldDescriptor{&L2, "density0"}
   };

   std::array density_parameters =
   {
      FieldDescriptor{&H1, "coordinates0"}
   };

   auto density_mf = new DifferentiableOperator(
      density_solutions,
      density_parameters,
      density_ops,
      mesh, ir);

   using density_mf_t = typename std::remove_pointer<decltype(density_mf)>::type;

   return LagrangianHydroOperator(
             H1,
             L2,
             ess_tdof,
             ir,
             rho0_coeff,
             x0_gf,
             rho0_gf,
             material_gf,
             std::shared_ptr<dt_est_t>(dt_est),
             std::shared_ptr<momentum_mf_t>(momentum_mf),
             std::shared_ptr<energy_conservation_mf_t>(energy_conservation_mf),
             std::shared_ptr<total_internal_energy_mf_t>(total_internal_energy_mf),
             std::shared_ptr<total_kinetic_energy_mf_t>(total_kinetic_energy_mf),
             std::shared_ptr<density_mf_t>(density_mf),
             qdata);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *mesh_file =
      "/Users/andrej1/repos/Laghos/data/rectangle01_quad.mesh";

   int refinements = 0;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   double t_final = 0.0;
   double blast_energy = 0.25;
   double blast_position[] = {0.0, 0.0, 0.0};

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&refinements, "-r", "--ref", "");
   args.AddOption(&order_v, "-ov", "--ov", "");
   args.AddOption(&order_e, "-oe", "--oe", "");
   args.AddOption(&order_q, "-oq", "--oq", "");
   args.AddOption(&t_final, "-tf", "--tf", "");
   args.AddOption(&problem, "-p", "--p", "");
   args.AddOption(&cfl, "-cfl", "--cfl", "");
   args.AddOption(&use_viscosity, "-av", "--av", "-no-av", "--no-av", "");
   args.ParseCheck();

   Mesh serial_mesh = Mesh(mesh_file, true, true);

   if (problem == 0 || problem == 1)
   {
      serial_mesh = Mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                               true));

      const int NBE = serial_mesh.GetNBE();
      for (int b = 0; b < NBE; b++)
      {
         Element *bel = serial_mesh.GetBdrElement(b);
         const int attr = (b < NBE/2) ? 2 : 1;
         bel->SetAttribute(attr);
      }
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
      out << "el: " << global_ne << "\n";
      out << "kinematic dofs: " << global_h1tsize << "\n";
      out << "thermodynamic dofs: " << global_l2tsize << "\n";
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
   BlockVector S(offset, Device::GetMemoryType());

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
            const double denom = 2.0 / 3.0;  // (5/3 - 1) * density.
            double val;
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

   auto hydro = CreateLagrangianHydroOperator(H1FESpace,
                                              L2FESpace,
                                              ess_tdof,
                                              rho0_coeff,
                                              x0_gf,
                                              rho0_gf,
                                              material_gf,
                                              ir);

   ImplicitMidpointSolver ode_solver;
   ode_solver.Init(hydro);

   hydro.ComputeDensity(rho_gf);
   const double energy_init = hydro.InternalEnergy(e_gf) +
                              hydro.KineticEnergy(v_gf);

   if (Mpi::Root())
   {
      out << "energy initial: " << energy_init << "\n";
   }

   out << "IE " << hydro.InternalEnergy(e_gf) << "\n"
       << "KE "<< hydro.KineticEnergy(v_gf) << "\n";


   double t = 0.0;
   double dt = hydro.GetTimeStepEstimate(S);
   double t_old;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);

   ParGridFunction verr_gf(v_gf);
   verr_gf.ProjectCoefficient(v_coeff);
   for (int i = 0; i < verr_gf.Size(); i++)
   {
      verr_gf(i) = abs(verr_gf(i) - v_gf(i));
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
   paraview_dc.RegisterField("velocity_error", &verr_gf);

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
      ode_solver.Step(S, t, dt);
      steps++;

      // Adaptive time step control.
      const double dt_est = hydro.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < std::numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         hydro.ResetQuadratureData();
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

      // out << ">>> x_gf outer loop\n";
      // print_vector(x_gf);

      if (Mpi::Root())
      {
         out << "step " << std::setw(5) << ti
             << ",\tt = " << std::setw(5) << std::setprecision(4) << t
             << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt;
         out << std::endl;
      }

      verr_gf.ProjectCoefficient(v_coeff);
      for (int i = 0; i < verr_gf.Size(); i++)
      {
         verr_gf(i) = abs(verr_gf(i) - v_gf(i));
      }

      // hydro.ComputeDensity(rho_gf);

      paraview_dc.SetCycle(ti);
      paraview_dc.SetTime(t);
      paraview_dc.Save();
   }

   const double energy_final = hydro.InternalEnergy(e_gf)
                               + hydro.KineticEnergy(v_gf);
   const real v_err_max = v_gf.ComputeMaxError(v_coeff);
   const real v_err_l1 = v_gf.ComputeL1Error(v_coeff);
   const real v_err_l2 = v_gf.ComputeL2Error(v_coeff);

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
