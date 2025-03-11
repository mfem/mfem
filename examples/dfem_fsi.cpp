#include "dfem/dfem_refactor.hpp"
#include "examples/dfem/dfem_util.hpp"
#include "linalg/hypre.hpp"
#include "linalg/sparsemat.hpp"
#include <fstream>

using namespace mfem;
using mfem::internal::tensor;

constexpr int DIMENSION = 2;

enum ProblemType
{
   CFD_EX_TEST = 0,
   CFD1_TEST = 1,
   CFD2_TEST = 2,
   FSI_CSM1_TEST = 3,
   FSI1 = 4,
};

float clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f)
{
   if (x < lowerlimit) { return lowerlimit; }
   if (x > upperlimit) { return upperlimit; }
   return x;
}

struct CFD_TEST_CTX
{
   const int L = 1;
} cfd_test_ctx;

template <int dim = 2>
struct DisplacementMassQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &u,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      return mfem::tuple{u * det(J) * w};
   }
};

template <int dim = 2>
struct DisplacementLaplacianQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      auto invJ = inv(J);
      return mfem::tuple{(dudxi * invJ) * det(J) * w * transpose(invJ)};
   }
};

template <int dim = 2>
struct VelocityMassQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &v,
                   const tensor<real_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto Ju = det(I + dudxi * inv(J));
      return mfem::tuple{Ju * v * det(J) * w};
      // auto detFxiW = det(J + dudxi) * w;
      // return mfem::tuple{detFxiW * v};
   }
};

template <int dim = 2>
struct SolidVelocityMassQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &v,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      return mfem::tuple{v * det(J) * w};
   }
};

template <int dim = 2>
struct PressureMassQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const real_t &p,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      return mfem::tuple{p * det(J) * w};
   }
};

template <int dim = DIMENSION>
struct SolidMomentumQFunction
{
   SolidMomentumQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const tensor<real_t, dim, dim> &dudxi,
      const tensor<real_t, dim, dim> &J,
      const double &w) const
   {
      constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      constexpr real_t nu = 0.4;
      constexpr real_t mu = 0.5 * 1e6;
      constexpr real_t lambda = 2.0 * mu * nu / (1.0 - 2.0 * nu);

      auto invJ = inv(J);

      // Linear elasticity
      // auto E = sym(dudx);
      // auto JxW = det(J) * w * transpose(inv(J));
      // return mfem::tuple{(lambda * tr(E) * I + 2.0 * mu * E) * JxW};

      // St. Venant-Kirchhoff model
      auto Fu = I + dudxi * invJ;
      auto C = transpose(Fu) * Fu;
      auto E = 0.5 * (C - I);
      auto PK2 = 2.0 * mu * E + lambda * tr(E) * I;
      auto JxW = det(J) * w * transpose(inv(J));
      return mfem::tuple{Fu * PK2 * JxW};
   }
};

template <int dim = 2>
struct SolidBodyForceQFunction
{
   SolidBodyForceQFunction(real_t body_force) :
      body_force(body_force) {};

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto Fu = I + dudxi * inv(J);
      auto Ju = det(Fu);
      tensor<real_t, dim> b = {0.0, -body_force};
      return mfem::tuple{Ju * b * det(J) * w};
   }

   const real_t body_force;
};

template <int dim = 2>
struct SolidLFQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &f,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      return mfem::tuple{f * det(J) * w};
   }
};

template <int dim = 2>
struct NavierStokesMomentumConvectiveQFunction
{
   NavierStokesMomentumConvectiveQFunction(real_t density) :
      density(density) {}

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &v,
                   const tensor<real_t, dim, dim> &dvdxi,
                   const tensor<real_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto Fu = I + dudxi * inv(J);
      auto Ju = det(Fu) * density;
      auto dvdx = dvdxi * inv(J);
      return mfem::tuple{dot(Ju * dvdx * inv(Fu), v) * det(J) * w};
   }
   const real_t density;
};

template <int dim = 2>
struct NavierStokesMomentumConvectiveDisplacementQFunction
{
   NavierStokesMomentumConvectiveDisplacementQFunction(real_t density) :
      density(density) {}

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &v,
                   const tensor<real_t, dim, dim> &dvdxi,
                   const tensor<real_t, dim> &u,
                   const tensor<real_t, dim, dim> &dudxi,
                   const tensor<real_t, dim> &uprev,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto Fu = I + dudxi * inv(J);
      auto Ju = det(Fu) * density;
      auto dvdx = dvdxi * inv(J);
      return mfem::tuple{dot(Ju * dvdx * inv(Fu), u - uprev) * det(J) * w};
   }

   const real_t density;
};

template <int dim = 2>
struct NavierStokesMomentumViscousQFunction
{
   NavierStokesMomentumViscousQFunction(
      real_t density,
      real_t kinematic_viscosity) :
      density(density),
      kinematic_viscosity(kinematic_viscosity) {};

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim, dim> &dvdxi,
                   const real_t &p,
                   const tensor<real_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      // auto Fxi = (J + dudxi);
      // auto invFxi = inv(Fxi);
      // auto detFxi = det(Fxi);
      auto Fu = I + dudxi * inv(J);
      auto invFu = inv(Fu);
      auto Ju = det(Fu);
      auto viscous_stress = -Ju * p * I + density * kinematic_viscosity * 2.0 * sym(
                               Ju * (dvdxi * inv(J)) * invFu);
      return mfem::tuple{viscous_stress * transpose(invFu) * transpose(inv(J)) * w * det(J)};
   }
   const real_t kinematic_viscosity;
   const real_t density;
};

template <int dim = 2>
struct NavierStokesContinuityQFunction
{
   NavierStokesContinuityQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim, dim> &dvdxi,
                   const tensor<real_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto Fu = I + dudxi * inv(J);
      auto invFu = inv(Fu);
      auto Ju = det(Fu);
      auto dvdx = dvdxi * inv(J);
      return mfem::tuple{Ju * tr(dvdx * invFu) * det(J) * w};
   }
};

class ALEFSIOperator : public TimeDependentOperator
{
   static constexpr int Position = 0;
   static constexpr int Displacement = 1;
   static constexpr int Velocity = 2;
   static constexpr int Pressure = 3;
   // This is a placeholder ID used for terms like
   // an auxiliary displacement variable.
   static constexpr int Aux = 4;

   class ALEFSIResidual : public Operator
   {
   public:
      class ALEFSIResJac : public Operator
      {
      public:
         ALEFSIResJac(const ALEFSIResidual &res, const Vector &S) :
            Operator(S.Size()),
            res(res),
            S(S)
         {
            fd_jacobian = std::make_shared<FDJacobian>(res, S);

            // auto jacout = std::ofstream("jacout.m");
            // fd_jacobian->PrintMatlab(jacout);
            // jacout.close();
            // exit(0);
         }

         void Mult(const Vector &x, Vector &y) const override
         {
            fd_jacobian->Mult(x, y);
         }

         const ALEFSIResidual &res;
         const Vector S;
         mutable std::shared_ptr<FDJacobian> fd_jacobian;
      };

      class ALEFSIJacPrec : public Solver
      {
      public:
         ALEFSIJacPrec() : Solver() {}

         void SetOperator(const Operator &jac) override
         {
            this->height = jac.Height();
            this->width = jac.Width();

            auto alefsi_jac = dynamic_cast<const ALEFSIResJac*>(&jac);
            MFEM_VERIFY(alefsi_jac != nullptr, "invalid operator");

            const ALEFSIResidual &res = alefsi_jac->res;
            ALEFSIOperator &op = res.op;

            Array<int> combined_ess_tdof(op.def_ess_tdof.Size() +
                                         op.vel_ess_tdof.Size() +
                                         op.pres_ess_tdof.Size());
            for (int i = 0; i < op.def_ess_tdof.Size(); i++)
            {
               combined_ess_tdof[i] = op.def_ess_tdof[i];
            }
            int offset = op.def_ess_tdof.Size();
            for (int i = 0; i < op.vel_ess_tdof.Size(); i++)
            {
               combined_ess_tdof[i + offset] = op.vel_ess_tdof[i] + op.H1vtsize;
            }
            offset += op.vel_ess_tdof.Size();
            for (int i = 0; i < op.pres_ess_tdof.Size(); i++)
            {
               combined_ess_tdof[i + offset] = op.pres_ess_tdof[i] + 2*op.H1vtsize;
            }

            Vector Sd, Sv, Sp;
            auto Sptr = const_cast<Vector*>(&alefsi_jac->S);
            Sd.MakeRef(*Sptr, 0, op.H1vtsize);
            Sv.MakeRef(*Sptr, op.H1vtsize, op.H1vtsize);
            Sp.MakeRef(*Sptr, 2*op.H1vtsize, op.H1tsize);

            Vector prevSd;
            auto prevSptr = const_cast<Vector*>(&res.prevS);
            prevSd.MakeRef(*prevSptr, 0, op.H1vtsize);

            auto x_gf = static_cast<ParGridFunction*>(op.H1vfes.GetParMesh()->GetNodes());
            op.d_gf.SetFromTrueDofs(Sd);
            op.aux_gf.SetFromTrueDofs(prevSd);
            op.v_gf.SetFromTrueDofs(Sv);
            op.p_gf.SetFromTrueDofs(Sp);

            HypreParMatrix Mds, Mdf, Mv, Mp, Aconv, Avisc, B, A001, A011, A10_0, A10_1,
                           A10_2, A11_1;

            auto solid_disp_mass_dd = op.solid_displacement_mass->GetDerivative(
                                         Displacement, {&op.d_gf}, {x_gf});
            auto fluid_disp_mass_dd = op.fluid_displacement_mass->GetDerivative(
                                         Displacement, {&op.d_gf}, {x_gf});

            auto dMDv = op.fluid_velocity_mass->GetDerivative(Velocity, {&op.v_gf}, {&op.d_gf, x_gf});
            auto dMDp = op.fluid_pressure_mass->GetDerivative(Pressure, {&op.p_gf}, {x_gf});
            auto dFcvDv = op.fluid_momentum_convective->GetDerivative(Velocity, {&op.v_gf}, {&op.d_gf, x_gf});
            auto dFvvDv = op.fluid_momentum_viscous->GetDerivative(Velocity, {&op.v_gf}, {&op.p_gf, &op.d_gf, x_gf});
            auto dCDv = op.fluid_continuity->GetDerivative(Velocity, {&op.v_gf}, {&op.p_gf, &op.d_gf, x_gf});

            auto a11_1 = op.fluid_momentum_convective_displacement->GetDerivative(Velocity, {&op.v_gf}, {&op.d_gf, &op.aux_gf, x_gf});

            // auto dKddd = op.solid_momentum->GetDerivative(Displacement, {&op.d_gf}, {x_gf});
            auto a001 = op.fluid_displacement_laplacian->GetDerivative(Displacement, {&op.d_gf}, {x_gf});

            auto a01 = op.solid_lf->GetDerivative(Velocity, {&op.v_gf}, {&op.d_gf, x_gf});

            auto a10_0 = op.fluid_momentum_viscous->GetDerivative(Displacement, {&op.v_gf}, {&op.p_gf, &op.d_gf, x_gf});
            auto a10_1 = op.fluid_momentum_convective->GetDerivative(Displacement, {&op.v_gf}, {&op.d_gf, x_gf});
            auto a10_2 = op.fluid_momentum_convective_displacement->GetDerivative(
                            Displacement, {&op.v_gf}, {&op.d_gf, &op.aux_gf, x_gf});

            solid_disp_mass_dd->Assemble(Mds);
            fluid_disp_mass_dd->Assemble(Mdf);

            dMDv->Assemble(Mv);
            dMDp->Assemble(Mp);
            dFcvDv->Assemble(Aconv);
            dFvvDv->Assemble(Avisc);
            dCDv->Assemble(B);

            // dKddd->Assemble(Kd);

            a01->Assemble(A011);

            a001->Assemble(A001);
            a10_0->Assemble(A10_0);
            a10_1->Assemble(A10_1);
            a10_2->Assemble(A10_2);

            a11_1->Assemble(A11_1);

            std::shared_ptr<HypreParMatrix> A00;
            A00.reset(Add(1.0/res.dt, Mds, res.relax/res.dt, Mdf));
            A00.reset(Add(1.0, *A00, res.relax, A001));

            std::shared_ptr<HypreParMatrix> A10;
            A10.reset(Add(1.0, A10_0, 1.0, A10_1));
            A10.reset(Add(1.0, *A10, -1.0/res.dt, A10_2));

            std::shared_ptr<HypreParMatrix> A11;
            A11.reset(Add(1.0/res.dt, Mv, 1.0, Aconv));
            A11.reset(Add(1.0, *A11, 1.0, Avisc));
            A11.reset(Add(1.0, *A11, -1.0/res.dt, A11_1));

            auto Bt = B.Transpose();

            Array2D<const HypreParMatrix*> blocks(3, 3);
            blocks(0, 0) = A00.get(); // D D
            blocks(0, 1) = nullptr; // V D
            blocks(0, 2) = nullptr; // P D

            blocks(1, 0) = nullptr; // D V
            blocks(1, 1) = A11.get(); // V V
            blocks(1, 2) = Bt; // P V

            blocks(2, 0) = nullptr; // D P
            blocks(2, 1) = &B; // V P
            blocks(2, 2) = &Mp; // P P

            Array2D<real_t> blockCoeff(3, 3);
            blockCoeff = 1.0;
            blockCoeff(1, 2) = -1.0; // P V
            blockCoeff(2, 1) = -1.0; // V P
            blockCoeff(2, 2) = 1e-8; // P P

            K.reset(HypreParMatrixFromBlocks(blocks, &blockCoeff));

            // std::ofstream kmout("K.m");
            // kmout.precision(16);
            // K->PrintMatlab(kmout);
            // kmout.close();

            auto Ke = K->EliminateRowsCols(combined_ess_tdof);
            delete Ke;

            // std::ofstream kemout("Kelim.m");
            // kemout.precision(16);
            // K->PrintMatlab(kemout);
            // kemout.close();

            // auto fd_jacobian = std::make_shared<FDJacobian>(res, alefsi_jac->u);
            // std::ofstream fdout("Kfd.m");
            // fdout.precision(16);
            // fd_jacobian->PrintMatlab(fdout);
            // fdout.close();

            solver = std::make_shared<SuperLUSolver>(MPI_COMM_WORLD);
            solver->SetPrintStatistics(false);
            A_SLU = std::make_shared<SuperLURowLocMatrix>(*K);
            solver->SetOperator(*A_SLU);
         }

         void Mult(const Vector &x, Vector &y) const override
         {
            // GMRESSolver krylov(MPI_COMM_WORLD);
            // krylov.SetRelTol(1e-8);
            // krylov.SetMaxIter(1000);
            // krylov.SetOperator(*K);
            // krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());
            // krylov.Mult(x, y);

            // slu->Mult(x, y);
            solver->Mult(x, y);
         }


         std::shared_ptr<HypreParMatrix> K;
         std::shared_ptr<SuperLURowLocMatrix> A_SLU;
         std::shared_ptr<SuperLUSolver> solver;
      };

      ALEFSIResidual(ALEFSIOperator &op, const real_t &dt, const Vector &S,
                     const Vector &prevS) :
         Operator(op.offsets.Last()),
         op(op),
         dt(dt),
         relax(1e-8 * op.density_solid),
         prevS(prevS),
         z(S.Size()),
         d_gf(&op.H1vfes),
         d_prev_gf(&op.H1vfes),
         v_gf(&op.H1vfes),
         p_gf(&op.H1fes),
         H1vtsize(op.H1vfes.GetTrueVSize()),
         H1tsize(op.H1fes.GetTrueVSize()) {}

      void Mult(const Vector &S, Vector &R) const override
      {
         this->MultNoBC(S, R);

         Vector Rd, Rv, Rp;
         Rd.MakeRef(R, 0, H1vtsize);
         Rv.MakeRef(R, H1vtsize, H1vtsize);
         Rp.MakeRef(R, 2*H1vtsize, H1tsize);

         Rd.SetSubVector(op.def_ess_tdof, 0.0);
         Rv.SetSubVector(op.vel_ess_tdof, 0.0);
         Rp.SetSubVector(op.pres_ess_tdof, 0.0);
      }

      void MultNoBC(const Vector &S, Vector &R) const
      {
         auto uptr = const_cast<Vector*>(&S);

         Vector Sd, Sv, Sp;
         Sd.MakeRef(*uptr, 0, H1vtsize);
         Sv.MakeRef(*uptr, H1vtsize, H1vtsize);
         Sp.MakeRef(*uptr, 2*H1vtsize, H1tsize);

         Vector prevSd, prevSv, prevSp;
         prevSd.MakeRef(prevS, 0, H1vtsize);
         prevSv.MakeRef(prevS, H1vtsize, H1vtsize);
         prevSp.MakeRef(prevS, 2*H1vtsize, H1tsize);

         Vector Zd, Zv, Zp;
         Zd.MakeRef(z, 0, H1vtsize);
         Zv.MakeRef(z, H1vtsize, H1vtsize);
         Zp.MakeRef(z, 2*H1vtsize, H1tsize);

         Zd = 0.0;
         Zv = 0.0;
         Zp = 0.0;

         Vector Rd, Rv, Rp;
         Rd.MakeRef(R, 0, H1vtsize);
         Rv.MakeRef(R, H1vtsize, H1vtsize);
         Rp.MakeRef(R, 2*H1vtsize, H1tsize);

         Rd = 0.0;
         Rv = 0.0;
         Rp = 0.0;

         const real_t rhos = op.density_solid;
         const real_t rhof = op.density_fluid;

         auto x_gf = static_cast<ParGridFunction*>(op.H1vfes.GetParMesh()->GetNodes());

         d_gf.SetFromTrueDofs(Sd);

         subtract(Sd, prevSd, Zd);
         op.solid_displacement_mass->SetParameters({x_gf});
         op.solid_displacement_mass->AddMult(Zd, Rd, 1.0/dt);

         op.fluid_displacement_mass->SetParameters({x_gf});
         op.fluid_displacement_mass->AddMult(Zd, Rd, relax/dt);

         subtract(Sv, prevSv, Zv);
         op.solid_velocity_mass->SetParameters({&d_gf, x_gf});
         op.solid_velocity_mass->AddMult(Zv, Rv, rhos/dt);

         op.fluid_velocity_mass->SetParameters({&d_gf, x_gf});
         op.fluid_velocity_mass->AddMult(Zv, Rv, rhof/dt);

         {
            op.solid_lf->SetParameters({&d_gf, x_gf});
            op.solid_lf->AddMult(Sv, Rd, -1.0);

            op.fluid_displacement_laplacian->SetParameters({x_gf});
            op.fluid_displacement_laplacian->AddMult(Sd, Rd, relax);

            op.solid_momentum->SetParameters({x_gf});
            op.solid_momentum->AddMult(Sd, Rv);

            // op.solid_body_force->SetParameters({&v_gf, x_gf});
            // op.solid_body_force->AddMult(Sd, Rv, -1.0);
         }

         p_gf.SetFromTrueDofs(Sp);
         op.fluid_momentum_viscous->SetParameters({&p_gf, &d_gf, x_gf});
         op.fluid_momentum_viscous->AddMult(Sv, Rv);

         op.fluid_momentum_convective->SetParameters({&d_gf, x_gf});
         op.fluid_momentum_convective->AddMult(Sv, Rv);

         d_prev_gf.SetFromTrueDofs(prevSd);
         op.fluid_momentum_convective_displacement->SetParameters({&d_gf, &d_prev_gf, x_gf});
         op.fluid_momentum_convective_displacement->AddMult(Sv, Rv, -1.0/dt);

         p_gf.SetFromTrueDofs(Sp);
         op.fluid_continuity->SetParameters({&p_gf, &d_gf, x_gf});
         op.fluid_continuity->AddMult(Sv, Rp, -1.0);

         // op.solid_pressure_mass->SetParameters({x_gf});
         // op.solid_pressure_mass->AddMult(Sp, Rp, 1.0e3 * rhos);

         op.fluid_pressure_mass->SetParameters({x_gf});
         op.fluid_pressure_mass->AddMult(Sp, Rp, 1.0e-8);
      }

      Operator& GetGradient(const Vector &S) const override
      {
         jacobian.reset(new ALEFSIResJac(*this, S));
         return *jacobian;
         // fd_jacobian.reset(new FDJacobian(*this, S));
         // return *fd_jacobian;
      }

      ALEFSIOperator &op;
      const real_t dt;
      const real_t relax;

      mutable ParGridFunction d_gf, d_prev_gf, v_gf, p_gf;

      const int H1vtsize;
      const int H1tsize;

      mutable Vector z, prevS;

      mutable std::shared_ptr<ALEFSIResJac> jacobian;
      mutable std::shared_ptr<FDJacobian> fd_jacobian;
   };

public:
   ALEFSIOperator(
      real_t theta,
      real_t density_solid,
      real_t density_fluid,
      real_t kinematic_viscosity,
      ParFiniteElementSpace &H1vfes,
      ParFiniteElementSpace &H1fes,
      Array<int> &offsets,
      Array<int> &solid_domain_attr,
      Array<int> &fluid_domain_attr,
      Array<int> &def_ess_bdr,
      Array<int> &vel_ess_bdr,
      Array<int> &pres_ess_bdr,
      VectorCoefficient &def_bdr_coeff,
      VectorCoefficient &vel_bdr_coeff,
      Coefficient &pres_bdr_coeff,
      const IntegrationRule &ir,
      bool enable_tps) :
      TimeDependentOperator(offsets.Last()),
      theta(theta),
      kinematic_viscosity(kinematic_viscosity),
      density_solid(density_solid),
      density_fluid(density_fluid),
      offsets(offsets),
      H1vfes(H1vfes),
      H1fes(H1fes),
      H1vtsize(H1vfes.GetTrueVSize()),
      H1tsize(H1fes.GetTrueVSize()),
      dis_ess_bdr(def_ess_bdr),
      vel_ess_bdr(vel_ess_bdr),
      pres_ess_bdr(pres_ess_bdr),
      dis_bdr_coeff(&def_bdr_coeff),
      vel_bdr_coeff(&vel_bdr_coeff),
      pres_bdr_coeff(&pres_bdr_coeff),
      ir(ir),
      prevS(offsets.Last()),
      d_gf(&H1vfes),
      aux_gf(&H1vfes),
      v_gf(&H1vfes),
      p_gf(&H1fes)
   {
      auto mesh = H1vfes.GetParMesh();
      x_gf = static_cast<ParGridFunction*>(mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *x_gf->ParFESpace();

      Array<int> all_domains(mesh->attributes.Size());
      for (int i = 0; i < all_domains.Size(); i++)
      {
         all_domains[i] = solid_domain_attr[i] || fluid_domain_attr[i] ? 1 : 0;
      }

      H1vfes.GetEssentialTrueDofs(def_ess_bdr, def_ess_tdof);
      H1vfes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_tdof);
      H1fes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_tdof);

      // solid_displacement_mass
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         solid_displacement_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            solid_displacement_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Displacement>{}};
         auto qf = DisplacementMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Displacement> {};
         solid_displacement_mass->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                      solid_domain_attr,
                                                      derivatives);
      }

      // fluid displacement mass
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_displacement_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_displacement_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Displacement>{}};
         auto qf = DisplacementMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Displacement> {};
         fluid_displacement_mass->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                      fluid_domain_attr,
                                                      derivatives);
      }

      // displacement mass
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         displacement_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            displacement_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Displacement>{}};
         auto qf = DisplacementMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Displacement> {};
         displacement_mass->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                all_domains,
                                                derivatives);
      }

      // fluid displacement laplacian
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_displacement_laplacian =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_displacement_laplacian->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Gradient<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Gradient<Displacement>{}};
         auto qf = DisplacementLaplacianQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Displacement> {};
         fluid_displacement_laplacian->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                           fluid_domain_attr,
                                                           derivatives);
      }

      // solid velocity mass
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
            FieldDescriptor{Position, &mesh_fes}
         };

         solid_velocity_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            solid_velocity_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Velocity>{}, Gradient<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Velocity>{}};
         auto qf = VelocityMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         solid_velocity_mass->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                  solid_domain_attr,
                                                  derivatives);
      }

      // solid momentum
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes}
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         solid_momentum =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);
         if (!enable_tps)
         {
            solid_momentum->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Gradient<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Gradient<Displacement>{}};

         auto qf = SolidMomentumQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Displacement> {};
         solid_momentum->AddDomainIntegrator(qf, inputs, outputs, ir, solid_domain_attr,
                                             derivatives);
      }

      // solid body force
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
            FieldDescriptor{Position, &mesh_fes}
         };

         solid_body_force =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);
         if (!enable_tps)
         {
            solid_body_force->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Gradient<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Velocity>{}};
         auto qf = SolidBodyForceQFunction<DIMENSION>(2.0*density_solid);
         solid_body_force->AddDomainIntegrator(qf, inputs, outputs, ir,
                                               solid_domain_attr);
      }

      // solid linear form
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
            FieldDescriptor{Position, &mesh_fes}
         };

         solid_lf = std::make_shared<DifferentiableOperator>(solutions, parameters,
                                                             *mesh);
         if (!enable_tps)
         {
            solid_lf->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Velocity>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Displacement>{}};

         auto qf = SolidLFQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         solid_lf->AddDomainIntegrator(qf, inputs, outputs, ir, solid_domain_attr,
                                       derivatives);
      }

      // fluid velocity mass
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_velocity_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_velocity_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Velocity>{}, Gradient<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Velocity>{}};
         auto qf = VelocityMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         fluid_velocity_mass->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                  fluid_domain_attr,
                                                  derivatives);
      }

      // fluid momentum convective
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_momentum_convective =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_momentum_convective->DisableTensorProductStructure();
         }

         mfem::tuple inputs
         {
            Value<Velocity>{},
            Gradient<Velocity>{},
            Gradient<Displacement>{},
            Gradient<Position>{},
            Weight{}
         };

         mfem::tuple outputs{Value<Velocity>{}};

         auto qf = NavierStokesMomentumConvectiveQFunction<DIMENSION>(density_fluid);
         auto derivatives = std::integer_sequence<size_t, Velocity, Displacement> {};
         fluid_momentum_convective->AddDomainIntegrator(
            qf, inputs, outputs, ir, fluid_domain_attr, derivatives);
      }

      // fluid momentum convective displacement
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Displacement, &H1vfes},
            FieldDescriptor{Aux, &H1vfes},
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_momentum_convective_displacement =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_momentum_convective_displacement->DisableTensorProductStructure();
         }

         mfem::tuple inputs
         {
            Value<Velocity>{},
            Gradient<Velocity>{},
            Value<Displacement>{},
            Gradient<Displacement>{},
            Value<Aux>{},
            Gradient<Position>{},
            Weight{}
         };

         mfem::tuple outputs{Value<Velocity>{}};

         auto qf = NavierStokesMomentumConvectiveDisplacementQFunction<DIMENSION>
                   (density_fluid);
         auto derivatives = std::integer_sequence<size_t, Velocity, Displacement> {};
         fluid_momentum_convective_displacement->AddDomainIntegrator(
            qf, inputs, outputs, ir, fluid_domain_attr, derivatives);
      }

      // fluid momentum viscous
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Pressure, &H1fes},
            FieldDescriptor{Displacement, &H1vfes},
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_momentum_viscous =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_momentum_viscous->DisableTensorProductStructure();
         }

         mfem::tuple inputs
         {
            Gradient<Velocity>{},
            Value<Pressure>{},
            Gradient<Displacement>{},
            Gradient<Position>{},
            Weight{}
         };

         mfem::tuple outputs{Gradient<Velocity>{}};

         auto qf = NavierStokesMomentumViscousQFunction<DIMENSION>(
                      density_fluid,
                      kinematic_viscosity);
         auto derivatives = std::integer_sequence<size_t, Velocity, Displacement> {};
         fluid_momentum_viscous->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                     fluid_domain_attr, derivatives);
      }

      // fluid continuity
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Pressure, &H1fes},
            FieldDescriptor{Displacement, &H1vfes},
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_continuity =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_continuity->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Gradient<Velocity>{}, Gradient<Displacement>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Pressure>{}};
         auto qf = NavierStokesContinuityQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         fluid_continuity->AddDomainIntegrator(
            qf, inputs, outputs, ir, fluid_domain_attr, derivatives);
      }

      // fluid pressure mass
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Pressure, &H1fes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_pressure_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_pressure_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Pressure>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Pressure>{}};
         auto qf = PressureMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Pressure> {};
         fluid_pressure_mass->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                  fluid_domain_attr,
                                                  derivatives);
      }

      // solid pressure mass
      {
         auto solutions = std::vector
         {
            FieldDescriptor{Pressure, &H1fes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         solid_pressure_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            solid_pressure_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Pressure>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Pressure>{}};
         auto qf = PressureMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Pressure> {};
         solid_pressure_mass->AddDomainIntegrator(qf, inputs, outputs, ir,
                                                  solid_domain_attr,
                                                  derivatives);
      }
   }

   void SetTime(const real_t t) override
   {
      dis_bdr_coeff->SetTime(t);
      vel_bdr_coeff->SetTime(t);
      pres_bdr_coeff->SetTime(t);
   }

   void Step(Vector &S, real_t &t, const real_t &dt)
   {
      this->SetTime(t);
      prevS = S;

      this->SetTime(t + dt);
      Vector Sd, Sv, Sp;
      Sd.MakeRef(S, 0, H1vtsize);
      Sv.MakeRef(S, H1vtsize, H1vtsize);
      Sp.MakeRef(S, 2*H1vtsize, H1tsize);

      d_gf.SetFromTrueDofs(Sd);
      d_gf.ProjectBdrCoefficient(*dis_bdr_coeff, dis_ess_bdr);
      d_gf.GetTrueDofs(Sd);

      v_gf.SetFromTrueDofs(Sv);
      v_gf.ProjectBdrCoefficient(*vel_bdr_coeff, vel_ess_bdr);
      v_gf.GetTrueDofs(Sv);

      p_gf.SetFromTrueDofs(Sp);
      p_gf.ProjectBdrCoefficient(*pres_bdr_coeff, pres_ess_bdr);
      p_gf.GetTrueDofs(Sp);

      ALEFSIResidual residual(*this, dt, S, prevS);

      ALEFSIOperator::ALEFSIResidual::ALEFSIJacPrec prec;

      GMRESSolver krylov(MPI_COMM_WORLD);
      krylov.SetRelTol(1e-8);
      krylov.SetMaxIter(2000);
      krylov.SetKDim(1000);
      krylov.SetPreconditioner(prec);
      krylov.SetPrintLevel(IterativeSolver::PrintLevel().FirstAndLast());

      NewtonSolver newton(MPI_COMM_WORLD);
      newton.SetSolver(krylov);
      newton.SetOperator(residual);
      newton.SetRelTol(1e-6);
      newton.SetAbsTol(1e-12);
      newton.SetMaxIter(50);
      // newton.SetAdaptiveLinRtol();
      newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());

      Vector zero;
      newton.Mult(zero, S);

      t += dt;
   }

   real_t theta;
   real_t density_solid;
   real_t kinematic_viscosity;
   real_t density_fluid;

   Array<int> offsets;

   std::shared_ptr<DifferentiableOperator> solid_lf;
   std::shared_ptr<DifferentiableOperator> solid_body_force;
   std::shared_ptr<DifferentiableOperator> solid_displacement_mass;
   std::shared_ptr<DifferentiableOperator> solid_velocity_mass;
   std::shared_ptr<DifferentiableOperator> solid_momentum;
   std::shared_ptr<DifferentiableOperator> solid_pressure_mass;

   std::shared_ptr<DifferentiableOperator> fluid_displacement_mass;
   std::shared_ptr<DifferentiableOperator> fluid_displacement_laplacian;
   std::shared_ptr<DifferentiableOperator> fluid_velocity_mass;
   std::shared_ptr<DifferentiableOperator> fluid_pressure_mass;
   std::shared_ptr<DifferentiableOperator> fluid_momentum_convective;
   std::shared_ptr<DifferentiableOperator> fluid_momentum_convective_displacement;
   std::shared_ptr<DifferentiableOperator> fluid_momentum_viscous;
   std::shared_ptr<DifferentiableOperator> fluid_continuity;

   std::shared_ptr<DifferentiableOperator> displacement_mass;

   ParGridFunction *x_gf, d_gf, aux_gf, v_gf, p_gf;

   Vector prevS;

   const Array<int> dis_ess_bdr, vel_ess_bdr, pres_ess_bdr;
   Array<int> def_ess_tdof, vel_ess_tdof, pres_ess_tdof;

   VectorCoefficient *dis_bdr_coeff, *vel_bdr_coeff;
   Coefficient *pres_bdr_coeff;

   ParFiniteElementSpace &H1vfes;
   ParFiniteElementSpace &H1fes;
   const int H1vtsize, H1tsize;
   IntegrationRule ir;
};

int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();
   Hypre::Init();

   const char* device_config = "cpu";
   const char* mesh_file = "";
   int polynomial_order_h1v = 2;
   int refinements = 0;
   int problem_type = 0;
   real_t t_final = 0.0;
   real_t dt = 1e-3;
   real_t density_solid = 1e3;
   real_t density_fluid = 1e3;
   real_t kinematic_viscosity = 1.0e-3;
   int vis_steps = 1;
   real_t theta = 1.0;
   real_t ubar = 0.2;
   bool enable_tps = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&problem_type, "-prob", "--problem", "Problem #");
   args.AddOption(&polynomial_order_h1v, "-ov", "--order-velocity", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&t_final, "-tf", "--tf", "");
   args.AddOption(&dt, "-dt", "--dt", "");
   args.AddOption(&density_solid, "-rhos", "--rhos", "");
   args.AddOption(&density_fluid, "-rhof", "--rhof", "");
   args.AddOption(&kinematic_viscosity, "-kv", "--kinematic-viscosity", "");
   args.AddOption(&theta, "-theta", "--theta", "");
   args.AddOption(&ubar, "-ubar", "--ubar", "");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&enable_tps, "-tps", "--enable-tps", "-no-tps",
                  "--no-tps", "Enable tensor product structure for quad/hex.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root() == 0)
   {
      device.Print();
   }

   int polynomial_order_h1 = polynomial_order_h1v - 1;

   Mesh mesh_serial;
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      mesh_serial = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      mesh_serial.EnsureNodes();
      auto nodes = mesh_serial.GetNodes();
      *nodes -= 0.5;
      *nodes *= cfd_test_ctx.L;
   }
   else if (problem_type == ProblemType::CFD1_TEST ||
            problem_type == ProblemType::CFD2_TEST ||
            problem_type == ProblemType::FSI1)
   {
      mesh_serial = Mesh::LoadFromFile(mesh_file);
      mesh_serial.EnsureNodes();
   }
   else if (problem_type == ProblemType::FSI_CSM1_TEST)
   {
      auto m = Mesh::LoadFromFile(mesh_file);
      m.EnsureNodes();
      Array<int> beam(1);
      beam[0] = 2;
      mesh_serial = SubMesh::CreateFromDomain(m, beam);
   }
   else
   {
      MFEM_ABORT("invalid problem type");
   }
   MFEM_ASSERT(mesh_serial.Dimension() == dim, "incorrect mesh dimension");

   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.EnsureNodes();
   mesh_serial.Clear();

   out << "#el: " << mesh.GetNE() << "\n";

   H1_FECollection h1vfec(polynomial_order_h1v, dim);
   H1_FECollection h1fec(polynomial_order_h1);

   ParFiniteElementSpace H1vfes(&mesh, &h1vfec, dim);
   ParFiniteElementSpace H1fes(&mesh, &h1fec);

   HYPRE_BigInt global_size_h1v = H1vfes.GlobalTrueVSize();
   HYPRE_BigInt global_size_h1 = H1fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      out << "displacement #dof: " << global_size_h1v << "\n";
      out << "velocity #dof: " << global_size_h1v << "\n";
      out << "pressure #dof: " << global_size_h1 << "\n";
      out << "Total #dof: " << 2*global_size_h1v + global_size_h1 << "\n";
   }

   const IntegrationRule &integration_rule =
      IntRules.Get(H1vfes.GetFE(0)->GetGeomType(),
                   2 * H1vfes.GetFE(0)->GetOrder() + 1);

   Array<int> solid_domain_attr(mesh.attributes.Max());
   solid_domain_attr = 0;
   Array<int> fluid_domain_attr(mesh.attributes.Max());
   fluid_domain_attr = 0;

   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      // no solid domain
      fluid_domain_attr[0] = 1;
   }
   else if (problem_type == ProblemType::CFD1_TEST ||
            problem_type == ProblemType::CFD2_TEST)
   {
      // all domains are fluid
      fluid_domain_attr[0] = 1;
      fluid_domain_attr[1] = 1;
   }
   else if (problem_type == ProblemType::FSI_CSM1_TEST)
   {
      // no fluid domain
      solid_domain_attr[1] = 1;
   }
   else if (problem_type == ProblemType::FSI1)
   {
      fluid_domain_attr[0] = 1;
      solid_domain_attr[1] = 1;
   }

   Array<int> disp_ess_attr(mesh.bdr_attributes.Max());
   Array<int> vel_ess_attr(mesh.bdr_attributes.Max());
   Array<int> pres_ess_attr(mesh.bdr_attributes.Max());

   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      disp_ess_attr = 1;
      vel_ess_attr = 1;
      pres_ess_attr = 1;
   }
   else if (problem_type == ProblemType::CFD1_TEST ||
            problem_type == ProblemType::CFD2_TEST)
   {
      disp_ess_attr = 1;

      vel_ess_attr = 1;
      // beam
      vel_ess_attr[4] = 0;
      // outlet
      vel_ess_attr[1] = 0;

      pres_ess_attr = 0;
   }
   else if (problem_type == ProblemType::FSI_CSM1_TEST)
   {
      disp_ess_attr = 0;
      // clamped beam edge on cylinder curve
      disp_ess_attr[6] = 1;

      vel_ess_attr = 0;
      // clamped beam edge on cylinder curve
      vel_ess_attr[6] = 1;

      pres_ess_attr = 0;
   }
   else if (problem_type == ProblemType::FSI1)
   {
      disp_ess_attr = 1;
      // beam
      disp_ess_attr[4] = 0;

      // everywhere
      vel_ess_attr = 1;
      // beam
      vel_ess_attr[4] = 0;
      // outlet
      vel_ess_attr[1] = 0;

      pres_ess_attr = 0;
   }

   Array<int> block_offsets(4);
   block_offsets[0] = 0;
   block_offsets[1] = H1vfes.GetTrueVSize();
   block_offsets[2] = H1vfes.GetTrueVSize();
   block_offsets[3] = H1fes.GetTrueVSize();
   block_offsets.PartialSum();

   BlockVector S(block_offsets, Device::GetDeviceMemoryType());

   ParGridFunction d_gf(&H1vfes), v_gf(&H1vfes), p_gf(&H1fes);

   std::function<void(const Vector &, real_t, Vector &)> displacement_exact;
   if (problem_type == ProblemType::CFD_EX_TEST ||
       problem_type == ProblemType::CFD1_TEST ||
       problem_type == ProblemType::CFD2_TEST ||
       problem_type == ProblemType::FSI_CSM1_TEST ||
       problem_type == ProblemType::FSI1)
   {
      displacement_exact = [](const Vector &, real_t, Vector &u)
      {
         u(0) = 0.0;
         u(1) = 0.0;
      };
   }

   std::function<void(const Vector &, real_t, Vector &)> velocity_exact;
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      velocity_exact = [nu = kinematic_viscosity](
                          const Vector &coords, real_t t, Vector &u)
      {
         const real_t x = coords(0);
         const real_t y = coords(1);
         const real_t f = exp(-4.0 * nu * M_PI * M_PI * t);
         u(0) = -sin(2.0 * M_PI * y) * f;
         u(1) = sin(2.0 * M_PI * x) * f;
      };
   }
   else if (problem_type == ProblemType::CFD1_TEST ||
            problem_type == ProblemType::CFD2_TEST ||
            problem_type == ProblemType::FSI1)
   {
      velocity_exact = [problem_type, ubar](const Vector &coords, real_t t, Vector &u)
      {
         const real_t x = coords(0);
         const real_t y = coords(1);
         const real_t H = 0.41;

         real_t U = ubar;

         auto smoothstep = [t](const real_t edge0, const real_t edge1, real_t x)
         {
            x = clamp((x - edge0) / (edge1 - edge0));
            return x * x * (3.0 - 2.0 * x);
         };
         real_t inflow_scaling = (1-cos(M_PI/2.0*t))/2.0;
         if (x == 0.0)
         {
            u(0) = 4.0 * U * y * (H - y) / powf(H, 2.0);
         }
         else
         {
            u(0) = 0.0;
         }

         if (t <= 2.0)
         {
            u(0) *= inflow_scaling;
         }

         u(1) = 0.0;
      };
   }
   else if (problem_type == ProblemType::FSI_CSM1_TEST)
   {
      velocity_exact = [](const Vector &coords, real_t t, Vector &u)
      {
         u = 0.0;
      };
   }

   std::function<real_t(const Vector &, real_t)> pressure_exact;
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      pressure_exact = [nu = kinematic_viscosity](const Vector &coords, real_t t)
      {
         const real_t x = coords(0);
         const real_t y = coords(1);
         const real_t f = exp(-8.0 * nu * M_PI * M_PI * t);
         return -cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y) * f;
      };
   }
   else if (problem_type == ProblemType::CFD1_TEST ||
            problem_type == ProblemType::CFD2_TEST ||
            problem_type == ProblemType::FSI_CSM1_TEST ||
            problem_type == ProblemType::FSI1)
   {
      pressure_exact = [](const Vector &, real_t)
      {
         return 0.0;
      };
   }

   VectorFunctionCoefficient dis_exact(dim, displacement_exact);
   VectorFunctionCoefficient vel_exact(dim, velocity_exact);
   FunctionCoefficient pres_exact(pressure_exact);

   d_gf.ProjectCoefficient(dis_exact);
   d_gf = 0.0;
   d_gf.GetTrueDofs(S.GetBlock(0));

   v_gf.ProjectCoefficient(vel_exact);
   v_gf.GetTrueDofs(S.GetBlock(1));

   p_gf.ProjectCoefficient(pres_exact);
   p_gf.GetTrueDofs(S.GetBlock(2));

   ALEFSIOperator alefsi(
      theta,
      density_solid,
      density_fluid,
      kinematic_viscosity,
      H1vfes,
      H1fes,
      block_offsets,
      solid_domain_attr,
      fluid_domain_attr,
      disp_ess_attr,
      vel_ess_attr,
      pres_ess_attr,
      dis_exact,
      vel_exact,
      pres_exact,
      integration_rule,
      enable_tps);

   real_t t = 0.0;
   out << "time step: " << dt << "\n";
   real_t t_old;
   bool last_step = false;

   ParGridFunction verr_gf(&H1vfes), vex_gf(&H1vfes);
   verr_gf = 0.0;

   ParGridFunction perr_gf(&H1fes), pex_gf(&H1fes);
   perr_gf = 0.0;

   ParaViewDataCollection dc("dfem_fsi", &mesh);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(polynomial_order_h1v);
   dc.RegisterField("displacement", &d_gf);
   dc.RegisterField("velocity", &v_gf);
   dc.RegisterField("pressure", &p_gf);
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      dc.RegisterField("velocity_exact", &vex_gf);
      dc.RegisterField("velocity_error", &verr_gf);
      dc.RegisterField("pressure_exact", &pex_gf);
      dc.RegisterField("pressure_error", &perr_gf);
   }
   dc.SetCycle(0);
   dc.SetTime(0);
   dc.Save();

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - 1e-8*dt)
      {
         dt = t_final - t;
         last_step = true;
      }

      alefsi.Step(S, t, dt);

      if (last_step || (ti % vis_steps == 0))
      {
         d_gf.SetFromTrueDofs(S.GetBlock(0));
         v_gf.SetFromTrueDofs(S.GetBlock(1));
         p_gf.SetFromTrueDofs(S.GetBlock(2));

         if (Mpi::Root())
         {
            out << "step " << std::setw(5) << ti
                << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt;
            out << std::endl;
         }

         dc.SetCycle(ti);
         dc.SetTime(t);
         dc.Save();
      }
   }

   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      vel_exact.SetTime(t);
      vex_gf.ProjectCoefficient(vel_exact);
      real_t vel_l2err = v_gf.ComputeL2Error(vel_exact);

      pres_exact.SetTime(t);
      pex_gf.ProjectCoefficient(pres_exact);
      real_t pres_l2err = p_gf.ComputeL2Error(pres_exact);

      if (Mpi::Root())
      {
         out << "|u - u_exact|_L2 = " << vel_l2err
             << "\n|p - p_exact|_L2 = " << pres_l2err;
      }

      for (int i = 0; i < verr_gf.Size(); i++)
      {
         verr_gf(i) = abs(vex_gf(i) - v_gf(i));
      }

      for (int i = 0; i < perr_gf.Size(); i++)
      {
         perr_gf(i) = abs(pex_gf(i) - p_gf(i));
      }

      dc.Save();
   }

   if (problem_type == ProblemType::CFD1_TEST ||
       problem_type == ProblemType::CFD2_TEST)
   {
      DenseMatrix points(dim, 1);
      Vector pointA(2), pointB(2);

      pointA(0) = 0.25;
      pointA(1) = 0.2;

      pointB(0) = 0.15;
      pointB(1) = 0.2;

      Array<int> elem_ids;
      Array<IntegrationPoint> ips;
      points.SetCol(0, pointA);
      mesh.FindPoints(points, elem_ids, ips);
      real_t pA = p_gf.GetValue(elem_ids[0], ips[0]);

      points.SetCol(0, pointB);
      mesh.FindPoints(points, elem_ids, ips);
      real_t pB = p_gf.GetValue(elem_ids[0], ips[0]);

      out << "p(B) - p(A) = " << pB - pA << "\n";
   }

   Hypre::Finalize();
   Mpi::Finalize();

   return 0;
}
