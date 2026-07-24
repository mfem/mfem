#ifndef HEATTRANSFER_OPT_HPP
#define HEATTRANSFER_OPT_HPP

#include "mfem.hpp"
#include "../topopt_transient/ObjectiveFunctional.hpp"     // TimeIntegratedObjective (J, dJ/du)
#include "../../pde_filter.hpp"
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>

namespace mfem
{

   // =============================================================================
// SIMP MATERIAL INTERPOLATION
// =============================================================================
// Computes r(ρ̃) = r_min + ρ̃^p (r_max - r_min)
class SIMPCoefficient : public Coefficient
{
private:
   ParGridFunction *rho_filter;  // Filtered density ρ̃
   real_t r_min, r_max;
   real_t exponent;

public:
   SIMPCoefficient(ParGridFunction *rho_filt, real_t rmin, real_t rmax, real_t p)
      : rho_filter(rho_filt), r_min(rmin), r_max(rmax), exponent(p) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);  // Clamp to [0,1]
      real_t rho_pow = std::pow(rho_val, exponent);
      return r_min + rho_pow * (r_max - r_min);
   }
};

// SIMP derivative: r'(ρ̃) = p ρ̃^(p-1) (r_max - r_min)
class SIMPGradCoefficient : public Coefficient
{
private:
   GridFunction *rho_filter;
   real_t r_min, r_max;
   real_t exponent;

public:
   SIMPGradCoefficient(GridFunction *rho_filt, real_t rmin, real_t rmax, real_t p)
      : rho_filter(rho_filt), r_min(rmin), r_max(rmax), exponent(p) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);
      if (rho_val < 1e-12) return 0.0;  // Avoid singularity at ρ=0
      real_t rho_pow = std::pow(rho_val, exponent - 1.0);
      return exponent * rho_pow * (r_max - r_min);
      // return 1.0;
   }
};

// =============================================================================
// REUSABLE ADJOINT + DESIGN SENSITIVITY
// =============================================================================
inline real_t SimpDerivative(const ParGridFunction &rho_tilde,
                             ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   real_t rho = rho_tilde.GetValue(T, ip);
   rho = std::min(std::max(rho, real_t(0.0)), real_t(1.0));
   if (rho <= 0.0) { return 0.0; }
   return 3.0 * std::pow(rho, 3.0 - 1.0)
          * (1.0 - 1e-6);
}
// =============================================================================
// ABSTRACT BASE CLASS: TerminalObjective
// =============================================================================
// Interface for terminal objective functionals J = ∫_Ω j(u,T) dx
//
// Subclasses must implement:
//   - ComputeObjective: actually computes cost 
//   - ComputeObjectiveGradient: compute ∂J/∂u at one timestep (for adjoint)
//
class TerminalObjective
{
   protected:
   ParFiniteElementSpace *fespace;
   real_t cost;
   MPI_Comm comm;
   int myid;
   
   public:
   TerminalObjective(ParFiniteElementSpace *fes, MPI_Comm comm_)
      : fespace(fes), cost(0.0), comm(comm_)
   {
      MPI_Comm_rank(comm, &myid);
   }

   virtual ~TerminalObjective() = default;

   void Reset() { cost = 0.0; }

   real_t GetObjective() const { return cost; }

   inline void ComputeObjective(const ParGridFunction &u)
   {
      (void)u;
      cost = 0.0;
   }


   /// Compute objective gradient ∂J/∂u (for adjoint)
   ParGridFunction ComputeObjectiveGradient(const ParGridFunction &u, ParLinearForm &grad_form);
};

// =============================================================================
// Terminal L2 OBJECTIVE: minimize ∫ |u(t)|² dx in subdomain
// =============================================================================
class TerminalL2Objective : public TerminalObjective
{
   private:
   Coefficient *subdomain_indicator; // non-owning view used in hot paths
   std::unique_ptr<Coefficient> owned_indicator;

   public:
   /// Borrow an externally-owned indicator coefficient.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           Coefficient &indicator,
                           MPI_Comm comm_)
      : TerminalObjective(fes, comm_),
        subdomain_indicator(&indicator) {}

   /// Take ownership of an indicator coefficient.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           std::unique_ptr<Coefficient> indicator,
                           MPI_Comm comm_)
      : TerminalObjective(fes, comm_),
        subdomain_indicator(indicator.get()),
        owned_indicator(std::move(indicator)) {}

   /// Backward-compatible constructor for legacy call sites.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           Coefficient *indicator,
                           MPI_Comm comm_,
                           bool own_indicator = true)
      : TerminalObjective(fes, comm_),
        subdomain_indicator(indicator),
        owned_indicator(own_indicator ? indicator : nullptr) {}

   void ComputeObjective(const ParGridFunction &u)
   {
      // ConstantCoefficient zero(0.0);
      // cost = u.ComputeL2Error(zero)*u.ComputeL2Error(zero);
      real_t local_integral = 0.0;
      Vector u_val;

      for (int e = 0; e < fespace->GetNE(); e++)
      {
         const FiniteElement *el = fespace->GetFE(e);
         ElementTransformation *T = fespace->GetElementTransformation(e);
         const int int_order = 2 * el->GetOrder() + 2;
         const IntegrationRule &ir = IntRules.Get(el->GetGeomType(), int_order);

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            u.GetVectorValue(*T, ip, u_val);

            const real_t u_norm_sq = u_val * u_val;
            const real_t chi_val = subdomain_indicator->Eval(*T, ip);
            local_integral += ip.weight * T->Weight() * chi_val * u_norm_sq;
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&local_integral, &global_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

      cost = global_integral;
   }

   virtual ~TerminalL2Objective() = default;

/// Compute objective gradient ∂J/∂u (for adjoint) NEED TO FIX
   void ComputeObjectiveGradient(const ParGridFunction &u, ParLinearForm &grad_form)
   {
      GridFunctionCoefficient u_coef(&u);
      class ObjectiveGradientCoef : public Coefficient
      {
      private:
         GridFunctionCoefficient *u_cf;
         Coefficient *chi;

      public:
         ObjectiveGradientCoef(GridFunctionCoefficient *uc, Coefficient *c)
            : u_cf(uc), chi(c) {}

         real_t Eval(ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            const real_t chi_val = chi->Eval(T, ip);
            return 2.0 * chi_val * (u_cf->Eval(T, ip));
            // V *= 2.0;
         }
      };

      ObjectiveGradientCoef grad_coef(&u_coef, subdomain_indicator);

      class HighOrderDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape; 
         real_t q_val; 
         ObjectiveGradientCoef &q;

      public:
         HighOrderDomainLFIntegrator(ObjectiveGradientCoef &q_)
            : q(q_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int dof = el.GetDof();

            shape.SetSize(dof);
            elvect.SetSize(dof);
            elvect = 0.0;

            const int int_order = 2 * el.GetOrder() + 2;
            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), int_order);

            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               T.SetIntPoint(&ip);

               el.CalcPhysShape(T, shape);
               q_val = q.Eval(T, ip);
               const real_t trans_weight = T.Weight();
               const real_t coeff = ip.weight * trans_weight * q_val;
               for (int s = 0; s < dof; s++)
               {
                  elvect(s) += coeff * shape(s);
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(
         new HighOrderDomainLFIntegrator(grad_coef));
      grad_form.Assemble();
   }
};

// =============================================================================
// FORWARD TRAJECTORY STORAGE
// =============================================================================
// Storage for forward state needed by adjoint solver
struct ForwardTrajectoryStorage
{
   Array<Vector*> q_traj;       // Displacement at each timestep

   int num_steps;
   bool storage_enabled;

   ForwardTrajectoryStorage(int n) : num_steps(n), storage_enabled(false)
   {
      q_traj.SetSize(n);

      for (int i = 0; i < n; i++)
      {
         q_traj[i] = nullptr;
      }
   }

   void EnableStorage() { storage_enabled = true; }

   void Store(int step, const Vector &q)
   {
      if (!storage_enabled) return;

      if (step >= num_steps) return;

      if (q_traj[step]) delete q_traj[step];

      q_traj[step] = new Vector(q);
   }

   Vector Get(int step){return *q_traj[step]; }

   real_t Size(){return q_traj.Size();}

   ~ForwardTrajectoryStorage()
   {
      for (int i = 0; i < num_steps; i++)
      {
         delete q_traj[i];
      }
   }
};


class Implicit_Solver : public Solver
{
private:
   HypreParMatrix &M, &S;
   HypreParMatrix *A;
   CGSolver linear_solver;
   real_t dt;
   SparseMatrix M_diag;
   MPI_Comm comm;
public:
   Implicit_Solver(HypreParMatrix &M_, HypreParMatrix &S_,
                   const ParFiniteElementSpace &fes, real_t &dt_, MPI_Comm comm_)
      : M(M_),
        S(S_),
        A(nullptr),
        comm(comm_),
        linear_solver(comm_),
        dt(dt_)
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(-1);

      M.GetDiag(M_diag);
      // Form initial operator A = M + dt*S so the linear solver has an operator
      A = Add(dt, S, 1.0, M);
      linear_solver.SetOperator(*A);
   }

   void SetTimeStep(real_t dt_)
   {
      real_t ddt = dt-dt_;

      // syncronize ddt across all processes
      // MPI_Comm comm = M.GetComm();
      int myrank;
      MPI_Comm_rank(comm, &myrank);
      MPI_Bcast(&ddt, 1, MPI_DOUBLE, 0, comm);

      real_t epsilon;
      epsilon = std::numeric_limits<real_t>::epsilon();
      // allow for some tolerance in the time stepping process
      epsilon*=10;

      if (fabs(ddt) > epsilon)
      {
         if (0==myrank)
         {
            // std::cout << "Updating Implicit_Solver time step from " << dt 
            //      << " to " << dt_ << std::endl;
         }
         delete A;
         dt = dt_;
         // Form operator A = M + dt*S
         A = Add(dt, S, 1.0, M);
         linear_solver.SetOperator(*A);
      }
   }

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // int myrank;
      // MPI_Comm_rank(comm, &myrank);
      // std::cout << "My rank " << myrank << std::endl;
      linear_solver.Mult(x, y);
   }

   void SetPreconditioner(Solver &precond)
   {
      linear_solver.SetPreconditioner(precond);
   }

   ~Implicit_Solver() override
   {
      delete A;
   }
};


class DGStiffnessDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &rho_tilde;
   ParGridFunction &u;
   ParGridFunction &z;
   real_t diff_term;
   
   // Pre-allocated data for the Domain Integrator
   Vector shape;
   Vector grad_u, grad_z; // Vectors instead of DenseMatrix for scalar field

   // Pre-allocated data for the Face Integrator
   Vector shape1, shape2;
   Vector grad_u1, grad_u2, grad_z1, grad_z2;
   real_t kappa;

public:
   DGStiffnessDesignLFIntegrator(ParGridFunction &rho_tilde_,
                                       ParGridFunction &u_,
                                       ParGridFunction &z_, real_t &diff_term_, real_t kappa_)
      : rho_tilde(rho_tilde_), u(u_), diff_term(diff_term_), z(z_), kappa(kappa_){}

   // -------------------------------------------------------------------------
   // 1. Domain Integrator (Scalar Diffusion)
   // -------------------------------------------------------------------------
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &T,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      const int dim = T.GetSpaceDim();
      
      shape.SetSize(dof);
      grad_u.SetSize(dim);
      grad_z.SetSize(dim);
      
      elvect.SetSize(dof);
      elvect = 0.0;

      const int int_order = 2 * T.OrderGrad(&el);
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape);

         // For scalar fields, GetGradient populates a Vector
         u.GetGradient(T, grad_u);
         z.GetGradient(T, grad_z);

         // Diffusion energy density: dot product of gradients
         real_t diffusion_density = 0.0;
         for (int i = 0; i < dim; i++)
         {
            diffusion_density += grad_u(i) * grad_z(i);
         }
         diffusion_density *= diff_term;

         const real_t rp = SimpDerivative(rho_tilde, T, ip);
         
         // Assuming the same negative adjoint/compliance convention
         const real_t density = rp * diffusion_density;
         const real_t weight = ip.weight * T.Weight() * density;

         for (int i = 0; i < dof; i++)
         {
            elvect(i) += weight * shape(i);
         }
      }
   }

   // -------------------------------------------------------------------------
   // 2. Interior Face Integrator (Scalar DG Jump and Average Terms)
   // -------------------------------------------------------------------------
   void AssembleRHSElementVect(const FiniteElement &el1,
                               const FiniteElement &el2,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      const int dof1 = el1.GetDof();
      const int dof2 = el2.GetDof();
      const int dim = Tr.GetSpaceDim();

      shape1.SetSize(dof1);
      shape2.SetSize(dof2);
      elvect.SetSize(dof1 + dof2);
      elvect = 0.0;
      
      grad_u1.SetSize(dim); grad_u2.SetSize(dim);
      grad_z1.SetSize(dim); grad_z2.SetSize(dim);

      const int int_order = 2 * std::max(el1.GetOrder(), el2.GetOrder());
      const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetAllIntPoints(&ip);

         el1.CalcPhysShape(*Tr.Elem1, shape1);
         el2.CalcPhysShape(*Tr.Elem2, shape2);

         // Scalar values at the interface
         const real_t u1 = u.GetValue(*Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t u2 = u.GetValue(*Tr.Elem2, Tr.Elem2->GetIntPoint());
         const real_t z1 = z.GetValue(*Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t z2 = z.GetValue(*Tr.Elem2, Tr.Elem2->GetIntPoint());

         u.GetGradient(*Tr.Elem1, grad_u1);
         u.GetGradient(*Tr.Elem2, grad_u2);
         z.GetGradient(*Tr.Elem1, grad_z1);
         z.GetGradient(*Tr.Elem2, grad_z2);

         // Normal vector from Elem1 to Elem2
         Vector nor(dim);
         CalcOrtho(Tr.Jacobian(), nor);
         const real_t weight = nor.Norml2(); 
         nor /= weight; // Normalize

         // Scalar jumps [u] = u1 - u2
         const real_t jump_u = u1 - u2;
         const real_t jump_z = z1 - z2;
         const real_t jump_dot = jump_u * jump_z;

         // Directional derivatives (grad \cdot n)
         real_t grad_u1_n = 0.0, grad_u2_n = 0.0;
         real_t grad_z1_n = 0.0, grad_z2_n = 0.0;
         for (int i = 0; i < dim; i++)
         {
            grad_u1_n += grad_u1(i) * nor(i);
            grad_u2_n += grad_u2(i) * nor(i);
            grad_z1_n += grad_z1(i) * nor(i);
            grad_z2_n += grad_z2(i) * nor(i);
         }

         // Interface flux terms
         const real_t flux_u1 = grad_u1_n * jump_z;
         const real_t flux_u2 = grad_u2_n * jump_z;
         const real_t flux_z1 = grad_z1_n * jump_u;
         const real_t flux_z2 = grad_z2_n * jump_u;

         // Characteristic length scale (h_f)
         const real_t h1 = Tr.Elem1->Weight() / weight;
         const real_t h2 = Tr.Elem2->Weight() / weight;
         const real_t h_f = 0.5 * (h1 + h2);
         // Penalty applied to scalar jump
         const real_t penalty_val = (kappa / h_f) * jump_dot;

         const real_t rp1 = diff_term*SimpDerivative(rho_tilde, *Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t rp2 = diff_term*SimpDerivative(rho_tilde, *Tr.Elem2, Tr.Elem2->GetIntPoint());

         // DG face energy density derivative
         const real_t D1 = -0.5 * rp1 * (flux_u1 + flux_z1 - penalty_val);
         const real_t D2 = -0.5 * rp2 * (flux_u2 + flux_z2 - penalty_val);

         const real_t w_D1 = ip.weight * weight * D1;
         const real_t w_D2 = ip.weight * weight * D2;

         for (int i = 0; i < dof1; i++)
         {
            elvect(i) += w_D1 * shape1(i);
         }
         for (int i = 0; i < dof2; i++)
         {
            elvect(dof1 + i) += w_D2 * shape2(i);
         }
      }
   }

   // -------------------------------------------------------------------------
   // 3. Boundary Face Integrator (Optional)
   // -------------------------------------------------------------------------
   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      elvect.SetSize(dof);
      elvect = 0.0;
      
      // Implement scalar Nitsche boundary flux differentiation here if needed
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
};

class DGAdvectionDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &rho_tilde;
   const ParGridFunction &u;
   const ParGridFunction &z;
   VectorCoefficient &v_base;
   real_t alpha;
   
   // Pre-allocated data for the Domain Integrator
   Vector shape;
   Vector grad_u;
   Vector v_val;

   // Pre-allocated data for the Face Integrators
   Vector shape1, shape2;

public:
   DGAdvectionDesignLFIntegrator(ParGridFunction &rho_tilde_,
                                 const ParGridFunction &u_,
                                 const ParGridFunction &z_,
                                 VectorCoefficient &v_base_,
                                 real_t alpha_ = -1.0)
      : rho_tilde(rho_tilde_), u(u_), z(z_), v_base(v_base_), alpha(alpha_) {}

   // -------------------------------------------------------------------------
   // 1. Domain Integrator (Advection)
   // -------------------------------------------------------------------------
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &T,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      const int dim = T.GetSpaceDim();
      
      shape.SetSize(dof);
      grad_u.SetSize(dim);
      v_val.SetSize(dim);
      
      elvect.SetSize(dof);
      elvect = 0.0;

      const int int_order = 2 * T.OrderGrad(&el);
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape);

         u.GetGradient(T, grad_u);
         const real_t z_val = z.GetValue(T, ip);
         
         // Evaluate the base velocity vector
         v_base.Eval(v_val, T, ip);

         real_t v_dot_gradu = 0.0;
         for (int i = 0; i < dim; i++)
         {
            v_dot_gradu += v_val(i) * grad_u(i);
         }

         // Advection energy density
         const real_t D = alpha * v_dot_gradu * z_val;

         // Derivative of SIMP scaling function w.r.t rho
         const real_t rp = SimpDerivative(rho_tilde, T, ip);
         
         // Negative sign matches adjoint convention 
         const real_t weight = rp * ip.weight * T.Weight() * D;

         for (int i = 0; i < dof; i++)
         {
            elvect(i) += weight * shape(i);
         }
      }
   }

   // -------------------------------------------------------------------------
   // 2. Interior Face Integrator (DG Upwind Advection)
   // -------------------------------------------------------------------------
   void AssembleRHSElementVect(const FiniteElement &el1,
                               const FiniteElement &el2,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      const int dof1 = el1.GetDof();
      const int dof2 = el2.GetDof();
      const int dim = Tr.GetSpaceDim();

      shape1.SetSize(dof1);
      shape2.SetSize(dof2);
      elvect.SetSize(dof1 + dof2);
      elvect = 0.0;
      v_val.SetSize(dim);

      const int int_order = std::min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) + 2 * std::max(el1.GetOrder(), el2.GetOrder()) ;
      const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetAllIntPoints(&ip);

         el1.CalcPhysShape(*Tr.Elem1, shape1);
         el2.CalcPhysShape(*Tr.Elem2, shape2);

         const real_t u1 = u.GetValue(*Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t u2 = u.GetValue(*Tr.Elem2, Tr.Elem2->GetIntPoint());
         const real_t z1 = z.GetValue(*Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t z2 = z.GetValue(*Tr.Elem2, Tr.Elem2->GetIntPoint());

         v_base.Eval(v_val, Tr, ip);

         // Extract geometric normal vector
         Vector nor(dim);
         CalcOrtho(Tr.Jacobian(), nor);
         const real_t face_weight = nor.Norml2(); 
         nor /= face_weight; // Normalize to get true normal dot product

         // Calculate normal velocity component
         real_t vn = 0.0;
         for (int i = 0; i < dim; i++)
         {
             vn += v_val(i) * nor(i);
         }

         // Nonconservative upwind logic
         const real_t jump_u = u2 - u1;
         const real_t z_up = (vn >= 0.0) ? z2 : z1;
         
         const real_t d_integrand = alpha * vn * jump_u * z_up;

         // Evaluate SIMP derivatives on both sides of the face
         const real_t rp1 = SimpDerivative(rho_tilde, *Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t rp2 = SimpDerivative(rho_tilde, *Tr.Elem2, Tr.Elem2->GetIntPoint());

         // Assuming the velocity coeff evaluates standard DG face averaging {v}
         const real_t w_D1 = 0.5 * rp1 * ip.weight * face_weight * d_integrand;
         const real_t w_D2 = 0.5 * rp2 * ip.weight * face_weight * d_integrand;

         for (int i = 0; i < dof1; i++) elvect(i) += w_D1 * shape1(i);
         for (int i = 0; i < dof2; i++) elvect(dof1 + i) += w_D2 * shape2(i);
      }
   }

   // -------------------------------------------------------------------------
   // 3. Boundary Face Integrator (Outflow/Inflow)
   // -------------------------------------------------------------------------
   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      const int dim = Tr.GetSpaceDim();
      real_t beta = alpha/2.0;

      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;
      v_val.SetSize(dim);

      const int int_order = Tr.Elem1->OrderW() + 2 * el.GetOrder();
      const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetAllIntPoints(&ip);
         el.CalcPhysShape(*Tr.Elem1, shape);

         const real_t u1 = u.GetValue(Tr, ip);
         const real_t z1 = z.GetValue(Tr, ip);

         v_base.Eval(v_val, Tr, ip);

         Vector nor(dim);
         CalcOrtho(Tr.Jacobian(), nor);

         // if (nor(0) > 0.0)
         // {
         //    nor(0) = 1.0;
         // }
         // else{
         //    nor(1) = -1.0;
         // }
         // std::cout << "nor(0) = " << nor(0) << "nor(1) = " << nor(1) << std::endl; 
         const real_t face_weight = nor.Norml2(); 
         // nor /= face_weight;

         real_t vn = 0.0;
         vn = v_val*nor;
         // for (int i = 0; i < dim; i++) vn += v_val(i) * nor(i);

         const real_t d_integrand = -0.5 * alpha * vn * (u1) * z1 + beta*fabs(vn) * (u1) * z1; 
         const real_t rp = SimpDerivative(rho_tilde, *Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t weight = rp * ip.weight * d_integrand;
         for (int i = 0; i < dof; i++)
         {
             elvect(i) += weight * shape(i);
         }
         // Nonconservative DG Trace only penalizes the inflow boundary (vn < 0)
         // Treating the exterior value as u_ext = 0
         // if (vn < 0.0)
         // {
         //     const real_t d_integrand = alpha * vn * (-u1) * z1; 
             
         //     const real_t rp = SimpDerivative(rho_tilde, *Tr.Elem1, Tr.Elem1->GetIntPoint());
         //     const real_t weight = rp * ip.weight*face_weight*d_integrand;

         //     for (int i = 0; i < dof; i++)
         //     {
         //         elvect(i) += weight * shape(i);
         //     }
         // }
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
};

class BdrFlowDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &rho_tilde;
   ParGridFunction &z;            // The adjoint state
   Coefficient &inflow;           // The prescribed inflow boundary data
   VectorCoefficient &v_base;     // The unscaled base velocity field
   real_t alpha;

   Vector shape;
   Vector v_val;

public:
   BdrFlowDesignLFIntegrator(ParGridFunction &rho_tilde_,
                             ParGridFunction &z_,
                             Coefficient &inflow_,
                             VectorCoefficient &v_base_,
                             real_t alpha_ = -1.0)
      : rho_tilde(rho_tilde_), z(z_), inflow(inflow_), 
        v_base(v_base_), alpha(alpha_) {}

   // Provide a no-op domain integrator implementation so this class
   // is not abstract (we only use it for boundary face integration).
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      elvect.SetSize(el.GetDof());
      elvect = 0.0;
   }

   // -------------------------------------------------------------------------
   // Boundary Face Integrator (Sensitivity of linear form 'b')
   // -------------------------------------------------------------------------
   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      const int dim = Tr.GetSpaceDim();
      real_t beta = 0.5*alpha;

      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;
      v_val.SetSize(dim);

      // Integration rule order
      // const int int_order = 2 * el.GetOrder(); 
      const int int_order = Tr.Elem1->OrderW() + 2*el.GetOrder();
      const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), int_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetAllIntPoints(&ip);
         const IntegrationPoint &eip = Tr.GetElement1IntPoint();
         el.CalcShape(eip, shape);

         // Evaluate the adjoint state (z) and prescribed inflow at the boundary
         const real_t z_val = z.GetValue(*Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t u_in = inflow.Eval(Tr, ip);

         // Evaluate the unscaled velocity vector
         v_base.Eval(v_val, *Tr.Elem1, eip);

         // Calculate and normalize the geometric normal vector
         Vector nor(dim);
         CalcOrtho(Tr.Jacobian(), nor);
         // const real_t face_weight = nor.Norml2(); 
         // nor /= face_weight;

         // Compute normal velocity component (v_base \cdot n)
         real_t vn = 0.0;
         for (int i = 0; i < dim; i++)
         {
             vn += v_val(i) * nor(i);
         }

         // The original integrand was: alpha * (v_n) * inflow * test_function
         const real_t b_integrand = 0.5 * alpha * vn * u_in * z_val + beta * alpha * fabs(vn) * u_in * z_val;

         // Derivative of SIMP scaling function w.r.t rho
         const real_t rp = SimpDerivative(rho_tilde, *Tr.Elem1, Tr.Elem1->GetIntPoint());

         // Note the POSITIVE sign here! 
         // This matches the + z^T (\delta b) term in the adjoint equation.
         const real_t weight = rp * ip.weight * b_integrand;

         for (int i = 0; i < dof; i++)
         {
             elvect(i) += weight * shape(i);
         }
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of the advection-diffusion equation is (M + dt S) du/dt = Su - K u + b
    , where M and K are the mass and advection matrices, and b describes the
    flow on the boundary. In the case of IMEX evolution, the diffusion term is
    treated implicitly, and the advection term is treated explicitly.  */
class IMEXAdvectionDiffusionSolver : public TimeDependentOperator
{
   protected:
   ParFiniteElementSpace *fespace;
   ParFiniteElementSpace *filter_fes;
   ParBilinearForm *M, *K, *S, *A; 
   mutable ParBilinearForm *Kd;
   std::unique_ptr<HypreParMatrix> M_mat, S_mat, K_mat;
   mutable std::unique_ptr<HypreParMatrix> Kd_mat;
   ParLinearForm *b;
   std::unique_ptr<HypreParVector> b_vec;
   Solver *M_prec;
   CGSolver *M_solver;
   Implicit_Solver *implicit_solver;
   LORSolver<HypreBoomerAMG>* lor_solver;
   mutable ParGridFunction q_gf;
   Array<int> ess_bdr_attr;
   Array<int> ess_tdof_list;
   mutable Array<int> inflow_bdr_attr;
   int true_size;
   MPI_Comm comm;
   bool adjoint;
   int current_step;
   real_t kappa;
   real_t raw_diff_term;
   //  std::unique_ptr<ODESolver> ode_solver;

   ScalarVectorProductCoefficient velocity_coeff;
   mutable VectorFunctionCoefficient v_base;
   ProductCoefficient diffusion_coeff;
   ForwardTrajectoryStorage *trajectory;
   TerminalObjective *objective;
   GridFunctionCoefficient q0;
   mutable FunctionCoefficient inflow;
   // mutable int current_adjoint_step;
   real_t dt;
   ProductCoefficient dt_diff_coeff;
   mutable Vector z;
   mutable Vector w;
   mutable Vector design_gradient;
   real_t t_final;
   mutable ParGridFunction rho_tilde;
   // real_t curr_time;
   public:
   IMEXAdvectionDiffusionSolver(ParFiniteElementSpace &fes, ScalarVectorProductCoefficient &velocity_coeff, VectorFunctionCoefficient &v_base, ProductCoefficient &dt_diff_coeff, ProductCoefficient &diffusion_coeff, FunctionCoefficient &inflow, Array<int> &inflow_bdr_attr_, GridFunctionCoefficient &q0, ParGridFunction &rho_tilde, real_t dt, real_t t_final, real_t raw_diff_term, MPI_Comm comm, Array<int> &ess_bdr_attr_, TerminalObjective *obj = nullptr);
   void Mult1(const Vector &x, Vector &y) const;
   void ImplicitSolve2(const real_t dt, const Vector &x, Vector &k);
   void JacobianMult1Transpose(const Vector &lam, Vector &lam_rhs, Vector &x) const;
   void AdjointImplicitSolve2(const real_t dt, const Vector &lam, Vector &x, Vector &k);
   void Mult(const Vector &x, Vector &y) const override
   {
      Mult1(x,y);
   }
   void ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k) override 
   {
      ImplicitSolve2(dt_pass,x,k);
   }
   void AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const
   {
      JacobianMult1Transpose(lam, lam_rhs, x);
   }
   void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &x,  Vector &k) 
   {
      AdjointImplicitSolve2(dt_pass,lam,x,k);
   }


   const Array<int>& GetEssentialTrueDofs() const { return ess_tdof_list; }
   //  void InitTimeStepping(); 
   //  void Step();

   void UpdateDt(real_t dt_real)
   {
      MPI_Bcast(&dt_real, 1, MPI_DOUBLE, 0, comm);
      dt = dt_real;
   }

   ParGridFunction& Getq() { return q_gf; }


   void Updateq(ParGridFunction &new_q_gf) {q_gf = new_q_gf;}

    

   void SetTrajectory(ForwardTrajectoryStorage *traj) { trajectory = traj; }

   void TakeAdjoint(){ adjoint = true; }

   void SetObjective(TerminalObjective *obj) { objective = obj; }

   void StoreTraj(int step, Vector &q_vec){trajectory->Store(step, q_vec);}

   void GetTraj(int step, Vector &q_vec){q_vec = trajectory->Get(step);}

   void SetStep(int new_step){current_step = new_step;}

   int GetStep(){return current_step;}

   Vector GetDesignGrad(){return design_gradient;}


   void ComputeObjectiveGradient(Vector &grad_vec) const
   {
      grad_vec = 0.0;
      if (!objective || !trajectory) return;
      // Get the state variable;
      // if (!q_gf) return;
      // Set grid function from stored state
      // Compute ∂J_Ω/∂u = 2 χ_Ω̃ u (from ObjectiveFunctional)
      ParLinearForm grad_form(fespace);
      // objective->ComputeObjectiveGradient(q_gf, grad_form);
      // grad_form.ParallelAssemble(grad_vec);
    }

    // Update Destructor
   virtual ~IMEXAdvectionDiffusionSolver()
   {
      delete implicit_solver;
      delete lor_solver;
      delete M_prec;
      delete M_solver;
      delete trajectory;
      delete M;
      delete K;
      delete S;
      delete A;
      delete b;
   }
};




IMEXAdvectionDiffusionSolver::IMEXAdvectionDiffusionSolver(ParFiniteElementSpace &fes_, ScalarVectorProductCoefficient &velocity_coeff_, VectorFunctionCoefficient &v_base_, ProductCoefficient &dt_diff_coeff_, ProductCoefficient &diffusion_coeff_, FunctionCoefficient &inflow_, Array<int> &inflow_bdr_attr_, GridFunctionCoefficient &q0_, ParGridFunction &rho_tilde_, real_t dt_, real_t t_final_, real_t raw_diff_term_, MPI_Comm comm_, Array<int> &ess_bdr_attr_, TerminalObjective *obj_)
   : TimeDependentOperator(fes_.GetTrueVSize()), 
   fespace(&fes_), 
   velocity_coeff(velocity_coeff_), 
   diffusion_coeff(diffusion_coeff_), 
   dt_diff_coeff(dt_diff_coeff_),
   inflow(inflow_), 
   q0(q0_),
   objective(obj_),
   inflow_bdr_attr(inflow_bdr_attr_),
   comm(comm_),
   v_base(v_base_),
   z(fes_.GetTrueVSize()),
   w(fes_.GetTrueVSize()),
   ess_bdr_attr(ess_bdr_attr_),
   rho_tilde(rho_tilde_),
   t_final(t_final_),
   raw_diff_term(raw_diff_term_),
   dt(dt_)
{
   adjoint = false;
   int order = fespace->GetOrder(0);
   kappa = (order + 1)*(order + 1);
   const real_t sigma = -1.0;
   int myid = Mpi::WorldRank();
   fespace->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);
   //  std::unique_ptr<ODESolver> ode_solver = ODESolver::SelectIMEX(ode_solver_type);
   //  *ode_solver = *ode_solver_up;
   ParMesh *pmesh = fespace->GetParMesh();
   int max_bdr_attr = pmesh->bdr_attributes.Max();
   // inflow_bdr.SetSize(max_bdr_attr);
   // inflow_bdr = 0;
   // inflow_bdr[1] = 1;

   // Array<int> outflow_bdr;
   // outflow_bdr.SetSize(max_bdr_attr);
   // outflow_bdr = 0;
   // outflow_bdr[3] = 1;

    // Form the Mass Integrator 
   rho_tilde.ExchangeFaceNbrData();
   M = new ParBilinearForm(fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   // Form the DG Conevection Matrix
   constexpr real_t alpha = -1.0;
   K = new ParBilinearForm(fespace);
   K->AddDomainIntegrator(new ConvectionIntegrator(velocity_coeff, alpha));
   K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_coeff, alpha));                                                       
   K->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_coeff, alpha));
   // K->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_coeff, alpha), outflow_bdr);
   // Form DG Stiffness Matrix
   S = new ParBilinearForm(fespace);

   S->AddDomainIntegrator(new DiffusionIntegrator(diffusion_coeff));
   S->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diffusion_coeff, sigma, kappa));
   //S->AddBdrFaceIntegrator(new DGDiffusionIntegrator(diffusion_coeff, sigma, kappa));
   // For the preconditioner - create billinear form corresponding to
   // operator (M + dt S)
   A = new ParBilinearForm(fespace);
   A->AddDomainIntegrator(new MassIntegrator);
   A->AddDomainIntegrator(new DiffusionIntegrator(dt_diff_coeff));
   A->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_diff_coeff, sigma, kappa));
   //A->AddBdrFaceIntegrator(new DGDiffusionIntegrator(dt_diff_coeff, sigma, kappa));
   M->Assemble();
   K->Assemble();
   // Sanity-check: evaluate diffusion coefficient at first element/int point
   S->Assemble();
   A->Assemble();
   M->Finalize();
   K->Finalize();
   S->Finalize();
   A->Finalize();
   //  Array<int> inflow_bdr = ess_bdr_attr;
   //  inflow_bdr = 0;
   //  inflow_bdr[0] = 1;
   b = new ParLinearForm(fespace);
   b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity_coeff, alpha), inflow_bdr_attr);
   b->Assemble();
   // b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity_coeff, alpha), outflow_bdr);
   // b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(inflow, diffusion_coeff, sigma, kappa));
   b_vec.reset(b->ParallelAssemble());


   //  A->Reset(A->ParallelAssemble(), true);
   M_mat.reset(M->ParallelAssemble());
   S_mat.reset(S->ParallelAssemble());
   K_mat.reset(K->ParallelAssemble());
   HypreSmoother *hypre_prec = new HypreSmoother(*M_mat, HypreSmoother::Jacobi);
   M_prec = hypre_prec;
   implicit_solver = new Implicit_Solver(*M_mat, *S_mat, *fespace, dt, comm);
   lor_solver = new LORSolver<HypreBoomerAMG>(*A, ess_tdof_list);
   lor_solver->GetSolver().SetSystemsOptions(fespace->GetVDim(), true);
   lor_solver->GetSolver().SetPrintLevel(-1);
   implicit_solver -> SetPreconditioner(*lor_solver);
   t = 0.0;
   // allocate temporary vectors used by Mult/ImplicitSolve
   //  z.SetSize(fespace.GetTrueVSize());
   //  w.SetSize(fespace.GetTrueVSize());
   q_gf.SetSpace(fespace);
   q_gf.ProjectCoefficient(q0);
   q_gf.ExchangeFaceNbrData();
   M_solver = new CGSolver(comm);
   M_solver->SetOperator(*M_mat);
   M_solver->SetPreconditioner(*M_prec);
   M_solver->iterative_mode = false;
   M_solver->SetRelTol(1e-13);
   M_solver->SetAbsTol(0.0);
   M_solver->SetMaxIter(100);
   M_solver->SetPrintLevel(0);


   filter_fes = rho_tilde.ParFESpace();
   design_gradient.SetSize(filter_fes->GetTrueVSize());
   design_gradient = 0.0;



   int n_steps = (int)ceil(t_final / dt);
   trajectory = new ForwardTrajectoryStorage(n_steps);
   trajectory->EnableStorage();
   Vector q_vec = q_gf;
   trajectory->Store(0, q_vec);
}

void IMEXAdvectionDiffusionSolver::Mult1(const Vector &x, Vector &y) const
{
   int myrank;
   MPI_Comm_rank(comm, &myrank);
   // Perform the explicit step
   // y = M^{-1} (K x + b)
   K_mat->Mult(x, z);
   z += *b_vec;
   M_solver->Mult(z, y);
   //std::cout << "explicit my_rank = " << myrank  << ", ||q_in|| = " << x.Norml2() << ", ||b|| = " << b_vec->Norml2() << ", ||z|| = " << z.Norml2() << ", ||q_out|| = " << y.Norml2() << std::endl;
}

void IMEXAdvectionDiffusionSolver::ImplicitSolve2(const real_t dt_pass, const Vector &x, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(implicit_solver != NULL,
               "Implicit time integration is not supported with partial assembly");

   int myrank;
   MPI_Comm_rank(comm, &myrank);
   S_mat->Mult(x, z);
   z *= -1.0;
   implicit_solver->SetTimeStep(dt_pass);
   implicit_solver->Mult(z, k);
}

void IMEXAdvectionDiffusionSolver::AdjointImplicitSolve2(const real_t dt_pass, const Vector &lam, Vector &x, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(implicit_solver != NULL,
               "Implicit time integration is not supported with partial assembly");

   implicit_solver->SetTimeStep(dt_pass);
   implicit_solver->Mult(lam, z);
   z *= -1.0;
   S_mat->Mult(z, k);
   //lam A^{-1} dS/drho A^{-1} S q
   Vector k_d(lam.Size()); 
   Vector y(lam.Size());
   Vector u(lam.Size());
   implicit_solver->Mult(lam, w); // w = A^{-1} lam, A is self adjoint
   //Vector q_vec = trajectory->Get(current_step-1);
   M_mat->Mult(x, u);
   implicit_solver->Mult(u, y); // y = A^{-1}S q
   ParLinearForm stiff_lf1(filter_fes); 
   ParGridFunction w_gf(fespace);
   ParGridFunction y_gf(fespace);
   w_gf.SetFromTrueDofs(w);
   y_gf.SetFromTrueDofs(y);
   rho_tilde.ExchangeFaceNbrData();
   w_gf.ExchangeFaceNbrData();
   y_gf.ExchangeFaceNbrData();
   stiff_lf1.AddDomainIntegrator(new DGStiffnessDesignLFIntegrator(rho_tilde, y_gf, w_gf, raw_diff_term, kappa));
   stiff_lf1.AddInteriorFaceIntegrator(new DGStiffnessDesignLFIntegrator(rho_tilde, y_gf, w_gf, raw_diff_term, kappa));
   stiff_lf1.Assemble();
   std::unique_ptr<HypreParVector> stiff_vec1(stiff_lf1.ParallelAssemble());    
   design_gradient.Add(dt, *stiff_vec1); 
}



void IMEXAdvectionDiffusionSolver::JacobianMult1Transpose(const Vector &lam, Vector &lam_rhs, Vector &x) const
{
   // Plain transpose of the forward RHS Jacobian:
   // G(u) = M^{-1} (K u + b)
   // lam_rhs = 0.0;
   // Adjoint RHS evaluation for discrete adjoint 
   // Jac(G) = M^{-1} K 
   // Jac(G)^T = K^{T} M^{-T} 
   z = 0.0;
   M_solver->Mult(lam, z);
   K_mat->MultTranspose(z, lam_rhs);

   // Update the design gradient
   M_solver->Mult(lam, w);
   ParLinearForm adv_lf(filter_fes);
   // Vector q_vec = trajectory->Get(current_step-1);
   // std::cout<<"current step = "<<current_step << std::endl;
   // Vector wf(filter_fes->GetTrueVSize()), qf(filter_fes->GetTrueVSize());
   // Mixed_Mass_mat->Mult(w, wf);
   // Mixed_Mass_mat->Mult(q_vec, qf);

   ParGridFunction lam_gf(fespace);
   lam_gf.SetFromTrueDofs(w);
   ParGridFunction qq_gf(fespace);
   qq_gf.SetFromTrueDofs(x);
   rho_tilde.ExchangeFaceNbrData();
   lam_gf.ExchangeFaceNbrData();
   qq_gf.ExchangeFaceNbrData();

   adv_lf.AddDomainIntegrator(new DGAdvectionDesignLFIntegrator(rho_tilde, qq_gf, lam_gf, v_base));
   adv_lf.AddBdrFaceIntegrator(new DGAdvectionDesignLFIntegrator(rho_tilde, qq_gf, lam_gf, v_base));
   adv_lf.AddInteriorFaceIntegrator(new DGAdvectionDesignLFIntegrator(rho_tilde, qq_gf, lam_gf, v_base));
   adv_lf.Assemble();
   std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
   design_gradient.Add(-dt, *adv_vec);


   ParLinearForm bdr_flow_lf(filter_fes);
   bdr_flow_lf.AddBdrFaceIntegrator(new BdrFlowDesignLFIntegrator(rho_tilde, lam_gf, inflow, v_base), inflow_bdr_attr);
   bdr_flow_lf.Assemble();
   std::unique_ptr<HypreParVector> bdr_flow_vec(bdr_flow_lf.ParallelAssemble());
   design_gradient.Add(dt, *bdr_flow_vec);
}

// =============================================================================
// IMEX ODESolvers for Design Opt
// =============================================================================
// Note, Time dependent operator f must have adjointmult

class TopOptIMEXSolver : public ODESolver
{
protected:
   IMEXAdvectionDiffusionSolver *f;
public:
   virtual void Init(IMEXAdvectionDiffusionSolver &f_) = 0;
   virtual void AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x) = 0;
   virtual void Step(Vector &x, real_t &t, real_t &dt) = 0;
   // virtual ~TopOptIMEXSolver();
};

void TopOptIMEXSolver::Init(IMEXAdvectionDiffusionSolver &f_)
{
   this->f = &f_;
   mem_type = GetMemoryType(f_.GetMemoryClass());
}


class TopOptIMEXExpImplEuler : public TopOptIMEXSolver
{
private:
   Vector k1; Vector k2;
public:
   void Init(IMEXAdvectionDiffusionSolver &f_) override;

   void Step(Vector &x, real_t &t, real_t &dt) override;

   void AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x) override;
};

void TopOptIMEXExpImplEuler::Init(IMEXAdvectionDiffusionSolver &f_)
{
   TopOptIMEXSolver::Init(f_);
   int n = f->Width();
   k1.SetSize(n, mem_type);
   k2.SetSize(n, mem_type);
}

void TopOptIMEXExpImplEuler::Step(Vector &x, real_t &t, real_t &dt)
{
   f->SetTime(t);
   f->Mult(x, k1);

   f->SetTime(t+dt);
   f->ImplicitSolve(dt, x, k2);

   f->SetTime(t);
   x.Add(dt, k1);
   x.Add(dt, k2);
   t += dt;
}

void TopOptIMEXExpImplEuler::AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x)
{
   f->SetTime(t);
   f->AdjointMult(lam, k1, x);

   f->SetTime(t+dt);
   f->AdjointImplicitSolve(dt, lam, x, k2);

   f->SetTime(t);
   lam.Add(dt, k1);
   lam.Add(dt, k2);
   t += dt;
}

/// Second order, two-stage implicit-explicit (IMEX) Runge-Kutta (RK) method
/** L-stable IMEX RK2 method adopted from "On the Stability of IMEX Upwind gSBP
    Schemes for 1D Linear Advection‑Difusion Equations" by Sigrun Ortleb. Same
    as (2,2,2) from "Implicit-explicit Runge-Kutta methods for time-dependent
    partial differential equations" by Ascher, Ruuth and Spiteri, Applied
    Numerical Mathematics (1997). */
class TopOptIMEXRK2 : public TopOptIMEXSolver
{
private:
   Vector k1_exp; Vector k2_exp; Vector k_imp;
   //helper vector
   Vector y;
public:
   void Init(IMEXAdvectionDiffusionSolver &f_) override;

   void Step(Vector &x, real_t &t, real_t &dt) override;

   void AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x) override;
};

void TopOptIMEXRK2::Init(IMEXAdvectionDiffusionSolver &f_)
{
   TopOptIMEXSolver::Init(f_);
   int n = f->Width();
   k1_exp.SetSize(n, mem_type);
   k2_exp.SetSize(n, mem_type);
   k_imp.SetSize(n, mem_type);
   y.SetSize(n, mem_type);
}

void TopOptIMEXRK2::Step(Vector &x, real_t &t, real_t &dt)
{
   double gamma = 1 - sqrt(2)/2;
   double delta = 1 - 1/(2*gamma);

   f->SetTime(t);

   //K1 exp is just f_1(t, x)
   f->Mult(x, k1_exp);

   //K2 exp is f_1(t + gamma dt, x + dt gamma K1)
   f->SetTime(t + gamma*dt);
   add(x, dt*gamma, k1_exp, y);
   f->Mult(y, k2_exp);

   //K2_imp = f_2(t + gamma dt, x + dt gamma K2_imp)
   f->ImplicitSolve(dt*gamma, x, k_imp);
   //reuse k_imp to avoid extra vector

   //K3_imp = f_2(t+dt,x + dt(1-gamma)K2_imp + dt gamma K3_imp)
   f -> SetTime(t + dt);
   //add(x, dt*(1-gamma), k2_imp, z);
   //optimization to avoid extra vector
   x.Add(dt*(1-gamma), k_imp);
   //f->ImplicitSolve(dt*gamma, z, k3_imp);
   //reuse k_imp to avoid extra vector
   f->ImplicitSolve(dt*gamma, x, k_imp);

   //add it all up
   f->SetTime(t);
   x.Add(dt*delta, k1_exp);
   x.Add(dt*(1-delta), k2_exp);
   //x.Add(dt*(1-gamma), k2_imp); it is already added to x above
   x.Add(dt*gamma, k_imp);
   t += dt;
}

void TopOptIMEXRK2::AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x)
{
   double gamma = 1 - sqrt(2)/2;
   double delta = 1 - 1/(2*gamma);
   int n = lam.Size();

   f->SetTime(t);

   Vector x1(n), x2(n), x3(n), ys(n), yi(n), x4(n);
   f->Mult(x, x1);

   //K2 exp is f_1(t + gamma dt, x + dt gamma K1)
   f->SetTime(t + gamma*dt);
   add(x, dt*gamma, k1_exp, ys);
   f->UpdateDt(gamma*dt);
   f->Mult(ys, x2);
   f->UpdateDt(dt);

   //K2_imp = f_2(t + gamma dt, x + dt gamma K2_imp)
   f->ImplicitSolve(dt*gamma, x, x3);
   //reuse k_imp to avoid extra vector

   //K3_imp = f_2(t+dt,x + dt(1-gamma)K2_imp + dt gamma K3_imp)
   f -> SetTime(t + dt);
   //add(x, dt*(1-gamma), k2_imp, z);
   //optimization to avoid extra vector
   add(x, dt*(1-gamma), x3, yi);
   //f->ImplicitSolve(dt*gamma, z, k3_imp);
   //reuse k_imp to avoid extra vector
   f->ImplicitSolve(dt*gamma, yi, x4);

   /////////////////////////////

   //K1 exp is just f_1(t, x)
   f->UpdateDt(delta*dt);
   f->AdjointMult(lam, k1_exp, x);

   //K2 exp is f_1(t + gamma dt, x + dt gamma K1)
   f->SetTime(t + gamma*dt);
   add(lam, dt*gamma, k1_exp, y);
   f->UpdateDt((1-delta)*dt);
   f->AdjointMult(y, k2_exp, x);
   // f->UpdateDt(dt);

   //K2_imp = f_2(t + gamma dt, x + dt gamma K2_imp)
   f->UpdateDt((1-gamma)*dt);
   f->AdjointImplicitSolve(dt*gamma, lam, x, k_imp);
   //reuse k_imp to avoid extra vector

   //K3_imp = f_2(t+dt,x + dt(1-gamma)K2_imp + dt gamma K3_imp)
   f -> SetTime(t + dt);
   //add(x, dt*(1-gamma), k2_imp, z);
   //optimization to avoid extra vector
   lam.Add(dt*(1-gamma), k_imp);
   //f->ImplicitSolve(dt*gamma, z, k3_imp);
   //reuse k_imp to avoid extra vector
   f->UpdateDt(gamma*dt);
   f->AdjointImplicitSolve(dt*(gamma), lam, yi, k_imp);
   f->UpdateDt(dt);

   f->SetTime(t);

   //add it all up
   lam.Add(dt*delta, k1_exp);
   lam.Add(dt*(1-delta), k2_exp);
   //x.Add(dt*(1-gamma), k2_imp); it is already added to x above
   lam.Add(dt*gamma, k_imp);
   t += dt;
}



std::unique_ptr<TopOptIMEXSolver> SelectDesignOptIMEX(const int ode_solver_type)
{
   using ode_ptr = std::unique_ptr<TopOptIMEXSolver>;
   switch (ode_solver_type)
   {
      // L-stable IMEX methods for design opt
      case 1: return ode_ptr(new TopOptIMEXExpImplEuler);
      case 2: return ode_ptr(new TopOptIMEXRK2);

      default: MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type );
   }
}



class DesignSolver
{
   private:
   ParFiniteElementSpace state_fes;
   ParFiniteElementSpace filter_fes;
   ParFiniteElementSpace control_fes;
   toopt::PDEFilter &filter;
   Array<int> ess_tdof_list;
   Array<int> ess_bdr_attr;
   Array<int> inflow_bdr;
   TerminalL2Objective &objective;
   ScalarVectorProductCoefficient velocity_coeff;
   VectorFunctionCoefficient v_base;
   ProductCoefficient diffusion_coeff;
   ProductCoefficient dt_diff_coeff;
   FunctionCoefficient inflow;
   int nsteps;
   real_t dt;
   real_t t_final;
   ParGridFunction &rho;         // working density (also the driver's ParaView field)
   ParGridFunction &rho_tilde;   // filtered density
   GridFunctionCoefficient q0;          // initial condition
   ParGridFunction q_gf;
   HypreParVector *q_vec;
   real_t raw_diff_term;

   bool paraview_vis;

   IMEXAdvectionDiffusionSolver *oper;
   std::vector<Vector> states;
   std::vector<real_t> times;
   Vector dJ_drho_tilde;

   int outer_it;

   MPI_Comm comm;
   int imex_integrator;

   public:
   DesignSolver(ParFiniteElementSpace &state_fes_,
                         ParFiniteElementSpace &filter_fes_,
                         ParFiniteElementSpace &control_fes_,
                         toopt::PDEFilter &filter_,
                         Array<int> &ess_bdr_attr_,
                         Array<int> &inflow_bdr_,
                         TerminalL2Objective &objective_,
                         ScalarVectorProductCoefficient &velocity_coeff_,
                         VectorFunctionCoefficient &v_base_,
                         real_t raw_diff_term_,
                         ProductCoefficient &diffusion_coeff_,
                         ProductCoefficient &dt_diff_coeff_,
                         FunctionCoefficient &inflow_,
                         GridFunctionCoefficient &q0_,
                         int nsteps_, real_t dt_, real_t t_final_,
                         ParGridFunction &rho_,
                         ParGridFunction &rho_tilde_,
                         int imex_integrator_, MPI_Comm comm_)
      : state_fes(state_fes_), filter_fes(filter_fes_), control_fes(control_fes_),
        filter(filter_),
        ess_bdr_attr(ess_bdr_attr_), inflow_bdr(inflow_bdr_),
        objective(objective_), velocity_coeff(velocity_coeff_), diffusion_coeff(diffusion_coeff_),
        dt_diff_coeff(dt_diff_coeff_), inflow(inflow_), q0(q0_), raw_diff_term(raw_diff_term_),
        nsteps(nsteps_), dt(dt_), t_final(t_final_),
        rho(rho_), rho_tilde(rho_tilde_), q_gf(&state_fes_), imex_integrator(imex_integrator_), v_base(v_base_), 
        q_vec(nullptr), oper(nullptr), comm(comm_)
   { 
      outer_it = 0;
   }

   ~DesignSolver() 
   { 
      if (oper) delete oper; 
      if (q_vec) delete q_vec;
   }

   int NumSteps() const {return nsteps;}
   real_t Time_Step() const {return dt;}

   // 1. Forward Filter. Raw control density -> filtered density (Helmholtz solve).
   void FilterFSolve(const Vector &rho_tv)
   {
      rho.SetFromTrueDofs(rho_tv);
      filter.Mult(rho, rho_tilde);
      rho_tilde.ExchangeFaceNbrData();
   }

   // 2. Forward physics: (re)assemble the operator for the current rho_tilde_, run
   //    the IMEX Forward Integration, store the trajectory, return J.
   real_t PhysicsFSolve()
   {
      if (oper) { delete oper; oper = nullptr; }
      if (q_vec) { delete q_vec; q_vec = nullptr; }
      std::unique_ptr<TopOptIMEXSolver> ode_solver = SelectDesignOptIMEX(imex_integrator);
      oper = new IMEXAdvectionDiffusionSolver(state_fes, velocity_coeff, v_base, dt_diff_coeff, diffusion_coeff, inflow, inflow_bdr, q0, rho_tilde, dt, t_final, raw_diff_term, comm, ess_bdr_attr);
      // if (Mpi::Root())
      // {
      //    std::cout << "    [it " << outer_it + 1 << "] forward integration (" << nsteps << " steps)\n";
      // }

      q_gf = oper->Getq();
      // q_gf.ExchangeFaceNbrData();
      q_vec = q_gf.GetTrueDofs();
      // ParaViewDataCollection *pd = NULL;
      // if (paraview_vis)
      // {
      //    pd = new ParaViewDataCollection("forward", state_fes.GetParMesh());
      //    pd->SetPrefixPath("ParaView");
      //    pd->RegisterField("solution", &q_gf);
      //    pd->SetLevelsOfDetail(state_fes.GetOrder(0));
      //    pd->SetDataFormat(VTKFormat::BINARY);
      //    pd->SetHighOrderOutput(false);
      //    pd->SetCycle(0);
      //    pd->SetTime(0.0);
      //    pd->Save();
      // }
      real_t t = 0.0;
      times.resize(nsteps);
      ode_solver->Init(*oper);
      oper->SetTime(t);
      bool done = false;
      int myrank;
      MPI_Comm_rank(comm, &myrank);
      // std::cout << "my_rank = " << myrank << "time step initial " << ", time: 0 " << ", ||q|| = " << q_vec->Norml2() << std::endl;
      for (int ti = 0; !done; )
      {
         real_t dt_real = std::min(dt, t_final - t);  
         oper->UpdateDt(dt_real);
         times[ti] = dt_real;
         ode_solver->Step(*q_vec, t, dt_real);
         // std::cout << "my_rank = " << myrank << "time step: " << ti << ", time: " << t << ", ||q|| = " << q_vec->Norml2() << std::endl;
         ti++;
         oper->SetStep(ti);
         oper->StoreTraj(ti, *q_vec);
         done = (t >= t_final - 1e-8*dt); 
         if (done || ti % 10 == 0)
         {
            // if (Mpi::Root())
            // {
            //    std::cout << "time step: " << ti << ", time: " << t << ", dt = " << dt_real << std::endl;  
            // }
            q_gf.SetFromTrueDofs(*q_vec);
            // if (paraview_vis)
            // {
            //    pd->SetCycle(ti);
            //    pd->SetTime(t);
            //    pd->Save();
            // }
         }
      }
      q_gf.SetFromTrueDofs(*q_vec);
      oper->Updateq(q_gf);
      objective.ComputeObjective(q_gf);
      return objective.GetObjective();
   }

   // 3. Adjoint physics: backward discrete-adjoint sweep -> dJ/d(rho_tilde).
   void PhysicsASolve()
   {
      std::unique_ptr<TopOptIMEXSolver> ode_solver = SelectDesignOptIMEX(imex_integrator);
      MFEM_VERIFY(oper, "PhysicsASolve() requires a preceding PhysicsFSolve().");
      const int myid = Mpi::WorldRank();
      //Vector grad_vec(q_vec->Size());
      //oper->ComputeObjectiveGradient(grad_vec);
      // for(int i = 0; i < grad_vec.Size(); i++)
      // {
      //    std::cout << "gv = " << grad_vec(i) << std::endl;
      // }
      // ParGridFunction lam_gf(&state_fes);
      // Vector two_qvec = *q_vec;
      // two_qvec *= 2.0;
      // q_gf.SetFromTrueDofs(two_qvec);

      // GridFunctionCoefficient qq_cf;
      // qq_cf.SetGridFunction(&q_gf);
      ParLinearForm grad_form(&state_fes);
      // grad_form.AddDomainIntegrator(new DomainLFIntegrator(qq_cf));
      // grad_form.Assemble();
      // grad_vec = grad_form;

      // q_gf = *q_vec;
      objective.ComputeObjectiveGradient(q_gf, grad_form);
      HypreParVector* grad_vec = grad_form.ParallelAssemble();
      // std::cout<< "grad u norm2 " << grad_vec->Norml2() << std::endl;
      // grad_vec *= -1.0;
      // HypreParVector lambda(grad_vec);

      // ParBilinearForm M(&state_fes);
      // M.AddDomainIntegrator(new MassIntegrator());
      // M.Assemble();
      // M.Finalize();
      // std::unique_ptr<HypreParMatrix> M_mat(M.ParallelAssemble());

      // HypreSmoother M_prec(*M_mat, HypreSmoother::Jacobi);
      // CGSolver M_solver(comm);
      // M_solver.SetOperator(*M_mat);
      // M_solver.SetPreconditioner(M_prec);
      // M_solver.SetRelTol(1e-12);
      // M_solver.SetMaxIter(200);
      // M_solver.SetPrintLevel(-1);

      // Vector lam_tdof(grad_vec.Size());
      // lam_tdof = 0.0;
      // M_solver.Mult(grad_vec, lam_tdof); // Solve M * lam = grad_vec

      // 3. Set the primal GridFunction from the True-Dofs
      HypreParVector lam_vec = *grad_vec;
      // lam_gf.SetFromTrueDofs(*grad_vec);
      lam_vec *= -1.0;
      // GridFunctionCoefficient q_coef(q_gf);
      // ProductCoefficient two_q_coef(2.0, q_coef);
      // lam_gf.ProjectCoefficient(two_q_coef);
      // Include an MFEM_Verify Symplectic RK IMEX Integrator
      // ParLinearForm grad_form(&state_fes);
      // Vector grad_vec;
      // oper->ComputeObjectiveGradient(grad_vec);
      // Vector lambda(grad_vec.Size());
      // std::cout << "grad vec size " << grad_vec.Size() << std::endl;
      // *lambda *= 2.0;
      // ParGridFunction lam_gf(&state_fes);
      // lam_gf.SetFromTrueDofs(*grad_vec);
      oper->SetStep(nsteps);
      oper->TakeAdjoint();
      ode_solver->Init(*oper);
      // oper->UpdateDt(dt);
      // ParaViewDataCollection *pd_adj = NULL;
      // if (paraview_vis)
      // {
      //    pd_adj = new ParaViewDataCollection("adjoint", state_fes.GetParMesh());
      //    pd_adj->SetPrefixPath("ParaView");
      //    pd_adj->RegisterField("solution", &lam_gf);
      //    pd_adj->SetLevelsOfDetail(state_fes.GetOrder(0));
      //    pd_adj->SetDataFormat(VTKFormat::BINARY);
      //    pd_adj->SetHighOrderOutput(false);
      //    pd_adj->SetCycle(0);
      //    pd_adj->SetTime(t_final);
      //    pd_adj->Save();
      // } 
      real_t t = t_final;
      bool done = false;
      for (int ti = 0; !done;)
      {
         real_t dti = times[nsteps-ti-1]; 
         oper->UpdateDt(dti);
         real_t t_dummy = t;
         // Vector x(lam_vec.Size());
         // x = 0.0;
         oper->GetTraj(oper->GetStep() - 1, *q_vec);
         ode_solver->AdjointStep(lam_vec, t_dummy, dti, *q_vec); 
         ti++;
         oper->SetStep(nsteps-ti);
         t -= dti;
         done = (t <= 1e-8*dt); 
         if (myid == 0)
         {
            // std::cout << "time step: " << ti << ", time: " << t << "------------------" << std::endl; 
            // lam_gf = *lambda;
            // if (paraview_vis)
            // {
            //    pd_adj->SetCycle(ti);
            //    pd_adj->SetTime(t_final-t);
            //    pd_adj->Save();
            // }
         }
      }
      dJ_drho_tilde = oper->GetDesignGrad();
      delete grad_vec;
   } 

   // 4. Adjoint filter: transpose the filter, dJ/d(rho_tilde) -> dJ/d(rho).
   void FilterASolve(Vector &dJ_drho)
   {
      filter.MultTranspose(dJ_drho_tilde, dJ_drho);
      MFEM_VERIFY(dJ_drho.Size() == control_fes.GetTrueVSize(),
                  "Raw design gradient has unexpected size.");
      // dJ_drho = dJ_drho_tilde;

   }

   // Convenience: the four steps in sequence (forward filter + physics, adjoint
   // physics + filter). Returns J and fills dJ_drho.
   real_t ObjectiveAndGradient(const Vector &rho_tv, Vector &dJ_drho,
                               int outer_it = -1)
   {
      // FilterFSolve(rho_tv);
      // const real_t J = PhysicsFSolve();
      // PhysicsASolve();
      // FilterASolve(dJ_drho);
      const real_t J = 0.0;
      std::cout << "Not implemented " << std::endl;
      return J;
   }

   // Forward-only objective J(rho) (no gradient / no stored trajectory).
   real_t Objective(const Vector &rho_tv)
   {
      return 0.0;
      // return EvaluateDesignObjective(
      //           rho_tv, x0_, state_fes_, control_fes_, rho_, rho_tilde_, filter_,
      //           gamma_coef_, exterior_bdr_attr_, ess_bdr_attr_, objective_, mat_,
      //           load_spec_, load_coef_, impedance_, nsteps_, h_, mass_type_);
   }

};

}
#endif 