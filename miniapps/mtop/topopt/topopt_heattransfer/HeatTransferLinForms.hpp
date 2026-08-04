#ifndef LIN_FORMS_HPP
#define LIN_FORMS_HPP

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include "HeatTransferTopOpt.hpp"
namespace mfem
{
class DGStiffnessDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &rho_tilde;
   ParGridFunction &u;
   ParGridFunction &z;
   real_t diff_term;
   SIMPCoefficient SIMP_cf;
   
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
                                       ParGridFunction &z_, real_t &diff_term_, real_t kappa_, SIMPCoefficient SIMP_cf_)
      : rho_tilde(rho_tilde_), u(u_), diff_term(diff_term_), z(z_), kappa(kappa_), SIMP_cf(SIMP_cf_){}

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

         const real_t rp = SIMP_cf.Eval_Derivative(T, ip);
         
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

         const real_t rp1 = diff_term*SIMP_cf.Eval_Derivative(*Tr.Elem1, ip);
         const real_t rp2 = diff_term*SIMP_cf.Eval_Derivative(*Tr.Elem2, ip);

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
   SIMPCoefficient SIMP_cf;

   // Pre-allocated data for the Face Integrators
   Vector shape1, shape2;

public:
   DGAdvectionDesignLFIntegrator(ParGridFunction &rho_tilde_,
                                 const ParGridFunction &u_,
                                 const ParGridFunction &z_,
                                 VectorCoefficient &v_base_, SIMPCoefficient SIMP_cf_,
                                 real_t alpha_ = -1.0)
      : rho_tilde(rho_tilde_), u(u_), z(z_), v_base(v_base_), SIMP_cf(SIMP_cf_), alpha(alpha_) {}

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
         const real_t rp = SIMP_cf.Eval_Derivative(T, ip);
         
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
         const real_t rp1 = SIMP_cf.Eval_Derivative(*Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t rp2 = SIMP_cf.Eval_Derivative(*Tr.Elem2, Tr.Elem2->GetIntPoint());

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
         const real_t face_weight = nor.Norml2(); 
         real_t vn = 0.0;
         vn = v_val*nor;
         const real_t d_integrand = -0.5 * alpha * vn * (u1) * z1 + beta*fabs(vn) * (u1) * z1; 
         const real_t rp = SIMP_cf.Eval_Derivative(*Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t weight = rp * ip.weight * d_integrand;
         for (int i = 0; i < dof; i++)
         {
             elvect(i) += weight * shape(i);
         }
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
   SIMPCoefficient SIMP_cf;

public:
   BdrFlowDesignLFIntegrator(ParGridFunction &rho_tilde_,
                             ParGridFunction &z_,
                             Coefficient &inflow_,
                             VectorCoefficient &v_base_, SIMPCoefficient SIMP_cf_,
                             real_t alpha_ = -1.0)
      : rho_tilde(rho_tilde_), z(z_), inflow(inflow_), 
        v_base(v_base_), alpha(alpha_), SIMP_cf(SIMP_cf_){}

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

         // Compute normal velocity component (v_base \cdot n)
         real_t vn = 0.0;
         for (int i = 0; i < dim; i++)
         {
             vn += v_val(i) * nor(i);
         }

         // The original integrand was: alpha * (v_n) * inflow * test_function
         const real_t b_integrand = 0.5 * alpha * vn * u_in * z_val + beta * alpha * fabs(vn) * u_in * z_val;

         // Derivative of SIMP scaling function w.r.t rho
         const real_t rp = SIMP_cf.Eval_Derivative(*Tr.Elem1, Tr.Elem1->GetIntPoint());
         const real_t weight = rp * ip.weight * b_integrand;

         for (int i = 0; i < dof; i++)
         {
             elvect(i) += weight * shape(i);
         }
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
};

class DomainDesignLFIntegrator : public LinearFormIntegrator
{
private:
   ParGridFunction &z;           
   Coefficient &inflow;           

   Vector shape;
   Vector v_val;

public:
   DomainDesignLFIntegrator(ParGridFunction &z_,
                             Coefficient &inflow_)
      : z(z_), inflow(inflow_) {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      elvect.SetSize(dof);
      shape.SetSize(dof);
      elvect = 0.0;
      const IntegrationRule *ir = GetIntegrationRule(el, Tr);

      ir = &IntRules.Get(el.GetGeomType(), 2*el.GetOrder());

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Tr.SetIntPoint(&ip);
         const real_t z_val = z.GetValue(Tr, Tr.GetIntPoint());
         const real_t u_in = inflow.Eval(Tr, ip);
         real_t val = Tr.Weight() * inflow.Eval(Tr, ip)*z_val;

         el.CalcPhysShape(Tr, shape);

         add(elvect, ip.weight * val, shape, elvect);
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
};
}
#endif 