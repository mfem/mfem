#include "hyperbolic_conservation_laws.hpp"

namespace mfem
{
//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace *vfes_, HyperbolicFormIntegrator &formIntegrator_,
   HyperbolicFormIntegrator &faceFormIntegrator_,
   const int num_equations_)
   : TimeDependentOperator(vfes_->GetNDofs() * num_equations_),
     dim(vfes_->GetFE(0)->GetDim()),
     num_equations(num_equations_),
     vfes(vfes_),
     formIntegrator(formIntegrator_),
     faceFormIntegrator(faceFormIntegrator_),
     Me_inv(0),
     z(vfes_->GetNDofs() * num_equations_)
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();
#ifndef MFEM_USE_MPI
   nonlinearForm = new NonlinearForm(vfes);
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(vfes);
   if (pvfes)
   {
      nonlinearForm = new ParNonlinearForm(pvfes);
   }
   else
   {
      nonlinearForm = new NonlinearForm(vfes);
   }
#endif
   formIntegrator.resetMaxCharSpeed();
   faceFormIntegrator.resetMaxCharSpeed();

   nonlinearForm->AddDomainIntegrator(&formIntegrator);
   nonlinearForm->AddInteriorFaceIntegrator(&faceFormIntegrator);

   height = z.Size();
   width = z.Size();
}

void DGHyperbolicConservationLaws::ComputeInvMass()
{
   DenseMatrix Me;     // auxiliary local mass matrix
   MassIntegrator mi;  // mass integrator
   // resize it to the current number of elements
   Me_inv.resize(vfes->GetNE());
   for (int i = 0; i < vfes->GetNE(); i++)
   {
      Me.SetSize(vfes->GetFE(i)->GetDof());
      mi.AssembleElementMatrix(*vfes->GetFE(i),
                               *vfes->GetElementTransformation(i), Me);
      DenseMatrixInverse inv(&Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv[i]);
   }
}

void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator.resetMaxCharSpeed();
   faceFormIntegrator.resetMaxCharSpeed();
   // 1. Create the vector z with the face terms (F(u), grad v) - <F.n(u), [w]>.
   nonlinearForm->Mult(x, z);
   max_char_speed = std::max(formIntegrator.getMaxCharSpeed(),
                             faceFormIntegrator.getMaxCharSpeed());

   // 2. Multiply element-wise by the inverse mass matrices.
   Vector zval;             // local dual vector storage
   Array<int> vdofs;        // local degrees of freedom storage
   DenseMatrix zmat, ymat;  // local dual vector storage

   for (int i = 0; i < vfes->GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes->GetElementVDofs(i, vdofs);
      // get local dual vector
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), vfes->GetFE(i)->GetDof(),
                           num_equations);
      ymat.SetSize(Me_inv[i].Height(), num_equations);
      // mass matrix inversion and pass it to global vector
      mfem::Mult(Me_inv[i], zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

void HyperbolicFormIntegrator::AssembleElementVector(const FiniteElement &el,
                                                     ElementTransformation &Tr,
                                                     const Vector &elfun,
                                                     Vector &elvect)
{
   // current element's the number of degrees of freedom
   // does not consider the number of equations
   const int dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   // Local storages for element integration

   Vector shape(dof); // shape function value at an integration point
   DenseMatrix dshape(dof,
                      el.GetDim()); // derivative of shape function at an integration point
   Vector state(num_equations); // state value at an integration point
   DenseMatrix flux(num_equations,
                    el.GetDim()); // flux value at an integration point
#else
   // resize shape and gradient shape storage
   shape.SetSize(dof);
   dshape.SetSize(dof, el.GetDim());
#endif

   // setDegree-up output vector
   elvect.SetSize(dof * num_equations);
   elvect = 0.0;

   // make state variable and output dual vector matrix form.
   const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
   DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

   // obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el);
   // loop over interation points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcPhysDShape(Tr, dshape);
      // compute current state value with given shape function values
      elfun_mat.MultTranspose(shape, state);
      // compute F(u,x) and point maximum characteristic speed

      const double mcs = ComputeFlux(state, Tr, flux);
      // update maximum characteristic speed
      max_char_speed = std::max(mcs, max_char_speed);
      // integrate (F(u,x), grad v)
      AddMult_a_ABt(ip.weight * Tr.Weight(), dshape, flux, elvect_mat);
   }
}

void HyperbolicFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   // current elements' the number of degrees of freedom
   // does not consider the number of equations
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
   // Local storages for element integration

   Vector shape1(
      dof1);  // shape function value at an integration point - first elem
   Vector shape2(
      dof2);  // shape function value at an integration point - second elem
   Vector nor(el1.GetDim());     // normal vector (usually not a unit vector)
   Vector state1(
      num_equations);  // state value at an integration point - first elem
   Vector state2(
      num_equations);  // state value at an integration point - second elem
   Vector fluxN1(
      num_equations);  // flux dot n value at an integration point - first elem
   Vector fluxN2(
      num_equations);  // flux dot n value at an integration point - second elem
   Vector fluxN(num_equations);   // hat(F)(u,x)
#else
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
#endif

   elvect.SetSize((dof1 + dof2) * num_equations);
   elvect = 0.0;

   const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
   const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
                                num_equations);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2,
                           num_equations);

   // obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el1, el2);
   // loop over integration points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip);  // setDegree face and element int. points

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, state1);
      elfun2_mat.MultTranspose(shape2, state2);

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         // This assume the 1D integration point is in (0,1). This may not work if
         // this chages.
         nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      // Compute F(u+, x) and F(u-, x) with maximum characteristic speed
      const double mcs = std::max(
                            ComputeFluxDotN(state1, nor, Tr.GetElement1Transformation(), fluxN1),
                            ComputeFluxDotN(state2, nor, Tr.GetElement2Transformation(), fluxN2));
      // Compute hat(F) using evaluated quantities
      rsolver->Eval(state1, state2, fluxN1, fluxN2, mcs, nor, fluxN);

      // Update the global max char speed
      max_char_speed = std::max(mcs, max_char_speed);

      // pre-multiply integration weight to flux
      fluxN *= ip.weight;
      for (int k = 0; k < num_equations; k++)
      {
         // this loop structure can increase cache hit because
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN(k) * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN(k) * shape2(s);
         }
      }
   }
}


class AdvectionFormIntegrator : public HyperbolicFormIntegrator
{
private:
   VectorCoefficient &b;  // velocity coefficient
   Vector bval;           // velocity value storage

public:
   /**
    * @brief Compute F(u)
    *
    * @param U U (u) at current integration point
    * @param Tr current element transformation with integration point
    * @param FU F(u) = ubᵀ
    * @return double maximum characteristic speed, |b|
    */
   double ComputeFlux(const Vector &U, ElementTransformation &Tr,
                      DenseMatrix &FU)
   {
      b.Eval(bval, Tr, Tr.GetIntPoint());
      MultVWt(U, bval, FU);
      return bval.Norml2();
   }
   /**
    * @brief Compute normal flux, F(u)n
    *
    * @param U U (u) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param FUdotN F(u)n = u(b⋅n)
    * @return double maximum characteristic speed, |b|
    */
   double ComputeFluxDotN(const Vector &U, const Vector &normal,
                          ElementTransformation &Tr, Vector &FUdotN)
   {
      b.Eval(bval, Tr, Tr.GetIntPoint());
      const double bN = bval * normal;
      FUdotN = U;
      FUdotN *= bN;
      return bval.Norml2();
   }

   /**
    * @brief Construct a new Advection Element Form Integrator object with given
    * integral order offset
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param b_ velocity coefficient, possibly depends on space
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   AdvectionFormIntegrator(RiemannSolver *rsolver_, const int dim,
                           VectorCoefficient &b_,
                           const int IntOrderOffset_ = 3)
      : HyperbolicFormIntegrator(rsolver_, dim, 1, IntOrderOffset_), b(b_),
        bval(dim) {}
   /**
    * @brief Construct a new Advection Element Form Integrator object with given
    * integral rule
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param b_ velocity coefficient, possibly depends on space
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   AdvectionFormIntegrator(RiemannSolver *rsolver_, const int dim,
                           VectorCoefficient &b_,
                           const IntegrationRule *ir)
      : HyperbolicFormIntegrator(rsolver_, dim, 1, ir), b(b_), bval(dim) {}
};

DGHyperbolicConservationLaws getAdvectionEquation(FiniteElementSpace *vfes,
                                                  RiemannSolver *numericalFlux,
                                                  VectorCoefficient &b,
                                                  const int IntOrderOffset)
{
   const int dim = vfes->GetMesh()->Dimension();
   const int num_equations = 1;

   AdvectionFormIntegrator *enfi =
      new AdvectionFormIntegrator(numericalFlux, dim, b, IntOrderOffset);
   AdvectionFormIntegrator *fnfi =
      new AdvectionFormIntegrator(numericalFlux, dim, b, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *enfi, *fnfi, num_equations);
}
class BurgersFormIntegrator : public HyperbolicFormIntegrator
{
public:
   /**
    * @brief Compute F(u)
    *
    * @param U U (u) at current integration point
    * @param Tr current element transformation with integration point
    * @param FU F(u) = ½u²*1ᵀ where 1 is (dim x 1) vector
    * @return double maximum characteristic speed, |u|
    */
   double ComputeFlux(const Vector &U, ElementTransformation &Tr,
                      DenseMatrix &FU)
   {
      FU = U * U * 0.5;
      return abs(U(0));
   }
   /**
    * @brief Compute normal flux, F(u)n
    *
    * @param U U (u) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param FUdotN F(u)n = ½u² 1⋅n
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &U, const Vector &normal,
                          ElementTransformation &Tr, Vector &FUdotN)
   {
      FUdotN = normal.Sum() * (U * U) * 0.5;
      return abs(U(0));
   }

   /**
    * @brief Construct a new Burgers Element Form Integrator object with given
    * integral order offset
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   BurgersFormIntegrator(RiemannSolver *rsolver_, const int dim,
                         const int IntOrderOffset_ = 3)
      : HyperbolicFormIntegrator(rsolver_, dim, 1, IntOrderOffset_) {}
   /**
    * @brief Construct a new Burgers Element Form Integrator object with given
    * integral rule
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   BurgersFormIntegrator(RiemannSolver *rsolver_, const int dim,
                         const IntegrationRule *ir)
      : HyperbolicFormIntegrator(rsolver_, dim, 1, ir) {}
};


DGHyperbolicConservationLaws getBurgersEquation(FiniteElementSpace *vfes,
                                                RiemannSolver *numericalFlux,
                                                const int IntOrderOffset)
{
   const int dim = vfes->GetMesh()->Dimension();
   const int num_equations = 1;

   BurgersFormIntegrator *enfi =
      new BurgersFormIntegrator(numericalFlux, dim, IntOrderOffset);
   BurgersFormIntegrator *fnfi =
      new BurgersFormIntegrator(numericalFlux, dim, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *enfi, *fnfi, num_equations);
}

class ShallowWaterFormIntegrator : public HyperbolicFormIntegrator
{
private:
   const double g;  // gravity constant

public:
   /**
    * @brief Compute F(h, hu)
    *
    * @param U U (h, hu) at current integration point
    * @param Tr current element transformation with integration point
    * @param FU F(h, hu) = [huᵀ; huuᵀ + ½gh²I]
    * @return double maximum characteristic speed, |u| + √(gh)
    */
   double ComputeFlux(const Vector &U, ElementTransformation &Tr,
                      DenseMatrix &FU)
   {
      const int dim = U.Size() - 1;
      const double height = U(0);
      const Vector h_vel(U.GetData() + 1, dim);

      const double energy = 0.5 * g * (height * height);

      MFEM_ASSERT(height >= 0, "Negative Height");

      for (int d = 0; d < dim; d++)
      {
         FU(0, d) = h_vel(d);
         for (int i = 0; i < dim; i++)
         {
            FU(1 + i, d) = h_vel(i) * h_vel(d) / height;
         }
         FU(1 + d, d) += energy;
      }

      const double sound = sqrt(g * height);
      const double vel = sqrt(h_vel * h_vel) / height;

      return vel + sound;
   }
   /**
    * @brief Compute normal flux, F(h, hu)
    *
    * @param U U (h, hu) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param FUdotN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &U, const Vector &normal,
                          ElementTransformation &Tr, Vector &FUdotN)
   {
      const int dim = normal.Size();
      const double height = U(0);
      const Vector h_vel(U.GetData() + 1, dim);

      const double energy = 0.5 * g * (height * height);

      MFEM_ASSERT(height >= 0, "Negative Height");
      FUdotN(0) = h_vel * normal;
      const double normal_vel = FUdotN(0) / height;
      for (int i = 0; i < dim; i++)
      {
         FUdotN(1 + i) = normal_vel * h_vel(i) + energy * normal(i);
      }

      const double sound = sqrt(g * height);
      const double vel = sqrt(h_vel * h_vel) / height;

      return vel + sound;
   }

   /**
    * @brief Construct a new Shallow Water Element Form Integrator object with
    * given integral order offset
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param g_ gravity constant
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   ShallowWaterFormIntegrator(RiemannSolver *rsolver_, const int dim,
                              const double g_,
                              const int IntOrderOffset_ = 3)
      : HyperbolicFormIntegrator(rsolver_, dim, dim + 1, IntOrderOffset_), g(g_) {}
   /**
    * @brief Construct a new Shallow Water Element Form Integrator object with
    * given integral rule
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param g_ gravity constant
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   ShallowWaterFormIntegrator(RiemannSolver *rsolver_, const int dim,
                              const double g_,
                              const IntegrationRule *ir)
      : HyperbolicFormIntegrator(rsolver_, dim, dim + 1, ir), g(g_) {}
};

DGHyperbolicConservationLaws getShallowWaterEquation(
   FiniteElementSpace *vfes, RiemannSolver *numericalFlux, const double g,
   const int IntOrderOffset)
{
   const int dim = vfes->GetMesh()->Dimension();
   const int num_equations = dim + 1;

   ShallowWaterFormIntegrator *enfi =
      new ShallowWaterFormIntegrator(numericalFlux, dim, g, IntOrderOffset);
   ShallowWaterFormIntegrator *fnfi =
      new ShallowWaterFormIntegrator(numericalFlux, dim, g, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *enfi, *fnfi, num_equations);
}
class EulerFormIntegrator : public HyperbolicFormIntegrator
{
private:
   const double specific_heat_ratio;  // specific heat ratio, γ
   // const double gas_constant;         // gas constant

   /**
    * @brief Compute F(ρ, ρu, E)
    *
    * @param U U (ρ, ρu, E) at current integration point
    * @param Tr current element transformation with integration point
    * @param FU F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
public:
   double ComputeFlux(const Vector &U, ElementTransformation &Tr,
                      DenseMatrix &FU)
   {
      const int dim = U.Size() - 2;

      // 1. Get states
      const double density = U(0);                  // ρ
      const Vector momentum(U.GetData() + 1, dim);  // ρu
      const double energy = U(1 + dim);             // E, internal energy ρe
      // pressure, p = (γ-1)*(ρu - ½ρ|u|²)
      const double pressure = (specific_heat_ratio - 1.0) *
                              (energy - 0.5 * (momentum * momentum) / density);

      // Check whether the solution is physical only in debug mode
      MFEM_ASSERT(density >= 0, "Negative Density");
      MFEM_ASSERT(pressure >= 0, "Negative Pressure");
      MFEM_ASSERT(energy >= 0, "Negative Energy");

      // 2. Compute Flux
      for (int d = 0; d < dim; d++)
      {
         FU(0, d) = momentum(d);  // ρu
         for (int i = 0; i < dim; i++)
         {
            // ρuuᵀ
            FU(1 + i, d) = momentum(i) * momentum(d) / density;
         }
         // (ρuuᵀ) + p
         FU(1 + d, d) += pressure;
      }
      // enthalpy H = e + p/ρ = (E + p)/ρ
      const double H = (energy + pressure) / density;
      for (int d = 0; d < dim; d++)
      {
         // u(E+p) = ρu*(E + p)/ρ = ρu*H
         FU(1 + dim, d) = momentum(d) * H;
      }

      // 3. Compute maximum characteristic speed

      // sound speed, √(γ p / ρ)
      const double sound = sqrt(specific_heat_ratio * pressure / density);
      // fluid speed |u|
      const double speed = sqrt(momentum * momentum) / density;
      // max characteristic speed = fluid speed + sound speed
      return speed + sound;
   }
   /**
    * @brief Compute normal flux, F(ρ, ρu, E)n
    *
    * @param x x (ρ, ρu, E) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param FUdotN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &x, const Vector &normal,
                          ElementTransformation &Tr, Vector &FUdotN)
   {
      const int dim = normal.Size();

      // 1. Get states
      const double density = x(0);                  // ρ
      const Vector momentum(x.GetData() + 1, dim);  // ρu
      const double energy = x(1 + dim);             // E, internal energy ρe
      // pressure, p = (γ-1)*(E - ½ρ|u|^2)
      const double pressure = (specific_heat_ratio - 1.0) *
                              (energy - 0.5 * (momentum * momentum) / density);

      // Check whether the solution is physical only in debug mode
      MFEM_ASSERT(density >= 0, "Negative Density");
      MFEM_ASSERT(pressure >= 0, "Negative Pressure");
      MFEM_ASSERT(energy >= 0, "Negative Energy");

      // 2. Compute normal flux

      FUdotN(0) = momentum * normal;  // ρu⋅n
      // u⋅n
      const double normal_velocity = FUdotN(0) / density;
      for (int d = 0; d < dim; d++)
      {
         // (ρuuᵀ + pI)n = ρu*(u⋅n) + pn
         FUdotN(1 + d) = normal_velocity * momentum(d) + pressure * normal(d);
      }
      // (u⋅n)(E + p)
      FUdotN(1 + dim) = normal_velocity * (energy + pressure);

      // 3. Compute maximum characteristic speed

      // sound speed, √(γ p / ρ)
      const double sound = sqrt(specific_heat_ratio * pressure / density);
      // fluid speed |u|
      const double speed = sqrt(momentum * momentum) / density;
      // max characteristic speed = fluid speed + sound speed
      return speed + sound;
   }

   /**
    * @brief Construct a new Euler Element Form Integrator object with given
    * integral order offset
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param specific_heat_ratio_ specific heat ratio, γ
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   EulerFormIntegrator(RiemannSolver *rsolver_, const int dim,
                       const double specific_heat_ratio_,
                       const int IntOrderOffset_)
      : HyperbolicFormIntegrator(rsolver_, dim, dim + 2, IntOrderOffset_),
        specific_heat_ratio(specific_heat_ratio_) {}

   /**
    * @brief Construct a new Euler Element Form Integrator object with given
    * integral rule
    *
    * @param[in] rsolver_ numerical flux
    * @param dim spatial dimension
    * @param specific_heat_ratio_ specific heat ratio, γ
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   EulerFormIntegrator(RiemannSolver *rsolver_, const int dim,
                       const double specific_heat_ratio_,
                       const IntegrationRule *ir)
      : HyperbolicFormIntegrator(rsolver_, dim, dim + 2, ir),
        specific_heat_ratio(specific_heat_ratio_) {}
};

DGHyperbolicConservationLaws getEulerSystem(FiniteElementSpace *vfes,
                                            RiemannSolver *numericalFlux,
                                            const double specific_heat_ratio,
                                            const int IntOrderOffset)
{
   const int dim = vfes->GetMesh()->Dimension();
   const int num_equations = dim + 2;

   EulerFormIntegrator *enfi = new EulerFormIntegrator(
      numericalFlux, dim, specific_heat_ratio, IntOrderOffset);

   EulerFormIntegrator *fnfi = new EulerFormIntegrator(
      numericalFlux, dim, specific_heat_ratio, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *enfi, *fnfi, num_equations);
}

}