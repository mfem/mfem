//           MFEM Hyperbolic Conservation Laws Serial/Parallel Shared Class
//
// Description:  This file contains general hyperbolic conservation element/face
//               form integrators.
//
//               Abstract RiemannSolver, HyperbolicFormIntegrator, and
//               DGHyperbolicConservationLaws are defined. Also, several example
//               hyperbolic equations are defined; including advection, burgers,
//               shallow water, and euler equations.
//
//               To implement a specific hyperbolic conservation laws, users can
//               create derived classes from HyperbolicFormIntegrator with overloaded
//               ComputeFlux. One can optionally overload ComputeFluxDotN to avoid
//               creating dense matrix when computing normal flux.
//
//               FormIntegrator use high-order quadrature points to implement
//               the given form to handle nonlinear flux function. During the
//               implementation of forms, it updates maximum characteristic
//               speed and collected by DGHyperbolicConservationLaws. The global
//               maximum characteristic speed can be obtained by public method
//               getMaxCharSpeed so that the time step can be determined by CFL
//               condition. Also, resetMaxCharSpeed should be called after each Mult.
//
//               @note For parallel version, users should reduce all maximum
//               characteristic speed from all processes using MPI_Allreduce.
//
// @todo: Implement limiter @dohyun-cse
// @todo: Implement boundary condition @dohyun-cse
//
//
// Class structure: DGHyperbolicConservationLaws
//                  |- HyperbolicFormIntegrator: (F(u,x), grad v) and <F̂(u, x, n), [[v]])
//                  |  |- RiemannSolver: (F±, u±) ↦ F̂(u,x,n)
//                  |
//                  |- (Par)NonlinearForm: Evaluate form integrators
//

#ifndef MFEM_HYPERBOLIC_CONSERVATION_LAWS
#define MFEM_HYPERBOLIC_CONSERVATION_LAWS
#include "../config/config.hpp"
#include "fespace.hpp"
#include "bilininteg.hpp"
#include "nonlinearform.hpp"
#include "estimators.hpp"
#ifdef MFEM_USE_MPI
#include "pnonlinearform.hpp"
#endif
// using namespace std;
// using namespace mfem;
namespace mfem
{
/**
 * @brief Average two states with sqrt of density as a weight
 *
 * @param state1 first state
 * @param state2 second state
 * @param avgs averaged state, (state1*sqrt(den1) + state2*sqrt(den2)) / (sqrt(den1) + sqrt(den2))
 * @param densityIndex density index of the given state
 *
 * @note avgs[densityIndex] = sqrt(den1*den2)
 */
void densityWeightedAverage(const Vector &state1, const Vector &state2,
                            Vector &avgs, const int densityIndex=0)
{
   const double den1 = state1[densityIndex];
   const double den2 = state2[densityIndex];
   const double sqrtden1 = sqrt(den1);
   const double sqrtden2 = sqrt(den2);
   avgs = state1;
   avgs *= sqrtden1;
   avgs.Add(sqrtden2, state2);
   avgs /= sqrtden1*sqrtden2;
   avgs[densityIndex] = sqrtden1*sqrtden2;
}

enum PRefineType
{
   elevation,  // order <- order + value
   setDegree,  // order <- value
};

/**
 * @brief Abstract class for numerical flux for an hyperbolic conservation laws
 * on a face with states, fluxes and characteristic speed
 *
 */
class RiemannSolver
{
public:
   RiemannSolver() {}
   /**
    * @brief Evaluates numerical flux for given states and fluxes. Must be
    * overloaded in a derived class
    *
    * @param[in] state1 state value at a point from the first element
    * (num_equations x 1)
    * @param[in] state2 state value at a point from the second element
    * (num_equations x 1)
    * @param[in] fluxN1 normal flux value at a point from the first element
    * (num_equations x dim)
    * @param[in] fluxN2 normal flux value at a point from the second element
    * (num_equations x dim)
    * @param[in] maxE maximum characteristic speed (eigenvalue of flux jacobian)
    * @param[in] nor normal vector (not a unit vector) (dim x 1)
    * @param[out] flux numerical flux (num_equations)
    */
   virtual void Eval(const Vector &state1, const Vector &state2,
                     const Vector &fluxN1, const Vector &fluxN2,
                     const double maxE, const Vector &nor, Vector &flux) = 0;
   virtual ~RiemannSolver() = default;
};

/**
 * @brief Abstract hyperbolic form integrator, (F(u, x), ∇v) and (F̂(u±, x, n))
 *
 */
class HyperbolicFormIntegrator : public NonlinearFormIntegrator
{
private:
   const int num_equations;  // the number of equations
   // The maximum characterstic speed, updated during element/face vector assembly
   double max_char_speed;
   const int IntOrderOffset;  // 2*p + IntOrderOffset will be used for quadrature
   RiemannSolver *rsolver;    // Numerical flux that maps F(u±,x) to hat(F)
#ifndef MFEM_THREAD_SAFE
   // Local storages for element integration
   Vector shape;              // shape function value at an integration point
   Vector state;              // state value at an integration point
   DenseMatrix flux;          // flux value at an integration point
   DenseMatrix dshape;  // derivative of shape function at an integration point

   Vector shape1;  // shape function value at an integration point - first elem
   Vector shape2;  // shape function value at an integration point - second elem
   Vector state1;  // state value at an integration point - first elem
   Vector state2;  // state value at an integration point - second elem
   Vector fluxN1;  // flux dot n value at an integration point - first elem
   Vector fluxN2;  // flux dot n value at an integration point - second elem
   Vector nor;     // normal vector (usually not a unit vector)
   Vector fluxN;   // hat(F)(u,x)
#endif

public:
   /**
    * @brief Compute flux F(u, x) for given state u and physical point x.
    * @a flux should be provided with correct size
    *
    * @param[in] state value of state at the current integration point
    * @param[in] Tr Transformation to find physical location x of the current
    * integration point
    * @param[out] flux F(u, x)
    * @return double maximum characteristic speed
    *
    * @note One can put assertion in here to detect non-physical solution
    */
   virtual double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                              DenseMatrix &flux) = 0;
   /**
    * @brief Compute flux ∂F(u, x) / ∂u for given state u and physical point x.
    * @a flux: vdim x vdim x sdim and @a eigs: vdim x sdim should be provided with correct size
    *
    * @param state
    * @param Tr
    * @param Jacobian
    * @param eigs
    */
   virtual void ComputeFluxJacobian(const Vector &state,
                                    ElementTransformation &Tr, DenseTensor &Jacobian, DenseMatrix &eigs)
   {
      mfem_error("Derived class does not implement flux jacobian.");
   };
   /**
    * @brief Compute normal flux. Optionally overloadded in the
    * derived class to avoid creating full dense matrix for flux.
    *
    * @param[in] state state value at the current integration point
    * @param[in] nor normal vector (usually not a unit vector)
    * @param[in] Tr element transformation
    * @param[out] fluxN normal flux from the given element at the current
    * integration point
    * @return double maximum characteristic velocity
    */
   virtual double ComputeFluxDotN(const Vector &state, const Vector &nor,
                                  ElementTransformation &Tr, Vector &fluxN)
   {
      double val = ComputeFlux(state, Tr, flux);
      flux.Mult(nor, fluxN);
      return val;
   }

   /**
    * @brief Construct a new Hyperbolic Form Integrator object
    *
    * @param[in] rsolver_ numerical flux
    * @param[in] dim physical dimension
    * @param[in] num_equations_ the number of equations
    * @param[in] IntOrderOffset_ 2*p+IntOrderOffset order Gaussian quadrature
    * will be used
    */
   HyperbolicFormIntegrator(RiemannSolver *rsolver_, const int dim,
                            const int num_equations_,
                            const int IntOrderOffset_ = 3)
      : NonlinearFormIntegrator(),
        num_equations(num_equations_),
        IntOrderOffset(IntOrderOffset_),
        rsolver(rsolver_)
   {
#ifndef MFEM_THREAD_SAFE
      state.SetSize(num_equations);
      flux.SetSize(num_equations, dim);
      state1.SetSize(num_equations);
      state2.SetSize(num_equations);
      fluxN1.SetSize(num_equations);
      fluxN2.SetSize(num_equations);
      fluxN.SetSize(num_equations);
      nor.SetSize(dim);
#endif
   }
   /**
    * @brief Construct an object with a fixed integration rule
    *
    * @param[in] rsolver_ numerical flux
    * @param[in] dim physical dimension
    * @param[in] num_equations_ the number of equations
    * @param[in] ir integration rule to be used
    */
   HyperbolicFormIntegrator(RiemannSolver *rsolver_, const int dim,
                            const int num_equations_,
                            const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir),
        num_equations(num_equations_),
        IntOrderOffset(0),
        rsolver(rsolver_)
   {
#ifndef MFEM_THREAD_SAFE
      state.SetSize(num_equations);
      flux.SetSize(num_equations, dim);
      state1.SetSize(num_equations);
      state2.SetSize(num_equations);
      fluxN1.SetSize(num_equations);
      fluxN2.SetSize(num_equations);
      fluxN.SetSize(num_equations);
      nor.SetSize(dim);
#endif
   }

   /**
    * @brief Get the element integration rule based on IntOrderOffset, @see
    * AssembleElementVector. Used only when ir is not provided
    *
    * @param[in] el given finite element space
    * @return const IntegrationRule& with order 2*p + IntOrderOffset
    */
   const IntegrationRule &GetRule(const FiniteElement &el)
   {
      int order;
      order = 2 * el.GetOrder() + IntOrderOffset;
      return IntRules.Get(el.GetGeomType(), order);
   }

   /**
    * @brief Get the face integration rule based on IntOrderOffset, @see
    * AssembleFaceVector. Used only when ir is not provided
    *
    * @param[in] trial_fe trial finite element space
    * @param[in] test_fe test finite element space
    * @return const IntegrationRule& with order 2*p + IntOrderOffset
    */
   const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                  const FiniteElement &test_fe)
   {
      int order;
      order = std::max(trial_fe.GetOrder(), test_fe.GetOrder()) * 2 + IntOrderOffset;
      return IntRules.Get(trial_fe.GetGeomType(), order);
   }

   /**
    * @brief Reset the Max Char Speed 0
    *
    */
   inline void resetMaxCharSpeed()
   {
      max_char_speed = 0.0;
   }

   inline double getMaxCharSpeed()
   {
      return max_char_speed;
   }

   /**
    * @brief Compute element flux F(u,x) to be used in ZZ error estimator
    */
   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans, Vector &u,
                                   const FiniteElement &fluxelem, Vector &flux,
                                   bool with_coef = true,
                                   const IntegrationRule *ir = NULL)
   {
      mfem_error("NOT IMPLEMENTED");
   }

   /**
    * @brief implement (F(u), grad v) with abstract F computed by ComputeFlux
    *
    * @param[in] el local finite element
    * @param[in] Tr element transformation
    * @param[in] elfun local coefficient of basis
    * @param[out] elvect evaluated dual vector
    */
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

   /**
    * @brief implement <-hat(F)(u,x) n, [[v]]> with abstract hat(F) computed by
    * ComputeFluxDotN and numerical flux object
    *
    * @param[in] el1 finite element of the first element
    * @param[in] el2 finite element of the second element
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis from both elements
    * @param[out] elvect evaluated dual vector <-hat(F)(u,x) n, [[v]]>
    */
   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);

   virtual ~HyperbolicFormIntegrator() {}
};


// Base Hyperbolic conservation law class.
// This contains all methods needed except the flux function.
class DGHyperbolicConservationLaws : public TimeDependentOperator
{
private:
   const int dim;
   const int num_equations;
   // Vector finite element space containing conserved variables
   FiniteElementSpace *vfes;
   // Element integration form. Should contain ComputeFlux
   HyperbolicFormIntegrator &formIntegrator;
   // Base Nonlinear Form
   NonlinearForm *nonlinearForm;
   // element-wise inverse mass matrix
   std::vector<DenseMatrix> Me_inv;
   // global maximum characteristic speed. Updated by form integrators
   mutable double max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();

public:
   /**
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ (F(u,x), grad v) and (F̂(u±, x, n), [[v]])
    * @param num_equations_ the number of equations
    */
   DGHyperbolicConservationLaws(
      FiniteElementSpace *vfes_,
      HyperbolicFormIntegrator &formIntegrator_,
      const int num_equations_);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   virtual void Mult(const Vector &x, Vector &y) const;
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   inline double getMaxCharSpeed()
   {
      return max_char_speed;
   }

   void Update()
   {
      nonlinearForm->Update();
      height = nonlinearForm->Height();
      width = height;
      z.SetSize(height);

      ComputeInvMass();
   }

   virtual ~DGHyperbolicConservationLaws() {}
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace *vfes_, HyperbolicFormIntegrator &formIntegrator_,
   const int num_equations_)
   : TimeDependentOperator(vfes_->GetNDofs() * num_equations_),
     dim(vfes_->GetFE(0)->GetDim()),
     num_equations(num_equations_),
     vfes(vfes_),
     formIntegrator(formIntegrator_),
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

   nonlinearForm->AddDomainIntegrator(&formIntegrator);
   nonlinearForm->AddInteriorFaceIntegrator(&formIntegrator);

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
   // 1. Create the vector z with the face terms (F(u), grad v) - <F.n(u), [w]>.
   nonlinearForm->Mult(x, z);
   max_char_speed = formIntegrator.getMaxCharSpeed();

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
      #pragma omp atomic
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

//////////////////////////////////////////////////////////////////
///                      NUMERICAL FLUXES                      ///
//////////////////////////////////////////////////////////////////

/**
 * @brief Rusanov flux hat(F)n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
 * where λ is the maximum characteristic velocity
 *
 */
class RusanovFlux : public RiemannSolver
{
public:
   /**
    * @brief  hat(F)n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    *
    * @param[in] state1 state value at a point from the first element
    * (num_equations x 1)
    * @param[in] state2 state value at a point from the second element
    * (num_equations x 1)
    * @param[in] fluxN1 normal flux value at a point from the first element
    * (num_equations x dim)
    * @param[in] fluxN2 normal flux value at a point from the second element
    * (num_equations x dim)
    * @param[in] maxE maximum characteristic speed (eigenvalue of flux jacobian)
    * @param[in] nor normal vector (not a unit vector) (dim x 1)
    * @param[out] flux ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    */
   void Eval(const Vector &state1, const Vector &state2, const Vector &fluxN1,
             const Vector &fluxN2, const double maxE, const Vector &nor,
             Vector &flux)
   {
      // NOTE: nor in general is not a unit normal

      flux = state1;
      flux -= state2;
      // here, sqrt(nor*nor) is multiplied to match the scale with fluxN
      flux *= maxE * sqrt(nor * nor);
      flux += fluxN1;
      flux += fluxN2;
      flux *= 0.5;
   }
};

// Upwind Flux, Not Yet Implemented
class UpwindFlux : public RiemannSolver
{
public:
   // Upwind Flux, Not Yet Implemented
   void Eval(const Vector &state1, const Vector &state2, const Vector &fluxN1,
             const Vector &fluxN2, const double maxE, const Vector &nor,
             Vector &flux)
   {
      // NOTE: nor in general is not a unit normal
      mfem_error("Not Implemented");
   }
};

//////////////////////////////////////////////////////////////////
///                        EULER SYSTEM                        ///
//////////////////////////////////////////////////////////////////

/**
 * @brief Euler element intgration (F(ρ, ρu, E), grad v)
 */
class EulerFormIntegrator : public HyperbolicFormIntegrator
{
private:
   const double specific_heat_ratio;  // specific heat ratio, γ
   // const double gas_constant;         // gas constant

   /**
    * @brief Compute F(ρ, ρu, E)
    *
    * @param state state (ρ, ρu, E) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
public:
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux)
   {
      const int dim = Tr.GetDimension();

      // 1. Get states
      const double density = state(0);                  // ρ
      const Vector momentum(state.GetData() + 1, dim);  // ρu
      const double energy = state(1 + dim);             // E, internal energy ρe
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
         flux(0, d) = momentum(d);  // ρu
         for (int i = 0; i < dim; i++)
         {
            // ρuuᵀ
            flux(1 + i, d) = momentum(i) * momentum(d) / density;
         }
         // (ρuuᵀ) + p
         flux(1 + d, d) += pressure;
      }
      // enthalpy H = e + p/ρ = (E + p)/ρ
      const double H = (energy + pressure) / density;
      for (int d = 0; d < dim; d++)
      {
         // u(E+p) = ρu*(E + p)/ρ = ρu*H
         flux(1 + dim, d) = momentum(d) * H;
      }

      // 3. Compute maximum characteristic speed

      // sound speed, √(γ p / ρ)
      const double sound = sqrt(specific_heat_ratio * pressure / density);
      // fluid speed |u|
      const double speed = sqrt(momentum * momentum) / density;
      // max characteristic speed = fluid speed + sound speed
      return speed + sound;
   }

   virtual void ComputeFluxJacobian(const Vector &state,
                                    ElementTransformation &Tr, DenseTensor &Jacobian, DenseMatrix &eigs)
   {
      const int dim = Tr.GetDimension();
      const int nvars = state.Size();

      const double density = state(0);                  // ρ
      const Vector momentum(state.GetData() + 1, dim);  // ρu
      const double energy = state(1 + dim);             // E, internal energy ρe
      // pressure, p = (γ-1)*(ρu - ½ρ|u|²)
      const double pressure = (specific_heat_ratio - 1.0) *
                              (energy - 0.5 * (momentum * momentum) / density);
      // ∂ p / ∂ ρ = ½(γ -1)|u|^2
      const double dpdrho = -0.5*(specific_heat_ratio - 1)*(momentum*momentum)/
                            (density*density);
      // ∂ p / ∂ (ρ u) = (γ - 1)u
      Vector dpdmom(dim);
      dpdmom = momentum;
      dpdmom *= (specific_heat_ratio - 1.0)/density;
      // ∂ p / ∂ E = γ - 1
      const double dpdE = specific_heat_ratio - 1.0;

      Jacobian = 0.0;
      for (int i=0; i<dim; i++)
      {
         double* current_col = Jacobian.GetData(i);

         // dF/dρ
         for (int j=0; j<dim; j++) { current_col[1 + j] = -(momentum[i]/(density*density))*momentum[j]; }
         current_col[1 + i] += dpdrho;
         current_col[1 + dim] += momentum[i]/density*dpdrho;

         // dF/d(ρu)
         Jacobian(0, 1 + i, 0) = 1.0; // dF_1/d(ρu)
         for (int j=0; j<dim; j++)
         {
            Jacobian(j + 1, i + 1, i) += momentum[j]/density; // (ρu * e_i^T)/ρ, column
            Jacobian(j + 1, j + 1, i) += momentum[i]/density; // (ρu_i / ρ) I, diagonal
            Jacobian(i + 1, j + 1, i) += dpdmom[j]; // e_i * dp/d(ρu)^T, row
            Jacobian(dim + 1, j + 1,
                     i) = momentum[i]/density*dpdmom[j]; // ((ρu)⋅e_i)*dp/d(ρu)^T, entry
         }
         Jacobian(dim + 1, i + 1, i) += (energy + pressure)/density; // (E + p)/ρ e_i^T

         // dF/dE
         current_col += nvars*(1 + dim); // move it to current column
         current_col[i + 1] = dpdE; // dp/dE e_i
         current_col[dim + 1] = momentum[i]/density*(energy +
                                                     dpdE); // (ρu)/ρ * (E  + dp/dE)
      }
   }


   /**
    * @brief Compute normal flux, F(ρ, ρu, E)n
    *
    * @param state state (ρ, ρu, E) at current integration point
    * @param nor normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param fluxN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &state, const Vector &nor,
                          ElementTransformation &Tr, Vector &fluxN)
   {
      const int dim = nor.Size();

      // 1. Get states
      const double density = state(0);                  // ρ
      const Vector momentum(state.GetData() + 1, dim);  // ρu
      const double energy = state(1 + dim);             // E, internal energy ρe
      // pressure, p = (γ-1)*(E - ½ρ|u|^2)
      const double pressure = (specific_heat_ratio - 1.0) *
                              (energy - 0.5 * (momentum * momentum) / density);

      // Check whether the solution is physical only in debug mode
      MFEM_ASSERT(density >= 0, "Negative Density");
      MFEM_ASSERT(pressure >= 0, "Negative Pressure");
      MFEM_ASSERT(energy >= 0, "Negative Energy");

      // 2. Compute normal flux

      fluxN(0) = momentum * nor;  // ρu⋅n
      // u⋅n
      const double normal_velocity = fluxN(0) / density;
      for (int d = 0; d < dim; d++)
      {
         // (ρuuᵀ + pI)n = ρu*(u⋅n) + pn
         fluxN(1 + d) = normal_velocity * momentum(d) + pressure * nor(d);
      }
      // (u⋅n)(E + p)
      fluxN(1 + dim) = normal_velocity * (energy + pressure);

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

   EulerFormIntegrator *elfi = new EulerFormIntegrator(
      numericalFlux, dim, specific_heat_ratio, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *elfi, num_equations);
}

//////////////////////////////////////////////////////////////////
///                      BURGERS EQUATION                      ///
//////////////////////////////////////////////////////////////////

/**
 * @brief Burgers element intgration (F(u), grad v)
 */
class BurgersFormIntegrator : public HyperbolicFormIntegrator
{
public:
   /**
    * @brief Compute F(u)
    *
    * @param state state (u) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(u) = ½u²*1ᵀ where 1 is (dim x 1) vector
    * @return double maximum characteristic speed, |u|
    */
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux)
   {
      flux = state * state * 0.5;
      return abs(state(0));
   }
   /**
    * @brief Compute normal flux, F(u)n
    *
    * @param state state (u) at current integration point
    * @param nor normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param fluxN F(u)n = ½u² 1⋅n
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &state, const Vector &nor,
                          ElementTransformation &Tr, Vector &fluxN)
   {
      fluxN = nor.Sum() * (state * state) * 0.5;
      return abs(state(0));
   }


   virtual void ComputeFluxJacobian(const Vector &state,
                                    ElementTransformation &Tr, DenseTensor &Jacobian, DenseMatrix &eigs)
   {
      // const int dim = Tr.GetDimension();
      Jacobian = state(0);
      eigs = state(0);
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

   BurgersFormIntegrator *elfi =
      new BurgersFormIntegrator(numericalFlux, dim, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *elfi, num_equations);
}

//////////////////////////////////////////////////////////////////
///                     ADVECTION EQUATION                     ///
//////////////////////////////////////////////////////////////////

// Advection equation main class. Overload ComputeFlux
class AdvectionFormIntegrator : public HyperbolicFormIntegrator
{
private:
   VectorCoefficient &b;  // velocity coefficient
   Vector bval;           // velocity value storage

public:
   /**
    * @brief Compute F(u)
    *
    * @param state state (u) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(u) = ubᵀ
    * @return double maximum characteristic speed, |b|
    */
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux)
   {
      b.Eval(bval, Tr, Tr.GetIntPoint());
      MultVWt(state, bval, flux);
      return bval.Norml2();
   }

   virtual void ComputeFluxJacobian(const Vector &state,
                                    ElementTransformation &Tr, DenseTensor &Jacobian, DenseMatrix &eigs)
   {
      const int dim = Tr.GetDimension();
      b.Eval(bval, Tr, Tr.GetIntPoint());
      for (int i=0; i<dim; i++) { Jacobian(0, 0, i) = bval(i); }
      for (int i=0; i<dim; i++) { eigs(0,i) = bval(i); }
   }
   /**
    * @brief Compute normal flux, F(u)n
    *
    * @param state state (u) at current integration point
    * @param nor normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param fluxN F(u)n = u(b⋅n)
    * @return double maximum characteristic speed, |b|
    */
   double ComputeFluxDotN(const Vector &state, const Vector &nor,
                          ElementTransformation &Tr, Vector &fluxN)
   {
      b.Eval(bval, Tr, Tr.GetIntPoint());
      const double bN = bval * nor;
      fluxN = state;
      fluxN *= bN;
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

   AdvectionFormIntegrator *elfi =
      new AdvectionFormIntegrator(numericalFlux, dim, b, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *elfi, num_equations);
}

//////////////////////////////////////////////////////////////////
///                        SHALLOW WATER                       ///
//////////////////////////////////////////////////////////////////

// ShallowWater equation element integration. Overload ComputeFlux
class ShallowWaterFormIntegrator : public HyperbolicFormIntegrator
{
private:
   const double g;  // gravity constant

public:
   /**
    * @brief Compute F(h, hu)
    *
    * @param state state (h, hu) at current integration point
    * @param Tr current element transformation with integration point
    * @param flux F(h, hu) = [huᵀ; huuᵀ + ½gh²I]
    * @return double maximum characteristic speed, |u| + √(gh)
    */
   double ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux)
   {
      const int dim = state.Size() - 1;
      const double height = state(0);
      const Vector h_vel(state.GetData() + 1, dim);

      const double energy = 0.5 * g * (height * height);

      MFEM_ASSERT(height >= 0, "Negative Height");

      for (int d = 0; d < dim; d++)
      {
         flux(0, d) = h_vel(d);
         for (int i = 0; i < dim; i++)
         {
            flux(1 + i, d) = h_vel(i) * h_vel(d) / height;
         }
         flux(1 + d, d) += energy;
      }

      const double sound = sqrt(g * height);
      const double vel = sqrt(h_vel * h_vel) / height;

      return vel + sound;
   }
   /**
    * @brief Compute normal flux, F(h, hu)
    *
    * @param state state (h, hu) at current integration point
    * @param nor normal vector, usually not a unit vector
    * @param Tr current element transformation with integration point
    * @param fluxN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return double maximum characteristic speed, |u| + √(γp/ρ)
    */
   double ComputeFluxDotN(const Vector &state, const Vector &nor,
                          ElementTransformation &Tr, Vector &fluxN)
   {
      const int dim = nor.Size();
      const double height = state(0);
      const Vector h_vel(state.GetData() + 1, dim);

      const double energy = 0.5 * g * (height * height);

      MFEM_ASSERT(height >= 0, "Negative Height");
      fluxN(0) = h_vel * nor;
      const double normal_vel = fluxN(0) / height;
      for (int i = 0; i < dim; i++)
      {
         fluxN(1 + i) = normal_vel * h_vel(i) + energy * nor(i);
      }

      const double sound = sqrt(g * height);
      const double vel = sqrt(h_vel * h_vel) / height;

      return vel + sound;
   }

   virtual void ComputeFluxJacobian(const Vector &state,
                                    ElementTransformation &Tr, DenseTensor &Jacobian, DenseMatrix &eigs)
   {
      const int dim = Tr.GetDimension();
      const int nvars = state.Size();

      const double height = state(0);                   // h
      const Vector momentum(state.GetData() + 1, dim);  // hu

      Jacobian = 0.0;
      for (int i=0; i<dim; i++)
      {
         // double* current_col = Jacobian.GetData(i);
         // dF/dh
         for (int j=0; j<dim; j++) { Jacobian(1 + j, 0, i) = -(momentum[i]/(height*height))*momentum[j]; }
         Jacobian(1 + i, 0, i) += g*height;

         // dF/d(hu)
         Jacobian(0, 1 + i, i) = 1.0; // dF_1/d(ρu)
         for (int j=0; j<dim; j++)
         {
            Jacobian(j + 1, i + 1, i) += momentum[j]/height; // (hu * e_i^T)/h, column
            Jacobian(j + 1, j + 1, i) += momentum[i]/height; // (hu_i / h) I, diagonal
         }
      }
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

   ShallowWaterFormIntegrator *elfi =
      new ShallowWaterFormIntegrator(numericalFlux, dim, g, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *elfi, num_equations);
}

// void weno(const GridFunction &u, FiniteElementSpace &fespace,
//           const HyperbolicFormIntegrator &form)
// {
//    Mesh *mesh = fespace.GetMesh();
//    DG_FECollection fec(0, mesh->Dimension());
//    FiniteElementSpace p0_space(mesh, &fec, u.VectorDim(), Ordering::byNODES);
//    GridFunction avg_u(&p0_space);
//    u.GetElementAverages(avg_u);
//    for (int i=0; i<mesh->GetNFaces(); i++)
//    {
//       auto face = mesh->GetFace(i);
//       int e1, e2;
//       mesh->GetFaceElements(i, &e1, &e2);
//       if (e2)
//       {

//       }
//    }
// }

/**
 * @brief Error estimator based on RT projection of flux and numerical flux
 *
 */
class FluxReconErrorEstimator : public ErrorEstimator
{
protected:
   long current_sequence; // current FE sequence
   Vector error_estimates; // element-wise error
   double total_error; // total error

   GridFunction &solution;

   HyperbolicFormIntegrator *hpblfi;

   RT_FECollection *RTfec;
   

   /// Check if the finite element of the solution was modified.
   bool FESpaceIsModified()
   {
      long fe_sequence = solution.FESpace()->GetSequence();
      MFEM_ASSERT(fe_sequence >= current_sequence, "");
      return (fe_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   /** @brief Construct a new FluxReconErrorEstimator object. ||π_RT(F(u_h), F̂(u_h, n)) - F(u_h)||
       @param sol      The solution field whose error is to be estimated.
       @param hyperbolicFormIntegrator 
   */
   FluxReconErrorEstimator(GridFunction &sol, HyperbolicFormIntegrator *hyperbolicFormIntegrator)
      : current_sequence(-1),
        total_error(-1.0),
        solution(sol),
        hpblfi(hyperbolicFormIntegrator)
   { 
      RTfec = new RT_FECollection(0, solution.FESpace()->GetMesh()->Dimension());
   }

   /// Return the total error from the last error estimate.
   virtual double GetTotalError() const override { return total_error; }

   /// Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors() override
   {
      if (FESpaceIsModified())
      {
         ComputeEstimates();
      }
      return error_estimates;
   }

   /// Reset the error estimator.
   virtual void Reset() override { current_sequence = -1; }

   virtual ~FluxReconErrorEstimator() { }
};

} // end mfem namespace


#endif