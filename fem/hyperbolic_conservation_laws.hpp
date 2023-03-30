//           MFEM Hyperbolic Conservation Laws Serial/Parallel Shared Class
//
// Description:  This is a refactored version of ex18.hpp to handle general
//               class of hyperbolic conservation laws.
//               Abstract RiemannSolver, HyperbolicElementFormIntegrator,
//               HyperbolicFaceFormIntegrator, and DGHyperbolicConservationLaws
//               are defined.
//
//               To implement a specific hyperbolic conservation laws, users can
//               create derived classes from HyperbolicElementFormIntegrator and
//               HyperbolicFaceFormIntegrator with specific flux evaluation.
//               Examples of derived classes are included such as compressible
//               Euler equations, Burgers equation, Advection equation, and
//               Shallow water equations.
//
//               FormIntegrators use high-order quadrature points to implement
//               the given form to handle nonlinear flux function. During the
//               implementation of forms, it updates maximum characteristic
//               speed and collected by DGHyperbolicConservationLaws. The global
//               maximum characteristic speed can be obtained by public method
//               getMaxCharSpeed so that the time step can be determined by CFL
//               condition.
//               @note For parallel version, users should reduce all maximum
//               characteristic speed from all processes using MPI_Allreduce.
//
// TODO: Implement limiter. IDEA:
//
//
// Class structure: DGHyperbolicConservationLaws
//                  |- HyperbolicElementFormIntegrator: (F(u,x), grad v)
//                  |- HyperbolicFaceFormIntegrator: <hat(F)(u+,x;u-,x), [[v]])
//                  |  |- RiemannSolver: hat(F)
//                  |
//                  |- (Par)NonlinearForm: Evaluate form integrators
//

#ifndef MFEM_HYPERBOLIC_CONSERVATION_LAWS
#define MFEM_HYPERBOLIC_CONSERVATION_LAWS
#include "mfem.hpp"

using namespace std;
using namespace mfem;

enum PRefineType
{
   elevation,  // order <- order + value
   setDegree,  // order <- value
};
GridFunction ProlongToMaxOrderDG(const GridFunction &x);

/**
 * @brief Abstract class for numerical flux for an hyperbolic conservation laws
 * on a face with states, fluxes and characteristic speed
 *
 */
class RiemannSolver
{
public:
   RiemannSolver() {};
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
 * @brief Abstract hyperbolic element form integrator, (F(u, x), ∇v)
 *
 */
class HyperbolicElementFormIntegrator : public NonlinearFormIntegrator
{
private:
   const int num_equations;  // the number of equations
   // The maximum characterstic speed, updated during element vector assembly
   double max_char_speed;
   const int IntOrderOffset;  // 2*p + IntOrderOffset will be used for quadrature
   Vector shape;              // shape function value at an integration point
   Vector state;              // state value at an integration point
   DenseMatrix flux;          // flux value at an integration point
   DenseMatrix dshape;  // derivative of shape function at an integration point

public:
   /**
    * @brief Compute flux F(u, x) for given state u and physical point x
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
    * @brief Construct a new Hyperbolic Element Form Integrator object
    *
    * @param[in] dim physical dimension
    * @param[in] num_equations_ the number of equations
    * @param[in] IntOrderOffset_ 2*p+IntOrderOffset order Gaussian quadrature
    * will be used
    */
   HyperbolicElementFormIntegrator(const int dim, const int num_equations_,
                                   const int IntOrderOffset_ = 3)
      : NonlinearFormIntegrator(),
        num_equations(num_equations_),
        IntOrderOffset(IntOrderOffset_),
        state(num_equations_),
        flux(num_equations_, dim) {};
   /**
    * @brief Construct an object with a fixed integration rule
    *
    * @param[in] dim physical dimension
    * @param[in] num_equations_ the number of equations
    * @param[in] ir integration rule to be used
    */
   HyperbolicElementFormIntegrator(const int dim, const int num_equations_,
                                   const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir),
        num_equations(num_equations_),
        IntOrderOffset(0),
        state(num_equations_),
        flux(num_equations_, dim) {};

   /**
    * @brief Get the integration rule based on IntOrderOffset, @see
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
    * @brief Set the Max Char Speed pointer to be collected later
    *
    * @param max_char_speed_ maximum characteristic speed (its pointer will be
    * used)
    */
   inline void setMaxCharSpeed(double max_char_speed_)
   {
      max_char_speed = max_char_speed_;
   }

   inline double getMaxCharSpeed() { return max_char_speed; }

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
   };

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

   virtual ~HyperbolicElementFormIntegrator() {}
};

/**
 * @brief Abstract class for hyperbolic face form integrator <-hat(F)(u, x) n,
 * [[v]]> where hat(F)(u, x)n is defined by ComputeFluxDotN and then given
 * Numerical Flux
 *
 */
class HyperbolicFaceFormIntegrator : public NonlinearFormIntegrator
{
private:
   const int num_equations;  // the number of equations
   // The maximum characterstic speed, updated during face vector assembly
   double max_char_speed;
   const int IntOrderOffset;  // 2*p + IntOrderOffset will be used for quadrature
   RiemannSolver *rsolver;    // Numerical flux that maps F(u±,x) to hat(F)
   Vector shape1;  // shape function value at an integration point - first elem
   Vector shape2;  // shape function value at an integration point - second elem
   Vector state1;  // state value at an integration point - first elem
   Vector state2;  // state value at an integration point - second elem
   Vector fluxN1;  // flux dot n value at an integration point - first elem
   Vector fluxN2;  // flux dot n value at an integration point - second elem
   Vector nor;     // normal vector (usually not a unit vector)
   Vector fluxN;   // hat(F)(u,x)

public:
   /**
    * @brief Abstract method to compute normal flux. Must be overloaded in the
    * derived class
    *
    * @param[in] state state value at the current integration point
    * @param[in] nor normal vector (usually not a unit vector)
    * @param[in] Tr element transformation
    * @param[out] fluxN normal flux from the given element at the current
    * integration point
    * @return double maximum characteristic velocity
    */
   virtual double ComputeFluxDotN(const Vector &state, const Vector &nor,
                                  ElementTransformation &Tr, Vector &fluxN) = 0;

   /**
    * @brief Construct a new Hyperbolic Face Form Integrator object
    *
    * @param[in] rsolver_ numerical flux
    * @param[in] dim spatial dimension
    * @param[in] num_equations_ the number of equations
    * @param[in] IntOrderOffset_ integration order offset, 2*p + IntOrderOffset
    * will be used
    */
   HyperbolicFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                                const int num_equations_,
                                const int IntOrderOffset_ = 3)
      : NonlinearFormIntegrator(),
        num_equations(num_equations_),
        max_char_speed(0.0),
        IntOrderOffset(IntOrderOffset_),
        rsolver(rsolver_),
        state1(num_equations_),
        state2(num_equations_),
        fluxN1(num_equations_),
        fluxN2(num_equations_),
        nor(dim),
        fluxN(num_equations_) {};
   /**
    * @brief Construct an object with given integration rule
    *
    * @param[in] rsolver_ numerical flux
    * @param[in] dim spatial dimension
    * @param[in] num_equations_ the number of equations
    * @param[in] ir integration rule
    */
   HyperbolicFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                                const int num_equations_,
                                const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir),
        num_equations(num_equations_),
        max_char_speed(0.0),
        IntOrderOffset(0),
        rsolver(rsolver_),
        state1(num_equations_),
        state2(num_equations_),
        fluxN1(num_equations_),
        fluxN2(num_equations_),
        nor(dim),
        fluxN(num_equations_) {};

   /**
    * @brief Get the integration rule based on IntOrderOffset, @see
    * AssembleFaceVector. Used only when ir is not provided
    *
    * @param[in] trial_el trial finite element space
    * @param[in] test_el test finite element space
    * @return const IntegrationRule& with order 2*p + IntOrderOffset
    */
   const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                  const FiniteElement &test_fe)
   {
      int order;
      order = max(trial_fe.GetOrder(), test_fe.GetOrder()) * 2 + IntOrderOffset;
      return IntRules.Get(trial_fe.GetGeomType(), order);
   }

   /**
    * @brief Set the Max Char Speed pointer to be collected later
    *
    * @param max_char_speed_ maximum characteristic speed (its pointer will be
    * used)
    */
   inline void setMaxCharSpeed(double max_char_speed_)
   {
      max_char_speed = max_char_speed_;
   }

   inline double getMaxCharSpeed() { return max_char_speed; }

   /**
    * @brief implement <-hat(F)(u,x) n, [[v]]> with abstract hat(F) computed by
    * ComputeFluxDotN and numerical flux object
    *
    * @param[in] el finite element of the first element
    * @param[in] el finite element of the second element
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis from both elements
    * @param[out] elvect evaluated dual vector <-hat(F)(u,x) n, [[v]]>
    */
   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
   virtual ~HyperbolicFaceFormIntegrator() {}
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
   HyperbolicElementFormIntegrator &elementFormIntegrator;
   // Face integration form. Should contain ComputeFluxDotN and Riemann Solver
   HyperbolicFaceFormIntegrator &faceFormIntegrator;
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

   void Update()
   {
      nonlinearForm->Update();
      height = nonlinearForm->Height();
      width = height;
      z.SetSize(height);

      ComputeInvMass();
   };

public:
   /**
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param nonlinForm_ empty nonlinear form with vfes. This is an input to
    * support ParNonlinearForm
    * @param elementFormIntegrator_ (F(u,x), grad v)
    * @param faceFormIntegrator_ <-hat(F)(u,x), [[v]]>
    * @param num_equations_ the number of equations
    */
   DGHyperbolicConservationLaws(
      FiniteElementSpace *vfes_,
      HyperbolicElementFormIntegrator &elementFormIntegrator_,
      HyperbolicFaceFormIntegrator &faceFormIntegrator_,
      const int num_equations_);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   virtual void Mult(const Vector &x, Vector &y) const;

   void pRefine(const Array<int> &orders, GridFunction &sol,
                const PRefineType pRefineType = PRefineType::elevation);

   void hRefine(Vector &errors, GridFunction &sol, const double z = 0.842);
   void hDerefine(Vector &errors, GridFunction &sol, const double z = 0.842);
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   inline double getMaxCharSpeed() { return max_char_speed; }

   virtual ~DGHyperbolicConservationLaws() {}
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace *vfes_,
   HyperbolicElementFormIntegrator &elementFormIntegrator_,
   HyperbolicFaceFormIntegrator &faceFormIntegrator_, const int num_equations_)
   : TimeDependentOperator(vfes_->GetNDofs() * num_equations_),
     dim(vfes_->GetFE(0)->GetDim()),
     num_equations(num_equations_),
     vfes(vfes_),
     elementFormIntegrator(elementFormIntegrator_),
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
   elementFormIntegrator.setMaxCharSpeed(max_char_speed);
   faceFormIntegrator.setMaxCharSpeed(max_char_speed);

   nonlinearForm->AddDomainIntegrator(&elementFormIntegrator);
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

/**
 * @brief Refine/Derefine FiniteElementSpace for DGHyperbolic.
 *
 * For each 0 ≤ i ≤ the number of elements,
 * If pRefineType == PRefineType::elevation, order(i) += orders(i).
 * If pRefineType == PRefineType::setDegree, order(i) = orders(i).
 *
 * @param[in] orders element-wise order offset or target order.
 * @param[out] sol The solution to be updated
 * @param[in] pRefineType p-refinement type. Default PRefineType::elevation.
 */
void DGHyperbolicConservationLaws::pRefine(const Array<int> &orders,
                                           GridFunction &sol,
                                           const PRefineType pRefineType)
{
   Mesh *mesh = vfes->GetMesh();
   mesh->EnsureNCMesh();  // Variable order needs nonconforming mesh
   const int numElem = mesh->GetNE();
   MFEM_VERIFY(orders.Size() == numElem, "Incorrect size of array is provided.");

   // FiniteElementSpace old_vfes(*vfes);  // save old fes
   DG_FECollection *fec = new DG_FECollection(0, mesh->Dimension());
   FiniteElementSpace old_vfes(mesh, fec, vfes->GetVDim(), vfes->GetOrdering());
   for (int i = 0; i < numElem; i++)
   {
      old_vfes.SetElementOrder(i, vfes->GetElementOrder(i));
   }
   old_vfes.Update(false);

   switch (pRefineType)
   {
      case PRefineType::elevation:
         for (int i = 0; i < numElem; i++)
         {
            if (orders[i] != 0)
            {
               const int order = vfes->GetElementOrder(i) + orders[i];
               MFEM_VERIFY(order >= 0, "Order should be non-negative.");
               vfes->SetElementOrder(i, vfes->GetElementOrder(i) + orders[i]);
            }
         }
         break;
      case PRefineType::setDegree:
         MFEM_VERIFY(orders.Min() >= 0, "Order should be non-negative.");
         for (int i = 0; i < numElem; i++)
         {
            vfes->SetElementOrder(i, orders[i]);
         }
         break;
      default:
         mfem_error("Undefined PRefineType.");
   }
   vfes->Update(false);  // p-refine transfer matrix is not provided.

   PRefinementTransferOperator T(old_vfes, *vfes);
   GridFunction new_sol(vfes);
   T.Mult(sol, new_sol);
   sol = new_sol;
   Update();
}

void DGHyperbolicConservationLaws::hRefine(Vector &errors, GridFunction &sol,
                                           const double z)
{
   Vector logErrors(errors);
   for (auto &err : logErrors) { err = log(err); }
   const double mean = logErrors.Sum() / logErrors.Size();
   const double sigma =
      pow(logErrors.Norml2(), 2) / logErrors.Size() - pow(mean, 2);
   const double upper_bound = exp(mean + z * pow(sigma / logErrors.Size(), 0.5));
   Mesh *mesh = vfes->GetMesh();
   mesh->RefineByError(errors, upper_bound * 1.5, 1, 2);
   vfes->Update();
   sol.Update();
   Update();
}

/**
 * @brief Perform h-derefinement with given error and z-score
 *
 * Threshold is set to log(err[i]) < mean(log(err)) - z*std(log(err))
 * and error is aggregate by using max(neighbors).
 *
 * @param errors element-wise error
 * @param sol solution (updated by using interpolation)
 * @param z z-score for the threshold
 */
void DGHyperbolicConservationLaws::hDerefine(Vector &errors, GridFunction &sol,
                                             const double z)
{
   Vector logErrors(errors);
   for (auto &err : logErrors) { err = log(err); }
   const double mean = logErrors.Sum() / logErrors.Size();
   const double sigma =
      pow(logErrors.Norml2(), 2) / logErrors.Size() - pow(mean, 2);
   const double lower_bound = exp(mean - z * pow(sigma / logErrors.Size(), 0.5));
   Mesh *mesh = vfes->GetMesh();
   mesh->DerefineByError(errors, lower_bound, 2, 0);
   vfes->Update();
   sol.Update();
   Update();
}

void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   elementFormIntegrator.setMaxCharSpeed(0.0);
   faceFormIntegrator.setMaxCharSpeed(0.0);
   // 1. Create the vector z with the face terms (F(u), grad v) - <F.n(u), [w]>.
   nonlinearForm->Mult(x, z);
   max_char_speed = max(elementFormIntegrator.getMaxCharSpeed(),
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

//////////////////////////////////////////////////////////////////
///                      ELEMENT INTEGRATOR                    ///
//////////////////////////////////////////////////////////////////
void HyperbolicElementFormIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   Vector &elvect)
{
   // current element's the number of degrees of freedom
   // does not consider the number of equations
   const int dof = el.GetDof();
   // resize shape and gradient shape storage
   shape.SetSize(dof);
   dshape.SetSize(dof, el.GetDim());
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
      max_char_speed = max(mcs, max_char_speed);
      // integrate (F(u,x), grad v)
      AddMult_a_ABt(ip.weight * Tr.Weight(), dshape, flux, elvect_mat);
   }
}
//////////////////////////////////////////////////////////////////
///                       FACE INTEGRATOR                      ///
//////////////////////////////////////////////////////////////////

void HyperbolicFaceFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   // current elements' the number of degrees of freedom
   // does not consider the number of equations
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1.SetSize(dof1);
   shape2.SetSize(dof2);

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
      if (nor.Size() == 1)    // if 1D, use 1 or -1.
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
      const double mcs = max(
                            ComputeFluxDotN(state1, nor, Tr.GetElement1Transformation(), fluxN1),
                            ComputeFluxDotN(state2, nor, Tr.GetElement2Transformation(), fluxN2));
      // Compute hat(F) using evaluated quantities
      rsolver->Eval(state1, state2, fluxN1, fluxN2, mcs, nor, fluxN);

      // Update the global max char speed
      max_char_speed = max(mcs, max_char_speed);

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
class EulerElementFormIntegrator : public HyperbolicElementFormIntegrator
{
private:
   const double specific_heat_ratio;  // specific heat ratio, γ
   const double gas_constant;         // gas constant

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
      const int dim = state.Size() - 2;

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

   /**
    * @brief Construct a new Euler Element Form Integrator object with given
    * integral order offset
    *
    * @param dim spatial dimension
    * @param specific_heat_ratio_ specific heat ratio, γ
    * @param gas_constant_ gas constant
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   EulerElementFormIntegrator(const int dim, const double specific_heat_ratio_,
                              const double gas_constant_,
                              const int IntOrderOffset_ = 3)
      : HyperbolicElementFormIntegrator(dim, dim + 2, IntOrderOffset_),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_) {}

   /**
    * @brief Construct a new Euler Element Form Integrator object with given
    * integral rule
    *
    * @param dim spatial dimension
    * @param specific_heat_ratio_ specific heat ratio, γ
    * @param gas_constant_ gas constant
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   EulerElementFormIntegrator(const int dim, const double specific_heat_ratio_,
                              const double gas_constant_,
                              const IntegrationRule *ir)
      : HyperbolicElementFormIntegrator(dim, dim + 2, ir),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_) {}
};

/**
 * @brief Euler face intgration <hat(F)(ρ, ρu, E)n, [[v]]>
 */
class EulerFaceFormIntegrator : public HyperbolicFaceFormIntegrator
{
private:
   const double specific_heat_ratio;  // specific heat ratio, γ
   const double gas_constant;         // gas constant

public:
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
    * @brief Construct a new Euler Face Form Integrator object with given
    * integral order offset
    *
    * @param rsolver_ Numerical flux to compute hat(F)
    * @param dim spatial dimension
    * @param specific_heat_ratio_ specific heat ratio, γ
    * @param gas_constant_ gas constant
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   EulerFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                           const double specific_heat_ratio_,
                           const double gas_constant_,
                           const int IntOrderOffset_ = 3)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, dim + 2, IntOrderOffset_),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_) {};

   /**
    * @brief Construct a new Euler Face Form Integrator object with given
    * integral rule
    *
    * @param rsolver_ Numerical flux to compute hat(F)
    * @param dim spatial dimension
    * @param specific_heat_ratio_ specific heat ratio, γ
    * @param gas_constant_ gas constant
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   EulerFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                           const double specific_heat_ratio_,
                           const double gas_constant_, const IntegrationRule *ir)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, dim + 2, ir),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_) {};
};

DGHyperbolicConservationLaws getEulerSystem(FiniteElementSpace *vfes,
                                            RiemannSolver *numericalFlux,
                                            const double specific_heat_ratio,
                                            const double gas_constant,
                                            const int IntOrderOffset)
{
   const int dim = vfes->GetMesh()->Dimension();
   const int num_equations = dim + 2;

   EulerElementFormIntegrator *elfi = new EulerElementFormIntegrator(
      dim, specific_heat_ratio, gas_constant, IntOrderOffset);

   EulerFaceFormIntegrator *fnfi = new EulerFaceFormIntegrator(
      numericalFlux, dim, specific_heat_ratio, gas_constant, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *elfi, *fnfi, num_equations);
}

//////////////////////////////////////////////////////////////////
///                      BURGERS EQUATION                      ///
//////////////////////////////////////////////////////////////////

/**
 * @brief Burgers element intgration (F(u), grad v)
 */
class BurgersElementFormIntegrator : public HyperbolicElementFormIntegrator
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
    * @brief Construct a new Burgers Element Form Integrator object with given
    * integral order offset
    *
    * @param dim spatial dimension
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   BurgersElementFormIntegrator(const int dim, const int IntOrderOffset_ = 3)
      : HyperbolicElementFormIntegrator(dim, 1, IntOrderOffset_) {};
   /**
    * @brief Construct a new Burgers Element Form Integrator object with given
    * integral rule
    *
    * @param dim spatial dimension
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   BurgersElementFormIntegrator(const int dim, const IntegrationRule *ir)
      : HyperbolicElementFormIntegrator(dim, 1, ir) {};
};

/**
 * @brief Burgers face intgration <hat(F)(u), [[v]]>
 */
class BurgersFaceFormIntegrator : public HyperbolicFaceFormIntegrator
{
public:
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

   /**
    * @brief Construct a new Burgers Face Form Integrator object with given
    * integral order offset
    *
    * @param rsolver_ Numerical flux to compute hat(F)
    * @param dim spatial dimension
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   BurgersFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                             const int IntOrderOffset_ = 3)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, 1, IntOrderOffset_) {};

   /**
    * @brief Construct a new Burgers Face Form Integrator object with given
    * integral rule
    *
    * @param rsolver_ Numerical flux to compute hat(F)
    * @param dim spatial dimension
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   BurgersFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                             const IntegrationRule *ir)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, 1, ir) {};
};

DGHyperbolicConservationLaws getBurgersEquation(FiniteElementSpace *vfes,
                                                RiemannSolver *numericalFlux,
                                                const int IntOrderOffset)
{
   const int dim = vfes->GetMesh()->Dimension();
   const int num_equations = 1;

   BurgersElementFormIntegrator *elfi =
      new BurgersElementFormIntegrator(dim, IntOrderOffset);

   BurgersFaceFormIntegrator *fnfi =
      new BurgersFaceFormIntegrator(numericalFlux, dim, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *elfi, *fnfi, num_equations);
}

//////////////////////////////////////////////////////////////////
///                     ADVECTION EQUATION                     ///
//////////////////////////////////////////////////////////////////

// Advection equation main class. Overload ComputeFlux
class AdvectionElementFormIntegrator : public HyperbolicElementFormIntegrator
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

   /**
    * @brief Construct a new Advection Element Form Integrator object with given
    * integral order offset
    *
    * @param dim spatial dimension
    * @param b_ velocity coefficient, possibly depends on space
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   AdvectionElementFormIntegrator(const int dim, VectorCoefficient &b_,
                                  const int IntOrderOffset_ = 3)
      : HyperbolicElementFormIntegrator(dim, 1, IntOrderOffset_),
        b(b_),
        bval(dim) {};
   /**
    * @brief Construct a new Advection Element Form Integrator object with given
    * integral rule
    *
    * @param dim spatial dimension
    * @param b_ velocity coefficient, possibly depends on space
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   AdvectionElementFormIntegrator(const int dim, VectorCoefficient &b_,
                                  const IntegrationRule *ir)
      : HyperbolicElementFormIntegrator(dim, 1, ir), b(b_), bval(dim) {};
};

// Advection equation face integration. Overload ComputeFluxDotN
class AdvectionFaceFormIntegrator : public HyperbolicFaceFormIntegrator
{
private:
   VectorCoefficient &b;  // velocity coefficient
   Vector bval;           // velocity value storage
public:
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
    * @brief Construct a new Advection Face Form Integrator object with given
    * integral order offset
    *
    * @param rsolver_ Numerical flux to compute hat(F)
    * @param dim spatial dimension
    * @param b_ velocity coefficient, possibly depends on space
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   AdvectionFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                               VectorCoefficient &b_,
                               const int IntOrderOffset_ = 3)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, 1, IntOrderOffset_),
        b(b_),
        bval(dim) {};

   /**
    * @brief Construct a new Advection Face Form Integrator object with given
    * integral rule
    *
    * @param rsolver_ Numerical flux to compute hat(F)
    * @param dim spatial dimension
    * @param b_ velocity coefficient, possibly depends on space
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   AdvectionFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                               VectorCoefficient &b_, const IntegrationRule *ir)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, 1, ir), b(b_), bval(dim) {};
};

DGHyperbolicConservationLaws getAdvectionEquation(FiniteElementSpace *vfes,
                                                  RiemannSolver *numericalFlux,
                                                  VectorCoefficient &b,
                                                  const int IntOrderOffset)
{
   const int dim = vfes->GetMesh()->Dimension();
   const int num_equations = 1;

   AdvectionElementFormIntegrator *elfi =
      new AdvectionElementFormIntegrator(dim, b, IntOrderOffset);

   AdvectionFaceFormIntegrator *fnfi =
      new AdvectionFaceFormIntegrator(numericalFlux, dim, b, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *elfi, *fnfi, num_equations);
}

//////////////////////////////////////////////////////////////////
///                        SHALLOW WATER                       ///
//////////////////////////////////////////////////////////////////

// ShallowWater equation element integration. Overload ComputeFlux
class ShallowWaterElementFormIntegrator
   : public HyperbolicElementFormIntegrator
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
    * @brief Construct a new Shallow Water Element Form Integrator object with
    * given integral order offset
    *
    * @param dim spatial dimension
    * @param g_ gravity constant
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   ShallowWaterElementFormIntegrator(const int dim, const double g_,
                                     const int IntOrderOffset_ = 3)
      : HyperbolicElementFormIntegrator(dim, dim + 1, IntOrderOffset_), g(g_) {};
   /**
    * @brief Construct a new Shallow Water Element Form Integrator object with
    * given integral rule
    *
    * @param dim spatial dimension
    * @param g_ gravity constant
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   ShallowWaterElementFormIntegrator(const int dim, const double g_,
                                     const IntegrationRule *ir)
      : HyperbolicElementFormIntegrator(dim, dim + 1, ir), g(g_) {};
};

// ShallowWater equation face integration. Overload ComputeFluxDotN
class ShallowWaterFaceFormIntegrator : public HyperbolicFaceFormIntegrator
{
private:
   const double g;

public:
   /**
    * @brief Compute normal flux, F(h, hu)
    *
    * @param state state (h, hu) at current integration point
    * @param flux F(h, hu) = [huᵀ; huuᵀ + ½gh²I]
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

   /**
    * @brief Construct a new shallow water Face Form Integrator object with given
    * integral order offset
    *
    * @param rsolver_ Numerical flux to compute hat(F)
    * @param dim spatial dimension
    * @param g_ gravity constant
    * @param IntOrderOffset_ 2*p + IntOrderOffset will be used for quadrature
    */
   ShallowWaterFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                                  const double g_, const int IntOrderOffset_ = 3)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, dim + 1, IntOrderOffset_),
        g(g_) {};

   /**
    * @brief Construct a new shallow water Face Form Integrator object with given
    * integral rule
    *
    * @param rsolver_ Numerical flux to compute hat(F)
    * @param dim spatial dimension
    * @param g_ gravity constant
    * @param ir this integral rule will be used for the Gauss quadrature
    */
   ShallowWaterFaceFormIntegrator(RiemannSolver *rsolver_, const int dim,
                                  const double g_, const IntegrationRule *ir)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, dim + 1, ir), g(g_) {};
};

// Experimental - required for visualizing functions on p-refined spaces.
GridFunction ProlongToMaxOrderDG(const GridFunction &x)
{
   const FiniteElementSpace *fes = x.FESpace();
   Mesh *mesh = fes->GetMesh();
   const int max_order = fes->GetMaxElementOrder() + 1;
   FiniteElementCollection *fec =
      new DG_FECollection(max_order, mesh->Dimension());
   FiniteElementSpace *new_fes =
      new FiniteElementSpace(mesh, fec, fes->GetVDim(), fes->GetOrdering());
   PRefinementTransferOperator T(*fes, *new_fes);
   GridFunction u(new_fes);
   T.Mult(x, u);
   u.MakeOwner(fec);
   return u;
}

DGHyperbolicConservationLaws getShallowWaterEquation(
   FiniteElementSpace *vfes, RiemannSolver *numericalFlux, const double g,
   const int IntOrderOffset)
{
   const int dim = vfes->GetMesh()->Dimension();
   const int num_equations = dim + 1;

   ShallowWaterElementFormIntegrator *elfi =
      new ShallowWaterElementFormIntegrator(dim, g, IntOrderOffset);

   ShallowWaterFaceFormIntegrator *fnfi =
      new ShallowWaterFaceFormIntegrator(numericalFlux, dim, g, IntOrderOffset);

   return DGHyperbolicConservationLaws(vfes, *elfi, *fnfi, num_equations);
}

#endif