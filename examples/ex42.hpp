//               MFEM Example 42 - MHD with Godunov-Powell source
//                      (Time-dependent DG operator and model problem)
//
// This code defines the Orszag-Tang model problem and the DG operator for
// ideal MHD with the nonconservative Godunov-Powell source:
//
//                 U_t + div(F(U)) = -S(U) div(B),
//                 U = (rho, rho*v, B, E)^T,
//                 S(U) = (0, B, v, v dot B)^T.
//
// For each element K, the semidiscrete weak formulation is
//
//   (U_t, w)_K - (F(U), grad(w))_K + <Fhat, w>_dK
//     = -(S(U) div(B), w)_K
//       - <S(U-)(Bhat - B-) dot n_K, w>_dK.
//
// Fhat is the numerical MHD flux, n_K is the outward normal, and the normal
// trace Bhat dot n_K is the arithmetic average of the adjacent traces.
// GPsourceFormIntegrator assembles both source contributions with their
// negative signs. The face terms use each element's own state and normal.
//
// DG_MHD_GPsource::Mult evaluates the explicit time derivative
//
//                  U -> M^{-1}(R_flux(U) + R_GP(U)),
//
// where R_flux is the weak volume flux minus the numerical face flux, and
// R_GP contains the negative volume and face Godunov-Powell source terms.
// R_GP is included only when use_GPsource is true (-gp in ex42.cpp).
// The block-diagonal DG mass matrix M is inverted element by element.
// HyperbolicFormIntegrator assembles the flux residual and records the
// maximum characteristic speed used in the CFL timestep estimate.
// With use_LDF enabled, the magnetic time derivative is L2-projected onto
// the local divergence-free subspace after application of the mass inverse.
//

#include <functional>
#include "mfem.hpp"

namespace mfem
{

/// @brief Godunov-Powell source term
class GPsource
{
public:
   const int num_equations;
   const int dim;

   /// @brief Constructor for the Godunov-Powell source term
   GPsource(const int dim)
      : num_equations(2*dim+2), dim(dim) { }

   /// @brief Destructor for the Godunov-Powell source term
   ~GPsource() {}

   /**
   * @brief Compute source -S(U)*divB in cell interiors.
   *
   * Used in GPsourceFormIntegrator::AssembleElementVector() for evaluation
   * of (-S(U)*divB, w). Here S(U) = (0, B, v, v dot B)^T
   * for the state ordering U = (rho, rho*v, B, E)^T.
   * @param[in] state state at the current integration point (num_equations)
   * @param[in] divB divergence of the magnetic field at the current integration point (scalar)
   * @param[in] Tr element transformation
   * @param[out] source source from the given element at the current integration point (num_equations)
   *
   * @note One can put assertion in here to detect non-physical solution
   */
   void ComputeSource(const Vector &state, const real_t &divB,
                      ElementTransformation &Tr, Vector &source) const
   {
      source.SetSize(num_equations);
      source(0) = 0.0;
      for (auto i=0; i<dim; i++) { source(i+1) = -state(i+1+dim) * divB; }
      for (auto i=0; i<dim; i++) { source(i+1+dim) = -state(i+1) / state(0) * divB; }
      source(1+2*dim) = 0.0;
      for (auto i=0; i<dim; i++) { source(1+2*dim) -= (state(i+1) / state(0)) * state(i+1+dim); }
      source(1+2*dim) *= divB;
      return;
   }
};

class NumericalGPsource
{
protected:
   const GPsource &gpSource;
public:
   NumericalGPsource(const GPsource &gpSource)
      : gpSource(gpSource) { }

   virtual ~NumericalGPsource() = default;

   /**
   * @brief Evaluates normal numerical source for the given states and normal.
   * Must be implemented in a derived class.
   *
   * Used in GPsourceFormIntegrator::AssembleFaceVector() for evaluation of
   * the signed source -S(U^-)*((Bhat - B^-) dot n^-) tested by w^-.
   * @param[in] state1 state value at a point from the first element
   * (num_equations)
   * @param[in] state2 state value at a point from the second element
   * (num_equations)
   * @param[in] nor scaled normal vector, see mfem::CalcOrtho() (dim)
   * @param[in] Tr face transformation
   * @param[out] source1 signed source on the first side (num_equations)
   * @param[out] source2 signed source on the second side (num_equations)
   */
   virtual void Eval(const Vector &state1, const Vector &state2, const Vector &nor,
                     FaceElementTransformations &Tr, Vector &source1, Vector &source2) const = 0;

   /**
   * @brief Evaluates Jacobian of the normal numerical source for the given
   * states and normal. OPTIONALLY overloaded in a derived class.
   *
   * Intended for GPsourceFormIntegrator::AssembleFaceGrad() for the Jacobian
   * of the signed source -S(U^-)*((Bhat - B^-) dot n^-) at the face.
   * @param[in] side indicates gradient w.r.t. the first (side = 1)
   * or second (side = 2) state
   * @param[in] state1 state value of the beginning of the interval
   * (num_equations)
   * @param[in] state2 state value of the end of the interval
   * (num_equations)
   * @param[in] nor scaled normal vector, see mfem::CalcOrtho() (dim)
   * @param[in] Tr face transformation
   * @param[out] grad Jacobian of the numerical source with respect to a state
   * (num_equations, num_equations); not implemented here
   */
   virtual void Grad(int side, const Vector &state1, const Vector &state2,
                     const Vector &nor, FaceElementTransformations &Tr,
                     DenseMatrix &grad) const
   { MFEM_ABORT("Not implemented."); }

   /// @brief Get source function
   /// @return constant reference to the source function.
   const GPsource &GetSourceFunction() const { return gpSource; }

};

class RusanovGPsource : public NumericalGPsource
{
public:
   RusanovGPsource(const GPsource &gpSource)
      : NumericalGPsource(gpSource) { }

   // at a face quadrature node with normal vector "nor" (not a unit vector) pointing from el1 to el2,
   // compute source1 that is to be tested by local test function from el1, directly (coefficient is +1)
   // and source2 that is to be tested by local test function from el2, directly (coefficient is also +1)
   void Eval(const Vector &state1, const Vector &state2, const Vector &nor,
             FaceElementTransformations &Tr, Vector &source1, Vector &source2) const override
   {
      const auto dim = gpSource.dim;
      // nor points from el1 to el2, when they are obtained through "GetFaceElementTransformations"
      // inner product with "nor" (nor is not a unit vector, just use a form consistent with "S(U) B dot n")
      real_t B1n = 0.0;
      real_t B2n = 0.0;
      for (auto i=0; i<dim; i++) { B1n += state1(i+1+dim) * nor(i); }
      for (auto i=0; i<dim; i++) { B2n += state2(i+1+dim) * nor(i); }

      // Averaged normal magnetic trace for the Rusanov Godunov-Powell correction
      const real_t Bhatn = 0.5 * (B1n + B2n);
      const real_t Bjump1 = Bhatn - B1n;
      const real_t Bjump2 = B2n -
                            Bhatn; // because of the negative sign of nor from el2

      source1.SetSize(gpSource.num_equations);
      source2.SetSize(gpSource.num_equations);
      source1(0) = 0.0;
      source2(0) = 0.0;
      for (auto i=0; i<dim; i++) { source1(i+1) = -state1(i+1+dim) * Bjump1; }
      for (auto i=0; i<dim; i++) { source2(i+1) = -state2(i+1+dim) * Bjump2; }
      for (auto i=0; i<dim; i++) { source1(i+1+dim) = -state1(i+1) / state1(0) * Bjump1; }
      for (auto i=0; i<dim; i++) { source2(i+1+dim) = -state2(i+1) / state2(0) * Bjump2; }
      source1(1+2*dim) = 0.0;
      source2(1+2*dim) = 0.0;
      for (auto i=0; i<dim; i++) { source1(1+2*dim) -= (state1(i+1) / state1(0)) * state1(i+1+dim); }
      for (auto i=0; i<dim; i++) { source2(1+2*dim) -= (state2(i+1) / state2(0)) * state2(i+1+dim); }
      source1(1+2*dim) *= Bjump1;
      source2(1+2*dim) *= Bjump2;
   }

   void Grad(int side, const Vector &state1, const Vector &state2,
             const Vector &nor, FaceElementTransformations &Tr,
             DenseMatrix &grad) const override
   {MFEM_ABORT("Not implemented.");}
};
// We can also make other "numerical sources", such as the HLL version (recall Kailiang Wu's paper on PP MHD DG schemes).

// similar to HyperbolicFormIntegrator
class GPsourceFormIntegrator : public NonlinearFormIntegrator
{
private:
   const NumericalGPsource &numGPsource;
   const GPsource &gpsource;
   const int IntOrderOffset; // integration order offset, 2*p + IntOrderOffset.
   const real_t sign;
#ifndef MFEM_THREAD_SAFE
   // Local storage for element integration
   Vector shape;              // shape function value at an integration point
   Vector state;              // state value at an integration point
   Vector source;             // GP source value at an integration point
   DenseTensor J;             // Jacobian matrix at an integration point
   DenseMatrix
   dshape;        // derivative of shape function at an integration point

   Vector shape1;  // shape function value at an integration point - first elem
   Vector shape2;  // shape function value at an integration point - second elem
   Vector state1;  // state value at an integration point - first elem
   Vector state2;  // state value at an integration point - second elem
   Vector nor;     // normal vector, see mfem::CalcOrtho()
   Vector source1; // source vector at the first element
   Vector source2; // source vector at the second element
   DenseMatrix JDotN;   // Reserved local Jacobian workspace
#endif

public:
   const int num_equations;  // the number of equations
   /**
    * @brief Construct a new GPsource Integrator object
    *
    * @param[in] numGPsource_ numerical GP source
    * @param[in] IntOrderOffset_ integration order offset
    * @param[in] sign_ multiplier for the volume and face source contributions
    */
   GPsourceFormIntegrator(
      const NumericalGPsource &numGPsource_,
      const int IntOrderOffset_ = 0,
      const real_t sign_ = 1.):
      NonlinearFormIntegrator(),
      numGPsource(numGPsource_),
      gpsource(numGPsource_.GetSourceFunction()),
      IntOrderOffset(IntOrderOffset_),
      sign(sign_),
      num_equations(gpsource.num_equations)
   {
#ifndef MFEM_THREAD_SAFE
      state.SetSize(num_equations);
      source.SetSize(num_equations);
      state1.SetSize(num_equations);
      state2.SetSize(num_equations);
      source1.SetSize(num_equations);
      source2.SetSize(num_equations);
      JDotN.SetSize(num_equations);
      nor.SetSize(gpsource.dim);
#endif
   }

   const GPsource &GetGPsource() { return gpsource; }

   /**
    * @brief Assemble (-S(U)*divB, w) using the source computed by
    * GPsource::ComputeSource()
    *
    * @param[in] el local finite element
    * @param[in] Tr element transformation
    * @param[in] elfun local coefficient of basis
    * @param[out] elvect evaluated dual vector
    */
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      // current element's the number of degrees of freedom
      // does not consider the number of equations
      const auto dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
      // Local storage for element integration

      // shape function value at an integration point
      Vector shape(dof);
      // derivative of shape function at an integration point
      DenseMatrix dshape(dof, Tr.GetSpaceDim());
      // divergence of vector shape function (ByNode) at an integration point
      // state value at an integration point
      Vector state(num_equations);
      // GP source value at an integration point
      Vector source(num_equations);
#else
      // resize shape and gradient shape storage
      shape.SetSize(dof);
      dshape.SetSize(dof, Tr.GetSpaceDim());
#endif

      // setDegree-up output vector
      elvect.SetSize(dof * num_equations);
      elvect = 0.0;

      // make state variable and output dual vector matrix form.
      const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
      DenseMatrix elvect_mat(elvect.GetData(), dof,
                             num_equations); // make a view (not a copy)

      // obtain integration rule. If integration is rule is given, then use it.
      // Otherwise, get (2*p + IntOrderOffset) order integration rule
      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         const auto order = el.GetOrder()*2 + IntOrderOffset;
         ir = &IntRules.Get(Tr.GetGeometryType(), order);
      }

      // loop over integration points
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetIntPoint(&ip);

         el.CalcShape(ip, shape);
         el.CalcPhysDShape(Tr, dshape);

         // compute current state value with given shape function values
         elfun_mat.MultTranspose(shape, state);

         // compute current divB value with given shape function gradient values
         const real_t divB = Vector(dshape.GetData(),
                                    dof*Tr.GetSpaceDim()) * Vector(elfun.GetData() + (1+Tr.GetSpaceDim())*dof,
                                                                   dof*Tr.GetSpaceDim());

         // compute source term value
         gpsource.ComputeSource(state, divB, Tr, source);
         // integrate (source, v) at the current quadrature point, and contribute to the final output
         AddMult_a_VWt(ip.weight * Tr.Weight() * sign, shape, source, elvect_mat);
      }
   }

   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &grad) override
   {MFEM_ABORT("Not implemented.");}

   /**
    * @brief Assemble the two element-side Godunov-Powell face corrections
    * -<S(U_side)*((Bhat - B_side) dot n_side), w_side>.
    * NumericalGPsource::Eval() supplies both signed source vectors.
    *
    * @param[in] el1 finite element of the first element
    * @param[in] el2 finite element of the second element
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis from both elements
    * @param[out] elvect evaluated dual vector <source, v>
    */
   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override
   {
      // current elements' the number of degrees of freedom does not consider the number of equations
      const int dof1 = el1.GetDof();
      const int dof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
      // Local storage for element integration

      // shape function value at an integration point - first elem
      Vector shape1(dof1);
      // shape function value at an integration point - second elem
      Vector shape2(dof2);
      // normal vector (usually not a unit vector)
      Vector nor(Tr.GetSpaceDim());
      // state value at an integration point - first elem
      Vector state1(num_equations);
      // state value at an integration point - second elem
      Vector state2(num_equations);
      // numerical source on the faces
      Vector source1(num_equations);
      Vector source2(num_equations);
#else
      shape1.SetSize(dof1);
      shape2.SetSize(dof2);
#endif

      // overall DOFs for two elements
      elvect.SetSize((dof1 + dof2) * num_equations);
      elvect = 0.0;

      // setup views
      const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
      const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
                                   num_equations);
      DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
      DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2,
                              num_equations);

      // Obtain integration rule. If integration is rule is given, then use it.
      // Otherwise, get (2*p + IntOrderOffset) order integration rule
      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         const int order = 2*std::max(el1.GetOrder(), el2.GetOrder()) + IntOrderOffset;
         ir = &IntRules.Get(Tr.GetGeometryType(), order);
      }
      // loop over integration points
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Tr.SetAllIntPoints(&ip); // set face and element int. points

         // Calculate basis functions on both elements at the face
         el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
         el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

         // Interpolate elfun at the point
         elfun1_mat.MultTranspose(shape1, state1);
         elfun2_mat.MultTranspose(shape2, state2);

         // Get the scaled normal vector on the face
         if (nor.Size() == 1)  // if 1D, use 1 or -1.
         {
            // This assume the 1D integration point is in (0,1). This may not work
            // if this changes.
            nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
         }
         else
         {
            CalcOrtho(Tr.Jacobian(), nor);
         }
         // Compute source1 and source2 using evaluated quantities
         numGPsource.Eval(state1, state2, nor, Tr, source1, source2);
         // We have adapted it so that only plus sign is needed below.

         // Weight the signed source contributions on both sides of the face
         AddMult_a_VWt(ip.weight*sign, shape1, source1, elvect1_mat);
         AddMult_a_VWt(ip.weight*sign, shape2, source2, elvect2_mat);
      }
   }

   void AssembleFaceGrad(const FiniteElement &el1,
                         const FiniteElement &el2,
                         FaceElementTransformations &Tr,
                         const Vector &elfun, DenseMatrix &elmat) override
   {MFEM_ABORT("Not implemented.");}
};



/// @brief Time dependent DG operator for MHD equation with GP source term
class DG_MHD_GPsource : public TimeDependentOperator
{
private:
   const bool use_GPsource;
   const bool use_LDF;

   const int num_equations; // the number of equations
   const int dim;
   FiniteElementSpace &vfes; // vector finite element space (ByNode)

   // Element integration form. Should contain ComputeFlux
   // only for setting up "nonlinearForm" and calling "GetMaxCharSpeed" (max speed obtained within computation)
   std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int = nullptr;

   std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int = nullptr;

   // Base Nonlinear Form
   std::unique_ptr<NonlinearForm> nonlinearForm;

   // element-wise inverse mass matrix
   std::vector<DenseMatrix> invmass; // local scalar inverse mass
   std::vector<DenseMatrix>
   ldf_projectors; // local magnetic L2 projectors to locally divergence-free subspace

   // global maximum characteristic speed. Updated by form integrators
   mutable real_t max_char_speed;

   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();
   void ComputeLDFProjectors();

public:
   /**
    * @brief Construct a new DG_MHD_GPsource object
    *
    * @param vfes_ DG vector finite element space with Ordering::byNODES
    * @param hyperb_form_Int_ integrator for the volume and numerical face fluxes
    * @param gpsource_form_Int_ integrator for the volume and face GP sources
    * @param use_GPsource_ whether to include the Godunov-Powell sources
    * @param use_LDF_ whether to project magnetic updates to the LDF space
    */
   DG_MHD_GPsource(
      FiniteElementSpace &vfes_,
      std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int_,
      std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int_,
      const bool use_GPsource_, const bool use_LDF_ = false);
   /** @brief Project the magnetic coefficients of a state or time derivative
       onto the element-wise LDF space, leaving the other fields unchanged.
       Requires use_LDF_ = true and scalar VALUE-mapped DG elements on affine
       2D triangles or bilinear quadrilaterals (no high-order curved geometry).
       This preserves magnetic cell averages but allows normal face jumps.
       @param state full DG state or time derivative, modified in place */
   void ApplyLDF(Vector &state) const;
   /** @brief Return the local magnetic L2 projector for inspection.
       @param element local element index */
   const DenseMatrix &GetLDFProjector(int element) const
   {
      MFEM_VERIFY(use_LDF && element >= 0 &&
                  element < static_cast<int>(ldf_projectors.size()),
                  "An enabled LDF projector and a valid element are required.");
      return ldf_projectors[element];
   }
   /**
    * @brief Compute M^{-1}(R_flux(x) + R_GP(x)) for explicit time stepping.
    * R_GP is included only when use_GPsource is true.
    * When use_LDF is true, the magnetic time derivative is L2-projected
    * onto the local divergence-free space.
    *
    * @param x current solution vector
    * @param y resulting time derivative vector for an explicit ODE solver
    */
   void Mult(const Vector &x, Vector &y) const override;
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   real_t GetMaxCharSpeed() { return max_char_speed; }
   void Update();

};

//////////////////////////////////////////////////////////////////
///        MHD with Godunov-Powell term IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DG_MHD_GPsource
DG_MHD_GPsource::DG_MHD_GPsource(
   FiniteElementSpace &vfes_,
   std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int_,
   std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int_,
   const bool use_GPsource_, const bool use_LDF_)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     num_equations(hyperb_form_Int_->num_equations),
     dim(vfes_.GetMesh()->SpaceDimension()),
     vfes(vfes_),
     hyperb_form_Int(std::move(hyperb_form_Int_)),
     gpsource_form_Int(std::move(gpsource_form_Int_)),
     use_GPsource(use_GPsource_),
     use_LDF(use_LDF_),
     z(vfes_.GetTrueVSize())
{
   // Assemble and invert the scalar mass matrix on each element for all fields.
   ComputeInvMass();
   if (use_LDF) { ComputeLDFProjectors(); }

#ifndef MFEM_USE_MPI
   nonlinearForm.reset(new NonlinearForm(&vfes));
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(&vfes);
   if (pvfes)
   {
      nonlinearForm.reset(new ParNonlinearForm(pvfes));
   }
   else
   {
      nonlinearForm.reset(new NonlinearForm(&vfes));
   }
#endif
   /** For nonlinear operators, the "matrix" assembly levels usually do not make
       sense, so only PARTIAL and NONE (matrix-free) are supported. */

   // Add the integrators to the nonlinear form
   nonlinearForm->AddDomainIntegrator(hyperb_form_Int.get());
   nonlinearForm->AddInteriorFaceIntegrator(
      hyperb_form_Int.get()); // only interior faces

   if (use_GPsource)
   {
      // Add the Godunov-Powell source term
      nonlinearForm->AddDomainIntegrator(gpsource_form_Int.get());
      nonlinearForm->AddInteriorFaceIntegrator(
         gpsource_form_Int.get()); // only interior faces
   }

   nonlinearForm->UseExternalIntegrators(); // indicate that integrators are not owned by the NonlinearForm

}

void DG_MHD_GPsource::ComputeInvMass()
{
   InverseIntegrator inv_mass(new MassIntegrator());

   invmass.resize(vfes.GetNE());
   for (auto i=0; i<vfes.GetNE(); i++)
   {
      const int dof = vfes.GetFE(i)->GetDof();
      invmass[i].SetSize(dof);
      inv_mass.AssembleElementMatrix(*vfes.GetFE(i),
                                     *vfes.GetElementTransformation(i),
                                     invmass[i]);
   }
}


void DG_MHD_GPsource::ComputeLDFProjectors()
{
   MFEM_VERIFY(dim == 2 && vfes.GetMesh()->Dimension() == 2,
               "LDF projection requires a two-dimensional mesh.");
   MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES &&
               vfes.GetVDim() == num_equations &&
               vfes.FEColl()->GetContType() == FiniteElementCollection::DISCONTINUOUS,
               "LDF projection requires the component-wise DG state space.");
   ldf_projectors.resize(vfes.GetNE());
   for (int k = 0; k < vfes.GetNE(); k++)
   {
      const FiniteElement &fe = *vfes.GetFE(k);
      ElementTransformation &Tr = *vfes.GetElementTransformation(k);
      const bool triangle = fe.GetGeomType() == Geometry::TRIANGLE;
      MFEM_VERIFY((triangle || fe.GetGeomType() == Geometry::SQUARE) &&
                  Tr.Order() == 1 &&
                  fe.GetRangeType() == FiniteElement::SCALAR &&
                  fe.GetMapType() == FiniteElement::VALUE &&
                  fe.Space() == (triangle ? FunctionSpace::Pk : FunctionSpace::Qk),
                  "LDF projection requires affine triangles or bilinear quads"
                  " with scalar VALUE-mapped DG elements.");
      const int dof = fe.GetDof();
      const int p = fe.GetOrder();
      DenseMatrix &P = ldf_projectors[k];
      P.SetSize(2*dof);
      P = 0.0;
      for (int j = 0; j < 2*dof; j++) { P(j,j) = 1.0; }
      if (p == 0) { continue; } // Cell-wise constants are already LDF.

      // On triangles div(B) is in P^(p-1). On bilinear quads det(J)*div(B)
      // is in Q^p with the highest tensor monomial absent. Test against a
      // complete basis of that numerator space; det(J) cancels in the integral.
      // In particular, Q^(p-1) alone would miss some quadrilateral constraints.
      L2_FECollection constraint_fec(triangle ? p-1 : p, dim);
      const FiniteElement &qfe =
         *constraint_fec.FiniteElementForGeometry(fe.GetGeomType());
      const int constraints = triangle ? qfe.GetDof() : (p+1)*(p+1)-1;
      DenseMatrix D(constraints, 2*dof);
      D = 0.0;
      Vector qshape(constraints);
      Vector legendre_x(p+1), legendre_y(p+1);
      DenseMatrix dshape(dof, dim);
      const IntegrationRule &ir =
         IntRules.Get(fe.GetGeomType(), 2*p + Tr.OrderW());
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         if (triangle) { qfe.CalcShape(ip, qshape); }
         else
         {
            Poly_1D::CalcLegendre(p, ip.x, legendre_x.GetData());
            Poly_1D::CalcLegendre(p, ip.y, legendre_y.GetData());
            int a = 0;
            for (int j = 0; j <= p; j++)
            {
               for (int i = 0; i <= p; i++)
               {
                  if (i == p && j == p) { continue; }
                  qshape(a++) = legendre_x(i) * legendre_y(j);
               }
            }
         }
         fe.CalcPhysDShape(Tr, dshape);
         const real_t weight = ip.weight * Tr.Weight();
         for (int a = 0; a < constraints; a++)
         {
            for (int j = 0; j < dof; j++)
            {
               D(a,j) += weight * qshape(a) * dshape(j,0);
               D(a,dof+j) += weight * qshape(a) * dshape(j,1);
            }
         }
      }

      // H = diag(M,M), A = H^{-1} D^T, S = D A.
      // P = I - A S^{-1} D is the H-orthogonal projector onto ker(D).
      DenseMatrix A(2*dof, constraints);
      A = 0.0;
      for (int c = 0; c < dim; c++)
      {
         for (int j = 0; j < dof; j++)
         {
            for (int a = 0; a < constraints; a++)
            {
               for (int l = 0; l < dof; l++)
               {
                  A(c*dof+j,a) += invmass[k](j,l) * D(a,c*dof+l);
               }
            }
         }
      }
      DenseMatrix S(constraints), SinvD(constraints, 2*dof);
      mfem::Mult(D, A, S);
      DenseMatrixInverse schur_inverse(S, true);
      schur_inverse.Mult(D, SinvD);
      DenseMatrix correction(2*dof);
      mfem::Mult(A, SinvD, correction);
      P -= correction;
   }
}

void DG_MHD_GPsource::ApplyLDF(Vector &state) const
{
   MFEM_VERIFY(use_LDF && state.Size() == vfes.GetVSize(),
               "ApplyLDF requires enabled LDF projection and a full DG vector.");
   Vector local_state, magnetic, projected;
   Array<int> vdofs;
   for (int k = 0; k < vfes.GetNE(); k++)
   {
      const int dof = vfes.GetFE(k)->GetDof();
      vfes.GetElementVDofs(k, vdofs);
      state.GetSubVector(vdofs, local_state);
      // Local fields are contiguous: (rho, momentum, magnetic field, E).
      magnetic.MakeRef(local_state, (1+dim)*dof, dim*dof);
      projected.SetSize(dim*dof);
      ldf_projectors[k].Mult(magnetic, projected);
      magnetic = projected;
      state.SetSubVector(vdofs, local_state);
   }
}


void DG_MHD_GPsource::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   hyperb_form_Int->ResetMaxCharSpeed();

   // 1. Apply Nonlinear form to obtain an auxiliary result
   nonlinearForm->Mult(x, z);

   // 2. Apply Linear form
   // Apply block inverse mass
   Vector zval; // z_loc, dof*num_eq
   DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
   DenseMatrix current_ymat; // view of element result, dof x num_eq
   Array<int> vdofs;
   for (auto i=0; i<vfes.GetNE(); i++)
   {
      const auto dof = vfes.GetFE(i)->GetDof();
      // ALWAYS returns ByNodes ordering (like separate fields)
      vfes.GetElementVDofs(i, vdofs);
      // copy from the global vector to the local vector
      z.GetSubVector(vdofs, zval);
      // each column is a field and each row is a DOF
      current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
      current_ymat.SetSize(dof, num_equations);
      // invmass left multiplies the auxiliary result
      mfem::Mult(invmass[i], current_zmat, current_ymat);
      // return the result to the global vector
      y.SetSubVector(vdofs, current_ymat.GetData());
   }
   // Project the time derivative after applying the mass inverse. With an
   // LDF initial state on a fixed mesh, explicit RK stages remain LDF.
   if (use_LDF) { ApplyLDF(y); }
   max_char_speed = hyperb_form_Int->GetMaxCharSpeed();
}

void DG_MHD_GPsource::Update()
{
   nonlinearForm->Update();
   height = nonlinearForm->Height();
   width = height;
   z.SetSize(height);

   ComputeInvMass();
   if (use_LDF) { ComputeLDFProjectors(); }
}


// Default model mesh: triangular periodic unit square [0,1]^2.
Mesh MHDMesh(const int problem)
{
   return Mesh("../data/periodic-square-tri.msh");
}

// Problem 1: two-dimensional Orszag-Tang vortex on the periodic unit square.
// rho = 25/(36*pi), p = 5/(12*pi),
// v = (-sin(2*pi*y), sin(2*pi*x)),
// B = (-sin(2*pi*y), sin(4*pi*x))/sqrt(4*pi).
// The initial magnetic field is divergence-free. E includes internal,
// kinetic, and magnetic energy; the six-component state is
// (rho, rho*v_x, rho*v_y, B_x, B_y, E).
// These are initial data for nonlinear MHD evolution, not an exact solution.
VectorFunctionCoefficient MHDInitialCondition(const int problem,
                                              real_t specific_heat_ratio)
{
   switch (problem)
   {
      case 1: // Orszag-Tang vortex
         return VectorFunctionCoefficient(6, [specific_heat_ratio](const Vector &x,
                                                                   Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            const real_t density = 25.0/(36.0*M_PI);
            const real_t velocity_x = -std::sin(2*M_PI*x(1));
            const real_t velocity_y = std::sin(2*M_PI*x(0));
            const real_t magnetic_x = -std::sin(2*M_PI*x(1))/std::sqrt(4*M_PI);
            const real_t magnetic_y = std::sin(4*M_PI*x(0))/std::sqrt(4*M_PI);
            const real_t pressure = 5.0/(12.0*M_PI);
            const real_t energy =
               pressure / (specific_heat_ratio - 1.0) +
               0.5 * (density * (velocity_x * velocity_x + velocity_y * velocity_y) +
                      (magnetic_x*magnetic_x + magnetic_y*magnetic_y));

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = magnetic_x;
            y(4) = magnetic_y;
            y(5) = energy;
         });
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

} // namespace mfem
