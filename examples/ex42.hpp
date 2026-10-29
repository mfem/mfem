//               MFEM Example 42 - MHD with Godunov-Powell source
//                      (Time-dependent DG operator and model problem)
//
// This code defines the Orszag-Tang and Alfven wave problems and DG operator for
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
// LDF_DG_GP_MHD::Mult evaluates the explicit time derivative
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
// dim is the mesh/physical dimension, while MHDdim counts velocity and magnetic
// components: 1 <= dim <= MHDdim <= 3. Fields are stored component by component
// as (rho, rho*v[0:MHDdim], B[0:MHDdim], E). Divergence and face normals use
// only the first dim magnetic components; magnetic energy and v dot B use all
// MHDdim components. No exterior boundary source or flux is supplied here.
//

#include <functional>
#include <limits>
#include "mfem.hpp"

namespace mfem
{

/// ex42 quadrature policy: degree 2*k in cells, 2*k+1 on faces by default.
/// A caller-supplied integration rule takes precedence. This specialization
/// leaves the common HyperbolicFormIntegrator defaults unchanged.
class MHDFormIntegrator : public HyperbolicFormIntegrator
{
private:
   const int integration_offset;

public:
   MHDFormIntegrator(const NumericalFlux &flux, int offset = 0)
      : HyperbolicFormIntegrator(flux, offset), integration_offset(offset) { }

   void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &state, Vector &rhs) override
   {
      const IntegrationRule *saved = IntRule;
      if (!saved)
      {
         const int degree = std::max(el1.GetOrder(), el2.GetOrder());
         IntRule = &IntRules.Get(Tr.GetGeometryType(),
                                 2*degree+1+integration_offset);
      }
      HyperbolicFormIntegrator::AssembleFaceVector(el1, el2, Tr, state, rhs);
      IntRule = saved;
   }
};

/// Godunov-Powell volume source -div(B)*(0, B, v, v dot B).
class GPsource
{
public:
   const int num_equations; ///< State size, 2*MHDdim + 2.
   const int dim;           ///< Spatial dimension for derivatives and normals.
   const int MHDdim;        ///< Velocity/magnetic components, at least dim.

   /// Construct the source; mhd_dim=-1 uses dim vector components.
   GPsource(const int dim, const int mhd_dim = -1)
      : num_equations(2*(mhd_dim < 0 ? dim : mhd_dim)+2), dim(dim),
        MHDdim(mhd_dim < 0 ? dim : mhd_dim)
   {
      MFEM_VERIFY(1 <= dim && dim <= MHDdim && MHDdim <= 3,
                  "Require 1 <= mesh dimension <= MHDdim <= 3.");
   }

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
   * @param[in] Tr element transformation (unused by this state-only source)
   * @param[out] source source from the given element at the current integration point (num_equations)
   */
   void ComputeSource(const Vector &state, const real_t &divB,
                      ElementTransformation &Tr, Vector &source) const
   {
      source.SetSize(num_equations);
      source(0) = 0.0;
      for (auto i=0; i<MHDdim; i++) { source(i+1) = -state(i+1+MHDdim) * divB; }
      for (auto i=0; i<MHDdim; i++) { source(i+1+MHDdim) = -state(i+1) / state(0) * divB; }
      source(1+2*MHDdim) = 0.0;
      for (auto i=0; i<MHDdim; i++) { source(1+2*MHDdim) -= (state(i+1) / state(0)) * state(i+1+MHDdim); }
      source(1+2*MHDdim) *= divB;
      return;
   }
};

/// Interface for the two element-side, nonconservative GP face corrections.
/// The referenced volume source must outlive this object.
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
   * @param[in] state1 state value on the first side
   * (num_equations)
   * @param[in] state2 state value on the second side
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

/// GP face correction using the arithmetic magnetic trace with Rusanov flux.
/// Unlike a conservative numerical flux, it returns a separate signed source
/// for each side, evaluated using that side's own S(U).
class RusanovGPsource : public NumericalGPsource
{
public:
   RusanovGPsource(const GPsource &gpSource)
      : NumericalGPsource(gpSource) { }

   // nor points from element 1 to element 2 and includes the face measure.
   // Return both signed RHS terms; the assembler adds each with a plus sign.
   void Eval(const Vector &state1, const Vector &state2, const Vector &nor,
             FaceElementTransformations &Tr, Vector &source1, Vector &source2) const override
   {
      const auto MHDdim = gpSource.MHDdim;
      // Only spatial components enter B dot nor, even in 1D/2D three-component MHD.
      real_t B1n = 0.0;
      real_t B2n = 0.0;
      for (auto i=0; i<gpSource.dim; i++) { B1n += state1(i+1+MHDdim) * nor(i); }
      for (auto i=0; i<gpSource.dim; i++) { B2n += state2(i+1+MHDdim) * nor(i); }

      // Averaged normal magnetic trace for the Rusanov Godunov-Powell correction
      const real_t Bhatn = 0.5 * (B1n + B2n);
      const real_t Bjump1 = Bhatn - B1n;
      // (Bhat-B2) dot n2 = (B2-Bhat) dot nor because n2=-nor.
      const real_t Bjump2 = B2n - Bhatn;

      source1.SetSize(gpSource.num_equations);
      source2.SetSize(gpSource.num_equations);
      source1(0) = 0.0;
      source2(0) = 0.0;
      for (auto i=0; i<MHDdim; i++) { source1(i+1) = -state1(i+1+MHDdim) * Bjump1; }
      for (auto i=0; i<MHDdim; i++) { source2(i+1) = -state2(i+1+MHDdim) * Bjump2; }
      for (auto i=0; i<MHDdim; i++) { source1(i+1+MHDdim) = -state1(i+1) / state1(0) * Bjump1; }
      for (auto i=0; i<MHDdim; i++) { source2(i+1+MHDdim) = -state2(i+1) / state2(0) * Bjump2; }
      source1(1+2*MHDdim) = 0.0;
      source2(1+2*MHDdim) = 0.0;
      for (auto i=0; i<MHDdim; i++) { source1(1+2*MHDdim) -= (state1(i+1) / state1(0)) * state1(i+1+MHDdim); }
      for (auto i=0; i<MHDdim; i++) { source2(1+2*MHDdim) -= (state2(i+1) / state2(0)) * state2(i+1+MHDdim); }
      source1(1+2*MHDdim) *= Bjump1;
      source2(1+2*MHDdim) *= Bjump2;
   }

   void Grad(int side, const Vector &state1, const Vector &state2,
             const Vector &nor, FaceElementTransformations &Tr,
             DenseMatrix &grad) const override
   {MFEM_ABORT("Not implemented.");}
};
// Other interface traces require matching side corrections; this arithmetic
// choice alone does not implement the complete positivity-preserving method.

/// Assemble the already signed GP volume and interior-face sources.
/// The numerical source must outlive the integrator. Jacobian assembly is not
/// implemented, so this example uses explicit time integration.
class GPsourceFormIntegrator : public NonlinearFormIntegrator
{
private:
   const NumericalGPsource &numGPsource;
   const GPsource &gpsource;
   // Defaults: cell order 2*p, face order 2*max(p_left,p_right)+1.
   const int IntOrderOffset;
   const real_t sign;
#ifndef MFEM_THREAD_SAFE
   // Local storage for element integration
   Vector shape;              // shape function value at an integration point
   Vector state;              // state value at an integration point
   Vector source;             // GP source value at an integration point
   DenseMatrix
   dshape;        // derivative of shape function at an integration point

   Vector shape1;  // shape function value at an integration point - first elem
   Vector shape2;  // shape function value at an integration point - second elem
   Vector state1;  // state value at an integration point - first elem
   Vector state2;  // state value at an integration point - second elem
   Vector nor;     // normal vector, see mfem::CalcOrtho()
   Vector source1; // source vector at the first element
   Vector source2; // source vector at the second element
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
      nor.SetSize(gpsource.dim);
#endif
   }

   const GPsource &GetGPsource() const { return gpsource; }

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
      // Scalar element DOFs; the local state contains num_equations blocks.
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

      // Allocate the local RHS in component-block order.
      elvect.SetSize(dof * num_equations);
      elvect = 0.0;

      // Non-owning, column-major views: each column holds all scalar DOF
      // coefficients for one state component (rho, rho*v, B, E).
      const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
      DenseMatrix elvect_mat(elvect.GetData(), dof,
                             num_equations); // make a view (not a copy)

      // obtain integration rule. If integration is rule is given, then use it.
      // Otherwise, use cell precision 2*p+IntOrderOffset.
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

         // At this quadrature point, dshape(j,d) = d phi_j / d x_d uses
         // physical coordinates, including the inverse-Jacobian transform.
         // Its column-major data, viewed as a Vector without copying, is
         // [d_x phi_0, ..., d_x phi_{dof-1}, d_y phi_0, ...].
         // Skip rho and MHDdim momentum blocks to reach the magnetic DOFs:
         // [B_x,0, ..., B_x,dof-1, B_y,0, ...]. Each block has dof entries.
         // Vector * Vector takes their dot product, giving the cell-interior
         // div(B_h) = sum_d sum_j B_{d,j} * (d phi_j / d x_d).
         // Only GetSpaceDim() directions contribute; e.g. in 2D, a stored
         // B_z does not contribute. DG jumps are handled by the face terms.
         const real_t divB = Vector(dshape.GetData(),
                                    dof*Tr.GetSpaceDim()) * Vector(elfun.GetData() + (1+gpsource.MHDdim)*dof,
                                                                   dof*Tr.GetSpaceDim());

         // compute source term value
         gpsource.ComputeSource(state, divB, Tr, source);
         // Physical volume weight includes the element Jacobian determinant.
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

      // elfun concatenates all coefficients of element 1, then element 2:
      // [element 1: dof1*num_equations | element 2: dof2*num_equations].
      // These non-owning views do not copy data. Each column holds one
      // equation component, and each row corresponds to a scalar basis DOF.
      // The second view skips the entire first element's coefficient block;
      // its row count can differ if the two elements have different DOF counts.
      const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
      const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
                                   num_equations);
      // The output residual uses the same two-element/component layout.
      DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
      DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2,
                              num_equations);

      // Obtain integration rule. If integration is rule is given, then use it.
      // Otherwise, use face precision 2*max(p_left,p_right)+1+IntOrderOffset.
      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         const int order = 2*std::max(el1.GetOrder(),
                                      el2.GetOrder()) + 1 + IntOrderOffset;
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

         // Evaluate U_side,k(x_q) = sum_j phi_side,j(x_q)*u_side,j,k.
         // Both states refer to the same face point, evaluated from each
         // element's full coefficient block; DG permits state1 != state2.
         elfun1_mat.MultTranspose(shape1, state1);
         elfun2_mat.MultTranspose(shape2, state2);

         // Get the scaled normal vector on the face
         if (nor.Size() == 1)  // if 1D, use 1 or -1.
         {
            // MFEM's reference segment is [0,1]; endpoints have normals -1,+1.
            nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
         }
         else
         {
            CalcOrtho(Tr.Jacobian(), nor);
         }
         // Compute source1 and source2 using evaluated quantities
         numGPsource.Eval(state1, state2, nor, Tr, source1, source2);
         // The normal already includes the face Jacobian, so only the
         // reference-face quadrature weight is needed for these signed terms.
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



/// Explicit DG operator for ideal MHD with optional GP sources and LDF projection.
/// The finite element space and the referenced flux/source objects must outlive
/// the operator. Mass inverses and projectors are cached for the fixed mesh.
class LDF_DG_GP_MHD : public TimeDependentOperator
{
private:
   const bool use_GPsource;
   const bool use_LDF;

   const int num_equations; // the number of equations
   const int dim;    ///< Physical/spatial dimension (no embedded surfaces).
   const int MHDdim; ///< Number of velocity and magnetic components.
   FiniteElementSpace &vfes; // vector finite element space (ByNode)

   // Own the integrators; NonlinearForm borrows them with UseExternalIntegrators().
   // The flux integrator also collects local characteristic speed estimates.
   std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int = nullptr;

   std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int = nullptr;

   // Base Nonlinear Form
   std::unique_ptr<NonlinearForm> nonlinearForm;

   // element-wise inverse mass matrix
   std::vector<DenseMatrix> invmass; // local scalar inverse mass
   std::vector<DenseMatrix>
   ldf_projectors; // local magnetic L2 projectors to locally divergence-free subspace

   // Process-local quadrature-point maximum from the latest Mult(), not an
   // MPI reduction. A parallel driver must perform its own global reduction.
   mutable real_t max_char_speed;

   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();
   void ComputeLDFProjectors();

public:
   /**
    * @brief Construct a new LDF_DG_GP_MHD object
    *
    * @param vfes_ DG vector finite element space with Ordering::byNODES
    * @param hyperb_form_Int_ non-null integrator for the volume and face fluxes
    * @param gpsource_form_Int_ integrator for the volume and face GP sources;
    *        may be null only when use_GPsource_ is false
    * @param use_GPsource_ whether to include the Godunov-Powell sources
    * @param use_LDF_ whether to project magnetic updates to the LDF space
    */
   LDF_DG_GP_MHD(
      FiniteElementSpace &vfes_,
      std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int_,
      std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int_,
      const bool use_GPsource_, const bool use_LDF_ = false);
   /** @brief Project the magnetic coefficients of a state or time derivative
       onto the element-wise LDF space, leaving the other fields unchanged.
       Requires use_LDF_ = true and scalar VALUE-mapped P^p simplex or Q^p
       tensor-product elements, or P^p(triangle) x P^p(segment) prisms.
       Geometry must be affine, except that the
       existing bilinear 2D quadrilateral projection is also supported.
       In 1D, replace B_x by its mean; in 2D/3D project all spatial components.
       Remaining magnetic components and all nonmagnetic fields, including E,
       are unchanged. Cell means are preserved but normal face jumps can remain.
       No positivity limiter or normal-continuity constraint is provided.
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
   /// Process-local maximum speed; call Mult() before querying it.
   real_t GetMaxCharSpeed() const { return max_char_speed; }
   /// Rebuild residual data, mass inverses and projectors after a mesh/space
   /// update. The driver must separately transfer and reproject the state.
   void Update();

};

//////////////////////////////////////////////////////////////////
///        MHD with Godunov-Powell term IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class LDF_DG_GP_MHD
LDF_DG_GP_MHD::LDF_DG_GP_MHD(
   FiniteElementSpace &vfes_,
   std::unique_ptr<HyperbolicFormIntegrator> hyperb_form_Int_,
   std::unique_ptr<GPsourceFormIntegrator> gpsource_form_Int_,
   const bool use_GPsource_, const bool use_LDF_)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     use_GPsource(use_GPsource_),
     use_LDF(use_LDF_),
     num_equations(hyperb_form_Int_ ? hyperb_form_Int_->num_equations : 0),
     dim(vfes_.GetMesh()->SpaceDimension()),
     MHDdim((num_equations-2)/2),
     vfes(vfes_),
     hyperb_form_Int(std::move(hyperb_form_Int_)),
     gpsource_form_Int(std::move(gpsource_form_Int_)),
     z(vfes_.GetTrueVSize())
{
   MFEM_VERIFY(hyperb_form_Int, "A flux integrator is required.");
   MFEM_VERIFY(vfes.GetVDim() == num_equations,
               "vfes vdim must match the flux's number of equations.");
   MFEM_VERIFY(!use_GPsource ||
               (gpsource_form_Int &&
                gpsource_form_Int->num_equations == num_equations),
               "GP source integrator is missing or has a different "
               "number of equations than the flux.");
   MFEM_VERIFY(vfes.FEColl()->GetContType() ==
               FiniteElementCollection::DISCONTINUOUS &&
               vfes.GetOrdering() == Ordering::byNODES,
               "The DG operator requires a component-wise discontinuous space.");
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
   // Use the standard element/face NonlinearForm assembly path. No partial
   // assembly kernels or implicit Jacobians are implemented by this example.

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

void LDF_DG_GP_MHD::ComputeInvMass()
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


void LDF_DG_GP_MHD::ComputeLDFProjectors()
{
   MFEM_VERIFY(dim >= 1 && dim <= 3 && vfes.GetMesh()->Dimension() == dim,
               "LDF requires a full-dimensional mesh in dimensions 1 to 3.");
   MFEM_VERIFY(!vfes.GetMesh()->NURBSext,
               "The LDF constraint space requires polynomial, not NURBS, maps.");
   MFEM_VERIFY(vfes.GetOrdering() == Ordering::byNODES &&
               vfes.GetVDim() == num_equations &&
               vfes.FEColl()->GetContType() == FiniteElementCollection::DISCONTINUOUS,
               "LDF projection requires the component-wise DG state space.");
   ldf_projectors.resize(vfes.GetNE());
   for (int k = 0; k < vfes.GetNE(); k++)
   {
      const FiniteElement &fe = *vfes.GetFE(k);
      ElementTransformation &Tr = *vfes.GetElementTransformation(k);
      if (dim == 1)
      {
         MFEM_VERIFY(fe.GetGeomType() == Geometry::SEGMENT && Tr.Order() == 1 &&
                     fe.GetRangeType() == FiniteElement::SCALAR &&
                     fe.GetMapType() == FiniteElement::VALUE,
                     "1D LDF requires affine segments and scalar VALUE elements.");
         // In one spatial dimension, div(B)=dB_x/dx. Its kernel consists
         // exactly of element constants, so the L2 projection is the mean.
         const int dof = fe.GetDof();
         DenseMatrix &P = ldf_projectors[k];
         P.SetSize(dof);
         Vector shape(dof), integrals(dof);
         integrals = 0.0;
         real_t volume = 0.0;
         const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT,
                                                  2*fe.GetOrder()+Tr.OrderW());
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            fe.CalcShape(ip, shape);
            const real_t weight = ip.weight*Tr.Weight();
            integrals.Add(weight, shape);
            volume += weight;
         }
         for (int i = 0; i < dof; i++)
         {
            for (int j = 0; j < dof; j++) { P(i,j) = integrals(j)/volume; }
         }
         continue;
      }
      const auto geom = fe.GetGeomType();
      const bool simplex = geom == Geometry::TRIANGLE ||
                           geom == Geometry::TETRAHEDRON;
      const bool tensor = geom == Geometry::SQUARE || geom == Geometry::CUBE;
      const bool prism = geom == Geometry::PRISM;
      MFEM_VERIFY(geom != Geometry::PYRAMID,
                  "Pyramid LDF needs a rational divergence space; it is not implemented.");
      MFEM_VERIFY((simplex || tensor || prism) && Tr.Order() == 1 &&
                  fe.GetRangeType() == FiniteElement::SCALAR &&
                  fe.GetMapType() == FiniteElement::VALUE &&
                  fe.Space() == (simplex ? FunctionSpace::Pk : FunctionSpace::Qk),
                  "LDF requires first-order simplex, tensor or prism geometry and scalar VALUE elements.");
      if (dim == 3 && (tensor || prism))
      {
         // First-order hex/prism maps need not be affine. Check the Jacobian
         // at every reference vertex; these determine its variation for
         // Q^1 hex and P^1(triangle) x P^1(segment) prism geometry.
         const IntegrationPoint &center = Geometries.GetCenter(geom);
         Tr.SetIntPoint(&center);
         const DenseMatrix reference_jacobian(Tr.Jacobian());
         const real_t tolerance = 64*std::numeric_limits<real_t>::epsilon()*
                                  reference_jacobian.MaxMaxNorm();
         const IntegrationRule &vertices = *Geometries.GetVertices(geom);
         for (int q = 0; q < vertices.GetNPoints(); q++)
         {
            Tr.SetIntPoint(&vertices.IntPoint(q));
            DenseMatrix difference(Tr.Jacobian());
            difference -= reference_jacobian;
            MFEM_VERIFY(difference.MaxMaxNorm() <= tolerance,
                        "3D LDF currently requires affine hexahedra or prisms.");
         }
      }
      const int dof = fe.GetDof();
      const int p = fe.GetOrder();
      DenseMatrix &P = ldf_projectors[k];
      P.SetSize(dim*dof);
      P = 0.0;
      for (int j = 0; j < dim*dof; j++) { P(j,j) = 1.0; }
      if (p == 0) { continue; } // Cell-wise constants are already LDF.

      // Affine simplex divergence lies in P^(p-1). For affine tensor-product cells,
      // it lies in Q^p without the highest tensor monomial (p,...,p).
      // The same numerator space represents det(J)*div(B) on bilinear 2D
      // quads. Testing it gives independent rows and avoids a singular Schur
      // complement. Q^(p-1) would miss tensor-product divergence modes.
      L2_FECollection constraint_fec(simplex ? p-1 : p, dim);
      const FiniteElement &qfe =
         *constraint_fec.FiniteElementForGeometry(fe.GetGeomType());
      // A prism divergence is
      // P^p(triangle) x P^(p-1)(z) + P^(p-1)(triangle) x P^p(z).
      // Use full triangular P^p below the top z mode, and P^(p-1)
      // for that top mode. This removes exactly p+1 redundant constraints.
      const int constraints = simplex ? qfe.GetDof() :
                              prism ? qfe.GetDof()-(p+1) : qfe.GetDof()-1;
      L2_FECollection triangle_fec(p, 2), lower_triangle_fec(p-1, 2);
      const FiniteElement &triangle_fe =
         *triangle_fec.FiniteElementForGeometry(Geometry::TRIANGLE);
      const FiniteElement &lower_triangle_fe =
         *lower_triangle_fec.FiniteElementForGeometry(Geometry::TRIANGLE);
      Vector triangle_shape(triangle_fe.GetDof());
      Vector lower_triangle_shape(lower_triangle_fe.GetDof());
      DenseMatrix D(constraints, dim*dof);
      D = 0.0;
      Vector qshape(constraints);
      Vector legendre_x(p+1), legendre_y(p+1), legendre_z(p+1);
      DenseMatrix dshape(dof, dim);
      const IntegrationRule &ir =
         IntRules.Get(fe.GetGeomType(), 2*p + Tr.OrderW());
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         if (simplex) { qfe.CalcShape(ip, qshape); }
         else if (prism)
         {
            triangle_fe.CalcShape(ip, triangle_shape);
            lower_triangle_fe.CalcShape(ip, lower_triangle_shape);
            Poly_1D::CalcLegendre(p, ip.z, legendre_z.GetData());
            int a = 0;
            for (int l = 0; l <= p; l++)
            {
               const Vector &shape = l == p ? lower_triangle_shape : triangle_shape;
               for (int j = 0; j < shape.Size(); j++)
               {
                  qshape(a++) = shape(j)*legendre_z(l);
               }
            }
         }
         else
         {
            Poly_1D::CalcLegendre(p, ip.x, legendre_x.GetData());
            Poly_1D::CalcLegendre(p, ip.y, legendre_y.GetData());
            if (dim == 3) { Poly_1D::CalcLegendre(p, ip.z, legendre_z.GetData()); }
            int a = 0;
            for (int l = 0; l <= (dim == 3 ? p : 0); l++)
            {
               for (int j = 0; j <= p; j++)
               {
                  for (int i = 0; i <= p; i++)
                  {
                     if (i == p && j == p && (dim == 2 || l == p)) { continue; }
                     qshape(a++) = legendre_x(i)*legendre_y(j)*
                                   (dim == 3 ? legendre_z(l) : 1_r);
                  }
               }
            }
         }
         fe.CalcPhysDShape(Tr, dshape);
         const real_t weight = ip.weight * Tr.Weight();
         for (int a = 0; a < constraints; a++)
         {
            for (int j = 0; j < dof; j++)
            {
               for (int c = 0; c < dim; c++)
               {
                  D(a,c*dof+j) += weight*qshape(a)*dshape(j,c);
               }
            }
         }
      }

      // H = diag(M,...,M), A = H^{-1} D^T, S = D A.
      // P = I - A S^{-1} D is the H-orthogonal projector onto ker(D).
      DenseMatrix A(dim*dof, constraints);
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
      DenseMatrix S(constraints), SinvD(constraints, dim*dof);
      mfem::Mult(D, A, S);
      DenseMatrixInverse schur_inverse(S, true);
      schur_inverse.Mult(D, SinvD);
      DenseMatrix correction(dim*dof);
      mfem::Mult(A, SinvD, correction);
      P -= correction;
   }
}

void LDF_DG_GP_MHD::ApplyLDF(Vector &state) const
{
   MFEM_VERIFY(use_LDF && state.Size() == vfes.GetVSize(),
               "ApplyLDF requires enabled LDF projection and a full DG vector.");
   Vector magnetic, projected;
   Array<int> vdofs;
   DofTransformation doftrans;
   real_t *state_data = state.HostReadWrite();
   // DG element DOFs are disjoint. Keep all scratch storage thread-local,
   // including the DOF transformation, and access shared vectors on the host.
   // When built with OpenMP and thread safety, use the runtime's thread count
   // for every mesh size; OMP_NUM_THREADS=1 explicitly selects one thread.
#if defined(MFEM_USE_OPENMP) && defined(MFEM_THREAD_SAFE)
   #pragma omp parallel for schedule(static) \
   private(magnetic, projected, vdofs, doftrans)
#endif
   for (int k = 0; k < vfes.GetNE(); k++)
   {
      const int dof = vfes.GetFE(k)->GetDof();
      vfes.GetElementVDofs(k, vdofs, doftrans);
      // Local fields are contiguous: (rho, momentum, magnetic field, E).
      magnetic.SetSize(dim*dof);
      projected.SetSize(dim*dof);
      for (int j = 0; j < dim*dof; j++)
      {
         magnetic(j) = state_data[vdofs[(1+MHDdim)*dof+j]];
      }
      ldf_projectors[k].Mult(magnetic, projected);
      for (int j = 0; j < dim*dof; j++)
      {
         state_data[vdofs[(1+MHDdim)*dof+j]] = projected(j);
      }
   }
}


void LDF_DG_GP_MHD::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   hyperb_form_Int->ResetMaxCharSpeed();

   // 1. Assemble the residual serially. The flux integrator's speed accumulator
   // and the form's shared element/face transformations are not parallelized.
   nonlinearForm->Mult(x, z);

   // 2. Apply each scalar inverse mass to all state components in its element.
   Vector zval; // z_loc, dof*num_eq
   DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
   DenseMatrix current_ymat; // view of element result, dof x num_eq
   Array<int> vdofs;
   DofTransformation doftrans;
   const real_t *z_data = z.HostRead();
   real_t *y_data = y.HostWrite();
#if defined(MFEM_USE_OPENMP) && defined(MFEM_THREAD_SAFE)
   #pragma omp parallel for schedule(static) \
   private(zval, current_zmat, current_ymat, vdofs, doftrans)
#endif
   for (auto i=0; i<vfes.GetNE(); i++)
   {
      const auto dof = vfes.GetFE(i)->GetDof();
      // ALWAYS returns ByNodes ordering (like separate fields)
      vfes.GetElementVDofs(i, vdofs, doftrans);
      // copy from the global vector to the local vector
      zval.SetSize(vdofs.Size());
      for (int j = 0; j < vdofs.Size(); j++) { zval(j) = z_data[vdofs[j]]; }
      // each column is a field and each row is a DOF
      current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
      current_ymat.SetSize(dof, num_equations);
      // invmass left multiplies the auxiliary result
      mfem::Mult(invmass[i], current_zmat, current_ymat);
      // return the result to the global vector
      for (int j = 0; j < vdofs.Size(); j++)
      {
         y_data[vdofs[j]] = current_ymat.GetData()[j];
      }
   }
   // Project the time derivative after applying the mass inverse. With an
   // LDF initial state on a fixed mesh, explicit RK stages remain LDF.
   if (use_LDF) { ApplyLDF(y); }
   max_char_speed = hyperb_form_Int->GetMaxCharSpeed();
}

void LDF_DG_GP_MHD::Update()
{
   nonlinearForm->Update();
   height = nonlinearForm->Height();
   width = height;
   z.SetSize(height);

   ComputeInvMass();
   if (use_LDF) { ComputeLDFProjectors(); }
}


/** Construct the periodic model mesh.
    Problem 1 uses the triangular unit square. Problem 2 uses an interval in
    1D, the paper's rectangle in 2D, or a fully periodic diagonal wave in 3D.
    @param problem 1 for Orszag-Tang, 2 for the smooth Alfven wave
    @param dim spatial dimension; Orszag-Tang requires dim=2
    @param simplex use triangles/tetrahedra for generated multidimensional meshes
    @param prism use prisms for the generated 3D wave mesh, instead of hexes */
Mesh MHDMesh(const int problem, const int dim = 2, const bool simplex = false,
             const bool prism = false)
{
   MFEM_VERIFY(problem == 1 || problem == 2, "Unknown MHD problem.");
   MFEM_VERIFY(dim >= 1 && dim <= 3 && (problem == 2 || dim == 2),
               "Orszag-Tang is 2D; the Alfven wave supports dimensions 1 to 3.");
   MFEM_VERIFY(!prism || (problem == 2 && dim == 3 && !simplex),
               "Prism generation requires problem 2, dimension 3, and no -simp.");
   if (problem == 2)
   {
      if (dim == 1)
      {
         Mesh mesh = Mesh::MakeCartesian1D(4);
         std::vector<Vector> translations = {Vector({1_r})};
         return Mesh::MakePeriodic(mesh,
                                   mesh.CreatePeriodicVertexMapping(translations));
      }
      const real_t lx = 1.0/std::cos(M_PI/6.0);
      const real_t ly = 1.0/std::sin(M_PI/6.0);
      if (dim == 3)
      {
         const real_t length = std::sqrt(3.0);
         Mesh mesh = Mesh::MakeCartesian3D(4, 4, 4,
                                           simplex ? Element::TETRAHEDRON :
                                           prism ? Element::WEDGE : Element::HEXAHEDRON,
                                           length, length, length);
         std::vector<Vector> translations =
         {Vector({length, 0_r, 0_r}), Vector({0_r, length, 0_r}), Vector({0_r, 0_r, length})};
         return Mesh::MakePeriodic(mesh,
                                   mesh.CreatePeriodicVertexMapping(translations));
      }
      Mesh mesh = Mesh::MakeCartesian2D(4, 4,
                                        simplex ? Element::TRIANGLE : Element::QUADRILATERAL,
                                        true, lx, ly);
      std::vector<Vector> translations = {Vector({lx, 0_r}), Vector({0_r, ly})};
      return Mesh::MakePeriodic(mesh,
                                mesh.CreatePeriodicVertexMapping(translations));
   }
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
/** Return initial data, or a time-dependent exact solution for problem 2.
    The Alfven wave always uses eight components; its 1D form has alpha=0,
    and its 3D form propagates along (1,1,1)/sqrt(3). The 2D form follows Section 4.1 of
    arXiv:2604.23885v2. The state order is (rho, rho*v, B, E).
    @param problem 1 for Orszag-Tang, 2 for the smooth Alfven wave
    @param specific_heat_ratio ratio of specific heats, greater than one
    @param dim spatial dimension (1 to 3); problem 1 requires 2 */
VectorFunctionCoefficient MHDInitialCondition(const int problem,
                                              real_t specific_heat_ratio,
                                              const int dim = 2)
{
   MFEM_VERIFY(dim >= 1 && dim <= 3 && (problem == 2 || dim == 2),
               "Invalid model problem dimension.");
   switch (problem)
   {
      case 2: // Wu and Shu, arXiv:2604.23885v2, Section 4.1.
         // In 2D alpha = pi/6 (zero in 1D), rho = B_parallel = 1, p = 0.1.
         // v_perp = B_perp = 0.1*sin(phi), v_z = B_z = 0.1*cos(phi),
         // phi = 2*pi*(x*cos(alpha)+y*sin(alpha)+t) in 1D/2D.
         // The 3D generalization below uses a diagonal propagation direction.
         // The wave travels in the negative parallel direction with speed
         // one and period one. Use T=1 here (the paper uses five periods).
         return VectorFunctionCoefficient(8, [specific_heat_ratio, dim](
                                             const Vector &x, real_t t, Vector &y)
         {
            MFEM_VERIFY(x.Size() == dim, "Alfven wave coordinate dimension mismatch.");
            if (dim == 3)
            {
               // Orthonormal frame: n=(1,1,1)/sqrt(3),
               // e1=(-1,1,0)/sqrt(2), e2=(-1,-1,2)/sqrt(6).
               const real_t normal = 1.0/std::sqrt(3.0);
               const real_t phase = 2*M_PI*((x(0)+x(1)+x(2))*normal+t);
               const real_t a = 0.1*std::sin(phase)/std::sqrt(2.0);
               const real_t b = 0.1*std::cos(phase)/std::sqrt(6.0);
               y(0) = 1.0;
               y(1) = -a-b;
               y(2) = a-b;
               y(3) = 2*b;
               for (int i = 0; i < 3; i++) { y(4+i) = normal+y(1+i); }
               y(7) = 0.1/(specific_heat_ratio-1.0)+0.51;
               return;
            }
            const real_t ca = dim == 1 ? 1_r : std::cos(M_PI/6.0);
            const real_t sa = dim == 1 ? 0_r : std::sin(M_PI/6.0);
            const real_t parallel = x(0)*ca + (dim > 1 ? x(1)*sa : 0_r);
            const real_t phase = 2*M_PI*(parallel+t);
            const real_t perpendicular = 0.1*std::sin(phase);
            const real_t transverse = 0.1*std::cos(phase);
            y(0) = 1.0;
            y(1) = -perpendicular*sa;
            y(2) = perpendicular*ca;
            y(3) = transverse;
            y(4) = ca-perpendicular*sa;
            y(5) = sa+perpendicular*ca;
            y(6) = transverse;
            // |v|^2=0.01, |B|^2=1.01; add internal, kinetic, magnetic energy.
            y(7) = 0.1/(specific_heat_ratio-1.0) + 0.5 + 0.01;
         });
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
