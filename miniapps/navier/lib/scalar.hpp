#ifndef MFEM_NAVIER_SCALAR_HPP
#define MFEM_NAVIER_SCALAR_HPP

#include <mfem.hpp>
#include <vector>

namespace mfem
{
namespace navier
{

/**
 * @brief ScalarEquation discretizes the advection-diffusion-reaction equation
 * of a scalar quantity k in residual form R(k) = 0.
 *
 * The advection-diffusion-reaction equation of a scalar quantity k in residual
 * form
 *
 * R(k) = \nabla \cdot D \nabla k - \nabla \cdot (v k) + R(k) + F
 *
 * with diffusion coefficient D, velocity v, reaction term R(k) and forcing term
 * F. All coefficients can depend on the state k since they are only evaluated
 * in residual form. This class is designed to work with explicit time
 * integrators.
 *
 * The equation is discritizes using a Discontinuous Galerkin (DG) method that
 * relies on upwinding for the convection term and the Symmetric Interior
 * Penalty Galerkin (SIPG) stabilization for diffusion. Additionally, collocated
 * nodes and quadrature points (Gauss-Lobatto-Legendre) are used to exploit a
 * diagonalized mass matrix.
 */
class ScalarEquation : public TimeDependentOperator
{
public:
   /**
    * @brief Construct a new Scalar Equation object
    *
    * @param[in] mesh
    * @param[in] order
    * @param[in] velgf
    */
   ScalarEquation(ParMesh &mesh, const int order,
                  const ParGridFunction &velgf);

   /**
    * @brief Evaluate the residual y = R(x)
    *
    * @param[in] x Scalar values on the true dofs
    * @param[out] y Residual values on the true dofs
    */
   void Mult(const Vector &x, Vector &y) const override;

   void SetTime(double t) override;

   /// Prepare all integrators and forms
   void Setup();

   /// Reassemble all forms
   void Reassemble() const;

   /**
    * @brief Add a reaction term with coefficient c
    *
    * @param[in] c
    */
   void AddReaction(Coefficient &c);

   /**
    * @brief Set the scalar to a fixed value at specified boundary attributes.
    * This enforces a Dirichlet like boundary condition.
    *
    * @param[in] c
    * @param[in] bdr_attributes
    */
   void SetFixedValue(Coefficient &c, std::vector<int> bdr_attributes);

   /**
    * @brief Add a forcing term with coefficient c
    *
    * @param[in] c
    */
   void AddForcing(Coefficient &c);

   /**
    * @brief Set the viscosity coefficient
    *
    * @param[in] c
    */
   void SetViscosityCoefficient(Coefficient &c)
   {
      q_coeff = &c;
   };

   /**
    * @brief Get the scalar ParGridFunction
    *
    * @return ParGridFunction&
    */
   ParGridFunction &GetScalar() { return k_gf; };

protected:
   ParMesh &mesh;
   const int order;
   const DG_FECollection fec;
   ParFiniteElementSpace fes;
   const IntegrationRule *ir = nullptr, *ir_face = nullptr; /// not owned
   IntegrationRules *gll_rules = nullptr;

   ParBilinearForm mform;
   mutable ParBilinearForm kform;
   ParLinearForm bform;
   mutable ParLinearForm rform;

   mutable ParGridFunction k_gf;

   ConstantCoefficient zero_coeff;
   ConstantCoefficient one_coeff;
   VectorGridFunctionCoefficient vel_coeff;
   Coefficient *q_coeff = nullptr;
   TransformedCoefficient *neg_q_coeff = nullptr;

   mutable OperatorHandle K;

   Vector m, minv;

   Vector B;
   mutable Vector z, R;

   // SIPG coefficient
   const double alpha = -1.0, beta = -0.5;
   // SIPG coefficient
   double sigma = -1.0, kappa;

   const int skip_zeros = 0;
};

} // namespace navier

} // namespace mfem

#endif