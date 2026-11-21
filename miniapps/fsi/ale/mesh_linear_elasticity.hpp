#pragma once

#include "mfem.hpp"
#include "mesh_deformation.hpp"

using namespace mfem;

/**
 * @brief Class for computing mesh deformation using linear elasticity theory for Arbitrary Lagrangian-Eulerian (ALE) 
 * mesh motion in fluid-structure interaction (FSI) simulations.
 * 
 * This class treats the fluid domain as an elastic body and constructs ALE displacement as a solution to the 
 * equations of linear elasticity: div(σ(u_a)) = 0, where σ is the Cauchy stress tensor related to the 
 * linearized strain tensor ε_a = (∇u_a + ∇u_a^T)/2 by Hooke's law: σ = λ_a tr(ε_a)I + 2μ_a ε_a
 */
class MeshLinearElasticity : public MeshDeformation
{
public:
   /**
    * @brief Construct a new MeshLinearElasticity object.
    *
    * We assume that the internal mesh nodes and the fluid GridFunctions
    * are defined on the same mesh, but may use different FE collections.
    *
    * @param fluid_mesh_ Finite element space for the fluid velocity field.
    * @param incremental If true, we use the incremental version (solve on the deformed mesh).
    */
   MeshLinearElasticity(ParMesh *fluid_mesh_, bool incremental_ = false);

   /**
    * @brief Destructor for MeshLinearElasticity.
    */
   ~MeshLinearElasticity();

   /**
    * @brief Enable local mesh stiffening for specific wall attributes.
    *
    * @param wall_attr_ Array marking the wall attributes to apply stiffening.
    * @param stiffening_type_ Type of stiffening to apply.
    *
    * @note See MeshDeformation::EnableLocalStiffening. If called after Setup the system is re-assembled.
    * @note Default arguments must match the base class ones (defaults bind to the static type).
    */
   void EnableLocalStiffening(Array<int> &wall_attr_, int stiffening_type_, int distance_solver_type_ = 0, real_t stiffening_degree_ = 1.0, bool filter_ls_ = false) override;
   void EnableLocalStiffening(int wall_attr_, int stiffening_type_, int distance_solver_type_ = 0, real_t stiffening_degree_ = 1.0, bool filter_ls_ = false) override;

   /**
    * @brief Setup the linear elasticity problem.
    * @param E    Young's modulus (elastic modulus).
    * @param nu   Poisson's ratio, must be in (-1, 0.5).
    * @param pa_        Enable partial assembly. Requires a byNODES mesh-node space
    *                   (the default SetCurvature/EnsureNodes space is byVDIM, so this currently aborts).
    * @param prec_type_ Type of preconditioner to use. (Ignored: there's only one type of preconditioner, elasticity AMG)
    */
   void Setup(real_t E_, real_t nu_, bool pa_ = false, int prec_type_ = 0);

   void Setup() override
   {
      Setup(1.0, 0.3, false, 0); // E = 1, nu = 0.3 (nu = 1 gives lambda < 0, an indefinite operator)
   }

   /**
    * @brief Update the linear elasticity problem after mesh is deformed.
    */
   void Update() override;

   /**
    * @brief Solve the linear elasticity system for a given field and optionally deform the mesh.
    * @param field        Grid function representing the original field to solve for.
    * @param deform_mesh  Flag to enable mesh deformation.
    * @param dt           Time step size for the mesh motion. 
    * @note This method is used to solve the linear elasticity equations for a given field, and uses it to deform the mesh.
    * If no dt is provided (dt<0) the optional mesh deformation is performed assuming the field is a displacement field.
    * Otherwise if a positive dt is provided, the field is assumed to be a velocity field and the mesh is deformed accordingly.
    */
   void Solve(ParGridFunction &field, bool deform_mesh = false, real_t dt = -1.0) override;

   void DeformMeshWithVelocity(ParGridFunction &mesh_velocity_gf, real_t dt, bool update_gf = false) override;
   

   Coefficient *GetCoefficient() override
   {
      return lambda_stiff;
   }

private:

   /**
    * @brief Reset the solver and preconditioner.
    */
   void ResetSolver();

   /**
    * @brief (Re)create the coefficients and bilinear form, assemble the system and reset the solver.
    */
   void AssembleOperator();
   
   ParMesh *reference_mesh = nullptr;                   ///< Reference parallel mesh for the fluid simulation. //< OWNED

   ParBilinearForm *elasticity_form = nullptr;         ///< Bilinear form for the linear elasticity operator.
   OperatorHandle opA;                                  ///< System matrix operator handle.

   bool pa = false;                                     ///< Flag for partial assembly (default false).
   ParLORDiscretization *lor = nullptr;                 ///< Low-order refined discretization for the linear elasticity problem (if using pa) //< OWNED

   real_t E = 1.0;                                      ///< Young's modulus (elastic modulus). (Doesn't affect the mesh deformation)
   real_t nu = 0.3;                                     ///< Poisson's ratio. (Affects computation; for stability recommends 0.3 to 0.45)
   ConstantCoefficient lambda;                          ///< Lamé's first parameter (λ).
   ConstantCoefficient mu;                              ///< Lamé's second parameter (μ, shear modulus).

   ProductCoefficient *lambda_stiff = nullptr;          ///< Product coefficient for the Lamé's first parameter (λ) with local stiffening.
   ProductCoefficient *mu_stiff = nullptr;              ///< Product coefficient for the Lamé's second parameter (μ) with local stiffening.
};