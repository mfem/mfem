#pragma once

#include "mfem.hpp"
#include "mesh_deformation.hpp"

using namespace mfem;

/**
 * @brief Class for computing the mesh velocity field required for Arbitrary Lagrangian-Eulerian (ALE) mesh motion
 * in fluid-structure interaction (FSI) simulations, solving the Harmonic extension problem (mesh Laplacian).
 * This is the classic Harmonic extension problem, formulated in the reference configuration.
 */
class MeshLaplacian : public MeshDeformation
{
public:
   /**
    * @brief Construct a new MeshLaplacian object.
    *
    * We assume that the internal mesh nodes and the fluid GridFunctions
    * are defined on the same mesh, but may use different FE collections.
    *
    * @param fluid_mesh_ Finite element space for the fluid velocity field.
    * @param incremental If true, we use the incremental version (solve on the deformed mesh).
    */
   MeshLaplacian(ParMesh *fluid_mesh_, bool incremental_ = false);

   /**
    * @brief Destructor for MeshLaplacian.
    */
   ~MeshLaplacian();

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
    * @brief Setup the mesh Laplacian problem.
    * @param alpha_     Diffusion coefficient for the mesh Laplacian.
    * @param pa_        Enable partial assembly.
    * @param prec_type_ Type of preconditioner to use (0: Jacobi, 1: AMG (full assembly) / LOR-AMG (partial assembly)).
    */
   void Setup(real_t alpha_, bool pa_ = false, int prec_type_ = 1);

   void Setup() override
   {
      Setup(1.0, false, 1);
   }

   /**
    * @brief Update the mesh Laplacian problem after mesh is deformed.
    */
   void Update() override;

   /**
    * @brief Solve the Laplacian system for a given field and optionally deform the mesh.
    * @param field        Grid function representing the original field to solve for.
    * @param deform_mesh  Flag to enable mesh deformation.
    * @param dt           Time step size for the mesh motion. 
    * @note This method is used to solve the Harmonic extension for a given field, and uses it to deform the mesh.
    * If no dt is provided (dt<0) the optional mesh deformation is performed assuming the field is a displacement field.
    * Otherwise if a positive dt is provided, the field is assumed to be a velocity field and the mesh is deformed accordingly.
    */
   void Solve(ParGridFunction &field, bool deform_mesh = false, real_t dt = -1.0) override;


   Coefficient *GetCoefficient() override
   {
      return diffusion_coeff;
   }
   
   private:

   /**
    * @brief Reset the solver and preconditioner.
    */
   void ResetSolver();

   /**
    * @brief (Re)create the coefficient and bilinear form, assemble the system and reset the solver.
    */
   void AssembleOperator();
   
   ParMesh *reference_mesh = nullptr;                   ///< Reference parallel mesh for the fluid simulation. //< OWNED

   ParBilinearForm *laplacian_form = nullptr; ///< Bilinear form for the Laplacian operator.
   OperatorHandle opA;                        ///< System matrix operator handle.

   bool pa = false;                                ///< Flag for partial assembly (default false).
   int prec_type = 1;                              ///< Type of preconditioner to use.
   ConstantCoefficient alpha;                      ///< Diffusion coefficient for the mesh deformation.
   ProductCoefficient *diffusion_coeff = nullptr;  ///< Product coefficient for the diffusion term (for local stiffening)

   bool incremental;                               ///< Flag for enabling incremental version.
};