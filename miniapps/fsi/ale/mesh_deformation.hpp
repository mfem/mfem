#pragma once

#include "mfem.hpp"
#include "local_stiffening_coefficient.hpp"

using namespace mfem;

/**
 * @brief Abstract base class for mesh deformation algorithms used in Arbitrary Lagrangian-Eulerian (ALE) mesh motion
 * in fluid-structure interaction (FSI) simulations.
 */

enum class MeshDeformationType : int
{
   Laplacian = 0,
   IncrementalLaplacian = 1,
   LinearElasticity = 2
};

class MeshDeformation
{
public:
   /**
    * @brief Construct a new MeshDeformation object.
    *
    * @param fluid_mesh_ Parallel mesh for the fluid simulation.
    */
   MeshDeformation(ParMesh *fluid_mesh_);

   /**
    * @brief Virtual destructor for MeshDeformation.
    */
   virtual ~MeshDeformation();

   /**
    * @brief Initialize the mesh deformation problem.
    *
    * @param mesh Parallel mesh for the fluid simulation
    * (doesn't use fluid_mesh directly as Incremental/NonIncremental mesh deformations have different setup).
    */
   virtual void Initialize(ParMesh *mesh);

   /**
    * @brief Set the essential boundary attributes for the mesh deformation problem.
    *
    * @param ess_bdr_ Array marking the essential boundary attributes.
    *
    * @note Must be called before Setup (the essential dofs are extracted in Setup).
    */
   void SetEssentialBoundary(Array<int> &ess_bdr_);

   /**
    * @brief Enable local mesh stiffening for specific wall attributes.
    *
    * @param wall_attr_ Array marking the wall attributes to apply stiffening.
    * @param stiffening_type_ Type of stiffening to apply (0: Jacobian-based, 1: Distance-based).
    * @param distance_solver_type Distance solver for the distance-based stiffening (0: Heat, 1: PLaplacian, 2: Normalization).
    * @param stiffening_degree_ Stiffening degree (power applied to the stiffening coefficient).
    * @param filter_ls Filter the wall level set before computing the distance (required by the Normalization solver).
    *
    * @note This method should be called before Setup. If called after Setup the system is re-assembled,
    *       and any pointer previously returned by GetCoefficient() is invalidated.
    * @note The distance-based stiffening uses a transition zone [Rmin, Rmax] = [2h, 6h] (h = average element size),
    *       alpha = 10, Amax = 100, Amin = 1.
    */
   virtual void EnableLocalStiffening(Array<int> &wall_attrs_,
                                      int stiffening_type_, int distance_solver_type = 0,
                                      real_t stiffening_degree_ = 1.0, bool filter_ls = false);
   virtual void EnableLocalStiffening(int wall_attr_, int stiffening_type_,
                                      int distance_solver_type = 0, real_t stiffening_degree_ = 1.0,
                                      bool filter_ls = false);

   /**
    * @brief Setup the mesh deformation problem.
    */
   virtual void Setup() = 0;

   /**
    * @brief Update the mesh deformation problem after mesh is deformed.
    */
   virtual void Update();

   /**
    * @brief Solve the deformation system for a given field and optionally deform the mesh.
    * @param field        Grid function representing the field to solve for.
    * @param deform_mesh  Flag to enable mesh deformation.
    * @param dt           Time step size for the mesh motion (negative means displacement field).
    */
   virtual void Solve(ParGridFunction &field, bool deform_mesh = false,
                      real_t dt = -1.0) = 0;

   /**
    * @brief Deform the mesh nodes based on the provided displacement field.
    * @param displacement_gf Grid function representing the displacement field.
    * @param update_gf        Flag to update the grid function after deformation. (Should be disabled for non-incremental deformation methods)
    * @note This method deforms the mesh based on the displacement field in the reference configuration:
    *       new_nodes = ref_nodes + displacement_gf.
    */
   virtual void DeformMeshWithDisplacement(ParGridFunction &displacement_gf,
                                           bool update_gf = false);

   /**
    * @brief Deform the mesh nodes based on the provided mesh velocity.
    * @param mesh_velocity_gf Grid function representing the mesh velocity field.
    * @param dt              Time step size.
    * @param update_gf        Flag to update the grid function after deformation. (Should be disabled for non incremental deformation methods)
    */
   virtual void DeformMeshWithVelocity(ParGridFunction &mesh_velocity_gf,
                                       real_t dt, bool update_gf = false);

   virtual void DeformMeshWithRelativeDisplacement(ParGridFunction
                                                   &relative_displacement_gf, bool update_gf = false)
   {
      // Deform the mesh with relative displacement
      DeformMeshWithVelocity(relative_displacement_gf, 1.0, update_gf);
   }

   /**
    * @brief Get the finite element space for the mesh nodes.
    *
    * @note: If the mesh is periodic, creates an H1 space for the mesh nodes,
    *        otherwise returns the original fes for the mesh nodes.
    *
    * @return Pointer to the finite element space for the mesh nodes.
    */
   ParFiniteElementSpace *GetFESpace() const { return fes_mesh; }

   /**
    * @brief Get the original finite element space for the mesh nodes.
    *
    * @return Pointer to the original finite element space for the mesh nodes.
    */
   ParFiniteElementSpace *GetOriginalFESpace() const { return original_fes_mesh; }


   /**
   * @brief Get the finite element space for the stiffening of mesh nodes.
   *
   * @return Pointer to the finite element space for the stiffening of mesh nodes.
   */
   ParFiniteElementSpace *GetStiffeningFESpace() const { return fes_stiffening; }

   /// @note The returned pointer is invalidated by Setup() and EnableLocalStiffening().
   virtual Coefficient *GetCoefficient() = 0;

   LocalStiffeningCoefficient *GetLocalStiffeningCoefficient() const { return local_stiffening_coeff; }

   bool IsNonIncremental() const { return is_nonincremental_mdt; }

protected:
   /// Notify the fluid mesh that its nodes were modified in place (geometric factors, face-neighbor nodes)
   void NodesUpdated();

   ParMesh *fluid_mesh;                       ///< Parallel mesh for the fluid simulation.               //< NOT OWNED

   ParFiniteElementSpace *fes_mesh =
      nullptr; ///< Finite element space for the mesh nodes.              //< OWNED if periodic
   const FiniteElementCollection *fec =
      nullptr;    ///< Finite element collection for the mesh nodes.         //< OWNED if periodic
   ParFiniteElementSpace *original_fes_mesh =
      nullptr; ///< Original finite element space for the mesh nodes.     //< NOT OWNED
   bool periodic =
      false;                     ///< Flag indicating if the mesh fes is periodic (discontinuous Galerkin (DG) space for Nodes).

   //ParGridFunction *nodes = nullptr;          ///< Grid function for the deformed mesh nodes.            //< OWNED
   ParGridFunction *ref_nodes =
      nullptr;      ///< Grid function for the reference mesh nodes.           //< OWNED

   Array<int> ess_bdr;                        ///< Essential boundary attributes.
   Array<int>
   ess_tdof_list;                  ///< List of true DOFs for essential boundaries.

   IterativeSolver *linear_solver = nullptr;  ///< Iterative solver.
   Solver *prec = nullptr;                    ///< Preconditioner.
   Vector X, B;                               ///< Solution and RHS vectors in true DOF space.
   Vector b;

   int my_id =
      0;                             ///< Rank of the current processor in the MPI communicator.
   int num_procs =
      1;                         ///< Total number of processors in the MPI communicator.

   bool  is_nonincremental_mdt =
      true;       ///< Flag indicating if the mesh deformation is non-incremental. (in the interface transfer we can just use undeformed coordinates)

   bool enable_local_stiffening =
      false;                          ///< Flag to enable local stiffening.
   int stiffening_type =
      1;                                       ///< Type of local stiffening (0: Jacobian-based, 1: Distance-based).
   Array<int>
   wall_attrs;                                         ///< Wall attributes for local stiffening.
   LocalStiffeningCoefficient *local_stiffening_coeff =
      nullptr;  ///< Local stiffening coefficient.
   ParFiniteElementSpace *fes_stiffening =
      nullptr;               ///< Finite element space for local stiffening (on the mesh the operator is assembled on).
   real_t stiffening_degree =
      1.0;                                ///< Stiffening degree for local stiffening.
};