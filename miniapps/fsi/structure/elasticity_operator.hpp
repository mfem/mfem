#ifndef ELASTICITY_HPP
#define ELASTICITY_HPP

#include <memory>
#include <mfem.hpp>

#include "elasticity_utils.hpp"
#include "fsi_utils.hpp"

using namespace mfem::common_fsi_utils;

namespace mfem
{
namespace elasticity
{

/**
 * @brief ElasticityOperator for time-dependent linear elasticity problems.
 *
 * This class encapsulates the setup and solution of the linear elasticity equations
 * using MFEM's parallel finite element infrastructure. It supports essential and
 * Neumann boundary conditions, body traction, and both explicit and implicit time integration.
 */
class ElasticityOperator : public SecondOrderTimeDependentOperator
{

   friend class
      ElasticitySolver; // Allow ElasticitySolver to access private members

public:
   using SecondOrderTimeDependentOperator::ImplicitSolve;
   using SecondOrderTimeDependentOperator::Mult;

   /**
    * @brief Construct a new ElasticityOperator.
    * @param mesh The parallel mesh to use.
    * @param polynomial_order The polynomial order of the finite element basis.
    * @param density The material density.
    * @param lambda The first Lamé parameter.
    * @param mu The second Lamé parameter (shear modulus).
    */
   ElasticityOperator(ParMesh *pmesh_, int order, const real_t density_,
                      const real_t lambda_, const real_t mu_, bool verbose_ = true);

   /**
    * @brief Get the parallel finite element space used by the solver.
    * @return Reference to the ParFiniteElementSpace.
    */
   ParFiniteElementSpace *GetFESpace();

   /**
    * @brief Destructor.
    */
   virtual ~ElasticityOperator();

   /**
    * @brief Set the boundaries to be fixed.
    * @param fixed_bdr Array marking the boundary attributes for fixed constraints.
    * @param component The component to apply the fixed constraint to (-1 for all components).
    */
   void AddFixedConstraint(const Array<int> &fixed_bdr, int component = -1);
   void AddFixedConstraint(int attr, int component = -1);

   /**
    * @brief Set the prescribed an analytic (time-dependent) displacement (Dirichlet BC) on specified boundary attributes.
    * API allows setting the prescribed displacement:
    * 1. Using either VectorCoefficient/VecFuncT (alias for void(const Vector &x, double t, Vector &u))
    * 2. Specifying single boundary attribute or an array of attributes.
    * 3. Velocity and acceleration should also be specified.
    */
   void AddPrescribedDisplacement(VectorCoefficient *disp, VectorCoefficient *vel,
                                  VectorCoefficient *acc, Array<int> &attr, bool own = true);
   void AddPrescribedDisplacement(VecFuncT disp, VecFuncT vel, VecFuncT acc,
                                  Array<int> &attr);
   void AddPrescribedDisplacement(VectorCoefficient *disp, VectorCoefficient *vel,
                                  VectorCoefficient *acc, int attr, bool own = true);
   void AddPrescribedDisplacement(VecFuncT disp, VecFuncT vel, VecFuncT acc,
                                  int attr);


   /**
    * @brief Set the prescribed a time-dependent velocity (Dirichlet BC) on specified boundary attributes.
    * This version assumed the velocity is prescribed from discrete velocity data, and computes the derived acceleration consistent with the time integration method used.
    */
   void AddPrescribedVelocity(GridFunction *prescribed_velocity_gf,
                              GridFunction *previous_velocity_gf, GridFunction *previous_acceleration_gf,
                              SecondOrderODESolver *time_integrator, Array<int> &attr);
   void AddPrescribedVelocity(GridFunction *prescribed_velocity_gf,
                              GridFunction *previous_velocity_gf, GridFunction *previous_acceleration_gf,
                              SecondOrderODESolver *time_integrator, int attr);


   /**
    * @brief Set the BoundaryLoad (Traction BC) on specified boundary attributes.
    * API allows setting the BoundaryLoad:
    * 1. Using either VectorCoefficient/VecFuncT (alias for void(const Vector &x, double t, Vector &u))
    * 2. Specifying single boundary attribute or an array of attributes.
    */
   void AddBoundaryLoad(VectorCoefficient *coeff, Array<int> &attr,
                        real_t scaling = 1.0, bool own = true);
   void AddBoundaryLoad(VecFuncT func, Array<int> &attr, real_t scaling = 1.0);
   void AddBoundaryLoad(VectorCoefficient *coeff, int attr, real_t scaling = 1.0,
                        bool own = true);
   void AddBoundaryLoad(VecFuncT func, int attr, real_t scaling = 1.0);

   /**
    * @brief Set the Robin boundary condition alpha * du/dt + sigma*n = g on specified boundary attributes.
    * @param alpha Scalar Robin penalty coefficient.
    * @param g     Vector coefficient for the Robin forcing (e.g. alpha*u_f + sigma_f*n in FSI).
    * @param attr  Boundary attributes marking the Robin boundary.
    * @param own   Whether the operator owns the coefficient.
    */
   void AddRobinBC(real_t alpha, VectorCoefficient *g, Array<int> &attr,
                   bool own = true);
   void AddRobinBC(real_t alpha, VectorCoefficient *g, int attr, bool own = true);

   /**
    * @brief Set the body force (volumetric load) on specified domain attributes.
    * API allows setting the BodyForce:
    * 1. Using either VectorCoefficient/VecFuncT (alias for void(const Vector &x, double t, Vector &u))
    * 2. Specifying single domain attribute or an array of attributes.
    */
   void AddBodyForce(VectorCoefficient *coeff, Array<int> &attrs, bool own = true);
   void AddBodyForce(VecFuncT func, Array<int> &attrs);
   void AddBodyForce(VectorCoefficient *coeff, int attr, bool own = true);
   void AddBodyForce(VecFuncT func, int attr);

   /**
    * @brief Assemble and initialize the system matrices and solvers.
    */
   void Setup();

   /// Set the current time.
   void SetTime(real_t time) override;

   /// Set the current time step
   void SetTimeStep(real_t dt);

   /**
    * @brief Compute the second time derivative (acceleration) for explicit time integration.
    * @param u Displacement vector.
    * @param du_dt Velocity vector.
    * @param d2udt2 Output acceleration vector.
    */
   void Mult(const Vector &u, const Vector &du_dt, Vector &d2udt2) const override;

   /**
    * @brief Solve the implicit system for the second time derivative (acceleration).
    * @param fac0 Scaling factor for the stiffness matrix.
    * @param fac1 Scaling factor for the mass matrix.
    * @param u Displacement vector.
    * @param dudt Velocity vector.
    * @param d2udt2 Output acceleration vector.
    */
   void ImplicitSolve(const real_t fac0, const real_t fac1, const Vector &u,
                      const Vector &dudt, Vector &d2udt2) override;

private:
   /**
    * @brief Update the system after mesh and/or coefficient changes.
    * For now it just updates the right-hand side vector.
    */
   void Update();

   ParMesh *pmesh;     ///< Reference to the parallel mesh.     // NOT OWNED
   const int dim;      ///< Spatial dimension.

   bool setup = false;   ///< Flag indicating if the system is set up.
   bool verbose = false; ///< Verbosity flag for debugging.

   FiniteElementCollection *fec; ///< Finite element collection.   // OWNED
   ParFiniteElementSpace *fes;   ///< Parallel finite element space. // OWNED
   int fes_truevsize;

   mutable ParGridFunction
   d2udt2_gf; ///< Grid function for second time derivative (acceleration).
   mutable ParGridFunction
   dudt_gf;   ///< Grid function for first time derivative (velocity).       CHECK: do we need that?
   mutable ParGridFunction
   u_gf;      ///< Grid function for displacement.                           CHECK: do we need that?

   ParBilinearForm *Mform = nullptr; ///< Mass bilinear form.
   ParBilinearForm *Kform = nullptr; ///< Stiffness bilinear form.
   ParLinearForm *Fform = nullptr;   ///< Right-hand side linear form.

   ConstantCoefficient lambda_coef, mu_coef,
                       density_coef; ///< Material coefficients.

   OperatorHandle opMfull, opK, opM, opMe;     ///< System matrix handles.
   HypreParMatrix *A = nullptr,
                   *Ae = nullptr; ///< System matrices for implicit solve.
   real_t A_fac0 = -1.0, A_fac1 = -1.0;       ///< fac0/fac1 that A was built with

   CGSolver M_solver;      ///< Mass matrix solver.
   Solver *M_pc = nullptr; ///< Preconditioner for mass solver.
   GMRESSolver A_solver;   ///< System solver for implicit integration.
   HypreBoomerAMG *A_pc;   ///< Preconditioner for system solver.

   mutable Vector z, F; ///< Solution and work vectors.

   const real_t beta = 0.25; ///< Newmark-beta parameter.
   const real_t gamma = 0.5; ///< Newmark-gamma parameter.
   real_t current_dt = -1.0; ///< Current time step.
   const real_t density;     ///< Material density.

   // Boundary conditions
   int max_bdr_attributes;
   int max_domain_attributes;
   std::vector<FixedConstraint>
   fixed_constraints;                  ///< Fixed constraints.
   std::vector<MultiVectorCoeffContainer>
   prescribed_displacements; ///< Prescribed displacements (Dirichlet conditions).
   std::vector<VectorCoeffContainer>
   prescribed_velocities;         ///< Prescribed velocities (Dirichlet conditions on velocity).
   std::vector<VectorCoeffContainer>
   boundary_loads;                ///< Boundary loads (Neumann conditions).
   std::vector<VectorCoeffContainer>
   body_forces;                   ///< Body forces (on domain).
   std::vector<RobinBC_T>
   robin_bcs;                                ///< Robin BCs: alpha * du/dt + sigma*n = g.

   ParBilinearForm *Mgamma_form =
      nullptr; ///< Boundary bilinear form for Robin mass M_gamma = int_Gamma alpha phi_i phi_j.
   OperatorHandle opMgamma;              ///< Assembled Robin boundary mass matrix.

   Array<int>
   tmp_bdr_attrs;                         ///< Temporary boundary attributes.
   Array<int>
   tmp_domain_attrs;                      ///< Temporary domain attributes.
   Array<int>
   fixed_attrs;                           ///< Fixed boundary attributes.
   Array<int>
   fixed_attrs_xyz;                       ///< Fixed boundary attributes for each component.
   Array<int>
   prescribed_disp_attr;                  ///< Prescribed displacement attributes.
   Array<int>
   prescribed_vel_attr;                   ///< Prescribed velocity attributes.
   Array<int> boundary_load_attr;                    ///< Boundary load attributes.
   Array<int>
   body_force_attr;                       ///< Body force attributes.  --> Domain attributes

   Array<int>
   fixed_tdof_list;                       ///< List of true DOFs for fixed constraint (subset ess_tdof_list).
   Array<int>
   prescribed_disp_tdof_list;             ///< List of true DOFs for prescribed displacements (subset ess_tdof_list).
   Array<int>
   prescribed_vel_tdof_list;              ///< List of true DOFs for prescribed velocities (subset ess_tdof_list).
   Array<int>
   ess_tdof_list;                         ///< List of essential DOFs for essential BCs.
};

}
}

#endif // ELASTICITY_HPP

// NOTES:
//
// 1. We can extend the API to allow to remove BCs from a boundary (e.g. if you prescribe a displacement and want to release after)
//    This would require to iterate over each element in the std::vector, check if the attribute/s match, and then remove the element with erase.
//    We need to check each individual element in the attribute array, as specific boundaries might have been passed as part of a bigger Array<int>,
//    and we need to ensure that we only remove the specific boundary attributes that match the ones passed to the function.
//
// 2. For a prescribed displacement, without a velocity and acceleration provided, we need to compute an approximation for the velocity and acceleration.
//    We can add this type of bcs to a eparate category, so that this approximatino is not computed for fixed constraints, or prescribed displacements
//    with analytic velocity and acceleration provided.
//
//
// Take care of derivatives approximation for time dependent essential boundary conditions
// Maybe we should just use a d2udt2_approx vector that is owned by the ElasticityOperator.
// Don't use solver->GetState() here, as it will not guarantee to be used in every Mult and ImplicitSolve call.

// 1.  Vector &d2udt2_approx = op->GetDerivativeApproximation(); We can make it zero everywhere, besides at essential TDofs.
//  Vector tmp_d2udt2, tmp_dudt;
// 2. ComputeDerivativesApproximation(x, dxdt, dt, tmp_dudt, tmp_d2udt2);
// 3. Set the acceleration and velocity to zero for FixedConstraint or TimeIndependent essential BCs
// auto &TI_ess_tdof_list = op->GetTIEssentialTDofList();
// auto &TD_ess_tdof_list = op->GetTDEssentialTDofList();
// auto &fixed_tdof_list = op->GetFixedTDofList();
// d2udt2_approx.SetSubVector(0.0, TI_ess_tdof_list);
// d2udt2_approx.SetSubVector(0.0, fixed_tdof_list);
// d2udt2_approx.SetSubVector(tmp_d2udt2, TD_ess_tdof_list);
// dxdt.SetSubVector(0.0, TI_ess_tdof_list);
// dxdt.SetSubVector(0.0, fixed_tdof_list);
// dxdt.SetSubVector(tmp_dudt, TD_ess_tdof_list);