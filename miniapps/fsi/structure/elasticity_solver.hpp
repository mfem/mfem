#ifndef ELASTICITY_SOLVER_HPP
#define ELASTICITY_SOLVER_HPP

#include "elasticity_operator.hpp" // Include the ElasticityOperator definition

namespace mfem
{

   namespace elasticity
   {
      /** @brief A wrapper class for second order ODE solvers for elasticity problems.
          This class holds a pointer to a SecondOrderODESolver, which is selected at runtime,
          and delegates all solver-related calls to it.
       */
      class ElasticitySolver
      {
      public:
         /** @brief Construct an ElasticitySolver.
             @param solver_type The type of SecondOrderODESolver to use (see ode.hpp).
             @param op_ The ElasticityOperator to use (not owned).
          */
         ElasticitySolver(int solver_type, ParMesh *pmesh_, int order, const real_t density_, const real_t lambda_, const real_t mu_, bool verbose_ = true);

         /// Destructor
         virtual ~ElasticitySolver();

         /// Setup the underlying solver with the operator.
         void Setup();

         Vector *GetLatestState();

         /// Reset the internal state of the solver. (acceleration)
         void ResetState(Vector *state_prev = nullptr);

         /// Compute the initial acceleration a_n = f(x_n, dxdt_n, t_n) and store it as the ODE state.
         void InitializeState(const Vector &x, const Vector &dxdt, real_t t, real_t dt);

         /// Update the time history for the solver.
         void UpdateTimestepHistory(Vector &x, Vector &dxdt);
         
         /// Perform a single time step.
         void Step(Vector &x, Vector &dxdt, real_t &t, real_t &dt, bool provisional = false);

         /// Get the FE space used by the operator.
         ParFiniteElementSpace *GetFESpace() { return op->GetFESpace(); }

         /// Access the underlying solver.
         SecondOrderODESolver *GetSolver() { return solver.get(); }

         HYPRE_BigInt GetProblemSize();

         // Expose the interface for adding bcs to the operator
         void AddFixedConstraint(const Array<int> &fixed_bdr, int component = -1);
         void AddFixedConstraint(int attr, int component = -1);

         void AddPrescribedDisplacement(VectorCoefficient *disp, VectorCoefficient *vel, VectorCoefficient *acc, Array<int> &attr, bool own = true);
         void AddPrescribedDisplacement(VecFuncT disp, VecFuncT vel, VecFuncT acc, Array<int> &attr);
         void AddPrescribedDisplacement(VectorCoefficient *disp, VectorCoefficient *vel, VectorCoefficient *acc, int attr, bool own = true);
         void AddPrescribedDisplacement(VecFuncT disp, VecFuncT vel, VecFuncT acc, int attr);

         void AddPrescribedVelocity(GridFunction *prescribed_velocity_gf,
                                    GridFunction *previous_velocity_gf,
                                    GridFunction *previous_acceleration_gf,
                                    Array<int> &attr);
         void AddPrescribedVelocity(GridFunction *prescribed_velocity_gf,
                                    GridFunction *previous_velocity_gf,
                                    GridFunction *previous_acceleration_gf,
                                    int attr);

         void AddBoundaryLoad(VectorCoefficient *coeff, Array<int> &attr, real_t scaling = 1.0, bool own = true);
         void AddBoundaryLoad(VecFuncT func, Array<int> &attr, real_t scaling = 1.0);
         void AddBoundaryLoad(VectorCoefficient *coeff, int attr, real_t scaling = 1.0, bool own = true);
         void AddBoundaryLoad(VecFuncT func, int attr, real_t scaling = 1.0);

         void AddBodyForce(VectorCoefficient *coeff, Array<int> &attrs, bool own = true);
         void AddBodyForce(VecFuncT func, Array<int> &attrs);
         void AddBodyForce(VectorCoefficient *coeff, int attr, bool own = true);
         void AddBodyForce(VecFuncT func, int attr);

         void AddRobinBC(real_t alpha, VectorCoefficient *g, Array<int> &attr, bool own = true);
         void AddRobinBC(real_t alpha, VectorCoefficient *g, int attr, bool own = true);

      private:
         std::unique_ptr<SecondOrderODESolver> solver;
         ElasticityOperator *op;                                          //< OWNED
         ODEStateDataVector *dudt_prev;   // State data for velocity      //< OWNED
         ODEStateDataVector *u_prev;      // State data for displacement  //< OWNED

         void ComputeDerivativesApproximation(const Vector &u, const Vector &du_dt, real_t dt);
      };

   } // namespace elasticity

} // namespace mfem

#endif // ELASTICITY_SOLVER_HPP

