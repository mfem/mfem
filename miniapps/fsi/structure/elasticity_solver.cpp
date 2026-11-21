#include "elasticity_solver.hpp" // Include the full definition for ElasticityOperator

using namespace mfem;
using namespace mfem::elasticity;

ElasticitySolver::ElasticitySolver(int solver_type, ParMesh *pmesh_, int order, const real_t density_, const real_t lambda_, const real_t mu_, bool verbose_)
    : op(new ElasticityOperator(pmesh_, order, density_, lambda_, mu_, verbose_))
{
   // If solver_type < 14 the we can use select, otherwise we do it manually
   if (solver_type < 15)
      solver.reset(SecondOrderODESolver::Select(solver_type));
   else if (solver_type == 15)
      solver.reset(new HHTAlphaSolver());
   else if (solver_type == 16)
      solver.reset(new WBZAlphaSolver());
   else {
      mfem::err << "Unknown ODE solver type: " << solver_type << "\n";
      mfem::err << SecondOrderODESolver::Types << "\n";
      mfem::err << "  15: HHTAlphaSolver\n";
      mfem::err << "  16: WBZAlphaSolver\n";
      mfem_error();
   }

   // Create state data vectors for the solver
   // We keep it 2 for now (second order solvers are either order 1 or 2), but we should adjust based on the solver type (maybe SecondOrderODESolver can store the order internally)
   // The acceleration can be accessed using solver->GetState(), which returns an ODEStateDataVector.
   int state_size = 2; 
   dudt_prev = new ODEStateDataVector(state_size);
   u_prev = new ODEStateDataVector(state_size); 
}


ElasticitySolver::~ElasticitySolver()
{
   // Cleanup the operator
   delete op;
   op = nullptr;

   // Cleanup the state data vectors
   delete dudt_prev;
   dudt_prev = nullptr;
   delete u_prev;
   u_prev = nullptr;
}

HYPRE_BigInt
ElasticitySolver::GetProblemSize()
{
   return op->GetFESpace()->GlobalTrueVSize();
}

void ElasticitySolver::Setup()
{
   // Setup the operator
   op->Setup();

   // Init the stats and ode solver
   Vector tmp(op->GetFESpace()->GetTrueVSize()); tmp = 0.0;
   auto mem_type = GetMemoryType(op->GetMemoryClass());
   u_prev->SetSize(op->Width(), mem_type);     // NOTE: we can just use the SubVector associated to the BCs instead of the whole vector
   dudt_prev->SetSize(op->Width(), mem_type);
   dudt_prev->Append(tmp);
   u_prev->Append(tmp);

   solver->Init(*op);
}

Vector *ElasticitySolver::GetLatestState()
{
   auto& state = static_cast<ODEStateDataVector&>(solver->GetState());
   if (state.Size() > 0)
   {
      return &state[0];
   }
   else
   {
      return nullptr; // No state available
   }
}

void ElasticitySolver::InitializeState(const Vector &x, const Vector &dxdt, real_t t, real_t dt)
{
   // a_n = M^{-1}(F(t_n) - K x_n - ...), computed once from the t_n data (instead of lazily inside the
   // first Step(), when the boundary data may already hold t_{n+1} interface values).
   op->SetTimeStep(dt); // needed by DerivedAccelerationCoefficient
   op->SetTime(t);
   auto &state = static_cast<ODEStateDataVector&>(solver->GetState());
   state.Reset();
   op->Mult(x, dxdt, state[0]);
   state.Increment();
}

void ElasticitySolver::ResetState( Vector *state_prev )
{
   // If state is not provided, we assume this was a reset after the first step
   // hence we need to reset the ODEStateDataVector
   if (state_prev == nullptr)
   {
      auto& state_vector = static_cast<ODEStateDataVector&>(solver->GetState());
      state_vector.Reset();
      //solver->Init(*op);  // If we use Init this will also recreate the interal vectors of the ODESolver 
      return;
   }

   // Reset the state to the previous state
   MFEM_VERIFY(state_prev != GetLatestState(),
               "ResetState: state_prev aliases the live ODE state (GetLatestState()), which Step() overwrites; pass a copy.");
   solver->GetState().Set(0, *state_prev);
}


void ElasticitySolver::Step(Vector &x, Vector &dxdt, real_t &t, real_t &dt, bool provisional)
{
   // Set operator time to the current step time.
   // The ODE solver updates stage/end times internally (e.g. t+dt in implicit solves).
   // Keeping the operator at t here ensures the first-call acceleration initialization
   // (state[0] = f.Mult(x, dxdt, ...)) is evaluated at the correct time level.
   // Only needed for the first-call a_n initialisation (f->Mult at time t); the ODE solver sets t+dt itself
   // before ImplicitSolve, so otherwise this would be an extra boundary-load assembly per step.
   if (solver->GetState().Size() == 0) { op->SetTime(t); }

   // Set operator timestep 
   op->SetTimeStep(dt);

   // Solve current step
   solver->Step(x, dxdt, t, dt);

   // Enforce essential boundary conditions (shouldn't be needed, but just to avoid round-off errors)
   op->u_gf.SetFromTrueDofs(x);
   op->dudt_gf.SetFromTrueDofs(dxdt);
   for (auto &prescribed_disp : op->prescribed_displacements)
   {
      op->u_gf.ProjectBdrCoefficient(*prescribed_disp.GetCoeff(0), prescribed_disp.attr);
      op->dudt_gf.ProjectBdrCoefficient(*prescribed_disp.GetCoeff(1), prescribed_disp.attr);
   }
   for (auto &prescribed_vel : op->prescribed_velocities)
   {
      auto derived_acc_coef = dynamic_cast<DerivedAccelerationCoefficient *>(prescribed_vel.original_coeff);
      if (derived_acc_coef)
      {
         op->dudt_gf.ProjectBdrCoefficient(*derived_acc_coef->GetPrescribedVelocityCoefficient(), prescribed_vel.attr);
      }
   }

   op->u_gf.GetTrueDofs(x);
   op->dudt_gf.GetTrueDofs(dxdt);
   x.SetSubVector(op->fixed_tdof_list, 0.0);     // fixed_tdof_list holds true-dof indices
   dxdt.SetSubVector(op->fixed_tdof_list, 0.0);

   // Update the time history
   if (provisional)
   {
      t -= dt; // If provisional, we don't update the history, but we need to adjust the time to avoid double counting
      return;
   }

   UpdateTimestepHistory(x, dxdt);
}


void ElasticitySolver::UpdateTimestepHistory(Vector &x, Vector &dxdt)
{
   // Update the time history for the solver
   u_prev->Append(x);
   dudt_prev->Append(dxdt);
}


// Expose the interface for adding bcs to the operator
/**
 * @brief Set the boundaries to be fixed.
 * @param fixed_bdr Array marking the boundary attributes for fixed constraints.
 * @param component The component to apply the fixed constraint to (-1 for all components).
 */
void ElasticitySolver::AddFixedConstraint(const Array<int> &fixed_bdr, int component)
{
   op->AddFixedConstraint(fixed_bdr, component);
}

void ElasticitySolver::AddFixedConstraint(int attr, int component)
{
   op->AddFixedConstraint(attr, component);
}

void ElasticitySolver::AddPrescribedDisplacement(VectorCoefficient *disp, VectorCoefficient *vel, VectorCoefficient *acc, Array<int> &attr, bool own)
{
   op->AddPrescribedDisplacement(disp, vel, acc, attr, own);
}

void ElasticitySolver::AddPrescribedDisplacement(VecFuncT disp, VecFuncT vel, VecFuncT acc, Array<int> &attr)
{
   op->AddPrescribedDisplacement(disp, vel, acc, attr);
}

void ElasticitySolver::AddPrescribedDisplacement(VectorCoefficient *disp, VectorCoefficient *vel, VectorCoefficient *acc, int attr, bool own)
{
   op->AddPrescribedDisplacement(disp, vel, acc, attr, own);
}

void ElasticitySolver::AddPrescribedDisplacement(VecFuncT disp, VecFuncT vel, VecFuncT acc, int attr)
{
   op->AddPrescribedDisplacement(disp, vel, acc, attr);
}

void ElasticitySolver::AddPrescribedVelocity(GridFunction *prescribed_velocity_gf,
                                              GridFunction *previous_velocity_gf,
                                              GridFunction *previous_acceleration_gf,
                                              Array<int> &attr)
{
   op->AddPrescribedVelocity(prescribed_velocity_gf, previous_velocity_gf, previous_acceleration_gf, solver.get(), attr);
}

void ElasticitySolver::AddPrescribedVelocity(GridFunction *prescribed_velocity_gf,
                                              GridFunction *previous_velocity_gf,
                                              GridFunction *previous_acceleration_gf,
                                              int attr)
{
   op->AddPrescribedVelocity(prescribed_velocity_gf, previous_velocity_gf, previous_acceleration_gf, solver.get(), attr);
}

void ElasticitySolver::AddBoundaryLoad(VectorCoefficient *coeff, Array<int> &attr, real_t scaling, bool own)
{
   op->AddBoundaryLoad(coeff, attr, scaling, own);
}

void ElasticitySolver::AddBoundaryLoad(VecFuncT func, Array<int> &attr, real_t scaling)
{
   op->AddBoundaryLoad(func, attr, scaling);
}

void ElasticitySolver::AddBoundaryLoad(VectorCoefficient *coeff, int attr, real_t scaling, bool own)
{
   op->AddBoundaryLoad(coeff, attr, scaling, own);
}

void ElasticitySolver::AddBoundaryLoad(VecFuncT func, int attr, real_t scaling)
{
   op->AddBoundaryLoad(func, attr, scaling);
}

/**
 * @brief Set the body force (volumetric load) on specified domain attributes.
 * API allows setting the BodyForce:
 * 1. Using either VectorCoefficient/VecFuncT (alias for void(const Vector &x, double t, Vector &u))
 * 2. Specifying single domain attribute or an array of attributes.
 */
void ElasticitySolver::AddBodyForce(VectorCoefficient *coeff, Array<int> &attrs, bool own)
{
   op->AddBodyForce(coeff, attrs, own);
}

void ElasticitySolver::AddBodyForce(VecFuncT func, Array<int> &attrs)
{
   op->AddBodyForce(func, attrs);
}

void ElasticitySolver::AddBodyForce(VectorCoefficient *coeff, int attr, bool own)
{
   op->AddBodyForce(coeff, attr, own);
}

void ElasticitySolver::AddBodyForce(VecFuncT func, int attr)
{
   op->AddBodyForce(func, attr);
}

void ElasticitySolver::AddRobinBC(real_t alpha, VectorCoefficient *g, Array<int> &attr, bool own)
{
   op->AddRobinBC(alpha, g, attr, own);
}

void ElasticitySolver::AddRobinBC(real_t alpha, VectorCoefficient *g, int attr, bool own)
{
   op->AddRobinBC(alpha, g, attr, own);
}
