#include "elasticity_operator.hpp"
#include <cstdio>

using namespace mfem;
using namespace mfem::elasticity;

ElasticityOperator::ElasticityOperator(ParMesh *pmesh_, int order,
                                       const real_t density_, const real_t lambda_, const real_t mu_, bool verbose_)
   : pmesh(pmesh_),
     dim(pmesh_->Dimension()),
     density(density_),
     verbose(verbose_)
{
   // Create the finite element space and set height/width of Operator
   fec = new H1_FECollection(order, dim);
   fes = new ParFiniteElementSpace(pmesh_, fec, dim);

   fes_truevsize  = fes->GetTrueVSize();
   this->height = fes_truevsize;
   this->width = fes_truevsize;

   // elasticity_operator.cpp, in the constructor
   HYPRE_BigInt global_tdofs =
      fes->GlobalTrueVSize();  // collective — all ranks must call this
   if (Mpi::Root() && verbose)
   {
      printf("Elasticity #tdofs: %lld\n", global_tdofs);
   }

   // Set Coefficients
   density_coef.constant = density_;

   mu_coef.constant = mu_;
   lambda_coef.constant = lambda_;

   // Initialize bcs
   max_bdr_attributes = pmesh->bdr_attributes.Max();
   max_domain_attributes = pmesh->attributes.Max();
   fixed_attrs.SetSize(max_bdr_attributes); fixed_attrs = 0;
   fixed_attrs_xyz.SetSize(max_bdr_attributes); fixed_attrs_xyz = 0;
   prescribed_disp_attr.SetSize(max_bdr_attributes); prescribed_disp_attr = 0;
   boundary_load_attr.SetSize(max_bdr_attributes); boundary_load_attr = 0;
   body_force_attr.SetSize(max_domain_attributes); body_force_attr = 0;
   tmp_bdr_attrs.SetSize(max_bdr_attributes); tmp_bdr_attrs = 0;
   tmp_domain_attrs.SetSize(max_domain_attributes); tmp_domain_attrs = 0;

   // Initialize vectors
   F.SetSize(fes_truevsize);
   z.SetSize(fes_truevsize);

   // Initialize gfs
   d2udt2_gf.SetSpace(fes); d2udt2_gf = 0.0;
   dudt_gf.SetSpace(fes); dudt_gf = 0.0;
   u_gf.SetSpace(fes); u_gf = 0.0;
}

ParFiniteElementSpace *ElasticityOperator::GetFESpace()
{
   return fes;
}



void ElasticityOperator::AddFixedConstraint(const Array<int> &attr,
                                            int component)
{
   MFEM_ASSERT(attr.Size() >= max_bdr_attributes,
               "Size of attributes array does not match mesh attributes.");

   // Append to the list of fixed constraints
   fixed_constraints.emplace_back(attr, component);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_VERIFY(!(attr[i] && (prescribed_disp_attr[i] || fixed_attrs[i] ||
                                (component < 0 && boundary_load_attr[i]))),
                  "Duplicate boundary definition detected (attribute " << i + 1 << ").");
      if (attr[i] == 1 && component < 0)
      {
         fixed_attrs[i] = 1; // Mark the attribute as fixed
      }
      else if (attr[i] == 1 && component >= 0)
      {
         fixed_attrs_xyz[i] = 1; // Mark the attribute as fixed for a specific component
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out <<
                "ElasticitySolver:: Adding FixedConstraint to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      if (component >= 0)
      {
         mfem::out << " for component " << component;
      }
      mfem::out << std::endl;
   }
}

void ElasticityOperator::AddFixedConstraint(int attr, int component)
{
   tmp_bdr_attrs = 0;
   tmp_bdr_attrs[attr - 1] = 1; // Set the specific attribute
   AddFixedConstraint(tmp_bdr_attrs, component);
}


void ElasticityOperator::AddPrescribedDisplacement(VectorCoefficient *disp,
                                                   VectorCoefficient *vel, VectorCoefficient *acc, Array<int> &attr, bool own)
{
   MFEM_ASSERT(attr.Size() >= max_bdr_attributes,
               "Size of attributes array does not match mesh attributes.");

   // Append to the list of prescribed displacements
   std::vector<VectorCoefficient *> coeffs = {disp, vel, acc};
   std::vector<real_t> scaling = {1.0, 1.0, 1.0};
   prescribed_displacements.emplace_back(attr, coeffs, scaling, own);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_VERIFY(!(attr[i] && (prescribed_disp_attr[i] || fixed_attrs[i] ||
                                boundary_load_attr[i])),
                  "Duplicate boundary definition detected (attribute " << i + 1 << ").");
      if (attr[i] == 1)
      {
         prescribed_disp_attr[i] = 1; // Mark the attribute as prescribed displacement
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out <<
                "ElasticitySolver:: Adding PrescribedDisplacement to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void ElasticityOperator::AddPrescribedDisplacement(VecFuncT disp, VecFuncT vel,
                                                   VecFuncT acc, Array<int> &attr)
{
   AddPrescribedDisplacement(new VectorFunctionCoefficient(pmesh->Dimension(),
                                                           disp),
                             new VectorFunctionCoefficient(pmesh->Dimension(), vel),
                             new VectorFunctionCoefficient(pmesh->Dimension(), acc),
                             attr, true);
}

void ElasticityOperator::AddPrescribedDisplacement(VectorCoefficient *disp,
                                                   VectorCoefficient *vel, VectorCoefficient *acc, int attr, bool own)
{
   tmp_bdr_attrs = 0;
   tmp_bdr_attrs[attr - 1] = 1; // Set the specific attribute
   AddPrescribedDisplacement(disp, vel, acc, tmp_bdr_attrs, own);
}

void ElasticityOperator::AddPrescribedDisplacement(VecFuncT disp, VecFuncT vel,
                                                   VecFuncT acc, int attr)
{
   AddPrescribedDisplacement(new VectorFunctionCoefficient(pmesh->Dimension(),
                                                           disp),
                             new VectorFunctionCoefficient(pmesh->Dimension(), vel),
                             new VectorFunctionCoefficient(pmesh->Dimension(), acc),
                             attr, true);
}


void ElasticityOperator::AddPrescribedVelocity(GridFunction
                                               *prescribed_velocity_gf, GridFunction *previous_velocity_gf,
                                               GridFunction *previous_acceleration_gf, SecondOrderODESolver *time_integrator,
                                               Array<int> &attr)
{
   // Create a DerivedAccelerationCoefficient to compute the consistent acceleration for the prescribed velocity
   auto derived_acc_coef = new DerivedAccelerationCoefficient(
      prescribed_velocity_gf, previous_velocity_gf, previous_acceleration_gf,
      time_integrator);

   // Add the prescribed velocity to the coeff container
   prescribed_velocities.emplace_back(attr, derived_acc_coef, 1.0, true);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_VERIFY(!(attr[i] && (prescribed_disp_attr[i] || fixed_attrs[i] ||
                                boundary_load_attr[i])),
                  "Duplicate boundary definition detected (attribute " << i + 1 << ").");
      if (attr[i] == 1)
      {
         prescribed_disp_attr[i] =
            1; // Mark the attribute as prescribed displacement (since velocity is the time derivative of displacement)
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out <<
                "ElasticitySolver:: Adding PrescribedVelocity to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

}

void ElasticityOperator::AddPrescribedVelocity(GridFunction
                                               *prescribed_velocity_gf, GridFunction *previous_velocity_gf,
                                               GridFunction *previous_acceleration_gf, SecondOrderODESolver *time_integrator,
                                               int attr)
{
   tmp_bdr_attrs = 0;
   tmp_bdr_attrs[attr - 1] = 1; // Set the specific attribute
   AddPrescribedVelocity(prescribed_velocity_gf, previous_velocity_gf,
                         previous_acceleration_gf, time_integrator, tmp_bdr_attrs);
}


void ElasticityOperator::AddBoundaryLoad(VectorCoefficient *coeff,
                                         Array<int> &attr, real_t scaling, bool own)
{
   MFEM_ASSERT(attr.Size() >= max_bdr_attributes,
               "Size of attributes array does not match mesh attributes.");

   // Append to the list of boundary loads
   boundary_loads.emplace_back(attr, coeff, scaling, own);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_VERIFY(!(attr[i] && (prescribed_disp_attr[i] || fixed_attrs[i] ||
                                boundary_load_attr[i])),
                  "Duplicate boundary definition detected (attribute " << i + 1 << ").");
      if (attr[i] == 1)
      {
         boundary_load_attr[i] = 1;
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "ElasticitySolver:: Adding BoundaryLoad to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void ElasticityOperator::AddBoundaryLoad(VecFuncT func, Array<int> &attr,
                                         real_t scaling)
{
   AddBoundaryLoad(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr,
                   scaling, true);
}

void ElasticityOperator::AddBoundaryLoad(VectorCoefficient *coeff, int attr,
                                         real_t scaling, bool own)
{
   tmp_bdr_attrs = 0;
   tmp_bdr_attrs[attr - 1] = 1; // Set the specific attribute
   AddBoundaryLoad(coeff, tmp_bdr_attrs, scaling, own);
}

void ElasticityOperator::AddBoundaryLoad(VecFuncT func, int attr,
                                         real_t scaling)
{
   AddBoundaryLoad(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr,
                   scaling, true);
}

void ElasticityOperator::AddRobinBC(real_t alpha, VectorCoefficient *g,
                                    Array<int> &attr, bool own)
{
   MFEM_ASSERT(attr.Size() >= max_bdr_attributes,
               "Size of attributes array does not match mesh attributes.");

   robin_bcs.emplace_back(attr, alpha, g, own);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "ElasticitySolver:: Adding Robin BC (alpha=" << alpha <<
                ") to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1) { mfem::out << i << " "; }
      }
      mfem::out << std::endl;
   }
}

void ElasticityOperator::AddRobinBC(real_t alpha, VectorCoefficient *g,
                                    int attr, bool own)
{
   tmp_bdr_attrs = 0;
   tmp_bdr_attrs[attr - 1] = 1;
   AddRobinBC(alpha, g, tmp_bdr_attrs, own);
}

void ElasticityOperator::AddBodyForce(VectorCoefficient *coeff,
                                      Array<int> &attr, bool own)
{
   MFEM_ASSERT(attr.Size() >= max_domain_attributes,
               "Size of attributes array does not match mesh attributes.");

   // Append to the list of body forces
   body_forces.emplace_back(attr, coeff, own);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((body_force_attr[i] &&
                   attr[i]) == 0, "Duplicate domain definition detected.");
      if (attr[i] == 1)
      {
         body_force_attr[i] = 1;
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "ElasticitySolver:: Adding BodyForce to domain attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i  << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void ElasticityOperator::AddBodyForce(VecFuncT func, Array<int> &attr)
{
   AddBodyForce(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr,
                true);
}

void ElasticityOperator::AddBodyForce(VectorCoefficient *coeff, int attr,
                                      bool own)
{
   tmp_domain_attrs = 0;
   tmp_domain_attrs[attr - 1] = 1; // Set the specific attribute
   AddBodyForce(coeff, tmp_domain_attrs, own);
}

void ElasticityOperator::AddBodyForce(VecFuncT func, int attr)
{
   AddBodyForce(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr,
                true);
}


/// SETUP
void ElasticityOperator::Setup()
{
   Array<int> empty;

   //// 1. Extract the essential dof list
   // For now we include only fixed constraints, but we can extend to prescribed displacements, velocity and acceleration
   for ( auto &fixed_constraint : fixed_constraints )
   {
      Array<int> tmp_tdof_list;
      fes->GetEssentialTrueDofs(fixed_constraint.attr, tmp_tdof_list,
                                fixed_constraint.component);
      fixed_tdof_list.Append(tmp_tdof_list);
   }

   for ( auto &prescribed_disp : prescribed_displacements )
   {
      Array<int> tmp_tdof_list;
      fes->GetEssentialTrueDofs(prescribed_disp.attr,
                                tmp_tdof_list); // -1 for all components
      prescribed_disp_tdof_list.Append(tmp_tdof_list);
   }

   for ( auto &prescribed_vel : prescribed_velocities )
   {
      Array<int> tmp_tdof_list;
      fes->GetEssentialTrueDofs(prescribed_vel.attr,
                                tmp_tdof_list); // -1 for all components
      prescribed_vel_tdof_list.Append(tmp_tdof_list);
   }

   ess_tdof_list.Append(fixed_tdof_list);
   ess_tdof_list.Append(prescribed_disp_tdof_list);
   ess_tdof_list.Append(prescribed_vel_tdof_list);
   ess_tdof_list.Sort();
   ess_tdof_list.Unique();

   //// 2. Assemble Bilinear forms
   int skip_zeros = 0;
   Mform = new ParBilinearForm(fes);
   Mform->AddDomainIntegrator(new VectorMassIntegrator);
   Mform->Assemble(skip_zeros);
   Mform->FormSystemMatrix(empty, opMfull);

   opM.Reset(new HypreParMatrix(*opMfull.As<HypreParMatrix>()));
   opMe.Reset(opM.As<HypreParMatrix>()->EliminateRowsCols(ess_tdof_list));

   Kform = new ParBilinearForm(fes);
   Kform->AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
   Kform->Assemble(skip_zeros);
   Kform->FormSystemMatrix(empty, opK);

   /// 3. Assemble linear form
   Fform = new ParLinearForm(fes);

   // 3.1 Boundary loads
   for (auto &boundary_load : boundary_loads)
   {
      Fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(
                                      *boundary_load.coeff), boundary_load.attr);
   }

   // 3.2 Body acceleration (volumetric terms)
   for (auto &body_force : body_forces)
   {
      Fform->AddDomainIntegrator(new VectorDomainLFIntegrator(*body_force.coeff),
                                 body_force.attr);
   }

   // 3.3 Robin forcing (g term)
   for (auto &robin_bc : robin_bcs)
   {
      Fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*robin_bc.g),
                                   robin_bc.attr);
   }

   Fform->Assemble();
   Fform->ParallelAssemble(
      F); // Might be redundant, but ensures F is assembled even if SetTime is not called before Mult.

   // 4. Robin boundary mass matrix M_gamma = int_Gamma alpha phi_i phi_j dGamma
   if (!robin_bcs.empty())
   {
      Mgamma_form = new ParBilinearForm(fes);
      for (auto &robin_bc : robin_bcs)
      {
         Mgamma_form->AddBoundaryIntegrator(new VectorMassIntegrator(
                                               *robin_bc.alpha_coef), robin_bc.attr);
      }
      Mgamma_form->Assemble(skip_zeros);
      Mgamma_form->FormSystemMatrix(empty, opMgamma);
   }

   // Setup solvers and preconditioners
   M_pc = new HypreSmoother();
   static_cast<HypreSmoother *>(M_pc)->SetType(
      HypreSmoother::Jacobi); // See hypre.hpp for more options

   M_solver = CGSolver(fes->GetComm());
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-12);
   M_solver.SetMaxIter(500);
   M_solver.SetPreconditioner(*M_pc);
   M_solver.SetOperator(*opM);

   A_pc = new HypreBoomerAMG();
   bool amg_elast = true;
   A_pc->SetPrintLevel(0);
   if (amg_elast)
   {
      common_fsi_utils::SetVectorAMGOptions(*A_pc, fes);
   }
   else
   {
      A_pc->SetSystemsOptions(pmesh->Dimension());
   }

   A_solver = GMRESSolver(fes->GetComm());
   A_solver.iterative_mode = true;
   A_solver.SetRelTol(1e-8);
   A_solver.SetPrintLevel(0);
   A_solver.SetMaxIter(1000);
   A_solver.SetPreconditioner(*A_pc);

   setup = true;
   tmp_bdr_attrs.DeleteAll();
   tmp_domain_attrs.DeleteAll();
}


void ElasticityOperator::SetTime( real_t time )
{
   TimeDependentOperator::SetTime(time);

   // Set the time for the coefficients
   // At the moment this doesn't do anything because they're constant, but it may be useful in the future.
   density_coef.SetTime(time);
   mu_coef.SetTime(time);
   lambda_coef.SetTime(time);

   // Set the time for bcs
   for (auto &boundary_load : boundary_loads)
   {
      boundary_load.coeff->SetTime(time);
   }

   // Set the time for prescribed displacements
   for (auto &prescribed_disp : prescribed_displacements)
   {
      prescribed_disp.GetCoeff(0)->SetTime(time); // displacement
      prescribed_disp.GetCoeff(1)->SetTime(time); // velocity
      prescribed_disp.GetCoeff(2)->SetTime(time); // acceleration
   }

   for (auto &prescribed_vel : prescribed_velocities)
   {
      prescribed_vel.coeff->SetTime(time); // velocity
   }

   // Set the time for body forces
   for (auto &body_force : body_forces)
   {
      body_force.coeff->SetTime(time);
   }

   // Set the time for Robin loads
   for (auto &robin_bc : robin_bcs)
   {
      robin_bc.SetTime(time);
   }

   // Update the operator
   this->Update();
}

void ElasticityOperator::SetTimeStep( real_t dt )
{
   current_dt = dt;

   for (auto &prescribed_vel : prescribed_velocities)
   {
      auto derived_acc_coef = dynamic_cast<DerivedAccelerationCoefficient *>
                              (prescribed_vel.original_coeff);
      if (derived_acc_coef)
      {
         derived_acc_coef->SetTimeStep(dt);
      }
   }
}

void ElasticityOperator::Update()
{
   // The system matrix A (and its BoomerAMG setup) depends only on fac0/fac1, i.e. on dt: it is rebuilt in
   // ImplicitSolve when those change, not at every SetTime().
   Fform->Update();
   Fform->Assemble();
   Fform->ParallelAssemble(F);
}

// --------------------------------------------------------------------------------- //
//// ----- Mult() and ImplicitSolve() methods -----
// --------------------------------------------------------------------------------- //

void ElasticityOperator::Mult(const Vector &u, const Vector &du_dt,
                              Vector &d2udt2) const
{
   // Compute rhs
   opK->Mult(u, z);
   z.Neg();
   z.Add(1.0, F);

   z *= (1.0 / density);

   // Robin boundary damping: -M_gamma * du_dt / density (g is already in F)
   if (!robin_bcs.empty())
   {
      opMgamma->AddMult(du_dt, z, -1.0 / density);
   }

   // Apply bcs
   d2udt2 = 0.0;
   d2udt2_gf = 0.0;
   for (auto &prescribed_disp : prescribed_displacements)
   {
      d2udt2_gf.ProjectBdrCoefficient(*prescribed_disp.GetCoeff(2),
                                      prescribed_disp.attr);
   }
   for (auto &prescribed_vel : prescribed_velocities)
   {
      // Mult() computes a_n: use the previous acceleration, not the derived end-of-step a_{n+1}
      auto *derived_acc_coef = dynamic_cast<DerivedAccelerationCoefficient *>
                               (prescribed_vel.original_coeff);
      VectorCoefficient *acc_n = derived_acc_coef ?
                                 derived_acc_coef->GetPreviousAccelerationCoefficient() : prescribed_vel.coeff;
      d2udt2_gf.ProjectBdrCoefficient(*acc_n, prescribed_vel.attr);
   }

   d2udt2_gf.GetTrueDofs(d2udt2);
   d2udt2.SetSubVector(fixed_tdof_list,
                       0.0); // should not be necessary as we set the vector d2udt2 to zero, and only project coefficients on the prescribed displacements bdry.
   opM.EliminateBC(opMe, ess_tdof_list, d2udt2, z);

   // Solve
   M_solver.Mult(z, d2udt2);

   // Enforce essential boundary conditions again (avoid round-off)  Is there a better way to enforce the inhomogeneous Dirichlet BCs?
   d2udt2_gf.SetFromTrueDofs(d2udt2);
   for (auto &prescribed_disp : prescribed_displacements)
   {
      d2udt2_gf.ProjectBdrCoefficient(*prescribed_disp.GetCoeff(2),
                                      prescribed_disp.attr);
   }
   for (auto &prescribed_vel : prescribed_velocities)
   {
      // Mult() computes a_n: use the previous acceleration, not the derived end-of-step a_{n+1}
      auto *derived_acc_coef = dynamic_cast<DerivedAccelerationCoefficient *>
                               (prescribed_vel.original_coeff);
      VectorCoefficient *acc_n = derived_acc_coef ?
                                 derived_acc_coef->GetPreviousAccelerationCoefficient() : prescribed_vel.coeff;
      d2udt2_gf.ProjectBdrCoefficient(*acc_n, prescribed_vel.attr);
   }
   d2udt2_gf.GetTrueDofs(d2udt2);
   d2udt2.SetSubVector(fixed_tdof_list, 0.0);
}

void ElasticityOperator::ImplicitSolve(const real_t fac0, const real_t fac1,
                                       const Vector &u, const Vector &dudt, Vector &d2udt2)
{
   // Construct the operator for implicit solve
   if (A == nullptr || fac0 != A_fac0 || fac1 != A_fac1)
   {
      delete A;
      A = nullptr;
      delete Ae;
      Ae = nullptr;
      A_fac0 = fac0;
      A_fac1 = fac1;
      A = Add(1.0, *opMfull.As<HypreParMatrix>(), fac0/density,
              *opK.As<HypreParMatrix>());
      // Robin: add fac1/density * M_gamma to A (implicit velocity damping)
      if (!robin_bcs.empty())
      {
         HypreParMatrix *Atmp = Add(1.0, *A, fac1/density,
                                    *opMgamma.As<HypreParMatrix>());
         delete A;
         A = Atmp;
      }
      Ae = A->EliminateRowsCols(ess_tdof_list);
      A_solver.SetOperator(*A);
   }

   // Compute rhs
   opK->Mult(u, z);
   z.Neg();
   z.Add(1.0, F);

   z *= (1.0 / density);

   // Robin boundary damping: -M_gamma * dudt / density  (fac1 * M_gamma * a part is on the LHS)
   if (!robin_bcs.empty())
   {
      opMgamma->AddMult(dudt, z, -1.0 / density);
   }

   // Apply bcs
   //d2udt2 = 0.0;
   d2udt2_gf = 0.0;
   for (auto &prescribed_disp : prescribed_displacements)
   {
      d2udt2_gf.ProjectBdrCoefficient(*prescribed_disp.GetCoeff(2),
                                      prescribed_disp.attr);
   }
   for (auto &prescribed_vel : prescribed_velocities)
   {
      d2udt2_gf.ProjectBdrCoefficient(*prescribed_vel.coeff, prescribed_vel.attr);
   }
   d2udt2_gf.GetTrueDofs(d2udt2);
   d2udt2.SetSubVector(fixed_tdof_list,
                       0.0); // should not be necessary as we set the vector d2udt2 to zero, and only project coefficients on the prescribed displacements bdry.
   A->EliminateBC(*Ae, ess_tdof_list, d2udt2, z);

   // Solve
   A_solver.Mult(z, d2udt2);

   // Enforce essential boundary conditions again (avoid round-off)  Is there a better way to enforce the inhomogeneous Dirichlet BCs?
   d2udt2_gf.SetFromTrueDofs(d2udt2);
   for (auto &prescribed_disp : prescribed_displacements)
   {
      d2udt2_gf.ProjectBdrCoefficient(*prescribed_disp.GetCoeff(2),
                                      prescribed_disp.attr);
   }
   for (auto &prescribed_vel : prescribed_velocities)
   {
      d2udt2_gf.ProjectBdrCoefficient(*prescribed_vel.coeff, prescribed_vel.attr);
   }
   d2udt2_gf.GetTrueDofs(d2udt2);
   d2udt2.SetSubVector(fixed_tdof_list, 0.0);
}



ElasticityOperator::~ElasticityOperator()
{
   delete Mform;
   Mform = nullptr;
   delete Kform;
   Kform = nullptr;

   delete fec;
   fec = nullptr;
   delete fes;
   fes = nullptr;

   delete Fform;
   Fform = nullptr;

   delete Mgamma_form;
   Mgamma_form = nullptr;

   delete A;
   A = nullptr;
   delete Ae;
   Ae = nullptr;

   opMfull.Clear();
   opK.Clear();
   opM.Clear();
   opMe.Clear();
   opMgamma.Clear();

   delete M_pc;
   M_pc = nullptr;
   delete A_pc;
   A_pc = nullptr;
}