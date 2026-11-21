#include "mesh_linear_elasticity.hpp"
#include "fsi_utils.hpp"

MeshLinearElasticity::MeshLinearElasticity(ParMesh *fluid_mesh_,
                                           bool incremental_)
   : MeshDeformation(fluid_mesh_), pa(false)
{
   // We keep the incremental flag for future use, and to be able to use the same interface for different mesh deformation methods.
   MFEM_VERIFY(!incremental_,
               "Incremental mesh deformation is not supported yet in MeshLinearElasticity.");

   // Create a copy of the reference mesh
   reference_mesh = new ParMesh(*fluid_mesh_);
   Initialize(reference_mesh);
}

MeshLinearElasticity::~MeshLinearElasticity()
{
   // Release solver/preconditioner/operator before the form they reference
   delete linear_solver;
   linear_solver = nullptr;
   delete prec;
   prec = nullptr;
   delete lor;
   lor = nullptr;
   opA.Clear();
   delete elasticity_form;
   elasticity_form = nullptr;
   delete lambda_stiff;
   lambda_stiff = nullptr;
   delete mu_stiff;
   mu_stiff = nullptr;
   // The stiffening space lives on reference_mesh: release it first
   delete local_stiffening_coeff;
   local_stiffening_coeff = nullptr;
   delete fes_stiffening;
   fes_stiffening = nullptr;
   delete reference_mesh;
}

void MeshLinearElasticity::Setup(real_t E_, real_t nu_, bool pa_,
                                 int prec_type_)
{
   // TODO:
   // 1. For PA we need to use Ordering::byNODES for the mesh space (see reorder_space in ex2p.cpp)
   // 2. we can also integrate StaticCondensation

   // Set the parameters for the elasticity problem
   E = E_;
   nu = nu_;
   mu.constant = E / (2.0 * (1.0 +
                             nu));                        // G, Shear modulus, Lame's second parameter
   lambda.constant  = E * nu / ((1.0 + nu) * (1.0 - 2.0 *
                                              nu)); // Lame's first parameter

   pa = pa_;

   // Extract essential tdofs
   ess_tdof_list.SetSize(0);
   fes_mesh->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Create the elasticity bilinear form, assemble, and create solver and preconditioner
   AssembleOperator();
}

void MeshLinearElasticity::AssembleOperator()
{
   // Release the old solver/operator before the form and coefficients they reference
   delete linear_solver;
   linear_solver = nullptr;
   delete prec;
   prec = nullptr;
   delete lor;
   lor = nullptr;
   opA.Clear();
   delete elasticity_form;
   elasticity_form = nullptr;
   delete lambda_stiff;
   lambda_stiff = nullptr;
   delete mu_stiff;
   mu_stiff = nullptr;

   // Create new elasticity bilinear form
   lambda_stiff = local_stiffening_coeff ? new ProductCoefficient(
                     *local_stiffening_coeff, lambda) : new ProductCoefficient(1.0, lambda);
   mu_stiff = local_stiffening_coeff ? new ProductCoefficient(
                 *local_stiffening_coeff, mu) : new ProductCoefficient(1.0, mu);

   elasticity_form = new ParBilinearForm(fes_mesh);
   elasticity_form->AddDomainIntegrator(new ElasticityIntegrator(*lambda_stiff,
                                                                 *mu_stiff));

   if (pa)
   {
      elasticity_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   elasticity_form->Assemble();
   elasticity_form->FormSystemMatrix(ess_tdof_list, opA);

   // Solver and preconditioner
   ResetSolver();
}

void MeshLinearElasticity::Update()
{
   // Nothing to update for linear elasticity in non-incremental case
   // (the operator is assembled on the reference mesh)
}

void MeshLinearElasticity::EnableLocalStiffening(Array<int> &wall_attr_,
                                                 int stiffening_type_, int distance_solver_type, real_t stiffening_degree,
                                                 bool filter_ls)
{
   MeshDeformation::EnableLocalStiffening(wall_attr_, stiffening_type_,
                                          distance_solver_type, stiffening_degree, filter_ls);

   if (elasticity_form)
   {
      // If elasticity form is not empty it means that Setup was called already
      AssembleOperator();
   }
}

void MeshLinearElasticity::EnableLocalStiffening(int wall_attr_,
                                                 int stiffening_type_, int distance_solver_type, real_t stiffening_degree,
                                                 bool filter_ls)
{
   Array<int> tmp_wall_attrs(fluid_mesh->bdr_attributes.Max());
   tmp_wall_attrs = 0;
   MFEM_VERIFY(wall_attr_ >= 1 &&
               wall_attr_ <= tmp_wall_attrs.Size(), "Invalid wall attribute " << wall_attr_);
   tmp_wall_attrs[wall_attr_ - 1] = 1; // boundary attributes are 1-based
   EnableLocalStiffening(tmp_wall_attrs, stiffening_type_, distance_solver_type,
                         stiffening_degree, filter_ls);
}

void MeshLinearElasticity::Solve(ParGridFunction &field, bool deform_mesh,
                                 real_t dt)
{
   MFEM_ASSERT(field.ParFESpace()->GetTrueVSize() == fes_mesh->GetTrueVSize(),
               "Field size does not match the fluid finite element space size.");

   field.GetTrueDofs(X);

   // Apply boundary conditions
   B = 0.0;
   if (pa)
   {
      auto *opAC = opA.As<ConstrainedOperator>();
      opAC->EliminateRHS(X, B);
   }
   else
   {
      elasticity_form->EliminateVDofsInRHS(ess_tdof_list, X, B);
   }

   // Solve
   linear_solver->Mult(B, X);
   if (!linear_solver->GetConverged() && my_id == 0)
   {
      mfem::err <<
                "WARNING: MeshLinearElasticity solve did not converge (iterations: "
                << linear_solver->GetNumIterations() << ", final norm: "
                << linear_solver->GetFinalNorm() << ")" << std::endl;
   }
   field.SetFromTrueDofs(X);

   // Evaluate the mesh displacement and deform the mesh
   if (deform_mesh)
   {
      if (dt <= 0.0)
      {
         // If dt is negative, we assume the field is a displacement field
         // For non-incremental deformation, we deform the mesh without updating the grid function
         DeformMeshWithDisplacement(field, false);
      }
      else
      {
         // If dt is positive, we assume the field is a velocity field
         DeformMeshWithVelocity(field, dt, false);
      }
   }
}

void MeshLinearElasticity::DeformMeshWithVelocity(ParGridFunction
                                                  &mesh_velocity_gf, real_t dt, bool update_gf)
{
   // Deform the mesh nodes based on the provided mesh velocity
   MFEM_ABORT("DeformMeshWithVelocity is not allowed in MeshLinearElasticity, use DeformMeshWithDisplacement instead.");
}

void MeshLinearElasticity::ResetSolver()
{
   delete linear_solver;
   linear_solver = nullptr;

   delete prec;
   prec = nullptr;

   if (pa)
   {
      delete lor;
      lor = new ParLORDiscretization(*elasticity_form, ess_tdof_list);
      auto *lor_amg = new HypreBoomerAMG(lor->GetAssembledMatrix());
      common_fsi_utils::SetVectorAMGOptions(*lor_amg, &(lor->GetParFESpace()));
      lor_amg->SetPrintLevel(0);
      prec = lor_amg;
   }
   else
   {
      prec = new HypreBoomerAMG(*opA.As<HypreParMatrix>());
      common_fsi_utils::SetVectorAMGOptions(*static_cast<HypreBoomerAMG *>(prec),
                                            fes_mesh);
      static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
   }

   linear_solver = new CGSolver(fes_mesh->GetComm());
   linear_solver->SetRelTol(1e-8);
   linear_solver->iterative_mode = false;
   linear_solver->SetPrintLevel(0);
   linear_solver->SetMaxIter(2000);
   linear_solver->SetOperator(*opA);
   linear_solver->SetPreconditioner(*prec);
}
