#include "mesh_laplacian.hpp"

using namespace mfem;

MeshLaplacian::MeshLaplacian(ParMesh *fluid_mesh_, bool incremental_)
    : MeshDeformation(fluid_mesh_), incremental(incremental_)
{
   // Deep copy of the fluid mesh in the deformed state
   if (incremental)
   {
      is_nonincremental_mdt = false; 
      Initialize(fluid_mesh);
   }
   else
   {
      reference_mesh = new ParMesh(*fluid_mesh_, true);
      Initialize(reference_mesh);
   }
}

MeshLaplacian::~MeshLaplacian()
{
   // Release solver/preconditioner/operator before the form they reference
   delete linear_solver;
   linear_solver = nullptr;
   delete prec;
   prec = nullptr;
   opA.Clear();
   delete laplacian_form;
   laplacian_form = nullptr;
   delete diffusion_coeff;
   diffusion_coeff = nullptr;
   // The stiffening space may live on reference_mesh: release it first
   delete local_stiffening_coeff;
   local_stiffening_coeff = nullptr;
   delete fes_stiffening;
   fes_stiffening = nullptr;
   delete reference_mesh;
}

void MeshLaplacian::Setup(real_t alpha_, bool pa_, int prec_type_)
{
   // Store parameters in base class members
   alpha.constant = alpha_;
   pa = pa_;
   prec_type = prec_type_;

   // Extract essential tdofs
   ess_tdof_list.SetSize(0);
   fes_mesh->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Create the bilinear form for the mesh Laplacian problem, assemble, and create solver and preconditioner
   AssembleOperator();
}

void MeshLaplacian::AssembleOperator()
{
   // Release the old solver/operator before the form and coefficient they reference
   delete linear_solver;
   linear_solver = nullptr;
   delete prec;
   prec = nullptr;
   opA.Clear();
   delete laplacian_form;
   laplacian_form = nullptr;
   delete diffusion_coeff;
   diffusion_coeff = nullptr;

   diffusion_coeff = local_stiffening_coeff ? new ProductCoefficient(alpha, *local_stiffening_coeff) : new ProductCoefficient(1.0, alpha);

   laplacian_form = new ParBilinearForm(fes_mesh);
   laplacian_form->AddDomainIntegrator(new VectorDiffusionIntegrator(*diffusion_coeff));
   if (pa)
      laplacian_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   laplacian_form->Assemble();
   laplacian_form->FormSystemMatrix(ess_tdof_list, opA);

   // Solver and preconditioner
   ResetSolver();
}

void MeshLaplacian::Solve(ParGridFunction &field, bool deform_mesh, real_t dt)
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
      laplacian_form->EliminateVDofsInRHS(ess_tdof_list, X, B);

   // Solve
   linear_solver->Mult(B, X);
   if (!linear_solver->GetConverged() && my_id == 0)
   {
      mfem::err << "WARNING: MeshLaplacian solve did not converge (iterations: "
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
         if (incremental)
         {
            // For incremental deformation, we deform the mesh based on the displacement field
            DeformMeshWithRelativeDisplacement(field, true);
         }
         else
         {
            // For non-incremental deformation, we deform the mesh without updating the grid function
            DeformMeshWithDisplacement(field, false);
         }
      }
      else
      {
         // If dt is positive, we assume the field is a velocity field
         DeformMeshWithVelocity(field, dt, false);
      }
   }
}

void MeshLaplacian::EnableLocalStiffening(Array<int> &wall_attr_, int stiffening_type_, int distance_solver_type, real_t stiffening_degree, bool filter_ls)
{
   MeshDeformation::EnableLocalStiffening(wall_attr_, stiffening_type_, distance_solver_type, stiffening_degree, filter_ls);

   // If laplacian_form exists, it means that Setup has been already called
   // So we re-assemble the system
   if (laplacian_form)
   {
      AssembleOperator();
   }
}

void MeshLaplacian::EnableLocalStiffening(int wall_attr_, int stiffening_type_, int distance_solver_type, real_t stiffening_degree, bool filter_ls)
{
   Array<int> tmp_wall_attrs(fluid_mesh->bdr_attributes.Max());
   tmp_wall_attrs = 0;
   MFEM_VERIFY(wall_attr_ >= 1 && wall_attr_ <= tmp_wall_attrs.Size(), "Invalid wall attribute " << wall_attr_);
   tmp_wall_attrs[wall_attr_ - 1] = 1; // boundary attributes are 1-based
   EnableLocalStiffening(tmp_wall_attrs, stiffening_type_, distance_solver_type, stiffening_degree, filter_ls);
}


void MeshLaplacian::Update()
{
   if (!incremental)
   {
      // No update needed as we work on the reference mesh
      return;
   }

   // Update the local stiffening coefficient first: the operator below is assembled with it
   if (enable_local_stiffening)
   {
      local_stiffening_coeff->Update();
   }

   // Update the mesh Laplacian problem after mesh is deformed (if using incremental version)
   opA.Clear();
   laplacian_form->Update();
   laplacian_form->Assemble();
   laplacian_form->FormSystemMatrix(ess_tdof_list, opA);

   ResetSolver();
}

void MeshLaplacian::ResetSolver()
{
   // Systems AMG: one function per mesh-node component, with the dof ordering of the mesh space
   const int vdim = fes_mesh->GetVDim();
   const bool bynodes = fes_mesh->GetOrdering() == Ordering::byNODES;

   // Solver and preconditioner (delete the solver first: it references the preconditioner)
   delete linear_solver;
   linear_solver = nullptr;
   delete prec;
   prec = nullptr;
   if (pa)
   {
      switch (prec_type)
      {
      case 0: // Jacobi Smoother
         prec = new OperatorJacobiSmoother(*laplacian_form, ess_tdof_list);
         break;
      case 1: // LOR
      {
         auto *lor_prec = new LORSolver<HypreBoomerAMG>(*laplacian_form, ess_tdof_list);
         lor_prec->GetSolver().SetSystemsOptions(vdim, bynodes);
         lor_prec->GetSolver().SetPrintLevel(0);
         prec = lor_prec;
         break;
      }
      default:
         MFEM_ABORT("Unknown preconditioner type.");
      }
   }
   else
   {
      switch (prec_type)
      {
      case 0:
         prec = new HypreSmoother(*opA.As<HypreParMatrix>());
         dynamic_cast<HypreSmoother *>(prec)->SetType(HypreSmoother::Jacobi, 1);
         break;
      case 1:
         prec = new HypreBoomerAMG(*opA.As<HypreParMatrix>());
         static_cast<HypreBoomerAMG *>(prec)->SetSystemsOptions(vdim, bynodes);
         static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
         break;
      }
   }

   linear_solver = new CGSolver(fes_mesh->GetComm());
   linear_solver->SetRelTol(1e-8);
   linear_solver->iterative_mode = false;
   linear_solver->SetPrintLevel(0);
   linear_solver->SetMaxIter(2000);
   linear_solver->SetOperator(*opA);
   linear_solver->SetPreconditioner(*prec);
}
