
#include "pde_solvers.hpp"

DiffusionSolver::DiffusionSolver(Mesh * mesh_, int order_, 
                     Coefficient * diffcf_, Coefficient * rhscf_)
: mesh(mesh_), order(order_), diffcf(diffcf_), rhscf(rhscf_)
{

#ifdef MFEM_USE_MPI
   pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh) { parallel = true; }
#endif

   SetupFEM();
}

void DiffusionSolver::SetupFEM()
{
   dim = mesh->Dimension();
   fec = new H1_FECollection(order, dim);
   
#ifdef MFEM_USE_MPI   
   if (parallel)
   {
      pfes = new ParFiniteElementSpace(pmesh, fec);
      u = new ParGridFunction(pfes);
      b = new ParLinearForm(pfes);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, fec);
      u = new GridFunction(fes);
      b = new LinearForm(fes);
   }
#else
   fes = new FiniteElementSpace(mesh, fec);
   u = new GridFunction(fes);
   b = new LinearForm(fes);
#endif
   *u=0.0;
   
   if (!ess_bdr.Size())
   {
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
   }
}

void DiffusionSolver::Solve()
{
   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;

#ifdef MFEM_USE_MPI   
   if (parallel)
   {
      pfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
   else
   {
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
#else
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
#endif
   *u=0.0;
   if (b) 
   {
      delete b;
#ifdef MFEM_USE_MPI   
      if (parallel)
      {
         b = new ParLinearForm(pfes);
      }
      else
      {
         b = new LinearForm(fes);
      }
#else
      b = new LinearForm(fes);
#endif   
   }   
   if (rhscf)
   {
      b->AddDomainIntegrator(new DomainLFIntegrator(*rhscf));
   }
   if (neumann_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryLFIntegrator(*neumann_cf),neumann_bdr);
   }
   else if (gradient_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*gradient_cf),neumann_bdr);
   }

   b->Assemble(false);

   BilinearForm * a = nullptr;

#ifdef MFEM_USE_MPI   
   if (parallel)
   {
      a = new ParBilinearForm(pfes);
   }
   else
   {
      a = new BilinearForm(fes);
   }
#else
   a = new BilinearForm(fes);
#endif  
   a->AddDomainIntegrator(new DiffusionIntegrator(*diffcf));
   if (masscf)
   {
      a->AddDomainIntegrator(new MassIntegrator(*masscf));
   }
   a->Assemble();
   if (essbdr_cf)
   {
      u->ProjectBdrCoefficient(*essbdr_cf,ess_bdr);
   }
   a->FormLinearSystem(ess_tdof_list, *u, *b, A, X, B);

   CGSolver * cg = nullptr;
   Solver * M = nullptr;
#ifdef MFEM_USE_MPI   
   if (parallel)
   {
      M = new HypreBoomerAMG;
      dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
      cg = new CGSolver(pmesh->GetComm());
   }
   else
   {
      M = new GSSmoother((SparseMatrix&)(*A));
      cg = new CGSolver;
   }
#else
   M = new GSSmoother((SparseMatrix&)(*A));
   cg = new CGSolver;
#endif
   cg->SetRelTol(1e-12);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   delete M;
   delete cg;
   a->RecoverFEMSolution(X, *b, *u);
   delete a;
}

GridFunction * DiffusionSolver::GetFEMSolution()
{
   return u;
}

#ifdef MFEM_USE_MPI  
ParGridFunction * DiffusionSolver::GetParFEMSolution()
{
   if (parallel)
   {
      return dynamic_cast<ParGridFunction*>(u);
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
      return nullptr;
   }
}
#endif

void DiffusionSolver::ResetFEM()
{
   delete u; u = nullptr;
   delete fes; fes = nullptr;
   delete fec; fec = nullptr;
   delete b;
}


DiffusionSolver::~DiffusionSolver()
{
   ResetFEM();
}



// -----------------------------------------------------------------------
// ------------------      Elasticity solver     -------------------------
// -----------------------------------------------------------------------

LinearElasticitySolver::LinearElasticitySolver(Mesh * mesh_, int order_, 
    Coefficient * lambda_cf_, Coefficient * mu_cf_)
: mesh(mesh_), order(order_), lambda_cf(lambda_cf_), mu_cf(mu_cf_)
{
#ifdef MFEM_USE_MPI
   pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh) { parallel = true; }
#endif
   SetupFEM();
}

void LinearElasticitySolver::SetupFEM()
{
   dim = mesh->Dimension();
   fec = new H1_FECollection(order, dim,BasisType::Positive);
   
#ifdef MFEM_USE_MPI   
   if (parallel)
   {
      pfes = new ParFiniteElementSpace(pmesh, fec, dim);
      u = new ParGridFunction(pfes);
      b = new ParLinearForm(pfes);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, fec,dim);
      u = new GridFunction(fes);
      b = new LinearForm(fes);
   }
#else
   fes = new FiniteElementSpace(mesh, fec, dim);
   u = new GridFunction(fes);
   b = new LinearForm(fes);
#endif
   *u=0.0;
   
   if (!ess_bdr.Size())
   {
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
   }
}

void LinearElasticitySolver::Solve()
{
   GridFunction * x = nullptr;
   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;

#ifdef MFEM_USE_MPI   
   if (parallel)
   {
      x = new ParGridFunction(pfes);
      pfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
   else
   {
      x = new GridFunction(fes);
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
#else
   x = new ParGridFunction(fes);
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
#endif
   *u=0.0;
   if (b) 
   {
      delete b;
#ifdef MFEM_USE_MPI   
      if (parallel)
      {
         b = new ParLinearForm(pfes);
      }
      else
      {
         b = new LinearForm(fes);
      }
#else
      b = new LinearForm(fes);
#endif   
   }   
   if (rhs_cf)
   {
      b->AddDomainIntegrator(new VectorDomainLFIntegrator(*rhs_cf));
   }

   b->Assemble(false);

   *x = 0.0;

   BilinearForm * a = nullptr;

#ifdef MFEM_USE_MPI   
   if (parallel)
   {
      a = new ParBilinearForm(pfes);
   }
   else
   {
      a = new BilinearForm(fes);
   }
#else
   a = new BilinearForm(fes);
#endif  
   a->AddDomainIntegrator(new ElasticityIntegrator(*lambda_cf, *mu_cf));
   a->Assemble();
   if (essbdr_cf)
   {
      u->ProjectBdrCoefficient(*essbdr_cf,ess_bdr);
   }
   a->FormLinearSystem(ess_tdof_list, *x, *b, A, X, B);

   CGSolver * cg = nullptr;
   Solver * M = nullptr;
#ifdef MFEM_USE_MPI   
   if (parallel)
   {
      M = new HypreBoomerAMG;
      dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
      cg = new CGSolver(pmesh->GetComm());
   }
   else
   {
      M = new GSSmoother((SparseMatrix&)(*A));
      cg = new CGSolver;
   }
#else
   M = new GSSmoother((SparseMatrix&)(*A));
   cg = new CGSolver;
#endif
   cg->SetRelTol(1e-10);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   delete M;
   delete cg;
   a->RecoverFEMSolution(X, *b, *x);
   *u+=*x;
   delete a;
   delete x;
}

GridFunction * LinearElasticitySolver::GetFEMSolution()
{
   return u;
}

#ifdef MFEM_USE_MPI  
ParGridFunction * LinearElasticitySolver::GetParFEMSolution()
{
   if (parallel)
   {
      return dynamic_cast<ParGridFunction*>(u);
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
      return nullptr;
   }
}
#endif

void LinearElasticitySolver::ResetFEM()
{
   delete u; u = nullptr;
   delete fes; fes = nullptr;
   delete fec; fec = nullptr;
   delete b;
}

LinearElasticitySolver::~LinearElasticitySolver()
{
   ResetFEM();
}
