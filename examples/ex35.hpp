//                  MFEM Example 35 - Serial/Parallel Shared Code
//
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

//  Class for solving Poisson's equation with MFEM:
//
//  - ∇ ⋅(κ ∇ u) = f  in Ω 
class DiffusionSolver 
{
private:
    Mesh * mesh = nullptr;
    int order = 1;
    Coefficient * diffcf = nullptr;
    Coefficient * masscf = nullptr;
    Coefficient * rhscf = nullptr;
    Coefficient * essbdr_cf = nullptr;
    Coefficient * neumann_cf = nullptr;
    VectorCoefficient * gradient_cf = nullptr;

    // FEM solver
    int dim;
    FiniteElementCollection * fec = nullptr;    
    FiniteElementSpace * fes = nullptr;    
    Array<int> ess_bdr;
    Array<int> neumann_bdr;
    GridFunction * u = nullptr;
    LinearForm * b = nullptr;
    bool parallel = false;
#ifdef MFEM_USE_MPI
    ParMesh * pmesh = nullptr;
    ParFiniteElementSpace * pfes = nullptr;
#endif

public:
    DiffusionSolver() { }
    DiffusionSolver(Mesh * mesh_, int order_, Coefficient * diffcf_, Coefficient * cf_);

    void SetMesh(Mesh * mesh_) 
    { 
        mesh = mesh_; 
#ifdef MFEM_USE_MPI
        pmesh = dynamic_cast<ParMesh *>(mesh);
        if (pmesh) { parallel = true; }
#endif    
    }
    void SetOrder(int order_) { order = order_ ; }
    void SetDiffusionCoefficient(Coefficient * diffcf_) { diffcf = diffcf_; }
    void SetMassCoefficient(Coefficient * masscf_) { masscf = masscf_; }
    void SetRHSCoefficient(Coefficient * rhscf_) { rhscf = rhscf_; }
    void SetEssentialBoundary(const Array<int> & ess_bdr_){ ess_bdr = ess_bdr_;};
    void SetNeumannBoundary(const Array<int> & neumann_bdr_){ neumann_bdr = neumann_bdr_;};
    void SetNeumannData(Coefficient * neumann_cf_) {neumann_cf = neumann_cf_;}
    void SetEssBdrData(Coefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}
    void SetGradientData(VectorCoefficient * gradient_cf_) {gradient_cf = gradient_cf_;}

    void ResetFEM();
    void SetupFEM();

    void Solve();
    GridFunction * GetFEMSolution();
    LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
    ParGridFunction * GetParFEMSolution();
    ParLinearForm * GetParLinearForm() 
    {
        if (parallel)
        {
           return dynamic_cast<ParLinearForm *>(b);
        }
        else
        {
            MFEM_ABORT("Wrong code path. Call GetLinearForm");
            return nullptr;
        }
    }
#endif

    ~DiffusionSolver();

};


//  Class for solving linear elasticity with MFEM:
//
//  -∇ ⋅ (σ(u)) u = f  in Ω
// σ(u) = λ ∇⋅u I + μ (∇ u + (∇u)^T)
class LinearElasticitySolver
{
private:
    Mesh * mesh = nullptr;
    int order = 1;
    Coefficient * lambda_cf = nullptr;
    Coefficient * mu_cf = nullptr;
    VectorCoefficient * essbdr_cf = nullptr;
    VectorCoefficient * rhs_cf = nullptr;

    // FEM solver
    int dim;
    FiniteElementCollection * fec = nullptr;    
    FiniteElementSpace * fes = nullptr;    
    Array<int> ess_bdr;
    Array<int> neumann_bdr;
    GridFunction * u = nullptr;
    LinearForm * b = nullptr;
    bool parallel = false;
#ifdef MFEM_USE_MPI
    ParMesh * pmesh = nullptr;
    ParFiniteElementSpace * pfes = nullptr;
#endif

public:
    LinearElasticitySolver() { }
    LinearElasticitySolver(Mesh * mesh_, int order_, 
    Coefficient * lambda_cf_, Coefficient * mu_cf_);

    void SetMesh(Mesh * mesh_) 
    { 
        mesh = mesh_; 
#ifdef MFEM_USE_MPI
        pmesh = dynamic_cast<ParMesh *>(mesh);
        if (pmesh) { parallel = true; }
#endif    
    }
    void SetOrder(int order_) { order = order_ ; }
    void SetLameCoefficients(Coefficient * lambda_cf_, Coefficient * mu_cf_) { lambda_cf = lambda_cf_; mu_cf = mu_cf_;  }
    void SetRHSCoefficient(VectorCoefficient * rhs_cf_) { rhs_cf = rhs_cf_; }
    void SetEssentialBoundary(const Array<int> & ess_bdr_){ ess_bdr = ess_bdr_;};
    void SetNeumannBoundary(const Array<int> & neumann_bdr_){ neumann_bdr = neumann_bdr_;};
    void SetEssBdrData(VectorCoefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}

    void ResetFEM();
    void SetupFEM();

    void Solve();
    GridFunction * GetFEMSolution();
    LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
    ParGridFunction * GetParFEMSolution();
    ParLinearForm * GetParLinearForm() 
    {
        if (parallel)
        {
           return dynamic_cast<ParLinearForm *>(b);
        }
        else
        {
            MFEM_ABORT("Wrong code path. Call GetLinearForm");
            return nullptr;
        }
    }
#endif

    ~LinearElasticitySolver();

};

// -----------------------------------------------------------------------
// --------------------      Poisson solver     --------------------------
// -----------------------------------------------------------------------

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

   b->Assemble();

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

   b->Assemble();

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