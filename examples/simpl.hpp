#include "mfem.hpp"
#ifndef SIMPL
#define SIMPL

namespace mfem
{

inline void SolveEllipticProblem(BilinearForm &a, LinearForm &b,
                                 GridFunction &x, Array<int> ess_tdof_list)
{
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, 1);

   GSSmoother M;
   CGSolver cg;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(A);
   cg.SetPrintLevel(0);
   cg.iterative_mode=true;
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);
}

#ifdef MFEM_USE_MPI
inline void ParSolveEllipticProblem(ParBilinearForm &a, ParLinearForm &b,
                                    ParGridFunction &x, Array<int> ess_tdof_list)
{
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, 1);

   HypreBoomerAMG M;
   M.SetPrintLevel(0);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(A);
   cg.SetPrintLevel(0);
   cg.iterative_mode=true;
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);
}
#endif

class LinearProblem
{
protected:
   FiniteElementSpace *fes;
   Mesh *mesh;
   std::unique_ptr<BilinearForm> a;
   std::unique_ptr<LinearForm> b;
   std::unique_ptr<LinearForm> adj_b;
   bool isAstationary=false;
   bool isBstationary=false;
   bool isAdjBstationary=false;
   Array<int> ess_tdof_list;

   bool parallel;
#ifdef MFEM_USE_MPI
   ParMesh *pmesh;
   ParFiniteElementSpace *pfes;
   ParBilinearForm *par_a;
   ParLinearForm *par_b;
   ParLinearForm *par_adj_b;
#endif

public:
   LinearProblem(FiniteElementSpace &fes, bool has_dualRHS=false)
      : fes(&fes), isAstationary(false), isBstationary(false),
        isAdjBstationary(false), ess_tdof_list(0), parallel(false)
   {
      mesh = fes.GetMesh();
#ifdef MFEM_USE_MPI
      pmesh = dynamic_cast<ParMesh *>(mesh);
      if (pmesh)
      {
         parallel = true;
         pfes = dynamic_cast<ParFiniteElementSpace *>(&fes);
         par_a = new ParBilinearForm(pfes);
         par_b = new ParLinearForm(pfes);
         if (has_dualRHS) { par_adj_b = new ParLinearForm(pfes); }
         a.reset(par_a);
         b.reset(par_b);
         adj_b.reset(par_adj_b);
      }
      else
      {
         a.reset(new BilinearForm(&fes));
         b.reset(new LinearForm(&fes));
         if (has_dualRHS) {adj_b.reset(new LinearForm(&fes));}
      }
#else
      a.reset(new BilinearForm(&fes));
      b.reset(new LinearForm(&fes));
      if (has_dualRHS) {adjoint_b.reset(new LinearForm(&fes));}
#endif
   }

   ~LinearProblem() = default;

   void SetAstationary(bool isstationary=true) {isAstationary=isstationary;}
   void SetBstationary(bool isstationary=true) {isBstationary=isstationary;}
   void SetAdjBstationary(bool isstationary=true) {isAdjBstationary=isstationary;}
   void AssembleStationaryOperators()
   {
      if (isAstationary) {a->Update(); a->Assemble();}
      if (isBstationary) {b->Assemble();}
      if (isAdjBstationary) {adj_b->Assemble();}
   }

   BilinearForm & GetBilinearForm() {return *a;}
   LinearForm & GetLinearForm() {return *b;}
   LinearForm & GetAdjointLinearForm() {return *adj_b;}
   virtual void Solve(GridFunction &x, bool assembleA, bool assembleB) = 0;
   virtual void SolveDual(GridFunction &x, bool assembleA, bool assembleB) = 0;


   void SetEssentialBoundary(Array<int> ess_bdr)
   {
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         pfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      else
      {
         fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
#else
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
#endif
   }

   void SetEssentialBoundary(Array2D<int> ess_bdr)
   {
      Array<int> ess_bdr_comp;
      Array<int> ess_tdof_list_comp;
      for (int i=-1; i<fes->GetVDim(); i++)
      {
         ess_bdr.GetRow(i+1, ess_bdr_comp);
         ess_tdof_list_comp.SetSize(0);
#ifdef MFEM_USE_MPI
         if (parallel)
         {
            pfes->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i);
         }
         else
         {
            fes->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i);
         }
#else
         fes->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i);
#endif
         ess_tdof_list.Append(ess_tdof_list_comp);
      }
   }
};

class LinearEllipticProblem : public LinearProblem
{
public:
   LinearEllipticProblem(FiniteElementSpace &fes,
                         bool hasDualRHS):LinearProblem(fes, hasDualRHS) {}
   ~LinearEllipticProblem() = default;
   void Solve(GridFunction &x, bool assembleA=false,
              bool assembleB=true) override final
   {
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         if (assembleA) {par_a->Update(); par_a->Assemble(); }
         if (assembleB) {par_b->Assemble(); }
         ParGridFunction *par_x = dynamic_cast<ParGridFunction*>(&x);
         ParSolveEllipticProblem(*par_a, *par_b, *par_x, ess_tdof_list);
      }
      else
      {
         if (assembleA) {a->Update(); a->Assemble(); }
         if (assembleB) {b->Assemble(); }
         SolveEllipticProblem(*a, *b, x, ess_tdof_list);
      }
#else
      if (assembleA) {a->Update(); a->Assemble(); }
      if (assembleB) {b->Assemble(); }
      SolveEllipticProblem(*a, *b, x, ess_tdof_list);
#endif
   }

   void SolveDual(GridFunction &x, bool assembleA=false,
                  bool assembleB=true) override final
   {
      if (!adj_b) {MFEM_ABORT("Adjoint problem undefined");}
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         if (assembleA) {par_a->Update(); par_a->Assemble(); }
         if (assembleB) {par_adj_b->Assemble(); }
         ParGridFunction *par_x = dynamic_cast<ParGridFunction*>(&x);
         ParSolveEllipticProblem(*par_a, *par_adj_b, *par_x, ess_tdof_list);
      }
      else
      {
         if (assembleA) {a->Update(); a->Assemble(); }
         if (assembleB) {adj_b->Assemble(); }
         SolveEllipticProblem(*a, *adj_b, x, ess_tdof_list);
      }
#else
      if (assembleA) {a->Update(); a->Assemble(); }
      if (assembleB) {adj_b->Assemble(); }
      SolveEllipticProblem(*a, *adj_b, x, ess_tdof_list);
#endif
   }
};

class LinearElasticityProblem final: public LinearEllipticProblem
{
protected:
   Coefficient *lambda;
   Coefficient *mu;

public:
   LinearElasticityProblem(FiniteElementSpace &fes, Coefficient *lambda,
                           Coefficient *mu, bool has_dualRHS=false):LinearEllipticProblem(fes,
                                    has_dualRHS),
      lambda(lambda), mu(mu)
   {
      a->AddDomainIntegrator(new ElasticityIntegrator(*lambda, *mu));
   }
};

class HelmholtzFilter final: public LinearEllipticProblem
{
protected:
   ConstantCoefficient eps2;
   Coefficient *rho;
   Coefficient *energy;

public:
   HelmholtzFilter(FiniteElementSpace &fes, real_t filter_radius, Coefficient *rho,
                   Coefficient *energy):LinearEllipticProblem(fes, true),
      eps2(std::pow(filter_radius/(2.0*std::sqrt(3)),2))
   {
      a->AddDomainIntegrator(new DiffusionIntegrator(eps2));
      a->AddDomainIntegrator(new MassIntegrator());
      b->AddDomainIntegrator(new DomainLFIntegrator(*rho));
      adj_b->AddDomainIntegrator(new DomainLFIntegrator(*energy));
      isAstationary = true;
   }
};

class L2Projector final : public LinearEllipticProblem
{
protected:
   Coefficient *target;

public:
   L2Projector(FiniteElementSpace &fes,
               Coefficient *target):LinearEllipticProblem(fes, false),
      target(target)
   {
      a->AddDomainIntegrator(new MassIntegrator());
      b->AddDomainIntegrator(new DomainLFIntegrator(*target));
      isAstationary = true;
   }
};

} // end of namespace mfem
#endif // end of define SIMPL
