#ifndef LINEAR_SOLVER_HPP
#define LINEAR_SOLVER_HPP
#include "mfem.hpp"

namespace mfem
{

class EllipticSolver
{
private:
   BilinearForm &a;
   Array<int> ess_tdof_list;
   std::unique_ptr<CGSolver> solver;
   std::unique_ptr<GSSmoother> prec;
   OperatorHandle A;
   Vector B;
   Vector X;
   bool parallel;
#ifdef MFEM_USE_MPI
   ParBilinearForm *par_a;
   MPI_Comm comm;
   std::unique_ptr<HyprePCG> par_solver;
   std::unique_ptr<HypreBoomerAMG> par_prec;
   HypreParVector par_B;
#endif
public:
private:
   void BuildEssTdofList();
   void SetupSolver();
public:
   EllipticSolver(BilinearForm &a, Array<int> &ess_bdr);
   EllipticSolver(BilinearForm &a, Array2D<int> &ess_bdr);
   void UseElasticityOption();
   void Solve(LinearForm &b, GridFunction &x);
   // This must be called after reassembly
   void Update();
};

class EllipticProblem
{
private:
   FiniteElementSpace &fes;
   bool isAStationary;
   bool isBStationary;
   bool isAdjBStationary;
   std::unique_ptr<EllipticSolver> solver;
   Array<int> *ess_bdr_vec;
   Array2D<int> *ess_bdr_mat;
   bool parallel;
   bool hasAdjoint;
   Array<Coefficient*> owned_coeffs;
   Array<VectorCoefficient*> owned_vcoeffs;
protected:
   std::unique_ptr<BilinearForm> a;
   std::unique_ptr<LinearForm> b;
   std::unique_ptr<LinearForm> adjb;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *par_fes;
   MPI_Comm comm;
   ParBilinearForm *par_a;
   ParLinearForm *par_b;
   ParLinearForm *par_adjb;
#endif
   void InitializeForms()
   {
#ifdef MFEM_USE_MPI
      par_fes = dynamic_cast<ParFiniteElementSpace*>(&fes);
      if (par_fes)
      {
         parallel = true;
         comm = par_fes->GetComm();
         par_a = new ParBilinearForm(par_fes);
         par_b = new ParLinearForm(par_fes);
         a.reset(par_a);
         b.reset(par_b);
         if (hasAdjoint)
         {
            par_adjb = new ParLinearForm(par_fes);
            adjb.reset(par_adjb);
         }
      }
      else
      {
         a.reset(new BilinearForm(&fes));
         b.reset(new LinearForm(&fes));
         if (hasAdjoint)
         {
            adjb.reset(new LinearForm(&fes));
         }
      }
#else
      a.reset(new BilinearForm(&fes));
      b.reset(new LinearForm(&fes));
      if (hasAdjoint)
      {
         adjb.reset(new LinearForm(&fes));
      }
#endif
   }
public:
   EllipticProblem(FiniteElementSpace &fes, Array<int> &ess_bdr,
                   bool hasAdjoint=false)
      :fes(fes), ess_bdr_vec(&ess_bdr),
       isAStationary(false), isBStationary(false), isAdjBStationary(false),
       parallel(false), hasAdjoint(hasAdjoint)
   {
      InitializeForms();
   }

   EllipticProblem(FiniteElementSpace &fes, Array2D<int> &ess_bdr,
                   bool hasAdjoint=false)
      :fes(fes), ess_bdr_mat(&ess_bdr),
       isAStationary(false), isBStationary(false), isAdjBStationary(false),
       parallel(false), hasAdjoint(hasAdjoint)
   {
      InitializeForms();
   }

   ~EllipticProblem()
   {
      for (Coefficient *coeff:owned_coeffs) { if (coeff) {delete coeff;} }
      for (VectorCoefficient *coeff:owned_vcoeffs) { if (coeff) {delete coeff;} }
   }

   void MakeCoefficientOwner(Coefficient *coeff) {owned_coeffs.Append(coeff);}
   void MakeCoefficientOwner(VectorCoefficient *coeff) {owned_vcoeffs.Append(coeff);}

   void SetAStationary(bool stationary=false) {isAStationary=stationary;}
   void SetBStationary(bool stationary=false) {isBStationary=stationary;}
   void SetAdjBStationary(bool stationary=false) {isAdjBStationary=stationary;}

   BilinearForm *GetBilinearForm() {return a.get();}
   LinearForm *GetLinearForm() {return b.get();}
   LinearForm *GetAdjLinearForm() {return adjb.get();}
#ifdef MFEM_USE_MPI
   ParBilinearForm *GetParBilinearForm() {return par_a;}
   ParLinearForm *GetParLinearForm() {return par_b;}
   ParLinearForm *GetAdjParLinearForm() {return par_adjb;}
#endif

   // Must be called after assembly
   void Update()
   {
      if (solver)
      {
         solver->Update();
         return;
      }
      if (ess_bdr_vec)
      {
         solver.reset(new EllipticSolver(*a, *ess_bdr_vec));
      }
      else
      {
         solver.reset(new EllipticSolver(*a, *ess_bdr_mat));
      }
   }

   void Solve(GridFunction &x)
   {
      if (!isAStationary) {a->Assemble();}
      if (!isBStationary) {b->Assemble();}
      if (!solver || !isAStationary) { Update();}
      solver->Solve(*b, x);
   }

   void SolveAdjoint(GridFunction &x)
   {
      MFEM_ASSERT(hasAdjoint,
                  "SolveAdjoint(GridFunction &) is called without setting hasAdjoint=true.");
      if (!isAStationary) {a->Assemble();}
      if (!isAdjBStationary) {adjb->Assemble();}
      if (!solver || !isAStationary) {Update();}
      solver->Solve(*b, x);
   }
   bool HasAdjoint() {return hasAdjoint;}
   bool IsParallel() { return parallel; }
#ifdef MFEM_USE_MPI
   bool GetComm() {return comm;}
#endif
};

class HelmholtzFilter: public EllipticProblem
{
private:
   real_t r_min;
   std::unique_ptr<ConstantCoefficient> eps2;
public:
   HelmholtzFilter(FiniteElementSpace &fes, Array<int> &ess_bdr,
                   const double r_min, bool hasAdjoint)
      :EllipticProblem(fes, ess_bdr, hasAdjoint), r_min(r_min)
   {
      eps2.reset(new ConstantCoefficient(r_min*r_min/12.0));
      a->AddDomainIntegrator(new DiffusionIntegrator(*eps2));
      a->AddDomainIntegrator(new MassIntegrator());
      a->Assemble();
      Update();
      SetAStationary(true);
   }
   real_t GetFilterRadius() {return r_min;}
   void SetFilterRadius(real_t new_r_min)
   {
      r_min = new_r_min;
      eps2->constant=r_min*r_min/12.0;
      a->Update(); a->Assemble();
      Update();
   }
};

class ElasticityProblem: public EllipticProblem
{
private:
public:
   ElasticityProblem(FiniteElementSpace &fes, Array2D<int> &ess_bdr,
                     Coefficient &lambda, Coefficient &mu, bool hasAdjoint)
      :EllipticProblem(fes, ess_bdr, hasAdjoint)
   {
      a->AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
      a->Assemble();
      Update();
      SetAStationary(true);
   }
};

}// namespace mfem
#endif
