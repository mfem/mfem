#ifndef LINEAR_SOLVER_HPP
#define LINEAR_SOLVER_HPP
#include "mfem.hpp"

namespace mfem
{

class EllipticSolver
{
private:
   BilinearForm &a;
   Array<int> &ess_tdof_list;
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
   Array<int> ess_tdof_list;
   bool parallel;
   bool hasAdjoint;
   bool elast;
   Array<Coefficient*> owned_coeffs;
   Array<VectorCoefficient*> owned_vcoeffs;
   Array<Vector*> owned_vectors;
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

   void BuildTDofList(Array<int> &ess_bdr);
   void BuildTDofList(Array2D<int> &ess_bdr);
   void InitializeForms();

public:
   EllipticProblem(FiniteElementSpace &fes, Array<int> &ess_bdr,
                   bool hasAdjoint=false)
      :fes(fes),
       isAStationary(false), isBStationary(false), isAdjBStationary(false),
       parallel(false), hasAdjoint(hasAdjoint), elast(false),
       owned_coeffs(0), owned_vcoeffs(0), owned_vectors(0)
   {
      InitializeForms();
      BuildTDofList(ess_bdr);
   }

   EllipticProblem(FiniteElementSpace &fes, Array2D<int> &ess_bdr,
                   bool hasAdjoint=false)
      :fes(fes),
       isAStationary(false), isBStationary(false), isAdjBStationary(false),
       parallel(false), hasAdjoint(hasAdjoint), elast(false),
       owned_coeffs(0), owned_vcoeffs(0), owned_vectors(0)
   {
      InitializeForms();
      BuildTDofList(ess_bdr);
   }

   ~EllipticProblem()
   {
      for (Coefficient *coeff:owned_coeffs) { if (coeff) {delete coeff;} }
      for (VectorCoefficient *coeff:owned_vcoeffs) { if (coeff) {delete coeff;} }
      for (Vector *v:owned_vectors) { if (v) {delete v;} }
   }

   void MakeCoefficientOwner(Coefficient *coeff) {owned_coeffs.Append(coeff);}
   void MakeCoefficientOwner(VectorCoefficient *coeff) {owned_vcoeffs.Append(coeff);}
   void MakeVectorOwner(Vector *v) {owned_vectors.Append(v);}

   void SetAStationary(bool stationary=true) {isAStationary=stationary;}
   void SetBStationary(bool stationary=true) {isBStationary=stationary;}
   void SetAdjBStationary(bool stationary=true) {isAdjBStationary=stationary;}

   void ResetSolver()
   {
      solver.reset(new EllipticSolver(*a, ess_tdof_list));
      if (elast) {solver->UseElasticityOption();}
   }

   BilinearForm *GetBilinearForm() {return a.get();}
   LinearForm *GetLinearForm() {return b.get();}
   LinearForm *GetAdjLinearForm() {return adjb.get();}
#ifdef MFEM_USE_MPI
   ParBilinearForm *GetParBilinearForm() {return par_a;}
   ParLinearForm *GetParLinearForm() {return par_b;}
   ParLinearForm *GetAdjParLinearForm() {return par_adjb;}
#endif

   void Solve(GridFunction &x, bool assembleA=false, bool assembleB=false);
   void SolveAdjoint(GridFunction &x, bool assembleA=false, bool assembleB=false);
   bool HasAdjoint() {return hasAdjoint;}
   bool IsParallel() { return parallel; }
#ifdef MFEM_USE_MPI
   MPI_Comm GetComm() {return comm;}
#endif
};

class HelmholtzFilter: public EllipticProblem
{
private:
   real_t r_min;
   std::unique_ptr<ConstantCoefficient> eps2;
public:
   HelmholtzFilter(FiniteElementSpace &fes, Array<int> &ess_bdr,
                   const real_t r_min, bool hasAdjoint)
      :EllipticProblem(fes, ess_bdr, hasAdjoint), r_min(r_min)
   {
      eps2.reset(new ConstantCoefficient(r_min*r_min/12.0));
      a->AddDomainIntegrator(new DiffusionIntegrator(*eps2));
      a->AddDomainIntegrator(new MassIntegrator());
      a->Assemble();
      SetAStationary(true);
   }
   real_t GetFilterRadius() {return r_min;}
   void SetFilterRadius(real_t new_r_min)
   {
      r_min = new_r_min;
      eps2->constant=r_min*r_min/12.0;
      a->Update(); a->Assemble();
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
      SetAStationary(true);
   }
};

class L2Projection: public EllipticProblem
{
private:
public:
   L2Projection(FiniteElementSpace &fes, Array<int> &ess_bdr)
      :EllipticProblem(fes, ess_bdr, false)
   {
      a->AddDomainIntegrator(new MassIntegrator());
      a->Assemble();
      SetAStationary(true);
   }
};

}// namespace mfem
#endif
