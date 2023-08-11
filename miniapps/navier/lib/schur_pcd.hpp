#pragma once

#include <mfem.hpp>

namespace mfem
{

class PCD : public Solver
{
public:
   PCD(Solver &Mp_inv_, Solver &Lp_inv_, OperatorHandle &Fp_) :
      Solver(Mp_inv_.Width()),
      Mp_inv(Mp_inv_),
      Lp_inv(Lp_inv_),
      Fp(Fp_) { }

   void Mult(const Vector &x, Vector &y) const override
   {
      z.SetSize(y.Size());
      w.SetSize(y.Size());

      Lp_inv.Mult(x, z);
      Fp->Mult(z, w);
      Mp_inv.Mult(w, y);
   }

   void SetOperator(const Operator &op) override {}

   Solver &Mp_inv;
   Solver &Lp_inv;
   OperatorHandle Fp;
   mutable Vector z, w;
};

class PCDBuilder
{
public:
   PCDBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs,
              Coefficient *am_coeff, Coefficient *ak_coeff) :
      mp_form(&pres_fes),
      lp_form(&pres_fes),
      fp_form(&pres_fes)
   {
      mp_form.AddDomainIntegrator(new MassIntegrator);
      mp_form.Assemble();
      mp_form.Finalize();
      mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);

      lp_form.AddDomainIntegrator(new DiffusionIntegrator);
      lp_form.Assemble();
      lp_form.Finalize();
      lp_form.FormSystemMatrix(pres_ess_tdofs, Lp);

      fp_form.AddDomainIntegrator(new MassIntegrator(*am_coeff));
      fp_form.AddDomainIntegrator(new DiffusionIntegrator(*ak_coeff));
      fp_form.Assemble();
      fp_form.Finalize();
      fp_form.FormSystemMatrix(pres_ess_tdofs, Fp);

      Mp.As<HypreParMatrix>()->GetDiag(M_local);
      Mp_inv = new UMFPackSolver(M_local);

      Lp.As<HypreParMatrix>()->GetDiag(Lp_local);
      Lp_inv = new UMFPackSolver(Lp_local);

      pcd = new PCD(*Mp_inv, *Lp_inv, Fp);
   }

   ~PCDBuilder()
   {
      delete pcd;
      delete Lp_inv;
      delete Mp_inv;
   }

   Solver& GetSolver() { return *pcd; }

   ParBilinearForm mp_form;
   OperatorHandle Mp;

   ParBilinearForm lp_form;
   OperatorHandle Lp;

   ParBilinearForm fp_form;
   OperatorHandle Fp;

   SparseMatrix M_local;
   SparseMatrix Lp_local;

   UMFPackSolver *Lp_inv;
   UMFPackSolver *Mp_inv;

   PCD *pcd = nullptr;
};

}