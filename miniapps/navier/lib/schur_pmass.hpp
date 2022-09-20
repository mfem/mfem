#include <mfem.hpp>

namespace mfem
{

class PMass : public Solver
{
public:
   PMass(Solver &Mp_inv_) :
      Solver(Mp_inv_.Width()),
      Mp_inv(Mp_inv_)
   { }

   void Mult(const Vector &x, Vector &y) const override
   {
      Mp_inv.Mult(x, y);
   }

   void SetOperator(const Operator &op) override {}

   Solver &Mp_inv;
};

class PMassBuilder
{
public:
   PMassBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs,
                Coefficient *am_coeff) :
      mp_form(&pres_fes)
   {
      mp_form.AddDomainIntegrator(new MassIntegrator);
      mp_form.Assemble();
      mp_form.Finalize();
      mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);
      Mp.As<HypreParMatrix>()->GetDiag(M_local);
      Mp_inv = new UMFPackSolver(M_local);

      pmass = new PMass(*Mp_inv);
   }

   ~PMassBuilder()
   {
      delete pmass;
      delete Mp_inv;
   }

   Solver& GetSolver() { return *pmass; }

   ParBilinearForm mp_form;
   OperatorHandle Mp;
   SparseMatrix M_local;
   UMFPackSolver *Mp_inv;
   PMass *pmass = nullptr;
};

}