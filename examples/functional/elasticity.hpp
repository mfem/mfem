#ifndef MFEM_ELASTICITY_HPP
#define MFEM_ELASTICITY_HPP
#include "mfem.hpp"
#include "myad.hpp"

namespace mfem
{

MAKE_AD_FUNCTION(SIMPFunction, T, x, E,
{
   T result = T();
   real_t simp_exp = E[x.Size()];
   for (int i = 0; i < x.Size(); i++)
   {
      result += pow(max(x[i], 0.0), simp_exp) * E[i];
   }
   return result + 1e-06;
});

MAKE_AD_FUNCTION(DensityFunction, T, x, rho,
{
   T result = T();
   for (int i=0; i<x.Size(); i++)
   {
      result += rho[i]*x[i];
   }
   return result;
});

inline void NormalizeLatent(GridFunction &latent)
{
   const int vdim = latent.VectorDim();
   const int n = latent.Size() / vdim;
   MFEM_VERIFY(latent.Size() % vdim == 0,
               "Latent vector size is not divisible by vector dimension");
   Vector curr_latent(latent.VectorDim());
   for (int i=0; i<n; i++)
   {
      for (int d=0; d<vdim; d++)
      {
         curr_latent[d] = latent[d*n + i];
      }
      curr_latent -= curr_latent.Max();
      for (int d=0; d<vdim; d++)
      {
         latent[d*n + i] = curr_latent[d];
      }
   }
}

// Find mass correction. Also it normalizes the latent value so that
// max_i psi_i(x) = 0.0
inline void MassProjection(Coefficient &rho_cf, QuadratureFunction &rho_qf,
                           GridFunction &latent, GridFunction &drho_gf,
                           const real_t target_mass,
                           const int max_iter=1000,
                           const real_t tol=1e-10,
                           int print_level=0)
{
   rho_cf.Project(rho_qf);
   real_t mass_diff = rho_qf.Integrate();
   real_t dc = std::pow(2, 6);
   real_t fa(mass_diff), fb(mass_diff);
   real_t a(0), b(0);
   if (mass_diff < target_mass)
   {
      while (mass_diff < target_mass)
      {
         fa = mass_diff;
         latent.Add(dc, drho_gf);
         rho_cf.Project(rho_qf);
         NormalizeLatent(latent);
         mass_diff = rho_qf.Integrate();
      }
      fb = mass_diff;
      b = 0;
      a = -dc;
   }
   else
   {
      while (mass_diff > target_mass)
      {
         fb = mass_diff;
         latent.Add(-dc, drho_gf);
         NormalizeLatent(latent);
         rho_cf.Project(rho_qf);
         mass_diff = rho_qf.Integrate();
      }
      fa = mass_diff;
      a = 0;
      b = dc;
   }
   NormalizeLatent(latent);
   fa -= target_mass;
   fb -= target_mass;
   bool converged = false;
   int side = 0;
   for (int i=0; i<max_iter; i++)
   {
      real_t c = (fa*b - fb*a) / (fa - fb);
      if (fabs(b-a) < 1e-8) { break; }
      latent.Add(c, drho_gf);
      NormalizeLatent(latent);
      rho_cf.Project(rho_qf);
      mass_diff = rho_qf.Integrate() - target_mass;
      if (print_level > 1)
      {
         out << "Mass Projection Iter " << i+1 << " Mass diff: " << mass_diff << std::endl;
      }
      if (mass_diff * fb > 0)
      {
         b = 0; fb = mass_diff; a = a-c;
         if (side == -1) { fa *= 0.5; }
         side = -1;
      }
      else
      {
         a = 0; fa = mass_diff; b = b-c;
         if (side == 1) { fb *= 0.5; }
         side = 1;
      }
      if (fabs(mass_diff) < 1e-08)
      {
         converged = true;
         if (print_level > 0)
         {
            out << "Volume correction converged: " << i << std::endl;
         }
         break;
      }
   }
}

// State solver with given control.
// For simplicity, this solver maps control L-vector to state L-vector.
class StateSolver : public Solver
{
   // member variables
public:
protected:
   std::vector<FiniteElementSpace*> ctrl_fes;
   std::vector<FiniteElementSpace*> state_fes;
   mutable std::vector<std::unique_ptr<GridFunction>> ctrl_gfs;
   mutable std::vector<std::unique_ptr<GridFunction>> state_gfs;
   mutable std::vector<std::unique_ptr<Vector>> state_vecs;
   mutable std::vector<LinearForm*> loads;
   mutable std::vector<std::unique_ptr<Vector>> load_vecs;
   std::vector<Array<int>> ess_tdof_list;
   bool reassemble_load=false;
#ifdef MFEM_USE_MPI
   std::vector<ParFiniteElementSpace*> ctrl_pfes;
   std::vector<ParFiniteElementSpace*> state_pfes;
   const bool parallel=false;
   const MPI_Comm comm=MPI_COMM_NULL;
#endif
private:
   // member functions
public:
   StateSolver(std::vector<FiniteElementSpace*> ctrl_fes,
               std::vector<FiniteElementSpace*> state_fes)
      : Solver(), ctrl_fes(ctrl_fes), state_fes(state_fes)
      , loads(state_fes.size())
   {
      MFEM_VERIFY(ctrl_fes.size() > 0,
                  "At least one control space should be provided");
      MFEM_VERIFY(state_fes.size() > 0,
                  "At least one state space should be provided");
      Initialize();
   }

#ifdef MFEM_USE_MPI
   StateSolver(std::vector<ParFiniteElementSpace*> ctrl_fes,
               std::vector<ParFiniteElementSpace*> state_fes)
      : Solver(), loads(state_fes.size()), load_vecs(state_fes.size())
      , ctrl_pfes(ctrl_fes), state_pfes(state_fes)
      , parallel(true)
      , comm(ctrl_fes.size() > 0 ? ctrl_fes[0]->GetComm() : MPI_COMM_NULL)
   {
      MFEM_VERIFY(ctrl_fes.size() > 0,
                  "At least one control space should be provided");
      MFEM_VERIFY(state_fes.size() > 0,
                  "At least one state space should be provided");
      for (auto fes : ctrl_pfes) { this->ctrl_fes.push_back(fes); }
      for (auto fes : state_pfes) { this->state_fes.push_back(fes); }
      Initialize();
   }
   MPI_Comm GetComm() { return comm; }
#endif
   bool IsParallel() { return parallel; }

   void SetOperator(const Operator &op) override
   {
      MFEM_ABORT("StateSolver does not support SetOperator");
   }

   // state solve with given control L-vector.
   virtual void Mult(const Vector &x, Vector &y) const override = 0;

   // Adjoint state solve (if exists)
   virtual void MultTranspose(const Vector &x, Vector &y) const override
   {
      MFEM_ABORT("Not implemented. Provide adjoint solver");
   }

   virtual void MarkEssentialBC(Array<int> ess_bdr, const int space_id=0,
                                const int comp=-1)
   {
      state_fes[space_id]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list[space_id],
         comp);
   }

   // Set Dirichlet Boundary Conditions for multiple components.
   // ess_bdr is a 2D array with d+1 rows and num attributes columns.
   // For i=0,...,d-1, ess_bdr[i,:] is the essential boundary for component i.
   // For i=d, ess_bdr[i,:] is the essential boundary for all components.
   virtual void MarkEssentialBC(Array2D<int> ess_bdr, const int space_id=0)
   {
      Array<int> ess_bdr_comp;
      Array<int> ess_tdof_list_comp;
      for (int i=0; i<ess_bdr.NumRows()-1; i++)
      {
         ess_bdr.GetRow(i, ess_bdr_comp);
         state_fes[space_id]->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i);
         ess_tdof_list[space_id].Append(ess_tdof_list_comp);
      }
      ess_bdr.GetRow(ess_bdr.NumRows()-1, ess_bdr_comp);
      state_fes[space_id]->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, -1);
   }

   virtual void SetLoad(LinearForm &lf, const int space_id=0)
   {
      loads[space_id] = &lf;
   }

protected:

   void Initialize()
   {
      ctrl_gfs.resize(ctrl_fes.size());
      state_gfs.resize(ctrl_fes.size());
      state_vecs.resize(state_fes.size());
      loads.resize(state_fes.size());
      load_vecs.resize(state_fes.size());
      ess_tdof_list.resize(state_fes.size());
      width = 0;
      for (auto fes : ctrl_fes) { width += fes->GetVSize(); }
      height = 0;
      for (auto fes : state_fes) { height += fes->GetVSize(); }

      if (parallel) // par grid functions
      {
#ifdef MFEM_USE_MPI
         for (int i=0; i<ctrl_pfes.size(); i++)
         {
            ctrl_gfs[i] = std::make_unique<ParGridFunction>(ctrl_pfes[i], (real_t*)nullptr);
         }
         for (int i=0; i<state_pfes.size(); i++)
         {
            state_gfs[i] = std::make_unique<ParGridFunction>(state_pfes[i],
                           (real_t*)nullptr);
            load_vecs[i].reset(state_pfes[i]->NewTrueDofVector());
            state_vecs[i].reset(state_pfes[i]->NewTrueDofVector());
         }
#endif
      }
      else // serial grid functions
      {
         for (int i=0; i<ctrl_fes.size(); i++)
         {
            ctrl_gfs[i] = std::make_unique<GridFunction>(ctrl_fes[i], (real_t*)nullptr);
         }
         for (int i=0; i<state_fes.size(); i++)
         {
            state_gfs[i] = std::make_unique<GridFunction>(state_fes[i], (real_t*)nullptr);
            load_vecs[i] = std::make_unique<Vector>(state_fes[i]->GetTrueVSize());
            state_vecs[i] = std::make_unique<Vector>(state_fes[i]->GetTrueVSize());
         }
      }
   }
private:
};

class ElasticityStateSolver : public StateSolver
{
public:
   ElasticityStateSolver(FiniteElementSpace &ctrl_fes,
                         FiniteElementSpace &state_fes,
                         ADVecGridFuncCF &lambda_cf,
                         ADVecGridFuncCF &mu_cf)
      : StateSolver(std::vector<FiniteElementSpace*>({&ctrl_fes}),
   std::vector<FiniteElementSpace*>({&state_fes}))
   , lambda_cf(lambda_cf), mu_cf(mu_cf)
   {
      this->lambda_cf.SetGridFunction(*ctrl_gfs[0]);
      this->mu_cf.SetGridFunction(*ctrl_gfs[0]);
      CreateSolvers();
   }

#ifdef MFEM_USE_MPI
   ElasticityStateSolver(ParFiniteElementSpace &ctrl_fes,
                         ParFiniteElementSpace &state_fes,
                         ADVecGridFuncCF &lambda_cf,
                         ADVecGridFuncCF &mu_cf)
      : StateSolver(std::vector<ParFiniteElementSpace*>({&ctrl_fes}),
   std::vector<ParFiniteElementSpace*>({&state_fes}))
   , lambda_cf(lambda_cf), mu_cf(mu_cf)
   {
      this->lambda_cf.SetGridFunction(*ctrl_gfs[0]);
      this->mu_cf.SetGridFunction(*ctrl_gfs[0]);
      CreateSolvers();
   }
#endif

   void Mult(const Vector &x, Vector &y) const override
   {
      ctrl_gfs[0]->MakeRef(ctrl_fes[0], const_cast<Vector&>(x), 0);
      state_gfs[0]->MakeRef(state_fes[0], y, 0);

      state_bf->Update(); // Reset previously assembled data
      state_bf->Assemble();
      loads[0]->Assemble();

      OperatorHandle A;
      state_bf->FormLinearSystem(ess_tdof_list[0], *state_gfs[0], *loads[0], A,
                                 *state_vecs[0], *load_vecs[0], true);

      solver->SetOperator(*A);
      solver->Mult(*load_vecs[0], *state_vecs[0]);

      state_bf->RecoverFEMSolution(*state_vecs[0], *loads[0],
                                   *state_gfs[0]);
   }

protected:
   void CreateSolvers()
   {
      if (parallel)
      {
#ifdef MFEM_USE_MPI
         state_bf = std::make_unique<ParBilinearForm>(state_pfes[0]);
#endif
      }
      else { state_bf = std::make_unique<BilinearForm>(state_fes[0]); }

      state_bf->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf, mu_cf));

      if (parallel)
      {
#ifdef MFEM_USE_MPI
         prec = std::make_unique<HypreBoomerAMG>();
         static_cast<HypreBoomerAMG&>(*prec).SetPrintLevel(0);
         static_cast<HypreBoomerAMG&>(*prec).SetElasticityOptions(state_pfes[0]);
         solver = std::make_unique<HyprePCG>(GetComm());
         static_cast<HyprePCG&>(*solver).SetPreconditioner(
            static_cast<HypreBoomerAMG&>(*prec));
         static_cast<HyprePCG&>(*solver).SetPrintLevel(0);
         static_cast<HyprePCG&>(*solver).SetAbsTol(1e-9);
         static_cast<HyprePCG&>(*solver).SetMaxIter(1e08);
         static_cast<HyprePCG&>(*solver).iterative_mode = true;
#endif
      }
      else
      {
         prec = std::make_unique<GSSmoother>();
         solver = std::make_unique<CGSolver>();
         static_cast<CGSolver&>(*solver).SetPreconditioner(*prec);
         static_cast<CGSolver&>(*solver).SetPrintLevel(0);
         static_cast<CGSolver&>(*solver).SetAbsTol(1e-9);
         static_cast<CGSolver&>(*solver).SetRelTol(1e-9);
         static_cast<CGSolver&>(*solver).SetMaxIter(1e08);
         static_cast<CGSolver&>(*solver).iterative_mode = true;
      }
   }

private:
protected:
   mutable ADVecGridFuncCF lambda_cf, mu_cf;  // Lame parameters
   std::unique_ptr<Solver> prec;
   std::unique_ptr<Solver> solver;
private:
   mutable std::unique_ptr<BilinearForm> state_bf;
};

class VectorGradientGridFunctionCoefficient : public MatrixCoefficient
{
public:
   VectorGradientGridFunctionCoefficient(GridFunction &gf)
      : MatrixCoefficient(gf.VectorDim(), gf.FESpace()->GetMesh()->SpaceDimension())
      , gf(gf) { }

   void Eval(DenseMatrix &m, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      m.SetSize(height, width);
      gf.GetVectorGradient(T, m);
   }
private:
   GridFunction &gf;
};

class ComplianceCoefficient : public Coefficient
{
public:
   ComplianceCoefficient(GridFunction &u,
                         Coefficient &lambda,
                         Coefficient &mu)
      : u(u), lambda(lambda), mu(mu)
   { }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // v = dlambda_drho * gradu + dmu_drho * gradadju
      real_t L = lambda.Eval(T, ip);
      real_t M = mu.Eval(T, ip);

      u.GetVectorGradient(T, gradu_mat);

      real_t divu = gradu_mat.Trace();
      gradu_mat.Symmetrize();

      real_t result = divu*divu*L;
      int dim = T.GetSpaceDim();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            result += M*gradu_mat(i,j)*(gradu_mat(i,j)+gradu_mat(j,i));
         }
      }
      return result;
   }
private:
   GridFunction &u;
   Coefficient &lambda, &mu;
   DenseMatrix gradu_mat;
};
class DensityDerivative : public VectorCoefficient
{
public:
   DensityDerivative(GridFunction &u,
                     VectorCoefficient &dlambda_drho,
                     VectorCoefficient &dmu_drho)
      : VectorCoefficient(dlambda_drho.GetVDim()),
        u(u), dlambda_drho(dlambda_drho), dmu_drho(dmu_drho)
   { }

   void Eval(Vector &v, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      lambda_vec.SetSize(dlambda_drho.GetVDim());
      mu_vec.SetSize(dmu_drho.GetVDim());
      v.SetSize(lambda_vec.Size());
      v = 0.0;

      // v = dlambda_drho * gradu + dmu_drho * gradadju
      dlambda_drho.Eval(lambda_vec, T, ip);
      dmu_drho.Eval(mu_vec, T, ip);

      u.GetVectorGradient(T, gradu_mat);

      MFEM_VERIFY(lambda_vec.CheckFinite() == 0,
                  "Lambda not finite (" << lambda_vec.Min() << ", " << lambda_vec.Max() << ").");
      MFEM_VERIFY(lambda_vec.CheckFinite() == 0,
                  "Mu not finite (" << mu_vec.Min() << ", " << mu_vec.Max() << ").");

      real_t divu = gradu_mat.Trace();
      gradu_mat.Symmetrize();

      v.Set(divu*divu, lambda_vec);
      int dim = T.GetSpaceDim();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            v.Add(gradu_mat(i,j)*(gradu_mat(i,j)+gradu_mat(j,i)), mu_vec);
         }
      }
      v.Neg();
   }
private:
   GridFunction &u;
   VectorCoefficient &dlambda_drho, &dmu_drho;
   Vector lambda_vec, mu_vec;
   DenseMatrix gradu_mat;
};

class MappedVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   std::function<void(const Vector &, Vector &)> f;
   GridFunction &gf;
   Vector v;
public:
   MappedVectorGridFunctionCoefficient(GridFunction &gf,
                                       std::function<void(const Vector &, Vector &)> f)
      : VectorCoefficient(gf.VectorDim()), gf(gf), f(f), v(gf.VectorDim()) { }

   void Eval(Vector &w, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      w.SetSize(gf.VectorDim());
      gf.GetVectorValue(T, ip, v);
      f(v, w); // Apply the mapping function
   }
};

class DGRieszLFIntegrator : public LinearFormIntegrator
{
public:
   DGRieszLFIntegrator(VectorCoefficient &targ)
      : dual_form(targ)
      , mass()
   {}

   void AssembleRHSElementVect(const FiniteElement &dg_fe,
                               ElementTransformation &Trans,
                               Vector &elvec) override
   {
      elvec.SetSize(dg_fe.GetDof());
      elvec = 0.0;
      dual_vec.SetSize(dg_fe.GetDof());
      dual_vec = 0.0;
      M.SetSize(dg_fe.GetDof());
      M = 0.0;

      dual_form.AssembleRHSElementVect(dg_fe, Trans, dual_vec);
      mass.AssembleElementMatrix(dg_fe, Trans, M);
      invM.Factor(M);
      invM.Mult(dual_vec, elvec);
   }

private:
   VectorDomainLFIntegrator dual_form;
   MassIntegrator mass;
   mutable Vector dual_vec;
   mutable DenseMatrix M;
   mutable DenseMatrixInverse invM;
};

// Make Autodiff Functor
// @param name will be the name of the structure
// @param T will be the name of the templated type
// @param param is additional parameter name (will not be differentiated)
// @param body is the main function body. Use T() to create 0 T-typed value.
#define SEQUENTIAL_PRINT(comm, stream, body) \
{                                            \
int myid_, num_procs_;                       \
MPI_Comm_rank(comm, &myid_);                 \
MPI_Comm_size(comm, &num_procs_);            \
for (int i_=0; i_<num_procs_; i_++)          \
{                                            \
   if (i_ == myid_)                          \
   {                                         \
     stream << body << std::endl;            \
   }                                         \
   MPI_Barrier(comm);                        \
}}


} // namespace mfem

#endif // MFEM_ELASTICITY_HPP
