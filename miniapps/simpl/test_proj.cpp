#include "mfem.hpp"
#include "logger.hpp"
#include "funs.hpp"


using namespace mfem;


class HellingerDerivativeMatrixCoefficient : public MatrixCoefficient
{
   // attributes
private:
   GridFunction * latent_gf;
   Coefficient *r_min;
   bool own_rmin;
   Vector latent_val;
protected:
public:
   // methods
private:
protected:
public:
   HellingerDerivativeMatrixCoefficient(GridFunction * latent_gf,
                                        const real_t r_min)
      :MatrixCoefficient(latent_gf->VectorDim()), latent_gf(latent_gf),
       r_min(new ConstantCoefficient(r_min)), own_rmin(true),
       latent_val(latent_gf->VectorDim()) { }
   HellingerDerivativeMatrixCoefficient(GridFunction * latent_gf,
                                        Coefficient &r_min)
      :MatrixCoefficient(latent_gf->VectorDim()), latent_gf(latent_gf), r_min(&r_min),
       own_rmin(false), latent_val(latent_gf->VectorDim()) { }
   ~HellingerDerivativeMatrixCoefficient() {if (own_rmin) {delete r_min;}}

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      latent_gf->GetVectorValue(T.ElementNo, T.GetIntPoint(), latent_val);
      const real_t norm2 = latent_val*latent_val;
      const real_t g = std::pow(norm2 + 1.0, -3.0/2.0);
      K.Diag(norm2 + 1, latent_val.Size());
      AddMult_a_VVt(-1.0, latent_val, K);
      K *= g;
   }
};

class HellingerLatent2PrimalCoefficient : public VectorCoefficient
{
   // attributes
private:
   GridFunction * latent_gf;
   Coefficient *r_min;
   bool own_rmin;
protected:
public:
   // methods
private:
protected:
public:
   HellingerLatent2PrimalCoefficient(GridFunction *latent_gf, const real_t r_min)
      :VectorCoefficient(latent_gf->VectorDim()), latent_gf(latent_gf),
       r_min(new ConstantCoefficient(r_min)), own_rmin(true) {}
   HellingerLatent2PrimalCoefficient(GridFunction *latent_gf, Coefficient &r_min)
      :VectorCoefficient(latent_gf->VectorDim()), latent_gf(latent_gf),
       r_min(&r_min), own_rmin(false) {}
   void SetLengthScale(const real_t new_rmin)
   {
      if (own_rmin) {delete r_min;}
      r_min = new ConstantCoefficient(new_rmin);
      own_rmin = true;
   }

   void SetLengthScale(Coefficient &new_rmin)
   {
      if (own_rmin) {delete r_min;}
      r_min = &new_rmin;
      own_rmin = false;
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      latent_gf->GetVectorValue(T.ElementNo, T.GetIntPoint(), V);
      V /= std::sqrt(V*V + 1.0);
   }
};

class FermiDiracDerivativeVectorCoefficient : public Coefficient
{
   // attributes
private:
   GridFunction *latent_gf;
   Coefficient *minval;
   Coefficient *maxval;
   bool own_minmax;
protected:
public:
   // methods
private:
protected:
public:
   FermiDiracDerivativeVectorCoefficient(GridFunction *latent_gf,
                                         const real_t minval=0,
                                         const real_t maxval=0)
      :Coefficient(), latent_gf(latent_gf),
       minval(new ConstantCoefficient(minval)),
       maxval(new ConstantCoefficient(maxval)), own_minmax(true) {}
   FermiDiracDerivativeVectorCoefficient(GridFunction *latent_gf,
                                         Coefficient &minval,
                                         Coefficient &maxval)
      :Coefficient(), latent_gf(latent_gf),
       minval(&minval), maxval(&maxval), own_minmax(false) {}
   ~FermiDiracDerivativeVectorCoefficient() { if (own_minmax) {delete minval; delete maxval;} }

   void SetGridFunction(GridFunction *new_gf) { latent_gf = new_gf; }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      const real_t m = minval->Eval(T, T.GetIntPoint());
      const real_t M = maxval->Eval(T, T.GetIntPoint());
      const real_t val = sigmoid(latent_gf->GetValue(T.ElementNo, T.GetIntPoint()));
      return (M - m)*(val*(1.0-val));
   }
};


class FermiDiracLatent2PrimalCoefficient : public Coefficient
{
   // attributes
private:
   GridFunction *latent_gf;
   Coefficient *minval;
   Coefficient *maxval;
   bool own_minmax;
protected:
public:
   // methods
private:
protected:
public:
   FermiDiracLatent2PrimalCoefficient(GridFunction *gf, const real_t minval=0,
                                      const real_t maxval=0)
      :Coefficient(), latent_gf(gf),
       minval(new ConstantCoefficient(minval)),
       maxval(new ConstantCoefficient(maxval)), own_minmax(true) {}
   FermiDiracLatent2PrimalCoefficient(GridFunction *gf, Coefficient &minval,
                                      Coefficient &maxval)
      :Coefficient(), latent_gf(gf),
       minval(&minval), maxval(&maxval), own_minmax(false) {}
   ~FermiDiracLatent2PrimalCoefficient() { if (own_minmax) {delete minval; delete maxval;} }
   void SetGridFunction(GridFunction *new_gf) { latent_gf = new_gf; }
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      const real_t m = minval->Eval(T, T.GetIntPoint());
      const real_t M = maxval->Eval(T, T.GetIntPoint());
      const real_t val = sigmoid(latent_gf->GetValue(T.ElementNo, T.GetIntPoint()));
      return (M - m)*val + m;
   }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int ref_levels = 5;
   const int order = 2;
   const int dim = 2;
   const real_t length_scale = 0.1;
   const real_t rho_min = 1e-08;

   std::unique_ptr<ParMesh> pmesh;
   {
      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
      pmesh.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      mesh.Clear();
      for (int i=0; i<ref_levels; i++) {pmesh->UniformRefinement();}
   }
   FunctionCoefficient rho_targ([](const Vector &x) { return (real_t)((x[0] > 0.5) | (x[1]<0.5)); });

   RT_FECollection RT_fec(order, dim);
   DG_FECollection DG_fec(order, dim);

   ParFiniteElementSpace RT_fes(pmesh.get(), &RT_fec);
   ParFiniteElementSpace DG_fes(pmesh.get(), &DG_fec);

   Array<int> offsets(4), true_offsets(4);
   offsets[0] = 0;
   offsets[1] = RT_fes.GetVSize();
   offsets[2] = DG_fes.GetVSize();
   offsets[3] = DG_fes.GetVSize();
   offsets.PartialSum();
   BlockVector x(offsets), b(offsets);
   BlockVector x_old(offsets);

   true_offsets[0] = 0;
   true_offsets[1] = RT_fes.GetTrueVSize();
   true_offsets[2] = DG_fes.GetTrueVSize();
   true_offsets[3] = DG_fes.GetTrueVSize();
   offsets.PartialSum();
   BlockVector x_tv(offsets), b_tv(offsets);
   x_tv = 0.0;
   x_tv.GetBlock(1) = 0.5;
   b_tv = 0.0;

   ParGridFunction Psi(&RT_fes, x.GetBlock(0));
   ParGridFunction rho(&DG_fes, x.GetBlock(1));
   ParGridFunction psi(&DG_fes, x.GetBlock(2));
   ParGridFunction Psi_k(&RT_fes);
   ParGridFunction psi_k(&DG_fes);

   Psi.Distribute(x_tv.GetBlock(0));
   rho.Distribute(x_tv.GetBlock(1));
   psi.Distribute(x_tv.GetBlock(2));
   HellingerDerivativeMatrixCoefficient DSigma(&Psi, length_scale);
   HellingerLatent2PrimalCoefficient grad_rho_cf(&Psi, length_scale);
   HellingerLatent2PrimalCoefficient grad_rho_k_cf(&Psi_k, length_scale);
   FermiDiracDerivativeVectorCoefficient dsigma(&psi, rho_min, 1.0);
   FermiDiracLatent2PrimalCoefficient rho_cf(&psi, rho_min, 1.0);
   FermiDiracLatent2PrimalCoefficient rho_k_cf(&psi_k, rho_min, 1.0);

   BlockOperator blockOp(true_offsets);
   blockOp.owns_blocks = true;

   ParBilinearForm DSigmaM(&RT_fes);
   DSigmaM.AddDomainIntegrator(new VectorFEMassIntegrator(DSigma));

   ParBilinearForm dsigmaM(&DG_fes);
   dsigmaM.AddDomainIntegrator(new MassIntegrator(dsigma));

   ParMixedBilinearForm divOp(&RT_fes, &DG_fes);
   divOp.AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   {
      auto D = divOp.ParallelAssemble();
      blockOp.SetBlock(1, 0, D);
      blockOp.SetBlock(0, 1, new TransposeOperator(D));
   }

   ParLinearForm gradL2(&DG_fes, b.GetBlock(1).GetData());
   ProductCoefficient alpha_rho_k_cf(1.0, rho_k_cf);
   gradL2.AddDomainIntegrator(new DomainLFIntegrator(alpha_rho_k_cf));
   ProductCoefficient neg_alpha_rho_targ(-1.0, rho_targ);
   gradL2.AddDomainIntegrator(new DomainLFIntegrator(neg_alpha_rho_targ));

   real_t alpha = 1.0;
   while (true)
   {
      alpha_rho_k_cf.SetAConst(alpha);
      neg_alpha_rho_targ.SetAConst(alpha);

      gradL2.Assemble();
      gradL2.SyncAliasMemory(b);
      gradL2.ParallelAssemble(b_tv.GetBlock(1));
      b_tv.GetBlock(1).SyncAliasMemory(b_tv);
   }


   return 0;
}

