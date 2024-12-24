#include "mfem.hpp"
#include "logger.hpp"
#include "funs.hpp"


using namespace mfem;

HypreParMatrix *reassemble(ParBilinearForm &op)
{
   SparseMatrix *S = op.LoseMat();
   if (S) { delete S; }
   op.Assemble();
   op.Finalize();
   return op.ParallelAssemble();
}

void MPISequential(std::function<void(int)> f)
{
   for (int i=0; i<Mpi::WorldSize(); i++)
   {
      if (i == Mpi::WorldRank())
      {
         f(i);
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
}

void reassemble(ParLinearForm &b, BlockVector &b_v, BlockVector &b_tv,
                const int block_id)
{
   b.Assemble();
   b.SyncAliasMemory(b_v);
   b.ParallelAssemble(b_tv.GetBlock(block_id));
   b_tv.GetBlock(block_id).SyncAliasMemory(b_tv);
}

HypreParMatrix * LinearFormToSparseMatrix(ParLinearForm &lf)
{
   ParFiniteElementSpace *pfes = lf.ParFESpace();
   // std::unique_ptr<int> i;
   // std::unique_ptr<HYPRE_BigInt> j;
   // std::unique_ptr<real_t> *d;
   Array<int> i; Array<HYPRE_BigInt> j; Vector d;
   Array<HYPRE_BigInt> cols;
   int local_siz = pfes->TrueVSize();
   HYPRE_BigInt global_siz = pfes->GlobalTrueVSize();
   int current;
   i.SetSize(local_siz+1); std::iota(i.begin(), i.end(), 0);
   j.SetSize(local_siz);
   if (HYPRE_AssumedPartitionCheck())
   {
      std::fill(j.begin(), j.end(), 0);
      cols.SetSize(2);
      cols[0] = Mpi::Root() ? 0 : 1;
      cols[1] = 1;
   }
   else
   {
      std::fill(j.begin(), j.end(), 0);
      MFEM_ABORT("Not yet implemented");
      // NOTE: Not sure how to construct since one column is shared with other processors.
      // Doesn't make sense to define increasing cols.
      cols.SetSize(Mpi::WorldSize() + 1);
      cols = 0;
      cols[Mpi::WorldRank()+1] = Mpi::Root() ? 0 :1;
      MPI_Allreduce(MPI_IN_PLACE, cols.GetData(), Mpi::WorldSize() + 1, MPI_INT,
                    MPI_SUM, pfes->GetComm());
      cols.PartialSum();

   }
   d.SetSize(local_siz);

   lf.Assemble(); lf.ParallelAssemble(d);

   // return nullptr;
   return new HypreParMatrix(pfes->GetComm(),
                             local_siz, global_siz, 1,
                             i.GetData(), j.GetData(), d.GetData(),
                             pfes->GetTrueDofOffsets(), cols.GetData());
}

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
      K *= g / r_min->Eval(T, ip);
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
      V /= std::sqrt(V*V + 1.0)*r_min->Eval(T, ip);
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


class VolumeConstraintOperator : public Operator
{
   // attributes
private:
   Coefficient &rho_cf;
   const real_t vol_frac;
   std::unique_ptr<ParLinearForm> b;
protected:
public:
   // methods
private:
protected:
public:
   VolumeConstraintOperator(const int n, Coefficient &rho_cf,
                            ParFiniteElementSpace &pfes, const real_t vol_frac)
      :Operator(1,n), rho_cf(rho_cf), vol_frac(vol_frac)
   {
      b.reset(new ParLinearForm(&pfes));
      b->AddDomainIntegrator(new DomainLFIntegrator(rho_cf));
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      b->Assemble();
      y.SetSize(1);
      real_t vol = b->Sum();
      MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MFEM_MPI_REAL_T, MPI_SUM,
                    b->ParFESpace()->GetComm());
      y = vol - vol_frac;
   }
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   int ref_levels = 5;
   int order = 2;
   int dim = 2;
   real_t length_scale = 0.1;
   real_t rho_min = 1e-08;
   real_t vol_frac = 0.5;

   int max_md_it = 1e04;
   int max_newton_it = 30;
   real_t tol_md = 1e-04;
   real_t tol_newt = 1e-08;

   bool use_paraview = true;

   std::unique_ptr<ParMesh> pmesh;
   {
      int ref_serial = 0;
      while (std::pow(2.0, ref_serial*dim) < Mpi::WorldSize()) { ref_serial++; }
      ref_serial = std::min(ref_serial, ref_levels);
      int ref_parallel = ref_levels - ref_serial;
      Mesh mesh = Mesh::MakeCartesian2D(std::pow(2, ref_serial),
                                        std::pow(2, ref_serial), Element::QUADRILATERAL);
      pmesh.reset(new ParMesh(MPI_COMM_WORLD, mesh));
      mesh.Clear();
      for (int i=0; i<ref_parallel; i++) {pmesh->UniformRefinement();}
   }
   FunctionCoefficient rho_targ([](const Vector &x) { return (real_t)((x[0] > 0.5) | (x[1]<0.5)); });

   RT_FECollection RT_fec(order, dim);
   DG_FECollection DG_fec(order, dim);

   ParFiniteElementSpace RT_fes(pmesh.get(), &RT_fec);
   ParFiniteElementSpace DG_fes(pmesh.get(), &DG_fec);

   HYPRE_BigInt global_RT_dof = RT_fes.GlobalTrueVSize();
   HYPRE_BigInt global_DG_dof = DG_fes.GlobalTrueVSize();
   MPISequential([global_DG_dof, global_RT_dof, &RT_fes, &DG_fes](int i)
   {
      out << "Processor " << i << ": "
          << RT_fes.TrueVSize() << " / " << global_RT_dof << ", "
          << DG_fes.TrueVSize() << " / " << global_DG_dof << std::endl;
   });

   Array<int> offsets(5);
   offsets[0] = 0;
   offsets[1] = RT_fes.GetVSize();
   offsets[2] = DG_fes.GetVSize();
   offsets[3] = DG_fes.GetVSize();
   offsets[4] = 1;
   offsets.PartialSum();
   BlockVector x(offsets), b(offsets);
   BlockVector x_old(offsets);

   Array<int> true_offsets(5);
   true_offsets[0] = 0;
   true_offsets[1] = RT_fes.GetTrueVSize();
   true_offsets[2] = DG_fes.GetTrueVSize();
   true_offsets[3] = DG_fes.GetTrueVSize();
   true_offsets[4] = Mpi::Root() ? 1 : 0;
   true_offsets.PartialSum();
   BlockVector x_tv(true_offsets), b_tv(true_offsets), dummy(true_offsets);
   x_tv.GetBlock(0) = 0.0;                   // Constant solution -> Psi=0
   x_tv.GetBlock(1) = vol_frac;              // Initial Design
   x_tv.GetBlock(2) = invsigmoid(vol_frac);  // initial latent
   b_tv = 0.0;

   ParGridFunction Psi, rho, psi;
   Psi.MakeRef(&RT_fes, x.GetBlock(0), 0);
   rho.MakeRef(&DG_fes, x.GetBlock(1), 0);
   psi.MakeRef(&DG_fes, x.GetBlock(2), 0);
   Psi.Distribute(&(x_tv.GetBlock(0)));
   rho.Distribute(&(x_tv.GetBlock(1)));
   psi.Distribute(&(x_tv.GetBlock(2)));
   ParGridFunction Psi_k(&RT_fes);
   ParGridFunction psi_k(&DG_fes);

   HellingerDerivativeMatrixCoefficient DSigma(&Psi, length_scale);
   HellingerLatent2PrimalCoefficient grad_rho_cf(&Psi, length_scale);
   HellingerLatent2PrimalCoefficient grad_rho_k_cf(&Psi_k, length_scale);
   FermiDiracDerivativeVectorCoefficient dsigma(&psi, rho_min, 1.0);
   FermiDiracLatent2PrimalCoefficient rho_cf(&psi, rho_min, 1.0);
   FermiDiracLatent2PrimalCoefficient rho_k_cf(&psi_k, rho_min, 1.0);

   // Global block operator, matrices are not owned.
   Array2D<HypreParMatrix*> blockOp(4,4);
   std::unique_ptr<HypreParMatrix> glbMat;

   // <Sigma(Psi), Xi>
   std::unique_ptr<HypreParMatrix> DSigmaM;
   ParBilinearForm DSigmaOp(&RT_fes);
   DSigmaOp.AddDomainIntegrator(new VectorFEMassIntegrator(DSigma));

   // <sigma(psi), xi>
   std::unique_ptr<HypreParMatrix> dsigmaM;
   ParBilinearForm dsigmaOp(&DG_fes);
   dsigmaOp.AddDomainIntegrator(new MassIntegrator(dsigma));

   // <div Psi, q>
   ParMixedBilinearForm divOp(&RT_fes, &DG_fes);
   divOp.AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   divOp.Assemble();
   divOp.Finalize();
   std::unique_ptr<HypreParMatrix> D(divOp.ParallelAssemble());
   std::unique_ptr<HypreParMatrix> G(D->Transpose());
   blockOp(1, 0) = D.get();
   blockOp(0, 1) = G.get();

   // -<rho, xi>
   ParBilinearForm neg_Mass(&DG_fes);
   ConstantCoefficient neg_one(-1.0);
   neg_Mass.AddDomainIntegrator(new MassIntegrator(neg_one));
   neg_Mass.Assemble();
   neg_Mass.Finalize();
   std::unique_ptr<HypreParMatrix> neg_M(neg_Mass.ParallelAssemble());
   blockOp(2, 1) = neg_M.get();
   blockOp(1, 2) = neg_M.get();

   // <rho, 1>
   std::unique_ptr<HypreParMatrix> M1;
   ParLinearForm volform(&DG_fes);
   ConstantCoefficient one_cf(1.0);
   volform.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
   M1.reset(LinearFormToSparseMatrix(volform));
   std::unique_ptr<HypreParMatrix> M1T(M1->Transpose());
   M1T->Mult(x_tv.GetBlock(1), b_tv.GetBlock(3)); // compute current volume!
   blockOp(1,3) = M1.get();
   blockOp(3,1) = M1T.get();

   ParLinearForm first_order_optimality(&DG_fes, b.GetBlock(1).GetData());
   ProductCoefficient alpha_rho_k(1.0, rho_k_cf);
   first_order_optimality.AddDomainIntegrator(new DomainLFIntegrator(alpha_rho_k));
   DivergenceGridFunctionCoefficient divPsi_k(&Psi_k);
   ParLinearForm int_rho_targ(&DG_fes, b.GetBlock(1).GetData());
   int_rho_targ.AddDomainIntegrator(new DomainLFIntegrator(rho_targ));
   int_rho_targ.Assemble();

   ParLinearForm GradNewtResidual(&RT_fes, b.GetBlock(0).GetData());
   ScalarVectorProductCoefficient neg_grad_rho(-1.0, grad_rho_cf);
   GradNewtResidual.AddDomainIntegrator(new VectorFEDomainLFIntegrator(
                                           neg_grad_rho));
   MatrixVectorProductCoefficient DSigmaGrad(DSigma, grad_rho_cf);
   GradNewtResidual.AddDomainIntegrator(new VectorFEDomainLFIntegrator(
                                           DSigmaGrad));

   ParLinearForm RhoNewtResidual(&DG_fes, b.GetBlock(2).GetData());
   ProductCoefficient neg_rho_cf(-1.0, rho_cf);
   RhoNewtResidual.AddDomainIntegrator(new DomainLFIntegrator(neg_rho_cf));
   ProductCoefficient dsigma_rho_cf(dsigma, rho_cf);
   RhoNewtResidual.AddDomainIntegrator(new DomainLFIntegrator(dsigma_rho_cf));


   real_t alpha = 1.0;
   int it_md(0), it_newt(0);
   real_t res_md,res_newt, res_linsolver;
   real_t curr_vol;
   TableLogger logger;
   logger.SaveWhenPrint("grad_proj_md.csv");
   logger.Append("it_md", it_md);
   logger.Append("res_md", res_md);
   logger.Append("it_newt", it_newt);
   logger.Append("res_newt", res_newt);
   logger.Append("volume", curr_vol);
   logger.Append("solver_res", res_linsolver);
   std::unique_ptr<ParaViewDataCollection> paraview_dc;
   if (use_paraview)
   {
      paraview_dc.reset(new mfem::ParaViewDataCollection("grad_proj_md",
                                                         pmesh.get()));
      if (paraview_dc->Error()) { use_paraview=false; }
      else
      {
         paraview_dc->SetPrefixPath("ParaView");
         paraview_dc->SetLevelsOfDetail(order);
         paraview_dc->SetDataFormat(VTKFormat::BINARY);
         paraview_dc->SetHighOrderOutput(true);
         // paraview_dc->RegisterField("displacement", &state_gf);
         paraview_dc->RegisterField("density", &rho);
      }
   }

   int ctr=0;

   Vector con_vec(Mpi::Root() ? 1 : 0);
   for (; it_md<max_md_it; it_md++)
   {
      // TODO: Custom update rule for alpha
      alpha = it_md; // update alpha

      // Store the previous
      // We do not store rho_k as it is replaced by sigma(psi_k).
      Psi_k = Psi;
      psi_k = psi;


      // Update RHS of first-order optimality condition
      // without the int_rho_targ which is pre-assembled
      first_order_optimality.Assemble();
      // Add preassembled int_rho_targ with alpha
      first_order_optimality.Add(-alpha, int_rho_targ);
      first_order_optimality.SyncAliasMemory(b);
      first_order_optimality.ParallelAssemble(b_tv.GetBlock(1));
      b_tv.GetBlock(1).SyncAliasMemory(b_tv);

      it_newt = 0;
      for (; it_newt < max_newton_it; it_newt++)
      {
         DSigmaM.reset(reassemble(DSigmaOp));
         dsigmaM.reset(reassemble(dsigmaOp));
         blockOp(0,0) = DSigmaM.get();
         blockOp(2,2) = dsigmaM.get();

         reassemble(GradNewtResidual, b, b_tv, 0);
         reassemble(RhoNewtResidual, b, b_tv, 2);
         MPI_Barrier(MPI_COMM_WORLD);

         glbMat.reset(HypreParMatrixFromBlocks(blockOp));
         MUMPSSolver mumps(MPI_COMM_WORLD);
         mumps.SetPrintLevel(0);
         mumps.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
         mumps.SetOperator(*glbMat);
         mumps.Mult(b_tv, x_tv);
         glbMat->Mult(x_tv, dummy);
         res_linsolver = b_tv.DistanceTo(dummy);
         M1T->Mult(x_tv.GetBlock(1), con_vec);

         Psi.Distribute(&(x_tv.GetBlock(0)));
         rho.Distribute(&(x_tv.GetBlock(1)));
         psi.Distribute(&(x_tv.GetBlock(2)));
         if (Mpi::Root()) { curr_vol = con_vec[0]; }
         // MPISequential([con_vec](int i) {con_vec.Print();});

         res_newt = rho.ComputeH1Error(&rho_cf, &grad_rho_cf);
         logger.Print();
         paraview_dc->SetTime(ctr);
         paraview_dc->SetCycle(ctr++);
         paraview_dc->Save();
         if (res_newt < tol_newt) { break; }
      }
      res_md = rho.ComputeH1Error(&rho_k_cf, &grad_rho_k_cf);
      ctr+=10;
      paraview_dc->SetCycle(ctr);
      paraview_dc->Save();
      if (it_md >= 1 && res_md < tol_md) { break; }
   }
   return 0;
}

