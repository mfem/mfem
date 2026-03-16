#include "mfem.hpp"

using namespace mfem;

void InitialCondition(const Vector &x, Vector &u);
real_t GapFunction(const Vector &x);

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u;
   Coefficient *gap;
   Vector *n_tilde;
   real_t min_val;

public:
   LogarithmGridFunctionCoefficient(GridFunction &u_, Coefficient &gap_, 
                                    Vector &n_tilde_, real_t min_val_=-36)
      : u(&u_), gap(&gap_), n_tilde(&n_tilde_), min_val(min_val_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *psi;
   real_t min_val;
   real_t max_val;

public:
   ExponentialGridFunctionCoefficient(GridFunction &psi_, real_t min_val_=0.0, real_t max_val_=1e6)
      : psi(&psi_), min_val(min_val_), max_val(max_val_)
   {
      MFEM_VERIFY(min_val <= max_val,
                  "ExponentialGridFunctionCoefficient: min_val must not exceed max_val");
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

/// Mixed Bilinear Form between parent mesh trial space and sub mesh test space. 
/// <A\hat{u}, v>_{\Gamma} 
/// where \hat{u} is defined on a parent mesh and v is defined on the sub mesh. 
/// Use @a AddBoundaryDomainIntegrator to add boundary "volume" integrators on the coupling boundary.
class ParentToSubMixedBilinearForm : public Operator
{
private:
   FiniteElementSpace *trial_parent_fes; // trial space defined on a parent mesh
   FiniteElementSpace *test_sub_fes; // test space defined on a sub mesh

   std::unique_ptr<FiniteElementCollection> parent_trace_fec; // FEC for trial space restricted to the sub mesh
   std::unique_ptr<FiniteElementSpace> parent_trace_fes; // trial space restricted to the sub mesh
   std::unique_ptr<MixedBilinearForm> sub_bf; // serial sub mesh bilinear form

   std::unique_ptr<SparseMatrix> R; // restriction matrix that maps from parent to sub mesh trial space
   std::unique_ptr<SparseMatrix> mat; // the final coupling matrix
   Array<int> sub_to_parent; // map from trace dof on sub mesh to dof on parent mesh

#ifdef MFEM_USE_MPI
   std::unique_ptr<ParMixedBilinearForm> p_sub_bf; // parallel sub mesh bilinear form
   std::unique_ptr<OperatorHandle> p_R; // parallel restriction operator that maps from parent to sub mesh trial space
#endif

public:
   ParentToSubMixedBilinearForm(FiniteElementSpace *trial_parent_fes_,
                                FiniteElementSpace *test_sub_fes_)
      : trial_parent_fes(trial_parent_fes_), test_sub_fes(test_sub_fes_)
   {
      auto *parent_mesh = trial_parent_fes->GetMesh();
      MFEM_VERIFY(parent_mesh != nullptr, "trial_parent_fes mesh is null");

#ifdef MFEM_USE_MPI
      auto *sub_mesh = dynamic_cast<SubMesh*>(test_sub_fes->GetMesh());
      auto *sub_pmesh = dynamic_cast<ParSubMesh*>(test_sub_fes->GetMesh());

      MFEM_VERIFY(sub_mesh != nullptr || sub_pmesh != nullptr,
                  "SubMeshToParentCouplingTransfer: test_sub_fes should be defined on a SubMesh");

      if (sub_mesh)
      {
         MFEM_VERIFY(sub_mesh->GetParent() == parent_mesh,
                     "SubMeshToParentCouplingTransfer: trial_parent_fes mesh should be the parent of test_sub_fes mesh");
      }
      else
      {
         MFEM_VERIFY(sub_pmesh->GetParent() == parent_mesh,
                     "SubMeshToParentCouplingTransfer: trial_parent_fes mesh should be the parent of test_sub_fes mesh");
      }
#else
      auto *sub_mesh = dynamic_cast<SubMesh*>(test_sub_fes->GetMesh());
      MFEM_VERIFY(sub_mesh != nullptr,
                  "SubMeshToParentCouplingTransfer: test_sub_fes should be defined on a SubMesh");
      MFEM_VERIFY(sub_mesh->GetParent() == parent_mesh,
                  "SubMeshToParentCouplingTransfer: trial_parent_fes mesh should be the parent of test_sub_fes mesh");
#endif

      bool parallel = false;  

#ifdef MFEM_USE_MPI
      auto *trial_parent_pfes = dynamic_cast<ParFiniteElementSpace*>(trial_parent_fes);
      auto *test_sub_pfes = dynamic_cast<ParFiniteElementSpace*>(test_sub_fes);

      if (!(trial_parent_pfes == nullptr && test_sub_pfes == nullptr))
      {
         MFEM_VERIFY(trial_parent_pfes != nullptr && test_sub_pfes != nullptr,
                     "ParentToSubMixedBilinearForm: trial and test must both be parallel or both be serial");
         MFEM_VERIFY(trial_parent_pfes->GetParMesh()->GetComm() ==
                     test_sub_pfes->GetParMesh()->GetComm(),
                     "ParentToSubMixedBilinearForm: communicator mismatch");
         MFEM_VERIFY(sub_pmesh != nullptr,
                     "ParentToSubMixedBilinearForm: parallel test space must live on ParSubMesh");
         parallel = true;
      }
#endif

      auto *parent_fec = trial_parent_fes->FEColl();
      MFEM_VERIFY(parent_fec != nullptr, "trial_parent_fes finite element collection is null");
      parent_trace_fec.reset(parent_fec->GetTraceCollection());
      MFEM_VERIFY(parent_trace_fec != nullptr,
                  "failed to create trace finite element collection");

#ifdef MFEM_USE_MPI
      if (parallel)
      {
         parent_trace_fes = std::make_unique<ParFiniteElementSpace>(
                               sub_pmesh, parent_trace_fec.get(),
                               trial_parent_fes->GetVDim());

         SubMeshUtils::BuildVdofToVdofMap(*parent_trace_fes, *trial_parent_fes_,
                                          sub_pmesh->GetFrom(),
                                          sub_pmesh->GetParentElementIDMap(),
                                          sub_to_parent);

         auto *parent_trace_pfes = static_cast<ParFiniteElementSpace*>(parent_trace_fes.get());
         p_sub_bf = std::make_unique<ParMixedBilinearForm>(parent_trace_pfes,
                                                           test_sub_pfes);
      }
      else
#endif
      {
         parent_trace_fes = std::make_unique<FiniteElementSpace>(
                               sub_mesh, parent_trace_fec.get(),
                               trial_parent_fes->GetVDim());

         SubMeshUtils::BuildVdofToVdofMap(*parent_trace_fes, *trial_parent_fes_,
                                          sub_mesh->GetFrom(),
                                          sub_mesh->GetParentElementIDMap(),
                                          sub_to_parent);

         sub_bf = std::make_unique<MixedBilinearForm>(parent_trace_fes.get(),
                                                      test_sub_fes);
      }

      const int ndof_sub = parent_trace_fes->GetVSize();
      const int ndof_parent = trial_parent_fes_->GetVSize();
      MFEM_VERIFY(sub_to_parent.Size() == ndof_sub,
                  "sub_to_parent map size does not match the trace space size");

      R = std::make_unique<SparseMatrix>(ndof_sub, ndof_parent);
      for (int i = 0; i < ndof_sub; i++)
      {
         int I = sub_to_parent[i];
         if (I >= 0) { R->Add(i, I, 1.0); }
      }
      R->Finalize();

#ifdef MFEM_USE_MPI
      if (parallel)
      {
         auto *parent_trace_pfes = static_cast<ParFiniteElementSpace*>(parent_trace_fes.get());

         // tdof parent -> vdof parent
         HypreParMatrix *P_parent = trial_parent_pfes->Dof_TrueDof_Matrix();

         // vdof parent -> vdof sub
         HypreParMatrix R_vdof(
            trial_parent_pfes->GetComm(),
            parent_trace_pfes->GlobalVSize(),
            trial_parent_pfes->GlobalVSize(),
            parent_trace_pfes->GetDofOffsets(),
            trial_parent_pfes->GetDofOffsets(),
            R.get());

         // tdof parent -> vdof sub
         std::unique_ptr<HypreParMatrix> tdof_parent_to_vdof_sub(
            ParMult(&R_vdof, P_parent));

         // vdof sub -> tdof sub
         std::unique_ptr<HypreParMatrix> vdof_sub_to_tdof_sub(
            parent_trace_pfes->Dof_TrueDof_Matrix()->Transpose());

         // tdof parent -> tdof sub
         HypreParMatrix *pR_hypre = ParMult(vdof_sub_to_tdof_sub.get(),
                    tdof_parent_to_vdof_sub.get());

         // Rescale the nonzeros of the restriction operator to be 1.0
         *pR_hypre = 1.0;
         p_R = std::make_unique<OperatorHandle>(Operator::Hypre_ParCSR);
         p_R->Reset(pR_hypre);
      }
#endif
   }
   virtual ~ParentToSubMixedBilinearForm() = default;

   FiniteElementSpace &GetTrialTraceSpace() const { return *parent_trace_fes; }

   virtual void AddBoundaryDomainIntegrator(BilinearFormIntegrator *bdr_intg)
   {
      MFEM_VERIFY(bdr_intg != nullptr,
                  "ParentToSubMixedBilinearForm: boundary integrator is null");
#ifdef MFEM_USE_MPI
      if (p_sub_bf)
      {
         p_sub_bf->AddDomainIntegrator(bdr_intg);
         return;
      }
#endif
      MFEM_VERIFY(sub_bf != nullptr,
                  "ParentToSubMixedBilinearForm: serial mixed bilinear form is not initialized");
      sub_bf->AddDomainIntegrator(bdr_intg);
   }

   virtual void Assemble()
   {
#ifdef MFEM_USE_MPI
      if (p_sub_bf)
      {
         p_sub_bf->Assemble();
         return;
      }
#endif
      MFEM_VERIFY(sub_bf != nullptr,
                  "ParentToSubMixedBilinearForm: serial mixed bilinear form is not initialized");
      sub_bf->Assemble();
   }

   virtual void Finalize()
   {
#ifdef MFEM_USE_MPI
      if (p_sub_bf)
      {
         p_sub_bf->Finalize();
      }
      else
#endif
      {
         sub_bf->Finalize();
      }
      
#ifdef MFEM_USE_MPI
      if (p_R) { return; } // skip to parallel assemble
#endif

      MFEM_VERIFY(sub_bf != nullptr,
                  "ParentToSubMixedBilinearForm: serial mixed bilinear form is not initialized");
      MFEM_VERIFY(R != nullptr,
                  "ParentToSubMixedBilinearForm: restriction matrix is not initialized");
      mat.reset(mfem::Mult(sub_bf->SpMat(), *R));
      mat->Finalize();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(mat->Finalized(),
                  "ParentToSubMixedBilinearForm: matrix not finalized");
      mat->Mult(x, y);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      MFEM_VERIFY(mat->Finalized(),
                  "ParentToSubMixedBilinearForm: matrix not finalized");
      return *mat;
   }

   SparseMatrix &SpMat() const
   {
      MFEM_VERIFY(mat->Finalized(),
                  "ParentToSubMixedBilinearForm: matrix not finalized");
      return *mat;
   }

#ifdef MFEM_USE_MPI
   HypreParMatrix *ParallelAssemble()
   {
      OperatorHandle Mh(Operator::Hypre_ParCSR);
      ParallelAssemble(Mh);
      Mh.SetOperatorOwner(false);
      return Mh.As<HypreParMatrix>();
   }

   void ParallelAssemble(OperatorHandle &A)
   {
      MFEM_VERIFY(p_sub_bf != nullptr,
                  "ParentToSubMixedBilinearForm: p_sub_bf is not initialized");
      MFEM_VERIFY(p_R != nullptr,
                  "ParentToSubMixedBilinearForm: p_R is not initialized");

      OperatorHandle Mh_restricted(Operator::Hypre_ParCSR);
      p_sub_bf->ParallelAssemble(Mh_restricted);

      MFEM_VERIFY(Mh_restricted.Type() == Operator::Hypre_ParCSR,
                  "ParentToSubMixedBilinearForm: expected Hypre_ParCSR matrix");
      MFEM_VERIFY(p_R->Type() == Operator::Hypre_ParCSR,
                  "ParentToSubMixedBilinearForm: expected Hypre_ParCSR restriction");

      A.Reset(ParMult(Mh_restricted.As<HypreParMatrix>(),
                      p_R->As<HypreParMatrix>()));
   }
#endif
};

/// Integrator for the mixed form (u · n~, w)_Γ where u is a vector field
/// represented as scalar FE replicated by vdim and w is a scalar test function.
/// The vector coefficient n~ is evaluated at quadrature points.
class MixedFormIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient *N;
   int dim_parent; 
   Vector nval;

public:
   MixedFormIntegrator(VectorCoefficient &n_tilde, int dim_)
      : N(&n_tilde), dim_parent(dim_), nval(dim_) {}

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override
   {
      MFEM_ASSERT(this->VerifyFiniteElementTypes(trial_fe, test_fe),
                  this->FiniteElementTypeFailureMessage());

      const int trial_nd = trial_fe.GetDof();
      const int test_nd  = test_fe.GetDof();

      Vector trial_shape(trial_nd), test_shape(test_nd);

      elmat.SetSize(test_nd, trial_nd * dim_parent);
      elmat = 0.0;

      const IntegrationRule *ir = GetIntegrationRule(trial_fe, test_fe, Trans);
      if (ir == NULL)
      {
         int ir_order = this->GetIntegrationOrder(trial_fe, test_fe, Trans);
         ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
      }

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         Trans.SetIntPoint(&ip);

         // scalar shapes
         trial_fe.CalcPhysShape(Trans, trial_shape);
         test_fe.CalcPhysShape(Trans, test_shape);

         // n~ at this quad point
         N->Eval(nval, Trans, ip);

         const real_t wq = Trans.Weight() * ip.weight;

         for (int i = 0; i < test_nd; i++)
         {
            const real_t wi = test_shape(i) * wq;
            for (int j = 0; j < trial_nd; j++)
            {
               const real_t wij = wi * trial_shape(j);
               for (int c = 0; c < dim_parent; c++)
               {
                  elmat(i, j + c*trial_nd) += wij * nval(c);
               }
            }
         }
      }
   }
protected:
   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW(); }

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == FiniteElement::SCALAR && 
               test_fe.GetRangeType() == FiniteElement::SCALAR);
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedFormIntegrator:  "
             "Trial space must be H1 scalar FE replicated by vdim"
             "and the test space must be L2 scalar";
   }
};
