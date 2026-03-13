#include "mfem.hpp"

using namespace mfem;

void InitialCondition(const Vector &x, Vector &u);
real_t GapFunction(const Vector &x);
void VisualizeSolution(socketstream &sol_sock, const Mesh &mesh, const GridFunction &u_gf);

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
      : psi(&psi_), min_val(min_val_), max_val(max_val_) { }

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
   std::unique_ptr<FiniteElementCollection> parent_trace_fec;
   std::unique_ptr<FiniteElementSpace> parent_trace_fes; // trial space restricted to the sub mesh
   std::unique_ptr<MixedBilinearForm> sub_bf; // sub mesh bilinear form

   std::unique_ptr<SparseMatrix> P; // Prolongation from sub to parent test space
   std::unique_ptr<SparseMatrix> mat; // the final coupling matrix

public:
   ParentToSubMixedBilinearForm(FiniteElementSpace *trial_parent_fes_,
                                FiniteElementSpace *test_sub_fes_)
      : trial_parent_fes(trial_parent_fes_), test_sub_fes(test_sub_fes_)
   {
      auto *sub_mesh = dynamic_cast<SubMesh*>(test_sub_fes->GetMesh());
      MFEM_VERIFY(sub_mesh != nullptr,
                  "SubMeshToParentCouplingTransfer: sub_fes should be defined on a SubMesh");

      auto *parent_mesh = trial_parent_fes->GetMesh();
      MFEM_VERIFY(sub_mesh->GetParent() == parent_mesh,
                  "SubMeshToParentCouplingTransfer: parent_fes mesh should be the parent of sub_fes mesh");
      
      auto parent_fec = trial_parent_fes->FEColl();
      parent_trace_fec.reset(parent_fec->GetTraceCollection());
      parent_trace_fes = std::make_unique<FiniteElementSpace>(
                            sub_mesh, parent_trace_fec.get(), trial_parent_fes->GetVDim()
                         );      
      
      Array<int> sub_to_parent;
      SubMeshUtils::BuildVdofToVdofMap(*parent_trace_fes, *trial_parent_fes_,
                                       sub_mesh->GetFrom(), sub_mesh->GetParentElementIDMap(),
                                       sub_to_parent);

      const int ndof_sub = parent_trace_fes->GetVSize();
      const int ndof_parent = trial_parent_fes_->GetVSize();

      P = std::make_unique<SparseMatrix>(ndof_sub, ndof_parent);

      for (int i = 0; i < ndof_sub; i++)
      {
         int I = sub_to_parent[i];
         if (I >= 0) { P->Add(i, I, 1.0); }
      }
      P->Finalize();

      sub_bf = std::make_unique<MixedBilinearForm>(parent_trace_fes.get(), test_sub_fes);
   }
   virtual ~ParentToSubMixedBilinearForm() = default;

   FiniteElementSpace &GetTrialTraceSpace() const { return *parent_trace_fes; }

   virtual void AddBoundaryDomainIntegrator(BilinearFormIntegrator *bdr_intg)
   {
      sub_bf->AddDomainIntegrator(bdr_intg);
   }

   virtual void Assemble()
   {
      sub_bf->Assemble();
      sub_bf->Finalize();
      mat.reset(mfem::Mult(sub_bf->SpMat(), *P));
   }

   virtual void Finalize()
   {
      MFEM_VERIFY(mat!=nullptr, "ParentToSubMixedBilinearForm: mat not built");
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


