
#include "parproblems_util.hpp"

class ElasticityOperator
{
private:
   MPI_Comm comm;
   bool nonlinear = false;
   bool formsystem = false;
   ParMesh * pmesh = nullptr;
   Array<int> ess_bdr_attr, ess_bdr_attr_comp;
   Array<int> ess_bdr, ess_tdof_list;
   int order=1, ndofs, ntdofs, gndofs;
   FiniteElementCollection * fec = nullptr;
   ParFiniteElementSpace * fes = nullptr;
   Operator * op = nullptr; // Bilinear or Nonlinear form
   ParLinearForm * b = nullptr;
   ParGridFunction x;
   HypreParMatrix *K=nullptr; // Gradient
   Vector B, X; // Rhs and Solution vector

   ConstantCoefficient pressure_cf;
   // linear elasticity:     
   // c1 = λ (1ˢᵗ Lame parameter), c2 = μ (2ⁿᵈ Lame parameter or shear modulus)
   // non linear elasticity: 
   // c1 = G (shear modulus μ ), c2 = K (bulk modulus) 
   Vector c1, c2;
   PWConstCoefficient c1_cf, c2_cf;
   NeoHookeanModel * material_model = nullptr;

   Vector xref;
   void Init();
   void SetEssentialBC();
   void SetUpOperator();

public:
   ElasticityOperator(ParMesh * pmesh_, Array<int> & ess_bdr_attr_, Array<int> & ess_bdr_attr_comp_,
                       const Vector & E, const Vector & nu, bool nonlinear_ = false); 
   void SetParameters(const Vector & E, const Vector & nu); 
   void SetNeumanPressureData(ConstantCoefficient &f, Array<int> & bdr_marker);
   void SetDisplacementDirichletData(const Vector & delta, Array<int> essbdr); 
   void ResetDisplacementDirichletData();
   void UpdateEssentialBC(Array<int> & ess_bdr_attr_, Array<int> & ess_bdr_attr_comp_);
   void FormLinearSystem();
   void UpdateLinearSystem();
   void UpdateRHS();

   ParMesh * GetMesh() const { return pmesh; }
   MPI_Comm GetComm() const { return comm; }

   ParFiniteElementSpace * GetFESpace() const { return fes; }
   const FiniteElementCollection * GetFECol() const { return fec; }
   int GetNumDofs() const { return ndofs; }
   int GetNumTDofs() const { return ntdofs; }
   int GetGlobalNumDofs() const { return gndofs; }
   const HypreParMatrix * GetOperator() const { return K; }  
   const Vector & GetRHS() const { return B; }

   const ParGridFunction & GetDisplacementGridFunction() const { return x; }
   const Array<int> & GetEssentialDofs() const { return ess_tdof_list; }

   real_t GetEnergy(const Vector & u) const;
   void GetGradient(const Vector & u, Vector & gradE) const;
   HypreParMatrix * GetHessian(const Vector & u);
   bool IsNonlinear() { return nonlinear; }

   ~ElasticityOperator();
};

class OptContactProblem
{
private:
   MPI_Comm comm;
   ElasticityOperator * problem = nullptr;
   ParFiniteElementSpace * vfes = nullptr;
   int dim;
   int dimU, dimM, dimC;
   int dimG;
   Vector ml;
   HypreParMatrix * NegId = nullptr;
   HypreParMatrix * Kref=nullptr;
   Vector grad_ref;
   real_t energy_ref;

   ParMesh * pmesh = nullptr;
   std::set<int> mortar_attrs;
   std::set<int> nonmortar_attrs;
   ParGridFunction * coords = nullptr;
   bool doublepass = false;

   Vector xref;
   Vector xrefbc;
   bool qp;

   void ComputeGapJacobian();
   Vector gapv;
   // Jacobian of gap
   HypreParMatrix * J = nullptr;
   // Transpose of the Jacobian of gap
   HypreParMatrix * Jt = nullptr;
   // Restriction operator to the contact dofs
   HypreParMatrix * Pc = nullptr;
   // Restriction operator to the non-contact dofs
   HypreParMatrix * Pnc = nullptr;
   Array<HYPRE_BigInt> constraints_starts;
   Array<HYPRE_BigInt> dof_starts;


   // with additional constraints
   //         [ g ]
   // g_new = [ eps + (d - dl) ]
   //         [ eps - (d - dl) ]
   // there are additional components to the Jacobian
   //         [ J ] 
   // J_new = [ I ]
   //         [-I ]      
   HypreParMatrix * Iu = nullptr;
   HypreParMatrix * negIu = nullptr;

   HypreParMatrix * dcdu = nullptr;

   Vector dl;
   Vector eps;
   Array<int> block_offsetsg;
   bool bound_constraints;
   int tribol_nranks;
   double tribol_ratio;
public:
   OptContactProblem(ElasticityOperator * problem_, 
                     const std::set<int> & mortar_attrs_, 
                     const std::set<int> & nonmortar_attrs_,
                     ParGridFunction * coords_, bool doublepass_,
                     const Vector & xref_, 
                     const Vector & xrefbc_, 
                     double tribol_ratio_,
                     int tribol_nranks_,
                     bool qp_ = true,
		     bool bound_constraints_=true);
   int GetDimU() {return dimU;}
   int GetDimM() {return dimM;}
   int GetDimC() {return dimC;}
   Vector & Getml() {return ml;}
   MPI_Comm GetComm() {return comm ;}
   HYPRE_BigInt * GetConstraintsStarts() {return constraints_starts.GetData();} 
   HYPRE_BigInt GetGlobalNumConstraints() {
	   if (bound_constraints)
	   {	   
	      return J->GetGlobalNumRows() + 2 * J->GetGlobalNumCols();
	   }
	   else
	   {
	      return J->GetGlobalNumRows();
	   }
   }

   HYPRE_BigInt * GetDofStarts() {return dof_starts.GetData();}
   HYPRE_BigInt GetGlobalNumDofs() {return J->GetGlobalNumCols(); }
   ElasticityOperator * GetElasticityOperator() {return problem;}

   HypreParMatrix * Duuf(const BlockVector &);
   HypreParMatrix * Dumf(const BlockVector &);
   HypreParMatrix * Dmuf(const BlockVector &);
   HypreParMatrix * Dmmf(const BlockVector &);
   HypreParMatrix * Duc(const BlockVector &);
   HypreParMatrix * Dmc(const BlockVector &);
   HypreParMatrix * lDuuc(const BlockVector &, const Vector &);

   HypreParMatrix * GetRestrictionToInteriorDofs();
   HypreParMatrix * GetRestrictionToContactDofs(); 

   void c(const BlockVector &, Vector &);
   void g(const Vector &, Vector &);
   double CalcObjective(const BlockVector &, int &);
   void CalcObjectiveGrad(const BlockVector &, BlockVector &);

   //double E(const Vector & d);
   double E(const Vector & d, int & eval_err);
   void DdE(const Vector & d, Vector & gradE);
   HypreParMatrix * DddE(const Vector & d);
   
   void SetBoundConstraints(const Vector & dl_, const Vector & eps_);
   
   ~OptContactProblem();
};

HypreParMatrix *  SetupTribol(ParMesh * pmesh, ParGridFunction * coords,
                              const Array<int> & ess_tdofs,
                              const std::set<int> & mortar_attrs, 
                              const std::set<int> & non_mortar_attrs, 
                              Vector &gap,  double tribol_ratio, int tribol_nranks);

