#include "mfem.hpp"


using namespace std;
using namespace mfem;

#include "axom/slic.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
// #include "tribol/mesh/CouplingScheme.hpp"


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
   Vector xrefbc;
   Vector eps;
   real_t eps_min = 1.e-4;
   int bound_constraint_step = 3;
   void Init();
   void SetEssentialBC();
   void SetUpOperator();

public:
   ElasticityOperator(ParMesh * pmesh_, Array<int> & ess_bdr_attr_,
                      Array<int> & ess_bdr_attr_comp_,
                      const Vector & E, const Vector & nu, bool nonlinear_ = false);
   void SetParameters(const Vector & E, const Vector & nu);
   void SetNeumanPressureData(ConstantCoefficient &f, Array<int> & bdr_marker);
   void SetDisplacementDirichletData(const Vector & delta, Array<int> essbdr);
   void SetTimeStepDisplacement(int i, const Vector & dx);
   void ResetDisplacementDirichletData();
   void UpdateEssentialBC(Array<int> & ess_bdr_attr_,
                          Array<int> & ess_bdr_attr_comp_);
   void FormLinearSystem();
   void UpdateLinearSystem();
   void UpdateRHS();

   ParMesh * GetMesh() const { return pmesh; };
   MPI_Comm GetComm() const { return comm; };

   ParFiniteElementSpace * GetFESpace() const { return fes; };
   const FiniteElementCollection * GetFECol() const { return fec; };
   int GetNumDofs() const { return ndofs; };
   int GetNumTDofs() const { return ntdofs; };
   int GetGlobalNumDofs() const { return gndofs; };
   const HypreParMatrix * GetOperator() const { return K; };
   const Vector & GetRHS() const { return B; };

   const ParGridFunction & GetDisplacementGridFunction() const { return x; };
   const Array<int> & GetEssentialDofs() const { return ess_tdof_list; };
   void Getxrefbc(Vector & xrefbc_) const;
   void Geteps(Vector & eps_) const;
   int GetBoundConstraintStep() const { return bound_constraint_step; };
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
   ParMesh * pmesh = nullptr;
   int dim, dimU, dimM, dimC, dimG;
   int num_constraints;
   real_t energy_ref;
   Vector ml, grad_ref, xref, xrefbc, gapv;
   std::set<int> mortar_attrs;
   std::set<int> nonmortar_attrs;

   HypreParMatrix * NegId = nullptr;
   HypreParMatrix * Kref=nullptr;
   // Jacobian of gap
   HypreParMatrix * J = nullptr;
   // Transpose of the Jacobian of gap
   HypreParMatrix * Jt = nullptr;
   // Transfer operator from the contact space to the displacement space
   HypreParMatrix * Pc = nullptr;
   ParGridFunction * coords = nullptr;

   void ReleaseMemory();
   void ComputeGapJacobian();

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

   HypreParMatrix * Mv = nullptr; // mass matrix in the volume
   HypreParMatrix * Mcs = nullptr; // mass matrix on the contact surface
   Vector Mvlump;
   Vector Mcslumpfull;
   Vector Mcslump;


   Vector dl;
   Vector eps;
   real_t eps_min = 1.e-4; // > 0
   int bound_constraint_step = 3;
   Array<int> block_offsetsg;
   bool bound_constraints;
   bool enable_bound_constraints = false;
   real_t tribol_ratio;
public:
   OptContactProblem(ElasticityOperator * problem_,
                     const std::set<int> & mortar_attrs_,
                     const std::set<int> & nonmortar_attrs_,
                     real_t tribol_ratio_,
                     bool bound_constraints_=true);
   void FormContactSystem(ParGridFunction * coords_, const Vector & xref);
   void UpdateContactSystem(ParGridFunction * coords_, const Vector & xref);
   int GetDimU() {return dimU;}
   int GetDimM() {return dimM;}
   int GetDimC() {return dimC;}
   Vector & Getml() {return ml;}
   MPI_Comm GetComm() {return comm ;}
   HYPRE_BigInt * GetConstraintsStarts() {return constraints_starts.GetData();}
   HYPRE_BigInt GetGlobalNumConstraints() { return num_constraints; }

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

   HypreParMatrix * GetContactSubspaceTransferOperator();

   void c(const BlockVector &, Vector &);
   void g(const Vector &, Vector &);
   real_t CalcObjective(const BlockVector &, int &);
   void CalcObjectiveGrad(const BlockVector &, BlockVector &);

   //real_t E(const Vector & d);
   real_t E(const Vector & d, int & eval_err);
   void DdE(const Vector & d, Vector & gradE);
   HypreParMatrix * DddE(const Vector & d);

   void SetTimeStepDisplacement(int i, const Vector & dx);
   void SetBoundConstraints(const Vector & dl_, const Vector & eps_);
   void SetBoundConstraints(int i);
   HypreParMatrix *  SetupTribol(ParMesh * pmesh, ParGridFunction * coords,
                                 const Array<int> & ess_tdofs,
                                 const std::set<int> & mortar_attrs,
                                 const std::set<int> & non_mortar_attrs,
                                 Vector &gap,  real_t tribol_ratio);

   void GetLumpedMassWeights(Vector & Mcslump_, Vector & Mvlump_)
   {
      Mcslump_.SetSize(Mcslump.Size()); Mcslump_ = 0.0;
      Mcslump_.Set(1.0, Mcslump);
      Mvlump_.SetSize(Mvlump.Size()); Mvlump_ = 0.0;
      Mvlump_.Set(1.0, Mvlump);
   };
   ~OptContactProblem();
};


