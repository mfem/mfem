#include "mfem.hpp"
#include "axom/slic.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

using namespace std;
using namespace mfem;


/**
 *  @class ElasticityOperator
 *  @brief Parallel finite element operator for linear and nonlinear elasticity.
 *
 *  This class sets up and evaluates finite element operators for elasticity on a parallel mesh.
 *  It supports both linear and nonlinear (Neo-Hookean) formulations.
 *  Features include evaluation of energy, gradient, and Hessian for use in optimization solvers.
 */
class ElasticityOperator
{
private:
   MPI_Comm comm;

   /// Toggle for nonlinear formulation (true = Neo-Hookean, false = linear elasticity).
   bool nonlinear = false;

   /// Tracks whether the linear system has been formed.
   bool formsystem = false;

   ParMesh * pmesh = nullptr;

   /// Essential boundary attribute markers.
   Array<int> ess_bdr, ess_bdr_attr;

   /// Essential DOFs (true DOF list) and component-based attributes.
   Array<int> ess_tdof_list, ess_bdr_attr_comp;

   /// Polynomial order of the FE basis (default = 1).
   int order=1;

   /// Global number of true DOFs in the FE space.
   int globalntdofs;

   /// Finite element collection (H1).
   FiniteElementCollection * fec = nullptr;

   /// Parallel finite element space for displacement.
   ParFiniteElementSpace * fes = nullptr;

   /// Underlying operator: bilinear form (linear) or nonlinear form (nonlinear).
   Operator * op = nullptr;

   /// Linear form for RHS assembly.
   ParLinearForm * b = nullptr;

   /// Current solution
   ParGridFunction x;

   // System matrix
   HypreParMatrix *K=nullptr;

   /// Right-hand side (B) and solution vector (X).
   Vector B, X;

   /// Neumann pressure coefficient (for traction BCs).
   ConstantCoefficient pressure_cf;

   /// Material parameters:
   /// - Linear: c1 = λ (1ˢᵗ Lame parameter), c2 = μ (2ⁿᵈ Lame parameter or shear modulus)
   /// - Nonlinear: c1 = G (shear modulus), c2 = K (bulk modulus)
   Vector c1, c2;
   PWConstCoefficient c1_cf, c2_cf;

   /// Hyperelastic material model (only for nonlinear case).
   NeoHookeanModel * material_model = nullptr;

   /// Reference configuration displacement
   Vector xref;

   /// Internal setup functions
   void Init();
   void SetEssentialBC();
   void SetUpOperator();

public:
   /**  @brief Construct an ElasticityOperator.
    *   @param pmesh_ Parallel mesh.
    *   @param ess_bdr_attr_ Array of essential boundary attributes.
    *   @param ess_bdr_attr_comp_ Component index for each essential boundary attribute.
    *   @param E Vector of Young’s modulus values (per attribute).
    *   @param nu Vector of Poisson’s ratio values (per attribute).
    *   @param nonlinear_ If true, setup nonlinear hyperelasticity.
    */
   ElasticityOperator(ParMesh * pmesh_, Array<int> & ess_bdr_attr_,
                      Array<int> & ess_bdr_attr_comp_,
                      const Vector & E, const Vector & nu, bool nonlinear_ = false);

   /// Set material parameters from vectors of Young’s modulus (E) and Poisson’s ratio (ν).
   void SetParameters(const Vector & E, const Vector & nu);

   /// Apply Neumann (pressure) boundary condition on a set of boundary markers.
   void SetNeumanPressureData(ConstantCoefficient &f, Array<int> & bdr_marker);

   /// Apply Dirichlet (displacement) boundary condition on a set of boundary markers.
   void SetDisplacementDirichletData(const Vector & delta, Array<int> essbdr);

   /// Assemble and form the linear system (matrix and RHS).
   void FormLinearSystem();

   /// Reset and reassemble the RHS linear form.
   void UpdateRHS();

   ParMesh * GetMesh() const { return pmesh; };
   MPI_Comm GetComm() const { return comm; };

   ParFiniteElementSpace * GetFESpace() const { return fes; };

   int GetGlobalNumDofs() const { return globalntdofs; };

   const Array<int> & GetEssentialDofs() const { return ess_tdof_list; };

   /// Get the displacement with essential boundary conditions applied.
   void Getxrefbc(Vector & xrefbc) const {x.GetTrueDofs(xrefbc);}

   /// Compute the elastic energy functional at a given displacement vector.
   real_t GetEnergy(const Vector & u) const;

   /// Compute the gradient of the energy functional at a given displacement vector.
   void GetGradient(const Vector & u, Vector & gradE) const;

   /// Get the Hessian (stiffness matrix) at a given displacement vector.
   HypreParMatrix * GetHessian(const Vector & u);

   /// Check if the operator is nonlinear.
   bool IsNonlinear() { return nonlinear; }

   /// Destructor (cleans up FE space, operator, and material model).
   ~ElasticityOperator();
};


/** @class OptContactProblem
 *  @brief Contact optimization problem with mortar and non-mortar interfaces.
 *
 *  This class formulates and manages a parallel finite element contact problem
 *  built on top of an `ElasticityOperator`. It sets up the contact system using
 *  Tribol, handles mortar/non-mortar interface attributes, and provides
 *  operators for objective evaluation, gradients, Hessians, and constraints.
 *
 *  Features include:
 * - Construction of gap constraints and Jacobians through Tribol.
 * - Optional bound constraints.
 * - Objective and gradient evaluation for optimization solvers.
 */
class OptContactProblem
{
private:
   /// MPI communicator for the problem.
   MPI_Comm comm;

   /// Underlying elasticity problem.
   ElasticityOperator * problem = nullptr;

   /// Finite element space for displacements
   ParFiniteElementSpace * vfes = nullptr;

   /// Dimensions: displacement, multiplier, constraint spaces.
   int dimU, dimM, dimC, dimG;

   /// Global number of constraints.
   int num_constraints;

   /// Energy value at reference configuration.
   real_t energy_ref;

   /// Energy gradient at reference configuration.
   Vector  grad_ref;

   /// Reference configuration displacement vector.
   Vector xref;

   /// Reference configuration displacement vector with updated BCs.
   Vector xrefbc;

   /// Constraint multiplier vector.
   Vector ml;

   /// Gap vector on contact interface.
   Vector gapv;

   /// Mortar and non-mortar attribute sets.
   std::set<int> mortar_attrs;
   std::set<int> nonmortar_attrs;

   /// Negative identity matrix (used in constraints).
   HypreParMatrix * NegId = nullptr;

   /// Reference stiffness (Hessian) matrix.
   HypreParMatrix * Kref=nullptr;

   /// Jacobian of the gap function.
   HypreParMatrix * J = nullptr;

   /// Transpose of gap Jacobian.
   HypreParMatrix * Jt = nullptr;

   /// Transfer operator from contact space to displacement space.
   HypreParMatrix * Pc = nullptr;

   /// Coordinates of mesh nodes (grid function).
   ParGridFunction * coords = nullptr;

   /// Free allocated matrices/vectors.
   void ReleaseMemory();

   /// Compute gap and its Jacobian using Tribol.
   void ComputeGapJacobian();

   /// Constraint partition offsets (for distributed data).
   Array<HYPRE_BigInt> constraints_starts;

   /// DOF partition offsets (for distributed data).
   Array<HYPRE_BigInt> dof_starts;

   // with additional constraints
   //         [ g ]
   // g_new = [ eps + (d - dl) ]
   //         [ eps - (d - dl) ]
   // there are additional components to the Jacobian
   //         [ J ]
   // J_new = [ I ]
   //         [-I ]

   /// Identity matrices for bound constraints.
   HypreParMatrix * Iu = nullptr;
   HypreParMatrix * negIu = nullptr;

   /// Cached constraint Jacobian with bounds.
   HypreParMatrix * dcdu = nullptr;

   /// Mass matrix in the volume.
   HypreParMatrix * Mv = nullptr;

   /// Mass matrix on the contact surface.
   HypreParMatrix * Mcs = nullptr;

   /// Lumped volume mass vector.
   Vector Mvlump;

   /// Lumped contact surface mass (full).
   Vector Mcslumpfull;

   /// Lumped contact surface mass (reduced).
   Vector Mcslump;

   /// Bound displacement vector for constraints.
   Vector dl;

   /// Epsilon vector (slack/bounds).
   Vector eps;

   /// Minimum epsilon value (>0).
   real_t eps_min = 1.e-4;

   /// Offsets for block vector partitioning of constraints.
   Array<int> block_offsetsg;

   /// Proximity ratio for Tribol binning.
   real_t tribol_ratio;

   /// Flag: whether bound constraints are enabled.
   bool bound_constraints;

   /// Flag: whether bound constraints have been activated.
   bool bound_constraints_activated = false;

public:
   OptContactProblem(ElasticityOperator * problem_,
                     const std::set<int> & mortar_attrs_,
                     const std::set<int> & nonmortar_attrs_,
                     real_t tribol_ratio_,
                     bool bound_constraints_);

   /// Build contact system, assemble gap Jacobian and mass matrices.
   void FormContactSystem(ParGridFunction * coords_, const Vector & xref);

   /// Return displacement space dimension.
   int GetDimU() {return dimU;}

   /// Return multiplier space dimension.
   int GetDimM() {return dimM;}

   /// Return constraint space dimension.
   int GetDimC() {return dimC;}

   /// Access Lagrange multiplier vector.
   Vector & Getml() {return ml;}

   /// Get MPI communicator.
   MPI_Comm GetComm() {return comm ;}

   /// Get distributed constraint partition offsets.
   HYPRE_BigInt * GetConstraintsStarts() {return constraints_starts.GetData();}

   /// Return global number of constraints.
   HYPRE_BigInt GetGlobalNumConstraints() {return num_constraints;}

   /// Get distributed DOF partition offsets.
   HYPRE_BigInt * GetDofStarts() {return dof_starts.GetData();}

   /// Return global number of DOFs (from Jacobian).
   HYPRE_BigInt GetGlobalNumDofs() {return J->GetGlobalNumCols();}

   /// Return underlying elasticity operator.
   ElasticityOperator * GetElasticityOperator() {return problem;}

   /// Block derivative of energy wrt displacement (U-U block).
   HypreParMatrix * Duuf(const BlockVector &x) {return DddE(x.GetBlock(0));}

   /// Block derivative U-M.
   HypreParMatrix * Dumf(const BlockVector &) {return nullptr;}

   /// Block derivative M-U.
   HypreParMatrix * Dmuf(const BlockVector &) {return nullptr;}

   /// Block derivative M-M.
   HypreParMatrix * Dmmf(const BlockVector &) {return nullptr;}

   /// Block derivative U-C (constraints).
   HypreParMatrix * Duc(const BlockVector &);

   /// Block derivative M-C.
   HypreParMatrix * Dmc(const BlockVector &);

   /// Block derivative U-U with constraints.
   HypreParMatrix * lDuuc(const BlockVector &, const Vector &) {return nullptr;}

   /// Return transfer operator from contact to displacement subspace.
   HypreParMatrix * GetContactSubspaceTransferOperator();

   /// Evaluate gap function
   void g(const Vector &, Vector &);

   /// Evaluate contact constraints
   void c(const BlockVector &, Vector &);

   /// Compute objective functional value.
   real_t CalcObjective(const BlockVector &, int &);

   /// Compute gradient of objective functional.
   void CalcObjectiveGrad(const BlockVector &, BlockVector &);

   /// Evaluate elastic energy functional.
   real_t E(const Vector & d, int & eval_err);

   /// Evaluate gradient of energy functional.
   void DdE(const Vector & d, Vector & gradE);

   /// Return Hessian of energy functional.
   HypreParMatrix * DddE(const Vector & d);

   /// Update displacement and eps for bound constraints.
   void SetDisplacement(const Vector & dx, bool active_constraints);

   /// Activate bound constraints (if enabled).
   void ActivateBoundConstraints();

   /// Get Gap and its Jacobian from Tribol.
   HypreParMatrix *  SetupTribol(ParMesh * pmesh, ParGridFunction * coords,
                                 const Array<int> & ess_tdofs,
                                 const std::set<int> & mortar_attrs,
                                 const std::set<int> & non_mortar_attrs,
                                 Vector &gap,  real_t tribol_ratio);

   /// Get lumped mass weights for contact and volume spaces.
   void GetLumpedMassWeights(Vector & Mcslump_, Vector & Mvlump_)
   {
      Mcslump_.SetSize(Mcslump.Size()); Mcslump_ = 0.0;
      Mcslump_.Set(1.0, Mcslump);
      Mvlump_.SetSize(Mvlump.Size()); Mvlump_ = 0.0;
      Mvlump_.Set(1.0, Mvlump);
   };
   ~OptContactProblem() { ReleaseMemory(); }
};
