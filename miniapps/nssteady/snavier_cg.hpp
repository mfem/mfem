// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SNAVIER_CG_HPP
#define MFEM_SNAVIER_CG_HPP

#define SNAVIER_CG_VERSION 0.1

#include "mfem.hpp"
#include "custom_bilinteg.hpp"

// Include for mkdir
#include <sys/stat.h>

namespace mfem
{

/// Typedefs

// Struct to pass slver parameters
struct SolverParams {   
    double rtol = 1e-6;
    double atol = 1e-10;
    int maxIter = 1000;
    int      pl = 0;

    SolverParams(double rtol_ = 1e-6, double atol_ = 1e-10, int maxIter_ = 1000, int pl_ = 0)
        : rtol(rtol_), atol(atol_), maxIter(maxIter_), pl(pl_) {}
};

// Type of segregation coefficient (if ADAPTIVE it will be adjusted to enforce Peclet < 2 )
enum AlphaType{CONSTANT, ADAPTIVE}; 

/// Container for vector coefficient holding coeff and mesh attribute (useful for BCs and forcing terms).
class VecCoeffContainer
{
public:
   VecCoeffContainer(Array<int> attr, VectorCoefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   VecCoeffContainer(VecCoeffContainer &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~VecCoeffContainer() { delete coeff; }

   Array<int> attr;
   VectorCoefficient *coeff;
};

/// Container for coefficient holding coeff, mesh attribute id (i.e. not the full array) and direction (x,y,z) (useful for componentwise BCs).
class CompCoeffContainer
{
public:
   CompCoeffContainer(Array<int> attr, Coefficient *coeff, int dir)
      : attr(attr), coeff(coeff), dir(dir)
   {}

   CompCoeffContainer(CompCoeffContainer &&obj)
   {
      // Deep copy the attribute and direction
      this->attr = obj.attr;
      this->dir = obj.dir;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~CompCoeffContainer() { delete coeff; }

   Array<int> attr;
   int dir;
   Coefficient *coeff;
};



/**
 * \class SNavierPicardCGSolver
 * \brief Steady-state Incompressible Navier Stokes solver with (Picard) Chorin-Temam algebraic splitting formulation.
 *
 * This implementation of a steady-state incompressible Navier Stokes solver uses
 * Picard iteration to linearize the nonlinear convective term.
 * The formulation introduces a user-defined parameter to adapt the scheme to use
 * Algebraic Chorin-Temam splitting scheme as in [1]
 *
 * The segregated schemes follows three steps:
 *
 * 1. Velocity prediction: Compute tentative velocity from decoupled momentum eqn without pressure term.
 *
 * 2. Pressure correction: Compute the pressure field using predicted velocity field.
 *
 * 3. Velocity correction: Project velocity onto divergence free space using the previously computed pressure field
 *
 * The numerical solver setup for each step are as follows.
 *
 * 1. is solved using CG with HypreBoomerAMG as preconditioner.
 *
 * 2. is solved using CG with HypreBoomerAMG as preconditioner (Note: pressure mass matrix can be used for preconditioning).
 *
 * 3. is solved using CG with HypreBoomerAMG as preconditioner.
 *
 *
 * A detailed description is available in [1] 
 *
 * [1] Viguerie, Alex, and Mengying Xiao. "Effective Chorin–Temam algebraic splitting schemes
       for the steady Navier–stokes equations." Numerical Methods for Partial Differential
       Equations 35.2 (2019): 805-829.
 
   [2] Viguerie, Alex, and Alessandro Veneziani. "Algebraic splitting methods for the steady
       incompressible Navier–Stokes equations at moderate Reynolds numbers." Computer Methods
       in Applied Mechanics and Engineering 330 (2018): 271-291.

   [3] Rebholz, Leo, Alex Viguerie, and Mengying Xiao. "Efficient nonlinear iteration schemes
       based on algebraic splitting for the incompressible Navier-Stokes equations." Mathematics
       of Computation 88.318 (2019): 1533-1557.
 */

class SNavierPicardCGSolver{
public:

    using DiagonalPolicy = Operator::DiagonalPolicy;

    SNavierPicardCGSolver(ParMesh* mesh_,int vorder_=2, int porder_=1, double kin_vis_=0, bool verbose_=false);

    ~SNavierPicardCGSolver();


    /// Boundary conditions/Forcing terms

    /**
    * \brief Add Dirichlet velocity BC. 
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs (they will be applied at setup time). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);

    /**
    * \brief Add Dirichlet velocity BC. 
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs (they will be applied at setup time). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Boundary attribute
    *
    */
    void AddVelDirichletBC(VectorCoefficient *coeff, int &attr);

    /**
    * \brief Add Dirichlet velocity BC componentwise to multiple boundaries.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs (they will be applied at setup time). 
    *
    * \param coeff Pointer to Coefficient 
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    * \param dir Component of bc constrained (0=x, 1=y, 2=z)
    *
    * \note dir=2 only if mesh is three dimensional.
    *
    */
    void AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir);

    /**
    * \brief Add Dirichlet velocity BC componentwise to a single boundary.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs (they will be applied at setup time). 
    *
    * \param coeff Pointer to Coefficient 
    * \param attr Boundary attribute 
    * \param dir Component of bc constrained (0=x, 1=y, 2=z)
    *
    * \note dir=2 only if mesh is three dimensional.
    *
    */
    void AddVelDirichletBC(Coefficient *coeff, int &attr, int &dir);

    /**
    * \brief Add Traction (Neumann) BC.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs (they will be applied at setup time by adding BoundaryIntegrators to the rhs). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Boundary attributes
    *
    */
    void AddTractionBC(VectorCoefficient *coeff, Array<int> &attr);

    /**
    * \brief Add forcing term to the rhs.
    *
    * Add a forcing term (acceleration) to internal list of acceleration terms (they will be applied at setup time by adding DomainIntegrators to the rhs). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Domain attributes
    *
    */
    void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);



    /// Solver setup and Solution

    /**
    * \brief Set the Fixed Point Solver parameters
    *
    * Set parameters ( @a rtol, @a atol, @a maxiter, @a print level), for the outer loop of the segregated scheme.
    * 
    */
    void SetFixedPointSolver(SolverParams params);

    /**
    * \brief Set parameter alpha. 
    *
    * Set segregation parameter @a alpha
    * 
    * \param alpha_ value for the parameter alpha (initial value if type_=ADAPTIVE)
    * \param type_ type of parameter (AlphaType::CONSTANT, AlphaType::ADAPTIVE)
    *
    * * \note If AlphaType::ADAPTIVE, alpha will be computed to enforce Peclet Number Pe < 2
    * 
    * Given the Peclet number
    *           \f$ \mathbb{P}_k = \frac{ \|v_k\|_K }{2 \nu h_K} \f$
    * we can thus set @a alpha to
    *           \f$ \alpha_k = 0.75 min_{K \in \Tau} \frac{ 4 \nu  }{ \| v_k \|_K h_K} \f$
    * 
    * ADAPTIVE alpha NYI!
    */
    void SetAlpha(double &alpha_, const AlphaType &type_);

    /**
    * \brief Set the Linear Solvers parameters
    *
    * Set parameters ( @a rtol, @a atol, @a maxiter, @a print level) for the three internal linear solvers involved in the 
    * segregated scheme. 
    *
    */
    void SetLinearSolvers(SolverParams params1, SolverParams params2, SolverParams params3);

    /**
    * \brief Finalizes setup.
    *
    * Finalizes setup of steady NS solver: initialize forms, linear solvers, and preconditioners.
    *
    * \note This method should be called only after:
    * - Setting the boundary conditions and forcing terms (AddVelDirichletBC/AddTractionBC/AddAccelTerm).
    * - Setting the FixedPoint and Linear solvers.
    */
    void Setup();

    /**
    * \brief Setup output.
    *
    * Setup output (V
    * ISit, Paraview oe both) and enable output during execution.
    *
    * \param folderPath output folder
    * \param filename filename
    * \param visit_ enable VISit output
    * \param paraview_ enable Paraview output
    */
    void SetupOutput( const char* folderPath= "./", bool visit=false, bool paraview=false, DataCollection::Format par_format=DataCollection::SERIAL_FORMAT );

    /**
    * \brief Set the initial condition for Velocity.
    *
    */
    void SetInitialConditionVel(VectorCoefficient &v_in);

    /**
    * \brief Set the initial condition for Pressure.
    *
    */
    void SetInitialConditionPres(Coefficient &p_in);

    /**
    * \brief Solve forward problem.
    *
    * Solves the forward problem until convergence of the steady NS solver is reached.
    *
    */
    void FSolve();



    /// Getter methods

    /**
    * \brief Returns pointer to the velocity FE space.
    */
    ParFiniteElementSpace* GetVFes(){return vfes;}

    /**
    * \brief Returns pointer to the pressure FE space.
    */
    ParFiniteElementSpace* GetPFes(){return pfes;}

    /**
    * \brief Returns the velocity solution vector.
    */
    Vector& GetVSol(){return *v;}

    /**
    * \brief Returns the pressure solution vector.
    */
    Vector& GetPSol(){return *p;}

    /**
    * \brief Returns the intermedite velocity vector.
    */
    Vector& GetProvisionalVSol(){return *z;}

    /**
    * \brief Returns the velocity solution (GridFunction).
    */
    ParGridFunction& GetVelocity()
    {
        v_gf.SetFromTrueDofs(*v);
        return v_gf;
    }

    /**
    * \brief Returns the pressure solution (GridFunction).
    */
    ParGridFunction& GetPressure()
    {
        p_gf.SetFromTrueDofs(*p);
        return p_gf;
    }

    /**
    * \brief Returns the intermediate velocity (GridFunction).
    */
    ParGridFunction& GetProvisionalVelocity()
    {
        z_gf.SetFromTrueDofs(*z);
        return z_gf;
    }

private:
    /// mesh
    ParMesh* pmesh;
    int dim;

    /// Velocity and Pressure FE spaces
    ParFiniteElementSpace* vfes;
    ParFiniteElementSpace* pfes;
    FiniteElementCollection* vfec;
    FiniteElementCollection* pfec;
    int vorder;
    int porder;
    int vdim;
    int pdim;

    /// Grid functions
    ParGridFunction v_gf;           // velocity
    ParGridFunction p_gf;           // pressure
    ParGridFunction z_gf;           // intermediate velocity    
    ParGridFunction vk_gf;          // velocity from previous iteration
    ParGridFunction pk_gf;          // pressure from previous iteration

    /// (Vector) GridFunction coefficients wrapping vk_gf and pk_gf (for error computation)
    VectorGridFunctionCoefficient *vk_vc = nullptr;
    GridFunctionCoefficient       *pk_c  = nullptr;

    /// Dirichlet conditions 
    Array<int> vel_ess_attr;          // Essential attributes (full velocity applied).
    Array<int> vel_ess_attr_x;        // Essential attributes (x component applied).
    Array<int> vel_ess_attr_y;        // Essential attributes (y component applied).
    Array<int> vel_ess_attr_z;        // Essential attributes (z component applied).
    Array<int> ess_attr_tmp;          // Temporary variable for essential attributes.

    Array<int> vel_ess_tdof;          // All essential true dofs.
    Array<int> vel_ess_tdof_full;     // All essential true dofs from VectorCoefficient.
    Array<int> vel_ess_tdof_x;        // All essential true dofs x component.
    Array<int> vel_ess_tdof_y;        // All essential true dofs y component.
    Array<int> vel_ess_tdof_z;        // All essential true dofs z component.

    // Bookkeeping for velocity dirichlet bcs (full Vector coefficient).
    std::vector<VecCoeffContainer> vel_dbcs;

    // Bookkeeping for velocity dirichlet bcs (componentwise).
    std::string dir_string;    // string for direction name for printing output
    std::vector<CompCoeffContainer> vel_dbcs_xyz;

    // Bookkeeping for traction (neumann) bcs.
    std::vector<VecCoeffContainer> traction_bcs;

    // Bookkeeping for acceleration (forcing) terms.
    std::vector<VecCoeffContainer> accel_terms;

    /// Bilinear/linear forms 
    ParBilinearForm      *K_form;
    ParMixedBilinearForm *B_form;
    ParBilinearForm      *C_form; // NOTE: or BilinearForm if VectorConvectionIntegrator works
    ParLinearForm        *f_form;

    /// Vectors
    Vector *v    = nullptr;    // corrected velocity vector
    Vector *p    = nullptr;    // pressure solution vector
    Vector *z    = nullptr;    // predicted velocity vector
    Vector *vk   = nullptr;    // corrected velocity at previous iteration
    Vector *pk   = nullptr;    // pressure at previous iteration
    Vector *fv   = nullptr;    // load vector for velocity (modified with ess bcs)
    Vector *fp   = nullptr;    // load vector for pressure (modified with ess bcs)
    Vector *rhs1 = nullptr;    // rhs for first solve  
    Vector *rhs2 = nullptr;    // rhs for second solve
    Vector *rhs3 = nullptr;    // rhs for third solve
    Vector *tmp = nullptr;     // tmp var to assemble rhs

    /// Matrices/operators
    HypreParMatrix     *K = nullptr;         // diffusion term
    HypreParMatrix     *B = nullptr;         // divergence
    HypreParMatrix     *C = nullptr;         // (linearized) convective term
    HypreParMatrix     *A = nullptr;         // A = K + alpha C
    HypreParMatrix     *S = nullptr;         // S = B Kdiag-1 Bt
    HypreParMatrix    *Bt = nullptr;
    HypreParMatrix    *Ke = nullptr;         // Matrices after bc elimination
    HypreParMatrix    *Be = nullptr;
    HypreParMatrix   *Bte = nullptr;
    HypreParMatrix    *Ce = nullptr;

    /// Kinematic viscosity.
    ConstantCoefficient kin_vis;

    /// Load vector coefficient
    VectorFunctionCoefficient *fcoeff = nullptr;

    /// Traction coefficient for Neumann
    VectorFunctionCoefficient *traction = nullptr;

    /// Coefficient for steady NS segregation
    double alpha;
    double alpha0;
    AlphaType alphaType;

    /// Newton/Fixed point solver parameters
    SolverParams sParams;
    int iter;

    /// Linear solvers parameters
    SolverParams s1Params;
    SolverParams s2Params;
    SolverParams s3Params;

    /// Solvers and Preconditioners
    CGSolver *invA= nullptr;     // solver for velocity prediction
    CGSolver *invS= nullptr;     // solver for pressure correction
    CGSolver *invK= nullptr;     // solver for velocity correction

    HypreBoomerAMG *invA_pc= nullptr;  //preconditioner for velocity prediction
    HypreBoomerAMG *invS_pc= nullptr;  //preconditioner for pressure correction
    HypreBoomerAMG *invK_pc= nullptr;  //preconditioner for velocity correction

    /// Error variables
    const IntegrationRule *irs[Geometry::NumGeom];
    int order_quad;
    double err_v;
    double norm_v;
    double err_p;
    double norm_p;
    
    /// Timers
    StopWatch timer;

    /// Enable/disable verbose output.
    bool verbose;

    /// Exit flag
    int flag; 

    /// Output
    bool    visit, paraview;
    ParaViewDataCollection* paraview_dc = nullptr;;
    VisItDataCollection*       visit_dc = nullptr;


    /// Solve the a single iteration of the problem
    void Step();

    /// Compute error for pressure and velocity
    void ComputeError();

    /// Update solution for next iteration
    void UpdateSolution();

    /**
    * \brief Modify rhs for essential bcs.
    *
    * Eliminates essential boundary conditions from rhs 
    *
    * \param ess_tdof_list List of essential degrees of freedom
    * \param mat_e elimiated matrix (obtained by EliminateRowsCols)
    * \param sol, rhs reference to solution vector and rhs to be modified
    * \param copy_sol if true (default) copies solution into rhs for ess_tdofs
    * 
    * \note Performs the following transformation to the rhs
    *       f(dofs)   = v(dofs)
    *       f(~dofs) -= mat_e*v(dofs)
    */
    void ModifyRHS(Array<int> &ess_tdof_list, HypreParMatrix* mat_e, Vector &sol, Vector &rhs, bool copy_sol=true);

    /**
    * \brief Matrix vector multiplication, using matrices obtained by ess dofs elimination.
    *
    * Multiply matrix and vector, given original matrix split into modified and eliminated matrices after 
    * ess dofs elimination. 
    *
    * \param ess_tdof_list List of essential degrees of freedom
    * \param mat   modified matrix (obtained by EliminateRowsCols)
    * \param mat_e eliminated matrix (obtained by EliminateRowsCols)
    * \param x     vector being multiplied
    * \param y     vector for storing the result
    * 
    * \note Since modified matrix has ones on the diagonal we need to remove offset at ess_tdofs
    *       M x = mat x + mat_e x - [vec(tdofs); 0]
    */
    void FullMult(Array<int> &ess_tdof_list, HypreParMatrix* mat, HypreParMatrix* mat_e,
                  Vector &x, Vector &y, Operator::DiagonalPolicy diag_policy);

    // Update alpha parameter
    void UpdateAlpha();

    /// Print information about the Navier version.
    void PrintInfo();
};

}

#endif

