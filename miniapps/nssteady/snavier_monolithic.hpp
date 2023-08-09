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

#ifndef MFEM_SNAVIER_MONOLITHIC_HPP
#define MFEM_SNAVIER_MONOLITHIC_HPP

#define MFEM_SNAVIER_MONOLITHIC_VERSION 0.1

#include "mfem.hpp"
#include "custom_bilinteg.hpp"

// Include for mkdir
#include <sys/stat.h>

namespace mfem
{

/// Typedefs

// Vector and Scalar functions (time independent)
using VecFunc = void(const Vector &x, Vector &v);
using ScalarFunc = double(const Vector &x);

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
   VecCoeffContainer(Array<int> attr, VectorCoefficient *coeff_)
      : attr(attr)
   {
      this->coeff = coeff_;
   }

   VecCoeffContainer(VecCoeffContainer &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~VecCoeffContainer()
   {
        delete coeff;
        coeff=nullptr;
    }

   Array<int> attr;
   VectorCoefficient *coeff = nullptr;
};

/// Container for coefficient holding coeff, mesh attribute id (i.e. not the full array)
class CoeffContainer
{
public:
   CoeffContainer(Array<int> attr, Coefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   CoeffContainer(CoeffContainer &&obj)
   {
      // Deep copy the attribute and direction
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~CoeffContainer()
   {
        delete coeff;
        coeff=nullptr;
    }

   Array<int> attr;
   Coefficient *coeff;
};

/// Container for coefficient holding coeff, mesh attribute id (i.e. not the full array) and direction (x,y,z) (useful for componentwise BCs).
class CompCoeffContainer : public CoeffContainer
{
public:
    // Constructor for CompCoeffContainer
    CompCoeffContainer(Array<int> attr, Coefficient *coeff, int dir)
        : CoeffContainer(attr, coeff), dir(dir)
    {}

    // Move Constructor
    CompCoeffContainer(CompCoeffContainer &&obj)
        : CoeffContainer(std::move(obj))
    {
        dir = obj.dir;
    }

    // Destructor
    ~CompCoeffContainer() {}

    int dir;
};



/**
 * \class SNavierMonolithicSolver
 * \brief Steady-state Incompressible Navier Stokes solver with (Picard) Chorin-Temam algebraic splitting formulation.
 *
 * Navier-Stokes problem corresponding,
 *               to the saddle point system:
 *
 *              -nu \nabla^2 u + u \cdot \nabla u + \nabla p = f
 *                                                  \div u   = 0
 *
 * This implementation of a steady-state incompressible Navier Stokes solver uses
 * Picard iteration to linearize the nonlinear convective term.
 * The formulation introduces a user-defined parameter to adapt the scheme.
 *
 * 
 * The algebraic form of the linearize system is:
 *
 *                [  K + alpha*C   G ] [v] = [ fv + (alpha-1)*C*vk]                 
 *                [      -B        0 ] [p]   [  0 ] 
 * 
 * 
 * The numerical solver setup for each step are as follows.
 *
 * The BlockOperator is solved using GMRES.
 *
 * The solver uses a diagonal BlockPreconditioner with HypreBoomerAMG approximation of the convective-diffusive term,
 * and HypreBoomerAMG approximation of the pressure mass matrix to precondition the Schur Complement.
 *
 *
 * A detailed description is available in [1-3] 
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

class SNavierMonolithicSolver{
public:

    SNavierMonolithicSolver(ParMesh* mesh,int vorder=2, int porder=1, double kin_vis_=0, bool verbose=false);

    ~SNavierMonolithicSolver();


    /// Boundary conditions/Forcing terms

    /**
    * \brief Add Dirichlet velocity BC using VectorCoefficient and list of essential mesh attributes. 
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VectorCoefficient
    * and list of essential mesh attributes (they will be applied at setup time). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);

    /**
    * \brief Add Dirichlet velocity BC using Vector function and list of essential mesh attributes. 
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing Vector function
    * and list of essential mesh attributes (they will be applied at setup time). 
    *
    * \param func Pointer to VecFunc
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddVelDirichletBC(VecFunc *func, Array<int> &attr);

    /**
    * \brief Add Dirichlet velocity BC componentwise using Coefficient and list of active mesh boundaries.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing
    * Coefficient, list of essential mesh attributes, and constrained component (they will be applied at setup time). 
    *
    * \param coeff Pointer to Coefficient 
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    * \param dir Component of bc constrained (0=x, 1=y, 2=z)
    *
    */
    void AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir);

    /**
    * \brief Add Dirichlet velocity BC using VectorCoefficient and specific mesh attribute. 
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VectorCoefficient,
    * and integer for specific mesh attribute (they will be applied at setup time). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Boundary attribute
    *
    */
    void AddVelDirichletBC(VectorCoefficient *coeff, int &attr);

    /**
    * \brief Add Dirichlet velocity BC passing VecFunc and specific mesh attribute. 
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VecFunc
    * and integer for specific mesh attribute (they will be applied at setup time). 
    *
    * \param func Pointer to VecFunc 
    * \param attr Boundary attribute
    *
    */
    void AddVelDirichletBC(VecFunc *func, int &attr);

    /**
    * \brief Add Dirichlet velocity BC componentwise passing coefficient and specific mesh attribute.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs, passing 
    * Coefficient, specific mesh attribute, and constrained component (they will be applied at setup time). 
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
    * \brief Add Traction (Neumann) BC using VectorCoefficient and list of essential boundaries.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using VectorCoefficient, and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddTractionBC(VectorCoefficient *coeff, Array<int> &attr);

    /**
    * \brief Add Traction (Neumann) BC using VecFunc and list of essential boundaries.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using VecFunc and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddTractionBC(VecFunc *coeff, Array<int> &attr);

    /**
    * \brief Add Traction (Neumann) BC using VectorCoefficient and specific mesh attribute.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using VectorCoefficient, and specific mesh attribute (they will be applied at setup time by adding BoundaryIntegrators to the rhs). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Boundary attribute
    *
    */
    void AddTractionBC(VectorCoefficient *coeff, int &attr);

        /**
    * \brief Add Traction (Neumann) BC using VecFunc and specific mesh attribute.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using VecFunc and specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs). 
    *
    * \param func Pointer to VecFunc 
    * \param attr Boundary attribute
    *
    */
    void AddTractionBC(VecFunc *func, int &attr);

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

    /**
    * \brief Add forcing term to the rhs passing VecFunc.
    *
    * Add a forcing term (acceleration) to internal list of acceleration terms, passing
    * VecFunc and list of domain attributes (they will be applied at setup time by adding DomainIntegrators to the rhs). 
    *
    * \param coeff Pointer to VectorCoefficient 
    * \param attr Domain attributes
    *
    */
    void AddAccelTerm(VecFunc *func, Array<int> &attr);

    /// Solver setup and Solution

    /**
    * \brief Set the Fixed Point Solver parameters
    *
    * Set parameters ( @a rtol, @a atol, @a maxiter, @a print level), for the outer loop of the segregated scheme.
    * 
    * \param params struct containing parameters for outer loop and GMRES solver
    * \param maxPicardIterations_ control after how many iterations solver switches to Newton linearization
    *                           Possible values: 
    *                           -  -1: Picard solver.
    *                           -   0: Newton solver.
    *                           - n>0: Picard-Newton solver (switch to newton after n iterations) 
    * 
    */
    void SetOuterSolver(SolverParams params, int maxPicardIterations_ = -1 );

    /**
    * \brief Set parameter alpha. 
    *
    * Set segregation parameter @a alpha
    * 
    * \param alpha_ value for the parameter alpha (initial value if type_=ADAPTIVE) [0,1]
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
    * \brief Set gamma parameter for relaxation step. 
    *
    * \param gamma_ parameter for relaxation parameter [0,1]
    * 
    * 
    * \note gamma must satisfy the bound $\gamma < 2 \alpha$
    */
    void SetGamma(double &gamma_);

    /**
    * \brief Set lift for pressure solution, in case of analytic functions. 
    *
    * \param lift_ lift for pressure solution
    */
    void SetLift(double &lift_);

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
    Vector& GetVSol(){return x->GetBlock(0);}

    /**
    * \brief Returns the pressure solution vector.
    */
    Vector& GetPSol(){return x->GetBlock(0);}

    /**
    * \brief Returns the velocity solution (GridFunction).
    */
    ParGridFunction& GetVelocity()
    {
        v_gf->SetFromTrueDofs(x->GetBlock(0));
        return *v_gf;
    }

    /**
    * \brief Returns the pressure solution (GridFunction).
    */
    ParGridFunction& GetPressure()
    {
        p_gf->SetFromTrueDofs(x->GetBlock(1));
        return *p_gf;
    }

private:
    /// mesh
    ParMesh* pmesh = nullptr;
    int dim;

    /// Velocity and Pressure FE spaces
    ParFiniteElementSpace*   vfes = nullptr;
    ParFiniteElementSpace*   pfes = nullptr;
    FiniteElementCollection* vfec = nullptr;
    FiniteElementCollection* pfec = nullptr;
    int vorder;
    int porder;
    int vdim;
    int pdim;
    Array<int> block_offsets; // number of variables + 1

    /// Grid functions
    ParGridFunction *v_gf=nullptr;           // velocity
    ParGridFunction *p_gf=nullptr;           // pressure
    ParGridFunction *vk_gf=nullptr;          // velocity from previous iteration
    ParGridFunction *pk_gf=nullptr;          // pressure from previous iteration
    ParGridFunction *p_gf_out=nullptr;       // pressure output (including lift)

    /// (Vector) GridFunction coefficients wrapping vk_gf and pk_gf (for error computation)
    VectorGridFunctionCoefficient *vk_vc = nullptr;
    GridFunctionCoefficient       *pk_c  = nullptr;

    /// Dirichlet conditions 
    Array<int> vel_ess_attr;          // Essential mesh attributes (full velocity applied).
    Array<int> vel_ess_attr_x;        // Essential mesh attributes (x component applied).
    Array<int> vel_ess_attr_y;        // Essential mesh attributes (y component applied).
    Array<int> vel_ess_attr_z;        // Essential mesh attributes (z component applied).
    Array<int> ess_attr_tmp;          // Temporary variable for essential mesh attributes.
    Array<int> trac_attr_tmp;         // Temporary variable for traction mesh attributes.

    Array<int> vel_ess_tdof;          // All essential velocity true dofs.
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
    ParBilinearForm      *K_form  = nullptr;
    ParMixedBilinearForm *B_form  = nullptr;
    ParBilinearForm      *C_form  = nullptr; 
    ParBilinearForm      *C2_form = nullptr; 
    ParBilinearForm      *Mp_form = nullptr;
    ParLinearForm         *f_form = nullptr;

    /// Vectors
    BlockVector*   x = nullptr;
    BlockVector* x_k = nullptr;
    BlockVector* rhs = nullptr;
    Vector       *fv = nullptr;    // load vector for velocity (modified with ess bcs)

    /// Matrices/operators
    HypreParMatrix      *K = nullptr;         // diffusion term
    HypreParMatrix      *B = nullptr;         // divergence
    HypreParMatrix      *A = nullptr;         // A = K + alpha C
    HypreParMatrix      *C = nullptr;         // convective term   w . grad u
    HypreParMatrix     *C2 = nullptr;         // convective term   u . grad w
    HypreParMatrix     *Mp = nullptr;         // pressure mass matrix
    HypreParMatrix      *G = nullptr;
    HypreParMatrix     *Be = nullptr;
    HypreParMatrix     *Ge = nullptr;
    HypreParMatrix     *Ae = nullptr;

    /// Linear form to compute the mass matrix to set pressure mean to zero.
    ParLinearForm *mass_lf = nullptr;
    ConstantCoefficient *onecoeff = nullptr;
    double volume = 0.0;

    /// Kinematic viscosity.
    ConstantCoefficient kin_vis;

    /// Load vector coefficient
    VectorFunctionCoefficient *fcoeff = nullptr;

    /// Traction coefficient for Neumann
    VectorFunctionCoefficient *traction = nullptr;

    /// Coefficient for relaxation step
    double gamma;

    /// Coefficient for steady NS segregation
    double alpha;
    double alpha0;
    AlphaType alphaType;

    /// Coefficient controlling linearization type (Picard/Newton)
    bool newton = false;
    bool switched = false;
    int maxPicardIterations;

    /// Coefficient for convective term at rhs
    double rhs_coeff = 0.0;

    /// Newton/Fixed point solver parameters
    SolverParams sParams;
    int iter;

    /// Linear solvers parameters
    SolverParams s1Params;
    SolverParams s2Params;
    SolverParams s3Params;

    /// Solvers and Preconditioners
    GMRESSolver    *solver = nullptr;               // solver for Navier Stokes system
    BlockOperator    *nsOp = nullptr;               // Navier-Stokes block operator
    BlockDiagonalPreconditioner* nsPrec = nullptr;  // diagonal block preconditioner

    HypreBoomerAMG    *invA = nullptr;      // preconditioner for velocity block
    HypreBoomerAMG   *invMp = nullptr;      // approximation for pressure mass matrix   
    OrthoSolver *invMpOrtho = nullptr;      // preconditioner for pressure correction (remove nullspace)

    /// Error variables
    const IntegrationRule *irs[Geometry::NumGeom];
    int order_quad;
    double err_v;
    double norm_v;
    double err_p;
    double norm_p;
    
    // Lift
    double lift;

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
    const char* outfolder = nullptr;

    /// Solve the a single iteration of the problem
    void Step();

    /// Compute error for pressure and velocity
    void ComputeError();

    /// Update solution for next iteration
    void UpdateSolution();

    /// Remove mean from a Vector.
    /**
     * Modify the Vector @a v by subtracting its mean using
     * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
     */
    void Orthogonalize(Vector &v);

    /// Remove the mean from a ParGridFunction.
    /**
     * Modify the ParGridFunction @a v by subtracting its mean using
     * \f$ v = v - \int_\Omega \frac{v}{vol(\Omega)} dx \f$.
     */
    void MeanZero(ParGridFunction &v);

    // Update alpha parameter
    void UpdateAlpha();

    // Save results in Paraview or GLVis format
    void SaveResults( int iter );

    /// Print information about the Navier version.
    void PrintInfo();

    /// Output matrices in matlab format (for debug).
    void PrintMatricesVectors( const char* id, int num );
};

}

#endif

