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

#ifndef MFEM_SNAVIER_DG_HPP
#define MFEM_SNAVIER_DG_HPP

#define SNAVIER_DG_VERSION 0.1

#include "mfem.hpp"
#include "custom_bilinteg_dg.hpp"

// Include for mkdir
#include <sys/stat.h>

namespace mfem
{
/// Typedefs

// Struct to pass slver parameters
// TODO: copy. I use Picard iteration along with Oseen equations. I may only need i) tol ii) maxIter
struct SolverParams {
    double rtol = 1e-6;
    double atol = 1e-10;
    int maxIter = 1000;
    int      pl = 0;

    SolverParams(double rtol_ = 1e-6, double atol_ = 1e-10, int maxIter_ = 1000, int pl_ = 0)
        : rtol(rtol_), atol(atol_), maxIter(maxIter_), pl(pl_) {}
};

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

   ~CompCoeffContainer()
   {
        delete coeff;
        coeff=nullptr;
    }

   Array<int> attr;
   int dir;
   Coefficient *coeff;
};

/**
 * \class SNavierPicardDGSolver
 * \brief Steady-state Incompressible Navier Stokes solver with Picard iteration, where in each
 * iteration we solve the Oseen equations.
 *
 * This implementation of a steady-state incompressible Navier Stokes solver uses
 * Picard iteration to linearize the nonlinear convective term.
 * As consequence, we have to solve the Oseen equaionts in each iteration.
 *
 * The Oseen equations are distcretized by a local discontinuous Galerkin (LDG) method.
 * In particular, there are several options of numerical flux [1][2][3]. In this code,
 * we simply adopt the simplest one that is analyzed in [2]. Note that additional procedure is
 * required to ensure that the divergence-free constraint is satisfied point-wisely
 * (see the TODO below).
 *
 * TODO: dealing with divergence-free constraint. There are three ways:
 *   1.Perform post-processing defined in [2]. However, a BDM type of space is needed and it
 *     is not implemented in mfem yet.
 *
 *   2.Use RT element in the approximation of velocity. In this case, we need to use discretization
 *     analyzed in [3]. However, we need "CalcGradVshape" to compute the convective term
 *     (see also "Add CalcGradVshape for Raviart-Thomas fes #2599"). Currently, it is only implemented
 *     for two-dimensional case and is not merged into master branch yet.
 *
 *   3.Perform global projection used in [4]. Currently, this method is implemented.
 *    Theoretically speaking, we can achieve local divergence-free. However, I did not obsereve
 *    such correction in the numerical result. May need to check again the formula and/or implementation.
 *
 * references:
 *
 * [1] Cockburn, Bernardo, Guido Kanschat, and Dominik Schötzau. "The local discontinuous
 *     Galerkin method for the Oseen equations." Mathematics of Computation 73.246
 *     (2004): 569-593.
 *
 * [2] Cockburn, Bernardo, Guido Kanschat, and Dominik Schötzau. "A locally conservative
 *     LDG method for the incompressible Navier-Stokes equations." Mathematics of computation
 *     74.251 (2005): 1067-1095.
 *
 * [3] Cockburn, Bernardo, Guido Kanschat, and Dominik Schötzau. "A note on discontinuous
 *     Galerkin divergence-free solutions of the Navier–Stokes equations." Journal of
 *     Scientific Computing 31 (2007): 61-73.
 *
 * [4] Botti, Lorenzo, and Daniele A. Di Pietro. "A pressure-correction scheme for
 *     convection-dominated incompressible flows with discontinuous velocity and
 *     continuous pressure." Journal of computational physics 230.3 (2011): 572-585.
 */

class SNavierPicardDGSolver{
public:

    SNavierPicardDGSolver(ParMesh* mesh, int sorder=1, int vorder=2, int porder=1, double kin_vis_=0, bool verbose=false);

    ~SNavierPicardDGSolver();

    //TODO: add comments later
    void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);
    void AddVelDirichletBC(VectorCoefficient *coeff, int &attr);
    void AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir);
    void AddVelDirichletBC(Coefficient *coeff, int &attr, int &dir);
    void AddTractionBC(VectorCoefficient *coeff, Array<int> &attr);
    void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);

    /// Solver setup and Solution

    /**
    * \brief Set the Fixed Point Solver parameters
    *
    * Set parameters ( @a rtol, @a atol, @a maxiter, @a print level), for the outer loop of the scheme.
    *
    */
    void SetFixedPointSolver(SolverParams params);

    /**
    * \brief Set the Linear Solvers parameters
    *
    * Set parameters ( @a rtol, @a atol, @a maxiter, @a print level) for the linear solver in solving
    * the LDG formulation of the Oseen equations.
    *
    */
    void SetLinearSolvers(SolverParams params1, SolverParams params2, SolverParams params3);

    /**
    * \brief Finalizes setup.
    *
    * Finalizes setup of steady NS solver: initialize forms and linear solvers.
    *
    * \note This method should be called only after:
    * - Setting the boundary conditions and forcing terms (AddVelDirichletBC/AddTractionBC/AddAccelTerm).
    * - Setting the FixedPoint and Linear solvers.
    */
    void Setup();

    /**
    * \brief Setup output.
    *
    * Setup output (VISit, Paraview oe both) and enable output during execution.
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
    * \brief Solve forward problem.
    *
    * Solves the sub-problems until the convergence of the Picard iteration of
    * the steady NS solver is reached.
    *
    */
    void FSolve();

    /// Getter methods

    /**
    * \brief Returns pointer to the gradient of velocity FE space.
    */
    ParFiniteElementSpace* GetSigFes(){return sigfes;}

    /**
    * \brief Returns pointer to the velocity FE space.
    */
    ParFiniteElementSpace* GetVFes(){return ufes;}

    /**
    * \brief Returns pointer to the pressure FE space.
    */
    ParFiniteElementSpace* GetPFes(){return pfes;}

    /**
    * \brief Returns the velocity solution vector.
    */
    Vector& GetVSol(){return *u;}

    /**
    * \brief Returns the pressure solution vector.
    */
    Vector& GetPSol(){return *p;}

    /**
    * \brief Returns the pressure solution (GridFunction).
    */
    ParGridFunction& GetPressure()
    {
        p_gf.SetFromTrueDofs(*p);
        return p_gf;
    }

private:
    /// mesh
    ParMesh* pmesh=nullptr;
    int dim;

    /// Velocity and Pressure FE spaces
    ParFiniteElementSpace* sigfes=nullptr;
    ParFiniteElementSpace* ufes=nullptr;
    ParFiniteElementSpace* pfes=nullptr;
    FiniteElementCollection* sigfec=nullptr;
    FiniteElementCollection* ufec=nullptr;
    FiniteElementCollection* pfec=nullptr;
    int sigorder;
    int uorder;
    int porder;
    int tdim;
    int vdim;

    /// Grid functions
    ParGridFunction u_gf;           // velocity
    ParGridFunction p_gf;           // pressure
    ParGridFunction uk_gf;          // velocity from previous iteration

    /// (Vector) GridFunction coefficients wrapping uk_gf
    VectorGridFunctionCoefficient *uk_vc = nullptr;

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
    ParBilinearForm      *ainv=nullptr;
    ParMixedBilinearForm *b=nullptr;
    ParBilinearForm      *c=nullptr;
    ParMixedBilinearForm *d=nullptr;

    ParLinearForm *f=nullptr;
    ParLinearForm *g=nullptr;
    ParLinearForm *h=nullptr

	/// Vectors
    ParGridFunction *u    = nullptr;    // velocity solution
    ParGridFunction *p    = nullptr;    // pressure solution vector
    ParGridFunction *uk   = nullptr;    // velocity at previous iteration
    BlockVector *x      = nullptr;
    BlockVector *truex  = nullptr;
	BlockVector *rhs    = nullptr;
	BlockVector *truerhs= nullptr;

	/// Matrices/operators
	HypreParMatrix *stokeMono = nullptr;
	HypreParMatrix *Ainv = nullptr;
	HypreParMatrix *B = nullptr;
	HypreParMatrix *Bt = nullptr;
	HypreParMatrix *C = nullptr;
	HypreParMatrix *D = nullptr;
	HypreParMatrix *nD = nullptr;
	HypreParMatrix *BtAinv = nullptr;
	HypreParMatrix *BtAinvB_C = nullptr;
	HypreParMatrix *zero= nullptr;

	HypreParVector *F=nullptr;
	HypreParVector *BtAinvF=nullptr;
	HypreParVector *G=nullptr;

    /// Kinematic viscosity.
    ConstantCoefficient kin_vis;

    /// Load vector coefficient
    VectorFunctionCoefficient *fcoeff = nullptr;

    /// Traction coefficient for Neumann
    VectorFunctionCoefficient *traction = nullptr;

    /// Fixed point solver parameters
    SolverParams sParams;
    int iter;

    /// Linear solvers parameters
    SolverParams s1Params;
    SolverParams s2Params;
    SolverParams s3Params;

    /// Solvers and Preconditioners

    /// Error variables
    const IntegrationRule *irs[Geometry::NumGeom];
    int order_quad;
    double err_u;
    double norm_u;
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
    const char* outfolder = nullptr;

    /// Solve the a single iteration of the problem
    void Step();

    /// Compute error for pressure and velocity
    void ComputeError();

    /// Update solution for next iteration
    void UpdateSolution();

    /**
    * \brief Matrix vector multiplication.
    *
    * Multiply matrix and vector.
    *
    * \param ess_tdof_list List of essential degrees of freedom
    * \param mat   modified matrix (obtained by EliminateRowsCols)
    * \param mat_e eliminated matrix (obtained by EliminateRowsCols)
    * \param x     vector being multiplied
    * \param y     vector for storing the result
    *
    */
    void FullMult(HypreParMatrix* mat, HypreParMatrix* mat_e, Vector &x, Vector &y);

    // Save results in Paraview or GLVis format
    void SaveResults( int iter );

    /// Print information about the Navier version.
    void PrintInfo();

    /// Output matrices in matlab format (for debug).
    void PrintMatricesVectors( const char* id, int num );
};
} // end of namespace "mfem"

#endif
