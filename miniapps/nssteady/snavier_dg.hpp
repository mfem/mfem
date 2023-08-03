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
    const char *petscrc_file = "rc_direct"; // this will only be used in PETSc.
    bool petsc=true;

    SolverParams(double rtol_ = 1e-6,
    		     double atol_ = 1e-10,
				 int maxIter_ = 1000,
				 int pl_ = 0,
				 bool petsc=true,
				 const char *petscrc_file_ = "rc_direct")
        : rtol(rtol_), atol(atol_), maxIter(maxIter_), pl(pl_), petsc(petsc), petscrc_file(petscrc_file_){}
};

/// Container for vector coefficient holding coeff and mesh attribute (useful for BCs and forcing terms).
class VecFcnCoeffContainer
{
public:
   VecFcnCoeffContainer(Array<int> attr, VectorFunctionCoefficient *coeff_)
      : attr(attr)
   {
      this->coeff = coeff_;
   }

   VecFcnCoeffContainer(VecFcnCoeffContainer &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~VecFcnCoeffContainer()
   {
        delete coeff;
        coeff=nullptr;
    }

   Array<int> attr;
   VectorFunctionCoefficient *coeff = nullptr;
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

	SNavierPicardDGSolver(ParMesh* mesh_,
			             int sorder_=1,
						 int vorder_=2,
						 int porder_=1,
						 double kin_vis_=1,
						 double kappa_0_=1,
						 bool verbose_=false);
	~SNavierPicardDGSolver();

    //TODO: add comments later
    void AddVelDirichletBC(VectorFunctionCoefficient *coeff, Array<int> &attr);
    void AddVelDirichletBC(VectorFunctionCoefficient *coeff, int &attr);
    void AddTractionBC(VectorFunctionCoefficient *coeff, Array<int> &attr);
    void AddAccelTerm(VectorFunctionCoefficient *coeff, Array<int> &attr);

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
    void SetLinearSolvers(SolverParams params);

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
    void SetInitialConditionVel(VectorCoefficient &u_in, VectorCoefficient &w_in);

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
    ParGridFunction* GetVSol(){return &u_gf;}

    /**
    * \brief Returns the pressure solution vector.
    */
    ParGridFunction* GetPSol(){return &p_gf;}

private:
    /// mesh
    ParMesh* pmesh=nullptr;
    int dim;

    /// Velocity and Pressure FE spaces
    FiniteElementCollection* sigfec=nullptr;
    FiniteElementCollection* ufec=nullptr;
    FiniteElementCollection* pfec=nullptr;
    ParFiniteElementSpace* sigfes=nullptr;
    ParFiniteElementSpace* ufes=nullptr;
    ParFiniteElementSpace* pfes=nullptr;
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
    VectorGridFunctionCoefficient *uk_coeff; // velocity at previous iteration

    /// Dirichlet conditions
    Array<int> nbc_bdr;          	// Neumann attributes (pseudo-traction applied)
    Array<int> dbc_bdr;				// Dirichlet attributes (full velocity applied).
    Array<int> tmp_bdr;

    // Bookkeeping for velocity dirichlet bcs (full Vector coefficient).
    std::vector<VecFcnCoeffContainer> vel_dbcs;

    // Bookkeeping for traction (neumann) bcs.
    std::vector<VecFcnCoeffContainer> traction_bcs;

    // Bookkeeping for acceleration (forcing) terms.
    std::vector<VecFcnCoeffContainer> accel_terms;

    /// Bilinear/linear forms
    ParBilinearForm      *ainv_form=nullptr;
    ParMixedBilinearForm *b_form=nullptr;
    ParBilinearForm      *c_form=nullptr;
    ParMixedBilinearForm *d_form=nullptr;

    ParLinearForm *f_form=nullptr;
    ParLinearForm *g_form=nullptr;
    ParLinearForm *h_form=nullptr;

    /// Arrays
    Array<int> block_offsets;     // number of variables + 1
    Array<int> trueblock_offsets; // number of variables + 1

	/// Vectors
//    ParGridFunction *u    = nullptr;    // velocity solution
//    ParGridFunction *p    = nullptr;    // pressure solution vector
//    ParGridFunction *uk   = nullptr;    // velocity at previous iteration

    BlockVector *x_bvec      = nullptr;
    BlockVector *truex_bvec  = nullptr;
	BlockVector *rhs_bvec    = nullptr;
	BlockVector *truerhs_bvec= nullptr;

	/// Matrices/operators
	Array2D< HypreParMatrix* > OseenOp;
	HypreParMatrix *OseenMono = nullptr;
	HypreParMatrix *Ainv_mat = nullptr;
	HypreParMatrix *B_mat = nullptr;
	HypreParMatrix *Bt_mat = nullptr;
	HypreParMatrix *C_mat = nullptr;
	HypreParMatrix *D_mat = nullptr;
	HypreParMatrix *nD_mat = nullptr;
	HypreParMatrix *BtAinv_mat = nullptr;
	HypreParMatrix *BtAinvB_mat = nullptr;
	HypreParMatrix *BtAinvB_C_mat = nullptr;
	HypreParMatrix *zero_mat= nullptr;

	HypreParVector *F_vec=nullptr;
	HypreParVector *BtAinvF_vec=nullptr;
	HypreParVector *G_vec=nullptr;

    /// Kinematic viscosity.
    ConstantCoefficient kin_vis;

    /// Reynolds number
    double Re;

    /// Stabilization parameter
    double kappa_0;

    /// Load vector coefficient
    VectorFunctionCoefficient *fcoeff = nullptr;

    /// Traction coefficient for Neumann
    VectorFunctionCoefficient *traction = nullptr;

    /// Fixed point solver parameters
    SolverParams sParams_Picard;
    int iter;

    /// Linear solvers parameters
    SolverParams sParams_Lin;

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

    /// Compute the residual of the momentum equation
    void ComputeRes();

    /// Update solution for next iteration
    void UpdateSolution();

    // Save results in Paraview or GLVis format
    void SaveResults( int iter );

    /// Print information about the Navier version.
    void PrintInfo();

    /// Output matrices in matlab format (for debug).
    void PrintMatricesVectors( const char* id, int num );
};
} // end of namespace "mfem"

#endif
