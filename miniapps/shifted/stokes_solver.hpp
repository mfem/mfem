#ifndef STOKES_SOLVER_HPP
#define STOKES_SOLVER_HPP

#include "mfem.hpp"

namespace mfem {

  class Viscosity:public Coefficient
  {
  public:
    Viscosity(mfem::ParMesh* mesh_): pmesh(mesh_),mu(pmesh->attributes.Max()),counter(0)
    {}
     
    ~Viscosity(){delete pmesh;}

    void AddViscosity(double val)
    {
      mu[counter] = val;
      counter++;
    }


    virtual
    double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
      int att = T.Attribute;
      return (mu(att-1));
    }



  private:
    ParMesh* pmesh;
    Vector mu;
    int counter;
  };

  class StokesSolver
  {
  public:
    StokesSolver(mfem::ParMesh* mesh_, int vorder=2, int porder = 1, bool vis = false);

    ~StokesSolver();

    /// Set the Newton Solver
    void SetNewtonSolver(double rtol=1e-7, double atol=1e-12,int miter=1000, int prt_level=1);

    /// Solves the forward problem.
    void FSolve();

    /// Adds velocity BC in direction 0(x),1(y),2(z), or 4(all).
    void AddVelocityBC(int id, mfem::VectorCoefficient& val);

    /// Add surface load
    void AddSurfLoad(int id, mfem::VectorCoefficient& ff)
    {
      mfem::Array<int> * ess_bdr_tmp = new Array<int>(pmesh->bdr_attributes.Max());
      (*ess_bdr_tmp) = 0;
      (*ess_bdr_tmp)[id-1] = 1;
      surf_loads.insert({ess_bdr_tmp,&ff});
    }

    /// Associates coefficient to the volumetric force.
    void SetVolForce(mfem::VectorCoefficient& ff);

    /// Set exact velocity solution.
    void SetExactVelocitySolution(mfem::VectorCoefficient& ff);

    /// Set exact pressure solution.
    void SetExactPressureSolution(mfem::Coefficient& ff);

    /// Returns the velocities.
    mfem::ParGridFunction& GetVelocities()
    {
      return *fvelocity;
    }

    /// Returns the pressures.
    mfem::ParGridFunction& GetPressures()
    {
      return *fpressure;
    }

    /// Add material to the solver. The solver owns the data.
    void AddMaterial(double val)
    {
      materials->AddViscosity(val);
    }

    /// Returns the solution vector.
    mfem::BlockVector& GetFullSol(){return *trueX;}

    void GetVelocitySol(ParGridFunction& sgf){
      sgf.SetSpace(vfes); sgf.SetFromTrueDofs(*fvelocity);}

    void GetPressureSol(ParGridFunction& sgf){
      sgf.SetSpace(vfes); sgf.SetFromTrueDofs(*fpressure);}

    void ComputeL2Errors();

    void VisualizeFields();
      
  private:
    mfem::ParMesh* pmesh;

    //solution true vector
    mfem::BlockVector * x;
    mfem::BlockVector * trueX;
    //RHS
    mfem::BlockVector * rhs;
    mfem::BlockVector * trueRhs;

    int velocityOrder;
    int pressureOrder;

    bool visualization;
    
    Array<int> block_offsets; // number of variables + 1
    Array<int> block_trueOffsets; // number of variables + 1

    //forward solution
    mfem::ParGridFunction * fvelocity;
    mfem::ParGridFunction * fpressure;

    /// Volumetric force coefficient can point to the
    /// one created by the solver or to external vector
    /// coefficient.
    mfem::VectorCoefficient* volforce;

    BlockDiagonalPreconditioner *stokesPr; //preconditioner

    mfem::GMRESSolver *ns;

    // velocity space and fe
    mfem::ParFiniteElementSpace* vfes;
    mfem::FiniteElementCollection* vfec;
    // pressure space and fe
    mfem::ParFiniteElementSpace* pfes;
    mfem::FiniteElementCollection* pfec;

    Viscosity* materials;

    BlockOperator * stokesOp;

    mfem::VectorCoefficient* exactVelocity;

    mfem::Coefficient* exactPressure;
        
    // holds the displacement contrained DOFs
    mfem::Array<int> ess_tdofv;

    // velocity BC
    std::map<int ,mfem::VectorCoefficient*> velocity_BC;

    //surface loads
    std::map<mfem::Array<int> *,mfem::VectorCoefficient*> surf_loads;


    //Newton solver parameters
    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;
  };

}

#endif
