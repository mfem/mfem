#ifndef LINEAR_ELASTICITY_SOLVER_HPP
#define LINEAR_ELASTICITY_SOLVER_HPP

#include "mfem.hpp"
#include "volume_weighted_integrals.hpp"
#include "nitsche_weighted_solver.hpp"
#include "volume_fractions.hpp"
#include "shifted_weighted_solver.hpp"
#include "ghost_penalty.hpp"
#include "dist_solver.hpp"
#include "marking.hpp"

class Dist_Level_Set_Coefficient;
class Combo_Level_Set_Coefficient;

namespace mfem {
double AvgElementSize(ParMesh &pmesh);

  class ShearModulus:public Coefficient
  {
  public:
    ShearModulus(mfem::ParMesh* mesh_): pmesh(mesh_),mu(pmesh->attributes.Max())
    {}
     
    ~ShearModulus(){}

    void AddShearModulus(double val, int att = -1)
    {
      if (att == -1){
	for (int i = 0; i < pmesh->attributes.Max(); i++){
	  mu[i] = val;
	}
      }
      else if (att < pmesh->attributes.Max()) {
	mu[att-1] = val; 
      }
      else{
	std::cout << " Invalid attribute" << std::endl;
      }
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
  };

  class BulkModulus:public Coefficient
  {
  public:
    BulkModulus(mfem::ParMesh* mesh_): pmesh(mesh_), kappa(pmesh->attributes.Max())
    {}
     
    ~BulkModulus(){
    }

    void AddBulkModulus(double val, int att = -1)
    {
      if (att == -1){
	for (int i = 0; i < pmesh->attributes.Max(); i++){
	  kappa[i] = val;
	}
      }
      else if (att < pmesh->attributes.Max()) {
	kappa[att-1] = val; 
      }
      else{
	std::cout << " Invalid attribute" << std::endl;
      }
    }

    virtual
    double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
      int att = T.Attribute;
      return (kappa(att-1));
    }

  private:
    ParMesh* pmesh;
    Vector kappa;
  };

  class LinearElasticitySolver
  {
  public:
    LinearElasticitySolver(mfem::ParMesh* mesh_, int vorder, bool useEmb, int gS, int nT, int nStrainTerms, double ghostPenCoeff, bool useMumps, bool useAS, bool vis);

    ~LinearElasticitySolver();

    /// Set the Newton Solver
    void SetNewtonSolver(double rtol=1e-7, double atol=1e-12,int miter=1000, int prt_level=1);

    /// Solves the forward problem.
    void FSolve();

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDisplacementBC(int id, mfem::VectorCoefficient& val)
    {
      mfem::Array<int> * ess_bdr_tmp = new Array<int>(pmesh->bdr_attributes.Max());
      (*ess_bdr_tmp) = 0;
      (*ess_bdr_tmp)[id-1] = 1;
      displacement_BC.insert({ess_bdr_tmp,&val});
    }

    /// Adds shifted displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddShiftedNormalStressBC(mfem::ShiftedVectorFunctionCoefficient& val)
    {
      shifted_traction_BC = &val;
    }

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

    /// Set exact displacement solution.
    void SetExactDisplacementSolution(mfem::VectorCoefficient& ff);

    /// Returns the velocities.
    mfem::ParGridFunction& GetVelocities()
    {
      return *fdisplacement;
    }

    /// Add material to the solver. The solver owns the data.
    void AddMaterial(double shearModCoef, double bulkModCoef)
    {
      shearMod->AddShearModulus(shearModCoef);
      bulkMod->AddBulkModulus(bulkModCoef);
    }

    void GetDisplacementSol(ParGridFunction& sgf){
      sgf.SetSpace(vfes); sgf.SetFromTrueDofs(*fdisplacement);}

    void ComputeL2Errors();

    void VisualizeFields();
      
  private:
    mfem::ParMesh* pmesh;

    int displacementOrder;

    bool useEmbedded;
    int geometricShape;
    int nTerms;
    int numberStrainTerms;
    double ghostPenaltyCoefficient;
    bool mumps_solver;
    bool visualization;
  
    //forward solution
    mfem::ParGridFunction * fdisplacement;

    /// Volumetric force coefficient can point to the
    /// one created by the solver or to external vector
    /// coefficient.
    mfem::VectorCoefficient* volforce;

    Solver *prec;    
    mfem::GMRESSolver *ns;

    // displacement space and fe
    mfem::ParFiniteElementSpace* vfes;
    mfem::FiniteElementCollection* vfec;

    ShearModulus* shearMod;
    BulkModulus* bulkMod;

    mfem::VectorCoefficient* exactDisplacement;
        
    // holds the displacement contrained DOFs
    mfem::Array<int> ess_vdofs;

    // displacement BC
    std::map<mfem::Array<int> * ,mfem::VectorCoefficient*> displacement_BC;

    // shifted displacement BC
    mfem::ShiftedVectorFunctionCoefficient* shifted_traction_BC;

    //surface loads
    std::map<mfem::Array<int> *,mfem::VectorCoefficient*> surf_loads;

    Array<int> ess_elem;

    double C_I;

    Dist_Level_Set_Coefficient *neumann_dist_coef;
    Combo_Level_Set_Coefficient *combo_dist_coef;
    // in case we are using level set to get distance and normal vectors
    ParFiniteElementSpace *distance_vec_space;
    ParGridFunction *distance;
    ParFiniteElementSpace *normal_vec_space;
    ParGridFunction *normal;
    ParGridFunction *ls_func;
    ParGridFunction *level_set_gf;
    ParGridFunction *filt_gf;

    //

    ShiftedFaceMarker *analyticalSurface;
    bool useAnalyticalShape;
    VectorCoefficient *dist_vec;
    VectorCoefficient *normal_vec;

    //Newton solver parameters
    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;
  };

}

#endif
