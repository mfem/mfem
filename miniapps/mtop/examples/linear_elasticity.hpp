#ifndef LINEAR_ELASTICITY_HPP
#define LINEAR_ELASTICITY_HPP
#include "mfem.hpp"


class LinearElasticityTimeDependentOperator : public mfem::TimeDependentOperator
{
public:

LinearElasticityTimeDependentOperator(mfem::ParMesh &mesh_, int vorder =1);

virtual ~LinearElasticityTimeDependentOperator() override
{
}


virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

virtual void MultTranspose(const mfem::Vector &x,mfem::Vector &y) const override;

virtual void ImplicitSolve(const mfem::real_t dt,
                             const mfem::Vector &x,
                             mfem::Vector &k) override;

// sets the objective integraand which will be integrated with the state of the system
void SetObjective(std::shared_ptr<mfem::Operator> op_);


// Assemble the explicit operators
// must be called after setting all material coefficients
// and before time stepping
void AssembleExplicit();

void SetDensity(mfem::Coefficient &rho)
{
    density = std::make_shared<mfem::CoefficientVector>(*qs, mfem::CoefficientStorage::FULL); 
    cdensity = &rho;
    density->Project(rho);
}


// l1, m1 are the Lame parameters for material 1
// l2, m2 are the Lame parameters for material 2
void SetElasticityCoefficients(mfem::Coefficient& l1_,
                               mfem::Coefficient& m1_,
                               mfem::Coefficient& l2_,
                               mfem::Coefficient& m2_)
{
    l1.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 
    l2.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 
    m1.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 
    m2.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));

    l1->Project(l1_);
    m1->Project(m1_);
    l2->Project(l2_);
    m2->Project(m2_);
}


// rho1 and rho2 are the density coefficients for material 1 and 2
void SetDensityMaterialCoefficients(mfem::Coefficient& rho1_,
                                    mfem::Coefficient& rho2_)
{
    dens1.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 
    dens2.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 

    cdens1 = &rho1_;
    cdens2 = &rho2_;

    dens1->Project(rho1_);
    dens2->Project(rho2_);
}

// c1 and c2 are velocity proportional damping coefficients for material 1 and 2
// both of them are considered to be time dependent
void SetDampingMaterialCoefficients(mfem::Coefficient& c1_,
                                    mfem::Coefficient& c2_)
{
    cm1.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));     
    cm2.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));     

    cm1->Project(c1_);
    cm2->Project(c2_);
}

// dl1, dm1 are the strain velocity proportional damping coefficients
// for material 1 and dl2, dm2 for material 2
// all of them are considered to be time dependent 
void SetDampingMaterialCoefficients(mfem::Coefficient& dl1_,
                                    mfem::Coefficient& dm1_,
                                    mfem::Coefficient& dl2_,
                                    mfem::Coefficient& dm2_)
{
    dl1.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 
    dl2.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 
    dm1.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL)); 
    dm2.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));    

    dl1->Project(dl1_);
    dm1->Project(dm1_);
    dl2->Project(dl2_);
    dm2->Project(dm2_);
}             

mfem::ParGridFunction& GetDisplacement() { return displ; }
mfem::ParGridFunction& GetVelocity() { return veloc; }
mfem::Array<int>& GetTrueBlockOffsets(){ return block_true_offsets;}

mfem::Vector& GetState(){return sol;}

const mfem::ParFiniteElementSpace* GetFESpace(){ return fespace.get();}

void SetZeroBdr(int bdr_attr)
{
    zero_bdrs.insert(bdr_attr);
}

void SetBdrLoad(int attr)
{
    bdr_loads_markers.insert(attr);
}

void SetVolForce(mfem::real_t period, mfem::real_t amplitude, mfem::real_t rad,
                     mfem::real_t xc=0.0, mfem::real_t yc=0.0, mfem::real_t zc=0.0,
                     mfem::real_t L=5.0, mfem::real_t t0=0.0, mfem::real_t n=2.0)
{
    // copy data to the host 
    mfem::real_t* pvol_force_mem = vol_force_mem.HostReadWrite();

    pvol_force_mem[1] = period;
    pvol_force_mem[2] = amplitude;
    pvol_force_mem[3] = rad;
    pvol_force_mem[4] = xc;
    pvol_force_mem[5] = yc;
    pvol_force_mem[6] = zc;

    pvol_force_mem[7] = L;
    pvol_force_mem[8] = t0;
    pvol_force_mem[9] = n;

    // copy data to the device
    vol_force_mem.Read();
}

private:
mfem::ParMesh &mesh;
int order;

std::unique_ptr<mfem::FiniteElementCollection>  fec;
std::unique_ptr<mfem::ParFiniteElementSpace>  fespace;
int dim;
int space_dim;

int myrank;

mfem::ParGridFunction *nodes;
mfem::ParFiniteElementSpace *mfes;
mfem::Array<int> domain_attributes;
const mfem::IntegrationRule *ir;

mutable mfem::ParGridFunction displ;
mutable mfem::ParGridFunction veloc;
mutable mfem::ParGridFunction accel;

mutable mfem::BlockVector sol;
mutable mfem::BlockVector rhs;
mutable mfem::BlockVector tmp;
mutable mfem::Vector  res;

mfem::Array<int> block_true_offsets;

std::unique_ptr<mfem::future::UniformParameterSpace> ups;
std::unique_ptr<mfem::QuadratureSpace> qs;
std::unique_ptr<mfem::FaceQuadratureSpace> fqs;

// linear elasticty coefficients in dFEM form
// l1, m1 - material 1
// l2, m2 - material 2
std::unique_ptr<mfem::CoefficientVector> l1, l2;
std::unique_ptr<mfem::CoefficientVector> m1, m2;

// linear elasticity damping coefficients in dFEM form
// strain proportional damping for material 1 and 2
std::unique_ptr<mfem::CoefficientVector> dl1, dl2;
std::unique_ptr<mfem::CoefficientVector> dm1, dm2;

// damping mass coefficients in dFEM form 
// (velocity proportional damping) for material 1 and 2
std::unique_ptr<mfem::CoefficientVector> cm1, cm2;

// density coefficients in dFEM form for material 1 and 2
std::unique_ptr<mfem::CoefficientVector> dens1, dens2;
mfem::Coefficient *cdens1, *cdens2;

// density coefficient for topology optimization
std::shared_ptr<mfem::CoefficientVector> density;
mfem::Coefficient *cdensity;



static constexpr int FDispl = 0; //grid function displacement
static constexpr int FVeloc = 1; //grid function velocity
// elasticity Coefficient Vectors
static constexpr int Lambda1 = 2, Lambda2 = 3, Mu1 = 4, Mu2 = 5;
// damping Coefficient Vectors
static constexpr int DLambda1 = 6, DMu1 = 7, DLambda2 = 8, DMu2 = 9;
static constexpr int CMass1 = 10, CMass2 = 11; // damping mass coeff vectors
static constexpr int Dens1 = 12, Dens2 = 13; // density Coefficient Vectors   
// density for topology optimization
static constexpr int Density = 14; // coefficient vector
static constexpr int Coords = 15; // coordinates grid function

// DFEM forward related definitions
std::unique_ptr<mfem::future::DifferentiableOperator> dfem_forward_op;
std::unique_ptr<mfem::future::DifferentiableOperator> dfem_mass_op;
std::unique_ptr<mfem::future::DifferentiableOperator> dfem_damp_op;
std::unique_ptr<mfem::future::DifferentiableOperator> dfem_vol_force_op;

std::unique_ptr<mfem::HypreParMatrix> M_lor;

std::unique_ptr<mfem::CGSolver> cg;
std::unique_ptr<mfem::HypreBoomerAMG> amg;

// zero bdrs 
std::set<int> zero_bdrs;

// time dependent memory vector for dynamic force
// the force is applied on all boundary attributes in bdr_loads_markers
std::set<int> bdr_loads_markers;
mutable mfem::Vector bdr_force_mem; // [0] - time, [1] - period, [2] - amplitude

// volumetric force parameters
// A*sin(2*pi*t/T)*cos^n (pi (t-t_0)/L)
// [0] - time, [1] - period, [2] - amplitude, [3] - radius
// [4],[5],[6] - point coordinates of the center of the force application
// [7] - L total train length - could be proportional to the period [1]
// [8] - t_0  center of the train
// [9] - n the envelope power
mutable mfem::Vector vol_force_mem; 

// zero bdr dofs - constructed during the corrsponding Assemble calls
mfem::Array<int> ess_tdof_list;

//objective/constraints integrand 
//obj->Mult(x,y)
//takes state vector s  and returns y which consists of multiple objectives/constraints 
std::shared_ptr<mfem::Operator> obj;


};




class ExampleObjectiveIntegrand: public mfem::Operator
{
public:

    ExampleObjectiveIntegrand(mfem::ParFiniteElementSpace* fes_, 
                                std::shared_ptr<mfem::Coefficient> objc);


    void SetCoefficients( std::shared_ptr<mfem::Coefficient> objc);

    //evaluates the QoIs y[1] for a given state x[2 x fes_->GetTrueVSize()]
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

    mfem::real_t EvalScalar(const mfem::Vector &x) const
    {
        mfem::Vector y(1);
        Mult(x, y);
        return y[0];
    }

    void EvalGradient(const mfem::Vector &x, mfem::Vector &grad) const;
    

private:
    std::shared_ptr<mfem::Coefficient> co;
    mfem::ParFiniteElementSpace* fes;
    mutable mfem::ParGridFunction disp;
    mutable mfem::ParGridFunction velo;

    mfem::Operator* grad;

    static constexpr int FDispl = 0; //grid function displacement
    static constexpr int FVeloc = 1; //grid function velocity
    static constexpr int Density = 14; // coefficient vector
    static constexpr int Coords = 15; // coordinates grid function

    // DFEM related definitions (3 objectives)
    std::unique_ptr<mfem::future::DifferentiableOperator> obj;

    std::shared_ptr<mfem::future::DerivativeOperator> dobj_du;

    // density coefficient for computing the objective function
    std::shared_ptr<mfem::CoefficientVector> density;

    mfem::Array<int> block_true_offsets;

    //uniform parameter space
    std::unique_ptr<mfem::future::UniformParameterSpace> ups;
    //quadrature space for the coefficient
    std::unique_ptr<mfem::QuadratureSpace> qs;

    mfem::ParGridFunction *nodes;
    mfem::ParFiniteElementSpace *mfes;
    mfem::Array<int> domain_attributes;
    const mfem::IntegrationRule *ir;

    mutable mfem::Vector res; 

};

#endif // LINEAR_ELASTICITY_HPP