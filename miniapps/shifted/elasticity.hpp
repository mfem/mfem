#ifndef ELASTICITY_SOLVER_HPP
#define ELASTICITY_SOLVER_HPP

#include "mfem.hpp"
#include "sbm_solver.hpp"

namespace mfem {

namespace elast {

template<class TFloat, class TMat>
void IsotropicStiffnessTensor3D(TFloat E, TFloat nu, TMat& CC)
{
    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
            CC(i,j)=0.0;
        }
    }

    CC(0,0)=E*(1.0-nu)/((1.0+nu)*(1.0-2.0*nu));
    CC(0,1)=E*nu/((1.0+nu)*(1.0-2.0*nu));
    CC(0,2)=CC(0,1);

    CC(1,0)=CC(0,1); CC(1,1)=CC(0,0); CC(1,2)=CC(0,1);
    CC(2,0)=CC(0,1); CC(2,1)=CC(0,1); CC(2,2)=CC(0,0);

    CC(3,3)=E/(2.0*(1.0+nu));
    CC(4,4)=CC(3,3);
    CC(5,5)=CC(3,3);

}

template<class TFloat, class TMat>
void IsotropicStiffnessTensor2D(TFloat E, TFloat nu, TMat& CC)
{
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            CC(i,j)=0.0;
        }
    }

    CC(0,0)=E*(1.0-nu)/((1.0+nu)*(1.0-2.0*nu));
    CC(0,1)=E*nu/((1.0+nu)*(1.0-2.0*nu));
    CC(1,0)=CC(0,1); CC(1,1)=CC(0,0);
    CC(2,2)=E/(2.0*(1.0+nu));
}


}


class BasicElasticityCoefficient:public MatrixCoefficient
{
public:
    BasicElasticityCoefficient():MatrixCoefficient(9)
    {}

    virtual
    ~BasicElasticityCoefficient(){}

    virtual
    void SetDisplacementField(mfem::GridFunction& disp){}

    virtual
    double EvalEnergy(ElementTransformation &T, const IntegrationPoint &ip)
    {
        return 0.0;
    }

    virtual
    void EvalStress(mfem::DenseMatrix& ss,ElementTransformation &T, const IntegrationPoint &ip)
    {
        ss.SetSize(3);
        ss=0.0;
    }

    virtual
    void EvalStrain(mfem::DenseMatrix& ee,ElementTransformation &T, const IntegrationPoint &ip)
    {
        ee.SetSize(3);
        ee=0.0;
    }

    // returns matrix 9x9
     virtual
     void EvalTangent(DenseMatrix &mm, Vector& gradu, ElementTransformation &T, const IntegrationPoint &ip)
     {
         mm.SetSize(9);
         mm=0.0;
     }

     // returns vector of length 9
     virtual
     void EvalResidual(Vector &rr, Vector& gradu, ElementTransformation &T, const IntegrationPoint &ip)
     {
         rr.SetSize(9);
         rr=0.0;
     }

     // returns the energy for displacements' gradients gradu
     virtual
     double EvalEnergy(Vector& gradu, ElementTransformation &T, const IntegrationPoint &ip)
     {
         return 0.0;
     }

     // inverse of the stiffness if exists
     virtual
     void EvalCompliance(DenseMatrix &C, Vector& stress,
                         ElementTransformation &T, const IntegrationPoint &ip)
     {
         C.SetSize(6);
         C=0.0;
     }

     // material stiffness at an integration point
      virtual
      void EvalStiffness(DenseMatrix &D, Vector& strain,
                         ElementTransformation &T, const IntegrationPoint &ip)
      {
          D.SetSize(6);
          D=0.0;
      }

};


class LinIsoElasticityCoefficient:public BasicElasticityCoefficient
{
public:
    LinIsoElasticityCoefficient(double E_=1.0, double nu_=0.3)
    {
        lE.constant=E_;
        lnu.constant=nu_;
        E=&lE;
        nu=&lnu;

        disp=nullptr;
        tmpm.SetSize(3);
    }

    LinIsoElasticityCoefficient(Coefficient& E_, double nu_=0.3)
    {
        lnu.constant=nu_;
        E=&E_;
        nu=&lnu;

        disp=nullptr;
        tmpm.SetSize(3);
    }

    LinIsoElasticityCoefficient(Coefficient &E_,Coefficient& nu_)
    {
        E=&E_;
        nu=&nu_;

        disp=nullptr;
        tmpm.SetSize(3);
    }

    LinIsoElasticityCoefficient(LinIsoElasticityCoefficient& co):lE(co.lE),lnu(co.lnu)
    {
        if(&(co.lE)==co.E)
        {
            E=&lE;
        }
        else
        {
            E=co.E;//external coefficient
        }

        if(&(co.lnu)==co.nu)
        {
            nu=&lnu;
        }
        else
        {
            nu=co.nu;//external coefficient
        }

        disp=nullptr;
        tmpm.SetSize(3);
    }

    virtual ~LinIsoElasticityCoefficient(){}

    //return the stiffness tensor
    virtual void Eval(DenseMatrix &C, ElementTransformation &T, const IntegrationPoint &ip)
    {
        MFEM_ASSERT(C.Size()==9,"The size of the stiffness tensor should be set to 9.");
        double EE=E->Eval(T,ip);
        double nnu=nu->Eval(T,ip);
        Eval(EE,nnu,C.Data());
    }

    //return the stress at the integration point
    virtual
    void EvalStress(DenseMatrix &ss, ElementTransformation &T, const IntegrationPoint &ip);

    virtual
    void EvalStrain(DenseMatrix &ee, ElementTransformation &T, const IntegrationPoint &ip);

    virtual
    void SetDisplacementField(GridFunction &disp_)
    {
        disp=&disp_;
        tmpg.SetSize(disp->VectorDim());

    }

    virtual
    double EvalEnergy(ElementTransformation &T, const IntegrationPoint &ip)
    {
        //to implement
        return 0.0;
    }

    //returns vector of length 9
    virtual
    void EvalResidual(Vector &rr, Vector& gradu, ElementTransformation &T, const IntegrationPoint &ip);

    //returns matrix 9x9
    virtual
    void EvalTangent(DenseMatrix &mm, Vector& gradu, ElementTransformation &T, const IntegrationPoint &ip);

    virtual
    double EvalEnergy(Vector& gradu, ElementTransformation &T, const IntegrationPoint &ip);

    virtual
    void EvalCompliance(DenseMatrix &C, Vector& stress,
                            ElementTransformation &T, const IntegrationPoint &ip);

    virtual
    void EvalStiffness(DenseMatrix &D, Vector& strain,
                            ElementTransformation &T, const IntegrationPoint &ip);


private:
    mfem::ConstantCoefficient lE;
    mfem::ConstantCoefficient lnu;

    mfem::Coefficient* E;
    mfem::Coefficient* nu;

    mfem::GridFunction* disp;

    mfem::DenseMatrix tmpm;
    mfem::DenseMatrix tmpg;

    void Eval(double EE,double nnu,double* CC);
    void EvalRes(double EE, double nnu, double* gradu, double* res);
};

class NLVolForceIntegrator:public NonlinearFormIntegrator
{
public:
    NLVolForceIntegrator()
    {
        force=nullptr;
    }

    NLVolForceIntegrator(mfem::VectorCoefficient* coeff)
    {
        force=coeff;
    }

    NLVolForceIntegrator(mfem::VectorCoefficient& coeff)
    {
        force=&coeff;
    }

    void SetVolForce(mfem::VectorCoefficient& coeff)
    {
        force=&coeff;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun);

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat);

private:
    mfem::VectorCoefficient* force;
};

class NLElasticityIntegrator:public NonlinearFormIntegrator
{
public:
    NLElasticityIntegrator()
    {
        elco=nullptr;
    }

    NLElasticityIntegrator(mfem::BasicElasticityCoefficient& coeff)
    {
        elco=&coeff;
    }

    NLElasticityIntegrator(mfem::BasicElasticityCoefficient* coeff)
    {
        elco=coeff;
    }

    void SetElasticityCoefficient(mfem::BasicElasticityCoefficient& coeff)
    {
        elco=&coeff;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun);

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat);

private:

    BasicElasticityCoefficient* elco;
    mfem::VectorCoefficient* force;

};

class ElasticitySolver
{
public:
    ElasticitySolver(mfem::ParMesh* mesh_, int vorder=1);

    ~ElasticitySolver();

    /// Set the Newton Solver
    void SetNewtonSolver(double rtol=1e-7, double atol=1e-12,int miter=1000, int prt_level=1);

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1000);

    /// Solves the forward problem.
    void FSolve();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::Vector& rhs);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, double val);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, mfem::Coefficient& val);

    /// Adds displacement BC specified by the vector coefficient val.
    void AddDispBC(int id, mfem::VectorCoefficient& val);

    /// Set the values of the volumetric force.
    void SetVolForce(double fx,double fy, double fz);

    /// Associates coefficient to the volumetric force.
    void SetVolForce(mfem::VectorCoefficient& ff);

    /// Returns the displacements.
    mfem::ParGridFunction& GetDisplacements()
    {
        fdisp.SetFromTrueDofs(sol);
        return fdisp;
    }

    /// Returns the adjoint displacements.
    mfem::ParGridFunction& GetADisplacements()
    {
        adisp.SetFromTrueDofs(adj);
        return adisp;
    }

    /// Add material to the solver. The solver owns the data.
    void AddMaterial(BasicElasticityCoefficient* nmat){
        if(nf!=nullptr)
        {
            delete nf; nf=nullptr;
        }
        materials.push_back(nmat);
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(vfes); sgf.SetFromTrueDofs(sol);}

    void GetAdj(ParGridFunction& agf){
        agf.SetSpace(vfes); agf.SetFromTrueDofs(adj);}

private:
    mfem::ParMesh* pmesh;

    //solution true vector
    mfem::Vector sol;
    //adjoint true vector
    mfem::Vector adj;
    //RHS
    mfem::Vector rhs;

    /// Volumetric force created by the solver.
    mfem::VectorConstantCoefficient* lvforce;
    /// Volumetric force coefficient can point to the
    /// one created by the solver or to external vector
    /// coefficient.
    mfem::VectorCoefficient* volforce;

    // boundary conditions for x,y, and z directions
    std::map<int, mfem::ConstantCoefficient> bcx;
    std::map<int, mfem::ConstantCoefficient> bcy;
    std::map<int, mfem::ConstantCoefficient> bcz;

    // holds BC in coefficient form
    std::map<int, mfem::Coefficient*> bccx;
    std::map<int, mfem::Coefficient*> bccy;
    std::map<int, mfem::Coefficient*> bccz;
    std::map<int, mfem::VectorCoefficient*> bcca;

    // holds the displacement contrained DOFs
    mfem::Array<int> ess_tdofv;

    //forward solution
    mfem::ParGridFunction fdisp;
    //adjoint solution
    mfem::ParGridFunction adisp;

    //Newton solver parameters
    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    mfem::HypreBoomerAMG *prec; //preconditioner
    mfem::CGSolver *ls;  //linear solver
    mfem::NewtonSolver *ns;

    mfem::ParNonlinearForm *nf;
    mfem::ParFiniteElementSpace* vfes;
    mfem::FiniteElementCollection* vfec;

    std::vector<mfem::BasicElasticityCoefficient*> materials;

};


class SBM2NNIntegrator:public NonlinearFormIntegrator
{
public:
    SBM2NNIntegrator(mfem::ParMesh* pm, mfem::ParFiniteElementSpace* pf,
                     mfem::Array<int>& elm_tags, mfem::Array<int>& fcl_tags)
    {
        pmesh=pm;
        pfes=pf;
        elco=nullptr;

        elem_attributes=&elm_tags;
        face_attributes=&fcl_tags;

        dist_coeff=nullptr;
        dist_grad=nullptr;

        bdr_force=nullptr;

        shift_order=1.0;
        elint=nullptr;

        assembly_interior=false;
        alpha=0.0;
    }

    SBM2NNIntegrator(mfem::ParMesh* pm, mfem::ParFiniteElementSpace* pf,
                      mfem::BasicElasticityCoefficient* coeff,
                      mfem::Array<int>& elm_tags, mfem::Array<int>& fcl_tags)
    {
        pmesh=pm;
        pfes=pf;
        elco=coeff;

        elem_attributes=&elm_tags;
        face_attributes=&fcl_tags;

        dist_coeff=nullptr;
        dist_grad=nullptr;

        bdr_force=nullptr;

        shift_order=1.0;
        elint=new NLElasticityIntegrator(*coeff);

        assembly_interior=false;
        alpha=0.0;
    }

    virtual ~SBM2NNIntegrator()
    {
        delete elint;
    }

    void SetElasticityCoefficient(mfem::BasicElasticityCoefficient& coeff)
    {
        elco=&coeff;
        delete elint;
        elint=new NLElasticityIntegrator(coeff);
    }

    void SetElasticityCoefficient(mfem::BasicElasticityCoefficient* coeff)
    {
        elco=coeff;
        delete elint;
        elint=new NLElasticityIntegrator(*coeff);
    }

    void SetShiftOrder(int new_order)
    {
        shift_order=new_order;
    }

    void SetAssemblyInterior(bool flag)
    {
        assembly_interior=flag;
    }

    void SetPenalty(double val)
    {
        alpha=val;
    }

    ///Sets the distance function dist and the gradients of the distance function grad.
    void SetDistance(mfem::Coefficient* dist, mfem::VectorCoefficient* grad)
    {
        dist_coeff=dist;
        dist_grad=grad;
    }

    void SetForceCoefficient(mfem::ShiftedVectorFunctionCoefficient* bdr)
    {
        bdr_force=bdr;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun)
    {
        return 0.0;
    }

    //Assemble the vector on the inner surrogate
    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                                const Vector &elfun, Vector &elvect);

    //Assemble the gradient on the inner surrogate
    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                              const Vector &elfun, DenseMatrix &elmat);

private:
    void EvalShiftOperator(mfem::DenseMatrix& grad_phys, double dist,
                           mfem::Vector& dir, int order,
                           mfem::DenseMatrix& shift_op, mfem::DenseMatrix& shift_test);

    void EvalDispOp(int dim, mfem::Vector& shape, mfem::DenseMatrix& disp_op)
    {
        int dof=shape.Size();
        disp_op.SetSize(dim,dim*dof); disp_op=0.0;
        for(int i=0;i<dof;i++){
            for(int j=0;j<dim;j++){
                disp_op(j,i+j*dof)=shape(i);
            }
        }
    }

    void StrainOperator(int dim, mfem::DenseMatrix& B, mfem::DenseMatrix& strain_op);

    void StressOperator(int dim, mfem::DenseMatrix &B, mfem::DenseMatrix &D,
                                                        mfem::DenseMatrix &stress_op);

    void BndrStressOperator(int dim, mfem::DenseMatrix& B, mfem::DenseMatrix &D,
                            mfem::Vector& normv, mfem::DenseMatrix &bdrstress_op);

    void EvalG(int dim, mfem::Vector &normv, mfem::DenseMatrix& G);

    void EvalShiftShapes(mfem::Vector& shape, mfem::DenseMatrix& grad, double dist,
                         mfem::Vector& dir, mfem::Vector& shift_shape);

    void FormL2Grad(const FiniteElement &el, ElementTransformation &Tr,
                                DenseMatrix &gradop);


private:

    int shift_order;
    mfem::ParMesh* pmesh;
    mfem::ParFiniteElementSpace* pfes;
    BasicElasticityCoefficient* elco;

    mfem::Coefficient* dist_coeff;
    mfem::VectorCoefficient* dist_grad;
    mfem::Array<int>* elem_attributes;
    mfem::Array<int>* face_attributes;

    mfem::ShiftedVectorFunctionCoefficient* bdr_force;

    mfem::NLElasticityIntegrator *elint;

    bool assembly_interior;
    double alpha;

};

class SBM2ELIntegrator:public NonlinearFormIntegrator
{

public:
    SBM2ELIntegrator(mfem::ParMesh* pm, mfem::ParFiniteElementSpace* pf,
                     mfem::Array<int>& elm_tags, mfem::Array<int>& fcl_tags)
    {
        pmesh=pm;
        pfes=pf;
        elco=nullptr;

        elem_attributes=&elm_tags;
        face_attributes=&fcl_tags;

        dist_coeff=nullptr;
        dist_grad=nullptr;

        bdr_disp=nullptr;

        shift_order=1.0;
        alpha=1.0;

        elint=nullptr;
    }

    SBM2ELIntegrator(mfem::ParMesh* pm, mfem::ParFiniteElementSpace* pf,
                     mfem::BasicElasticityCoefficient* coeff,
                     mfem::Array<int>& elm_tags, mfem::Array<int>& fcl_tags)
    {
        pmesh=pm;
        pfes=pf;
        elco=coeff;

        elem_attributes=&elm_tags;
        face_attributes=&fcl_tags;

        dist_coeff=nullptr;
        dist_grad=nullptr;

        bdr_disp=nullptr;

        shift_order=1.0;
        alpha=1.0;

        elint=new NLElasticityIntegrator(*coeff);
    }

    virtual ~SBM2ELIntegrator()
    {
        delete elint;
    }

    void SetElasticityCoefficient(mfem::BasicElasticityCoefficient& coeff)
    {
        elco=&coeff;
        delete elint;
        elint=new NLElasticityIntegrator(coeff);
    }

    void SetElasticityCoefficient(mfem::BasicElasticityCoefficient* coeff)
    {
        elco=coeff;
        delete elint;
        elint=new NLElasticityIntegrator(*coeff);
    }

    void SetShiftOrder(int new_order)
    {
        shift_order=new_order;
    }

    void SetPenalization(double alpha_)
    {
        alpha=alpha_;
    }

    ///Sets the distance function dist and the gradients of the distance function grad.
    void SetDistance(mfem::Coefficient* dist, mfem::VectorCoefficient* grad)
    {
        dist_coeff=dist;
        dist_grad=grad;
    }

    void SetBdrCoefficient(mfem::ShiftedVectorFunctionCoefficient* bdr)
    {
        bdr_disp=bdr;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun);

    //Assemble the vector on the inner surrogate
    /*
    virtual
    void AssembleElementVectorI(const FiniteElement &el, ElementTransformation &Tr,
                                const Vector &elfun, Vector &elvect);
     */

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                                const Vector &elfun, Vector &elvect);

    //Assemble the gradient on the inner surrogate
    virtual
    void AssembleElementGradI(const FiniteElement &el, ElementTransformation &Tr,
                              const Vector &elfun, DenseMatrix &elmat);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                              const Vector &elfun, DenseMatrix &elmat);



private:
    int shift_order;
    double alpha;
    mfem::ParMesh* pmesh;
    mfem::ParFiniteElementSpace* pfes;
    BasicElasticityCoefficient* elco;

    mfem::Coefficient* dist_coeff;
    mfem::VectorCoefficient* dist_grad;

    mfem::ShiftedVectorFunctionCoefficient* bdr_disp;


    mfem::Array<int>* elem_attributes;
    mfem::Array<int>* face_attributes;

    mfem::NLElasticityIntegrator *elint;

    ///Computes shift operator for given grad_phys matrix, distance dist, and
    /// unit directional vector dir. The matrix grad_phys consists of the
    /// nodal derivatives of the shape functions with respect to x, y and z(in 3D).
    /// The matrix grad_phys has dimensions [d*ndof, ndof] where d is the number of
    /// dimensions.
    void EvalShiftOperator(mfem::DenseMatrix& grad_phys, double dist,
                           mfem::Vector& dir, int order, mfem::DenseMatrix& shift_op);

    ///Same as the above, however it returns the test operator which is only first order
    void EvalShiftOperator(mfem::DenseMatrix& grad_phys, double dist,
                           mfem::Vector& dir, int order,
                           mfem::DenseMatrix& shift_op, mfem::DenseMatrix& shift_test);

    void EvalShiftShapes(mfem::Vector& shape, mfem::DenseMatrix& grad, double dist,
                           mfem::Vector& dir, mfem::Vector& shift_shape);

    void EvalTestShiftOperator(mfem::DenseMatrix& grad_phys, double dist,
                           mfem::Vector& dir,
                           mfem::DenseMatrix& shift_test);

    void EvalDispOp(int dim, mfem::Vector& shape, mfem::DenseMatrix& disp_op)
    {
        int dof=shape.Size();
        disp_op.SetSize(dim,dim*dof); disp_op=0.0;
        for(int i=0;i<dof;i++){
            for(int j=0;j<dim;j++){
                disp_op(j,i+j*dof)=shape(i);
            }
        }
    }

    void StrainOperator(int dim, mfem::DenseMatrix& B, mfem::DenseMatrix& strain_op);

    void StressOperator(int dim, mfem::DenseMatrix &B, mfem::DenseMatrix &D,
                                                        mfem::DenseMatrix &stress_op);

    void BndrStressOperator(int dim, mfem::DenseMatrix& B, mfem::DenseMatrix &D,
                            mfem::Vector& normv, mfem::DenseMatrix &bdrstress_op);

    void EvalG(int dim, mfem::Vector &normv, mfem::DenseMatrix& G);



};


class LevelSetElasticitySolver
{
public:
    LevelSetElasticitySolver(mfem::ParMesh& mesh, int vorder=1);

    ~LevelSetElasticitySolver();

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1000, int restart=50);

    void SetPrintLevel(int prtl=0);

    /// Set the level-set information
    void SetLSF(mfem::ParGridFunction* lf);


    /// Solves the forward problem.
    void FSolve();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::Vector& rhs);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, double val);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, mfem::Coefficient& val);

    void AddDispBC(int id, mfem::VectorCoefficient& val);

    /// Set the values of the volumetric force.
    void SetVolForce(double fx,double fy, double fz);

    /// Associates coefficient to the volumetric force.
    void SetVolForce(mfem::VectorCoefficient& ff);

    /// Returns the displacements.
    mfem::ParGridFunction& GetDisplacements()
    {
        fdisp.SetFromTrueDofs(sol);
        return fdisp;
    }

    /// Returns the adjoint displacements.
    mfem::ParGridFunction& GetADisplacements()
    {
        adisp.SetFromTrueDofs(adj);
        return adisp;
    }

    /// Add material to the solver. The solver owns the data.
    void AddMaterial(BasicElasticityCoefficient* nmat){
        if(nf!=nullptr)
        {
            delete nf; nf=nullptr;
            delete pf; pf=nullptr;
        }
        materials.push_back(nmat);
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(vfes); sgf.SetFromTrueDofs(sol);}

    void GetAdj(ParGridFunction& agf){
        agf.SetSpace(vfes); agf.SetFromTrueDofs(adj);}

private:
    mfem::ParMesh* pmesh;

    //solution true vector
    mfem::Vector sol;
    //adjoint true vector
    mfem::Vector adj;
    //RHS
    mfem::Vector rhs;
    //tmp vector
    mfem::Vector tmv;

    /// Volumetric force created by the solver.
    mfem::VectorConstantCoefficient* lvforce;
    /// Volumetric force coefficient can point to the
    /// one created by the solver or to external vector
    /// coefficient.
    mfem::VectorCoefficient* volforce;

    // boundary conditions for x,y, and z directions
    std::map<int, mfem::ConstantCoefficient> bcx;
    std::map<int, mfem::ConstantCoefficient> bcy;
    std::map<int, mfem::ConstantCoefficient> bcz;

    // holds BC in coefficient form
    std::map<int, mfem::Coefficient*> bccx;
    std::map<int, mfem::Coefficient*> bccy;
    std::map<int, mfem::Coefficient*> bccz;

    // holds BC in vector coefficient form
    std::map<int, mfem::VectorCoefficient*> bcca;

    // holds the displacement contrained DOFs
    mfem::Array<int> ess_tdofv;

    //forward solution
    mfem::ParGridFunction fdisp;
    //adjoint solution
    mfem::ParGridFunction adisp;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;
    int linear_rest;
    int print_level;

    mfem::ParNonlinearForm *nf;
    mfem::ParNonlinearForm *pf; //nf for preconditioning
    mfem::ParFiniteElementSpace* vfes;
    mfem::FiniteElementCollection* vfec;

    std::vector<mfem::BasicElasticityCoefficient*> materials;


    //Level-set function and its derivatives
    mfem::ParGridFunction* lsfunc;
    mfem::GridFunctionCoefficient* distco;
    mfem::GradientGridFunctionCoefficient* gradco;

    mfem::Array<int> element_markers;
    mfem::Array<int> face_markers;
    mfem::Array<int> markers_tdof_list;



};


}

#endif
