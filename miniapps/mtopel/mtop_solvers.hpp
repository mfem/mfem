#ifndef MTOP_SOLVERS_HPP
#define MTOP_SOLVERS_HPP

#include "mfem.hpp"
#include "mtop_filters.hpp"

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


template<class TFloat,class TVec>
void EvalStressIsoMat2D(TFloat EE, TFloat nnu, TVec& strain, TVec& ss)
{
    TFloat mu=EE/(2.0*(1.0+nnu));
    TFloat ll=nnu*EE/((1.0+nnu)*(1.0-2.0*nnu));

    for(int j=0;j<2;j++){
    for(int i=0;i<2;i++){
        ss(i,j)=2.0*mu*strain(i,j);
    }}

    ss(0,0)=ss(0,0)+ll*(strain(0,0)+strain(1,1));
    ss(1,1)=ss(1,1)+ll*(strain(0,0)+strain(1,1));
}

template<class TFloat,class TMat>
void EvalStressIsoMat3D(TFloat EE, TFloat nnu, TMat& strain, TMat& ss)
{

    TFloat mu=EE/(2.0*(1.0+nnu));
    TFloat ll=nnu*EE/((1.0+nnu)*(1.0-2.0*nnu));

    for(int j=0;j<3;j++){
    for(int i=0;i<3;i++){
        ss(i,j)=2.0*mu*strain(i,j);
    }}

    ss(0,0)=ss(0,0)+ll*(strain(0,0)+strain(1,1)+strain(2,2));
    ss(1,1)=ss(1,1)+ll*(strain(0,0)+strain(1,1)+strain(2,2));
    ss(2,2)=ss(2,2)+ll*(strain(0,0)+strain(1,1)+strain(2,2));
}


template<class TMat, class TVec>
void Convert3DVoigtStrain(TMat& mat,TVec& vec)
{
    vec[0]=mat(0,0);
    vec[1]=mat(1,1);
    vec[2]=mat(2,2);
    vec[3]=2.0*mat(1,2);
    vec[4]=2.0*mat(0,2);
    vec[5]=2.0*mat(0,1);
}

template<class TMat, class TVec>
void Convert2DVoigtStrain(TMat& mat,TVec& vec)
{
    vec[0]=mat(0,0);
    vec[1]=mat(1,1);
    vec[2]=2.0*mat(0,1);
}

template<class TMat>
void EvalLinStrain3D(TMat& gradu, TMat& strain)
{
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            strain(i,j)=0.5*(gradu(i,j)+gradu(j,i));
        }
    }
}

template<class TMat>
void EvalLinStrain2D(TMat& gradu, TMat& strain)
{
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            strain(i,j)=0.5*(gradu(i,j)+gradu(j,i));
        }
    }
}

template<class TFloat,class TMat>
TFloat vonMisesStress3D(TMat stress)
{
    TFloat vms=0.0;
    vms=vms+(stress(0,0)-stress(1,1))*(stress(0,0)-stress(1,1));
    vms=vms+(stress(1,1)-stress(2,2))*(stress(1,1)-stress(2,2));
    vms=vms+(stress(2,2)-stress(0,0))*(stress(2,2)-stress(0,0));
    vms=vms+(stress(0,1)*stress(0,1)+stress(1,0)*stress(1,0))*3.0;
    vms=vms+(stress(0,2)*stress(0,2)+stress(2,0)*stress(2,0))*3.0;
    vms=vms+(stress(1,2)*stress(1,2)+stress(2,1)*stress(2,1))*3.0;
    return sqrt(vms/2.0);
}

template<class TFloat,class TMat>
TFloat vonMisesStress2D(TMat stress)
{
    TFloat vms=0.0;
    vms=vms+(stress(0,0)-stress(1,1))*(stress(0,0)-stress(1,1));
    vms=vms+(stress(1,1)-stress(2,2))*(stress(1,1)-stress(2,2));
    vms=vms+(stress(0,1)*stress(0,1)+stress(1,0)*stress(1,0))*3.0;
    return sqrt(vms/2.0);
}

template<class TFloat,class TMat>
TFloat vonMisesStress3DGrad(TMat stress, TMat grad)
{
    TFloat vms=vonMisesStress3D(stress);

    grad(0,0)=(2.0*stress(0,0)-stress(1,1)-stress(2,2))/(2.0*vms);
    grad(1,1)=(-stress(0,0)+2.0*stress(1,1)-stress(2,2))/(2.0*vms);
    grad(2,2)=(-stress(0,0)-stress(1,1)+2.0*stress(2,2))/(2.0*vms);

    grad(0,1)=3.0*stress(0,1)/(2.0*vms);
    grad(0,2)=3.0*stress(0,2)/(2.0*vms);
    grad(1,2)=3.0*stress(1,2)/(2.0*vms);

    grad(1,0)=3.0*stress(1,0)/(2.0*vms);
    grad(2,0)=3.0*stress(2,0)/(2.0*vms);
    grad(2,1)=3.0*stress(2,1)/(2.0*vms);

    return vms;
}

template<class TFloat,class TMat>
TFloat vonMisesStress2DGrad(TMat stress, TMat grad)
{
    TFloat vms=vonMisesStress2D(stress);

    grad(0,0)=(2.0*stress(0,0)-stress(1,1))/(2.0*vms);
    grad(1,1)=(-stress(0,0)+2.0*stress(1,1))/(2.0*vms);

    grad(0,1)=3.0*stress(0,1)/(2.0*vms);
    grad(1,0)=3.0*stress(1,0)/(2.0*vms);

    return vms;
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


class NLSurfLoadIntegrator:public NonlinearFormIntegrator
{
public:
    NLSurfLoadIntegrator(int id_, mfem::VectorCoefficient* vc_)
    {
        vc=vc_;
        sid=id_;
    }

    virtual
    void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                            FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect);

    virtual
    void AssembleFaceGrad (const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Tr, const Vector &elfun, DenseMatrix &elmat);

private:
    mfem::VectorCoefficient* vc;
    int sid;
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

    /// Forms the tangent matrix
    void AssembleTangent();

    /// Solves the linearized system Ku=f,
    /// where K is the assembled tangent matrix
    void LSolve();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::Vector& rhs);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, double val);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, mfem::Coefficient& val);

    /// Clear all displacement BC
    void DelDispBC();

    /// Set the values of the volumetric force.
    void SetVolForce(double fx,double fy, double fz);

    /// Add surface load
    void AddSurfLoad(int id, double fx,double fy, double fz=0.0)
    {
        mfem::Vector vec; vec.SetSize(pmesh->SpaceDimension());
        vec[0]=fx;
        vec[1]=fy;
        if(pmesh->SpaceDimension()==3){vec[2]=fz;}
        mfem::VectorConstantCoefficient* vc=new mfem::VectorConstantCoefficient(vec);
        if(load_coeff.find(id)!=load_coeff.end()){ delete load_coeff[id];}
        load_coeff[id]=vc;
    }

    /// Add surface load
    void AddSurfLoad(int id, mfem::VectorCoefficient& ff)
    {
        surf_loads[id]=&ff;
    }

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

    //surface loads
    std::map<int,mfem::VectorConstantCoefficient*> load_coeff;
    std::map<int,mfem::VectorCoefficient*> surf_loads;

    // boundary conditions for x,y, and z directions
    std::map<int, mfem::ConstantCoefficient> bcx;
    std::map<int, mfem::ConstantCoefficient> bcy;
    std::map<int, mfem::ConstantCoefficient> bcz;

    // holds BC in coefficient form
    std::map<int, mfem::Coefficient*> bccx;
    std::map<int, mfem::Coefficient*> bccy;
    std::map<int, mfem::Coefficient*> bccz;

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

    HypreParMatrix* A;
};



class YoungModulus:public Coefficient
{
public:
    YoungModulus()
    {
        dens=nullptr;
        Emax=1.0;
        Emin=1e-6;
        loc_eta=new ConstantCoefficient(0.5);
        eta=loc_eta;
        beta=8.0;
        pp=1.0;
    }

    virtual
    ~YoungModulus(){
        delete loc_eta;
    }

    virtual
    void SetDens(ParGridFunction* dens_)
    {
        dens=dens_;
    }

    virtual
    void SetDens(Coefficient* coef_)
    {
        coef=coef_;
    }

    virtual
    void SetProjParam(Coefficient& eta_, double beta_)
    {
        eta=&eta_;
        beta=beta_;
    }

    virtual
    void SetProjParam(double eta_, double beta_)
    {
        delete loc_eta;
        loc_eta=new ConstantCoefficient(eta_);
        eta=loc_eta;
        beta=beta_;
    }

    virtual
    void SetEMaxMin(double Emin_,double Emax_)
    {
        Emax=Emax_;
        Emin=Emin_;
    }

    virtual
    void SetPenal(double pp_)
    {
        pp=pp_;
    }

    virtual
    double 	Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        //evaluate density
        double dd=1.0;
        if(dens!=nullptr){dd=dens->GetValue(T,ip);}
        else if(coef!=nullptr){dd=coef->Eval(T,ip);}

        if(dd>1.0){dd=1.0;}
        if(dd<0.0){dd=0.0;}
        //eval eta
        double deta=eta->Eval(T,ip);
        //do the projection
        double pd=PointwiseTrans::HProject(dd,deta,beta);
        //evaluate the E modulus
        return Emin+(Emax-Emin)*std::pow(pd,pp);
    }

    ///returnas the pointwise gradient with respect to the density
    virtual
    double Grad(ElementTransformation &T, const IntegrationPoint &ip)
    {
        //evaluate density
        double dd=1.0;
        if(dens!=nullptr){dd=dens->GetValue(T,ip);}
        else if(coef!=nullptr){dd=coef->Eval(T,ip);}

        if(dd>1.0){dd=1.0;}
        if(dd<0.0){dd=0.0;}
        //eval eta
        double deta=eta->Eval(T,ip);
        //do the projection
        double pd=PointwiseTrans::HProject(dd,deta,beta);
        //evaluate hte gradient of the projection
        double pg=PointwiseTrans::HGrad(dd,deta,beta);
        //evaluate the gradient with respect to the density field
        return (Emax-Emin)*pg*pp*std::pow(pd,pp-1.0);
    }



private:
    ParGridFunction* dens;
    Coefficient* coef;
    double Emax;
    double Emin;
    Coefficient* eta;
    Coefficient* loc_eta;
    double beta;
    double pp;

};

class YoungModulusSIMP:public YoungModulus
{
public:
    YoungModulusSIMP()
    {
        dens=nullptr;
        Emax=1.0;
        Emin=1e-6;
        pp=3.0;
    }

    ~YoungModulusSIMP(){}

    void SetDens(ParGridFunction* dens_)
    {
        dens=dens_;
    }

    void SetProjParam(double eta_, double beta_)
    {
    }

    void SetEMaxMin(double Emin_,double Emax_)
    {
        Emax=Emax_;
        Emin=Emin_;
    }

    void SetPenal(double pp_)
    {
        pp=pp_;
    }

    virtual
    double 	Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        //evaluate density
        double dd=dens->GetValue(T,ip);
        if(dd<0.0){dd=0.0;}
        if(dd>1.0){dd=1.0;}
        //evaluate the E modulus
        return Emin+(Emax-Emin)*std::pow(dd,pp);
    }

    ///returnas the pointwise gradient with respect to the density
    virtual
    double Grad(ElementTransformation &T, const IntegrationPoint &ip)
    {
        //evaluate density
        double dd=dens->GetValue(T,ip);
        if(dd<0.0){dd=0.0;}
        //evaluate the gradient with respect to the density field
        return (Emax-Emin)*pp*std::pow(dd,pp-1.0);
    }

private:
    double pp;
    double Emax;
    double Emin;
    mfem::ParGridFunction* dens;
};

class ComplianceNLIntegrator:public NonlinearFormIntegrator
{
public:

    ComplianceNLIntegrator()
    {
        //The class does not own any the following objects
        disp=nullptr;
        volforce=nullptr;
        nu=0.2;
        Ecoef=nullptr;
    }

    void SetDisp(ParGridFunction* disp_)
    {
        disp=disp_;
    }

    void SetE(YoungModulus* E_)
    {
        Ecoef=E_;
    }

    void SetPoissonRatio(double nu_)
    {
        nu=nu_;
    }

    void SetVolForce(VectorCoefficient* force_)
    {
        volforce=force_;
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
    ParGridFunction* disp;

    mfem::VectorCoefficient* volforce;

    double nu;
    YoungModulus* Ecoef;
};



class ComplianceObjective
{
public:
    ComplianceObjective()
    {
        esolv=nullptr;
        dfes=nullptr;
        nf=nullptr;
        dens=nullptr;
        nu=0.2;
    }

    ~ComplianceObjective(){ delete nf;};

    void SetElastSolver(ElasticitySolver* esolv_){
        esolv=esolv_;
    }

    void SetFilter(FilterSolver* fsolv_){
        //dfes=fsolv_->GetDesignFES();
        dfes=fsolv_->GetFilterFES();
    }

    void SetDesignFES(ParFiniteElementSpace* fes)
    {
        dfes=fes;
    }

    void SetE(YoungModulus* E_){
        Ecoef=E_;
    }

    void SetPoissonRatio(double nu_)
    {
        nu=nu_;
    }

    void SetDens(Vector& dens_)
    {
        dens=&dens_;
    }

    double Eval();

    double Eval(mfem::ParGridFunction& sol);

    void Grad(Vector& grad);

    void Grad(mfem::ParGridFunction& sol, Vector& grad);

private:
    ElasticitySolver* esolv;
    ParFiniteElementSpace* dfes;//design space
    ParNonlinearForm* nf;
    ComplianceNLIntegrator* intgr;
    YoungModulus* Ecoef;
    double nu;
    Vector* dens;

};


class InvMechObjIntegrator:public NonlinearFormIntegrator
{
public:
    InvMechObjIntegrator()
    {
        //The class does not own any the following objects
        disp=nullptr;
        adj=nullptr;
        nu=0.2;
        Ecoef=nullptr;
    }

    void SetDisp(ParGridFunction* disp_)
    {
        disp=disp_;
    }

    void SetAdj(ParGridFunction* adj_)
    {
        adj=adj_;
    }

    void SetE(YoungModulus* E_)
    {
        Ecoef=E_;
    }

    void SetPoissonRatio(double nu_)
    {
        nu=nu_;
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
    ParGridFunction* disp;
    ParGridFunction* adj;

    double nu;
    YoungModulus* Ecoef;

};


class InvMechObjective
{
public:
    InvMechObjective()
    {
        dfes=nullptr;
        nf=nullptr;
        dens=nullptr;
        nu=0.2;
    }

    ~InvMechObjective(){ delete nf;};

    void SetFilter(FilterSolver* fsolv_){
        //dfes=fsolv_->GetDesignFES();
        dfes=fsolv_->GetFilterFES();
    }

    void SetDesignFES(ParFiniteElementSpace* fes)
    {
        dfes=fes;
    }

    void SetE(YoungModulus* E_){
        Ecoef=E_;
    }

    void SetPoissonRatio(double nu_)
    {
        nu=nu_;
    }

    void SetDens(Vector& dens_)
    {
        dens=&dens_;
    }

    double Eval(mfem::ParGridFunction& sol, mfem::ParGridFunction& adj);

    void Grad(mfem::ParGridFunction& sol, mfem::ParGridFunction& adj, Vector& grad);

private:
    ParFiniteElementSpace* dfes;//design space
    ParNonlinearForm* nf;
    InvMechObjIntegrator* intgr;
    YoungModulus* Ecoef;
    double nu;
    Vector* dens;
};


/*

class StressObjNLIntegrator:public NonlinearFormIntegrator
{
private:

    GridFunction* disp; //displacements
    GridFunction* adj; //adjoint

    double nu;
    YoungModulus* Ecoef;

public:
    StressObjNLIntegrator();

    void SetDisp(GridFunction* disp_)
    {
        disp=disp_;
    }

    void SetAdjoint(GridFunction* adj_)
    {
        adj=adj_;
    }

    void SetE(YoungModulus* E_)
    {
        Ecoef=E_;
    }

    void SetPoissonRatio(double nu_)
    {
        nu=nu_;
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
};



///Evaluates the RHS for adjoint
class StressRHSAdjointLFIntegrator:public LinearFormIntegrator
{
private:
    GridFunction* disp;
    YoungModulus* Ecoef;
    double nu;


public:
    StressRHSAdjointLFIntegrator()
    {

    }


    virtual
    void AssembleRHSElementVect (const FiniteElement &el,
                                 ElementTransformation &Tr,
                                 Vector &elvect);

};

*/

class DiffusionSolver
{
private:

    mfem::ParMesh* pmesh;

    //solution true vector
    mfem::Vector sol;
    //adjoint true vector
    mfem::Vector adj;
    //RHS
    mfem::Vector rhs;
    //tmp vector
    mfem::Vector tmpv;

    // boundary conditions
    std::map<int, mfem::ConstantCoefficient> bc;
    // holds BC in coefficient form
    std::map<int, mfem::Coefficient*> bcc;

    // holds the  contrained DOFs
    mfem::Array<int> ess_tdofv;


    //forward solution
    mfem::ParGridFunction fsol;
    //adjoint solution
    mfem::ParGridFunction asol;

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


    mfem::ParFiniteElementSpace* vfes;
    mfem::FiniteElementCollection* vfec;

    std::vector<mfem::MatrixCoefficient*> materials;

    /// Volumetric force created by the solver.
    mfem::ConstantCoefficient* lvforce;
    /// Volumetric force coefficient can point to the
    /// one created by the solver or to external vector
    /// coefficient.
    mfem::Coefficient* volforce;

    HypreParMatrix* A;
    HypreParMatrix* Ae;


public:
    DiffusionSolver(mfem::ParMesh* mesh_,int vorder=1);

    ~DiffusionSolver();

    /// Set the Newton Solver
    void SetNewtonSolver(double rtol=1e-7, double atol=1e-12,int miter=100, int prt_level=0);

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=100);

    /// Solves the forward problem.
    void FSolve();


    /// Forms the tangent matrix
    void AssembleTangent();

    /// Solves the linearized system Ku=f,
    /// where K is the assembled tangent matrix
    void LSolve();


    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::Vector& rhs);

    /// Adds Dirichlet BC
    void AddDirichletBC(int id, double val);

    /// Adds Dirichlet BC
    void AddDirichletBC(int id, mfem::Coefficient& val);

    /// Clear all Dirichlet BC
    void DelDirichletBC();

    /// Set the values of the volumetric source
    void SetVolInput(double ff);

    /// Set vol source
    void SetVolInput(mfem::Coefficient& ff);

    /*
    /// Add surface flux
    void AddFlux(int id, double fl);

    /// Add surface flux
    void AddFlux(int id, mfem::Coefficient& fl);
    */


    /// Return the primal solution
    mfem::ParGridFunction& GetFSolution()
    {
        fsol.SetFromTrueDofs(sol);
        return fsol;
    }

    mfem::ParGridFunction& GetASolution()
    {
        asol.SetFromTrueDofs(adj);
        return asol;
    }

    /// Add material to the solver. The solver owns the data.
    void AddMaterial(MatrixCoefficient* nmat){
        materials.push_back(nmat);
    }

    void DelMaterials(){
        for(unsigned int i=0;i<materials.size();i++){
            delete materials[i];
        }
        materials.clear();
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(vfes); sgf.SetFromTrueDofs(sol);}

    void GetAdj(ParGridFunction& agf){
        agf.SetSpace(vfes); agf.SetFromTrueDofs(adj);}

};

class DiffusionMaterial: public MatrixCoefficient
{
public:
    DiffusionMaterial(int dim): MatrixCoefficient(dim)
    {
        dens=nullptr;
        Dmax=1.0;
        Dmin=1e-6;
        loc_eta=new ConstantCoefficient(0.5);
        eta=loc_eta;
        beta=8.0;
        pp=1.0;
        projection=true;
    }

    ~DiffusionMaterial()
    {
        delete loc_eta;
    }

    void SetDens(ParGridFunction* dens_)
    {
        dens=dens_;
    }

    void SetDens(Coefficient* coef_)
    {
        coef=coef_;
    }

    void SetProjParam(Coefficient& eta_, double beta_)
    {
        eta=&eta_;
        beta=beta_;
    }

    void SetProjection(bool pr=true)
    {
        projection=pr;
    }

    void SetProjParam(double eta_, double beta_)
    {
        delete loc_eta;
        loc_eta=new ConstantCoefficient(eta_);
        eta=loc_eta;
        beta=beta_;
    }

    void SetEMaxMin(double Emin_,double Emax_)
    {
        Dmax=Emax_;
        Dmin=Emin_;
    }

    void SetPenal(double pp_)
    {
        pp=pp_;
    }

    virtual
    void Eval(DenseMatrix& m,
              ElementTransformation &T, const IntegrationPoint &ip)
    {
        //m.SetSize(GetWidth());
        //evaluate density
        double dd=1.0;
        if(dens!=nullptr){dd=dens->GetValue(T,ip);}
        else if(coef!=nullptr){dd=coef->Eval(T,ip);}

        if(dd>1.0){dd=1.0;}
        if(dd<0.0){dd=0.0;}
        double cd;
        if(projection==false)
        {
            cd=Dmin+(Dmax-Dmin)*std::pow(dd,pp);
        }else{
            //eval eta
            double deta=eta->Eval(T,ip);
            //do the projection
            double pd=PointwiseTrans::HProject(dd,deta,beta);
            cd=Dmin+(Dmax-Dmin)*std::pow(pd,pp);
        }

        m=0.0;
        for(int i=0;i<GetWidth();i++)
        {
            m(i,i)=cd;
        }
    }

    ///returnas the pointwise gradient with respect to the density
    virtual
    void Grad(DenseMatrix& m,
              ElementTransformation &T, const IntegrationPoint &ip)
    {
        //evaluate density
        double dd=1.0;
        if(dens!=nullptr){dd=dens->GetValue(T,ip);}
        else if(coef!=nullptr){dd=coef->Eval(T,ip);}

        if(dd>1.0){dd=1.0;}
        if(dd<0.0){dd=0.0;}
        double cd;
        if(projection==false)
        {
            cd=(Dmax-Dmin)*pp*std::pow(dd,pp-1.0);
        }else{
            //eval eta
            double deta=eta->Eval(T,ip);
            //do the projection
            double pd=PointwiseTrans::HProject(dd,deta,beta);
            //evaluate hte gradient of the projection
            double pg=PointwiseTrans::HGrad(dd,deta,beta);
            cd=(Dmax-Dmin)*pg*pp*std::pow(pd,pp-1.0);
        }

        m=0.0;
        for(int i=0;i<GetWidth();i++)
        {
            m(i,i)=cd;
        }
    }

private:
    ParGridFunction* dens;
    Coefficient* coef;
    double Dmax;
    double Dmin;
    Coefficient* eta;
    Coefficient* loc_eta;
    double beta;
    double pp;
    bool projection;

};


class ComplianceDiffIntegrator:public NonlinearFormIntegrator
{
public:
    ComplianceDiffIntegrator()
    {
        //The class does not own any the following objects
        temp=nullptr;
        volsrc=nullptr;
        diff_tensor=nullptr;
    }

    void SetTemperature(ParGridFunction& temp_)
    {
        temp=&temp_;
    }

    void SetTemperature(ParGridFunction* temp_)
    {
        temp=temp_;
    }

    void SetDiffTensor(DiffusionMaterial* t)
    {
        diff_tensor=t;
    }

    void SetVolSrc(Coefficient* src_)
    {
        volsrc=src_;
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

    ParGridFunction* temp;
    mfem::Coefficient* volsrc;
    DiffusionMaterial* diff_tensor;


};

class DiffusionComplianceObj
{
public:

    DiffusionComplianceObj()
    {
        esolv=nullptr;
        dfes=nullptr;
        nf=nullptr;
        dens=nullptr;
    }

    ~DiffusionComplianceObj()
    {
        delete nf;
    }

    void SetDiffSolver(DiffusionSolver* esolv_){
        esolv=esolv_;
    }

    void SetDesignFES(ParFiniteElementSpace* fes)
    {
        dfes=fes;
    }

    void SetDiffMaterial(DiffusionMaterial* mat_)
    {
        mat=mat_;
    }

    void SetDens(Vector& dens_)
    {
        dens=&dens_;
    }

    double Eval();

    double Eval(mfem::ParGridFunction& sol);

    void Grad(Vector& grad);

    void Grad(mfem::ParGridFunction& sol, Vector& grad);



private:
    DiffusionSolver* esolv;
    ParFiniteElementSpace* dfes;//design space
    ParNonlinearForm* nf;
    ComplianceDiffIntegrator* intgr;
    DiffusionMaterial* mat;
    Vector* dens;

    Vector tmpv;

};


}

#endif
