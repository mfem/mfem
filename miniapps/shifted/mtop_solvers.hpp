#ifndef MTOP_SOLVERS_HPP
#define MTOP_SOLVERS_HPP

#include "mfem.hpp"
#include "mtop_filters.hpp"
#include "marking.hpp"

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

    ~YoungModulus(){
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

    void SetProjParam(double eta_, double beta_)
    {
        delete loc_eta;
        loc_eta=new ConstantCoefficient(eta_);
        eta=loc_eta;
        beta=beta_;
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

#ifdef MFEM_USE_ALGOIM

class VolObjCutIntegrator:public NonlinearFormIntegrator{
public:
    VolObjCutIntegrator()
    {
        coef=nullptr;
    }

    ~VolObjCutIntegrator()
    {

    }

    void SetCoeff(Coefficient* coef_)
    {
        coef=coef_;
    }

    void SetCutIntegrationRules(Array<int>& el_markers_, CutIntegrationRules* cut_int_)
    {
        marks=&el_markers_;
        cut_int=cut_int_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun)
    {
        if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::OUTSIDE){
            return 0.0;
        }

        const IntegrationRule *ir = nullptr;
        if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::INSIDE)
        {
            int order= 2 * el.GetOrder() + Tr.OrderGrad(&el);//this might be too big
            ir=&IntRules.Get(Tr.GetGeometryType(),order);
        }
        else
        {
            ir=cut_int->GetVolIntegrationRule(Tr.ElementNo);
        }

        double w;
        double energy=0.0;
        double mass;
        for(int i=0; i<ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w=Tr.Weight();
            w = ip.weight * w;
            mass=coef->Eval(Tr,ip);
            energy=energy+w*mass;
        }

        return energy;

    }

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect)
    {
        int ndof = el.GetDof();

        elvect.SetSize(ndof); elvect=0.0;

        if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::OUTSIDE){
            return;
        }

        if((*marks)[Tr.ElementNo]==ElementMarker::SBElementType::INSIDE){
            return;
        }

        //cut element

        const IntegrationRule *ir = nullptr;
        ir=cut_int->GetSurfIntegrationRule(Tr.ElementNo);

        double w;
        double f;
        Vector shf(ndof);

        for (int j = 0; j < ir->GetNPoints(); j++)
        {
           const IntegrationPoint &ip = ir->IntPoint(j);
           w=ip.weight;
           f=coef->Eval(Tr,ip);
           el.CalcPhysShape(Tr,shf);
           elvect.Add(w * f , shf);
        }
    }

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat)
    {
        int ndof = el.GetDof();
        elmat.SetSize(ndof); elmat=0.0;
        return;
    }


private:
    Coefficient* coef;
    CutIntegrationRules* cut_int;
    Array<int>* marks;
};

class VolObjectiveCut
{

public:
    VolObjectiveCut():volw(1)
    {
        volc=&volw;
        nf=nullptr;

    }

    ~VolObjectiveCut()
    {
        delete nf;
    }

    void SetCutIntegrationRules(Array<int>& el_markers_, CutIntegrationRules* cut_int_)
    {
        marks=&el_markers_;
        cut_int=cut_int_;
    }


    void SetWeight(Coefficient* coeff){
        volc=coeff;
    }

    double Eval(ParGridFunction& lsf){
        if(nf==nullptr){//allocate the nonlinear form
            nf=new ParNonlinearForm(lsf.ParFESpace());
            itgr=new VolObjCutIntegrator();
            itgr->SetCoeff(volc);
            itgr->SetCutIntegrationRules(*marks,cut_int);
            nf->AddDomainIntegrator(itgr);
        }

        return nf->GetEnergy(lsf.GetTrueVector());

    }

    void Grad(ParGridFunction& lsf, Vector& grad){
        grad.SetSize(lsf.GetTrueVector().Size());
        nf->Mult(lsf.GetTrueVector(),grad);
    }

private:
    ConstantCoefficient volw; //volume weight
    Coefficient* volc; //points either to volw or to user supplied coefficient

    CutIntegrationRules* cut_int;
    Array<int>* marks;

    ParNonlinearForm* nf;
    VolObjCutIntegrator* itgr;


};

#endif



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


class ElastGhostPenaltyIntegrator:public BilinearFormIntegrator
{
private:
    double beta;//penalty parameter
    Array<int>* markers;
public:

    ElastGhostPenaltyIntegrator(Array<int>& el_marks, double penal)
    {
        beta=penal;
        markers=&el_marks;
    }

    virtual void AssembleFaceMatrix(const FiniteElement &fe1,
                                        const FiniteElement &fe2,
                                        FaceElementTransformations &Tr,
                                        DenseMatrix &elmat);
};


#ifdef MFEM_USE_ALGOIM

class CFNLElasticityIntegrator:public NonlinearFormIntegrator
{
public:
    CFNLElasticityIntegrator()
    {
        elco=nullptr;
        forc=nullptr;
        surfl=nullptr;
        cut_int=nullptr;
        stiffness_ratio=1e-6;
    }

    CFNLElasticityIntegrator(mfem::BasicElasticityCoefficient& coeff)
    {
        elco=&coeff;
        forc=nullptr;
        surfl=nullptr;
        cut_int=nullptr;
        stiffness_ratio=1e-6;
    }

    CFNLElasticityIntegrator(mfem::BasicElasticityCoefficient* coeff,
                             mfem::VectorCoefficient* vol_forc=nullptr)
    {
        elco=coeff;
        forc=vol_forc;
        surfl=nullptr;
        cut_int=nullptr;
        stiffness_ratio=1e-6;
    }

    void SetElasticityCoefficient(mfem::BasicElasticityCoefficient& coeff)
    {
        elco=&coeff;
    }

    void SetVolumetricForce(mfem::VectorCoefficient& ff)
    {
        forc=&ff;
    }

    void SetSurfaceLoad(mfem::VectorCoefficient& ss)
    {
        surfl=&ss;
    }

    void SetStiffnessRatio(double val)
    {
        stiffness_ratio=val;
    }

    /// Set level-set function and the element markers
    void SetLSF(ParGridFunction& lsf_, Array<int>& el_markers_,
                CutIntegrationRules& irs)
    {
        lsf=&lsf_;
        marks=&el_markers_;
        cut_int=&irs;
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
    VectorCoefficient* forc;  //volumetric load
    VectorCoefficient* surfl; //surface load
    Array<int>* marks; //markings of the elements
    ParGridFunction* lsf; //level set function
    double stiffness_ratio; //ratio between the stiffness in the level set and outside
    CutIntegrationRules* cut_int;
};



class CFElasticitySolver
{
public:
    CFElasticitySolver(mfem::ParMesh* mesh_, int vorder=1);

    ~CFElasticitySolver();

    /// Set level-set function and the element markers
    void SetLSF(ParGridFunction& lsf_, Array<int>& el_markers_,
                CutIntegrationRules& cint)
    {
        level_set_function=&lsf_;
        el_markers=&el_markers_;
        cut_int=&cint;
    }

    /// Set the ration of the void stiffness to the full stiffness in the
    /// cut elements
    void SetStiffnessRatio(double sr=1e-6)
    {
        stiffness_ratio=sr;
    }

    /// Set the Newton Solver
    void SetNewtonSolver(double rtol=1e-7, double atol=1e-12,int miter=10, int prt_level=1);

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

    /// Adds displacement BC in all directions.
    void AddDispBC(int id, mfem::VectorCoefficient& val);

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
        vforces.push_back(nullptr);
        sforces.push_back(nullptr);
    }

    /// Add material and force to the solver. The solver owns the data.
    void AddMaterial(BasicElasticityCoefficient* nmat,
                     VectorCoefficient* vol_force,
                     VectorCoefficient* surf_force){
        if(nf!=nullptr)
        {
            delete nf; nf=nullptr;
        }
        materials.push_back(nmat);
        vforces.push_back(vol_force);
        sforces.push_back(surf_force);
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

    // holds BC in vector coefficient form
    std::map<int, mfem::VectorCoefficient*> bccv;


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
    std::vector<mfem::VectorCoefficient*> vforces; //volumetric load for every material
    std::vector<mfem::VectorCoefficient*> sforces; //surface load for every material

    mfem::ParGridFunction* level_set_function;
    Array<int>*  el_markers;
    CutIntegrationRules* cut_int;
    double stiffness_ratio;

};

#endif


class QubicPenalty{
private:
    double eta;
    double a;
    double b;
    double c;
    // the maximum is 1.0 and is located at point eta/3.0
    // all penalties above eta are set to zero and
    // the gradient at eta and above eta is zero
public:
    QubicPenalty(double eta_){
        eta=eta_;
        a=27.0/(4.0*eta*eta*eta);
        b=-27.0/(2.0*eta*eta);
        c=27.0/(4.0*eta);
    }

    double Eval(double x){
        if(x<0.0){return 0.0;}
        if(x>eta){return 0.0;}
        return a*x*x*x+b*x*x+c*x;
    }

    double Grad(double x){
        if(x<0.0){return 0.0;}
        if(x>eta){return 0.0;}
        return 3*a*x*x+2*b*x+c;
    }

};

class VolPenalIntegrator:public NonlinearFormIntegrator
{
public:
    VolPenalIntegrator(Coefficient* lsf, double eta):qpntl(eta)
    {
        level_set=lsf;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun)
    {
        int ndof = el.GetDof();
        int ndim = Tr.GetSpaceDim();
        double val=0.0;
        double tal=0.0;
        int order= 2 * el.GetOrder() +Tr.OrderGrad(&el);

        const IntegrationRule * ir = nullptr;
        ir = &IntRules.Get(el.GetGeomType(), order);
        double w;
        double f;
        for(int i=0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w = Tr.Weight();
            w = ip.weight * w;
            f=level_set->Eval(Tr,ip);
            if(f>0.0){
                val=val+w;
            }
            tal=tal+w;
        }

        return qpntl.Eval(val/tal);

    }

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect)
    {
        int ndof = el.GetDof();
        int ndim = Tr.GetSpaceDim();
        double val=0.0;
        double tal=0.0;
        int order= 2 * el.GetOrder() +Tr.OrderGrad(&el);

        elvect.SetSize(ndof*ndim); elvect=0.0;

        DenseMatrix bmat(ndof,ndim); //gradients of the shape functions in isoparametric space
        DenseMatrix pmat(ndof,ndim);
        DenseMatrix gp; gp.UseExternalData(elvect.GetData(),ndof,ndim);

        const IntegrationRule * ir = nullptr;
        ir = &IntRules.Get(el.GetGeomType(), order);
        double w;
        double f;

        //evaluate the gradients with respect to nodal displacements
        for(int i=0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w = Tr.Weight();
            w = ip.weight * w;
            f=level_set->Eval(Tr,ip);
            if(f>0.0){
                val=val+w;

                el.CalcDShape(ip,bmat);
                Mult(bmat,Tr.AdjugateJacobian(),pmat);
                gp.Add(w,pmat);

            }
            tal=tal+w;
        }
        elvect*=(qpntl.Grad(val/tal)/tal);

    }

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat)
    {


    }


private:
    QubicPenalty qpntl;
    Coefficient* level_set;

};



}

#endif
