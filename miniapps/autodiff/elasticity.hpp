#ifndef ELASTICITY_SOLVER_HPP
#define ELASTICITY_SOLVER_HPP

#include "mfem.hpp"
#include "admfem.hpp"


namespace mfem {


namespace elast {

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

template<class TMat, class TVec>
void Convert2DVoigtStress(TMat& mat,TVec& vec)
{
    vec[0]=mat(0,0);
    vec[1]=mat(1,1);
    vec[2]=mat(0,1);
}

template<class TMat, class TVec>
void Convert2DVoigtStrain(TMat& mat,TVec& vec)
{
    vec[0]=mat(0,0);
    vec[1]=mat(1,1);
    vec[2]=2.0*mat(0,1);
}

template<class TMat, class TVec>
void Convert3DVoigtStress(TMat& mat,TVec& vec)
{
    vec[0]=mat(0,0);
    vec[1]=mat(1,1);
    vec[2]=mat(2,2);
    vec[3]=mat(1,2);
    vec[4]=mat(0,2);
    vec[5]=mat(0,1);
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
void ConvertVoightStress2D(TVec& vec, TMat& mat)
{
    mat(0,0)=vec[0]; mat(0,1)=vec[2];
    mat(1,0)=vec[2]; mat(1,1)=vec[1];
}

template<class TMat, class TVec>
void ConvertVoightStrain2D(TVec& vec, TMat& mat)
{
    mat(0,0)=vec[0];     mat(0,1)=vec[2]*0.5;
    mat(1,0)=vec[2]*0.5; mat(1,1)=vec[1];
}

template<class TMat, class TVec>
void ConvertVoightStress3D(TVec& vec, TMat& mat)
{
    mat(0,0)=vec[0];    mat(0,1)=vec[5];    mat(0,2)=vec[4];
    mat(1,0)=vec[5];    mat(1,1)=vec[1];    mat(1,2)=vec[3];
    mat(2,0)=vec[4];    mat(2,1)=vec[3];    mat(2,2)=vec[2];
}

template<class TMat, class TVec>
void ConvertVoightStrain3D(TVec& vec, TMat& mat)
{
    mat(0,0)=vec[0];        mat(0,1)=vec[5]*0.5;    mat(0,2)=vec[4]*0.5;
    mat(1,0)=vec[5]*0.5;    mat(1,1)=vec[1];        mat(1,2)=vec[3]*0.5;
    mat(2,0)=vec[4]*0.5;    mat(2,1)=vec[3]*0.5;    mat(2,2)=vec[2];
}

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

/// LC - lower diagonal matrix columnwise storage
template<class TMat,class TVec>
void StiffnessTensor3DVec2Mat(TVec& LC, TMat& CC)
{
    int count=0;
    for(int i=0;i<6;i++){
        for(int j=i;j<6;j++){
            CC(i,j)=LC[count];
            CC(j,i)=LC[count];
            count++;
        }
    }
}

template<class TMat>
void GreenStrain(TMat& gradu, TMat& ee)
{

    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            ee(i,j)=gradu(i,j)+gradu(j,i);
            for(int k=0;k<3;k++){
                ee(i,j)=ee(i,j)+gradu(i,k)*gradu(k,j);
            }
            ee(i,j)=ee(i,j)*0.5;
        }
    }
}

template<class TMat>
void DeformationGrad(TMat& gradu, TMat& F)
{
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            F(i,j)=gradu(j,i);
            if(i==j){
                F(i,j)=F(i,j)+1.0;
            }
        }
    }
}

template<class TMat, class TFloat>
void IsoMatGradUStress(TFloat E, TFloat nu, TMat& gradu, TMat& sigma)
{
    TFloat lambda=E*nu/((1.0+nu)*(1.0-2.0*nu));
    TFloat mu=0.5*E/(1.0+nu);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            sigma(i,j)=mu*(gradu(i,j)+gradu(j,i));
            if(i==j){
                sigma(i,j)=lambda*gradu(i,i);
            }
        }
    }
}


}// end namespace elast


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

private:
    mfem::ConstantCoefficient lE;
    mfem::ConstantCoefficient lnu;

    mfem::Coefficient* E;
    mfem::Coefficient* nu;

    mfem::GridFunction* disp;

    mfem::DenseMatrix tmpm;
    mfem::DenseMatrix tmpg;

    void Eval(double EE,double nnu,double* CC);
};

class NLElasticityIntegrator:public NonlinearFormIntegrator
{
public:
    NLElasticityIntegrator()
    {
        elco=nullptr;
    }

    void SetElasticityCoefficient(mfem::BasicElasticityCoefficient& coeff)
    {
        elco=&coeff;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
    {

    }

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun, Vector &elvect)
    {

    }

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun, DenseMatrix &elmat)
    {

    }


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

    /// Return adj*d(residual(sol))/d(design). The dimension
    /// of the vector grad is the save as the dimension of the
    /// true design vector.
    void GradD(mfem::Vector& grad);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, double val);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, mfem::Coefficient& val);

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



private:
    mfem::ParMesh* pmesh;

    mfem::ParFiniteElementSpace* vfes;
    mfem::FiniteElementCollection* vfec;

    mfem::ParNonlinearForm* nf;

    //solution true vector
    mfem::Vector sol;
    //adjoint true vector
    mfem::Vector adj;
    //RHS
    mfem::Vector rhs;

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

};

}

#endif
