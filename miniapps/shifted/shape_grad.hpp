#ifndef SHAPE_GRAD_HPP
#define SHAPE_GRAD_HPP


#include "mfem.hpp"

#ifdef MFEM_USE_ALGOIM
namespace mfem{

/// Volumetric shape integrator - discrete
class DVolShapeIntegrator: public NonlinearFormIntegrator
{

public:
    DVolShapeIntegrator(Coefficient& coeff_, Array<int> &elem_markers_, int order_=-1)
    {
        coeff=&coeff_;
        elem_markers=&elem_markers_;
        lorder=order_;
    }

    virtual
    ~DVolShapeIntegrator()
    {

    }

    /// Perform the local action of the NonlinearFormIntegrator
    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       const Vector &elfun, Vector &elvect);

    /// Compute the local energy
    virtual double GetElementEnergy(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun);

private:
    Coefficient* coeff;
    Array<int>* elem_markers;
    int lorder;
};


/// Volumetric shape integrator
class VolShapeIntegrator: public NonlinearFormIntegrator
{

public:
    VolShapeIntegrator(Coefficient& coeff_, Array<int> &elem_markers_, int order_=-1)
    {
        coeff=&coeff_;
        elem_markers=&elem_markers_;
        lorder=order_;
    }

    virtual
    ~VolShapeIntegrator()
    {

    }

    /// Perform the local action of the NonlinearFormIntegrator
    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       const Vector &elfun, Vector &elvect);

    /// Compute the local energy
    virtual double GetElementEnergy(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun);

private:
    Coefficient* coeff;
    Array<int>* elem_markers;
    int lorder;
};

class VolObjectiveCutA
{
public:
    VolObjectiveCutA():volw(1.0)
    {
        volc=&volw;
    }

    void SetWeight(Coefficient* coeff){
        volc=coeff;
    }

    void SetCutIntegrationRules(Array<int>& el_markers_)
    {
        marks=&el_markers_;
    }


    double Eval(ParGridFunction& lsf){
        VolShapeIntegrator* itg=new VolShapeIntegrator(*volc,*marks);
        ParNonlinearForm* nf=new ParNonlinearForm(lsf.ParFESpace());
        nf->AddDomainIntegrator(itg);
        double vol=nf->GetEnergy(lsf.GetTrueVector());
        delete nf;
        return vol;
    }

    void Grad(ParGridFunction& lsf, Vector& grad){
        VolShapeIntegrator* itg=new VolShapeIntegrator(*volc,*marks);
        ParNonlinearForm* nf=new ParNonlinearForm(lsf.ParFESpace());
        nf->AddDomainIntegrator(itg);
        grad.SetSize(lsf.GetTrueVector().Size());
        nf->Mult(lsf.GetTrueVector(),grad);
        delete nf;
    }


private:
    ConstantCoefficient volw; //volume weight
    Coefficient* volc; //points either to volw or to user supplied coefficient

    Array<int>* marks;

};



/// Surface shape integrator
class SurfShapeIntegrator: public NonlinearFormIntegrator
{
public:
    SurfShapeIntegrator(Coefficient& coeff_, Array<int> &elem_markers_, int order_=-1)
    {
        coeff=&coeff_;
        elem_markers=&elem_markers_;
        lorder=order_;
        gradco=nullptr;
    }

    SurfShapeIntegrator(Coefficient& coeff_, VectorCoefficient& gradco_, Array<int> &elem_markers_, int order_=-1)
    {
        coeff=&coeff_;
        elem_markers=&elem_markers_;
        lorder=order_;
        gradco=&gradco_;
    }

    virtual
    ~SurfShapeIntegrator(){}

    /// Perform the local action of the NonlinearFormIntegrator
    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       const Vector &elfun, Vector &elvect);

    /// Compute the local energy
    virtual double GetElementEnergy(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun);

private:
    Coefficient* coeff;
    VectorCoefficient* gradco;
    Array<int>* elem_markers;
    int lorder;
};


/// Evaluates div(n/|n|) for a level set function
class MeanCurvImplicitSurf:public Coefficient
{
public:
    MeanCurvImplicitSurf(GridFunction & gf_)
    {
        gf=&gf_;
    }

    virtual
    double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        T.SetIntPoint(&ip);

        const FiniteElement* el= gf->FESpace()->GetFE(T.ElementNo);
        int ndim = el->GetDim();
        int ndof = el->GetDof();

        Vector elfun(ndof);
        Array<int> dofs;
        gf->FESpace()->GetElementDofs(T.ElementNo, dofs);
        gf->FESpace()->DofsToVDofs(dofs);
        gf->GetSubVector(dofs,elfun);

        DenseMatrix proj(ndim*ndof,ndof);
        el->ProjectGrad(*el,T,proj);


        Vector gradv(ndim*ndof);
        proj.Mult(elfun,gradv);
        Vector gradx; gradx.SetDataAndSize(gradv.GetData()+0*ndof,ndof);
        Vector grady; grady.SetDataAndSize(gradv.GetData()+1*ndof,ndof);
        Vector gradz;
        if(ndim==3){gradz.SetDataAndSize(gradv.GetData()+2*ndof,ndof);}


        Vector dgradx(ndim*ndof); proj.Mult(gradx,dgradx);
        Vector dgrady(ndim*ndof); proj.Mult(grady,dgrady);
        Vector dgradz(ndim*ndof);
        if(ndim==3){ proj.Mult(gradz,dgradz);}

        Vector sh(ndof);
        el->CalcPhysShape(T,sh);

        if(ndim==2){
            Vector tv;

            tv.SetDataAndSize(gradv.GetData()+0*ndof,ndof);
            double tx=sh*tv;
            tv.SetDataAndSize(gradv.GetData()+1*ndof,ndof);
            double ty=sh*tv;

            tv.SetDataAndSize(dgradx.GetData()+0*ndof,ndof);
            double txx=sh*tv;
            tv.SetDataAndSize(dgradx.GetData()+1*ndof,ndof);
            double txy=sh*tv;

            tv.SetDataAndSize(dgrady.GetData()+1*ndof,ndof);
            double tyy=sh*tv;


            double nr=sqrt(tx*tx+ty*ty);
            double rez=tx*tx*(tyy)+ty*ty*(txx);
            rez=rez-2.0*tx*ty*txy;
            rez=rez/(nr*nr*nr);
            return rez;

        }else{
            //ndim==3
            Vector tv;

            tv.SetDataAndSize(gradv.GetData()+0*ndof,ndof);
            double tx=sh*tv;
            tv.SetDataAndSize(gradv.GetData()+1*ndof,ndof);
            double ty=sh*tv;
            tv.SetDataAndSize(gradv.GetData()+2*ndof,ndof);
            double tz=sh*tv;

            tv.SetDataAndSize(dgradx.GetData()+0*ndof,ndof);
            double txx=sh*tv;
            tv.SetDataAndSize(dgradx.GetData()+1*ndof,ndof);
            double txy=sh*tv;
            tv.SetDataAndSize(dgradx.GetData()+2*ndof,ndof);
            double txz=sh*tv;

            tv.SetDataAndSize(dgrady.GetData()+1*ndof,ndof);
            double tyy=sh*tv;
            tv.SetDataAndSize(dgrady.GetData()+2*ndof,ndof);
            double tyz=sh*tv;

            tv.SetDataAndSize(dgradz.GetData()+2*ndof,ndof);
            double tzz=sh*tv;

            double nr=sqrt(tx*tx+ty*ty+tz*tz);
            double rez=tx*tx*(tyy+tzz)+ty*ty*(txx+tzz)+tz*tz*(txx+tyy);
            rez=rez-2.0*tx*ty*txy-2.0*tx*tz*txz-2.0*ty*tz*tyz;
            rez=rez/(nr*nr*nr);
            return rez;
        }
    }

private:
    GridFunction *gf;
};

/// Evaluates the mean curvature div(n/|n|) for all integration points
/// elfun - nodal values of a level-set function
void MeanCurvImplicitFunction(const Vector& elfun,
                    const FiniteElement & el,
                    ElementTransformation &T,
                    const IntegrationRule& ir,
                    Vector& H);

/// Evaluates the mean curvature div(n/|n|) for all integration points
/// elfun - nodal values of a level-set function
void MeanCurvImplicitFunction(int elno,
                              GridFunction& gf,
                              const IntegrationRule& ir,
                              Vector& H);

}

#endif

#endif
