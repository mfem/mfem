#include "shape_grad.hpp"
#include "marking.hpp"


#ifdef MFEM_USE_ALGOIM
#include "integ_algoim.hpp"
#endif

namespace mfem{

#ifdef MFEM_USE_ALGOIM

void DVolShapeIntegrator::AssembleElementVector(const FiniteElement &el,
                                          ElementTransformation &Tr,
                                          const Vector &elfun, Vector &elvect)
{

    if((*elem_markers)[Tr.ElementNo]==ElementMarker::SBElementType::CUT)
    {
        int ndof = el.GetDof();
        int ndim = Tr.GetSpaceDim();

        elvect.SetSize(ndof); elvect=0.0;

        DenseMatrix bmat(ndof,ndim); //gradients of the shape functions in isoparametric space
        Vector tnormal(ndim); //normal to the level set in true space
        Vector shf(ndof);
        DenseMatrix gradn(ndof,ndim); //nodal gradient values

        DenseMatrix proj(ndim*ndof,ndof);
        el.ProjectGrad(el,Tr,proj);
        DenseMatrix ngrad[ndim];
        for(int i=0;i<ndim;i++){
            ngrad[i].SetSize(ndof,ndof);
            proj.GetSubMatrix(i*ndof,(i+1)*ndof,0,ndof,ngrad[i]);
        }

        {
            Vector gradv; gradv.SetDataAndSize(gradn.GetData(),ndim*ndof);
            proj.Mult(elfun,gradv);
        }

        int order;
        if(lorder>0){
            order=lorder;
        }else{
            order = 2 * el.GetOrder() +Tr.OrderGrad(&el);
        }

        order=10;

        AlgoimIntegrationRule air(order, el, Tr, elfun);
        const IntegrationRule *ir = air.GetVolumeIntegrationRule();

        double w;
        double f;
        DenseMatrix pmat(ndof,ndim);

        //evaluate the gradients with respect to nodal displacements
        /*
        DenseMatrix gp(ndof,ndim); gp=0.0;
        {
            for (int j = 0; j < ir->GetNPoints(); j++)
            {
                const IntegrationPoint &ip = ir->IntPoint(j);
                Tr.SetIntPoint(&ip);
                el.CalcDShape(ip,bmat);
                Mult(bmat,Tr.AdjugateJacobian(),pmat);
                w = ip.weight ;
                f = coeff->Eval(Tr,ip);
                gp.Add(w*f,pmat);
            }
            std::cout<<"GP="<<std::endl;
            gp.PrintMatlab(std::cout);
        }*/

        Vector gradf(ndim); gradf=0.0;
        Vector lap(ndof);
        Vector tv(ndof);
        Vector gv;
        for (int j = 0; j < ir->GetNPoints(); j++)
        {
           const IntegrationPoint &ip = ir->IntPoint(j);
           Tr.SetIntPoint(&ip);
           el.CalcPhysDShape(Tr,bmat);

           f = coeff->Eval(Tr,ip);
           w = ip.weight*Tr.Weight();


           for(int d=0;d<ndim;d++){
               gv.SetDataAndSize(bmat.GetData()+d*ndof,ndof);
               ngrad[d].MultTranspose(gv,tv);
               elvect.Add(w*f,tv);
           }

           bmat.Mult(gradf,tv);
           elvect.Add(w,tv);
        }


        elvect.Print(std::cout);

    }
    else
    {
        elvect.SetSize(elfun.Size());
        elvect=0.0;
    }


}

double DVolShapeIntegrator::GetElementEnergy(const FiniteElement &el,
                                            ElementTransformation &Tr,
                                            const Vector &elfun)
{
    if((*elem_markers)[Tr.ElementNo]==ElementMarker::SBElementType::INSIDE){
        double val=0.0;
        const IntegrationRule * ir = nullptr;
        int order;
        if(lorder>0){
            order=lorder;
        }else{
            order = 2 * el.GetOrder() +Tr.OrderGrad(&el);
        }
        ir = &IntRules.Get(el.GetGeomType(), order);
        double w;
        double f;
        for(int i=0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w = Tr.Weight();
            w = ip.weight * w;
            f = coeff->Eval(Tr,ip);
            val = val + f*w;

        }

        return val;
    }else
    if((*elem_markers)[Tr.ElementNo]==ElementMarker::SBElementType::OUTSIDE)
    {
        return 0.0;
    }

    // integrate cut element
    double val = 0.0;
    int order;
    if(lorder>0){
        order=lorder;
    }else{
        order = 2 * el.GetOrder() +Tr.OrderGrad(&el);
    }
    AlgoimIntegrationRule air(order, el, Tr, elfun);
    const IntegrationRule * ir = air.GetVolumeIntegrationRule();
    for (int j = 0; j < ir->GetNPoints(); j++)
    {
       const IntegrationPoint &ip = ir->IntPoint(j);
       Tr.SetIntPoint(&ip);
       val += ip.weight * Tr.Weight();
    }

    return val;
}

void VolShapeIntegrator::AssembleElementVector(const FiniteElement &el,
                                          ElementTransformation &Tr,
                                          const Vector &elfun, Vector &elvect)
{

    if((*elem_markers)[Tr.ElementNo]==ElementMarker::SBElementType::CUT)
    {
        int ndof = el.GetDof();
        int ndim = Tr.GetSpaceDim();

        elvect.SetSize(ndof); elvect=0.0;

        DenseMatrix bmat(ndof,ndim); //gradients of the shape functions in isoparametric space
        Vector inormal(ndim); //normal to the level set in isoparametric space
        Vector shf(ndof);

        int order;
        if(lorder>0){
            order=lorder;
        }else{
            order = 2 * el.GetOrder() +Tr.OrderGrad(&el);
        }

        AlgoimIntegrationRule air(order, el, Tr, elfun);
        const IntegrationRule * ir = air.GetSurfaceIntegrationRule();

        double w;
        double f;

        for (int j = 0; j < ir->GetNPoints(); j++)
        {
           const IntegrationPoint &ip = ir->IntPoint(j);
           Tr.SetIntPoint(&ip);
           el.CalcDShape(ip,bmat);
           //compute the normal to the LS in isoparametric space
           bmat.MultTranspose(elfun,inormal);
           if(ndim==2) { w = ip.weight * sqrt(Tr.Weight());} /// inormal.Norml2();
           else { w= ip.weight * pow(Tr.Weight(), 2.0/3.0);}
           el.CalcPhysShape(Tr,shf);
           f = coeff->Eval(Tr,ip);
           elvect.Add(w * f , shf);
        }
    }
    else
    {
        elvect.SetSize(elfun.Size());
        elvect=0.0;
    }


}


double VolShapeIntegrator::GetElementEnergy(const FiniteElement &el,
                                            ElementTransformation &Tr,
                                            const Vector &elfun)
{
    if((*elem_markers)[Tr.ElementNo]==ElementMarker::SBElementType::INSIDE){
        double val=0.0;
        const IntegrationRule * ir = nullptr;
        int order;
        if(lorder>0){
            order=lorder;
        }else{
            order = 2 * el.GetOrder() +Tr.OrderGrad(&el);
        }
        ir = &IntRules.Get(el.GetGeomType(), order);
        double w;
        double f;
        for(int i=0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w = Tr.Weight();
            w = ip.weight * w;
            f = coeff->Eval(Tr,ip);
            val = val + f*w;

        }

        return val;
    }else
    if((*elem_markers)[Tr.ElementNo]==ElementMarker::SBElementType::OUTSIDE)
    {
        return 0.0;
    }

    // integrate cut element
    double val = 0.0;
    int order;
    if(lorder>0){
        order=lorder;
    }else{
        order = 2 * el.GetOrder() +Tr.OrderGrad(&el);
    }
    AlgoimIntegrationRule air(order, el, Tr, elfun);
    const IntegrationRule * ir = air.GetVolumeIntegrationRule();
    for (int j = 0; j < ir->GetNPoints(); j++)
    {
       const IntegrationPoint &ip = ir->IntPoint(j);
       Tr.SetIntPoint(&ip);
       val += ip.weight * Tr.Weight();
    }

    return val;
}

double SurfShapeIntegrator::GetElementEnergy(const FiniteElement &el,
                                            ElementTransformation &Tr,
                                            const Vector &elfun)
{
    if((*elem_markers)[Tr.ElementNo]!=ElementMarker::SBElementType::CUT)
    {
        return 0.0;
    }

    // integrate cut element
    double val = 0.0;
    int order;
    if(lorder>0){
        order=lorder;
    }else{
        order = 2 * el.GetOrder() +Tr.OrderGrad(&el);
    }

    AlgoimIntegrationRule air(order, el, Tr, elfun);
    const IntegrationRule * ir = air.GetSurfaceIntegrationRule();

    int ndof = el.GetDof();
    int ndim = Tr.GetSpaceDim();

    DenseMatrix bmat(ndof,ndim); //gradients of the shape functions in isoparametric space
    DenseMatrix pmat(ndof,ndim); //gradients of the shape functions in physical space
    Vector inormal(ndim); //normal to the level set in isoparametric space
    Vector tnormal(ndim);

    double w;
    double f;

    for (int j = 0; j < ir->GetNPoints(); j++)
    {
       const IntegrationPoint &ip = ir->IntPoint(j);
       Tr.SetIntPoint(&ip);
       el.CalcDShape(ip,bmat);
       Mult(bmat, Tr.AdjugateJacobian(), pmat);
       //compute the normal to the LS in isoparametric space
       bmat.MultTranspose(elfun,inormal);
       //compute the normal to the LS in physical space
       pmat.MultTranspose(elfun,tnormal);

       //std::cout<<" sca1="<< tnormal.Norml2()/ inormal.Norml2()<< " (adjJ)="<<Tr.AdjugateJacobian().Det() <<" |J|=" <<Tr.Weight()<<" det(J)="<< Tr.Jacobian().Det()<<std::endl;

       w = ip.weight * tnormal.Norml2()/ inormal.Norml2();
       f = coeff->Eval(Tr,ip);

       val=val + w*f;
    }

    return val;
}

double SurfMeanCurv3D(Vector& sh, Vector& gradv, Vector& dgradx, Vector& dgrady, Vector& dgradz)
{
    int ndof=sh.Size();
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

double SurfMeanCurv2D(Vector& sh, Vector& gradv, Vector& dgradx, Vector& dgrady, Vector& dgradz)
{
    int ndof=sh.Size();
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
}

void SurfShapeIntegrator::AssembleElementVector(const FiniteElement &el,
                                          ElementTransformation &Tr,
                                          const Vector &elfun, Vector &elvect)
{
    elvect.SetSize(elfun.Size()); elvect=0.0;
    if((*elem_markers)[Tr.ElementNo]!=ElementMarker::SBElementType::CUT)
    {
        return;
    }

    int ndof = el.GetDof();
    int ndim = Tr.GetSpaceDim();

    DenseMatrix proj(ndim*ndof,ndof);
    el.ProjectGrad(el,Tr,proj);

    Vector gradv(ndim*ndof);
    proj.Mult(elfun,gradv);

    Vector gradx; gradx.SetDataAndSize(gradv.GetData()+0*ndof,ndof);
    Vector grady; grady.SetDataAndSize(gradv.GetData()+1*ndof,ndof);
    Vector gradz;
    if(ndim==3){
        gradz.SetDataAndSize(gradv.GetData()+2*ndof,ndof);
    }else{
        gradz.SetSize(ndof); gradz=0.0;
    }

    Vector dgradx(ndim*ndof); proj.Mult(gradx,dgradx);
    Vector dgrady(ndim*ndof); proj.Mult(grady,dgrady);
    Vector dgradz(ndim*ndof); proj.Mult(gradz,dgradz);

    DenseMatrix bmat(ndof,ndim); //gradients of the shape functions in isoparametric space
    Vector inormal(ndim); //normal to the level set in isoparametric space
    Vector shf(ndof);

    int order;
    if(lorder>0){
        order=lorder;
    }else{
        order = 2 * el.GetOrder() +Tr.OrderGrad(&el);
    }

    AlgoimIntegrationRule air(order, el, Tr, elfun);
    const IntegrationRule * ir = air.GetSurfaceIntegrationRule();

    double w;
    double f;
    Vector H;
    Vector nn(ndim);
    Vector gr(ndim); gr=0.0;

    //evaluate the curvature at the integrations points
    MeanCurvImplicitFunction(elfun,el,Tr,*ir,H);

    for (int j = 0; j < ir->GetNPoints(); j++)
    {
       const IntegrationPoint &ip = ir->IntPoint(j);
       Tr.SetIntPoint(&ip);
       el.CalcDShape(ip,bmat);
       //compute the normal to the LS in isoparametric space
       bmat.MultTranspose(elfun,inormal);
       w = ip.weight * Tr.Weight() / inormal.Norml2();

       el.CalcPhysShape(Tr,shf);

       double H1;
       //Compute the mean curvature H
       if(ndim==3){
           H1=SurfMeanCurv3D(shf,gradv,dgradx,dgrady,dgradz);
       }else{
           H1=SurfMeanCurv2D(shf,gradv,dgradx,dgrady,dgradz);
       }

       nn[0]=shf*gradx;
       nn[1]=shf*grady;
       if(ndim==3){ nn[2]=shf*gradz;}

       std::cout<<"nn="<<nn.Norml2()<<" H="<<H[j]<<" H1="<<H1<<std::endl;

       f = coeff->Eval(Tr,ip);

       if(gradco!=nullptr){
           gradco->Eval(gr,Tr,ip);
       }

       gr.Print(std::cout); std::cout<<" dot="<<nn*gr/nn.Norml2()<<std::endl;

       //elvect.Add(w*(-H*f),shf);
       elvect.Add(w*(-H[j]*f-nn*gr/nn.Norml2()),shf);



    }
}


/// Evaluates the mean curvature div(n/|n|) for all integration points
/// elfun - nodal values of a level-set function
void MeanCurvImplicitFunction(const Vector& elfun,
                    const FiniteElement & el,
                    ElementTransformation &T,
                    const IntegrationRule& ir,
                    Vector& H)
{
    H.SetSize(ir.GetNPoints()); H=0.0;

    int ndim = el.GetDim();
    int ndof = el.GetDof();

    DenseMatrix proj(ndim*ndof,ndof);
    el.ProjectGrad(el,T,proj);

    Vector gradv(ndim*ndof);
    proj.Mult(elfun,gradv);

    Vector grad[ndim];
    for(int i=0;i<ndim;i++){
        grad[i].SetDataAndSize(gradv.GetData()+i*ndof,ndof);
    }

    Vector dgrad[ndim];
    for(int i=0;i<ndim;i++){
        dgrad[i].SetSize(ndim*ndof);
        proj.Mult(grad[i],dgrad[i]);
    }

    Vector sh(ndof);
    Vector tv;

    if(ndim==2){
    for (int j = 0; j < ir.GetNPoints(); j++)
    {
        const IntegrationPoint &ip = ir.IntPoint(j);
        T.SetIntPoint(&ip);
        el.CalcPhysShape(T,sh);

        tv.SetDataAndSize(gradv.GetData()+0*ndof,ndof);
        double tx=sh*tv;
        tv.SetDataAndSize(gradv.GetData()+1*ndof,ndof);
        double ty=sh*tv;

        tv.SetDataAndSize(dgrad[0].GetData()+0*ndof,ndof);
        double txx=sh*tv;
        tv.SetDataAndSize(dgrad[0].GetData()+1*ndof,ndof);
        double txy=sh*tv;

        tv.SetDataAndSize(dgrad[1].GetData()+1*ndof,ndof);
        double tyy=sh*tv;

        double nr=sqrt(tx*tx+ty*ty);
        double rez=tx*tx*(tyy)+ty*ty*(txx);
        rez=rez-2.0*tx*ty*txy;
        rez=rez/(nr*nr*nr);
        H[j]=rez;

    }}else{//ndim=3
    for (int j = 0; j < ir.GetNPoints(); j++){
        const IntegrationPoint &ip = ir.IntPoint(j);
        T.SetIntPoint(&ip);
        el.CalcPhysShape(T,sh);

        tv.SetDataAndSize(gradv.GetData()+0*ndof,ndof);
        double tx=sh*tv;
        tv.SetDataAndSize(gradv.GetData()+1*ndof,ndof);
        double ty=sh*tv;
        tv.SetDataAndSize(gradv.GetData()+2*ndof,ndof);
        double tz=sh*tv;

        tv.SetDataAndSize(dgrad[0].GetData()+0*ndof,ndof);
        double txx=sh*tv;
        tv.SetDataAndSize(dgrad[0].GetData()+1*ndof,ndof);
        double txy=sh*tv;
        tv.SetDataAndSize(dgrad[0].GetData()+2*ndof,ndof);
        double txz=sh*tv;

        tv.SetDataAndSize(dgrad[1].GetData()+1*ndof,ndof);
        double tyy=sh*tv;
        tv.SetDataAndSize(dgrad[1].GetData()+2*ndof,ndof);
        double tyz=sh*tv;

        tv.SetDataAndSize(dgrad[2].GetData()+2*ndof,ndof);
        double tzz=sh*tv;

        double nr=sqrt(tx*tx+ty*ty+tz*tz);
        double rez=tx*tx*(tyy+tzz)+ty*ty*(txx+tzz)+tz*tz*(txx+tyy);
        rez=rez-2.0*tx*ty*txy-2.0*tx*tz*txz-2.0*ty*tz*tyz;
        rez=rez/(nr*nr*nr);
        H[j]=rez;
    }}
}

/// Evaluates the mean curvature div(n/|n|) for all integration points
/// elfun - nodal values of a level-set function
void MeanCurvImplicitFunction(int elno,
                              GridFunction& gf,
                              const IntegrationRule& ir,
                              Vector& H)
{
    const FiniteElement* el= gf.FESpace()->GetFE(elno);
    int ndof = el->GetDof();

    Vector elfun(ndof);
    Array<int> dofs;
    gf.FESpace()->GetElementDofs(elno, dofs);
    gf.FESpace()->DofsToVDofs(dofs);
    gf.GetSubVector(dofs,elfun);

    ElementTransformation* T=gf.FESpace()->GetElementTransformation(elno);
    MeanCurvImplicitFunction(elfun,*el,*T,ir,H);
}



#endif

}
