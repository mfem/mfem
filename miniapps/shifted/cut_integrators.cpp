#include "cut_integrators.hpp"

namespace mfem {

void CutVolLagrangeIntegrator::AssembleElementVector(const FiniteElement &el,
                           ElementTransformation &tr,
                           const Vector &elfun, Vector &elvect)
{
    elvect.SetSize(elfun.Size());
    elvect=0.0;

    if(cint==nullptr){return;}

    int elmark=cint->GetElementMarker(tr.ElementNo);

    //only cut elements will have gradients
    if(elmark!=ElementMarker::SBElementType::OUTSIDE){
        int ndof = el.GetDof();
        int ndim = tr.GetSpaceDim();

        DenseMatrix bmat(ndof,ndim); //gradients of the shape functions in isoparametric space
        DenseMatrix pmat(ndof,ndim);
        Vector sh(ndof);
        double f=1.0;

        DenseMatrix gp; gp.Reset(elvect.GetData(),ndof,ndim);


        const IntegrationRule* ir;
        if(elmark==ElementMarker::SBElementType::CUT){
            ir=cint->GetVolIntegrationRule(tr.ElementNo);
        }else{
            ir=&IntRules.Get(el.GetGeomType(),int_order);
        }

        for (int j = 0; j < ir->GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir->IntPoint(j);
            tr.SetIntPoint(&ip);
            el.CalcDShape(ip,bmat);
            Mult(bmat,tr.AdjugateJacobian(),pmat);
            if(coeff!=nullptr){f = coeff->Eval(tr,ip);}
            gp.Add(ip.weight*f,pmat);

        }
    }
}


/// Compute the local energy, i.e., the volume
double CutVolLagrangeIntegrator::GetElementEnergy(const FiniteElement &el,
                            ElementTransformation &tr,
                            const Vector &elfun)
{
    double rez=0.0;
    if(cint==nullptr){return rez;}

    int elmark=cint->GetElementMarker(tr.ElementNo);
    if(elmark!=ElementMarker::SBElementType::OUTSIDE){

        const IntegrationRule* ir;
        if(elmark==ElementMarker::SBElementType::CUT){
            ir=cint->GetVolIntegrationRule(tr.ElementNo);
        }else{
            ir=&IntRules.Get(el.GetGeomType(),int_order);
        }

        double f=1.0;
        double w;
        for (int j = 0; j < ir->GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir->IntPoint(j);
            tr.SetIntPoint(&ip);
            w = ip.weight ;
            if(coeff!=nullptr){f = coeff->Eval(tr,ip);}

            rez=rez+w*f*tr.Weight();

            //std::cout<<"Tr.Weight="<<tr.Weight()<<std::endl;
        }

    }
    return rez;
}


void VolGhostPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe1,
                                                const FiniteElement &fe2,
                                                FaceElementTransformations &Tr,
                                                DenseMatrix &elmat)
{
    const int ndim=Tr.GetSpaceDim();
    const int vdim=fe1.GetVDim();
    int elem2 = Tr.Elem2No;
    if(elem2<0){
        elmat.SetSize(fe1.GetDof()*vdim);
        elmat=0.0;
        return;
    }

    const int ndof1 = fe1.GetDof();
    const int ndof2 = fe2.GetDof();
    const int ndofe = ndof1+ndof2;

    elmat.SetSize(ndofe*vdim);
    elmat=0.0;

    int order=std::max(fe1.GetOrder(), fe2.GetOrder());

    int ndofg;
    if(ndim==1){ndofg=order+1;}
    else if(ndim==2){ ndofg=(order+1)*(order+2)/2;}
    else if(ndim==3){ ndofg=(order+1)*(order+2)*(order+3)/6;}

    Vector sh1(ndof1);
    Vector sh2(ndof2);
    Vector shg(ndofg);

    Vector xx(ndim);

    DenseMatrix Mge(ndofg,ndofe); Mge=0.0;
    DenseMatrix Mgg(ndofg,ndofg); Mgg=0.0;
    DenseMatrix Mee(ndofe,ndofe); Mee=0.0;

    ElementTransformation &Tr1 = Tr.GetElement1Transformation();
    ElementTransformation &Tr2 = Tr.GetElement2Transformation();

    //elements' volumes
    double vol1=0.0;
    double vol2=0.0;

    const IntegrationRule* ir;

    //element 1
    double w;
    ir=&IntRules.Get(Tr1.GetGeometryType(), 2*order+2);

    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr1.SetIntPoint(&ip);
        Tr1.Transform(ip,xx);
        fe1.CalcPhysShape(Tr1,sh1);
        Shape(xx,ndim,order,shg);

        w = Tr1.Weight();
        w = ip.weight * w;
        vol1=vol1+w;
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg(i,j)=Mgg(i,j)+shg(i)*shg(j)*w;
                Mgg(j,i)=Mgg(j,i)+shg(i)*shg(j)*w;
            }
            Mgg(i,i)=Mgg(i,i)+shg(i)*shg(i)*w;
        }

        for(int i=0;i<ndof1;i++){
            for(int j=0;j<i;j++){
                Mee(i,j)=Mee(i,j)+sh1(i)*sh1(j)*w;
                Mee(j,i)=Mee(j,i)+sh1(i)*sh1(j)*w;
            }
            Mee(i,i)=Mee(i,i)+sh1(i)*sh1(i)*w;
        }

        for(int i=0;i<ndof1;i++){
            for(int j=0;j<ndofg;j++){
                Mge(j,i)=Mge(j,i)+shg(j)*sh1(i)*w;
            }}
    }

    //element 2
    ir=&IntRules.Get(Tr2.GetGeometryType(), 2*order+2);
    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr2.SetIntPoint(&ip);
        Tr2.Transform(ip,xx);

        fe2.CalcPhysShape(Tr2,sh2);
        Shape(xx,ndim,order,shg);

        w = Tr2.Weight();
        w = ip.weight * w;
        vol2=vol2+w;
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg(i,j)=Mgg(i,j)+shg(i)*shg(j)*w;
                Mgg(j,i)=Mgg(j,i)+shg(i)*shg(j)*w;
            }
            Mgg(i,i)=Mgg(i,i)+shg(i)*shg(i)*w;
        }

        for(int i=0;i<ndof2;i++){
            for(int j=0;j<i;j++){
                Mee(ndof1+i,ndof1+j)=Mee(ndof1+i,ndof1+j)+sh2(i)*sh2(j)*w;
                Mee(ndof1+j,ndof1+i)=Mee(ndof1+j,ndof1+i)+sh2(i)*sh2(j)*w;
            }
            Mee(ndof1+i,ndof1+i)=Mee(ndof1+i,ndof1+i)+sh2(i)*sh2(i)*w;
        }

        for(int i=0;i<ndof2;i++){
            for(int j=0;j<ndofg;j++){
                Mge(j,ndof1+i)=Mge(j,ndof1+i)+shg(j)*sh2(i)*w;
            }}
    }


    DenseMatrixInverse Mii(Mgg);
    DenseMatrix Mre(ndofg,ndofe);
    DenseMatrix Mff(ndofe,ndofe);
    Mii.Mult(Mge,Mre);
    MultAtB(Mge,Mre,Mff);


    double hh;
    if(ndim==1){
        hh=(vol1+vol2)/2.0;
    }else if(ndim==2){
        hh=std::sqrt((vol1+vol2)/2.0);
    }else{
        hh=std::cbrt((vol1+vol2)/2.0);
    }

    double tv;

    hh=penal/(hh*hh);

    for(int i=0;i<ndofe;i++){
        for(int j=0;j<ndofe;j++){
            tv=hh*(Mee(i,j)+Mee(j,i)-Mff(i,j)-Mff(j,i))/(2.0);
            for(int d=0;d<vdim;d++){
                elmat(i+d*ndofe,j+d*ndofe)=tv;
            }
        }
    }
}

}
