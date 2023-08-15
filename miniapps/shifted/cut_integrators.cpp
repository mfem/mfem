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



}
