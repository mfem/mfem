#include "mfem.hpp"
#include "CurlCurlIntegrator.hpp"

using namespace mfem;
using namespace std;

void BmatCoeff::Eval(DenseMatrix &K, ElementTransformation &T, const IntegrationPoint &ip)
{
    DenseMatrix grad;
    K.SetSize(height, width);
    K=0.0;
    B->GetVectorGradient(T, grad);
    double div = grad(0,0)+grad(1,1)+grad(2,2);
    K.Diag(-div, height);
    K+=grad;
    
    // debug grad and div
    /*
    cout<<grad(0,0)<<" "<<grad(0,1)<<" "<<grad(0,2)<<" "
        <<grad(1,0)<<" "<<grad(1,1)<<" "<<grad(1,2)<<" "
        <<grad(2,0)<<" "<<grad(2,1)<<" "<<grad(2,2)
        <<endl;
    cout<<div<<endl;
    std::exit(1);
    */
}

// Custom integrator
//  <B (div x) + [(grad B) - (div B)I].x - B.grad x, B (div g) + [(grad B) - (div B)I].g - B.grad g>
//
//Idea: get a rectangular matrix of vdim by dof*dim first and then Mult_a_AAt
//Note this operator only makes sense for dim==vdim==3
void SpecialVectorCurlCurlIntegrator::AssembleElementMatrix(const FiniteElement &el,
                           ElementTransformation &Trans,DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    double w;

    shape.SetSize(dof);
    divshape.SetSize(dim*dof);
    tmp.SetSize(dim);
    BdotGrad.SetSize(dof);

    dshape.SetSize(dof, dim);
    dshapedxt.SetSize(dof, dim);
    gshape.SetSize(dof, dim);
    recmat.SetSize(dof*dim, dim);   //intermediate mat
    partrecmat.SetSize(dof, dim);
    partrecmat2.SetSize(dof, 1);

    //final output:
    elmat.SetSize(dof*dim);
    elmat=0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = 2 * Trans.OrderGrad(&el); // correct order?
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir -> GetNPoints(); i++){
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDShape(ip, dshape);

        Trans.SetIntPoint(&ip);
        w = ip.weight * Trans.Weight();
        Mult(dshape, Trans.InverseJacobian(), gshape);
        gshape.GradToDiv(divshape);

        //include B.div^T [verified]
        BC->Eval(Bvec, Trans, ip);
        MultVWt(divshape, Bvec, recmat);

        //include [(grad B) - (div B)I].x  [??]
        el.CalcPhysShape(Trans, shape);
        BmatC->Eval(Bmat, Trans, ip);
        for (int j = 0; j < dim; j++){
            for (int ii = 0; ii < dim; ii++){
                tmp(ii)=Bmat(ii,j);
            }
            MultVWt(shape, tmp, partrecmat);
            recmat.AddMatrix(1.0, partrecmat, dof*j, 0);
        }

        //include B.grad x [??]
        gshape.Mult(Bvec, BdotGrad);
        partrecmat2.SetCol(0, BdotGrad);
        for (int j = 0; j < dim; j++){
            recmat.AddMatrix(-1.0, partrecmat2, dof*j, j);
        }

        AddMult_a_AAt(w, recmat, elmat);
    }
}

void VectorGradDivIntegrator::AssembleElementMatrix(const FiniteElement &el,
                           ElementTransformation &Trans,DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    double w;

    shape.SetSize(dof);
    divshape.SetSize(dim*dof);

    dshape.SetSize(dof, dim);
    dshapedxt.SetSize(dof, dim);
    gshape.SetSize(dof, dim);

    //final output:
    elmat.SetSize(dof*dim);
    elmat=0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = 2 * Trans.OrderGrad(&el); // correct order?
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir -> GetNPoints(); i++){
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDShape(ip, dshape);

        Trans.SetIntPoint(&ip);
        w = ip.weight * Trans.Weight();
        Mult(dshape, Trans.InverseJacobian(), gshape);
        gshape.GradToDiv(divshape);

        BC->Eval(Bvec, Trans, ip);
        double L = Bvec.Norml2();
        L = L*L;

        AddMult_a_VVt(L * w, divshape, elmat);
    }
}

void SpecialVectorDiffusionIntegrator::AssembleElementMatrix(const FiniteElement &el,
                           ElementTransformation &Trans,DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    double w;

    shape.SetSize(dof);
    divshape.SetSize(dim*dof);
    BdotGrad.SetSize(dof);

    dshape.SetSize(dof, dim);
    dshapedxt.SetSize(dof, dim);
    gshape.SetSize(dof, dim);
    recmat.SetSize(dof*dim, dim);   //intermediate mat
    partrecmat.SetSize(dof, dim);
    partrecmat2.SetSize(dof, 1);

    //final output:
    elmat.SetSize(dof*dim);
    elmat=0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = 2 * Trans.OrderGrad(&el); // correct order?
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir -> GetNPoints(); i++){
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDShape(ip, dshape);

        Trans.SetIntPoint(&ip);
        w = ip.weight * Trans.Weight();
        Mult(dshape, Trans.InverseJacobian(), gshape);

        recmat = 0.0;
        BC->Eval(Bvec, Trans, ip);
        //include B.grad x [??]
        gshape.Mult(Bvec, BdotGrad);
        partrecmat2.SetCol(0, BdotGrad);
        for (int j = 0; j < dim; j++){
            recmat.AddMatrix(-1.0, partrecmat2, dof*j, j);
        }

        AddMult_a_AAt(w, recmat, elmat);
    }
}
