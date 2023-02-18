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
    recmat.SetSize(dim, dof*dim);   //intermediate mat
    recmatT.SetSize(dof*dim, dim);   //intermediate mat
    partrecmat.SetSize(dim, dof);
    partrecmat2.SetSize(1, dof);

    //final output:
    elmat.SetSize(dof*dim);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = 2 * Trans.OrderGrad(&el); // correct order?
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir -> GetNPoints(); i++){
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDShape(ip, dshape);
        el.CalcShape(ip, shape);

        Trans.SetIntPoint(&ip);
        w = ip.weight * Trans.Weight();
        Mult(dshape, Trans.InverseJacobian(), gshape);
        gshape.GradToDiv (divshape);

        //include B.div^T
        BC->Eval(Bvec, Trans, ip);
        MultVWt(Bvec, divshape, recmat);

        //include [(grad B) - (div B)I].x  [ok]
        BmatC->Eval(Bmat, Trans, ip);
        for (int j = 0; j < dim; j++){
            for (int i = 0; i < dim; i++){
                tmp(i)=Bmat(i,j);
            }
            MultVWt(tmp, shape, partrecmat);
            recmat.AddMatrix(1.0, partrecmat, 0, dof*j);
        }

        //include B.grad x [ok]
        // Note w = ip.weight / Trans.Weight() in VectorDiffusionIntegrator
        // Here we let w = ip.weight*Trans.Weight() and thus need to adjust
        Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);  
        dshapedxt.Mult(Bvec, BdotGrad);
        partrecmat2.SetRow(0,BdotGrad);
        for (int i = 0; i < dim; i++){
            recmat.AddMatrix(-1.0/Trans.Weight(), partrecmat2, i, dof*i);
        }

        recmatT.Transpose(recmat);
        Mult_a_AAt(w, recmatT, elmat);
    }
}
