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
    recmat.SetSize(dim, dof*dim);   //intermediate mat
    recmatT.SetSize(dof*dim, dim);   //intermediate mat
    partrecmat.SetSize(dim, dof);
    partrecmat2.SetSize(1, dof);

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
        MultVWt(Bvec, divshape, recmat);

        //include [(grad B) - (div B)I].x  [??]
        el.CalcPhysShape(Trans, shape);
        BmatC->Eval(Bmat, Trans, ip);
        for (int j = 0; j < dim; j++){
            for (int ii = 0; ii < dim; ii++){
                tmp(ii)=Bmat(ii,j);
            }
            MultVWt(tmp, shape, partrecmat);
            recmat.AddMatrix(1.0, partrecmat, 0, dof*j);
        }

        //include B.grad x [??]
        gshape.Mult(Bvec, BdotGrad);
        partrecmat2.SetRow(0,BdotGrad);
        for (int j = 0; j < dim; j++){
            recmat.AddMatrix(-1.0, partrecmat2, j, dof*j);
        }

        recmatT.Transpose(recmat);
        AddMult_a_AAt(w, recmatT, elmat);
    }
}
