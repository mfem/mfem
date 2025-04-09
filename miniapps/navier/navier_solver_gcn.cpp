#include "navier_solver_gcn.hpp"
#include "../../general/forall.hpp"
#include <fstream>
#include <iomanip>

namespace mfem {


NavierSolverGCN::NavierSolverGCN(ParMesh* mesh, int order, std::shared_ptr<Coefficient> visc_):
    pmesh(mesh), order(order), visc(visc),
    thet1(real_t(0.5)), thet2(real_t(0.5)),thet3(real_t(0.5)),thet4(real_t(0.5))
{

    vfec.reset(new H1_FECollection(order, pmesh->Dimension()));
    pfec.reset(new H1_FECollection(order));
    vfes.reset(new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension()));
    pfes.reset(new ParFiniteElementSpace(pmesh, pfec));

    int vfes_truevsize = vfes->GetTrueVSize();
    int pfes_truevsize = pfes->GetTrueVSize();

    cvel.SetSpace(vfes);
    pvel.SetSpace(vfes);
    pres.SetSpace(pfes);

    ConvectionIntegrator* bla=new ConvectionIntegrator();


}

NavierSolverGCN::~NavierSolverGCN()
{

}

void NavierSolverGCN::Setup(real_t dt)
{

}

void NavierSolverGCN::Step(real_t &time, real_t dt, int cur_step, bool provisional = false)
{

}

//VectorConvectionIntegrator

void VectorConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{

    int nd = el.GetDof();
    dim = el.GetDim();

 #ifdef MFEM_THREAD_SAFE
    DenseMatrix dshape, adjJ, Q_ir;
    Vector shape, vec2, BdFidxT;
 #endif
    elmat.SetSize(nd*dim);
    dshape.SetSize(nd,dim);
    adjJ.SetSize(dim);
    shape.SetSize(nd);
    vec2.SetSize(dim);
    BdFidxT.SetSize(nd);
    partelmat.SetSize(nd);

    Vector vec1;


    const IntegrationRule *ir = GetIntegrationRule(el, Trans);
    if (ir == NULL)
    {
       int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
       ir = &IntRules.Get(el.GetGeomType(), order);
    }

    Q->Eval(Q_ir, Trans, *ir);

    elmat = 0.0;
    partelmat=0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       el.CalcDShape(ip, dshape);
       el.CalcShape(ip, shape);

       Trans.SetIntPoint(&ip);
       CalcAdjugate(Trans.Jacobian(), adjJ);
       Q_ir.GetColumnReference(i, vec1);
       vec1 *= alpha * ip.weight;

       adjJ.Mult(vec1, vec2);
       dshape.Mult(vec2, BdFidxT);

       AddMultVWt(shape, BdFidxT, partelmat);
    }

    for (int k = 0; k < dim; k++)
    {
       elmat.AddMatrix(partelmat, nd*k, nd*k);
    }

}

const IntegrationRule &VectorConvectionIntegrator::GetRule(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   const ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + Trans.Order() + test_fe.GetOrder();

   return IntRules.Get(trial_fe.GetGeomType(), order);
}

const IntegrationRule &VectorConvectionIntegrator::GetRule(
   const FiniteElement &el, const ElementTransformation &Trans)
{
   return GetRule(el,el,Trans);
}

}//end namespace mfem
