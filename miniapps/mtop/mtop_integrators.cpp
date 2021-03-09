#include "mtop_integrators.hpp"

namespace mfem {

double ParametricLinearDiffusion::GetElementEnergy(const Array<const FiniteElement *> &el,
                                                   const Array<const FiniteElement *> &pel,
                                                   ElementTransformation &Tr,
                                                   const Array<const Vector *> &elfun,
                                                   const Array<const Vector *> &pelfun)
{
    int dof_u0 = el[0]->GetDof();
    int dof_r0 = pel[0]->GetDof();

    int dim = el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricLinearDiffusion::GetElementEnergy"
                         " is not defined on manifold meshes");
    }

    //shape functions
    Vector shu0(dof_u0);
    Vector shr0(dof_r0);
    DenseMatrix dsu0(dof_u0,dim);
    DenseMatrix B(dof_u0, 4);
    B=0.0;

    double w;

    Vector param(1); param=0.0;
    Vector uu(4); uu=0.0;

    double energy =0.0;

    const IntegrationRule *ir;
    {
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysDShape(Tr,dsu0);
        el[0]->CalcPhysShape(Tr,shu0);
        pel[0]->CalcPhysShape(Tr,shr0);

        param[0]=shr0*(*pelfun[0]);

        //set the matrix B
        for(int jj=0;jj<dim;jj++)
        {
            B.SetCol(jj,dsu0.GetColumn(jj));
        }
        B.SetCol(3,shu0);
        B.MultTranspose(*elfun[0],uu);
        energy=energy+w * qfun.QEnergy(Tr,ip,param,uu);
    }
    return energy;
}


void ParametricLinearDiffusion::AssembleElementVector(const Array<const FiniteElement *> &el,
                                                      const Array<const FiniteElement *> &pel,
                                                      ElementTransformation &Tr,
                                                      const Array<const Vector *> &elfun,
                                                      const Array<const Vector *> &pelfun,
                                                      const Array<Vector *> &elvec)
{
    int dof_u0 = el[0]->GetDof();
    int dof_r0 = pel[0]->GetDof();

    int dim = el[0]->GetDim();

    elvec[0]->SetSize(dof_u0);
    *elvec[0]=0.0;
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricLinearDiffusion::AssembleElementVector"
                         " is not defined on manifold meshes");
    }

    //shape functions
    Vector shu0(dof_u0);
    Vector shr0(dof_r0);
    DenseMatrix dsu0(dof_u0,dim);
    DenseMatrix B(dof_u0, 4);
    B=0.0;

    double w;

    Vector param(1); param=0.0;
    Vector uu(4); uu=0.0;
    Vector rr(4);
    Vector lvec; lvec.SetSize(dof_u0);

    const IntegrationRule *ir = nullptr;
    if(ir==nullptr){
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysDShape(Tr,dsu0);
        el[0]->CalcPhysShape(Tr,shu0);
        pel[0]->CalcPhysShape(Tr,shr0);

        param[0]=shr0*(*pelfun[0]);

        //set the matrix B
        for(int jj=0;jj<dim;jj++)
        {
            B.SetCol(jj,dsu0.GetColumn(jj));
        }
        B.SetCol(3,shu0);
        B.MultTranspose(*elfun[0],uu);
        qfun.QResidual(Tr,ip,param, uu, rr);

        B.Mult(rr,lvec);
        elvec[0]->Add(w,lvec);
    }
}

void ParametricLinearDiffusion::AssembleElementGrad(const Array<const FiniteElement *> &el,
                                                    const Array<const FiniteElement *> &pel,
                                                    ElementTransformation &Tr,
                                                    const Array<const Vector *> &elfun,
                                                    const Array<const Vector *> &pelfun,
                                                    const Array2D<DenseMatrix *> &elmats)
{
     int dof_u0 = el[0]->GetDof();
     int dof_r0 = pel[0]->GetDof();

     int dim = el[0]->GetDim();

     //elmats[0]->Size(dof_u0, dof_u0);
     //*elmats[0]=0.0;

     DenseMatrix* K=elmats(0,0);
     K->SetSize(dof_u0,dof_u0);
     (*K)=0.0;

     int spaceDim = Tr.GetSpaceDim();
     if (dim != spaceDim)
     {
         mfem::mfem_error("ParametricLinearDiffusion::AssembleElementGrad"
                          " is not defined on manifold meshes");
     }

     //shape functions
     Vector shu0(dof_u0);
     Vector shr0(dof_r0);
     DenseMatrix dsu0(dof_u0,dim);
     DenseMatrix B(dof_u0, 4);
     DenseMatrix A(dof_u0, 4);
     B=0.0;
     double w;

     Vector param(1); param=0.0;
     Vector uu(4); uu=0.0;
     DenseMatrix hh(4,4);
     Vector lvec; lvec.SetSize(dof_u0);

     const IntegrationRule *ir = nullptr;
     if(ir==nullptr){
         int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                             +pel[0]->GetOrder();
         ir=&IntRules.Get(Tr.GetGeometryType(),order);
     }

     for (int i = 0; i < ir->GetNPoints(); i++)
     {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetIntPoint(&ip);
         w = Tr.Weight();
         w = ip.weight * w;

         el[0]->CalcPhysDShape(Tr,dsu0);
         el[0]->CalcPhysShape(Tr,shu0);
         pel[0]->CalcPhysShape(Tr,shr0);

         param[0]=shr0*(*pelfun[0]);

         //set the matrix B
         for(int jj=0;jj<dim;jj++)
         {
             B.SetCol(jj,dsu0.GetColumn(jj));
         }
         B.SetCol(3,shu0);
         B.MultTranspose(*elfun[0],uu);
         qfun.QGradResidual(Tr,ip,param,uu,hh);
         Mult(B,hh,A);
         AddMult_a_ABt(w,A,B,*K);
     }
}

void ParametricLinearDiffusion::AssemblePrmElementVector(const Array<const FiniteElement *> &el,
                                                         const Array<const FiniteElement *> &pel,
                                                         ElementTransformation &Tr,
                                                         const Array<const Vector *> &elfun,
                                                         const Array<const Vector *> &alfun,
                                                         const Array<const Vector *> &pelfun,
                                                         const Array<Vector *> &elvec)
{
    int dof_u0 = el[0]->GetDof();
    int dof_r0 = pel[0]->GetDof();

    int dim = el[0]->GetDim();
    Vector& e0 = *(elvec[0]);

    e0.SetSize(dof_r0);
    e0=0.0;

    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricLinearDiffusion::AssemblePrmElementVector"
                         " is not defined on manifold meshes");
    }

    //shape functions
    Vector shu0(dof_u0);
    Vector shr0(dof_r0);
    DenseMatrix dsu0(dof_u0,dim);
    DenseMatrix B(dof_u0, 4);
    B=0.0;

    double w;

    Vector param(1); param=0.0;
    Vector uu(4); uu=0.0;
    Vector aa(4); aa=0.0;
    Vector rr(1);
    Vector lvec0; lvec0.SetSize(dof_r0);

    const IntegrationRule *ir;
    {
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysDShape(Tr,dsu0);
        el[0]->CalcPhysShape(Tr,shu0);
        pel[0]->CalcPhysShape(Tr,shr0);

        param[0]=shr0*(*pelfun[0]);

        //set the matrix B
        for(int jj=0;jj<dim;jj++)
        {
            B.SetCol(jj,dsu0.GetColumn(jj));
        }
        B.SetCol(3,shu0);
        B.MultTranspose(*elfun[0],uu);
        B.MultTranspose(*alfun[0],aa);

        qfun.AQResidual(Tr, ip, param, uu, aa, rr);

        lvec0=shr0;
        lvec0*=rr[0];

        e0.Add(w,lvec0);
    }
}

double DiffusionObjIntegrator::GetElementEnergy(const Array<const FiniteElement *> &el,
                                                ElementTransformation &Tr,
                                                const Array<const Vector *> &elfun)
{

    int dof_u0 = el[0]->GetDof();
    int dim = el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("DiffusionObjIntegrator::GetElementEnergy"
                         " is not defined on manifold meshes");
    }

    //shape functions
    Vector shu0(dof_u0);

    double w;
    double val;

    double energy = 0.0;

    const IntegrationRule *ir;
    {
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();

        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu0);

        val=shu0*(*elfun[0]);
        energy=energy + w * std::pow(val,ppc);
    }
    return energy/ppc;
}

void DiffusionObjIntegrator::AssembleElementVector(const Array<const FiniteElement *> &el,
                                                   ElementTransformation &Tr,
                                                   const Array<const Vector *> &elfun,
                                                   const Array<Vector *> &elvec)
{
    int dof_u0 = el[0]->GetDof();
    int dim = el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();

    elvec[0]->SetSize(dof_u0);
    *elvec[0]=0.0;

    if (dim != spaceDim)
    {
        mfem::mfem_error("DiffusionObjIntegrator::GetElementEnergy"
                         " is not defined on manifold meshes");
    }

    //shape functions
    Vector shu0(dof_u0);

    double w;
    double val;

    const IntegrationRule *ir;
    {
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();

        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu0);

        val=shu0*(*elfun[0]);

        elvec[0]->Add(w*std::pow(val,ppc-1.0),shu0);
    }
}


double
ThermalComplianceIntegrator::GetElementEnergy(const Array<const FiniteElement *> &el,
                                              ElementTransformation &Tr,
                                              const Array<const Vector *> &elfun)
{
    int dof_u0 = el[0]->GetDof();
    int dim = el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ThermalComplianceIntegrator::GetElementEnergy"
                         " is not defined on manifold meshes");
    }

    //shape functions
    Vector shu0(dof_u0);

    double w;
    double val;
    double inp;

    double energy = 0.0;

    const IntegrationRule *ir;
    {
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        inp=load.Eval(Tr,ip);

        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu0);

        val=shu0*(*elfun[0]);
        energy=energy + w * val * inp;
    }
    return energy;
}

void
ThermalComplianceIntegrator::AssembleElementVector(const Array<const FiniteElement *> &el,
                                                   ElementTransformation &Tr,
                                                   const Array<const Vector *> &elfun,
                                                   const Array<Vector *> &elvec)
{
    int dof_u0 = el[0]->GetDof();
    int dim = el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();

    elvec[0]->SetSize(dof_u0);
    *elvec[0]=0.0;

    if (dim != spaceDim)
    {
        mfem::mfem_error("ThermalComplianceIntegrator::AssembleElementVector"
                         " is not defined on manifold meshes");
    }

    //shape functions
    Vector shu0(dof_u0);

    double w;
    double val;
    double inp;

    const IntegrationRule *ir;
    {
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();

        w = ip.weight * w;

        inp=load.Eval(Tr,ip);

        el[0]->CalcPhysShape(Tr,shu0);

        val=shu0*(*elfun[0]);

        elvec[0]->Add(w*inp,shu0);
    }

}


double
QAdvectionDiffusion::QEnergy(ElementTransformation &T, const IntegrationPoint &ip, Vector &dd, Vector &uu)
{
    //parameters
    double& rho(dd[0]);
    double& bx(dd[1]);
    double& by(dd[2]);
    double& bz(dd[3]);

    //state fields and derivatives
    double& u(uu[0]);  //state field
    double& gx(uu[1]); //grad x
    double& gy(uu[2]); //grad y
    double& gz(uu[3]); //grad z
    double& qx(uu[4]); //flux x
    double& qy(uu[5]); //flux y
    double& qz(uu[6]); //flux z
    double& divq(uu[7]); //div(flux)
    double& xl(uu[8]); //Lagrange multiplier

    //compute the density at the integration point
    double rhop=PointHeavisideProj::Project(rho,eta_co,beta_co);

    mfem::DenseMatrix A(3,3); A=0.0;
    mfem::Vector As(6); dtensor.Eval(As,T,ip);
    //scale the tensor values
    As*=rhop;

    A(0,0)=As(0); A(1,1)=As(1); A(2,2)=As(2);
    A(1,2)=As(3); A(0,2)=As(4); A(0,1)=As(5);
    A(2,1)=As(3); A(2,0)=As(4); A(1,0)=As(5);


    mfem::Vector lres(3); //local res
    lres(0)=bx*u-A(0,0)*gx-A(0,1)*gy-A(0,2)*gz-qx;
    lres(1)=by*u-A(1,0)*gx-A(1,1)*gy-A(1,2)*gz-qy;
    lres(2)=bz*u-A(2,0)*gx-A(2,1)*gy-A(2,2)*gz-qz;

    double mu=muc.Eval(T,ip);
    double f=load.Eval(T,ip);

    double rez=0.5*(lres*lres)+xl*(divq+mu*u)-xl*f;

}

void QAdvectionDiffusion::QResidual(ElementTransformation &T, const IntegrationPoint &ip,
                                    Vector &dd, Vector &uu, Vector &rr)
{
    //parameters
    double& rho(dd[0]);

    //compute the density at the integration point
    double rhop=PointHeavisideProj::Project(rho,eta_co,beta_co);

    //diffusion tensor (only upper triangle)
    mfem::Vector As(6); dtensor.Eval(As,T,ip);
    double mu=muc.Eval(T,ip);
    double f=load.Eval(T,ip);

    rr.SetSize(9);

    //Maple generated code
    double t3,t5,t8,t12,t16,t23,t27,t29,t31;
    t3 = rhop*As[4];
    t5 = rhop*As[5];
    t8 = -rhop*As[0]*uu[1]-t3*uu[3]-t5*uu[2]+dd[1]*uu[0]-uu[4];
    t12 = rhop*As[3];
    t16 = -rhop*As[1]*uu[2]-t12*uu[3]-t5*uu[1]+dd[2]*uu[0]-uu[5];
    t23 = -rhop*As[2]*uu[3]-t12*uu[2]-t3*uu[1]+dd[3]*uu[0]-uu[6];
    t27 = t8*rhop;
    t29 = t16*rhop;
    t31 = t23*rhop;
    rr[0] = uu[8]*mu+t16*dd[2]+t23*dd[3]+t8*dd[1];
    rr[1] = -t27*As[0]-t29*As[5]-t31*As[4];
    rr[2] = -t27*As[5]-t29*As[1]-t31*As[3];
    rr[3] = -t27*As[4]-t29*As[3]-t31*As[2];
    rr[4] = -t8;
    rr[5] = -t16;
    rr[6] = -t23;
    rr[7] = uu[8];
    rr[8] = mu*uu[0]-f+uu[7];
}

void QAdvectionDiffusion::QGradResidual(ElementTransformation &T, const IntegrationPoint &ip,
                                        Vector &dd, Vector &uu, DenseMatrix &hh)
{
    //parameters
    double& rho(dd[0]);

    //compute the density at the integration point
    double rhop=PointHeavisideProj::Project(rho,eta_co,beta_co);

    mfem::Vector As(6); dtensor.Eval(As,T,ip);
    double mu=muc.Eval(T,ip);
    double f=load.Eval(T,ip);

    hh.SetSize(9);
    double* hv = hh.GetData(); // get pointer to the matrix
    //Maple generated code
    double t1,t2,t3,t5,t7,t9,t11,t12,t14,t17,t18,t22,t23,t24,t26,t27;
    double t28,t29,t35,t37,t38,t40,t43,t44,t46,t47,t52,t53;

    t1 = dd[1]*dd[1];
    t2 = dd[2]*dd[2];
    t3 = dd[3]*dd[3];
    t5 = rhop*As[0];
    t7 = rhop*As[4];
    t9 = rhop*As[5];
    t11 = -t5*dd[1]-t7*dd[3]-t9*dd[2];
    t12 = rhop*As[1];
    t14 = rhop*As[3];
    t17 = -t12*dd[2]-t14*dd[3]-t9*dd[1];
    t18 = rhop*As[2];
    t22 = -t14*dd[2]-t18*dd[3]-t7*dd[1];
    t23 = rhop*rhop;
    t24 = As[0]*As[0];
    t26 = As[4]*As[4];
    t27 = t23*t26;
    t28 = As[5]*As[5];
    t29 = t23*t28;
    t35 = t23*As[3];
    t37 = t23*As[5]*As[0]+t23*As[1]*As[5]+t35*As[4];
    t38 = t23*As[4];
    t40 = t23*As[2];
    t43 = t35*As[5]+t38*As[0]+t40*As[4];
    t44 = As[1]*As[1];
    t46 = As[3]*As[3];
    t47 = t23*t46;
    t52 = t35*As[1]+t38*As[5]+t40*As[3];
    t53 = As[2]*As[2];
    hv[0] = t1+t2+t3;
    hv[1] = t11;
    hv[2] = t17;
    hv[3] = t22;
    hv[4] = -dd[1];
    hv[5] = -dd[2];
    hv[6] = -dd[3];
    hv[7] = 0.0;
    hv[8] = mu;
    hv[9] = t11;
    hv[10] = t23*t24+t27+t29;
    hv[11] = t37;
    hv[12] = t43;
    hv[13] = t5;
    hv[14] = t9;
    hv[15] = t7;
    hv[16] = 0.0;
    hv[17] = 0.0;
    hv[18] = t17;
    hv[19] = t37;
    hv[20] = t23*t44+t29+t47;
    hv[21] = t52;
    hv[22] = t9;
    hv[23] = t12;
    hv[24] = t14;
    hv[25] = 0.0;
    hv[26] = 0.0;
    hv[27] = t22;
    hv[28] = t43;
    hv[29] = t52;
    hv[30] = t23*t53+t27+t47;
    hv[31] = t7;
    hv[32] = t14;
    hv[33] = t18;
    hv[34] = 0.0;
    hv[35] = 0.0;
    hv[36] = -dd[1];
    hv[37] = t5;
    hv[38] = t9;
    hv[39] = t7;
    hv[40] = 1.0;
    hv[41] = 0.0;
    hv[42] = 0.0;
    hv[43] = 0.0;
    hv[44] = 0.0;
    hv[45] = -dd[2];
    hv[46] = t9;
    hv[47] = t12;
    hv[48] = t14;
    hv[49] = 0.0;
    hv[50] = 1.0;
    hv[51] = 0.0;
    hv[52] = 0.0;
    hv[53] = 0.0;
    hv[54] = -dd[3];
    hv[55] = t7;
    hv[56] = t14;
    hv[57] = t18;
    hv[58] = 0.0;
    hv[59] = 0.0;
    hv[60] = 1.0;
    hv[61] = 0.0;
    hv[62] = 0.0;
    hv[63] = 0.0;
    hv[64] = 0.0;
    hv[65] = 0.0;
    hv[66] = 0.0;
    hv[67] = 0.0;
    hv[68] = 0.0;
    hv[69] = 0.0;
    hv[70] = 0.0;
    hv[71] = 1.0;
    hv[72] = mu;
    hv[73] = 0.0;
    hv[74] = 0.0;
    hv[75] = 0.0;
    hv[76] = 0.0;
    hv[77] = 0.0;
    hv[78] = 0.0;
    hv[79] = 1.0;
    hv[80] = 0.0;

}

void QAdvectionDiffusion::AQResidual(ElementTransformation &T, const IntegrationPoint &ip,
                                     Vector &dd, Vector &uu, Vector &aa, Vector &rr)
{

}

double
ParametricAdvecDiffusIntegrator::GetElementEnergy(const Array<const FiniteElement *> &el,
                                                  const Array<const FiniteElement *> &pel,
                                                  ElementTransformation &Tr,
                                                  const Array<const Vector *> &elfun,
                                                  const Array<const Vector *> &pelfun)
{
    int dof_u = el[0]->GetDof();//state field H1
    int dof_q = el[1]->GetDof();//flux - H(div)
    int dof_x = el[2]->GetDof();//Lagrange multiplier field L2

    int dof_r = pel[0]->GetDof(); //density
    int dof_v = pel[1]->GetDof(); //velocity

    int dim= el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricAdvecDiffusIntegrator::GetElementEnergy"
                         " is not defined on manifold meshes!");
    }

    //shape functions
    Vector shu(dof_u);
    DenseMatrix shq(dof_q,dim);
    Vector shx(dof_x);

    Vector shr(dof_r);
    Vector shv(dof_v);

    DenseMatrix dsu(dof_u,dim);
    Vector dsq(dof_q);



    double w;
    Vector param(4); //[density, b_x, b_y, b_z] where b is the velocity field
    param=0.0;

    Vector uu(9); uu=0.0;

    double energy=0.0; //the value of the Lagrangian

    const IntegrationRule *ir;
    {
        int order;
        if(pel[0]->GetOrder() > pel[1]->GetOrder()){
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        }else{
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[1]->GetOrder();
        }
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu);
        el[1]->CalcPhysVShape(Tr,shq);
        el[2]->CalcPhysShape(Tr,shx);

        pel[0]->CalcPhysShape(Tr,shr);
        pel[1]->CalcPhysShape(Tr,shv);

        el[0]->CalcPhysDShape(Tr,dsu);
        el[1]->CalcPhysDivShape(Tr,dsq);


        uu[0]=shu*(*elfun[0]);
        double* duu=uu.GetData(); // state field
        dsu.MultTranspose(elfun[0]->GetData(),duu+1); //gradients
        shq.MultTranspose(elfun[1]->GetData(),duu+4); //fluxes
        uu[7]=dsq*(*elfun[1]); //div(q)
        uu[8]=shx*(*elfun[2]); //Lagrange multiplier

        //compute the parameters
        double* pv=pelfun[1]->GetData();
        Vector  velc;
        param[0]=shr*(*pelfun[0]);
        for(int ii=0;ii<dim;ii++){
            velc.SetDataAndSize(pv+ii*dof_v,dof_v);
            param[1+ii]=shv*velc;
        }
        energy = energy +w * qfun.QEnergy(Tr,ip,param,uu);

    }

    return energy;
}

void
ParametricAdvecDiffusIntegrator::AssembleElementVector(const Array<const FiniteElement *> &el,
                                                       const Array<const FiniteElement *> &pel,
                                                       ElementTransformation &Tr,
                                                       const Array<const Vector *> &elfun,
                                                       const Array<const Vector *> &pelfun,
                                                       const Array<Vector *> &elvec)
{
    int dof_u = el[0]->GetDof();//state field H1
    int dof_q = el[1]->GetDof();//flux - H(div)
    int dof_x = el[2]->GetDof();//Lagrange multiplier field L2

    int dof_r = pel[0]->GetDof(); //density
    int dof_v = pel[1]->GetDof(); //velocity

    int dim= el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricAdvecDiffusIntegrator::GetElementEnergy"
                         " is not defined on manifold meshes!");
    }

    //shape functions
    Vector shu(dof_u);
    DenseMatrix shq(dof_q,dim);
    Vector shx(dof_x);

    Vector shr(dof_r);
    Vector shv(dof_v);

    DenseMatrix dsu(dof_u,dim);
    Vector dsq(dof_q);

    DenseMatrix Bu(dof_u,4); Bu=0.0;
    DenseMatrix Bq(dof_q,4); Bq=0.0;

    Vector r0(dof_u);
    Vector r1(dof_q);




    double w;
    Vector param(4); //[density, b_x, b_y, b_z] where b is the velocity field
    param=0.0;

    Vector uu(9); uu=0.0;
    Vector rr(9); rr=0.0;

    const IntegrationRule *ir;
    {
        int order;
        if(pel[0]->GetOrder() > pel[1]->GetOrder()){
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        }else{
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[1]->GetOrder();
        }
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu);
        el[1]->CalcPhysVShape(Tr,shq);
        el[2]->CalcPhysShape(Tr,shx);

        pel[0]->CalcPhysShape(Tr,shr);
        pel[1]->CalcPhysShape(Tr,shv);

        el[0]->CalcPhysDShape(Tr,dsu);
        el[1]->CalcPhysDivShape(Tr,dsq);

        //compute the parameters
        double* pv=pelfun[1]->GetData();
        Vector  velc;
        param[0]=shr*(*pelfun[0]);
        for(int i=0;i<dim;i++){
            velc.SetDataAndSize(pv+i*dof_v,dof_v);
            param[1+i]=shv*velc;
        }

        //Set Bu
        Bu.SetCol(0, shu);
        for(int jj=0;jj<dim;jj++)
        {
            Bu.SetCol(jj+1, dsu.GetColumn(jj));
        }

        //set Bq
        for(int jj=0;jj<dim;jj++)
        {
            Bq.SetCol(jj,shq.GetColumn(jj));
        }
        Bq.SetCol(3,dsq);

        double* puu=uu.GetData();

        Bu.MultTranspose(elfun[0]->GetData(),puu);
        Bq.MultTranspose(elfun[1]->GetData(),puu+4);
        uu[8]=shx*(*elfun[2]);

        //compute the residual
        qfun.QResidual(Tr,ip,param,uu,rr);
        double* prr=rr.GetData();
        Bu.Mult(prr,r0.GetData());
        elvec[0]->Add(w,r0);

        Bq.Mult(prr+4,r1.GetData());
        elvec[1]->Add(w,r1);

        elvec[2]->Add(w*rr[8],shx);
    }
}

void
ParametricAdvecDiffusIntegrator::AssembleElementGrad(const Array<const FiniteElement *> &el,
                                                     const Array<const FiniteElement *> &pel,
                                                     ElementTransformation &Tr,
                                                     const Array<const Vector *> &elfun,
                                                     const Array<const Vector *> &pelfun,
                                                     const Array2D<DenseMatrix *> &elmats)
{
    int dof_u = el[0]->GetDof();//state field H1
    int dof_q = el[1]->GetDof();//flux - H(div)
    int dof_x = el[2]->GetDof();//Lagrange multiplier field L2

    int dof_r = pel[0]->GetDof(); //density
    int dof_v = pel[1]->GetDof(); //velocity

    int dim= el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricAdvecDiffusIntegrator::GetElementEnergy"
                         " is not defined on manifold meshes!");
    }

    //shape functions
    Vector shu(dof_u);
    DenseMatrix shq(dof_q,dim);
    Vector shx(dof_x);

    Vector shr(dof_r);
    Vector shv(dof_v);

    DenseMatrix dsu(dof_u,dim);
    Vector dsq(dof_q);

    DenseMatrix Bu(dof_u,4); Bu=0.0;
    DenseMatrix Bq(dof_q,4); Bq=0.0;
    DenseMatrix Bx(dof_x,1);
    DenseMatrix Bt;

    double w;
    Vector param(4); //[density, b_x, b_y, b_z] where b is the velocity field
    param=0.0;

    Vector uu(9); uu=0.0;

    DenseMatrix hh(9);

    DenseMatrix* K;
    DenseMatrix lh;
    int ind0[4]={0,1,2,3};
    int ind1[4]={4,5,6,7};
    int ind2[1]={8};

    const IntegrationRule *ir;
    {
        int order;
        if(pel[0]->GetOrder() > pel[1]->GetOrder()){
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        }else{
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[1]->GetOrder();
        }
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu);
        el[1]->CalcPhysVShape(Tr,shq);
        el[2]->CalcPhysShape(Tr,shx);

        pel[0]->CalcPhysShape(Tr,shr);
        pel[1]->CalcPhysShape(Tr,shv);

        el[0]->CalcPhysDShape(Tr,dsu);
        el[1]->CalcPhysDivShape(Tr,dsq);

        //compute the parameters
        double* pv=pelfun[1]->GetData();
        Vector  velc;
        param[0]=shr*(*pelfun[0]);
        for(int i=0;i<dim;i++){
            velc.SetDataAndSize(pv+i*dof_v,dof_v);
            param[1+i]=shv*velc;
        }

        //Set Bu
        Bu.SetCol(0, shu);
        for(int jj=0;jj<dim;jj++)
        {
            Bu.SetCol(jj+1, dsu.GetColumn(jj));
        }

        //set Bq
        for(int jj=0;jj<dim;jj++)
        {
            Bq.SetCol(jj,shq.GetColumn(jj));
        }
        Bq.SetCol(3,dsq);

        //set Bx
        Bx.SetCol(0,shx);

        double* puu=uu.GetData();

        Bu.MultTranspose(elfun[0]->GetData(),puu);
        Bq.MultTranspose(elfun[1]->GetData(),puu+4);
        uu[8]=shx*(*elfun[2]);

        qfun.QGradResidual(Tr,ip,param,uu,hh);

        //block(0,0)
        K=elmats(0,0);

        lh.SetSize(4);
        for(int ii=0;ii<4;ii++){
        for(int jj=0;jj<4;jj++){
            lh(ii,jj)=hh(ind0[ii],ind0[jj]);
        }}

        Bt.SetSize(dof_u,4);
        Mult(Bu,lh,Bt);
        AddMult_a_ABt(w,Bt,Bu,*K);

        //block(0,1)
        K=elmats(0,1);

        lh.SetSize(4);
        for(int ii=0;ii<4;ii++){
        for(int jj=0;jj<4;jj++){
            lh(ii,jj)=hh(ind0[ii],ind1[jj]);
        }}

        Bt.SetSize(dof_u,4);
        Mult(Bu,lh,Bt);
        AddMult_a_ABt(w,Bt,Bq,*K);

        //block(0,2)
        K=elmats(0,2);

        lh.SetSize(4,1);
        for(int ii=0;ii<4;ii++){
        for(int jj=0;jj<1;jj++){
            lh(ii,jj)=hh(ind0[ii],ind2[jj]);
        }}

        Bt.SetSize(dof_u,1);
        Mult(Bu,lh,Bt);
        AddMult_a_ABt(w,Bt,Bx,*K);

        //block(1,1)
        K=elmats(1,1);

        lh.SetSize(4,4);
        for(int ii=0;ii<4;ii++){
        for(int jj=0;jj<4;jj++){
            lh(ii,jj)=hh(ind1[ii],ind1[jj]);
        }}

        Bt.SetSize(dof_q,4);
        Mult(Bq,lh,Bt);
        AddMult_a_ABt(w,Bt,Bq,*K);

        //block(1,2)
        K=elmats(1,2);
        lh.SetSize(4,1);
        for(int ii=0;ii<4;ii++){
        for(int jj=0;jj<1;jj++){
            lh(ii,jj)=hh(ind1[ii],ind2[jj]);
        }}

        Bt.SetSize(dof_q,1);
        Mult(Bq,lh,Bt);
        AddMult_a_ABt(w,Bt,Bx,*K);

        //block(2,2) - is always 0
    }

    //set block(1,0)
    K=elmats(1,0);
    *K = (*elmats(0,1));
    K->Transpose();

    //set block(2,0)
    K=elmats(2,0);
    *K = (*elmats(0,2));
    K->Transpose();

    //set block(2,1)
    K=elmats(2,1);
    *K = (*elmats(1,2));
    K->Transpose();
}

void
ParametricAdvecDiffusIntegrator::AssemblePrmElementVector(const Array<const FiniteElement *> &el,
                                                          const Array<const FiniteElement *> &pel,
                                                          ElementTransformation &Tr,
                                                          const Array<const Vector *> &elfun,
                                                          const Array<const Vector *> &alfun,
                                                          const Array<const Vector *> &pelfun,
                                                          const Array<Vector *> &elvec)
{

}


double QAdvectionDiffusionLSFEM::QEnergy(ElementTransformation &T,
                                         const IntegrationPoint &ip,
                                         Vector &dd, Vector &uu)
{
    //parameters
    double& rho(dd[0]);
    double& bx(dd[1]);
    double& by(dd[2]);
    double& bz(dd[3]);

    //state fields and derivatives
    double& u(uu[0]);  //state field
    double& gx(uu[1]); //grad x
    double& gy(uu[2]); //grad y
    double& gz(uu[3]); //grad z
    double& qx(uu[4]); //flux x
    double& qy(uu[5]); //flux y
    double& qz(uu[6]); //flux z
    double& divq(uu[7]); //div(flux)

    //compute the density at the integration point
    double rhop=PointHeavisideProj::Project(rho,eta_co,beta_co);

    mfem::DenseMatrix A(3,3); A=0.0;
    mfem::Vector As(6); dtensor.Eval(As,T,ip);
    //scale the tensor values
    As*=rhop;

    A(0,0)=As(0); A(1,1)=As(1); A(2,2)=As(2);
    A(1,2)=As(3); A(0,2)=As(4); A(0,1)=As(5);
    A(2,1)=As(3); A(2,0)=As(4); A(1,0)=As(5);


    mfem::Vector lres(3); //local res
    lres(0)=bx*u-A(0,0)*gx-A(0,1)*gy-A(0,2)*gz-qx;
    lres(1)=by*u-A(1,0)*gx-A(1,1)*gy-A(1,2)*gz-qy;
    lres(2)=bz*u-A(2,0)*gx-A(2,1)*gy-A(2,2)*gz-qz;

    double mu=muc.Eval(T,ip);
    double f=load.Eval(T,ip);

    double qres=divq+mu*u-f;

    double rez=0.5*(lres*lres)+0.5*qres*qres;

}

void
QAdvectionDiffusionLSFEM::QResidual(ElementTransformation &T,
                                    const IntegrationPoint &ip,
                                    Vector &dd, Vector &uu, Vector &rr)
{
    //parameters
    double& rho(dd[0]);

    //compute the density at the integration point
    double rhop=PointHeavisideProj::Project(rho,eta_co,beta_co);

    //diffusion tensor (only upper triangle)
    mfem::Vector As(6); dtensor.Eval(As,T,ip);
    double mu=muc.Eval(T,ip);
    double f=load.Eval(T,ip);

    rr.SetSize(9);

    //code generated with Maple
    double t3,t5,t8,t12,t16,t23,t26,t29,t31,t33;
    t3 = rhop*As[4];
    t5 = rhop*As[5];
    t8 = -rhop*As[0]*uu[1]-t3*uu[3]-t5*uu[2]+dd[1]*uu[0]-uu[4];
    t12 = rhop*As[3];
    t16 = -rhop*As[1]*uu[2]-t12*uu[3]-t5*uu[1]+dd[2]*uu[0]-uu[5];
    t23 = -rhop*As[2]*uu[3]-t12*uu[2]-t3*uu[1]+dd[3]*uu[0]-uu[6];
    t26 = mu*uu[0]-f+uu[7];
    t29 = t8*rhop;
    t31 = t16*rhop;
    t33 = t23*rhop;
    rr[0] = t26*mu+t16*dd[2]+t23*dd[3]+t8*dd[1];
    rr[1] = -t29*As[0]-t31*As[5]-t33*As[4];
    rr[2] = -t29*As[5]-t31*As[1]-t33*As[3];
    rr[3] = -t29*As[4]-t31*As[3]-t33*As[2];
    rr[4] = -t8;
    rr[5] = -t16;
    rr[6] = -t23;
    rr[7] = t26;
}

void
QAdvectionDiffusionLSFEM::QGradResidual(ElementTransformation &T,
                                        const IntegrationPoint &ip,
                                        Vector &dd, Vector &uu, DenseMatrix &hh)
{
    //parameters
    double& rho(dd[0]);

    //compute the density at the integration point
    double rhop=PointHeavisideProj::Project(rho,eta_co,beta_co);

    mfem::Vector As(6); dtensor.Eval(As,T,ip);
    double mu=muc.Eval(T,ip);
    double f=load.Eval(T,ip);

    hh.SetSize(9);
    double* hv = hh.GetData(); // get pointer to the matrix

    //code generated with Maple
    double t1,t2,t3,t4,t6,t8,t10,t12,t13,t15,t18,t19;
    double t23,t24,t25,t27,t28,t29,t30,t36,t38,t39;
    double t41,t44,t45,t47,t48,t53,t54;
    t1 = mu*mu;
    t2 = dd[1]*dd[1];
    t3 = dd[2]*dd[2];
    t4 = dd[3]*dd[3];
    t6 = rhop*As[0];
    t8 = rhop*As[4];
    t10 = rhop*As[5];
    t12 = -t10*dd[2]-t6*dd[1]-t8*dd[3];
    t13 = rhop*As[1];
    t15 = rhop*As[3];
    t18 = -t10*dd[1]-t13*dd[2]-t15*dd[3];
    t19 = rhop*As[2];
    t23 = -t15*dd[2]-t19*dd[3]-t8*dd[1];
    t24 = rhop*rhop;
    t25 = As[0]*As[0];
    t27 = As[4]*As[4];
    t28 = t24*t27;
    t29 = As[5]*As[5];
    t30 = t24*t29;
    t36 = t24*As[3];
    t38 = t24*As[5]*As[0]+t24*As[1]*As[5]+t36*As[4];
    t39 = t24*As[4];
    t41 = t24*As[2];
    t44 = t36*As[5]+t39*As[0]+t41*As[4];
    t45 = As[1]*As[1];
    t47 = As[3]*As[3];
    t48 = t24*t47;
    t53 = t36*As[1]+t39*As[5]+t41*As[3];
    t54 = As[2]*As[2];
    hv[0] = t1+t2+t3+t4;
    hv[1] = t12;
    hv[2] = t18;
    hv[3] = t23;
    hv[4] = -dd[1];
    hv[5] = -dd[2];
    hv[6] = -dd[3];
    hv[7] = mu;
    hv[8] = t12;
    hv[9] = t24*t25+t28+t30;
    hv[10] = t38;
    hv[11] = t44;
    hv[12] = t6;
    hv[13] = t10;
    hv[14] = t8;
    hv[15] = 0.0;
    hv[16] = t18;
    hv[17] = t38;
    hv[18] = t24*t45+t30+t48;
    hv[19] = t53;
    hv[20] = t10;
    hv[21] = t13;
    hv[22] = t15;
    hv[23] = 0.0;
    hv[24] = t23;
    hv[25] = t44;
    hv[26] = t53;
    hv[27] = t24*t54+t28+t48;
    hv[28] = t8;
    hv[29] = t15;
    hv[30] = t19;
    hv[31] = 0.0;
    hv[32] = -dd[1];
    hv[33] = t6;
    hv[34] = t10;
    hv[35] = t8;
    hv[36] = 1.0;
    hv[37] = 0.0;
    hv[38] = 0.0;
    hv[39] = 0.0;
    hv[40] = -dd[2];
    hv[41] = t10;
    hv[42] = t13;
    hv[43] = t15;
    hv[44] = 0.0;
    hv[45] = 1.0;
    hv[46] = 0.0;
    hv[47] = 0.0;
    hv[48] = -dd[3];
    hv[49] = t8;
    hv[50] = t15;
    hv[51] = t19;
    hv[52] = 0.0;
    hv[53] = 0.0;
    hv[54] = 1.0;
    hv[55] = 0.0;
    hv[56] = mu;
    hv[57] = 0.0;
    hv[58] = 0.0;
    hv[59] = 0.0;
    hv[60] = 0.0;
    hv[61] = 0.0;
    hv[62] = 0.0;
    hv[63] = 1.0;

}

void
QAdvectionDiffusionLSFEM::AQResidual(ElementTransformation &T,
                                     const IntegrationPoint &ip,
                                     Vector &dd, Vector &uu,
                                     Vector &aa, Vector &rr)
{

}

double
ParametricAdvecDiffusLSFEM::GetElementEnergy(const Array<const FiniteElement *> &el,
                                             const Array<const FiniteElement *> &pel,
                                             ElementTransformation &Tr,
                                             const Array<const Vector *> &elfun,
                                             const Array<const Vector *> &pelfun)
{
    int dof_u = el[0]->GetDof();//state field H1
    int dof_q = el[1]->GetDof();//flux - H(div)

    int dof_r = pel[0]->GetDof(); //density
    int dof_v = pel[1]->GetDof(); //velocity

    int dim= el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricAdvecDiffusIntegratorLSFEM::GetElementEnergy"
                         " is not defined on manifold meshes!");
    }

    //shape functions
    Vector shu(dof_u);
    DenseMatrix shq(dof_q,dim);

    Vector shr(dof_r);
    Vector shv(dof_v);

    DenseMatrix dsu(dof_u,dim);
    Vector dsq(dof_q);

    double w;
    Vector param(4); //[density, b_x, b_y, b_z] where b is the velocity field
    param=0.0;

    Vector uu(8); uu=0.0;

    double energy=0.0; //the value of the Lagrangian

    const IntegrationRule *ir;
    {
        int order;
        if(pel[0]->GetOrder() > pel[1]->GetOrder()){
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        }else{
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[1]->GetOrder();
        }
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu);
        el[1]->CalcPhysVShape(Tr,shq);

        pel[0]->CalcPhysShape(Tr,shr);
        pel[1]->CalcPhysShape(Tr,shv);

        el[0]->CalcPhysDShape(Tr,dsu);
        el[1]->CalcPhysDivShape(Tr,dsq);

        uu[0]=shu*(*elfun[0]);
        double* duu=uu.GetData(); // state field
        dsu.MultTranspose(elfun[0]->GetData(),duu+1); //gradients
        shq.MultTranspose(elfun[1]->GetData(),duu+4); //fluxes
        uu[7]=dsq*(*elfun[1]); //div(q)

        //compute the parameters
        double* pv=pelfun[1]->GetData();
        Vector  velc;
        param[0]=shr*(*pelfun[0]);
        for(int ii=0;ii<dim;ii++){
            velc.SetDataAndSize(pv+ii*dof_v,dof_v);
            param[1+ii]=shv*velc;
        }
        energy = energy + w * qfun.QEnergy(Tr,ip,param,uu);
    }
    return energy;
}

void
ParametricAdvecDiffusLSFEM::AssembleElementVector(const Array<const FiniteElement *> &el,
                           const Array<const FiniteElement *> &pel,
                           ElementTransformation &Tr,
                           const Array<const Vector *> &elfun,
                           const Array<const Vector *> &pelfun,
                           const Array<Vector *> &elvec)
{
    int dof_u = el[0]->GetDof();//state field H1
    int dof_q = el[1]->GetDof();//flux - H(div)

    int dof_r = pel[0]->GetDof(); //density
    int dof_v = pel[1]->GetDof(); //velocity

    int dim= el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();

    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricAdvecDiffusIntegrator::GetElementEnergy"
                         " is not defined on manifold meshes!");
    }

    //shape functions
    Vector shu(dof_u);
    DenseMatrix shq(dof_q,dim);

    Vector shr(dof_r);
    Vector shv(dof_v);

    DenseMatrix dsu(dof_u,dim);
    Vector dsq(dof_q);

    DenseMatrix Bu(dof_u,4); Bu=0.0;
    DenseMatrix Bq(dof_q,4); Bq=0.0;

    Vector r0(dof_u);
    Vector r1(dof_q);

    double w;
    Vector param(4); //[density, b_x, b_y, b_z] where b is the velocity field
    param=0.0;

    Vector uu(8); uu=0.0;
    Vector rr(8); rr=0.0;

    const IntegrationRule *ir;
    {
        int order;
        if(pel[0]->GetOrder() > pel[1]->GetOrder()){
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        }else{
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[1]->GetOrder();
        }
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu);
        el[1]->CalcPhysVShape(Tr,shq);

        pel[0]->CalcPhysShape(Tr,shr);
        pel[1]->CalcPhysShape(Tr,shv);

        el[0]->CalcPhysDShape(Tr,dsu);
        el[1]->CalcPhysDivShape(Tr,dsq);

        //compute the parameters
        double* pv=pelfun[1]->GetData();
        Vector  velc;
        param[0]=shr*(*pelfun[0]);
        for(int i=0;i<dim;i++){
            velc.SetDataAndSize(pv+i*dof_v,dof_v);
            param[1+i]=shv*velc;
        }

        //Set Bu
        Bu.SetCol(0, shu);
        for(int jj=0;jj<dim;jj++)
        {
            Bu.SetCol(jj+1, dsu.GetColumn(jj));
        }

        //set Bq
        for(int jj=0;jj<dim;jj++)
        {
            Bq.SetCol(jj,shq.GetColumn(jj));
        }
        Bq.SetCol(3,dsq);

        double* puu=uu.GetData();

        Bu.MultTranspose(elfun[0]->GetData(),puu);
        Bq.MultTranspose(elfun[1]->GetData(),puu+4);

        //compute the residual
        qfun.QResidual(Tr,ip,param,uu,rr);
        double* prr=rr.GetData();
        Bu.Mult(prr,r0.GetData());
        elvec[0]->Add(w,r0);

        Bq.Mult(prr+4,r1.GetData());
        elvec[1]->Add(w,r1);
    }


}


void
ParametricAdvecDiffusLSFEM::AssembleElementGrad(const Array<const FiniteElement *> &el,
                         const Array<const FiniteElement *> &pel,
                         ElementTransformation &Tr,
                         const Array<const Vector *> &elfun,
                         const Array<const Vector *> &pelfun,
                         const Array2D<DenseMatrix *> &elmats)
{

    int dof_u = el[0]->GetDof();//state field H1
    int dof_q = el[1]->GetDof();//flux - H(div)

    int dof_r = pel[0]->GetDof(); //density
    int dof_v = pel[1]->GetDof(); //velocity

    int dim= el[0]->GetDim();
    int spaceDim = Tr.GetSpaceDim();
    if (dim != spaceDim)
    {
        mfem::mfem_error("ParametricAdvecDiffusIntegrator::GetElementEnergy"
                         " is not defined on manifold meshes!");
    }

    //shape functions
    Vector shu(dof_u);
    DenseMatrix shq(dof_q,dim);

    Vector shr(dof_r);
    Vector shv(dof_v);

    DenseMatrix dsu(dof_u,dim);
    Vector dsq(dof_q);

    DenseMatrix Bu(dof_u,4); Bu=0.0;
    DenseMatrix Bq(dof_q,4); Bq=0.0;
    DenseMatrix Bt;

    double w;
    Vector param(4); //[density, b_x, b_y, b_z] where b is the velocity field
    param=0.0;

    Vector uu(8); uu=0.0;

    DenseMatrix hh(8);

    DenseMatrix* K;
    DenseMatrix lh;
    int ind0[4]={0,1,2,3};
    int ind1[4]={4,5,6,7};

    const IntegrationRule *ir;
    {
        int order;
        if(pel[0]->GetOrder() > pel[1]->GetOrder()){
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[0]->GetOrder();
        }else{
            order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                            +pel[1]->GetOrder();
        }
        ir=&IntRules.Get(Tr.GetGeometryType(),order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        w=Tr.Weight();
        w = ip.weight * w;

        el[0]->CalcPhysShape(Tr,shu);
        el[1]->CalcPhysVShape(Tr,shq);

        pel[0]->CalcPhysShape(Tr,shr);
        pel[1]->CalcPhysShape(Tr,shv);

        el[0]->CalcPhysDShape(Tr,dsu);
        el[1]->CalcPhysDivShape(Tr,dsq);

        //compute the parameters
        double* pv=pelfun[1]->GetData();
        Vector  velc;
        param[0]=shr*(*pelfun[0]);
        for(int i=0;i<dim;i++){
            velc.SetDataAndSize(pv+i*dof_v,dof_v);
            param[1+i]=shv*velc;
        }

        //Set Bu
        Bu.SetCol(0, shu);
        for(int jj=0;jj<dim;jj++)
        {
            Bu.SetCol(jj+1, dsu.GetColumn(jj));
        }

        //set Bq
        for(int jj=0;jj<dim;jj++)
        {
            Bq.SetCol(jj,shq.GetColumn(jj));
        }
        Bq.SetCol(3,dsq);

        double* puu=uu.GetData();

        Bu.MultTranspose(elfun[0]->GetData(),puu);
        Bq.MultTranspose(elfun[1]->GetData(),puu+4);

        qfun.QGradResidual(Tr,ip,param,uu,hh);

        //block(0,0)
        K=elmats(0,0);

        lh.SetSize(4);
        for(int ii=0;ii<4;ii++){
        for(int jj=0;jj<4;jj++){
            lh(ii,jj)=hh(ind0[ii],ind0[jj]);
        }}

        Bt.SetSize(dof_u,4);
        Mult(Bu,lh,Bt);
        AddMult_a_ABt(w,Bt,Bu,*K);

        //block(0,1)
        K=elmats(0,1);

        lh.SetSize(4);
        for(int ii=0;ii<4;ii++){
        for(int jj=0;jj<4;jj++){
            lh(ii,jj)=hh(ind0[ii],ind1[jj]);
        }}

        Bt.SetSize(dof_u,4);
        Mult(Bu,lh,Bt);
        AddMult_a_ABt(w,Bt,Bq,*K);

        //block(1,1)
        K=elmats(1,1);

        lh.SetSize(4,4);
        for(int ii=0;ii<4;ii++){
        for(int jj=0;jj<4;jj++){
            lh(ii,jj)=hh(ind1[ii],ind1[jj]);
        }}

        Bt.SetSize(dof_q,4);
        Mult(Bq,lh,Bt);
        AddMult_a_ABt(w,Bt,Bq,*K);
    }

    //set block(1,0)
    K=elmats(1,0);
    *K = (*elmats(0,1));
    K->Transpose();

}


void
ParametricAdvecDiffusLSFEM::AssemblePrmElementVector(const Array<const FiniteElement *> &el,
                              const Array<const FiniteElement *> &pel,
                              ElementTransformation &Tr,
                              const Array<const Vector *> &elfun,
                              const Array<const Vector *> &alfun,
                              const Array<const Vector *> &pelfun,
                              const Array<Vector *> &elvec)
{

}

double FScreenedPoisson::GetElementEnergy(const FiniteElement &el,
                                         ElementTransformation &trans,
                                         const Vector &elfun)
{
   double energy = 0.0;
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
   ir = &IntRules.Get(el.GetGeomType(), order);

   Vector shapef(ndof);
   double fval;
   double pval;
   DenseMatrix B(ndof, ndim);
   Vector qval(ndim);

   B=0.0;

   double w;
   double ngrad2;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      fval=func->Eval(trans,ip);

      el.CalcPhysDShape(trans, B);
      el.CalcPhysShape(trans,shapef);

      B.MultTranspose(elfun,qval);

      ngrad2=0.0;
      for (int jj=0; jj<ndim; jj++)
      {
         ngrad2 = ngrad2 + qval(jj)*qval(jj);
      }

      energy = energy + w * ngrad2 * diffcoef * 0.5;

      // add the external load -1 if fval > 0.0; 1 if fval < 0.0;
      pval=shapef*elfun;

      energy = energy + w * pval * pval * 0.5;

      energy = energy + w * pval * fval;
   }

   return energy;
}

void FScreenedPoisson::AssembleElementVector(const FiniteElement &el,
                                            ElementTransformation &trans,
                                            const Vector &elfun,
                                            Vector &elvect)
{
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
   ir = &IntRules.Get(el.GetGeomType(), order);

   elvect.SetSize(ndof);
   elvect=0.0;

   Vector shapef(ndof);
   double fval;
   double pval;

   DenseMatrix B(ndof, ndim); //[diff_x,diff_y,diff_z]

   Vector qval(ndim); //[diff_x,diff_y,diff_z,u]
   Vector lvec(ndof); //residual at ip

   B=0.0;
   qval=0.0;

   double w;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      fval=func->Eval(trans,ip);

      el.CalcPhysDShape(trans, B);
      el.CalcPhysShape(trans,shapef);

      B.MultTranspose(elfun,qval);
      B.Mult(qval,lvec);

      elvect.Add(w * diffcoef,lvec);

      pval=shapef*elfun;

      elvect.Add(w * pval, shapef);

      elvect.Add(-w * fval, shapef);

   }
}

void FScreenedPoisson::AssembleElementGrad(const FiniteElement &el,
                                          ElementTransformation &trans,
                                          const Vector &elfun,
                                          DenseMatrix &elmat)
{
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
   ir = &IntRules.Get(el.GetGeomType(), order);

   elmat.SetSize(ndof,ndof);
   elmat=0.0;

   Vector shapef(ndof);

   DenseMatrix B(ndof, ndim); //[diff_x,diff_y,diff_z]
   B = 0.0;

   double w;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      el.CalcPhysDShape(trans, B);
      el.CalcPhysShape(trans,shapef);

      AddMult_a_VVt(w , shapef, elmat);
      AddMult_a_AAt(w * diffcoef, B, elmat);
   }
}

void PDEFilterTO::Filter(Coefficient &func, ParGridFunction &ffield)
{
   if (sint == nullptr)
   {
      sint = new FScreenedPoisson(func, rr);
      nf->AddDomainIntegrator(sint);
      *sv = 0.0;
      gmres->SetOperator(nf->GetGradient(*sv));
   }
   else { sint->SetInput(func); }

   // form RHS
   *sv = 0.0;
   Vector rhs(sv->Size());
   nf->Mult(*sv, rhs);
   // filter the input field
   gmres->Mult(rhs, *sv);

   gf.SetFromTrueDofs(*sv);
   gf.Neg();

   GridFunctionCoefficient gfc(&gf);
   ffield.ProjectCoefficient(gfc);
}


}//end mfem namespace
