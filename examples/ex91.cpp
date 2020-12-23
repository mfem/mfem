#include"mfem.hpp"

#include<memory>
#include<iostream>
#include<fstream>

namespace mfem {


class LinDiffQFunc
{
public:
    LinDiffQFunc(mfem::Coefficient& dd, mfem::Coefficient& ll, double gg_,
                 double pp0_, double pp1_):diff(dd),load(ll),gg(gg_),pp0(pp0_),pp1(pp1_)
    {

    }

    double QEnergy(ElementTransformation &T,
                  const IntegrationPoint &ip,
                  mfem::Vector& param, mfem::Vector& uu)
    {
        double dd=diff.Eval(T,ip);
        double ll=load.Eval(T,ip);

        double rho0=param[0];
        double rho1=param[1];
        double fd=dd*std::pow(rho0,pp0)*std::pow(rho1,pp1);

        double rez = 0.5*(uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2])*fd
                        + 0.5*gg*uu[3]*uu[3] -uu[3]*ll;

        return rez;
    }

    void QResidual(ElementTransformation &T,
                   const IntegrationPoint &ip,
                   mfem::Vector& param, mfem::Vector& uu, mfem::Vector& rr)
    {
        rr.SetSize(4);
        double dd=diff.Eval(T,ip);
        double ll=load.Eval(T,ip);

        double rho0=param[0];
        double rho1=param[1];

        double fd=dd*std::pow(rho0,pp0)*std::pow(rho1,pp1);

        rr[0]=uu[0]*fd;
        rr[1]=uu[1]*fd;
        rr[2]=uu[2]*fd;
        rr[3]=gg*uu[3]-ll;
    }

    void AQResidual(ElementTransformation &T,
                    const IntegrationPoint &ip,
                    mfem::Vector& param,
                    mfem::Vector& uu, mfem::Vector& aa, mfem::Vector& rr)
    {
         rr.SetSize(2);
         double dd=diff.Eval(T,ip);
         double ll=load.Eval(T,ip);

         double rho0=param[0];
         double rho1=param[1];

         double fd0=dd*pp0*std::pow(rho0,pp0-1.0)*std::pow(rho1,pp1);
         double fd1=dd*std::pow(rho0,pp0)*pp1*std::pow(rho1,pp1-1.0);

         rr[0] = (aa[0]*uu[0]+aa[1]*uu[1]+aa[2]*uu[2])*fd0;
         rr[1] = (aa[0]*uu[0]+aa[1]*uu[1]+aa[2]*uu[2])*fd1;

    }

    void QGradResidual(ElementTransformation &T,
                       const IntegrationPoint &ip,
                       mfem::Vector& param, mfem::Vector& uu, mfem::DenseMatrix& hh)
    {
        hh.SetSize(4);
        double dd=diff.Eval(T,ip);
        //double ll=load.Eval(T,ip);


        double rho0=param[0];
        double rho1=param[1];

        double fd=dd*std::pow(rho0,pp0)*std::pow(rho1,pp1);
        hh=0.0;

        hh(0,0)=fd;
        hh(1,1)=fd;
        hh(2,2)=fd;
        hh(3,3)=gg;
    }


private:
    mfem::Coefficient& diff;
    mfem::Coefficient& load;
    double gg;
    double pp0;
    double pp1;
};


class PrmBlockLSFEMDiffusion: public PrmBlockNonlinearFormIntegrator
{
public:
    PrmBlockLSFEMDiffusion(LinDiffQFunc& qfun_)
    {
        qfunc=&qfun_;
    }

    /// Compute the local energy
    virtual double GetElementEnergy(const Array<const FiniteElement *>&el,
                                    const Array<const FiniteElement *>&pel,
                                    ElementTransformation &Tr,
                                    const Array<const Vector *>&elfun,
                                    const Array<const Vector *>&pelfun)
    {
        int dof_u0 = el[0]->GetDof();
        int dof_r0 = pel[0]->GetDof();
        int dof_r1 = pel[1]->GetDof();

        int dim = el[0]->GetDim();
        int spaceDim = Tr.GetSpaceDim();
        if (dim != spaceDim)
        {
            mfem::mfem_error(" PrmBlockLSFEMDiffusion::GetElementEnergy"
                             " is not defined on manifold meshes");
        }

        //shape functions
        Vector shu0(dof_u0);
        Vector shr0(dof_r0);
        Vector shr1(dof_r1);
        DenseMatrix dsu0(dof_u0,dim);
        DenseMatrix B(dof_u0, 4);
        B=0.0;

        double w;

        Vector param(2); param=0.0;
        Vector uu(4); uu=0.0;

        double energy =0.0;

        const IntegrationRule *ir = nullptr;
        if(ir==nullptr){
            int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                                +pel[0]->GetOrder()+pel[1]->GetOrder();
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
            pel[1]->CalcPhysShape(Tr,shr1);

            param[0]=shr0*(*pelfun[0]);
            param[1]=shr1*(*pelfun[1]);

            //set the matrix B
            for(int jj=0;jj<dim;jj++)
            {
                B.SetCol(jj,dsu0.GetColumn(jj));
            }
            B.SetCol(3,shu0);
            B.MultTranspose(*elfun[0],uu);
            energy=energy+w * qfunc->QEnergy(Tr,ip,param,uu);
        }
        return energy;
    }

    /// Perform the local action of the BlockNonlinearFormIntegrator
    virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                       const Array<const FiniteElement *>&pel,
                                       ElementTransformation &Tr,
                                       const Array<const Vector *> &elfun,
                                       const Array<const Vector *>&pelfun,
                                       const Array<Vector *> &elvec)
    {
        int dof_u0 = el[0]->GetDof();
        int dof_r0 = pel[0]->GetDof();
        int dof_r1 = pel[1]->GetDof();

        int dim = el[0]->GetDim();

        elvec[0]->SetSize(dof_u0);
        *elvec[0]=0.0;
        int spaceDim = Tr.GetSpaceDim();
        if (dim != spaceDim)
        {
            mfem::mfem_error(" PrmBlockLSFEMDiffusion::AssembleElementVector"
                             " is not defined on manifold meshes");
        }

        //shape functions
        Vector shu0(dof_u0);
        Vector shr0(dof_r0);
        Vector shr1(dof_r1);
        DenseMatrix dsu0(dof_u0,dim);
        DenseMatrix B(dof_u0, 4);
        B=0.0;

        double w;

        Vector param(2); param=0.0;
        Vector uu(4); uu=0.0;
        Vector rr;
        Vector lvec; lvec.SetSize(dof_u0);

        const IntegrationRule *ir = nullptr;
        if(ir==nullptr){
            int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                                +pel[0]->GetOrder()+pel[1]->GetOrder();
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
            pel[1]->CalcPhysShape(Tr,shr1);

            param[0]=shr0*(*pelfun[0]);
            param[1]=shr1*(*pelfun[1]);

            //set the matrix B
            for(int jj=0;jj<dim;jj++)
            {
                B.SetCol(jj,dsu0.GetColumn(jj));
            }
            B.SetCol(3,shu0);
            B.MultTranspose(*elfun[0],uu);
            qfunc->QResidual(Tr,ip,param, uu, rr);

            B.Mult(rr,lvec);
            elvec[0]->Add(w,lvec);
        }

    }

    virtual void AssembleFaceVector(const Array<const FiniteElement *> &el1,
                                    const Array<const FiniteElement *> &el2,
                                    const Array<const FiniteElement *> &pel1,
                                    const Array<const FiniteElement *> &pel2,
                                    FaceElementTransformations &Tr,
                                    const Array<const Vector *> &elfun,
                                    const Array<const Vector *>&pelfun,
                                    const Array<Vector *> &elvect)
    {

    }

    /// Assemble the local gradient matrix
    virtual void AssembleElementGrad(const Array<const FiniteElement*> &el,
                                     const Array<const FiniteElement *>&pel,
                                     ElementTransformation &Tr,
                                     const Array<const Vector *> &elfun,
                                     const Array<const Vector *>&pelfun,
                                     const Array2D<DenseMatrix *> &elmats)
    {
        int dof_u0 = el[0]->GetDof();
        int dof_r0 = pel[0]->GetDof();
        int dof_r1 = pel[1]->GetDof();

        int dim = el[0]->GetDim();

        //elmats[0]->Size(dof_u0, dof_u0);
        //*elmats[0]=0.0;

        DenseMatrix* K=elmats(0,0);
        K->SetSize(dof_u0,dof_u0);
        (*K)=0.0;

        int spaceDim = Tr.GetSpaceDim();
        if (dim != spaceDim)
        {
            mfem::mfem_error(" PrmBlockLSFEMDiffusion::AssembleElementVector"
                             " is not defined on manifold meshes");
        }

        //shape functions
        Vector shu0(dof_u0);
        Vector shr0(dof_r0);
        Vector shr1(dof_r1);
        DenseMatrix dsu0(dof_u0,dim);
        DenseMatrix B(dof_u0, 4);
        DenseMatrix A(dof_u0, 4);
        B=0.0;

        double w;

        Vector param(2); param=0.0;
        Vector uu(4); uu=0.0;
        DenseMatrix hh;
        Vector lvec; lvec.SetSize(dof_u0);

        const IntegrationRule *ir = nullptr;
        if(ir==nullptr){
            int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                                +pel[0]->GetOrder()+pel[1]->GetOrder();
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
            pel[1]->CalcPhysShape(Tr,shr1);

            param[0]=shr0*(*pelfun[0]);
            param[1]=shr1*(*pelfun[1]);

            //set the matrix B
            for(int jj=0;jj<dim;jj++)
            {
                B.SetCol(jj,dsu0.GetColumn(jj));
            }
            B.SetCol(3,shu0);
            B.MultTranspose(*elfun[0],uu);
            qfunc->QGradResidual(Tr,ip,param,uu,hh);
            Mult(B,hh,A);
            AddMult_a_ABt(w,A,B,*K);
        }
    }

    virtual void AssembleFaceGrad(const Array<const FiniteElement *>&el1,
                                  const Array<const FiniteElement *>&el2,
                                  const Array<const FiniteElement *> &pel1,
                                  const Array<const FiniteElement *> &pel2,
                                  FaceElementTransformations &Tr,
                                  const Array<const Vector *> &elfun,
                                  const Array<const Vector *>&pelfun,
                                  const Array2D<DenseMatrix *> &elmats)
    {

    }

    virtual void AssemblePrmElementVector(const Array<const FiniteElement *> &el,
                                          const Array<const FiniteElement *> &pel,
                                          ElementTransformation &Tr,
                                          const Array<const Vector *> &elfun,
                                          const Array<const Vector *> &alfun,
                                          const Array<const Vector *> &pelfun,
                                          const Array<Vector *> &elvec)
    {
        int dof_u0 = el[0]->GetDof();
        int dof_r0 = pel[0]->GetDof();
        int dof_r1 = pel[1]->GetDof();

        int dim = el[0]->GetDim();

        Vector& e0 = *(elvec[0]);
        Vector& e1 = *(elvec[1]);

        e0.SetSize(dof_r0);
        e0=0.0;
        e1.SetSize(dof_r1);
        e1=0.0;

        int spaceDim = Tr.GetSpaceDim();
        if (dim != spaceDim)
        {
            mfem::mfem_error(" PrmBlockLSFEMDiffusion::AssembleElementVector"
                             " is not defined on manifold meshes");
        }

        //shape functions
        Vector shu0(dof_u0);
        Vector shr0(dof_r0);
        Vector shr1(dof_r1);
        DenseMatrix dsu0(dof_u0,dim);
        DenseMatrix B(dof_u0, 4);
        B=0.0;

        double w;

        Vector param(2); param=0.0;
        Vector uu(4); uu=0.0;
        Vector aa(4); aa=0.0;
        Vector rr;
        Vector lvec0; lvec0.SetSize(dof_r0);
        Vector lvec1; lvec1.SetSize(dof_r1);

        const IntegrationRule *ir = nullptr;
        if(ir==nullptr){
            int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0])
                                +pel[0]->GetOrder()+pel[1]->GetOrder();
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
            pel[1]->CalcPhysShape(Tr,shr1);

            param[0]=shr0*(*pelfun[0]);
            param[1]=shr1*(*pelfun[1]);

            //set the matrix B
            for(int jj=0;jj<dim;jj++)
            {
                B.SetCol(jj,dsu0.GetColumn(jj));
            }
            B.SetCol(3,shu0);
            B.MultTranspose(*elfun[0],uu);
            B.MultTranspose(*alfun[0],aa);

            qfunc->AQResidual(Tr, ip, param, uu, aa, rr);

            lvec0=shr0;
            lvec0*=rr[0];
            lvec1=shr1;
            lvec1*=rr[1];

            e0.Add(w,lvec0);
            e1.Add(w,lvec1);
        }

    }

    virtual void AssemblePrmFaceVector(const Array<const FiniteElement *> &el1,
                                       const Array<const FiniteElement *> &el2,
                                       const Array<const FiniteElement *> &pel1,
                                       const Array<const FiniteElement *> &pel2,
                                       FaceElementTransformations &Tr,
                                       const Array<const Vector *> &elfun,
                                       const Array<const Vector *> &alfun,
                                       const Array<const Vector *> &pelfun,
                                       const Array<Vector *> &elvect)
    {

    }



private:
    LinDiffQFunc* qfunc;
};

}

int main(int argc, char *argv[])
{
    const char *mesh_file = "../../data/beam-tet.mesh";
    int ser_ref_levels = 1;
    int order = 2;
    bool visualization = true;
    double newton_rel_tol = 1e-4;
    double newton_abs_tol = 1e-6;
    int newton_iter = 10;
    int print_level = 0;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&ser_ref_levels,
                     "-rs",
                     "--refine-serial",
                     "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&order,
                      "-o",
                      "--order",
                      "Order (degree) of the finite elements.");
    args.AddOption(&visualization,
                      "-vis",
                      "--visualization",
                      "-no-vis",
                      "--no-visualization",
                      "Enable or disable GLVis visualization.");
    args.AddOption(&newton_rel_tol,
                      "-rel",
                      "--relative-tolerance",
                      "Relative tolerance for the Newton solve.");
    args.AddOption(&newton_abs_tol,
                      "-abs",
                      "--absolute-tolerance",
                      "Absolute tolerance for the Newton solve.");
    args.AddOption(&newton_iter,
                      "-it",
                      "--newton-iterations",
                      "Maximum iterations for the Newton solve.");

    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        return 1;
    }
    args.PrintOptions(std::cout);
    // 3. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    //    with the same code.
    mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }

    /// Define the q-function
    mfem::ConstantCoefficient* dc=new mfem::ConstantCoefficient(1.0);
    mfem::ConstantCoefficient* lc=new mfem::ConstantCoefficient(1.0);
    mfem::LinDiffQFunc* qfun=new mfem::LinDiffQFunc(*dc,*lc,1.0,1.0,1.0);

    mfem::H1_FECollection fec00(order, dim);
    mfem::L2_FECollection fec01(order, dim);
    mfem::FiniteElementSpace* bfes00=new mfem::FiniteElementSpace(mesh,&fec00,1,mfem::Ordering::byVDIM);
    mfem::FiniteElementSpace* pfes00=new mfem::FiniteElementSpace(mesh,&fec00,1,mfem::Ordering::byVDIM);
    mfem::FiniteElementSpace* pfes01=new mfem::FiniteElementSpace(mesh,&fec01,1,mfem::Ordering::byVDIM);

    /// Define parametric nonlinear form
    mfem::Array<mfem::FiniteElementSpace*> bfes;
    mfem::Array<mfem::FiniteElementSpace*> pfes;

    bfes.Append(bfes00);
    pfes.Append(pfes00);
    pfes.Append(pfes01);

    mfem::PrmBlockNonlinearForm* nf=new mfem::PrmBlockNonlinearForm(bfes,pfes);
    nf->AddDomainIntegrator(new mfem::PrmBlockLSFEMDiffusion(*qfun));

    /// Define the grid functions
    mfem::GridFunction* bgf00=new mfem::GridFunction(bfes00);
    mfem::GridFunction* pgf00=new mfem::GridFunction(pfes00);
    mfem::GridFunction* pgf01=new mfem::GridFunction(pfes01);
    mfem::GridFunction* ggf00=new mfem::GridFunction(pfes00);
    mfem::GridFunction* ggf01=new mfem::GridFunction(pfes01);

    *bgf00=0.0;
    *pgf00=1.0;
    *pgf01=1.0;

    mfem::BlockVector solbv; solbv.Update(nf->GetBlockTrueOffsets()); solbv=0.0;
    mfem::BlockVector resbv; resbv.Update(nf->GetBlockTrueOffsets()); resbv=0.0;
    mfem::BlockVector adjbv; adjbv.Update(nf->GetBlockTrueOffsets()); adjbv=0.0;
    mfem::BlockVector prmbv; prmbv.Update(nf->PrmGetBlockTrueOffsets()); prmbv=1.0;
    mfem::BlockVector grdbv; grdbv.Update(nf->PrmGetBlockTrueOffsets()); grdbv=0.0;

    bgf00->SetFromTrueDofs(solbv.GetBlock(0));
    pgf00->SetFromTrueDofs(prmbv.GetBlock(0));
    pgf01->SetFromTrueDofs(prmbv.GetBlock(1));

    nf->SetPrmFields(prmbv);
    double energy = nf->GetEnergy(solbv);

    nf->Mult(solbv,resbv);
    std::cout<<"Norm res="<<resbv.Norml2()<<std::endl;

    //mfem::Operator& K=nf->GetGradient(solbv);
    std::cout<<"energy ="<< energy<<std::endl;

    nf->SetStateFields(solbv);
    nf->SetAdjointFields(adjbv);
    nf->PrmMult(prmbv,grdbv);


    //set the BC for the physics
    mfem::Array<mfem::Array<int> *> ess_bdr;
    mfem::Array<mfem::Vector*>      ess_rhs;
    ess_bdr.Append(new mfem::Array<int>(mesh->bdr_attributes.Max()));
    ess_rhs.Append(nullptr);
    (*ess_bdr[0]) = 1;
    nf->SetEssentialBC(ess_bdr,ess_rhs);

    //define the solvers
    mfem::UMFPackSolver* umfsolv=new mfem::UMFPackSolver();

    mfem::GMRESSolver *gmres;
    gmres = new mfem::GMRESSolver();
    gmres->SetAbsTol(newton_abs_tol/10);
    gmres->SetRelTol(newton_rel_tol/10);
    gmres->SetMaxIter(100);
    gmres->SetPrintLevel(print_level);
    //gmres->SetPreconditioner(*prec);


    mfem::NewtonSolver *ns;
    ns = new mfem::NewtonSolver();
    ns->iterative_mode = true;
    ns->SetSolver(*gmres);
    ns->SetOperator(*nf);
    ns->SetPrintLevel(print_level);
    ns->SetRelTol(newton_rel_tol);
    ns->SetAbsTol(newton_abs_tol);
    ns->SetMaxIter(newton_iter);

    mfem::Vector b; //RHS is zero
    solbv=0.0;
    ns->Mult(b, solbv);


    nf->SetStateFields(solbv);
    nf->SetAdjointFields(solbv);
    nf->PrmMult(prmbv,grdbv);




    mfem::ParaViewDataCollection *dacol = new mfem::ParaViewDataCollection("Example91",
                                                                      mesh);

    ggf00->SetFromTrueDofs(grdbv.GetBlock(0));
    ggf01->SetFromTrueDofs(grdbv.GetBlock(1));
    pgf00->SetFromTrueDofs(solbv.GetBlock(0));

    dacol->SetLevelsOfDetail(order);
    dacol->RegisterField("sol", pgf00);
    dacol->RegisterField("grad00", ggf00);
    dacol->RegisterField("grad01", ggf01);

    dacol->SetTime(1.0);
    dacol->SetCycle(1);
    dacol->Save();

    delete dacol;

    delete ns;
    delete umfsolv;
    delete gmres;
    delete ess_bdr[0];

    delete bgf00;
    delete pgf00;
    delete pgf01;
    delete ggf00;
    delete ggf01;


    delete nf;
    delete pfes01;
    delete pfes00;
    delete bfes00;

    delete qfun;
    delete lc;
    delete dc;

    delete mesh;

}
