#ifndef STOKES_HPP
#define STOKES_HPP

#include "mfem.hpp"


namespace mfem {

namespace PointwiseTrans
{

/*  Standrd "Heaviside" projection in topology optimization with threhold eta
 * and steepness of the projection beta.
 * */
inline
double HProject(double rho, double eta, double beta)
{
    // tanh projection - Wang&Lazarov&Sigmund2011
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double c=std::tanh(beta*(rho-eta));
    double rez=(a+c)/(a+b);
    return rez;
}

/// Gradient of the "Heaviside" projection with respect to rho.
inline
double HGrad(double rho, double eta, double beta)

{
    double c=std::tanh(beta*(rho-eta));
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double rez=beta*(1.0-c*c)/(a+b);
    return rez;
}

/// Second derivative of the "Heaviside" projection with respect to rho.
inline
double HHess(double rho,double eta, double beta)
{
    double c=std::tanh(beta*(rho-eta));
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double rez=-2.0*beta*beta*c*(1.0-c*c)/(a+b);
    return rez;
}


inline
double FluidInterpolation(double rho,double q)
{
    return q*(1.0-rho)/(q+rho);
}

inline
double GradFluidInterpolation(double rho, double q)
{
    double tt=q+rho;
    return -q/tt-q*(1.0-rho)/(tt*tt);
}


}




// Taylor-Hood finite elements
class StokesIntegratorTH:public mfem::BlockNonlinearFormIntegrator
{
public:
    StokesIntegratorTH()
    {
        mu=nullptr;
        bc=nullptr;
        ff=nullptr;

        ss.SetSize(13);
        rr.SetSize(13);
        mm.SetSize(13);
    }

    StokesIntegratorTH(mfem::Coefficient* mu_, mfem::Coefficient* bc_, mfem::VectorCoefficient* ff_)
    {
        mu=mu_;
        bc=bc_;
        ff=ff_;

        ss.SetSize(13);
        rr.SetSize(13);
        mm.SetSize(13);
    }

    virtual
    double GetElementEnergy(const Array<const FiniteElement *> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elfun)
    {
        return 0.0;
    }

    virtual
    void AssembleElementVector(const Array<const FiniteElement *> &el,
                               ElementTransformation &Tr,
                               const Array<const Vector *> &elfun,
                               const Array<Vector *> &elvec)
    {
        int dof_u = el[0]->GetDof();
        int dof_p = el[1]->GetDof();

        int dim =  el[0]->GetDim();

        elvec[0]->SetSize(dim*dof_u);
        elvec[1]->SetSize(dof_p);

        int spaceDim = Tr.GetDimension();
        if (dim != spaceDim)
        {
           mfem::mfem_error("StokesIntegrator::AssembleElementVector"
                            " is not defined on manifold meshes");
        }

        // gradients
        bsu.SetSize(dof_u,4);
        bsp.SetSize(dof_p,1);

        Vector uu(elfun[0]->GetData()+0*dof_u, dof_u);
        Vector vv(elfun[0]->GetData()+1*dof_u, dof_u);

        Vector ru(elvec[0]->GetData()+0*dof_u, dof_u); ru=0.0;
        Vector rv(elvec[0]->GetData()+1*dof_u, dof_u); rv=0.0;

        Vector ww;
        Vector rw;
        if(dim==2){
            ww.SetSize(dof_u); ww=0.0;
        }
        else{
            ww.SetDataAndSize(elfun[0]->GetData()+2*dof_u, dof_u);
            rw.SetDataAndSize(elvec[0]->GetData()+2*dof_u, dof_u); rw=0.0;
        }

        Vector pp(elfun[1]->GetData(), dof_p);
        Vector rp(elvec[1]->GetData(), dof_p); rp=0.0;

        // temp storages for vectors and matrices
        Vector sh;
        DenseMatrix dh;

        const IntegrationRule *ir = nullptr;
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);

        double bpenal; // Brinkmann penalization
        double mmu;
        Vector fv(3); fv=0.0;
        double w;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ip = ir->IntPoint(i);
           Tr.SetIntPoint(&ip);
           w=Tr.Weight();
           w = ip.weight * w;

           sh.SetDataAndSize(bsu.GetData(),dof_u);
           el[0]->CalcPhysShape(Tr,sh);
           sh.SetDataAndSize(bsp.GetData(),dof_p);
           el[1]->CalcPhysShape(Tr,sh);

           dh.UseExternalData(bsu.GetData()+dof_u,dof_u,dim);
           el[0]->CalcPhysDShape(Tr,dh);
           if(dim=2){
               sh.SetDataAndSize(bsu.GetData()+3*dof_u,dof_u);
               sh=0.0;
           }

           sh.SetDataAndSize(ss.GetData(),4);
           bsu.MultTranspose(uu,sh);
           sh.SetDataAndSize(ss.GetData()+4,4);
           bsu.MultTranspose(vv,sh);
           sh.SetDataAndSize(ss.GetData()+8,4);
           if(dim==3){
               bsu.MultTranspose(ww,sh);}
           else{
               sh=0.0;}
           sh.SetDataAndSize(ss.GetData()+12,1);
           bsp.MultTranspose(pp,sh);

           mmu=1.0;
           if(mu!=nullptr){
               mmu=mu->Eval(Tr,ip);}

           bpenal=0.0;
           if(bc!=nullptr){
               bpenal=bc->Eval(Tr,ip);}

           fv=0.0;
           if(ff!=nullptr){
               ff->Eval(fv,Tr,ip);}

           EvalQres(mmu,bpenal,fv[0],fv[1],fv[2],ss.GetData(),rr.GetData());

           sh.SetDataAndSize(rr.GetData(),4);
           bsu.AddMult_a(w,sh,ru);
           sh.SetDataAndSize(rr.GetData()+4,4);
           bsu.AddMult_a(w,sh,rv);
           if(dim==3){
               sh.SetDataAndSize(rr.GetData()+8,4);
               bsu.AddMult_a(w,sh,rw);
           }
           sh.SetDataAndSize(rr.GetData()+12,1);
           bsp.AddMult_a(w,sh,rp);
        }

    }

    virtual
    void AssembleElementGrad(const Array<const FiniteElement *> &el,
                             ElementTransformation &Tr,
                             const Array<const Vector *> &elfun,
                             const Array2D<DenseMatrix *> &elmats)
    {
        int dof_u = el[0]->GetDof();
        int dof_p = el[1]->GetDof();

        int dim =  el[0]->GetDim();

        int spaceDim = Tr.GetDimension();
        if (dim != spaceDim)
        {
           mfem::mfem_error("StokesIntegrator::AssembleElementVector"
                            " is not defined on manifold meshes");
        }

        elmats(0,0)->SetSize(dof_u*dim);
        elmats(0,1)->SetSize(dof_u*dim,dof_p);
        elmats(1,0)->SetSize(dof_p,dim*dof_u);
        elmats(1,1)->SetSize(dof_p,dof_p);

        (*elmats(0,0))=0.0;
        (*elmats(0,1))=0.0;
        (*elmats(1,0))=0.0;
        (*elmats(1,1))=0.0;

        // gradients
        DenseMatrix bsu(dof_u,4);
        DenseMatrix bsp(dof_p,1);

        mm.SetSize(13,13); // state matrix

        // temp storages for vectors and matrices
        Vector sh;
        DenseMatrix mh;
        DenseMatrix th;
        DenseMatrix rh;
        DenseMatrix dh;

        const IntegrationRule *ir = nullptr;
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);


        double mmu;
        double bpenal; // Brinkmann penalization
        double w;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ip = ir->IntPoint(i);
           Tr.SetIntPoint(&ip);
           w=Tr.Weight();

           // Primal
           sh.SetDataAndSize(bsu.GetData(),dof_u);
           el[0]->CalcPhysShape(Tr,sh);
           sh.SetDataAndSize(bsp.GetData(),dof_p);
           el[1]->CalcPhysShape(Tr,sh);


           // Gradients
           dh.UseExternalData(bsu.GetData()+dof_u,dof_u,dim);
           el[0]->CalcPhysDShape(Tr,dh);

           if(dim=2){
               sh.SetDataAndSize(bsu.GetData()+3*dof_u,dof_u);
               sh=0.0;
           }

           mmu=1.0;
           if(mu!=nullptr)
           {
               mmu=mu->Eval(Tr,ip);
           }

           bpenal=0.0;
           if(bc!=nullptr){
               bpenal=bc->Eval(Tr,ip);}

           EvalQMat(mmu,bpenal,mm.GetData());

           w = ip.weight * w;

           th.SetSize(dof_u,4);
           rh.SetSize(dof_u);
           mh.SetSize(4,4);
           for(int ii=0;ii<dim;ii++){
           for(int jj=0;jj<dim;jj++){
               mh.CopyMN(mm,4,4,ii*4,jj*4);
               mh.Transpose();
               MultABt(bsu,mh,th);
               MultABt(th,bsu,rh);
               elmats(0,0)->AddMatrix(w,rh,ii*dof_u,jj*dof_u);
           }}

           th.SetSize(dof_u,1);
           rh.SetSize(dof_u,dof_p);
           mh.SetSize(4,1);
           for(int jj=0;jj<dim;jj++){
               mh.CopyMN(mm,4,1,jj*4,12);
               mh.Transpose();
               MultABt(bsu,mh,th);
               MultABt(th,bsp,rh);
               elmats(0,1)->AddMatrix(w,rh,jj*dof_u,0);
           }

           th.SetSize(dof_p,1);
           rh.SetSize(dof_p,dof_p);
           mh.SetSize(1,1);
           mh.CopyMN(mm,1,1,12,12);
           mh.Transpose();
           MultABt(bsp,mh,th);
           MultABt(th,bsp,rh);
           elmats(1,1)->AddMatrix(w,rh,0,0);

        }

        elmats(1,0)->CopyMNt(*elmats(0,1),0,0);
    }



private:
    mfem::Coefficient* mu;
    mfem::Coefficient* bc;
    mfem::VectorCoefficient* ff;

    mfem::DenseMatrix bsu;
    mfem::DenseMatrix bsp;
    DenseMatrix mm;
    Vector rr;
    Vector ss;

    void EvalQres(double mmu, double bpenal,
                  double fx, double fy, double fz,
                  double* uu, double* rr)
    {
        double t7,t9,t16;
        t7 = mmu*(uu[2]+uu[5]);
        t9 = mmu*(uu[3]+uu[9]);
        t16 = mmu*(uu[7]+uu[10]);
        rr[0] = bpenal*uu[0]-fx;
        rr[1] = 2.0*mmu*uu[1]-uu[12];
        rr[2] = t7;
        rr[3] = t9;
        rr[4] = bpenal*uu[4]-fy;
        rr[5] = t7;
        rr[6] = 2.0*mmu*uu[6]-uu[12];
        rr[7] = t16;
        rr[8] = bpenal*uu[8]-fz;
        rr[9] = t9;
        rr[10] = t16;
        rr[11] = 2.0*mmu*uu[11]-uu[12];
        rr[12] = -uu[1]-uu[6]-uu[11];
    }

    void EvalQMat(double mmu, double bpenal, double* kmat)
    {
        for(int i=0;i<169;i++){
            kmat[i]=0.0;
        }
        double t1 = 2.0*mmu;
        kmat[0] = bpenal;
        kmat[14] = t1;
        kmat[25] = -1.0;
        kmat[28] = mmu;
        kmat[31] = mmu;
        kmat[42] = mmu;
        kmat[48] = mmu;
        kmat[56] = bpenal;
        kmat[67] = mmu;
        kmat[70] = mmu;
        kmat[84] = t1;
        kmat[90] = -1.0;
        kmat[98] = mmu;
        kmat[101] = mmu;
        kmat[112] = bpenal;
        kmat[120] = mmu;
        kmat[126] = mmu;
        kmat[137] = mmu;
        kmat[140] = mmu;
        kmat[154] = t1;
        kmat[155] = -1.0;
        kmat[157] = -1.0;
        kmat[162] = -1.0;
        kmat[167] = -1.0;
    }

};


class FluidInterpolationCoefficient: public mfem::Coefficient
{
public:
    FluidInterpolationCoefficient()
    {
        eta=0.5;
        beta=8.0;
        q=1.0;
        lambda=100;
    }

    FluidInterpolationCoefficient(double eta_, double beta_,
                                  double q_, double lambda_, mfem::GridFunction& gf):gfc(&gf)
    {
        eta=eta_;
        beta=beta_;
        q=q_;
        lambda=lambda_;
    }

    virtual
    double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double rhop=PointwiseTrans::HProject(gfc.Eval(T,ip),eta,beta);
        return lambda*PointwiseTrans::FluidInterpolation(rhop,q);

    }

    virtual
    double GradEval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double rhoo=gfc.Eval(T,ip);
        double rhop=PointwiseTrans::HProject(rhoo,eta,beta);
        double g1=lambda*PointwiseTrans::GradFluidInterpolation(rhop,q);
        double g2=PointwiseTrans::HGrad(rhoo,eta,beta);
        return g1*g2;

    }

    void SetGridFunction(mfem::GridFunction* gf)
    {
        gfc.SetGridFunction(gf);
    }

    void SetGridFunction(mfem::GridFunction& gf)
    {
        gfc.SetGridFunction(&gf);
    }


private:
    mfem::GridFunctionCoefficient gfc;
    double eta;  // threshold level
    double beta; // steepness of the projection
    double q; // interpolation parameter
    double lambda; // penalization

};

class StokesGradIntergrator: public mfem::LinearFormIntegrator
{
public:
    StokesGradIntergrator(mfem::GridFunction& velocity,
                          mfem::GridFunction& adjoint,
                          mfem::FluidInterpolationCoefficient& bcoef,
                          int intorder_)
        :vel(velocity), adj(adjoint), brm(bcoef)
    {
        intorder=intorder_; // should be equal to the order of the velocity field
    }

    virtual
    void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, mfem::Vector &elvect)
    {
         int dof = el.GetDof();
         int dim = el.GetDim();
         elvect.SetSize(dof);
         elvect=0.0;

         const mfem::IntegrationRule *ir = nullptr;
         int order= 2 * intorder + Tr.OrderGrad(&el);
         ir=&IntRules.Get(Tr.GetGeometryType(),order);

         mfem::Vector sh(dof);
         mfem::Vector vv(dim);
         mfem::Vector aa(dim);

         double gbpenal; //gradient of the Brinkmann penalization
         double dp;
         double w;


         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w=Tr.Weight();
            w = ip.weight * w;

            gbpenal=brm.GradEval(Tr,ip);

            vel.GetVectorValue(Tr,ip,vv);
            adj.GetVectorValue(Tr,ip,aa);

            el.CalcPhysShape(Tr,sh);

            for(int i=0;i<dof;i++)
            {
                dp=vv*aa;
                elvect[i]=elvect[i]-w*sh[i]*gbpenal*dp;
            }

         }
    }

private:
    mfem::GridFunction& vel;
    mfem::GridFunction& adj;
    mfem::FluidInterpolationCoefficient& brm;
    int intorder;

};



class StokesSolver
{
public:
    StokesSolver(mfem::ParMesh* pmesh_, int vorder=2)
    {
        if(vorder<2){vorder=2;}
        int porder=vorder-1;
        pmesh=pmesh_;

        int dim=pmesh->Dimension();

        vfec=new H1_FECollection(vorder,dim);
        pfec=new H1_FECollection(porder,dim);


        vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim, Ordering::byVDIM);
        pfes=new mfem::ParFiniteElementSpace(pmesh,pfec);

        sfes.Append(vfes);
        sfes.Append(pfes);

        bpenal=nullptr;
        load=nullptr;
        viscosity=nullptr;

        mfem::Array<mfem::ParFiniteElementSpace*> pf;
        pf.Append(vfes);
        pf.Append(pfes);

        nf=new mfem::ParBlockNonlinearForm(pf);
        nfin=nullptr;

        rhs.Update(nf->GetBlockTrueOffsets()); rhs=0.0;
        sol.Update(nf->GetBlockTrueOffsets()); sol=0.0;
        adj.Update(nf->GetBlockTrueOffsets()); adj=0.0;

        fvelocity.SetSpace(vfes); fvelocity=0.0;
        fpressure.SetSpace(pfes); fpressure=0.0;
        avelocity.SetSpace(vfes); avelocity=0.0;


        pmat=nullptr;
        prec=nullptr;
        psol=nullptr;

        dfes=nullptr;
        ltopopt.fcoef=nullptr;
        SetDesignParameters();

        SetSolver();

    }

    ~StokesSolver()
    {
        delete ltopopt.fcoef;

        delete psol;
        delete prec;
        delete pmat;

        delete nf;

        delete vfes;
        delete pfes;
        delete vfec;
        delete pfec;

    }

    void SetViscosity(mfem::Coefficient& coef_)
    {
        viscosity=&coef_;
    }

    // Set the penalization field to coef_.
    void SetBrinkmanPenal(mfem::Coefficient& coef_)
    {
        bpenal=&coef_;
    }

    void SetVolForces(mfem::VectorCoefficient& load_)
    {
        load=&load_;
    }

    void SetSolver(double rtol=1e-8, double atol=1e-12,int miter=1000, int prt_level=1)
    {
        rel_tol=rtol;
        abs_tol=atol;
        max_iter=miter;
        print_level=prt_level;
    }


    /// Solves the forward problem.
    void FSolve();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::BlockVector& rhs);

    /// Return adj*d(residual(sol))/d(design). The dimension
    /// of the vector grad is the save as the dimension of the
    /// true design vector.
    void GradD(mfem::Vector& grad);

    mfem::ParGridFunction& GetVelocity()
    {
        fvelocity.SetFromTrueDofs(sol.GetBlock(0));
        return fvelocity;
    }

    mfem::ParGridFunction& GetPressure()
    {
        fpressure.SetFromTrueDofs(sol.GetBlock(1));
        return fpressure;
    }

    void AddVelocityBC(int id, int dir, double val)
    {
        if(dir==0){
            bcx[id]=mfem::ConstantCoefficient(val);
            AddVelocityBC(id,dir,bcx[id]);
        }
        if(dir==1){
            bcy[id]=mfem::ConstantCoefficient(val);
            AddVelocityBC(id,dir,bcy[id]);

        }
        if(dir==2){
            bcz[id]=mfem::ConstantCoefficient(val);
            AddVelocityBC(id,dir,bcz[id]);
        }
        if(dir==4){
            bcx[id]=mfem::ConstantCoefficient(val);
            bcy[id]=mfem::ConstantCoefficient(val);
            bcz[id]=mfem::ConstantCoefficient(val);
            AddVelocityBC(id,0,bcx[id]);
            AddVelocityBC(id,1,bcy[id]);
            AddVelocityBC(id,2,bcz[id]);
        }
    }

    void AddVelocityBC(int id, int dir, mfem::Coefficient& val)
    {
        if(dir==0){ bccx[id]=&val; }
        if(dir==1){ bccy[id]=&val; }
        if(dir==2){ bccz[id]=&val; }
        if(dir==4){ bccx[id]=&val; bccy[id]=&val; bccz[id]=&val;}
        if(pmesh->Dimension()==2)
        {
            bccz.clear();
        }
    }


    mfem::BlockVector& GetSol(){return sol;}
    mfem::BlockVector& GetAdj(){return adj;}

    mfem::ParFiniteElementSpace* GetVelocityFES(){return vfes;}
    mfem::ParFiniteElementSpace* GetPressureFES(){return pfes;}

    ///Get state spaces
    mfem::Array<mfem::ParFiniteElementSpace*>& GetStateFES()
    {
        return sfes;
    }

    void SetDesignSpace(mfem::ParFiniteElementSpace* dfes_)
    {
        dfes=dfes_;
        density.SetSpace(dfes);
        density=0.0;

        if(ltopopt.fcoef!=nullptr)
        {
            delete ltopopt.fcoef;
        }

        ltopopt.fcoef=new FluidInterpolationCoefficient(ltopopt.eta,ltopopt.beta,
                                                        ltopopt.q,ltopopt.lambda,density);
        bpenal=ltopopt.fcoef;
    }

    // Sets the design field using true vector designvec
    // and set the penalization field
    void SetDesign(mfem::Vector& designvec)
    {
        density.SetFromTrueDofs(designvec);
        if(ltopopt.fcoef!=nullptr)
        {
            delete ltopopt.fcoef;
        }

        ltopopt.fcoef=new FluidInterpolationCoefficient(ltopopt.eta,ltopopt.beta,
                                                        ltopopt.q,ltopopt.lambda,density);
        bpenal=ltopopt.fcoef;
    }

    void SetDesignParameters(double eta_=0.5, double beta_=8.0, double q_=1, double lambda_=100)
    {
        ltopopt.eta=eta_;
        ltopopt.beta=beta_;
        ltopopt.lambda=lambda_;
        ltopopt.q=q_;
    }


private:
    double mu;
    double alpha;
    mfem::Coefficient* viscosity;
    mfem::Coefficient* bpenal;
    mfem::VectorCoefficient* load;

    mfem::ParMesh* pmesh;

    mfem::ParFiniteElementSpace* vfes;
    mfem::ParFiniteElementSpace* pfes;
    mfem::FiniteElementCollection* vfec;
    mfem::FiniteElementCollection* pfec;
    mfem::Array<mfem::ParFiniteElementSpace*> sfes;

    // boundary conditions
    std::map<int, mfem::ConstantCoefficient> bcx;
    std::map<int, mfem::ConstantCoefficient> bcy;
    std::map<int, mfem::ConstantCoefficient> bcz;

    std::map<int, mfem::Coefficient*> bccx;
    std::map<int, mfem::Coefficient*> bccy;
    std::map<int, mfem::Coefficient*> bccz;

    mfem::Array<int> ess_tdofv;


    mfem::BlockNonlinearFormIntegrator* nfin;
    mfem::ParBlockNonlinearForm* nf;

    mfem::BlockVector rhs;
    mfem::BlockVector sol;
    mfem::BlockVector adj;

    mfem::ParGridFunction fvelocity;
    mfem::ParGridFunction avelocity;
    mfem::ParGridFunction fpressure;

    /// The PETSc objects are allocated once the problem is
    /// assembled. They are utilized in computing the adjoint
    /// solutions.
    mfem::PetscParMatrix* pmat;
    mfem::PetscPreconditioner* prec;
    mfem::PetscLinearSolver*   psol;

    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;

    mfem::ParFiniteElementSpace* dfes; //design space
    mfem::ParGridFunction density;

    struct{
      double eta;
      double beta;
      double lambda;
      double q;
      mfem::FluidInterpolationCoefficient* fcoef;
    } ltopopt;

};


}


#endif
