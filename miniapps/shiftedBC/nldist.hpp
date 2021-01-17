#ifndef NLDIST_H
#define NLDIST_H

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include "dist_solver.hpp"

namespace mfem {

//Product of the modulus of the first coefficient and the second coefficient
class PProductCoefficient : public Coefficient
{
private:
    Coefficient *basef, *corrf;

public:
    PProductCoefficient(Coefficient& basec_,Coefficient& corrc_)
    {
        basef=&basec_;
        corrf=&corrc_;
    }

    virtual
    double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        T.SetIntPoint(&ip);
        double u=basef->Eval(T,ip);
        double c=corrf->Eval(T,ip);
        if(u<0.0){ u*=-1.0;}
        return u*c;
    }
};



//Formulation for the  ScreenedPoisson equation
//The possitive part of the input coefficient supply unit volumetric loading
//The negative part - negative unit volumetric loading
//The parameter rh is the radius of a linear cone filter which will deliver
//similar smoothing effect as the Screened Poisson euation
//It determines the length scale of the smoothing.
class ScreenedPoisson: public NonlinearFormIntegrator
{
protected:
    double diffcoef;
    mfem::Coefficient* func;

public:
    ScreenedPoisson(mfem::Coefficient& nfunc, double rh):func(&nfunc)
    {
        double rd=rh/(2*std::sqrt(3.0));
        diffcoef= rd*rd;
    }

    ~ScreenedPoisson()
    {

    }

    void SetInput(mfem::Coefficient& nfunc)
    {
        func=&nfunc;
    }

    virtual double GetElementEnergy(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun) override
    {
        double energy = 0.0;
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
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
        double detJ;
        double ngrad2;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w * w);
            w = ip.weight * w;

            fval=func->Eval(trans,ip);

            el.CalcPhysDShape(trans, B);
            el.CalcPhysShape(trans,shapef);

            B.MultTranspose(elfun,qval);

            ngrad2=0.0;
            for(int jj=0;jj<ndim;jj++)
            {
                ngrad2 = ngrad2 + qval(jj)*qval(jj);
            }

            energy = energy + w * ngrad2 * diffcoef * 0.5;

            //add the external load -1 if fval > 0.0; 1 if fval < 0.0;
            pval=shapef*elfun;

            energy = energy + w * pval * pval * 0.5;

            if(fval>0.0){
                energy = energy - w*pval;
            }else  if(fval<0.0){
                energy = energy + w*pval;
            }
        }

        return energy;
    }

    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &trans,
                                       const Vector &elfun,
                                       Vector &elvect) override
    {

        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
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
        Vector tmpv(ndof);

        B=0.0;
        qval=0.0;

        double w;
        double detJ;
        double ngrad2;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w * w);
            w = ip.weight * w;

            fval=func->Eval(trans,ip);

            el.CalcPhysDShape(trans, B);
            el.CalcPhysShape(trans,shapef);

            B.MultTranspose(elfun,qval);
            B.Mult(qval,lvec);

            elvect.Add(w * diffcoef,lvec);

            pval=shapef*elfun;

            elvect.Add(w * pval, shapef);


            //add the load
            //add the external load -1 if fval > 0.0; 1 if fval < 0.0;
            pval=shapef*elfun;
            if(fval>0.0){
                elvect.Add( -w , shapef);
            }else  if(fval<0.0){
                elvect.Add(  w , shapef);
            }
        }

    }

    virtual void AssembleElementGrad(const FiniteElement &el,
                                        ElementTransformation &trans,
                                        const Vector &elfun,
                                        DenseMatrix &elmat) override
    {
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
        const IntegrationRule *ir = NULL;
        int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
        ir = &IntRules.Get(el.GetGeomType(), order);

        elmat.SetSize(ndof,ndof);
        elmat=0.0;

        Vector shapef(ndof);

        DenseMatrix B(ndof, ndim); //[diff_x,diff_y,diff_z]

        B=0.0;

        double w;
        double detJ;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w * w);
            w = ip.weight * w;

            el.CalcPhysDShape(trans, B);
            el.CalcPhysShape(trans,shapef);

            AddMult_a_VVt(w , shapef, elmat);
            AddMult_a_AAt(w * diffcoef, B, elmat);
        }
    }

};

///the vector coefficent should return a vector eith elements:
/// [0] - derivative with respect to x
/// [1] - derivative with respect to y
/// [2] - derivative with respect to z
class PUMPLaplacian: public NonlinearFormIntegrator
{

protected:
    mfem::Coefficient *func;
    mfem::VectorCoefficient *fgrad;
    bool ownership;
    double pp;
    double ee;

public:
    PUMPLaplacian(mfem::Coefficient* nfunc, mfem::VectorCoefficient* nfgrad, bool ownership_=true)
    {
        func=nfunc;
        fgrad=nfgrad;
        ownership=ownership_;
        pp=2.0;
        ee=1e-7;
    }

    void SetPower(double pp_)
    {
        pp=pp_;
    }

    void SetReg(double ee_)
    {
        ee=ee_;
    }

    virtual
    ~PUMPLaplacian(){
        if(ownership){
            delete func;
            delete fgrad;
        }
    }

    virtual double GetElementEnergy(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun) override
    {
        double energy = 0.0;
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
        const IntegrationRule *ir = NULL;
        int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
        ir = &IntRules.Get(el.GetGeomType(), order);

        Vector shapef(ndof);
        double fval;
        double pval;
        double tval;
        Vector vgrad(ndim);
        DenseMatrix dshape(ndof, ndim);
        DenseMatrix B(ndof, ndim);
        Vector qval(ndim);
        Vector tmpv(ndof);

        B=0.0;

        double w;
        double detJ;

        double ngrad2;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w * w);
            w = ip.weight * w;

            fval=func->Eval(trans,ip);
            fgrad->Eval(vgrad,trans,ip);
            tval=fval;
            if(fval<0.0)
            {
                fval=-fval;
                vgrad*=-1.0;
            }

            el.CalcPhysDShape(trans, dshape);
            el.CalcPhysShape(trans,shapef);

            for(int jj=0;jj<ndim;jj++)
            {
                dshape.GetColumn(jj,tmpv);
                tmpv*=fval;
                tmpv.Add(vgrad[jj],shapef);
                B.SetCol(jj,tmpv);
            }
            B.MultTranspose(elfun,qval);

            ngrad2=0.0;
            for(int jj=0;jj<ndim;jj++)
            {
                ngrad2 = ngrad2 + qval(jj)*qval(jj);
            }

            energy = energy + w * std::pow(ngrad2+ee*ee,pp/2.0)/pp;

            //add the external load -1 if fval > 0.0; 1 if fval < 0.0;
            pval=shapef*elfun;
            if(tval>0.0){
                energy = energy - w * pval * tval;
            }else  if(tval<0.0){
                energy = energy + w * pval * tval;
            }
        }

        return energy;
    }

    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &trans,
                                       const Vector &elfun,
                                       Vector &elvect) override
    {

        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
        const IntegrationRule *ir = NULL;
        int order = 2 * el.GetOrder() + trans.OrderGrad(&el)+1;
        ir = &IntRules.Get(el.GetGeomType(), order);

        elvect.SetSize(ndof);
        elvect=0.0;

        Vector shapef(ndof);
        double fval;
        double pval;
        double tval;
        Vector vgrad(3);

        DenseMatrix dshape(ndof, ndim);
        DenseMatrix B(ndof, ndim); //[diff_x,diff_y,diff_z]

        Vector qval(ndim); //[diff_x,diff_y,diff_z,u]
        Vector lvec(ndof); //residual at ip
        Vector tmpv(ndof);

        B=0.0;
        qval=0.0;

        double w;
        double detJ;
        double ngrad2;
        double aa;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w * w);
            w = ip.weight * w;

            fval=func->Eval(trans,ip);
            fgrad->Eval(vgrad,trans,ip);
            tval=fval;
            if(fval<0.0)
            {
                fval=-fval;
                vgrad*=-1.0;
            }

            el.CalcPhysDShape(trans, dshape);
            el.CalcPhysShape(trans,shapef);

            for(int jj=0;jj<ndim;jj++)
            {
                dshape.GetColumn(jj,tmpv);
                tmpv*=fval;
                tmpv.Add(vgrad[jj],shapef);
                B.SetCol(jj,tmpv);
            }

            B.MultTranspose(elfun,qval);

            ngrad2=0.0;
            for(int jj=0;jj<ndim;jj++)
            {
                ngrad2 = ngrad2 + qval(jj)*qval(jj);
            }

            aa = ngrad2 + ee*ee;
            aa = std::pow(aa, (pp - 2.0) / 2.0);
            B.Mult(qval,lvec);
            elvect.Add(w * aa,lvec);

            //add the load
            //add the external load -1 if tval > 0.0; 1 if tval < 0.0;
            pval=shapef*elfun;
            if(tval>0.0){
                elvect.Add( -w*fval , shapef);
            }else  if(tval<0.0){
                elvect.Add(  w*fval , shapef);
            }
        }

    }

    virtual void AssembleElementGrad(const FiniteElement &el,
                                        ElementTransformation &trans,
                                        const Vector &elfun,
                                        DenseMatrix &elmat) override
    {
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        int spaceDim = trans.GetSpaceDim();
        bool square = (ndim == spaceDim);
        const IntegrationRule *ir = NULL;
        int order = 2 * el.GetOrder() + trans.OrderGrad(&el)+1;
        ir = &IntRules.Get(el.GetGeomType(), order);

        elmat.SetSize(ndof,ndof);
        elmat=0.0;

        Vector shapef(ndof);
        double fval;
        double tval;
        Vector vgrad(ndim);

        Vector qval(ndim); //[diff_x,diff_y,diff_z,u]
        DenseMatrix dshape(ndof, ndim);
        DenseMatrix B(ndof, ndim); //[diff_x,diff_y,diff_z]
        Vector lvec(ndof);
        Vector tmpv(ndof);

        B=0.0;

        double w;
        double detJ;
        double ngrad2;
        double aa;
        double aa0;
        double aa1;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            detJ = (square ? w : w * w);
            w = ip.weight * w;

            fval=func->Eval(trans,ip);
            fgrad->Eval(vgrad,trans,ip);
            tval=fval;
            if(fval<0.0)
            {
                fval=-fval;
                vgrad*=-1.0;
            }

            el.CalcPhysDShape(trans, dshape);
            el.CalcPhysShape(trans,shapef);

            for(int jj=0;jj<ndim;jj++)
            {
                dshape.GetColumn(jj,tmpv);
                tmpv*=fval;
                tmpv.Add(vgrad[jj],shapef);
                B.SetCol(jj,tmpv);
            }

            B.MultTranspose(elfun,qval);
            B.Mult(qval,lvec);

            ngrad2=0.0;
            for(int jj=0;jj<ndim;jj++)
            {
                ngrad2 = ngrad2 + qval(jj)*qval(jj);
            }



            aa = ngrad2 + ee * ee;
            aa1 = std::pow(aa, (pp - 2.0) / 2.0);
            aa0 = (pp-2.0) * std::pow(aa, (pp - 4.0) / 2.0);

            AddMult_a_VVt(w * aa0, lvec, elmat);
            AddMult_a_AAt(w * aa1, B, elmat);
        }
    }

};

class PDEFilter
{
public:
    PDEFilter(mfem::ParMesh& mesh, double rh, int order_=2,
              int maxiter=100, double rtol=1e-7, double atol=1e-15, int print_lv=0)
    {
        int dim=mesh.Dimension();
        lcom=mesh.GetComm();

        rr=rh;

        fecp=new mfem::H1_FECollection(order_,dim);
        fesp=new mfem::ParFiniteElementSpace(&mesh,fecp,1,mfem::Ordering::byVDIM);

        sv = fesp->NewTrueDofVector();
        bv = fesp->NewTrueDofVector();

        gf = new mfem::ParGridFunction(fesp);


        nf=new mfem::ParNonlinearForm(fesp);
        prec=new mfem::HypreBoomerAMG();
        prec->SetPrintLevel(print_lv);

        gmres = new mfem::GMRESSolver(lcom);

        gmres->SetAbsTol(atol);
        gmres->SetRelTol(rtol);
        gmres->SetMaxIter(maxiter);
        gmres->SetPrintLevel(print_lv);
        gmres->SetPreconditioner(*prec);

        K=nullptr;
        sint=nullptr;
    }

    ~PDEFilter()
    {

        delete gmres;
        delete prec;
        delete nf;
        delete gf;
        delete bv;
        delete sv;
        delete fesp;
        delete fecp;
    }

    void Filter(mfem::ParGridFunction& func, mfem::ParGridFunction ffield)
    {
            mfem::GridFunctionCoefficient gfc(&func);
            Filter(gfc,ffield);

    }

    void Filter(mfem::Coefficient& func, mfem::ParGridFunction& ffield)
    {
        if(sint==nullptr)
        {
            sint=new mfem::ScreenedPoisson(func,rr);
            nf->AddDomainIntegrator(sint);
            *sv=0.0;
            K=&(nf->GetGradient(*sv));
            gmres->SetOperator(*K);

        }else{
            sint->SetInput(func);
        }

        //form RHS
        *sv=0.0;
        nf->Mult(*sv,*bv);
        //filter the input field
        gmres->Mult(*bv,*sv);

        gf->SetFromTrueDofs(*sv);

        mfem::GridFunctionCoefficient gfc(gf);
        ffield.ProjectCoefficient(gfc);
    }

private:
    MPI_Comm lcom;
    mfem::H1_FECollection* fecp;
    mfem::ParFiniteElementSpace* fesp;
    mfem::ParNonlinearForm* nf;
    mfem::HypreBoomerAMG* prec;
    mfem::GMRESSolver *gmres;
    mfem::HypreParVector *sv;
    mfem::HypreParVector *bv;

    mfem::ParGridFunction* gf;

    mfem::Operator* K;
    mfem::ScreenedPoisson* sint;
    double rr;

};


class PLapDistanceSolver : public DistanceSolver
{
public:
    PLapDistanceSolver(int maxp_=30,
                       int newton_iter_=10, double rtol=1e-7, double atol=1e-12,
                       int print_lv=0)
    {
        maxp=maxp_;
        newton_iter=newton_iter_;
        newton_rel_tol=rtol;
        newton_abs_tol=atol;
        print_level=print_lv;
    }

    void SetMaxPower(int new_pp)
    {
        maxp=new_pp;
    }

    void DistanceField(mfem::ParGridFunction& gfunc, mfem::ParGridFunction& fdist)
    {
        mfem::GridFunctionCoefficient gfc(&gfunc);
        ComputeDistance(gfc, fdist);
    }

    void ComputeDistance(mfem::Coefficient& func, mfem::ParGridFunction& fdist)
    {
        mfem::ParFiniteElementSpace* fesd=fdist.ParFESpace();

        auto check_h1 = dynamic_cast<const H1_FECollection *>(fesd->FEColl());
        auto check_l2 = dynamic_cast<const L2_FECollection *>(fesd->FEColl());
        MFEM_VERIFY((check_h1 || check_l2) && fesd->GetVDim() == 1,
                    "This solver supports only scalar H1 or L2 spaces.");

        mfem::ParMesh* mesh=fesd->GetParMesh();
        const int dim=mesh->Dimension();

        MPI_Comm lcomm=fesd->GetComm();

        const int order = fesd->GetOrder(0);
        mfem::H1_FECollection fecp(order, dim);
        mfem::ParFiniteElementSpace fesp(mesh, &fecp, 1, mfem::Ordering::byVDIM);

        mfem::ParGridFunction wf(&fesp);
        wf.ProjectCoefficient(func);
        mfem::GradientGridFunctionCoefficient gf(&wf); //gradient of wf


        mfem::ParGridFunction xf(&fesp);
        mfem::HypreParVector *sv = xf.GetTrueDofs();
        *sv=1.0;


        mfem::ParNonlinearForm* nf=new mfem::ParNonlinearForm(&fesp);

        mfem::PUMPLaplacian* pint = new mfem::PUMPLaplacian(&func,&gf,false);
        nf->AddDomainIntegrator(pint);


        pint->SetPower(2);

        //define the solvers
        mfem::HypreBoomerAMG* prec=new mfem::HypreBoomerAMG();
        prec->SetPrintLevel(print_level);


        mfem::GMRESSolver *gmres;
        gmres = new mfem::GMRESSolver(lcomm);
        gmres->SetAbsTol(newton_abs_tol/10);
        gmres->SetRelTol(newton_rel_tol/10);
        gmres->SetMaxIter(100);
        gmres->SetPrintLevel(print_level);
        gmres->SetPreconditioner(*prec);


        mfem::NewtonSolver *ns;
        ns = new mfem::NewtonSolver(lcomm);
        ns->iterative_mode = true;
        ns->SetSolver(*gmres);
        ns->SetOperator(*nf);
        ns->SetPrintLevel(print_level);
        ns->SetRelTol(newton_rel_tol);
        ns->SetAbsTol(newton_abs_tol);
        ns->SetMaxIter(newton_iter);


        mfem::Vector b; //RHS is zero
        ns->Mult(b, *sv);

        for(int pp=3;pp<maxp;pp++)
        {
           std::cout<<"pp="<<pp<<std::endl;
           pint->SetPower(pp);
           ns->Mult(b, *sv);
        }

        xf.SetFromTrueDofs(*sv);
        mfem::GridFunctionCoefficient gfx(&xf);
        mfem::PProductCoefficient tsol(func,gfx);
        fdist.ProjectCoefficient(tsol);


        for (int i = 0; i < fdist.Size(); i++)
        {
           fdist(i) = fabs(fdist(i));
        }

        delete ns;
        delete gmres;
        delete prec;
        delete nf;
        delete sv;
    }

private:
    int maxp; //maximum value of the power p
    double newton_abs_tol;
    double newton_rel_tol;
    int newton_iter;
    int print_level;
};

}
#endif
