#ifndef REMAP_OBJ
#define REMAP_OBJ

#include "mfem.hpp"

class SphCoefficient:public mfem::Coefficient
{
public:
   SphCoefficient(mfem::real_t r_=0.30):r(r_)
   {

   }

   ~SphCoefficient()
   {


   }

   virtual
   mfem::real_t Eval(mfem::ElementTransformation &T,
                     const mfem::IntegrationPoint &ip)
   {
      mfem::Vector tmpv;
      tmpv.SetSize(T.GetSpaceDim());
      T.Transform(ip,tmpv);

      for (int i=0; i<tmpv.Size(); i++)
      {
         tmpv[i]-=0.40;
      }

      mfem::real_t rez=tmpv.Norml2();


      if (rez<r) {return 1.0;}
      else {return 0.0;}
   }


private:
   mfem::real_t r;
};


class QVolFunctional:public mfem::Operator
{
public:
    QVolFunctional(mfem::QuadratureSpace & qes_,mfem::real_t vol_):
        mfem::Operator(1,qes_.GetSize()), qspace(qes_), vol_bound(vol_)
    {

        gr.SetSize(qes_.GetSize());//set the size of the gradient vector
        const int NE  = qes_.GetMesh()->GetNE();
        for (int e = 0; e < NE; e++)
        {
            const mfem::IntegrationRule &ir = qes_.GetElementIntRule(e);
            const int nip = ir.GetNPoints();

            // Transformation of the element with the pos_mesh coordinates.
            mfem::IsoparametricTransformation Tr;
            //check out if the mesh is set up with the correct nodal values
            //qspace.GetMesh()->GetElementTransformation(e, pos_mesh, &Tr);
            qes_.GetMesh()->GetElementTransformation(e,&Tr);

            for (int q = 0; q < nip; q++)
            {
                const mfem::IntegrationPoint &ip = ir.IntPoint(q);
                Tr.SetIntPoint(&ip);
                mfem::real_t w=Tr.Weight();
                gr[e*nip+q]=w;
            }
        }

        grad=new locGrad(gr);
    }

    ~QVolFunctional()
    {
        delete grad;
    }

    virtual
    void Mult(const mfem::Vector &x, mfem::Vector& y) const
    {
        //x is assumed to be a quadrature function

        mfem::real_t loc_vol=mfem::real_t(0.0);

        for(int i=0;i<gr.Size();i++)
        {
            loc_vol=loc_vol+x[i]*gr[i];
        }

        auto pmesh = dynamic_cast<mfem::ParMesh*>(qspace.GetMesh());
        MFEM_VERIFY(pmesh, "Broken QuadratureSpace.");

        double dloc_vol=loc_vol;
        double dglb_vol=0.0;
        MPI_Allreduce(&dloc_vol,&dglb_vol,1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

        y[0]=mfem::real_t(dglb_vol)-vol_bound;
    }


    //evaluate the gradient of the functional
    virtual
    mfem::Operator& GetGradient(const mfem::Vector& x) const
    {
        return *grad;
    }


private:
    mfem::QuadratureSpace& qspace;
    mfem::real_t vol_bound;
    mfem::Vector gr; //the gradient is constant vector

    class locGrad:public mfem::Operator
    {
    public:
        locGrad(mfem::Vector& gr_)
            :mfem::Operator(gr_.Size(),gr_.Size()), lgr(gr_)
        {
        }

        virtual
        void Mult(const mfem::Vector &x, mfem::Vector& y) const
        {
            y=lgr;
        }

    private:
        mfem::Vector& lgr;
    };

    locGrad* grad;
};

class QL2Objective:public mfem::Operator
{
public:
    QL2Objective(mfem::QuadratureSpace & qes_):mfem::Operator(1,qes_.GetSize())
    {

    }

    ~QL2Objective()
    {

    }

    virtual
    void Mult(const mfem::Vector &x, mfem::Vector& y) const
    {

    }
private:
};


class VolFunctional:public mfem::Operator
{
public:
    VolFunctional(mfem::ParFiniteElementSpace& pfes_,
                  double vol_): mfem::Operator(1,pfes_.GetTrueVSize())
                                ,pfes(&pfes_),vol(vol_)
    {
        l=new mfem::ParLinearForm(pfes);
        l->AddDomainIntegrator(new mfem::DomainLFIntegrator(one,
                                                            pfes->GetMaxElementOrder()));
        l->Assemble();
        gf=new mfem::ParGridFunction(pfes);

        mfem::Vector y; y.SetSize(pfes->GetTrueVSize());
        l->ParallelAssemble(y);

        mfem::DenseMatrix* dmat=new mfem::DenseMatrix(1,pfes_.GetTrueVSize());
        dmat->SetRow(0,y);
        grad=dmat;
    }

    ~VolFunctional()
    {
        delete l;
        delete gf;
        delete grad;
    }

    //evaluate the functional
    virtual
    void Mult(const mfem::Vector &x, mfem::Vector& y) const
    {
        gf->SetFromTrueDofs(x);
        y[0]=(*l)(*gf)-vol;
    }


    //evaluate the gradient of the functional
    virtual
    mfem::Operator& GetGradient(const mfem::Vector& x) const
    {
        return *grad;
    }

    void Test()
    {
        mfem::ConstantCoefficient cc(0.5);
        mfem::ParGridFunction lgf(pfes);
        lgf.ProjectCoefficient(cc);

        mfem::Vector x=lgf.GetTrueVector();
        mfem::Vector p; p.SetSize(x.Size()); p.Randomize();
        mfem::Vector g; g.SetSize(x.Size());
        mfem::Vector tmpv; tmpv.SetSize(x.Size());

        mfem::Vector r(1);

        this->Mult(x,r);
        double lo=r[0];
        this->GetGradient(x).Mult(x,g);

        mfem::real_t nd=mfem::InnerProduct(pfes->GetComm(),p,p);
        mfem::real_t td=mfem::InnerProduct(pfes->GetComm(),p,g);

        td=td/nd;

        double lsc=1.0;
        double lqoi;

        for (int l=0; l<10; l++)
        {
           lsc/=10.0;
           p/=10.0;
           add(p,x,tmpv);
           this->Mult(tmpv,r);
           lqoi=r[0];
           mfem::real_t ld=(lqoi-lo)/lsc;
           if (pfes->GetMyRank()==0)
           {
              std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                        << " gradient=" << td
                        << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
        }
    }


private:

    mfem::ConstantCoefficient one;
    mfem::ParFiniteElementSpace* pfes;
    double vol;

    mfem::ParLinearForm* l;
    mfem::ParGridFunction* gf;

    mfem::Operator* grad;
};

class L2Objective:public mfem::Operator
{
public:
    L2Objective(mfem::ParFiniteElementSpace& pfes_,
                mfem::ParGridFunction& tgf_)
                        :mfem::Operator(1,pfes_.GetTrueVSize())
                        ,tgf(&tgf_)
    {
        grad=new locGrad(pfes_,tgf_);
        cgf=new mfem::ParGridFunction(&pfes_);
    }

    ~L2Objective()
    {
       delete grad;
    }

    //evaluate the functional
    virtual
    void Mult(const mfem::Vector &x, mfem::Vector& y) const
    {
        mfem::GridFunctionCoefficient gfc;
        cgf->SetFromTrueDofs(x);
        gfc.SetGridFunction(cgf);
        mfem::real_t vv=tgf->ComputeL2Error(gfc);
        y[0]=0.5*vv*vv;
    }

    //evaluate the gradient of the functional
    virtual
    mfem::Operator& GetGradient(const mfem::Vector& x) const
    {
        return *grad;
    }

    void Test()
    {
        mfem::ConstantCoefficient cc(0.5);
        mfem::ParGridFunction lgf(*tgf);
        lgf.ProjectCoefficient(cc);

        mfem::Vector x=lgf.GetTrueVector();
        mfem::Vector p; p.SetSize(x.Size()); p.Randomize();
        mfem::Vector g; g.SetSize(x.Size());
        mfem::Vector tmpv; tmpv.SetSize(x.Size());

        mfem::Vector r(1);


        this->Mult(x,r);
        double lo=r[0];
        this->GetGradient(x).Mult(x,g);

        mfem::real_t nd=mfem::InnerProduct(cgf->ParFESpace()->GetComm(),p,p);
        mfem::real_t td=mfem::InnerProduct(cgf->ParFESpace()->GetComm(),p,g);

        td=td/nd;

        double lsc=1.0;
        double lqoi;

        for (int l=0; l<10; l++)
        {
           lsc/=10.0;
           p/=10.0;

           add(p,x,tmpv);
           this->Mult(tmpv,r);
           lqoi=r[0];
           mfem::real_t ld=(lqoi-lo)/lsc;
           if (cgf->ParFESpace()->GetMyRank()==0)
           {
              std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                        << " gradient=" << td
                        << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
        }
    }

private:

    mutable mfem::ParGridFunction* cgf;
    mfem::ParGridFunction* tgf;

    class locGrad:public mfem::Operator
    {
    public:
        locGrad(mfem::ParFiniteElementSpace& pfes_,
                mfem::ParGridFunction& tgf_)
                    :mfem::Operator(pfes_.GetTrueVSize())
        {
            bf=new mfem::ParBilinearForm(&pfes_);
            bf->AddDomainIntegrator(new mfem::MassIntegrator());
            bf->Assemble();

            tgv.SetSize(pfes_.GetTrueVSize());
            bf->Mult(tgf_.GetTrueVector(),tgv);
        }


        virtual
        void Mult(const mfem::Vector &x, mfem::Vector& y) const
        {
            bf->Mult(x,y);
            y-=tgv;
        }

        virtual
        mfem::Operator& GetGradient(const mfem::Vector& x) const
        {
            return *bf;
        }
    private:
        mfem::ParBilinearForm* bf;
        mfem::Vector tgv;
    };

    locGrad* grad;

};




#endif
