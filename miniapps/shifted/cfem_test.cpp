#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "marking.hpp"
#include "mtop_solvers.hpp"
#include "mtop_filters.hpp"


using namespace mfem;
using namespace std;

class GyroidCoeff:public Coefficient
{
public:
    GyroidCoeff(double cell_size=1.0){
        ll=cell_size;
    }

    virtual
    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);
        double x = xx[0]*ll;
        double y = xx[1]*ll;
        double z = (xx.Size()==3) ? xx[2]*ll : 0.0;

        double r=std::sin(x)*std::cos(y) +
                std::sin(y)*std::cos(z) +
               std::sin(z)*std::cos(x) ;

        if(r>0.0){return 1.0;}
        return -1.0;
    }

private:
    double ll;
};


class CheseCoeff:public Coefficient
{
public:
    CheseCoeff(double cell_size=1.0)
    {
        ll=cell_size;
    }

    virtual
    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        double x = xx[0]*ll;
        double y = xx[1]*ll;
        double z = (xx.Size()==3) ? xx[2]*ll : 0.0;

        double r=std::cos(x)*std::cos(y)*std::cos(z)-0.1;

        if(r>0.0){return -1.0;}
        return 1.0;

    }

private:
    double ll;

};

class DispSol2D:public VectorCoefficient
{
public:
    DispSol2D(double E_=1.0, double nu_=0.3,
              double lx_=1.0,double ly_=1.0):VectorCoefficient(2)
    {
        E=E_;
        nu=nu_;
        lx=lx_;
        ly=ly_;

    }

    virtual
    void  Eval(Vector &V, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        V.SetSize(2);

        V[0] = sin(lx*xx[0])+cos(ly*xx[1]);
        V[1] = cos(lx*xx[0]+ly*xx[1]);
    }

private:

    double E;
    double nu;
    double lx;
    double ly;
};

class StressSol2D:public MatrixCoefficient
{
public:
    StressSol2D(double E_=1.0, double nu_=0.3,
                double lx_=1.0,double ly_=1.0):MatrixCoefficient(2)
    {
        E=E_;
        nu=nu_;
        lx=lx_;
        ly=ly_;
    }

    virtual
    void  Eval(DenseMatrix &ss, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);


        ss(0,0) = 2.0*E/(2.0+2.0*nu)*lx*cos(lx*xx[0])
                +E*nu/(1.0+nu)/(1.0-2.0*nu)*(lx*cos(lx*xx[0])-ly*sin(lx*xx[0]+ly*xx[1]));
        ss(0,1) = -E*(ly*sin(ly*xx[1])+lx*sin(lx*xx[0]+ly*xx[1]))/(2.0+2.0*nu);
        ss(1,0) = -E*(ly*sin(ly*xx[1])+lx*sin(lx*xx[0]+ly*xx[1]))/(2.0+2.0*nu);
        ss(1,1) = -2.0*E/(2.0+2.0*nu)*ly*sin(lx*xx[0]+ly*xx[1])
                +E*nu/(1.0+nu)/(1.0-2.0*nu)*(lx*cos(lx*xx[0])-ly*sin(lx*xx[0]+ly*xx[1]));
    }

private:

    double E;
    double nu;
    double lx;
    double ly;

};

class BdrLoadSol2D:public VectorCoefficient
{
public:
    BdrLoadSol2D(StressSol2D* sco_, ParGridFunction* lsf_):VectorCoefficient(2)
    {
        sco=sco_;
        lsf=lsf_;
    }
    virtual
    void  Eval(Vector &vv, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        vv.SetSize(2);
        Vector n(2);
        DenseMatrix ss(2);
        sco->Eval(ss,T,ip);
        T.SetIntPoint(&ip);
        lsf->GetGradient(T,n);
        double nr=n.Norml2();
        n/=-nr;
        ss.Mult(n,vv);
    }


private:
    StressSol2D* sco;
    ParGridFunction* lsf;


};

class ForceSol2D:public VectorCoefficient
{
public:
    ForceSol2D(double E_=1.0, double nu_=0.3,
               double lx_=1.0,double ly_=1.0):VectorCoefficient(2)
    {
         E=E_;
         nu=nu_;
         lx=lx_;
         ly=ly_;
    }

    virtual
    void  Eval(Vector &ff, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        ff.SetSize(2);

        ff[0] = 2.0*E/(2.0+2.0*nu)*lx*lx*sin(lx*xx[0])
                -E*nu/(1.0+nu)/(1.0-2.0*nu)*(-lx*lx*sin(lx*xx[0])-ly*lx*cos(lx*xx[0]+ly*xx[1]))
                +E*(ly*ly*cos(ly*xx[1])+ly*lx*cos(lx*xx[0]+ly*xx[1]))/(2.0+2.0*nu);

        ff[1] = E*lx*lx*cos(lx*xx[0]+ly*xx[1])/(2.0+2.0*nu)
                +2.0*E/(2.0+2.0*nu)*ly*ly*cos(lx*xx[0]+ly*xx[1])
                +E*nu/(1.0+nu)/(1.0-2.0*nu)*ly*ly*cos(lx*xx[0]+ly*xx[1]);
    }

private:

    double E;
    double nu;
    double lx;
    double ly;
};

class DispSol3D:public VectorCoefficient
{
public:
    DispSol3D(double E_=1.0, double nu_=0.3,
              double lx_=1.0,double ly_=1.0, double lz_=1.0):VectorCoefficient(3)
    {
        E=E_;
        nu=nu_;
        lx=lx_;
        ly=ly_;
        lz=lz_;

    }

    virtual
    void  Eval(Vector &u, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        u.SetSize(3);

        u[0] = sin(lx*xx[0])+cos(ly*xx[1]);
        u[1] = sin(lx*xx[1])+cos(lz*xx[2]);
        u[2] = sin(lz*xx[2])+cos(lx*xx[0]);
    }

private:
    double E;
    double nu;
    double lx;
    double ly;
    double lz;

};




class StressSol3D:public MatrixCoefficient
{
public:
    StressSol3D(double E_=1.0, double nu_=0.3,
              double lx_=1.0,double ly_=1.0, double lz_=1.0):MatrixCoefficient(3)
    {
        E=E_;
        nu=nu_;
        lx=lx_;
        ly=ly_;
        lz=lz_;
   }

    virtual
    void  Eval(DenseMatrix &ss, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        double nnu=nu;

        ss(0,0) = 2.0*E/(2.0+2.0*nnu)*lx*cos(lx*xx[0])+E*nnu/(1.0+nnu)/(1.0-2.0*
        nnu)*(lx*cos(lx*xx[0])+lx*cos(lx*xx[1])+lz*cos(lz*xx[2]));

        ss(0,1) = -E/(2.0+2.0*nnu)*ly*sin(ly*xx[1]);
        ss(0,2) = -E/(2.0+2.0*nnu)*lx*sin(lx*xx[0]);
        ss(1,0) = -E/(2.0+2.0*nnu)*ly*sin(ly*xx[1]);
        ss(1,1) = 2.0*E/(2.0+2.0*nnu)*lx*cos(lx*xx[1])+E*nnu/(1.0+nnu)/(1.0-2.0*
        nnu)*(lx*cos(lx*xx[0])+lx*cos(lx*xx[1])+lz*cos(lz*xx[2]));

        ss(1,2) = -E/(2.0+2.0*nnu)*lz*sin(lz*xx[2]);
        ss(2,0) = -E/(2.0+2.0*nnu)*lx*sin(lx*xx[0]);
        ss(2,1) = -E/(2.0+2.0*nnu)*lz*sin(lz*xx[2]);
        ss(2,2) = 2.0*E/(2.0+2.0*nnu)*lz*cos(lz*xx[2])+E*nnu/(1.0+nnu)/(1.0-2.0*
        nnu)*(lx*cos(lx*xx[0])+lx*cos(lx*xx[1])+lz*cos(lz*xx[2]));
    }



private:
    double E;
    double nu;
    double lx;
    double ly;
    double lz;

};

class ForceSol3D:public VectorCoefficient
{
public:
    ForceSol3D(double E_=1.0, double nu_=0.3,
              double lx_=1.0,double ly_=1.0, double lz_=1.0):VectorCoefficient(3)
    {
        E=E_;
        nu=nu_;
        lx=lx_;
        ly=ly_;
        lz=lz_;

    }

    virtual
    void  Eval(Vector &ff, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        ff.SetSize(3);

        double nnu=nu;
        ff[0] = 2.0*E/(2.0+2.0*nnu)*lx*lx*sin(lx*xx[0])+E*nnu/(1.0+nnu)/(1.0-2.0*
  nnu)*lx*lx*sin(lx*xx[0])+E/(2.0+2.0*nnu)*ly*ly*cos(ly*xx[1]);
        ff[1] = 2.0*E/(2.0+2.0*nnu)*lx*lx*sin(lx*xx[1])+E*nnu/(1.0+nnu)/(1.0-2.0*
  nnu)*lx*lx*sin(lx*xx[1])+E/(2.0+2.0*nnu)*lz*lz*cos(lz*xx[2]);
        ff[2] = E/(2.0+2.0*nnu)*lx*lx*cos(lx*xx[0])+2.0*E/(2.0+2.0*nnu)*lz*lz*sin
  (lz*xx[2])+E*nnu/(1.0+nnu)/(1.0-2.0*nnu)*lz*lz*sin(lz*xx[2]);

    }

private:
    double E;
    double nu;
    double lx;
    double ly;
    double lz;

};

class StressCompCoef:public Coefficient
{
public:
    StressCompCoef(StressSol2D* sco_,LinIsoElasticityCoefficient* lco_)
    {
        sco2d=sco_;
        lco=lco_;
    }

    StressCompCoef(StressSol3D* sco_,LinIsoElasticityCoefficient* lco_)
    {
        sco2d=nullptr;
        sco3d=sco_;
        lco=lco_;
    }


    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip)
    {
        DenseMatrix ss(3);
        lco->EvalStress(ss, T, ip);

        DenseMatrix ssc(3);
        double r=0.0;
        if(sco2d!=nullptr){
            ssc=0.0;
            sco2d->Eval(ssc,T,ip);
            for(int i=0;i<2;i++){
            for(int j=0;j<2;j++){
                r=r+(ssc(i,j)-ss(i,j))*(ssc(i,j)-ss(i,j));
            }}
        }else{
            sco3d->Eval(ssc,T,ip);
            for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                r=r+(ssc(i,j)-ss(i,j))*(ssc(i,j)-ss(i,j));
            }}
        }

        return sqrt(r);


    }

private:
    StressSol3D* sco3d;
    StressSol2D* sco2d;
    LinIsoElasticityCoefficient* lco;
};


void CompDisplH1Norm(VectorCoefficient& ut, ParGridFunction* uu,
                       Array<int> marks, CutIntegrationRules& cut_int, double& L2err, double& H1err)
{
    ParMesh* mesh=uu->ParFESpace()->GetParMesh();
    int order = uu->ParFESpace()->GetMaxElementOrder();
    int dim= mesh->SpaceDimension();
    H1_FECollection fec(order+3,dim);
    ParFiniteElementSpace fes(mesh,&fec,dim);

    ParGridFunction up(&fes);
    up.ProjectCoefficient(ut);

    //add the cuts
    double cerr=0.0;
    double derr=0.0;
    double rerr=0.0;

    {
        ElementTransformation *trans;
        Vector duu(dim);
        Vector dtt(dim);
        DenseMatrix dduu(dim);
        DenseMatrix ddtt(dim);
        double w;
        const IntegrationRule* ir;
        for(int i=0;i<fes.GetNE();i++)
        {
            const FiniteElement* el=fes.GetFE(i);
            //get the element transformation
            trans = fes.GetElementTransformation(i);

            if(marks[i]==ElementMarker::SBElementType::INSIDE){
                ir=&IntRules.Get(el->GetGeomType(), order+3);
            }else
            if(marks[i]==ElementMarker::SBElementType::CUT){
                ir=cut_int.GetSurfIntegrationRule(i);
            }else{
                continue;
            }

            for(int j=0; j<ir->GetNPoints();j++){
                const IntegrationPoint &ip = ir->IntPoint(j);
                trans->SetIntPoint(&ip);
                w=ip.weight * trans->Weight();

                uu->GetVectorValue(*trans,ip,duu);
                up.GetVectorValue(*trans,ip,dtt);

                duu.Add(-1.0,dtt);
                cerr=cerr+w*(duu*duu);

                uu->GetVectorGradient(*trans,dduu);
                up.GetVectorGradient(*trans,ddtt);

                dduu.Add(-1.0,ddtt);
                for(int pp=0;pp<dim;pp++){
                for(int kk=0;kk<dim;kk++){
                    derr=derr+w*dduu(kk,pp)*dduu(kk,pp);
                }}
            }
        }

    }

    MPI_Reduce(&cerr,&rerr,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    cerr=sqrt(rerr); L2err=cerr;
    rerr=0.0;
    MPI_Reduce(&derr,&rerr,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    cerr=cerr+sqrt(rerr); H1err=cerr;

}


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myrank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 2;
   int order = 2;
   int cut_int_order = order;
   const char *device_config = "cpu";
   double stiff_ratio=1e-6;
   bool visualization = true;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");

   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&cut_int_order, "-co", "--corder",
                  "Cut integration order");
   args.AddOption(&stiff_ratio,"-sr", "--stiff_ratio",
                  "Stiffness ratio");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myrank == 0) { args.PrintOptions(cout); }

   // Enable hardware devices such as GPUs, and programming models such as CUDA,
   // OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myrank == 0) { device.Print(); }

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   if(myrank==0){
       std::cout<<"Num elements="<<mesh.GetNE()<<std::endl;
   }



   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   FilterSolver* filter=new FilterSolver(0.1,&pmesh,2);

   ParFiniteElementSpace* dfes=filter->GetDesignFES();
   ParFiniteElementSpace* ffes=filter->GetFilterFES();

   ParGridFunction desgf(dfes); desgf=0.0;
   ParGridFunction filgf(ffes); filgf=0.0;

   {// project the coefficient and filter
       //GyroidCoeff gc(6.0*M_PI);
       CheseCoeff  gc(2.0*M_PI);
       desgf.ProjectCoefficient(gc);
       Vector tdes(dfes->GetTrueVSize()); tdes=0.0;
       desgf.GetTrueDofs(tdes);
       Vector tfil(ffes->GetTrueVSize()); tfil=0.0;
       filter->Mult(tdes,tfil);
       filgf.SetFromTrueDofs(tfil);
   }

   ElementMarker* elmark=new ElementMarker(pmesh,false,true);
   elmark->SetLevelSetFunction(filgf);

   Array<int> marks;
   elmark->MarkElements(marks);
   Array<int> ghost_penalty_marks;
   elmark->MarkGhostPenaltyFaces(ghost_penalty_marks);


   //define the cut integration rules
   CutIntegrationRules cut_int(2*cut_int_order, filgf, marks);

   for(int i=0;i<pmesh.GetNE();i++){
       pmesh.SetAttribute(i,marks[i]);
   }

   DispSol2D dsol2d(1.0,0.3,2.0*M_PI,2.0*M_PI);
   StressSol2D ssol2d(1.0,0.3,2.0*M_PI,2.0*M_PI);
   DispSol3D dsol3d(1.0,0.3,2.0*M_PI,2.0*M_PI,2.0*M_PI);
   StressSol3D ssol3D(1.0,0.3,2.0*M_PI,2.0*M_PI,2.0*M_PI);




   CFElasticitySolver* elsolv=new CFElasticitySolver(&pmesh,order);
   elsolv->SetGhostPenalty(0.001,ghost_penalty_marks);

   Vector vf(dim); vf=0.0; vf(1)=0.0;
   //VectorConstantCoefficient* ff=new VectorConstantCoefficient(vf);
   LinIsoElasticityCoefficient* lec=new LinIsoElasticityCoefficient(1.0,0.3);
   elsolv->SetLinearSolver(1e-12,1e-12,400);
   elsolv->SetNewtonSolver(1e-10,1e-12,20,1);
   if(dim==2){
       ForceSol2D* fsol2d= new ForceSol2D(1.0,0.3,2.0*M_PI,2.0*M_PI);
       BdrLoadSol2D* surf_load2d=new BdrLoadSol2D(&ssol2d,&filgf);
       elsolv->AddMaterial(lec,fsol2d,surf_load2d);
       //elsolv->AddMaterial(lec,fsol2d,nullptr);
       //elsolv->AddMaterial(lec,nullptr,nullptr);
   }else{//3D
       ForceSol3D* fsol3d=new ForceSol3D(1.0,0.3,2.0*M_PI,2.0*M_PI,2.0*M_PI);
       elsolv->AddMaterial(lec,fsol3d,nullptr);
   }
   //elsolv->AddDispBC(2,4,0.0);
   if(dim==2){
       elsolv->AddDispBC(1,dsol2d);
       elsolv->AddDispBC(2,dsol2d);
       elsolv->AddDispBC(3,dsol2d);
       elsolv->AddDispBC(4,dsol2d);
   }else{ //3D
       elsolv->AddDispBC(2,dsol3d);
   }
   elsolv->SetLSF(filgf,marks, cut_int);
   elsolv->SetStiffnessRatio(stiff_ratio);
   elsolv->FSolve();
   ParGridFunction& u=elsolv->GetDisplacements();

   //chack the displacement and the stress coefficients
   //displacement field
   ParGridFunction ug(u);
   L2_FECollection l2fec(4,dim);
   ParFiniteElementSpace l2fes(&pmesh,&l2fec,1);
   ParGridFunction errgf(&l2fes);
   if(dim==2){
       ug.ProjectCoefficient(dsol2d);
       lec->SetDisplacementField(ug);
       StressSol2D sco(1.0,0.3,2.0*M_PI,2.0*M_PI);
       StressCompCoef scco(&sco,lec);
       errgf.ProjectCoefficient(scco);

   }else{
       ug.ProjectCoefficient(dsol3d);
       lec->SetDisplacementField(ug);
       StressSol3D sco(1.0,0.3,2.0*M_PI,2.0*M_PI,2.0*M_PI);
       StressCompCoef scco(&sco,lec);
       errgf.ProjectCoefficient(scco);
   }


   double L2err;
   double H1err;
   if(dim==2){
       CompDisplH1Norm(dsol2d,&u,marks,cut_int,L2err,H1err);
   }else{
       CompDisplH1Norm(dsol3d,&u,marks,cut_int,L2err,H1err);
   }


   if(myrank==0){
       std::cout.setf( std::ios_base::scientific, std::ios_base::floatfield );
       std::cout<<"L2 err = "<<L2err<<std::endl;
       std::cout<<"H1 err = "<<H1err<<std::endl;
   }


   // ParaView output.
   ParaViewDataCollection dacol("ParaViewDistance", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.SetHighOrderOutput(true);
   dacol.RegisterField("design", &desgf);
   dacol.RegisterField("flter", &filgf);
   dacol.RegisterField("disp",&u);
   dacol.RegisterField("disp_sol",&ug);
   dacol.RegisterField("err",&errgf);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();


   delete elsolv;
   delete elmark;
   delete filter;

   return 0;
}
