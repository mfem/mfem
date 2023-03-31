#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "marking.hpp"
#include "mtop_solvers.hpp"
#include "mtop_filters.hpp"
#include "shape_grad.hpp"


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

        return r;

        //if(r>0.0){return 1.0;}
        //return -1.0;
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

        if(r>0.0){return 0.0;}
        return 1.0;
    }

private:
    double ll;

};

class ThresholdCoeff:public Coefficient
{
public:
    ThresholdCoeff(Coefficient& coeff_,Coefficient& threshold_)
    {
        coeff=&coeff_;
        threshold=&threshold_;
        loc_thres=nullptr;
        loc_coeff=nullptr;
    }


    ThresholdCoeff(Coefficient& coeff_,double thr)
    {
        coeff=&coeff_;
        loc_thres=new ConstantCoefficient(thr);
        threshold=loc_thres;
        loc_coeff=nullptr;

    }

    ThresholdCoeff(GridFunction& coeff_,double thr)
    {
        loc_coeff=new GridFunctionCoefficient(&coeff_);
        coeff=loc_coeff;
        loc_thres=new ConstantCoefficient(thr);
        threshold=loc_thres;

    }

    ~ThresholdCoeff()
    {
        delete loc_thres;
        delete loc_coeff;
    }

    virtual
    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip)
    {
        double cc=coeff->Eval(T,ip);
        double hh=threshold->Eval(T,ip);
        if(cc>hh){return 1.0;}
        else{return 0.0;}
    }

private:
    Coefficient* coeff;
    Coefficient* threshold;
    Coefficient* loc_thres;
    Coefficient* loc_coeff;
};

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myrank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int solver_type = 0;
   int rs_levels = 2;
   int order = 2;
   int cut_int_order = order;
   const char *device_config = "cpu";
   bool visualization = true;
   double stiff_ratio=1e-6;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");

   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&stiff_ratio,"-sr", "--stiff_ratio",
                  "Stiffness ratio");

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

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.SpaceDimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   if(myrank==0){
       std::cout<<"Num elements="<<mesh.GetNE()<<std::endl;
   }

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   FilterSolver* filter=new FilterSolver(0.1,&pmesh,2);
   filter->SetSolver(1e-8,1e-12,100,0);
if(myrank==0){std::cout<<"step 1"<<std::endl;}

   ParGridFunction desgf(filter->GetDesignFES());
   ParGridFunction filgf(filter->GetFilterFES());

   {
       //project the coefficient and filter
       GyroidCoeff gc(6.0*M_PI);
       ThresholdCoeff tc(gc,0.0);

       CheseCoeff cc(2*M_PI);
       desgf.ProjectCoefficient(cc);
       Vector tdes(filter->GetDesignFES()->GetTrueVSize()); tdes=0.0;
       desgf.GetTrueDofs(tdes);
       Vector tfil(filter->GetFilterFES()->GetTrueVSize()); tfil=0.0;
       filter->Mult(tdes,tfil);
       filgf.SetFromTrueDofs(tfil);
   }

if(myrank==0){std::cout<<"step 1"<<std::endl;}

   ElementMarker* elmark=new ElementMarker(pmesh,false,true);
   ParGridFunction lsgf(filgf); lsgf-=0.5;
   elmark->SetLevelSetFunction(lsgf);
   Array<int> marks;
   elmark->MarkElements(marks);

if(myrank==0){std::cout<<"step 2"<<std::endl;}
   //define the cut integration rule
   CutIntegrationRules* cut_int=new CutIntegrationRules(2*cut_int_order, lsgf,marks);
if(myrank==0){std::cout<<"step 3"<<std::endl;}
   CFElasticitySolver* elsolv=new CFElasticitySolver(&pmesh,order);
   Vector vf(dim); vf=0.0; vf(1)=1.0;
   VectorConstantCoefficient* ff=new VectorConstantCoefficient(vf);
   LinIsoElasticityCoefficient* lec=new LinIsoElasticityCoefficient(1.0,0.3);
   elsolv->AddMaterial(lec,ff,nullptr);
   elsolv->SetLinearSolver(1e-12,1e-12,400);
   elsolv->SetNewtonSolver(1e-10,1e-12,20,0);
   elsolv->SetLSF(lsgf,marks,*cut_int);
   elsolv->AddDispBC(2,4,0.0);
   elsolv->SetStiffnessRatio(stiff_ratio);
   elsolv->FSolve();
   ParGridFunction& u=elsolv->GetDisplacements();


if(myrank==0){std::cout<<"step 4"<<std::endl;}
   ComplianceObjectiveCut* cobj=new ComplianceObjectiveCut();
   cobj->SetCutIntegrationRules(marks,cut_int);
   cobj->SetE(1.0);
   cobj->SetPoissonRatio(0.3);
   cobj->SetVolForce(ff);
   double cpl=cobj->Eval(u,lsgf);
   std::cout<<"Compliance="<<cpl<<std::endl;
   Vector cfgrad(filter->GetFilterFES()->GetTrueVSize()); cfgrad=0.0;
   Vector cograd(filter->GetDesignFES()->GetTrueVSize()); cograd=0.0;

   cobj->Grad(u,lsgf,cfgrad);
   filter->MultTranspose(cfgrad,cograd);

   ParGridFunction cgr(filter->GetFilterFES()); cgr.SetFromTrueDofs(cfgrad);
   ParGridFunction cor(filter->GetDesignFES()); cor.SetFromTrueDofs(cograd);


   //displ objective
   DisplObjectiveCut* dobj=new DisplObjectiveCut();
   GyroidCoeff gc(2.0);
   GyroidCoeff gu(4.0);
   VectorArrayCoefficient utg(dim);
   for(int i=0;i<dim;i++){
       utg.Set(i,&gu,false);
   }
   dobj->SetCutIntegrationRules(cut_int);
   dobj->SetE(1.0);
   dobj->SetPoissonRatio(0.3);
   dobj->SetVolForce(ff);
   dobj->SetPower(2.0);
   dobj->SetSolver(elsolv);
   dobj->SetIntegrationWeight(&gc);
   dobj->SetTargetDisplacement(&utg);

   double dpl=dobj->Eval(lsgf);
   if(myrank==0){
       std::cout<<" Displ obj="<<dpl<<std::endl;
   }

   Vector dfgrad(filter->GetFilterFES()->GetTrueVSize()); dfgrad=0.0;
   Vector dograd(filter->GetDesignFES()->GetTrueVSize()); dograd=0.0;

   std::cout<<"grad in"<<std::endl;
   dobj->Grad(lsgf,dfgrad);
   std::cout<<"grad on"<<std::endl;
   filter->MultTranspose(dfgrad,dograd);
   std::cout<<"grad out"<<std::endl;

   ParGridFunction dgr(filter->GetFilterFES()); dgr.SetFromTrueDofs(dfgrad);
   ParGridFunction dor(filter->GetDesignFES()); dor.SetFromTrueDofs(dograd);


   //stress objective
   StressObjectiveCut* sobj=new StressObjectiveCut();
   sobj->SetCutIntegrationRules(cut_int);
   sobj->SetE(1.0);
   sobj->SetPoissonRatio(0.3);
   sobj->SetVolForce(ff);
   sobj->SetPower(2.0);
   sobj->SetSolver(elsolv);
   sobj->SetIntegrationWeight(&gc);

   double spl=sobj->Eval(lsgf);
   if(myrank==0){
       std::cout<<" Stress obj="<<spl<<std::endl;
   }

   Vector sfgrad(filter->GetFilterFES()->GetTrueVSize()); sfgrad=0.0;
   Vector sograd(filter->GetDesignFES()->GetTrueVSize()); sograd=0.0;

   std::cout<<"grad in"<<std::endl;
   sobj->Grad(lsgf,sfgrad);
   std::cout<<"grad on"<<std::endl;
   filter->MultTranspose(sfgrad,sograd);
   std::cout<<"grad out"<<std::endl;

   ParGridFunction sgr(filter->GetFilterFES()); sgr.SetFromTrueDofs(sfgrad);
   ParGridFunction sor(filter->GetDesignFES()); sor.SetFromTrueDofs(sograd);


   VolObjectiveCut* vobj=new VolObjectiveCut();
   vobj->SetCutIntegrationRules(marks,cut_int);


   double vol00=vobj->Eval(lsgf);
   Vector fvolgrad(filter->GetFilterFES()->GetTrueVSize());
   Vector ovolgrad(filter->GetDesignFES()->GetTrueVSize());
   vobj->Grad(lsgf,fvolgrad);
   filter->MultTranspose(fvolgrad,ovolgrad);
   ParGridFunction vgr(filter->GetFilterFES()); vgr.SetFromTrueDofs(fvolgrad);
   ParGridFunction vor(filter->GetDesignFES()); vor.SetFromTrueDofs(ovolgrad);

   std::cout<<"vol="<<vol00<<std::endl;

   //FD check
   {
       mfem::Vector prtv;
       mfem::Vector tmpv;

       mfem::Vector lsfv; desgf.GetTrueDofs(lsfv);

       prtv.SetSize(ovolgrad.Size());
       tmpv.SetSize(ovolgrad.Size());

       prtv.Randomize();
       double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
       double td=mfem::InnerProduct(pmesh.GetComm(),prtv,ovolgrad);
       double cd=mfem::InnerProduct(pmesh.GetComm(),prtv,cograd);
       double dd=mfem::InnerProduct(pmesh.GetComm(),prtv,dograd);
       double ss=mfem::InnerProduct(pmesh.GetComm(),prtv,sograd);

       td=td/nd;
       cd=cd/nd;
       dd=dd/nd;
       ss=ss/nd;
       double lsc=1.0;
       double lqoi;
       double dqoi;
       double sqoi;

       for(int l=0; l<8;l++){
           lsc/=10.0;
           prtv/=10.0;

           add(prtv,lsfv,tmpv);
           filter->Mult(tmpv,fvolgrad);
           lsgf.SetFromTrueDofs(fvolgrad); lsgf-=0.5;
           elmark->SetLevelSetFunction(lsgf);
           elmark->MarkElements(marks);
           delete cut_int;
           cut_int=new CutIntegrationRules(2*cut_int_order, lsgf, marks);

           //compute the linear elastic model
           elsolv->SetLSF(lsgf,marks,*cut_int);
           elsolv->FSolve();
           u=elsolv->GetDisplacements();

           cobj->SetCutIntegrationRules(marks,cut_int);
           double cqoi=cobj->Eval(u,lsgf);
           double sd=(cqoi-cpl)/lsc;

           dobj->SetCutIntegrationRules(cut_int);
           dqoi=dobj->Eval(lsgf);
           double md=(dqoi-dpl)/lsc;

           sobj->SetCutIntegrationRules(cut_int);
           sqoi=sobj->Eval(lsgf);
           double ssd=(sqoi-spl)/lsc;



           vobj->SetCutIntegrationRules(marks, cut_int);
           lqoi=vobj->Eval(lsgf);
           double ld=(lqoi-vol00)/lsc;
           if(myrank==0){
               std::cout<<" obj="<<vol00<<" lvo="<< lqoi<<" dx="<<lsc<<" FD app="<< ld/nd<<
                          " gr="<< td <<" err="<< std::fabs(ld/nd-td) <<std::endl;

               std::cout<<" cpl="<<cpl<<" lcpl="<< cqoi<<" dx="<<lsc<<" FD app="<< sd/nd<<
                          " gr="<< cd <<" err="<< std::fabs(sd/nd-cd) <<std::endl;

               std::cout<<" dpl="<<dpl<<" lcpl="<< dqoi<<" dx="<<lsc<<" FD app="<< md/nd<<
                          " gr="<< dd <<" err="<< std::fabs(md/nd-dd) <<std::endl;

               std::cout<<" spl="<<spl<<" scpl="<< sqoi<<" dx="<<lsc<<" FD app="<< ssd/nd<<
                          " gr="<< ss <<" err="<< std::fabs(ssd/nd-ss) <<std::endl;
           }
       }
   }


   // ParaView output.
   ParaViewDataCollection dacol("ParaViewDistance", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.SetHighOrderOutput(true);
   dacol.RegisterField("design", &desgf);
   dacol.RegisterField("flter", &filgf);
   dacol.RegisterField("fgrad",&vgr);
   dacol.RegisterField("ograd",&vor);
   dacol.RegisterField("lsf",&lsgf);
   dacol.RegisterField("disp",&u);
   dacol.RegisterField("cfgrad",&cgr);
   dacol.RegisterField("cograd",&cor);
   dacol.RegisterField("dfgrad",&dgr);
   dacol.RegisterField("dograd",&dor);
   dacol.RegisterField("adjoint",&(elsolv->GetADisplacements()));
   dacol.RegisterField("sfgrad",&sgr);
   dacol.RegisterField("sograd",&sor);

   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();


   delete cobj;
   delete sobj;
   delete dobj;
   delete elsolv;
   delete vobj;
   delete cut_int;
   delete elmark;
   delete filter;
}
