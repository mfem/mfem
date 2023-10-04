#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_coefficients.hpp"
#include "mtop_solvers.hpp"



class CoeffHoles:public mfem::Coefficient
{
public:
    CoeffHoles(double pr=0.5)
    {
        period=pr;
    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {

        double x[3];
        mfem::Vector transip(x, T.GetSpaceDim());
        T.Transform(ip, transip);

        int nx=x[0]/period;
        int ny=x[1]/period;

        x[0]=x[0]-double(nx)*period-0.5*period;
        x[1]=x[1]-double(ny)*period-0.5*period;

        double r=sqrt(x[0]*x[0]+x[1]*x[1]);
        if(r<(0.45*period)){return 0.2;}
        return 0.8;
    }


private:
    double period;
};


class YoungModulusLoc:public mfem::YoungModulus
{
public:
    YoungModulusLoc()
    {
        dens=nullptr;
        Emax=1.0;
        Emin=1e-6;
        loc_eta=new mfem::ConstantCoefficient(0.5);
        eta=loc_eta;
        loc_sca=new mfem::ConstantCoefficient(1.0);
        sca=loc_sca;
        beta=8.0;
        pp=1.0;
        simp=true;
    }

    virtual
    ~YoungModulusLoc(){
        delete loc_eta;
        delete loc_sca;
    }

    virtual
    bool GetSIMP(){return simp;}

    virtual
    void SetSIMP(bool simp_=true){
        simp=simp_;
    }

    virtual
    void SetDens(mfem::ParGridFunction* dens_)
    {
        dens=dens_;
    }

    virtual
    void SetDens(mfem::Coefficient* coef_)
    {
        coef=coef_;
    }

    virtual
    void SetProjParam(mfem::Coefficient& eta_, double beta_)
    {
        eta=&eta_;
        beta=beta_;
    }

    virtual
    void SetProjParam(double eta_, double beta_)
    {
        delete loc_eta;
        loc_eta=new mfem::ConstantCoefficient(eta_);
        eta=loc_eta;
        beta=beta_;
    }

    virtual
    void SetEMaxMin(double Emin_,double Emax_)
    {
        Emax=Emax_;
        Emin=Emin_;
    }

    virtual
    void SetPenal(double pp_)
    {
        pp=pp_;
    }

    virtual
    void SetScaling(mfem::Coefficient* sca_)
    {
        if(sca_!=nullptr){
            sca=sca_;
        }else{
            sca=loc_sca;
        }
    }

    virtual
    double 	Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {

        //evaluate the actual coordinates
        mfem::Vector xx(T.GetSpaceDim()); xx=0.0;
        T.Transform(ip,xx);

        double dd=1.0;
        if(dens!=nullptr){dd=dens->GetValue(T,ip);}
        else if(coef!=nullptr){dd=coef->Eval(T,ip);}

        if(dd>1.0){dd=1.0;}
        if(dd<0.0){dd=0.0;}

        //eval eta
        double deta=eta->Eval(T,ip);
        double lsc=sca->Eval(T,ip);
        //do the projection
        double pd=mfem::PointwiseTrans::HProject(dd,deta,beta);
        //evaluate the E modulus
        if(simp==false){
            return Emin+lsc*(Emax-Emin)*std::pow(pd,pp);
        }else{
            return Emin+lsc*(Emax-Emin)*std::pow(dd,pp);
        }
    }

    ///returnas the pointwise gradient with respect to the density
    virtual
    double Grad(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {

        //evaluate the actual coordinates
        mfem::Vector xx(T.GetSpaceDim()); xx=0.0;
        T.Transform(ip,xx);

        //evaluate density
        double dd=1.0;
        if(dens!=nullptr){dd=dens->GetValue(T,ip);}
        else if(coef!=nullptr){dd=coef->Eval(T,ip);}

        if(dd>1.0){dd=1.0;}
        if(dd<0.0){dd=0.0;}
        //eval eta
        double deta=eta->Eval(T,ip);
        double lsc=sca->Eval(T,ip);
        //do the projection
        double pd=mfem::PointwiseTrans::HProject(dd,deta,beta);
        //evaluate hte gradient of the projection
        double pg=mfem::PointwiseTrans::HGrad(dd,deta,beta);
        //evaluate the gradient with respect to the density field
        if(simp==false){
            return lsc*(Emax-Emin)*pg*pp*std::pow(pd,pp-1.0);
        }else{
            return lsc*(Emax-Emin)*std::pow(dd,pp-1.0);
        }
    }



private:
    mfem::ParGridFunction* dens;
    mfem::Coefficient* coef;
    double Emax;
    double Emin;
    mfem::Coefficient* eta;
    mfem::Coefficient* loc_eta;
    mfem::Coefficient* sca;
    mfem::Coefficient* loc_sca;
    double beta;
    double pp;
    bool simp;
};



class AlcoaBracket
{
public:
    AlcoaBracket(mfem::ParMesh* pmesh_, int vorder=1,
                 int seed=std::numeric_limits<int>::max()):E(),nu(0.2)
    {

        dorder=vorder;
        pmesh=pmesh_;

        esolv=new mfem::ElasticitySolver(pmesh,vorder);
        esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
        esolv->SetNewtonSolver(1e-8,1e-12,1,0);
        esolv->SetLinearSolver(1e-10,1e-12,400);

        dfes=nullptr;
        cobj=new mfem::ComplianceObjective();

        generator.seed(seed);
        rf=new mfem::RandFieldCoefficient(pmesh,vorder);
        rf->SetCorrelationLen(0.2);
        rf->SetMaternParameter(2.0);
        rf->SetScale(1.0);
        //set all bc
        /*
        for(int i=3;i<23;i++){
            rf->SetZeroDirichletBC(i);
        }*/
        rf->SetZeroDirichletBC(1);


        rf->Sample(seed);

        num_samples=100;

    }

    void SetDesignFES(mfem::ParFiniteElementSpace* fes)
    {
        dfes=fes;
        pdens.SetSpace(dfes);
        vdens.SetSize(dfes->GetTrueVSize());
    }

    ~AlcoaBracket()
    {
        delete rf;
        delete cobj;
        delete esolv;
    }

    void GetRandomField(mfem::ParGridFunction& pgf)
    {
        pgf.ProjectCoefficient(*rf);
    }

    void GetE(mfem::ParGridFunction& pgf)
    {
        pgf.ProjectCoefficient(E);
    }

    void SetNumSamples(int ns)
    {
        num_samples=ns;
    }

    void SetCorrelationLen(double l)
    {
        rf->SetCorrelationLen(l);
    }


    void SetSIMP(bool vv)
    {
        E.SetSIMP(vv);
    }

    double MeanCompliance(mfem::Vector& grad)
    {
        mfem::Vector lgr(grad.Size());
        grad=0.0;

        //set all bc
        esolv->DelDispBC();
        for(int i=3;i<23;i++){
            esolv->AddDispBC(i,4,0.0);
        }
        esolv->AddSurfLoad(1,0.00,1.00,0.0);

        //esolv->FSolve();
        //esolv->GetSol(sol);

        //mfem::UniformDistributionCoefficient sca(rf,0.001,0.999);

        int myrank;
        MPI_Comm_rank(pmesh->GetComm(),&myrank);

        double threshold=2.5;
        mfem::ThresholdCoefficient sca(rf,threshold,0.00,1.00);
        std::uniform_int_distribution<int> uint(1,std::numeric_limits<int>::max());

        //seeds are conditioned on single or multiple threshold crossings
        int bulk_samples=0;
        if(seeds.size()<num_samples)
        {
            mfem::ThresholdCoefficient lsca(rf,threshold,1.00,0.00);
            mfem::L2_FECollection fec(dorder,pmesh->Dimension());
            mfem::ParFiniteElementSpace fes(pmesh,&fec);
            mfem::ParLinearForm lf(&fes);
            lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(lsca));

            lsca.SetThresholdValues(1.0,1.0);

            lf.Assemble();
            double ovol=lf.Sum();
            MPI_Allreduce(MPI_IN_PLACE, &ovol, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

            if(myrank==0){std::cout<<"tot_vol="<<ovol<<std::endl;}

            lsca.SetThresholdValues(1.0,0.0);

            for(int ii=0;ii<num_samples;ii++){
                //generate seeds
                int flag=0;
                int seed;
                while(flag==0){
                    int seed1 = uint(generator);
                    int seed2 = static_cast<int>(std::time(nullptr));
                    seed=seed1/2+seed2/2;
                    rf->Sample(seed);
                    lf.Assemble();
                    double vtr=lf.Sum();
                    MPI_Allreduce(MPI_IN_PLACE, &vtr, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

                    if(myrank==0){std::cout<<"vtr="<<vtr<<std::endl;}

                    if(vtr>0.01*ovol){flag=1;}
                    else{bulk_samples++;}
                }
                seeds.push_back(seed);
            }

            std::cout<<"bulk_samples="<<bulk_samples<<" num_samples="<<num_samples<<std::endl;
        }



        int n=num_samples;
        double obj=0.0;
        double var=0.0;

        for(int i=0;i<n;i++){

            if(seeds.size()<(i+1)){
                int seed1 = uint(generator);
                int seed2 = static_cast<int>(std::time(nullptr));
                int seed=seed1/2+seed2/2;
                seeds.push_back(seed);
            }

            rf->Sample(seeds[i]);
            E.SetScaling(&sca);
            esolv->FSolve();
            esolv->GetSol(sol);

            double rez=cobj->Eval(sol);
            cobj->Grad(sol,lgr);
            grad.Add(1.0,lgr);
            obj=obj+rez;
            var=var+rez*rez;
        }

        E.SetScaling(nullptr);

        grad/=double(n);
        var=var/double(n);
        obj=obj/double(n);

        if(myrank==0){
        std::cout<<"Var="<<var-obj*obj<<std::endl;}

        return obj;
    }



    void SetDensity(mfem::Vector& vdens_,
                    double eta=0.5, double beta=8.0,double pen=3.0){

        vdens=vdens_;
        pdens.SetFromTrueDofs(vdens);

        E.SetDens(&pdens);
        E.SetProjParam(eta,beta);
        E.SetEMaxMin(1e-6,1.0);
        E.SetPenal(pen);

        cobj->SetE(&E);
        cobj->SetDens(vdens);
        cobj->SetDesignFES(dfes);

    }


    mfem::Coefficient* GetE(){
        return &E;
    }

    /// Evaluates the compliance
    double  Compliance(mfem::Vector& grad)
    {
        //set all bc
        esolv->DelDispBC();
        for(int i=3;i<23;i++){
            esolv->AddDispBC(i,4,0.0);
        }
        esolv->AddSurfLoad(1,0.00,1.00,0.0);
        esolv->FSolve();
        esolv->GetSol(sol);

        double rez=cobj->Eval(sol);
        cobj->Grad(sol,grad);
        return rez;
    }


private:
    int dorder; //discretization order
    mfem::ParMesh* pmesh;
    YoungModulusLoc E;
    double nu;

    mfem::ParFiniteElementSpace* dfes; //design FES
    mfem::ParGridFunction pdens;
    mfem::Vector vdens;

    mfem::ElasticitySolver* esolv;
    mfem::ComplianceObjective* cobj;

    mfem::ParGridFunction sol;

    mfem::RandFieldCoefficient* rf;
    int num_samples;
    std::vector<int> seeds;
    std::default_random_engine generator;

};



int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "./canti_2D_m.msh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   double corr_len = 0.2;
   int tot_iter = 100;
   int max_it = 51;
   int print_level = 1;
   int num_samples=100;
   bool visualization = false;
   const char *petscrc_file = "";
   int restart=0;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&tot_iter,
                  "-it",
                  "--linear-iterations",
                  "Maximum iterations for the linear solve.");
   args.AddOption(&max_it,
                  "-mit",
                  "--max-optimization-iterations",
                  "Maximum iterations for the linear optimizer.");
   args.AddOption(&fradius,
                  "-r",
                  "--radius",
                  "Filter radius");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                     "PetscOptions file to use.");
   args.AddOption(&restart,
                     "-rstr",
                     "--restart",
                     "Restart the optimization from previous design.");
   args.AddOption(&num_samples,
                  "-ns",
                  "--num-samples",
                  "Number of samples.");
   args.AddOption(&corr_len,
                  "-crl",
                  "--corr",
                  "Correlation length");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }
   mfem::MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   {
       mfem::Vector vert;
       mesh.GetVertices(vert);
       vert*=0.01;
       mesh.SetVertices(vert);
       mfem::Vector xmin(dim), xmax(dim);
       mesh.GetBoundingBox(xmin,xmax);
       if(myrank==0){
           std::cout<<"Xmin:";xmin.Print(std::cout);
           std::cout<<"Xmax:";xmax.Print(std::cout);
       }
   }

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);

      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh.UniformRefinement();
       }
   }

   if(myrank==0)
   {
       std::cout<<"num el="<<pmesh.GetNE()<<std::endl;
   }


   //allocate the filter
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.04,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,1.0);
   fsolv->AddBC(2,0.0);
   for(int i=3;i<23;i++){
       fsolv->AddBC(i,1.0);
   }

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction emod(fsolv->GetFilterFES()); emod=0.0;
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);


   AlcoaBracket* alco=new AlcoaBracket(&pmesh,1);
   alco->SetDesignFES(pgdens.ParFESpace());
   alco->SetDensity(vdens);
   alco->SetCorrelationLen(corr_len);
   alco->SetNumSamples(num_samples);
   //mfem::ParGridFunction disp;
   //alco->GetSol(4,1,1,1,disp);

   //check gradients
   /*
   {
       mfem::Vector prtv;
       mfem::Vector tmpv;
       mfem::Vector tgrad;
       mfem::Vector fgrad;
       prtv.SetSize(vtmpv.Size());
       tmpv.SetSize(vtmpv.Size());
       tgrad.SetSize(vtmpv.Size());
       fgrad.SetSize(vdens.Size()); fgrad=0.0;
       double val=alco->MeanCompliance();
       alco->MeanCompliance(fgrad);
       fsolv->MultTranspose(fgrad,tgrad);

       prtv.Randomize();
       double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
       double td=mfem::InnerProduct(pmesh.GetComm(),prtv,tgrad);
       td=td/nd;
       double lsc=1.0;
       double lqoi;

       for(int l=0;l<10;l++){
           lsc/=10.0;
           prtv/=10.0;
           add(prtv,vtmpv,tmpv);
           fsolv->Mult(tmpv,vdens);
           alco->SetDensity(vdens);
           alco->Solve();
           lqoi=alco->MeanCompliance();
           double ld=(lqoi-val)/lsc;
           if(myrank==0){
               std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                         << " adjoint gradient=" << td
                         << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
       }
   }*/

   mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(fsolv->GetFilterFES());
   //mfem::VolumeQoI* vobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   vobj->SetProjection(0.5,8.0);//threshold 0.2

   //compute the total volume
   double tot_vol;
   {
       vdens=1.0;
       tot_vol=vobj->Eval(vdens);
   }
   double max_vol=0.4*tot_vol;
   if(myrank==0){ std::cout<<"tot vol="<<tot_vol<<std::endl;}

   //intermediate volume
   mfem::VolumeQoI* ivobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   ivobj->SetProjection(0.5,32);

   //gradients with respect to the filtered field
   mfem::Vector ograd(fsolv->GetFilterFES()->GetTrueVSize()); ograd=0.0; //of the objective
   mfem::Vector vgrad(fsolv->GetFilterFES()->GetTrueVSize()); vgrad=0.0; //of the volume contr.

   //the input design field and the filtered one might not have the same dimensionality
   mfem::Vector ogrado(fsolv->GetDesignFES()->GetTrueVSize()); ogrado=0.0;
   mfem::Vector vgrado(fsolv->GetDesignFES()->GetTrueVSize()); vgrado=0.0;

   mfem::Vector xxmax(fsolv->GetDesignFES()->GetTrueVSize()); xxmax=1.0;
   mfem::Vector xxmin(fsolv->GetDesignFES()->GetTrueVSize()); xxmin=0.0;

   mfem::NativeMMA* mma;
   {
       double a=0.0;
       double c=1000.0;
       double d=0.0;
       mma=new mfem::NativeMMA(MPI_COMM_WORLD,1, ogrado,&a,&c,&d);
   }

   double max_ch=0.1; //max design change

   double cpl; //compliance
   double vol; //volume
   double ivol; //intermediate volume
   double dcpl;


   mfem::ParGridFunction solx;
   mfem::ParGridFunction soly;

   {
      mfem::ParaViewDataCollection paraview_dc("TopOpt", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);

      paraview_dc.RegisterField("design",&pgdens);
      paraview_dc.RegisterField("E",&emod);
      alco->GetRandomField(emod);

      //alco->GetSol(6,0.0,1.0,0.0,solx);
      //alco->GetSol(1,0.0,1.0,0.0,soly);
      //paraview_dc.RegisterField("solx",&solx);
      //paraview_dc.RegisterField("soly",&soly);

      //spdegf.ProjectCoefficient(spderf);
      //paraview_dc.RegisterField("reta",&spdegf);

      paraview_dc.Save();

      CoeffHoles holes;
      oddens.ProjectCoefficient(holes);
      oddens.GetTrueDofs(vtmpv);
      vtmpv=0.3;
      fsolv->Mult(vtmpv,vdens);
      pgdens.SetFromTrueDofs(vdens);

      double cvar;
      double erisk;
      double meanstd;
      double std;

      for(int i=1;i<max_it;i++){

          /*
          if((i%11)==0){
              spderf.Generate();
              spdegf.ProjectCoefficient(spderf);
          }*/

          vobj->SetProjection(0.5,8.0);
          alco->SetDensity(vdens,0.5,8.0,3.0);
          alco->SetSIMP(true);
          /*
          if(i<11){alco->SetDensity(vdens,0.5,1.0,3.0); vobj->SetProjection(0.5,1.0); }
          else if(i<80){alco->SetDensity(vdens,0.5,8.0,3.0);}
          else if(i<120){alco->SetDensity(vdens,0.5,8.0,3.0);}
          else if(i<160){alco->SetDensity(vdens,0.6,8.0,3.0); vobj->SetProjection(0.4,8.0); }
          else {alco->SetDensity(vdens,0.7,8.0,3.0);  vobj->SetProjection(0.3,8.0);}
          */

          vol=vobj->Eval(vdens);
          ivol=ivobj->Eval(vdens);
          //cpl=alco->Compliance(ograd);
          cpl=alco->MeanCompliance(ograd);

          if(myrank==0){
              std::cout<<"it: "<<i<<" obj="<<cpl<<" vol="<<vol<<" cvol="<<max_vol<<" ivol="<<ivol<<std::endl;
          }
          //compute the gradients
          //alco->MeanCompliance(ograd);
          //alco->CVar(ograd);
          //alco->MeanSTD(ograd,0.5);
          //alco->GetComplianceGrad(0,0.0,0.0,1.0,ograd);
          //alco->CVar(ograd);
          //alco->CVaRe(ograd,0.98,1.0);
          //alco->EVaR(ograd,0.70);

          vobj->Grad(vdens,vgrad);
          //compute the original gradients
          fsolv->MultTranspose(ograd,ogrado);
          fsolv->MultTranspose(vgrad,vgrado);

          {
              //set xxmin and xxmax
              xxmin=vtmpv; xxmin-=max_ch;
              xxmax=vtmpv; xxmax+=max_ch;
              for(int li=0;li<xxmin.Size();li++){
                  if(xxmin[li]<0.0){xxmin[li]=0.0;}
                  if(xxmax[li]>1.0){xxmax[li]=1.0;}
              }
          }

          double con=vol-max_vol;
          mma->Update(vtmpv,ogrado,&con,&vgrado,xxmin,xxmax);

          fsolv->Mult(vtmpv,vdens);
          pgdens.SetFromTrueDofs(vdens);

          //alco->GetSol(1,0.0,1.0,0.0,solx);
          //alco->GetSol(6,0.0,1.0,0.0,soly);

          //paraview_dc.RegisterField("solx",&solx);
          //paraview_dc.RegisterField("soly",&soly);

          //save the design
          if(i%4==0)
          {
              alco->GetE(emod);
              paraview_dc.SetCycle(i);
              paraview_dc.SetTime(i*1.0);
              paraview_dc.Save();
          }
      }

   }


   delete mma;
   delete vobj;
   delete ivobj;
   delete alco;
   delete fsolv;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
