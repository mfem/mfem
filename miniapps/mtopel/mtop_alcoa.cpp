#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_solvers.hpp"


class MyCoefficient:public mfem::Coefficient
{
public:
    MyCoefficient(mfem::GridFunction* gridf)
    {
        gf=gridf;
    }

    void SetGridFunction(mfem::GridFunction* gridf){
        gf=gridf;
    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {
        double gf_val=gf->GetValue(T,ip);
        return gf_val*(1.0-gf_val);
    }

private:
    mfem::GridFunction* gf;
};

class AlcoaBracket
{
public:
    AlcoaBracket(mfem::ParMesh* pmesh, int vorder=1):E(),nu(0.2)
    {
        esolv=new mfem::ElasticitySolver(pmesh,vorder);
        esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
        esolv->SetNewtonSolver(1e-8,1e-12,1,0);
        esolv->SetLinearSolver(1e-10,1e-12,200);

        dfes=nullptr;
        cobj=new mfem::ComplianceObjective();


    }

    void SetDesignFES(mfem::ParFiniteElementSpace* fes)
    {
        dfes=fes;
        pdens.SetSpace(dfes);
        vdens.SetSize(dfes->GetTrueVSize());
    }

    ~AlcoaBracket()
    {
        delete cobj;
        delete esolv;

    }

    void Solve()
    {
        //solve the problem for the base loads
        bsolx.resize(5);
        bsoly.resize(5);
        bsolz.resize(5);

        /*
        for(int i=0;i<4;i++){
            esolv->DelDispBC();
            //set BC
            for(int j=0;j<4;j++){if(j!=i){ esolv->AddDispBC(2+j,4,0.0);}}
            esolv->AddSurfLoad(1,.001,0.0,0.0);
            esolv->FSolve();
            esolv->GetSol(bsolx[i]);
            esolv->AddSurfLoad(1,0.0,.001,0.0);
            esolv->FSolve();
            esolv->GetSol(bsoly[i]);
            esolv->AddSurfLoad(1,0.0,0.0,.001);
            esolv->FSolve();
            esolv->GetSol(bsolz[i]);
        }
        */

        //set all bc
        esolv->DelDispBC();
        for(int j=0;j<4;j++){esolv->AddDispBC(2+j,4,0.0);}
        esolv->AddSurfLoad(1,.001,0.0,0.0);
        esolv->FSolve();
        esolv->GetSol(bsolx[4]);
        esolv->AddSurfLoad(1,0.0,.001,0.0);
        esolv->FSolve();
        esolv->GetSol(bsoly[4]);
        esolv->AddSurfLoad(1,0.0,0.0,.001);
        esolv->FSolve();
        esolv->GetSol(bsolz[4]);

        // all solutions are stored in bsolx,bsoly,bsolz
    }

    void SetDensity(mfem::Vector& vdens_,
                    double eta=0.5, double beta=8.0,double pen=1.0){

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

    /// Evaluates the compliance
    double Compliance(int i, double fx, double fy, double fz){

        return cobj->Eval(GetSol(i,fx,fy,fz));
    }

    void GetComplianceGrad(int i, double fx, double fy, double fz, mfem::Vector& grad){
        if(dfes==nullptr)
        {
             mfem::mfem_error("AlcoaBracket dfes is not defined!");
        }
        cobj->Grad(GetSol(i,fx,fy,fz),grad);
    }

    double MeanCompliance()
    {
        int myrank=dfes->GetMyRank();
        int num_samples=10;
        int scase[num_samples];
        double fx[num_samples];
        double fy[num_samples];
        double an[num_samples];

        //integer random generator
        if(myrank==0){
            std::default_random_engine generator;
            std::uniform_int_distribution<int> idist(4,4);
            //std::normal_distribution<double> ndistx(0.0,1.0);
            //std::normal_distribution<double> ndisty(0.0,1.0);
            std::uniform_real_distribution<double> adist(-M_PI,M_PI);

            for(int i=0;i<num_samples;i++){
                scase[i]=idist(generator);
                //fx[i]=ndistx(generator);
                //fy[i]=ndisty(generator);
                an[i]=adist(generator);
            }
        }

        MPI_Bcast(scase,num_samples,MPI_INT,0,dfes->GetComm());
        //MPI_Bcast(fx,num_samples,MPI_DOUBLE,0,dfes->GetComm());
        //MPI_Bcast(fy,num_samples,MPI_DOUBLE,0,dfes->GetComm());
        MPI_Bcast(an,num_samples,MPI_DOUBLE,0,dfes->GetComm());



        double mean=0.0;
        double var=0.0;

        double cobj;
        for(int i=0;i<num_samples;i++){
            fx[i]=sin(an[i]); fy[i]=cos(an[i]);
            //eval the objective
            cobj=Compliance(scase[i],fx[i],fy[i],0.0);
            mean=mean+cobj;
            var=var+cobj*cobj;
        }

        mean=mean/(double(num_samples));
        var=var/(double(num_samples)); var=var-mean*mean;
        if(myrank==0){
            std::cout<<"mean="<<mean<<" var="<<var<<std::endl;
        }
        return mean;
    }

    void MeanCompliance(mfem::Vector& grad)
    {
        int myrank=dfes->GetMyRank();
        int num_samples=10;
        int scase[num_samples];
        double fx[num_samples];
        double fy[num_samples];
        double an[num_samples];
        //integer random generator
        if(myrank==0){
            std::default_random_engine generator;
            std::uniform_int_distribution<int> idist(4,4);
            std::normal_distribution<double> ndistx(0.0,1.0);
            std::normal_distribution<double> ndisty(0.0,1.0);
            std::uniform_real_distribution<double> adist(-M_PI,M_PI);

            for(int i=0;i<num_samples;i++){
                scase[i]=idist(generator);
                //fx[i]=ndistx(generator);
                //fy[i]=ndisty(generator);
                an[i]=adist(generator);
            }
        }

        MPI_Bcast(scase,num_samples,MPI_INT,0,dfes->GetComm());
        //MPI_Bcast(fx,num_samples,MPI_DOUBLE,0,dfes->GetComm());
        //MPI_Bcast(fy,num_samples,MPI_DOUBLE,0,dfes->GetComm());
        MPI_Bcast(an,num_samples,MPI_DOUBLE,0,dfes->GetComm());


        grad=0.0;
        mfem::Vector cgrad(grad.Size()); cgrad=0.0;
        for(int i=0;i<num_samples;i++){
            fx[i]=sin(an[i]); fy[i]=cos(an[i]);
            //eval the objective
            GetComplianceGrad(scase[i],fx[i],fy[i],0.0,cgrad);
            grad.Add(1.0,cgrad);
        }

        grad/=(double(num_samples));
    }

    mfem::ParGridFunction& GetSol(int i, double fx, double fy, double fz)
    {

        sol.SetSpace((esolv->GetDisplacements()).ParFESpace()); sol=0.0;
        sol.Add(fx,bsolx[i]);
        sol.Add(fy,bsoly[i]);
        sol.Add(fz,bsolz[i]);
        return sol;
    }

    void GetSol(int i, double fx, double fy, double fz, mfem::ParGridFunction& msol)
    {
        msol.SetSpace((esolv->GetDisplacements()).ParFESpace()); msol=0.0;
        msol.Add(fx,bsolx[i]);
        msol.Add(fy,bsoly[i]);
        msol.Add(fz,bsolz[i]);
    }

private:
    mfem::YoungModulus E;
    double nu;

    mfem::ParFiniteElementSpace* dfes; //design FES
    mfem::ParGridFunction pdens;
    mfem::Vector vdens;

    mfem::ElasticitySolver* esolv;
    mfem::ComplianceObjective* cobj;

    //base solution vectors x,y,z direction loads
    std::vector<mfem::ParGridFunction> bsolx;
    std::vector<mfem::ParGridFunction> bsoly;
    std::vector<mfem::ParGridFunction> bsolz;


    mfem::ParGridFunction sol;

};



int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   int tot_iter = 100;
   int max_it = 51;
   int print_level = 1;
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
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(1.0,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,1.0);
   fsolv->AddBC(2,1.0);
   fsolv->AddBC(3,1.0);
   fsolv->AddBC(4,1.0);
   fsolv->AddBC(5,1.0);
   fsolv->AddBC(6,1.0);

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);


   AlcoaBracket* alco=new AlcoaBracket(&pmesh,1);
   alco->SetDesignFES(pgdens.ParFESpace());
   alco->SetDensity(vdens);
   //alco->Solve();
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
   vobj->SetProjection(0.2,8.0);//threshold 0.2

   //compute the total volume
   double tot_vol;
   {
       vdens=1.0;
       tot_vol=vobj->Eval(vdens);
   }
   double max_vol=0.5*tot_vol;

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

   double max_ch=0.02; //max design change

   double cpl; //compliance
   double vol; //volume
   double ivol; //intermediate volume

   {
      mfem::ParaViewDataCollection paraview_dc("TopOpt", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);

      paraview_dc.RegisterField("design",&pgdens);
      paraview_dc.Save();

      vtmpv=0.2;
      fsolv->Mult(vtmpv,vdens);
      pgdens.SetFromTrueDofs(vdens);

      for(int i=1;i<max_it;i++){

          alco->SetDensity(vdens,0.8);
          alco->Solve();

          cpl=alco->MeanCompliance();
          vol=vobj->Eval(vdens);
          ivol=ivobj->Eval(vdens);

          if(myrank==0){
              std::cout<<"it: "<<i<<" obj="<<cpl<<" vol="<<vol<<" ivol="<<ivol<<std::endl;
          }
          //compute the gradients
          alco->MeanCompliance(ograd);
          ivobj->Grad(vdens,vgrad);
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

          double con=ivol-max_vol*0.5;
          mma->Update(vtmpv,ogrado,&con,&vgrado,xxmin,xxmax);

          fsolv->Mult(vtmpv,vdens);
          pgdens.SetFromTrueDofs(vdens);

          //save the design
          {
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
