#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_solvers.hpp"

#include "spde_solvers.hpp"

class SPDERandField:public mfem::Coefficient
{
public:
    SPDERandField(mfem::ParMesh* mesh_,int order_=1)
    {
        mesh=mesh_;
        int dim=mesh->SpaceDimension();
        fec=new mfem::H1_FECollection(order_,dim);
        fes=new mfem::ParFiniteElementSpace(mesh,fec);
        u.SetSpace(fes); u=0.0;

        double nu=1.0;
        double r=0.25;
        //allocate the solvers
        mfem::DenseMatrix MatCoeff=mfem::DenseMatrix(dim); MatCoeff=0.0;
        for(int i=0;i<dim;i++){
            MatCoeff(i,i)=r*r/(1.0*nu);
        }
        diff_co=new mfem::MatrixConstantCoefficient(MatCoeff);

        //allocate the WhiteNose
        int seed=0;
        WhiteNoise=new mfem::WhiteGaussianNoiseDomainLFIntegrator(mesh->GetComm(),seed);


        lf=new mfem::ParLinearForm(fes);
        //(*lf)=0.0;
        lf->AddDomainIntegrator(WhiteNoise);
        lf->Assemble();
        double normalization = 1.0;
        lf->operator*=(normalization);

        /*
        bc.AddHomogeneousBoundaryCondition(1, mfem::materials::BoundaryType::kNeumann);
        bc.AddHomogeneousBoundaryCondition(2, mfem::materials::BoundaryType::kNeumann);
        bc.AddHomogeneousBoundaryCondition(3, mfem::materials::BoundaryType::kNeumann);
        bc.AddHomogeneousBoundaryCondition(4, mfem::materials::BoundaryType::kNeumann);
        bc.AddHomogeneousBoundaryCondition(5, mfem::materials::BoundaryType::kNeumann);
        bc.AddHomogeneousBoundaryCondition(6, mfem::materials::BoundaryType::kNeumann);
        */

        solver=new mfem::materials::SPDESolver(*diff_co,nu,bc,fes);

        solver->Solve(*lf,u);
        gfc.SetGridFunction(&u);
    }

    virtual
    ~SPDERandField()
    {
        delete solver;
        delete lf;
        delete fes;
        delete fec;
        delete diff_co;

    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {
        double rv=gfc.Eval(T,ip);
        return 0.5+0.5*std::erf(rv/std::sqrt(2.0*8.0));
    }

    void Generate()
    {
        u=0.0;
        //(*lf)=0.0;
        lf->Assemble();
        solver->Solve(*lf,u);
        gfc.SetGridFunction(&u);
    }

private:

    mfem::ParMesh* mesh;

    mfem::H1_FECollection* fec;
    mfem::ParFiniteElementSpace* fes;

    mfem::WhiteGaussianNoiseDomainLFIntegrator* WhiteNoise;

    mfem::ParGridFunction u;
    mfem::MatrixConstantCoefficient* diff_co;

    mfem::ParLinearForm* lf;
    mfem::materials::Boundary bc;
    mfem::materials::SPDESolver* solver;

    mfem::GridFunctionCoefficient gfc;

};

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

class AlcoaBracket
{
public:
    AlcoaBracket(mfem::ParMesh* pmesh, int vorder=1):E(),nu(0.2)
    {
        esolv=new mfem::ElasticitySolver(pmesh,vorder);
        esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
        esolv->SetNewtonSolver(1e-8,1e-12,1,0);
        esolv->SetLinearSolver(1e-10,1e-12,400);

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
        bsolx.resize(16);
        bsoly.resize(16);
        bsolz.resize(16);

        Solve(bsolx,0.5);
        Solve(bsoly,0.6);
        Solve(bsolz,0.7);

    }

    void Solve(std::vector<mfem::ParGridFunction>& bsol,double eta)
    {

        E.SetProjParam(eta,8.0);
        //set all bc
        esolv->DelDispBC();
        for(int j=0;j<6;j++){esolv->AddDispBC(2+j,4,0.0);}
        esolv->AddSurfLoad(1,0.00,1.00,0.0);
        esolv->FSolve();
        esolv->GetSol(bsol[0]);

        for(int i=0;i<6;i++){
            esolv->DelDispBC();
            for(int j=0;j<6;j++){if(j!=i){ esolv->AddDispBC(2+j,4,0.0);}}
            esolv->AddSurfLoad(1,0.00,1.00,0.0);
            esolv->FSolve();
            esolv->GetSol(bsol[1+i]);
        }

        for(int i=0;i<5;i++){
            esolv->DelDispBC();
            for(int j=0;j<6;j++){if((j!=i)&&(j!=(i+1))){ esolv->AddDispBC(2+j,4,0.0);}}
            esolv->AddSurfLoad(1,0.00,1.00,0.0);
            esolv->FSolve();
            esolv->GetSol(bsol[7+i]);
        }

        for(int i=0;i<4;i++){
            esolv->DelDispBC();
            for(int j=0;j<6;j++){if((j!=i)&&(j!=(i+1))&&(j!=(i+2))){ esolv->AddDispBC(2+j,4,0.0);}}
            esolv->AddSurfLoad(1,0.00,1.00,0.0);
            esolv->FSolve();
            esolv->GetSol(bsol[12+i]);
        }
        // all solutions are stored in bsolx,bsoly,bsolz
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
        double mean=0.0;
        mean=mean+1.0*Compliance(0,1.0,0.0,0.0);
        mean=mean+1.0*Compliance(0,0.0,1.0,0.0);
        mean=mean+1.0*Compliance(0,0.0,0.0,1.0);
        for(int i=1;i<7;i++){
            mean=mean+0.01*Compliance(i,1.0,0.0,0.0);
            mean=mean+0.01*Compliance(i,0.0,1.0,0.0);
            mean=mean+0.01*Compliance(i,0.0,0.0,1.0);
        }
        for(int i=7;i<12;i++){
            mean=mean+0.001*Compliance(i,1.0,0.0,0.0);
            mean=mean+0.001*Compliance(i,0.0,1.0,0.0);
            mean=mean+0.001*Compliance(i,0.0,0.0,1.0);
        }
        for(int i=12;i<16;i++){
            mean=mean+0.0001*Compliance(i,1.0,0.0,0.0);
            mean=mean+0.0001*Compliance(i,0.0,1.0,0.0);
            mean=mean+0.0001*Compliance(i,0.0,0.0,1.0);
        }
        return mean;
    }

    void MeanCompliance(mfem::Vector& grad)
    {
        grad=0.0;
        mfem::Vector cgrad(grad.Size()); cgrad=0.0;

        GetComplianceGrad(0,1.0,0.0,0.0,cgrad);grad.Add(1.0,cgrad);
        GetComplianceGrad(0,0.0,1.0,0.0,cgrad);grad.Add(1.0,cgrad);
        GetComplianceGrad(0,0.0,0.0,1.0,cgrad);grad.Add(1.0,cgrad);

        for(int i=1;i<7;i++){
            GetComplianceGrad(i,1.0,0.0,0.0,cgrad);grad.Add(0.01,cgrad);
            GetComplianceGrad(i,0.0,1.0,0.0,cgrad);grad.Add(0.01,cgrad);
            GetComplianceGrad(i,0.0,0.0,1.0,cgrad);grad.Add(0.01,cgrad);
        }
        for(int i=7;i<12;i++){
            GetComplianceGrad(i,1.0,0.0,0.0,cgrad);grad.Add(0.001,cgrad);
            GetComplianceGrad(i,0.0,1.0,0.0,cgrad);grad.Add(0.001,cgrad);
            GetComplianceGrad(i,0.0,0.0,1.0,cgrad);grad.Add(0.001,cgrad);
        }
        for(int i=12;i<16;i++){
            GetComplianceGrad(i,1.0,0.0,0.0,cgrad);grad.Add(0.0001,cgrad);
            GetComplianceGrad(i,0.0,1.0,0.0,cgrad);grad.Add(0.0001,cgrad);
            GetComplianceGrad(i,0.0,0.0,1.0,cgrad);grad.Add(0.0001,cgrad);
        }
    }

    mfem::ParGridFunction& GetSol(int i, double fx, double fy, double fz)
    {

        sol.SetSpace((esolv->GetDisplacements()).ParFESpace()); sol=0.0;
        //sol.Add(fx,bsolx[i]);
        sol.Add(fy,bsoly[i]);
        //sol.Add(fz,bsolz[i]);
        return sol;
    }

    void GetSol(int i, double fx, double fy, double fz, mfem::ParGridFunction& msol)
    {
        msol.SetSpace((esolv->GetDisplacements()).ParFESpace()); msol=0.0;
        //msol.Add(fx,bsolx[i]);
        msol.Add(fy,bsoly[i]);
        //msol.Add(fz,bsolz[i]);
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

   SPDERandField spderf(&pmesh,order);

   //allocate the filter
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.07,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,1.0);
   fsolv->AddBC(2,1.0);
   fsolv->AddBC(3,1.0);
   fsolv->AddBC(4,1.0);
   fsolv->AddBC(5,1.0);
   fsolv->AddBC(6,1.0);
   fsolv->AddBC(7,1.0);
   fsolv->AddBC(8,0.0);

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);


   AlcoaBracket* alco=new AlcoaBracket(&pmesh,1);
   alco->SetDesignFES(pgdens.ParFESpace());
   alco->SetDensity(vdens);
   alco->Solve();
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


   mfem::ParGridFunction solx;
   mfem::ParGridFunction soly;
   spderf.Generate();

   {
      mfem::ParaViewDataCollection paraview_dc("TopOpt", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);

      paraview_dc.RegisterField("design",&pgdens);

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

      for(int i=1;i<max_it;i++){

          /*
          if((i%11)==0){
              spderf.Generate();
              spdegf.ProjectCoefficient(spderf);
          }*/

          vobj->SetProjection(0.3,8.0);
          alco->SetDensity(vdens,0.5,8.0,3.0);
          /*
          if(i<11){alco->SetDensity(vdens,0.5,1.0,3.0); vobj->SetProjection(0.5,1.0); }
          else if(i<80){alco->SetDensity(vdens,0.5,8.0,3.0);}
          else if(i<120){alco->SetDensity(vdens,0.5,8.0,3.0);}
          else if(i<160){alco->SetDensity(vdens,0.6,8.0,3.0); vobj->SetProjection(0.4,8.0); }
          else {alco->SetDensity(vdens,0.7,8.0,3.0);  vobj->SetProjection(0.3,8.0);}
          */
          alco->Solve();

          cpl=alco->MeanCompliance();
          vol=vobj->Eval(vdens);
          ivol=ivobj->Eval(vdens);

          if(myrank==0){
              std::cout<<"it: "<<i<<" obj="<<cpl<<" vol="<<vol<<" cvol="<<max_vol<<" ivol="<<ivol<<std::endl;
          }
          //compute the gradients
          alco->MeanCompliance(ograd);
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
