#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_solvers.hpp"
#include "mtop_coefficients.hpp"

enum geom_type
{
   square,    
   sphere     
};
geom_type geom;

class CoeffHoles:public mfem::Coefficient
{
public:
    CoeffHoles(double pr=0.2)
    {
        period=pr;
    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {

        double x[3]={0.0,0.0,0.0};
        mfem::Vector transip(x, T.GetSpaceDim());
        T.Transform(ip, transip);

        double ang = atan2 (x[1],x[0]);

        double r=sin(ang*5);

        /*
        x[0]=x[0]*2.0*M_PI/period;
        x[1]=x[1]*2.0*M_PI/period;
        x[2]=x[2]*2.0*M_PI/period;
        double r=sin(x[0])*cos(x[1])+sin(x[1])*cos(x[2])+sin(x[2])*cos(x[0]);
        return 0.5+r/8.0;
        */

        return r*0.5+0.5;

    }

private:
    double period;

};


class HeatSink
{
public:

    HeatSink(mfem::ParMesh* pmesh_, int vorder=1, int seed=std::numeric_limits<int>::max())
    {
        mat=new mfem::DiffusionMaterial(pmesh_->SpaceDimension());
        dsolv=new mfem::DiffusionSolver(pmesh_,vorder);
        dsolv->AddMaterial(mat);
        dsolv->SetVolInput(1.0);

        pmesh=pmesh_;
        dfes=nullptr;
        obj=new mfem::DiffusionComplianceObj();

        generator.seed(seed);

        gf=new mfem::RandFieldCoefficient(pmesh_,vorder);
        gf->SetCorrelationLen(0.2, 0.2,0.2);
        gf->SetMaternParameter(1.0);
        gf->SetScale(1.0);
        gf->Sample(seed);

        af=new mfem::RandFieldCoefficient(pmesh_,vorder);
        af->SetCorrelationLen(0.2, 0.2, 0.2);
        af->SetMaternParameter(1.0);
        af->SetScale(1.0);
        af->SetZeroDirichletBC(2);
        af->Sample(seed+1347);

        num_samples=50;
    }

    ~HeatSink()
    {
        delete obj;
        delete dsolv;

        delete af;
        delete gf;
    }

    void GetSRand(mfem::ParGridFunction& pgf){
        pgf.ProjectCoefficient(*gf);
    }

    void GetARand(mfem::ParGridFunction& pgf){
        pgf.ProjectCoefficient(*af);
    }

    void SetNumSamples(int ns)
    {
        num_samples=ns;
    }

    void SetCorrelationLen(double lx, double ly, double lz)
    {
        gf->SetCorrelationLen(lx, ly, lz);
        af->SetCorrelationLen(lx, ly, lz);
    }

    void SetRotationAngles(double angle_x, double angle_y, double angle_z)
    {
        gf->SetRotationAngles(angle_x, angle_y, angle_z);
        af->SetRotationAngles(angle_x, angle_y, angle_z);
    }

    void SetMaternParameter(double nu)
    {
        gf->SetMaternParameter(nu);
        af->SetMaternParameter(nu);
    }

    void SetDesignFES(mfem::ParFiniteElementSpace* fes)
    {
        dfes=fes;
        pdens.SetSpace(dfes);
        vdens.SetSize(dfes->GetTrueVSize());
    }

    void SetDensity(mfem::Vector& vdens_,
                    double eta=0.5, double beta=8.0,double pen=1.0){

        vdens=vdens_;
        pdens.SetFromTrueDofs(vdens);

        mat->SetDens(&pdens);
        mat->SetProjParam(eta,beta);
        mat->SetEMaxMin(1e-6,1.0);
        mat->SetPenal(pen);

        obj->SetDiffMaterial(mat);
        obj->SetDens(vdens);
        obj->SetDesignFES(dfes);

    }

    void SetSIMP(bool simp_=false){
        mat->SetProjection(!simp_);
    }

    double Compliance(mfem::Vector& grad)
    {
        grad=0.0;
        dsolv->DelDirichletBC();
        dsolv->AddDirichletBC(2,0.0);
        dsolv->FSolve();
        obj->SetDiffSolver(dsolv);
        double vv=obj->Eval();
        obj->Grad(grad);
        return vv;
    }

    double MeanCompl(mfem::Vector& grad)
    {
        mfem::Vector lgr(grad.Size());
        grad=0.0;

        dsolv->DelDirichletBC();
        if (geom == geom_type::square)
        {
          dsolv->AddDirichletBC(2,0.0);
        }
        else
        {
          dsolv->AddDirichletBC(1,0.0);
        }
        dsolv->AssembleTangent();

        std::uniform_int_distribution<int> uint(1,std::numeric_limits<int>::max());
        int n=num_samples;
        double robj=0.0;
        double rvar=0.0;
        for(int i=0;i<n;i++){
            if(seeds.size()<(i+1)){
                int seed = uint(generator);
                seeds.push_back(seed);
            }
            gf->Sample(seeds[i]);

            dsolv->SetVolInput(*gf);
            dsolv->LSolve();

            obj->SetDiffSolver(dsolv);
            double vv=obj->Eval();
            obj->Grad(lgr);

            robj=robj+vv;
            grad.Add(1.0,lgr);
            rvar=rvar+vv*vv;
        }
        grad/=double(n);
        rvar=rvar/double(n);
        robj=robj/double(n);

        int myrank;
        MPI_Comm_rank(pmesh->GetComm(),&myrank);
        if(myrank==0){
        std::cout<<"Var="<<rvar-robj*robj<<std::endl;}

        return robj;
    }

private:
    mfem::DiffusionMaterial* mat;
    mfem::DiffusionSolver* dsolv;
    mfem::ParMesh* pmesh;

    mfem::ParFiniteElementSpace* dfes;
    mfem::ParGridFunction pdens;
    mfem::Vector vdens;

    mfem::DiffusionComplianceObj* obj;

    mfem::RandFieldCoefficient* gf;//symmetric load
    mfem::RandFieldCoefficient* af;//asymmetric load

    int num_samples;
    std::vector<int> seeds;
    std::vector<int> seeda;

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
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   double corr_len_x = 0.2;
   double corr_len_y = 0.2;
   double corr_len_z = 0.2;
   double angle_x = 0.0;
   double angle_y = 0.0;
   double angle_z = 0.0;
   int num_samples=100;
   int tot_iter = 100;
   int max_it = 51;
   int print_level = 1;
   bool visualization = false;
   const char *petscrc_file = "";
   int restart=0;
   double volume_fraction = 0.35;
   int igeom = 0;
   double nu = 1.0;

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
   args.AddOption(&volume_fraction,
                  "-vf",
                  "--volume-fraction",
                  "Volume fraction");                   
   args.AddOption(&num_samples,
                  "-ns",
                  "--num-samples",
                  "Number of samples.");
   args.AddOption(&corr_len_x,
                  "-crlx",
                  "--corr-x",
                  "Correlation length in x");
   args.AddOption(&corr_len_y,
                  "-crly",
                  "--corr-y",
                  "Correlation length in y");  
   args.AddOption(&corr_len_z,
                  "-crlz",
                  "--corr-z",
                  "Correlation length in z"); 
   args.AddOption(&angle_x, "-e1", "--e1",
                  "Rotation angle in x direction");
   args.AddOption(&angle_y, "-e2", "--e2",
                  "Rotation angle in y direction");             
   args.AddOption(&angle_z, "-e3", "--e3",
                  "Rotation angle in z direction");   
   args.AddOption(&nu, "-nu", "--nu",
                  "Matern Parameter (Smoothness of random field)");                     
   args.AddOption(&igeom, "-g", "--geom", "Geometry type"
                  "0: square, 1: sphere");                                                                                   
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


   MFEM_VERIFY((igeom == 0 || igeom == 1), "Wrong choice of geometry kind");
   geom = (geom_type)igeom;

   int seed;
   {
       std::default_random_engine generator;
       std::uniform_int_distribution<int> distribution(1,std::numeric_limits<int>::max());

       for(int i=0;i<nprocs;i++){
           if(i!=0){
               if(myrank==0){
                   int rnum=distribution(generator);
                   MPI_Send(&rnum,1,MPI_INT,i,100,MPI_COMM_WORLD);}
               if(myrank==i){
                   MPI_Recv(&seed,1,MPI_INT,0,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);}
           }else{
               seed=distribution(generator);
           }
       }
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
   // mfem::out << mesh.bdr_attributes.Max() << std::endl;

   /*
   {
       mesh.EnsureNodes();
       mfem::Vector vert;
       //mesh.GetVertices(vert);
       mfem::GridFunction* nod=mesh.GetNodes();

       std::cout<<"Size="<<nod->Size()<<std::endl;

       int nv=nod->Size()/3;

       for(int i=0;i<nv;i++){
           double xx=(*nod)[3*i+0];
           double yy=(*nod)[3*i+1];
           double zz=(*nod)[3*i+2];

           double dd=pow(xx,8.0)+pow(yy,8.0)+pow(zz,8.0);
           dd=pow(dd,1.0/8.0);

           (*nod)[3*i+0] = xx/dd;
           (*nod)[3*i+1] = yy/dd;
           (*nod)[3*i+2] = zz/dd;
       }


       //mesh.SetVertices(vert);
       mfem::Vector xmin(dim), xmax(dim);
       mesh.GetBoundingBox(xmin,xmax);
       if(myrank==0){
           std::cout<<"Xmin:";xmin.Print(std::cout);
           std::cout<<"Xmax:";xmax.Print(std::cout);
       }
   }*/

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels = ser_ref_levels;
        //  (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);

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
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(fradius,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   
   if (geom == geom_type::square)
   {
      fsolv->AddBC(2,1.0);
   }
   else
   {
      fsolv->AddBC(1,1.0);
   }

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   //mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);

   HeatSink* sink=new HeatSink(&pmesh,1,seed);
   sink->SetDesignFES(pgdens.ParFESpace());
   sink->SetDensity(vdens);
   sink->SetCorrelationLen(corr_len_x,corr_len_y, corr_len_z);
   sink->SetRotationAngles(angle_x, angle_y, angle_z);
   sink->SetMaternParameter(nu);
   sink->SetNumSamples(num_samples);

   mfem::VolumeQoI* vobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   //mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(fsolv->GetFilterFES());
   vobj->SetProjection(0.5,8.0);//threshold 0.2


   //compute the total volume
   double tot_vol;
   {
       vdens=1.0;
       tot_vol=vobj->Eval(vdens);
   }
   double max_vol=volume_fraction*tot_vol;
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

   {

      vtmpv=0.2;
      CoeffHoles cf;
      oddens.ProjectCoefficient(cf);
      oddens.GetTrueDofs(vtmpv);
      if (geom == geom_type::square)
      {
        vtmpv=0.2;
      }
      fsolv->Mult(vtmpv,vdens);
      pgdens.SetFromTrueDofs(vdens);


      std::ostringstream paraview_file_name;
      paraview_file_name << "TopOpt:ser_ref_" << ser_ref_levels  
                                         << "_par_ref_" << par_ref_levels 
                                         << "_order_" << order 
                                         << "_num_samples_" << num_samples
                                         << "_max_it_" << max_it
                                         << "_crlx_" << corr_len_x 
                                         << "_crly_" << corr_len_y 
                                         << "_angle_x_" << angle_x 
                                         << "_volume_fraction_" << volume_fraction
                                         << "_radius_" << fradius
                                         << "_maternparam_" << nu
                                         << "_geom_" << igeom;
      mfem::ParaViewDataCollection paraview_dc(paraview_file_name.str(), &pmesh);
      {
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetCycle(0);
         paraview_dc.SetTime(0.0);
         paraview_dc.RegisterField("design",&pgdens);
         paraview_dc.Save();
      }

      for(int i=1;i<max_it;i++){

          if(i<10){
              vobj->SetProjection(0.5,2.0);
              sink->SetDensity(vdens,0.5,8.0,1.0);
              sink->SetSIMP(true);
          }else if(i<300){
              vobj->SetProjection(0.5,2.0);
              sink->SetDensity(vdens,0.5,2.0,3.0);
              sink->SetSIMP(true);
          }else{
              vobj->SetProjection(0.5,2.0);
              sink->SetDensity(vdens,0.5,2.0,4.0);
              sink->SetSIMP(true);
          }

          vol=vobj->Eval(vdens);
          ivol=ivobj->Eval(vdens);
        //   cpl=sink->Compliance(ograd);
          cpl=sink->MeanCompl(ograd);


          if(myrank==0){
              std::cout<<"it: "<<i<<" obj="<<cpl<<" vol="<<vol<<" cvol="<<max_vol<<" ivol="<<ivol<<std::endl;
          }
          //compute the gradients
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

          //save the design
        //   if(i%4==0)
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
   delete sink;
   delete fsolv;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
