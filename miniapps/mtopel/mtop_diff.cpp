#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_solvers.hpp"

#include "spde_solvers.hpp"



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

    HeatSink(mfem::ParMesh* pmesh_, int vorder=1)
    {
        mat=new mfem::DiffusionMaterial(pmesh_->SpaceDimension());
        dsolv=new mfem::DiffusionSolver(pmesh_,vorder);
        dsolv->AddMaterial(mat);
        dsolv->SetVolInput(0.1);


        pmesh=pmesh_;
        dfes=nullptr;
        obj=new mfem::DiffusionComplianceObj();

    }

    ~HeatSink()
    {
        delete obj;
        delete dsolv;
    }


    void Solve()
    {

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

    double Compliance(mfem::Vector& grad)
    {
        grad=0.0;
        mat->SetProjParam(0.7,8.0);
        dsolv->DelDirichletBC();
        dsolv->AddDirichletBC(1,0.0);
        dsolv->FSolve();
        obj->SetDiffSolver(dsolv);
        double vv=obj->Eval();
        obj->Grad(grad);
        return vv;
    }


private:
    mfem::DiffusionMaterial* mat;
    mfem::DiffusionSolver* dsolv;
    mfem::ParMesh* pmesh;

    mfem::ParFiniteElementSpace* dfes;
    mfem::ParGridFunction pdens;
    mfem::Vector vdens;

    mfem::DiffusionComplianceObj* obj;

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
       //vert*=0.01;
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
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.05,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,1.0);


   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   //mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);

   HeatSink* sink=new HeatSink(&pmesh,1);
   sink->SetDesignFES(pgdens.ParFESpace());
   sink->SetDensity(vdens);

   mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(fsolv->GetFilterFES());
   //mfem::VolumeQoI* vobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   vobj->SetProjection(0.2,8.0);//threshold 0.2


   //compute the total volume
   double tot_vol;
   {
       vdens=1.0;
       tot_vol=vobj->Eval(vdens);
   }
   double max_vol=0.35*tot_vol;
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
      fsolv->Mult(vtmpv,vdens);
      pgdens.SetFromTrueDofs(vdens);

      mfem::ParaViewDataCollection paraview_dc("TopOpt", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);

      paraview_dc.RegisterField("design",&pgdens);
      paraview_dc.Save();

      for(int i=1;i<max_it;i++){

          vobj->SetProjection(0.2,8.0);
          vol=vobj->Eval(vdens);
          ivol=ivobj->Eval(vdens);

          sink->SetDensity(vdens,0.8,8.0,1.0);
          cpl=sink->Compliance(ograd);

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
   delete sink;
   delete fsolv;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
