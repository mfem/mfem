#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "stokes.hpp"

// example runs
// mpirun -np 4 ./stokes -m ./ball2D.msh -petscopts ./stokes_fieldsplit
// mpirun -np 4 ./stokes -m ./ball2D.msh -petscopts ./stokes_fieldsplit_01

double inlet_vel(const mfem::Vector &x)
{
    double d=(x[1]-1.4);
    if(fabs(d)>0.2){return 0.0;}
    return 1.5*(1-d*d/(0.2*0.2));
}

double charfunc(const mfem::Vector &x)
{
    double nx=(x[0]-1.5)*(x[0]-1.5)+(x[1]-0.5)*(x[1]-0.5);
    if(x.Size()==3){
        nx=nx+(x[2]-0.5)*(x[2]-0.5);
    }

    nx=std::sqrt(nx);
    double r=0.25;

    double rez=1.0;
    if(nx<=1.5*r){
        double a=-26.0/27.0;;
        double b=62.0/27.0;
        double c=-5.0/6.0;
        nx=nx/r;
        rez=a*nx*nx*nx+b*nx*nx+c*nx;
        if(rez<0.0){rez=0.0;}
    }
    return rez;
}

/// returns a refinement threshold which will respect the max_elements number and the ratio
/// of the ref_elements/tot_elements<=perc
double LocErrorThresholdRef(MPI_Comm comm, mfem::Vector& merr, mfem::Vector& derr, double perc)
{

    //remove all elements with error smaller than the model error
    double threshold=0.0;
    double loc_sum=0.0;
    int tot_num=0;
    int loc_num=0;
    int ref_num;
    for(int i=0;i<derr.Size();i++)
    {
        derr[i]=fabs(derr[i]);
        if(derr[i]<fabs(merr[i])){derr[i]=0.0;}
        if(loc_sum<derr[i]){loc_sum=derr[i];}
    }
    MPI_Allreduce(&loc_sum, &threshold, 1, MPI_DOUBLE, MPI_MAX, comm);
    loc_num=derr.Size();
    MPI_Allreduce(&loc_num, &tot_num, 1, MPI_INT, MPI_SUM, comm);

    double min_tr;
    double max_tr=threshold;
    min_tr=threshold/100.0;
    bool flag=true;
    while(flag)
    {
        loc_num=0;
        ref_num=0;
        for(int i=0;i<derr.Size();i++)
        {
            if(derr[i]>min_tr){loc_num=loc_num+1;}
        }
        MPI_Allreduce(&loc_num, &ref_num, 1, MPI_INT, MPI_SUM, comm);
        if(ref_num>perc*tot_num)
        {
            min_tr=min_tr+threshold/100.0;
        }
        else
        {
            flag=false;
        }

    }

    return min_tr;
}


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
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   int tot_iter = 100;
   int print_level = 1;
   bool visualization = false;

   const char *petscrc_file = "stokes_fieldsplit";

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
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                                 "PetscOptions file to use.");
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
   mesh.EnsureNCMesh(true);

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(100./mesh.GetNE())/log(2.)/dim);
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
   /*
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }*/


   // Define design space
   //mfem::FiniteElementCollection* dfec=new mfem::L2_FECollection(1,dim);
   mfem::FiniteElementCollection* dfec=new mfem::H1_FECollection(1,dim);
   mfem::ParFiniteElementSpace* dfes=new mfem::ParFiniteElementSpace(&pmesh,dfec);
   mfem::ParGridFunction design; design.SetSpace(dfes); design=0.0;
   mfem::Vector tdesign(dfes->GetTrueVSize());
   //project the characteristic function onto the design space
   mfem::FunctionCoefficient fco(charfunc);
   design.ProjectCoefficient(fco);
   design.GetTrueDofs(tdesign);


   mfem::StokesSolver* solver=new mfem::StokesSolver(&pmesh,2);

   solver->SetSolver(rel_tol,abs_tol,tot_iter,print_level);

   mfem::ConstantCoefficient viscosity(1);
   solver->SetViscosity(viscosity);

   mfem::FunctionCoefficient vin(inlet_vel);
   solver->AddVelocityBC(2,0,vin);
   //solver->AddVelocityBC(2,0,1.0);
   solver->AddVelocityBC(1,4,0.0);

   mfem::Vector vload(2); vload(0)=0.0; vload(1)=0.0; //vload(2)=0.0;
   mfem::VectorConstantCoefficient load(vload);
   solver->SetVolForces(load);

   solver->SetDesignSpace(dfes);
   solver->SetDesignParameters(0.5,8.0,1,200*1000);
   solver->SetTargetDesignParameters(0.5,64.0,1,1000*1000);
   solver->SetDesign(tdesign);
   //solver->SetSolver(1e-15,1e-12,1000,1);
   solver->SetSolver(1e-12,1e-12,100,0);

   solver->FSolve();

   mfem::Vector gradd(dfes->GetTrueVSize()); gradd=0.0;
   mfem::ParGridFunction model_error;
   mfem::ParGridFunction discr_error;

   //refinement loop
   {
       if(myrank==0){
           std::cout<<"pdofs="<<solver->GetPressureFES()->GlobalTrueVSize()<<std::endl;
           std::cout<<"vdofs="<<solver->GetVelocityFES()->GlobalTrueVSize()<<std::endl;
           std::cout<<"ddofs="<<solver->GetDesignFES()->GlobalTrueVSize()<<std::endl;
       }



       mfem::VelocityIntQoI pdqoi(solver,3);
       double qoi=pdqoi.Eval();
       if(myrank==0){std::cout<<"QoI="<<qoi<<std::endl;}

       double derr=pdqoi.DiscretizationError(discr_error);
       if(myrank==0){std::cout<<"Discr. error="<<derr<<std::endl;}

       //pdqoi.Grad(gradd);
       double merr=pdqoi.ModelError(model_error);
       if(myrank==0){std::cout<<"Model error="<<merr<<std::endl;}

       mfem::ParGridFunction& veloc=solver->GetVelocity();
       mfem::ParGridFunction& press=solver->GetPressure();
       mfem::ParGridFunction& aveloc=solver->GetAVelocity();
       mfem::ParGridFunction& apress=solver->GetAPressure();

       mfem::ParGridFunction pdesign; pdesign.SetSpace(dfes);
       pdesign.ProjectCoefficient(*(solver->GetBrinkmanPenal()));

       mfem::ParGridFunction desgrad; desgrad.SetSpace(dfes);
       desgrad.SetFromTrueDofs(gradd);

       mfem::ParaViewDataCollection paraview_dc("Stokes", &pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("velocity",&veloc);
       paraview_dc.RegisterField("pressure",&press);
       paraview_dc.RegisterField("idesign",&design);
       paraview_dc.RegisterField("pdesign",&pdesign);
       paraview_dc.RegisterField("grads",&desgrad);
       paraview_dc.RegisterField("aveloc",&aveloc);
       paraview_dc.RegisterField("apress",&apress);
       paraview_dc.RegisterField("merr",&model_error);
       paraview_dc.RegisterField("derr",&discr_error);
       paraview_dc.Save();


       int max_elements=10000;


       for(int re=1;re<10;re++)
       {
           //refine
           //make all errors positive
           double threshold=0.0;
           /*
           {
               double loc_max=0.0;
               double loc_sum=0.0;
               int loc_siz=discr_error.Size();
               for(int i=0;i<discr_error.Size();i++)
               {
                   if(discr_error[i]<0.0){discr_error[i]=-discr_error[i];}
                   if(discr_error[i]>loc_max){loc_max=discr_error[i];}
                   loc_sum=loc_sum+discr_error[i];
               }

               double global_sum;
               int global_siz;

               MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
               MPI_Allreduce(&loc_max, &threshold, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
               MPI_Allreduce(&loc_siz,&global_siz,1,MPI_INT, MPI_SUM, pmesh.GetComm());


               if(myrank==0)
               {
                   std::cout<<"Total discretization error="<<global_sum<<" loc="<<loc_siz<<" tot_siz="<<global_siz<<std::endl;
               }

               threshold=global_sum/global_siz;

           }*/

           model_error=0.0;
           threshold=LocErrorThresholdRef(pmesh.GetComm(),model_error,discr_error,0.2);
           if(myrank==0){
               std::cout<<"num="<<re<<"  threshold="<<threshold<<std::endl;
           }

           pmesh.RefineByError(discr_error, threshold, 1, 1);

           dfes->Update();
           design.Update();
           design.ProjectCoefficient(fco);
           tdesign.SetSize(dfes->GetTrueVSize());
           design.GetTrueDofs(tdesign);

           //mesh.Rebalance();
           solver->Update();
           solver->SetDesign(tdesign);
           solver->FSolve();
           pdesign.Update();
           desgrad.Update();
           model_error.Update();
           discr_error.Update();
           pdqoi.Update();
           qoi=pdqoi.Eval();
           if(myrank==0){std::cout<<"QoI="<<qoi<<std::endl;}
           derr=pdqoi.DiscretizationError(discr_error);
           if(myrank==0){std::cout<<"Discr. error="<<derr<<std::endl;}
           merr=pdqoi.ModelError(model_error);
           if(myrank==0){std::cout<<"Model error="<<merr<<std::endl;}

           paraview_dc.SetCycle(re);
           paraview_dc.SetTime(1.0*re);
           paraview_dc.Save();

           pmesh.Rebalance();
           solver->Update();
           pdesign.Update();
           desgrad.Update();
           model_error.Update();
           discr_error.Update();
           pdqoi.Update();

           if(myrank==0){
               std::cout<<"pdofs="<<solver->GetPressureFES()->GlobalTrueVSize()<<std::endl;
               std::cout<<"vdofs="<<solver->GetVelocityFES()->GlobalTrueVSize()<<std::endl;
               std::cout<<"ddofs="<<solver->GetDesignFES()->GlobalTrueVSize()<<std::endl;
           }

       }



       //update design
       //compute error
       //save paraview
   }

   delete solver;
   delete dfes;
   delete dfec;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;

}



