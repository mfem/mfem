#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "stokes.hpp"

double inlet_vel(const mfem::Vector &x)
{
    double d=(x[1]-1.4);
    if(fabs(d)>0.2){return 0.0;}
    return 1.5*(1-d*d/(0.2*0.2));
}


double AvgErrorThresholdRef(MPI_Comm comm, mfem::Vector& merr, mfem::Vector& derr)
{
    //remove all elements with error smaller than the model error
    double threshold=0.0;
    double loc_sum=0.0;
    int tot_num=0;
    int loc_num=0;
    for(int i=0;i<derr.Size();i++)
    {
        derr[i]=fabs(derr[i]);
        if(derr[i]<fabs(merr[i])){derr[i]=0.0;}
        loc_sum=loc_sum+derr[i];
    }
    MPI_Allreduce(&loc_sum, &threshold, 1, MPI_DOUBLE, MPI_SUM, comm);
    loc_num=derr.Size();
    MPI_Allreduce(&loc_num, &tot_num, 1, MPI_INT, MPI_SUM, comm);

    threshold=threshold/double(tot_num);

    return threshold;
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
   int print_level = 0;
   bool visualization = false;

   const char *petscrc_file = "stokes_fieldsplit";

   int restart=0;



   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&restart,
                  "-rstr",
                  "--restart",
                  "Restart the optimization from previous design.");
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
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels+1; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   if(restart)
   {
       //read the mesh and the design
       std::ostringstream oss;
       oss << std::setw(10) << std::setfill('0') << myrank;
       std::string mname="pmesh_"+oss.str()+".msh";

       std::ifstream in;
       in.open(mname.c_str(),std::ios::in);
       pmesh.Load(in,1,0);
       in.close();
   }



   // Define design space
   //mfem::FiniteElementCollection* dfec=new mfem::L2_FECollection(0,dim);
   mfem::FiniteElementCollection* dfec=new mfem::H1_FECollection(1,dim);
   mfem::ParFiniteElementSpace* dfes=new mfem::ParFiniteElementSpace(&pmesh,dfec);
   mfem::ParGridFunction odesign; odesign.SetSpace(dfes); odesign=0.5;
   mfem::Vector ovector(dfes->GetTrueVSize());
   odesign.GetTrueDofs(ovector);

   if(restart)//read the design from file
   {
       //read the mesh and the design
       std::ostringstream oss;
       oss << std::setw(10) << std::setfill('0') << myrank;
       std::string gname="design_"+oss.str()+".gf";
       std::ifstream in;
       in.open(gname.c_str(),std::ios::in);
       mfem::ParGridFunction ndes(&pmesh,in);
       in.close();
       odesign.ProjectGridFunction(ndes);//avoids mixing the FE spaces
       odesign.GetTrueDofs(ovector);
   }

   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.05,&pmesh,dfes);
   fsolv->SetSolver(1e-8,1e-12,100,print_level);

   //define filtered design
   mfem::ParFiniteElementSpace* ffes=fsolv->GetFilterFES();
   mfem::Vector fvector(ffes->GetTrueVSize()); fvector=0.0;
   fsolv->Mult(ovector,fvector);

   //define the state solver
   mfem::StokesSolver* solver=new mfem::StokesSolver(&pmesh,2);
   solver->SetSolver(rel_tol,abs_tol,tot_iter,print_level);

   mfem::ConstantCoefficient viscosity(1);
   solver->SetViscosity(viscosity);

   mfem::FunctionCoefficient vin(inlet_vel);
   solver->AddVelocityBC(2,0,vin);
   //solver->AddVelocityBC(2,0,1.0);
   solver->AddVelocityBC(1,4,0.0);

   //set design space for the solver - filtered design
   solver->SetDesignSpace(ffes);
   solver->SetDesignParameters(0.1,4.0,1.0,1000);
   solver->SetTargetDesignParameters(0.5,64.0,1,1000*10);
   solver->SetDesign(fvector);
   solver->SetSolver(1e-8,1e-12,100,0);

   if(myrank==0)
   {
       std::cout<<"Design dofs="<<dfes->GetTrueVSize()<<std::endl;
       std::cout<<"Filtered design dofs="<<fsolv->GetFilterFES()->GetTrueVSize()<<std::endl;
       std::cout<<"vel dofs="<<solver->GetVelocityFES()->GetTrueVSize();
       std::cout<<" pres dofs="<<solver->GetPressureFES()->GetTrueVSize()<<std::endl;
   }

   mfem::Vector ograd(ffes->GetTrueVSize()); ograd=0.0; //objective gradeints
   mfem::Vector vgrad(ffes->GetTrueVSize()); vgrad=0.0; //volume gradients

   mfem::Vector ogrado(dfes->GetTrueVSize()); ogrado=0.0;
   mfem::Vector vgrado(dfes->GetTrueVSize()); vgrado=0.0;

   mfem::Vector ndesign(dfes->GetTrueVSize());

   double maxch=0.1; //max design change
   double lam, lam_max, lam_min; //Lagrange multiplier
   int max_it=55;

   //mfem::PowerDissipationQoI objqoi(solver);
   //mfem::AveragePressureDropQoI objqoi(solver);
   mfem::PowerDissipationTGQoI  objqoi(solver);
   mfem::VolumeQoI         volqoi(solver,0.5);

   double obj;
   double vol;
   double Be;
   double xx;

   mfem::ParGridFunction model_error;
   mfem::ParGridFunction discr_error;



   {
       mfem::ParGridFunction& veloc=solver->GetVelocity();
       mfem::ParGridFunction& press=solver->GetPressure();
       mfem::ParGridFunction& aveloc=solver->GetAVelocity();
       mfem::ParGridFunction& apress=solver->GetAPressure();
       mfem::ParGridFunction& fdesign=solver->GetDesign();
       mfem::ParGridFunction  ogradgf; ogradgf.SetSpace(dfes);

       mfem::ParaViewDataCollection paraview_dc("Stokes", &pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("velocity",&veloc);
       paraview_dc.RegisterField("pressure",&press);
       paraview_dc.RegisterField("fdesign",&fdesign);
       paraview_dc.RegisterField("odesign",&odesign);
       paraview_dc.RegisterField("aveloc",&aveloc);
       paraview_dc.RegisterField("apress",&apress);
       paraview_dc.RegisterField("ograd",&ogradgf);

       fsolv->Mult(ovector,fvector);
       solver->SetDesign(fvector);

       for(int i=1;i<max_it;i++){
           solver->FSolve();

           //compute the objective and the vol constraint
           obj=objqoi.Eval();
           vol=volqoi.Eval();
           if(myrank==0){
               std::cout<<"it: "<<i<<" obj="<<obj<<" vol="<<vol<<std::endl;}

           //compute the gradients
           objqoi.Grad(ograd);
           volqoi.Grad(vgrad);
           //compute the original gradients
           fsolv->MultTranspose(ograd,ogrado);
           fsolv->MultTranspose(vgrad,vgrado);


           //OC update
           lam_min=0.0; lam_max=1e9;
           while( (lam_max-lam_min)/(lam_max+lam_min) > 1e-3 ){
               lam=0.5*(lam_max+lam_min);
               for(int j=0;j<ndesign.Size();j++)
               {
                   Be=-ogrado[j]/(lam*vgrado[j]);
                   if(Be>=0.0){ xx=ovector[j]*sqrt(Be);}
                   else{   xx=-ovector[j]*sqrt(-Be); }

                   xx=std::min(ovector[j]+maxch,xx);
                   if(xx>1.0){xx=1.0;}//project to 1.0
                   xx=std::max(ovector[j]-maxch,xx);
                   if(xx<0.0){xx=0.0;}//project to 0.0
                   ndesign[j]=xx;
               }
               fsolv->Mult(ndesign,fvector);
               if( volqoi.Eval(fvector) > 0.0 ){  lam_min=lam;}
               else{lam_max = lam;}
           }

           ovector=ndesign;
           fsolv->Mult(ovector,fvector);
           solver->SetDesign(fvector);

           //save the design and the solution
           if((i%3)==0)
           {
               odesign.SetFromTrueDofs(ovector);
               ogradgf.SetFromTrueDofs(ogrado);
               paraview_dc.SetCycle(i);
               paraview_dc.SetTime(i*1.0);
               paraview_dc.Save();
           }

       //refine

       if((i%8)==0)
       {

           solver->FSolve();
           obj=objqoi.Eval();

           //refine de-reffine
           double lderr=objqoi.DiscretizationError(discr_error);
           double lmerr=objqoi.ModelError(model_error);

           if(pmesh.GetMyRank()==0){
               std::cout<<"obj="<<obj<<" merr="<<lmerr<<" derr="<<lderr<<std::endl;
           }

           double threshold=0.0;
           threshold=AvgErrorThresholdRef(pmesh.GetComm(),model_error,discr_error);

           pmesh.RefineByError(discr_error, threshold, 1, 1);
           //update the design
           odesign.SetFromTrueDofs(ovector);
           dfes->Update();
           odesign.Update();
           ovector.SetSize(dfes->GetTrueVSize());
           odesign.GetTrueDofs(ovector);
           ogradgf.Update();

           //update the filter
           fsolv->Update();
           fvector.SetSize(ffes->GetTrueVSize());
           fsolv->Mult(ovector,fvector);

           //update the solver
           solver->Update();
           solver->SetDesign(fvector);
           model_error.Update();
           discr_error.Update();
           objqoi.Update();
           volqoi.Update();

           pmesh.Rebalance();
           dfes->Update();
           odesign.Update();
           ovector.SetSize(dfes->GetTrueVSize());
           odesign.GetTrueDofs(ovector);
           ogradgf.Update();

           fsolv->Update();
           fvector.SetSize(ffes->GetTrueVSize());
           fsolv->Mult(ovector,fvector);

           solver->Update();
           solver->SetDesign(fvector);
           model_error.Update();
           discr_error.Update();
           objqoi.Update();
           volqoi.Update();

           ogrado.SetSize(dfes->GetTrueVSize()); ograd=0.0; //objective gradeints
           vgrado.SetSize(dfes->GetTrueVSize()); vgrad=0.0; //volume gradients
           ndesign.SetSize(dfes->GetTrueVSize());

           ograd.SetSize(ffes->GetTrueVSize());
           vgrad.SetSize(ffes->GetTrueVSize());
       }


   }

   }

   /*
   {
       mfem::ParGridFunction& veloc=solver->GetVelocity();
       mfem::ParGridFunction& press=solver->GetPressure();
       mfem::ParGridFunction& aveloc=solver->GetAVelocity();
       mfem::ParGridFunction& apress=solver->GetAPressure();

       mfem::ParGridFunction& design=solver->GetDesign();


       mfem::ParGridFunction desograd; desograd.SetSpace(dfes);
       desograd.SetFromTrueDofs(ograd);
       mfem::ParGridFunction desvgrad; desvgrad.SetSpace(dfes);
       desvgrad.SetFromTrueDofs(vgrad);

       mfem::ParaViewDataCollection paraview_dc("Stokes", &pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("velocity",&veloc);
       paraview_dc.RegisterField("pressure",&press);
       paraview_dc.RegisterField("design",&design);
       paraview_dc.RegisterField("ograds",&desograd);
       paraview_dc.RegisterField("vgrads",&desvgrad);
       paraview_dc.RegisterField("aveloc",&aveloc);
       paraview_dc.RegisterField("apress",&apress);
       //paraview_dc.RegisterField("merr",&model_error);
       paraview_dc.Save();
   }
   */


   {
       //save the mesh and the design
       std::ostringstream oss;
       oss << std::setw(10) << std::setfill('0') << myrank;
       std::string mname="pmesh_"+oss.str()+".msh";
       std::string gname="design_"+oss.str()+".gf";
       std::ofstream out;
       out.open(mname.c_str(),std::ios::out);
       pmesh.ParPrint(out);
       out.close();

       //save the design
       odesign.SetFromTrueDofs(ovector);
       out.open(gname.c_str(),std::ios::out);
       odesign.Save(out);
       out.close();
   }



   delete solver;
   delete fsolv;
   delete dfes;
   delete dfec;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
