#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "stokes.hpp"


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
   //mfem::FiniteElementCollection* dfec=new mfem::L2_FECollection(1,dim);
   mfem::FiniteElementCollection* dfec=new mfem::H1_FECollection(1,dim);
   mfem::ParFiniteElementSpace* dfes=new mfem::ParFiniteElementSpace(&pmesh,dfec);
   mfem::ParGridFunction design; design.SetSpace(dfes); design=0.5;
   mfem::Vector tdesign(dfes->GetTrueVSize());
   design.GetTrueDofs(tdesign);

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
       design.ProjectGridFunction(ndes);//avoids mixing the FE spaces
       design.GetTrueDofs(tdesign);
   }


   mfem::StokesSolver* solver=new mfem::StokesSolver(&pmesh,2);

   solver->SetSolver(rel_tol,abs_tol,tot_iter,print_level);

   mfem::ConstantCoefficient viscosity(1);
   solver->SetViscosity(viscosity);
   solver->AddVelocityBC(2,0,1.0);
   solver->AddVelocityBC(1,4,0.0);


   solver->SetDesignSpace(dfes);
   solver->SetDesignParameters(0.1,4.0,1.0,100000);
   solver->SetDesign(tdesign);
   solver->SetSolver(1e-8,1e-12,100,1);


   mfem::Vector ograd(dfes->GetTrueVSize()); ograd=0.0; //objective gradeints
   mfem::Vector vgrad(dfes->GetTrueVSize()); vgrad=0.0; //volume gradients
   mfem::Vector ndesign(dfes->GetTrueVSize());

   double maxch=0.1; //max design change
   double lam, lam_max, lam_min; //Lagrange multiplier
   int max_it=5;

   //mfem::PowerDissipationQoI objqoi(solver);
   mfem::AveragePressureDropQoI objqoi(solver);
   mfem::VolumeQoI           volqoi(solver,0.85);

   double obj;
   double vol;
   double Be;
   double xx;
   double cvol;

   for(int i=0;i<max_it;i++){
       solver->SetDesign(tdesign);
       solver->FSolve();
       //compute the objective and the vol constraint
       obj=objqoi.Eval();
       vol=volqoi.Eval();

       if(myrank==0){
       std::cout<<"it: "<<i<<" obj="<<obj<<" vol="<<vol<<std::endl;}
       //compute the gradients
       objqoi.Grad(ograd);
       volqoi.Grad(vgrad);
       //OC update
       lam_min=0.0; lam_max=1e9;
       while( (lam_max-lam_min)/(lam_max+lam_min) > 1e-3 ){
           lam=0.5*(lam_max+lam_min);
           for(int j=0;j<ndesign.Size();j++)
           {
               Be=-ograd[j]/(lam*vgrad[j]);
               if(Be>=0.0){ xx=tdesign[j]*sqrt(Be);}
               else{   xx=-tdesign[j]*sqrt(-Be); }

               xx=std::min(tdesign[j]+maxch,xx);
               if(xx>1.0){xx=1.0;}//project to 1.0
               xx=std::max(tdesign[j]-maxch,xx);
               if(xx<0.0){xx=0.0;}//project to 0.0
               ndesign[j]=xx;
           }
           if( volqoi.Eval(ndesign) > 0.0 ){  lam_min=lam;}
           else{lam_max = lam;}
       }

       tdesign=ndesign;
   }

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
       solver->SetDesign(tdesign);
       mfem::ParGridFunction& design=solver->GetDesign();
       out.open(gname.c_str(),std::ios::out);
       design.Save(out);
       out.close();

   }





   delete solver;
   delete dfes;
   delete dfec;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
