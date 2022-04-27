#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_filters.hpp"


// example runs
// mpirun -np 4 ./stokes -m ./ball2D.msh -petscopts ./stokes_fieldsplit
// mpirun -np 4 ./stokes -m ./ball2D.msh -petscopts ./stokes_fieldsplit_01


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
        rez=0.0;
    }
    return rez;
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
   double fradius = 0.15;
   int tot_iter = 100;
   int print_level = 1;
   bool visualization = false;

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
   args.AddOption(&fradius,
                  "-r",
                  "--radius",
                  "Filter radius");
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
   /*
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }*/

   mfem::FilterSolver* filter=new mfem::FilterSolver(fradius,&pmesh);
   filter->SetSolver(1e-8,1e-12,100,0);

   mfem::ParFiniteElementSpace* dfes=filter->GetDesignFES();
   mfem::ParGridFunction design; design.SetSpace(dfes); design=0.0;
   mfem::Vector tdesign(dfes->GetTrueVSize());
   //project the characteristic function onto the design space
   {
        mfem::FunctionCoefficient fco(charfunc);
        design.ProjectCoefficient(fco);
   }
   design.GetTrueDofs(tdesign);


   mfem::ParGridFunction fdesign; fdesign.SetSpace(filter->GetFilterFES());
   mfem::Vector tfdesign(filter->GetFilterFES()->GetTrueVSize()); tfdesign=0.0;

   filter->Mult(tdesign,tfdesign);
   fdesign.SetFromTrueDofs(tfdesign);

   if(myrank==0)
   {
       std::cout<<" idof="<<tdesign.Size()<<" fdof="<<tfdesign.Size()<<std::endl;
   }

   {

       mfem::ParaViewDataCollection paraview_dc("Stokes", &pmesh);
       paraview_dc.SetPrefixPath("Filter");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("idesign",&design);
       paraview_dc.RegisterField("fdesign",&fdesign);
       paraview_dc.Save();

   }

   if(myrank==0)
   {
       std::cout<<"Design is filtered! Nexxt step: Test gradients.";
   }

   //test gradients
   {

       mfem::PVolumeQoI qoi(filter->GetFilterFES());
       qoi.SetProjection(0.5,4.0);

       mfem::Vector ograd(filter->GetFilterFES()->GetTrueVSize());
       mfem::Vector tgrad(dfes->GetTrueVSize()); tgrad=0.0;

       //scale the design
       tdesign*=0.5;
       //filter it
       filter->Mult(tdesign,tfdesign);

       double val=qoi.Eval(tfdesign);
       qoi.Grad(tfdesign,ograd);
       filter->MultTranspose(ograd,tgrad);

       mfem::Vector prtv;
       mfem::Vector tmpv;

       prtv.SetSize(tgrad.Size());
       tmpv.SetSize(tgrad.Size());

       prtv.Randomize();

       double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
       double td=mfem::InnerProduct(pmesh.GetComm(),prtv,tgrad);

       td=td/nd;
       double lsc=1.0;
       double lqoi;

       for(int l=0;l<10;l++){
           lsc/=10.0;
           prtv/=10.0;
           add(prtv,tdesign,tmpv);
           filter->Mult(tmpv,tfdesign);
           lqoi=qoi.Eval(tfdesign);
           double ld=(lqoi-val)/lsc;
           if(myrank==0){
               std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                         << " adjoint gradient=" << td
                         << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
       }

   }

   delete filter;

   MPI_Finalize();
   return 0;
}
