#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_solvers.hpp"


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

      double x[3]= {0.0,0.0,0.0};
      mfem::Vector transip(x, T.GetSpaceDim());
      T.Transform(ip, transip);

      x[0]=x[0]*2.0*M_PI/period;
      x[1]=x[1]*2.0*M_PI/period;
      x[2]=x[2]*2.0*M_PI/period;
      double r=sin(x[0])*cos(x[1])+sin(x[1])*cos(x[2])+sin(x[2])*cos(x[0]);

      return 0.5+r/8.0;

   }

private:
   double period;

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
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
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
   int sdim = mesh.SpaceDimension();
   if (myrank==0)
   {
      std::cout<<"Dim:"<<dim<<" Sdim:"<<sdim<<std::endl;
   }


   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
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

   double r=0.1;
   mfem::FilterSolver* filt=new mfem::FilterSolver(r,&pmesh);
   filt->SetSolver(1e-8,1e-12,100,0);
   filt->AddBC(1,1.0);


   mfem::ParGridFunction pgdens(filt->GetFilterFES());
   mfem::ParGridFunction oddens(filt->GetDesignFES());
   mfem::Vector vdens; vdens.SetSize(filt->GetFilterFES()->GetTrueVSize());
   vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(filt->GetDesignFES()->GetTrueVSize());
   vtmpv=0.5;

   CoeffHoles cfh(0.5);
   oddens.ProjectCoefficient(cfh);
   oddens.GetTrueDofs(vtmpv);
   filt->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);

   //define the material tensor
   mfem::DiffusionMaterial* dmat=new mfem::DiffusionMaterial(sdim);
   dmat->SetDens(&pgdens);


   mfem::DiffusionSolver* solver=new mfem::DiffusionSolver(&pmesh,1);
   solver->SetNewtonSolver(1e-7, 1e-12,1000, 0);

   /*
   solver->AddDirichletBC(2,0.0);
   solver->AddDirichletBC(3,0.0);
   solver->AddDirichletBC(4,0.0);
   solver->AddDirichletBC(5,0.0);
   */
   solver->AddDirichletBC(1,0.0);
   solver->SetVolInput(1.0);
   solver->AddMaterial(dmat);


   /*
   mfem::DenseMatrix dcoef(pmesh.SpaceDimension()); dcoef=0.0;
   for(int d=0;d<pmesh.SpaceDimension();d++){
       dcoef(d,d)=1.0;
   }
   solver->AddMaterial(new mfem::MatrixConstantCoefficient(dcoef));
   */

   solver->FSolve();


   //define compliance objective
   mfem::DiffusionComplianceObj* obj=new mfem::DiffusionComplianceObj();
   obj->SetDiffSolver(solver);
   obj->SetDesignFES(filt->GetFilterFES()); //the FEM which enters the material
   obj->SetDiffMaterial(dmat);
   obj->SetDens(vdens);


   double oo=0.0;
   oo=obj->Eval();
   std::cout<<"Obj: "<<oo<<std::endl;

   //gradients
   mfem::Vector ograd; ograd.SetSize(filt->GetFilterFES()->GetTrueVSize());
   mfem::Vector ogrado; ogrado.SetSize(filt->GetDesignFES()->GetTrueVSize());

   obj->Grad(ograd);
   filt->MultTranspose(ograd,ogrado);

   mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(filt->GetFilterFES());
   vobj->SetProjection(0.2,8.0);//threshold 0.2
   double tot_vol;
   tot_vol=vobj->Eval(vdens);
   std::cout<<"Vol: "<<tot_vol<<std::endl;




   {
      //finite difference check
      mfem::Vector prtv;
      mfem::Vector tmpv;
      mfem::Vector tgrad;
      mfem::Vector fgrad;
      prtv.SetSize(vtmpv.Size());
      tmpv.SetSize(vtmpv.Size());
      tgrad.SetSize(vtmpv.Size());
      fgrad.SetSize(vdens.Size()); fgrad=0.0;


      pgdens.SetFromTrueDofs(vdens);
      solver->FSolve();
      obj->SetDens(vdens);


      double val=obj->Eval();
      obj->Grad(fgrad);
      filt->MultTranspose(fgrad,tgrad);

      prtv.Randomize();
      double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
      double td=mfem::InnerProduct(pmesh.GetComm(),prtv,tgrad);
      td=td/nd;
      double lsc=1.0;
      double lqoi;

      for (int l=0; l<10; l++)
      {
         lsc/=10.0;
         prtv/=10.0;
         add(prtv,vtmpv,tmpv);
         filt->Mult(tmpv,vdens);
         pgdens.SetFromTrueDofs(vdens);
         solver->FSolve();
         obj->SetDens(vdens);
         lqoi=obj->Eval();
         double ld=(lqoi-val)/lsc;
         if (myrank==0)
         {
            std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                      << " adjoint gradient=" << td
                      << " err=" << std::fabs(ld/nd-td) << std::endl;
         }
      }
   }





   mfem::ParGridFunction& gsol=solver->GetFSolution();

   {
      mfem::ParaViewDataCollection paraview_dc("Diffusion",&pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("temp",&gsol);
      paraview_dc.RegisterField("dens",&pgdens);
      paraview_dc.Save();
   }



   delete obj;
   delete vobj;
   delete solver;
   delete filt;

   MPI_Finalize();
   return 0;

}
