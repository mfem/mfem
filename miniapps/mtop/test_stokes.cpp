#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_solvers.hpp"

using namespace mfem;

class BrinkCoeff :public Coefficient
{
public:
    BrinkCoeff(real_t penal_=10.0):penalty(penal_)
    {
    }

    virtual real_t Eval(ElementTransformation &T,
                        const IntegrationPoint &ip) override
    {
        real_t x[3];
        Vector transip(x, 3);
        T.Transform(ip, transip);

        real_t c[3]={0.2,0.2,0.0};
        real_t r=0.0;
        real_t d;
        for(int i=0;i<3;i++){
            d=x[i]-c[i];
            r=r+d*d;
        }

        r=sqrt(r);
        if(r>0.05){return 0.0;}
        else{ return penalty;}
    }

private:
    real_t penalty;
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   mfem::Mpi::Init(argc, argv);
   int myrank = mfem::Mpi::WorldRank();
   mfem::Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "./dfg_bench_flow_tri.msh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   real_t newton_rel_tol = 1e-7;
   real_t newton_abs_tol = 1e-12;
   int newton_iter = 10;
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
   args.AddOption(&newton_rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter,
                  "-it",
                  "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
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
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   std::cout<<"My rank="<<pmesh.GetMyRank()<<std::endl;

   StokesSolver* solver=new StokesSolver(&pmesh,2);


   mfem::Vector bci(dim); bci=1.0; bci(1)=0.0;
   mfem::Vector zvi(dim); zvi=0.0;
   std::shared_ptr<VectorCoefficient> cvci;
   cvci.reset(new VectorConstantCoefficient(bci));
   std::shared_ptr<VectorCoefficient> zvci;
   zvci.reset(new VectorConstantCoefficient(zvi));

   solver->AddVelocityBC(1,cvci);
   solver->AddVelocityBC(2,cvci);
   solver->AddVelocityBC(3,cvci);
   //solver->AddVelocityBC(4,cvci);
   //solver->AddVelocityBC(5,zvci);

   std::shared_ptr<Coefficient> brink;
   brink.reset(new BrinkCoeff(1000.0));

   solver->SetBrink(brink);

   ParGridFunction pg(solver->GetVelocitySpace()); pg=0.0;
   ParGridFunction ng(solver->GetVelocitySpace()); ng=0.0;
   ng.SetTrueVector();
   solver->SetEssVBC(pg);
   Vector pgv(ng.GetTrueVector());

   solver->SetEssTDofsV(pgv);

   ng.SetFromTrueDofs(pgv);

   //solver->SetZeroMeanPressure(true);
   solver->SetLinearSolver(1e-8,1e-12,50);

   solver->Assemble();
   solver->FSolve();

   //dump the solution
   {
      ParGridFunction& vel=solver->GetVelocity();
      ParGridFunction& pre=solver->GetPressure();

      ParaViewDataCollection paraview_dc("stokes_flow", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("vel",&vel);
      paraview_dc.RegisterField("pres",&pre);
      paraview_dc.RegisterField("pg",&pg);
      paraview_dc.RegisterField("ng",&ng);
      paraview_dc.Save();
   }

   delete solver;

   MPI::Finalize();
   return 0;
}


