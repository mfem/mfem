#include "mfem.hpp"
#include "problems.hpp"
#include "IPsolver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



double dmanufacturedFun(const Vector &);

int main(int argc, char *argv[])
{
  // Initialize MPI
  MPI::Init(argc, argv);
  int nProcs = Mpi::WorldSize();
  int rank   = Mpi::WorldRank();
  bool iAmRoot = 1;
  MFEM_ASSERT(nProcs == 1, "work needs to be done before a parallel run");
	

  int FEorder = 1; // order of the finite elements
  int linSolver = 0;
  int maxIPMiters = 30;
  const char *device_config = "cpu";
  OptionsParser args(argc, argv);
  args.AddOption(&FEorder, "-o", "--order",\
		  "Order of the finite elements.");
  args.AddOption(&linSolver, "-linSolver", "--linearSolver", \
       "IP-Newton linear system solution strategy.");
  args.AddOption(&maxIPMiters, "-IPMiters", "--IPMiters",\
		  "Maximum number of IPM iterations");
  
  args.Parse();
  if(!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }
  else
  {
    if(iAmRoot)
    {  
      args.PrintOptions(cout);
    }
  }

  const char *meshFile = "../../data/inline-quad.mesh";
  Mesh *mesh = new Mesh(meshFile, 1, 1);
  int dim = mesh->Dimension(); // geometric dimension of the meshed domain
  {
     int ref_levels =
        (int)floor(log(200./mesh->GetNE())/log(2.)/dim);
     for (int l = 0; l < ref_levels; l++)
     {
        mesh->UniformRefinement();
     }
  }

  FiniteElementCollection *fec = new H1_FECollection(FEorder, dim);
  FiniteElementSpace      *Vh  = new FiniteElementSpace(mesh, fec);
  ObstacleProblem problem(Vh);
  
  int dimD = problem.GetDimD();
  Array<int> offsets(3);
  offsets[0] = 0;
  offsets[1] = dimD;
  offsets[2] = dimD;
  offsets.PartialSum();


  InteriorPointSolver optimizer(&problem); 
  BlockVector x0(offsets); x0 = 100.0;
  BlockVector xf(offsets); xf = 0.0;
  optimizer.SetTol(1.e-6);
  optimizer.SetLinearSolver(linSolver);
  optimizer.SetMaxIter(maxIPMiters);
  optimizer.Mult(x0, xf);

  GridFunction d_gf(Vh);
  GridFunction s_gf(Vh);

  d_gf = xf.GetBlock(0);
  s_gf = xf.GetBlock(1);

  FunctionCoefficient dm_fc(dmanufacturedFun); // manufactured solution
  GridFunction dm_gf(Vh);
  dm_gf.ProjectCoefficient(dm_fc);

  ParaViewDataCollection paraview_dc("BarrierProblemSolution", mesh);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(FEorder);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetCycle(0);
  paraview_dc.SetTime(0.0);
  paraview_dc.RegisterField("d", &d_gf);
  paraview_dc.RegisterField("dm", &dm_gf);
  paraview_dc.RegisterField("s", &s_gf);
  paraview_dc.Save();




  delete Vh;
  delete fec;
  delete mesh;
  MPI::Finalize();
  return 0;
}


double dmanufacturedFun(const Vector &x)
{
  return cos(2*M_PI*x(0)) + 0.2 - 2.0*(pow(x(0),3) - 1.5*pow(x(0),2));
}

