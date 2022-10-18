//                                MFEM Example 
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "stokes_solver.hpp"

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f );
void trac_Left(const Vector & x, Vector & tN );
void trac_Right(const Vector & x, Vector & tN );

double pi = 3.141592653589793e0;

int main(int argc, char *argv[])
{
  StopWatch chrono;
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();
  bool verbose = (myid == 0);


   // 1. Parse command line options
  //  const char *mesh_file = "./mesh_1.exo";
  //  const char *mesh_file = "./square01_tri.mesh";
  const char *mesh_file = "./square01_quad.mesh";

  int velocityOrder = 2;
   int pressureOrder = 1;
   int ser_ref_levels = 0;
   const char *device_config = "cpu";
   double viscosityCoefficient = 0.0;
   bool visualization = true;
     
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&velocityOrder, "-vo", "--velocityOrder", "Finite element velocity polynomial degree");
   args.AddOption(&pressureOrder, "-po", "--pressureOrder", "Finite element pressure polynomial degree");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&viscosityCoefficient, "-viscCoef", "--viscosityCoefficient",
                  "Value of viscosity.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
		  "--no-visualization",
                  "Enable or disable GLVis visualization.");
  
   args.ParseCheck();
   Device device(device_config);

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh *mesh;
   mesh = new Mesh(mesh_file, true, true);
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); }

   int dim = mesh->Dimension();

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

   delete mesh;

   // 5. Define the coefficients, analytical solution, and rhs of the PDE.
   VectorFunctionCoefficient fcoeff(dim, fFun);
   VectorFunctionCoefficient trac_LeftCoeff(dim, trac_Left);
   VectorFunctionCoefficient trac_RightCoeff(dim, trac_Right);
   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   mfem::StokesSolver* ssolv=new mfem::StokesSolver(pmesh,velocityOrder,pressureOrder,visualization);
   ssolv->AddMaterial(viscosityCoefficient);
   ssolv->SetVolForce(fcoeff);
   ssolv->AddSurfLoad(2,trac_LeftCoeff);
   ssolv->AddSurfLoad(4,trac_RightCoeff);
   ssolv->AddVelocityBC(1,ucoeff);
   ssolv->AddVelocityBC(3,ucoeff);
   ssolv->SetExactVelocitySolution(ucoeff);
   ssolv->SetExactPressureSolution(pcoeff);
   ssolv->SetNewtonSolver(1.0e-10,0.0,10000000,1);
   ssolv->FSolve();
   ssolv->ComputeL2Errors();
   ssolv->VisualizeFields();
   
   //  delete MinvBt;
   delete ssolv;
   
   return 0;
}


void fFun(const Vector & x, Vector & f )
{
  f(0) = -(2*pi*cos(2*pi*x(1))*sin(2*pi*x(0)) - 16*pi*pi*pi*sin(2*pi*x(0))*sin(2*pi*x(1)));
  f(1) = -(2*pi*cos(2*pi*x(0))*sin(2*pi*x(1)) - 16*pi*pi*pi*cos(2*pi*x(0))*cos(2*pi*x(1)));
}

void uFun_ex(const Vector & x, Vector & u)
{
  u(0) = 2*pi*sin(2*pi*x(1))*sin(2*pi*x(0));
  u(1) = 2*pi*cos(2*pi*x(0))*cos(2*pi*x(1));
}

double pFun_ex(const Vector & x){
  return cos(2*pi*x(0))*cos(2*pi*x(1)) - 1;
}

void trac_Left(const Vector & x, Vector & tN )
{
  tN(0) = -(8*pi*pi*cos(2*pi*x(0))*sin(2*pi*x(1)) - cos(2*pi*x(0))*cos(2*pi*x(1)) + 1);
  tN(1) = 0.0;
}

void trac_Right(const Vector & x, Vector & tN )
{
  tN(0) = (8*pi*pi*cos(2*pi*x(0))*sin(2*pi*x(1)) - cos(2*pi*x(0))*cos(2*pi*x(1)) + 1);
  tN(1) = 0.0;
}
