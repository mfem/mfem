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
#include "nitsche_solver.hpp"
#include "shifted_solver.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "p_divW.hpp"
#include "../common/mfem-common.hpp"
#include "AnalyticalSurface.hpp"

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
  const char *mesh_file = "./square01_tri.mesh";
  // const char *mesh_file = "./square01_quad.mesh";
  //  const char *mesh_file = "./OneElement_tri.mesh";
 
   int velocityOrder = 2;
   int pressureOrder = 1;
   int ser_ref_levels = 0;
   double penaltyParameter = 1.0;
   const char *device_config = "cpu";
   bool useEmbedded = false;
   int geometricShape = 0;
  
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&velocityOrder, "-vo", "--velocityOrder", "Finite element velocity polynomial degree");
   args.AddOption(&pressureOrder, "-po", "--pressureOrder", "Finite element pressure polynomial degree");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&penaltyParameter, "-penPar", "--penaltyParameter",
                  "Value of penalty parameter.");
   args.AddOption(&useEmbedded, "-emb", "--use-embedded", "-no-emb",
                  "--no-embedded",
                  "Use Embedded when there is surface that will be embedded in a pre-existing mesh");
   args.AddOption(&geometricShape, "-gS", "--geometricShape",
                  "Shape of the embedded geometry that will be embedded");
  
   args.ParseCheck();
   Device device(device_config);

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh *mesh;
   mesh = new Mesh(mesh_file, true, true);
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); }

   int dim = mesh->Dimension();

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

   delete mesh;

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection V_fec(velocityOrder, pmesh->Dimension());
   H1_FECollection P_fec(pressureOrder, pmesh->Dimension());

   ParFiniteElementSpace V_H1FESpace(pmesh, &V_fec, pmesh->Dimension());
   ParFiniteElementSpace P_H1FESpace(pmesh, &P_fec);
   V_H1FESpace.ExchangeFaceNbrData();
   P_H1FESpace.ExchangeFaceNbrData();
 
   // Weak Boundary condition imposition: all tests use v.n = 0 on the boundary
   // We need to define ess_tdofs and ess_vdofs, but they will be kept empty
   AnalyticalSurface *analyticalSurface = NULL;
   Array<int> ess_tdofs, ess_vdofs;
   if (useEmbedded){
     analyticalSurface = new AnalyticalSurface(geometricShape, V_H1FESpace);
     analyticalSurface->ResetData();
     analyticalSurface->SetupElementStatus();
 
     Array<int> ess_inactive_dofs = analyticalSurface->GetEss_Vdofs();
     V_H1FESpace.GetRestrictionMatrix()->BooleanMult(ess_inactive_dofs, ess_tdofs);
     V_H1FESpace.MarkerToList(ess_tdofs, ess_vdofs);
     //  ess_vdofs.Print(std::cout,1);
     //     std::cout << " ess size " << ess_vdofs.Size() << std::endl;
   }
 
   const int max_elem_attr = pmesh->attributes.Max();
   Array<int> ess_elem(max_elem_attr);
   ess_elem = 1;
   if (useEmbedded){
     ess_elem[max_elem_attr-1] = 0;
   }
   //  ess_elem.Print();
   // 4. Create "marker arrays" to define the portions of boundary associated
   //    with each type of boundary condition. These arrays have an entry
   //    corresponding to each boundary attribute.  Placing a '1' in entry i
   //    marks attribute i+1 as being active, '0' is inactive.
   Array<int> dbc_bdr_top_list, dbc_bdr_top(pmesh->bdr_attributes.Max());
   Array<int> dbc_bdr_bottom_list, dbc_bdr_bottom(pmesh->bdr_attributes.Max());
   Array<int> nbc_bdr_left_list, nbc_bdr_left(pmesh->bdr_attributes.Max());
   Array<int> nbc_bdr_right_list, nbc_bdr_right(pmesh->bdr_attributes.Max());
   Array<int> dbc_bdr_dir_list, dbc_bdr_dir(pmesh->bdr_attributes.Max());
   dbc_bdr_top = 0; dbc_bdr_top[0] = 1;
   nbc_bdr_left = 0; nbc_bdr_left[1] = 1;
   dbc_bdr_bottom = 0; dbc_bdr_bottom[2] = 1;
   nbc_bdr_right = 0; nbc_bdr_right[3] = 1;
   dbc_bdr_dir = 0;  dbc_bdr_dir[0] = 1; dbc_bdr_dir[2] = 1;


   V_H1FESpace.GetEssentialTrueDofs(dbc_bdr_top, dbc_bdr_top_list);
   V_H1FESpace.GetEssentialTrueDofs(nbc_bdr_left, nbc_bdr_left_list);
   V_H1FESpace.GetEssentialTrueDofs(dbc_bdr_bottom, dbc_bdr_bottom_list);
   V_H1FESpace.GetEssentialTrueDofs(nbc_bdr_right, nbc_bdr_right_list);   
   V_H1FESpace.GetEssentialTrueDofs(dbc_bdr_dir, dbc_bdr_dir_list);

   // 5. Define the coefficients, analytical solution, and rhs of the PDE.
   VectorFunctionCoefficient fcoeff(dim, fFun);
   VectorFunctionCoefficient trac_LeftCoeff(dim, trac_Left);
   VectorFunctionCoefficient trac_RightCoeff(dim, trac_Right);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   ShiftedVectorFunctionCoefficient ucoeff_shifted(dim, uFun_ex);
   
   FunctionCoefficient pcoeff(pFun_ex);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = V_H1FESpace.GetVSize();
   block_offsets[2] = P_H1FESpace.GetVSize();
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(3); // number of variables + 1
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = V_H1FESpace.GetTrueVSize();
   block_trueOffsets[2] = P_H1FESpace.GetTrueVSize();
   block_trueOffsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(P) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(V+P) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   Vector lambda(pmesh->attributes.Max());
   lambda = 0.0;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   mu = 1.0;
   PWConstCoefficient mu_func(mu);

   // 7. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
   BlockVector x(block_offsets), rhs(block_offsets);
   BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
   trueX = 0.0;
   x = 0.0;
   rhs = 0.0;
   trueRhs = 0.0;
   ParLinearForm *fform(new ParLinearForm);
   fform->Update(&V_H1FESpace, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff),ess_elem);
   fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(trac_LeftCoeff),nbc_bdr_left);
   fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(trac_RightCoeff),nbc_bdr_right);
 
   // Nitsche
   fform->AddBdrFaceIntegrator(new StrainNitscheBCForceIntegrator(mu_func,ucoeff),dbc_bdr_dir);
   // Penalty
   fform->AddBdrFaceIntegrator(new VelocityBCPenaltyIntegrator(penaltyParameter,mu_func,ucoeff),dbc_bdr_dir);
   if (useEmbedded){
     fform->AddInteriorFaceIntegrator(new ShiftedVelocityBCPenaltyIntegrator(pmesh, penaltyParameter, mu_func, ucoeff_shifted, analyticalSurface));
     // Nitsche
     fform->AddInteriorFaceIntegrator(new ShiftedStrainNitscheBCForceIntegrator(pmesh, mu_func, ucoeff_shifted, analyticalSurface));
   }

   fform->Assemble();
   fform->ParallelAssemble(trueRhs.GetBlock(0));

   ParLinearForm *gform(new ParLinearForm);
   gform->Update(&P_H1FESpace, rhs.GetBlock(1), 0);
   // Nitsche
   gform->AddBdrFaceIntegrator(new PressureNitscheBCForceIntegrator(ucoeff),dbc_bdr_dir);
   if (useEmbedded){
     // Nitsche
     gform->AddInteriorFaceIntegrator(new ShiftedPressureNitscheBCForceIntegrator(pmesh, ucoeff_shifted, analyticalSurface));    
   }
   gform->Assemble();
   gform->ParallelAssemble(trueRhs.GetBlock(1));
   trueRhs.GetBlock(1).SyncAliasMemory(trueRhs);
   
   ParBilinearForm *mVarf(new ParBilinearForm(&V_H1FESpace));
   mVarf->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func),ess_elem);
   // Nitsche
   mVarf->AddBdrFaceIntegrator(new StrainBoundaryForceIntegrator(mu_func),dbc_bdr_dir);
   // IP
   mVarf->AddBdrFaceIntegrator(new StrainBoundaryForceTransposeIntegrator(mu_func),dbc_bdr_dir);
   // Penalty
   mVarf->AddBdrFaceIntegrator(new VelocityPenaltyIntegrator(penaltyParameter,mu_func),dbc_bdr_dir);

   if (useEmbedded){
     // Nitsche
     mVarf->AddInteriorFaceIntegrator(new ShiftedStrainBoundaryForceIntegrator(pmesh, mu_func, analyticalSurface));
     // IP
     mVarf->AddInteriorFaceIntegrator(new ShiftedStrainBoundaryForceTransposeIntegrator(pmesh, mu_func, analyticalSurface));
     // Penalty
     mVarf->AddInteriorFaceIntegrator(new ShiftedVelocityPenaltyIntegrator(pmesh, penaltyParameter, mu_func, analyticalSurface));
   }

   mVarf->Assemble();
   mVarf->Finalize();
   
   ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(&V_H1FESpace, &P_H1FESpace));
   bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(),ess_elem);
   // Nitsche
   bVarf->AddBdrFaceIntegrator(new PressureBoundaryForceIntegrator(),dbc_bdr_dir);
   if (useEmbedded){
     // Nitsche
     bVarf->AddInteriorFaceIntegrator(new ShiftedPressureBoundaryForceIntegrator(pmesh, analyticalSurface));
   }
   
   bVarf->Assemble();
   bVarf->Finalize();

   ParMixedBilinearForm *btVarf(new ParMixedBilinearForm(&P_H1FESpace, &V_H1FESpace));
   btVarf->AddDomainIntegrator(new PDivWForceIntegrator(),ess_elem);
   // IP
   btVarf->AddBdrFaceIntegrator(new PressureBoundaryForceTransposeIntegrator(),dbc_bdr_dir);
   if (useEmbedded){
      // IP
     btVarf->AddInteriorFaceIntegrator(new ShiftedPressureBoundaryForceTransposeIntegrator(pmesh, analyticalSurface));
   }
   
   btVarf->Assemble();
   btVarf->Finalize();

   BlockOperator stokesOp(block_trueOffsets);

   HypreParMatrix *M = NULL;
   HypreParMatrix *B = NULL;
   HypreParMatrix *Bt = NULL;

   M = mVarf->ParallelAssemble();
   B = bVarf->ParallelAssemble();
   Bt = btVarf->ParallelAssemble();
   
   stokesOp.SetBlock(0,0, M);
   stokesOp.SetBlock(0,1, Bt);
   stokesOp.SetBlock(1,0, B);

   ConstrainedOperator A(&stokesOp,ess_vdofs);
   A.EliminateRHS(trueX,trueRhs);
   // 11. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.
   int maxIter(100000000);
   double rtol(1.e-14);
   double atol(1.e-14);

   chrono.Clear();
   chrono.Start();
   GMRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   //  solver.SetOperator(stokesOp);
   solver.SetOperator(A);
   solver.SetPrintLevel(1);
   trueX = 0.0;
   solver.Mult(trueRhs, trueX);
 
   chrono.Stop();

   // 8. Create the grid functions u and p and enforce BC on u 
   ParGridFunction *u(new ParGridFunction);
   ParGridFunction *p(new ParGridFunction);
   u->MakeRef(&V_H1FESpace, x.GetBlock(0), 0);
   p->MakeRef(&P_H1FESpace, x.GetBlock(1), 0);
   u->Distribute(&(trueX.GetBlock(0)));
   p->Distribute(&(trueX.GetBlock(1)));

   // 13. Compute the L2 error norms.
   int order_quad = max(2, 10*velocityOrder+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u = 0.0;
   double err_p = 0.0;
   if (useEmbedded){
     Array<int> elem_marker(pmesh->GetNE());
     elem_marker = 0;
     Array<int> &elemStatus = analyticalSurface->GetElement_Status();
   for (int e = 0; e < V_H1FESpace.GetNE(); e++)
     {
       if (elemStatus[e] == AnalyticalGeometricShape::SBElementType::INSIDE){
	 elem_marker[e] = 1;
       }
     }
   err_u  = u->ComputeL2Error(ucoeff, irs, &elem_marker);
   err_p  = p->ComputeL2Error(pcoeff, irs, &elem_marker);
   }
   else{
     err_u  = u->ComputeL2Error(ucoeff, irs);
     err_p  = p->ComputeL2Error(pcoeff, irs);
   }
 

   if (myid == 0)
   {
      std::cout << "|| u_h - u_ex || = " << err_u << "\n";
      std::cout << "|| p_h - p_ex || = " << err_p << "\n";
   }

   int size = 500;
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock_u;
   common::VisualizeField(sol_sock_u, vishost, visport, *u,
                          "Velocity", 0, 0, size, size);
   socketstream sol_sock_p;
   common::VisualizeField(sol_sock_p, vishost, visport, *p,
                          "Pressure", size, 0, size, size);

     // 14. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example5_mesh3", pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(velocityOrder);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",u);
   paraview_dc.RegisterField("pressure",p);
   paraview_dc.Save();

   // 15. Free the used memory.
   delete fform;
   delete mVarf;
   delete bVarf;
   delete pmesh;
   
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
  tN(0) = -(8*pi*pi*cos(2*pi*x[0])*sin(2*pi*x[1]) - cos(2*pi*x[0])*cos(2*pi*x[1]) + 1);
  tN(1) = 0.0;
}

void trac_Right(const Vector & x, Vector & tN )
{
  tN(0) = (8*pi*pi*cos(2*pi*x(0))*sin(2*pi*x(1)) - cos(2*pi*x(0))*cos(2*pi*x(1)) + 1);
  tN(1) = 0.0;
}
