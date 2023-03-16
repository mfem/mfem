//                                MFEM Example 
//
// Compile with: make ex0
//
// Sample runs: mpirun -np 1 ./ex_Stokes_shifted -m ./square01_quad.mesh -fP -supgPar 0.0 -penPar 10.0 -emb -gS 1 -viscCoef 1.0 -rs 2 -tO 3 -vo 3 -po 2 -gPenCoef 1.0 -nST 3 -nPT 2 -mumps
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include "nitsche_weighted_solver.hpp"
#include "ghost_penalty.hpp"
#include "shifted_weighted_solver.hpp"
#include "volume_fractions.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "volume_integrals.hpp"
#include "AnalyticalSurface.hpp"
#include "../common/mfem-common.hpp"

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void uFun_ex(const Vector & x, Vector & u);
void uFun_exShifted(const Vector & x, Vector & u);
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
  //  const char *mesh_file = "./square01_quad.mesh";
  //  const char *mesh_file = "./OneElement_tri.mesh";
 
   int velocityOrder = 2;
   int pressureOrder = 1;
   int ser_ref_levels = 0;
   double penaltyParameter = 1.0;
   const char *device_config = "cpu";
   bool useEmbedded = false;
   int geometricShape = 0;
   int nTerms = 1;
   bool visualization = true;
   bool mumps_solver = false;
   double viscosityCoefficient = 0.0;
   double SUPGParameter = 1.0;
   bool fullPenalty = false;
   int numberStrainTerms = 1;
   int numberPressureTerms = 1;
   double ghostPenaltyCoefficient = 1.0;
   
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
   args.AddOption(&SUPGParameter, "-supgPar", "--supgParameter",
		  "Value of SUPG parameter.");
   args.AddOption(&geometricShape, "-gS", "--geometricShape",
                  "Shape of the embedded geometry that will be embedded");
   args.AddOption(&nTerms, "-tO", "--taylorOrder",
                  "Number of terms in the Taylor expansion");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
		  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&fullPenalty, "-fP", "--full-Penalty", "-first-order-penalty",
                  "--first-order-penalty",
                  "Use full or first order for SBM penalty.");
   args.AddOption(&viscosityCoefficient, "-viscCoef", "--viscosityCoefficient",
                  "Value of viscosity.");
   args.AddOption(&numberStrainTerms, "-nST", "--numberStrainTerms",
                  "Number of terms in the strain ghost penalty operator.");
   args.AddOption(&numberPressureTerms, "-nPT", "--numberPressureTerms",
                  "Number of terms in the pressure ghost penalty operator.");
   args.AddOption(&ghostPenaltyCoefficient, "-gPenCoef", "--ghost-penalty-coefficient",
		  "Ghost penalty scaling.");
   
#ifdef MFEM_USE_MUMPS
   args.AddOption(&mumps_solver, "-mumps", "--mumps-solver", "-no-mumps",
                  "--no-mumps-solver", "Use the MUMPS Solver.");
#endif

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

   ParFiniteElementSpace V_H1FESpace(pmesh, &V_fec, pmesh->Dimension(),Ordering::byVDIM);
   ParFiniteElementSpace P_H1FESpace(pmesh, &P_fec);
   V_H1FESpace.ExchangeFaceNbrData();
   P_H1FESpace.ExchangeFaceNbrData();
 
   // Weak Boundary condition imposition: all tests use v.n = 0 on the boundary
   // We need to define ess_tdofs and ess_vdofs, but they will be kept empty
   AnalyticalSurface *analyticalSurface = NULL;
   Array<int> ess_tdofs, ess_vdofs;
   Array<int> ess_tdofs_p, ess_vdofs_p, ess_vdofs_total;
   if (useEmbedded){
     analyticalSurface = new AnalyticalSurface(geometricShape, V_H1FESpace, P_H1FESpace, 1);
     analyticalSurface->ResetData();
     analyticalSurface->SetupElementStatus();
 
     Array<int> ess_inactive_dofs = analyticalSurface->GetEss_Vdofs();
     V_H1FESpace.GetRestrictionMatrix()->BooleanMult(ess_inactive_dofs, ess_tdofs);
     V_H1FESpace.MarkerToList(ess_tdofs, ess_vdofs);
     Array<int> ess_inactive_dofs_p = analyticalSurface->GetEss_Vdofs_P();
     P_H1FESpace.GetRestrictionMatrix()->BooleanMult(ess_inactive_dofs_p, ess_tdofs_p);
     P_H1FESpace.MarkerToList(ess_tdofs_p, ess_vdofs_p);
     //  ess_vdofs_p.Print(std::cout,1);
     //  std::cout << " end " << std::endl;
     //     std::cout << " ess size " << ess_vdofs.Size() << std::endl;
   }
 
   const int max_elem_attr = pmesh->attributes.Max();
   Array<int> ess_elem(max_elem_attr);
   ess_elem = 1;
   if (useEmbedded && (max_elem_attr >= 2)){
     ess_elem[max_elem_attr-1] = 0;
   }
   //  ess_elem.Print();

   double Volume, vol = 0.0;
   int NE = pmesh->GetNE();
   int Ne, ne = NE;
  
   for (int e = 0; e < NE; e++) { vol += pmesh->GetElementVolume(e); }

   MPI_Allreduce(&vol, &Volume, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&ne, &Ne, 1, MPI_INT, MPI_SUM, pmesh->GetComm());
   double hinscribed = 0.0;
   double hcircumscribed = 0.0;
   switch (pmesh->GetElementBaseGeometry(0))
     {
       //     case Geometry::SEGMENT: h0 = Volume / Ne; break;
     case Geometry::TRIANGLE:
       hcircumscribed = sqrt(2)*(1.0 / sqrt(Ne/2));
       hinscribed = ((1.0 / sqrt(Ne/2))*2-sqrt(2)*(1.0 / sqrt(Ne/2)));
       break;
     case Geometry::SQUARE:
       hcircumscribed = sqrt(2)*sqrt(Volume/Ne);
       hinscribed = sqrt(Volume/Ne);
       break;
       //  case Geometry::CUBE: h0 = pow(Volume / Ne, 1./3.); break;
       // case Geometry::TETRAHEDRON: h0 = pow(6.0 * Volume / Ne, 1./3.); break;
     default: MFEM_ABORT("Unknown zone type!");
     }
   //  h0 /= (double) V_H1FESpace.GetOrder(0);
   std::cout << " hinsc " << hinscribed << " hcircum " << hcircumscribed << std::endl;
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
   mu = viscosityCoefficient;
   PWConstCoefficient mu_func(mu);

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection alpha_fec(0, pmesh->Dimension());
   ParFiniteElementSpace alpha_fes(pmesh, &alpha_fec);
   ParGridFunction alphaCut(&alpha_fes);
   alphaCut = 1;
   if (useEmbedded){
     UpdateAlpha(*analyticalSurface,alphaCut,P_H1FESpace);
     alphaCut.ExchangeFaceNbrData();
   }

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

   double C_I = 0.0;
   switch (pmesh->GetElementBaseGeometry(0))
     {
     case Geometry::TRIANGLE:
     case Geometry::TETRAHEDRON:{
       //  C_I = (velocityOrder+1)*(velocityOrder+2)/dim;
       C_I = (velocityOrder)*(velocityOrder+1)/dim;
	 break;
     }
     case Geometry::SQUARE: 
     case Geometry::CUBE:{
       C_I = velocityOrder*velocityOrder;
       break;
     }
     default: MFEM_ABORT("Unknown zone type!");
     }

   double C_inv  = 1.0;
   switch (velocityOrder)
     {
     case 1:{
       C_inv = 1.0;
	 break;
     }
     case 2:{
       C_inv = 1.0/36.0;
       break;
     }
     case 3:{
       C_inv = 1.0/155.05;
       break;
     }
     case 4:{
       C_inv = 1.0/397.50;
       break;
     }
     default:
       C_inv = 1.0;
     }
   
   double eta = -0.5*(3*pow(C_inv,-0.5)+1.5-(1/(4*C_I)))+pow(0.25*pow(3*pow(C_inv,-0.5)+1.5-(1/(4*C_I)),2)+(pow(C_inv,-0.5)+1)*(3+(1/(2*C_I))),0.5);
   // if (fullPenalty){
   penaltyParameter *= 2*C_I*2*viscosityCoefficient;
     /*   }
   else{
     penaltyParameter *= 2*C_I*eta*2*viscosityCoefficient;
   }*/
   //   penaltyParameter *= 2*C_I*2*viscosityCoefficient;
   std::cout << " pen; " << penaltyParameter << std::endl;
   
   double penaltyParameter_bf = 10.0*2*C_I*2*viscosityCoefficient;

   ConstantCoefficient TauU(SUPGParameter*hcircumscribed*hinscribed*C_inv/(2*viscosityCoefficient));
   //  ConstantCoefficient TauU(SUPGParameter*h0*h0/(2*viscosityCoefficient));
   std::cout << " gammaSUPG " << SUPGParameter*hcircumscribed*hinscribed*C_inv/(2*viscosityCoefficient) << std::endl;
 
   ParLinearForm *fform(new ParLinearForm);
   fform->Update(&V_H1FESpace, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new WeightedVectorForceIntegrator(alphaCut, fcoeff),ess_elem);
   fform->AddBdrFaceIntegrator(new WeightedTractionBCIntegrator(alphaCut, trac_LeftCoeff),nbc_bdr_left);
   fform->AddBdrFaceIntegrator(new WeightedTractionBCIntegrator(alphaCut, trac_RightCoeff),nbc_bdr_right);
   
   // Nitsche
   fform->AddBdrFaceIntegrator(new WeightedStrainNitscheBCForceIntegrator(alphaCut, mu_func, ucoeff),dbc_bdr_dir);
   // Penalty
   fform->AddBdrFaceIntegrator(new WeightedVelocityBCPenaltyIntegrator(alphaCut, penaltyParameter_bf, mu_func, ucoeff),dbc_bdr_dir);
   if (useEmbedded){
     // Penalty
     fform->AddInteriorFaceIntegrator(new WeightedShiftedVelocityBCPenaltyIntegrator(pmesh, alphaCut, penaltyParameter, mu_func, ucoeff_shifted, analyticalSurface, nTerms, 1, fullPenalty));
     // Nitsche
     fform->AddInteriorFaceIntegrator(new WeightedShiftedStrainNitscheBCForceIntegrator(pmesh, alphaCut, mu_func, ucoeff_shifted, analyticalSurface, 1));
   }
   fform->Assemble();
   fform->ParallelAssemble(trueRhs.GetBlock(0));

   ParLinearForm *gform(new ParLinearForm);
   gform->Update(&P_H1FESpace, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new StabilizedDomainLFGradIntegrator(fcoeff,TauU, alphaCut),ess_elem);
   // Nitsche
   gform->AddBdrFaceIntegrator(new WeightedPressureNitscheBCForceIntegrator(alphaCut, ucoeff),dbc_bdr_dir);
   if (useEmbedded){
     // Nitsche
     gform->AddInteriorFaceIntegrator(new WeightedShiftedPressureNitscheBCForceIntegrator(pmesh, alphaCut, ucoeff_shifted, analyticalSurface, 1));    
   }
  
   gform->Assemble();
   gform->ParallelAssemble(trueRhs.GetBlock(1));
   trueRhs.GetBlock(1).SyncAliasMemory(trueRhs);
  
   ParBilinearForm *mVarf(new ParBilinearForm(&V_H1FESpace));
   mVarf->AddDomainIntegrator(new WeightedStrainStrainForceIntegrator(alphaCut, mu_func),ess_elem);
   // Nitsche
   mVarf->AddBdrFaceIntegrator(new WeightedStrainBoundaryForceIntegrator(alphaCut, mu_func),dbc_bdr_dir);
   // IP
   mVarf->AddBdrFaceIntegrator(new WeightedStrainBoundaryForceTransposeIntegrator(alphaCut, mu_func),dbc_bdr_dir);
   // Penalty
   mVarf->AddBdrFaceIntegrator(new WeightedVelocityPenaltyIntegrator(alphaCut, penaltyParameter_bf, mu_func),dbc_bdr_dir);
   if (useEmbedded){
     // Nitsche
     mVarf->AddInteriorFaceIntegrator(new WeightedShiftedStrainBoundaryForceIntegrator(pmesh, alphaCut, mu_func, analyticalSurface, nTerms, 1));
     // IP
     mVarf->AddInteriorFaceIntegrator(new WeightedShiftedStrainBoundaryForceTransposeIntegrator(pmesh, alphaCut, mu_func, analyticalSurface, 1));
     // Penalty
     mVarf->AddInteriorFaceIntegrator(new WeightedShiftedVelocityPenaltyIntegrator(pmesh, alphaCut, penaltyParameter, mu_func, analyticalSurface, nTerms, 1, fullPenalty));

     mVarf->AddInteriorFaceIntegrator(new GhostStrainPenaltyIntegrator(pmesh, mu_func, alphaCut, ghostPenaltyCoefficient, analyticalSurface, 1));
     for (int i = 2; i <= numberStrainTerms; i++){
       // best to use 1.0 / i!
       double factorial = 1.0;	
       for (int s = 1; s <= i; s++){
	 factorial = factorial*s;
       }
       // ghost penalty
       mVarf->AddInteriorFaceIntegrator(new GhostStrainFullGradPenaltyIntegrator(pmesh, mu_func, alphaCut, ghostPenaltyCoefficient/factorial, analyticalSurface, i));
     }
   }
   mVarf->Assemble();
   mVarf->Finalize();
   
   ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(&V_H1FESpace, &P_H1FESpace));
   bVarf->AddDomainIntegrator(new WeightedQDivUForceIntegrator(alphaCut),ess_elem);
   // Nitsche
   bVarf->AddBdrFaceIntegrator(new WeightedPressureBoundaryForceIntegrator(alphaCut),dbc_bdr_dir);
   // STAB
   bVarf->AddDomainIntegrator(new DivergenceStrainStabilizedIntegrator(TauU, alphaCut, mu_func),ess_elem);
   if (useEmbedded){
     // Nitsche
     bVarf->AddInteriorFaceIntegrator(new WeightedShiftedPressureBoundaryForceIntegrator(pmesh, alphaCut, analyticalSurface, nTerms, 1));
   }
   bVarf->Assemble();
   bVarf->Finalize();
   
   ParMixedBilinearForm *btVarf(new ParMixedBilinearForm(&P_H1FESpace, &V_H1FESpace));
   btVarf->AddDomainIntegrator(new WeightedPDivWForceIntegrator(alphaCut),ess_elem);
   // IP
   btVarf->AddBdrFaceIntegrator(new WeightedPressureBoundaryForceTransposeIntegrator(alphaCut),dbc_bdr_dir);
   if (useEmbedded){
      // IP
     btVarf->AddInteriorFaceIntegrator(new WeightedShiftedPressureBoundaryForceTransposeIntegrator(pmesh, alphaCut, analyticalSurface, 1));
   }
   btVarf->Assemble();
   btVarf->Finalize();

   ParBilinearForm *pVarf(new ParBilinearForm(&P_H1FESpace));
   pVarf->AddDomainIntegrator(new WeightedDiffusionIntegrator(TauU, alphaCut),ess_elem);
   for (int i = 1; i <= numberPressureTerms; i++){
     // best to use 1.0 / i!
     double factorial = 1.0;	
     for (int s = 1; s <= i; s++){
       factorial = factorial*s;
     }
     // ghost penalty
     pVarf->AddInteriorFaceIntegrator(new GhostPenaltyFullGradIntegrator(pmesh, mu_func, alphaCut, ghostPenaltyCoefficient/factorial, analyticalSurface, i));
   }
   pVarf->Assemble();
   pVarf->Finalize();
     
   HypreParMatrix *M = NULL;
   HypreParMatrix *B = NULL;
   HypreParMatrix *Bt = NULL;
   HypreParMatrix *P = NULL;

   M = mVarf->ParallelAssemble();
   B = bVarf->ParallelAssemble();
   Bt = btVarf->ParallelAssemble();
   P = pVarf->ParallelAssemble();
   if (mumps_solver){
     Array2D< HypreParMatrix * > bm(2,2);
     bm(0,0) = M; bm(0,1) = Bt;
     bm(1,0) = B; bm(1,1) = P;
     HypreParMatrix* MM = mfem::HypreParMatrixFromBlocks(bm);  
     MM->EliminateZeroRows();
     
     MUMPSSolver mumps;
     mumps.SetPrintLevel(1);
     mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
     mumps.SetOperator(*MM);
     trueX = 0.0;
     mumps.Mult(trueRhs, trueX);
   }
   else{  
     BlockOperator stokesOp(block_trueOffsets);
     
     stokesOp.SetBlock(0,0, M);
     stokesOp.SetBlock(0,1, Bt);
     stokesOp.SetBlock(1,0, B);
     stokesOp.SetBlock(1,1, P);
       
     ConstrainedOperator A(&stokesOp,ess_vdofs);
     A.EliminateRHS(trueX,trueRhs);
     
     PowerMethod powerMethod(MPI_COMM_WORLD);
     Vector ev(A.Width());
     double max_eig_estimate = 1.0;
     max_eig_estimate = powerMethod.EstimateLargestEigenvalue(A, ev, 1000, 1e-10);

     // PRECONDITIONER
     ConstantCoefficient scale(std::sqrt(std::abs(max_eig_estimate)));
     ParBilinearForm *pMass = new ParBilinearForm(&P_H1FESpace);
     pMass->AddDomainIntegrator(new MassIntegrator(scale));
     for (int i = 1; i <= pressureOrder; i++){
       // ghost penalty
       pMass->AddInteriorFaceIntegrator(new GhostPenaltyFullGradIntegrator(pmesh, mu_func, alphaCut, penaltyParameter, analyticalSurface, i));
     }

     pMass->Assemble();
     pMass->Finalize();
     
     ParBilinearForm *vMass(new ParBilinearForm(&V_H1FESpace));
     vMass->AddDomainIntegrator(new  WeightedStrainStrainForceIntegrator(alphaCut, mu_func),ess_elem);
     if (useEmbedded){
       vMass->AddInteriorFaceIntegrator(new WeightedShiftedVelocityPenaltyIntegrator(pmesh, alphaCut, penaltyParameter, mu_func, analyticalSurface, nTerms));
       vMass->AddInteriorFaceIntegrator(new GhostStrainPenaltyIntegrator(pmesh, mu_func, alphaCut, penaltyParameter, analyticalSurface, 1));
       for (int i = 2; i <= velocityOrder; i++){
	 // ghost penalty
	 vMass->AddInteriorFaceIntegrator(new GhostStrainFullGradPenaltyIntegrator(pmesh, mu_func, alphaCut, penaltyParameter, analyticalSurface, i));
       }
     }
     vMass->AddBdrFaceIntegrator(new WeightedVelocityPenaltyIntegrator(alphaCut, penaltyParameter,mu_func),dbc_bdr_dir);
     vMass->Assemble();
     vMass->Finalize();
     
     HypreParMatrix *velocity_Mass = NULL;
     HypreParMatrix *pressure_Mass = NULL;
     velocity_Mass = vMass->ParallelAssemble();
     pressure_Mass = pMass->ParallelAssemble();
     
     HypreParMatrix *VMe = NULL;
     HypreParMatrix *PMe = NULL;
     VMe = velocity_Mass->EliminateRowsCols(ess_vdofs);
     mfem::HypreBoomerAMG* Vamg = new HypreBoomerAMG(*velocity_Mass);
     Vamg->SetSystemsOptions(dim);
     Vamg->SetElasticityOptions(&V_H1FESpace);
     mfem::HypreBoomerAMG* Pamg = new HypreBoomerAMG(*pressure_Mass);
     BlockDiagonalPreconditioner *stokesPr = new BlockDiagonalPreconditioner(block_trueOffsets);
     stokesPr->SetDiagonalBlock(0,Vamg);
     stokesPr->SetDiagonalBlock(1,Pamg);
     ///
     
     // 11. Solve the linear system with MINRES.
     //     Check the norm of the unpreconditioned residual.
     int maxIter(100000000);
     double rtol(1.e-13);
     double atol(0.0);
     
     chrono.Clear();
     chrono.Start();
     GMRESSolver solver(MPI_COMM_WORLD);
     solver.SetAbsTol(atol);
     solver.SetRelTol(rtol);
     solver.SetMaxIter(maxIter);
     solver.SetOperator(A);
     //    solver.SetPreconditioner(*stokesPr);
     solver.SetPrintLevel(1);
     trueX = 0.0;
     solver.Mult(trueRhs, trueX);
     chrono.Stop();
   }
   
   // 8. Create the grid functions u and p and enforce BC on u 
   ParGridFunction *u(new ParGridFunction);
   ParGridFunction *p(new ParGridFunction);
   u->MakeRef(&V_H1FESpace, x.GetBlock(0), 0);
   p->MakeRef(&P_H1FESpace, x.GetBlock(1), 0);
   u->Distribute(&(trueX.GetBlock(0)));
   p->Distribute(&(trueX.GetBlock(1)));

   // 13. Compute the L2 error norms.
   int order_quad = max(5, 10*velocityOrder+1);
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
	 if ( (elemStatus[e] == AnalyticalGeometricShape::SBElementType::INSIDE) || (elemStatus[e] == AnalyticalGeometricShape::SBElementType::CUT)) {
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

   if (visualization){
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
     ParaViewDataCollection paraview_dc("TH_Example5_tri_mesh_"+std::string(mesh_file)+std::string("_velocityOrder_")+std::to_string(velocityOrder)+std::string("_meshSize_")+std::to_string(hinscribed)+std::string("_supg_")+std::to_string(SUPGParameter)+std::string("_fP_")+std::to_string(fullPenalty), pmesh);

     paraview_dc.SetPrefixPath("ParaView");
     paraview_dc.SetLevelsOfDetail(velocityOrder);
     paraview_dc.SetCycle(0);
     paraview_dc.SetDataFormat(VTKFormat::BINARY);
     paraview_dc.SetHighOrderOutput(true);
     paraview_dc.SetTime(0.0); // set the time
     paraview_dc.RegisterField("velocity",u);
     paraview_dc.RegisterField("pressure",p);
     paraview_dc.Save();
   }

   // 15. Free the used memory.
   delete fform;
   delete gform;
   delete mVarf;
   delete bVarf;
   delete btVarf;
   delete pmesh;
   if (useEmbedded){
     delete analyticalSurface;
   }
   return 0;
}


void fFun(const Vector & x, Vector & f )
{
  f(0) = -(2*pi*cos(2*pi*x(1))*sin(2*pi*x(0)) - 16*pi*pi*pi*sin(2*pi*x(0))*sin(2*pi*x(1)));
  f(1) = -(2*pi*cos(2*pi*x(0))*sin(2*pi*x(1)) - 16*pi*pi*pi*cos(2*pi*x(0))*cos(2*pi*x(1)));
  // f(0) = 0.0;
  //  f(1) = 0.0;
}

void uFun_ex(const Vector & x, Vector & u)
{
  u(0) = 2*pi*sin(2*pi*x(1))*sin(2*pi*x(0));
  u(1) = 2*pi*cos(2*pi*x(0))*cos(2*pi*x(1));
}

void uFun_exShifted(const Vector & x, Vector & u)
{
  u(0) = 0.0;
  u(1) = 0.0;
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
