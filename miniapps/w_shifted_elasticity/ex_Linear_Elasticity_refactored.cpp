// Sample run:  
// mpirun -np 1 ./ex_Linear_Elasticity_WSBM -m ./square01_tri.mesh -poissonsRatio 0.3 -youngsModulus 200 -emb -gS 1 -rs 2 -tO 2 -do 2 -gPenCoef 1.0 -nST 2 -mumps -level-set
//              -poissonsRatio is the flag for setting Poisson's ratio.
//              -youngsModulus is the flag for setting Young's modulus. 
//              -emb is the flag, when added, for activating the embedded data structure of the code.
//              -gS is the flag for setting the type of geometry for the level set to use.
//              -do is the flag for setting the polynomial order for the displacement.
//              -gPenCoef is the flag for setting the penalty parameter for the strain ghost penalty.
//              -nST is the flag for setting the number of terms to add in the strain ghost penalty.
//              -mumps is the flag, when added, for using a mumps solver.
//              -level-set is the flag, when added, for using a level set for computing the distance and normal vectors otherwise an analytical shape is used instead.    
//
// Description: This code implements WSBM for the linear elasticity operator. This Neumann conditions are embedded
//              while any Dirichlet condition has to be body-fitted. So far we assume one level set to define
//              the geometry to be embedded, but the code can be easily modifed to handle multiple level sets.
//              The body-fitted Dirichlet conditions are enforced weakly with a fixed penalty parameter equal to
//              40.0*C_I, where C_I is the trace inequality constant. Code also works for pure body-fitted calculation.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "linear_elasticity_solver.hpp"
#include "sbm_aux.hpp"

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void uFun_ex(const Vector & x, Vector & u);
void fFun(const Vector & x, Vector & f );
void traction_ex(const Vector & x, DenseMatrix & tN );

void uFun_ex3D(const Vector & x, Vector & u);
void fFun3D(const Vector & x, Vector & f );
void traction_ex3D(const Vector & x, DenseMatrix & tN );

double pi = 3.141592653589793e0;

int main(int argc, char *argv[])
{
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();
     
   // 1. Parse command line options
  const char *mesh_file = "./square01_quad.mesh";

  int displacementOrder = 1;
  int ser_ref_levels = 0;
  double poissonsRatio = 0.3;
  double youngsModulus = 200.0; // in MPA
  bool visualization = false;
  bool useEmbedded = false;
  int geometricShape = 0;
  int nTerms = 1; 
  bool mumps_solver = false;
  int numberStrainTerms = 1;
  double ghostPenaltyCoefficient = 1.0;
  bool useAnalyticalShape = true;
  int problemType = 1;
  
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&problemType, "-pT", "--problemType", "Select the problem type.");
  args.AddOption(&displacementOrder, "-do", "--displacementOrder", "Finite element displacement polynomial degree");
  args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
		 "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&poissonsRatio, "-poissonsRatio", "--poissonsRatioCoefficient",
		 "Value of Poisson's ratio.");
  args.AddOption(&youngsModulus, "-youngsModulus", "--youngsModulusCoefficient",
		 "Value of Young's modulus.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
		 "--no-visualization",
		 "Enable or disable GLVis visualization.");
  args.AddOption(&useEmbedded, "-emb", "--use-embedded", "-no-emb",
		 "--no-embedded",
		 "Use Embedded when there is surface that will be embedded in a pre-existing mesh");
  args.AddOption(&geometricShape, "-gS", "--geometricShape",
		 "Shape of the embedded geometry that will be embedded");
  args.AddOption(&nTerms, "-tO", "--taylorOrder",
		 "Number of terms in the Taylor expansion");
  args.AddOption(&numberStrainTerms, "-nST", "--numberStrainTerms",
		 "Number of terms in the strain ghost penalty operator.");
  args.AddOption(&ghostPenaltyCoefficient, "-gPenCoef", "--ghost-penalty-coefficient",
		 "Ghost penalty scaling.");
  args.AddOption(&useAnalyticalShape, "-analyticalShape", "--use-analytical-shape", "-level-set",
		 "--use-level-set",
		 "Use analytical shape for computing distance and normals, otherwise a level set is used.");
  
#ifdef MFEM_USE_MUMPS
   args.AddOption(&mumps_solver, "-mumps", "--mumps-solver", "-no-mumps",
                  "--no-mumps-solver", "Use the MUMPS Solver.");
#endif

   args.Parse();
   if (!args.Good())
     {
       if (myid == 0)
	 {
	   args.PrintUsage(cout);
	 }
       return 1;
     }
   if (myid == 0)
     {
       args.PrintOptions(cout);
     }

   // 2. Calculate the shear and bulk moduli from Poisson's ratio and Young's modulus 
   double shearModCoefficient = youngsModulus/(2.0*(1.0+poissonsRatio));
   double bulkModCoefficient = youngsModulus*poissonsRatio/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio)) + (2.0/3.0)*shearModCoefficient;
 
   // 3. Read the mesh from the given mesh file, and refine uniformly.
   Mesh *mesh;
   mesh = new Mesh(mesh_file, 1, 1);
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); }
   int dim = mesh->Dimension();
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   
   // 4. Define the coefficients, analytical solution, and rhs of the PDE.
   VectorFunctionCoefficient fcoeff(dim, fFun);
   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   ShiftedMatrixFunctionCoefficient traction_shifted(dim, traction_ex);
   VectorFunctionCoefficient fcoeff3D(dim, fFun3D);
   VectorFunctionCoefficient ucoeff3D(dim, uFun_ex3D);
   ShiftedMatrixFunctionCoefficient traction_shifted3D(dim, traction_ex3D);
   MatrixFunctionCoefficient traction_3D(dim, traction_ex3D);
   
   // 5. Create the Linear Elasticity solver
   mfem::LinearElasticitySolver* ssolv=new mfem::LinearElasticitySolver(pmesh, displacementOrder, useEmbedded, geometricShape, nTerms, numberStrainTerms, ghostPenaltyCoefficient, mumps_solver, useAnalyticalShape, visualization);

   // 6. Create the level set grid function and the marking class
   ParGridFunction * level_set_gf = NULL;
   // 7. Define the FECollection and FESpace for the level-set      
   mfem::FiniteElementCollection* lsvec = NULL;
   mfem::ParFiniteElementSpace* lsfes = NULL;
   
   if (useEmbedded){
     // 8a. Define the FECollection and FESpace for the level-set      
     lsvec = new H1_FECollection(displacementOrder+2,dim);
     lsfes = new mfem::ParFiniteElementSpace(pmesh,lsvec);
     lsfes->ExchangeFaceNbrData();
     // 8b. Create the level set grid function and the marking class
     level_set_gf = new ParGridFunction(lsfes);
     level_set_gf->ExchangeFaceNbrData();
     
     if (useAnalyticalShape){
       // 8c. if we are using an analytical surface to describe the geometry
       //     Dist_Coefficient just returns +1 or -1 for inside or outside domain 
       Dist_Coefficient *neumann_dist_coef = new Dist_Coefficient(geometricShape);
       // 8d. project the neumann dist coef to the level set gf 
       level_set_gf->ProjectCoefficient(*neumann_dist_coef);
       // Exchange information for ghost elements i.e. elements that share a face
       // with element on the current processor, but belong to another processor.
       level_set_gf->ExchangeFaceNbrData();     
     }
     else {
       // 8c. if we are using a level set to describe the geometry
       //     Dist_Level_Set_Coefficient just returns +1 or -1 for inside or outside domain     
       Dist_Level_Set_Coefficient * neumann_dist_coef = new Dist_Level_Set_Coefficient(geometricShape);
       
       // 8d. project the neumann dist coef to the level set gf and smooth it.
       //    To smooth it, code is using DiffuseH1, but one can also use the PDEFilter
       //    Just uncomment the filter lines and comment out the DiffuseH1
       level_set_gf->ProjectCoefficient(*neumann_dist_coef);
       // PDEFilter filter(*pmesh, 0.1);
       //  filter.Filter(*neumann_dist_coef, *level_set_gf);
       DiffuseH1(*level_set_gf, 1.0);
       // Exchange information for ghost elements i.e. elements that share a face
       // with element on the current processor, but belong to another processor.      
       level_set_gf->ExchangeFaceNbrData();            
     }
     // 9. Set the Level set grid function in the linear elasticity solver 
     ssolv->SetLevelSetGridFunction(*level_set_gf);
     // 10. Create the distance and normal grid functions in the linear elaticity solver
     ssolv->CreateDistanceAndNormalGridFunctions();
     // 11. Calculate volume fractions in the linear elasticity solver
     ssolv->CalculateVolumeFractions();
     // 12. Mark mlements
     ssolv->MarkElements();
   }
   // 13. Extract the inactive dofs, create the ess_elem array and set the inactive entries to 0. 
   ssolv->ExtractInactiveDofsAndElements();     
   
   // 14. Set material coefficients, boundary conditions, shifted boundary condition and exact solution
   ssolv->AddMaterial(shearModCoefficient,bulkModCoefficient);
 
   if ( problemType == 1){
     ssolv->SetVolForce(fcoeff);
     ssolv->AddDisplacementBC(1,ucoeff);
     ssolv->AddDisplacementBC(2,ucoeff);
     ssolv->AddDisplacementBC(3,ucoeff);
     ssolv->AddDisplacementBC(4,ucoeff);
     ssolv->AddShiftedNormalStressBC(traction_shifted);
     ssolv->SetExactDisplacementSolution(ucoeff);
   }
   else if (problemType == 2){
     ssolv->SetVolForce(fcoeff3D);
     ssolv->AddDisplacementBC(1,ucoeff3D);
     ssolv->AddDisplacementBC(2,ucoeff3D);
     ssolv->AddDisplacementBC(3,ucoeff3D);
     ssolv->AddDisplacementBC(4,ucoeff3D);
     ssolv->AddDisplacementBC(5,ucoeff3D);
     ssolv->AddDisplacementBC(6,ucoeff3D);
     ssolv->AddShiftedNormalStressBC(traction_shifted3D);
     ssolv->SetExactDisplacementSolution(ucoeff3D);
   }
   // 15. Set Newton solver parameters, solve, compute L2 error and visualize fields    
   ssolv->SetNewtonSolver(1.0e-14,0.0,100000,1);
   ssolv->FSolve();
   ssolv->ComputeL2Errors();
   ssolv->VisualizeFields();
   
   delete ssolv;
   
   return 0;
}


void fFun(const Vector & x, Vector & f )
{
  double kappa = 500.0/3.0;
  double mu = 1000.0/13.0;
  
  f(0) = -(0.2*(pi*pi*mu*cos(pi*(x(0)+0.5))*sin(pi*(x(1)+0.5))) + (kappa-(2.0/3.0)*mu)*((pi*pi*cos((pi*(x(0)+0.5))/7.0)*cos((pi*(x(1)+0.5))/3.0))/210.0 + 0.1*pi*pi*cos(pi*(x(0)+0.5))*sin(pi*(x(1)+0.5))) + mu*0.1*pi*pi*cos(pi*(x(0)+0.5))*sin(pi*(x(1)+0.5)) + mu*(pi*pi/210.0)*cos((pi/7.0)*(x(0)+0.5))*cos((pi/3.0)*(x(1)+0.5)) ) * 1.0; 

  f(1) = - (-(pi/3.0)*(pi/15.0)*mu*sin((pi/7.0)*(x(0)+0.5))*sin((pi/3.0)*(x(1)+0.5))+(kappa-(2.0/3.0)*mu)*((pi/10.0)*pi*sin(pi*(x(0)+0.5))*cos(pi*(x(1)+0.5))-(pi/30.0)*(pi/3.0)*sin((pi/7.0)*(x(0)+0.5))*sin((pi/3.0)*(x(1)+0.5))) + mu*((pi/10.0)*pi*sin(pi*(x(0)+0.5))*cos(pi*(x(1)+0.5))-(pi/70.0)*(pi/7.0)*sin((pi/7.0)*(x(0)+0.5))*sin((pi/3.0)*(x(1)+0.5))) ) * 1.0;
}

void uFun_ex(const Vector & x, Vector & u)
{
  u(0) = -0.1*cos(pi*(x(0)+0.5))*sin(pi*(x(1)+0.5))*1.0;
  u(1) = 0.1*sin(pi*(x(0)+0.5)/7.0)*sin(pi*(x(1)+0.5)/3.0)*1.0;
}

void traction_ex(const Vector & x, DenseMatrix & tN )
{
  double kappa = 500.0/3.0;
  double mu = 1000.0/13.0;
  double sigma_xx = (pi/5.0)*mu*sin(pi*(x(0)+0.5))*sin(pi*(x(1)+0.5))+(kappa-(2.0/3.0)*mu)*((pi/10.0)*sin(pi*(x(0)+0.5))*sin(pi*(x(1)+0.5))+(pi/30.0)*sin((pi/7.0)*(x(0)+0.5))*cos((pi/3.0)*(x(1)+0.5)));
  double sigma_xy = mu*(-(pi/10.0)*cos(pi*(x(0)+0.5))*cos(pi*(x(1)+0.5))+(pi/70.0)*cos((pi/7.0)*(x(0)+0.5))*sin((pi/3.0)*(x(1)+0.5)));
  double sigma_yx = mu*(-(pi/10.0)*cos(pi*(x(0)+0.5))*cos(pi*(x(1)+0.5))+(pi/70.0)*cos((pi/7.0)*(x(0)+0.5))*sin((pi/3.0)*(x(1)+0.5)));
  double sigma_yy = (pi/15.0)*mu*sin((pi/7.0)*(x(0)+0.5))*cos((pi/3.0)*(x(1)+0.5))+(kappa-(2.0/3.0)*mu)*((pi/10.0)*sin(pi*(x(0)+0.5))*sin(pi*(x(1)+0.5))+(pi/30.0)*sin((pi/7.0)*(x(0)+0.5))*cos((pi/3.0)*(x(1)+0.5)));
  tN = 0.0; 
  tN(0,0) = sigma_xx*1.0;
  tN(0,1) = sigma_xy*1.0;
  tN(1,0) = sigma_yx*1.0;
  tN(1,1) = sigma_yy*1.0;
}


void fFun3D(const Vector & x, Vector & f )
{
  // assumes kappa = 500/3 and mu = 76.9230769231
  double kappa = 500.0/3.0;
  double mu = 1000.0/13.0;

  f(0) = -((kappa+(1.0/3.0)*mu)*pi*x(1)*(1.0/20.0)*cos(pi*x(0)*x(1)*x(2)*0.5)+(kappa+(1.0/3.0)*mu)*(pi*pi/210.0)*cos(pi*x(0)/7.0)*cos(pi*x(1)/3.0)*cos(pi*x(2)/5.0)-(kappa+(1.0/3.0)*mu)*pi*pi*(1.0/40.0)*x(0)*x(1)*x(1)*x(2)*sin(pi*x(0)*x(1)*x(2)*0.5)+(4*mu+(kappa-(2.0/3.0)*mu))*0.1*pi*pi*cos(pi*x(0))*cos(pi*x(2))*sin(pi*x(1)));
  
  f(1) = -(2.0*(kappa+(1.0/3.0)*mu)*(pi*x(0)*0.025*cos(pi*x(0)*x(1)*x(2)*0.5)-pi*pi*0.0125*x(0)*x(0)*x(1)*x(2)*sin(pi*x(0)*x(1)*x(2)*0.5)+0.05*pi*pi*cos(pi*x(1))*cos(pi*x(2))*sin(pi*x(0)))-pi*pi*(mu*((1.0/45.0)+(1.0/250.0)+(1.0/490.0)-(1.0/135.0))+kappa/90.0)*sin(pi*x(0)/7.0)*sin(pi*x(1)/3.0)*cos(pi*x(2)/5.0));

  f(2) = (kappa+(4.0/3.0)*mu)*pi*pi*0.025*x(0)*x(0)*x(1)*x(1)*sin(pi*x(0)*x(1)*x(2)*0.5)+0.025*mu*pi*pi*x(0)*x(0)*x(2)*x(2)*sin(pi*x(0)*x(1)*x(2)*0.5)+0.025*mu*pi*pi*x(1)*x(1)*x(2)*x(2)*sin(pi*x(0)*x(1)*x(2)*0.5)+(kappa+(1.0/3.0)*mu)*(1.0/150.0)*pi*pi*sin(pi*x(0)/7.0)*cos(pi*x(1)/3.0)*sin(pi*x(2)/5.0)+(kappa+(1.0/3.0)*mu)*0.1*pi*pi*sin(pi*x(0))*sin(pi*x(1))*sin(pi*x(2));

}

void uFun_ex3D(const Vector & x, Vector & u)
{
  u(0) = -0.1*cos(pi*x(0))*sin(pi*x(1))*cos(pi*x(2));
  u(1) = 0.1*sin(pi*x(0)/7.0)*sin(pi*x(1)/3.0)*cos(pi*x(2)/5.0);
  u(2) = 0.1*sin(pi*x(0)*x(1)*x(2)/2.0);
}

void traction_ex3D(const Vector & x, DenseMatrix & tN )
{
  double kappa = 500.0/3.0;
  double mu = 1000.0/13.0;

  double u_11 = 0.1 * pi * sin(pi*x(0)) * sin(pi*x(1)) * cos(pi*x(2));
  double u_12 = -0.1 * pi * cos(pi*x(0)) * cos(pi*x(1)) * cos(pi*x(2));
  double u_13 = 0.1 * pi * cos(pi*x(0)) * sin(pi*x(1)) * sin(pi*x(2));
  double u_21 = (pi/70.0) * cos(pi*x(0)/7.0) * sin(pi*x(1)/3.0) * cos(pi*x(2)/5.0);
  double u_22 = (pi/30.0) * sin(pi*x(0)/7.0) * cos(pi*x(1)/3.0) * cos(pi*x(2)/5.0);
  double u_23 = -(pi/50.0) * sin(pi*x(0)/7.0) * sin(pi*x(1)/3.0) * sin(pi*x(2)/5.0);
  double u_31 = (pi/20.0)*x(1)*x(2) * cos((pi/2.0)*x(0)*x(1)*x(2));
  double u_32 = (pi/20.0)*x(0)*x(2) * cos((pi/2.0)*x(0)*x(1)*x(2));
  double u_33 = (pi/20.0)*x(0)*x(1) * cos((pi/2.0)*x(0)*x(1)*x(2));
  
  double div_u = u_11 + u_22 + u_33;
  double epsilon_11 = u_11;
  double epsilon_12 = (u_12+u_21)*0.5;
  double epsilon_13 = (u_13+u_31)*0.5;
  double epsilon_22 = u_22;
  double epsilon_23 = (u_23+u_32)*0.5;
  double epsilon_33 = u_33;

  double sigma_11 = 2 * mu * epsilon_11 + (kappa - (2.0/3.0) * mu)*div_u;
  double sigma_12 = 2 * mu * epsilon_12;
  double sigma_13 = 2 * mu * epsilon_13;
  double sigma_22 = 2 * mu * epsilon_22 + (kappa - (2.0/3.0) * mu)*div_u;
  double sigma_23 = 2 * mu * epsilon_23;
  double sigma_33 = 2 * mu * epsilon_33 + (kappa - (2.0/3.0) * mu)*div_u;
        
  tN = 0.0; 
  tN(0,0) = sigma_11;
  tN(0,1) = sigma_12;
  tN(0,2) = sigma_13;
  
  tN(1,0) = sigma_12;
  tN(1,1) = sigma_22;
  tN(1,2) = sigma_23;
  
  tN(2,0) = sigma_13;
  tN(2,1) = sigma_23;
  tN(2,2) = sigma_33;
  
}
