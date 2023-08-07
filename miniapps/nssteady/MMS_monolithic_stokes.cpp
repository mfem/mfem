//
// Description:  This example code solves a 2D Stokes problem corresponding,
//               to the saddle point system:
//
//              -nu \nabla^2 u + \nabla p = f
//                               \div u   = 0
//
// The algebraic form of the system is:
//
//                [  K    G ] [v] = [ fv ]                 
//                [  -B   0 ] [p]   [  0 ] 
//
//               with essential boundary conditions.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g). The discretization uses the
//               inf-sup stable finite element pair Pn-Pn-1 (Taylor-Hood pair).
//

#include "mfem.hpp"

// Include for mkdir
#include <sys/stat.h>

#ifdef M_PI
#define PI M_PI
#else
#define M_PI 3.14159265358979
#endif

using namespace mfem;

// Forward declarations of functions and variables
ParMesh *pmesh = nullptr;
ParLinearForm *mass_lf = nullptr;
double volume = 0.0;
void Orthogonalize(Vector &v);
void MeanZero(ParGridFunction &p);
double ComputeLift(ParGridFunction &p);


void   V_exact1(const Vector &X, Vector &v);
double P_exact1(const Vector &X);
std::function<void(const Vector &, Vector &)> RHS1(const double &kin_vis);

void   V_exact2(const Vector &X, Vector &v);
double P_exact2(const Vector &X);
std::function<void(const Vector &, Vector &)> RHS2(const double &kin_vis);

void   V_exact3(const Vector &X, Vector &v);
double P_exact3(const Vector &X);
std::function<void(const Vector &, Vector &)> RHS3(const double &kin_vis);

double pZero(const Vector &X);
void   vZero(const Vector &X, Vector &v);

// Test
int main(int argc, char *argv[])
{
   //
   /// 1. Initialize MPI and HYPRE.
   //
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   Hypre::Init();


   //
   /// 2. Parse command-line options. 
   //
   int fun = 1;               // exact solution

   int porder = 1;            // fe
   int vorder = 2;

   int n = 10;                // mesh
   int dim = 2;
   int elem = 0;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   double kin_vis = 0.01;     // kinematic viscosity

   double rel_tol = 1e-7;     // solvers
   double abs_tol = 1e-10;
   int tot_iter = 1000;
   int print_level = 0;

   double alpha = 1.;         // steady-state scheme
   double gamma = 1.;         // relaxation

   bool paraview = false;     // postprocessing
   bool verbose = false;
   const char *folderPath = "./";

   // TODO: check parsing and assign variables
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&dim,
                     "-d",
                     "--dimension",
                     "Dimension of the problem (2 = 2d, 3 = 3d)");
   args.AddOption(&elem,
                     "-e",
                     "--element-type",
                     "Type of elements used (0: Quad/Hex, 1: Tri/Tet)");
   args.AddOption(&n,
                     "-n",
                     "--num-elements",
                     "Number of elements in uniform mesh.");
   args.AddOption(&ser_ref_levels,
                     "-rs",
                     "--refine-serial",
                     "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                     "-rp",
                     "--refine-parallel",
                     "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&vorder, "-ov", "--order_vel",
                     "Finite element order for velocity (polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&porder, "-op", "--order_pres",
                     "Finite element order for pressure(polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&kin_vis,
                     "-kv",
                     "--kin-viscosity",
                     "Kinematic viscosity");
   args.AddOption(&rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Outer solve.");
   args.AddOption(&tot_iter,
                  "-it",
                  "--linear-iterations",
                  "Maximum iterations for the linear solve.");
   args.AddOption(&folderPath,
                  "-o",
                  "--output-folder",
                  "Output folder.");
   args.AddOption(&paraview, "-p", "--paraview", "-no-p",
                  "--no-paraview",
                  "Enable Paraview output.");
   args.AddOption(&fun, "-f", "--test-function",
                     "Analytic function to test");    
   args.AddOption(&print_level,
                  "-pl",
                  "--print-level",
                  "Print level.");              

   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(mfem::out);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
       args.PrintOptions(mfem::out);
   }


   //
   /// 3. Read the (serial) mesh from the given mesh file on all processors.
   //
   Element::Type type;
   switch (elem)
   {
      case 0: // quad
         type = (dim == 2) ? Element::QUADRILATERAL: Element::HEXAHEDRON;
         break;
      case 1: // tri
         type = (dim == 2) ? Element::TRIANGLE: Element::TETRAHEDRON;
         break;
   }

   Mesh mesh;
   switch (dim)
   {
      case 2: // 2d
         mesh = Mesh::MakeCartesian2D(n,n,type,true);	
         break;
      case 3: // 3d
         mesh = Mesh::MakeCartesian3D(n,n,n,type,true);	
         break;
   }


   for (int l = 0; l < ser_ref_levels; l++)
   {
       mesh.UniformRefinement();
   }


   //
   /// 4. Define a parallel mesh by a partitioning of the serial mesh. 
   // Refine this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   //
   pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh->UniformRefinement();
       }
   }

   // mesh
   dim=pmesh->Dimension();

   // FE collection and spaces for velocity and pressure
   H1_FECollection *vfec=new H1_FECollection(vorder,dim);
   H1_FECollection *pfec=new H1_FECollection(porder);
   ParFiniteElementSpace *vfes=new ParFiniteElementSpace(pmesh,vfec,dim);
   ParFiniteElementSpace *pfes=new ParFiniteElementSpace(pmesh,pfec,1);

   // initialize vectors of essential attributes
   Array<int> vel_ess_tdof, empty;
   Array<int> vel_ess_attr(pmesh->bdr_attributes.Max());
   vel_ess_attr=1;
   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof); 

   //
   // 5. Define the two BlockStructure of the problem.  block_offsets is used
   //    for Vector based on dof (like ParGridFunction or ParLinearForm),
   //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
   //    for the rhs and solution of the linear system).  The offsets computed
   //    here are local to the processor.
   int vdim = vfes->GetTrueVSize();
   int pdim = pfes->GetTrueVSize();

   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = vdim;
   block_offsets[2] = pdim;
   block_offsets.PartialSum();

   if (myrank == 0)
   {
      mfem::out << "Velocity dofs: " << vfes->GlobalVSize() << std::endl;
      mfem::out << "Pressure dofs: " << pfes->GlobalVSize() << std::endl;
   }

   BlockVector x(block_offsets), rhs(block_offsets), x_k(block_offsets), x_rel(block_offsets);

   x = 0.0; rhs = 0.0;


   //
   /// 6. Define the coefficients (e.g. parameters, analytical solution/s).
   //
   VectorFunctionCoefficient*    V_ex = nullptr;
   VectorFunctionCoefficient* f_coeff = nullptr;
   FunctionCoefficient*          P_ex = nullptr;
   switch (fun)
   {
   case 1:
      {
         V_ex = new VectorFunctionCoefficient(dim, V_exact1);
         P_ex = new FunctionCoefficient(P_exact1);
         f_coeff = new VectorFunctionCoefficient(dim, RHS1(kin_vis));
         break;
      }
   case 2:
      {
         V_ex = new VectorFunctionCoefficient(dim, V_exact2);
         P_ex = new FunctionCoefficient(P_exact2);
         f_coeff = new VectorFunctionCoefficient(dim, RHS2(kin_vis));
         break;
      }
   case 3:
      {
         V_ex = new VectorFunctionCoefficient(dim, V_exact3);
         P_ex = new FunctionCoefficient(P_exact3);
         f_coeff = new VectorFunctionCoefficient(dim, RHS3(kin_vis));
         break;
      }
   default:
      break;
   }


   // Project the correct boundary condition to the grid function
   // of v and sets the dofs in the velocity offset part of the x vector.
   ParGridFunction *v_gf(new ParGridFunction(vfes));
   ParGridFunction *p_gf(new ParGridFunction(pfes));
   v_gf->ProjectBdrCoefficient(*V_ex, vel_ess_attr);
   v_gf->GetTrueDofs(x.GetBlock(0));
   p_gf->GetTrueDofs(x.GetBlock(1));

   ParGridFunction *vk_gf(new ParGridFunction(vfes)); *vk_gf = 0.0;
   ParGridFunction *pk_gf(new ParGridFunction(pfes)); *pk_gf = 0.0;
   VectorGridFunctionCoefficient *vk_vc = new VectorGridFunctionCoefficient(vk_gf);
   VectorGridFunctionCoefficient *pk_c = new VectorGridFunctionCoefficient(pk_gf);


   //
   // 7. Assemble the finite element matrices for the Navier-Stokes operator
   //
   //           A = [  K   G ]                 
   //               [ -B   0 ]            
   //     where:
   //
   //     K = nu \int_\Omega \nabla u_h \cdot \nabla v_h d\Omega   u_h, v_h \in Vfes
   //     B  =  \int_\Omega \div u_h q_h d\Omega                   u_h \in Vfes q_h \in Pfes
   //     G  = -Bt =  \int_\Omega p_h \div v_h d\Omega             v_h \in Vfes p_h \in Pfes
   //

   // Bilinear forms to be assembled out of loop
   HypreParMatrix* K = new HypreParMatrix();
   HypreParMatrix* B = new HypreParMatrix();
   HypreParMatrix* Bt = new HypreParMatrix();
   HypreParMatrix* Mp = new HypreParMatrix();

   ParLinearForm *fform = new ParLinearForm();
   fform->Update(vfes, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(*f_coeff));
   fform->Assemble();

   ConstantCoefficient kin_vis_coeff(kin_vis);
   ParBilinearForm      *K_form=nullptr;
   K_form = new ParBilinearForm(vfes);
;
   K_form->AddDomainIntegrator(new VectorDiffusionIntegrator(kin_vis_coeff));
   K_form->Assemble();  K_form->Finalize();

   // Form the linear system for the first block (apply ess bcs modification to matrix and rhs). 
   //K_form->FormLinearSystem(vel_ess_tdof, x.GetBlock(0), rhs.GetBlock(0), *K,
   //                       x.GetBlock(0), rhs.GetBlock(0));
   K = K_form->ParallelAssemble();
   HypreParMatrix* Ke = K->EliminateRowsCols(vel_ess_tdof);
   K->EliminateBC(*Ke, vel_ess_tdof, x.GetBlock(0), rhs.GetBlock(0));   
   
   ParMixedBilinearForm *B_form = new ParMixedBilinearForm(vfes, pfes);
   B_form->AddDomainIntegrator(new VectorDivergenceIntegrator);
   B_form->Assemble();  B_form->Finalize();
   B = B_form->ParallelAssemble();
   HypreParMatrix* Be = B->EliminateCols(vel_ess_tdof);
   Be->Mult(-1.0, x.GetBlock(0), 1.0, rhs.GetBlock(1)); // rhs_p -= Be*x_v
   Orthogonalize(rhs.GetBlock(1));

   // Form a rectangular matrix of the block system and eliminate the
   // columns (trial dofs). Like in FormLinearSystem, the boundary values
   // are moved from the first block of the x vector into the second block
   // of the rhs.
   //B_form->FormRectangularLinearSystem(vel_ess_tdof, empty, x.GetBlock(0), rhs.GetBlock(1), *B,
   //                           x.GetBlock(0), rhs.GetBlock(1));

   // Define the gradient matrix by transposing the discretized divergence matrix.
   Bt = B->Transpose();
   (*Bt) *= -1.0;

   // Flip signs of the second block part to make system symmetric.
   (*B) *= -1.0;
   rhs.GetBlock(1) *= -1.0;

   BlockOperator *stokesop = new BlockOperator(block_offsets);
   stokesop->SetBlock(0, 0, K);
   stokesop->SetBlock(0, 1, Bt);
   stokesop->SetBlock(1, 0, B);


   //
   // 8. Construct the operators for the block preconditioner
   //
   //                 P = [  A           0        ]
   //                     [  0    (kin_vis)^-1 Mp ]
   //
   //     Here we use a single VCycle of AMG for the diffusive part K,
   //     and AMG on pressure mass matrix Mp to approximate the 
   //     inverse of the pressure Schur Complement (the pressure uses an OrthoSolver
   //     which removes the nullspace).

   // The preconditioning technique is to approximate a Schur complement, which
   // is achieved here by forming the pressure mass matrix.
   ParBilinearForm *Mpform = new ParBilinearForm(pfes);
   Mpform->AddDomainIntegrator(new MassIntegrator);
   Mpform->Assemble(); Mpform->Finalize();
   Mp = Mpform->ParallelAssemble();
   *Mp *= 1.0/kin_vis;
   //HypreDiagScale *invMp = new HypreDiagScale(*Mp);
   HypreBoomerAMG *invMp = new HypreBoomerAMG(*Mp);
   invMp->SetPrintLevel(0);
   OrthoSolver* invMpOrtho = new OrthoSolver(pmesh->GetComm());
   invMpOrtho->SetSolver(*invMp);

   HypreBoomerAMG *invK = new HypreBoomerAMG(*K);
   invK->SetPrintLevel(0);
   invK->iterative_mode = false;

   BlockDiagonalPreconditioner *stokesprec = new BlockDiagonalPreconditioner(block_offsets);
   stokesprec->SetDiagonalBlock(0, invK);
   stokesprec->SetDiagonalBlock(1, invMpOrtho);


   //
   // 9. Solve using MINRES (system is symmetric).
   //
   MINRESSolver solver(pmesh->GetComm());
   solver.iterative_mode = false;
   solver.SetAbsTol(0.0);
   solver.SetRelTol(rel_tol);
   solver.SetMaxIter(tot_iter);
   solver.SetOperator(*stokesop);
   solver.SetPreconditioner(*stokesprec);
   solver.SetPrintLevel(print_level);
   solver.Mult(rhs, x);

   // Retrieve solution
   v_gf->SetFromTrueDofs(x.GetBlock(0));
   p_gf->SetFromTrueDofs(x.GetBlock(1));
   MeanZero(*p_gf);

   //
   // 10. Compute error and output
   // Define a quadrature rule used to compute error wrt exactr solution
   int order_quad = std::max(2, 2*vorder+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   // Compute lift to add to the output pressure
   ParGridFunction *p_ex_gf(new ParGridFunction(pfes));
   p_ex_gf->ProjectCoefficient(*P_ex);
   double lift = ComputeLift(*p_ex_gf);
   delete p_ex_gf; p_ex_gf = nullptr;

   ParaViewDataCollection *paraview_dc = nullptr;
   ParGridFunction* p_gf_out = new ParGridFunction(pfes); // need additional p_gf_out to be lifted
   p_gf_out->ProjectGridFunction(*p_gf);
   *p_gf_out += lift;
   double err_v = v_gf->ComputeL2Error(*V_ex, irs);
   double norm_v = ComputeGlobalLpNorm(2, *V_ex, *pmesh, irs);
   double err_p = p_gf_out->ComputeL2Error(*P_ex, irs);
   double norm_p = ComputeGlobalLpNorm(2, *P_ex, *pmesh, irs);


   if (myrank == 0)
   {
      mfem::out << "|| v_h - v_ex || = " << err_v << "\n";
      mfem::out << "|| v_h - v_ex || / || v_ex || = " << err_v / norm_v << "\n";
      mfem::out << "|| p_h - p_ex || = " << err_p << "\n";
      mfem::out << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
   }

   if( paraview )
   {
      // Creating output directory if not existent
      if (mkdir(folderPath, 0777) == -1) {mfem::err << "Error :  " << strerror(errno) << std::endl;}

      // exact solution
      ParGridFunction* velocityExactPtr = new ParGridFunction(vfes);
      ParGridFunction* pressureExactPtr = new ParGridFunction(pfes);
      ParGridFunction*           rhsPtr = new ParGridFunction(vfes);
      velocityExactPtr->ProjectCoefficient(*V_ex);
      pressureExactPtr->ProjectCoefficient(*P_ex);
      rhsPtr->ProjectCoefficient(*f_coeff);

      // Create output data collections
      ParaViewDataCollection *paraview_dc = new ParaViewDataCollection("Results-Paraview", pmesh);
      paraview_dc->SetPrefixPath(folderPath);
      paraview_dc->SetLevelsOfDetail(vorder);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->RegisterField("velocity",v_gf);
      paraview_dc->RegisterField("pressure",p_gf_out);
      paraview_dc->RegisterField("velocity_exact",velocityExactPtr);
      paraview_dc->RegisterField("pressure_exact",pressureExactPtr);
      paraview_dc->RegisterField("rhs",rhsPtr);      
      paraview_dc->SetCycle(tot_iter);
      paraview_dc->SetTime(tot_iter);
      paraview_dc->Save();
   }   

   // Free used memory.
   delete vfec;
   delete pfec;
   delete vfes;
   delete pfes;
   delete v_gf;
   delete p_gf;
   delete p_ex_gf;

   delete fform;
   delete K_form;
   delete B_form;
   delete Mpform;

   delete stokesop;
   delete stokesprec;

   delete invK;
   delete invMp;
   delete invMpOrtho;

   delete K;
   delete Ke;
   delete B;
   delete Be;
   delete Bt;
   delete Mp;

   delete P_ex;
   delete V_ex;
   delete f_coeff;

   delete pmesh;

   return 0;
}


void Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, pmesh->GetComm());

   v -= global_sum / static_cast<double>(global_size);
}

void MeanZero(ParGridFunction &p)
{
   // Make sure not to recompute the inner product linear form every
   // application.
   if (mass_lf == nullptr)
   {
      ConstantCoefficient onecoeff(1.0);
      mass_lf = new ParLinearForm(p.ParFESpace());
      auto *dlfi = new DomainLFIntegrator(onecoeff);
      mass_lf->AddDomainIntegrator(dlfi);
      mass_lf->Assemble();

      ParGridFunction one_gf(p.ParFESpace());
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   }

   double integ = mass_lf->operator()(p);

   p -= integ / volume;
}

double ComputeLift(ParGridFunction &p)
{
   // Make sure not to recompute the inner product linear form every
   // application.
   if (mass_lf == nullptr)
   {
      ConstantCoefficient onecoeff(1.0);
      mass_lf = new ParLinearForm(p.ParFESpace());
      auto *dlfi = new DomainLFIntegrator(onecoeff);
      mass_lf->AddDomainIntegrator(dlfi);
      mass_lf->Assemble();

      ParGridFunction one_gf(p.ParFESpace());
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   }

   double integ = mass_lf->operator()(p);

   return integ/volume;   // CHECK should we scale by volume
}



// Exact solutions
void V_exact1(const Vector &X, Vector &v)
{
   const int dim = X.Size();

   double x = X[0];
   double y = X[1];
   if( dim == 3) {
      double z = X[2];
   }

   v = 0.0;

   v(0) = -cos(M_PI * x) * sin(M_PI * y);
   v(1) = sin(M_PI * x) * cos(M_PI * y);

   if( dim == 3) { v(2) = 0; }
}

std::function<void(const Vector &, Vector &)> RHS1(const double &kin_vis)
{
   return [kin_vis](const Vector &X, Vector &v)
   {
      const int dim = X.Size();

      double x = X[0];
      double y = X[1];
      
      if( dim == 3) {
         double z = X[2];
      }
      
      v = 0.0;

      v(0) = 1.0 - 2.0 * kin_vis * M_PI * M_PI * cos(M_PI * x) * sin(M_PI * y);
      v(1) = 1.0 + 2.0 * kin_vis * M_PI * M_PI * cos(M_PI * y) * sin(M_PI * x);
      if( dim == 3) {
         v(2) = 0;
      }
   };
}

double P_exact1(const Vector &X)
{
   double x = X[0];
   double y = X[1];

   double p = x + y - 1.0;;

   return p;
}



void V_exact2(const Vector &X, Vector &v)
{
   const int dim = X.Size();

   double x = X[0];
   double y = X[1];
   if( dim == 3) {
      double z = X[2];
   }

   v = 0.0;

   v(0) = pow(sin(M_PI*x),2) * sin(M_PI*y) * cos(M_PI*y);
   v(1) = - pow(sin(M_PI*y),2) * sin(M_PI*x) * cos(M_PI*x);
   if( dim == 3) { v(2) = 0; }
}

std::function<void(const Vector &, Vector &)> RHS2(const double &kin_vis)
{
   return [kin_vis](const Vector &X, Vector &v)
   {
      const int dim = X.Size();

      double x = X[0];
      double y = X[1];
      
      if( dim == 3) {
         double z = X[2];
      }
      
      v = 0.0;

      v(0) =  y*(2*x - 1)*(y - 1)
              - kin_vis * M_PI * M_PI * 2 * sin(M_PI*y) * cos(M_PI*y) *(pow(cos(M_PI*x),2) - 3*pow(sin(M_PI*x),2));
      v(1) =  x*(2*y - 1)*(x - 1)
              + kin_vis * M_PI * M_PI * 2 * sin(M_PI*x) * cos(M_PI*x) *(pow(cos(M_PI*y),2) - 3*pow(sin(M_PI*y),2));

      if( dim == 3) {
         v(2) = 0;
      }
   };
}

double P_exact2(const Vector &X)
{
   double x = X[0];
   double y = X[1];

   double p = x*y*(1-x)*(1-y);

   return p;
}



void V_exact3(const Vector &X, Vector &v)
{
   const int dim = X.Size();

   double x = X[0];
   double y = X[1];
   if( dim == 3) {
      double z = X[2];
   }

   v = 0.0;

   v(0) = -sin(x)*cos(y);
   v(1) = cos(x)*sin(y);
   if( dim == 3) { v(2) = 0; }
}

std::function<void(const Vector &, Vector &)> RHS3(const double &kin_vis)
{
   return [kin_vis](const Vector &X, Vector &v)
   {
      const int dim = X.Size();

      double x = X[0];
      double y = X[1];
      
      if( dim == 3) {
         double z = X[2];
      }
      
      v = 0.0;

      v(0) = cos(x) - 2*kin_vis*cos(y)*sin(x);
      v(1) = cos(y) + 2*kin_vis*cos(x)*sin(y);
      if( dim == 3) {
         v(2) = 0;
      }
   };
}

double P_exact3(const Vector &X)
{
   double x = X[0];
   double y = X[1];

   double p = sin(x) + sin(y);

   return p;
}

void vZero(const Vector &X, Vector &v)
{
   v = 0.0;
}

double pZero(const Vector &X)
{
   return 0;
}