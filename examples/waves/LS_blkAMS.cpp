//                                MFEM Example multigrid-grid Cycle
//
// Compile with: make mg_maxwellp
//
// Sample runs:  mg_maxwellp -m ../data/one-hex.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "AMS_LS.hpp"

using namespace std;
using namespace mfem;

class Block_AMSSolver : public Solver {
private:
   /// The linear system matrix
   Array2D<HypreParMatrix* > A_array;
   Array2D<HypreParMatrix* > Pi;
   HypreParMatrix *Grad, *Pix, *Piy, *Piz;
   HypreParMatrix *l1A00, *l1A11;
   BlockOperator* GtAG;
   BlockOperator* PxtAPx;
   BlockOperator* PytAPy;
   BlockOperator* PztAPz;
   Array<int> offsets;
   Array<int> offsetsG;
   Array<int> offsetsPi;
   BlockOperator * D;
   BlockOperator * A;
   BlockOperator * G;
   BlockOperator * Px;
   BlockOperator * Py;
   BlockOperator * Pz;
   HypreBoomerAMG *G00_inv, *Px00_inv, *Py00_inv, *Pz00_inv;
   HypreBoomerAMG *G11_inv, *Px11_inv, *Py11_inv, *Pz11_inv;;
   BlockDiagonalPreconditioner * blkAMG_G;
   BlockDiagonalPreconditioner * blkAMG_Px;
   BlockDiagonalPreconditioner * blkAMG_Py;
   BlockDiagonalPreconditioner * blkAMG_Pz;
   double theta = 1.0;
   string cycle_type = "023414320"; // 0-Smoother, 1-Grad, 2,3,4-Pix,Piy,Piz
   HypreSmoother * Dh;
   HypreParMatrix* Ah;
   int NumberOfCycles=1;
public:
   Block_AMSSolver(Array<int> offsets_, ParFiniteElementSpace *fespace)
      :  offsets(offsets_), offsetsG(3), offsetsPi(3)
   {
      Grad = new HypreParMatrix(*GetDiscreteGradientOp(fespace));
      Pi = GetNDInterpolationOp(fespace);
      Pix = new HypreParMatrix(*Pi(0,0));
      Piy = new HypreParMatrix(*Pi(0,1));
      Piz = new HypreParMatrix(*Pi(0,2));
      offsetsG[0]=0;
      offsetsG[1]=Grad->Width();
      offsetsG[2]=Grad->Width();
      offsetsG.PartialSum();
      offsetsPi[0]=0;
      offsetsPi[1]=Pix->Width();
      offsetsPi[2]=Pix->Width();
      offsetsPi.PartialSum();
      G      = new BlockOperator(offsets, offsetsG);
      Px     = new BlockOperator(offsets, offsetsPi);
      Py     = new BlockOperator(offsets, offsetsPi);
      Pz     = new BlockOperator(offsets, offsetsPi);
      GtAG   = new BlockOperator(offsetsG);
      PxtAPx = new BlockOperator(offsetsPi);
      PytAPy = new BlockOperator(offsetsPi);
      PztAPz = new BlockOperator(offsetsPi);
      A = new BlockOperator(offsets);
      this->height = 2*Grad->Height();
      this->width = 2*Grad->Height();
      blkAMG_G = new BlockDiagonalPreconditioner(offsetsG);
      blkAMG_Px = new BlockDiagonalPreconditioner(offsetsPi);
      blkAMG_Py = new BlockDiagonalPreconditioner(offsetsPi);
      blkAMG_Pz = new BlockDiagonalPreconditioner(offsetsPi);
   }
   virtual void SetOperator(const Operator & ) {}
   virtual void SetOperator(Array2D<HypreParMatrix*> Op) {
      A_array = Op;
      l1A00 = new HypreParMatrix(*A_array(0,0));
      l1A11 = new HypreParMatrix(*A_array(1,1));
      // DiagAddL1norm();
      HypreSmoother * D_00 = new HypreSmoother;  
      D_00->SetType(HypreSmoother::l1GS);
      // D_00->SetType(HypreSmoother::Jacobi);
      D_00->SetOperator(*l1A00);
      HypreSmoother * D_11 = new HypreSmoother;  
      D_11->SetType(HypreSmoother::l1GS);
      // D_11->SetType(HypreSmoother::Jacobi);
      D_11->SetOperator(*l1A11);
      
      D = new BlockOperator(offsets);
      D->SetDiagonalBlock(0, D_00);
      D->SetDiagonalBlock(1, D_11);
      SetOperators();
   }

   virtual void SetOperators() {
      int i,j;
      for (i=0; i<2 ; i++)
      {
         A->SetBlock(i,i,A_array(i,i));
         G->SetBlock(i,i,Grad);
         Px->SetBlock(i,i,Pix);
         Py->SetBlock(i,i,Piy);
         Pz->SetBlock(i,i,Piz);
         for (j=0; j<2 ; j++)
         {
            A->SetBlock(i,j,A_array(i,j));
            GtAG->SetBlock(i,j,RAP(A_array(i,j),Grad));
            PxtAPx->SetBlock(i,j,RAP(A_array(i,j),Pix));
            PytAPy->SetBlock(i,j,RAP(A_array(i,j),Piy));
            PztAPz->SetBlock(i,j,RAP(A_array(i,j),Piz));
         }
      }

      for (i=0; i<2 ; i++)
      { 
         HypreBoomerAMG * G_AMG = new HypreBoomerAMG(*RAP(A_array(i,i),Grad));
         HypreBoomerAMG * Px_AMG = new HypreBoomerAMG(*RAP(A_array(i,i),Pix));
         HypreBoomerAMG * Py_AMG = new HypreBoomerAMG(*RAP(A_array(i,i),Piy));
         HypreBoomerAMG * Pz_AMG = new HypreBoomerAMG(*RAP(A_array(i,i),Piz));
         G_AMG->SetPrintLevel(0);
         G_AMG->SetErrorMode(HypreSolver::ErrorMode::IGNORE_HYPRE_ERRORS);
         Px_AMG->SetPrintLevel(0);
         Px_AMG->SetErrorMode(HypreSolver::ErrorMode::IGNORE_HYPRE_ERRORS);
         Py_AMG->SetPrintLevel(0);
         Py_AMG->SetErrorMode(HypreSolver::ErrorMode::IGNORE_HYPRE_ERRORS);
         Pz_AMG->SetPrintLevel(0);
         Pz_AMG->SetErrorMode(HypreSolver::ErrorMode::IGNORE_HYPRE_ERRORS);
         blkAMG_G->SetDiagonalBlock(i,G_AMG);
         blkAMG_Px->SetDiagonalBlock(i,Px_AMG);
         blkAMG_Py->SetDiagonalBlock(i,Py_AMG);
         blkAMG_Pz->SetDiagonalBlock(i,Pz_AMG);
      }   
   }
   virtual void SetTheta(const double a) {theta = a;}
   virtual void SetCycleType(const string c_type) {cycle_type = c_type;}
   virtual void SetNumberofCycles(const int k) {NumberOfCycles = k;}


   virtual void DiagAddL1norm()
   {
      
      int n=A_array(1,1)->Height();
      Vector l1norm0(n);
      Vector l1norm1(n);

      Getrowl1norm(A_array(0,1), l1norm0);
      Getrowl1norm(A_array(1,0), l1norm1);

      hypre_ParCSRMatrix * A_00 = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*l1A00);
      // Add the L1 norms on the diagonal
      for (int j = 0; j < n; j++)
      {
         A_00->diag->data[A_00->diag->i[j]] += l1norm0(j);
      }

      hypre_ParCSRMatrix * A_11 = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*l1A11);
      // Add the L1 norms on the diagonal
      for (int j = 0; j < n; j++)
      {
         A_11->diag->data[A_11->diag->i[j]] += l1norm1(j);
      }
   }

   virtual void Getrowl1norm(HypreParMatrix *A , Vector &l1norm)
   {
      // First cast as hypre_ParCSRMatrix
      hypre_ParCSRMatrix * Ah = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*A);
      HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(Ah);
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(Ah);
      HYPRE_Int *A_diag_I = hypre_CSRMatrixI(A_diag);
      HYPRE_Int *A_diag_J = hypre_CSRMatrixJ(A_diag);
      HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);

      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(Ah);
      HYPRE_Int *A_offd_I = hypre_CSRMatrixI(A_offd);
      HYPRE_Int *A_offd_J = hypre_CSRMatrixJ(A_offd);
      HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
      HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

      //Initialize vector;

      l1norm = 0.0;

      for (int i = 0; i < num_rows; i++)
      {
         /* Add the l1 norm of the diag part of the ith row */
         for (int j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
            l1norm(i) += fabs(A_diag_data[j]);
         /* Add the l1 norm of the offd part of the ith row */
         if (num_cols_offd)
         {
            for (int j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
               l1norm(i) += fabs(A_offd_data[j]);
         }
      }

   }

   virtual void Mult(const Vector &r, Vector &z) const
   {
      int n = r.Size();
      int m = A->Height();
      int Numit = 0;
      // int k = G->Width();
      if (n != m ) {cout << "Size inconsistency" << endl;}

      Vector res(n), raux(n),zaux(n);
      //initialization
      res = r; z = 0.0;
      //
      Array<BlockOperator *> Tr_v(4);
      Array<BlockOperator *> PtAP_v(4);
      Array<BlockDiagonalPreconditioner *> blkAMG_v(4);
      Tr_v[0]     = G;        Tr_v[1]     = Px;        Tr_v[2]     = Py;        Tr_v[3]     = Pz;
      PtAP_v[0]   = GtAG;     PtAP_v[1]   = PxtAPx;    PtAP_v[2]   = PytAPy;    PtAP_v[3]   = PztAPz;
      blkAMG_v[0] = blkAMG_G; blkAMG_v[1] = blkAMG_Px; blkAMG_v[2] = blkAMG_Py; blkAMG_v[3] = blkAMG_Pz;  
      //
      int len = cycle_type.length();
      Array<int> ii(len);
      for (int i=0; i<len; i++){ii[i]=cycle_type[i]-'0';}
      //
      for (int ic = 0; ic<NumberOfCycles; ic++)
      {
         for (int j = 0; j<len ; j++)
         {
            int i = ii[j];
            if (i ==0)
            {
               D->Mult(res,zaux); zaux *= theta;
            }
            else
            {
               GetCorrection(Tr_v[i-1], PtAP_v[i-1], blkAMG_v[i-1], res, zaux);
            }
            z +=zaux; 
            A->Mult(zaux,raux); res -=raux;
         }
         // Numit++;
         // // double beta = Norm(res);
         // double beta = sqrt(InnerProduct(MPI_COMM_WORLD, res, res));
         // if(beta < 1e-6)
         // {
         //    int myid;
         //    MPI_Comm_rank(MPI_COMM_WORLD, &myid); // Determine process identifier
         //    if (myid == 0){
         //       mfem::out << "Convergend in " << Numit << " iterations. " << 
         //          "||r||_L2 = " << beta << "\n";
         //    }      
         // break;
         // } 
      }
   }

   void GetCorrection(BlockOperator* Tr, BlockOperator* op, BlockDiagonalPreconditioner *prec, Vector &r, Vector &z) const
   {
      int k = Tr->Width();
      Vector raux(k), zaux(k);
      // Map trough the Transpose of the Transfer operator
      Tr->MultTranspose(r,raux);
      zaux = 0.0;

      int maxit(3000);
      double rtol(0.0);
      double atol(1e-8);
      
      // CGSolver cg(MPI_COMM_WORLD);
      // cg.SetAbsTol(atol);
      // cg.SetRelTol(rtol);
      // cg.SetMaxIter(maxit);
      // cg.SetOperator(*op);
      // cg.SetPreconditioner(*prec);
      // cg.SetPrintLevel(0);
      // cg.Mult(raux, zaux);
         prec->Mult(raux,zaux);

      // Map back to the original space through the Tranfer operator
      Tr->Mult(zaux, z);
   }
   virtual ~Block_AMSSolver(){}
};
// Define exact solution
void E_exact(const Vector & x, Vector & E);
void H_exact(const Vector & x, Vector & H);
void scaledf_exact_E(const Vector & x, Vector & f_E);
void scaledf_exact_H(const Vector & x, Vector & f_H);
void f_exact_E(const Vector & x, Vector & f_E);
void f_exact_H(const Vector & x, Vector & f_H);
void get_maxwell_solution(const Vector & x, double E[], double curlE[], double curl2E[]);

int dim;
double omega;
int isol = 1;


int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialise MPI
   int num_procs, myid;
   MPI_Init(&argc, &argv); // Initialise MPI
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs); //total number of processors available
   MPI_Comm_rank(MPI_COMM_WORLD, &myid); // Determine process identifier
   // 1. Parse command-line options.
   // geometry file
   // const char *mesh_file = "../data/star.mesh";
   const char *mesh_file = "../../data/one-hex.mesh";
   // finite element order of approximation
   int order = 1;
   // static condensation flag
   bool static_cond = false;
   // visualization flag
   bool visualization = 1;
   // number of wavelengths
   double k = 1.0;
   // number of mg levels
   int maxref = 1;
   // number of initial ref
   int initref = 1;
   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&maxref, "-ref", "--maxref",
                  "Number of Refinements.");
   args.AddOption(&initref, "-initref", "--initref",
                  "Number of initial refinements.");
   args.AddOption(&isol, "-isol", "--exact",
                  "Exact solution flag - "
                  " 1:sinusoidal, 2: point source, 3: plane wave");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   // check if the inputs are correct
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Angular frequency
   omega = 2.0*k*M_PI;
   // omega = k;

   // 2. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 4. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   // ParFiniteElementSpace *fespace = new ParFiniteElementSpace(mesh, fec);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = fespace->GetVSize();
   block_offsets[2] = fespace->GetVSize();
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = fespace->TrueVSize();
   block_trueOffsets[2] = fespace->TrueVSize();
   block_trueOffsets.PartialSum();

   //    _           _    _  _       _  _
   //   |             |  |    |     |    |
   //   |  A00   A01  |  | E  |     |F_E |
   //   |             |  |    |  =  |    |
   //   |  A10   A11  |  | H  |     |F_G |
   //   |_           _|  |_  _|     |_  _|
   //
   // A00 = (curl E, curl F) + \omega^2 (E,F)
   // A01 = - \omega *( (curl E, F) + (E,curl F)
   // A10 = - \omega *( (curl H, G) + (H,curl G)
   // A11 = (curl H, curl H) + \omega^2 (H,G)

   BlockVector x(block_offsets), rhs(block_offsets);
   BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);

   x = 0.0;
   rhs = 0.0;
   trueX = 0.0;
   trueRhs = 0.0;

   VectorFunctionCoefficient Eex(sdim, E_exact);
   ParGridFunction * E_gf = new ParGridFunction;
   E_gf->MakeRef(fespace, x.GetBlock(0));
   E_gf->ProjectCoefficient(Eex);

   VectorFunctionCoefficient Hex(sdim, H_exact);
   ParGridFunction * H_gf = new ParGridFunction;   
   H_gf->MakeRef(fespace, x.GetBlock(1));
   H_gf->ProjectCoefficient(Hex);

   // // 6. Set up the linear form
   VectorFunctionCoefficient sf_E(sdim,scaledf_exact_E);
   VectorFunctionCoefficient sf_H(sdim,scaledf_exact_H);
   VectorFunctionCoefficient f_E(sdim,f_exact_E);
   VectorFunctionCoefficient f_H(sdim,f_exact_H);

   ParLinearForm *b_E = new ParLinearForm;
   b_E->Update(fespace, rhs.GetBlock(0), 0);
   b_E->AddDomainIntegrator(new VectorFEDomainLFIntegrator(sf_H));
   b_E->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_E));
   b_E->Assemble();


   ParLinearForm *b_H = new ParLinearForm;
   b_H->Update(fespace, rhs.GetBlock(1), 0);
   b_H->AddDomainIntegrator(new VectorFEDomainLFIntegrator(sf_E));
   b_H->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_H));
   b_H->Assemble();



   // 7. Bilinear form a(.,.) on the finite element space
   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(pow(omega, 2));
   ConstantCoefficient neg(-abs(omega));
   ConstantCoefficient pos(abs(omega));
   //
   ParBilinearForm *a_EE = new ParBilinearForm(fespace);
   a_EE->AddDomainIntegrator(new CurlCurlIntegrator(one)); 
   a_EE->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a_EE->Assemble();
   a_EE->EliminateEssentialBC(ess_bdr,x.GetBlock(0), rhs.GetBlock(0));
   a_EE->Finalize();
   HypreParMatrix *A_EE = a_EE->ParallelAssemble();

   ParMixedBilinearForm *a_HE = new ParMixedBilinearForm(fespace,fespace);
   a_HE->AddDomainIntegrator(new MixedVectorCurlIntegrator(neg));
   a_HE->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(neg)); 
   a_HE->Assemble();
   a_HE->EliminateTrialDofs(ess_bdr, x.GetBlock(0), rhs.GetBlock(1));
   a_HE->Finalize();

   HypreParMatrix *A_HE = a_HE->ParallelAssemble();

   HypreParMatrix *A_EH = A_HE->Transpose();

   ParBilinearForm *a_HH = new ParBilinearForm(fespace);
   a_HH->AddDomainIntegrator(new CurlCurlIntegrator(one)); // one is the coeff
   a_HH->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a_HH->Assemble();
   a_HH->Finalize();
   HypreParMatrix *A_HH = a_HH->ParallelAssemble();

   BlockOperator *LS_Maxwellop = new BlockOperator(block_trueOffsets);
   LS_Maxwellop->SetBlock(0, 0, A_EE);
   LS_Maxwellop->SetBlock(0, 1, A_EH);
   LS_Maxwellop->SetBlock(1, 0, A_HE);
   LS_Maxwellop->SetBlock(1, 1, A_HH);
   

   fespace->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
   fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),trueRhs.GetBlock(0));

   fespace->GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));
   fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),trueRhs.GetBlock(1));

   if (myid == 0)
   {
      cout << "Size of fine grid system: "
           << 2.0 * A_EE->GetGlobalNumRows() << " x " << 2.0* A_EE->GetGlobalNumCols() << endl;
   }

   // Set up the preconditioner
   Array2D<HypreParMatrix*> blockA(2,2);
   blockA(0,0) = A_EE;
   blockA(0,1) = A_EH;
   blockA(1,0) = A_HE;
   blockA(1,1) = A_HH;

   Block_AMSSolver * blkAMS;
   blkAMS = new Block_AMSSolver(block_trueOffsets, fespace);
   blkAMS->SetOperator(blockA);
   blkAMS->SetTheta(1.0);
   //0-Smoother, 1-Grad, 2,3,4-Pix,Piy,Piz
   blkAMS->SetCycleType("023414320");
   // blkAMS->SetCycleType("000000000023414320000000000");
   blkAMS->SetNumberofCycles(1);
   // blkAMS->SetCycleType("012343210");


   int maxit(500);
   double rtol(1.e-6);
   double atol(0.0);
   trueX = 0.0;

   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetAbsTol(atol);
   pcg.SetRelTol(rtol);
   pcg.SetMaxIter(maxit);
   pcg.SetPreconditioner(*blkAMS);
   pcg.SetOperator(*LS_Maxwellop);
   pcg.SetPrintLevel(1);
   pcg.Mult(trueRhs, trueX);


   if (myid == 0)
   {
      cout << "PCG with Block AMS finished" << endl;
   }

   *E_gf = 0.0;
   *H_gf = 0.0;

   E_gf->Distribute(&(trueX.GetBlock(0)));
   H_gf->Distribute(&(trueX.GetBlock(1)));

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double Error_E = E_gf->ComputeL2Error(Eex, irs);
   double norm_E = ComputeGlobalLpNorm(2, Eex, *pmesh, irs);

   double Error_H = H_gf->ComputeL2Error(Hex, irs);
   double norm_H = ComputeGlobalLpNorm(2, Hex , *pmesh, irs);

   if (myid == 0)
   {
      // cout << "|| E_h - E || / || E || = " << Error_E / norm_E << "\n";
      // cout << "|| H_h - H || / || H || = " << Error_H / norm_H << "\n";
      cout << "|| E_h - E || = " << Error_E  << "\n";
      cout << "|| H_h - H || = " << Error_H  << "\n";

      cout << "Total error = " << sqrt(Error_H*Error_H+Error_E*Error_E) << "\n";

      // cout << "Total Relative error = " <<  Error_E / norm_E + Error_H / norm_H  << "\n";
      // cout << "E Relative error = " <<  Error_E / norm_E  << "\n";
      // cout << "H Relative error = " <<  Error_H / norm_H  << "\n";

      // cout << "|| E || = " << norm_E  << "\n";
      // cout << "|| H || = " << norm_H  << "\n";
   }

   if (visualization)
   {
      // ParGridFunction * Eex_gf = new ParGridFunction;   
      // Eex_gf->MakeRef(fespace, x.GetBlock(0));
      // Eex_gf->ProjectCoefficient(Eex);
      // ParGridFunction * Hex_gf = new ParGridFunction;   
      // Hex_gf->MakeRef(fespace, x.GetBlock(1));
      // Hex_gf->ProjectCoefficient(Hex);

      // 8. Connect to GLVis.
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream E_sock(vishost, visport);
      E_sock << "parallel " << num_procs << " " << myid << "\n";
      E_sock.precision(8);
      E_sock << "solution\n" << *pmesh << *E_gf << "window_title 'Electric field'" << endl;
      // MPI_Barrier(pmesh->GetComm());
      // socketstream Eex_sock(vishost, visport);
      // Eex_sock << "parallel " << num_procs << " " << myid << "\n";
      // Eex_sock.precision(8);
      // Eex_sock << "solution\n" << *pmesh << *Eex_gf << "window_title 'Exact Electric Field'" << endl;
      MPI_Barrier(pmesh->GetComm());
      socketstream H_sock(vishost, visport);
      H_sock << "parallel " << num_procs << " " << myid << "\n";
      H_sock.precision(8);
      H_sock << "solution\n" << *pmesh << *H_gf << "window_title 'Magnetic field'" << endl;
      // MPI_Barrier(pmesh->GetComm());
      // socketstream Hex_sock(vishost, visport);
      // Hex_sock << "parallel " << num_procs << " " << myid << "\n";
      // Hex_sock.precision(8);
      // Hex_sock << "solution\n" << *pmesh << *Hex_gf << "window_title 'Exact Magnetic field'" << endl;
   }


   delete a_EE;
   delete a_HE;
   delete a_HH;
   delete b_E;
   delete b_H;
   delete fec;
   delete fespace;
   delete pmesh;
   MPI_Finalize();
   return 0;
}


//define exact solution
void E_exact(const Vector &x, Vector &E)
{
   double curlE[3], curl2E[3];
   get_maxwell_solution(x, E, curlE, curl2E);
}

void H_exact(const Vector &x, Vector &H)
{
   double E[3], curlE[3], curl2E[3];
   get_maxwell_solution(x, E, curlE, curl2E);
   for (int i = 0; i<3; i++) {H(i) = curlE[i]/omega;}
}

//calculate RHS from exact solution
void f_exact_E(const Vector &x, Vector &f)
{
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   // curl E - omega H = 0 
   f(0) = curlE[0] - omega * (curlE[0]/omega); // = 0
   f(1) = curlE[1] - omega * (curlE[1]/omega); // = 0
   f(2) = curlE[2] - omega * (curlE[2]/omega); // = 0
}

void f_exact_H(const Vector &x, Vector &f)
{
   if (dim != 3)
   {
      cout << "2D not set up yet: " << endl;
      exit(0);
   }
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   // curl H - omega E = f 
   // = curl (curl E / omega) - omega E 
   f(0) = curl2E[0]/omega - omega * E[0];
   f(1) = curl2E[1]/omega - omega * E[1];
   f(2) = curl2E[2]/omega - omega * E[2];
}

void scaledf_exact_E(const Vector &x, Vector &f)
{
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   //  - omega *( curl E - omega H) = 0 
   f(0) =-omega * (curlE[0] - omega * (curlE[0]/omega)); // = 0
   f(1) =-omega * (curlE[1] - omega * (curlE[1]/omega)); // = 0
   f(2) =-omega * (curlE[2] - omega * (curlE[2]/omega)); // = 0
}

void scaledf_exact_H(const Vector &x, Vector &f)
{
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   // curl H - omega E = f 
   // = - omega *( curl (curl E / omega) - omega E) 
   
   f(0) = -omega * (curl2E[0]/omega - omega * E[0]);
   f(1) = -omega * (curl2E[1]/omega - omega * E[1]);
   f(2) = -omega * (curl2E[2]/omega - omega * E[2]);
}

void get_maxwell_solution(const Vector & X, double E[], double curlE[], double curl2E[])
{
   const double x = X[0];
   const double y = X[1];
   const double z = X[2];

   if (isol == 0) // polynomial
   {
      // Polynomial vanishing on the boundary
      E[0] = y * z * (1.0 - y) * (1.0 - z);
      E[1] = (1.0 - x) * x * y * (1.0 - z) * z;
      E[2] = (1.0 - x) * x * (1.0 - y) * y;
      //

      curlE[0] = -(-1.0 + x) * x * (1.0 + y * (-3.0 + 2.0 * z));
      curlE[1] = -2.0 * (-1.0 + y) * y * (x - z);
      curlE[2] = (1.0 + (-3.0 + 2.0 * x) * y) * (-1.0 + z) * z; 

      curl2E[0] = -2.0 * (-1.0 + y) * y + (-3.0 + 2.0 * x) * (-1.0 + z) * z;
      curl2E[1] = -2.0 * y * (-x + x*x + (-1.0 + z) * z);
      curl2E[2] = -2.0 * (-1.0 + y) * y + (-1.0 + x) * x * (-3.0 + 2.0 * z);

   }

   else if (isol == 1) // sinusoidal
   {
      E[0] = sin(omega * y);
      E[1] = sin(omega * z);
      E[2] = sin(omega * x);

      curlE[0] = -omega * cos(omega * z);
      curlE[1] = -omega * cos(omega * x);;
      curlE[2] = -omega * cos(omega * y);; 

      curl2E[0] = omega * omega * E[0];
      curl2E[1] = omega * omega * E[1];
      curl2E[2] = omega * omega * E[2];

   }
   else if (isol == 2) //simple polynomial
   {
      E[0] = y;
      E[1] = z;
      E[2] = x;

      curlE[0] = -1.0;
      curlE[1] = -1.0;
      curlE[2] = -1.0; 

      curl2E[0] =0.0;
      curl2E[1] =0.0;
      curl2E[2] =0.0;
   }
      else if (isol == 4) //constant
   {
      E[0] = 1.0;
      E[1] = 1.0;
      E[2] = 1.0;

      curlE[0] = 0.0;
      curlE[1] = 0.0;
      curlE[2] = 0.0; 

      curl2E[0] =0.0;
      curl2E[1] =0.0;
      curl2E[2] =0.0;
   }
   else if (isol == 3) // plane wave
   {
      double coeff = omega / sqrt(3.0);
      E[0] = cos(coeff * (x + y + z));
      E[1] = 0.0;
      E[2] = 0.0;


      curlE[0] = 0.0;
      curlE[1] = -coeff * sin(coeff * (x+y+z));
      curlE[2] = coeff * sin(coeff * (x+y+z)); 

      curl2E[0] = 2.0 * coeff * coeff * E[0];
      curl2E[1] = -coeff * coeff * E[0];
      curl2E[2] = -coeff * coeff * E[0];
   }

}

