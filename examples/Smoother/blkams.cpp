//                                MFEM Example multigrid-grid Cycle
//
// Compile with: make mg_maxwellp
//
// Sample runs:  mg_maxwellp -m ../data/one-hex.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "blkams.hpp"

using namespace std;
using namespace mfem;

Block_AMSSolver::Block_AMSSolver(Array<int> offsets_, std::vector<ParFiniteElementSpace *> fespaces_)
      : fespaces(fespaces_), offsets(offsets_), offsetsG(3), offsetsPi(3)
{
   nrmeshes = fespaces.size();
   Grad = GetDiscreteGradientOp(fespaces[nrmeshes-1]);
   Pi = GetNDInterpolationOp(fespaces[nrmeshes-1]);
   Pix = Pi(0,0);
   Piy = Pi(0,1);
   Piz = Pi(0,2);
   
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


void Block_AMSSolver::SetOperator(Array2D<HypreParMatrix*> Op)
{
   A_array = Op;
   if (sType == Block_AMS::BlkSmootherType::SCHWARZ)
   {
      // int lvls = std::min(1,nrmeshes);
      // int lvls = nrmeshes;
      int lvls = 1;
      D = new BlkParSchwarzSmoother(fespaces[nrmeshes-lvls]->GetParMesh(),lvls-1,fespaces[nrmeshes-1],Op);
      // dynamic_cast<BlkParSchwarzSmoother *>(D)->SetDumpingParam(1.0);
      // dynamic_cast<BlkParSchwarzSmoother *>(D)->SetNumSmoothSteps(1);
   }
   else
   {
      l1A00 = new HypreParMatrix(*A_array(0,0));
      l1A11 = new HypreParMatrix(*A_array(1,1));
      // DiagAddL1norm();
      HypreSmoother * D_00 = new HypreSmoother;  
      // D_00->SetType(HypreSmoother::l1GS);
      D_00->SetType(HypreSmoother::Jacobi);
      D_00->SetOperator(*l1A00);
      HypreSmoother * D_11 = new HypreSmoother;  
      // D_11->SetType(HypreSmoother::l1GS);
      D_11->SetType(HypreSmoother::Jacobi);
      D_11->SetOperator(*l1A11);
      D = new BlockOperator(offsets);
      static_cast<BlockOperator *>(D)->SetDiagonalBlock(0, D_00);
      static_cast<BlockOperator *>(D)->SetDiagonalBlock(1, D_11);
   }
   SetOperators();
}

void Block_AMSSolver::SetOperators()
{
   hRAP_G.SetSize(2,2);
   hRAP_Px.SetSize(2,2);
   hRAP_Py.SetSize(2,2);
   hRAP_Pz.SetSize(2,2);
   for (int i=0; i<2 ; i++)
   {
      A->SetBlock(i,i,A_array(i,i));
      G->SetBlock(i,i,Grad);
      Px->SetBlock(i,i,Pix);
      Py->SetBlock(i,i,Piy);
      Pz->SetBlock(i,i,Piz);
      for (int j=0; j<2 ; j++)
      {
         A->SetBlock(i,j,A_array(i,j));
         // hRAP_G(i,j) =  RAP(A_array(i,j),Grad);
         // hRAP_Px(i,j) =  RAP(A_array(i,j),Pix);
         // hRAP_Py(i,j) =  RAP(A_array(i,j),Piy);
         // hRAP_Pz(i,j) =  RAP(A_array(i,j),Piz);

         // GtAG->SetBlock(i,j,hRAP_G(i,j));
         // PxtAPx->SetBlock(i,j,hRAP_Px(i,j));
         // PytAPy->SetBlock(i,j,hRAP_Py(i,j));
         // PztAPz->SetBlock(i,j,hRAP_Pz(i,j));
         GtAG->SetBlock(i,j,RAP(A_array(i,j),Grad));
         PxtAPx->SetBlock(i,j,RAP(A_array(i,j),Pix));
         PytAPy->SetBlock(i,j,RAP(A_array(i,j),Piy));
         PztAPz->SetBlock(i,j,RAP(A_array(i,j),Piz));
      }
   }

   // create hypre matrices from blocks
   // Array2D<double> coefficients(2,2);
   // coefficients(0,0) = 1.0;  coefficients(0,1) = 1.0;
   // coefficients(1,0) = 1.0;  coefficients(1,1) = 1.0;
   // hGtAG = CreateHypreParMatrixFromBlocks(MPI_COMM_WORLD,offsetsG,hRAP_G,coefficients);
   // hPxtAPx = CreateHypreParMatrixFromBlocks(MPI_COMM_WORLD,offsetsPi,hRAP_Px,coefficients);
   // hPytAPy = CreateHypreParMatrixFromBlocks(MPI_COMM_WORLD,offsetsPi,hRAP_Py,coefficients);
   // hPztAPz = CreateHypreParMatrixFromBlocks(MPI_COMM_WORLD,offsetsPi,hRAP_Pz,coefficients);


   // AMG_G = new HypreBoomerAMG(*hGtAG); AMG_G->SetPrintLevel(0);
   // AMG_Px = new HypreBoomerAMG(*hPxtAPx); AMG_Px->SetPrintLevel(0);
   // AMG_Py = new HypreBoomerAMG(*hPytAPy); AMG_Py->SetPrintLevel(0);
   // AMG_Pz = new HypreBoomerAMG(*hPztAPz); AMG_Pz->SetPrintLevel(0);

   for (int i=0; i<2 ; i++)
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

void Block_AMSSolver::SetTheta(const double a) {theta = a;}
void Block_AMSSolver::SetCycleType(const string c_type) {cycle_type = c_type;}
void Block_AMSSolver::SetNumberofCycles(const int k) {NumberOfCycles = k;}

void Block_AMSSolver::DiagAddL1norm()
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

void Block_AMSSolver::Getrowl1norm(HypreParMatrix *A , Vector &l1norm)
{
   // First cast as hypre_ParCSRMatrix
   hypre_ParCSRMatrix * Ah = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*A);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(Ah);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(Ah);
   HYPRE_Int *A_diag_I = hypre_CSRMatrixI(A_diag);
   // HYPRE_Int *A_diag_J = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(Ah);
   HYPRE_Int *A_offd_I = hypre_CSRMatrixI(A_offd);
   // HYPRE_Int *A_offd_J = hypre_CSRMatrixJ(A_offd);
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

void Block_AMSSolver::Mult(const Vector &r, Vector &z) const
{
   int n = r.Size();
   int m = A->Height();
   // int Numit = 0;
   if (n != m ) {cout << "Size inconsistency" << endl;}

   Vector res(n), raux(n),zaux(n);
   //initialization
   res = r; z = 0.0;
   Array<BlockOperator *> Tr_v(4);
   Array<BlockOperator *> PtAP_v(4);
   Array<BlockDiagonalPreconditioner *> blkAMG_v(4);

   // Array<HypreParMatrix *> hPtAP_v(4);
   // Array<HypreBoomerAMG *> AMG_v(4);

   // hPtAP_v[0]  = hGtAG;    hPtAP_v[1]  = hPxtAPx;   hPtAP_v[2]  = hPytAPy;   hPtAP_v[3]  = hPztAPz;
   // AMG_v[0]    = AMG_G;    AMG_v[1]    = AMG_Px;    AMG_v[2]    = AMG_Py;    AMG_v[3]    = AMG_Pz;  


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
            // GetCorrection(Tr_v[i-1], hPtAP_v[i-1], AMG_v[i-1], res, zaux);
         }
         z +=zaux; 
         A->Mult(zaux,raux); res -=raux;
      }
   }
}

void Block_AMSSolver::GetCorrection(BlockOperator* Tr, BlockOperator* op, BlockDiagonalPreconditioner *prec, Vector &r, Vector &z) const
{
   int k = Tr->Width();
   Vector raux(k), zaux(k);
   // Map trough the Transpose of the Transfer operator
   Tr->MultTranspose(r,raux);
   zaux = 0.0;

   int maxit(20);
   double rtol(0.0);
   double atol(1e-6);
   
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(atol);
   cg.SetRelTol(rtol);
   cg.SetMaxIter(maxit);
   cg.SetOperator(*op);
   cg.SetPreconditioner(*prec);
   cg.SetPrintLevel(-1);
   cg.Mult(raux, zaux);
      // prec->Mult(raux,zaux);

   // Map back to the original space through the Tranfer operator
   Tr->Mult(zaux, z);
}


void Block_AMSSolver::GetCorrection(BlockOperator* Tr, HypreParMatrix* op, HypreBoomerAMG *prec, Vector &r, Vector &z) const
{
   int k = Tr->Width();
   Vector raux(k), zaux(k);
   // Map trough the Transpose of the Transfer operator
   Tr->MultTranspose(r,raux);
   zaux = 0.0;

   int maxit(100);
   double rtol(0.0);
   double atol(1e-6);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(atol);
   cg.SetRelTol(rtol);
   cg.SetMaxIter(maxit);
   cg.SetOperator(*op);
   cg.SetPreconditioner(*prec);
   cg.SetPrintLevel(-1);
   cg.Mult(raux, zaux);
      // prec->Mult(raux,zaux);

   // Map back to the original space through the Tranfer operator
   Tr->Mult(zaux, z);
}


Block_AMSSolver::~Block_AMSSolver()
{
   delete Grad;
   delete Pix;
   delete Piy;
   delete Piz;
   delete G;
   delete Px;
   delete Py;
   delete Pz;
   delete GtAG;
   delete PxtAPx;
   delete PytAPy;
   delete PztAPz;
   delete A;
   delete blkAMG_G; 
   delete blkAMG_Px;
   delete blkAMG_Py;
   delete blkAMG_Pz;
}

// Discrete gradient matrix
HypreParMatrix* GetDiscreteGradientOp(ParFiniteElementSpace *fespace)
{
   int dim = fespace->GetMesh()->Dimension();
   // int sdim = fespace->GetMesh()->SpaceDimension();
   // const FiniteElementCollection *fec = fespace->FEColl();
   int p = 1;
   if (fespace->GetNE() > 0)
   {
      p = fespace->GetOrder(0);
   }
   ParMesh *pmesh = fespace->GetParMesh();
   FiniteElementCollection *vert_fec;
   vert_fec = new H1_FECollection(p, dim);
   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh,vert_fec);
   // generate and set the discrete gradient
   ParDiscreteLinearOperator *grad;
   grad = new ParDiscreteLinearOperator(vert_fespace, fespace);
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   HypreParMatrix *G;
   G = grad->ParallelAssemble();
   delete vert_fespace;
   delete grad;
   return G;
}


// Discrete gradient matrix
Array2D<HypreParMatrix *> GetNDInterpolationOp(ParFiniteElementSpace *fespace)
{
   int dim = fespace->GetMesh()->Dimension();
   int sdim = fespace->GetMesh()->SpaceDimension();
   // const FiniteElementCollection *fec = fespace->FEColl();
   int p = 1;
   if (fespace->GetNE() > 0)
   {
      p = fespace->GetOrder(0);
   }
   ParMesh *pmesh = fespace->GetParMesh();
   FiniteElementCollection *vert_fec;
   vert_fec = new H1_FECollection(p, dim);
   
   Array2D<HypreParMatrix *> Pi_blocks;
   ParFiniteElementSpace *vert_fespace_d
         = new ParFiniteElementSpace(pmesh, vert_fec, sdim, Ordering::byVDIM);
   ParDiscreteLinearOperator *id_ND;
   id_ND = new ParDiscreteLinearOperator(vert_fespace_d, fespace);
   id_ND->AddDomainInterpolator(new IdentityInterpolator);
   id_ND->Assemble();
   id_ND->Finalize();
   id_ND->GetParBlocks(Pi_blocks);
   //
   delete id_ND;
   delete vert_fespace_d;
   delete vert_fec;
   return Pi_blocks;
}
