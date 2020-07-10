
#include "UMFPackC.hpp"

void ComplexUMFPackSolver::Init()
{
   mat = NULL;
   Numeric = NULL;
   AI = AJ = NULL;
   if (!use_long_ints)
   {
      umfpack_zi_defaults(Control);
   }
   else
   {
      umfpack_zl_defaults(Control);
   }
}

void ComplexUMFPackSolver::SetOperator(const Operator &op)
{
   int *Ap, *Ai;
   void *Symbolic;
   double *Ax;
   double *Az;

   if (Numeric)
   {
      if (!use_long_ints)
      {
         umfpack_zi_free_numeric(&Numeric);
      }
      else
      {
         umfpack_zl_free_numeric(&Numeric);
      }
   }

   mat = const_cast<ComplexSparseMatrix *>(dynamic_cast<const ComplexSparseMatrix *>(&op));
   MFEM_VERIFY(mat, "not a ComplexSparseMatrix");

   MFEM_VERIFY(mat->real().NumNonZeroElems() == mat->imag().NumNonZeroElems(),
      "Real and imag Sparsity patter missmatch: Try setting Assemble (skip_zeros = 0)");

   // UMFPack requires that the column-indices in mat corresponding to each
   // row be sorted.
   // Generally, this will modify the ordering of the entries of mat.

   mat->real().SortColumnIndices();
   mat->imag().SortColumnIndices();

   height = mat->real().Height();
   width = mat->real().Width();
   MFEM_VERIFY(width == height, "not a square matrix");

   Ap = mat->real().GetI(); // assuming real and imag have the same sparsity
   Ai = mat->real().GetJ();
   Ax = mat->real().GetData();
   Az = mat->imag().GetData();

   if (!use_long_ints)
   {
      int status = umfpack_zi_symbolic(width,width,Ap,Ai,Ax,Az,&Symbolic,
                                       Control,Info);     
      if (status < 0)
      {
         umfpack_zi_report_info(Control, Info);
         umfpack_zi_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::SetOperator :"
                    " umfpack_zi_symbolic() failed!");
      }

      status = umfpack_zi_numeric(Ap, Ai, Ax, Az, Symbolic, &Numeric,
                                  Control, Info);     
      if (status < 0)
      {
         umfpack_zi_report_info(Control, Info);
         umfpack_zi_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::SetOperator :"
                    " umfpack_zi_numeric() failed!");
      }
      umfpack_zi_free_symbolic(&Symbolic);
   }
   else
   {
      SuiteSparse_long status;

      delete [] AJ;
      delete [] AI;
      AI = new SuiteSparse_long[width + 1];
      AJ = new SuiteSparse_long[Ap[width]];
      for (int i = 0; i <= width; i++)
      {
         AI[i] = (SuiteSparse_long)(Ap[i]);
      }
      for (int i = 0; i < Ap[width]; i++)
      {
         AJ[i] = (SuiteSparse_long)(Ai[i]);
      }

      status = umfpack_zl_symbolic(width, width, AI, AJ, Ax, Az, &Symbolic,
                                   Control, Info);                           
      if (status < 0)
      {
         umfpack_zl_report_info(Control, Info);
         umfpack_zl_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::SetOperator :"
                    " umfpack_zl_symbolic() failed!");
      }

      status = umfpack_zl_numeric(AI, AJ, Ax, Az, Symbolic, &Numeric,
                                  Control, Info);
      if (status < 0)
      {
         umfpack_zl_report_info(Control, Info);
         umfpack_zl_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::SetOperator :"
                    " umfpack_zl_numeric() failed!");
      }
      umfpack_zl_free_symbolic(&Symbolic);
   }
}

void ComplexUMFPackSolver::Mult(const Vector &b, Vector &x) const
{
   if (mat == NULL)
      mfem_error("ComplexUMFPackSolver::Mult : matrix is not set!"
                 " Call SetOperator first!");
   int n = b.Size()/2;
   double * datax = x.GetData();
   double * datab = b.GetData();

   // For the Block Symmetric case data the imaginary part 
   // have to be scaled by -1 
   ComplexOperator::Convention conv = mat->GetConvention();
   Vector bimag;
   if (conv == ComplexOperator::Convention::BLOCK_SYMMETRIC)
   {
      bimag.SetDataAndSize(&datab[n],n);
      bimag *=-1.0;
   }

   //Solve the transpose, since UMFPack expects CCS instead of CRS format
   if (!use_long_ints)
   {
      //
      int status =
         umfpack_zi_solve(UMFPACK_Aat, mat->real().GetI(), mat->real().GetJ(),
                          mat->real().GetData(), mat->imag().GetData(), 
                          datax, &datax[n], datab, &datab[n], Numeric, Control, Info);
      umfpack_zi_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_zi_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::Mult : umfpack_zi_solve() failed!");
      }
   }
   else
   {
      SuiteSparse_long status =
         umfpack_zl_solve(UMFPACK_Aat,AI,AJ,mat->real().GetData(),
         mat->imag().GetData(),
         datax,&datax[n],datab,&datab[n],Numeric,Control,Info);    

      umfpack_zl_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_zl_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::Mult : umfpack_zl_solve() failed!");
      }
   }

   if (conv == ComplexOperator::Convention::BLOCK_SYMMETRIC)
   {
      bimag *=-1.0;
   }

}

void ComplexUMFPackSolver::MultTranspose(const Vector &b, Vector &x) const
{
   if (mat == NULL)
      mfem_error("ComplexUMFPackSolver::Mult : matrix is not set!"
                 " Call SetOperator first!");
   int n = b.Size()/2;
   double * datax = x.GetData();
   double * datab = b.GetData();

   ComplexOperator::Convention conv = mat->GetConvention();
   Vector bimag;
   bimag.SetDataAndSize(&datab[n],n);
   //To solve the Adjoint A^H x = b by solving 
   // the conjugate problem A^T \bar{x} = \bar{b}
   if ((!transpose && conv == ComplexOperator::HERMITIAN) ||
       ( transpose && conv == ComplexOperator::BLOCK_SYMMETRIC))
   {
      bimag *=-1.0;
   }
   
   if (!use_long_ints)
   {
      //
      int status =
         umfpack_zi_solve(UMFPACK_A, mat->real().GetI(), mat->real().GetJ(),
                          mat->real().GetData(), mat->imag().GetData(), 
                          datax, &datax[n], datab, &datab[n], Numeric, Control, Info);
      umfpack_zi_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_zi_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::Mult : umfpack_zi_solve() failed!");
      }
   }
   else
   {
      SuiteSparse_long status =
         umfpack_zl_solve(UMFPACK_A,AI,AJ,mat->real().GetData(),
         mat->imag().GetData(),
         datax,&datax[n],datab,&datab[n],Numeric,Control,Info);    

      umfpack_zl_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_zl_report_status(Control, status);
         mfem_error("ComplexUMFPackSolver::Mult : umfpack_zl_solve() failed!");
      }
   }
   if (!transpose)
   {
      Vector ximag;
      ximag.SetDataAndSize(&datax[n],n);
      ximag *=-1.0;
   }
   if ((!transpose && conv == ComplexOperator::HERMITIAN) ||
       ( transpose && conv == ComplexOperator::BLOCK_SYMMETRIC))
   {
      bimag *=-1.0;
   }
}


ComplexUMFPackSolver::~ComplexUMFPackSolver()
{
   delete [] AJ;
   delete [] AI;
   if (Numeric)
   {
      if (!use_long_ints)
      {
         umfpack_zi_free_numeric(&Numeric);
      }
      else
      {
         umfpack_zl_free_numeric(&Numeric);
      }
   }
}
