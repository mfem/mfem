// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor_batched.hpp"
#include "lor_restriction.hpp"
#include "../../general/forall.hpp"

// Specializations
#include "lor_diffusion.hpp"

namespace mfem
{

template<int D1D, int Q1D>
void NodalInterpolation2D(const int NE,
                          const Vector &localL,
                          Vector &localH,
                          const Array<double> &B);
template<int D1D, int Q1D>
void NodalInterpolation3D(const int NE,
                          const Vector &localL,
                          Vector &localH,
                          const Array<double> &B);

template <typename T>
bool HasIntegrator(BilinearForm &a)
{
   Array<BilinearFormIntegrator*> *integs = a.GetDBFI();
   if (integs != NULL && integs->Size() == 1)
   {
      BilinearFormIntegrator *i = (*integs)[0];
      if (dynamic_cast<T*>(i))
      {
         return true;
      }
   }
   return false;
}

template <typename T1, typename T2>
bool HasIntegrators(BilinearForm &a)
{
   Array<BilinearFormIntegrator*> *integs = a.GetDBFI();
   if (integs != NULL && integs->Size() == 2)
   {
      BilinearFormIntegrator *i0 = (*integs)[0];
      BilinearFormIntegrator *i1 = (*integs)[1];

      if ((dynamic_cast<T1*>(i0) && dynamic_cast<T2*>(i1)) ||
          (dynamic_cast<T2*>(i0) && dynamic_cast<T1*>(i1)))
      {
         return true;
      }
   }
   return false;
}

template <int Q1D>
void BatchedLORAssembly::GetLORVertexCoordinates()
{
   Mesh &mesh_ho = *fes_ho.GetMesh();
   mesh_ho.EnsureNodes();

   // Get nodal points at the LOR vertices
   const int dim = mesh_ho.Dimension();
   const int nel_ho = mesh_ho.GetNE();
   const int order = fes_ho.GetMaxElementOrder();
   const int nd1d = order + 1;
   const int ndof_per_el = pow(nd1d, dim);

   MFEM_VERIFY(nd1d == Q1D, "Bad template instantiation.");

   X_vert.SetSize(dim*ndof_per_el*nel_ho);
   X_vert.UseDevice(true);

   const GridFunction *nodal_gf = mesh_ho.GetNodes();
   const FiniteElementSpace *nodal_fes = nodal_gf->FESpace();
   const Operator *nodal_restriction = nodal_fes->GetElementRestriction(
                                          ElementDofOrdering::LEXICOGRAPHIC);
   const int nodal_nd1d = nodal_fes->GetMaxElementOrder() + 1;

   // Map from nodal E-vector to L-vector
   Vector nodes_loc(nodal_restriction->Height());
   nodes_loc.UseDevice(true);
   nodal_restriction->Mult(*nodal_gf, nodes_loc);

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   Geometry::Type geom = mesh_ho.GetElementGeometry(0);
   const IntegrationRule &ir = irs.Get(geom, 2*nd1d - 3);
   const DofToQuad& maps =
      nodal_fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);

   if (dim == 2)
   {
      switch (nodal_nd1d)
      {
         case 2: NodalInterpolation2D<2,Q1D>(nel_ho, nodes_loc, X_vert, maps.B); break;
         case 4: NodalInterpolation2D<4,Q1D>(nel_ho, nodes_loc, X_vert, maps.B); break;
         case 6: NodalInterpolation2D<6,Q1D>(nel_ho, nodes_loc, X_vert, maps.B); break;
         default: MFEM_ABORT("Unsuported mesh order!");
      }
   }
   else if (dim == 3)
   {
      switch (nodal_nd1d)
      {
         case 2: NodalInterpolation3D<2,Q1D>(nel_ho, nodes_loc, X_vert, maps.B); break;
         case 4: NodalInterpolation3D<4,Q1D>(nel_ho, nodes_loc, X_vert, maps.B); break;
         case 6: NodalInterpolation3D<6,Q1D>(nel_ho, nodes_loc, X_vert, maps.B); break;
         default: MFEM_ABORT("Unsuported mesh order!");
      }
   }
   else
   {
      MFEM_ABORT("Not supported.");
   }
}

bool BatchedLORAssembly::FormIsSupported(BilinearForm &a)
{
   // We want to support the following configurations:
   // H1, ND, and RT spaces: M, A, M + K
   if (HasIntegrator<DiffusionIntegrator>(a))
   {
      return true;
   }
   return false;
}

SparseMatrix *BatchedLORAssembly::AssembleWithoutBC()
{
   MFEM_VERIFY(UsesTensorBasis(fes_ho),
               "Batched LOR assembly requires tensor basis");

   // Set up the sparsity pattern for the matrix
   const int vsize = fes_ho.GetVSize();
   SparseMatrix *A = new SparseMatrix(vsize, vsize, 0);
   A->GetMemoryI().New(A->Height()+1, A->GetMemoryI().GetMemoryType());
   const int nnz = R.FillI(*A);
   A->GetMemoryJ().New(nnz, A->GetMemoryJ().GetMemoryType());
   A->GetMemoryData().New(nnz, A->GetMemoryData().GetMemoryType());
   R.FillJAndZeroData(*A); // J, A = 0.0

   // Get the LOR vertex coordinates
   const int order = fes_ho.GetMaxElementOrder();
   const int nd1d = order + 1;
   switch (nd1d)
   {
      case 2: GetLORVertexCoordinates<2>(); break;
      case 3: GetLORVertexCoordinates<3>(); break;
      case 4: GetLORVertexCoordinates<4>(); break;
      case 5: GetLORVertexCoordinates<5>(); break;
      case 6: GetLORVertexCoordinates<6>(); break;
      default: MFEM_ABORT("Unsupported order!")
   }

   // Assemble the matrix, using kernels from the derived classes
   AssemblyKernel(*A);
   A->Finalize();
   return A;
}

#ifdef MFEM_USE_MPI
void BatchedLORAssembly::ParAssemble(OperatorHandle &A)
{
   // Assemble the system matrix local to this partition
   SparseMatrix *A_local = AssembleWithoutBC();

   ParFiniteElementSpace *pfes_ho =
      dynamic_cast<ParFiniteElementSpace*>(&fes_ho);
   MFEM_VERIFY(pfes_ho != nullptr,
               "ParAssemble must be called with ParFiniteElementSpace");

   // Create a block diagonal parallel matrix
   OperatorHandle A_diag(Operator::Hypre_ParCSR);
   A_diag.MakeSquareBlockDiag(pfes_ho->GetComm(),
                              pfes_ho->GlobalVSize(),
                              pfes_ho->GetDofOffsets(),
                              A_local);

   // Parallel matrix assembly using P^t A P
   OperatorHandle P(Operator::Hypre_ParCSR);
   P.ConvertFrom(pfes_ho->Dof_TrueDof_Matrix());
   A.MakePtAP(A_diag, P);

   // Eliminate the boundary conditions
   HypreParMatrix *A_mat = A.As<HypreParMatrix>();
   hypre_ParCSRMatrix *A_hypre = *A_mat;
   A_mat->HypreReadWrite();

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A_hypre);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A_hypre);

   HYPRE_Int diag_nrows = hypre_CSRMatrixNumRows(diag);
   HYPRE_Int offd_ncols = hypre_CSRMatrixNumCols(offd);

   const int n_ess_dofs = ess_dofs.Size();
   const auto ess_dofs_d = ess_dofs.Read();

   // Start communication to figure out which columns need to be eliminated in
   // the off-diagonal block
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int *int_buf_data, *eliminate_row, *eliminate_col;
   {
      eliminate_row = hypre_CTAlloc(HYPRE_Int, diag_nrows, HYPRE_MEMORY_HOST);
      eliminate_col = hypre_CTAlloc(HYPRE_Int, offd_ncols, HYPRE_MEMORY_HOST);

      // Make sure A has a communication package
      hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A_hypre);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
      }

      // Which of the local rows are to be eliminated?
      for (int i = 0; i < diag_nrows; i++)
      {
         eliminate_row[i] = 0;
      }

      ess_dofs.HostRead();
      for (int i = 0; i < n_ess_dofs; i++)
      {
         eliminate_row[ess_dofs[i]] = 1;
      }

      // Use a matvec communication pattern to find (in eliminate_col) which of
      // the local offd columns are to be eliminated
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data =
         hypre_CTAlloc(HYPRE_Int,
                       hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                       HYPRE_MEMORY_HOST);
      int index = 0;
      for (int i = 0; i < num_sends; i++)
      {
         int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (int j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            int k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            int_buf_data[index++] = eliminate_row[k];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(
                       11, comm_pkg, int_buf_data, eliminate_col);
   }

   // Eliminate rows and columns in the diagonal block
   {
      const auto I = diag->i;
      const auto J = diag->j;
      auto data = diag->data;

      MFEM_FORALL(i, n_ess_dofs,
      {
         const int idof = ess_dofs_d[i];
         for (int j=I[idof]; j<I[idof+1]; ++j)
         {
            const int jdof = J[j];
            if (jdof == idof)
            {
               // Set eliminate diagonal equal to identity
               data[j] = 1.0;
            }
            else
            {
               data[j] = 0.0;
               for (int k=I[jdof]; k<I[jdof+1]; ++k)
               {
                  if (J[k] == idof)
                  {
                     data[k] = 0.0;
                     break;
                  }
               }
            }
         }
      });
   }

   // Eliminate rows in the off-diagonal block
   {
      const auto I = offd->i;
      auto data = offd->data;
      MFEM_FORALL(i, n_ess_dofs,
      {
         const int idof = ess_dofs_d[i];
         for (int j=I[idof]; j<I[idof+1]; ++j)
         {
            data[j] = 0.0;
         }
      });
   }

   // Wait for MPI communication to finish
   Array<HYPRE_Int> cols_to_eliminate;
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);

      // set the array cols_to_eliminate
      int ncols_to_eliminate = 0;
      for (int i = 0; i < offd_ncols; i++)
      {
         if (eliminate_col[i]) { ncols_to_eliminate++; }
      }

      cols_to_eliminate.SetSize(ncols_to_eliminate);
      cols_to_eliminate = 0.0;

      ncols_to_eliminate = 0;
      for (int i = 0; i < offd_ncols; i++)
      {
         if (eliminate_col[i])
         {
            cols_to_eliminate[ncols_to_eliminate++] = i;
         }
      }

      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
      hypre_TFree(eliminate_row, HYPRE_MEMORY_HOST);
      hypre_TFree(eliminate_col, HYPRE_MEMORY_HOST);
   }

   // Eliminate columns in the off-diagonal block
   {
      const int ncols_to_eliminate = cols_to_eliminate.Size();
      const int nrows_offd = hypre_CSRMatrixNumRows(offd);
      const auto cols = cols_to_eliminate.Read();
      const auto I = offd->i;
      const auto J = offd->j;
      auto data = offd->data;
      // Note: could also try a different strategy, looping over nnz in the
      // matrix and then doing a binary search in ncols_to_eliminate to see if
      // the column should be eliminated.
      MFEM_FORALL(idx, ncols_to_eliminate,
      {
         const int j = cols[idx];
         for (int i=0; i<nrows_offd; ++i)
         {
            for (int jj=I[i]; jj<I[i+1]; ++jj)
            {
               if (J[jj] == j)
               {
                  data[jj] = 0.0;
                  break;
               }
            }
         }
      });
   }
}
#endif

void BatchedLORAssembly::Assemble(OperatorHandle &A)
{
#ifdef MFEM_USE_MPI
   if (dynamic_cast<ParFiniteElementSpace*>(&fes_ho))
   {
      return ParAssemble(A);
   }
#endif

   SparseMatrix *A_mat = AssembleWithoutBC();

   // Eliminate essential DOFs (BCs) from the matrix (what we do here is
   // equivalent to  DiagonalPolicy::DIAG_KEEP).
   const int n_ess_dofs = ess_dofs.Size();
   const auto ess_dofs_d = ess_dofs.Read();
   const auto I = A_mat->ReadI();
   const auto J = A_mat->ReadJ();
   auto dA = A_mat->ReadWriteData();

   MFEM_FORALL(i, n_ess_dofs,
   {
      const int idof = ess_dofs_d[i];
      for (int j=I[idof]; j<I[idof+1]; ++j)
      {
         const int jdof = J[j];
         if (jdof != idof)
         {
            dA[j] = 0.0;
            for (int k=I[jdof]; k<I[jdof+1]; ++k)
            {
               if (J[k] == idof)
               {
                  dA[k] = 0.0;
                  break;
               }
            }
         }
      }
   });

   A.Reset(A_mat);
}

BatchedLORAssembly::BatchedLORAssembly(BilinearForm &a_,
                                       FiniteElementSpace &fes_ho_,
                                       const Array<int> &ess_dofs_)
   : R(fes_ho_), fes_ho(fes_ho_), ess_dofs(ess_dofs_)
{ }

void BatchedLORAssembly::Assemble(BilinearForm &a,
                                  FiniteElementSpace &fes_ho,
                                  const Array<int> &ess_dofs,
                                  OperatorHandle &A)
{
   if (HasIntegrator<DiffusionIntegrator>(a))
   {
      BatchedLORDiffusion(a, fes_ho, ess_dofs).Assemble(A);
   }
}

} // namespace mfem
