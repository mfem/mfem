// Implementation of class ParHDGBilinearForm3
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/sort_pairs.hpp"

#include "pHDGBilinearForm3.hpp"

namespace mfem
{

void ParHDGBilinearForm3::AssembleSC(const ParGridFunction *R,
                                     const ParGridFunction *F,
                                     Array<int> &bdr_attr_is_ess,
                                     const ParGridFunction sol,
                                     const double memA, const double memB,
                                     int skip_zeros)
{
   // ExchangeFaceNbrData to be able to use shared faces
   pfes1->ExchangeFaceNbrData();

   // Allocate the matrices and the RHS vectors on every processor.
   // Also, compute the el_to_face and Edge_to_be tables
   HDGBilinearForm3::Allocate(bdr_attr_is_ess, memA, memB);

   // Do the assembly in parallel
   ParallelAssemble(R, F, sol, memA, memB, skip_zeros);
}

/* Parallel Assembly of the Schur complement. Similar to the serial one,
 * but it computes the integrals over shared faces using the necessary
 * communication between the processors
 */
void ParHDGBilinearForm3::ParallelAssemble(const ParGridFunction *R,
                                           const ParGridFunction *F,
                                           const ParGridFunction sol,
                                           const double memA, const double memB,
                                           int skip_zeros)
{
   ParMesh *pmesh = pfes1 -> GetParMesh();
   DenseMatrix A_local;
   DenseMatrix AC_local, SC_local;
   Vector R_local, F_local, RF_local, AinvRF, CAinvRF;

   Array<int> fcs;

   Array<int> vdofs_q, vdofs_u, vdofs_e1, vdofs_e2;
   int ndof_q, ndof_u, ndof_e1, ndof_e2;

   int nsharedfaces = pmesh->GetNSharedFaces();
   int nedge = Edge_to_Be.Size();

   // Create an array to identify the shared faces. The entry is -1
   // if the face is not shared, otherwise, is gives the number of
   // the face in the shared face list, so that GetSharedFaceTransformation
   // can be used.
   Array<int> Edge_to_SharedEdge;
   Edge_to_SharedEdge.SetSize(nedge);
   Edge_to_SharedEdge = -1;
   for (int i = 0; i < nsharedfaces; i++)
   {
      Edge_to_SharedEdge[pmesh->GetSharedFace(i)] = i;
   }

   double *A_local_data, *B_local_data;

   for (int i=0; i< pfes1->GetNE(); i++)
   {
      pfes1 -> GetElementVDofs (i, vdofs_q);
      ndof_q  = vdofs_q.Size();

      pfes2 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      A_local.SetSize(ndof_q+ndof_u,ndof_q+ndof_u);
      A_local = 0.0;
      HDGBilinearForm3::compute_domain_integrals(i, &A_local);

      // Get the element faces
      el_to_face->GetRow(i, fcs);
      int no_faces = fcs.Size();

      DenseMatrix B_local[no_faces];
      DenseMatrix C_local[no_faces];
      DenseMatrix D_local[no_faces];
      Vector L_local;

      // Get the right hand side vectors and merge them into one,
      // so it has the same size as A_local
      R->GetSubVector(vdofs_q, R_local);
      F->GetSubVector(vdofs_u, F_local);
      RF_local.SetSize(ndof_q + ndof_u);
      for (int ii = 0; ii<ndof_q; ii++)
      {
         RF_local(ii) = R_local(ii);
      }
      for (int ii = 0; ii<ndof_u; ii++)
      {
         RF_local(ii+ndof_q) = F_local(ii);
      }

      for (int edge1=0; edge1<no_faces; edge1++)
      {
         fes3 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();

         B_local[edge1].SetSize(ndof_q+ndof_u, ndof_e1);
         C_local[edge1].SetSize(ndof_e1, ndof_q+ndof_u);
         D_local[edge1].SetSize(ndof_e1);

         B_local[edge1] = 0.0;
         C_local[edge1] = 0.0;
         D_local[edge1] = 0.0;

         // compute the face integrals for A, B, C and D
         if ( Edge_to_SharedEdge[fcs[edge1]] == -1 )
         {
            HDGBilinearForm3::compute_face_integrals(i, fcs[edge1],
                                                     Edge_to_Be[fcs[edge1]], false,
                                                     &A_local, &B_local[edge1], &C_local[edge1], &D_local[edge1]);
         }
         else
         {
            compute_face_integrals_shared(i, fcs[edge1], Edge_to_SharedEdge[fcs[edge1]],
                                          false,
                                          &A_local, &B_local[edge1], &C_local[edge1], &D_local[edge1]);
         }
      }

      // compute the negative inverse of A
      A_local.Invert();
      A_local.Neg();

      // Save A and B if necessary
      if (i<elements_A)
      {
         A_local_data = A_local.GetData();

         for (int j = 0; j<((ndof_u+ndof_q)*(ndof_u+ndof_q)); j++)
         {
            A_data[A_offsets[i] + j] = A_local_data[j];
         }
      }

      if (i<elements_B)
      {
         int size_B_copied = 0;
         for (int edge1=0; edge1<no_faces; edge1++)
         {
            B_local_data = B_local[edge1].GetData();
            fes3->GetFaceVDofs(fcs[edge1], vdofs_e1);

            for (int j = 0; j<((ndof_u+ndof_q)*(vdofs_e1.Size())); j++)
            {
               B_data[B_offsets[i] + size_B_copied + j] = B_local_data[j];
            }

            size_B_copied += (ndof_u+ndof_q)*(vdofs_e1.Size());
         }
      }

      // Eliminate the boundary conditions
      for (int edge1=0; edge1<no_faces; edge1++)
      {
         pfes3 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();

         L_local.SetSize(ndof_e1);
         L_local = 0.0;

         Eliminate_BC(vdofs_e1, ndof_u, ndof_q, sol, &RF_local,
                      &L_local, &B_local[edge1], &C_local[edge1], &D_local[edge1]);

         rhs_SC->AddElementVector(vdofs_e1, 1.0, L_local);

      }

      AinvRF.SetSize(ndof_q + ndof_u);

      // Compute -A^{-1}*F_local
      A_local.Mult(RF_local, AinvRF);

      // Loop over all the possible face pairs
      for (int edge1=0; edge1<fcs.Size(); edge1++)
      {
         // Get the unknowns belonging to the edge
         pfes3 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();

         (D_local[edge1]).Threshold(1.0e-16);
         mat->AddSubMatrix(vdofs_e1, vdofs_e1, D_local[edge1], skip_zeros);

         CAinvRF.SetSize(ndof_e1);
         (C_local[edge1]).Mult(AinvRF, CAinvRF);

         rhs_SC->AddElementVector(vdofs_e1, 1.0, CAinvRF);

         AC_local.SetSize(ndof_e1, ndof_q+ndof_u);
         Mult(C_local[edge1], A_local, AC_local);

         for (int edge2=0; edge2<fcs.Size(); edge2++)
         {
            // Get the unknowns belonging to the edge
            pfes3 -> GetFaceVDofs(fcs[edge2], vdofs_e2);
            ndof_e2 = vdofs_e2.Size();

            SC_local.SetSize(ndof_e1, ndof_e2);

            Mult(AC_local, B_local[edge2], SC_local);

            SC_local.Threshold(1.0e-16);
            mat->AddSubMatrix(vdofs_e1, vdofs_e2, SC_local, skip_zeros);
         }
      }
   }
}

// Compute the face integrals for A, B,C and D over a shared face
void ParHDGBilinearForm3::compute_face_integrals_shared(const int elem,
                                                        const int edge,
                                                        const int sf,
                                                        const bool is_reconstruction,
                                                        DenseMatrix *A_local,
                                                        DenseMatrix *B_local,
                                                        DenseMatrix *C_local,
                                                        DenseMatrix *D_local)
{
   ParMesh *pmesh = pfes1 -> GetParMesh();
   FaceElementTransformations *tr;

   // Over a shared edge get the shared face transformation
   tr = pmesh->GetSharedFaceTransformations(sf);

   const FiniteElement &trial_face_fe = *fes3->GetFaceElement(edge);
   const FiniteElement &testq_fe1 = *fes1->GetFE(tr->Elem1No);
   const FiniteElement &testq_fe2 = *fes1->GetFE(tr->Elem2No);
   const FiniteElement &testu_fe1 = *fes2->GetFE(tr->Elem1No);
   const FiniteElement &testu_fe2 = *fes2->GetFE(tr->Elem2No);

   // For the parallel case the element the processor owns is tr->Elem1No
   // Compute the integrals using element 1
   // For the serial there was an if condition to check if element 1 or 2
   // is needed. Over a shared face every processor uses the element it owns
   // as element 1
   hdg_bbfi[0]->AssembleFaceMatrixOneElement2and1FES(testq_fe1, testq_fe2,
                                                     testu_fe1, testu_fe2, trial_face_fe,
                                                     *tr, 1, is_reconstruction, elemmat1, elemmat2, elemmat3, elemmat4);

   // Add the face matrices to the local matrices
   A_local->Add(1.0, elemmat1);
   B_local->Add(1.0, elemmat2);

   // C and D are not necessary for recontruction, only when setting up
   // the Schur complement
   if (!is_reconstruction)
   {
      C_local->Add(1.0, elemmat3);
      D_local->Add(0.5, elemmat4);
   }
}

void ParHDGBilinearForm3::Eliminate_BC(const Array<int> &vdofs_e1,
                                       const int ndof_u, const int ndof_q,
                                       const ParGridFunction &sol,
                                       Vector *rhs_RF, Vector *rhs_L,
                                       DenseMatrix *B_local, DenseMatrix *C_local, DenseMatrix *D_local)
{
   int ndof_e1 = vdofs_e1.Size();
   double solution;

   // First we set the BC on the rhs vector for the unknowns on the boundary
   // and eliminate the rows and columns of D
   for (int j = 0; j < ndof_e1; j++) // j is the column
   {
      if (ess_dofs[vdofs_e1[j]] < 0)
      {
         (*rhs_L)(j) += sol(vdofs_e1[j]);
         for (int i = 0; i < ndof_e1; i++)
         {
            (*D_local)(j,i) = (i == j);
            (*D_local)(i,j) = (i == j);
         }
      }
   }

   // Eliminate BC from B, C and D
   // From the rhs we have to modify only the values
   // which do not belong to a boundary unknown,
   // since those values or the RHS are already set.
   for (int j = 0; j < ndof_e1; j++) // j is the column
   {
      if (ess_dofs[vdofs_e1[j]] < 0)
      {
         solution = sol(vdofs_e1[j]);
         for (int i = 0; i < ndof_e1; i++)
         {
            if (!(ess_dofs[vdofs_e1[i]] < 0))
            {
               (*rhs_L)(i) -= solution * (*D_local)(i,j);
            }
         }

         for (int i = 0; i < (ndof_q+ndof_u); i++)
         {
            (*rhs_RF)(i) -= solution * (*B_local)(i,j);
            (*B_local)(i,j) = 0.0;
            (*C_local)(j,i) = 0.0;
         }
      }
   }
}

// Reconstruct u and q from the facet solution
void ParHDGBilinearForm3::Reconstruct(const ParGridFunction *R,
                                      const ParGridFunction *F,
                                      ParGridFunction *ubar,
                                      ParGridFunction *q, ParGridFunction *u)
{
   ParMesh *pmesh = pfes1 -> GetParMesh();
   DenseMatrix A_local;
   Vector q_local, u_local, qu_local, R_local, F_local, RF_local, ubar_local,
          Bubar_local;

   Array<int> fcs;

   Array<int> vdofs_q, vdofs_u, vdofs_e1;
   int ndof_q, ndof_u, ndof_e1;

   int nsharedfaces = pmesh->GetNSharedFaces();
   int nedge = Edge_to_Be.Size();

   // same as for ParallelAssemble
   Array<int> Edge_to_SharedEdge;
   Edge_to_SharedEdge.SetSize(nedge);
   Edge_to_SharedEdge = -1;
   for (int i = 0; i < nsharedfaces; i++)
   {
      Edge_to_SharedEdge[pmesh->GetSharedFace(i)] = i;
   }

   for (int i=0; i< pfes1->GetNE(); i++)
   {
      pfes1 -> GetElementVDofs (i, vdofs_q);
      ndof_q  = vdofs_q.Size();

      pfes2 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      // if A is not saved
      A_local.SetSize(ndof_q + ndof_u, ndof_q+ndof_u);
      A_local = 0.0;
      if (i>=elements_A)
      {
         HDGBilinearForm3::compute_domain_integrals(i, &A_local);
      }

      // Get the element faces
      el_to_face->GetRow(i, fcs);

      int no_faces = fcs.Size();

      DenseMatrix B_local[no_faces], dummy_DM;

      R_local.SetSize(ndof_q);
      R_local = 0.0;
      F_local.SetSize(ndof_u);
      F_local = 0.0;

      R->GetSubVector(vdofs_q, R_local);
      F->GetSubVector(vdofs_u, F_local);

      RF_local.SetSize(ndof_q + ndof_u);
      for (int ii = 0; ii<ndof_q; ii++)
      {
         RF_local(ii) = R_local(ii);
      }
      for (int ii = 0; ii<ndof_u; ii++)
      {
         RF_local(ii+ndof_q) = F_local(ii);
      }

      Bubar_local.SetSize(ndof_q+ndof_u);

      int B_values_read = 0;

      for (int edge1=0; edge1<no_faces; edge1++)
      {
         fes3 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();
         B_local[edge1].SetSize(ndof_q+ndof_u, ndof_e1);
         B_local[edge1] = 0.0;

         // If B is not saved then compute the face integrals
         // otherwise load the matrices
         if (i>=elements_B)
         {
            if ( Edge_to_SharedEdge[fcs[edge1]] == -1 )
            {
               HDGBilinearForm3::compute_face_integrals(i, fcs[edge1], Edge_to_Be[fcs[edge1]],
                                                        true,
                                                        &A_local, &B_local[edge1], &dummy_DM, &dummy_DM);
            }
            else
            {
               compute_face_integrals_shared(i, fcs[edge1], Edge_to_SharedEdge[fcs[edge1]],
                                             true,
                                             &A_local, &B_local[edge1], &dummy_DM, &dummy_DM);
            }
         }
         else if (i < elements_B)
         {
            for (int row = 0; row < ndof_e1; row++)
               for (int col = 0; col < (ndof_q+ndof_u); col++)
               {
                  (B_local[edge1])(col,row) = B_data[B_offsets[i] + B_values_read + row*
                                                     (ndof_q+ndof_u) + col];
               }

            B_values_read += (ndof_q+ndof_u)*ndof_e1;
         }

         // Compute B*ubar
         ubar_local.SetSize(ndof_e1);
         ubar->GetSubVector(vdofs_e1, ubar_local);
         (B_local[edge1]).Mult(ubar_local, Bubar_local);

         // Subtract from RF
         RF_local.Add(-1.0, Bubar_local);
      }

      // Since the negative inverse of A is stored the negative of RF is needed
      RF_local *= -1.0;

      // Compute -A^{-1} if it is not saved or just load it
      if (i>=elements_A)
      {
         A_local.Invert();
         A_local.Neg();
      }
      else
      {
         for (int row = 0; row < (ndof_q+ndof_u); row++)
            for (int col = 0; col < (ndof_q+ndof_u); col++)
            {
               A_local(col,row) = A_data[A_offsets[i] + row*(ndof_q+ndof_u) + col];
            }
      }


      // (q,u) = -A^{-1}(B*ubar - RF)
      q_local.SetSize(ndof_q);
      q_local = 0.0;
      u_local.SetSize(ndof_u);
      u_local = 0.0;
      qu_local.SetSize(ndof_u+ndof_q);

      A_local.Mult(RF_local, qu_local);

      for (int ii = 0; ii<ndof_q; ii++)
      {
         q_local(ii) = qu_local(ii);
      }
      for (int ii = 0; ii<ndof_u; ii++)
      {
         u_local(ii) = qu_local(ii+ndof_q);
      }

      q->SetSubVector(vdofs_q, q_local);
      u->SetSubVector(vdofs_u, u_local);

   }
}

// Creates the parallel matrix from the local sparse matrices
HypreParMatrix *ParHDGBilinearForm3::ParallelAssembleSC(SparseMatrix *m)
{
   if (m == NULL) { return NULL; }

   MFEM_VERIFY(m->Finalized(), "local matrix needs to be finalized for "
               "ParallelAssemble3");

   int lvsize = pfes3->GetVSize();
   const HYPRE_Int *face_nbr_glob_ldof = pfes3->GetFaceNbrGlobalDofMap();
   HYPRE_Int ldof_offset = pfes3->GetMyDofOffset();

   Array<HYPRE_Int> glob_J(m->NumNonZeroElems());
   int *J = m->GetJ();
   for (int i = 0; i < glob_J.Size(); i++)
   {
      if (J[i] < lvsize)
      {
         glob_J[i] = J[i] + ldof_offset;
      }
      else
      {
         glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];
      }
   }

   HypreParMatrix *A = new HypreParMatrix(pfes3->GetComm(), lvsize,
                                          pfes3->GlobalVSize(),
                                          pfes3->GlobalVSize(), m->GetI(), glob_J,
                                          m->GetData(), pfes3->GetDofOffsets(),
                                          pfes3->GetDofOffsets());

   HypreParMatrix *rap = RAP(A, pfes3->Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

// Create the parallel vector from the local vectors
HypreParVector *ParHDGBilinearForm3::ParallelVectorSC()
{
   HypreParVector *tv = pfes3->NewTrueDofVector();

   pfes3->Dof_TrueDof_Matrix()->MultTranspose(*rhs_SC, *tv);
   return tv;
}

void ParHDGBilinearForm3::Update(FiniteElementSpace *nfes1,
                                 FiniteElementSpace *nfes2, FiniteElementSpace *nfes3)
{
   HDGBilinearForm3::Update(nfes1, nfes2, nfes3);
}

}

#endif
