// Implementation of class ParHDGBilinearForm2
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo


#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/sort_pairs.hpp"

#include "pHDGBilinearForm2.hpp"

namespace mfem
{

void ParHDGBilinearForm2::AssembleSC(const ParGridFunction *F,
                                     const double memA, const double memB,
                                     int skip_zeros)
{
   // ExchangeFaceNbrData to be able to use shared faces
   pfes1->ExchangeFaceNbrData();

   // Allocate the matrices and the RHS vectors on every processor.
   // Also, compute the el_to_face and Edge_to_be tables
   HDGBilinearForm2::Allocate(memA, memB);

   // Do the assembly in parallel
   ParallelAssemble(F, memA, memB, skip_zeros);
}

/* Parallel Assembly of the Schur complement. Similar to the serial one,
 * but it computes the integrals over shared faces using the necessary
 * communication between the processors
 */
void ParHDGBilinearForm2::ParallelAssemble(const ParGridFunction *F,
                                           const double memA, const double memB,
                                           int skip_zeros)
{
   ParMesh *pmesh = pfes1 -> GetParMesh();
   DenseMatrix A_local, AC_local, SC_local;
   Vector F_local, AinvF, CAinvF;
   Array<int> vdofs_u, vdofs_e1, vdofs_e2, fcs;
   int ndof_u, ndof_e1, ndof_e2;

   int nsharedfaces = pmesh->GetNSharedFaces();
   int nedge = Edge_to_Be.Size();

   double *A_local_data, *B_local_data;

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

   for (int i=0; i< pfes1->GetNE(); i++)
   {
      pfes1 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      A_local.SetSize(ndof_u, ndof_u);
      A_local = 0.0;
      HDGBilinearForm2::compute_domain_integrals(i, &A_local);

      // Get the element faces
      el_to_face->GetRow(i, fcs);
      int no_faces = fcs.Size();

      DenseMatrix B_local[no_faces];
      DenseMatrix C_local[no_faces];
      DenseMatrix D_local[no_faces];
      Vector G_local;

      F_local.SetSize(ndof_u);
      F_local = 0.0;

      F->GetSubVector(vdofs_u, F_local);

      // compute the face integrals for A, B, C and D
      for (int edge1=0; edge1<no_faces; edge1++)
      {
         fes2 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();

         B_local[edge1].SetSize(ndof_u, ndof_e1);
         C_local[edge1].SetSize(ndof_e1, ndof_u);
         D_local[edge1].SetSize(ndof_e1, ndof_e1);

         B_local[edge1] = 0.0;
         C_local[edge1] = 0.0;
         D_local[edge1] = 0.0;

         if ( Edge_to_SharedEdge[fcs[edge1]] == -1 )
         {
            HDGBilinearForm2::compute_face_integrals(i, fcs[edge1], Edge_to_Be[fcs[edge1]],
                                                     false,
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
      A_local.Neg();
      A_local.Invert();

      // Save A and B if necessary
      if (i<elements_A)
      {
         A_local_data = A_local.GetData();

         for (int j = 0; j<ndof_u*ndof_u; j++)
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
            fes2->GetFaceVDofs(fcs[edge1], vdofs_e1);

            for (int j = 0; j<(ndof_u*(vdofs_e1.Size())); j++)
            {
               B_data[B_offsets[i] + size_B_copied + j] = B_local_data[j];
            }

            size_B_copied += ndof_u*(vdofs_e1.Size());
         }
      }

      AinvF.SetSize(ndof_u);
      A_local.Mult(F_local, AinvF);

      // Loop over all the possible face pairs
      for (int edge1=0; edge1<no_faces; edge1++)
      {
         // Get the unknowns belonging to the edge
         pfes2 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();
         (D_local[edge1]).Threshold(1.0e-16);
         mat->AddSubMatrix(vdofs_e1, vdofs_e1, D_local[edge1], skip_zeros);

         CAinvF.SetSize(ndof_e1);
         (C_local[edge1]).Mult(AinvF, CAinvF);
         rhs_SC->AddElementVector(vdofs_e1, 1.0, CAinvF);

         AC_local.SetSize(ndof_e1, ndof_u);
         Mult(C_local[edge1], A_local, AC_local);

         for (int edge2=0; edge2<no_faces; edge2++)
         {
            // Get the unknowns belonging to the edge
            pfes2 -> GetFaceVDofs(fcs[edge2], vdofs_e2);

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
void ParHDGBilinearForm2::compute_face_integrals_shared(const int elem,
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

   const FiniteElement &trial_face_fe = *fes2->GetFaceElement(edge);
   const FiniteElement &test_fe1 = *fes1->GetFE(tr->Elem1No);
   const FiniteElement &test_fe2 = *fes1->GetFE(tr->Elem2No);

   // For the parallel case the element the processor owns is tr->Elem1No
   // Compute the integrals using element 1
   // For the serial there was an if condition to check if element 1 or 2
   // is needed. Over a shared face every processor uses the element it owns
   // as element 1
   hdg_bbfi[0]->AssembleFaceMatrixOneElement1and1FES(test_fe1, test_fe2,
                                                     trial_face_fe,
                                                     *tr, 1, is_reconstruction, elemmat1, elemmat2, elemmat3,
                                                     elemmat4);

   // Add the face matrices to the local matrices
   A_local->Add(1.0, elemmat1);
   B_local->Add(1.0, elemmat2);
   if (!is_reconstruction)
   {
      // If it is not reconstruction C and D are necessary
      C_local->Add(1.0, elemmat3);

      // Over an interior edge only 1/2*D_local has to be assembled
      // since the functions \lambda and \mu are defined only on the face
      // therefore the same integral will be computed for both
      // adjacent elements
      D_local->Add(0.5, elemmat4);
   }
}

// Reconstruct u from the facet solution
void ParHDGBilinearForm2::Reconstruct(const ParGridFunction *F,
                                      const ParGridFunction *ubar,
                                      ParGridFunction *u)
{
   ParMesh *pmesh = pfes1 -> GetParMesh();
   DenseMatrix A_local;
   Vector u_local, F_local, ubar_local, Bubar_local;

   Array<int> fcs;

   Array<int> vdofs_u, vdofs_e1;
   int ndof_u, ndof_e1;

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
      pfes1 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      // if A is not saved
      A_local.SetSize(ndof_u, ndof_u);
      A_local = 0.0;
      if (i>=elements_A)
      {
         HDGBilinearForm2::compute_domain_integrals(i, &A_local);
      }

      // Get the element faces
      el_to_face->GetRow(i, fcs);

      int no_faces = fcs.Size();

      DenseMatrix B_local[no_faces], dummy_DM;

      F_local.SetSize(ndof_u);
      F_local = 0.0;
      F->GetSubVector(vdofs_u, F_local);
      Bubar_local.SetSize(ndof_u);
      int B_values_read = 0;

      for (int edge1=0; edge1<no_faces; edge1++)
      {
         pfes2 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();

         B_local[edge1].SetSize(ndof_u, ndof_e1);
         B_local[edge1] = 0.0;

         // If B is not saved then compute the face integrals
         // otherwise load the matrices
         if (i>=elements_B)
         {
            if ( Edge_to_SharedEdge[fcs[edge1]] == -1 )
            {
               HDGBilinearForm2::compute_face_integrals(i, fcs[edge1],
                                                        Edge_to_Be[fcs[edge1]], true,
                                                        &A_local, &B_local[edge1], &dummy_DM, &dummy_DM);
            }
            else
            {
               compute_face_integrals_shared(i, fcs[edge1],
                                             Edge_to_SharedEdge[fcs[edge1]], true,
                                             &A_local, &B_local[edge1], &dummy_DM, &dummy_DM);
            }
         }
         else
         {
            for (int row = 0; row < ndof_e1; row++)
               for (int col = 0; col < (ndof_u); col++)
               {
                  (B_local[edge1])(col,row) = B_data[B_offsets[i] + B_values_read + row*ndof_u +
                                                     col];
               }

            B_values_read += ndof_u*ndof_e1;
         }

         ubar_local.SetSize(ndof_e1);

         // Compute B*ubar
         ubar->GetSubVector(vdofs_e1, ubar_local);
         (B_local[edge1]).Mult(ubar_local, Bubar_local);

         F_local.Add(-1.0, Bubar_local);
      }

      // Since the negative inverse of A is stored the negative of F is needed
      F_local *= -1.0;

      // Compute -A^{-1} if it is not saved or just load it
      if (i>=elements_A)
      {
         A_local.Invert();
         A_local.Neg();
      }
      else
      {
         for (int row = 0; row < ndof_u; row++)
            for (int col = 0; col < ndof_u; col++)
            {
               A_local(col,row) = A_data[A_offsets[i] + row*ndof_u + col];
            }
      }

      // u = -A^{-1}(B*ubar - F)
      u_local.SetSize(ndof_u);
      A_local.Mult(F_local, u_local);

      u->SetSubVector(vdofs_u, u_local);
   }
}

// Creates the parallel matrix from the local sparse matrices
HypreParMatrix *ParHDGBilinearForm2::ParallelAssembleSC(SparseMatrix *m)
{
   if (m == NULL) { return NULL; }

   MFEM_VERIFY(m->Finalized(), "local matrix needs to be finalized for "
               "ParallelAssemble3");

   int lvsize = pfes2->GetVSize();
   const HYPRE_Int *face_nbr_glob_ldof = pfes2->GetFaceNbrGlobalDofMap();
   HYPRE_Int ldof_offset = pfes2->GetMyDofOffset();

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

   HypreParMatrix *A = new HypreParMatrix(pfes2->GetComm(), lvsize,
                                          pfes2->GlobalVSize(),
                                          pfes2->GlobalVSize(), m->GetI(), glob_J,
                                          m->GetData(), pfes2->GetDofOffsets(),
                                          pfes2->GetDofOffsets());

   HypreParMatrix *rap = RAP(A, pfes2->Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

// Create the parallel vector from the local vectors
HypreParVector *ParHDGBilinearForm2::ParallelVectorSC()
{
   HypreParVector *tv = pfes2->NewTrueDofVector();

   pfes2->Dof_TrueDof_Matrix()->MultTranspose(*rhs_SC, *tv);
   return tv;
}

void ParHDGBilinearForm2::Update(FiniteElementSpace *nfes1,
                                 FiniteElementSpace *nfes2)
{
   HDGBilinearForm2::Update(nfes1, nfes2);
}


}

#endif
