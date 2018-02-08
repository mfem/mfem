// Implementation of class HDGBilinearForm3
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#include "HDGBilinearForm3.hpp"

#include "fem.hpp"
#include <cmath>

namespace mfem
{

HDGBilinearForm3::HDGBilinearForm3 (FiniteElementSpace * f1,
                                    FiniteElementSpace * f2, FiniteElementSpace * f3)
{
   fes1 = f1;
   fes2 = f2;
   fes3 = f3;

   mat = NULL;
   rhs_SC = NULL;
   el_to_face = NULL;
   A_data = NULL; B_data = NULL;
   elements_A = elements_B = 0;
}


double& HDGBilinearForm3::Elem (int i, int j)
{
   return mat -> Elem(i,j);
}

const double& HDGBilinearForm3::Elem (int i, int j) const
{
   return mat -> Elem(i,j);
}

void HDGBilinearForm3::Finalize (int skip_zeros)
{
   mat->Finalize(skip_zeros);
}

void HDGBilinearForm3::AddHDGDomainIntegrator(BilinearFormIntegrator * bfi)
{
   hdg_dbfi.Append (bfi);
}
void HDGBilinearForm3::AddHDGBdrIntegrator(BilinearFormIntegrator * bfi)
{
   hdg_bbfi.Append (bfi);
}

void HDGBilinearForm3::Update(FiniteElementSpace *nfes1,
                              FiniteElementSpace *nfes2, FiniteElementSpace *nfes3)
{
   if (nfes1)
   {
      fes1 = nfes1;
      fes2 = nfes2;
      fes3 = nfes3;
   }
   delete mat;
   mat = NULL;

   delete rhs_SC;
   rhs_SC = NULL;

   delete [] A_data;
   A_data = NULL;

   delete [] B_data;
   B_data = NULL;

   elements_A = elements_B = 0;
}

HDGBilinearForm3::~HDGBilinearForm3()
{
   delete mat;
   delete rhs_SC;
   delete A_data;
   delete B_data;

   int k;
   for (k=0; k < hdg_dbfi.Size(); k++) { delete hdg_dbfi[k]; }
   for (k=0; k < hdg_bbfi.Size(); k++) { delete hdg_bbfi[k]; }
}


/* To allocate the sparse matrix and the right hand side vector and to create the Edge_to_be and el_to_face tables.
 * This is also called for the parallel, since this information is important on every processor
 * Edge_to_be is an Array with size of the number of edges.
 * The entry Edge_to_be[i] is -1 if the i-th face is interior or shared. It is greater then -1 if the i-th face lies on the boundary
 * Moreover, Edge_to_be[i] = n means that the n-th boundary face is the i-th face.
 * el_to_faces has number of element rows and the i-th row contains the faces of the i-th element
 * elements_A and elements_B are the number of elements on which A and B are stored, respectively
 */
void HDGBilinearForm3::Allocate(Array<int> &bdr_attr_is_ess, const double memA,
                                const double memB)
{
   Mesh *mesh = fes1 -> GetMesh();
   mesh->GetEdgeToBdrFace(Edge_to_Be);

   // Get the list of the faces of every element
   if (mesh->Dimension() == 2)
      el_to_face = new Table(mesh->ElementToEdgeTable());
   else if (mesh->Dimension() == 3)
      el_to_face = new Table(mesh->ElementToFaceTable());


   if (mat == NULL)
   {
      mat = new SparseMatrix(fes3->GetVSize());
   }

   if (rhs_SC == NULL)
   {
      rhs_SC = new Vector(fes3->GetVSize());
      *rhs_SC = 0.0;
   }

   fes3->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   elements_A = (int)(memA * fes1->GetNE());
   elements_B = (int)(memB * fes1->GetNE());

   // Set the offset vectors
   A_offsets.SetSize(elements_A+1);
   B_offsets.SetSize(elements_B+1);
   A_offsets[0] = 0;
   B_offsets[0] = 0;

   Array<int> vdofs_q, vdofs_u, vdofs_e1, fcs;
   int ndof_q, ndof_u;

   // loop over the elements to find the offset entries
   for (int i=0; i< fes1->GetNE(); i++)
   {
      // Get the local number of dof for q and u
      fes1 -> GetElementVDofs (i, vdofs_q);
      ndof_q  = vdofs_q.Size();

      fes2 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // A will have the size (ndof_q + ndof_u)*(ndof_q + ndof_u)
      // The next offset entry can be set
      if (i < elements_A)
      {
         A_offsets[i+1] = A_offsets[i] + (ndof_q + ndof_u)*(ndof_q + ndof_u);
      }

      // To find the next offset entry of B the local number of dofs
      // from the faces are needed
      el_to_face->GetRow(i, fcs);
      int no_faces = fcs.Size();
      int ndof_edge_all = 0;

      // Sum up the face dofs for all faces
      if (i < elements_B)
      {
         for (int edge1=0; edge1<no_faces; edge1++)
         {
            fes3->GetFaceVDofs(fcs[edge1], vdofs_e1);
            ndof_edge_all += vdofs_e1.Size();
         }

         B_offsets[i+1] = B_offsets[i] + (ndof_q + ndof_u)*(ndof_edge_all);
      }

      // If i >= elements_A then i >= elements_B also, so the for loop can be stopped
      if (i >= elements_A)
      {
         break;
      }
   }

   // Create A_data and B_data as a vector with the proper size
   delete A_data;
   A_data = new double[A_offsets[elements_A]];
   delete B_data;
   B_data = new double[B_offsets[elements_B]];
}

// Compute the Schur complement element-wise.
void HDGBilinearForm3::AssembleSC(const Vector rhs_R, const Vector rhs_F,
                                  Array<int> &bdr_attr_is_ess,
                                  Vector &sol,
                                  const double memA, const double memB,
                                  int skip_zeros)
{
   DenseMatrix A_local;
   DenseMatrix AC_local, SC_local;
   Vector R_local, F_local, RF_local, AinvRF, CAinvRF;

   Array<int> fcs;

   Array<int> vdofs_q, vdofs_u, vdofs_e1, vdofs_e2;
   int ndof_q, ndof_u, ndof_e1, ndof_e2;

   double *A_local_data, *B_local_data;

   Allocate(bdr_attr_is_ess, memA, memB);

   DenseMatrix *B_local;
   DenseMatrix *C_local;
   DenseMatrix *D_local;
   Vector L_local;

   for (int i=0; i< fes1->GetNE(); i++)
   {
      fes1 -> GetElementVDofs (i, vdofs_q);
      ndof_q  = vdofs_q.Size();

      fes2 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      A_local.SetSize(ndof_q+ndof_u,ndof_q+ndof_u);
      A_local = 0.0;
      compute_domain_integrals(i, &A_local);

      // Get the element faces
      el_to_face->GetRow(i, fcs);

      int no_faces = fcs.Size();
      B_local = new DenseMatrix[no_faces];
      C_local = new DenseMatrix[no_faces];
      D_local = new DenseMatrix[no_faces];
      
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

         // compute the face integrals
         compute_face_integrals(i, fcs[edge1], Edge_to_Be[fcs[edge1]], false,
                                &A_local, &B_local[edge1], &C_local[edge1], &D_local[edge1]);
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

      // Get the right hand side vectors and merge them into one, so it has
      // the same size as A_local
      rhs_R.GetSubVector(vdofs_q, R_local);
      rhs_F.GetSubVector(vdofs_u, F_local);
      RF_local.SetSize(ndof_q + ndof_u);
      for (int ii = 0; ii<ndof_q; ii++)
      {
         RF_local(ii) = R_local(ii);
      }
      for (int ii = 0; ii<ndof_u; ii++)
      {
         RF_local(ii+ndof_q) = F_local(ii);
      }

      // eliminate BC
      for (int edge1=0; edge1<fcs.Size(); edge1++)
      {
         fes3 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();

         L_local.SetSize(ndof_e1);
         L_local = 0.0;

         Eliminate_BC(vdofs_e1,
                      ndof_u, ndof_q,
                      sol,
                      &RF_local, &L_local,
                      &B_local[edge1], &C_local[edge1], &D_local[edge1]);

         rhs_SC->AddElementVector(vdofs_e1, 1.0, L_local);
      }

      AinvRF.SetSize(ndof_q + ndof_u);
      A_local.Mult(RF_local, AinvRF);

      // Loop over all the possible face pairs
      for (int edge1=0; edge1<fcs.Size(); edge1++)
      {
         // Get the unknowns belonging to the edge
         fes3 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
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
            fes3 -> GetFaceVDofs(fcs[edge2], vdofs_e2);
            ndof_e2 = vdofs_e2.Size();

            SC_local.SetSize(ndof_e1, ndof_e2);

            Mult(AC_local, B_local[edge2], SC_local);

            SC_local.Threshold(1.0e-16);
            mat->AddSubMatrix(vdofs_e1, vdofs_e2, SC_local, skip_zeros);
         }
      }

      delete [] B_local;
      delete [] C_local;
      delete [] D_local;
   }

}

// compute all the domain based integrals in one loop over the quadrature nodes
void HDGBilinearForm3::compute_domain_integrals(const int elem,
                                                DenseMatrix *A_local)
{
   // get the element transformation and the finite elements for the variables
   ElementTransformation *eltrans;
   eltrans = fes1->GetElementTransformation(elem);
   const FiniteElement &fe_q = *fes1->GetFE(elem);
   const FiniteElement &fe_u = *fes2->GetFE(elem);

   // compute the integrals
   hdg_dbfi[0]->AssembleElementMatrix2FES(fe_q, fe_u, *eltrans, elemmat1);

   // add them to A_local
   A_local->Add(1.0, elemmat1);
}

// compute all the face based integrals in one loop over the quadrature nodes
void HDGBilinearForm3::compute_face_integrals(const int elem,
                                              const int edge,
                                              const int isboundary,
                                              const bool is_reconstruction,
                                              DenseMatrix *A_local,
                                              DenseMatrix *B_local,
                                              DenseMatrix *C_local,
                                              DenseMatrix *D_local)
{
   Mesh *mesh = fes1 -> GetMesh();
   FaceElementTransformations *tr;

   if (isboundary == -1)
   {
      // Over an interior edge get the face transformation
      tr = mesh->GetFaceElementTransformations(edge);

      // Get the finite elements on both physical elements and on the face
      const FiniteElement &trial_face_fe = *fes3->GetFaceElement(edge);
      const FiniteElement &testq_fe1 = *fes1->GetFE(tr->Elem1No);
      const FiniteElement &testq_fe2 = *fes1->GetFE(tr->Elem2No);
      const FiniteElement &testu_fe1 = *fes2->GetFE(tr->Elem1No);
      const FiniteElement &testu_fe2 = *fes2->GetFE(tr->Elem2No);

      // Compute the integrals depending on which element do we need to use
      if (tr->Elem2No == elem)
      {
         hdg_bbfi[0]->AssembleFaceMatrixOneElement2and1FES(testq_fe1, testq_fe2,
                                                           testu_fe1, testu_fe2, trial_face_fe,
                                                           *tr, 2, is_reconstruction, elemmat1, elemmat2, elemmat3, elemmat4);
      }
      else
      {
         hdg_bbfi[0]->AssembleFaceMatrixOneElement2and1FES(testq_fe1, testq_fe2,
                                                           testu_fe1, testu_fe2, trial_face_fe,
                                                           *tr, 1, is_reconstruction, elemmat1, elemmat2, elemmat3, elemmat4);
      }
   }
   else
   {
      // Over an interior edge get the boundary face transformation
      tr =  mesh->GetBdrFaceTransformations(isboundary);

      // Only 3 FiniteElements are needed, since the face belongs to
      // only one element
      const FiniteElement &trial_face_fe = *fes3->GetFaceElement(edge);
      const FiniteElement &testq_fe1 = *fes1->GetFE(tr->Elem1No);
      const FiniteElement &testu_fe1 = *fes2->GetFE(tr->Elem1No);

      // compute the integrals
      hdg_bbfi[0]->AssembleFaceMatrixOneElement2and1FES(testq_fe1, testq_fe1,
                                                        testu_fe1, testu_fe1, trial_face_fe,
                                                        *tr, 1, is_reconstruction, elemmat1, elemmat2, elemmat3, elemmat4);
   }

   // Add the face matrices to the local matrices
   A_local->Add(1.0, elemmat1);
   B_local->Add(1.0, elemmat2);

   // C and D are not necessary for recontruction, only when setting up
   // the Schur complement
   if (!is_reconstruction)
   {
      C_local->Add(1.0, elemmat3);
      D_local->Add(1.0, elemmat4);
   }

}

// Eliminate boundary conditions
void HDGBilinearForm3::Eliminate_BC(const Array<int> &vdofs_e1,
                                    const int ndof_u, const int ndof_q,
                                    const Vector &sol,
                                    Vector *rhs_RF, Vector *rhs_L,
                                    DenseMatrix *B_local, DenseMatrix *C_local, DenseMatrix *D_local)
{
   int ndof_e1 = vdofs_e1.Size();
   double solution;

   // First we set the BC on the rhs vector for
   // the unknowns on the boundary
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
   // From the rhs we have to modify only the values which do not
   // belong to a boundary unknown,
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
void HDGBilinearForm3::Reconstruct(const Vector *R, const Vector *F,
                                   Vector &sol,
                                   GridFunction *q, GridFunction *u)
{
   DenseMatrix A_local;
   Vector q_local, u_local, qu_local, R_local, F_local, RF_local, ubar_local,
          Bubar_local;

   Array<int> fcs;

   Array<int> vdofs_q, vdofs_u, vdofs_e1;
   int ndof_q, ndof_u, ndof_e1;

   DenseMatrix dummy_DM;
   DenseMatrix *B_local;

   for (int i=0; i< fes1->GetNE(); i++)
   {
      fes1 -> GetElementVDofs (i, vdofs_q);
      ndof_q  = vdofs_q.Size();

      fes2 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      // if A is not saved
      A_local.SetSize(ndof_q + ndof_u, ndof_q+ndof_u);
      A_local = 0.0;
      if (i>=elements_A)
      {
         compute_domain_integrals(i, &A_local);
      }

      // Get the element faces
      el_to_face->GetRow(i, fcs);

      int no_faces = fcs.Size();
      B_local = new DenseMatrix[no_faces];

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
            compute_face_integrals(i, fcs[edge1], Edge_to_Be[fcs[edge1]], true,
                                   &A_local, &B_local[edge1], &dummy_DM, &dummy_DM);
         else
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
         sol.GetSubVector(vdofs_e1, ubar_local);
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
      
      delete [] B_local;

   }
}

}
