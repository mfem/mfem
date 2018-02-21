// Implementation of class HDGBilinearForm2
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#include "HDGBilinearForm2.hpp"
#include "fem.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

namespace mfem
{


HDGBilinearForm2::HDGBilinearForm2 (FiniteElementSpace *_fes1,
                                    FiniteElementSpace *_fes2)
{
   fes1 = _fes1;
   fes2 = _fes2;

   mat = NULL;
   rhs_SC = NULL;
   el_to_face = NULL;

   A_data = NULL; B_data = NULL;
   elements_A = elements_B = 0;
}


double& HDGBilinearForm2::Elem (int i, int j)
{
   return mat -> Elem(i,j);
}

const double& HDGBilinearForm2::Elem (int i, int j) const
{
   return mat -> Elem(i,j);
}

void HDGBilinearForm2::Finalize (int skip_zeros)
{
   mat->Finalize(skip_zeros);
}

void HDGBilinearForm2::AddHDGDomainIntegrator(BilinearFormIntegrator * bfi)
{
   hdg_dbfi.Append (bfi);
}
void HDGBilinearForm2::AddHDGBdrIntegrator(BilinearFormIntegrator * bfi)
{
   hdg_bbfi.Append (bfi);
}

void HDGBilinearForm2::Update(FiniteElementSpace *nfes1,
                              FiniteElementSpace *nfes2)
{
   if (nfes1)
   {
      fes1 = nfes1;
      fes2 = nfes2;
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

HDGBilinearForm2::~HDGBilinearForm2()
{
   delete mat;
   delete rhs_SC;

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
 * * elements_A and elements_B are the number of elements on which A and B are stored, respectively
 */
void HDGBilinearForm2::Allocate(const double memA, const double memB)
{
   Mesh *mesh = fes1 -> GetMesh();
   mesh->GetEdgeToBdrFace(Edge_to_Be);

   // Get the list of the faces of every elementbdr
   if (mesh->Dimension() == 2)
      el_to_face = new Table(mesh->ElementToEdgeTable());
   else if (mesh->Dimension() == 3)
      el_to_face = new Table(mesh->ElementToFaceTable());

   if (mat == NULL)
   {
      mat = new SparseMatrix(fes2->GetVSize());
   }

   if (rhs_SC == NULL)
   {
      rhs_SC = new Vector(fes2->GetVSize());
      *rhs_SC = 0.0;
   }

   elements_A = (int)(memA * fes1->GetNE());
   elements_B = (int)(memB * fes1->GetNE());

   // Set the offset vectors
   A_offsets.SetSize(elements_A+1);
   B_offsets.SetSize(elements_B+1);
   A_offsets[0] = 0;
   B_offsets[0] = 0;

   Array<int> vdofs_u, vdofs_e1, fcs;
   int ndof_u;

   // loop over the elements to find the offset entries
   for (int i=0; i< fes1->GetNE(); i++)
   {
      // Get the local number of dof for u
      fes1 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // A will have the size ndof_u * ndof_u
      // The next offset entry can be set
      if (i < elements_A)
      {
         A_offsets[i+1] = A_offsets[i] + ndof_u * ndof_u;
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
            fes2->GetFaceVDofs(fcs[edge1], vdofs_e1);
            ndof_edge_all += vdofs_e1.Size();
         }

         B_offsets[i+1] = B_offsets[i] + ndof_u*ndof_edge_all;
      }

      // If i >= elements_A then i >= elements_B also, so the for loop can be cancelled
      if (i >= elements_A)
      {
         break;
      }
   }

   // Create A_data and B_data as a vector with the proper size
   delete [] A_data;
   A_data = new double[A_offsets[elements_A]];
   delete [] B_data;
   B_data = new double[B_offsets[elements_B]];
}

// Compute the Schur complement element-wise.
void HDGBilinearForm2::AssembleSC(const Vector rhs_F,
                                  const double memA, const double memB,
                                  int skip_zeros)
{
   DenseMatrix A_local, AC_local, SC_local;
   Vector F_local, CAinvF, AinvF;

   Array<int> fcs;
   Array<int> be_to_face;

   Array<int> vdofs_u, vdofs_e1, vdofs_e2;
   int ndof_u, ndof_e1, ndof_e2;

   double *A_local_data, *B_local_data;

   Allocate(memA, memB);

   DenseMatrix *B_local;
   DenseMatrix *C_local;
   DenseMatrix *D_local;
   Vector G_local;
   
   for (int i=0; i< fes1->GetNE(); i++)
   {
      fes1 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      A_local.SetSize(ndof_u, ndof_u);
      A_local = 0.0;
      compute_domain_integrals(i, &A_local);

      // Get the element faces
      el_to_face->GetRow(i, fcs);

      int no_faces = fcs.Size();
      B_local = new DenseMatrix[no_faces];
      C_local = new DenseMatrix[no_faces];
      D_local = new DenseMatrix[no_faces];

      F_local.SetSize(ndof_u);
      F_local = 0.0;
      rhs_F.GetSubVector(vdofs_u, F_local);

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

         compute_face_integrals(i, fcs[edge1], Edge_to_Be[fcs[edge1]], false,
                                &A_local, &B_local[edge1], &C_local[edge1], &D_local[edge1]);
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
         fes2 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
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
            fes2 -> GetFaceVDofs(fcs[edge2], vdofs_e2);
            ndof_e2 = vdofs_e2.Size();

            SC_local.SetSize(ndof_e1, ndof_e2);

            // Compute the product that will be added to the Schur complement
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
void HDGBilinearForm2::compute_domain_integrals(const int elem,
                                                DenseMatrix *A_local)
{
   // get the element transformation and the finite elements for the variables
   ElementTransformation *eltrans;
   eltrans = fes1->GetElementTransformation(elem);
   const FiniteElement &fe_u = *fes1->GetFE(elem);

   // compute the integrals
   hdg_dbfi[0]->AssembleElementMatrix(fe_u, *eltrans, elemmat1);

   // add them to A_local
   A_local->Add(1.0, elemmat1);
}

// compute all the face based integrals in one loop over the quadrature nodes
void HDGBilinearForm2::compute_face_integrals(const int elem,
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

      const FiniteElement &trial_face_fe = *fes2->GetFaceElement(edge);
      const FiniteElement &test_fe1 = *fes1->GetFE(tr->Elem1No);
      const FiniteElement &test_fe2 = *fes1->GetFE(tr->Elem2No);

      // Compute the integrals depending on which element do we need to use
      int elem_1or2 = 1 + (tr->Elem2No == elem);
      hdg_bbfi[0]->AssembleFaceMatrixOneElement1and1FES(test_fe1, test_fe2,
                                                        trial_face_fe,
                                                        *tr, elem_1or2, is_reconstruction, elemmat1, elemmat2, elemmat3,
                                                        elemmat4);
   }
   else
   {
      // Over an interior edge get the boundary face transformation
      tr =  mesh->GetBdrFaceTransformations(isboundary);

      // Only 2 FiniteElements are needed, since the face belongs to
      // only one element
      const FiniteElement &trial_face_fe = *fes2->GetFaceElement(edge);
      const FiniteElement &test_fe1 = *fes1->GetFE(tr->Elem1No);

      // compute the integrals
      hdg_bbfi[0]->AssembleFaceMatrixOneElement1and1FES(test_fe1, test_fe1,
                                                        trial_face_fe,
                                                        *tr, 1, is_reconstruction, elemmat1, elemmat2, elemmat3,
                                                        elemmat4);
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

// Reconstruct u from the facet solution
void HDGBilinearForm2::Reconstruct(const Vector *F,
                                   const Vector *ubar, GridFunction *u)
{
   DenseMatrix A_local;
   Vector u_local, F_local, ubar_local, Bubar_local;

   Array<int> fcs;

   Array<int> vdofs_u, vdofs_e1;
   int ndof_u, ndof_e1;

   DenseMatrix dummy_DM;
   DenseMatrix *B_local;

   for (int i=0; i< fes1->GetNE(); i++)
   {
      fes1 -> GetElementVDofs (i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      // if A is not saved
      A_local.SetSize(ndof_u, ndof_u);
      A_local = 0.0;
      if (i>=elements_A)
      {
         compute_domain_integrals(i, &A_local);
      }


      // Get the element faces
      el_to_face->GetRow(i, fcs);

      int no_faces = fcs.Size();
      B_local = new DenseMatrix[no_faces];

      Bubar_local.SetSize(ndof_u);

      int B_values_read = 0;

      F_local.SetSize(ndof_u);
      F_local = 0.0;
      F->GetSubVector(vdofs_u, F_local);

      for (int edge1=0; edge1<no_faces; edge1++)
      {
         fes2 -> GetFaceVDofs(fcs[edge1], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();

         B_local[edge1].SetSize(ndof_u, ndof_e1);

         B_local[edge1] = 0.0;

         // If B is not saved then compute the face integrals
         // otherwise load the matrices
         if (i>=elements_B)
            compute_face_integrals(i, fcs[edge1], Edge_to_Be[fcs[edge1]], true,
                                   &A_local, &B_local[edge1], &dummy_DM, &dummy_DM);
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
      
      delete [] B_local;
   }

}

}
