// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Implementation of class HDGBilinearForm
//
// Contributed by: T. Horvath: Oakland University
//                 S. Rhebergen, A. Sivas: University of Waterloo

#include "HDGBilinearForm.hpp"
#include "fem.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

namespace mfem
{


HDGBilinearForm::HDGBilinearForm (Array<FiniteElementSpace*> &_fes1,
                                  Array<FiniteElementSpace*> &_fes2,
                                  bool _parallel)
{
   NVolumeFES = _fes1.Size();
   NSkeletalFES = _fes2.Size();
   volume_fes.SetSize(NVolumeFES);
   skeletal_fes.SetSize(NSkeletalFES);
   for (int fes=0; fes < NVolumeFES; fes++)
   {
      volume_fes[fes] = _fes1[fes];
   }
   for (int fes=0; fes < NSkeletalFES; fes++)
   {
      skeletal_fes[fes] = _fes2[fes];
   }

   parallel = _parallel;
   mat = NULL;
   rhs_SC = NULL;
   el_to_face = NULL;

   A_data = NULL; B_data = NULL;
   elements_A = elements_B = 0;
}

HDGBilinearForm::HDGBilinearForm(FiniteElementSpace *_volume_fes,
                                 FiniteElementSpace *_skeletal_fes,
                                 bool _parallel)
{
   NVolumeFES = 1;
   NSkeletalFES = 1;
   volume_fes.SetSize(NVolumeFES);
   skeletal_fes.SetSize(NSkeletalFES);

   volume_fes[0] = _volume_fes;

   skeletal_fes[0] = _skeletal_fes;

   parallel = _parallel;
   mat = NULL;
   rhs_SC = NULL;
   el_to_face = NULL;

   A_data = NULL; B_data = NULL;
   elements_A = elements_B = 0;
}

HDGBilinearForm::HDGBilinearForm(FiniteElementSpace *_fes1,
                                 FiniteElementSpace *_fes2,
                                 FiniteElementSpace *_fes3,
                                 bool _parallel)
{
   NVolumeFES = 2;
   NSkeletalFES = 1;
   volume_fes.SetSize(NVolumeFES);
   skeletal_fes.SetSize(NSkeletalFES);

   volume_fes[0] = _fes1;
   volume_fes[1] = _fes2;

   skeletal_fes[0] = _fes3;

   parallel = _parallel;
   mat = NULL;
   rhs_SC = NULL;
   el_to_face = NULL;

   A_data = NULL; B_data = NULL;
   elements_A = elements_B = 0;
}

HDGBilinearForm::~HDGBilinearForm()
{
   delete mat;
   delete rhs_SC;
   delete A_data;
   delete B_data;

   int k;
   for (k=0; k < hdg_dbfi.Size(); k++) { delete hdg_dbfi[k]; }
   for (k=0; k < hdg_fbfi.Size(); k++) { delete hdg_fbfi[k]; }
}

void HDGBilinearForm::Finalize (int skip_zeros)
{
   mat->Finalize(skip_zeros);
}

void HDGBilinearForm::AddHDGDomainIntegrator(BilinearFormIntegrator * bfi)
{
   hdg_dbfi.Append (bfi);
}

void HDGBilinearForm::AddHDGFaceIntegrator(BilinearFormIntegrator * bfi)
{
   hdg_fbfi.Append (bfi);
}

void HDGBilinearForm::Update(FiniteElementSpace *nfes1,
                             FiniteElementSpace *nfes2)
{
   if (nfes1)
   {
      volume_fes = nfes1;
      skeletal_fes = nfes2;
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



void HDGBilinearForm::GetInteriorVDofs(int i, Array<int> &vdofs) const
{
   vdofs.SetSize(0);
   Array<int> vdofs_fes;
   for (int fes=0; fes < NVolumeFES; fes++)
   {
      volume_fes[fes]->GetElementVDofs(i, vdofs_fes);
      vdofs.Append(vdofs_fes);
   }
}

void HDGBilinearForm::GetInteriorSubVector(const Array<GridFunction*>
                                           &rhs_vector,
                                           int i, int ndof, Vector &SubVector) const
{
   SubVector.SetSize(ndof);
   Vector LocalVector;
   Array<int> vdofs_fes;

   int counter = 0;
   for (int fes=0; fes < NVolumeFES; fes++)
   {
      volume_fes[fes]->GetElementVDofs(i, vdofs_fes);
      rhs_vector[fes]->GetSubVector(vdofs_fes, LocalVector);
      for (int k = 0; k<vdofs_fes.Size(); k++)
      {
         SubVector(counter + k) = LocalVector(k);
      }

      counter += vdofs_fes.Size();
   }
}

void HDGBilinearForm::SetInteriorSubVector(Array<GridFunction*>
                                           &sol_gridfunctions,
                                           int i, int ndof, Vector &SubVector)
{
   SubVector.SetSize(ndof);
   Vector LocalVector;
   Array<int> vdofs_fes;

   int counter = 0;
   for (int fes=0; fes < NVolumeFES; fes++)
   {
      volume_fes[fes]->GetElementVDofs(i, vdofs_fes);
      LocalVector.SetSize(vdofs_fes.Size());
      for (int k = 0; k<vdofs_fes.Size(); k++)
      {
         LocalVector(k) = SubVector(counter + k);
      }

      sol_gridfunctions[fes]->SetSubVector(vdofs_fes, LocalVector);

      counter += vdofs_fes.Size();
   }
}

void HDGBilinearForm::GetFaceVDofs(int face, Array<int> &vdofs) const
{
   vdofs.SetSize(0);
   Array<int> vdofs_fes;
   for (int fes=0; fes < NSkeletalFES; fes++)
   {
      skeletal_fes[fes]->GetFaceVDofs(face, vdofs_fes);
      vdofs.Append(vdofs_fes);
   }
}

// compute all the domain based integrals in one loop over the quadrature nodes
void HDGBilinearForm::compute_domain_integrals(const int elem,
                                               DenseMatrix *A_local)
{
   // get the element transformation and the finite elements for the variables
   ElementTransformation *eltrans;
   eltrans = volume_fes[0]->GetElementTransformation(elem);
   const FiniteElement &fe_u1 = *volume_fes[0]->GetFE(elem);

   switch (NVolumeFES)
   {
      case 1:
      {
         // compute the integrals
         hdg_dbfi[0]->AssembleElementMatrix(fe_u1, *eltrans, elemmat1);
         break;
      }
      case 2:
      {
         const FiniteElement &fe_u2 = *volume_fes[1]->GetFE(elem);
         // compute the integrals
         hdg_dbfi[0]->AssembleElementMatrix2FES(fe_u1, fe_u2, *eltrans, elemmat1);
         break;
      }
      default:
      {
         mfem_error("HDGBilinearForm::compute_domain_integrals is defined only for 1 or 2 interior FES");
         break;
      }
   }

   // add them to the right matrices
   A_local->Add(1.0, elemmat1);
}

// Compute the face integrals for B and C
/* They both contain only TraceInteriorFaceIntegrators and TraceBoundaryFaceIntegrators.
 * The bool onlyB should be false when creating the Schur complement, true when reconstructing u from ubar.
*/
void HDGBilinearForm::compute_face_integrals(const int elem, const int edge,
                                             //                                             const int isshared,
                                             const bool onlyB,
                                             DenseMatrix *A_local,
                                             DenseMatrix *B_local,
                                             DenseMatrix *C_local,
                                             DenseMatrix *D_local)
{
   FaceElementTransformations *tr;

   Mesh *mesh = volume_fes[0] -> GetMesh();
   tr = mesh->GetFaceElementTransformations(edge);

   const FiniteElement &trace_fe = *skeletal_fes[0]->GetFaceElement(edge);
   const FiniteElement &volume_fe = *volume_fes[0]->GetFE(elem);

   // If elem is tr->Elem2No then normal is the outward normal,
   // otherwise it is the inward normal
   int elem_1or2 = 1 + (tr->Elem2No == elem);
   switch (NVolumeFES)
   {
      case 1:
      {
         hdg_fbfi[0]->AssembleFaceMatrixOneElement1and1FES(volume_fe,
                                                           trace_fe,
                                                           *tr, elem_1or2, onlyB,
                                                           elemmat1, elemmat2,
                                                           elemmat3, elemmat4);
         break;
      }
      case 2:
      {
         const FiniteElement &volume_fe2 = *volume_fes[1]->GetFE(elem);
         hdg_fbfi[0]->AssembleFaceMatrixOneElement2and1FES(volume_fe, volume_fe2,
                                                           trace_fe,
                                                           *tr, elem_1or2, onlyB,
                                                           elemmat1, elemmat2,
                                                           elemmat3, elemmat4);
         break;
      }
      default:
      {
         mfem_error("HDGBilinearForm::compute_face_integrals is defined only for 1 or 2 interior FES");
         break;
      }

   }

   // If it is not reconstruction C and D are needed
   A_local->Add(1.0, elemmat1);
   B_local->Add(1.0, elemmat2);
   if (!onlyB)
   {
      C_local->Add(1.0, elemmat3);
      D_local->Add(1.0, elemmat4);
   }

}

/* To allocate the sparse matrix and the right hand side vector and to create the el_to_face tables.
 * el_to_faces has number of element rows and the i-th row contains the faces of the i-th element
 */
void HDGBilinearForm::Allocate(const Array<int> &bdr_attr_is_ess,
                               const double memA, const double memB)
{
   Mesh *mesh = volume_fes[0] -> GetMesh();

   // Get the list of the faces of every element
   if (mesh->Dimension() == 2)
   {
      el_to_face = new Table(mesh->ElementToEdgeTable());
   }
   else if (mesh->Dimension() == 3)
   {
      el_to_face = new Table(mesh->ElementToFaceTable());
   }


#ifdef MFEM_USE_MPI
   ParFiniteElementSpace* pfes1 = dynamic_cast<ParFiniteElementSpace*>
                                  (volume_fes[0]);

   if (parallel)
   {
      pfes1->ExchangeFaceNbrData();
   }
#endif

   if (mat == NULL)
   {
      mat = new SparseMatrix(skeletal_fes[0]->GetVSize());
   }

   if (rhs_SC == NULL)
   {
      rhs_SC = new Vector(skeletal_fes[0]->GetVSize());
      *rhs_SC = 0.0;
   }

   skeletal_fes[0]->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   elements_A = (int)(memA * volume_fes[0]->GetNE());
   elements_B = (int)(memB * volume_fes[0]->GetNE());

   // Set the offset vectors
   A_offsets.SetSize(elements_A+1);
   B_offsets.SetSize(elements_B+1);
   A_offsets[0] = 0;
   B_offsets[0] = 0;

   Array<int> vdofs_u, fcs;
   int ndof_u;

   // loop over the elements to find the offset entries
   for (int i=0; i< volume_fes[0]->GetNE(); i++)
   {
      // Get the local number of dof for u
      GetInteriorVDofs(i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // A will have the size (ndof_q + ndof_u)*(ndof_q + ndof_u)
      // The next offset entry can be set
      if (i < elements_A)
      {
         A_offsets[i+1] = A_offsets[i] + ndof_u * ndof_u;
      }

      // To find the next offset entry of B the local number of dofs
      // are needed
      el_to_face->GetRow(i, fcs);
      int no_faces = fcs.Size();
      int ndof_edge_all = 0;

      // Sum up the face dofs for all faces
      if (i < elements_B)
      {
         for (int edge1=0; edge1<no_faces; edge1++)
         {
            GetFaceVDofs(fcs[edge1], vdofs3);
            ndof_edge_all += vdofs3.Size();
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
   delete A_data;
   A_data = new double[A_offsets[elements_A]];
   delete B_data;
   B_data = new double[B_offsets[elements_B]];

}

void HDGBilinearForm::AssembleSC(GridFunction *F,
                                 const double memA, const double memB,
                                 int skip_zeros)
{
   Array<GridFunction*> RHSGridFunctions;
   RHSGridFunctions.SetSize(1);
   RHSGridFunctions[0] = F;
   Array<int> ess_bdr;
   ess_bdr.SetSize(volume_fes[0]->GetMesh()->bdr_attributes.Max());
   ess_bdr = 0; // dummy

   GridFunction sol(skeletal_fes[0]);
   sol = 0.0;

   AssembleSC(RHSGridFunctions, ess_bdr, sol, 0, memA, memB, skip_zeros);
}

void HDGBilinearForm::AssembleSC(GridFunction *F1,
                                 GridFunction *F2,
                                 Array<int> &bdr_attr_is_ess,
                                 GridFunction &sol,
                                 const double memA, const double memB,
                                 int skip_zeros)
{
   Array<GridFunction*> RHSGridFunctions;
   RHSGridFunctions.SetSize(2);
   RHSGridFunctions[0] = F1;
   RHSGridFunctions[1] = F2;
   AssembleSC(RHSGridFunctions, bdr_attr_is_ess, sol, 1, memA, memB, skip_zeros);
}

void HDGBilinearForm::AssembleSC(Array<GridFunction*> &rhs_F,
                                 const Array<int> &bdr_attr_is_ess,
                                 const GridFunction &sol,
                                 const int elimBC,
                                 const double memA, const double memB,
                                 int skip_zeros)
{
   Allocate(bdr_attr_is_ess, memA, memB);

   DenseMatrix A_local, CA_local, SC_local;
   Vector F_local, F1_local, F2_local, CAinvF, AinvF;

   Array<int> fcs;
   Array<int> be_to_face;

   Array<int> vdofs_u, vdofs_e1, vdofs_e2;
   int ndof_u, ndof_e1, ndof_e2;

   double *A_local_data, *B_local_data;

   DenseMatrix *B_local;
   DenseMatrix *C_local;
   DenseMatrix *D_local;
   Vector G_local;

   for (int i=0; i< volume_fes[0]->GetNE(); i++)
   {
      GetInteriorVDofs(i, vdofs_u);
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

      GetInteriorSubVector(rhs_F, i, ndof_u, F_local);

      // compute the face integrals for A, B, C and D
      for (int edge=0; edge<no_faces; edge++)
      {
         GetFaceVDofs(fcs[edge], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();

         B_local[edge].SetSize(ndof_u, ndof_e1);
         C_local[edge].SetSize(ndof_e1, ndof_u);
         D_local[edge].SetSize(ndof_e1, ndof_e1);

         B_local[edge] = 0.0;
         C_local[edge] = 0.0;
         D_local[edge] = 0.0;
         compute_face_integrals(i, fcs[edge], false,
                                &A_local, &B_local[edge], &C_local[edge], &D_local[edge]);
      }

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
         for (int edge=0; edge<no_faces; edge++)
         {
            B_local_data = B_local[edge].GetData();
            GetFaceVDofs(fcs[edge], vdofs_e1);

            for (int j = 0; j<(ndof_u*(vdofs_e1.Size())); j++)
            {
               B_data[B_offsets[i] + size_B_copied + j] = B_local_data[j];
            }

            size_B_copied += ndof_u*(vdofs_e1.Size());
         }
      }

      for (int edge=0; edge<no_faces; edge++)
      {
         GetFaceVDofs(fcs[edge], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();
         G_local.SetSize(ndof_e1);
         G_local = 0.0;
         if (elimBC == 1)
         {
            Eliminate_BC(vdofs_e1, ndof_u, sol, &F_local, &G_local,
                         &B_local[edge], &C_local[edge], &D_local[edge]);
         }

         rhs_SC->AddElementVector(vdofs_e1, 1.0, G_local);
      }

      AinvF.SetSize(ndof_u);
      A_local.Mult(F_local, AinvF);

      // Loop over all the possible face pairs
      for (int edge_i=0; edge_i<no_faces; edge_i++)
      {
         GetFaceVDofs(fcs[edge_i], vdofs_e1);
         ndof_e1 = vdofs_e1.Size();
         D_local[edge_i].Threshold(1.0e-16);
         mat->AddSubMatrix(vdofs_e1, vdofs_e1, D_local[edge_i], skip_zeros);

         CAinvF.SetSize(ndof_e1);
         C_local[edge_i].Mult(AinvF, CAinvF);
         rhs_SC->AddElementVector(vdofs_e1, 1.0, CAinvF);

         CA_local.SetSize(ndof_e1, ndof_u);
         Mult(C_local[edge_i], A_local, CA_local);

         for (int edge_j=0; edge_j<no_faces; edge_j++)
         {
            // Get the unknowns belonging to the edge
            GetFaceVDofs(fcs[edge_j], vdofs_e2);
            ndof_e2 = vdofs_e2.Size();

            SC_local.SetSize(ndof_e1, ndof_e2);

            // Compute the product that will be added to the Schur complement
            Mult(CA_local, B_local[edge_j], SC_local);

            SC_local.Threshold(1.0e-16);
            mat->AddSubMatrix(vdofs_e1, vdofs_e2, SC_local, skip_zeros);
         }
      }

      delete [] B_local;
      delete [] C_local;
      delete [] D_local;
   }

}

// Eliminate the boundary condition from B, C and D
void HDGBilinearForm::Eliminate_BC(const Array<int> &vdofs_e1,
                                   const int ndof_u, const GridFunction &sol,
                                   Vector *rhs_F, Vector *rhs_G, DenseMatrix *B_local,
                                   DenseMatrix *C_local, DenseMatrix *D_local)
{
   int ndof_e1 = vdofs_e1.Size();
   double solution;

   // First we set the BC on the rhs vector for the unknowns on the boundary
   for (int j = 0; j < ndof_e1; j++) // j is the column
   {
      if (ess_dofs[vdofs_e1[j]] < 0)
      {
         (*rhs_G)(j) = sol(vdofs_e1[j]);
         for (int i = 0; i < ndof_e1; i++)
         {
            (*D_local)(j,i) = (i == j);
         }
      }
   }

   // Eliminate BC from B, C and D
   // From D we have to eliminate only the rows that do not belong to a boundary unknown,
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
               (*rhs_G)(i) -= solution * (*D_local)(i,j);
            }

            (*D_local)(i,j) = (i==j);
         }

         for (int i = 0; i < ndof_u; i++)
         {
            (*rhs_F)(i) -= solution * (*B_local)(i,j);
            (*B_local)(i,j) = 0.0;
            (*C_local)(j,i) = 0.0;
         }
      }
   }
}


void HDGBilinearForm::Reconstruct(GridFunction *F,
                                  const GridFunction *ubar,
                                  GridFunction *u)
{
   Array<GridFunction*> RHSGridFunctions, FacetGridFunctions, SolGridFunctions;
   RHSGridFunctions.SetSize(1);
   RHSGridFunctions[0] = F;
   SolGridFunctions.SetSize(1);
   SolGridFunctions[0] = u;
   Reconstruct(RHSGridFunctions, ubar, SolGridFunctions);
}

void HDGBilinearForm::Reconstruct(GridFunction *F1,
                                  GridFunction *F2,
                                  const GridFunction *ubar,
                                  GridFunction *q,
                                  GridFunction *u)
{
   Array<GridFunction*> RHSGridFunctions, FacetGridFunctions, SolGridFunctions;
   RHSGridFunctions.SetSize(2);
   RHSGridFunctions[0] = F1;
   RHSGridFunctions[1] = F2;
   SolGridFunctions.SetSize(2);
   SolGridFunctions[0] = q;
   SolGridFunctions[1] = u;
   Reconstruct(RHSGridFunctions, ubar, SolGridFunctions);
}



// Reconstruct u from the facet solution
void HDGBilinearForm::Reconstruct(Array<GridFunction*> &F,
                                  const GridFunction *ubar,
                                  Array<GridFunction*> &u)
{
   DenseMatrix A_local;
   Vector u_local, F_local, ubar_local, Bubar_local;

   Array<int> fcs;

   Array<int> vdofs_u, vdofs_e;
   int ndof_u, ndof_e;

   DenseMatrix dummy_DM;
   DenseMatrix *B_local;

   for (int i=0; i< volume_fes[0]->GetNE(); i++)
   {
      GetInteriorVDofs(i, vdofs_u);
      ndof_u  = vdofs_u.Size();

      // Set A_local and compute the domain integrals
      A_local.SetSize(ndof_u, ndof_u);
      A_local = 0.0;
      // If A is not saved then compute the domain integrals
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

      GetInteriorSubVector(F, i, ndof_u, F_local);


      for (int edge=0; edge<no_faces; edge++)
      {
         GetFaceVDofs(fcs[edge], vdofs_e);
         ndof_e = vdofs_e.Size();

         B_local[edge].SetSize(ndof_u, ndof_e);
         B_local[edge] = 0.0;
         // If B is not saved then compute the face integrals
         // otherwise load the matrices
         if (i>=elements_B)
         {
            compute_face_integrals(i, fcs[edge], true,
                                   &A_local, &B_local[edge], &dummy_DM, &dummy_DM);
         }
         else
         {
            for (int row = 0; row < ndof_e; row++)
               for (int col = 0; col < (ndof_u); col++)
               {
                  (B_local[edge])(col,row) = B_data[B_offsets[i] + B_values_read + row*ndof_u +
                                                    col];
               }

            B_values_read += ndof_u*ndof_e;
         }

         ubar_local.SetSize(ndof_e);

         ubar->GetSubVector(vdofs_e, ubar_local);
         B_local[edge].Mult(ubar_local, Bubar_local);

         F_local.Add(-1.0, Bubar_local);
      }

      F_local *= -1.0;
      // Compute A^{-1} if it is not saved or just load it
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

      u_local.SetSize(ndof_u);
      A_local.Mult(F_local, u_local);

      SetInteriorSubVector(u, i, ndof_u, u_local);

      delete [] B_local;
   }

}


#ifdef MFEM_USE_MPI
HypreParMatrix *HDGBilinearForm::ParallelAssembleSC(SparseMatrix *m)
{
   if (m == NULL) { return NULL; }

   MFEM_VERIFY(m->Finalized(), "local matrix needs to be finalized for "
               "ParallelAssemble3");

   ParFiniteElementSpace* pfes2 = dynamic_cast<ParFiniteElementSpace*>
                                  (skeletal_fes[0]);

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

HypreParVector *HDGBilinearForm::ParallelVectorSC()
{
   ParFiniteElementSpace* pfes = dynamic_cast<ParFiniteElementSpace*>
                                 (skeletal_fes[0]);
   HypreParVector *tv = pfes->NewTrueDofVector();

   pfes->Dof_TrueDof_Matrix()->MultTranspose(*rhs_SC, *tv);
   return tv;
}

#endif

}
