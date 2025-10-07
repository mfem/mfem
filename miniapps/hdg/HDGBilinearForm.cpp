// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "../../fem/fem.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std;

HDGBilinearForm::HDGBilinearForm (Array<FiniteElementSpace*> &_fes1,
                                  Array<FiniteElementSpace*> &_fes2,
                                  bool _parallel)
{
   NVolumeFES = _fes1.Size();
   NSkeletalFES = _fes2.Size();
   volume_fes.SetSize(NVolumeFES);
   skeletal_fes.SetSize(NSkeletalFES);
   rhs_SC.SetSize(NSkeletalFES);
   mat.SetSize(NSkeletalFES * NSkeletalFES);

   for (int fes=0; fes < NVolumeFES; fes++)
   {
      volume_fes[fes] = _fes1[fes];
   }
   for (int fes=0; fes < NSkeletalFES; fes++)
   {
      skeletal_fes[fes] = _fes2[fes];
      rhs_SC[fes] = NULL;
   }
   for (int fes=0; fes < NSkeletalFES*NSkeletalFES; fes++)
   {
      mat[fes] = NULL;
   }

   parallel = _parallel;
   el_to_face = NULL;

   A_data = NULL;
   B_data = NULL;
   elements_A = elements_B = 0;
}

HDGBilinearForm::HDGBilinearForm(FiniteElementSpace *_fes1,
                                 FiniteElementSpace *_fes2,
                                 bool _parallel)
{
   NVolumeFES = 1;
   NSkeletalFES = 1;
   volume_fes.SetSize(NVolumeFES);
   skeletal_fes.SetSize(NSkeletalFES);

   rhs_SC.SetSize(NSkeletalFES);
   mat.SetSize(NSkeletalFES * NSkeletalFES);

   volume_fes[0] = _fes1;

   skeletal_fes[0] = _fes2;

   parallel = _parallel;
   mat[0] = NULL;
   rhs_SC[0] = NULL;
   el_to_face = NULL;

   A_data = NULL;
   B_data = NULL;
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

   rhs_SC.SetSize(NSkeletalFES);
   mat.SetSize(NSkeletalFES * NSkeletalFES);


   volume_fes[0] = _fes1;
   volume_fes[1] = _fes2;

   skeletal_fes[0] = _fes3;

   parallel = _parallel;
   mat[0] = NULL;
   rhs_SC[0] = NULL;
   el_to_face = NULL;

   A_data = NULL;
   B_data = NULL;
   elements_A = elements_B = 0;
}

HDGBilinearForm::~HDGBilinearForm()
{
   for (int fes=0; fes < NSkeletalFES; fes++)
   {
      delete rhs_SC[fes];
   }
   for (int fes=0; fes < NSkeletalFES*NSkeletalFES; fes++)
   {
      delete mat[fes];
   }

   int k;
   for (k=0; k < hdg_dbfi.Size(); k++)
   {
      delete hdg_dbfi[k];
   }
   for (k=0; k < hdg_fbfi.Size(); k++)
   {
      delete hdg_fbfi[k];
   }
}

void HDGBilinearForm::Finalize (int skip_zeros)
{
   for (int fes=0; fes < NSkeletalFES*NSkeletalFES; fes++)
   {
      mat[fes] -> Finalize(skip_zeros);
   }
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

   for (int fes=0; fes < NSkeletalFES*NSkeletalFES; fes++)
   {
      delete mat[fes];
      mat[fes] = NULL;
   }

   for (int fes=0; fes < NSkeletalFES; fes++)
   {
      delete rhs_SC[fes];
      rhs_SC[fes] = NULL;
   }

   el_to_face = NULL;
}


// Loops over all volume FES and collects the dofs into the vdofs array.
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

/* Loops over all volume FES and collects the subvector from the correspoinding FES and appends it to SubVector
 */
void HDGBilinearForm::GetInteriorSubVector(const Array<GridFunction*>
                                           &rhs_gridfunctions,
                                           int i, int ndof, Vector &SubVector) const
{
   SubVector.SetSize(ndof);
   Vector LocalVector;
   Array<int> vdofs_fes;

   int counter = 0;
   for (int fes=0; fes < NVolumeFES; fes++)
   {
      volume_fes[fes]->GetElementVDofs(i, vdofs_fes);
      rhs_gridfunctions[fes]->GetSubVector(vdofs_fes, LocalVector);
      for (int k = 0; k<vdofs_fes.Size(); k++)
      {
         SubVector(counter + k) = LocalVector(k);
      }

      counter += vdofs_fes.Size();
   }
}

/* Loops over all facet FES and collects the subvector from the correspoinding FES and appends it to SubVector
 */
void HDGBilinearForm::GetFaceSubVector(const Array<GridFunction*>
                                       &face_gridfunctions,
                                       int i, int ndof, Vector &SubVector) const
{
   SubVector.SetSize(ndof);
   Vector LocalVector;
   Array<int> vdofs_fes;

   int counter = 0;
   for (int fes=0; fes < NSkeletalFES; fes++)
   {
      skeletal_fes[fes]->GetFaceVDofs(i, vdofs_fes);
      face_gridfunctions[fes]->GetSubVector(vdofs_fes, LocalVector);
      for (int k = 0; k<vdofs_fes.Size(); k++)
      {
         SubVector(counter + k) = LocalVector(k);
      }

      counter += vdofs_fes.Size();
   }
}

/* Loops over all volume FES and sets the subvector from the correspoinding FES by the corresponding parts of SubVector
 */
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

/* Loops over all volume FES and collects the dofs into the vdofs array.
* Also creates an array containing the lengths of the local dofs for every FES.
* This additional information is used in the Eliminate_BC function
*/
void HDGBilinearForm::GetFaceVDofs(int i, Array<int> &vdofs,
                                   Array<int> &dof_length) const
{
   vdofs.SetSize(0);
   Array<int> vdofs_fes;
   dof_length.SetSize(NSkeletalFES);

   for (int fes=0; fes < NSkeletalFES; fes++)
   {
      skeletal_fes[fes]->GetFaceVDofs(i, vdofs_fes);
      vdofs.Append(vdofs_fes);
      dof_length[fes] = vdofs_fes.Size();
   }

}

/* Loops over all volume FES and collects the dofs into the vdofs array.
* This version does not create an array containing the lengths of the local dofs for every FES.
*/
void HDGBilinearForm::GetFaceVDofs(int i, Array<int> &vdofs) const
{
   Array<int> dummy;

   GetFaceVDofs(i, vdofs, dummy);
}

// compute all the domain based integrals in one loop over the quadrature nodes
void HDGBilinearForm::compute_domain_integrals(const int elem,
                                               DenseMatrix *A_local)
{
   // get the element transformation and the finite elements for the variables
   ElementTransformation *eltrans;
   eltrans = volume_fes[0]->GetElementTransformation(elem);
   const FiniteElement &fe_volume = *volume_fes[0]->GetFE(elem);
   // CAUTION: using the fact that all volume FiniteElements have the same order

   // compute the integrals
   switch (NVolumeFES)
   {
      case 1:
      {
         // compute the integrals
         hdg_dbfi[0]->AssembleElementMatrix(fe_volume, *eltrans, elemmat1);
         break;
      }
      case 2:
      {
         const FiniteElement &fe_volume2 = *volume_fes[1]->GetFE(elem);
         // compute the integrals
         hdg_dbfi[0]->AssembleElementMatrix2FES(fe_volume, fe_volume2, *eltrans,
                                                elemmat1);
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

/* Compute the face integrals for A, B, C and D
 *The bool reconstruct_only should be false when creating the Schur complement, true when reconstructing u from ubar.
 */
void HDGBilinearForm::compute_face_integrals(const int elem, const int edge,
                                             const int isshared,
                                             const bool reconstruct_only,
                                             DenseMatrix *A_local,
                                             DenseMatrix *B_local,
                                             DenseMatrix *C_local,
                                             DenseMatrix *D_local)
{
   FaceElementTransformations *tr = NULL;

   if (isshared == -1)
   {
      Mesh *mesh = volume_fes[0] -> GetMesh();
      tr = mesh->GetFaceElementTransformations(edge);
   }
   else
   {
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace* pfes1 = dynamic_cast<ParFiniteElementSpace*>
                                     (volume_fes[0]);

      // do we need the next line?
      pfes1->ExchangeFaceNbrData();
      ParMesh *pmesh = pfes1 -> GetParMesh();
      // in the case of a shared mesh the 3rd input is isshared, not isboundary
      tr = pmesh->GetSharedFaceTransformations(isshared);
#endif
   }

   // CAUTION: using the fact that all volume FiniteElements have the same order
   const FiniteElement &trace_fe = *skeletal_fes[0]->GetFaceElement(edge);
   const FiniteElement &volume_fe = *volume_fes[0]->GetFE(elem);

   // Compute the integrals depending on which element do we need to use
   int elem_1or2 = 1 + (tr->Elem2No == elem);

   switch (NVolumeFES)
   {
      case 1:
      {
         hdg_fbfi[0]->AssembleFaceMatrixOneElement1and1FES(volume_fe,
                                                           trace_fe,
                                                           *tr, elem_1or2, reconstruct_only,
                                                           elemmat1, elemmat2,
                                                           elemmat3, elemmat4);
         break;
      }
      case 2:
      {
         const FiniteElement &volume_fe2 = *volume_fes[1]->GetFE(elem);
         hdg_fbfi[0]->AssembleFaceMatrixOneElement2and1FES(volume_fe, volume_fe2,
                                                           trace_fe,
                                                           *tr, elem_1or2, reconstruct_only,
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
   if (!reconstruct_only)
   {
      C_local->Add(1.0, elemmat3);
      D_local->Add(1.0, elemmat4);
   }

}

/* To allocate the sparse matrix and the right hand side vector and to create the Edge_to_SharedEdge and el_to_face tables.
 * This is also called for the parallel, since these information are important on every processor.
 * Edge_to_SharedEdge is an Array with size of the number of edges.
 * The entry Edge_to_SharedEdge[i] is -1 if the i-th face is not shared.
 * Edge_to_SharedEdge[i] = n means that the n-th shared face is the i-th face.
 * el_to_faces has number of element rows and the i-th row contains the faces of the i-th element
 */
void HDGBilinearForm::Allocate(const Array<int> &bdr_attr_is_ess,
                               const real_t memA, const real_t memB)
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

   int n_skeleton_elements = mesh->Dimension() == 2 ? mesh->GetNEdges() :
                             mesh->GetNFaces();

   // Edge_to_SharedEdge is an Array with size of the number of edges.
   // The entry Edge_to_SharedEdge[i] is -1 if the i-th face is not shared.
   //Edge_to_SharedEdge[i] = n means that the n-th shared face is the i-th face.
   Edge_to_SharedEdge.SetSize(n_skeleton_elements);
   Edge_to_SharedEdge = -1;
#ifdef MFEM_USE_MPI
   // ExchangeFaceNbrData to be able to use shared faces
   ParFiniteElementSpace* pfes1 = dynamic_cast<ParFiniteElementSpace*>
                                  (volume_fes[0]);

   if (parallel)
   {
      pfes1->ExchangeFaceNbrData();
      int nsharedfaces = pfes1 -> GetParMesh()->GetNSharedFaces();

      // Create an array to identify the shared faces. The entry is one of the face is not shared,
      // otherwise, is gives the number of the face in the shared face list, so that GetSharedFaceTransformation
      // can be used.
      for (int i = 0; i < nsharedfaces; i++)
      {
         Edge_to_SharedEdge[pfes1 -> GetParMesh()->GetSharedFace(i)] = i;
      }
   }
#endif

   // setting up the matrices and the right hand side vectors
   for (int i = 0; i< NSkeletalFES*NSkeletalFES; i++)
   {
      if (mat[i] == NULL)
      {
         int col = i%NSkeletalFES;
         int row = (i - col)/NSkeletalFES;

         mat[i] = new SparseMatrix(skeletal_fes[row]->GetVSize(),
                                   skeletal_fes[col]->GetVSize());
      }
   }

   for (int i = 0; i< NSkeletalFES; i++)
   {
      if (rhs_SC[i] == NULL)
      {
         rhs_SC[i] = new Vector(skeletal_fes[i]->GetVSize());
         *rhs_SC[i] = 0.0;
      }
   }

   // setting up the essential dofs
   if (bdr_attr_is_ess.Size())
   {
      skeletal_fes[0]->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);
   }

   // Local_A inverse is saved on the first elements_A elements
   // Local_B is saved on the first elements_B elements (all faces of those elements)
   elements_A = (int)(memA * volume_fes[0]->GetNE());
   elements_B = (int)(memB * volume_fes[0]->GetNE());

   // Set the offset vectors
   A_offsets.SetSize(elements_A+1);
   B_offsets.SetSize(elements_B+1);
   A_offsets[0] = 0;
   B_offsets[0] = 0;

   Array<int> vdofs_volume, vdofs_edge, fcs;
   int ndof_volume;

   // loop over the elements to find the offset entries
   for (int i=0; i< volume_fes[0]->GetNE(); i++)
   {
      // Get the local number of dof for the volume unknowns
      GetInteriorVDofs(i, vdofs_volume);
      ndof_volume  = vdofs_volume.Size();

      // A will have the size ndof_volume * ndof_volume
      // The next offset entry can be set
      if (i < elements_A)
      {
         A_offsets[i+1] = A_offsets[i] + ndof_volume * ndof_volume;
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
            GetFaceVDofs(fcs[edge1], vdofs_edge);
            ndof_edge_all += vdofs_edge.Size();
         }

         // B will have the size ndof_volume * ndof_edge
         B_offsets[i+1] = B_offsets[i] + ndof_volume*ndof_edge_all;
      }

      // If i >= elements_A then i >= elements_B also, so the for loop can be cancelled
      if (i >= elements_A)
      {
         break;
      }
   }

   // Create A_data and B_data as a vector with the proper size
   A_data = new real_t[A_offsets[elements_A]];
   B_data = new real_t[B_offsets[elements_B]];

}

/// Assembles the Schur complement - advection-reaction example
void HDGBilinearForm::AssembleSC(GridFunction *F,
                                 const real_t memA,
                                 const real_t memB,
                                 int skip_zeros)
{
   Array<GridFunction*> rhs_F;
   rhs_F.SetSize(1);
   rhs_F[0] = F;

   Array<GridFunction*> rhs_G;
   rhs_G.SetSize(0); // dummy

   Array<int> bdr_attr_is_ess;
   bdr_attr_is_ess.SetSize(0);
   bdr_attr_is_ess = 0; // dummy

   Array<GridFunction*> sol;
   sol.SetSize(0);

   AssembleReconstruct(rhs_F, rhs_G,
                       bdr_attr_is_ess, sol,
                       true, memA, memB, skip_zeros);

}

/// Assembles the Schur complement - diffusion example
void HDGBilinearForm::AssembleSC(GridFunction *F1,
                                 GridFunction *F2,
                                 Array<int> &bdr_attr_is_ess,
                                 GridFunction &sol,
                                 const real_t memA,
                                 const real_t memB,
                                 int skip_zeros)
{
   Array<GridFunction*> rhs_F;
   rhs_F.SetSize(2);
   rhs_F[0] = F1;
   rhs_F[1] = F2;

   Array<GridFunction*> rhs_G;
   rhs_G.SetSize(0); // dummy

   Array<GridFunction*> solution;
   solution.SetSize(1);
   solution[0] = &sol;

   AssembleReconstruct(rhs_F, rhs_G,
                       bdr_attr_is_ess, solution,
                       true, memA, memB, skip_zeros);
}

/// Assembles the Schur complement - general setup
void HDGBilinearForm::AssembleSC(Array<GridFunction*> rhs_F,
                                 Array<GridFunction*> rhs_G,
                                 const Array<int> &bdr_attr_is_ess,
                                 Array<GridFunction*> sol,
                                 int skip_zeros)
{
   AssembleReconstruct(rhs_F, rhs_G,
                       bdr_attr_is_ess, sol,
                       true, skip_zeros);
}

// Eliminate the boundary condition from B, C and D
// We are eliminating the rows as well, not just the columns, so B and rhs_Volume change as well.
void HDGBilinearForm::Eliminate_BC(const Array<int> &vdofs_e1,
                                   const Array<int> &vdofs_e1_length,
                                   const int ndof_u, Array<GridFunction*> sol,
                                   Vector *rhs_Volume, Vector *rhs_Skeleton, DenseMatrix *B_local,
                                   DenseMatrix *C_local, DenseMatrix *D_local)
{
   // To get the separators: vdofs_e1 contains the dofs for all skeletal FES on the given edge
   // vdofs_e1_length contains the number of dofs for the different FES on the edge
   // We create a partial sum vector starting at 0, so we have the separators of the different dofs in vdofs_e1
   Array<int> vdofs_e1_PS(
      vdofs_e1_length); // array for the partial sum without the 0
   vdofs_e1_PS.PartialSum();
   vdofs_e1_PS.Prepend(0);

   // At this point vdofs_e1_PS have
   // first entry:  0
   // second entry: lenght of DoFs for 1st skeletal space
   // third entry:  sum of lenghts of DoFs for 1st and 2nd skeletal space
   // i-th entry:   sum of lenghts of DoFs for skeletal spaces from fists to (i-1)st

   int ndof_e = vdofs_e1.Size();
   Array<int> local_vdof;

   // loop over all skeletal FES:
   // get the dofs for the skeletal space
   // eliminate if the dof is essential
   for (int sk_fes = 0; sk_fes < NSkeletalFES; sk_fes++)
   {
      int local_size = vdofs_e1_length[sk_fes];
      local_vdof.SetSize(local_size);

      for (int i= 0; i< local_size; i++)
      {
         local_vdof[i] = vdofs_e1[i + vdofs_e1_PS[sk_fes]];
      }
      // At this point local_vdof contains the dofs for the skeletal FES sk_fes

      real_t solution;

      // loop over the local_vdof and eliminate from C, D, and the rhs if necessary
      for (int j = 0; j < local_size; j++) // j is the column
      {
         if (ess_dofs[local_vdof[j]] < 0)
         {
            (*rhs_Skeleton)(j+vdofs_e1_PS[sk_fes]) = (*sol[sk_fes])(local_vdof[j]);
            // eliminate the row from D
            for (int i = 0; i < ndof_e; i++)
            {
               (*D_local)(j+vdofs_e1_PS[sk_fes],i) = (i == (j+vdofs_e1_PS[sk_fes]));
            }
         }
      }

      // Eliminate BC from B, C and D
      // From D we have to eliminate only the rows that do not belong to a boundary unknown,
      // since those values or the RHS are already set.
      // TODO: this needs more explanation
      for (int j = 0; j < ndof_e; j++) // j is the column
      {
         if (ess_dofs[vdofs_e1[j]] < 0)
         {
            solution = (*sol[sk_fes])(local_vdof[j]);
            for (int i = 0; i < ndof_e; i++)
            {
               if (!(ess_dofs[vdofs_e1[i]] < 0))
               {
                  (*rhs_Skeleton)(i+vdofs_e1_PS[sk_fes]) -= solution * (*D_local)(i,
                                                                                  j+vdofs_e1_PS[sk_fes]);
               }

               (*D_local)(i, j+vdofs_e1_PS[sk_fes]) = (i == (j+vdofs_e1_PS[sk_fes]));
            }

            for (int i = 0; i < ndof_u; i++)
            {
               (*rhs_Volume)(i) -= solution * (*B_local)(i,j+vdofs_e1_PS[sk_fes]);
               (*B_local)(i,j+vdofs_e1_PS[sk_fes]) = 0.0;
               (*C_local)(j+vdofs_e1_PS[sk_fes],i) = 0.0;
            }
         }
      }

   }
}

/// Reconstruction - advection-reaction example
void HDGBilinearForm::Reconstruct(GridFunction *F,
                                  GridFunction *ubar,
                                  GridFunction *u)
{
   Array<GridFunction*> RHSGridFunctions, FacetGridFunctions, SolGridFunctions;
   RHSGridFunctions.SetSize(1);
   RHSGridFunctions[0] = F;
   SolGridFunctions.SetSize(1);
   SolGridFunctions[0] = u;
   FacetGridFunctions.SetSize(1);
   FacetGridFunctions[0] = ubar;

   Array<int> dummy_Array;
   AssembleReconstruct(RHSGridFunctions, FacetGridFunctions,
                       dummy_Array, SolGridFunctions,
                       false);

}

/// Reconstruction - diffusion example
void HDGBilinearForm::Reconstruct(GridFunction *F1,
                                  GridFunction *F2,
                                  GridFunction *ubar,
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

   FacetGridFunctions.SetSize(1);
   FacetGridFunctions[0] = ubar;

   Array<int> dummy_Array;
   AssembleReconstruct(RHSGridFunctions, FacetGridFunctions,
                       dummy_Array, SolGridFunctions,
                       false);
}

// Reconstruct the volume unknowns from the facet solution
void HDGBilinearForm::Reconstruct(Array<GridFunction*> Volume_GF,
                                  Array<GridFunction*> Skeleton_GF,
                                  Array<GridFunction*> u)
{
   Array<int> dummy_Array;
   AssembleReconstruct(Volume_GF, Skeleton_GF,
                       dummy_Array, u,
                       false);
}

// Add the local vector to the proper part of the RHS vector
void HDGBilinearForm::AddToRHS(Array<int> &skeletal_vdofs,
                               Array<int> &skeletal_vdof_length, Vector v_add)
{
   // create the partial sum vector starting with 0 (these are the separators)
   Array<int> skeletal_vdof_separators(skeletal_vdof_length);
   skeletal_vdof_separators.PartialSum();
   skeletal_vdof_separators.Prepend(0);
   Vector auxVector;
   Array<int> local_vdof;

   // At this point skeletal_vdof_separators have
   // first entry:  0
   // second entry: lenght of DoFs for 1st skeletal space
   // third entry:  sum of lenghts of DoFs for 1st and 2nd skeletal space
   // i-th entry:   sum of lenghts of DoFs for skeletal spaces from fists to (i-1)st

   // Loop over all skeletal spaces
   for (int sk_fes = 0; sk_fes < NSkeletalFES; sk_fes++)
   {
      // Get the correspoinding DoFs
      int local_size = skeletal_vdof_length[sk_fes];
      auxVector.SetSize(local_size);

      // Copy the corresponding part of the input Vector into auxVector
      local_vdof.SetSize(local_size);
      for (int i= 0; i< local_size; i++)
      {
         auxVector[i] = v_add[i + skeletal_vdof_separators[sk_fes]];
         local_vdof[i] = skeletal_vdofs[i + skeletal_vdof_separators[sk_fes]];
      }

      // Add auxVector to the correspoinding right hand side vector
      rhs_SC[sk_fes]->AddElementVector(local_vdof, 1.0, auxVector);
   }
}

// Add the local matrix to the proper part of the Schur complement
void HDGBilinearForm::AddToMat(Array<int> &skeletal_vdofs_edge_i,
                               Array<int> &skeletal_vdof_length_edge_i,
                               Array<int> &skeletal_vdofs_edge_j, Array<int> &skeletal_vdof_length_edge_j,
                               DenseMatrix dm_add, int skip_zeros)
{
   // create the partial sum vector starting with 0 for edge_i (these are the separators)
   Array<int> skeletal_vdof_edge_i_separators(
      skeletal_vdof_length_edge_i); // array for the partial sum without the 0
   skeletal_vdof_edge_i_separators.PartialSum();
   skeletal_vdof_edge_i_separators.Prepend(0);

   // create the partial sum vector starting with 0 for edge_j (these are the separators)
   Array<int> skeletal_vdof_edge_j_separators(
      skeletal_vdof_length_edge_j); // array for the partial sum without the 0
   skeletal_vdof_edge_j_separators.PartialSum();
   skeletal_vdof_edge_j_separators.Prepend(0);

   // At this point both skeletal_vdof_edge i/j separators have
   // first entry:  0
   // second entry: lenght of DoFs for 1st skeletal space
   // third entry:  sum of lenghts of DoFs for 1st and 2nd skeletal space
   // i-th entry:   sum of lenghts of DoFs for skeletal spaces from fists to (i-1)st

   DenseMatrix auxDenseMatrix;
   Array<int> local_vdof_edge_i, local_vdof_edge_j;

   // Loop over all skeletal spaces - this will refer to the rows of the block sparse matrix
   for (int sk_fes_row = 0; sk_fes_row < NSkeletalFES; sk_fes_row++)
   {
      // Get the correspoinding DoFs
      int local_size_edge_i = skeletal_vdof_length_edge_i[sk_fes_row];
      local_vdof_edge_i.SetSize(local_size_edge_i);
      for (int i= 0; i< local_size_edge_i; i++)
      {
         local_vdof_edge_i[i] = skeletal_vdofs_edge_i[i +
                                                      skeletal_vdof_edge_i_separators[sk_fes_row]];
      }
      // Loop over all skeletal spaces - this will refer to the columns of the block sparse matrix
      for (int sk_fes_col = 0; sk_fes_col < NSkeletalFES; sk_fes_col++)
      {
         int local_size_edge_j = skeletal_vdof_length_edge_j[sk_fes_col];
         local_vdof_edge_j.SetSize(local_size_edge_j);
         for (int i= 0; i< local_size_edge_j; i++)
         {
            local_vdof_edge_j[i] = skeletal_vdofs_edge_j[i +
                                                         skeletal_vdof_edge_j_separators[sk_fes_col]];
         }

         // Copy the corresponding part of the input DenseMatrix into auxDenseMatrix
         auxDenseMatrix.SetSize(local_size_edge_i, local_size_edge_j);
         auxDenseMatrix.CopyMN(dm_add, local_size_edge_i, local_size_edge_j,
                               skeletal_vdof_edge_i_separators[sk_fes_row],
                               skeletal_vdof_edge_j_separators[sk_fes_col]);

         // Add auxDenseMatrix to the correspoinding SparseMatrix
         mat[sk_fes_row*NSkeletalFES + sk_fes_col]->AddSubMatrix(local_vdof_edge_i,
                                                                 local_vdof_edge_j, auxDenseMatrix, skip_zeros);
      }
   }
}

DenseMatrix HDGBilinearForm::CalculateInverse(DenseMatrix A_local)
{
   // This function can be used is A has a special structure that makes
   // the inversion cheaper
   int size = A_local.Width();
   DenseMatrix A_local_inv(size);
   A_local_inv = 0.0;

   A_local_inv.Add(1.0, A_local);
   A_local_inv.Invert();

   return A_local_inv;
}

#ifdef MFEM_USE_MPI
// 2025 Sept begins
HypreParMatrix *HDGBilinearForm::ParallelAssemble(int i, SparseMatrix *m)
{
   OperatorHandle Mh(Operator::Hypre_ParCSR);
   ParallelAssemble(i, Mh, m);
   Mh.SetOperatorOwner(false);
   return Mh.As<HypreParMatrix>();
}

void HDGBilinearForm::ParallelAssemble(int i, OperatorHandle &A,
                                       SparseMatrix *m)
{
   A.Clear();

   ParFiniteElementSpace* pfes = NULL;

   MFEM_VERIFY(m->Finalized(), "local matrix needs to be finalized for "
               "ParallelAssemble3");

   OperatorHandle dA(A.Type()), Ph(A.Type()), hdA;

   if (i%(NSkeletalFES + 1) == 0)
   {
      // diagonal terms this is based on ParBilinearform->ParallelAssemble
      pfes = dynamic_cast<ParFiniteElementSpace*>(skeletal_fes[i/(NSkeletalFES + 1)]);

      int lvsize = pfes->GetVSize();
      const HYPRE_BigInt *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      HYPRE_BigInt ldof_offset = pfes->GetMyDofOffset();

      Array<HYPRE_BigInt> glob_J(m->NumNonZeroElems());
      int *J = m->GetJ();
      for (int ii = 0; ii < glob_J.Size(); ii++)
      {
         if (J[ii] < lvsize)
         {
            glob_J[ii] = J[ii] + ldof_offset;
         }
         else
         {
            glob_J[ii] = face_nbr_glob_ldof[J[ii] - lvsize];
         }
      }

      // TODO - construct dA directly in the A format
      hdA.Reset(
         new HypreParMatrix(pfes->GetComm(), lvsize,
                            pfes->GlobalVSize(),
                            pfes->GlobalVSize(), m->GetI(), glob_J,
                            m->GetData(), pfes->GetDofOffsets(),
                            pfes->GetDofOffsets()));

      // - hdA owns the new HypreParMatrix
      // - the above constructor copies all input arrays
      glob_J.DeleteAll();

      cout << "HDGBilinearForm::ParallelAssemble before dA.ConvertFrom(hdA);" << endl
           << flush;
      // this line fails
      dA.ConvertFrom(hdA);
      cout << "HDGBilinearForm::ParallelAssemble after da.ConvertFrom(hdA);" << endl
           << flush;

      Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());

      A.MakePtAP(dA, Ph);
   }
   else
   {
      // diagonal terms this is based on ParMixedBilinearform->ParallelAssemble
      int col = i%NSkeletalFES;
      int row = (i - col)/NSkeletalFES;
      pfes = dynamic_cast<ParFiniteElementSpace*>(skeletal_fes[row]);
      ParFiniteElementSpace* pfes2 = dynamic_cast<ParFiniteElementSpace*>
                                     (skeletal_fes[col]);

      OperatorHandle dA(A.Type());
      dA.MakeRectangularBlockDiag(pfes2->GetComm(),
                                  pfes->GlobalVSize(),
                                  pfes2->GlobalVSize(),
                                  pfes->GetDofOffsets(),
                                  pfes2->GetDofOffsets(),
                                  m);

      OperatorHandle P_test(A.Type()), P_trial(A.Type());

      // TODO - construct the Dof_TrueDof_Matrix directly in the required format.
      P_test.ConvertFrom(pfes->Dof_TrueDof_Matrix());
      P_trial.ConvertFrom(pfes2->Dof_TrueDof_Matrix());

      A.MakeRAP(P_test, dA, P_trial);
   }

}
// 2025 Seot ends


HypreParMatrix *HDGBilinearForm::ParallelAssembleSC(int i, SparseMatrix *m)
{
   if (m == NULL)
   {
      return NULL;
   }

   MFEM_VERIFY(m->Finalized(), "local matrix needs to be finalized for "
               "ParallelAssemble3");

   ParFiniteElementSpace* pfes = NULL;
   HypreParMatrix *A = NULL, *rap = NULL;
   if (i%(NSkeletalFES + 1) == 0)
   {
      // diagonal terms this is based on ParBilinearform->ParallelAssemble
      pfes = dynamic_cast<ParFiniteElementSpace*>(skeletal_fes[i/(NSkeletalFES + 1)]);

      int lvsize = pfes->GetVSize();
      const HYPRE_Int *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      HYPRE_Int ldof_offset = pfes->GetMyDofOffset();

      Array<HYPRE_Int> glob_J(m->NumNonZeroElems());
      int *J = m->GetJ();
      for (int ii = 0; ii < glob_J.Size(); ii++)
      {
         if (J[ii] < lvsize)
         {
            glob_J[ii] = J[ii] + ldof_offset;
         }
         else
         {
            glob_J[ii] = face_nbr_glob_ldof[J[ii] - lvsize];
         }
      }

      A = new HypreParMatrix(pfes->GetComm(), lvsize,
                             pfes->GlobalVSize(),
                             pfes->GlobalVSize(), m->GetI(), glob_J,
                             m->GetData(), pfes->GetDofOffsets(),
                             pfes->GetDofOffsets());

      rap = RAP(A, pfes->Dof_TrueDof_Matrix());
   }
   else
   {
      // diagonal terms this is based on ParMixedBilinearform->ParallelAssemble
      int col = i%NSkeletalFES;
      int row = (i - col)/NSkeletalFES;
      pfes = dynamic_cast<ParFiniteElementSpace*>(skeletal_fes[row]);
      ParFiniteElementSpace* pfes2 = dynamic_cast<ParFiniteElementSpace*>
                                     (skeletal_fes[col]);
      A = new HypreParMatrix(pfes2->GetComm(),
                             pfes->GlobalVSize(),
                             pfes2->GlobalVSize(),
                             pfes->GetDofOffsets(),
                             pfes2->GetDofOffsets(),
                             m);

      rap = RAP(pfes->Dof_TrueDof_Matrix(), A, pfes2->Dof_TrueDof_Matrix());
   }

   delete A;

   return rap;
}

HypreParVector *HDGBilinearForm::ParallelVectorSC(int i)
{
   ParFiniteElementSpace* pfes = dynamic_cast<ParFiniteElementSpace*>
                                 (skeletal_fes[i]);
   HypreParVector *tv = pfes->NewTrueDofVector();

   pfes->Dof_TrueDof_Matrix()->MultTranspose(*rhs_SC[i], *tv);
   return tv;
}

void HDGBilinearForm::ParallelVectorSC(int i, Vector &tv)
{
   ParFiniteElementSpace* pfes = dynamic_cast<ParFiniteElementSpace*>
                                 (skeletal_fes[i]);
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*rhs_SC[i], tv);
}
#endif

// We used "edge" in the documentation for the skeletal mesh elements.
// For 3D calculations, they refer to the "faces" of the mesh.
void HDGBilinearForm::AssembleReconstruct(Array<GridFunction*> Vol_GF,
                                          Array<GridFunction*> Skel_GF,
                                          const Array<int> &bdr_attr_is_ess,
                                          Array<GridFunction*> bdr_sol_sol_GF,
                                          bool assemble,
                                          const real_t memA, const real_t memB,
                                          int skip_zeros)
{
   // Allocate the matrices, right hand sides, and all other necessary objects
   if (assemble)
   {
      Allocate(bdr_attr_is_ess, memA, memB);
   }

   DenseMatrix A_local, CA_local, SC_local;
   Vector F_local, CAinvF, AinvF;

   // The faces (edges) of a given element
   Array<int> fcs;

   // DoF arrays and DoF counters
   Array<int> vdofs_volume, vdofs_edge, vdofs_edge_i, vdofs_edge_j;
   Array<int> vdofs_edge_lenght, vdofs_edge_i_lenght, vdofs_edge_j_lenght;
   int ndof_volume, ndof_edge, ndof_edge_i, ndof_edge_j;

   // Arrays of the local matrices (one on each edge)
   DenseMatrix *B_local;
   DenseMatrix *C_local;
   DenseMatrix *D_local;
   Vector G_local;

   // To save A and B
   real_t *A_local_data, *B_local_data;

   // These are needed for the reconstructions
   Vector B_skeleton_local, skeleton_local, u_local;

   // Loop over the elements
   for (int i=0; i< volume_fes[0]->GetNE(); i++)
   {
      // Get the volume DoFs for a given element
      GetInteriorVDofs(i, vdofs_volume);
      ndof_volume  = vdofs_volume.Size();

      // Set A_local and compute the domain integrals
      A_local.SetSize(ndof_volume, ndof_volume);
      A_local = 0.0;
      // For assembly: compute A
      // For reconstruction: only compute A if it is not stored
      if ((assemble) || (i>=elements_A))
      {
         compute_domain_integrals(i, &A_local);
      }

      // Get the element edges
      el_to_face->GetRow(i, fcs);

      // Set the necessary number of the local edge matrices
      int no_faces = fcs.Size();
      B_local = new DenseMatrix[no_faces];
      C_local = new DenseMatrix[no_faces];
      D_local = new DenseMatrix[no_faces];

      // Get the entries from the volume GFs
      GetInteriorSubVector(Vol_GF, i, ndof_volume, F_local);

      // Loop over the edges of the element and compute the
      // edge integrals for A, B, C and D
      for (int edge=0; edge<no_faces; edge++)
      {
         // Get the skeletal DoFs for a given edge
         GetFaceVDofs(fcs[edge], vdofs_edge);
         ndof_edge = vdofs_edge.Size();

         // Set the sude of the local edge matrices
         B_local[edge].SetSize(ndof_volume, ndof_edge);
         C_local[edge].SetSize(ndof_edge, ndof_volume);
         D_local[edge].SetSize(ndof_edge, ndof_edge);

         B_local[edge] = 0.0;
         C_local[edge] = 0.0;
         D_local[edge] = 0.0;
         // For assembly: compute edge integrals
         // For reconstruction: only compute edge integrals if they are not stored
         if ((assemble) || (i>=elements_B))
            compute_face_integrals(i, fcs[edge], Edge_to_SharedEdge[fcs[edge]],
                                   !assemble,
                                   &A_local, &B_local[edge], &C_local[edge], &D_local[edge]);
      }

      // For assembly: invert A
      // For reconstruction: only compute A if it is not stored
      if ((assemble) || (i>=elements_A))
      {
         A_local = CalculateInverse(A_local);
      }
      else
      {
         for (int row = 0; row < ndof_volume; row++)
            for (int col = 0; col < ndof_volume; col++)
            {
               A_local(col,row) = A_data[A_offsets[i] + row*ndof_volume + col];
            }
      }

      if (assemble)
      {
         // Save A and B if necessary
         if (i<elements_A)
         {
            A_local_data = A_local.GetData();

            for (int j = 0; j<ndof_volume*ndof_volume; j++)
            {
               A_data[A_offsets[i] + j] = A_local_data[j];
            }
         }
         A_local.Neg();

         if (i<elements_B)
         {
            int size_B_copied = 0;
            for (int edge=0; edge<no_faces; edge++)
            {
               B_local_data = B_local[edge].GetData();
               // Get the skeletal DoFs for a given edge
               GetFaceVDofs(fcs[edge], vdofs_edge);

               for (int j = 0; j<(ndof_volume*(vdofs_edge.Size())); j++)
               {
                  B_data[B_offsets[i] + size_B_copied + j] = B_local_data[j];
               }

               size_B_copied += ndof_volume*(vdofs_edge.Size());
            }
         }

         // Loop over the edges, and eliminate if necessary
         for (int edge=0; edge<no_faces; edge++)
         {
            // Get the skeletal DoFs for a given edge
            GetFaceVDofs(fcs[edge], vdofs_edge, vdofs_edge_lenght);
            ndof_edge = vdofs_edge.Size();
            G_local.SetSize(ndof_edge);
            G_local = 0.0;

            if (bdr_attr_is_ess.Size())
            {
               Eliminate_BC(vdofs_edge, vdofs_edge_lenght, ndof_volume, bdr_sol_sol_GF,
                            &F_local, &G_local,
                            &B_local[edge], &C_local[edge], &D_local[edge]);
            }

            AddToRHS(vdofs_edge,  vdofs_edge_lenght, G_local);
         }

         // Calculate A^{-1}F
         AinvF.SetSize(ndof_volume);
         A_local.Mult(F_local, AinvF);

         // Loop over all the possible edge pairs, and added
         // D and the C A^{-1} B terms to the correct matrices
         for (int edge_i=0; edge_i<no_faces; edge_i++)
         {
            // Get the skeletal DoFs for a given edge, with the lengths of the
            // sub arrays - needed for the assembly
            GetFaceVDofs(fcs[edge_i], vdofs_edge_i, vdofs_edge_i_lenght);
            ndof_edge_i = vdofs_edge_i.Size();
            // Remove the meaninglessly small entries from D
            (D_local[edge_i]).Threshold(1.0e-16);

            // Assemble D to the proper sparse matrices
            AddToMat(vdofs_edge_i, vdofs_edge_i_lenght, vdofs_edge_i, vdofs_edge_i_lenght,
                     D_local[edge_i], skip_zeros);

            // Calculate C A^{-1}F
            CAinvF.SetSize(ndof_edge_i);
            (C_local[edge_i]).Mult(AinvF, CAinvF);

            // Add C A^{-1}F to the proper right hand side
            AddToRHS(vdofs_edge_i,  vdofs_edge_i_lenght, CAinvF);

            // C A^{-1} B is calculated in two steps, first calculate C A^{-1}
            CA_local.SetSize(ndof_edge_i, ndof_volume);
            Mult(C_local[edge_i], A_local, CA_local);

            for (int edge_j=0; edge_j<no_faces; edge_j++)
            {
               // Get the skeletal DoFs for a given edge, with the lengths of the
               // sub arrays - needed for the assembly
               GetFaceVDofs(fcs[edge_j], vdofs_edge_j, vdofs_edge_j_lenght);
               ndof_edge_j = vdofs_edge_j.Size();

               // Set the size of C A^{-1} B
               SC_local.SetSize(ndof_edge_i, ndof_edge_j);

               // Compute the product that will be added to the Schur complement
               Mult(CA_local, B_local[edge_j], SC_local);

               // Remove the meaninglessly small entries from C A^{-1} B
               SC_local.Threshold(1.0e-16);

               // Assemble C A^{-1} B to the proper sparse matrices
               AddToMat(vdofs_edge_i, vdofs_edge_i_lenght, vdofs_edge_j, vdofs_edge_j_lenght,
                        SC_local, skip_zeros);
            }
         }
      }
      else
      {
         // Set the sise of B ubar
         B_skeleton_local.SetSize(ndof_volume);

         // if B has been stored, we need to keep track of how many terms have beed read already
         int B_values_read = 0;

         // Loop over of edges of the elemnt and calculate B ubar
         for (int edge=0; edge<no_faces; edge++)
         {
            // Get the skeletal DoFs for a given edge
            GetFaceVDofs(fcs[edge], vdofs_edge);
            ndof_edge = vdofs_edge.Size();

            skeleton_local.SetSize(ndof_edge);

            // read B if it had been stored
            if (i < elements_B)
            {
               for (int row = 0; row < ndof_edge; row++)
                  for (int col = 0; col < ndof_volume; col++)
                  {
                     (B_local[edge])(col,row) = B_data[B_offsets[i] + B_values_read + row*ndof_volume
                                                       +
                                                       col];
                  }

               B_values_read += ndof_volume*ndof_edge;
            }

            // Get the entries from the skeletal GFs
            GetFaceSubVector(Skel_GF, fcs[edge], ndof_edge, skeleton_local);
            (B_local[edge]).Mult(skeleton_local, B_skeleton_local);

            // Cacluclate F - B ubar
            F_local.Add(-1.0, B_skeleton_local);
         }

         if (i < elements_A)
         {
            for (int row = 0; row < ndof_volume; row++)
               for (int col = 0; col < ndof_volume; col++)
               {
                  A_local(col,row) = A_data[A_offsets[i] + row*ndof_volume + col];
               }
         }

         // volume sol = A^{-1}(F - B ubar)
         u_local.SetSize(ndof_volume);
         A_local.Mult(F_local, u_local);

         // Get the entries of the volume GFs
         SetInteriorSubVector(bdr_sol_sol_GF, i, ndof_volume, u_local);
      }

      delete [] B_local;
      delete [] C_local;
      delete [] D_local;
   }

   if (!assemble)
   {
      delete el_to_face;
      delete [] A_data;
      delete [] B_data;
   }
}
