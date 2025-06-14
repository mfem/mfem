#include "block_hybridization.hpp"

using namespace std;

namespace mfem
{
namespace blocksolvers
{

/// Block hybridization solver
BlockHybridizationSolver::BlockHybridizationSolver(
   ParBilinearForm *mVarf,
   ParMixedBilinearForm *bVarf,
   IterSolveParameters param)
   : DarcySolver(mVarf->ParFESpace()->GetTrueVSize(),
                 bVarf->TestParFESpace()->GetTrueVSize()),
     hdiv_space(mVarf->ParFESpace()),
     l2_space(bVarf->TestParFESpace()),
     solver_(hdiv_space->GetComm())
{
   ParMesh *mesh(hdiv_space->GetParMesh());
   mesh->ExchangeFaceNbrData();
   const int dim = mesh->Dimension();
   const int num_elements(mesh->GetNE());
   const real_t eps = 1e-12;

   DG_Interface_FECollection fec(hdiv_space->FEColl()->GetOrder()-1,
                                 mesh->Dimension());
   multiplier_space = new ParFiniteElementSpace(mesh, &fec);
   SparseMatrix reduced_matrix(multiplier_space->GetNDofs());

   FaceElementTransformations *trans(nullptr);
   NormalTraceIntegrator integ;
   const FiniteElement *element(nullptr);

   saved_hdiv_matrices = new DenseMatrix[num_elements];
   saved_mixed_matrices = new DenseMatrix[num_elements];
   saved_l2_matrices = new DenseMatrix[num_elements];
   interior_indices = new Array<int>[num_elements];

   Table element_to_facet_table;
   if (2 == dim)
   {
      element_to_facet_table = mesh->ElementToEdgeTable();
   }
   else
   {
      element_to_facet_table = mesh->ElementToFaceTable();
   }

   // indirectly mark the interior faces using the be_to_face table
   Array<int> interior_face_marker(mesh->GetNumFaces());
   interior_face_marker = 1;
   for (int i = 0; i < mesh->GetNBE(); ++i)
   {
      interior_face_marker[mesh->GetBdrElementFaceIndex(i)] = 0;
   }
   interior_face_marker.Print(out, interior_face_marker.Size());


   int *I = element_to_facet_table.GetI();
   int *J = element_to_facet_table.GetJ();

   // create element-to-interior-face table
   // at some point really make this into a table
   // instead of just arrays
   int *nI = new int[num_elements+1];
   nI[0] = 0;

   // we already know the number of local interior faces summed over all elements
   // this should still work for mixed meshes
   int nnz(I[num_elements]-mesh->GetNBE());
   int *nJ = new int[nnz];

   int counter(0);
   for (int elem_idx = 0; elem_idx < num_elements; ++elem_idx)
   {
      for (int j = I[elem_idx]; j < I[elem_idx+1]; ++j)
      {
         const int face_idx(J[j]);
         if (interior_face_marker[face_idx]) // we check every face index usually twice
                                             // can this be made more efficient?
         {
            nJ[counter++] = face_idx;
         }
      }
      nI[elem_idx+1] = counter;
   }

   Array<int> aI(nI, num_elements+1);
   Array<int> aJ(nJ, nnz);

   aI.Print(out, aI.Size());
   aJ.Print(out, aJ.Size());

   const Table &face_to_dof_table(multiplier_space->GetFaceToDofTable());
   const int *faceI = face_to_dof_table.GetI();
   const int *faceJ = face_to_dof_table.GetJ();

   int *interior_faceI = new int[faceI[mesh->GetNumFaces()]];
   interior_faceI[0] = 0;
   int total_dofs(0);
   for (int elem_idx = 0; elem_idx < num_elements; ++elem_idx)
   {
      for (int k = nI[elem_idx]; k < nI[elem_idx+1]; ++k)
      {
         const int face_idx = nJ[k];
         total_dofs += faceI[face_idx+1] - faceI[face_idx];
      }
      interior_faceI[elem_idx+1] = total_dofs;
   }

   int *interior_faceJ = new int[interior_faceI[num_elements]];
   counter = 0;
   for (int elem_idx = 0; elem_idx < num_elements; ++elem_idx)
   {
      for (int k = nI[elem_idx]; k < nI[elem_idx+1]; ++k)
      {
         const int face_idx = nJ[k];
         for (int dof_idx = faceI[face_idx]; dof_idx < faceI[face_idx+1]; ++dof_idx)
         {
            interior_faceJ[counter++] = faceJ[dof_idx];
         }
      }
   }

   /*
   Array<int> arrI(interior_faceI, num_elements+1);
   Array<int> arrJ(interior_faceJ, interior_faceI[num_elements]);

   arrI.Print(out, arrI.Size());
   arrJ.Print(out, arrJ.Size());
   */

   for (int elem_idx = 0; elem_idx < num_elements; ++elem_idx)
   {
      mVarf->ComputeElementMatrix(elem_idx, saved_hdiv_matrices[elem_idx]);
      saved_hdiv_matrices[elem_idx].Threshold(eps * saved_hdiv_matrices[elem_idx].MaxMaxNorm());
      saved_hdiv_matrices[elem_idx].Invert(); // overwrite saved_hdiv_matrices[elem_idx]

      bVarf->ComputeElementMatrix(elem_idx, saved_mixed_matrices[elem_idx]);
      saved_mixed_matrices[elem_idx].Threshold(eps * saved_mixed_matrices[elem_idx].MaxMaxNorm());

      DenseMatrix product_matrix(saved_mixed_matrices[elem_idx].Height(),
                                 saved_hdiv_matrices[elem_idx].Width());
      mfem::Mult(saved_mixed_matrices[elem_idx],
                 saved_hdiv_matrices[elem_idx], product_matrix); // BA^{-1}

      saved_l2_matrices[elem_idx].SetSize(product_matrix.Height(),
                                               saved_mixed_matrices[elem_idx].Height());
      MultABt(product_matrix, saved_mixed_matrices[elem_idx],
              saved_l2_matrices[elem_idx]); // BA^{-1}B^t = -S
      saved_l2_matrices[elem_idx].Invert(); // (BA^{-1}B^t)^{-1} = -S^{-1}
      saved_l2_matrices[elem_idx].Neg(); // -(BA^{-1}B^t)^{-1} = S^{-1}

      mfem::Mult(saved_l2_matrices[elem_idx], product_matrix,
                 saved_mixed_matrices[elem_idx]); // overwrite saved_mixed_matrices[elem_idx]
                                                       // with S^{-1}BA^{-1}
      DenseMatrix temp_matrix(product_matrix.Width(),
                              saved_mixed_matrices[elem_idx].Width());
      MultAtB(product_matrix, saved_mixed_matrices[elem_idx], temp_matrix); // A^{-T}B^TS^{-1}BA^{-1}
      saved_hdiv_matrices[elem_idx] += temp_matrix;

      saved_mixed_matrices[elem_idx].Neg(); // -S^{-1}BA^{-1}

      element = hdiv_space->GetFE(elem_idx);
      const FiniteElement *face(nullptr);

      // exploit that dense matrices are in column-major format
      // we do not copy each face matrix into the boundary matrix
      const int num_multiplier_dofs = interior_faceI[elem_idx+1] - interior_faceI[elem_idx];
      Array<int> multiplier_dofs;
      multiplier_dofs.MakeRef(interior_faceJ+interior_faceI[elem_idx], num_multiplier_dofs);
      DenseMatrix bdr_matrix(saved_hdiv_matrices[elem_idx].Height(), num_multiplier_dofs);
      int data_offset = 0;
      const int num_faces = nI[elem_idx+1] - nI[elem_idx];
      const int height = bdr_matrix.Height();
      const int width = bdr_matrix.Width() / num_faces; // number of dofs per face

      for (int k = nI[elem_idx]; k < nI[elem_idx+1]; ++k)
      {
         const int face_idx = nJ[k];
         trans = mesh->GetFaceElementTransformations(face_idx);
         face = multiplier_space->GetFaceElement(face_idx);
         DenseMatrix face_matrix(bdr_matrix.Data()+data_offset, height, width);
         integ.AssembleTraceFaceMatrix(elem_idx, *face, *element, *trans,
                                       face_matrix);
         data_offset += height*width;
      }

      temp_matrix.SetSize(height, bdr_matrix.Width());
      mfem::Mult(saved_hdiv_matrices[elem_idx], bdr_matrix, temp_matrix);
      product_matrix.SetSize(bdr_matrix.Width());
      MultAtB(bdr_matrix, temp_matrix, product_matrix);
      reduced_matrix.AddSubMatrix(multiplier_dofs, multiplier_dofs, product_matrix);

   /*
      Array<int> face_indices_array;
      element_to_facet_table.GetRow(elem_idx, face_indices_array);
      DenseMatrix *face_matrices = new DenseMatrix[face_indices_array.Size()];
      Array<int> *face_dofs = new Array<int>[face_indices_array.Size()];

      for (int local_index = 0; local_index < face_indices_array.Size();
           ++local_index)
      {
         const int face_index(face_indices_array[local_index]);
         trans = mesh->GetFaceElementTransformations(face_index);
         if (!mesh->FaceIsTrueInterior(face_index))
         {
            continue;
         }
         interior_indices[elem_idx].Append(local_index);
         multiplier_space->GetFaceVDofs(face_index, face_dofs[local_index]);
         face = multiplier_space->GetFaceElement(face_index);
         integ.AssembleTraceFaceMatrix(elem_idx, *face, *element, *trans,
                                       face_matrices[local_index]);
      }
      for (int column_index : interior_indices[elem_idx])
      {
         DenseMatrix temp_matrix(saved_hdiv_matrices[elem_idx].Height(),
                                 face_matrices[column_index].Width());
         mfem::Mult(saved_hdiv_matrices[elem_idx], face_matrices[column_index],
                    temp_matrix);
         for (int row_index : interior_indices[elem_idx])
         {
            DenseMatrix product_matrix(face_matrices[row_index].Width(),
                                       temp_matrix.Width());
            MultAtB(face_matrices[row_index], temp_matrix, product_matrix);
            reduced_matrix.AddSubMatrix(face_dofs[row_index], face_dofs[column_index],
                                        product_matrix);
         }
      }
      delete []face_dofs;
      delete []face_matrices;
   */
   }
   delete []interior_faceI;
   delete []interior_faceJ;
   delete []nI;
   delete []nJ;

   reduced_matrix.Finalize(1, true);
   // if (Mpi::Root())
   //    reduced_matrix.Print();

   HypreParMatrix *P(multiplier_space->Dof_TrueDof_Matrix());
   HypreParMatrix *dH = new HypreParMatrix(multiplier_space->GetComm(),
                                           multiplier_space->GlobalVSize(),
                                           multiplier_space->GetDofOffsets(), &reduced_matrix);
   HypreParMatrix *dHP = ParMult(dH, P);
   HypreParMatrix *Pt(P->Transpose());
   pH = ParMult(Pt, dHP, true);

   delete dH;
   /*
      OperatorPtr pP(Operator::Hypre_ParCSR);
      pP.ConvertFrom(multiplier_space->Dof_TrueDof_Matrix());
      OperatorPtr dH(pP.Type());
      dH.MakeSquareBlockDiag(multiplier_space->GetComm(), multiplier_space->GlobalVSize(),
                             multiplier_space->GetDofOffsets(), &reduced_matrix);
      OperatorPtr AP(ParMult(dH.As<HypreParMatrix>(), pP.As<HypreParMatrix>()));
      OperatorPtr R(pP.As<HypreParMatrix>()->Transpose());
      pH = ParMult(R.As<HypreParMatrix>(), AP.As<HypreParMatrix>(), true);
   */
   preconditioner = new HypreBoomerAMG(*pH);
   preconditioner->SetPrintLevel(0);

   SetOptions(solver_, param);
   solver_.SetPreconditioner(*preconditioner);
   solver_.SetOperator(*pH);
}

BlockHybridizationSolver::~BlockHybridizationSolver()
{
   delete pH;
   delete []saved_hdiv_matrices;
   delete []saved_mixed_matrices;
   delete []saved_l2_matrices;
   delete []interior_indices;
   delete preconditioner;
   delete multiplier_space;
}

} // namespace blocksolvers
} // namespace mfem
