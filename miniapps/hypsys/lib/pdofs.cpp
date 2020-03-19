#include "pdofs.hpp"

ParDofInfo::ParDofInfo(ParFiniteElementSpace *pfes_sltn,
                       ParFiniteElementSpace *pfes_bounds)
   : DofInfo(pfes_sltn, pfes_bounds),
     pmesh(pfes_sltn->GetParMesh()), pfes(pfes_sltn),
     px_min(pfes_bounds), px_max(pfes_bounds)
{
   dim = pmesh->Dimension();

   int n = pfes->GetVSize();
   int ne = pmesh->GetNE();

   xi_min.SetSize(n);
   xi_max.SetSize(n);
   xe_min.SetSize(ne);
   xe_max.SetSize(ne);

   ExtractBdrDofs(pfes->GetFE(0)->GetOrder(),
                  pfes->GetFE(0)->GetGeomType(), BdrDofs);
   NumFaceDofs = BdrDofs.Height();
   NumBdrs = BdrDofs.Width();

   FillNeighborDofs();
   FillSubcell2CellDof();
}

void ParDofInfo::FillNeighborDofs()
{
   // Use the first mesh element as indicator.
   const FiniteElement &dummy = *pfes->GetFE(0);
   int i, j, e, nbr, ne = pmesh->GetNE();
   int nd = dummy.GetDof(), p = dummy.GetOrder();
   Array <int> bdrs, orientation;
   FaceElementTransformations *Trans;

   pmesh->ExchangeFaceNbrData();
   Table *face_to_el = pmesh->GetFaceToAllElementTable();

   NbrDofs.SetSize(NumBdrs, NumFaceDofs, ne);

   // Permutations of BdrDofs, taking into account all possible orientations.
   // Assumes BdrDofs are ordered in xyz order, which is true for 3D hexes,
   // but it isn't true for 2D quads.
   int orient_cnt = 1;
   if (dim == 2) { orient_cnt = 2; }
   if (dim == 3) { orient_cnt = 8; }
   const int dof1D_cnt = p+1;
   DenseTensor fdof_ids(NumFaceDofs, NumBdrs, orient_cnt);
   for (int ori = 0; ori < orient_cnt; ori++)
   {
      for (int face_id = 0; face_id < NumBdrs; face_id++)
      {
         for (int fdof_id = 0; fdof_id < NumFaceDofs; fdof_id++)
         {
            // Index of fdof_id in the current orientation.
            const int ori_fdof_id = GetLocalFaceDofIndex(dim, face_id, ori,
                                                         fdof_id, dof1D_cnt);
            fdof_ids(ori)(ori_fdof_id, face_id) = BdrDofs(fdof_id, face_id);
         }
      }
   }

   for (e = 0; e < ne; e++)
   {
      if (dim==1)
      {
         pmesh->GetElementVertices(e, bdrs);

         for (i = 0; i < NumBdrs; i++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[i]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               NbrDofs(i,0,e) = -1;
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[i], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == e) ? el2_id : el1_id;
            NbrDofs(i,0,e) = nbr_id*nd + BdrDofs(0,(i+1)%2);
         }
      }
      else if (dim==2)
      {
         pmesh->GetElementEdges(e, bdrs, orientation);

         for (i = 0; i < NumBdrs; i++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[i]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               for (j = 0; j < NumFaceDofs; j++) { NbrDofs(i,j,e) = -1; }
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[i], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == e) ? el2_id : el1_id;

            int el1_info, el2_info;
            pmesh->GetFaceInfos(bdrs[i], &el1_info, &el2_info);
            const int face_id_nbr = (nbr_id == el1_id) ? el1_info / 64
                                    : el2_info / 64;
            for (j = 0; j < NumFaceDofs; j++)
            {
               // Here it is utilized that the orientations of the face for
               // the two elements are opposite of each other.
               NbrDofs(i,j,e) = nbr_id*nd + BdrDofs(NumFaceDofs - 1 - j,
                                                    face_id_nbr);
            }
         }
      }
      else if (dim==3)
      {
         pmesh->GetElementFaces(e, bdrs, orientation);

         for (int f = 0; f < NumBdrs; f++)
         {
            const int nbr_cnt = face_to_el->RowSize(bdrs[f]);
            if (nbr_cnt == 1)
            {
               // No neighbor element.
               for (j = 0; j < NumFaceDofs; j++) { NbrDofs(f,j,e) = -1; }
               continue;
            }

            int el1_id, el2_id, nbr_id;
            pmesh->GetFaceElements(bdrs[f], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == e) ? el2_id : el1_id;

            // Local index and orientation of the face, when considered in
            // the neighbor element.
            int el1_info, el2_info;
            pmesh->GetFaceInfos(bdrs[f], &el1_info, &el2_info);
            const int face_id_nbr = (nbr_id == el1_id) ? el1_info / 64
                                    : el2_info / 64;
            const int face_or_nbr = (nbr_id == el1_id) ? el1_info % 64
                                    : el2_info % 64;
            for (j = 0; j < NumFaceDofs; j++)
            {
               // What is the index of the j-th dof on the face, given its
               // orientation.
               const int loc_face_dof_id =
                  GetLocalFaceDofIndex(dim, face_id_nbr, face_or_nbr,
                                       j, dof1D_cnt);
               // What is the corresponding local dof id on the element,
               // given the face orientation.
               const int nbr_dof_id =
                  fdof_ids(face_or_nbr)(loc_face_dof_id, face_id_nbr);

               NbrDofs(f,j,e) = nbr_id*nd + nbr_dof_id;
            }
         }
      }
   }
   for (e = 0; e < pfes->GetNBE(); e++)
   {
      const int bdr_attr = mesh->GetBdrAttribute(e);
      FaceElementTransformations *tr = mesh->GetBdrFaceTransformations(e);

      if (tr != NULL)
      {
         const int el = tr->Elem1No;

         if (dim == 1) { mesh->GetElementVertices(el, bdrs); }
         else if (dim == 2) { mesh->GetElementEdges(el, bdrs, orientation); }
         else if (dim == 3) { mesh->GetElementFaces(el, bdrs, orientation); }

         for (i = 0; i < NumBdrs; i++)
         {
            if (bdrs[i] == mesh->GetBdrElementEdgeIndex(e))
            {
               for (j = 0; j < NumFaceDofs; j++)
               {
                  NbrDofs(i, j, el) = -bdr_attr;
               }
               continue;
            }
         }
      }
      else
      {
         MFEM_ABORT("Something went wrong.");
      }
   }

   delete face_to_el;
}

void ParDofInfo::ComputeBounds()
{
   ParFiniteElementSpace *pfesCG = px_min.ParFESpace();
   GroupCommunicator &gcomm = pfesCG->GroupComm();
   Array<int> dofsCG;

   // Form min/max at each CG dof, considering element overlaps.
   px_min =   std::numeric_limits<double>::infinity();
   px_max = - std::numeric_limits<double>::infinity();
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      px_min.FESpace()->GetElementDofs(i, dofsCG);
      for (int j = 0; j < dofsCG.Size(); j++)
      {
         px_min(dofsCG[j]) = std::min(px_min(dofsCG[j]), xe_min(i));
         px_max(dofsCG[j]) = std::max(px_max(dofsCG[j]), xe_max(i));
      }
   }
   Array<double> minvals(px_min.GetData(), px_min.Size()),
         maxvals(px_max.GetData(), px_max.Size());

   gcomm.Reduce<double>(minvals, GroupCommunicator::Min);
   gcomm.Bcast(minvals);
   gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
   gcomm.Bcast(maxvals);

   // Use (px_min, px_max) to fill (xi_min, xi_max) for each DG dof.
   const TensorBasisElement *fe_cg =
      dynamic_cast<const TensorBasisElement *>(pfesCG->GetFE(0));
   const Array<int> &dof_map = fe_cg->GetDofMap();
   const int ndofs = dof_map.Size();
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      px_min.FESpace()->GetElementDofs(i, dofsCG);
      for (int j = 0; j < dofsCG.Size(); j++)
      {
         xi_min(i*ndofs + j) = px_min(dofsCG[dof_map[j]]);
         xi_max(i*ndofs + j) = px_max(dofsCG[dof_map[j]]);
      }
   }
}
