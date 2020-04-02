#include "dofs.hpp"

int DofInfo::GetLocalFaceDofIndex3D(int loc_face_id, int face_orient,
                                    int face_dof_id, int face_dof1D_cnt)
{
   int k1, k2;
   const int kf1 = face_dof_id % face_dof1D_cnt;
   const int kf2 = face_dof_id / face_dof1D_cnt;
   switch (loc_face_id)
   {
      case 0://BOTTOM
         switch (face_orient)
         {
            case 0://{0, 1, 2, 3}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 1://{0, 3, 2, 1}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 2://{1, 2, 3, 0}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3://{1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 4://{2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 5://{2, 1, 0, 3}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6://{3, 0, 1, 2}
               k1 = kf2;
               k2 = kf1;
               break;
            case 7://{3, 2, 1, 0}
               k1 = kf1;
               k2 = kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 1://SOUTH
         switch (face_orient)
         {
            case 0://{0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1://{0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2://{1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3://{1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4://{2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5://{2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6://{3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7://{3, 2, 1, 0}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 2://EAST
         switch (face_orient)
         {
            case 0://{0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1://{0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2://{1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3://{1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4://{2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5://{2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6://{3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7://{3, 2, 1, 0}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 3://NORTH
         switch (face_orient)
         {
            case 0://{0, 1, 2, 3}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 1://{0, 3, 2, 1}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 2://{1, 2, 3, 0}
               k1 = kf2;
               k2 = kf1;
               break;
            case 3://{1, 0, 3, 2}
               k1 = kf1;
               k2 = kf2;
               break;
            case 4://{2, 3, 0, 1}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5://{2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 6://{3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 7://{3, 2, 1, 0}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 4://WEST
         switch (face_orient)
         {
            case 0://{0, 1, 2, 3}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 1://{0, 3, 2, 1}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 2://{1, 2, 3, 0}
               k1 = kf2;
               k2 = kf1;
               break;
            case 3://{1, 0, 3, 2}
               k1 = kf1;
               k2 = kf2;
               break;
            case 4://{2, 3, 0, 1}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5://{2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 6://{3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 7://{3, 2, 1, 0}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      case 5://TOP
         switch (face_orient)
         {
            case 0://{0, 1, 2, 3}
               k1 = kf1;
               k2 = kf2;
               break;
            case 1://{0, 3, 2, 1}
               k1 = kf2;
               k2 = kf1;
               break;
            case 2://{1, 2, 3, 0}
               k1 = kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 3://{1, 0, 3, 2}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = kf2;
               break;
            case 4://{2, 3, 0, 1}
               k1 = face_dof1D_cnt-1-kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            case 5://{2, 1, 0, 3}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = face_dof1D_cnt-1-kf1;
               break;
            case 6://{3, 0, 1, 2}
               k1 = face_dof1D_cnt-1-kf2;
               k2 = kf1;
               break;
            case 7://{3, 2, 1, 0}
               k1 = kf1;
               k2 = face_dof1D_cnt-1-kf2;
               break;
            default:
               mfem_error("This orientation does not exist in 3D");
               break;
         }
         break;
      default: MFEM_ABORT("This face_id does not exist in 3D");
   }
   return k1 + face_dof1D_cnt * k2;
}

int DofInfo::GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                                  int face_dof_id, int face_dof1D_cnt)
{
   switch (dim)
   {
      case 1: return face_dof_id;
      case 2:
         if (loc_face_id <= 1)
         {
            // SOUTH or EAST (canonical ordering)
            return face_dof_id;
         }
         else
         {
            // NORTH or WEST (counter-canonical ordering)
            return face_dof1D_cnt - 1 - face_dof_id;
         }
      case 3: return GetLocalFaceDofIndex3D(loc_face_id, face_orient,
                                               face_dof_id, face_dof1D_cnt);
      default: MFEM_ABORT("Dimension too high!"); return 0;
   }
}

DofInfo::DofInfo(FiniteElementSpace *fes_sltn,
                 FiniteElementSpace *fes_bounds)
   : mesh(fes_sltn->GetMesh()), fes(fes_sltn),
     x_min(fes_bounds), x_max(fes_bounds)
{
   dim = mesh->Dimension();

   int n = fes->GetVSize();
   int ne = mesh->GetNE();

   xi_min.SetSize(n);
   xi_max.SetSize(n);
   xe_min.SetSize(ne);
   xe_max.SetSize(ne);

   ExtractBdrDofs(fes->GetFE(0)->GetOrder(),
                  fes->GetFE(0)->GetGeomType(), BdrDofs);
   NumFaceDofs = BdrDofs.Height();
   NumBdrs = BdrDofs.Width();

   FillNeighborDofs();
   FillSubcell2CellDof();
}

void DofInfo::FillNeighborDofs()
{
   // Use the first mesh element as indicator.
   const FiniteElement &el = *fes->GetFE(0);
   int i, j, e, nbr, ne = mesh->GetNE();
   int nd = el.GetDof(), p = el.GetOrder();
   Array <int> bdrs, orientation;
   FaceElementTransformations *Trans;

   Table *face_to_el = mesh->GetFaceToElementTable();

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
         mesh->GetElementVertices(e, bdrs);

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
            mesh->GetFaceElements(bdrs[i], &el1_id, &el2_id);
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
         mesh->GetElementEdges(e, bdrs, orientation);

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
            mesh->GetFaceElements(bdrs[i], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == e) ? el2_id : el1_id;

            int el1_info, el2_info;
            mesh->GetFaceInfos(bdrs[i], &el1_info, &el2_info);
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
         mesh->GetElementFaces(e, bdrs, orientation);

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
            mesh->GetFaceElements(bdrs[f], &el1_id, &el2_id);
            if (el2_id < 0)
            {
               // This element is in a different mpi task.
               el2_id = -1 - el2_id + ne;
            }
            nbr_id = (el1_id == e) ? el2_id : el1_id;

            // Local index and orientation of the face, when considered in
            // the neighbor element.
            int el1_info, el2_info;
            mesh->GetFaceInfos(bdrs[f], &el1_info, &el2_info);
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

   for (e = 0; e < fes->GetNBE(); e++)
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

void DofInfo::FillSubcell2CellDof()
{
   const FiniteElement &el = *fes->GetFE(0);
   int j, m, aux, p = el.GetOrder();
   Geometry::Type gtype = el.GetGeomType();

   if (dim==1)
   {
      numSubcells = p;
      numDofsSubcell = 2;
   }
   else if (dim==2)
   {
      if (gtype == Geometry::TRIANGLE)
      {
         numSubcells = 1;
         numDofsSubcell = 3;
         for (int i = 1; i < p; i++)
         {
            numSubcells += 2*i+1;
         }
      }
      else if (gtype == Geometry::SQUARE)
      {
         numSubcells = p*p;
         numDofsSubcell = 4;
      }
   }
   else if (dim==3)
   {
      numSubcells = p*p*p;
      numDofsSubcell = 8;
   }

   Sub2Ind.SetSize(numSubcells, numDofsSubcell);

   if (gtype == Geometry::TRIANGLE)
   {
      int nd = (p+1)*(p+2) / 2;

      int ctr = numSubcells - 1;
      int node = nd - 1;
      int old;
      aux = 1;
      for (int iy = p; iy > 0; iy--)
      {
         old = node;
         Sub2Ind(ctr,2) = node;
         Sub2Ind(ctr,1) = node - aux;
         Sub2Ind(ctr,0) = node - aux - 1;
         ctr--;

         for (int ix = p - iy; ix > 0; ix--)
         {
            Sub2Ind(ctr,2) = node;
            Sub2Ind(ctr,1) = node - aux - 1;
            Sub2Ind(ctr,0) = node - 1;
            ctr--;
            node--;

            Sub2Ind(ctr,2) = node;
            Sub2Ind(ctr,1) = node - aux;
            Sub2Ind(ctr,0) = node - aux - 1;
            ctr--;
         }

         node = old - aux;
         aux++;
      }

      return;
   }

   for (m = 0; m < numSubcells; m++)
   {
      for (j = 0; j < numDofsSubcell; j++)
      {
         if (dim == 1) { Sub2Ind(m,j) = m + j; }
         else if (dim == 2 && gtype == Geometry::SQUARE)
         {
            aux = m + m/p;
            switch (j)
            {
               case 0: Sub2Ind(m,j) =  aux; break;
               case 1: Sub2Ind(m,j) =  aux + 1; break;
               case 2: Sub2Ind(m,j) =  aux + p+1; break;
               case 3: Sub2Ind(m,j) =  aux + p+2; break;
            }
         }
         else if (dim == 3 && gtype == Geometry::CUBE)
         {
            aux = m + m/p + (p+1)*(m/(p*p));
            switch (j)
            {
               case 0: Sub2Ind(m,j) = aux; break;
               case 1: Sub2Ind(m,j) = aux + 1; break;
               case 2: Sub2Ind(m,j) = aux + p+1; break;
               case 3: Sub2Ind(m,j) = aux + p+2; break;
               case 4: Sub2Ind(m,j) = aux + (p+1)*(p+1); break;
               case 5: Sub2Ind(m,j) = aux + (p+1)*(p+1)+1; break;
               case 6: Sub2Ind(m,j) = aux + (p+1)*(p+1)+p+1; break;
               case 7: Sub2Ind(m,j) = aux + (p+1)*(p+1)+p+2; break;
            }
         }
         else
         {
            MFEM_ABORT("Tetraheadra are not supported.");
         }
      }
   }
}

// Assuming L2 elements.
void DofInfo::ExtractBdrDofs(int p, Geometry::Type gtype, DenseMatrix &dofs)
{
   switch (gtype)
   {
      case Geometry::SEGMENT:
      {
         dofs.SetSize(1,2);
         dofs(0,0) = 0;
         dofs(0,1) = p;
         break;
      }
      case Geometry::TRIANGLE:
      {
         int ctr = 0;
         dofs.SetSize(p+1, 3);
         for (int i = 0; i <= p; i++)
         {
            dofs(i,0) = i;
            dofs(i,1) = ctr + p;
            dofs(i,2) = ctr + i;
            ctr += p - i;
         }
         break;
      }
      case Geometry::SQUARE:
      {
         dofs.SetSize(p+1,4);
         for (int i = 0; i <= p; i++)
         {
            dofs(i,0) = i;
            dofs(i,1) = i*(p+1) + p;
            dofs(i,2) = (p+1)*(p+1) - 1 - i;
            dofs(i,3) = (p-i)*(p+1);
         }
         break;
      }
      case Geometry::TETRAHEDRON:
      {
         int ctr;
         dofs.SetSize((p+1)*(p+2)/2, 4);
         for (int bdrID = 0; bdrID < 4; bdrID++)
         {
            int o = 0;
            switch (bdrID)
            {
               case 0:
                  ctr = p;
                  for (int i = 0; i <= p; i++)
                  {
                     for (int j = 0; j <= p - i; j++)
                     {
                        dofs(o++,bdrID) = ctr;
                        ctr += p - i - j;
                     }
                     ctr += p - i;
                  }
                  break;
               case 1:
                  ctr = 0;
                  for (int i = 0; i <= p; i++)
                  {
                     for (int j = 0; j <= p - i; j++)
                     {
                        dofs(o++,bdrID) = ctr;
                        ctr += p + 1 - i - j;
                     }
                  }
                  break;
               case 2:
                  ctr = 0;
                  for (int i = 0; i <= p; i++)
                  {
                     for (int j = 0; j <= p - i; j++)
                     {
                        dofs(o++,bdrID) = ctr++;
                     }
                     ctr += - p + i - 1 + (p-i+1)*(p-i+2)/2;
                  }
                  break;
               case 3:
                  for (int i = 0; i < (p+1)*(p+2)/2; i++)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
            }
         }
         break;
      }
      case Geometry::CUBE:
      {
         dofs.SetSize((p+1)*(p+1), 6);
         for (int bdrID = 0; bdrID < 6; bdrID++)
         {
            int o(0);
            switch (bdrID)
            {
               case 0:
                  for (int i = 0; i < (p+1)*(p+1); i++)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
               case 1:
                  for (int i = 0; i <= p*(p+1)*(p+1); i+=(p+1)*(p+1))
                     for (int j = 0; j < p+1; j++)
                     {
                        dofs(o++,bdrID) = i+j;
                     }
                  break;
               case 2:
                  for (int i = p; i < (p+1)*(p+1)*(p+1); i+=p+1)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
               case 3:
                  for (int i = 0; i <= p*(p+1)*(p+1); i+=(p+1)*(p+1))
                     for (int j = p*(p+1); j < (p+1)*(p+1); j++)
                     {
                        dofs(o++,bdrID) = i+j;
                     }
                  break;
               case 4:
                  for (int i = 0; i <= (p+1)*((p+1)*(p+1)-1); i+=p+1)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
               case 5:
                  for (int i = p*(p+1)*(p+1); i < (p+1)*(p+1)*(p+1); i++)
                  {
                     dofs(o++,bdrID) = i;
                  }
                  break;
            }
         }
         break;
      }
      default: MFEM_ABORT("Geometry not implemented.");
   }
}

void DofInfo::ComputeBounds()
{
   FiniteElementSpace *fesCG = x_min.FESpace();
   Array<int> dofsCG;

   // Form min/max at each CG dof, considering element overlaps.
   x_min =   std::numeric_limits<double>::infinity();
   x_max = - std::numeric_limits<double>::infinity();
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      x_min.FESpace()->GetElementDofs(i, dofsCG);
      for (int j = 0; j < dofsCG.Size(); j++)
      {
         x_min(dofsCG[j]) = std::min(x_min(dofsCG[j]), xe_min(i));
         x_max(dofsCG[j]) = std::max(x_max(dofsCG[j]), xe_max(i));
      }
   }
   Array<double> minvals(x_min.GetData(), x_min.Size()),
         maxvals(x_max.GetData(), x_max.Size());

   // Use (x_min, x_max) to fill (xi_min, xi_max) for each DG dof.
   const TensorBasisElement *fe_cg =
      dynamic_cast<const TensorBasisElement *>(fesCG->GetFE(0));
   const Array<int> &dof_map = fe_cg->GetDofMap();
   const int ndofs = dof_map.Size();
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      x_min.FESpace()->GetElementDofs(i, dofsCG);
      for (int j = 0; j < dofsCG.Size(); j++)
      {
         xi_min(i*ndofs + j) = x_min(dofsCG[dof_map[j]]);
         xi_max(i*ndofs + j) = x_max(dofsCG[dof_map[j]]);
      }
   }
}
