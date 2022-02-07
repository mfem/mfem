#include "schwarz.hpp"

// Some utility functions 

bool is_a_patch(int iv, Array<int> patch_ids)
{
   return (patch_ids.FindSorted(iv) != -1);
}

bool owned(int tdof, int * offs)
{
   return  (offs[0] <= tdof && tdof < offs[1]); 
}

void GetColumnValues(const int tdof_i, const Array<int> & tdof_j, SparseMatrix & diag,
SparseMatrix & offd, const int * cmap, const int * row_start,  Array<int> &cols, Array<double> &vals)
{
   int row = tdof_i - row_start[0];
   int row_size = diag.RowSize(row);

   int *col = diag.GetRowColumns(row);
   double *cval = diag.GetRowEntries(row);
   for (int j = 0; j < row_size; j++)
   {
      int icol = col[j]+ row_start[0];
      int jj = tdof_j.FindSorted(icol);
      if (jj != -1)
      {
         double dval = cval[j];
         cols.Append(jj);
         vals.Append(dval);
      }
   }
   int crow_size = offd.RowSize(row);
   int *ccol = offd.GetRowColumns(row);
   double *ccval = offd.GetRowEntries(row);
   for (int j = 0; j < crow_size; j++)
   {
      int icol = cmap[ccol[j]];
      int jj = tdof_j.FindSorted(icol);
      if (jj != -1)
      {
         double dval = ccval[j];
         cols.Append(jj);
         vals.Append(dval);
      }
   }
}

int GetNumColumns(const int tdof_i, const Array<int> & tdof_j, SparseMatrix & diag,
SparseMatrix & offd, const int * cmap, const int * row_start)
{
   int row = tdof_i - row_start[0];
   int row_size = diag.RowSize(row);

   int *col = diag.GetRowColumns(row);
   int k = -1;
   for (int j = 0; j < row_size; j++)
   {
      int icol = col[j]+ row_start[0];
      int jj = tdof_j.FindSorted(icol);
      if (jj != -1)
      {
         k++;
      }
   }
   int crow_size = offd.RowSize(row);
   int *ccol = offd.GetRowColumns(row);
   for (int j = 0; j < crow_size; j++)
   {
      int icol = cmap[ccol[j]];
      int jj = tdof_j.FindSorted(icol);
      if (jj != -1)
      {
         k++;
      }
   }
   return k;
}

void GetOffdColumnValues(const Array<int> & tdof_i, const Array<int> & tdof_j, SparseMatrix & offd, const int * cmap, 
                         const int * row_start , SparseMatrix * PatchMat)
{
   int ndof = tdof_i.Size();
   for (int i = 0; i<ndof; i++)
   {
      int row = tdof_i[i] - row_start[0];
      int row_size = offd.RowSize(row);
      int *ccol = offd.GetRowColumns(row);
      double *ccval = offd.GetRowEntries(row);
      for (int j = 0; j < row_size; j++)
      {
         int icol = cmap[ccol[j]];
         int jj = tdof_j.FindSorted(icol);
         if (jj != -1)
         {
            double dval = ccval[j];
            PatchMat->Set(i,jj,dval);
         }
      }
   }
}

SparseMatrix * GetLocalRestriction(const Array<int> & tdof_i, const int * row_start, 
                              const int num_rows, const int num_cols)
{
   SparseMatrix * R = new SparseMatrix(num_cols,num_rows);
   for (int i=0; i<num_cols; i++)
   {
         int ii = tdof_i[i] - row_start[0];
         R->Set(i,ii,1.0);
   }
   R->Finalize();
   return R;
}

void GetLocal2GlobalMap(const Array<int> & tdof_i, const int * row_start, 
                  const int num_rows, const int num_cols, Array<int> & l2gmap)
{
   l2gmap.SetSize(num_cols);
   for (int i=0; i<num_cols; i++)
   {
      int ii = tdof_i[i] - row_start[0];
      l2gmap[i] = ii;
   }
}

void GetArrayIntersection(const Array<int> & A, const Array<int> & B, Array<int>  & C) 
{
   int i = 0, j = 0;
   while (i != A.Size() && j != B.Size())
   {
        if (A[i] == B[j]) 
        {
            C.Append(A[i]);
            i++;
            j++;
        }
      else if (A[i] > B[j]) 
      {
         j++;
      } 
      else 
      {
         i++;
      }
   }
}  


VertexPatchInfo::VertexPatchInfo(ParMesh *pmesh_, int ref_levels_)
    : pmesh(pmesh_), ref_levels(ref_levels_)
{
   int dim = pmesh->Dimension();
   // 1. Define an auxiliary parallel H1 finite element space on the parallel mesh.
   FiniteElementCollection * aux_fec = new H1_FECollection(1, dim);
   ParMesh * aux_pmesh = new ParMesh(*pmesh);
   ParFiniteElementSpace * aux_fespace = new ParFiniteElementSpace(aux_pmesh, aux_fec);
   int mycdofoffset = aux_fespace->GetMyDofOffset(); // dof offset for the coarse mesh

   // 2. Store the cDofTrueDof Matrix. Required after the refinements
   HypreParMatrix *cDofTrueDof = new HypreParMatrix(*aux_fespace->Dof_TrueDof_Matrix());
   
   // 3. Perform the refinements (if any) and Get the final Prolongation operator
   HypreParMatrix *Pr = nullptr;
   for (int i = 0; i < ref_levels; i++)
   {
      const ParFiniteElementSpace cfespace(*aux_fespace);
      aux_pmesh->UniformRefinement();
      // Update fespace
      aux_fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      aux_fespace->GetTrueTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      HypreParMatrix *P;
      Tr.Get(P);
      if (!Pr)
      {
         Pr = P;
      }
      else
      {
         Pr = ParMult(P, Pr);
      }
   }
   // if (Pr) Pr->Threshold(0.0);

   // 4. Get the DofTrueDof map on this mesh and convert the prolongation matrix
   // to correspond to global dof numbering (from true dofs to dofs)
   HypreParMatrix *DofTrueDof = aux_fespace->Dof_TrueDof_Matrix();
   HypreParMatrix *A = nullptr; 
   if (Pr)
   {
      A = ParMult(DofTrueDof, Pr);
   }
   else
   {
      // If there is no refinement then the prolongation is the identity
      A = DofTrueDof;
   }
   HypreParMatrix * cDofTrueDofT = cDofTrueDof->Transpose();
   HypreParMatrix *B = ParMult(A, cDofTrueDofT); 
   delete cDofTrueDofT;
   // 5. Now we compute the vertices that are owned by the process
   SparseMatrix cdiag, coffd;
   cDofTrueDof->GetDiag(cdiag);
   Array<int> cown_vertices;
   int cnv = 0;
   for (int k = 0; k < cdiag.Height(); k++)
   {
      int nz = cdiag.RowSize(k);
      int i = mycdofoffset + k;
      if (nz != 0)
      {
         cnv++;
         cown_vertices.SetSize(cnv);
         cown_vertices[cnv - 1] = i;
      }
   }
   // 6. Compute total number of patches
   MPI_Comm comm = pmesh->GetComm();
   mynrpatch = cown_vertices.Size();
   // Compute total number of patches.

   MPI_Allreduce(&mynrpatch, &nrpatch, 1, MPI_INT, MPI_SUM, comm);

   patch_global_dofs_ids.SetSize(nrpatch);
   // Create a list of patches identifiers to all procs
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   int count[num_procs];

   MPI_Allgather(&mynrpatch, 1, MPI_INT, &count[0], 1, MPI_INT, comm);
   int displs[num_procs];
   displs[0] = 0;
   for (int i = 1; i < num_procs; i++)
   {
      displs[i] = displs[i - 1] + count[i - 1];
   }

   int * cownvert_ptr = nullptr;
   int * dof_rank_id_ptr = nullptr;
   Array<int> dof_rank_id;
   if (cown_vertices.Size() >0) 
   {
      cownvert_ptr = &cown_vertices[0];
      dof_rank_id.SetSize(cown_vertices.Size());
      dof_rank_id = myid;
      dof_rank_id_ptr = &dof_rank_id[0];
   }
   // send also the rank number for each global dof
   host_rank.SetSize(nrpatch);
   MPI_Allgatherv(cownvert_ptr, mynrpatch, MPI_INT, &patch_global_dofs_ids[0], count, displs, MPI_INT, comm);
   MPI_Allgatherv(dof_rank_id_ptr, mynrpatch, MPI_INT, &host_rank[0], count, displs, MPI_INT, comm);

   int size = patch_global_dofs_ids[nrpatch - 1] + 1;
   patch_natural_order_idx.SetSize(size);
   // initialize with -1
   patch_natural_order_idx = -1;
   for (int i = 0; i < nrpatch; i++)
   {
      int k = patch_global_dofs_ids[i];
      patch_natural_order_idx[k] = i;
   }

   int nvert = aux_pmesh->GetNV();
   // first find all the contributions of the vertices
   vert_contr.resize(nvert);
   SparseMatrix H1pr_diag;
   B->GetDiag(H1pr_diag);
   for (int i = 0; i < nvert; i++)
   {
      int row = i;
      int row_size = H1pr_diag.RowSize(row);
      int *col = H1pr_diag.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = col[j] + mycdofoffset;
         if (is_a_patch(jv, patch_global_dofs_ids))
         {
            vert_contr[i].Append(jv);
         }
      }
   }

   SparseMatrix H1pr_offd;
   int *cmap;
   B->GetOffd(H1pr_offd, cmap);
   for (int i = 0; i < nvert; i++)
   {
      int row = i;
      int row_size = H1pr_offd.RowSize(row);
      int *col = H1pr_offd.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = cmap[col[j]];
         if (is_a_patch(jv, patch_global_dofs_ids))
         {
            vert_contr[i].Append(jv);
         }
      }
   }

   Array<int> edge_vertices;
   int nedge = aux_pmesh->GetNEdges();
   edge_contr.resize(nedge);
   for (int ie = 0; ie < nedge; ie++)
   {
      aux_pmesh->GetEdgeVertices(ie, edge_vertices);
      int nv = edge_vertices.Size(); // always 2 but ok
      // The edge will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = edge_vertices[iv];
         edge_contr[ie].Append(vert_contr[ivert]);
      }
      edge_contr[ie].Sort();
      edge_contr[ie].Unique();
   }
   // -----------------------------------------------------------------------
   // done with edges. Now the faces
   // -----------------------------------------------------------------------
   Array<int> face_vertices;
   int nface = aux_pmesh->GetNFaces();
   face_contr.resize(nface);
   for (int ifc = 0; ifc < nface; ifc++)
   {
      aux_pmesh->GetFaceVertices(ifc, face_vertices);
      int nv = face_vertices.Size();
      // The face will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = face_vertices[iv];
         face_contr[ifc].Append(vert_contr[ivert]);
      }
      face_contr[ifc].Sort();
      face_contr[ifc].Unique();
   }
   // -----------------------------------------------------------------------
   // Finally the elements
   // -----------------------------------------------------------------------
   Array<int> elem_vertices;
   int nelem = aux_pmesh->GetNE();
   elem_contr.resize(nelem);
   for (int iel = 0; iel < nelem; iel++)
   {
      aux_pmesh->GetElementVertices(iel, elem_vertices);
      int nv = elem_vertices.Size();
      // The element will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = elem_vertices[iv];
         elem_contr[iel].Append(vert_contr[ivert]);
      }
      elem_contr[iel].Sort();
      elem_contr[iel].Unique();
   }
   if(Pr) delete A;
   delete B;
   if(Pr) delete Pr;
   delete cDofTrueDof;
   delete aux_fespace;
   delete aux_fec;
   delete aux_pmesh;
}

PatchDofInfo::PatchDofInfo(ParMesh *pmesh_, int ref_levels_, ParFiniteElementSpace *fespace)
{
   VertexPatchInfo *patch_nodes = new VertexPatchInfo(pmesh_, ref_levels_);

   int num_procs, myid;
   comm = pmesh_->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // Build a list on each processor identifying the truedofs in each patch
   // First the vertices
   nrpatch = patch_nodes->nrpatch;
   host_rank = patch_nodes->host_rank;

   patch_local_tdofs.resize(nrpatch);
   int * offs = fespace->GetTrueDofOffsets();
   int nrvert = fespace->GetNV();

   for (int i = 0; i < nrvert; i++)
   {
      int np = patch_nodes->vert_contr[i].Size();
      if (np == 0) continue;
      Array<int> vertex_dofs;
      fespace->GetVertexDofs(i, vertex_dofs);
      int nv = vertex_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patch_nodes->vert_contr[i][j];
         int kk = patch_nodes->patch_natural_order_idx[k];
         for (int l = 0; l < nv; l++)
         {
            int m = fespace->GetGlobalTDofNumber(vertex_dofs[l]);
            if (owned(m,offs)) patch_local_tdofs[kk].Append(m);
         }
      }
   }

   int nedge = fespace->GetMesh()->GetNEdges();
   for (int i = 0; i < nedge; i++)
   {
      int np = patch_nodes->edge_contr[i].Size();
      if (np == 0) continue;
      Array<int> edge_dofs;
      fespace->GetEdgeInteriorDofs(i, edge_dofs);
      int nv = edge_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patch_nodes->edge_contr[i][j];
         int kk = patch_nodes->patch_natural_order_idx[k];
         for (int l = 0; l < nv; l++)
         {
            int m = fespace->GetGlobalTDofNumber(edge_dofs[l]);
            if (owned(m,offs)) patch_local_tdofs[kk].Append(m);
         }
      }
   }
   int nface = fespace->GetMesh()->GetNFaces();
   for (int i = 0; i < nface; i++)
   {
      int np = patch_nodes->face_contr[i].Size();
      if (np == 0) continue;
      Array<int> face_dofs;
      fespace->GetFaceInteriorDofs(i, face_dofs);
      int nv = face_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patch_nodes->face_contr[i][j];
         int kk = patch_nodes->patch_natural_order_idx[k];
         for (int l = 0; l < nv; l++)
         {
            int m = fespace->GetGlobalTDofNumber(face_dofs[l]);
            if (owned(m,offs)) patch_local_tdofs[kk].Append(m);
         }
      }
   }

   int nelem = fespace->GetNE();
   for (int i = 0; i < nelem; i++)
   {
      int np = patch_nodes->elem_contr[i].Size();
      if (np == 0) continue;
      Array<int> elem_dofs;
      fespace->GetElementInteriorDofs(i, elem_dofs);
      int nv = elem_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patch_nodes->elem_contr[i][j];
         int kk = patch_nodes->patch_natural_order_idx[k];
         for (int l = 0; l < nv; l++)
         {
            int m = fespace->GetGlobalTDofNumber(elem_dofs[l]);
            if (owned(m,offs)) patch_local_tdofs[kk].Append(m);
         }
      }
   }
   delete patch_nodes;

   patch_tdofs.resize(nrpatch);
   for (int i = 0; i < nrpatch; i++)
   {
      Array<int> count(num_procs);
      int size = patch_local_tdofs[i].Size();

      count[myid] = size;
      MPI_Allgather(&size, 1, MPI_INT, &count[0], 1, MPI_INT, comm);
      Array<int>displs(num_procs);
      displs[0] = 0;
      for (int j = 1; j < num_procs; j++)
      {
         displs[j] = displs[j-1] + count[j-1];
      }
      int tot_size = displs[num_procs - 1] + count[num_procs - 1];
      // Get a group identifier for comm.
      MPI_Group world_group_id;
      MPI_Comm new_comm = MPI_COMM_NULL;
      MPI_Group new_group_id;
      MPI_Comm_group (comm, &world_group_id);
      // count the ranks that do not have zero length
      int num_ranks = 0;
      for (int k = 0; k<num_procs; k++)
      {
         if (count[k] != 0) {num_ranks++;}
      }
      Array<int> new_count(num_ranks);
      Array<int> new_displs(num_ranks);
      int sub_comm_ranks[num_ranks];
      num_ranks = 0;
      for (int j = 0; j <num_procs ; j++ )
      {
         if (count[j] != 0)
         {
            sub_comm_ranks[num_ranks] = j;
            new_count[num_ranks] = count[j];
            new_displs[num_ranks] = displs[j];
            num_ranks++;
         }
      }
      MPI_Group_incl(world_group_id, num_ranks, sub_comm_ranks, &new_group_id);
      MPI_Comm_create(comm, new_group_id, &new_comm);
      if (size != 0)
      {
         patch_tdofs[i].SetSize(tot_size);
         MPI_Allgatherv(&patch_local_tdofs[i][0],size,MPI_INT,
                     &patch_tdofs[i][0],new_count,new_displs,MPI_INT,new_comm);
      }
      MPI_Group_free(&world_group_id);
      MPI_Group_free(&new_group_id);
      if (new_comm != MPI_COMM_NULL) MPI_Comm_free(&new_comm);
   }

}

PatchAssembly::PatchAssembly(ParMesh *cpmesh_, int ref_levels_, ParFiniteElementSpace *fespace_, HypreParMatrix * A_) 
   :  A(A_), fespace(fespace_)
{
   comm = A->GetComm();
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   compute_trueoffsets();
   SparseMatrix diag;
   SparseMatrix offd;
   int *cmap;
   A->GetDiag(diag);
   A->GetOffd(offd,cmap);
   int *row_start = A->GetRowStarts();
   SparseMatrix offdT;
   int *cmapT;
   HypreParMatrix * At = A->Transpose();
   At->GetOffd(offdT,cmapT);
   int *row_startT = At->GetRowStarts();
   diag.SortColumnIndices();

   patch_tdof_info = new PatchDofInfo(cpmesh_, ref_levels_,fespace);
   
   nrpatch = patch_tdof_info->nrpatch; 
   host_rank.SetSize(nrpatch); host_rank = -1;
   // This can be changed later. For now the required lists are
   // constructed from the whole list 
   patch_other_tdofs.resize(nrpatch);
   patch_owned_other_tdofs.resize(nrpatch);
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int ndof = patch_tdof_info->patch_tdofs[ip].Size();

      if (ndof !=0 )
      {
         host_rank[ip] = patch_tdof_info->host_rank[ip];
         // host_rank[ip] = get_rank(patch_tdof_info->patch_tdofs[ip][0]);
         for (int i=0; i<ndof; i++)
         {
            int tdof = patch_tdof_info->patch_tdofs[ip][i];
            int tdof_rank = get_rank(tdof);
            if (tdof_rank != host_rank[ip])
            {
               patch_other_tdofs[ip].Append(tdof);
            }
         }
         GetArrayIntersection(patch_other_tdofs[ip], patch_tdof_info->patch_local_tdofs[ip], patch_owned_other_tdofs[ip]);
      }
   }
   // For the construction of the matrix of a patch we follow the following procedure. 
   // The matrix will be split to a 2x2 block matrix where: 
   // Block (0,0) is constructed by the dofs owned by the processor (using diag and RAP)
   // Block (0,1) is constructed by the dofs owned by the processor (using offd) 
   // Block (1,0) is the Transpose of (0,1) (for now the support is only for symmetric matrices)
   // Block (1,1) has to be communicated among processors. Its constructed by the dofs not owned by the processor.

   Array<SparseMatrix * > PatchMat00(nrpatch);
   l2gmaps.resize(nrpatch);

   //--------------------------------------------------------------------------------------
   // Construction of (0,0): This is done with RAP
   //--------------------------------------------------------------------------------------
   
   for (int ip = 0; ip < nrpatch; ip++)
   {
      PatchMat00[ip]=nullptr;
      if (myid == host_rank[ip])
      {
         int num_cols = patch_tdof_info->patch_local_tdofs[ip].Size();
         int num_rows = diag.Height();
         // loop through rows
         // l2gmaps[ip].SetSize(num_cols);
         GetLocal2GlobalMap(patch_tdof_info->patch_local_tdofs[ip],
                                 row_start, num_rows, num_cols, l2gmaps[ip]);      
         // Build prolongation (temporary to perform RAP)
         SparseMatrix Prl(num_rows,num_cols);
         for (int i=0; i<num_cols; ++i)
         {
            int ii = l2gmaps[ip][i];
            Prl.Set(ii,i,1.0);
         }
         Prl.Finalize();
         PatchMat00[ip] = RAP(Prl,diag,Prl);
      }  
   }

   //--------------------------------------------------------------------------------------
   // Construction of (0,1) and its transpose
   //--------------------------------------------------------------------------------------
    // The matrix PatchMat10 is the same as PatchMat01 only for symmetric problems
   // For the case of FOSLS the offdiagonal matrices are not symmetric because of the essential BC
   // Therefore Patch10 has to be computed in the same way with (0,1) but using the traspose of A
   // loop through patches
   Array<SparseMatrix * > PatchMat01(nrpatch);
   Array<SparseMatrix * > PatchMat10(nrpatch);
   for (int ip = 0; ip < nrpatch; ++ip)
   {
       PatchMat01[ip] = nullptr;
       PatchMat10[ip] = nullptr;
      if (myid == host_rank[ip])
      {
         int num_rows = patch_tdof_info->patch_local_tdofs[ip].Size();
         int num_cols = patch_other_tdofs[ip].Size();
         if (num_rows*num_cols !=0)
         {
            PatchMat01[ip] = new SparseMatrix(num_rows, num_cols);
            GetOffdColumnValues(patch_tdof_info->patch_local_tdofs[ip],patch_other_tdofs[ip],offd, cmap,row_start, PatchMat01[ip]);
            PatchMat01[ip]->Finalize();
            // Now the transpose
            SparseMatrix Mat(num_rows, num_cols);
            GetOffdColumnValues(patch_tdof_info->patch_local_tdofs[ip],patch_other_tdofs[ip],offdT, cmapT,row_startT, &Mat);
            Mat.Finalize();
            PatchMat10[ip] = Transpose(Mat);
         }
      }  
   }
   delete patch_tdof_info;
   delete At;

   //--------------------------------------------------------------------------------------
   // Construction of (1,1)
   //--------------------------------------------------------------------------------------
   // 1. Send info (sendbuff, sentcounts, send sdispls)
   // Each proccess computes and groups together an array of sendbuff. There will be
   // one sendbuff for all patches
   Array<int> send_count(num_procs);
   Array<int> send_displ(num_procs);
   Array<int> recv_count(num_procs);
   Array<int> recv_displ(num_procs);
   send_count = 0; send_displ = 0;
   recv_count = 0; recv_displ = 0;

   for (int ip = 0; ip < nrpatch; ip++)
   {
      // loop through the dofs and identify their ranks
      int sendnum_rows = patch_owned_other_tdofs[ip].Size();
      for (int i =0; i<sendnum_rows; i++)
      {
         int tdof = patch_owned_other_tdofs[ip][i];
         int tdof_rank = get_rank(tdof);
         if (myid == tdof_rank)
         {
            int k = GetNumColumns(tdof,patch_other_tdofs[ip],diag, offd, cmap, row_start);
            // pass one more that holds how many
            send_count[host_rank[ip]] += k+2;
         }
      } 
   }

   // comunicate so that recv_count is constructed
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();
   // now allocate space for the send buffer
   Array<double> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> sendmap(sbuff_size);  sendmap = 0;
   Array<int> soffs(num_procs); soffs = 0;

   // now the data will be placed according to process offsets
   for (int ip = 0; ip < nrpatch; ip++)
   {
      // loop through the dofs and identify their ranks
      int sendnum_rows = patch_owned_other_tdofs[ip].Size();
      for (int i = 0; i<sendnum_rows; i++)
      {
         int tdof = patch_owned_other_tdofs[ip][i];
         // find its rank
         int tdof_rank = get_rank(tdof);
         if (myid == tdof_rank)
         {
            Array<int>cols;
            Array<double>vals;
            GetColumnValues(tdof,patch_other_tdofs[ip],diag,offd, cmap,row_start, cols,vals);
            int j = send_displ[host_rank[ip]] + soffs[host_rank[ip]];
            int size = cols.Size();
            // Pass one more to hold the size
            soffs[host_rank[ip]] += size+1;
            // need to save and communicate these offsets for extraction from recv_buff
            // // For now we do the copy (will be changed later)
            sendbuf[j] = 0.0;
            sendmap[j] = size;
            for (int k=0; k<size ; k++)
            {
               sendbuf[j+k+1] = vals[k];
               sendmap[j+k+1] = cols[k];
            }
         }
      }
   }

   // communication
   Array<double> recvbuf(rbuff_size);
   Array<int> recvmap(rbuff_size);

   double * sendbuf_ptr = nullptr;
   double * recvbuf_ptr = nullptr;
   int * sendmap_ptr = nullptr;
   int * recvmap_ptr = nullptr;
   if (sbuff_size !=0 ) 
   {
      sendbuf_ptr = &sendbuf[0]; 
      sendmap_ptr = &sendmap[0]; 
   }   
   if (rbuff_size !=0 ) 
   {
      recvbuf_ptr = &recvbuf[0]; 
      recvmap_ptr = &recvmap[0]; 
   }

   MPI_Alltoallv(sendbuf_ptr, send_count, send_displ, MPI_DOUBLE, recvbuf_ptr,
                 recv_count, recv_displ, MPI_DOUBLE, comm);

   MPI_Alltoallv(sendmap_ptr, send_count, send_displ, MPI_INT, recvmap_ptr,
                 recv_count, recv_displ, MPI_INT, comm);

   Array<SparseMatrix * > PatchMat11(nrpatch);

   Array<int> roffs(num_procs);
   roffs = 0;
   // Now each process will construct the SparseMatrix
   for (int ip = 0; ip < nrpatch; ip++)
   {
      PatchMat11[ip] = nullptr;
      if(myid == host_rank[ip]) 
      {
         int ndof = patch_other_tdofs[ip].Size();
         PatchMat11[ip] = new SparseMatrix(ndof,ndof);
         // extract the data from receiv buffer
         // loop through rows
         for (int i=0; i<ndof; i++)
         {
            // pick up the dof and find its tdof_rank
            int tdof = patch_other_tdofs[ip][i];
            int tdof_rank= get_rank(tdof);
            // offset
            int k = recv_displ[tdof_rank] + roffs[tdof_rank];
            roffs[tdof_rank] += recvmap[k]+1;
            // copy to the matrix
            for (int j =0; j<recvmap[k]; j++)
            {
                  int jj = recvmap[k+j+1];
                  PatchMat11[ip]->Set(i,jj,recvbuf[k+j+1]);
            } 
         }   
         PatchMat11[ip]->Finalize();
      }
   }

   Array<BlockMatrix * > BlkPatchMat(nrpatch);
   PatchMat.SetSize(nrpatch);

   for (int ip = 0; ip < nrpatch; ip++)
   {
      //initialise to nullptr
      PatchMat[ip] = nullptr;
      if (myid == host_rank[ip])
      {
         if (PatchMat11[ip]->Height() !=0)
         {
            Array<int>block_offsets(3);
            block_offsets[0] = 0;
            block_offsets[1] = PatchMat00[ip]->Height();
            block_offsets[2] = PatchMat11[ip]->Height();
            block_offsets.PartialSum();
            BlkPatchMat[ip] = new BlockMatrix(block_offsets);
            BlkPatchMat[ip]->SetBlock(0,0,PatchMat00[ip]);   
            BlkPatchMat[ip]->SetBlock(0,1,PatchMat01[ip]);   
            BlkPatchMat[ip]->SetBlock(1,0,PatchMat10[ip]);   
            BlkPatchMat[ip]->SetBlock(1,1,PatchMat11[ip]);   
            // Convert to sparse
            PatchMat[ip] = BlkPatchMat[ip]->CreateMonolithic();
            delete BlkPatchMat[ip];
         }
         else
         {
            PatchMat[ip] = new SparseMatrix(*PatchMat00[ip]);
         }
         delete PatchMat00[ip];
         delete PatchMat10[ip];
         delete PatchMat01[ip];
         delete PatchMat11[ip];
         PatchMat[ip]->Threshold(1e-13);
      }
   }
   PatchMat00.DeleteAll();
   PatchMat01.DeleteAll();
   PatchMat10.DeleteAll();
   PatchMat11.DeleteAll();

}


PatchRestriction::PatchRestriction(PatchAssembly * P_) : P(P_)
{
   comm = P->comm;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   nrpatch = P->nrpatch;
   host_rank = P->host_rank;

   send_count.SetSize(num_procs);
   send_displ.SetSize(num_procs);
   recv_count.SetSize(num_procs);
   recv_displ.SetSize(num_procs);

   send_count = 0; send_displ = 0;
   recv_count = 0; recv_displ = 0;

   // Precompute send_counts
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int sendnum_rows = P->patch_owned_other_tdofs[ip].Size();
      for (int i =0; i<sendnum_rows; i++)
      {
         int tdof = P->patch_owned_other_tdofs[ip][i];
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            send_count[host_rank[ip]]++;
         }
      } 
   }

   // comunicate so that recv_count is constructed
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);
   //
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   sbuff_size = send_count.Sum();
   rbuff_size = recv_count.Sum();
}

void PatchRestriction::Mult(const Vector & r , Array<BlockVector*> & res)
{
   int *row_start = P->A->GetRowStarts();
   std::vector<Vector> res0(nrpatch); // residual on the processor
   std::vector<Vector> res1(nrpatch); // residual off the processor
   //  Part of the residual on the processor
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (myid == host_rank[ip]) 
      {
         r.GetSubVector(P->l2gmaps[ip], res0[ip]);
      }
   }
   // now allocate space for the send buffer
   Array<double> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> soffs(num_procs); soffs = 0;
   // now the data will be placed according to process offsets
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int sendnum_rows = P->patch_owned_other_tdofs[ip].Size();
      for (int i = 0; i<sendnum_rows; i++)
      {
         int tdof = P->patch_owned_other_tdofs[ip][i];
         // find its rank
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            int j = send_displ[host_rank[ip]] + soffs[host_rank[ip]];
            soffs[host_rank[ip]]++;
            int k = tdof - row_start[0];
            sendbuf[j] = r[k];
         }
      }
   }
   // communication
   Array<double> recvbuf(rbuff_size);
   double * sendbuf_ptr = nullptr;
   double * recvbuf_ptr = nullptr;
   if (sbuff_size !=0 ) sendbuf_ptr = &sendbuf[0]; 
   if (rbuff_size !=0 ) recvbuf_ptr = &recvbuf[0]; 
      
   MPI_Alltoallv(sendbuf_ptr, send_count, send_displ, MPI_DOUBLE, recvbuf_ptr,
                 recv_count, recv_displ, MPI_DOUBLE, comm);            
   Array<int> roffs(num_procs);
   roffs = 0;
   // Now each process will construct the res1 vector
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if(myid == host_rank[ip]) 
      {
         int ndof = P->patch_other_tdofs[ip].Size();
         res1[ip].SetSize(ndof);
         // extract the data from receiv buffer
         // loop through rows
         for (int i=0; i<ndof; i++)
         {
            // pick up the dof and find its tdof_rank
            int tdof = P->patch_other_tdofs[ip][i];
            int tdof_rank= P->get_rank(tdof);
            // offset
            int k = recv_displ[tdof_rank] + roffs[tdof_rank];
            roffs[tdof_rank]++;
            res1[ip][i] = recvbuf[k]; 
         }   
      }
   }

   res.SetSize(nrpatch);
   for (int ip=0; ip<nrpatch; ip++)
   {
      res[ip] = nullptr;
      if(myid == host_rank[ip]) 
      {
         Array<int> block_offs(3);
         block_offs[0] = 0;
         block_offs[1] = res0[ip].Size();
         block_offs[2] = res1[ip].Size();
         block_offs.PartialSum();
         res[ip] = new BlockVector(block_offs);
         res[ip]->SetVector(res0[ip], 0);
         res[ip]->SetVector(res1[ip], res0[ip].Size());
      }
   }
}

void PatchRestriction::MultTranspose(const Array<BlockVector *> & sol, Vector & z)
{
   int *row_start = P->A->GetRowStarts();
   std::vector<Vector> sol0(nrpatch);
   std::vector<Vector> sol1(nrpatch);
   // Step 3: Propagate the information to the global solution vector
   // (the recv_buff becomes the sendbuff and vice-versa) 
   Array<double> sendbuf(sbuff_size);  sendbuf = 0.0;
   Array<double> recvbuf(rbuff_size);  recvbuf = 0.0;
   Array<int> roffs(num_procs); roffs = 0;
   Array<int> soffs(num_procs); soffs = 0;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if(myid == host_rank[ip]) 
      {
         sol1[ip] = sol[ip]->GetBlock(1);
         int ndof = P->patch_other_tdofs[ip].Size();
         // loop through rows
         for (int i=0; i<ndof; i++)
         {
            //  pick up the dof and find its tdof_rank
            int tdof = P->patch_other_tdofs[ip][i];
            int tdof_rank= P->get_rank(tdof);
            // offset
            int k = recv_displ[tdof_rank] + roffs[tdof_rank];
            roffs[tdof_rank]++;
            recvbuf[k] = sol1[ip][i]; 
         }   
      }
   }
// now communication
   double * sendbuf_ptr = nullptr;
   double * recvbuf_ptr = nullptr;
   if (sbuff_size !=0 ) sendbuf_ptr = &sendbuf[0]; 
   if (rbuff_size !=0 ) recvbuf_ptr = &recvbuf[0]; 
   
   MPI_Alltoallv(recvbuf_ptr, recv_count, recv_displ, MPI_DOUBLE, sendbuf_ptr,
                 send_count, send_displ, MPI_DOUBLE, comm);            

   // 1. Accummulate for the solution to other prosessors
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int sendnum_rows = P->patch_owned_other_tdofs[ip].Size();
      for (int i = 0; i<sendnum_rows; i++)
      {
         int tdof = P->patch_owned_other_tdofs[ip][i];
         // find its rank
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            int j = send_displ[host_rank[ip]] + soffs[host_rank[ip]];
            soffs[host_rank[ip]]++;
            int k = tdof - row_start[0];
            z[k] += sendbuf[j];
         }
      }
   }
   // 2. Accummulate for the solution on the processor
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (myid == host_rank[ip])
      {
         sol0[ip] = sol[ip]->GetBlock(0);
         z.AddElementVector(P->l2gmaps[ip],sol0[ip]);
      }
   }
}

void PatchAssembly::compute_trueoffsets()
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   tdof_offsets.resize(num_procs);
   int mytoffset = fespace->GetMyTDofOffset();
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

int PatchAssembly::get_rank(int tdof)
{
   int size = tdof_offsets.size();
   if (size == 1) {return 0;}
   std::vector<int>::iterator up;
   up=std::upper_bound (tdof_offsets.begin(), tdof_offsets.end(), tdof); //          ^
   return std::distance(tdof_offsets.begin(),up)-1;
}

PatchAssembly::~PatchAssembly()
{
   // // if(patch_tdof_info) delete patch_tdof_info;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (PatchMat[ip]) delete PatchMat[ip];
   }
   PatchMat.DeleteAll();
}


SchwarzSmoother::SchwarzSmoother(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_, HypreParMatrix * A_)
: Solver(A_->Height(), A_->Width()), A(A_)
{
   P = new PatchAssembly(cpmesh_,ref_levels_, fespace_, A_);

   comm = A->GetComm();
   nrpatch = P->nrpatch;
   host_rank.SetSize(nrpatch);
   host_rank = P->host_rank;
   PatchInv.SetSize(nrpatch);
   for (int ip = 0; ip < nrpatch; ip++)
   {
      PatchInv[ip] = nullptr;
      if (P->PatchMat[ip])
      {
         PatchInv[ip] = new UMFPackSolver;
         PatchInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
         PatchInv[ip]->SetOperator(*P->PatchMat[ip]);
      }
   }
   R = new PatchRestriction(P);
}

void SchwarzSmoother::Mult(const Vector &r, Vector &z) const
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   Array<int> tdof_i;

   z = 0.0;
   Vector rnew(r);
   Vector znew(z);
   
   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;
      Array<BlockVector * > res;
      R->Mult(rnew,res);

      Array<BlockVector*> sol(nrpatch);
      for (int ip=0; ip<nrpatch; ip++)
      {
         sol[ip] = nullptr;
         if(myid == host_rank[ip]) 
         {
            Array<int> block_offs(3);
            block_offs[0] = 0;
            block_offs[1] = res[ip]->GetBlock(0).Size();
            block_offs[2] = res[ip]->GetBlock(1).Size();
            block_offs.PartialSum();
            sol[ip] = new BlockVector(block_offs);
            PatchInv[ip]->Mult(*res[ip], *sol[ip]);
         }
      }
      for (auto p:res) delete p;
      res.DeleteAll();
      R->MultTranspose(sol,znew);
      for (auto p:sol) delete p;
      sol.DeleteAll();

      znew *= theta; // relaxation parameter
      z+= znew;
      // Update residual
      Vector raux(znew.Size());
      A->Mult(znew,raux); 
      rnew -= raux;
   } // end of loop through smoother iterations
}


SchwarzSmoother::~SchwarzSmoother()
{
   delete P; 
   delete R;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (PatchInv[ip]) delete PatchInv[ip];
   }
   PatchInv.DeleteAll();
}



ComplexSchwarzSmoother::ComplexSchwarzSmoother(ParMesh * cpmesh_, int ref_levels_,
ParFiniteElementSpace *fespace_, ComplexHypreParMatrix * A_)
: Solver(A_->Height(), A_->Width()), A(A_)
{
   P_r = new PatchAssembly(cpmesh_,ref_levels_, fespace_, &A_->real());
   P_i = new PatchAssembly(cpmesh_,ref_levels_, fespace_, &A_->imag());

   comm = fespace_->GetComm();
   nrpatch = P_r->nrpatch;
   host_rank.SetSize(nrpatch);
   host_rank = P_r->host_rank;
   PatchInv.SetSize(nrpatch);
   for (int ip = 0; ip < nrpatch; ip++)
   {
      PatchInv[ip] = nullptr;
      if (P_r->PatchMat[ip])
      {
         PatchInv[ip] = new UMFPackSolver;
         ComplexSparseMatrix * cA = new ComplexSparseMatrix(P_r->PatchMat[ip], P_i->PatchMat[ip], false, false);
         PatchInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
         PatchInv[ip]->SetOperator(*cA->GetSystemMatrix());
      }
   }
   R_r = new PatchRestriction(P_r);
   R_i = new PatchRestriction(P_i);
}

void ComplexSchwarzSmoother::Mult(const Vector &r, Vector &z) const
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   Array<int> tdof_i;

   z = 0.0;
   Vector rnew(r);
   Vector znew(z);
   

   Vector rnew_r, rnew_i;
   Vector znew_r, znew_i;
   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;

      double * zdata = znew.GetData();
      int zsize = znew.Size();
      znew_r.SetDataAndSize(zdata,zsize/2);
      znew_i.SetDataAndSize(&zdata[zsize/2],zsize/2);

      Array<BlockVector * > res_r;
      Array<BlockVector * > res_i;

      double * data = rnew.GetData();
      int size = rnew.Size();
      rnew_r.SetDataAndSize(data,size/2);
      rnew_i.SetDataAndSize(&data[size/2],size/2);

      R_r->Mult(rnew_r,res_r);
      R_i->Mult(rnew_i,res_i);

      Array<BlockVector*> sol_r(nrpatch);
      Array<BlockVector*> sol_i(nrpatch);
      for (int ip=0; ip<nrpatch; ip++)
      {
         sol_r[ip] = nullptr;
         sol_i[ip] = nullptr;
         if(myid == host_rank[ip]) 
         {
            Array<int> block_offs(3);
            block_offs[0] = 0;
            block_offs[1] = res_r[ip]->GetBlock(0).Size();
            block_offs[2] = res_i[ip]->GetBlock(1).Size();
            block_offs.PartialSum();
            sol_r[ip] = new BlockVector(block_offs);
            sol_i[ip] = new BlockVector(block_offs);
            Vector res(2*sol_r[ip]->Size());
            res.SetVector(*res_r[ip],0);
            res.SetVector(*res_i[ip],res_r[ip]->Size());
            Vector sol(2*sol_r[ip]->Size());
            PatchInv[ip]->Mult(res, sol);
            Vector solr, soli;
            solr.SetDataAndSize(sol.GetData(), sol.Size()/2);
            soli.SetDataAndSize(&(sol.GetData())[sol.Size()/2], sol.Size()/2);

            sol_r[ip]->SetVector(solr,0);
            sol_i[ip]->SetVector(soli,0);
            
         }
      }
      for (auto p:res_r) delete p;
      for (auto p:res_i) delete p;
      res_r.DeleteAll();
      res_i.DeleteAll();
      R_r->MultTranspose(sol_r,znew_r);
      R_i->MultTranspose(sol_i,znew_i);
      for (auto p:sol_r) delete p;
      for (auto p:sol_i) delete p;
      sol_r.DeleteAll();
      sol_i.DeleteAll();

      znew *= theta; // relaxation parameter
      z+= znew;
      // Update residual
      Vector raux(znew.Size());
      A->Mult(znew,raux); 
      rnew -= raux;
   } // end of loop through smoother iterations
}


ComplexSchwarzSmoother::~ComplexSchwarzSmoother()
{
   delete P_r; 
   delete P_i; 
   delete R_r;
   delete R_i;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (PatchInv[ip]) delete PatchInv[ip];
   }
   PatchInv.DeleteAll();
}