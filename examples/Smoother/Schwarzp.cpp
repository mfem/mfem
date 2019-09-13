
#include "mfem.hpp"
#include "Schwarzp.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

par_patch_nod_info::par_patch_nod_info(ParMesh *cpmesh_, int ref_levels_)
    : pmesh(*cpmesh_), ref_levels(ref_levels_)
{
   int dim = pmesh.Dimension();
   // 1. Define an auxiliary parallel H1 finite element space on the parallel mesh.
   aux_fec = new H1_FECollection(1, dim);
   aux_fespace = new ParFiniteElementSpace(&pmesh, aux_fec);
   int mycdofoffset = aux_fespace->GetMyDofOffset(); // dof offset for the coarse mesh

   // 2. Store the cDofTrueDof Matrix. Required after the refinements
   HypreParMatrix *cDofTrueDof = new HypreParMatrix(*aux_fespace->Dof_TrueDof_Matrix());

   // 3. Perform the refinements and Get the final Prolongation operator
   HypreParMatrix *Pr = nullptr;
   for (int i = 0; i < ref_levels; i++)
   {
      const ParFiniteElementSpace cfespace(*aux_fespace);
      pmesh.UniformRefinement();
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
   Pr->Threshold(0.0);

   // 4. Get the DofTrueDof map on this mesh and convert the prolongation matrix
   // to correspond to global dof numbering (from true dofs to dofs)
   HypreParMatrix *DofTrueDof = aux_fespace->Dof_TrueDof_Matrix();
   HypreParMatrix *A = ParMult(DofTrueDof, Pr);
   HypreParMatrix *B = ParMult(A, cDofTrueDof->Transpose()); // This should be changed to RAP

   // 5. Now we compute the vertice that are owned by the process
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
   MPI_Comm comm = pmesh.GetComm();
   mynrpatch = cown_vertices.Size();
   // Compute total number of patches.
   MPI_Allreduce(&mynrpatch, &nrpatch, 1, MPI_INT, MPI_SUM, comm);
   patch_global_dofs_ids.SetSize(nrpatch);
   // Create a list of patches identifiers to all procs
   // This has to be changed to MPI_allGather

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
   // MPI_Gather(&cown_vertices[0],mynrpatch,MPI_INT,&patch_global_dofs_ids[0],mynrpatch,MPI_INT,0,comm);
   // MPI_Gatherv(&cown_vertices[0],mynrpatch,MPI_INT,&patch_global_dofs_ids[0],count,displs,MPI_INT,0,comm);
   // MPI_Bcast(&patch_global_dofs_ids[0],nrpatch,MPI_INT,0,comm);
   MPI_Allgatherv(&cown_vertices[0], mynrpatch, MPI_INT, &patch_global_dofs_ids[0], count, displs, MPI_INT, comm);

   int size = patch_global_dofs_ids[nrpatch - 1] + 1;
   patch_natural_order_idx.SetSize(size);
   // initialize with -1
   patch_natural_order_idx = -1;
   for (int i = 0; i < nrpatch; i++)
   {
      int k = patch_global_dofs_ids[i];
      patch_natural_order_idx[k] = i;
   }

   // On each processor identify the vertices that it owns (fine grid)
   SparseMatrix diag;
   DofTrueDof->GetDiag(diag);
   Array<int> own_vertices;
   int nv = 0;
   for (int k = 0; k < diag.Height(); k++)
   {
      int nz = diag.RowSize(k);
      int i = aux_fespace->GetMyDofOffset() + k;
      if (nz != 0)
      {
         nv++;
         own_vertices.SetSize(nv);
         own_vertices[nv - 1] = i;
      }
   }

   // For each vertex construct the list of patches that belongs to
   // First the patches that are already on the processor
   int mynrvertices = own_vertices.Size();
   vector<Array<int>> own_vertex_contr(mynrvertices);
   SparseMatrix H1pr_diag;
   B->GetDiag(H1pr_diag);
   for (int i = 0; i < mynrvertices; i++)
   {
      int kv = 0;
      int iv = own_vertices[i];
      int row = iv - aux_fespace->GetMyDofOffset();
      int row_size = H1pr_diag.RowSize(row);
      int *col = H1pr_diag.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = col[j] + mycdofoffset;
         if (its_a_patch(jv, patch_global_dofs_ids))
         {
            kv++;
            own_vertex_contr[i].SetSize(kv);
            own_vertex_contr[i][kv - 1] = jv;
         }
      }
   }
   // Next for the patches which are not owned by the processor.
   SparseMatrix H1pr_offd;
   int *cmap;
   B->GetOffd(H1pr_offd, cmap);
   for (int i = 0; i < mynrvertices; i++)
   {
      int kv = own_vertex_contr[i].Size();
      int iv = own_vertices[i];
      int row = iv - aux_fespace->GetMyDofOffset();
      int row_size = H1pr_offd.RowSize(row);
      int *col = H1pr_offd.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = cmap[col[j]];
         if (its_a_patch(jv, patch_global_dofs_ids))
         {
            kv++;
            own_vertex_contr[i].SetSize(kv);
            own_vertex_contr[i][kv - 1] = jv;
         }
      }
   }
   // Include also the vertices an each processor that are not owned
   // This will be helpfull when creating the list for edges, faces, elements.
   // Have to modify above to do this at once
   int allmyvert = pmesh.GetNV();
   vert_contr.resize(allmyvert);
   for (int i = 0; i < mynrvertices; i++)
   {
      int idx = own_vertices[i] - aux_fespace->GetMyDofOffset();
      int size = own_vertex_contr[i].Size();
      vert_contr[idx].SetSize(size);
      vert_contr[idx] = own_vertex_contr[i];
   }
   // -----------------------------------------------------------------------
   // done with vertices. Now the edges
   // -----------------------------------------------------------------------
   Array<int> edge_vertices;
   int nedge = pmesh.GetNEdges();
   edge_contr.resize(nedge);
   for (int ie = 0; ie < nedge; ie++)
   {
      pmesh.GetEdgeVertices(ie, edge_vertices);
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
   int nface = pmesh.GetNFaces();
   face_contr.resize(nface);
   for (int ifc = 0; ifc < nface; ifc++)
   {
      pmesh.GetFaceVertices(ifc, face_vertices);
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
   int nelem = pmesh.GetNE();
   elem_contr.resize(nelem);
   for (int iel = 0; iel < nelem; iel++)
   {
      pmesh.GetElementVertices(iel, elem_vertices);
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
}

void par_patch_nod_info::Print(int rankid)
{
   int num_procs, myid;
   MPI_Comm_size(pmesh.GetComm(), &num_procs);
   MPI_Comm_rank(pmesh.GetComm(), &myid);
   if (myid == rankid)
   {
      for (int i = 0; i < pmesh.GetNV(); i++)
      {
         cout << "vertex number, vertex id: " << i << ", " << i + aux_fespace->GetMyDofOffset() << endl;
         cout << "contributes to: ";
         vert_contr[i].Print();
      }
      Array<int> edge_vertices;
      for (int i = 0; i < pmesh.GetNEdges(); i++)
      {
         pmesh.GetEdgeVertices(i, edge_vertices);
         cout << "edge vertices are: " << edge_vertices[0] + aux_fespace->GetMyDofOffset() << " and "
              << edge_vertices[1] + aux_fespace->GetMyDofOffset() << endl;
         cout << "edge number: " << i;
         cout << " contributes to: ";
         edge_contr[i].Print();
         cout << endl;
      }
      int elem_offset;
      int nelem = pmesh.GetNE();
      MPI_Scan(&nelem, &elem_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      elem_offset -= nelem;
      if (myid == 1)
      {
         for (int i = 0; i < nelem; i++)
         {
            cout << "Element number: " << i + elem_offset;
            cout << " contributes to: ";
            elem_contr[i].Print();
            cout << endl;
         }
      }
   }
}

par_patch_dof_info::par_patch_dof_info(ParMesh *cpmesh_, int ref_levels_, ParFiniteElementSpace *fespace)
{
   par_patch_nod_info *patch_nodes = new par_patch_nod_info(cpmesh_, ref_levels_);

   int num_procs, myid;
   comm = cpmesh_->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // Build a list on each processor identifying the truedofs in each patch
   // First the vertices
   nrpatch = patch_nodes->nrpatch;
   vector<Array<int>> patch_dofs(nrpatch);

   int nrvert = fespace->GetNV();
   for (int i = 0; i < nrvert; i++)
   {
      int np = patch_nodes->vert_contr[i].Size();
      if (np == 0)
      {
         continue;
      }
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
            Array<int> temp(1);
            temp[0] = m;
            patch_dofs[kk].Append(temp);
         }
      }
   }
   int nedge = fespace->GetMesh()->GetNEdges();
   for (int i = 0; i < nedge; i++)
   {
      int np = patch_nodes->edge_contr[i].Size();
      if (np == 0)
      {
         continue;
      }
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
            Array<int> temp(1);
            temp[0] = m;
            patch_dofs[kk].Append(temp);
         }
      }
   }
   int nface = fespace->GetMesh()->GetNFaces();
   for (int i = 0; i < nface; i++)
   {
      int np = patch_nodes->face_contr[i].Size();
      if (np == 0)
      {
         continue;
      }
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
            Array<int> temp(1);
            temp[0] = m;
            patch_dofs[kk].Append(temp);
         }
      }
   }
   int nelem = fespace->GetNE();
   for (int i = 0; i < nelem; i++)
   {
      int np = patch_nodes->elem_contr[i].Size();
      if (np == 0) {continue;}
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
            Array<int> temp(1);
            temp[0] = m;
            patch_dofs[kk].Append(temp);
         }
      }
   }

   patch_tdofs.resize(nrpatch);
   for (int i = 0; i < nrpatch; i++)
   {
      Array<int> count(num_procs);
      int size = patch_dofs[i].Size();

      count[myid] = size;
      MPI_Allgather(&size, 1, MPI_INT, &count[0], 1, MPI_INT, comm);
      //
      Array<int>displs(num_procs);
      displs[0] = 0;
      for (int j = 1; j < num_procs; j++)
      {
         displs[j] = displs[j-1] + count[j-1];
      }

      int tot_size = displs[num_procs - 1] + count[num_procs - 1];

      // Get a group identifier for MPI_COMM_WORLD.
      MPI_Group world_group_id;
      MPI_Comm new_comm;
      MPI_Group new_group_id;
      MPI_Comm_group (comm, &world_group_id);
      //
      //  List the even processes, and create their group.
      // count the ranks that does not have zero length
      int num_ranks = 0;
      for (int i = 0; i<num_procs; i++)
      {
         if (count[i] != 0) {num_ranks++;}
      }
      Array<int> new_count(num_ranks);
      Array<int> new_displs(num_ranks);

      int including_ranks[num_ranks];
      num_ranks = 0;
      for (int j = 0; j <num_procs ; j++ )
      {
         if (count[j] != 0)
         {
            including_ranks[num_ranks] = j;
            new_count[num_ranks] = count[j];
            new_displs[num_ranks] = displs[j];
            num_ranks++;
         }
      }
      MPI_Group_incl(world_group_id, num_ranks, including_ranks, &new_group_id);
      MPI_Comm_create(comm, new_group_id, &new_comm);
      if (size != 0)
      {
         patch_tdofs[i].SetSize(tot_size);
         MPI_Allgatherv(&patch_dofs[i][0],size,MPI_INT,
                     &patch_tdofs[i][0],new_count,new_displs,MPI_INT,new_comm);
      }
   }
}

void par_patch_dof_info::Print()
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   for (int i = 0; i < nrpatch; i++)
   {
      for (int ii = 0; ii<num_procs; ii++)
      {
         if (myid == ii)
         {
            if (patch_tdofs[i].Size() != 0)
            {
               cout << "patch no: " << i << ", myid: " << myid 
               << ", patch_tdofs: "; patch_tdofs[i].Print(cout, 20);
            }
         }
      MPI_Barrier(comm);
      }
   }
}


par_patch_assembly::par_patch_assembly(ParMesh *cpmesh_, int ref_levels_, ParFiniteElementSpace *fespace)
{
   par_patch_dof_info *patch_tdof_info = new par_patch_dof_info(cpmesh_, ref_levels_,fespace);

   patch_tdof_info->Print();
}


bool its_a_patch(int iv, Array<int> patch_ids)
{
   if (patch_ids.FindSorted(iv) == -1)
   {
      return false;
   }
   else
   {
      return true;
   }
}
