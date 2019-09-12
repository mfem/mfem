
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
   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(&pmesh, fec);
   int mycdofoffset = fespace->GetMyDofOffset(); // dof offset for the coarse mesh


   // 2. Store the cDofTrueDof Matrix. Required after the refinements
   HypreParMatrix * cDofTrueDof = new HypreParMatrix(*fespace->Dof_TrueDof_Matrix());

   // 3. Perform the refinements and Get the final Prolongation operator
   HypreParMatrix *Pr = nullptr;
   for (int i = 0; i < ref_levels; i++)
   {
      const ParFiniteElementSpace cfespace(*fespace);
      pmesh.UniformRefinement();
      // Update fespace
      fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      fespace->GetTrueTransferOperator(cfespace, Tr);
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
   HypreParMatrix *DofTrueDof = fespace->Dof_TrueDof_Matrix();
   HypreParMatrix * A = ParMult(DofTrueDof,Pr);
   HypreParMatrix * B = ParMult(A,cDofTrueDof->Transpose()); // This should be changed to RAP

   // 5. Now we compute the vertice that are owned by the process

   SparseMatrix cdiag, coffd;
   cDofTrueDof->GetDiag(cdiag);
   Array<int> cown_vertices;
   int cnv=0;
   for (int k=0; k<cdiag.Height(); k++)
   {
      int nz = cdiag.RowSize(k);
      int i = mycdofoffset + k;
      if (nz != 0)
      {
         cnv++;
         cown_vertices.SetSize(cnv);
         cown_vertices[cnv-1] = i;
      }
   }

   // 6. Compute total number of patches 
   MPI_Comm comm = pmesh.GetComm(); 
   int mynrpatch = cown_vertices.Size();
   int nrpatch;
   // Compute total number of patches.
   MPI_Allreduce(&mynrpatch,&nrpatch,1,MPI_INT,MPI_SUM,comm);
   Array <int> patch_global_dofs_ids(nrpatch);
   // Create a list of patches identifiers to all procs
   // This has to be changed to MPI_allGather
   MPI_Gather(&cown_vertices[0],mynrpatch,MPI_INT,&patch_global_dofs_ids[0],mynrpatch,MPI_INT,0,comm);
   MPI_Bcast(&patch_global_dofs_ids[0],nrpatch,MPI_INT,0,comm);

   // On each processor identify the vertices that it owns (fine grid)
   SparseMatrix diag;
   DofTrueDof->GetDiag(diag);
   Array<int> own_vertices;
   int nv=0;
   for (int k=0; k<diag.Height(); k++)
   {
      int nz = diag.RowSize(k);
      int i = fespace->GetMyDofOffset() + k;
      if (nz != 0)
      {
         nv++;
         own_vertices.SetSize(nv);
         own_vertices[nv-1] = i;
      }
   }

   // For each vertex construct the list of patches that belongs to
   // First the patches that are already on the processor
   int mynrvertices = own_vertices.Size();
   vector<Array<int>> own_vertex_contr(mynrvertices);
   SparseMatrix H1pr_diag;
   B->GetDiag(H1pr_diag);
   for (int i = 0; i<mynrvertices; i++)
   {
      int kv = 0;
      int iv = own_vertices[i];
      int row = iv - fespace->GetMyDofOffset();
      int row_size = H1pr_diag.RowSize(row);
      int *col = H1pr_diag.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = col[j] + mycdofoffset;
         if (its_a_patch(jv,patch_global_dofs_ids))
         {
            kv++;
            own_vertex_contr[i].SetSize(kv);
            own_vertex_contr[i][kv-1] = jv; 
         }
      }
   }
   // Next for the patches which are not owned by the processor.
   SparseMatrix H1pr_offd;
   int *cmap;
   B->GetOffd(H1pr_offd,cmap);
   for (int i = 0; i<mynrvertices; i++)
   {
      int kv = own_vertex_contr[i].Size();
      int iv = own_vertices[i];
      int row = iv - fespace->GetMyDofOffset();
      int row_size = H1pr_offd.RowSize(row);
      int *col = H1pr_offd.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = cmap[col[j]];
         if (its_a_patch(jv,patch_global_dofs_ids))
         {
            kv++;
            own_vertex_contr[i].SetSize(kv);
            own_vertex_contr[i][kv-1] = jv; 
         }
      }
   }
   // Include also the vertices an each processor that are not owned 
   // This will be helpfull when creating the list for edges, faces, elements. 
   // Have to modify above to do this at once
   int allmyvert = pmesh.GetNV();
   vert_contr.resize(allmyvert);
   for (int i = 0; i<mynrvertices; i++)
   {
      int idx = own_vertices[i]-fespace->GetMyDofOffset();
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

bool its_a_patch(int iv, Array<int> patch_ids)
{
   if (patch_ids.FindSorted(iv)== -1)
   {
      return false;
   }
   else
   {
      return true;
   }
}
