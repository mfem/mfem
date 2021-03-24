
#include "mfem.hpp"
#include "schwarz.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


void print(std::vector<int> const &input)
{
	for (int i = 0; i < (int)input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
}

// constructor
patch_nod_info::patch_nod_info(Mesh *mesh_, int ref_levels_)
    : mesh(mesh_), ref_levels(ref_levels_)
{
   /* The patches are defined by all the "active" vertices of the coarse mesh
   We define a low order H1 fespace and perform refinements so that we can get
   the H1 prolongation operator recursively. This way we can easily find  
   all the patches that the fine mesh vertices contribute to. After the vertices 
   are done the edges, faces and elements can be found easily because they
   contribute to the same patches as their vertices. */

   // Number of patches
   nrpatch = mesh->GetNV();
   int dim = mesh->Dimension();
   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // First we need to construct a list of non-essential coarse grid vertices

   // SparseMatrix *Pr = nullptr;
   //initialize Pr with the Identity
   Vector ones(fespace->GetTrueVSize());
   ones = 1.0;
   SparseMatrix * Pr = new SparseMatrix(ones);
   // 4. Refine the mesh
   for (int i = 0; i < ref_levels; i++)
   {
      const FiniteElementSpace cfespace(*fespace);
      mesh->UniformRefinement();
      // Update fespace
      fespace->Update();
      OperatorHandle Tr(Operator::MFEM_SPARSEMAT);
      fespace->GetTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      SparseMatrix *P;
      Tr.Get(P);
      if (!Pr)
      {
         Pr = P;
      }
      else
      {
         Pr = Mult(*P, *Pr);
      }
   }
   // if there is no refinement the prolongation is the identity
   Pr->Threshold(0.0);
   int nvert = mesh->GetNV();
   vertex_contr.resize(nvert);
   for (int iv = 0; iv < nvert; iv++)
   {
      int nz = Pr->RowSize(iv);
      vertex_contr[iv].SetSize(nz);
      int *col = Pr->GetRowColumns(iv);
      for (int i = 0; i < nz; i++)
      {
         vertex_contr[iv][i] = col[i];
      }
   }

   delete Pr;

   Array<int> edge_vertices;
   int nedge = mesh->GetNEdges();
   edge_contr.resize(nedge);
   for (int ie = 0; ie < nedge; ie++)
   {
      mesh->GetEdgeVertices(ie, edge_vertices);
      int nv = edge_vertices.Size(); // always 2 but ok
      // The edge will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = edge_vertices[iv];
         edge_contr[ie].Append(vertex_contr[ivert]);
      }
      edge_contr[ie].Sort();
      edge_contr[ie].Unique();
   }

   Array<int> face_vertices;
   int nface = mesh->GetNFaces();
   face_contr.resize(nface);
   for (int ifc = 0; ifc < nface; ifc++)
   {
      mesh->GetFaceVertices(ifc, face_vertices);
      int nv = face_vertices.Size();
      // The face will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = face_vertices[iv];
         face_contr[ifc].Append(vertex_contr[ivert]);
      }
      face_contr[ifc].Sort();
      face_contr[ifc].Unique();
   }

   Array<int> elem_vertices;
   int nelem = mesh->GetNE();
   elem_contr.resize(nelem);
   for (int iel = 0; iel < nelem; iel++)
   {
      mesh->GetElementVertices(iel, elem_vertices);
      int nv = elem_vertices.Size();
      // The element will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = elem_vertices[iv];
         elem_contr[iel].Append(vertex_contr[ivert]);
      }
      elem_contr[iel].Sort();
      elem_contr[iel].Unique();
   }
   delete fespace;
   delete fec;
}
// Constructor of patch local problems
patch_assembly::patch_assembly(Mesh *cmesh_, int ref_levels_, FiniteElementSpace *fespace)
    : cmesh(*cmesh_), ref_levels(ref_levels_)
{
   patch_nod_info *patches = new patch_nod_info(&cmesh, ref_levels);

   nrpatch = patches->nrpatch;
   Pid.SetSize(nrpatch);
   patch_dof_map.SetSize(nrpatch);
   // Build a sparse matrix out of this map to extract the patch submatrix
   Array<int> dofoffset(nrpatch);
   dofoffset = 0;
   int height = fespace->GetVSize();
   // allocation of sparse matrices.
   for (int i = 0; i < nrpatch; i++)
   {
      Pid[i] = new SparseMatrix(height);
   }
   // Now the filling of the matrices with vertex,edge,face,interior dofs
   Mesh *mesh = fespace->GetMesh();
   int nrvert = mesh->GetNV();
   int nredge = mesh->GetNEdges();
   int nrface = mesh->GetNFaces();
   int nrelem = mesh->GetNE();
   // First the vertices
   for (int i = 0; i < nrvert; i++)
   {
      int np = patches->vertex_contr[i].Size();
      Array<int> vertex_dofs;
      fespace->GetVertexDofs(i, vertex_dofs);
      int nv = vertex_dofs.Size();

      for (int j = 0; j < np; j++)
      {
         int k = patches->vertex_contr[i][j];
         for (int l = 0; l < nv; l++)
         {
            int m = vertex_dofs[l];
            Pid[k]->Set(m, dofoffset[k], 1.0);
            dofoffset[k]++;
         }
      }
   }
   // Edges
   for (int i = 0; i < nredge; i++)
   {
      int np = patches->edge_contr[i].Size();
      Array<int> edge_dofs;
      fespace->GetEdgeInteriorDofs(i, edge_dofs);
      int ne = edge_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patches->edge_contr[i][j];
         for (int l = 0; l < ne; l++)
         {
            int m = edge_dofs[l];
            Pid[k]->Set(m, dofoffset[k], 1.0);
            dofoffset[k]++;
         }
      }
   }
   // Faces
   for (int i = 0; i < nrface; i++)
   {
      int np = patches->face_contr[i].Size();
      Array<int> face_dofs;
      fespace->GetFaceInteriorDofs(i, face_dofs);
      int nfc = face_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patches->face_contr[i][j];
         for (int l = 0; l < nfc; l++)
         {
            int m = face_dofs[l];
            Pid[k]->Set(m, dofoffset[k], 1.0);
            dofoffset[k]++;
         }
      }
   }

   // The following can be skipped in case of static condensation
   // Elements
   for (int i = 0; i < nrelem; i++)
   {
      int np = patches->elem_contr[i].Size();
      Array<int> elem_dofs;
      fespace->GetElementInteriorDofs(i, elem_dofs);
      int nel = elem_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patches->elem_contr[i][j];
         for (int l = 0; l < nel; l++)
         {
            int m = elem_dofs[l];
            Pid[k]->Set(m, dofoffset[k], 1.0);
            dofoffset[k]++;
         }
      }
   }

   for (int i = 0; i < nrpatch; i++)
   {
      Pid[i]->SetWidth(dofoffset[i]);
      Pid[i]->Finalize();

      patch_dof_map[i].SetSize(Pid[i]->Width());
      // copy from sparse matrix to a simple injection map
      // use the traspose
      SparseMatrix * temp = Transpose(*Pid[i]);
      // Extract row by row of the transpose
      for (int k =0; k<temp->Height(); ++k)
      {
         int * col = temp->GetRowColumns(k);
         patch_dof_map[i][k] = col[0];
      }
      delete temp;
   }
   delete patches;
}

patch_assembly:: ~patch_assembly()
{
   for (int i=0; i<nrpatch; i++)
   {
      delete Pid[i];
   }
   Pid.DeleteAll();
}

// constructor
SchwarzSmoother::SchwarzSmoother(Mesh *cmesh_, int ref_levels_, FiniteElementSpace *fespace_, SparseMatrix *A_, Array<int> ess_bdr)
    : Solver(A_->Height(), A_->Width()), A(A_)
{
   P = new patch_assembly(cmesh_, ref_levels_, fespace_);

   ess_bdr = 0;
   GetNonEssentialPatches(cmesh_, ess_bdr, patch_ids);

   // nrpatch = P->nrpatch;
   nrpatch = patch_ids.size();
   A_local.SetSize(nrpatch);
   invA_local.SetSize(nrpatch);

   for (int i = 0; i < nrpatch; i++)
   {
      int k = patch_ids[i];
      SparseMatrix *Pr = P->Pid[k];
      // construct the local problems. Factor the patch matrices
      A_local[i] = RAP(*Pr, *A, *Pr);
      // if (i == 0) A_local[i]->PrintMatlab(cout);
      invA_local[i] = new KLUSolver;
      // invA_local[i] = new UMFPackSolver;
      // invA_local[i]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      invA_local[i]->SetOperator(*A_local[i]);
   }
}

void SchwarzSmoother::GetNonEssentialPatches(Mesh *cmesh, const Array<int> &ess_bdr, vector<int> &patch_ids)
{
   Array<int> ess_vertices;
   Array<int> bdr_vertices;

   for (int i = 0; i < cmesh->GetNBE(); i++)
   {
      int bdr = cmesh->GetBdrAttribute(i);
      //check if it's essential;
      if (ess_bdr[bdr - 1] == 1)
      {
         cmesh->GetBdrElementVertices(i, bdr_vertices);
         ess_vertices.Append(bdr_vertices);
      }
   }
   ess_vertices.Sort();
   ess_vertices.Unique();

   int nrpatch = cmesh->GetNV() - ess_vertices.Size();
   patch_ids.resize(nrpatch);

   if (ess_vertices.Size() > 0)
   {
      int m = 0;
      int l = 0;
      for (int i = 0; i < cmesh->GetNV(); i++)
      {
         if (m<ess_vertices.Size() && i == ess_vertices[m])
         {
            m++;
         }
         else
         {
            patch_ids[l] = i;
            l++;
         }
      }
   }
   else
   {
      for (int i = 0; i < cmesh->GetNV(); i++)
      {
         patch_ids[i] = i;
      }
   }
}

void SchwarzSmoother::Mult(const Vector &r, Vector &z) const
{
   // Apply the smoother patch on the restriction of the residual
   z = 0.0;
   Vector rnew(r);
   Vector znew(z);
   Vector raux(znew.Size());
   Vector res_local, sol_local;
   switch (sType)
   {
   case Schwarz::SmootherType::ADDITIVE:
   {
      for (int iter = 0; iter < maxit; iter++)
      {
         znew = 0.0;
         for (int i = 0; i < nrpatch; i++)
         {
            int k = patch_ids[i];
            Array<int> * dof_map = &P->patch_dof_map[k];
            // SparseMatrix *Pr = P->Pid[k];
            // res_local.SetSize(Pr->NumCols());
            // sol_local.SetSize(Pr->NumCols());
            // Pr->MultTranspose(rnew, res_local[i]);
            int ndofs = dof_map->Size();
            res_local.SetSize(ndofs);
            sol_local.SetSize(ndofs);
            rnew.GetSubVector(*dof_map, res_local);

            invA_local[i]->Mult(res_local, sol_local);
            znew.AddElementVector(*dof_map,sol_local);
            // Pr->Mult(sol_local[i], zaux[i]);
            // znew += zaux[i];
         }
         // Relaxation parameter
         znew *= theta;
         z += znew;

         //Update residual
         if (iter + 1 < maxit)
         {
            A->Mult(znew, raux);
            rnew -= raux;
         }
      }
   }
   break;
   case Schwarz::SmootherType::MULTIPLICATIVE:
   {
      //   TODO
   }
   break;
   case Schwarz::SmootherType::SYM_MULTIPLICATIVE:
   {
      //   TODO
   }
   break;
   }
}

SchwarzSmoother:: ~SchwarzSmoother() 
{
   delete P;
   for (int ip=0; ip<nrpatch; ++ip)
   {
      delete A_local[ip]; 
      delete invA_local[ip]; 
   }
   A_local.DeleteAll();
   invA_local.DeleteAll();
}



BlkSchwarzSmoother::BlkSchwarzSmoother(Mesh *cmesh_, int ref_levels_, FiniteElementSpace* fespace_, SparseMatrix *A_) 
: Solver(A_->Height(), A_->Width()), A(A_)
{
   P = new patch_assembly(cmesh_, ref_levels_, fespace_);

   nrpatch = cmesh_->GetNV();
	cout << "nrpatch = " << nrpatch << endl;
   patch_ids.resize(nrpatch);
   for (int i=0; i<nrpatch; i++) {patch_ids[i]=i;}

   nrpatch = patch_ids.size();
   A_local.SetSize(nrpatch);
   invA_local.SetSize(nrpatch);

   for (int i = 0; i < nrpatch; i++)
   {
      int k = patch_ids[i];
      SparseMatrix *Pr  = P->Pid[k];
      Array<int> offsets_i(3);
      Array<int> offsets_j(3);
      offsets_i[0] = 0; 
      offsets_i[1] = Pr->Height();
      offsets_i[2] = Pr->Height();
      offsets_i.PartialSum();
      offsets_j[0] = 0; 
      offsets_j[1] = Pr->Width();
      offsets_j[2] = Pr->Width();
      offsets_j.PartialSum();

      BlockMatrix * BlockPr = new BlockMatrix(offsets_i,offsets_j);
      BlockPr->SetBlock(0,0,Pr);
      BlockPr->SetBlock(1,1,Pr);
      // Fake blocks
      SparseMatrix * fakemat = new SparseMatrix(Pr->Height(),Pr->Width()); fakemat->Finalize();
      BlockPr->SetBlock(0,1,fakemat);
      BlockPr->SetBlock(1,0,fakemat);

      SparseMatrix *Bpr = BlockPr->CreateMonolithic();
      A_local[i] = RAP(*Bpr, *A, *Bpr);
      invA_local[i] = new UMFPackSolver;
      invA_local[i]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      invA_local[i]->SetOperator(*A_local[i]);
   }
}


void BlkSchwarzSmoother::Mult(const Vector &r, Vector &z) const
{
   // Apply the smoother patch on the restriction of the residual
   Array<Vector> res_local(nrpatch);
   Array<Vector> sol_local(nrpatch);
   Array<Vector> zaux(nrpatch);
   z = 0.0;
   Vector rnew(r);
   Vector znew(z);

   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;
      for (int i = 0; i < nrpatch; i++)
      {
         int k = patch_ids[i];
         SparseMatrix *Pr = P->Pid[k];
         Array<int> offsets_i(3);
         Array<int> offsets_j(3);
         offsets_i[0] = 0; 
         offsets_i[1] = Pr->Height();
         offsets_i[2] = Pr->Height();
         offsets_i.PartialSum();
         offsets_j[0] = 0; 
         offsets_j[1] = Pr->Width();
         offsets_j[2] = Pr->Width();
         offsets_j.PartialSum();
         BlockMatrix * BlockPr = new BlockMatrix(offsets_i,offsets_j);
         BlockPr->SetBlock(0,0,Pr);
         BlockPr->SetBlock(1,1,Pr);
         SparseMatrix * fakemat = new SparseMatrix(Pr->Height(),Pr->Width()); fakemat->Finalize();
         BlockPr->SetBlock(0,1,fakemat);
         BlockPr->SetBlock(1,0,fakemat);
         SparseMatrix *Bpr = BlockPr->CreateMonolithic();
         res_local[i].SetSize(Bpr->NumCols());
         sol_local[i].SetSize(Bpr->NumCols());
         Bpr->MultTranspose(rnew, res_local[i]);

         invA_local[i]->Mult(res_local[i], sol_local[i]);
         zaux[i].SetSize(r.Size());
         zaux[i] = 0.0;
         Bpr->Mult(sol_local[i], zaux[i]);
         znew += zaux[i];
      }

      // Relaxation parameter
      znew *= theta;
      z += znew;
      //Update residual
      Vector raux(znew.Size());
      A->Mult(znew, raux);
      rnew -= raux;
   }
}