#include "element-smoother.hpp"

using namespace std;
using namespace mfem;


ElementSmoother::ElementSmoother(ParFiniteElementSpace * fes_,  
Array<int> ess_bdr, Coefficient * cf_ )
: Solver(fes_->GetTrueVSize()), fes(fes_), cf(cf_)
{  
   comm = fes->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   Pr = fes->GetProlongationMatrix();
   ParMesh * pmesh = fes->GetParMesh();
   dim = pmesh->Dimension();
   eidx.SetSize(dim);
   eidx[0] = 0;
   eidx[1] = 1;
   if (dim == 3) eidx[2] = 8;
   nrelems = pmesh->GetNE();
   tpcf.SetSize(nrelems);
   int vsize = fes->GetVSize();
   ovlp_count.SetSize(vsize);
   ovlp_count = 0.0;
   // Construct overlap count for each dof &
   // count x y z edges sharing a vertex
   for (int i=0; i<nrelems; i++)
   {
      Array<int> elem_dofs;
      fes->GetElementDofs(i,elem_dofs);
      for (int j = 0; j<elem_dofs.Size(); j++)
      {
         ovlp_count(elem_dofs[j]) +=1.0;
      }
   }


   Vector tovlp_count(fes->TrueVSize());
   if (Pr)
   {
      Pr->MultTranspose(ovlp_count,tovlp_count);
      Pr->Mult(tovlp_count, ovlp_count);
   }

   DenseMatrix edge_counts;
   GetVertexToEdgeCount(pmesh,edge_counts);
   double * data = edge_counts.GetData();

   // helper H1 fespace for communication of vertex info
   H1_FECollection fec(1, dim); 
   ParFiniteElementSpace aux_fes(const_cast<ParMesh *>(pmesh), &fec);
   Vector tedge_counts(aux_fes.TrueVSize());
   Vector temp(aux_fes.GetVSize());


   const Operator * Ph = aux_fes.GetProlongationMatrix();

   if (Ph)
   {
      for (int d=0; d<dim; d++)
      {
         temp.SetData(&data[d*aux_fes.GetVSize()]);
         Ph->MultTranspose(temp,tedge_counts);
         Ph->Mult(tedge_counts, temp);
      }
   }

   for (int i = 0; i < vsize; i++)
   {
      ovlp_count(i) = 1.0/sqrt(ovlp_count(i)); 
   }

   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   TPElementTransformation TPTrans(*fes);
   
   int nredges = pmesh->GetNEdges();
   edge_orient.SetSize(nredges); // orientation of edges wrt the element
   Array<DenseMatrix *> EdgeGrad(nredges);
   Array<DenseMatrix *> EdgeMass(nredges);
   Array<DenseMatrix *> AssembledEdgeGrad(nredges);
   Array<DenseMatrix *> AssembledEdgeMass(nredges);
   // Initialize
   for (int i = 0; i<nredges; i++)
   {
      AssembledEdgeGrad[i] = nullptr;
      AssembledEdgeMass[i] = nullptr;
      EdgeGrad[i] = nullptr;
      EdgeMass[i] = nullptr;
   }

   // loop through element to calculate Edge matrices
   Array<int> emarker(nredges); emarker = 0;
   for (int iel = 0; iel<nrelems; iel++)
   {
      tpcf[iel] = new ElementTPFunctionCoefficient(*fes,iel,*cf);
      Array<int> edges, cor;
      pmesh->GetElementEdges(iel,edges,cor);
      for (int ii = 0; ii<dim; ii++) 
      {   
         int i = eidx[ii];
         int edge = edges[i];
         if (emarker[edge]) continue;
         edge_orient[edge] = cor[i];
         const FiniteElement * fe = fes->GetEdgeElement(edge);

         tpcf[iel]->SetCoord(ii);
         tpcf[iel]->SetOrient(edge_orient[edge]);
         tpcf[iel]->ResetCounter(ii);
         IntegrationRule *irs = TensorIntegrationRule(1,fe->GetOrder());
         EdgeGrad[edge] = new DenseMatrix(fe->GetDof());
         EdgeMass[edge] = new DenseMatrix(fe->GetDof());

         int j;
         Vector * Q;
         switch (ii)
         {
         case 0: 
            j = 1; 
            Q = tpcf[iel]->GetVecX();
            break;
         case 1: 
            j = 0; 
            Q = tpcf[iel]->GetVecY();
            break;  
         default:
            j=2; 
            Q = tpcf[iel]->GetVecZ();
            break;
         }

         Vector * vecG = TPTrans.GetTPTransformation(iel,ii,ii);
         Vector * vecM = nullptr;
         Vector *vecM1 = nullptr;
         Vector *vecM2 = nullptr;
         if (dim == 2)
         {
            vecM = TPTrans.GetTPTransformation(iel,ii,j);
         }
         else
         {
            switch (ii)
            {
            case 0:
            {
               vecM1 = TPTrans.GetTPTransformation(iel,ii,1);
               vecM2 = TPTrans.GetTPTransformation(iel,ii,2);
            }
            break;
            case 1:
            {
               vecM1 = TPTrans.GetTPTransformation(iel,ii,0);
               vecM2 = TPTrans.GetTPTransformation(iel,ii,2);
            }            
            default:
            {
               vecM1 = TPTrans.GetTPTransformation(iel,ii,0);
               vecM2 = TPTrans.GetTPTransformation(iel,ii,1);
            }
               break;
            }
            vecM = new Vector(vecM1->Size());
            for (int i=0; i<vecM->Size(); i++)
            {
               // (*vecM)(i) = ((*vecM1)(i)+(*vecM2)(i))/2.0;
               // (*vecM)(i) = (*vecM1)(i);
               (*vecM)(i) = 1.0/((1./(*vecM1)(i)+1./(*vecM2)(i))/2.0);
            }   
         }
         // GetDiffusionEdgeMatrix(edge,fes,*vecG,*Q,irs,*EdgeGrad[edge],edge_orient[edge]);
         // GetMassEdgeMatrix(edge,fes,*vecM,*Q,irs,*EdgeMass[edge],edge_orient[edge]);
         Get1DMatrices(fes,edge, edge_orient[edge],
                       *vecG, *vecM,*Q,irs,*EdgeGrad[edge],*EdgeMass[edge]);
         emarker[edge] = 1;
      }
   }

   
   for (int iel=0; iel<nrelems; iel++)
   {
      Array<int> edges, cor;
      pmesh->GetElementEdges(iel,edges,cor);

      for (int ii = 0; ii<dim; ii++) 
      {   
         int i = eidx[ii];
         int k = edges[i];
         const FiniteElement *fe = fes->GetEdgeElement(k);
         int ndof = fe->GetDof();
         DenseMatrix Grad(ndof); Grad = *EdgeGrad[k];
         DenseMatrix Mass(ndof); Mass = *EdgeMass[k];

         Array<int> vert;
         pmesh->GetEdgeVertices(k,vert);
         for (int i = 0; i<2; i++)
         {
            int vertex = vert[i];
            Array<int> vertdofs;
            fes->GetVertexDofs(vertex,vertdofs);
            double count = edge_counts(vertdofs[0],ii); 

            Grad(i,i) *= count;
            Mass(i,i) *= count;
         }
         const Array<int> &dmap =
         dynamic_cast<const TensorBasisElement&>(*fe).GetDofMap();
         SparseMatrix * P = new SparseMatrix(dmap.Size());
         for (int j = 0; j<dmap.Size(); j++)
         {
            P->Set(dmap[j],j, 1.0);
         }
         P->Finalize();
         // Map from MFEM ordering to TensorProduct Ordering
         AssembledEdgeGrad[k] = RAP(Grad,*P);
         AssembledEdgeMass[k] = RAP(Mass,*P);
         delete P;
      }
   }

   Array<Array<int> * > tmap;
   TensorProductEssentialDofsMaps(ess_tdof_list, fes, tmap, dofmap);


   Array<DenseMatrix * > G(nredges);
   Array<DenseMatrix * > M(nredges);

   for (int ie = 0; ie<nredges; ie++)
   {
      if (!emarker[ie]) continue;
      if (!AssembledEdgeMass[ie])
      {
         cout << "ie = " << ie << endl;
         MFEM_ABORT("Memory allocation incosistency 2");
      }
      const FiniteElement * fe = fes->GetEdgeElement(ie);
      int n = fe->GetDof() - tmap[ie]->Size();
      G[ie] = new DenseMatrix(n);
      M[ie] = new DenseMatrix(n);
      const Array<int> &dmap =
      dynamic_cast<const TensorBasisElement&>(*fe).GetDofMap();
      // modify tmap to use tensor product index;
      Array<int> dmapt(dmap.Size());
      for (int i = 0; i<dmap.Size(); i++)
      {
         dmapt[dmap[i]] = i;
      }

      // Eliminate indices corresponding to tmap from matrices Grad1D_A and Mass1D_A
      // construct Map;
      Array<int> tmap_marker(fe->GetDof());
      tmap_marker = 0;
      for (int i = 0; i<tmap[ie]->Size(); i++)
      {
         int j = (*tmap[ie])[i];
         tmap_marker[j] = 1;
      }
      Array<int> dof_list;
      for (int i =0; i<fe->GetDof(); i++)
      {
         if (tmap_marker[i]) continue;
         dof_list.Append(dmapt[i]);
      }
      dof_list.Sort();
      for (int i=0; i<dof_list.Size(); i++)
      {
         int iii = (edge_orient[ie] == 1) ? i : n - i - 1; 
         int ii = dof_list[i];
         for (int j=0; j<dof_list.Size(); j++)
         {
            int jjj = (edge_orient[ie] == 1) ? j : n - j - 1; 
            int jj = dof_list[j];
            (*G[ie])(iii,jjj) = (*AssembledEdgeGrad[ie])(ii,jj);
            (*M[ie])(iii,jjj) = (*AssembledEdgeMass[ie])(ii,jj);
         }
      }
   }

   elem_inv.SetSize(nrelems);
   for (int iel = 0; iel<nrelems; iel++)
   {
      elem_inv[iel] = nullptr;
      Array<int> edges, cor;
      pmesh->GetElementEdges(iel,edges,cor);
      Array<DenseMatrix *> Gv(dim);
      Array<DenseMatrix *> Mv(dim);
      bool msize = true;
      for (int d = 0; d<dim; d++)
      {
         Gv[dim - d - 1] = G[edges[eidx[d]]];
         Mv[dim - d - 1] = M[edges[eidx[d]]];
         if (Mv[dim-d-1]->Size() == 0) msize = false;
      }
      if (msize) elem_inv[iel] = new FDSolver(Gv,Mv);
   }
}


void ElementSmoother::Mult(const Vector &r, Vector &z) const
{
   z = r;
   z.SetSubVectorComplement(ess_tdof_list,0.0);
   Vector rnew(fes->GetVSize());
   Vector znew(fes->GetVSize()); 
   Vector ztemp(fes->GetTrueVSize()); 
   ztemp = 0.0;
   znew = 0.0;

   // const SparseMatrix * R = fes->GetRestrictionMatrix();
   if (Pr)
   {
      Pr->Mult(r,rnew);
   }
   else
   {
      rnew = r;
   }
   for (int iel=0; iel<nrelems; iel++)
   {
      if (!elem_inv[iel]) continue;
      int n = dofmap[iel]->Size();

      Vector rloc(n);
      rnew.GetSubVector(*dofmap[iel],rloc);
      // pre-scale
      for (int i = 0; i<n; i++)
      {
         int j = (*dofmap[iel])[i];
         rloc[i] *= ovlp_count[j]; 
      }
      Vector zloc(n);
      elem_inv[iel]->Mult(rloc,zloc);
      // post-scale
      for (int i = 0; i<n; i++)
      {
         int j = (*dofmap[iel])[i];
         zloc[i] *= ovlp_count[j]; 
      }

      znew.AddElementVector(*dofmap[iel],zloc);
   }
   if (Pr)
   {
      Pr->MultTranspose(znew,ztemp);
   }
   else
   {
      ztemp = znew;
   }
   // R->Mult(znew,ztemp);
   z += ztemp;
}
