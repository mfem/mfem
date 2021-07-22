

#include "smoother-util.hpp"

IntegrationRule * TensorIntegrationRule(const FiniteElementSpace & fes, int order) 
{
   Mesh *mesh = fes.GetMesh();
   IntegrationRule * ir;
   int ir_order = 2*order+2;
   IntegrationRules IntRule(0, Quadrature1D::GaussLobatto);
   // IntegrationRules IntRule(0, Quadrature1D::GaussLegendre);

   int dim = mesh->Dimension();
   switch (dim)
   {
   case 1: 
      ir = new IntegrationRule(IntRules.Get(mfem::Geometry::SEGMENT, ir_order));
      break;
   case 2: 
      ir = new IntegrationRule(IntRules.Get(mfem::Geometry::SQUARE, ir_order));
      break;   
   default:
      ir = new IntegrationRule(IntRules.Get(mfem::Geometry::CUBE, ir_order));
      break;
   }
   return ir;
}
IntegrationRule * TensorIntegrationRule(int dim, int order) 
{
   IntegrationRule * ir;
   int ir_order = 2*order+2;
   // IntegrationRules IntRule(0, Quadrature1D::GaussLegendre);
   IntegrationRules IntRule(0, Quadrature1D::GaussLobatto);
   switch (dim)
   {
   case 1: 
      ir = new IntegrationRule(IntRules.Get(mfem::Geometry::SEGMENT, ir_order));
      break;
   case 2: 
      ir = new IntegrationRule(IntRules.Get(mfem::Geometry::SQUARE, ir_order));
      break;   
   default:
      ir = new IntegrationRule(IntRules.Get(mfem::Geometry::CUBE, ir_order));
      break;
   }
   return ir;
}

void KronMult(const Vector & x, const Vector & y, Vector & z)
{
   int n = x.Size();
   int m = y.Size();
   z.SetSize(n*m);
   for (int i=0; i<n; i++)
   {
      for (int j = 0; j<m; j++)
      {
         z(i*m+j) = x(i)*y(j);
      }
   }
}

void KronMult(const Vector & x, const Vector & y, const Vector & z, Vector & w)
{
   Vector xy;
   KronMult(x,y,xy);
   KronMult(xy,z,w);
}

void AlterLS(DenseMatrix & T, Vector & vecA, Vector & vecB)
{
   int n = T.Height();
   int m = T.Width();
   Vector x(n);
   Vector temp(m);
   double s;
   Vector y(m); y.Randomize(1); y /= y.Norml2();
   int maxit = 3;
   for (int i=0; i<maxit; i++)
   {
      T.Mult(y,x); x /= x.Norml2();
      T.MultTranspose(x,temp); y=temp; y/= y.Norml2();
      s = InnerProduct(y,temp);
   }
   vecA = x; vecA *= sqrt(s);
   vecB = y; vecB *= sqrt(s);
}

void AlterLS(DenseTensor & T, Vector & vecA, Vector & vecB, Vector & vecC)
{
   int n = T.SizeI();
   int m = T.SizeJ();
   int l = T.SizeK();

   DenseMatrix A0(n,l*m);
   DenseMatrix A1(m,l*n);
   DenseMatrix A2(l,n*m);

   for (int i=0; i<n; ++i)
   {
      for (int j=0; j<m; ++j)
      {
         for (int k=0; k<l; ++k)
         {
            A0(i,j+k*m) = T(i,j,k);
            A1(j,i+k*n) = T(i,j,k);
            A2(k,i+j*n) = T(i,j,k);
         }
      }
   }

   double s;
   Vector x(n);
   Vector temp;
   // Vector temp;
   Vector y(m); y.Randomize(1); y/=y.Norml2();
   Vector z(l); z.Randomize(2); z/=z.Norml2();

   int maxit = 3;
   for (int i = 0; i<maxit; i++)
   {
      KronMult(z,y,temp); A0.Mult(temp,x); s=x.Norml2(); x/=s;
      KronMult(z,x,temp); A1.Mult(temp,y); s=y.Norml2(); y/=s;
      KronMult(y,x,temp); A2.Mult(temp,z); s=z.Norml2(); z/=s;
   }
   vecA = x; vecA *= cbrt(s);
   vecB = y; vecB *= cbrt(s);
   vecC = z; vecC *= cbrt(s);
}

ElementTPFunctionCoefficient::ElementTPFunctionCoefficient(FiniteElementSpace &fes, int iel, Coefficient &cf)
{
   coeff_avg = 0.0;
   dim = fes.GetMesh()->Dimension();
   ElementTransformation * Tr;
   const IntegrationRule * ir;
   const FiniteElement * fe = fes.GetFE(iel);
   ir = TensorIntegrationRule(fes,fe->GetOrder()); 
   int nint1D = (dim == 2) ? sqrt(ir->GetNPoints()) : cbrt(ir->GetNPoints());
   int nrintx = nint1D;
   int nrinty = nint1D;
   int nrintz = (dim ==2) ? 0 : nint1D;
   if (dim == 2)
   {
      A.SetSize(nrintx,nrinty);
      Tr = fes.GetElementTransformation(iel);
      nint = ir->GetNPoints();
      for (int i = 0; i < nint; i++)
      {
         int nint1D = sqrt(ir->GetNPoints());
         int iy = i/nint1D;
         int ix = i%nint1D;
         const IntegrationPoint &ip = ir->IntPoint(i);
         Vector x(dim);
         Tr->Transform(ip,x);
         double val = cf.Eval(*Tr, ip);
         A(ix,iy) = val;
         coeff_avg += val;
      }
      AlterLS(A,VecX,VecY);
      coeff_avg /= (double)nint;
   }
   else
   {
      T.SetSize(nrintx,nrinty,nrintz);
      Tr = fes.GetElementTransformation(iel);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         int nint1D = cbrt(ir->GetNPoints());
         int iz = i/(nint1D*nint1D);
         int iy = (i - iz*nint1D*nint1D)/nint1D;
         int ix = (i - iz*nint1D*nint1D)%nint1D;
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr->SetIntPoint (&ip);
         double val = cf.Eval(*Tr, ip);
         T(ix,iy,iz) = val;
      }
      AlterLS(T,VecX,VecY,VecZ);
   }
   delete ir;
}


double ElementTPFunctionCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   double val = 0.0;
   switch (coord)
   {
      case 0: 
      {
         if (orient == 1)
         {
            val = VecX(nintx++);
         }
         else
         {
            int nend = VecX.Size();   
            val = VecX(nend-1-nintx++);
         }
      }
      break;
      case 1: 
      {
         if (orient == 1)
         {
            val = VecY(ninty++);
         }
         else
         {
            int nend = VecY.Size();   
            val = VecY(nend-1-ninty++);
         }
      } 
      break;
      case 2: 
      {
         if (orient == 1)
         {
            val = VecZ(nintz++); 
         }
         else
         {
            int nend = VecZ.Size();   
            val = VecZ(nend-1-nintz++);
         }
      }
      break;
      case -1:
      {
         int nint1D = sqrt(nint);
         int iy = mint/nint1D;
         int ix = mint%nint1D;
         val = VecX(ix) * VecY(iy);
         mint++;
      }
      break;
      case -2:
      {
         int nint1D = cbrt(nint);
         int iz = mint/(nint1D*nint1D);
         int iy = (mint - iz*nint1D*nint1D)/nint1D;
         int ix = (mint - iz*nint1D*nint1D)%nint1D;
         val = VecX(ix) * VecY(iy) * VecZ(iz);
         mint++;
      }
      break;
      default: MFEM_ABORT("ElementTPFunctionCoefficient::Eval: Wrong coord choice"); 
         break;
   }
   return val;
}


void TPElementTransformation::Setup2D()
{
   Mesh * mesh = fes->GetMesh();
   MFEM_VERIFY(dim == 2, "Wrong dimension");
   int nel = mesh->GetNE();
   TransA1D.SetSize(nel,dim);
   TransB1D.SetSize(nel,dim);
   // Get ElementTransformations for the 2D elements
   for (int iel = 0; iel <nel; iel++)
   {
      // allocate memory for Trans1D
      for (int d = 0; d<dim; d++)
      {
         TransA1D[iel][d] = new Vector;
         TransB1D[iel][d] = new Vector;
      }
      ElementTransformation * T = mesh->GetElementTransformation(iel);
      // Populate integrations points and get the K = adj(J)/sqrt(detJ);
      // Store K_11^2 + K_12^2 
      //       K_21^2 + K_22^2 

      const FiniteElement * fe = fes->GetFE(iel);
      const IntegrationRule * ir = TensorIntegrationRule(*fes,fe->GetOrder()); 
      int nint = ir->GetNPoints();
      int nint1D = sqrt(nint);
      DenseMatrix A, B;
      A.SetSize(nint1D,nint1D);
      B.SetSize(nint1D,nint1D);

      for (int i = 0; i < nint; i++)
      {
         int iy = i/nint1D;
         int ix = i%nint1D;
         const IntegrationPoint &ip = ir->IntPoint(i);
         T->SetIntPoint(&ip);

         double detJ = T->Weight();
         const DenseMatrix & adjJ = T->AdjugateJacobian();
         DenseMatrix JtJ(adjJ.Height());
         MultAtB(adjJ,adjJ,JtJ);
         JtJ *= 1.0/abs(detJ);
         // DenseMatrix adjtt(adjJ.Height());
         // MultAtB(adjJ, adjJ, adjtt);
         double valA = pow(abs(adjJ(0,0))+abs(adjJ(0,1)),2)/abs(detJ);
         // double valA = abs(JtJ(0,0))+abs(JtJ(0,1));
         double valB = pow(abs(adjJ(1,1))+abs(adjJ(1,0)),2)/abs(detJ);
         // double valB = abs(JtJ(1,1))+abs(JtJ(1,0));
         A(ix,iy) = valA;
         B(ix,iy) = valB;
      }
      AlterLS(A,*TransA1D[iel][0],*TransA1D[iel][1]);
      AlterLS(B,*TransB1D[iel][0],*TransB1D[iel][1]);
   }
}
void TPElementTransformation::Setup3D()
{
   Mesh * mesh = fes->GetMesh();
   MFEM_VERIFY(dim == 3, "Wrong dimension");
   int nel = mesh->GetNE();
   TransA1D.SetSize(nel,dim);
   TransB1D.SetSize(nel,dim);
   TransC1D.SetSize(nel,dim);
   // Get ElementTransformations for the 3D elements
   for (int iel = 0; iel <nel; iel++)
   {
      // allocate memory for Trans1D
      for (int d = 0; d<dim; d++)
      {
         TransA1D[iel][d] = new Vector;
         TransB1D[iel][d] = new Vector;
         TransC1D[iel][d] = new Vector;
      }
      ElementTransformation * T = mesh->GetElementTransformation(iel);
      // Populate integrations points and get the K = adj(J)/sqrt(detJ);
      // Store K_11^2 + K_12^2 + K_13^2
      //       K_21^2 + K_22^2 + K_23^2
      //       K_31^2 + K_32^2 + K_33^2

      const FiniteElement * fe = fes->GetFE(iel);
      const IntegrationRule * ir = TensorIntegrationRule(*fes,fe->GetOrder()); 
      int nint = ir->GetNPoints();
      int nint1D = cbrt(nint);
      DenseTensor A, B, C;
      A.SetSize(nint1D,nint1D,nint1D);
      B.SetSize(nint1D,nint1D,nint1D);
      C.SetSize(nint1D,nint1D,nint1D);

      for (int i = 0; i < nint; i++)
      {
         int iz = i/(nint1D*nint1D);
         int iy = (i - iz*nint1D*nint1D)/nint1D;
         int ix = (i - iz*nint1D*nint1D)%nint1D;

         const IntegrationPoint &ip = ir->IntPoint(i);
         T->SetIntPoint(&ip);

         double detJ = T->Weight();
         const DenseMatrix & adjJ = T->AdjugateJacobian();
         // DenseMatrix JtJ(adjJ.Height());
         // MultAtB(adjJ,adjJ,JtJ);
         // JtJ *= 1.0/abs(detJ);
         // double valA = abs(JtJ(0,0))+abs(JtJ(0,1))+abs(JtJ(0,2));
         double valA = pow(abs(adjJ(0,0))+abs(adjJ(0,1))+abs(adjJ(0,2)),2)/abs(detJ);
         // double valB = abs(JtJ(1,0))+abs(JtJ(1,1))+abs(JtJ(1,2));
         double valB = pow(abs(adjJ(1,0))+abs(adjJ(1,1))+abs(adjJ(1,2)),2)/abs(detJ);
         // double valC = abs(JtJ(2,0))+abs(JtJ(2,1))+abs(JtJ(2,2));
         double valC = pow(abs(adjJ(2,0))+abs(adjJ(2,1))+abs(adjJ(2,2)),2)/abs(detJ);
         A(ix,iy,iz) = valA;
         B(ix,iy,iz) = valB;
         C(ix,iy,iz) = valC;
      }
      AlterLS(A,*TransA1D[iel][0],*TransA1D[iel][1],*TransA1D[iel][2]);
      AlterLS(B,*TransB1D[iel][0],*TransB1D[iel][1],*TransB1D[iel][2]);
      AlterLS(C,*TransC1D[iel][0],*TransC1D[iel][1],*TransC1D[iel][2]);
      // for (int i = 0; i<dim; i++)
      // {
      //    cout << "TransA["<<i<<"] = "; TransA1D[iel][i]->Print(cout, TransA1D[iel][i]->Size()); 
      //    cout << "TransB["<<i<<"] = "; TransB1D[iel][i]->Print(cout, TransB1D[iel][i]->Size()); 
      //    cout << "TransC["<<i<<"] = "; TransC1D[iel][i]->Print(cout, TransC1D[iel][i]->Size()); 
      // }
      // cin.get();
   }
}

TPElementTransformation::TPElementTransformation(FiniteElementSpace &fes_) 
: fes(&fes_)
{
   dim = fes->GetMesh()->Dimension();
   if (dim == 2) 
   {
      Setup2D();
   }
   else
   {
      Setup3D();
   }
} 


void GetVertexToEdgeCount(const Mesh * mesh, DenseMatrix & edge_counts)
{
   // serial
   int dim = mesh->Dimension();
   int nv = mesh->GetNV();
   int ne = mesh->GetNEdges();
   int nel = mesh->GetNE();

   Array<int> ibeg(dim), iend(dim), inc(dim);
   // loop through axis
   for (int axis = 0; axis<dim; axis++)
   {
      switch (axis)
      {
         case 0:  ibeg[axis] = 0; inc[axis] = 2; iend[axis] = (dim == 2) ? 4 : 8  ; break; // "x" edges
         case 1:  ibeg[axis] = 1; inc[axis] = 2; iend[axis] = (dim == 2) ? 4 : 8  ; break; // "y" edges
         case 2:  ibeg[axis] = 8; inc[axis] = 1; iend[axis] = 12; break;                   // "z edges"
         default: MFEM_ABORT("This should be unreachable"); break; 
      }
   }

   Array<bool> edge_marker(ne); 
   edge_counts.SetSize(nv,dim);
   edge_counts = 0.0;
   Array<int> edge_owned;
   bool par = false;
#ifdef MFEM_USE_MPI
   const ParMesh * pmesh = dynamic_cast<const ParMesh *>(mesh);
   if (pmesh) par = true;
#endif   
   edge_owned.SetSize(ne); edge_owned = 0;
   ND_FECollection fec(1, dim);
   if (par) 
   {
      ParFiniteElementSpace aux_fes(const_cast<ParMesh *>(pmesh), &fec);
      int mytoffset = aux_fes.GetMyTDofOffset();
      int tsize = aux_fes.GetTrueVSize();
      Array<int> dofs;
      for (int i=0; i<ne; ++i)
      {
         aux_fes.GetEdgeDofs(i, dofs);
         const int ldof = (dofs[0] >= 0) ? dofs[0] : -1 - dofs[0];
         int sign = aux_fes.GetLocalTDofNumber(ldof);
         if (sign == -1) continue; // just a hack for now to work with AMR, need to rethink this.
         int gdof = aux_fes.GetGlobalTDofNumber(ldof);
         if (gdof >= mytoffset && gdof < mytoffset+tsize)
         {
            edge_owned[i] = 1;
         }
      }
   }
   else
   {
      edge_owned = 1;
      if (mesh->Nonconforming())
      {
         NCMesh * ncmesh = mesh->ncmesh;
         const mfem::NCMesh::NCList &nclist = ncmesh->GetEdgeList();
         Array<mfem::NCMesh::Slave> ncslaves = nclist.slaves;
         int ns = ncslaves.Size();
         for (int i=0; i<ns; ++i)
         {
             int j = ncslaves[i].index;
            edge_owned[j] = 0;
         }
      }
   }

   for (int d=0; d<dim; d++)
   {
      edge_marker = false;
      for (int iel = 0; iel<nel; iel++)
      {
         Array<int> elem_edges, cor;
         mesh->GetElementEdges(iel,elem_edges,cor);
         for (int i = ibeg[d]; i<iend[d]; i += inc[d])
         {
            int edge = elem_edges[i];
            if (edge_owned.Size())
            {
               // skip contributions from edge not owned by the proc
               if (!edge_owned[edge]) continue; 
            }
            if (edge_marker[edge]) continue;
            Array<int>edge_vert;
            mesh->GetEdgeVertices(edge,edge_vert);
            for (int j = 0; j < 2; j++)
            {
               int vertex = edge_vert[j];
               edge_counts(vertex,d) += 1.0;
            }
            edge_marker[edge] = true;
         }
      }
   }   
}

void GetDiffusionEdgeMatrix(int iedge, FiniteElementSpace * fes, 
                               Vector & Jac1D, Vector & Coeff1D,
                               const IntegrationRule *ir,  
                               DenseMatrix &elmat, int orient)
{
   const FiniteElement * el = fes->GetEdgeElement(iedge);
   int nd = el->GetDof();
   int dim = el->GetDim();

   DenseMatrix dshape(nd,dim);
   elmat.SetSize(nd);
   elmat = 0.0;
   int nint = ir->GetNPoints();
   
   for (int i = 0; i < nint; i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      double w = ip.weight;
      el->CalcDShape(ip, dshape);
      int j = orient == -1 ? nint-i-1 : i;
      double val = Coeff1D(j) * Jac1D(i);
      w *= val;
      AddMult_a_AAt(w, dshape, elmat);
   }
}

void GetMassEdgeMatrix(int iedge, FiniteElementSpace * fes, 
                       Vector & Jac1D, Vector & Coeff1D,
                       const IntegrationRule *ir,  
                       DenseMatrix &elmat, int orient)
{
   const FiniteElement * el = fes->GetEdgeElement(iedge);
   int nd = el->GetDof();
   Vector shape(nd);
   elmat.SetSize(nd);
   elmat = 0.0;
   int nint = ir->GetNPoints();
   for (int i = 0; i < nint; i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      double w = ip.weight;
      el->CalcShape(ip, shape);
      int j = orient == -1 ? nint-i-1 : i;
      double val = Coeff1D(j) * Jac1D(i);
      w *= val;
      AddMult_a_VVt(w, shape, elmat);
   }
}

void TensorProductEssentialDofsMaps(const Array<int> & ess_tdof_list, 
                                    const ParFiniteElementSpace * fes, 
                                    Array<Array<int> *> & tmap, // local edge map
                                    Array<Array<int>* > & non_ess_dofs) // element map
{

   MPI_Comm comm = fes->GetComm();
   int num_procs,myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // 1. Find the element local dofs that are essential and then identify 
   //    edge vertices (in local numbering) that are produced from
   //    Gather the vertex dofs to be eliminated for each edge in tmap.
   // 2. Reconstruct the essential dofs for each element (this list might be different 
   //    for each element e.g. l-shape/fichera mesh, where to keep the kronecker product 
   //    structure an essential dof is not eliminated, see vertex (v) below)
   //   .  .  .
   //   .  .  .
   //   .  .  v  .  .
   //   .  .  .  .  .
   //   .  .  .  .  .
   // ----------------------------------------------------------------------
   //
   ParMesh * pmesh = fes->GetParMesh();
   int dim = pmesh->Dimension();
   int nredges = pmesh->GetNEdges();
   int tsize = fes->GetTrueVSize();
   int vsize = fes->GetVSize();
   Vector tess_tdof_marker(tsize); tess_tdof_marker = 0.0;
   Vector ess_tdof_marker(vsize); 
   
   for (int i = 0; i<ess_tdof_list.Size(); i++)
   {
      int tdof = ess_tdof_list[i];

      tess_tdof_marker[tdof] = 1.0;
   }

   fes->GetProlongationMatrix()->Mult(tess_tdof_marker, ess_tdof_marker);

   tmap.SetSize(nredges);
   for (int i = 0; i<nredges; i++) { tmap[i] = new Array<int>(0); } 
   int nel = pmesh->GetNE();
   non_ess_dofs.SetSize(nel);
   for (int iel = 0; iel<nel; iel++)
   {
      non_ess_dofs[iel] = new Array<int>(0);
      Array<int> local_dofs; 
      Array<int> local_tdofs; 
      const FiniteElement &fe = *fes->GetFE(iel);
      // mfem to Tensor basis map
      const Array<int> &dmap =
      dynamic_cast<const TensorBasisElement&>(fe).GetDofMap();
      Array<int> dmapt(dmap.Size());
      for (int i = 0; i<dmapt.Size(); i++)
      {
         dmapt[dmap[i]] = i;
      }
      Array<int> elem_dofs;
      fes->GetElementDofs(iel,elem_dofs);
      // get local index of ess_dofs
      int n = elem_dofs.Size();
      // loop through the vertices
      for (int i = 0; i< elem_dofs.Size(); i++)
      {
         int ldof = elem_dofs[i];
         if (!ess_tdof_marker[ldof]) // if not essential dof
         {
            local_dofs.Append(dmapt[i]); // append in local element dofs
            local_tdofs.Append(ldof); // append in local element dofs
         }
      }
      
      // Find the possible vertex local dofs on the edges to be removed
      int n1D = (dim == 2) ? sqrt(n) : cbrt(n);
      Array<int> edges, cor;
      pmesh->GetElementEdges(iel,edges,cor);
      Array<int> eidx(dim);
      eidx[0] = 0;
      eidx[1] = 1;
      if (dim == 3) eidx[2] = 8;
      // mark edge local dofs
      Array<Array<int> *> markers(dim);
      for (int d = 0; d<dim; d++)
      {
         Array<int> marker(n1D); marker = 0;
         for (int i = 0; i<local_dofs.Size(); i++)
         {
            int j = local_dofs[i];
            int c = j/(n1D*n1D);
            int l = c*n1D*n1D;
            int k = (d == 0) ? (j-l)%n1D : d == 1 ? (j-l)/n1D : c; 
            marker[k] = 1;
         }
         markers[d] = new Array<int>(marker);
         // pick up the edge and orientation
         int edge = edges[eidx[d]];
         int orient = cor[eidx[d]];
         const FiniteElement &fe = *fes->GetEdgeElement(edge);
         // edge tensor product map
         const Array<int> &emap =
         dynamic_cast<const TensorBasisElement&>(fe).GetDofMap();
         Array<int> edge_ldofs;
         for (int i = 0; i<marker.Size(); i++)
         {
            if (!marker[i])
            {
               if (orient == 1)
               {
                  edge_ldofs.Append(emap[i]);
               }
               else
               {
                  int k = (emap[i] == 1) ? 0 : 1;
                  edge_ldofs.Append(k);
               }
            }
         }
         edge_ldofs.Sort(); edge_ldofs.Unique();   
         tmap[edge]->Append(edge_ldofs);
         tmap[edge]->Sort();
         tmap[edge]->Unique();
      }
      if (dim == 2)
      {
         for (int j = 0; j<n1D; j++)
         {
            if ((*markers[1])[j])
            {
               for (int i = 0; i<n1D; i++)
               {
                  if ((*markers[0])[i])
                  {
                     int ldof = n1D*j+i;
                     non_ess_dofs[iel]->Append(ldof);
                  }
               }
            }
         }
      }
      else
      {
         for (int k = 0; k<n1D; k++)
         {
            if ((*markers[2])[k])
            {
               for (int j = 0; j<n1D; j++)
               {
                  if ((*markers[1])[j])
                  {
                     for (int i = 0; i<n1D; i++)
                     {
                        if ((*markers[0])[i])
                        {
                           int ldof = n1D*n1D*k + n1D*j+i;
                           non_ess_dofs[iel]->Append(ldof);
                        }
                     }
                  }
               }
            }
         }
      }
      non_ess_dofs[iel]->Sort();
      non_ess_dofs[iel]->Unique();
      for (int i = 0; i<non_ess_dofs[iel]->Size(); i++)
      {
         int ldof = (*non_ess_dofs[iel])[i];
         int tdof = elem_dofs[dmap[ldof]];
         (*non_ess_dofs[iel])[i] = tdof;
      }
      for (int d=0; d<dim; d++)
      {
         delete markers[d];
      }
   }
};