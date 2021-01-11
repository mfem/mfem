
#include "DofMaps.hpp"
#include "MeshPart.hpp"


void E_exact(const Vector &x, Vector &E)
{
   double kappa = 1.0;
   int dim = x.Size();
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}


void FindPtsGetCommonElements(Mesh & mesh0, Mesh & mesh1, 
                              Array<int> & elems0, Array<int> & elems1)
{
   int dim = mesh0.Dimension();
   const int ne0 = mesh0.GetNE();
   Vector centers(ne0*dim);
   elems0.SetSize(0);
   elems1.SetSize(0);
   for (int i = 0; i < ne0; i++)
   {
      Vector center(dim);
      mesh0.GetElementCenter(i,center);
      for (int d=0; d<dim; d++)
      {
         centers[ne0*d + i] = center[d];
      }
   }
   // Evaluate mesh 1 grid function.
   FindPointsGSLIB finder;
   finder.Setup(mesh1);
   finder.FindPoints(centers);
   Array<int> elem_map = finder.GetElem();
   Array<int> code = finder.GetCode();
   finder.FreeData();

   for (int i = 0; i<code.Size(); i++)
   {
      if (!code[i]) 
      {  // element is found
         elems0.Append(i);
         elems1.Append(elem_map[i]);
      }
   }
}                              

// Assuming there are no dublicated indices in the lists
void GetCommonIndices(const Array<int> & list0, const Array<int> & list1, Array<int> & idx0, Array<int> & idx1)
{
   unordered_map<int, int> map0, map1;
   int i = 0, j = 0;
   for (auto k : list0) map0[k] = i++;
   for (auto k : list1) map1[k] = j++;
   
   for (auto k : map0)
   {
      if (map1.find(k.first) != map1.end())
      {
         idx0.Append(k.second);
         idx1.Append(map1[k.first]);
      }
   }
}

// dofs0 the fes0 indices (Domain)
// dofs1 the fes1 indices (Range)
void GetDofMaps(const FiniteElementSpace &fes0, const FiniteElementSpace &fes1, 
               Array<int> & dofs0, Array<int> & dofs1,
               const Array<int> * elems0_, const Array<int> * elems1_)
{
   Array<int> elems0, elems1;
   if (!elems0_ || !elems1_)
   {  // construct the element lists using gslib
      FindPtsGetCommonElements(*fes0.GetMesh(), *fes1.GetMesh(), elems0, elems1);
   }
   else
   {
      GetCommonIndices(*elems0_, *elems1_, elems0, elems1);   
   }

   // construct dof maps fes0->fes1 (possibly not a subspace)
   int nel = elems0.Size();
   MFEM_VERIFY(elems1.Size() == nel, "Inconsistent number of elements");
   Array<int> dof_marker(fes0.GetTrueVSize()); dof_marker = 0;
   for (int ie = 0; ie<nel; ie++)
   {
      int iel0 = elems0[ie];
      int iel1 = elems1[ie];
      Array<int> ElemDofs0;
      Array<int> ElemDofs1;
      fes0.GetElementDofs(iel0,ElemDofs0);
      fes1.GetElementDofs(iel1,ElemDofs1);
      int ndof = ElemDofs0.Size();
      for (int i = 0; i<ndof; i++)
      {
         int dof0_ = ElemDofs0[i];
         int dof1_ = ElemDofs1[i];
         int dof0 = (dof0_ >= 0) ? dof0_ : - dof0_ - 1;
         int dof1 = (dof1_ >= 0) ? dof1_ : - dof1_ - 1;
         if (dof_marker[dof0]) continue;
         dofs0.Append(dof0);
         dofs1.Append(dof1);
         dof_marker[dof0] = 1;
      }
   }
}


void PartitionFE(const FiniteElementSpace * fes, int nrsubmeshes, double ovlp, 
                 Array<FiniteElementSpace*> & fespaces, 
                 Array<Array<int> * > & ElemMaps,
                 Array<Array<int> * > & DofMaps0, Array<Array<int> * > & DofMaps1,
                 Array<Array<int> * > & OvlpMaps0, Array<Array<int> * > & OvlpMaps1)
{
   Mesh * mesh = fes->GetMesh();
   Array<Mesh *> meshes;

   PartitionMesh(mesh,nrsubmeshes,ovlp,meshes,ElemMaps);

   // DofMaps from subdomains to global mesh

   const FiniteElementCollection * fec = fes->FEColl();
   Array<int> GlobalElems(mesh->GetNE());
   for (int i = 0; i<GlobalElems.Size(); i++) GlobalElems[i] = i;

   fespaces.SetSize(nrsubmeshes);
   DofMaps0.SetSize(nrsubmeshes);
   DofMaps1.SetSize(nrsubmeshes);
   for (int i = 0; i<nrsubmeshes; i++)
   {
      fespaces[i] = new FiniteElementSpace(meshes[i],fec);
      cout << " fespace size " << fespaces[i]->GetTrueVSize() << endl;
      DofMaps0[i] = new Array<int>();
      DofMaps1[i] = new Array<int>();
      GetDofMaps(*fespaces[i],*fes,*DofMaps0[i], *DofMaps1[i], ElemMaps[i], &GlobalElems);
   }
   int nroverlaps = nrsubmeshes-1;
   OvlpMaps0.SetSize(nroverlaps);
   OvlpMaps1.SetSize(nroverlaps);
   for (int i = 0; i<nroverlaps; i++)
   {
      OvlpMaps0[i] = new Array<int>();
      OvlpMaps1[i] = new Array<int>();
      GetDofMaps(*fespaces[i],*fespaces[i+1],*OvlpMaps0[i], *OvlpMaps1[i], 
                  ElemMaps[i], ElemMaps[i+1]);
   }
}

void GetElements(Mesh &mesh, double ovlp, int direction, Array<int> & elems)
{
   double amin, amax;
   GetMeshAngleRange(&mesh, amin, amax);
   int dim = mesh.Dimension();
   // loop through elements
   int ne = mesh.GetNE();
   for (int i=0; i<ne; i++)
   {
      Vector center(dim);
      mesh.GetElementCenter(i,center);
      double thetad = GetPointAngle(center);

      switch (direction)
      {
      case -1:
         if (thetad >= amin + ovlp)  
         {
            elems.Append(i); 
         }
         break;
      case  1: 
         if (thetad <= amax - ovlp)
         {

            elems.Append(i);
         }   
         break;
      default:
         if (thetad >= amin + ovlp && thetad <= amax - ovlp) 
         {
            elems.Append(i); 
         }
         break;
      }
   }
}

void GetRestrictionDofs(FiniteElementSpace &fes, int direction, double ovlp, Array<int> & rdofs)
{
   Array<int> elems;
   GetElements(*fes.GetMesh(),ovlp,direction,elems);
   int ne = elems.Size();
   int tsize = fes.GetTrueVSize();
   Array<int> tdof_marker(tsize); tdof_marker = 0;
   for (int i=0; i<ne; i++)
   {
      int ie = elems[i];
      Array<int> elem_dofs;
      fes.GetElementDofs(ie,elem_dofs);
      for (auto x : elem_dofs)
      {
         int tdof = (x>=0) ? x : -1 - x;
         tdof_marker[tdof] = 1;
      }
   }
   int n = tdof_marker.Sum();
   rdofs.SetSize(n);
   int k=0;
   for (int i=0; i<tsize; i++)
   {
      if (tdof_marker[i])
      {
         rdofs[k++] = i;
      }
   }
}

void RestrictDofs(const Array<int> & rdofs, int tsize, Vector & x)
{
   int n = rdofs.Size();
   Array<int> dofs(2*n);
   for (int i =0; i<n; i++)
   {
      dofs[i] = rdofs[i];
      dofs[n+i] = rdofs[i]+tsize;
   }
   x.SetSubVectorComplement(dofs,0.0);
}


void MapDofs(const Array<int> & dmap0, const Array<int> & dmap1,
             const Vector &gf0, Vector &gf1)
{
   int tsize0 = gf0.Size()/2;
   int tsize1 = gf1.Size()/2;
   for (int i = 0; i< dmap0.Size(); i++)
   {
      int j = dmap0[i];   
      int k = dmap1[i];   
      gf1[k] = gf0[j];
      gf1[k+tsize1] = gf0[j+tsize0];
   }
}

void AddMapDofs(const Array<int> & dmap0, const Array<int> & dmap1,
             const Vector &gf0, Vector &gf1)
{
   int tsize0 = gf0.Size()/2;
   int tsize1 = gf1.Size()/2;
   for (int i = 0; i< dmap0.Size(); i++)
   {
      int j = dmap0[i];   
      int k = dmap1[i];   
      gf1[k] += gf0[j];
      gf1[k+tsize1] += gf0[j+tsize0];
   }
}

void DofMapTests(FiniteElementSpace &fes0, FiniteElementSpace &fes1,
                 const Array<int> & dmap0, const Array<int> & dmap1)
{

   Mesh * mesh0=fes0.GetMesh();
   Mesh * mesh1=fes1.GetMesh();
   ComplexGridFunction gf0(&fes0);
   ComplexGridFunction gf1(&fes1); gf1 = 0.0;
   int dim = mesh0->Dimension();
   // Vector vone(dim); vone = 1.0;
   // VectorConstantCoefficient one(vone);
   VectorFunctionCoefficient cf(dim,E_exact);
   gf0.ProjectCoefficient(cf,cf);

   MapDofs(dmap0,dmap1,gf0,gf1);

   char vishost[] = "localhost";
   int  visport   = 19916;

   // socketstream sol_sock0(vishost, visport);
   // sol_sock0.precision(8);
   // sol_sock0 << "solution\n" << *mesh0 << gf0.real() 
   //          //  << "valuerange -1 1 \n"
   //           << "window_title ' gf_0 ' " << flush;                     

   // socketstream sol_sock1(vishost, visport);
   // sol_sock1.precision(8);
   // sol_sock1 << "solution\n" << *mesh1 << gf1.real() 
   //          //  << "valuerange -1 1 \n"
   //           << "window_title ' gf_1 ' " << flush;  
   int n = 2;
   {
      socketstream solsock(vishost, visport);
      solsock.precision(8);
      solsock << "parallel " << n << " " << 0 << "\n";
      solsock << "solution\n" << *mesh0 << gf0.real() << flush;     
   }
   {
      socketstream solsock(vishost, visport);
      solsock.precision(8);
      solsock << "parallel " << n << " " << 1 << "\n";
      solsock << "solution\n" << *mesh1 << gf1.real()  << flush;     
   }
}


void DofMapOvlpTest(FiniteElementSpace &fes, const Array<int> & dmap)
{
   Mesh * mesh = fes.GetMesh();
   int dim = mesh->Dimension();
   int tsize = fes.GetTrueVSize();
   ComplexGridFunction gf(&fes);
   VectorFunctionCoefficient cf(dim,E_exact);
   gf.ProjectCoefficient(cf,cf);
   
   RestrictDofs(dmap,tsize,gf);

   string keys = "keys mac\n" ;

   char vishost[] = "localhost";
   int  visport   = 19916;

   {
      socketstream solsock_re(vishost, visport);
      solsock_re.precision(8);
      solsock_re << "solution\n" << *mesh << gf.real() << keys << flush;     
      socketstream solsock_im(vishost, visport);
      solsock_im.precision(8);
      solsock_im << "solution\n" << *mesh << gf.imag() << keys << flush; 
   }
}           