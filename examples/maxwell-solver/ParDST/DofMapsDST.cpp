#include "DofMapsDST.hpp"


void ComputeTdofOffsets(const MPI_Comm & comm, const ParFiniteElementSpace * pfes, 
                        std::vector<int> & tdof_offsets)
{
   int num_procs;
   MPI_Comm_size(comm, &num_procs);
   tdof_offsets.resize(num_procs);
   int mytoffset = pfes->GetMyTDofOffset();
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

void GetSubdomainijk(int ip, const Array<int> nxyz, Array<int> & ijk)
{
   ijk.SetSize(3);
   ijk[2] = ip/(nxyz[0]*nxyz[1]);
   ijk[1] = (ip-ijk[2]*nxyz[0]*nxyz[1])/nxyz[0];
   ijk[0] = (ip-ijk[2]*nxyz[0]*nxyz[1])%nxyz[0];
}
void GetDirectionijk(int id, Array<int> & ijk)
{
   ijk.SetSize(3);
   int n = 3;
   ijk[2] = id/(n*n) - 1;
   ijk[1] = (id-(ijk[2]+1)*n*n)/n - 1;
   ijk[0] = (id-(ijk[2]+1)*n*n)%n - 1;
}

int GetSubdomainId(const Array<int> nxyz, Array<int> & ijk)
{
   int dim=ijk.Size();
   int k = (dim==2)? 0 : ijk[2];
   return k*nxyz[1]*nxyz[0] + ijk[1]*nxyz[0] + ijk[0];
}

int GetDirectionId(const Array<int> & ijk)
{
   int n = 3;
   int dim = ijk.Size();
   int k = (dim == 2) ? -1 : ijk[2];
   return (k+1)*n*n + (ijk[1]+1)*n + ijk[0]+1; 
}

void DofMaps::Init()
{
   comm = pfes->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   dim = pfes->GetParMesh()->Dimension();
   ComputeTdofOffsets(comm, pfes, tdof_offsets);
   myelemoffset = part->myelem_offset;
   mytoffset = pfes->GetMyTDofOffset();
   subdomain_rank = part->subdomain_rank;
   nrsubdomains = part->nrsubdomains;
   nxyz.SetSize(3);
   for (int i = 0; i<3; i++) { nxyz[i] = part->nxyz[i]; }
}

DofMaps::DofMaps(ParFiniteElementSpace *pfes_, ParMeshPartition * part_)
: pfes(pfes_), part(part_)
{
   Init();
   // cout << "myid = " << myid << ", nrsubdomains = " << nrsubdomains << endl;
   // cout << "subdomain_rank = " ; subdomain_rank.Print(cout, nrsubdomains);
   Setup();
}

void DofMaps::Setup()
{
   // Setup the local FiniteElementSpaces
   const FiniteElementCollection * fec = pfes->FEColl();
   fes.SetSize(nrsubdomains);
   for (int i = 0; i<nrsubdomains; i++)
   {
      fes[i] = nullptr; // initialize with null on all procs
      if (myid == subdomain_rank[i])
      {
         Mesh * mesh = part->subdomain_mesh[i];
         fes[i] = new FiniteElementSpace(mesh,fec);
      }
   }
   // cout << "Computing Overlap Tdofs" << endl;
   ComputeOvlpTdofs();
   // PrintOvlpTdofs();
}


void DofMaps::AddElementToOvlpLists(int l, int iel, 
                            const Array<bool> & neg, const Array<bool> & pos)
{
   int kbeg = (dim == 2) ? 0 : -1;
   int kend = (dim == 2) ? 0 :  1;
   Array<int> dijk(3);
   for (int k = kbeg; k<=kend; k++)
   {
      if (dim == 3)
      {
         if (k == -1 && !neg[2]) continue;   
         if (k ==  1 && !pos[2]) continue;   
      }   

      for (int j = -1; j<=1; j++)
      {
         if (j== -1 && !neg[1]) continue;   
         if (j==  1 && !pos[1]) continue;   
         for (int i = -1; i<=1; i++)
         {
            // cases to skip 
            if (i==-1 && !neg[0]) continue;   
            if (i== 1 && !pos[0]) continue;   

            if (i==0 && j==0 && k == 0) continue;
            dijk[0] = i; dijk[1] = j; dijk[2] = (dim==2)?-1 : k;  
            int DirId = GetDirectionId(dijk);
            OvlpElems[l][DirId].Append(iel);
         }
      }
   }
}

void DofMaps::ComputeOvlpElems()
{
   // first compute the element in the overlaps
   OvlpElems.resize(nrsubdomains);
   int nlayers = 2*part->OvlpNlayers; 
   // loop through subdomains
   for (int l = 0; l<nrsubdomains; l++)
   {
      if (myid == subdomain_rank[l])
      {
         Array<int> ijk;
         GetSubdomainijk(l,nxyz,ijk);
         Mesh * mesh = part->subdomain_mesh[l];
         OvlpElems[l].resize(pow(3,dim)); 
         Vector pmin, pmax;
         mesh->GetBoundingBox(pmin,pmax);
         double h = part->MeshSize;
         // loop through the elements in the mesh and assign them to the 
         // appropriate lists of overlaps
         for (int iel=0; iel< mesh->GetNE(); iel++)
         {
            // Get element center
            Vector center(dim);
            int geom = mesh->GetElementBaseGeometry(iel);
            ElementTransformation * tr = mesh->GetElementTransformation(iel);
            tr->Transform(Geometries.GetCenter(geom),center);

            Array<bool> pos(dim); pos = false;
            Array<bool> neg(dim); neg = false;
            // loop through dimensions   
            for (int d=0;d<dim; d++)
            {
               if (ijk[d]>0 && center[d] < pmin[d]+h*nlayers)
               {
                  neg[d] = true;
               }

               if (ijk[d]<nxyz[d]-1 && center[d] > pmax[d]-h*nlayers) 
               {
                  pos[d] = true;
               }
            }
            // Add the element to the appropriate lists
            AddElementToOvlpLists(l,iel,neg,pos);        
         }   
      }
   }
}

void DofMaps::ComputeOvlpTdofs()
{
   // first compute the element in the overlaps
   ComputeOvlpElems();

   OvlpTDofs.resize(nrsubdomains);
   int nrneighbors = pow(3,dim); // including its self

   // loop through subdomains
   for (int l = 0; l<nrsubdomains; l++)
   {
      if (myid == subdomain_rank[l])
      {
         int ntdofs = fes[l]->GetTrueVSize();
         Array<int> tdof_marker(ntdofs); 
         OvlpTDofs[l].resize(nrneighbors);
         // loop through neighboring directions/neighbors
         for (int d=0; d<nrneighbors; d++)
         {
            OvlpTDofs[l].resize(nrneighbors);
            tdof_marker = 0;
            Array<int> tdoflist;
            // Get the direction
            Array<int> dijk;
            GetDirectionijk(l,dijk);
            int nel = OvlpElems[l][d].Size();
            Array<int>Elems = OvlpElems[l][d];
            for (int iel = 0; iel<nel; ++iel)
            {
               int jel = Elems[iel];   
               Array<int> ElemDofs;

               fes[l]->GetElementDofs(jel,ElemDofs);
               int ndof = ElemDofs.Size();
               for (int i = 0; i<ndof; ++i)
               {
                  int dof_ = ElemDofs[i];
                  int dof = (dof_ >= 0) ? dof_ : abs(dof_) - 1;
                  if (!tdof_marker[dof])
                  {
                     tdoflist.Append(dof); // dofs of ip0 in ovlp
                     tdof_marker[dof] = 1;
                  }
               }
            }
            OvlpTDofs[l][d] = tdoflist; 
         }
      }
   }
}


void DofMaps::PrintOvlpTdofs()
{
   int nrneighbors = pow(3,dim); // including its self
   if (myid == 0)
   {
      for (int i = 0; i<nrsubdomains; i++)
      {
         if (myid == subdomain_rank[i])
         {
            Array<int> ijk;
            GetSubdomainijk(i,nxyz,ijk);
            cout << "subdomain = " ; ijk.Print();
            cout << "myid = " << myid << endl;
            cout << "ip   = " << i << endl;
            for (int d = 0; d<nrneighbors; d++)
            {
               Array<int> dijk;
               GetDirectionijk(d,dijk);
               cout << "direction = " ; dijk.Print();

               if (OvlpTDofs[i][d].Size())
               {
                  cout << "OvlpTdofs = " ; OvlpTDofs[i][d].Print(cout,OvlpTDofs[i][d].Size() );
               }
            }
         }
      }
   }

}


















