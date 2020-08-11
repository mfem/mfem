#include "DofMapsDST.hpp"

double testcoeff(const Vector & x)
{
   // return x(0)*x(0)+x(1)*x(1);
   return sin(3*M_PI*(x(0)+x(1)));
}

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



   if (myid == 0)
   {
      for (int i = 0; i<nrsubdomains; i++)
      {
         Array<int> ijk;
         GetSubdomainijk(i,nxyz,ijk);
         cout << "ip = " << i 
              << ", id = " << GetSubdomainId(nxyz,ijk)
              << ", (i,j) = (" << ijk[0] << "," << ijk[1] << ") " << endl;
      }

   }







   // Setup the local FiniteElementSpaces
   const FiniteElementCollection * fec = pfes->FEColl();
   fes.SetSize(nrsubdomains);
   for (int i = 0; i<nrsubdomains; i++)
   {
      fes[i] = nullptr; // initialize with null on all procs
      if (myid == subdomain_rank[i])
      {
         fes[i] = new FiniteElementSpace(part->subdomain_mesh[i],fec);
      }
   }
   // cout << "Computing Overlap Tdofs" << endl;
   ComputeOvlpTdofs();
   // // PrintOvlpTdofs();


   // Vector X, Y;
   FunctionCoefficient c1(testcoeff);
   
   // Y = 0.0;
   // GridFunction gf(fes[myid]);
   // gf.ProjectCoefficient(c1);
   // // X.SetSize(fes[0]->GetTrueVSize());
   // // X = 1.0;
   // X.SetDataAndSize(gf.GetData(),gf.Size());
   // // X.Randomize();
   cout << "nrsubdomains = " << nrsubdomains << endl;
   int nrsub = nrsubdomains;
   Array<int> subdomain_ids(nrsub);
   Array<Vector*> x(nrsub);
   for (int i = 0; i<nrsub; i++)
   {
      subdomain_ids[i] = i;
      if (fes[i])
      {
         GridFunction gf(fes[i]);
         gf = 0.0;
         gf.ProjectCoefficient(c1);
         x[i] = new Vector(fes[i]->GetTrueVSize());
         *x[i] = gf;
      }
   }
   MPI_Barrier(comm); // leave this here for now

   TransferToNeighbors(subdomain_ids,x);
   //    cout << "Y = "; Y.Print();
   int nrneighbors = pow(3,dim);

   // if (myid == 0)
   // {
      string keys = "keys amrRljc\n";
      for (int i0 = 0 ; i0< nrsubdomains; i0++)
      {
         if (fes[i0])
         {
            GridFunction gf0(fes[i0]);
            for (int d = 0; d<nrneighbors; d++)
            {

               if(OvlpSol[i0][d])
               {
                  Array<int>dijk;
                  GetDirectionijk(d,dijk);
                  Array<int>ijk;
                  GetSubdomainijk(i0,nxyz,ijk);
                  ostringstream oss;
                  oss << "myid: " << myid 
                      << ", subdomain: (" << ijk[0] << "," << ijk[1] <<")"
                      << ", direction: (" << dijk[0] << "," << dijk[1] <<")";

                  gf0 = 0.0;
                  gf0.SetVector(*OvlpSol[i0][d],0);
                  char vishost[] = "localhost";
                  int  visport   = 19916;
                  socketstream sol_sock(vishost, visport);
                  sol_sock.precision(8);
                  sol_sock << "solution\n" << *(part->subdomain_mesh[i0]) << gf0 
                  << keys
                  << "window_title '" << oss.str() << "'" << flush;
               }
            }
         }
      // }

   //    if (myid == i0) 
   //    {
   //       gf0.SetVector(X,0);
   //    }
   //    else
   //    {
   //       gf0.SetVector(Y,0);
   //    }
      
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock << "solution\n" << *(part->subdomain_mesh[myid]) << gf0 
   //    << keys
   //    << "window_title '" << oss.str() << "'" << flush;
   }
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
   OvlpSol.resize(nrsubdomains);
   int nrneighbors = pow(3,dim); // including its self

   // loop through subdomains
   cout << "nrsubdomains = " << nrsubdomains << endl;
   for (int l = 0; l<nrsubdomains; l++)
   {
      if (myid == subdomain_rank[l])
      {
         int ntdofs = fes[l]->GetTrueVSize();
         Array<int> tdof_marker(ntdofs); 
         OvlpTDofs[l].resize(nrneighbors);
         OvlpSol[l].resize(nrneighbors);
         // loop through neighboring directions/neighbors
         for (int d=0; d<nrneighbors; d++)
         {
            OvlpSol[l][d] = nullptr;
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

void DofMaps::TransferToNeighbor(int i0, const Array<int> & direction0, 
                                 const Vector & x0, Vector & x1)
{
   // Find neighbor id;
   // MPI_Request request, request2;
   // MPI_Status status,status1;
   // Array<int> ijk0;
   // GetSubdomainijk(i0,nxyz,ijk0);
   // Array<int> ijk1(3); ijk1=0;
   // Array<int> direction1(3); direction1 = -1; // default for dim = 2, 3rd direction
   // for (int d=0;d<dim;d++) 
   // { 
   //    ijk1[d] = ijk0[d] + direction0[d]; 
   //    direction1[d] = -direction0[d];
   // }

   // int i1 = GetSubdomainId(nxyz,ijk1);
   // int rank1 = subdomain_rank[i1];
   // if (myid == subdomain_rank[i0])
   // {
   //    int d0 = GetDirectionId(direction0);
   //    Array<int> tdofs0 = OvlpTDofs[i0][d0];
   //    if (myid == rank1)
   //    {
   //       // The subdomain is on the same processor 
   //       int d1 = GetDirectionId(direction1);
   //       Array<int> tdofs1 = OvlpTDofs[i1][d1];
   //       x1.SetSize(fes[i1]->GetTrueVSize()); x1 = 0.0;
   //       for (int i = 0; i<tdofs0.Size(); i++)
   //       {
   //          // pick up input possition
   //          int j = tdofs0[i];
   //          // destination
   //          int k = tdofs1[i];
   //          x1[k] = x0[j];
   //       }
   //    }
   //    else
   //    {
   //       // The subdomain is not on the processor 
   //       Vector y0(tdofs0.Size());
   //       x0.GetSubVector(tdofs0, y0);
   //       // For now we use blocking send/receive
   //       int dest = subdomain_rank[i1];
   //       int tag = i0;
   //       int count = tdofs0.Size();

   //       // MPI_Send(y0.GetData(),count,MPI_DOUBLE,dest,tag,comm);
   //       MPI_Isend(y0.GetData(),count,MPI_DOUBLE,dest,tag,comm,&request);
   //       MPI_Wait(&request, &status);
   //    }
   // }
   // else if (myid == rank1)
   // {
   //    int d1 = GetDirectionId(direction1);
   //    Array<int> tdofs1 = OvlpTDofs[i1][d1];
   //    Vector y1(tdofs1.Size());
   //    int count = tdofs1.Size();
   //    int src = subdomain_rank[i0];
   //    int tag = i0; 
   //    // receive data
   //    // MPI_Recv(y1.GetData(),count,MPI_DOUBLE,src,tag,comm,&status);
   //    MPI_Irecv(y1.GetData(),count,MPI_DOUBLE,src,tag,comm,&request);
   //    MPI_Wait(&request,&status);
   //    x1.SetSize(fes[i1]->GetTrueVSize()); x1 = 0.0;
   //    x1.SetSubVector(tdofs1,y1);
   // }
}


void DofMaps::TransferToNeighbors(const Array<int> & SubdomainIds, const Array<Vector *> & x)
{
   // 2D for now....
   MFEM_VERIFY(SubdomainIds.Size() == x.Size(), "TransferToNeighbors: Size inconsistency");
   int nrsendIds = SubdomainIds.Size();
   int nrneighbors = pow(3,dim);
   MPI_Request *recv_requests = new MPI_Request[nrsendIds*nrneighbors];
   MPI_Request *send_requests = new MPI_Request[nrsendIds*nrneighbors];
   MPI_Status  *recv_statuses = new MPI_Status[nrsendIds*nrneighbors];
   MPI_Status  *send_statuses = new MPI_Status[nrsendIds*nrneighbors];
   // Initialize
   for (int i = 0; i < nrsendIds*nrneighbors; i++)
   {
      recv_requests[i] = MPI_REQUEST_NULL;
      send_requests[i] = MPI_REQUEST_NULL;
   }
   Array<Vector * > send_buffer(nrsendIds*nrneighbors);
   Array<Vector * > recv_buffer(nrsendIds*nrneighbors);
   int send_counter = 0;
   int recv_counter = 0;
   for (int is = 0; is<nrsendIds; is++)
   {
      int i0 = SubdomainIds[is];
      Array<int> ijk;
      GetSubdomainijk(i0,nxyz,ijk);
      for (int d=0;d<nrneighbors; d++)
      {
         Array<int>directions;
         GetDirectionijk(d,directions);
         if (directions[0] == 0 && directions[1] == 0) continue;
         int i = ijk[0] + directions[0];
         if (i<0 || i>=nxyz[0]) continue;
         int j = ijk[1] + directions[1];
         if (j<0 || j>=nxyz[1]) continue;
         Array<int>ijk1(3);
         ijk1[0] = i;
         ijk1[1] = j;
         ijk1[2] = 0;
         int i1 = GetSubdomainId(nxyz,ijk1);
         if (myid == subdomain_rank[i0])
         {
            Array<int> tdofs0 = OvlpTDofs[i0][d]; // map of dofs in the overlap
            send_buffer[send_counter] = new Vector(tdofs0.Size());
            x[i0]->GetSubVector(tdofs0,*send_buffer[send_counter]);
            // Destination rank
            int dest = subdomain_rank[i1];
            cout << "sending from (id,sub) = (" << myid << "," << i0 << ") to: " 
                 << "(" << dest << ","<< i1 << ")" << endl;
            int tag = i0 * nrneighbors + d;

            int count = tdofs0.Size();
            MPI_Isend(send_buffer[send_counter]->GetData(),count,MPI_DOUBLE,dest,
                      tag,comm,&send_requests[send_counter]);
            send_counter++;
         }
         if (myid == subdomain_rank[i1])
         {
            Array<int> direction1(3); direction1 = -1;
            for (int d=0;d<dim;d++) 
            { 
               direction1[d] = -directions[d];
            }
            int d1 = GetDirectionId(direction1);

            int count = OvlpTDofs[i1][d1].Size();
            recv_buffer[recv_counter] = new Vector(count); 
            int src = subdomain_rank[i0];
            int tag = i0 * nrneighbors + d;
            MPI_Irecv(recv_buffer[recv_counter]->GetData(), count,MPI_DOUBLE,src,
                      tag,comm, &recv_requests[recv_counter]);
            recv_counter++;
         }
      }   
   }
   MPI_Waitall(send_counter, send_requests, send_statuses);
   MPI_Waitall(recv_counter, recv_requests, recv_statuses);

   delete [] send_statuses;
   delete [] send_requests;
   delete [] recv_statuses;
   delete [] recv_requests;

   // Extruct the transfered solutions
   recv_counter = 0;
   for (int is = 0; is<nrsendIds; is++)
   {
      int i0 = SubdomainIds[is];
      Array<int> ijk;
      GetSubdomainijk(i0,nxyz,ijk);
      for (int d=0;d<nrneighbors; d++)
      {
         Array<int>directions;
         GetDirectionijk(d,directions);
         if (directions[0] == 0 && directions[1] == 0) continue;
         int i = ijk[0] + directions[0];
         if (i<0 || i>=nxyz[0]) continue;
         int j = ijk[1] + directions[1];
         if (j<0 || j>=nxyz[1]) continue;
         Array<int>ijk1(3);
         ijk1[0] = i;
         ijk1[1] = j;
         ijk1[2] = 0;
         int i1 = GetSubdomainId(nxyz,ijk1);
         if (myid == subdomain_rank[i1])
         {
            Array<int> direction1(3); direction1 = -1;
            for (int d=0;d<dim;d++) 
            { 
               direction1[d] = -directions[d];
            }
            int d1 = GetDirectionId(direction1);
            Array<int> tdofs1 = OvlpTDofs[i1][d1];
            OvlpSol[i1][d1] = new Vector(fes[i1]->GetTrueVSize());
            *OvlpSol[i1][d1] = 0.0;
            OvlpSol[i1][d1]->SetSubVector(tdofs1,*recv_buffer[recv_counter]);
            recv_counter++;
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


















