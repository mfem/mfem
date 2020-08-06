//Diagonal Source Transfer Preconditioner

#include "ParDST.hpp"

//1. Computing the list of global tdofs for each subdomain 
//   on the host_rank
ParSubdomainDofInfo::ParSubdomainDofInfo(ParFiniteElementSpace *fespace_, 
                                         ParMeshPartition * part_)
: fespace(fespace_), part(part_)                                         
{
   // MPI workspace
   MPI_Comm comm = fespace->GetComm();
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   ComputeTdofOffsets();
   ParMesh * pmesh = fespace->GetParMesh();
   int myelemoffset = part->myelem_offset;
   subdomain_rank = part->subdomain_rank;
   nrsubdomains = part->nrsubdomains;
   // workspace for MPI_AlltoAll
   Array<int> send_count(num_procs);  send_count = 0;
   Array<int> send_displ(num_procs);  send_displ = 0; 
   Array<int> recv_count(num_procs);  recv_count = 0;
   Array<int> recv_displ(num_procs);  recv_displ = 0;
  

   //----------------------------------------------------------
   // Each rank constructs the list of its own tdofs in a subdomain 
   // After that they communicate it to the host rank to 
   // create the complete list
   SubdomainLocalTrueDofs.resize(nrsubdomains);
   Array<int> dof_marker(fespace->GetTrueVSize());
   for (int ip = 0; ip<nrsubdomains; ++ip)
   {
      dof_marker = 0;
      int nel = part->local_element_map[ip].Size();
      if (nel)
      {
         for (int iel = 0; iel<nel; iel++)
         {
            int elem_idx = part->local_element_map[ip][iel] - myelemoffset;
            Array<int> element_dofs;
            fespace->GetElementDofs(elem_idx, element_dofs);
            int ndofs = element_dofs.Size();
            for (int i=0; i<ndofs; i++)
            {
               int pdof_ = element_dofs[i];
               int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
               int tdof = fespace->GetGlobalTDofNumber(pdof);
               if (myid == get_rank(tdof))
               {
                  if (!dof_marker[tdof-mytoffset])
                  {
                     SubdomainLocalTrueDofs[ip].Append(tdof);
                     dof_marker[tdof-mytoffset] = 1;
                  }
               }
            }
         }
      }
   }
  
   // Communicate to the host rank
   // 1. Construct send count
   for (int ip = 0; ip<nrsubdomains; ++ip)
   {
      int ndofs = SubdomainLocalTrueDofs[ip].Size();
      if (ndofs)
      {
         // Data to send to the subdomain rank
         // 1. Subdomain no 
         // 2. Number of dofs
         // 3. The list of dofs
         send_count[subdomain_rank[ip]] += 1 + 1 + ndofs; 
      }
   }
   // 2. Construct receive count
   MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();
   // now allocate space for the send buffer
   Array<int> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> soffs(num_procs); soffs = 0;
   // 3. Fill up the send buffer
   for (int ip = 0; ip<nrsubdomains; ++ip)
   {
      int ndofs = SubdomainLocalTrueDofs[ip].Size();
      if (ndofs)
      {
         // Data to send to the subdomain rank
         // 1. Subdomain no 
         // 2. Number of dofs
         // 3. The list of dofs
         int j = send_displ[subdomain_rank[ip]] + soffs[subdomain_rank[ip]];
         sendbuf[j] = ip;
         sendbuf[j+1] = ndofs;
         for (int k = 0; k < ndofs ; ++k)
         {
            sendbuf[j+2+k] = SubdomainLocalTrueDofs[ip][k];
         }
         soffs[subdomain_rank[ip]] +=  2 + ndofs;
      }
   }

   // Communication
   Array<int> recvbuf(rbuff_size);
   MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_INT, recvbuf,
                 recv_count, recv_displ, MPI_INT, comm);

   SubdomainGlobalTrueDofs.resize(nrsubdomains);
   int k=0;
   while (k<rbuff_size)
   {
      int ip = recvbuf[k];
      k++;
      int ndofs = recvbuf[k];
      k++;
      for (int i = 0; i < ndofs; ++i)
      {
         SubdomainGlobalTrueDofs[ip].Append(recvbuf[k+i]);
      }
      k += ndofs;
   }

   // for (int ip = 0; ip<nrsubdomains; ++ip)
   // {
   //    if (SubdomainLocalTrueDofs[ip].Size())
   //    {
   //       cout << "myid, ip = " << myid << ", " << ip << ", SubdomainLocalTrueDofs: " ;
   //       SubdomainLocalTrueDofs[ip].Print(cout, SubdomainLocalTrueDofs[ip].Size());
   //    }
   //    if (SubdomainGlobalTrueDofs[ip].Size())
   //    {
   //       cout << "myid, ip = " << myid << ", " << ip << ", SubdomainGlobalTrueDofs: ";
   //       SubdomainGlobalTrueDofs[ip].Print(cout, SubdomainGlobalTrueDofs[ip].Size());
   //    }
   //    MPI_Barrier(comm);
   // }
   // MPI_Barrier(MPI_COMM_WORLD);

}








ParDST::ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, Coefficient * ws_,  int nrlayers_ , int nx_, int ny_, int nz_)
   : Solver(2*bf_->ParFESpace()->GetTrueVSize(), 2*bf_->ParFESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{
   pfes = bf->ParFESpace();
   fec = pfes->FEColl();
 
   //1. Indentify problem ... Helmholtz or Maxwell
   cout << " 1. Indentify problem to be solved ... " << endl;
   int prob_kind = fec->GetContType();
   if (prob_kind == 0) cout << "    Helmholtz" << endl;
   if (prob_kind == 1) cout << "    Maxwell" << endl; 


   //2. Create the parallel mesh partition
   pmesh = pfes->GetParMesh();
   dim = pmesh->Dimension();
   cout << "\n 2. Generating ParMesh partitioning ... " << endl;
   ParMeshPartition part(pmesh,nx_,ny_,nz_,nrlayers);
   nx = part.nxyz[0]; ny = part.nxyz[1];  nz = part.nxyz[2];
   nrsubdomains = part.nrsubdomains;
   cout << "    Done ! " << endl;


   //3. Setup info for sweeps
   cout << "\n 3. Computing sweeps info ..." << endl; 
   sweeps = new Sweep(dim);
   cout << "    Done ! " << endl;

   //4. Create LocalToGlobal maps 
   //   (local GridFunctions/Vector to Global ParGridFunction/Vector) 
   cout << "\n 4. Computing subdomain to global maps ..." << endl; 
   ParSubdomainDofInfo * test = new ParSubdomainDofInfo(pfes,&part);
   cout << "    Done ! " << endl;

}

void ParDST::Mult(const Vector &r, Vector &z) const
{
}


// void ParDST::Getijk(int ip, int & i, int & j, int & k) const
// {
//    k = ip/(nx*ny);
//    j = (ip-k*nx*ny)/nx;
//    i = (ip-k*nx*ny)%nx;
// }

// int ParDST::GetPatchId(const Array<int> & ijk) const
// {
//    int d=ijk.Size();
//    int z = (d==2)? 0 : ijk[2];
//    return part->subdomains(ijk[0],ijk[1],z);
// }

ParDST::~ParDST() {}
