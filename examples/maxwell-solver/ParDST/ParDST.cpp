//Diagonal Source Transfer Preconditioner

#include "ParDST.hpp"


ParSubdomainDofInfo::ParSubdomainDofInfo(ParFiniteElementSpace *fespace_, 
                                         ParMeshPartition * part_)
: fespace(fespace_), part(part_)                                         
{
   // MPI workspace
   MPI_Comm comm = fespace->GetComm();
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(comm, &myid);
   ParMesh * pmesh = fespace->GetParMesh();
   int myelemoffset = part->myelem_offset;
   subdomain_rank = part->subdomain_rank;
   nrsubdomains = part->nrsubdomains;
   // workspace for MPI_AlltoAll
   Array<int> send_count(num_procs);  send_count = 0;
   Array<int> send_displ(num_procs);  send_displ = 0; 
   Array<int> recv_count(num_procs);  recv_count = 0;
   Array<int> recv_displ(num_procs);  recv_displ = 0;
  
   // each element contributing to a subdomain has to communicate 
   // to the subdomain host rank the list
   // of its tdof numbers whether it owns them or not
   // After these lists are constructed to the host rank then they will be brodcasted to
   // the participating ranks
   
   for (int ip = 0; ip<nrsubdomains; ++ip)
   {
      int nrelems = part->local_element_map[ip].Size();
      if (nrelems >0 )
      {
         for (int iel=0; iel<nrelems; ++iel)
         {
            Array<int>element_dofs;
            int elem_idx = part->local_element_map[ip][iel] - myelemoffset;
            int nrdofs = fespace->GetFE(elem_idx)->GetDof();
            // send the number of dofs for each element and the tdof numbers
            // subdomain no, nrdofs the list of tdofs
           send_count[subdomain_rank[ip]] += 1 + 1 + nrdofs; 
         }
      }
   }

   // comunicate so that recv_count is constructed
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

   // fill up the send_buffer
   for (int ip = 0; ip<nrsubdomains; ++ip)
   {
      int nrelems = part->local_element_map[ip].Size();

      if (nrelems > 0)
      {
         for (int iel=0; iel<nrelems; ++iel)
         {
            Array<int>element_dofs;
            int elem_idx = part->local_element_map[ip][iel] - myelemoffset;
            fespace->GetElementDofs(elem_idx, element_dofs);
            int nrdofs = element_dofs.Size();
            int j = send_displ[subdomain_rank[ip]] + soffs[subdomain_rank[ip]];
            sendbuf[j] = ip;
            sendbuf[j+1] = nrdofs;
            for (int idof = 0; idof < nrdofs ; ++idof)
            {
               int pdof_ = element_dofs[idof];
               int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
               sendbuf[j+2+idof] = fespace->GetGlobalTDofNumber(pdof);
            }
            soffs[subdomain_rank[ip]] +=  2 + nrdofs;
         }
      }
   }
   cout << "fes size = " << fespace->GetTrueVSize() << endl;

   // Communication
   Array<int> recvbuf(rbuff_size);
   MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_INT, recvbuf,
                 recv_count, recv_displ, MPI_INT, comm);
   //  // Extract from the recv_buffer
   std::vector<Array<int>> subdomain_true_dofs(nrsubdomains);

   int k=0;
   while (k<rbuff_size)
   {
      int ip = recvbuf[k];
      k++;
      int nrdofs = recvbuf[k];
      k++;
      for (int idof = 0; idof < nrdofs; ++idof)
      {
         subdomain_true_dofs[ip].Append(recvbuf[k+idof]);
      }
      k += nrdofs;
   }

   // build the lists from of tdofs on each subdomain
   subdomain_fespaces.SetSize(nrsubdomains);
   subdomain_dof_map.resize(nrsubdomains);
   const FiniteElementCollection * fec = fespace->FEColl();
   for (int ip=0; ip<nrsubdomains; ++ip)
   {
      subdomain_fespaces[ip] = nullptr;
      if (part->subdomain_mesh[ip])
      {
         subdomain_fespaces[ip] = new FiniteElementSpace(part->subdomain_mesh[ip],fec);
         // create the dof map
         int nrdof = subdomain_fespaces[ip]->GetTrueVSize();
         subdomain_dof_map[ip].SetSize(nrdof);
         int nrelems = part->element_map[ip].Size();
         int k = 0;
         for (int iel = 0; iel<nrelems; ++iel)
         {
            Array<int> subdomain_elem_dofs;
            subdomain_fespaces[ip]->GetElementDofs(iel,subdomain_elem_dofs);
            int ndof = subdomain_elem_dofs.Size();
            for (int i = 0; i<ndof; ++i)
            {
               int pdof_ = subdomain_elem_dofs[i];
               int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
               subdomain_dof_map[ip][pdof] = subdomain_true_dofs[ip][i+k];
            }
            k += ndof;
         }
      }
   }

   // Append the complex dof tdofs indices.
   for (int ip = 0; ip<nrsubdomains; ip++)
   {
      int tsize = subdomain_dof_map[ip].Size();
      if (tsize)
      {
         Array<int> imag_tdofs(subdomain_dof_map[ip]);
         for (int i = 0; i<tsize; i++)
         {
            imag_tdofs[i] += fespace->GetTrueVSize();
         }
         subdomain_dof_map[ip].Append(imag_tdofs);
      }
   }

   if (myid == 1)
   {
      cout << "subdomain size = " << subdomain_dof_map[2].Size() << endl;
      subdomain_dof_map[2].Print(cout, subdomain_dof_map[2].Size()/2);
   }




   MPI_Barrier(MPI_COMM_WORLD);
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
