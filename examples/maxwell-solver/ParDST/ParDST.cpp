//Diagonal Source Transfer Preconditioner

#include "ParDST.hpp"

// //1. Computing the list of global tdofs for each subdomain 
// //   on the host_rank
// SubdomainToGlobalMap::SubdomainToGlobalMap(ParFiniteElementSpace *fespace_, 
//                                          ParMeshPartition * part_)
// : fespace(fespace_), part(part_)                                         
// {
//    // MPI workspace
//    comm = fespace->GetComm();
//    MPI_Comm_size(comm, &num_procs);
//    MPI_Comm_rank(comm, &myid);
//    ComputeTdofOffsets();
//    myelemoffset = part->myelem_offset;
//    subdomain_rank = part->subdomain_rank;
//    nrsubdomains = part->nrsubdomains;
//    Setup();
// }


// void SubdomainToGlobalMap::Setup()
// {
//    // workspace for MPI_AlltoAll
//    send_count.SetSize(num_procs);  send_count = 0;
//    send_displ.SetSize(num_procs);  send_displ = 0; 
//    recv_count.SetSize(num_procs);  recv_count = 0;
//    recv_displ.SetSize(num_procs);  recv_displ = 0;
//    SubdomainLocalTrueDofs.resize(nrsubdomains);
   
//    //----------------------------------------------------------
//    // Each rank constructs the list of its own tdofs in a subdomain 
//    // After that they communicate it to the host rank to 
//    // create the complete list
//    Array<int> dof_marker(fespace->GetTrueVSize());
//    for (int ip = 0; ip<nrsubdomains; ++ip)
//    {
//       dof_marker = 0;
//       int nel = part->local_element_map[ip].Size();
//       for (int iel = 0; iel<nel; iel++)
//       {
//          int elem_idx = part->local_element_map[ip][iel] - myelemoffset;
//          Array<int> element_dofs;
//          fespace->GetElementDofs(elem_idx, element_dofs);
//          int ndofs = element_dofs.Size();
//          for (int i=0; i<ndofs; i++)
//          {
//             int pdof_ = element_dofs[i];
//             int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
//             int tdof = fespace->GetGlobalTDofNumber(pdof);
//             if (myid != get_rank(tdof)) continue;
//             if (dof_marker[tdof-mytoffset]) continue;
//             SubdomainLocalTrueDofs[ip].Append(tdof);
//             dof_marker[tdof-mytoffset] = 1;
//          }
//       }
//    }
//    // Communicate to the host rank
//    // 1. Construct send count
//    for (int ip = 0; ip<nrsubdomains; ++ip)
//    {
//       int ndofs = SubdomainLocalTrueDofs[ip].Size();
//       if (ndofs)
//       {
//          // Data to send to the subdomain rank
//          // 1. Subdomain no 
//          // 2. Number of dofs
//          // 3. The list of dofs
//          send_count[subdomain_rank[ip]] += 1 + 1 + ndofs; 
//       }
//    }
//    // 2. Construct receive count
//    MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
//    for (int k=0; k<num_procs-1; k++)
//    {
//       send_displ[k+1] = send_displ[k] + send_count[k];
//       recv_displ[k+1] = recv_displ[k] + recv_count[k];
//    }
//    sbuff_size = send_count.Sum();
//    rbuff_size = recv_count.Sum();
//    // now allocate space for the send buffer
//    Array<int> sendbuf(sbuff_size);  sendbuf = 0;
//    Array<int> soffs(num_procs); soffs = 0;
//    // 3. Fill up the send buffer
//    for (int ip = 0; ip<nrsubdomains; ++ip)
//    {
//       int ndofs = SubdomainLocalTrueDofs[ip].Size();
//       if (ndofs)
//       {
//          // Data to send to the subdomain rank
//          // 1. Subdomain no 
//          // 2. Number of dofs
//          // 3. The list of dofs
//          int j = send_displ[subdomain_rank[ip]] + soffs[subdomain_rank[ip]];
//          sendbuf[j] = ip;
//          sendbuf[j+1] = ndofs;
//          for (int k = 0; k < ndofs ; ++k)
//          {
//             sendbuf[j+2+k] = SubdomainLocalTrueDofs[ip][k];
//          }
//          soffs[subdomain_rank[ip]] +=  2 + ndofs;
//       }
//    }

//    // Communication
//    Array<int> recvbuf(rbuff_size);
//    MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_INT, recvbuf,
//                  recv_count, recv_displ, MPI_INT, comm);

//    SubdomainGlobalTrueDofs.resize(nrsubdomains);
//    int k=0;
//    while (k<rbuff_size)
//    {
//       int ip = recvbuf[k];
//       k++;
//       int ndofs = recvbuf[k];
//       k++;
//       for (int i = 0; i < ndofs; ++i)
//       {
//          SubdomainGlobalTrueDofs[ip].Append(recvbuf[k+i]);
//       }
//       k += ndofs;
//    }
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
   // }





   // const FiniteElementCollection * fec = fespace->FEColl();

   // subdomain_fespaces.SetSize(nrsubdomains);
   // subdomain_dof_map.resize(nrsubdomains);  


   // need to construct subdomain true dofs to global true dofs maps
   // TODO

   // for (int ip=0; ip<nrsubdomains; ++ip)
   // {
   //    subdomain_fespaces[ip] = nullptr;
   //    if (part->subdomain_mesh[ip])
   //    {
   //       // patch_true_dofs[ip].Print(cout, 20);
   //       subdomain_fespaces[ip] = new FiniteElementSpace(part->subdomain_mesh[ip],fec);
   //       // create the dof map
   //       int nrdof = subdomain_fespaces[ip]->GetTrueVSize();
   //       Array<int> dof_marker(nrdof); dof_marker = 0;
   //       subdomain_dof_map[ip].SetSize(nrdof);
   //       int nrelems = part->element_map[ip].Size();
   //       int k = 0;
   //       for (int iel = 0; iel<nrelems; ++iel)
   //       {
   //          Array<int> subdomain_elem_dofs;
   //          subdomain_fespaces[ip]->GetElementDofs(iel,subdomain_elem_dofs);
   //          int ndof = subdomain_elem_dofs.Size();
   //          for (int i = 0; i<ndof; ++i)
   //          {
   //             int pdof_ = subdomain_elem_dofs[i];
   //             int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
   //             if (dof_marker[pdof]) continue;
   //             // rearranging dofs from serial fespace for the subdomain 
   //             // to the ordering of the pfespace
   //             subdomain_dof_map[ip][pdof] = SubdomainGlobalTrueDofs[ip][k];
   //             k++;
   //             dof_marker[pdof] = 1;
   //          }
   //       }
   //    }
   // }




   // for (int ip=0; ip<1; ++ip)
   // {
   //    if(subdomain_fespaces[ip])
   //    {
   //       cout << "subdomain_dof_map = " << endl;
   //       subdomain_dof_map[ip].Print();
   //       cout << endl;
   //       cout << "SubdomainGlobalTrueDofs = " << endl;
   //       SubdomainGlobalTrueDofs[ip].Print();
   //    }
   // }
// }

// void SubdomainToGlobalMap::MultTranspose(const Vector & r, std::vector<Vector> & res)
// {
//    send_count = 0;
//    send_displ = 0;
//    recv_count = 0;
//    recv_displ = 0;

//    for (int ip = 0; ip < nrsubdomains; ip++)
//    {
//       int ndofs = SubdomainLocalTrueDofs[ip].Size();
//       send_count[subdomain_rank[ip]] += ndofs;
//    }

//    // communicate so that recv_count is constructed
//    MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
//    //
//    for (int k=0; k<num_procs-1; k++)
//    {
//       send_displ[k+1] = send_displ[k] + send_count[k];
//       recv_displ[k+1] = recv_displ[k] + recv_count[k];
//    }
//    sbuff_size = send_count.Sum();
//    rbuff_size = recv_count.Sum();

//    Array<double> sendbuf(sbuff_size); sendbuf = 0.0;
//    Array<int> soffs(num_procs); soffs = 0;

//    for (int ip = 0; ip < nrsubdomains; ip++)
//    {
//       int ndofs = SubdomainLocalTrueDofs[ip].Size();
//       for (int i = 0; i<ndofs; i++)
//       {
//          int j = send_displ[subdomain_rank[ip]] + soffs[subdomain_rank[ip]];
//          soffs[subdomain_rank[ip]]++;
//          int tdof = SubdomainLocalTrueDofs[ip][i];
//          sendbuf[j] = r[tdof - mytoffset];
//       }
//    }

//    // communication
//    Array<double> recvbuf(rbuff_size);

//    MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_DOUBLE, recvbuf,
//                  recv_count, recv_displ, MPI_DOUBLE, comm);
//    Array<int> roffs(num_procs); roffs = 0;
//    // Now each process will construct the res vector
//    res.resize(nrsubdomains);

//    for (int ip = 0; ip < nrsubdomains; ip++)
//    {
//       if (myid == subdomain_rank[ip])
//       {
//          int ndof = SubdomainGlobalTrueDofs[ip].Size();
//          res[ip].SetSize(ndof);
//          // extract the data from receiv buffer
//          for (int i=0; i<ndof; i++)
//          {
//             // pick up the tdof and find its rank
//             int tdof = SubdomainGlobalTrueDofs[ip][i];
//             int tdof_rank= get_rank(tdof);
//             int k = recv_displ[tdof_rank] + roffs[tdof_rank];
//             roffs[tdof_rank]++;
//             res[ip][i] = recvbuf[k];
//          }
//       }
//    }
// }


// void SubdomainToGlobalMap::Mult(const std::vector<Vector> & sol, Vector & z)
// {
//    send_count = 0;
//    send_displ = 0;
//    recv_count = 0;
//    recv_displ = 0;

//    // Compute send count
//    for (int ip = 0; ip < nrsubdomains; ip++)
//    {
//       if (myid == subdomain_rank[ip])
//       {
//          int ndofs = SubdomainGlobalTrueDofs[ip].Size();
//          // loop through dofs
//          for (int i=0; i<ndofs; i++)
//          {
//             //  pick up the dof and find its tdof_rank
//             int tdof = SubdomainGlobalTrueDofs[ip][i];
//             int tdof_rank= get_rank(tdof);
//             send_count[tdof_rank]++;
//          }
//       }
//    }

//    // communicate so that recv_count is constructed
//    MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
//    //
//    for (int k=0; k<num_procs-1; k++)
//    {
//       send_displ[k+1] = send_displ[k] + send_count[k];
//       recv_displ[k+1] = recv_displ[k] + recv_count[k];
//    }
//    sbuff_size = send_count.Sum();
//    rbuff_size = recv_count.Sum();

//    Array<double> sendbuf(sbuff_size); sendbuf = 0.0;
//    Array<int> soffs(num_procs); soffs = 0;

//    for (int ip = 0; ip < nrsubdomains; ip++)
//    {
//       if (myid == subdomain_rank[ip])
//       {
//          int ndofs = SubdomainGlobalTrueDofs[ip].Size();
//          // loop through dofs
//          for (int i=0; i<ndofs; i++)
//          {
//             //  pick up the dof and find its tdof_rank
//             int tdof = SubdomainGlobalTrueDofs[ip][i];
//             int tdof_rank= get_rank(tdof);
//             int k = send_displ[tdof_rank] + soffs[tdof_rank];
//             soffs[tdof_rank]++;
//             sendbuf[k] = sol[ip][i];
//          }
//       }
//    }

//    // communication
//    Array<double> recvbuf(rbuff_size);
//    MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_DOUBLE, recvbuf,
//                  recv_count, recv_displ, MPI_DOUBLE, comm);
//    Array<int> roffs(num_procs); roffs = 0;

//    // 1. Accummulate for the solution
//    for (int ip = 0; ip < nrsubdomains; ip++)
//    {
//       int ndofs = SubdomainLocalTrueDofs[ip].Size();
//       for (int i = 0; i<ndofs; i++)
//       {
//          int j = recv_displ[subdomain_rank[ip]] + roffs[subdomain_rank[ip]];
//          roffs[subdomain_rank[ip]]++;
//          int tdof = SubdomainLocalTrueDofs[ip][i];
//          z[tdof - mytoffset] += recvbuf[j];
//       }
//    }

// }



ParDST::ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, Coefficient * ws_,  int nrlayers_ , int nx_, int ny_, int nz_)
   : Solver(2*bf_->ParFESpace()->GetTrueVSize(), 2*bf_->ParFESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{
   pfes = bf->ParFESpace();
   fec = pfes->FEColl();
 

   comm = pfes->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

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
   DofMaps * test = new DofMaps(pfes,&part);

   // Vector B(pfes->TrueVSize()); B.Randomize(1); B = 1.0;
   // std::vector<Vector> res(nrsubdomains);

   // test->MultTranspose(B,res);
   
   // cout << endl;
   // for (int ip = 0; ip < nrsubdomains; ip++)
   // {
   //    cout << "ip = " << ip << ", res = " ; res[ip].Print();
   // }

   // B = 0.0;
   // test->Mult(res,B);

   // cout << "myid = " << myid << ", B = " ; B.Print();

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
