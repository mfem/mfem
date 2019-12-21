#include "additive_schwarzp.hpp"

// constructor
ParMeshPartition::ParMeshPartition(ParMesh *pmesh_) : pmesh(pmesh_)
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int dim = pmesh->Dimension();
   FiniteElementCollection * aux_fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace * aux_fespace = new ParFiniteElementSpace(pmesh, aux_fec);
   int mydofoffset = aux_fespace->GetMyTDofOffset(); // dof offset 

   // the DofTrueDof matrix for the mesh FE Space will gives the information 
   // for the vertices owned by the processor
   HypreParMatrix *DofTrueDof = new HypreParMatrix(*aux_fespace->Dof_TrueDof_Matrix());

   // Compute the TrueVr
   SparseMatrix diag, offd;
   DofTrueDof->GetDiag(diag);
   HYPRE_Int * cmap;
   DofTrueDof->GetOffd(offd, cmap);
   Array<int> own_vertices;
   // Need a map from all vertices to the "true" vertices (remove dublicates)
   int nrvert = pmesh->GetNV();
   Array<int> myTrueVert(nrvert);
   int nv = 0;
   for(int row = 0; row<nrvert; ++row)
   {  // the size of the row will always be one or zero
      int diag_row_size = diag.RowSize(row);
      int offd_row_size = offd.RowSize(row);
      int *diagcol = diag.GetRowColumns(row);
      int *offdcol = offd.GetRowColumns(row);
      // Check that they add up to one
      MFEM_VERIFY(diag_row_size + offd_row_size == 1, "diag and offd row size inconcistent")
      if (diag_row_size > 0)
      {
         myTrueVert[row] = diagcol[0] + mydofoffset;
         nv++;
         own_vertices.SetSize(nv);
         own_vertices[nv - 1] = myTrueVert[row];
      }
      else
      {
         myTrueVert[row] = cmap[offdcol[0]];
      }
   }

   // 6. Compute total number of patches
   MPI_Comm comm = pmesh->GetComm();
   int mynrpatch = own_vertices.Size();
   // Compute total number of patches.
   MPI_Allreduce(&mynrpatch, &nrpatch, 1, MPI_INT, MPI_SUM, comm);

   // need to identify the elements in each patch from each rank
   // Create a list of patches identifiers to all procs
   Array<int> patch_global_id(nrpatch);
   patch_rank.SetSize(nrpatch);
   Array<int> patch_rank_id(nrpatch);

   int count[num_procs];

   MPI_Allgather(&mynrpatch, 1, MPI_INT, &count[0], 1, MPI_INT, comm);
   int displs[num_procs];
   displs[0] = 0;
   for (int i = 1; i < num_procs; i++)
   {
      displs[i] = displs[i - 1] + count[i - 1];
   }

   patch_rank_id.SetSize(own_vertices.Size());
   patch_rank_id = myid;
   MPI_Allgatherv(own_vertices, mynrpatch, MPI_INT, patch_global_id, count, displs, MPI_INT, comm);
   MPI_Allgatherv(patch_rank_id, mynrpatch, MPI_INT, patch_rank, count, displs, MPI_INT, comm);

   Array<int> patch_id_index(patch_global_id[nrpatch-1]+1);
   patch_id_index = -1;
    for (int i = 0; i < nrpatch; i++)
   {
      int k = patch_global_id[i];
      patch_id_index[k] = i;
   }
   // now loop over all the elements and using the global dof of their vertices
   // the ids of the patches that they contribute to can be identified.
   int mynrelem = pmesh->GetNE();
   MPI_Scan(&mynrelem, &myelem_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   myelem_offset -= mynrelem;
   for (int iel=0; iel<mynrelem; ++iel)
   {
      Array<int> vertices;
      pmesh->GetElementVertices(iel,vertices);
      int nrvert = vertices.Size();
      for (int iv=0; iv<nrvert; ++iv) vertices[iv] += mydofoffset;
   }
   // now construct the element contribution information
   local_element_map.resize(nrpatch); 
   // cout << "nrpatch = " << nrpatch << endl;
   for (int iel=0; iel<mynrelem; ++iel)
   {
      // get element vertex index
      Array<int> vertices;
      pmesh->GetElementVertices(iel,vertices);
      int nrvert = vertices.Size();
      // fill in the element contribution lists
      for (int iv = 0; iv< nrvert; ++iv)
      {
         // find the "true vertex"
         int vert = vertices[iv];
         int truevert = myTrueVert[vert];
         // natural ordering of this patch
         int ip = patch_id_index[truevert];
         local_element_map[ip].Append(iel+myelem_offset);
      }
   }

   // communicate the element map to every processor that is involved
   element_map.resize(nrpatch);
   for (int ip = 0; ip < nrpatch; ++ip)
   {
      Array<int> count(num_procs);
      int size = local_element_map[ip].Size();
      count[myid] = size;
      MPI_Allgather(&size, 1, MPI_INT, count, 1, MPI_INT, comm);
      Array<int>displs(num_procs);
      displs[0] = 0;
      for (int j = 1; j < num_procs; j++)
      {
         displs[j] = displs[j-1] + count[j-1];
      }
      int tot_size = displs[num_procs - 1] + count[num_procs - 1];
      // Get a group identifier for comm.
      MPI_Group world_group_id;
      MPI_Comm new_comm = MPI_COMM_NULL;
      MPI_Group new_group_id;
      MPI_Comm_group (comm, &world_group_id);
      // count the ranks that do not have zero length
      int num_ranks = 0;
      for (int k = 0; k<num_procs; k++)
      {
         if (count[k] != 0) {num_ranks++;}
      }
      Array<int> new_count(num_ranks);
      Array<int> new_displs(num_ranks);
      int sub_comm_ranks[num_ranks];
      num_ranks = 0;
      for (int j = 0; j <num_procs ; j++ )
      {
         if (count[j] != 0)
         {
            sub_comm_ranks[num_ranks] = j;
            new_count[num_ranks] = count[j];
            new_displs[num_ranks] = displs[j];
            num_ranks++;
         }
      }
      MPI_Group_incl(world_group_id, num_ranks, sub_comm_ranks, &new_group_id);
      MPI_Comm_create(comm, new_group_id, &new_comm);
      if (size != 0)
      {
         element_map[ip].SetSize(tot_size);
         MPI_Allgatherv(local_element_map[ip],size,MPI_INT,
                        element_map[ip],new_count,new_displs,MPI_INT,new_comm);
      }
      MPI_Group_free(&world_group_id);
      MPI_Group_free(&new_group_id);
      if (new_comm != MPI_COMM_NULL) MPI_Comm_free(&new_comm);
   }

   // Now each process will send the vertex coords and elements to the patch host rank
   Array<int> send_count(num_procs);
   Array<int> send_displ(num_procs);
   Array<int> recv_count(num_procs);
   Array<int> recv_displ(num_procs);
   send_count = 0; send_displ = 0;
   recv_count = 0; recv_displ = 0;

   // send buffer for coordinates
   Array<int> send_count_d(num_procs);
   Array<int> send_displ_d(num_procs);
   Array<int> recv_count_d(num_procs);
   Array<int> recv_displ_d(num_procs);
   send_count_d = 0; send_displ_d = 0;
   recv_count_d = 0; recv_displ_d = 0;



   for (int ip = 0; ip < nrpatch; ++ip)
   {
      // a) patch no
      // b) element global number
      // c) type of the element (int)
      // c) number of vertices
      // d) global index of vertices
      // e) the coordinates of the vertices (x,y,z) // leave this for now
      //---------------------------------------------
      // get local element_map size
      int patch_local_nelems = local_element_map[ip].Size();
      if (patch_local_nelems !=0) // the rank is contributing to the patch ip
      {
         // loop through the elements 
         for (int iel=0; iel<patch_local_nelems; ++iel)
         {
         // get the vertices list for the element
            Array<int> elem_vertices;
            int iel_idx = local_element_map[ip][iel]-myelem_offset;
            pmesh->GetElementVertices(iel_idx,elem_vertices);
            int nrvert = elem_vertices.Size();
            send_count[patch_rank[ip]] += 1 + 1 + 1 + 1 + nrvert; 
            send_count_d[patch_rank[ip]] += dim * nrvert; 
         }   
      }
   }

   // comunicate so that recv_count is constructed
   MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
   MPI_Alltoall(send_count_d,1,MPI_INT,recv_count_d,1,MPI_INT,comm);
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
      send_displ_d[k+1] = send_displ_d[k] + send_count_d[k];
      recv_displ_d[k+1] = recv_displ_d[k] + recv_count_d[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   int sbuff_size_d = send_count_d.Sum();
   int rbuff_size_d = recv_count_d.Sum();

   // now allocate space for the send buffer
   Array<int> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> soffs(num_procs); soffs = 0;

   Array<double> sendbuf_d(sbuff_size_d);  sendbuf_d = 0.0;
   Array<int> soffs_d(num_procs); soffs_d = 0;

   // now the data will be placed according to process offsets
   for (int ip = 0; ip < nrpatch; ++ip)
   {
      // The send_buffer contains the following:
      // a) patch no 
      // b) element global number
      // c) number of vertices
      // d) global index of vertices
      // e) the coordinates of the vertices (x,y,z)
      // f) The type of the element
      //---------------------------------------------
      // get local element_map size
      int patch_local_nelems = local_element_map[ip].Size();
      if (patch_local_nelems !=0) // the rank is contributing to the patch ip
      {
         // loop through the elements 
         for (int iel=0; iel<patch_local_nelems; ++iel)
         {
            // get the vertex list for the element
            Array<int> elem_vertices;
            int iel_idx = local_element_map[ip][iel]-myelem_offset;
            pmesh->GetElementVertices(iel_idx,elem_vertices);
            int nrvert = elem_vertices.Size();
            // cout << ", nrvert = " << nrvert << endl;
            int j = send_displ[patch_rank[ip]] + soffs[patch_rank[ip]];
            int j_d = send_displ_d[patch_rank[ip]] + soffs_d[patch_rank[ip]];
            sendbuf[j] = ip; 
            sendbuf[j+1] = iel_idx + myelem_offset; 
            sendbuf[j+2] = pmesh->GetElementType(iel_idx); 
            sendbuf[j+3] = nrvert;
            for (int iv = 0; iv<nrvert; ++iv)
            {
               sendbuf[j+4+iv] = myTrueVert[elem_vertices[iv]];
               for (int comp=0; comp<dim; ++comp)
               {
                  sendbuf_d[j_d+iv+comp] = pmesh->GetVertex(elem_vertices[iv])[comp];
               }
               // j_d++; 
               j_d += dim-1;
            }
            soffs[patch_rank[ip]] += 1 + 1 + 1 + 1 + nrvert; 
            soffs_d[patch_rank[ip]] += dim * nrvert; 
         }   
      }
   }

   // Communication
   Array<int> recvbuf(rbuff_size);
   MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_INT, recvbuf,
                 recv_count, recv_displ, MPI_INT, comm);

   Array<double> recvbuf_d(rbuff_size_d);
   MPI_Alltoallv(sendbuf_d, send_count_d, send_displ_d, MPI_DOUBLE, recvbuf_d,
                 recv_count_d, recv_displ_d, MPI_DOUBLE, comm);


   // Extract from the recv_buffer
   std::vector<Array<int>> patch_elements(nrpatch);
   std::vector<Array<int>> patch_elements_type(nrpatch);
   std::vector<Array<int>> patch_vertices(nrpatch);
   std::vector<Array<double>> patch_vertex_xcoord(nrpatch);
   std::vector<Array<double>> patch_vertex_ycoord(nrpatch);
   std::vector<Array<double>> patch_vertex_zcoord(nrpatch);
   int k=0;
   int kd=0;

   while (k<rbuff_size)
   {
      int ip = recvbuf[k]; k++;
      patch_elements[ip].Append(recvbuf[k]); k++;
      patch_elements_type[ip].Append(recvbuf[k]); k++;
      int nrvert = recvbuf[k]; k++;
      int id = 0;
      for (int iv = 0; iv < nrvert; ++iv) 
      {
         patch_vertices[ip].Append(recvbuf[k+iv]);
         patch_vertex_xcoord[ip].Append(recvbuf_d[kd+iv+id]);
         patch_vertex_ycoord[ip].Append(recvbuf_d[kd+iv+1+id]);
         if (dim == 3) patch_vertex_zcoord[ip].Append(recvbuf_d[kd+iv+2+id]);
         // id ++; 
         id += dim-1;
      }
      k += nrvert;
      kd += dim* nrvert;
   }
   patch_mesh.SetSize(nrpatch);
   for (int ip = 0; ip < nrpatch; ++ip)
   {
      patch_mesh[ip] = nullptr;
      if (myid == patch_rank[ip]) 
      {
         Array<int> vertices_local_id(patch_vertices[ip].Size());
         // loop through the patch vertices;
         UniqueIndexGenerator gen;
         gen.Reset();
         for (int iv = 0; iv< patch_vertices[ip].Size(); ++iv)
         {
            int global_idx = patch_vertices[ip][iv];
            int local_idx = gen.Get(global_idx);
            vertices_local_id[iv] = local_idx;
         }
         int patch_nrvertices = gen.counter;
         int patch_nrelems = patch_elements[ip].Size();
         patch_mesh[ip] = new Mesh(dim,patch_nrvertices,patch_nrelems);
         // Add the vertices
         int k = -1;
         for (int iv = 0; iv<patch_vertices[ip].Size(); ++iv)
         {
            int vert_local_idx = vertices_local_id[iv];
            if (vert_local_idx > k)
            {
               double vert[dim];
               vert[0] = patch_vertex_xcoord[ip][iv];
               vert[1] = patch_vertex_ycoord[ip][iv];
               if (dim == 3) vert[2] = patch_vertex_zcoord[ip][iv];
               patch_mesh[ip]->AddVertex(vert);
               k++;
            }
         }

         int l = 0;
         for (int iel=0; iel<patch_nrelems; ++iel)
         {
            enum mfem::Element::Type elem_type;
            int type = patch_elements_type[ip][iel];
            int nrvert;
            GetNumVertices(type, elem_type, nrvert);
            // get the vertices list for the element
            int ind[nrvert];
            for (int iv = 0; iv<nrvert; ++iv)
            {
               ind[iv] = vertices_local_id[iv+l];
            }
            l += nrvert;
            AddElementToMesh(patch_mesh[ip],elem_type,ind);
         }
         patch_mesh[ip]->FinalizeTopology();
      }   
   }
   save_mesh_partition();
}





void ParMeshPartition::AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind)
{
   switch (elem_type)
   {
   case Element::QUADRILATERAL:
      mesh->AddQuad(ind);
      break;
   case Element::TRIANGLE :
      mesh->AddTri(ind);
      break;
   case Element::HEXAHEDRON :
      mesh->AddHex(ind);
      break;
   case Element::TETRAHEDRON :
      mesh->AddTet(ind);
      break;
   case Element::WEDGE :
      mesh->AddWedge(ind);
      break;
   default:
      MFEM_ABORT("Unknown element type");
      break;
   }
}



void ParMeshPartition::GetNumVertices(int type, mfem::Element::Type & elem_type, int & nrvert)
{
   switch (type)
   {
   case 0:
      elem_type = Element::POINT;
      nrvert = 1;
      break;
   case 1:
      elem_type = Element::SEGMENT;
      nrvert = 2;
      break;
   case 2:
      elem_type = Element::TRIANGLE;
      nrvert = 3;
      break;
   case 3:
      elem_type = Element::QUADRILATERAL;
      nrvert = 4;
      break;
   case 4:
      elem_type = Element::TETRAHEDRON;
      nrvert = 4;
      break;
   case 5:
      elem_type = Element::HEXAHEDRON;
      nrvert = 8;
      break;
   case 6:
      elem_type = Element::WEDGE;
      nrvert = 6;
      break;               
   default:
      MFEM_ABORT("Unknown element type");
      break;
   }
}


void ParMeshPartition::save_mesh_partition()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      if (patch_mesh[ip])
      {
         ostringstream mesh_name;
         mesh_name << "output/mesh." << setfill('0') << setw(6) << ip;
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         patch_mesh[ip]->Print(mesh_ofs);
      }
   }
}

ParMeshPartition::~ParMeshPartition()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      delete patch_mesh[ip]; patch_mesh[ip] = nullptr;
   }
   patch_mesh.DeleteAll();
}




ParPatchDofInfo::ParPatchDofInfo(ParFiniteElementSpace *fespace)
{
   MPI_Comm comm = fespace->GetComm();
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(comm, &myid);
   ParMesh * pmesh = fespace->GetParMesh();
   p = new ParMeshPartition(pmesh); 
   nrpatch = p->nrpatch;
   int myelemoffset = p->myelem_offset;
   patch_rank = p->patch_rank;

   Array<int> send_count(num_procs);
   Array<int> send_displ(num_procs);
   Array<int> recv_count(num_procs);
   Array<int> recv_displ(num_procs);
   send_count = 0; send_displ = 0;
   recv_count = 0; recv_displ = 0;

   // each element contributing to the patch has to communicate to the patch_rank the list
   // of its tdof numbers whether it owns them or not
   // there is no problem with dublicates since they will be overwritten
   // After these lists are constructed to the host rank then they will be brodcasted to 
   // the participating ranks 

   // calculate the sent_count for each patch
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      int nrelems = p->local_element_map[ip].Size();
      if(nrelems >0 )
      {
         for (int iel=0; iel<nrelems; ++iel)
         {
            Array<int>element_dofs;
            int elem_idx = p->local_element_map[ip][iel] - myelemoffset;
            fespace->GetElementDofs(elem_idx, element_dofs);
            int nrdofs = element_dofs.Size();
            // send the number of dofs for each element and the tdof numbers
            send_count[patch_rank[ip]] += 1 + 1 + nrdofs; // patch no, nrdofs the tdofs
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
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      int nrelems = p->local_element_map[ip].Size();

      if(nrelems > 0)
      {
         for (int iel=0; iel<nrelems; ++iel)
         {
            Array<int>element_dofs;
            int elem_idx = p->local_element_map[ip][iel] - myelemoffset;
            fespace->GetElementDofs(elem_idx, element_dofs);
            int nrdofs = element_dofs.Size();
            int j = send_displ[patch_rank[ip]] + soffs[patch_rank[ip]];
            sendbuf[j] = ip; 
            sendbuf[j+1] = nrdofs; 
            for (int idof = 0; idof < nrdofs ; ++idof)
            {
               int pdof_ = element_dofs[idof];
               int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
               sendbuf[j+2+idof] = fespace->GetGlobalTDofNumber(pdof);
            }
            soffs[patch_rank[ip]] +=  2 + nrdofs; 
         }
      }
   }

   // Communication
   Array<int> recvbuf(rbuff_size);
   MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_INT, recvbuf,
                 recv_count, recv_displ, MPI_INT, comm);
   //  // Extract from the recv_buffer
   std::vector<Array<int>> patch_true_dofs(nrpatch);

   int k=0;
   while (k<rbuff_size)
   {
      int ip = recvbuf[k]; k++;
      int nrdofs = recvbuf[k]; k++;
      for (int idof = 0; idof < nrdofs; ++idof) 
      {
         patch_true_dofs[ip].Append(recvbuf[k+idof]);
      }
      k += nrdofs;
   }
   // for (int ip = 0; ip<nrpatch; ip++)
   // {
   //    if (patch_true_dofs[ip].Size()>0)
   //    {
   //       cout << "myid = " << myid << ", ip = " << ip << ", tdofs = " ; patch_true_dofs[ip].Print(cout, 20);
   //    }
   // }
   // build the maps from patch true dof to global truedof
   patch_fespaces.SetSize(nrpatch);
   patch_dof_map.resize(nrpatch);
   const FiniteElementCollection * fec = fespace->FEColl();
   for (int ip=0; ip<nrpatch; ++ip)
   {
      patch_fespaces[ip] = nullptr;
      if(p->patch_mesh[ip])
      {
         patch_fespaces[ip] = new FiniteElementSpace(p->patch_mesh[ip],fec);
         // create the dof map
         int nrdof = patch_fespaces[ip]->GetTrueVSize();
         patch_dof_map[ip].SetSize(nrdof);
         int nrelems = p->element_map[ip].Size();
         int k = 0;
         for (int iel = 0; iel<nrelems; ++iel)
         {
            Array<int> patch_elem_dofs;
            patch_fespaces[ip]->GetElementDofs(iel,patch_elem_dofs);
            int ndof = patch_elem_dofs.Size();
            for (int i = 0; i<ndof; ++i)
            {
               int pdof_ = patch_elem_dofs[i];
               int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
               patch_dof_map[ip][pdof] = patch_true_dofs[ip][i+k]; 
            }
            k += ndof;
         }
      }
   }
   // for (int ip=0; ip<nrpatch; ++ip)
   // {
   //    if(p->patch_mesh[ip])
   //    {
   //       cout << "ip = " << ip << " patch_dof_map = " ; patch_dof_map[ip].Print(cout, 20); 
   //    }
   // }
}


// constructor
ParPatchAssembly::ParPatchAssembly(ParBilinearForm * bf_) : bf(bf_)
{
   fespace = bf->ParFESpace();
   comm = fespace->GetComm();
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   compute_trueoffsets();
   ParPatchDofInfo * patch_dofs = new ParPatchDofInfo(fespace);

   nrpatch = patch_dofs->nrpatch;
   patch_rank = patch_dofs->patch_rank;
   // share the dof map with the contributing ranks
   Array<int> send_count(num_procs);
   Array<int> send_displ(num_procs);
   Array<int> recv_count(num_procs);
   Array<int> recv_displ(num_procs);
   send_count = 0; send_displ = 0;
   recv_count = 0; recv_displ = 0;

   for (int ip = 0; ip < nrpatch; ++ip)
   {
      int nrdofs = patch_dofs->patch_dof_map[ip].Size();
      if (nrdofs > 0)
      {   
         Array<int> patch_dofs_ranks(num_procs);
         patch_dofs_ranks = 0;
         // loop through the dofs and find their rank
         for (int i = 0; i<nrdofs; ++i)
         {
            int tdof = patch_dofs->patch_dof_map[ip][i];
            int rank = get_rank(tdof);
            patch_dofs_ranks[rank] = 1;
         }
         for (int irank = 0; irank<num_procs; ++irank)
         {
            if (patch_dofs_ranks[irank] == 1)
            {
               send_count[irank] += 2+nrdofs; // patch_number and size
            }
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
   Array<double> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> soffs(num_procs); soffs = 0;

   for (int ip = 0; ip < nrpatch; ++ip)
   {
      int nrdofs = patch_dofs->patch_dof_map[ip].Size();
      if (nrdofs > 0)
      {   
         Array<int> patch_dofs_ranks(num_procs);
         patch_dofs_ranks = 0;
         // loop through the dofs and find their rank
         for (int i = 0; i<nrdofs; ++i)
         {
            int tdof = patch_dofs->patch_dof_map[ip][i];
            int rank = get_rank(tdof);
            patch_dofs_ranks[rank] = 1;
         }
         for (int irank = 0; irank<num_procs; ++irank)
         {
            if (patch_dofs_ranks[irank] == 1)
            {
               int j = send_displ[irank] + soffs[irank];
               sendbuf[j] = ip;
               sendbuf[j+1] = nrdofs;
               for (int i = 0; i<nrdofs; ++i)
               {
                  sendbuf[j+2+i] = patch_dofs->patch_dof_map[ip][i];
               }
               soffs[irank] += nrdofs + 2;
            }
         }
      }
   }

   Array<double> recvbuf(rbuff_size);

   MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_DOUBLE, recvbuf,
                 recv_count, recv_displ, MPI_DOUBLE, comm);

   patch_true_dofs.resize(nrpatch);
   patch_local_dofs.resize(nrpatch);
   patch_other_dofs.resize(nrpatch);

   // recvbuf.Print(cout,10);
   int k=0;
   while (k<rbuff_size)
   {
      int ip = recvbuf[k]; k++;
      int nrdofs = recvbuf[k]; k++;
      for (int idof = 0; idof < nrdofs; ++idof) 
      {
         int tdof = recvbuf[k+idof];
         patch_true_dofs[ip].Append(tdof);
         if (get_rank(tdof) == myid)
         {
            patch_local_dofs[ip].Append(tdof);
         }
         else
         {
            patch_other_dofs[ip].Append(tdof);
         }
      }
      k += nrdofs;
   }

   // for (int ip=0; ip<nrpatch; ++ip)
   // {
   //    int nrdofs = patch_true_dofs[ip].Size();
   //    if (myid == patch_rank[ip])
   //    {
   //       if (patch_true_dofs[ip].Size() > 0)
   //       {
   //          cout << "myid: " << myid << ", ip: " << ip 
   //             << ", tdofs: " ; patch_true_dofs[ip].Print(cout,10); 
   //          cout << "myid: " << myid << ", ip: " << ip 
   //             << ", mydofs: " ; patch_local_dofs[ip].Print(cout,10);      
   //       }
   //    }
   // }

   AssemblePatchMatrices(patch_dofs);

}

void ParPatchAssembly::AssemblePatchMatrices(ParPatchDofInfo * p)
{
   patch_mat.SetSize(nrpatch);
   patch_mat_inv.SetSize(nrpatch);
   ess_tdof_list.resize(nrpatch);
   for (int ip=0; ip<nrpatch; ++ip)
   {
      if (p->p->patch_mesh[ip])
      {
         // Define the patch bilinear form and apply boundary conditions (only the LHS)
         FiniteElementSpace * patch_fespace = p->patch_fespaces[ip];
         Mesh * patch_mesh = p->p->patch_mesh[ip];
         if (patch_mesh->bdr_attributes.Size())
         {
            Array<int> ess_bdr(patch_mesh->bdr_attributes.Max());
            ess_bdr = 1;
            patch_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list[ip]);
         }
         BilinearForm a(patch_fespace, bf);
         a.Assemble(); 
         OperatorPtr Alocal;
         a.FormSystemMatrix(ess_tdof_list[ip],Alocal);
         patch_mat[ip] = new SparseMatrix((SparseMatrix&)(*Alocal));
         patch_mat[ip]->Threshold(0.0);
         // if (ip == 1) patch_mat[ip]->PrintMatlab(cout);
         // if (ip == 4) patch_mat[ip]->PrintMatlab(cout);
         // Vector ones(p->patch_fespaces[ip]->GetTrueVSize());
         // ones = 1.0;
         // patch_mat[ip] = new SparseMatrix(ones);
         // Save the inverse
         patch_mat_inv[ip] = new KLUSolver;
         patch_mat_inv[ip]->SetOperator(*patch_mat[ip]);
      }   
   }

}

void ParPatchAssembly::compute_trueoffsets()
{
   int num_procs;
   MPI_Comm_size(comm, &num_procs);
   tdof_offsets.resize(num_procs);
   int mytoffset = fespace->GetMyTDofOffset();
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

int ParPatchAssembly::get_rank(int tdof)
{
   int size = tdof_offsets.size();
   if (size == 1) {return 0;}
   std::vector<int>::iterator up;
   up=std::upper_bound (tdof_offsets.begin(), tdof_offsets.end(), tdof); //          ^
   return std::distance(tdof_offsets.begin(),up)-1;
}




ParPatchRestriction::ParPatchRestriction(ParPatchAssembly * P_) : P(P_)
{
   comm = P->comm;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   nrpatch = P->nrpatch;
   patch_rank = P->patch_rank;

   send_count.SetSize(num_procs);
   send_displ.SetSize(num_procs);
   recv_count.SetSize(num_procs);
   recv_displ.SetSize(num_procs);

   send_count = 0; send_displ = 0;
   recv_count = 0; recv_displ = 0;

   // Precompute send_counts
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int ndofs = P->patch_local_dofs[ip].Size();
      for (int i =0; i<ndofs; i++)
      {
         int tdof = P->patch_local_dofs[ip][i];
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            send_count[patch_rank[ip]]++;
         }
      }
   }

   // communicate so that recv_count is constructed
   MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
   //
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   sbuff_size = send_count.Sum();
   rbuff_size = recv_count.Sum();
}

void ParPatchRestriction::Mult(const Vector & r , Array<Vector> & res)
{
   int mytdofoffset = P->fespace->GetMyTDofOffset();
   // now allocate space for the send buffer
   Array<double> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> soffs(num_procs); soffs = 0;
   // now the data will be placed according to process offsets
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int ndofs = P->patch_local_dofs[ip].Size();
      for (int i = 0; i<ndofs; i++)
      {
         int tdof = P->patch_local_dofs[ip][i];
         // find its rank
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            int j = send_displ[patch_rank[ip]] + soffs[patch_rank[ip]];
            soffs[patch_rank[ip]]++;
            int k = tdof - mytdofoffset;
            sendbuf[j] = r[k];
         }
      }
   }
      
   // communication
   Array<double> recvbuf(rbuff_size);
      
   MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_DOUBLE, recvbuf,
                 recv_count, recv_displ, MPI_DOUBLE, comm);            
   Array<int> roffs(num_procs);
   roffs = 0;
   // Now each process will construct the res1 vector
   res.SetSize(nrpatch);
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if(myid == patch_rank[ip]) 
      {
         int ndof = P->patch_true_dofs[ip].Size();
         res[ip].SetSize(ndof);
         // extract the data from receiv buffer
         // loop through rows
         int l = 0;
         for (int i=0; i<ndof; i++)
         {
            // pick up the dof and find its tdof_rank
            int tdof = P->patch_true_dofs[ip][i];
            int tdof_rank= P->get_rank(tdof);
            // offset
            int k = recv_displ[tdof_rank] + roffs[tdof_rank];
            roffs[tdof_rank]++;
            res[ip][i] = recvbuf[k];
         }   
      }
   }
}

void ParPatchRestriction::MultTranspose(const Array<Vector > & sol, Vector & z)
{
   int mytdofoffset = P->fespace->GetMyTDofOffset();
   // Step 3: Propagate the information to the global solution vector
   // (the recv_buff becomes the sendbuff and vice-versa) 
   Array<double> sendbuf(sbuff_size);  sendbuf = 0.0;
   Array<double> recvbuf(rbuff_size);  recvbuf = 0.0;
   Array<int> roffs(num_procs); roffs = 0;
   Array<int> soffs(num_procs); soffs = 0;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if(myid == patch_rank[ip]) 
      {
         int ndofs = P->patch_true_dofs[ip].Size();
         // loop through dofs
         for (int i=0; i<ndofs; i++)
         {
            //  pick up the dof and find its tdof_rank
            int tdof = P->patch_true_dofs[ip][i];
            int tdof_rank= P->get_rank(tdof);
            // offset
            int k = recv_displ[tdof_rank] + roffs[tdof_rank];
            roffs[tdof_rank]++;
            recvbuf[k] = sol[ip][i];
         }
      }
   }
// now communication
   
   MPI_Alltoallv(recvbuf, recv_count, recv_displ, MPI_DOUBLE, sendbuf,
                 send_count, send_displ, MPI_DOUBLE, comm);            

   // 1. Accummulate for the solution 
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int ndofs = P->patch_local_dofs[ip].Size();
      for (int i = 0; i<ndofs; i++)
      {
         int tdof = P->patch_local_dofs[ip][i];
         // find its rank
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank) 
         {  
            int k = tdof - mytdofoffset;
            int j = send_displ[patch_rank[ip]] + soffs[patch_rank[ip]];
            soffs[patch_rank[ip]]++;
            z[k] += sendbuf[j];
         }
      }
   }
}


ParAddSchwarz::ParAddSchwarz(ParBilinearForm * bf_) 
: Solver(bf_->ParFESpace()->GetTrueVSize(), bf_->ParFESpace()->GetTrueVSize())
{
   comm = bf_->ParFESpace()->GetComm();
   p = new ParPatchAssembly(bf_);
   nrpatch = p->nrpatch;
   R = new ParPatchRestriction(p);
}


void ParAddSchwarz::Mult(const Vector &r, Vector &z) const
{
   int myid;
   MPI_Comm_rank(comm, &myid);

   z = 0.0;
   Vector rnew(r);
   Vector znew(z);
   
   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;
      Array<Vector > res;
      R->Mult(rnew,res);

      Array<Vector > sol(nrpatch);
      for (int ip=0; ip<nrpatch; ip++)
      {
         if(myid == p->patch_rank[ip]) 
         {
            sol[ip].SetSize(res[ip].Size());
            p->patch_mat_inv[ip]->Mult(res[ip], sol[ip]);
            // zero out the essential truedofs 
            Array<int> ess_bdr_indices = p->ess_tdof_list[ip];
            sol[ip].SetSubVector(ess_bdr_indices,0.0);
         }
      }
      R->MultTranspose(sol,znew);

      znew *= theta; // relaxation parameter
      z+= znew;
      // Update residual
      Vector raux(znew.Size());
      A->Mult(znew,raux); 
      rnew -= raux;
   } // end of loop through smoother iterations
}









// ParPatchRestriction::ParPatchRestriction(ParPatchAssembly * P_) : P(P_)
// {
//    comm = P->comm;
//    MPI_Comm_size(comm, &num_procs);
//    MPI_Comm_rank(comm, &myid);
//    nrpatch = P->nrpatch;
//    patch_rank = P->patch_rank;

//    send_count.SetSize(num_procs);
//    send_displ.SetSize(num_procs);
//    recv_count.SetSize(num_procs);
//    recv_displ.SetSize(num_procs);

//    send_count = 0; send_displ = 0;
//    recv_count = 0; recv_displ = 0;

//    // Precompute send_counts
//    for (int ip = 0; ip < nrpatch; ip++)
//    {
//       if (myid != patch_rank[ip])
//       {
//          int ndofs = P->patch_local_dofs[ip].Size();
//          for (int i =0; i<ndofs; i++)
//          {
//             int tdof = P->patch_local_dofs[ip][i];
//             int tdof_rank = P->get_rank(tdof);
//             if (myid == tdof_rank)
//             {
//                send_count[patch_rank[ip]]++;
//             }
//          }
//       }
       
//    }

//    // comunicate so that recv_count is constructed
//    MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);
//    //
//    for (int k=0; k<num_procs-1; k++)
//    {
//       send_displ[k+1] = send_displ[k] + send_count[k];
//       recv_displ[k+1] = recv_displ[k] + recv_count[k];
//    }
//    sbuff_size = send_count.Sum();
//    rbuff_size = recv_count.Sum();
// }



// void ParAddSchwarz::Mult(const Vector &r, Vector &z) const
// {
//    int myid;
//    MPI_Comm_rank(comm, &myid);

//    z = 0.0;
//    Vector rnew(r);
//    Vector znew(z);
   
//    for (int iter = 0; iter < maxit; iter++)
//    {
//       znew = 0.0;
//       Array<BlockVector * > res;
//       R->Mult(rnew,res);

//       Array<BlockVector*> sol(nrpatch);
//       for (int ip=0; ip<nrpatch; ip++)
//       {
//          sol[ip] = nullptr;
//          if(myid == p->patch_rank[ip]) 
//          {
//             Array<int> block_offs(3);
//             block_offs[0] = 0;
//             block_offs[1] = res[ip]->GetBlock(0).Size();
//             block_offs[2] = res[ip]->GetBlock(1).Size();
//             block_offs.PartialSum();
//             sol[ip] = new BlockVector(block_offs);
//             p->patch_mat_inv[ip]->Mult(*res[ip], *sol[ip]);
//             // zero out the essential truedofs 
//             // Array<int> ess_bdr_indices = p->ess_tdof_list[ip];
//             // sol[ip]->SetSubVector(ess_bdr_indices,0.0);
//          }
//       }
//       for (auto p:res) delete p;
//       res.DeleteAll();
//       R->MultTranspose(sol,znew);
//       for (auto p:sol) delete p;
//       sol.DeleteAll();

//       znew *= theta; // relaxation parameter
//       z+= znew;
//       // Update residual
//       Vector raux(znew.Size());
//       A->Mult(znew,raux); 
//       rnew -= raux;
//    } // end of loop through smoother iterations
// }










// void ParPatchRestriction::Mult(const Vector & r , Array<BlockVector *> & res)
// {
//    int mytdofoffset = P->fespace->GetMyTDofOffset();
//    Array<Vector> res0(nrpatch); // residual on the processor
//    Array<Vector> res1(nrpatch); // residual off the processor
//    //  Part of the residual on the processor
//    for (int ip = 0; ip < nrpatch; ip++)
//    {
//       if (myid == patch_rank[ip]) 
//       {
//          Array<int> local_dof_map(P->patch_local_dofs[ip]);
//          for (int i=0;i<local_dof_map.Size(); i++)
//          {
//             local_dof_map[i] -= mytdofoffset;
//          }
//          r.GetSubVector(local_dof_map, res0[ip]);
//       }
//    }
//    // now allocate space for the send buffer
//    Array<double> sendbuf(sbuff_size);  sendbuf = 0;
//    Array<int> soffs(num_procs); soffs = 0;
//    // now the data will be placed according to process offsets
//    for (int ip = 0; ip < nrpatch; ip++)
//    {
//       if (myid != patch_rank[ip])
//       {
//          int ndofs = P->patch_local_dofs[ip].Size();
//          for (int i = 0; i<ndofs; i++)
//          {
//             int tdof = P->patch_local_dofs[ip][i];
//             // find its rank
//             int tdof_rank = P->get_rank(tdof);
//             if (myid == tdof_rank)
//             {
//                int j = send_displ[patch_rank[ip]] + soffs[patch_rank[ip]];
//                soffs[patch_rank[ip]]++;
//                int k = tdof - mytdofoffset;
//                sendbuf[j] = r[k];
//             }
//          }
//       }
//    }
      
//    // communication
//    Array<double> recvbuf(rbuff_size);
//    double * sendbuf_ptr = nullptr;
//    double * recvbuf_ptr = nullptr;
//    if (sbuff_size !=0 ) sendbuf_ptr = &sendbuf[0]; 
//    if (rbuff_size !=0 ) recvbuf_ptr = &recvbuf[0]; 
      
//    MPI_Alltoallv(sendbuf_ptr, send_count, send_displ, MPI_DOUBLE, recvbuf_ptr,
//                  recv_count, recv_displ, MPI_DOUBLE, comm);            
//    Array<int> roffs(num_procs);
//    roffs = 0;
//    // Now each process will construct the res1 vector
//    for (int ip = 0; ip < nrpatch; ip++)
//    {
//       if(myid == patch_rank[ip]) 
//       {
//          int ndof = P->patch_other_dofs[ip].Size();
//          res1[ip].SetSize(ndof);
//          // extract the data from receiv buffer
//          // loop through rows
//          for (int i=0; i<ndof; i++)
//          {
//             // pick up the dof and find its tdof_rank
//             int tdof = P->patch_other_dofs[ip][i];
//             int tdof_rank= P->get_rank(tdof);
//             // offset
//             int k = recv_displ[tdof_rank] + roffs[tdof_rank];
//             roffs[tdof_rank]++;
//             res1[ip][i] = recvbuf[k]; 
//          }   
//       }
//    }

//    res.SetSize(nrpatch);
//    for (int ip=0; ip<nrpatch; ip++)
//    {
//       res[ip] = nullptr;
//       if(myid == patch_rank[ip]) 
//       {
//          Array<int> block_offs(3);
//          block_offs[0] = 0;
//          block_offs[1] = res0[ip].Size();
//          block_offs[2] = res1[ip].Size();
//          block_offs.PartialSum();
//          res[ip] = new BlockVector(block_offs);
//          res[ip]->SetVector(res0[ip], 0);
//          res[ip]->SetVector(res1[ip], res0[ip].Size());
//       }
//    }
// }

// void ParPatchRestriction::MultTranspose(const Array<BlockVector* > & sol, Vector & z)
// {
//    int mytdofoffset = P->fespace->GetMyTDofOffset();
//    Array<Vector> sol0(nrpatch);
//    Array<Vector> sol1(nrpatch);
//    // Step 3: Propagate the information to the global solution vector
//    // (the recv_buff becomes the sendbuff and vice-versa) 
//    Array<double> sendbuf(sbuff_size);  sendbuf = 0.0;
//    Array<double> recvbuf(rbuff_size);  recvbuf = 0.0;
//    Array<int> roffs(num_procs); roffs = 0;
//    Array<int> soffs(num_procs); soffs = 0;
//    for (int ip = 0; ip < nrpatch; ip++)
//    {
//       if(myid == patch_rank[ip]) 
//       {
//          sol1[ip] = sol[ip]->GetBlock(1);
//          int ndofs = P->patch_other_dofs[ip].Size();
//          // loop through dofs
//          for (int i=0; i<ndofs; i++)
//          {
//             //  pick up the dof and find its tdof_rank
//             int tdof = P->patch_other_dofs[ip][i];
//             int tdof_rank= P->get_rank(tdof);
//             // offset
//             int k = recv_displ[tdof_rank] + roffs[tdof_rank];
//             roffs[tdof_rank]++;
//             recvbuf[k] = sol1[ip][i]; 
//          }   
//       }
//    }
// // now communication
//    double * sendbuf_ptr = nullptr;
//    double * recvbuf_ptr = nullptr;
//    if (sbuff_size !=0 ) sendbuf_ptr = &sendbuf[0]; 
//    if (rbuff_size !=0 ) recvbuf_ptr = &recvbuf[0]; 
   
//    MPI_Alltoallv(recvbuf_ptr, recv_count, recv_displ, MPI_DOUBLE, sendbuf_ptr,
//                  send_count, send_displ, MPI_DOUBLE, comm);            

//    // 1. Accummulate for the solution to other prosessors
//    for (int ip = 0; ip < nrpatch; ip++)
//    {
//       if (myid != patch_rank[ip])
//       {
//          int ndofs = P->patch_local_dofs[ip].Size();
//          for (int i = 0; i<ndofs; i++)
//          {
//             int tdof = P->patch_local_dofs[ip][i];
//             // find its rank
//             int tdof_rank = P->get_rank(tdof);
//             if (myid == tdof_rank)
//             {
//                int j = send_displ[patch_rank[ip]] + soffs[patch_rank[ip]];
//                soffs[patch_rank[ip]]++;
//                int k = tdof - mytdofoffset;
//                z[k] += sendbuf[j];
//             }  
//          }
//       }
//    }
//    // 2. Accummulate for the solution on the processor
//    for (int ip = 0; ip < nrpatch; ip++)
//    {
//       if (myid == patch_rank[ip])
//       {
//          sol0[ip] = sol[ip]->GetBlock(0);
//          Array<int> local_dof_map(P->patch_local_dofs[ip]);
//          for (int i=0;i<local_dof_map.Size(); i++)
//          {
//             local_dof_map[i] -= mytdofoffset;
//          }
//          z.AddElementVector(local_dof_map,sol0[ip]);
//       }
//    }
// }
