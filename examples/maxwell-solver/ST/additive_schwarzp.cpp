
#include "additive_schwarzp.hpp"

// constructor
CartesianParMeshPartition::CartesianParMeshPartition(ParMesh *pmesh_) : pmesh(
      pmesh_)
{
   int num_procs,myid;
   MPI_Comm comm = pmesh->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   int dim = pmesh->Dimension();

   int nx = 2;
   int ny = 2;
   int nz = 1;
   int nxyz[3] = {nx,ny,nz};
   nrpatch = nx*ny*nz;
   double pmin[3] = { infinity(), infinity(), infinity() };
   double pmax[3] = { -infinity(), -infinity(), -infinity() };

   // find a bounding box using the vertices
   for (int vi = 0; vi < pmesh->GetNV(); vi++)
   {
      const double *p = pmesh->GetVertex(vi);
      for (int i = 0; i < dim; i++)
      {
         if (p[i] < pmin[i])
         {
            pmin[i] = p[i];
         }
         if (p[i] > pmax[i])
         {
            pmax[i] = p[i];
         }
      }
   }

   double global_min[dim];
   double global_max[dim];

   for (int comp = 0; comp<dim; comp++)
   {
      MPI_Allreduce(&pmin[comp], &global_min[comp], 1, MPI_DOUBLE, MPI_MIN, comm);
      MPI_Allreduce(&pmax[comp], &global_max[comp], 1, MPI_DOUBLE, MPI_MAX, comm);
   }

   int mynrelem = pmesh->GetNE();
   int partitioning[mynrelem];

   // determine the partitioning using the centers of the elements
   double ppt[dim];
   Vector pt(ppt, dim);
   for (int el = 0; el < mynrelem; el++)
   {
      pmesh->GetElementTransformation(el)->Transform(
         Geometries.GetCenter(pmesh->GetElementBaseGeometry(el)), pt);
      int part = 0;
      for (int i = dim-1; i >= 0; i--)
      {
         int idx = (int)floor(nxyz[i]*((pt(i) - global_min[i])/(global_max[i] -
                                                                global_min[i])));
         if (idx < 0)
         {
            idx = 0;
         }
         if (idx >= nxyz[i])
         {
            idx = nxyz[i]-1;
         }
         part = part * nxyz[i] + idx;
      }
      partitioning[el] = part;
   }

   int myelem_offset;
   MPI_Scan(&mynrelem, &myelem_offset, 1, MPI_INT, MPI_SUM, comm);
   myelem_offset -= mynrelem;

   // loop through elements and construct local_element maps
   local_element_map.resize(nrpatch);
   for (int iel = 0; iel < mynrelem; iel++)
   {
      int ip = partitioning[iel];
      local_element_map[ip].Append(iel+myelem_offset);
   }

   Array<int>patch_size(nrpatch);
   for (int ip = 0; ip < nrpatch; ++ip)
   {
      patch_size[ip] = local_element_map[ip].Size();
   }

   Array<int>patch_ranks(nrpatch*num_procs);
   MPI_Allgather(patch_size, nrpatch, MPI_INT, patch_ranks, nrpatch, MPI_INT,
                 comm);

   Array<int> max(nrpatch);
   max = -1;
   patch_rank.SetSize(nrpatch);
   patch_rank = -1;
   // loop through the patches and determine the rank with the max number of elements
   for (int irank = 0; irank < num_procs; ++irank)
   {
      int offset = irank*nrpatch;
      for (int ip = 0; ip<nrpatch; ++ip)
      {
         if (patch_ranks[ip+offset]>= max[ip])
         {
            max[ip] = patch_ranks[ip+offset];
            patch_rank[ip] = irank;
         }
      }
   }
}

// constructor
VertexParMeshPartition::VertexParMeshPartition(ParMesh *pmesh_) : pmesh(pmesh_)
{
   int num_procs;
   MPI_Comm comm = pmesh->GetComm();
   MPI_Comm_size(comm, &num_procs);
   int dim = pmesh->Dimension();
   FiniteElementCollection * aux_fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace * aux_fespace = new ParFiniteElementSpace(pmesh, aux_fec);
   int mytdofoffset = aux_fespace->GetMyTDofOffset(); // dof offset

   // 6. Compute total number of patches
   nrpatch = aux_fespace->GlobalTrueVSize();
   // Create a list of patch identifiers to all procs
   patch_rank.SetSize(nrpatch);
   Array<int> true_vert_offsets;
   true_vert_offsets.SetSize(num_procs);
   int myvert_offset = aux_fespace->GetMyTDofOffset();
   MPI_Allgather(&myvert_offset,1,MPI_INT,true_vert_offsets,1,MPI_INT,comm);

   int ip = 0;
   true_vert_offsets.Append(nrpatch);
   for (int i = 0; i<num_procs; ++i)
   {
      while (ip < true_vert_offsets[i+1])
      {
         patch_rank[ip] = i;
         ip++;
      }
   }

   // now loop over all the elements and using the global dof of their vertices
   // the ids of the patches that they contribute to can be identified.
   int mynrelem = pmesh->GetNE();
   int myelem_offset;
   MPI_Scan(&mynrelem, &myelem_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   myelem_offset -= mynrelem;
   for (int iel=0; iel<mynrelem; ++iel)
   {
      Array<int> vertices;
      pmesh->GetElementVertices(iel,vertices);
      int nrvert = vertices.Size();
      for (int iv=0; iv<nrvert; ++iv) { vertices[iv] += mytdofoffset; }
   }
   // now construct the element contribution information
   local_element_map.resize(nrpatch);
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
         int truevert = aux_fespace->GetGlobalTDofNumber(vert);
         // natural ordering of this patch
         local_element_map[truevert].Append(iel+myelem_offset);
      }
   }
   delete aux_fespace;
   delete aux_fec;
}

ParMeshPartition::ParMeshPartition(ParMesh *pmesh_, int part) : pmesh(pmesh_)
{
   int num_procs, myid;
   comm  = pmesh->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   int dim = pmesh->Dimension();

   if (part)
   {
      CartesianParMeshPartition partition(pmesh);
      local_element_map = partition.local_element_map;
      patch_rank = partition.patch_rank;
   }
   else
   {
      VertexParMeshPartition partition(pmesh);
      local_element_map = partition.local_element_map;
      patch_rank = partition.patch_rank;
   }

   nrpatch = local_element_map.size();
   int mynrelem = pmesh->GetNE();
   MPI_Scan(&mynrelem, &myelem_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   myelem_offset -= mynrelem;

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
         if (count[k] != 0)
         {
            num_ranks++;
         }
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
      if (new_comm != MPI_COMM_NULL) { MPI_Comm_free(&new_comm); }
   }

   // Now each process will send the vertex coords and elements to the patch host rank
   Array<int> send_count(num_procs);
   Array<int> send_displ(num_procs);
   Array<int> recv_count(num_procs);
   Array<int> recv_displ(num_procs);
   send_count = 0;
   send_displ = 0;
   recv_count = 0;
   recv_displ = 0;

   // send buffer for coordinates
   Array<int> send_count_d(num_procs);
   Array<int> send_displ_d(num_procs);
   Array<int> recv_count_d(num_procs);
   Array<int> recv_displ_d(num_procs);
   send_count_d = 0;
   send_displ_d = 0;
   recv_count_d = 0;
   recv_displ_d = 0;


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

   // communicate so that recv_count is constructed
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
   Array<int> sendbuf(sbuff_size);
   sendbuf = 0;
   Array<int> soffs(num_procs);
   soffs = 0;

   Array<double> sendbuf_d(sbuff_size_d);
   sendbuf_d = 0.0;
   Array<int> soffs_d(num_procs);
   soffs_d = 0;

   // now the data will be placed according to process offsets
   FiniteElementCollection * aux_fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace * aux_fespace = new ParFiniteElementSpace(pmesh, aux_fec);
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
            int j = send_displ[patch_rank[ip]] + soffs[patch_rank[ip]];
            int j_d = send_displ_d[patch_rank[ip]] + soffs_d[patch_rank[ip]];
            sendbuf[j] = ip;
            sendbuf[j+1] = iel_idx + myelem_offset;
            sendbuf[j+2] = pmesh->GetElementType(iel_idx);
            sendbuf[j+3] = nrvert;
            for (int iv = 0; iv<nrvert; ++iv)
            {
               sendbuf[j+4+iv] = aux_fespace->GetGlobalTDofNumber(elem_vertices[iv]);
               for (int comp=0; comp<dim; ++comp)
               {
                  sendbuf_d[j_d+iv+comp] = pmesh->GetVertex(elem_vertices[iv])[comp];
               }
               j_d += dim-1;
            }
            soffs[patch_rank[ip]] += 1 + 1 + 1 + 1 + nrvert;
            soffs_d[patch_rank[ip]] += dim * nrvert;
         }
      }
   }

   delete aux_fespace;
   delete aux_fec;

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
      int ip = recvbuf[k];
      k++;
      patch_elements[ip].Append(recvbuf[k]);
      k++;
      patch_elements_type[ip].Append(recvbuf[k]);
      k++;
      int nrvert = recvbuf[k];
      k++;
      int id = 0;
      for (int iv = 0; iv < nrvert; ++iv)
      {
         patch_vertices[ip].Append(recvbuf[k+iv]);
         patch_vertex_xcoord[ip].Append(recvbuf_d[kd+iv+id]);
         patch_vertex_ycoord[ip].Append(recvbuf_d[kd+iv+1+id]);
         if (dim == 3) { patch_vertex_zcoord[ip].Append(recvbuf_d[kd+iv+2+id]); }
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
               if (dim == 3) { vert[2] = patch_vertex_zcoord[ip][iv]; }
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
   // SaveMeshPartition();
}

void ParMeshPartition::AddElementToMesh(Mesh * mesh,
                                        mfem::Element::Type elem_type,int * ind)
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

void ParMeshPartition::GetNumVertices(int type, mfem::Element::Type & elem_type,
                                      int & nrvert)
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


void ParMeshPartition::SaveMeshPartition()
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
      delete patch_mesh[ip];
      patch_mesh[ip] = nullptr;
   }
   patch_mesh.DeleteAll();
}


ParPatchDofInfo::ParPatchDofInfo(ParFiniteElementSpace *fespace, int part)
{
   MPI_Comm comm = fespace->GetComm();
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(comm, &myid);
   ParMesh * pmesh = fespace->GetParMesh();
   p = new ParMeshPartition(pmesh, part);
   nrpatch = p->nrpatch;
   int myelemoffset = p->myelem_offset;
   patch_rank = p->patch_rank;

   Array<int> send_count(num_procs);
   Array<int> send_displ(num_procs);
   Array<int> recv_count(num_procs);
   Array<int> recv_displ(num_procs);
   send_count = 0;
   send_displ = 0;
   recv_count = 0;
   recv_displ = 0;

   // each element contributing to the patch has to communicate to the patch_rank the list
   // of its tdof numbers whether it owns them or not
   // there is no problem with dublicates since they will be overwritten
   // After these lists are constructed to the host rank then they will be brodcasted to
   // the participating ranks

   // calculate the sent_count for each patch
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      int nrelems = p->local_element_map[ip].Size();
      if (nrelems >0 )
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
   Array<int> sendbuf(sbuff_size);
   sendbuf = 0;
   Array<int> soffs(num_procs);
   soffs = 0;

   // fill up the send_buffer
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      int nrelems = p->local_element_map[ip].Size();

      if (nrelems > 0)
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
      int ip = recvbuf[k];
      k++;
      int nrdofs = recvbuf[k];
      k++;
      for (int idof = 0; idof < nrdofs; ++idof)
      {
         patch_true_dofs[ip].Append(recvbuf[k+idof]);
      }
      k += nrdofs;
   }


   // build the maps from patch true dof to global truedof
   patch_fespaces.SetSize(nrpatch);
   patch_dof_map.resize(nrpatch);
   const FiniteElementCollection * fec = fespace->FEColl();
   for (int ip=0; ip<nrpatch; ++ip)
   {
      patch_fespaces[ip] = nullptr;
      if (p->patch_mesh[ip])
      {
         // patch_true_dofs[ip].Print(cout, 20);
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
}


ParPatchDofInfo::~ParPatchDofInfo()
{
   delete p;
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      delete patch_fespaces[ip];
      patch_fespaces[ip] = nullptr;
   }
   patch_fespaces.DeleteAll();
}



// constructor
ParPatchAssembly::ParPatchAssembly(ParBilinearForm * bf_,int part) : bf(bf_)
{
   fespace = bf->ParFESpace();
   comm = fespace->GetComm();
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   compute_trueoffsets();
   ParPatchDofInfo * patch_dofs = new ParPatchDofInfo(fespace, part);

   nrpatch = patch_dofs->nrpatch;
   patch_rank = patch_dofs->patch_rank;
   // share the dof map with the contributing ranks
   Array<int> send_count(num_procs);
   Array<int> send_displ(num_procs);
   Array<int> recv_count(num_procs);
   Array<int> recv_displ(num_procs);
   send_count = 0;
   send_displ = 0;
   recv_count = 0;
   recv_displ = 0;

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

   // communicate so that recv_count is constructed
   MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();
   // now allocate space for the send buffer
   Array<double> sendbuf(sbuff_size);
   sendbuf = 0;
   Array<int> soffs(num_procs);
   soffs = 0;

   for (int ip = 0; ip < nrpatch; ++ip)
   {
      int nrdofs = patch_dofs->patch_dof_map[ip].Size();
      if (nrdofs > 0)
      {
         // patch_dofs->patch_dof_map[ip].Print(cout,20);
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

   // recvbuf.Print(cout,10);
   int k=0;
   while (k<rbuff_size)
   {
      int ip = recvbuf[k];
      k++;
      int nrdofs = recvbuf[k];
      k++;
      for (int idof = 0; idof < nrdofs; ++idof)
      {
         int tdof = recvbuf[k+idof];
         patch_true_dofs[ip].Append(tdof);
         if (get_rank(tdof) == myid)
         {
            patch_local_dofs[ip].Append(tdof);
         }
      }
      k += nrdofs;
   }
   AssemblePatchMatrices(patch_dofs);
   delete patch_dofs;
}

void ParPatchAssembly::AssemblePatchMatrices(ParPatchDofInfo * p)
{
   patch_mat.SetSize(nrpatch);
   patch_bilinear_forms.SetSize(nrpatch);
   patch_mat_inv.SetSize(nrpatch);
   ess_tdof_list.resize(nrpatch);
   for (int ip=0; ip<nrpatch; ++ip)
   {
      patch_bilinear_forms[ip] = nullptr;
      patch_mat_inv[ip] = nullptr;
      patch_mat[ip] = nullptr;
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
         patch_bilinear_forms[ip] = new BilinearForm(patch_fespace, bf);
         patch_bilinear_forms[ip]->Assemble();
         OperatorPtr Alocal;
         patch_bilinear_forms[ip]->FormSystemMatrix(ess_tdof_list[ip],Alocal);
         patch_mat[ip] = new SparseMatrix((SparseMatrix&)(*Alocal));
         patch_mat[ip]->Threshold(0.0);
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
   if (size == 1)
   {
      return 0;
   }
   std::vector<int>::iterator up;
   up=std::upper_bound (tdof_offsets.begin(), tdof_offsets.end(),
                        tdof); //          ^
   return std::distance(tdof_offsets.begin(),up)-1;
}


ParPatchAssembly::~ParPatchAssembly()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      delete patch_bilinear_forms[ip];
      patch_bilinear_forms[ip] = nullptr;
      delete patch_mat_inv[ip];
      patch_mat_inv[ip] = nullptr;
   }
   patch_bilinear_forms.DeleteAll();
   patch_mat_inv.DeleteAll();
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

   send_count = 0;
   send_displ = 0;
   recv_count = 0;
   recv_displ = 0;

   // Precompute send_counts
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int ndofs = P->patch_local_dofs[ip].Size();
      for (int i =0; i<ndofs; i++)
      {
         int tdof = P->patch_local_dofs[ip][i];
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank && tdof_rank != patch_rank[ip])
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

void ParPatchRestriction::Mult(const Vector & r , std::vector<Vector> & res)
{
   int mytdofoffset = P->fespace->GetMyTDofOffset();
   // now allocate space for the send buffer
   Array<double> sendbuf(sbuff_size);
   sendbuf = 0;
   Array<int> soffs(num_procs);
   soffs = 0;
   // the data are placed according to process offsets
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int ndofs = P->patch_local_dofs[ip].Size();
      for (int i = 0; i<ndofs; i++)
      {
         int tdof = P->patch_local_dofs[ip][i];
         // find its rank
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank && tdof_rank != patch_rank[ip])
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
   // Now each process will construct the res vector
   res.resize(nrpatch);
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (myid == patch_rank[ip])
      {
         int ndof = P->patch_true_dofs[ip].Size();
         res[ip].SetSize(ndof);
         // extract the data from receiv buffer
         for (int i=0; i<ndof; i++)
         {
            // pick up the tdof and find its rank
            int tdof = P->patch_true_dofs[ip][i];
            int tdof_rank= P->get_rank(tdof);
            if (tdof_rank != patch_rank[ip])
            {
               int k = recv_displ[tdof_rank] + roffs[tdof_rank];
               roffs[tdof_rank]++;
               res[ip][i] = recvbuf[k];
            }
            else
            {
               int kk = tdof - mytdofoffset;
               res[ip][i] = r[kk];
            }
         }
      }
   }
}

void ParPatchRestriction::MultTranspose(const std::vector<Vector > & sol,
                                        Vector & z)
{
   int mytdofoffset = P->fespace->GetMyTDofOffset();
   // Step 3: Propagate the information to the global solution vector
   // (the recv_buff becomes the sendbuff and vice-versa)
   Array<double> sendbuf(sbuff_size);
   sendbuf = 0.0;
   Array<double> recvbuf(rbuff_size);
   recvbuf = 0.0;
   Array<int> roffs(num_procs);
   roffs = 0;
   Array<int> soffs(num_procs);
   soffs = 0;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (myid == patch_rank[ip])
      {
         int ndofs = P->patch_true_dofs[ip].Size();
         // loop through dofs
         for (int i=0; i<ndofs; i++)
         {
            //  pick up the dof and find its tdof_rank
            int tdof = P->patch_true_dofs[ip][i];
            int tdof_rank= P->get_rank(tdof);
            // offset
            if (tdof_rank != patch_rank[ip])
            {
               int k = recv_displ[tdof_rank] + roffs[tdof_rank];
               roffs[tdof_rank]++;
               recvbuf[k] = sol[ip][i];
            }
         }
      }
   }
   // now communication

   MPI_Alltoallv(recvbuf, recv_count, recv_displ, MPI_DOUBLE, sendbuf,
                 send_count, send_displ, MPI_DOUBLE, comm);

   // 1. Accummulate for the solution
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int ndofs = P->patch_true_dofs[ip].Size();
      for (int i = 0; i<ndofs; i++)
      {
         int tdof = P->patch_true_dofs[ip][i];
         // find its rank
         int k = tdof - mytdofoffset;
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            if (tdof_rank != patch_rank[ip])
            {
               int j = send_displ[patch_rank[ip]] + soffs[patch_rank[ip]];
               soffs[patch_rank[ip]]++;
               z[k] += sendbuf[j];
            }
            else
            {
               z[k] += sol[ip][i];
            }
         }
      }
   }
}


ParAddSchwarz::ParAddSchwarz(ParBilinearForm * bf_, int i)
   : Solver(bf_->ParFESpace()->GetTrueVSize(), bf_->ParFESpace()->GetTrueVSize()),
     part(i)
{
   cout << "part = " << part << endl;
   comm = bf_->ParFESpace()->GetComm();
   p = new ParPatchAssembly(bf_, part);

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
      std::vector<Vector > res;
      R->Mult(rnew,res);

      std::vector<Vector > sol(nrpatch);
      for (int ip=0; ip<nrpatch; ip++)
      {
         if (myid == p->patch_rank[ip])
         {
            sol[ip].SetSize(res[ip].Size());
            p->patch_mat_inv[ip]->Mult(res[ip], sol[ip]);

            // zero out the essential truedofs
            Array<int> ess_bdr_indices = p->ess_tdof_list[ip];
            if (!part) { sol[ip].SetSubVector(ess_bdr_indices,0.0); }
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

ParAddSchwarz::~ParAddSchwarz()
{
   delete p;
   delete R;
}


