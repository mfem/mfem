#include "parproblems_util.hpp"

void FindPointsInMesh(Mesh & mesh, const Array<int> & gvert, const Vector & xyz, const Array<int> & s_conn, Array<int>& conn,
                      Vector & xyz2, Array<int> & s_conn2, Vector& xi, DenseMatrix & coords)
{
   const int dim = mesh.Dimension();
   const int np = xyz.Size() / dim;

   MFEM_VERIFY(np * dim == xyz.Size(), "");

   mesh.EnsureNodes();

   FindPointsGSLIB finder(MPI_COMM_WORLD);

   finder.SetDistanceToleranceForPointsFoundOnBoundary(0.5);

   const double bb_t = 0.5;
   finder.Setup(mesh, bb_t);

   finder.FindPoints(xyz,mfem::Ordering::byVDIM);

   Array<unsigned int> procs = finder.GetProc();

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   Array<unsigned int> codes = finder.GetCode();

   /// Return element number for each point found by FindPoints.
   Array<unsigned int> elems = finder.GetElem();

   /// Return reference coordinates for each point found by FindPoints.
   Vector refcrd = finder.GetReferencePosition();

   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   Vector dist = finder.GetDist();

   finder.FreeData();

   MFEM_VERIFY(dist.Size() == np, "");
   MFEM_VERIFY(refcrd.Size() == np * dim, "");
   MFEM_VERIFY(elems.Size() == np, "");
   MFEM_VERIFY(codes.Size() == np, "");

   bool allfound = true;
   for (auto code : codes)
      if (code == 2) { allfound = false; }

   MFEM_VERIFY(allfound, "A point was not found");

   // cout << "Maximum distance of projected points: " << dist.Max() << endl;


   Array<unsigned int> elems_recv, proc_recv;
   Vector ref_recv;
   Vector xyz_recv;
   Array<int> s_conn_recv;

   MPICommunicator mycomm(MPI_COMM_WORLD, procs);
   mycomm.Communicate(xyz,xyz_recv,3,mfem::Ordering::byNODES);
   mycomm.Communicate(elems,elems_recv,1,mfem::Ordering::byVDIM);
   mycomm.Communicate(refcrd,ref_recv,3,mfem::Ordering::byVDIM);
   mycomm.Communicate(s_conn,s_conn_recv,1,mfem::Ordering::byVDIM);

   proc_recv = mycomm.GetOriginProcs();

   int np_loc = elems_recv.Size();
   Array<int> conn_loc(np_loc*4);
   Vector xi_send(np_loc*(dim-1));
   for (int i=0; i<np_loc; ++i)
   {
      int refFace, refNormal;
      // int refNormalSide;
      bool is_interior = -1;

      Vector normal = GetNormalVector(mesh, elems_recv[i],
                                      ref_recv.GetData() + (i*dim),
                                      refFace, refNormal, is_interior);

      // continue;
      int phyFace;
      if (is_interior)
      {
         phyFace = -1; // the id of the face that has the closest point
         FindSurfaceToProject(mesh, elems_recv[i], phyFace); // seems that this works

         Array<int> cbdrVert;
         mesh.GetFaceVertices(phyFace, cbdrVert);
         Vector xs(dim);
         xs[0] = xyz_recv[i + 0*np_loc];
         xs[1] = xyz_recv[i + 1*np_loc];
         xs[2] = xyz_recv[i + 2*np_loc];

         Vector xi_tmp(dim-1);
         // get nodes!

         GridFunction *nodes = mesh.GetNodes();
         DenseMatrix coord(4,3);
         for (int j=0; j<4; j++)
         {
            for (int k=0; k<3; k++)
            {
               coord(j,k) = (*nodes)[cbdrVert[j]*3+k];
            }
         }
         SlaveToMaster(coord, xs, xi_tmp);

         for (int j=0; j<dim-1; ++j)
         {
            xi_send[i*(dim-1)+j] = xi_tmp[j];
         }
         // now get get the projection to the surface
      }
      else
      {
         Vector faceRefCrd(dim-1);
         {
            int fd = 0;
            for (int j=0; j<dim; ++j)
            {
               if (j == refNormal)
               {
                  // refNormalSide = (ref_recv[(i*dim) + j] > 0.5); // not used
               }
               else
               {
                  faceRefCrd[fd] = ref_recv[(i*dim) + j];
                  fd++;
               }
            }
            MFEM_VERIFY(fd == dim-1, "");
         }

         for (int j=0; j<dim-1; ++j)
         {
            xi_send[i*(dim-1)+j] = faceRefCrd[j]*2.0 - 1.0;
         }
      }
      // Get the element face
      Array<int> faces;
      Array<int> ori;
      int face;

      if (is_interior)
      {
         face = phyFace;
      }
      else
      {
         mesh.GetElementFaces(elems_recv[i], faces, ori);
         face = faces[refFace];
      }

      Array<int> faceVert;
      mesh.GetFaceVertices(face, faceVert);

      for (int p=0; p<4; p++)
      {
         conn_loc[4*i+p] = faceVert[p];
      }
   }

   if (0) // for debugging
   {
      int sz = xi_send.Size()/2;

      for (int i = 0; i<sz; i++)
      {
         mfem::out << "("<<xi_send[i*(dim-1)]<<","<<xi_send[i*(dim-1)+1]<<"): -> ";
         for (int j = 0; j<4; j++)
         {
            double * vc = mesh.GetVertex(conn_loc[4*i+j]);
            if (j<3)
            {
               mfem::out << "("<<vc[0]<<","<<vc[1]<<","<<vc[2]<<"), ";
            }
            else
            {
               mfem::out << "("<<vc[0]<<","<<vc[1]<<","<<vc[2]<<") \n " << endl;
            }
         }
      }
   }

   int sz = xi_send.Size()/2;
   DenseMatrix coordsm(sz*4, dim);
   for (int i = 0; i<sz; i++)
   {
      for (int j = 0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            coordsm(i*4+j,k) = mesh.GetVertex(conn_loc[i*4+j])[k];
         }
      }
   }

   // pass global indices for conn_loc
   for (int i = 0; i<conn_loc.Size(); i++)
   {
      conn_loc[i] = gvert[conn_loc[i]];
   }

   mycomm.UpdateDestinationProcs();
   mycomm.Communicate(xyz_recv,xyz2,3,mfem::Ordering::byNODES);
   mycomm.Communicate(xi_send,xi,2,mfem::Ordering::byVDIM);
   mycomm.Communicate(s_conn_recv,s_conn2,1,mfem::Ordering::byVDIM);
   mycomm.Communicate(conn_loc,conn,4,mfem::Ordering::byVDIM);
   mycomm.Communicate(coordsm,coords,4,mfem::Ordering::byVDIM);
}


void FindPointsInMesh(ParMesh & pmesh, const Array<int> & gvert, Array<int> & s_conn, const Vector &x1, Vector & xyz, Array<int>& conn,
                      Vector& xi, DenseMatrix & coords, bool singlemesh)
{
   const int dim = pmesh.Dimension();
   const int np = xyz.Size() / dim;
   MFEM_VERIFY(np * dim == xyz.Size(), "");

   pmesh.EnsureNodes();

   ParSubMesh * psubmesh = nullptr;
   ParMesh * mesh = nullptr;
   Array<int> elem_map;
   if (singlemesh)
   {
      Array<int> attr; attr.Append(1);
      psubmesh = new ParSubMesh(ParSubMesh::CreateFromDomain(pmesh,attr));
      mesh = (ParMesh *)psubmesh;
   }
   else
   {
      mesh = &pmesh;
   }

   FindPointsGSLIB finder(MPI_COMM_WORLD);

   finder.SetDistanceToleranceForPointsFoundOnBoundary(0.5);

   const double bb_t = 0.5;
   MFEM_VERIFY(mesh->GetNE(), "FindPointsGSLIB does not support empty partition");
   finder.Setup(*mesh, bb_t);

   finder.FindPoints(xyz,mfem::Ordering::byVDIM);

   Array<unsigned int> procs = finder.GetProc();

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   Array<unsigned int> codes = finder.GetCode();

   /// Return element number for each point found by FindPoints.
   Array<unsigned int> elems = finder.GetElem();

   /// Return reference coordinates for each point found by FindPoints.
   Vector refcrd = finder.GetReferencePosition();

   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   Vector dist = finder.GetDist();

   finder.FreeData();

   MFEM_VERIFY(dist.Size() == np, "");
   MFEM_VERIFY(refcrd.Size() == np * dim, "");
   MFEM_VERIFY(elems.Size() == np, "");
   MFEM_VERIFY(codes.Size() == np, "");

   bool allfound = true;
   for (auto code : codes)
      if (code == 2) { allfound = false; }

   MFEM_VERIFY(allfound, "A point was not found");

   // reorder data so that the procs are in ascending order
   // sort procs and save the permutation
   std::vector<unsigned int> procs_index(np);
   std::iota(procs_index.begin(),procs_index.end(),0); //Initializing
   sort( procs_index.begin(),procs_index.end(), [&](int i,int j){return procs[i]<procs[j];} );

   // map to sorted
   Array<unsigned int> procs_sorted(np);
   Array<unsigned int> elems_sorted(np);
   Vector xyz_sorted(np*dim);
   Vector refcrd_sorted(np*dim);
   Array<int> s_conn_sorted(np);
   for (int i = 0; i<np; i++)
   {
      int j = procs_index[i];
      procs_sorted[i] = procs[j];
      elems_sorted[i] = elems[j];
      s_conn_sorted[i] = s_conn[j];
      for (int d = 0; d<dim; d++)
      {
         xyz_sorted(i*dim+d) = xyz(j*dim+d);
         refcrd_sorted(i*dim+d) = refcrd(j*dim+d);
      }
   }

   Array<unsigned int> elems_recv, proc_recv;
   xyz = xyz_sorted;
   s_conn = s_conn_sorted;
   Vector ref_recv;
   Vector xyz_recv;

   MPICommunicator mycomm(MPI_COMM_WORLD, procs_sorted);
   mycomm.Communicate(xyz_sorted,xyz_recv,3,mfem::Ordering::byVDIM);
   mycomm.Communicate(elems_sorted,elems_recv,1,mfem::Ordering::byVDIM);
   mycomm.Communicate(refcrd_sorted,ref_recv,3,mfem::Ordering::byVDIM);

   proc_recv = mycomm.GetOriginProcs();
   int np_loc = elems_recv.Size();
   if (singlemesh)
   {
      elem_map = psubmesh->GetParentElementIDMap();
      for (int i = 0; i< np_loc; i++)
      {
         elems_recv[i] = elem_map[elems_recv[i]];
      }
      delete mesh;
   }


   Array<int> conn_loc(np_loc*4);
   Vector xi_send(np_loc*(dim-1));
   for (int i=0; i<np_loc; ++i)
   {
      int refFace, refNormal; 
      // int refNormalSide;
      bool is_interior = -1;
      Vector normal = GetNormalVector(pmesh, elems_recv[i],
                                      ref_recv.GetData() + (i*dim),
                                      refFace, refNormal, is_interior);

      // continue;
      int phyFace;
      if (is_interior)
      {
         phyFace = -1; // the id of the face that has the closest point
         FindSurfaceToProject(pmesh, elems_recv[i], phyFace); // seems that this works

         Array<int> cbdrVert;
         pmesh.GetFaceVertices(phyFace, cbdrVert);
         Vector xs(dim);
         xs[0] = xyz_recv[i*dim + 0];
         xs[1] = xyz_recv[i*dim + 1];
         xs[2] = xyz_recv[i*dim + 2];

         Vector xi_tmp(dim-1);
         // get nodes!

         GridFunction *nodes = pmesh.GetNodes();
         DenseMatrix coord(4,3);
         for (int j=0; j<4; j++)
         {
            for (int k=0; k<3; k++)
            {
               coord(j,k) = (*nodes)[cbdrVert[j]*3+k];
            }
         }
         SlaveToMaster(coord, xs, xi_tmp);

         for (int j=0; j<dim-1; ++j)
         {
            xi_send[i*(dim-1)+j] = xi_tmp[j];
         }
         // now get the projection to the surface
      }
      else
      {
         Vector faceRefCrd(dim-1);
         {
            int fd = 0;
            for (int j=0; j<dim; ++j)
            {
               if (j == refNormal)
               {
                  // refNormalSide = (ref_recv[(i*dim) + j] > 0.5); // not used
               }
               else
               {
                  faceRefCrd[fd] = ref_recv[(i*dim) + j];
                  fd++;
               }
            }
            MFEM_VERIFY(fd == dim-1, "");
         }

         for (int j=0; j<dim-1; ++j)
         {
            xi_send[i*(dim-1)+j] = faceRefCrd[j]*2.0 - 1.0;
         }
      }
      // Get the element face
      Array<int> faces;
      Array<int> ori;
      int face;

      if (is_interior)
      {
         face = phyFace;
      }
      else
      {
         pmesh.GetElementFaces(elems_recv[i], faces, ori);
         face = faces[refFace];
      }

      Array<int> faceVert;
      pmesh.GetFaceVertices(face, faceVert);

      for (int p=0; p<4; p++)
      {
         conn_loc[4*i+p] = faceVert[p];
      }
   }

   if (0) // for debugging
   {
      int sz = xi_send.Size()/2;

      for (int i = 0; i<sz; i++)
      {
         mfem::out << "("<<xi_send[i*(dim-1)]<<","<<xi_send[i*(dim-1)+1]<<"): -> ";
         for (int j = 0; j<4; j++)
         {
            double * vc = pmesh.GetVertex(conn_loc[4*i+j]);
            if (j<3)
            {
               mfem::out << "("<<vc[0]<<","<<vc[1]<<","<<vc[2]<<"), ";
            }
            else
            {
               mfem::out << "("<<vc[0]<<","<<vc[1]<<","<<vc[2]<<") \n " << endl;
            }
         }
      }
   }

   int sz = xi_send.Size()/2;
   DenseMatrix coordsm(sz*4, dim);
   for (int i = 0; i<sz; i++)
   {
      for (int j = 0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            coordsm(i*4+j,k) = pmesh.GetVertex(conn_loc[i*4+j])[k]+x1[dim*conn_loc[i*4+j]+k];
         }
      }
   }

   // pass global indices for conn_loc
   for (int i = 0; i<conn_loc.Size(); i++)
   {
      conn_loc[i] = gvert[conn_loc[i]];
   }

   mycomm.UpdateDestinationProcs();
   mycomm.Communicate(xi_send,xi,2,mfem::Ordering::byVDIM);
   mycomm.Communicate(conn_loc,conn,4,mfem::Ordering::byVDIM);
   mycomm.Communicate(coordsm,coords,4,mfem::Ordering::byVDIM);


}

int get_rank(int tdof, std::vector<int> & tdof_offsets)
{
   int size = tdof_offsets.size();
   if (size == 1) { return 0; }
   std::vector<int>::iterator up;
   up=std::upper_bound(tdof_offsets.begin(), tdof_offsets.end(),tdof); //
   return std::distance(tdof_offsets.begin(),up)-1;
}

void ComputeTdofOffsets(const ParFiniteElementSpace * pfes,
                        std::vector<int> & tdof_offsets)
{
   MPI_Comm comm = pfes->GetComm();
   int num_procs;
   MPI_Comm_size(comm, &num_procs);
   tdof_offsets.resize(num_procs);
   int mytoffset = pfes->GetMyTDofOffset();
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

void ComputeTdofOffsets(MPI_Comm comm, int mytoffset, std::vector<int> & tdof_offsets)
{
   int num_procs;
   MPI_Comm_size(comm,&num_procs);
   tdof_offsets.resize(num_procs);
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

void ComputeTdofs(MPI_Comm comm, int mytoffs, std::vector<int> & tdofs)
{
   int num_procs;
   MPI_Comm_size(comm,&num_procs);
   tdofs.resize(num_procs);
   MPI_Allgather(&mytoffs,1,MPI_INT,&tdofs,1,MPI_INT,comm);
}         


// Performs Pᵀ * A * P for BlockOperator  P (with blocks as HypreParMatrices)
// and A a HypreParMatrix, i.e., this handles the special case 
// where P = [P₁ P₂ ⋅⋅⋅ Pₙ] 
// C = Pᵀ * A * P 
void RAP(const HypreParMatrix & A, const BlockOperator & P, 
         BlockOperator & C)
{
   int nblocks = P.NumColBlocks();

   const HypreParMatrix * Pi = nullptr;
   const HypreParMatrix * Pj = nullptr;
   HypreParMatrix * PitAPj = nullptr; 

   for (int i = 0; i< nblocks; i++)
   {
      if (P.IsZeroBlock(0,i)) continue;
      Pi = dynamic_cast<const HypreParMatrix*>(&P.GetBlock(0,i));
      for (int j = 0; j<nblocks; j++)
      {
         if (P.IsZeroBlock(0,j)) continue;
         Pj = dynamic_cast<const HypreParMatrix*>(&P.GetBlock(0,j));
         if (i == j) 
         {
            PitAPj = RAP(&A, Pj); 
         }
         else
         {
            PitAPj = RAP(Pi, &A, Pj); 
         }
         C.SetBlock(i,j,PitAPj);
      }
   }
}

void ParAdd(const BlockOperator & A, const BlockOperator & B, BlockOperator & C)
{
   int n = A.NumRowBlocks();
   int m = A.NumColBlocks();
   MFEM_VERIFY(B.NumRowBlocks() == n, "Inconsistent number of row blocks");
   MFEM_VERIFY(B.NumColBlocks() == m, "Inconsistent number of column blocks");

   const HypreParMatrix * a;
   const HypreParMatrix * b;
   for (int i = 0; i<n; i++)
   {
      for (int j = 0; j<m; j++)
      {
         a = nullptr;
         b = nullptr;
         if (!A.IsZeroBlock(i,j))
         {
            a = dynamic_cast<const HypreParMatrix*>(&A.GetBlock(i,j));
         }
         if (!B.IsZeroBlock(i,j))
         {
            b = dynamic_cast<const HypreParMatrix*>(&B.GetBlock(i,j));
         }
         if (a && b)
         {
            C.SetBlock(i,j,ParAdd(a,b));
         }
         else if (a)
         {
            C.SetBlock(i,j,new HypreParMatrix(*a));
         }
         else if (b)
         {
            C.SetBlock(i,j,new HypreParMatrix(*b));
         }
         else
         {
            // do nothing
         }
      }
   }
}
