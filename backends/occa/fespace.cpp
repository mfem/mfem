// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "backend.hpp"
#include "fespace.hpp"
#include "interpolation.hpp"

#ifdef MFEM_USE_MPI
#include <mpi-ext.h> // Check for cuda support
#endif

namespace mfem
{

namespace occa
{

FiniteElementSpace::FiniteElementSpace(const Engine &e,
                                       mfem::FiniteElementSpace &fespace)
   : PFiniteElementSpace(e, fespace),
     e_layout(new Layout(e, 0)) // resized in SetupLocalGlobalMaps()
{
   vdim     = fespace.GetVDim();
   ordering = fespace.GetOrdering();

   SetupLocalGlobalMaps();
   SetupOperators(); // calls virtual methods of 'fes'
   SetupKernels();
}

FiniteElementSpace::~FiniteElementSpace()
{
   delete restrictionOp;
   delete prolongationOp;
}

void FiniteElementSpace::SetupLocalGlobalMaps()
{
   const int elements = fes->GetNE();

   if (elements == 0) { return; }

   // Assuming of finite elements are the same.
   const mfem::FiniteElement &fe = *fes->GetFE(0);
   const mfem::TensorBasisElement *el =
      dynamic_cast<const mfem::TensorBasisElement*>(&fe);

   const mfem::Table &e2dTable = fes->GetElementToDofTable();
   const int *elementMap = e2dTable.GetJ();

   globalDofs = fes->GetNDofs();
   localDofs  = fe.GetDof();

   e_layout->OccaResize(e2dTable.Size_of_connections());

   int *elementDofMap = new int[localDofs];
   if (el)
   {
      ::memcpy(elementDofMap,
               el->GetDofMap().GetData(),
               localDofs * sizeof(int));
   }
   else
   {
      for (int i = 0; i < localDofs; ++i)
      {
         elementDofMap[i] = i;
      }
   }

   // Allocate device offsets and indices
   globalToLocalOffsets.allocate(GetDevice(),
                                 globalDofs + 1);
   globalToLocalIndices.allocate(GetDevice(),
                                 localDofs, elements);
   localToGlobalMap.allocate(GetDevice(),
                             localDofs, elements);

   int *offsets = globalToLocalOffsets.ptr();
   int *indices = globalToLocalIndices.ptr();
   int *l2gMap  = localToGlobalMap.ptr();

   // We'll be keeping a count of how many local nodes point
   //   to its global dof
   for (int i = 0; i <= globalDofs; ++i)
   {
      offsets[i] = 0;
   }

   for (int e = 0; e < elements; ++e)
   {
      MFEM_ASSERT(e2dTable.RowSize(e) == localDofs, "");
      for (int d = 0; d < localDofs; ++d)
      {
         const int gid = elementMap[localDofs*e + d];
         ++offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= globalDofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // For each global dof, fill in all local nodes that point
   //   to it
   for (int e = 0; e < elements; ++e)
   {
      for (int d = 0; d < localDofs; ++d)
      {
         const int gid = elementMap[localDofs*e + elementDofMap[d]];
         const int lid = localDofs*e + d;
         indices[offsets[gid]++] = lid;
         l2gMap[lid] = gid;
      }
   }
   // We shifted the offsets vector by 1 by using it
   //   as a counter. Now we shift it back.
   for (int i = globalDofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;

   delete [] elementDofMap;

   globalToLocalOffsets.keepInDevice();
   globalToLocalIndices.keepInDevice();
   localToGlobalMap.keepInDevice();
}

void FiniteElementSpace::SetupOperators() const
{
   // Construct 'restrictionOp' and 'prolongationOp'.

   prolongationOp = restrictionOp = NULL;

   const mfem::SparseMatrix *R = fes->GetRestrictionMatrix();
   const mfem::Operator *P = fes->GetProlongationMatrix();

   if (!P) { return; }

   Layout &v_layout = OccaVLayout();
   Layout &t_layout = OccaTrueVLayout();

   // Assuming R has one entry per row equal to 1.
   MFEM_ASSERT(R->Finalized(), "");
   const int tdofs = R->Height();
   MFEM_ASSERT(tdofs == (int)t_layout.Size(), "");
   MFEM_ASSERT(tdofs == R->GetI()[tdofs], "");
   ::occa::array<int> ltdof_ldof(GetDevice(), tdofs, R->GetJ());
   ltdof_ldof.keepInDevice();

   restrictionOp = new RestrictionOperator(v_layout, t_layout, ltdof_ldof);

   const mfem::SparseMatrix *pmat = dynamic_cast<const mfem::SparseMatrix*>(P);
   if (pmat)
   {
      const mfem::SparseMatrix *pmatT = Transpose(*pmat);

      OccaSparseMatrix *occaP  =
         CreateMappedSparseMatrix(t_layout, v_layout, *pmat);
      OccaSparseMatrix *occaPT =
         CreateMappedSparseMatrix(v_layout, t_layout, *pmatT);

      prolongationOp = new ProlongationOperator(*occaP, *occaPT);

      delete occaPT;
      delete occaP;
   }
#ifdef MFEM_USE_MPI
   else if (fes->Conforming() && dynamic_cast<ParFiniteElementSpace*>(fes))
   {
      ParFiniteElementSpace *pfes = static_cast<ParFiniteElementSpace*>(fes);
      prolongationOp = new OccaConformingProlongation(*this, *pfes,
                                                      ltdof_ldof.memory());
   }
#endif
   else
   {
      prolongationOp = new ProlongationOperator(t_layout, v_layout, P);
   }
}

void FiniteElementSpace::SetupKernels()
{
   ::occa::properties props("defines: {"
                            "  TILESIZE: 256,"
                            "}");
   props["defines/NUM_VDIM"] = vdim;

   props["defines/ORDERING_BY_NODES"] = 0;
   props["defines/ORDERING_BY_VDIM"]  = 1;
   props["defines/VDIM_ORDERING"] = (int) (ordering == Ordering::byVDIM);

   ::occa::device device = GetDevice();
   const std::string &okl_path = OccaEngine().GetOklPath();
   globalToLocalKernel = device.buildKernel(okl_path + "fespace.okl",
                                            "GlobalToLocal",
                                            props);
   localToGlobalKernel = device.buildKernel(okl_path + "fespace.okl",
                                            "LocalToGlobal",
                                            props);
}


#ifdef MFEM_USE_MPI

OccaConformingProlongation::OccaConformingProlongation(
   const FiniteElementSpace &ofes, const mfem::ParFiniteElementSpace &pfes,
   ::occa::memory ltdof_ldof_)

   : Operator(ofes.OccaTrueVLayout(), ofes.OccaVLayout()),
     shr_ltdof(ofes.OccaEngine()),
     ext_ldof(ofes.OccaEngine()),
     shr_buf(shr_ltdof.OccaLayout(), sizeof(double)),
     ext_buf(ext_ldof.OccaLayout(), sizeof(double)),
     shr_buf_offsets(NULL), ext_buf_offsets(NULL),
     ltdof_ldof(ltdof_ldof_),
     gc(pfes.GroupComm())
{
   MFEM_ASSERT(pfes.Conforming(), "internal error");

   const Engine &engine = ofes.OccaEngine();
   const std::string &okl_path = engine.GetOklPath();
   ::occa::device device = engine.GetDevice();

   {
      Table nbr_ltdof;
      gc.GetNeighborLTDofTable(nbr_ltdof);
      shr_ltdof.OccaResize(nbr_ltdof.Size_of_connections(), sizeof(int));
      shr_ltdof.OccaPush(nbr_ltdof.GetJ());
      shr_buf.OccaResize(&shr_ltdof.OccaLayout(), sizeof(double));
      shr_buf_offsets = nbr_ltdof.GetI();
      {
         mfem::Array<int> shr_ltdof(nbr_ltdof.GetJ(),
                                    nbr_ltdof.Size_of_connections());
         mfem::Array<int> unique_ltdof(shr_ltdof);
         unique_ltdof.Sort();
         unique_ltdof.Unique();
         // Note: the next loop modifies the J array of nbr_ltdof
         for (int i = 0; i < shr_ltdof.Size(); i++)
         {
            shr_ltdof[i] = unique_ltdof.FindSorted(shr_ltdof[i]);
            MFEM_ASSERT(shr_ltdof[i] != -1, "internal error");
         }
         Table unique_shr;
         Transpose(shr_ltdof, unique_shr, unique_ltdof.Size());

         unq_ltdof = device.malloc(unique_ltdof.Size()*sizeof(int),
                                   unique_ltdof.GetData());
         unq_shr_i = device.malloc((unique_shr.Size()+1)*sizeof(int),
                                   unique_shr.GetI());
         unq_shr_j = device.malloc(unique_shr.Size_of_connections()*sizeof(int),
                                   unique_shr.GetJ());
      }
      delete [] nbr_ltdof.GetJ();
      nbr_ltdof.LoseData();
   }
   {
      Table nbr_ldof;
      gc.GetNeighborLDofTable(nbr_ldof);
      ext_ldof.OccaResize(nbr_ldof.Size_of_connections(), sizeof(int));
      ext_ldof.OccaPush(nbr_ldof.GetJ());
      ext_buf.OccaResize(&ext_ldof.OccaLayout(), sizeof(double));
      ext_buf_offsets = nbr_ldof.GetI();
      delete [] nbr_ldof.GetJ();
      nbr_ldof.LoseData();
   }
   host_shr_buf = NULL;
   host_ext_buf = NULL;
   // If the device has a separate memory space (e.g. CUDA device) and the MPI
   // library does not support buffers in that separate memory space, we
   // allocate separate host buffers to use for MPI communication.
   if (device.hasSeparateMemorySpace())
   {
      bool need_host_buf = true;
      if (device.mode() == "CUDA")
      {
#ifdef MPIX_CUDA_AWARE_SUPPORT
         need_host_buf = !MPIX_Query_cuda_support();
#endif
         if (engine.GetForceCudaAwareMPI()) { need_host_buf = false; }
         if (gc.GetGroupTopology().MyRank() == 0)
         {
            mfem::out << "\nOccaConformingProlongation: CUDA-aware MPI: "
                      << (need_host_buf ? "NO" : "YES") << "\n\n";
         }
      }
      if (need_host_buf)
      {
         host_shr_buf = new char[shr_buf.OccaMem().size()];
         host_ext_buf = new char[ext_buf.OccaMem().size()];
      }
   }

   ExtractSubVector = device.buildKernel(okl_path + "mappings.okl",
                                         "ExtractSubVector",
                                         "defines: { TILESIZE: 256 }");
   SetSubVector = device.buildKernel(okl_path + "mappings.okl",
                                     "SetSubVector",
                                     "defines: { TILESIZE: 256 }");
   AddSubVector = device.buildKernel(okl_path + "mappings.okl",
                                     "AddSubVector",
                                     "defines: { TILESIZE: 256 }");

   const GroupTopology &gtopo = gc.GetGroupTopology();
   int req_counter = 0;
   for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
   {
      const int send_offset = shr_buf_offsets[nbr];
      const int send_size = shr_buf_offsets[nbr+1] - send_offset;
      if (send_size > 0) { req_counter++; }

      const int recv_offset = ext_buf_offsets[nbr];
      const int recv_size = ext_buf_offsets[nbr+1] - recv_offset;
      if (recv_size > 0) { req_counter++; }
   }
   requests = new MPI_Request[req_counter];
}

OccaConformingProlongation::~OccaConformingProlongation()
{
   delete [] requests;
   delete [] host_ext_buf;
   delete [] host_shr_buf;
   delete [] ext_buf_offsets;
   delete [] shr_buf_offsets;
}

void OccaConformingProlongation::BcastBeginCopy(const ::occa::memory &src,
                                                std::size_t item_size) const
{
   // shr_buf[i] = src[shr_ltdof[i]]
   MFEM_ASSERT(item_size == sizeof(double), "");
   if (shr_ltdof.Size() == 0) { return; }
   ExtractSubVector((int)shr_ltdof.Size(), shr_ltdof.OccaMem(), src,
                    shr_buf.OccaMem());
   // If the above kernel is executed asynchronously, wait for it to complete:
   shr_buf.OccaMem().getDevice().finish();
   if (host_shr_buf)
   {
      shr_buf.OccaMem().copyTo(host_shr_buf);
   }
}

void OccaConformingProlongation::BcastLocalCopy(const ::occa::memory &src,
                                                ::occa::memory &dst,
                                                std::size_t item_size) const
{
   // dst[ltdof_ldof[i]] = src[i]
   MFEM_ASSERT(item_size == sizeof(double), "");
   if (ltdof_ldof.size<int>() == 0) { return; }
   SetSubVector((int)ltdof_ldof.size<int>(), ltdof_ldof, src, dst);
}

void OccaConformingProlongation::BcastEndCopy(::occa::memory &dst,
                                              std::size_t item_size) const
{
   // dst[ext_ldof[i]] = ext_buf[i]
   MFEM_ASSERT(item_size == sizeof(double), "");
   if (ext_ldof.Size() == 0) { return; }
   if (host_ext_buf)
   {
      ext_buf.OccaMem().copyFrom(host_ext_buf);
   }
   SetSubVector((int)ext_ldof.Size(), ext_ldof.OccaMem(),
                ext_buf.OccaMem(), dst);
}

void OccaConformingProlongation::ReduceBeginCopy(const ::occa::memory &src,
                                                 std::size_t item_size) const
{
   // ext_buf[i] = src[ext_ldof[i]]
   MFEM_ASSERT(item_size == sizeof(double), "");
   if (ext_ldof.Size() == 0) { return; }
   ExtractSubVector((int)ext_ldof.Size(), ext_ldof.OccaMem(), src,
                    ext_buf.OccaMem());
   // If the above kernel is executed asynchronously, wait for it to complete:
   ext_buf.OccaMem().getDevice().finish();
   if (host_ext_buf)
   {
      ext_buf.OccaMem().copyTo(host_ext_buf);
   }
}

void OccaConformingProlongation::ReduceLocalCopy(const ::occa::memory &src,
                                                 ::occa::memory &dst,
                                                 std::size_t item_size) const
{
   // dst[i] = src[ltdof_ldof[i]]
   MFEM_ASSERT(item_size == sizeof(double), "");
   if (ltdof_ldof.size<int>() == 0) { return; }
   ExtractSubVector((int)ltdof_ldof.size<int>(), ltdof_ldof, src, dst);
}

void OccaConformingProlongation::ReduceEndAssemble(::occa::memory &dst,
                                                   std::size_t item_size) const
{
   // dst[shr_ltdof[i]] += shr_buf[i]
   MFEM_ASSERT(item_size == sizeof(double), "");
   if (unq_ltdof.size<int>() == 0) { return; }
   if (host_shr_buf)
   {
      shr_buf.OccaMem().copyFrom(host_shr_buf);
   }
   AddSubVector((int)unq_ltdof.size<int>(), unq_ltdof, unq_shr_i, unq_shr_j,
                shr_buf.OccaMem(), dst);
}

void OccaConformingProlongation::Mult_(const Vector &x, Vector &y) const
{
   const GroupTopology &gtopo = gc.GetGroupTopology();

   BcastBeginCopy(x.OccaMem(), sizeof(double)); // copy to 'shr_buf'

   int req_counter = 0;
   for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
   {
      const int send_offset = shr_buf_offsets[nbr];
      const int send_size = shr_buf_offsets[nbr+1] - send_offset;
      if (send_size > 0)
      {
         void *send_buf;
         if (host_shr_buf)
         {
            send_buf = host_shr_buf + send_offset*sizeof(double);
         }
         else
         {
            send_buf = (shr_buf.OccaMem() + send_offset*sizeof(double)).ptr();
         }
         MPI_Isend(send_buf, send_size, MPI_DOUBLE, gtopo.GetNeighborRank(nbr),
                   41822, gtopo.GetComm(), &requests[req_counter++]);
      }

      const int recv_offset = ext_buf_offsets[nbr];
      const int recv_size = ext_buf_offsets[nbr+1] - recv_offset;
      if (recv_size > 0)
      {
         void *recv_buf;
         if (host_ext_buf)
         {
            recv_buf = host_ext_buf + recv_offset*sizeof(double);
         }
         else
         {
            recv_buf = (ext_buf.OccaMem() + recv_offset*sizeof(double)).ptr();
         }
         MPI_Irecv(recv_buf, recv_size, MPI_DOUBLE, gtopo.GetNeighborRank(nbr),
                   41822, gtopo.GetComm(), &requests[req_counter++]);
      }
   }

   BcastLocalCopy(x.OccaMem(), y.OccaMem(), sizeof(double));

   MPI_Waitall(req_counter, requests, MPI_STATUSES_IGNORE);

   BcastEndCopy(y.OccaMem(), sizeof(double)); // copy from 'ext_buf'
}

void OccaConformingProlongation::MultTranspose_(const Vector &x,
                                                Vector &y) const
{
   const GroupTopology &gtopo = gc.GetGroupTopology();

   ReduceBeginCopy(x.OccaMem(), sizeof(double)); // copy to 'ext_buf'

   int req_counter = 0;
   for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
   {
      const int send_offset = ext_buf_offsets[nbr];
      const int send_size = ext_buf_offsets[nbr+1] - send_offset;
      if (send_size > 0)
      {
         void *send_buf;
         if (host_ext_buf)
         {
            send_buf = host_ext_buf + send_offset*sizeof(double);
         }
         else
         {
            send_buf = (ext_buf.OccaMem() + send_offset*sizeof(double)).ptr();
         }
         MPI_Isend(send_buf, send_size, MPI_DOUBLE, gtopo.GetNeighborRank(nbr),
                   41823, gtopo.GetComm(), &requests[req_counter++]);
      }

      const int recv_offset = shr_buf_offsets[nbr];
      const int recv_size = shr_buf_offsets[nbr+1] - recv_offset;
      if (recv_size > 0)
      {
         void *recv_buf;
         if (host_shr_buf)
         {
            recv_buf = host_shr_buf + recv_offset*sizeof(double);
         }
         else
         {
            recv_buf = (shr_buf.OccaMem() + recv_offset*sizeof(double)).ptr();
         }
         MPI_Irecv(recv_buf, recv_size, MPI_DOUBLE, gtopo.GetNeighborRank(nbr),
                   41823, gtopo.GetComm(), &requests[req_counter++]);
      }
   }

   ReduceLocalCopy(x.OccaMem(), y.OccaMem(), sizeof(double));

   MPI_Waitall(req_counter, requests, MPI_STATUSES_IGNORE);

   ReduceEndAssemble(y.OccaMem(), sizeof(double)); // assemble from 'shr_buf'
}

#endif // MFEM_USE_MPI

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
