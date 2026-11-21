#include "interface_transfer.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace common_fsi_utils
{

void ReorderByVDIM(const Vector &v_in, Vector &v_out, int vdim)
{
   int npts = v_in.Size() / vdim;
   v_out.SetSize(vdim * npts);

   auto d_v_in = v_in.Read();
   auto d_v_out = v_out.Write();

   mfem::forall(npts, [=] MFEM_HOST_DEVICE(int i)
   {
      for (int d = 0; d < vdim; d++)
      {
         d_v_out[vdim * i + d] = d_v_in[d * npts + i];
      }
   });
}


/// INTERFACE TRANSFER CLASS
InterfaceTransfer::InterfaceTransfer(ParFiniteElementSpace *src_fes_,
                                     ParFiniteElementSpace *dst_fes_,
                                     Array<int> &bdr_attributes_, Backend backend_, MPI_Comm comm_,
                                     bool use_undeformed_coords_)
   : src_fes(src_fes_), dst_fes(dst_fes_), bdr_attributes(bdr_attributes_),
     backend(backend_),
     src_mesh(src_fes_->GetParMesh()), dst_mesh(dst_fes_->GetParMesh()),
     sdim(src_mesh->SpaceDimension()), comm(comm_),
     use_undeformed_coords(use_undeformed_coords_)
{
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nranks);
   recvcounts_gather.SetSize(nranks);
   recvcounts_scatter.SetSize(nranks);
   recvcounts.SetSize(nranks);
   displs_gather.SetSize(nranks);
   displs_scatter.SetSize(nranks);

   /// Store indices of boundary elements on both source and destination meshes
   // Attributes outside the marker (e.g. the parent_max+1 attribute ParSubMesh gives to new boundary faces)
   // are not part of the interface.
   auto marked = [&](int attr) { return attr >= 1 && attr <= bdr_attributes.Size() && bdr_attributes[attr - 1] != 0; };

   /// 1. Find boundary elements with the specified attributes (on the destination mesh) --> LOCAL
   for (int be = 0; be < dst_mesh->GetNBE(); be++)
   {
      if (marked(dst_mesh->GetBdrAttribute(be))) { dst_bdr_element_idx.Append(be); }
   }

   // For convenience, we will also store the boundary elements for the source mesh (to test the alternative Interpolation)
   for (int be = 0; be <  src_mesh->GetNBE(); be++)
   {
      if (marked(src_mesh->GetBdrAttribute(be))) { src_bdr_element_idx.Append(be); }
   }

   /// Setup the transfer (depending on the backend)
   if (backend == Backend::Native || backend == Backend::Hybrid)
   {
      CheckNativeCompatibility();
      transfer_map = new ParTransferMap(*src_fes, *dst_fes);
      src_fes_sequence = src_fes->GetSequence();
      dst_fes_sequence = dst_fes->GetSequence();
      BuildNativeInterfaceMask();
   }

   if (backend == Backend::GSLIB || backend == Backend::Hybrid)
   {
#ifdef MFEM_USE_GSLIB
      // Setup specific for GSLIB functionalities (determine QP coordinates on interface, setup FindPointsGSLIB instance)
      finder = new FindPointsGSLIB(comm);
      SetupGSLIB();
      gslib_initialized = true; // Mark that the initial setup has been done
#else
      MFEM_ABORT(name <<
                 ": the GSLIB and Hybrid backends need MFEM built with GSLIB (MFEM_USE_GSLIB); "
                 "use Backend::Native (ParSubMeshes of one parent mesh, same FE space on both sides).");
#endif
   }

   tmp_src_gf.SetSpace(src_fes); tmp_src_gf = 0.0;
}

InterfaceTransfer::~InterfaceTransfer()
{
   if (transfer_map)
   {
      delete transfer_map;
   }

#ifdef MFEM_USE_GSLIB
   if (finder)
   {
      finder->FreeData();
      delete finder;
   }
#endif

   // Nothing else to clean, InterfaceTransfer doesn't take ownership of the mesh or FiniteElementSpace or Mesh
}

void InterfaceTransfer::CheckNativeCompatibility() const
{
   // ParTransferMap (SubMeshToSubMesh) maps both spaces through one space on the common root parent, built
   // from the source FE collection: it needs ParSubMeshes of the same parent and identical FE spaces.
   // Release-mode checks: a violation would otherwise give silently wrong transfers.
   MFEM_VERIFY(ParSubMesh::IsParSubMesh(src_mesh) &&
               ParSubMesh::IsParSubMesh(dst_mesh),
               name << ": the Native backend needs source and destination meshes that are ParSubMeshes "
               "of the same parent mesh (use the GSLIB backend for independent meshes).");
   const ParMesh *src_root = SubMeshUtils::GetRootParent(
                                *static_cast<ParSubMesh *>(src_mesh));
   const ParMesh *dst_root = SubMeshUtils::GetRootParent(
                                *static_cast<ParSubMesh *>(dst_mesh));
   MFEM_VERIFY(src_root == dst_root,
               name << ": the Native backend needs ParSubMeshes of the same parent mesh.");
   MFEM_VERIFY(std::string(src_fes->FEColl()->Name()) == dst_fes->FEColl()->Name(),
               name << ": the Native backend needs the same FE collection on both sides (got "
               << src_fes->FEColl()->Name() << " and " << dst_fes->FEColl()->Name()
               << "); use the GSLIB backend for different orders.");
   MFEM_VERIFY(src_fes->GetVDim() == dst_fes->GetVDim(),
               name << ": the Native backend needs the same vdim on both sides.");
}

void InterfaceTransfer::BuildNativeInterfaceMask()
{
   // ParTransferMap overwrites every dst dof shared with the source submesh; the GSLIB backend only writes the
   // dofs of the marked interface. Same semantics for both: mark the vdofs of the marked dst boundary elements.
   dst_iface_mask.SetSize(dst_fes->GetVSize());
   dst_iface_mask = 0;
   Array<int> vdofs;
   for (int be : dst_bdr_element_idx)
   {
      dst_fes->GetBdrElementVDofs(be, vdofs);
      for (int v : vdofs) { dst_iface_mask[v >= 0 ? v : -1 - v] = 1; }
   }
   dst_fes->Synchronize(
      dst_iface_mask); // shared dofs on ranks without the boundary element
}

void InterfaceTransfer::NativeTransfer(const ParGridFunction &src_gf,
                                       ParGridFunction &dst_gf)
{
   // The transfer map was built for src_fes/dst_fes and indexes the grid functions directly
   MFEM_VERIFY(src_gf.Size() == src_fes->GetVSize() &&
               dst_gf.Size() == dst_fes->GetVSize(),
               name << ": the Native transfer needs grid functions on the spaces given at construction.");
   dst_backup = dst_gf;
   transfer_map->Transfer(src_gf, dst_gf);
   // Restore the dofs outside the marked interface
   for (int i = 0; i < dst_gf.Size(); i++)
   {
      if (!dst_iface_mask[i]) { dst_gf(i) = dst_backup(i); }
   }
}

void InterfaceTransfer::Update( VectorCoefficient *dx )
{
   if (transfer_map)
   {
      // The transfer map is topological (dof-to-dof through the parent mesh): mesh motion does not change it.
      MFEM_VERIFY(src_fes->GetSequence() == src_fes_sequence &&
                  dst_fes->GetSequence() == dst_fes_sequence,
                  name << ": the FE spaces changed after construction (e.g. refinement); recreate the InterfaceTransfer.");
   }

   // GSLIB: search again (FindPointsGSLIB::Setup frees the previous setup). With undeformed coordinates there
   // is nothing to update.
   if (finder && !use_undeformed_coords)
   {
      SetupGSLIB(dx);
   }
}

void InterfaceTransfer::GSLIBUseUndeformedCoords(bool use_undeformed_coords_)
{
   use_undeformed_coords = use_undeformed_coords_;

   Update(); // Reinitialize the finder with the new undeformed coordinates
}

void InterfaceTransfer::Interpolate(ParGridFunction &src_gf,
                                    ParGridFunction &dst_gf)
{
   if (backend == Backend::Native || backend == Backend::Hybrid)
   {
      NativeTransfer(src_gf, dst_gf);
   }
   else // Backend::GSLIB
   {
#ifdef MFEM_USE_GSLIB
      MFEM_VERIFY(src_gf.FESpace()->GetVDim() == dst_gf.FESpace()->GetVDim(),
                  name << ": Interpolate needs source and destination grid functions with the same vdim.");

      // Interpolate the source grid function (values for all the searched points, ordered like src_gf)
      finder->Interpolate(src_gf, gslib_vals);

      // Before scattering, ensure that for Vector GridFunctions (vdim > 1) with byNODES ordering,
      // we reorder the data byVDIM
      // @note: If src_gf is a Vector GridFunction (vdim > 1) with byNODES ordering,
      // then in parallel, interp_vals will be structured as:
      // [ x1 x2 ... xn | y1 y2 ... yn | ... | z1 z2 ... zn ] over all the (global) points,
      // so MPI_Scatterv would not distribute it correctly to each rank.
      int vdim = src_gf.FESpace()->GetVDim();
      if (src_gf.FESpace()->GetOrdering() == Ordering::byNODES && vdim > 1)
      {
         ReorderByVDIM(gslib_vals, interp_vals, vdim);
         TransferQoIToDestGf(interp_vals, dst_gf, Ordering::byVDIM);
      }
      else
      {
         // byVDIM or vdim == 1: use the GSLIB output directly
         TransferQoIToDestGf(gslib_vals, dst_gf, Ordering::byVDIM);
      }
#else
      MFEM_ABORT(name << ": the GSLIB backend needs MFEM built with GSLIB.");
#endif
   }
}

void InterfaceTransfer::InterpolateQoI(qoi_func_t qoi_func,
                                       ParGridFunction &dst_gf, bool bdr_version)
{
   if (bdr_version)
   {
      InterpolateQoIv2(qoi_func, dst_gf);
   }
   else if (backend == Backend::Native)
   {
      InterpolateQoINative(qoi_func, dst_gf);
   }
   else
   {
      InterpolateQoIv1(qoi_func, dst_gf);
   }
}

void InterfaceTransfer::AssembleSourceQoI(qoi_func_t qoi_func,
                                          bool element_side)
{
   // Evaluate the QoI at the source interface dofs into tmp_src_gf. A dof shared by several interface boundary
   // elements (on this or other ranks) gets the average of their values: SetSubVector + SetTrueVector would
   // keep an arbitrary copy, including the 0 of a rank that shares the dof without an interface face.
   const int vdim = src_fes->GetVDim();
   Vector qoi(vdim), loc_vals;
   Array<int> vdofs;

   if (!src_cnt_tv)
   {
      // Multiplicity of the source interface true dofs (topological, computed once)
      tmp_src_gf = 0.0;
      for (int be_idx : src_bdr_element_idx)
      {
         src_fes->GetBdrElementVDofs(be_idx, vdofs);
         loc_vals.SetSize(vdofs.Size());
         loc_vals = 1.0;
         tmp_src_gf.AddElementVector(vdofs, loc_vals);
      }
      src_cnt_tv.reset(tmp_src_gf.ParallelAssemble());
   }

   tmp_src_gf = 0.0;
   for (int be_idx : src_bdr_element_idx)
   {
      const IntegrationRule &ir_face = src_fes->GetBE(be_idx)->GetNodes();
      const int ndofs = ir_face.GetNPoints();
      src_fes->GetBdrElementVDofs(be_idx, vdofs);
      loc_vals.SetSize(vdofs.Size());

      FaceElementTransformations *FTr = element_side ?
                                        src_mesh->GetBdrFaceTransformations(be_idx) : nullptr;
      ElementTransformation *Tr = element_side ? nullptr :
                                  src_fes->GetBdrElementTransformation(be_idx);
      for (int qp = 0; qp < ndofs; qp++)
      {
         const IntegrationPoint &ip = ir_face.IntPoint(qp);
         if (element_side)
         {
            FTr->SetAllIntPoints(&ip);
            qoi_func(FTr->GetElement1Transformation(), FTr->GetElement1IntPoint(), qoi);
         }
         else
         {
            Tr->SetIntPoint(&ip);
            qoi_func(*Tr, ip, qoi);
         }
         // GetBdrElementVDofs is component-major for both orderings
         for (int d = 0; d < vdim; d++) { loc_vals(d * ndofs + qp) = qoi(d); }
      }
      tmp_src_gf.AddElementVector(vdofs, loc_vals);
   }

   std::unique_ptr<HypreParVector> tv(tmp_src_gf.ParallelAssemble());
   for (int i = 0; i < tv->Size(); i++)
   {
      if ((*src_cnt_tv)(i) > 0.0) { (*tv)(i) /= (*src_cnt_tv)(i); }
   }
   tmp_src_gf.SetFromTrueDofs(*tv);
}

void InterfaceTransfer::InterpolateQoINative(qoi_func_t qoi_func,
                                             ParGridFunction &dst_gf)
{
   // Same contract as InterpolateQoIv1 (qoi_func gets the transformation of the source *element*), evaluated at
   // the source interface dofs and moved with the native map.
   MFEM_VERIFY(dst_gf.ParFESpace()->GetVDim() == src_fes->GetVDim(),
               name << ": InterpolateQoI (Native) needs a destination with the same vdim as the source space.");
   AssembleSourceQoI(qoi_func, true);
   NativeTransfer(tmp_src_gf, dst_gf);
}

void InterfaceTransfer::InterpolateQoIv1(qoi_func_t qoi_func,
                                         ParGridFunction &dst_gf)
{
#ifdef MFEM_USE_GSLIB
   // Extract space dimension and FE space from the mesh
   ParFiniteElementSpace *fes_dst = dst_gf.ParFESpace();
   int vdim = fes_dst->GetVDim();

   // Distribute internal GSLIB info to the corresponding mpi-rank for each point.
   Array<unsigned int>
   recv_elem,
   recv_code;   // Element and GSLIB code
   Vector recv_rst; // (Reference) coordinates of the quadrature points
   finder->DistributePointInfoToOwningMPIRanks(recv_elem, recv_rst, recv_code);
   int npt_recv = recv_elem.Size();

   // Compute qoi locally (on source mesh)
   qoi_src.SetSize(npt_recv * vdim);
   qoi_loc.SetSize(vdim);
   for (int i = 0; i < npt_recv; i++)
   {
      // Get the element index
      const int e = recv_elem[i];

      // Get the quadrature point
      IntegrationPoint ip;
      if (sdim == 2)
      {
         ip.Set2(recv_rst(sdim * i + 0), recv_rst(sdim * i + 1));
      }
      else if (sdim == 3)
      {
         ip.Set3(recv_rst(sdim * i + 0), recv_rst(sdim * i + 1),
                 recv_rst(sdim * i + 2));
      }

      // Get the element transformation
      ElementTransformation *Tr = src_fes->GetElementTransformation(e);
      Tr->SetIntPoint(&ip);

      // Extract the local qoi vector
      qoi_loc.MakeRef(qoi_src, i * vdim, vdim);

      // Compute the qoi_src at quadrature point (it will change the qoi_src vector)
      qoi_func(*Tr, ip, qoi_loc);
   }

   // Transfer the QoI from the source mesh to the destination mesh at quadrature points
   finder->DistributeInterpolatedValues(qoi_src, vdim, Ordering::byVDIM, qoi_dst);

   // Transfer the QoI to the destination grid function
   TransferQoIToDestGf(qoi_dst, dst_gf, Ordering::byVDIM);
#else
   MFEM_ABORT(name <<
              ": InterpolateQoI(..., false) with the GSLIB/Hybrid backend needs MFEM built with GSLIB.");
#endif
}


void InterfaceTransfer::InterpolateQoIv2(qoi_func_t qoi_bdr_func,
                                         ParGridFunction &dst_gf)
{
   // Check that vdim of dst_gf matches the expected dimension
   MFEM_VERIFY(dst_gf.ParFESpace()->GetVDim() == src_fes->GetVDim(),
               name << ": InterpolateQoI(..., true) needs a destination with the same vdim as the source space.");

   // ----- 1. QoI at the source interface dofs, with the boundary element transformation (gives the normal) -----
   AssembleSourceQoI(qoi_bdr_func, false);

   // ----- 2. Transfer the QoI to the destination grid function -----
   this->Interpolate(tmp_src_gf, dst_gf);
}


inline void InterfaceTransfer::TransferQoIToDestGf(const Vector &dst_vec,
                                                   ParGridFunction &dst_gf, const mfem::Ordering::Type src_ordering)
{
   // @note: ordering of dst_vec will be:
   //       * byVDIM/byNODES, depending on field_in (if using Interpolate)
   //       * byVDIM (if using InterpolateQoI)
   //       either way, it's not necessarily the same as dst_gf ordering

   // Extract fe space from the destination grid function
   ParFiniteElementSpace *dst_gf_fes = dst_gf.ParFESpace();
   const int dof = dst_gf_fes->GetTypicalTraceElement()->GetNodes().GetNPoints();
   int vdim = dst_gf_fes->GetVDim();

   // In parallel we need to scatter the interpolated data so that each rank only has the right portion of the data
   // Compute recvcounts and displacements
   for (int i = 0; i < nranks; i++)
   {
      recvcounts_scatter[i] = vdim * recvcounts[i];
   }

   displs_scatter[0] = 0; // The first rank starts at index 0
   for (int i = 1; i < nranks; i++)
   {
      displs_scatter[i] = displs_scatter[i - 1] + recvcounts_scatter[i - 1];
   }

   // Scatter the global dst_vec to get the local part for this rank
   int local_size = recvcounts_scatter[myid]; // Local size for this rank
   if (local_dst_vec.Size() != local_size)
   {
      local_dst_vec.SetSize(local_size);
   }

   MPI_Scatterv(dst_vec.GetData(), recvcounts_scatter.GetData(),
                displs_scatter.GetData(), MFEM_MPI_REAL_T,
                local_dst_vec.GetData(), local_size, MFEM_MPI_REAL_T, 0, comm);

   // Accumulate the values (and their multiplicity) of the local interface boundary elements, then assemble across
   // ranks and average: a rank that shares an interface dof without holding an interface boundary element (e.g.
   // a vertex fan) would otherwise keep a stale copy, which can be the owner's copy. Collective.
   const int nbe = dst_bdr_element_idx.Size();
   MFEM_VERIFY(local_size == vdim * dof * nbe,
               name << ": the destination grid function must use the FE collection of the destination space given at construction.");
   if (dst_val_gf.FESpace() != dst_gf_fes) { dst_val_gf.SetSpace(dst_gf_fes); dst_cnt_gf.SetSpace(dst_gf_fes); }
   dst_val_gf = 0.0;
   dst_cnt_gf = 0.0;

   int idx_in, idx_out;
   Vector loc_values(dof * vdim), ones(dof * vdim);
   ones = 1.0;
   Array<int> vdofs(dof * vdim);
   for (int be = 0; be < nbe; be++)
   {
      dst_gf_fes->GetBdrElementVDofs(dst_bdr_element_idx[be], vdofs);
      for (int qp = 0; qp < dof; qp++)
      {
         for (int d = 0; d < vdim; d++)
         {
            idx_in = (src_ordering == Ordering::byVDIM) ? vdim * (dof * be + qp) +
                     d   // dst_vec ordered byVDIM
                     : (nbe * dof) * d + be * dof + qp; // byNODES
            idx_out = dof * d +
                      qp;  // vdofs is returned byNODES regardless of the dst_ordering, so we populate loc_values accordingly
            loc_values(idx_out) = local_dst_vec(idx_in);
         }
      }
      dst_val_gf.AddElementVector(vdofs, loc_values);
      dst_cnt_gf.AddElementVector(vdofs, ones);
   }

   std::unique_ptr<HypreParVector> tv(dst_val_gf.ParallelAssemble());
   std::unique_ptr<HypreParVector> tc(dst_cnt_gf.ParallelAssemble());
   for (int i = 0; i < tv->Size(); i++)
   {
      if ((*tc)(i) > 0.0) { (*tv)(i) /= (*tc)(i); }
   }
   dst_val_gf.SetFromTrueDofs(*tv);
   dst_cnt_gf.SetFromTrueDofs(*tc);

   // Only the interface dofs of dst_gf are overwritten
   for (int i = 0; i < dst_gf.Size(); i++)
   {
      if (dst_cnt_gf(i) > 0.0) { dst_gf(i) = dst_val_gf(i); }
   }
}



void InterfaceTransfer::GSLIBAttrToMarker(int max_attr,
                                          const Array<unsigned int> &elems, Array<int> &marker)
{
   // Convert the element indices to markers
   marker.SetSize(max_attr);
   marker = 0;
   for (const auto &elem : elems)
   {
      if (elem <= 0 || elem > static_cast<unsigned int>(max_attr))
      {
         continue;
      }
      int attr = static_cast<int>(elem);
      marker[attr - 1] = 1;
   }

   return;
}


void InterfaceTransfer::GetElementIdx(Array<int> &elem_idx)
{
   elem_idx.SetSize(src_mesh->GetNE());
   elem_idx = 0;

   if (backend == Backend::Native)
   {
      // Source elements adjacent to the interface boundary elements (a boundary element of a ParSubMesh is
      // always local to the rank owning its adjacent element)
      for (int be : src_bdr_element_idx)
      {
         int el, info;
         src_mesh->GetBdrElementAdjacentElement(be, el, info);
         elem_idx[el] = 1;
      }
      return;
   }

#ifdef MFEM_USE_GSLIB
   // Mark the local source elements that contain found points. finder->GetElem() holds 0-based element ids
   // local to the owning rank (finder->GetProc()) for the full point set, so it cannot be used directly
   // as a 1-based marker on this rank. Collective call.
   Array<unsigned int> recv_elem, recv_code;
   Vector recv_rst;
   finder->DistributePointInfoToOwningMPIRanks(recv_elem, recv_rst, recv_code);
   for (int i = 0; i < recv_elem.Size(); i++)
   {
      if (recv_code[i] != 2) { elem_idx[recv_elem[i]] = 1; } // code 2: point not found
   }
#else
   MFEM_ABORT(name <<
              ": GetElementIdx with the GSLIB/Hybrid backend needs MFEM built with GSLIB.");
#endif
}

void InterfaceTransfer::SetupGSLIB(  VectorCoefficient *dx )
{
#ifdef MFEM_USE_GSLIB
   // Check if VectorCoefficient dx has right dimension
   if (dx)
   {
      MFEM_ASSERT(dx->GetVDim() == sdim,
                  "InterfaceTransfer::SetupGSLIB: dx must have the same dimension as the mesh space dimension.");
   }

   if (!gslib_initialized || !use_undeformed_coords)
   {
      /// Compute coordinates of quadrature points on dst_bdr_element_idx (on the destination mesh) --> LOCAL
      const IntegrationRule &ir_face =
         (dst_fes->GetTypicalTraceElement())->GetNodes();
      int bdr_coords_size = dst_bdr_element_idx.Size() * ir_face.GetNPoints() * sdim;

      Vector bdr_element_coords_loc;
      bdr_element_coords_loc.SetSize(bdr_coords_size);
      bdr_element_coords_loc = 0.0;
      x.SetSize(sdim);
      x = 0.0;
      dx_val.SetSize(sdim);
      dx_val = 0.0;

      bool compute_offset = (dx != nullptr) && !use_undeformed_coords;

      auto pec = Reshape(bdr_element_coords_loc.ReadWrite(), sdim,
                         ir_face.GetNPoints(), dst_bdr_element_idx.Size());
      for (int be = 0; be < dst_bdr_element_idx.Size(); be++)
      {
         int be_idx = dst_bdr_element_idx[be];
         const FiniteElement *fe = dst_fes->GetBE(be_idx);
         ElementTransformation *Tr = dst_fes->GetBdrElementTransformation(be_idx);
         const IntegrationRule &ir_face = fe->GetNodes();

         for (int qp = 0; qp < ir_face.GetNPoints(); qp++)
         {
            const IntegrationPoint &ip = ir_face.IntPoint(qp);
            Tr->SetIntPoint(&ip);

            dx_val = 0.0;
            if (compute_offset)
            {
               // If dx is provided, compute the offset
               dx->Eval(dx_val, *Tr, ip);
            }

            Tr->Transform(ip, x);
            for (int d = 0; d < sdim; d++)
            {
               pec(d, qp, be) = x(d) + dx_val(d);
            }
         }
      }

      // In parallel we need to gather the coordinates from all ranks so the GSLIB finder can search all points
      // The FindPoints will take care of identifying on which rank the points are located
      // First reduce the size of the vector
      int global_size = 0;
      MPI_Allreduce(&bdr_coords_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
      bdr_element_coords.SetSize(global_size);

      // Gather the sizes from all ranks
      MPI_Allgather(&bdr_coords_size, 1, MPI_INT, recvcounts_gather.GetData(), 1,
                    MPI_INT, comm);

      for (int i = 0; i < nranks; i++)
      {
         recvcounts[i] = recvcounts_gather[i] /
                         sdim; // Convert from elements * sdim to elements
      }

      // Compute displacements
      displs_gather[0] = 0;
      for (int i = 1; i < nranks; i++)
      {
         displs_gather[i] = displs_gather[i - 1] + recvcounts_gather[i - 1];
      }

      // Now gather the coordinates from all ranks
      MPI_Allgatherv(bdr_element_coords_loc.GetData(), bdr_coords_size,
                     MFEM_MPI_REAL_T,
                     bdr_element_coords.GetData(), recvcounts_gather.GetData(),
                     displs_gather.GetData(), MFEM_MPI_REAL_T, comm);

      // Coordinates at construction (needed by the not-found report below already in the first setup)
      if (!gslib_initialized) { undeformed_bdr_element_coords = bdr_element_coords; }
   }

   // Set target bdry coordinates
   Vector &target_coords = (use_undeformed_coords &&
                            gslib_initialized) ? undeformed_bdr_element_coords : bdr_element_coords;

   // Default parameters
   const real_t rel_bbox_el = 0.1;
   const real_t newton_tol = 1.0e-12;
   const int npts_at_once = 256;
   finder->Setup(*src_mesh, rel_bbox_el, newton_tol, npts_at_once);
   finder->SetDistanceToleranceForPointsFoundOnBoundary(boundary_tol);
   finder->FindPoints(target_coords,
                      Ordering::byVDIM); // --> Because the coordinates are stored byVDIM
   auto procs = finder->GetProc();
   auto finder_codes = finder->GetCode();
   int idx = 0;
   for (auto &code : finder_codes)
   {
      if (code == 2)
      {
         mfem::out << "Point NOT found in the mesh, on rank " << myid << " for " << name
                   << std::endl;
         mfem::out << "Point coordinates: " << target_coords(idx*sdim) << ", " <<
                   target_coords(idx*sdim + 1);
         if (sdim == 3)
         {
            mfem::out << ", " << target_coords(idx*sdim + 2);
         }
         mfem::out << ")." << std::endl;
         mfem::out << "Point undeformed coordinates: " << undeformed_bdr_element_coords(
                      idx*sdim) << ", " << undeformed_bdr_element_coords(idx*sdim + 1);
         if (sdim == 3)
         {
            mfem::out << ", " << undeformed_bdr_element_coords(idx*sdim + 2);
         }
         mfem::out << ")." << std::endl;
         MFEM_ABORT("Aborting!");
      }
      idx ++; // Move to the next point
   }
#else
   MFEM_ABORT(name << ": the GSLIB point search needs MFEM built with GSLIB.");
#endif
}

} // namespace common_fsi_utils

} // namespace mfem

#endif // MFEM_USE_MPI
