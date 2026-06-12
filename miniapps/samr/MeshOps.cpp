#include "SAMRAI/hier/PatchLevel.h"

#include "MeshOps.hpp"

namespace mfem
{

MeshOps::MeshOps(std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy) :
   hierarchy(hierarchy), corners(getCorners( hierarchy->getDim().getValue())),
   fe_collection_node(1, hierarchy->getDim().getValue()),
   fe_collection_cell(0, hierarchy->getDim().getValue())
{
   int rank, ranks;
   MPI_Comm_rank(hierarchy->getMPI().getCommunicator(), &rank);
   MPI_Comm_size(hierarchy->getMPI().getCommunicator(), &ranks);
   const unsigned short dim = hierarchy->getDim().getValue();

   // build Cartesian MFEM mesh with unit width elements to easier facilitate
   // indexing
   SAMRAI::hier::Index domain_lower = SAMRAI::hier::Index::getMaxIndex(hierarchy->getDim());
   SAMRAI::hier::Index domain_upper = SAMRAI::hier::Index::getMinIndex(hierarchy->getDim());
   std::shared_ptr<SAMRAI::hier::PatchLevel> level0 = hierarchy->getPatchLevel(0);
   for (SAMRAI::hier::PatchLevel::iterator patch_iter=level0->begin();
         patch_iter != level0->end(); patch_iter++)
   {
      const SAMRAI::hier::Box& box = patch_iter->getBox();
      domain_lower.min(box.lower());
      domain_upper.max(box.upper());
   }

   // TODO: see if there's a more appropriate way to access the int[] object in
   //       the hier::Index class
   int* lowerData = &domain_lower[0];
   int* upperData = &domain_upper[0];
   MPI_Allreduce(MPI_IN_PLACE, lowerData, dim, MPI_INT, MPI_MIN,
      hierarchy->getMPI().getCommunicator());
   MPI_Allreduce(MPI_IN_PLACE, upperData, dim, MPI_INT, MPI_MAX,
      hierarchy->getMPI().getCommunicator());
   const SAMRAI::hier::IntVector num_cells = domain_upper - domain_lower + 1;

   // Create coarse mesh
   mfem::Mesh serial_mesh;
   switch(dim)
   {
      case 1:
      {
         const int num_x = num_cells(0);
         serial_mesh = mfem::Mesh::MakeCartesian1D(num_x, num_x);
         // label xlo and xhi as "1"
         serial_mesh.GetBdrElement(0)->SetAttribute(1);
         serial_mesh.GetBdrElement(1)->SetAttribute(1);
         break;
      }
      case 2:
      {
         const int num_x = num_cells(0);
         const int num_y = num_cells(1);
         serial_mesh = mfem::Mesh::MakeCartesian2D(num_x, num_y,
            mfem::Element::Type::QUADRILATERAL, true, num_x, num_y);
         break;
      }
      case 3:
      {
         const int num_x = num_cells(0);
         const int num_y = num_cells(1);
         const int num_z = num_cells(2);
         serial_mesh = mfem::Mesh::MakeCartesian3D(num_x, num_y, num_z,
            mfem::Element::Type::HEXAHEDRON, num_x, num_y, num_z);
         break;
      }
      default:
         MFEM_ABORT("dimension not valid");
   }
   serial_mesh.EnsureNCMesh();
   mesh = ParMesh(hierarchy->getMPI().getCommunicator(), serial_mesh);
   serial_mesh.Clear();

   // create global patch info list by adding all SAMRAI patches to it
   AddNewPatchesToGlobalPatchInfo();

   // refine the mesh according to the global patch info list
   std::vector<PatchLevelBounds> global_patch_bounds;
   GetGlobalPatchBounds(global_patch_bounds);
   RefineMesh(global_patch_bounds);

   // create transfer maps between SAMRAI grid and refined MFEM mesh
   CreateTransferMaps();

   // create the finite element space and grid function for the mesh topology
   fe_spaces_node[mesh.Dimension()] =
      std::make_unique<ParFiniteElementSpace>(&mesh, &fe_collection_node, mesh.Dimension());
   mesh_grid_function =
      std::make_shared<ParGridFunction>(fe_spaces_node[mesh.Dimension()].get());
   mesh.SetNodalGridFunction(mesh_grid_function.get());

   // store the current mesh topology values b/c they are in index coordinates
   mesh_grid_function->GetTrueDofs(mesh_index_space_tdofs);
}

void MeshOps::GatherGlobalPatchInfo(const std::vector<PatchInfo>& local_patch_info,
   std::vector<PatchInfo>& gathered_patch_info) const
{
   int rank, ranks;
   MPI_Comm_rank(mesh.GetComm(), &rank);
   MPI_Comm_size(mesh.GetComm(), &ranks);

   // serialize local patch data
   const unsigned patch_info_size = PatchInfo::Size(mesh.Dimension());
   const int local_patch_count = local_patch_info.size();
   BlockArray<int> local_patch_info_buffer(patch_info_size, local_patch_count);
   for (unsigned ind=0; ind < local_patch_info.size(); ind++)
   {
      const PatchInfo& patch_info = local_patch_info[ind];
      local_patch_info_buffer.SetBlock(ind, patch_info.AsArray());
   }

   // gather the number of patches from each rank
   Array<int> global_patch_counts(ranks);
   MPI_Allgather(&local_patch_count, 1, MPI_INT, global_patch_counts.GetData(),
      1, MPI_INT, mesh.GetComm());

   // gather the serialized global patch info on each rank
   BlockArray<int> gathered_patch_info_buffer(patch_info_size, global_patch_counts.Sum());
   {
      Array<int> gather_counts;
      gathered_patch_info_buffer.GetElementCounts(global_patch_counts, gather_counts);
      Array<int> gather_offsets = gather_counts;
      gather_offsets.Prepend(0);
      gather_offsets.PartialSum();
      MPI_Allgatherv(local_patch_info_buffer.GetData(),
         local_patch_info_buffer.Size(), MPI_INT,
         gathered_patch_info_buffer.GetData(), gather_counts.GetData(),
         gather_offsets.GetData(), MPI_INT, mesh.GetComm());
   }

   // unserialize patch data and note rank that has each patch
   gathered_patch_info.reserve(gathered_patch_info_buffer.NumBlocks());
   for (int block_ind=0; block_ind < gathered_patch_info_buffer.NumBlocks();
      block_ind++)
   {
      gathered_patch_info.emplace_back(
         PatchInfo::FromArray(gathered_patch_info_buffer.GetBlock(block_ind)));
   }
}

void MeshOps::AddNewPatchesToGlobalPatchInfo()
{
   int rank;
   MPI_Comm_rank(mesh.GetComm(), &rank);

   // collect local added patch info, followed by gathering global added patch
   // info
   std::vector<PatchInfo> local_added_patch_info;
   for (int level_number = 0; level_number <= hierarchy->getFinestLevelNumber();
      level_number++)
   {
      std::shared_ptr<SAMRAI::hier::PatchLevel> patch_level =
         hierarchy->getPatchLevel(level_number);
      for (SAMRAI::hier::PatchLevel::iterator patch_iter=patch_level->begin();
            patch_iter != patch_level->end(); patch_iter++)
      {
         bool added = true;
         for (unsigned patch_ind=0; patch_ind < global_patch_info.size(); patch_ind++)
         {
            const PatchInfo& patch_info = global_patch_info[patch_ind];
            if (patch_info.rank == rank &&
                  patch_info.lower_index == patch_iter->getBox().lower() &&
                  patch_info.upper_index == patch_iter->getBox().upper())
            {
               added = false;
               break;
            }
         }
         if (added)
         {
            local_added_patch_info.emplace_back(rank, level_number,
               patch_iter->getBox().lower(), patch_iter->getBox().upper());
         }
      }
   }
   std::vector<PatchInfo> global_added_patch_info;
   GatherGlobalPatchInfo(local_added_patch_info, global_added_patch_info);

   // add patches to theglobal patch info list
   for (const PatchInfo& patch_info : global_added_patch_info)
   {
      global_patch_info.push_back(patch_info);
   }
}

void MeshOps::RemoveOldPatchesFromGlobalPatchInfo()
{
   int rank;
   MPI_Comm_rank(mesh.GetComm(), &rank);

   // collect local removed patch info, followed by gathering global removed
   // patch info
   std::vector<PatchInfo> local_removed_patch_info;
   for (unsigned patch_ind=0; patch_ind < global_patch_info.size(); patch_ind++)
   {
      const PatchInfo& patch_info = global_patch_info[patch_ind];
      if (patch_info.rank != rank)
         continue;
      std::shared_ptr<SAMRAI::hier::PatchLevel> patch_level =
         hierarchy->getPatchLevel(patch_info.level_number);
      for (SAMRAI::hier::PatchLevel::iterator patch_iter=patch_level->begin();
            patch_iter != patch_level->end(); patch_iter++)
      {
         if (patch_info.lower_index == patch_iter->getBox().lower() &&
               patch_info.upper_index == patch_iter->getBox().upper())
         {
            local_removed_patch_info.push_back(patch_info);
            break;
         }
      }
   }
   std::vector<PatchInfo> global_removed_patch_info;
   GatherGlobalPatchInfo(local_removed_patch_info, global_removed_patch_info);

   // create new global patch info list with only the remaining patches
   std::vector<PatchInfo> new_global_patch_info;
   for (const PatchInfo& patch_info : global_patch_info)
   {
      bool not_removed = true;
      for (const PatchInfo& removed_patch_info : global_removed_patch_info)
      {
         if (patch_info == removed_patch_info)
         {
            not_removed = false;
            break;
         }
      }
      if (not_removed)
      {
         new_global_patch_info.push_back(patch_info);
      }
   }
   global_patch_info = std::move(new_global_patch_info);
}

void MeshOps::GetGlobalPatchBounds(std::vector<PatchLevelBounds>& global_patch_bounds) const
{
   // organize patch bounds by level
   global_patch_bounds.resize(hierarchy->getFinestLevelNumber()+1);
   for (int patch_ind=0; patch_ind < global_patch_info.size(); patch_ind++)
   {
      const PatchInfo& current_patch = global_patch_info[patch_ind];
      const int level_number = current_patch.level_number;
      global_patch_bounds[level_number].emplace_back(
         current_patch.lower_index, current_patch.upper_index);
   }
}

void MeshOps::DerefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds)
{
   const real_t error_threshold = 1.0;
   Array<real_t> pseudo_error(mesh.GetNE());

   for (int level_num=hierarchy->getFinestLevelNumber(); level_num >= 0; level_num--)
   {
      const Vector level0_ratio = toVector(
         hierarchy->getPatchLevel(level_num)->getRatioToLevelZero());
      const SAMRAI::hier::IntVector level_ratio =
         hierarchy->getPatchLevel(level_num)->getRatioToCoarserLevel();

      // TODO: support 3D
      MFEM_VERIFY(mesh.Dimension() == 2, "3D derefinement not yet supported")
      const double dx = 1.0/level0_ratio[0];
      const double dy = 1.0/level0_ratio[1];

      // determine which elements to derefine at this level
      Array<int> derefine_element_inds;
      pseudo_error = error_threshold;
      for (int element_ind=0; element_ind < mesh.GetNE(); element_ind++)
      {
         Vector x0, h;
         ElementTransformation* transform =
            mesh.GetElementTransformation(element_ind);
         // TODO: support 3D
         const IntegrationPoint ip0 = {0.0, 0.0};
         const IntegrationPoint ip1 = {1.0, 1.0};
         transform->Transform(ip0, x0);
         transform->Transform(ip1, h);
         h -= x0;

         if (std::abs(h[0] - dx) < 1e-12 && std::abs(h[1] - dy) < 1e-12)
         {
            // get center in current level coordinates
            Vector center;
            mesh.GetElementCenter(element_ind, center);
            center *= level0_ratio;
            SAMRAI::hier::Index index = toIndex(center);

            // check if center is in any patches from current level
            for (const auto& bounds : global_patch_bounds[level_num])
            {
               if (bounds.first <= index && index <= bounds.second + 1)
               {
                  derefine_element_inds.Append(element_ind);
                  pseudo_error[element_ind] = 0.0;
                  break;
               }
            }
         }
      }
      // coarsen/derefine elements
      mesh.DerefineByError(pseudo_error, error_threshold);
      // TODO: handle 3:1 derefinement
      MFEM_VERIFY(level_ratio == SAMRAI::hier::IntVector(SAMRAI::tbox::Dimension(2),2),
            "Coarsen/Derefinement ratio " << level_ratio << " not yet supported");
   }
}

void MeshOps::RefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds)
{
   for (int level_num=1; level_num <= hierarchy->getFinestLevelNumber(); level_num++)
   {
      const Vector level0_ratio = toVector(
         hierarchy->getPatchLevel(level_num)->getRatioToLevelZero());
      const SAMRAI::hier::IntVector level_ratio =
         hierarchy->getPatchLevel(level_num)->getRatioToCoarserLevel();

      // determine which elements to refine at this level
      Array<int> refine_element_inds;
      for (int element_ind=0; element_ind < mesh.GetNE(); element_ind++)
      {
         // get center in current level coordinates
         Vector center;
         mesh.GetElementCenter(element_ind, center);
         center *= level0_ratio;
         SAMRAI::hier::Index index = toIndex(center);

         // check if center is in any patches from current level
         for (const auto& bounds : global_patch_bounds[level_num])
         {
            if (bounds.first <= index && index <= bounds.second + 1)
            {
               refine_element_inds.Append(element_ind);
               break;
            }
         }
      }
      // refine elements
      Array<Refinement> refinements(refine_element_inds.Size());
      real_t ratioX, ratioY, ratioZ;
      ratioX = 1.0 / level_ratio[0];
      if (mesh.Dimension() > 1) ratioY = 1.0 / level_ratio[1];
      if (mesh.Dimension() > 2) ratioZ = 1.0 / level_ratio[2];
      for (int i=0; i < refinements.Size(); i++)
      {
         refinements[i] = Refinement(refine_element_inds[i],
            {{Refinement::X, ratioX}, {Refinement::Y, ratioY}});//, {Refinement::Z, ratioZ}});
      }
      mesh.GeneralRefinement(refinements);
      // refine elements further for 3:1 refinement
      if (level_ratio != SAMRAI::hier::IntVector(SAMRAI::tbox::Dimension(hierarchy->getDim()),2))
      {
         MFEM_VERIFY(level_ratio == SAMRAI::hier::IntVector(SAMRAI::tbox::Dimension(2),3),
            "Refinement ratio " << level_ratio << " not yet supported");
         // TODO: support 3D
         MFEM_VERIFY(mesh.Dimension() == 2, "3D refinement not yet supported")
         const double dx = 1.0/level0_ratio[0];
         const double dy = 1.0/level0_ratio[1];
         Table coarse_to_fine;
         mesh.ncmesh->GetRefinementTransforms().MakeCoarseToFineTable(coarse_to_fine);
         refinements.DeleteAll();
         for (const int& original_element_ind : refine_element_inds)
         {
            Array<int> new_element_inds;
            coarse_to_fine.GetRow(original_element_ind, new_element_inds);
            for (const int& new_element_ind : new_element_inds)
            {
               Vector x0, h;
               ElementTransformation* transform =
                  mesh.GetElementTransformation(new_element_ind);
               // TODO: support 3D
               const IntegrationPoint ip0 = {0.0, 0.0};
               const IntegrationPoint ip1 = {1.0, 1.0};
               transform->Transform(ip0, x0);
               transform->Transform(ip1, h);
               h -= x0;
               if (std::abs(h[0] - dx) > 1e-12 && std::abs(h[1] - dy) > 1e-12)
                  refinements.Append(Refinement(new_element_ind,
                     {{Refinement::X, dx/h[0]}, {Refinement::Y, dy/h[1]}}));
               else if (std::abs(h[0] - dx) > 1e-12)
                  refinements.Append(Refinement(new_element_ind,
                     Refinement::X, dx/h[0]));
               else if (std::abs(h[1] - dy) > 1e-12)
                  refinements.Append(Refinement(new_element_ind,
                     Refinement::Y, dy/h[1]));
            }
         }
         mesh.GeneralRefinement(refinements);
      }
   }
}

void MeshOps::CreateTransferMaps()
{
   int rank, ranks;
   MPI_Comm_rank(mesh.GetComm(), &rank);
   MPI_Comm_size(mesh.GetComm(), &ranks);

   // map local elements to global rank, patch level, and index
   std::vector<std::vector<ElementInfo>> local_element_info(ranks);
   local_element_inds.resize(ranks);
   for (int element_ind=0; element_ind < mesh.GetNE(); element_ind++)
   {
      // compute element level from element volume
      const real_t element_volume = mesh.GetElementVolume(element_ind);
      int level_number = 0;
      double level_volume = 1.0;
      while (element_volume < level_volume - 1e-12)
      {
         level_number++;
         const SAMRAI::hier::IntVector& ratio =
            hierarchy->getRatioToCoarserLevel(level_number);
         for (int i=0; i < mesh.Dimension(); i++)
            level_volume /= ratio[i];
      }
      // get center in level zero coordinates
      Vector center;
      mesh.GetElementCenter(element_ind, center);
      // convert center to patch level coordinates
      center *= toVector(hierarchy->getPatchLevel(level_number)->getRatioToLevelZero());
      SAMRAI::hier::Index index = toIndex(center);
      // use patch level coordinates to find associated patch to determine rank
      int index_rank = -1;
      for (const PatchInfo& patch : global_patch_info)
      {
         if (patch.level_number == level_number &&
               patch.lower_index <= index && index <= patch.upper_index)
         {
            index_rank = patch.rank;
            break;
         }
      }
      MFEM_VERIFY(index_rank >= 0, "Index not found in global patch list");
      local_element_info[index_rank].emplace_back(level_number, index);
      local_element_inds[index_rank].push_back(element_ind);
   }

   // send local element info to ranks that have corresponding cells
   const unsigned element_info_size = ElementInfo::Size(mesh.Dimension());
   std::vector<std::unique_ptr<BlockArray<int>>> element_info_buffers(ranks);
   std::vector<MPI_Request> requests(ranks);
   for (int remote_rank=0; remote_rank < ranks; remote_rank++)
   {
      // local elements that correspond to local cells will be accounted for later
      if (remote_rank == rank)
         continue;

      const std::vector<ElementInfo>& element_info_for_rank =
         local_element_info[remote_rank];

      element_info_buffers[remote_rank] =
         std::make_unique<BlockArray<int>>(element_info_size,
            element_info_for_rank.size());
      for (unsigned data_ind=0; data_ind < element_info_for_rank.size(); data_ind++)
      {
         element_info_buffers[remote_rank]->SetBlock(data_ind,
            element_info_for_rank[data_ind].AsArray());
      }
      MPI_Isend(element_info_buffers[remote_rank]->GetData(),
         element_info_buffers[remote_rank]->Size(), MPI_INT, remote_rank,
         element_info_tag, mesh.GetComm(), &requests[remote_rank]);
   }

   // define utility lambda function for setting local cell info
   auto SetLocalCellInfo = [&](int source_rank,
      const std::vector<ElementInfo>& element_info_list)
   {
      for (const ElementInfo& element_info : element_info_list)
      {
         const SAMRAI::hier::Index& index = element_info.index;
         std::shared_ptr<SAMRAI::hier::PatchLevel> patch_level =
            hierarchy->getPatchLevel(element_info.level_number);
         std::shared_ptr<SAMRAI::hier::Patch> patch;
         for (SAMRAI::hier::PatchLevel::iterator patch_iter=patch_level->begin();
               patch_iter != patch_level->end(); patch_iter++)
         {
            if (patch_iter->getBox().contains(index))
            {
               patch = *patch_iter;
               break;
            }
         }
         MFEM_VERIFY(patch != nullptr, "Level " << element_info.level_number
            << ", index " << index << " received from rank " << source_rank
            << " is not on any local patches");
         local_cell_info[source_rank].push_back({SAMRAI::pdat::CellIndex(index), patch});
      }
   };

   // receive remote element info that correspond to local cells
   local_cell_info.resize(ranks);
   int messages_received = 0;
   MPI_Status status;
   int count;
   while (messages_received < ranks - 1)
   {
      MPI_Probe(MPI_ANY_SOURCE, element_info_tag, mesh.GetComm(), &status);
      MPI_Get_count(&status, MPI_INT, &count);
      MFEM_ASSERT(count % element_info_size == 0, "Unexpected message size");
      BlockArray<int> remote_element_info(element_info_size, count / element_info_size);
      MPI_Recv(remote_element_info.GetData(), remote_element_info.Size(),
         MPI_INT, status.MPI_SOURCE, element_info_tag, mesh.GetComm(), &status);
      messages_received++;

      // unserialize and store cell info for received remote elements
      std::vector<ElementInfo> element_info_list;
      element_info_list.reserve(remote_element_info.NumBlocks());
      for (unsigned block_ind=0; block_ind < remote_element_info.NumBlocks();
            block_ind++)
      {
         element_info_list.emplace_back(
            ElementInfo::FromArray(remote_element_info.GetBlock(block_ind)));
      }
      SetLocalCellInfo(status.MPI_SOURCE, element_info_list);
   }

   // account for local elements that correspond to local cells
   SetLocalCellInfo(rank, local_element_info[rank]);

   MPI_Barrier(mesh.GetComm());
}

void MeshOps::setMeshGridFunction(std::shared_ptr<GridFunction> grid_function)
{
   mesh.SetNodalGridFunction(grid_function.get());
   mesh_grid_function = grid_function;
}

std::unique_ptr<ParFiniteElementSpace> MeshOps::createFESpace(
   FiniteElementCollection& fe_collection, int dim)
{
   return std::make_unique<ParFiniteElementSpace>(&mesh, &fe_collection, dim);
}

std::vector<std::unique_ptr<ParGridFunction>> MeshOps::transferToMFEM(
   const int position_id, const std::vector<int>& node_ids,
   const std::vector<int>& cell_ids)
{
   // check that finite element space under mesh grid function has not been
   // changed (it can later be replaced with a higher-order finite element space
   // and grid function via setMeshGridFunction)
   MFEM_ASSERT(mesh_grid_function->FESpace() == fe_spaces_node[mesh.Dimension()].get(),
      "An external mesh grid function can only be set after the call to transferToMFEM");

   // with mesh node positions, start maps of SAMRAI ids to grid functions
   std::vector<std::pair<int, GridFunction&>> node_fields;
   node_fields.emplace_back(position_id, *mesh_grid_function);
   std::vector<std::pair<int, ParGridFunction&>> cell_fields;

   // create finite element spaces and grid functions for each SAMRAI field
   std::vector<std::unique_ptr<ParGridFunction>> grid_functions;
   for (const int& node_id : node_ids)
   {
      // TODO: don't assume node ids have depth NDIM
      const int field_dimension = mesh.Dimension();
      if (fe_spaces_node.find(field_dimension) == fe_spaces_node.end())
         fe_spaces_node[field_dimension] =
            std::make_unique<ParFiniteElementSpace>(
               &mesh, &fe_collection_node, field_dimension);
      grid_functions.emplace_back(
         new ParGridFunction(fe_spaces_node[field_dimension].get()));
      node_fields.emplace_back(node_id, *grid_functions.back());
   }
   for (const int& cell_id : cell_ids)
   {
      // TODO: don't assume cell ids have depth 1
      const int field_dimension = 1;
      if (fe_spaces_cell.find(field_dimension) == fe_spaces_cell.end())
         fe_spaces_cell[field_dimension] =
            std::make_unique<ParFiniteElementSpace>(
               &mesh, &fe_collection_cell, field_dimension);
      grid_functions.emplace_back(
         new ParGridFunction(fe_spaces_cell[field_dimension].get()));
      cell_fields.emplace_back(cell_id, *grid_functions.back());
   }

   int rank, ranks;
   MPI_Comm_rank(mesh.GetComm(), &rank);
   MPI_Comm_size(mesh.GetComm(), &ranks);

   // extract buffer size information
   int num_variables;
   Array<int> node_field_vector_dimensions;
   Array<int> node_field_offsets;
   std::tie(num_variables, node_field_vector_dimensions, node_field_offsets)
      = ExtractBufferInfo(node_fields, cell_fields);

   // send local node/cell values to ranks that have corresponding elements
   std::vector<std::vector<double>> samrai_value_buffers(ranks);
   std::vector<MPI_Request> requests(ranks);
   for (int remote_rank=0; remote_rank < ranks; remote_rank++)
   {
      // local cells that correspond to local elements will be accounted for later
      if (remote_rank == rank)
         continue;

      const std::vector<CellInfo>& cell_info_for_rank = local_cell_info[remote_rank];
      std::vector<double>& local_samrai_values = samrai_value_buffers[remote_rank];
      local_samrai_values.resize(num_variables * cell_info_for_rank.size());
      for (int cell_ind=0; cell_ind < cell_info_for_rank.size(); cell_ind++)
      {
         const int samrai_values_ind = num_variables * cell_ind;
         const SAMRAI::pdat::CellIndex cell_index = cell_info_for_rank[cell_ind].index;
         const SAMRAI::hier::Patch& cell_patch = *cell_info_for_rank[cell_ind].patch;
         // node field values
         for (int i=0; i < node_fields.size(); i++)
         {
            const SAMRAI::pdat::NodeData<double>& node_values =
               *SAMRAI_SHARED_PTR_CAST<SAMRAI::pdat::NodeData<double>>(
                  cell_patch.getPatchData(std::get<int>(node_fields[i])));
            const int samrai_values_node_field_ind =
               samrai_values_ind + node_field_offsets[i];
            // order "by node" (aka inner loop around corners) to match what
            // Mesh::GetElementVDofs uses
            for (int d=0; d < node_field_vector_dimensions[i]; d++)
            {
               const int samrai_values_node_field_value_ind =
                  samrai_values_node_field_ind + corners.Size()*d;
               for (int c=0; c < corners.Size(); c++)
               {
                  const SAMRAI::pdat::NodeIndex corner_index(cell_index, corners[c]);
                  local_samrai_values[samrai_values_node_field_value_ind+c] =
                     node_values(corner_index,d);
               }
            }
         }
         // cell field values
         const int samrai_values_cell_field_ind =
            samrai_values_ind + node_field_offsets.Last();
         for (int i=0; i < cell_fields.size(); i++)
         {
            const SAMRAI::pdat::CellData<double>& field_SAMRAI =
               *SAMRAI_SHARED_PTR_CAST<SAMRAI::pdat::CellData<double>>(
                  cell_patch.getPatchData(std::get<int>(cell_fields[i])));
            local_samrai_values[samrai_values_cell_field_ind+i] =
               field_SAMRAI(cell_index);
         }

      }

      MPI_Isend(local_samrai_values.data(), local_samrai_values.size(),
         MPI_DOUBLE, remote_rank, samrai_values_tag, mesh.GetComm(),
         &requests[remote_rank]);
   }

   // receive remote node/cell values that correspond to local elements
   int messages_received = 0;
   MPI_Status status;
   while (messages_received < ranks - 1)
   {
      MPI_Probe(MPI_ANY_SOURCE, samrai_values_tag, mesh.GetComm(), &status);
      const std::vector<int>& element_inds = local_element_inds[status.MPI_SOURCE];
      std::vector<double> remote_samrai_values(num_variables * element_inds.size());
      MPI_Recv(remote_samrai_values.data(), remote_samrai_values.size(),
         MPI_DOUBLE, status.MPI_SOURCE, samrai_values_tag, mesh.GetComm(), &status);
      messages_received++;

      // set local element values from received cell values
      for (int cell_ind=0; cell_ind < element_inds.size(); cell_ind++)
      {
         const int samrai_values_ind = num_variables * cell_ind;
         const int element_ind = element_inds[cell_ind];
         // set node field element values
         for (int i=0; i < node_fields.size(); i++)
         {
            GridFunction& node_field_MFEM = std::get<GridFunction&>(node_fields[i]);
            const int samrai_values_node_field_ind =
               samrai_values_ind + node_field_offsets[i];
            Array<int> dof_indices;
            // the dof_indices will be ordered "by node" (aka inner loop over corners)
            node_field_MFEM.FESpace()->GetElementVDofs(element_ind, dof_indices);
            for (int d=0; d < node_field_vector_dimensions[i]; d++)
            {
               const int dof_indices_ind = corners.Size()*d;
               const int samrai_values_node_field_value_ind =
                  samrai_values_node_field_ind + dof_indices_ind;
               for (int c=0; c < corners.Size(); c++)
                  node_field_MFEM[dof_indices[dof_indices_ind+c]] =
                     remote_samrai_values[samrai_values_node_field_value_ind+c];
            }
         }
         // set cell field element values
         const int samrai_values_cell_field_ind = samrai_values_ind +
            node_field_offsets.Last();
         for (int i=0; i < cell_fields.size(); i++)
         {
            ParGridFunction& cell_field_MFEM =
               std::get<ParGridFunction&>(cell_fields[i]);
            Array<int> dof_indices;
            cell_field_MFEM.FESpace()->GetElementDofs(element_ind, dof_indices);
            cell_field_MFEM.SetSubVector(dof_indices,
               remote_samrai_values[samrai_values_cell_field_ind+i]);
         }
      }
   }

   // account for local cells that correspond to local elements
   for (int cell_ind=0; cell_ind < local_cell_info[rank].size(); cell_ind++)
   {
      const SAMRAI::pdat::CellIndex cell_index = local_cell_info[rank][cell_ind].index;
      const SAMRAI::hier::Patch& cell_patch = *local_cell_info[rank][cell_ind].patch;
      const int element_ind = local_element_inds[rank][cell_ind];
      // node field values
      for (int i=0; i < node_fields.size(); i++)
      {
         const SAMRAI::pdat::NodeData<double>& node_values =
            *SAMRAI_SHARED_PTR_CAST<SAMRAI::pdat::NodeData<double>>(
               cell_patch.getPatchData(std::get<int>(node_fields[i])));
         GridFunction &node_field_MFEM = std::get<GridFunction&>(node_fields[i]);
         Array<int> dof_indices;
         node_field_MFEM.FESpace()->GetElementVDofs(element_ind, dof_indices);
         for (int d=0; d < node_field_vector_dimensions[i]; d++)
         {
            const int dof_indices_ind = corners.Size()*d;
            for (int c=0; c < corners.Size(); c++)
            {
               const SAMRAI::pdat::NodeIndex corner_index(cell_index, corners[c]);
               node_field_MFEM[dof_indices[dof_indices_ind+c]] =
                  node_values(corner_index,d);
            }
         }
      }
      // cell field values
      for (int i=0; i < cell_fields.size(); i++)
      {
         const SAMRAI::pdat::CellData<double>& field_SAMRAI =
            *SAMRAI_SHARED_PTR_CAST<SAMRAI::pdat::CellData<double>>(
               cell_patch.getPatchData(std::get<int>(cell_fields[i])));
         ParGridFunction& cell_field_MFEM =
            std::get<ParGridFunction&>(cell_fields[i]);
         Array<int> dof_indices;
         cell_field_MFEM.FESpace()->GetElementDofs(element_ind, dof_indices);
         cell_field_MFEM.SetSubVector(dof_indices, field_SAMRAI(cell_index));
      }
   }

   MPI_Barrier(mesh.GetComm());

   mesh.NodesUpdated();
   for (int i=0; i < cell_fields.size(); i++)
   {
      cell_fields[i].second.ExchangeFaceNbrData();
   }

   return grid_functions;
}

void MeshOps::transferToSAMRAI(
   std::vector<std::pair<int, GridFunction&>> node_fields,
   std::vector<std::pair<int, ParGridFunction&>> cell_fields) const
{
   int num_variables;
   Array<int> node_field_vector_dimensions;
   Array<int> node_field_offsets;
   std::tie(num_variables, node_field_vector_dimensions, node_field_offsets)
      = ExtractBufferInfo(node_fields, cell_fields);

   // get the cell field values averaged over elements
   L2_FECollection fe_collection(0, mesh.Dimension());
   // use of const_cast here only b/c we know mesh isn't modified by fe_space or field_average
   FiniteElementSpace fe_space(&const_cast<ParMesh&>(mesh), &fe_collection);
   std::vector<GridFunction> cell_field_averages;
   for (const std::pair<int, ParGridFunction&>& field: cell_fields)
   {
      const ParGridFunction& cell_field = std::get<ParGridFunction&>(field);
      cell_field_averages.emplace_back(&fe_space);
      cell_field.GetElementAverages(cell_field_averages.back());
   }

   int rank, ranks;
   MPI_Comm_rank(mesh.GetComm(), &rank);
   MPI_Comm_size(mesh.GetComm(), &ranks);

   // send local element values to ranks that have corresponding cells
   std::vector<std::vector<double>> element_value_buffers(ranks);
   std::vector<MPI_Request> requests(ranks);
   for (int remote_rank=0; remote_rank < ranks; remote_rank++)
   {
      // local elements that correspond to local cells will be accounted for later
      if (remote_rank == rank)
         continue;

      const std::vector<int>& element_inds = local_element_inds[remote_rank];
      std::vector<double>& local_element_values = element_value_buffers[remote_rank];
      local_element_values.resize(num_variables * element_inds.size());
      for (int element_inds_ind=0; element_inds_ind < element_inds.size();
         element_inds_ind++)
      {
         const int element_values_ind = num_variables * element_inds_ind;
         const int element_ind = element_inds[element_inds_ind];
         // node field values
         for (int i=0; i < node_fields.size(); i++)
         {
            const GridFunction& node_field = std::get<GridFunction&>(node_fields[i]);
            Array<double> dof_values;
            const int element_values_node_field_ind =
               element_values_ind + node_field_offsets[i];
            for (int d=0; d < node_field_vector_dimensions[i]; d++)
            {
               node_field.GetNodalValues(element_ind, dof_values, d+1);
               MFEM_ASSERT(dof_values.Size() == corners.Size(),
                  "Received an unexpected number of node values for dimension.");
               const int element_values_node_field_value_ind =
                  element_values_node_field_ind + d*corners.Size();
               for (int c=0; c < corners.Size(); c++)
               {
                  local_element_values[element_values_node_field_value_ind+c] =
                     dof_values[c];
               }
            }
         }
         // cell field values
         const int element_values_cell_field_ind = element_values_ind +
            node_field_offsets.Last();
         for (int i=0; i < cell_fields.size(); i++)
         {
            Vector dof_values;
            cell_field_averages[i].GetElementDofValues(element_ind, dof_values);
            local_element_values[element_values_cell_field_ind+i] = dof_values[0];
         }
      }

      MPI_Isend(local_element_values.data(), local_element_values.size(),
         MPI_DOUBLE, remote_rank, element_values_tag, mesh.GetComm(),
         &requests[remote_rank]);
   }

   // receive remote element values that correspond to local cells
   int messages_received = 0;
   MPI_Status status;
   while (messages_received < ranks - 1)
   {
      MPI_Probe(MPI_ANY_SOURCE, element_values_tag, mesh.GetComm(), &status);
      const std::vector<CellInfo>& cell_info_for_rank = local_cell_info[status.MPI_SOURCE];
      std::vector<double> remote_element_values(num_variables * cell_info_for_rank.size());
      MPI_Recv(remote_element_values.data(), remote_element_values.size(),
         MPI_DOUBLE, status.MPI_SOURCE, element_values_tag, mesh.GetComm(),
         &status);
      messages_received++;

      // set local cell values from received element values
      for (int cell_ind=0; cell_ind < cell_info_for_rank.size(); cell_ind++)
      {
         const int element_values_ind = num_variables * cell_ind;
         const SAMRAI::pdat::CellIndex& cell_index = cell_info_for_rank[cell_ind].index;
         const SAMRAI::hier::Patch& cell_patch = *cell_info_for_rank[cell_ind].patch;
         // set node field values
         for (int i=0; i < node_fields.size(); i++)
         {
            SAMRAI::pdat::NodeData<double>& node_values =
               *SAMRAI_SHARED_PTR_CAST<SAMRAI::pdat::NodeData<double>>(
                  cell_patch.getPatchData(std::get<int>(node_fields[i])));
            const int element_values_node_field_ind = element_values_ind
               + node_field_offsets[i];
            // TODO: check this is the correct order using something like
            //       what is done in GetNodalValues
            //    const FiniteElement *FElem = fes->GetFE(i);
            //    const IntegrationRule *ElemVert =
            //    Geometries.GetVertices(FElem->GetGeomType());
            for (int d=0; d < node_field_vector_dimensions[i]; d++)
            {
               const int element_values_node_field_value_ind =
                  element_values_node_field_ind + corners.Size()*d;
               for (int c=0; c < corners.Size(); c++)
               {
                  const SAMRAI::pdat::NodeIndex corner_index(cell_index, corners[c]);
                  node_values(corner_index,d) =
                     remote_element_values[element_values_node_field_value_ind+c];
               }
            }
         }
         // set cell field values
         const int element_values_cell_field_ind = element_values_ind +
            node_field_offsets.Last();
         for (int i=0; i < cell_fields.size(); i++)
         {
            SAMRAI::pdat::CellData<double>& cell_values =
               *SAMRAI_SHARED_PTR_CAST<SAMRAI::pdat::CellData<double>>(
                  cell_patch.getPatchData(std::get<int>(cell_fields[i])));
            cell_values(cell_index) =
               remote_element_values[element_values_cell_field_ind+i];
         }
      }
   }

   // account for local elements that correspond to local cells
   for (int element_inds_ind=0; element_inds_ind < local_element_inds[rank].size();
      element_inds_ind++)
   {
      const int element_ind = local_element_inds[rank][element_inds_ind];
      const SAMRAI::pdat::CellIndex& cell_index = local_cell_info[rank][element_inds_ind].index;
      const SAMRAI::hier::Patch& cell_patch = *local_cell_info[rank][element_inds_ind].patch;
      // set node field values
      for (int i=0; i < node_fields.size(); i++)
      {
         SAMRAI::pdat::NodeData<double>& node_values =
            *SAMRAI_SHARED_PTR_CAST<SAMRAI::pdat::NodeData<double>>(
               cell_patch.getPatchData(std::get<int>(node_fields[i])));
         GridFunction &node_field_MFEM = std::get<GridFunction&>(node_fields[i]);
         Array<int> dof_indices;
         node_field_MFEM.FESpace()->GetElementVDofs(element_ind, dof_indices);
         // TODO: check this is the correct order using something like
         //       what is done in GetNodalValues
         //    const FiniteElement *FElem = fes->GetFE(i);
         //    const IntegrationRule *ElemVert =
         //    Geometries.GetVertices(FElem->GetGeomType());
         for (int d=0; d < node_field_vector_dimensions[i]; d++)
         {
            const int dof_indices_ind = corners.Size()*d;
            for (int c=0; c < corners.Size(); c++)
            {
               const SAMRAI::pdat::NodeIndex corner_index(cell_index, corners[c]);
               node_values(corner_index,d) =
                  node_field_MFEM[dof_indices[dof_indices_ind+c]];
            }
         }
      }
      // set cell field values
      for (int i=0; i < cell_fields.size(); i++)
      {
         SAMRAI::pdat::CellData<double>& cell_values =
            *SAMRAI_SHARED_PTR_CAST<SAMRAI::pdat::CellData<double>>(
               cell_patch.getPatchData(std::get<int>(cell_fields[i])));
         Vector dof_values;
         cell_field_averages[i].GetElementDofValues(element_ind, dof_values);
         cell_values(cell_index) = dof_values[0];
      }
   }

   MPI_Barrier(mesh.GetComm());
}

void MeshOps::synchronizeToHierarchy()
{
   // restore mesh topology to index space
   mesh_grid_function->SetFromTrueDofs(mesh_index_space_tdofs);

   // update global patch info and obtain corresponding patch bounds
   std::vector<PatchLevelBounds> global_patch_bounds;
   RemoveOldPatchesFromGlobalPatchInfo();
   AddNewPatchesToGlobalPatchInfo();
   GetGlobalPatchBounds(global_patch_bounds);

   // derefine and then refine mesh
   DerefineMesh(global_patch_bounds);
   RefineMesh(global_patch_bounds);

   // create new transfer maps
   CreateTransferMaps();

   // update the finite element spaces and mesh grid function
   for (auto& entry : fe_spaces_cell)
   {
      entry.second->Update();
   }
   for (auto& entry : fe_spaces_node)
   {
      entry.second->Update();
   }
   mesh_grid_function->Update();

}

std::tuple<int,Array<int>,Array<int>> MeshOps::ExtractBufferInfo(
   std::vector<std::pair<int, GridFunction&>> node_fields,
   std::vector<std::pair<int, ParGridFunction&>> cell_fields) const
{
   Array<int> node_field_vector_dimensions;
   Array<int> node_field_offsets;
   for (const std::pair<int, GridFunction&>& field : node_fields)
   {
      const GridFunction& node_field = std::get<GridFunction&>(field);
      MFEM_ASSERT(node_field.FESpace()->GetOrdering() == Ordering::byNODES,
            "GridFunction for node fields must have FE space ordered by node.");
      node_field_vector_dimensions.Append(field.second.FESpace()->GetVectorDim());
      node_field_offsets.Append(node_field_vector_dimensions.Last() * corners.Size());
   }
   node_field_offsets.Prepend(0);
   node_field_offsets.PartialSum();
   const int num_variables = node_field_offsets.Last() + cell_fields.size();

   return {num_variables, node_field_vector_dimensions, node_field_offsets};
}

Array<SAMRAI::pdat::NodeIndex::Corner> MeshOps::getCorners(const unsigned dimension)
{
   constexpr SAMRAI::pdat::NodeIndex::Corner corners1D[] =
      {SAMRAI::pdat::NodeIndex::Left, SAMRAI::pdat::NodeIndex::Right};
   constexpr SAMRAI::pdat::NodeIndex::Corner corners2D[] =
      {SAMRAI::pdat::NodeIndex::LowerLeft, SAMRAI::pdat::NodeIndex::LowerRight,
         SAMRAI::pdat::NodeIndex::UpperRight, SAMRAI::pdat::NodeIndex::UpperLeft};
   constexpr SAMRAI::pdat::NodeIndex::Corner corners3D[] =
      {SAMRAI::pdat::NodeIndex::LLL, SAMRAI::pdat::NodeIndex::ULL,
         SAMRAI::pdat::NodeIndex::UUL, SAMRAI::pdat::NodeIndex::LUL,
         SAMRAI::pdat::NodeIndex::LLU, SAMRAI::pdat::NodeIndex::ULU,
         SAMRAI::pdat::NodeIndex::UUU, SAMRAI::pdat::NodeIndex::LUU};

   switch (dimension)
   {
      case 1: return Array<SAMRAI::pdat::NodeIndex::Corner>(corners1D);
      case 2: return Array<SAMRAI::pdat::NodeIndex::Corner>(corners2D);
      case 3: return Array<SAMRAI::pdat::NodeIndex::Corner>(corners3D);
      default:
         MFEM_ABORT("Invalid dimension value: " << dimension);
   }
}

SAMRAI::hier::Index MeshOps::toIndex(const Vector& vector)
{
   return SAMRAI::hier::Index(std::vector<int>(vector.begin(),vector.end()));
}

SAMRAI::hier::Index MeshOps::toIndex(const Array<int>& array,
   const unsigned dim, const int start)
{
   MFEM_VERIFY(start + dim <= array.Size(), "size mismatch");

   const int* begin = array.begin() + start;
   const int* end = begin + dim;
   return SAMRAI::hier::Index(std::vector<int>(begin, end));
}

Vector MeshOps::toVector(const SAMRAI::hier::IntVector& vector)
{
   Vector result(vector.getDim().getValue());
   for (int i=0; i < result.Size(); i++)
      result[i] = vector[i];
   return result;
}

template<typename PODType>
MeshOps::BlockArray<PODType>::BlockArray(const unsigned block_size,
   const unsigned num_blocks) : block_size(block_size),num_blocks(num_blocks),
   data(num_blocks*block_size) {}

template<typename PODType>
void MeshOps::BlockArray<PODType>::SetBlock(const unsigned index,
   const Array<PODType> &values)
{
   MFEM_ASSERT(values.Size() == block_size,
      "Provided array is not the correct size.");
   // TODO: consider replacing with memcpy
   for(unsigned i=0; i < block_size; i++)
   {
      data[index*block_size + i] = values[i];
   }
}

template<typename PODType>
Array<PODType> MeshOps::BlockArray<PODType>::GetBlock(const unsigned index) const
{
   Array<PODType> values(block_size);
   // TODO: consider replacing with memcpy
   for(unsigned i=0; i < block_size; i++)
   {
      values[i] = data[index*block_size + i];
   }
   return values;
}

template<typename PODType>
unsigned MeshOps::BlockArray<PODType>::NumBlocks() const
{
   return num_blocks;
}

template<typename PODType>
void* MeshOps::BlockArray<PODType>::GetData()
{
   return data.GetData();
}

template<typename PODType>
int MeshOps::BlockArray<PODType>::Size() const
{
   return data.Size();
}

template<typename PODType>
void MeshOps::BlockArray<PODType>::GetElementCounts(const Array<PODType> &block_counts,
   Array<int> &element_counts) const
{
   element_counts.SetSize(block_counts.Size());
   for (int i=0; i < block_counts.Size(); i++)
      element_counts[i] = block_counts[i] * block_size;
}

MeshOps::PatchInfo MeshOps::PatchInfo::FromArray(const Array<int>& values)
{
   MFEM_ASSERT((values.Size() - 2) % 2 == 0,
      "Provided array is not a valid size.")
   const unsigned dimension = (values.Size()-2) / 2;
   const int rank = values[0];
   const int level_number = values[1];
   const SAMRAI::hier::Index lower_index = toIndex(values, dimension, 2);
   const SAMRAI::hier::Index upper_index = toIndex(values, dimension, 2+dimension);
   return PatchInfo(rank, level_number, lower_index, upper_index);
}

MeshOps::PatchInfo::PatchInfo(const int rank_, const int level_number_,
   const SAMRAI::hier::Index lower_index_, const SAMRAI::hier::Index upper_index_) :
   rank(rank_), level_number(level_number_), lower_index(lower_index_),
   upper_index(upper_index_) {}

Array<int> MeshOps::PatchInfo::AsArray() const
{
   const unsigned dimension = lower_index.getDim().getValue();
   Array<int> array(Size(dimension));
   array[0] = rank;
   array[1] = level_number;
   // TODO: consider replacing loops with memcpy
   for (int d=0; d < dimension; d++)
      array[2+d] = lower_index(d);
   for (int d=0; d < dimension; d++)
      array[2+dimension+d] = upper_index(d);
   return array;
}

unsigned MeshOps::PatchInfo::Size(const unsigned dimension)
{
   return 2 + 2*dimension;
}

bool MeshOps::PatchInfo::operator==(const PatchInfo& other) const
{
   return rank == other.rank && level_number == other.level_number &&
            lower_index == other.lower_index && upper_index == other.upper_index;
}

MeshOps::ElementInfo::ElementInfo(const int level_number_,
   const SAMRAI::hier::Index index_) : level_number(level_number_),
   index(index_) {}

MeshOps::ElementInfo MeshOps::ElementInfo::FromArray(const Array<int>& values)
{
   MFEM_ASSERT(values.Size() > 1, "Provided array is not a valid size.");
   const unsigned dimension = values.Size()-1;
   const int level_number = values[0];
   const SAMRAI::hier::Index index = toIndex(values, dimension, 1);
   return ElementInfo(level_number, index);
}

Array<int> MeshOps::ElementInfo::AsArray() const
{
   const unsigned dimension = index.getDim().getValue();
   Array<int> array(Size(dimension));
   array[0] = level_number;
   // TODO: consider replacing loops with memcpy
   for (int d=0; d < dimension; d++)
      array[1+d] = index(d);
   return array;
}

unsigned MeshOps::ElementInfo::Size(const unsigned dimension)
{
   return 1 + dimension;
}

}
