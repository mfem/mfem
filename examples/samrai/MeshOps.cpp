#include "SAMRAI/hier/PatchLevel.h"

#include "MeshOps.hpp"

namespace mfem
{

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

inline SAMRAI::hier::Index MeshOps::toIndex(const Vector& vector)
{
   return SAMRAI::hier::Index(std::vector<int>(vector.begin(),vector.end()));
}

inline SAMRAI::hier::Index MeshOps::toIndex(const Array<int>& array,
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

MeshOps::MeshOps(std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy,
   MPI_Comm communicator) :
   hierarchy(hierarchy), corners(getCorners( hierarchy->getDim().getValue())),
   fe_collection_node(1, hierarchy->getDim().getValue()),
   fe_collection_cell(0, hierarchy->getDim().getValue())
{
   int rank, ranks;
   MPI_Comm_rank(communicator, &rank);
   MPI_Comm_size(communicator, &ranks);
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
   MPI_Allreduce(MPI_IN_PLACE, lowerData, dim, MPI_INT, MPI_MIN, communicator);
   MPI_Allreduce(MPI_IN_PLACE, upperData, dim, MPI_INT, MPI_MAX, communicator);
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
         MFEM_ABORT("NDIM not valid");
   }
   serial_mesh.EnsureNCMesh();
   mesh = ParMesh(communicator, serial_mesh);
   serial_mesh.Clear();

   // serialize local patch data
   unsigned local_patch_count = 0;
   for (int level_num = 0; level_num <= hierarchy->getFinestLevelNumber();
      level_num++)
   {
      local_patch_count +=
         hierarchy->getPatchLevel(level_num)->getLocalNumberOfPatches();
   }
   // patch data block is (level number, lower index, upper index)
   const unsigned patch_data_size = 1 + 2*mesh.Dimension();
   BlockArray<int> local_patch_data(patch_data_size, local_patch_count);
   unsigned patch_ind = 0;
   Array<int> patch_data_values(patch_data_size);
   for (int level_number = 0; level_number <= hierarchy->getFinestLevelNumber();
      level_number++)
   {
      patch_data_values[0] = level_number;
      std::shared_ptr<SAMRAI::hier::PatchLevel> patch_level =
         hierarchy->getPatchLevel(level_number);
      for (SAMRAI::hier::PatchLevel::iterator patch_iter=patch_level->begin();
            patch_iter != patch_level->end(); patch_iter++)
      {
         // TODO: consider replacing loops with memcpy
         const SAMRAI::hier::Index& lower = patch_iter->getBox().lower();
         for (int d=0; d < mesh.Dimension(); d++)
            patch_data_values[1+d] = lower(d);
         const SAMRAI::hier::Index& upper = patch_iter->getBox().upper();
         for (int d=0; d < mesh.Dimension(); d++)
            patch_data_values[1+mesh.Dimension()+d] = upper(d);
         local_patch_data.SetBlock(patch_ind++, patch_data_values);
      }
   }

   // gather the number of patches from each rank
   Array<int> global_patch_counts(ranks);
   MPI_Allgather(&local_patch_count, 1, MPI_INT, global_patch_counts.GetData(),
      1, MPI_INT, mesh.GetComm());

   // gather the global patch data on each rank
   BlockArray<int> global_patch_data(patch_data_size, global_patch_counts.Sum());
   {
      Array<int> gather_counts;
      global_patch_data.GetElementCounts(global_patch_counts, gather_counts);
      Array<int> gather_offsets = gather_counts;
      gather_offsets.Prepend(0);
      gather_offsets.PartialSum();
      MPI_Allgatherv(local_patch_data.GetData(),
         local_patch_data.Size(), MPI_INT,
         global_patch_data.GetData(), gather_counts.GetData(),
         gather_offsets.GetData(), MPI_INT, mesh.GetComm());
   }

   // unserialize patch data and note rank that has each patch
   std::vector<PatchData> global_patch_list;
   global_patch_list.reserve(global_patch_data.NumBlocks());
   int current_rank = 0;
   int current_patch_count = 0;
   for (int patch_ind = 0; patch_ind < global_patch_data.NumBlocks(); patch_ind++)
   {
      if (current_patch_count == global_patch_counts[current_rank])
      {
         current_rank++;
         current_patch_count = 0;
      }
      global_patch_data.GetBlock(patch_ind, patch_data_values);
      global_patch_list.push_back({current_rank, patch_data_values[0],
         toIndex(patch_data_values, mesh.Dimension(), 1),
         toIndex(patch_data_values, mesh.Dimension(), 1+mesh.Dimension())});
      current_patch_count++;
   }

   // organize patch bounds by level
   using PatchLevelBounds = std::vector<std::pair<const SAMRAI::hier::Index, const SAMRAI::hier::Index>>;
   std::vector<PatchLevelBounds> global_patch_bounds(hierarchy->getFinestLevelNumber()+1);
   for (int patch_ind=0; patch_ind < global_patch_list.size(); patch_ind++)
   {
      const PatchData& current_patch = global_patch_list[patch_ind];
      const int level_number = current_patch.level_number;
      global_patch_bounds[level_number].emplace_back(
         current_patch.lower_index, current_patch.upper_index);
   }

   // refine mesh at each level
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

         // check if center is in any refinement patches from current level
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
      // refinement elements further for 3:1 refinement
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
               // TODO: support NDIM
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

   fe_spaces_node[mesh.Dimension()] =
      std::make_unique<ParFiniteElementSpace>(&mesh, &fe_collection_node, mesh.Dimension());
   mesh_grid_function =
      std::make_shared<ParGridFunction>(fe_spaces_node[mesh.Dimension()].get());
   mesh.SetNodalGridFunction(mesh_grid_function.get());

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
      for (const PatchData& patch : global_patch_list)
      {
         if (patch.level_number == level_number &&
               patch.lower_index <= index && index <= patch.upper_index)
         {
            index_rank = patch.rank;
            break;
         }
      }
      MFEM_VERIFY(index_rank >= 0, "Index not found in global patch list");
      local_element_info[index_rank].push_back({level_number, index});
      local_element_inds[index_rank].push_back(element_ind);
   }

   // send local element info to ranks that have corresponding cells
   const unsigned element_info_size = ElementInfo::Size(mesh.Dimension());
   Array<int> element_info_values(element_info_size);
   std::vector<std::unique_ptr<BlockArray<int>>> element_info_buffers(ranks);
   std::vector<MPI_Request> requests(ranks);
   for (int remote_rank=0; remote_rank < ranks; remote_rank++)
   {
      // local elements that correspond to local cells will be accounted for later
      if (remote_rank == rank)
         continue;

      const std::vector<ElementInfo>& element_info_for_rank
         = local_element_info[remote_rank];

      element_info_buffers[remote_rank] =
         std::make_unique<BlockArray<int>>(element_info_size, element_info_for_rank.size());
      for (unsigned data_ind=0; data_ind < element_info_for_rank.size(); data_ind++)
      {
         element_info_buffers[remote_rank]->SetBlock(data_ind,
            element_info_for_rank[data_ind].AsArray());
      }
      MPI_Isend(element_info_buffers[remote_rank]->GetData(),
         element_info_buffers[remote_rank]->Size(), MPI_INT, remote_rank,
         element_info_tag, mesh.GetComm(), &requests[remote_rank]);
   }

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

      for (int i=0; i < remote_element_info.NumBlocks(); i++)
      {
         remote_element_info.GetBlock(i, element_info_values);
         const int level_number = element_info_values[0];
         const SAMRAI::hier::Index& index =
            toIndex(element_info_values, mesh.Dimension(), 1);
         std::shared_ptr<SAMRAI::hier::PatchLevel> patch_level =
            hierarchy->getPatchLevel(level_number);
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
         MFEM_VERIFY(patch != nullptr, "Level " << level_number
            << ", index " << index << " received from rank " << status.MPI_SOURCE
            << " is not on any patches local to rank " << rank);
         local_cell_info[status.MPI_SOURCE].push_back({SAMRAI::pdat::CellIndex(index), patch});
      }
   }

   // account for local elements that correspond to local cells
   for (const ElementInfo& element_info : local_element_info[rank])
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
      MFEM_VERIFY(patch != nullptr, "Index is not on any local patches");
      local_cell_info[rank].push_back({SAMRAI::pdat::CellIndex(index), patch});
   }

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
      = extractBufferInfo(node_fields, cell_fields);

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
   int count;
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
      = extractBufferInfo(node_fields, cell_fields);

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
   int count;
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
   }

   MPI_Barrier(mesh.GetComm());
}

void MeshOps::reconstructField(const ParGridFunction& src, ParGridFunction& dst)
{
   // todo: improve checking when dst is L2, p=0
   if (dst.FESpace()->GetOrder(0) == 0)
   {
      dst = src;
      return;
   }

   MassIntegrator mass;
   const FiniteElementSpace& src_fe_space = *(src.FESpace()); // u_hat space
   const FiniteElementSpace& dst_fe_space = *(dst.FESpace()); // u space

   std::unique_ptr<HypreParMatrix> matrix;

   // compute <{u_hat} (xhat dot n), psi_hat>_E
   VectorConstantCoefficient xhat(Vector({1.0,0.0}));
   Vector b_xhat(mesh.GetNE());
   ParBilinearForm B_xhat(src.ParFESpace());
   B_xhat.AddInteriorFaceIntegrator(new DGTraceIntegrator(xhat, 1.0, 0.0));
   B_xhat.AddBdrFaceIntegrator(new DGTraceIntegrator(xhat, 2.0, 0.0)); // note the 2 enforces du/dx=0 at the x boundaries
   B_xhat.Assemble();
   B_xhat.Finalize();
   matrix = std::unique_ptr<HypreParMatrix>(B_xhat.ParallelAssemble());
   matrix->Mult(*src.GetTrueDofs(), b_xhat);

   // compute <{u_hat} (yhat dot n), psi_hat>_E
   VectorConstantCoefficient yhat(Vector({0.0,1.0}));
   Vector b_yhat(mesh.GetNE());
   ParBilinearForm B_yhat(src.ParFESpace());
   B_yhat.AddInteriorFaceIntegrator(new DGTraceIntegrator(yhat, 1.0, 0.0));
   B_yhat.AddBdrFaceIntegrator(new DGTraceIntegrator(yhat, 2.0, 0.0)); // note the 2 enforces du/dy=0 at the y boundaries
   B_yhat.Assemble();
   B_yhat.Finalize();
   matrix = std::unique_ptr<HypreParMatrix>(B_yhat.ParallelAssemble());
   matrix->Mult(*src.GetTrueDofs(), b_yhat);

   MixedDirectionalDerivativeIntegrator partial_x(xhat);
   MixedDirectionalDerivativeIntegrator partial_y(yhat);
   for (int element_ind=0; element_ind < mesh.GetNE(); element_ind++)
   {
      const FiniteElement& src_element = *(src_fe_space.GetFE(element_ind));
      const FiniteElement& dst_element = *(dst_fe_space.GetFE(element_ind));
      ElementTransformation& transform = *(src_fe_space.GetElementTransformation(element_ind));
      DenseMatrix A(dst_element.GetDof());
      Vector b(dst_element.GetDof());
      DenseMatrix Arow;

      // enforce (u, psi_hat)_E = ({u_hat}, psi_hat)_E
      mass.AssembleElementMatrix2(dst_element, src_element, transform, Arow);
      A.SetSubMatrix(0, 0, Arow);
      Vector bmean(b, 0, src_element.GetDof());
      DenseMatrix Bmean;
      mass.AssembleElementMatrix(src_element, transform, Bmean);
      Vector src_dof_values;
      src.GetElementDofValues(element_ind, src_dof_values);
      Bmean.Mult(src_dof_values, bmean);

      // enforce (div[ u xhat ], psi_hat)_E = <{u_hat} (xhat dot n), psi_hat>_E,
      // i.e., (du/dx, psi_hat)_E = <{u_hat} (xhat dot n), psi_hat>_E
      partial_x.AssembleElementMatrix2(dst_element, src_element, transform, Arow);
      A.SetSubMatrix(1, 0, Arow);
      b[1] = b_xhat[element_ind];

      // enforce (div[ u yhat ], psi_hat)_E = <{u_hat} (yhat dot n), psi_hat>_E,
      // i.e., (du/dy, psi_hat)_E = <{u_hat} (yhat dot n), psi_hat>_E
      partial_y.AssembleElementMatrix2(dst_element, src_element, transform, Arow);
      A.SetSubMatrix(2, 0, Arow);
      b[2] = b_yhat[element_ind];

      // enforce (div[ (xhat \otimes yhat) grad[u] ], psi_hat)_E = 0, i.e.,
      // < du/dy (xhat dot n), psi_hat>_E = 0
      // TODO: replace with actual face integration
      A(3,0) = 1.0;
      A(3,1) = -1.0;
      A(3,2) = -1.0;
      A(3,3) = 1.0;
      b[3] = 0.0;

      // solve for u dof values
      A.Invert();
      mfem::Vector solution(dst_element.GetDof());
      A.Mult(b, solution);
      Array<int> dst_dof_indices;
      dst_fe_space.GetElementDofs(element_ind, dst_dof_indices);
      dst.SetSubVector(dst_dof_indices, solution);
   }
   dst.ExchangeFaceNbrData();
}

std::tuple<int,Array<int>,Array<int>> MeshOps::extractBufferInfo(
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

}
