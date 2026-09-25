
#ifndef MFEM_SAMRAICOUPLINGMANAGER
#define MFEM_SAMRAICOUPLINGMANAGER

#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/pdat/NodeData.h"
#include "SAMRAI/pdat/CellData.h"

#include "mfem.hpp"

namespace mfem
{

class SAMRAICouplingManager
{
public:

   enum BDR_ATTRIBUTE {Ylower=1, Xupper=2, Yupper=3, Xlower=4};

   SAMRAICouplingManager(std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy);

   // read-only access to current MFEM mesh object (meant for temporary access)
   const ParMesh& GetMesh() const { return *mesh; }

   // create an "unmanaged" finite element space on the managed MFEM mesh, which
   // can be used for MFEM-only fields **the space becomes invalid after call to
   // SynchronizeMeshToHierarchy**.
   std::unique_ptr<ParFiniteElementSpace> CreateFESpace(
      FiniteElementCollection* fe_collection, int dim=1)
   {
      return std::make_unique<ParFiniteElementSpace>(mesh.get(), fe_collection, dim);
   }

   // specify the MFEM mesh topology by an externally provided grid function
   // defined on a finite element space created by CreateFESpace
   void SetMeshGridFunction(std::shared_ptr<GridFunction> grid_function)
   {
      mesh->SetNodalGridFunction(grid_function.get());
      mesh_grid_function = grid_function;
   }

   // synchronizes the MFEM mesh to the SAMRAI grid by either updating the
   // existing MFEM mesh or creating a new mesh, with all "managed" finite
   // element spaces updated in either case (**"unmanaged" spaces created by
   // CreateFESpace become invalid)**
   void SynchronizeMeshToHierarchy(bool create_new_mesh=false);

   // transfer SAMRAI node positions and specified node and cell values to the
   // MFEM mesh nodes and specified grid function. This method assumes the
   // following about the arguments:
   //   1) the PatchHierarchy object under position and the SAMRAI field ids
   //      is the same one passed to the MeshOps constructor (and has not
   //      changed since then)
   //   2) the SAMRAI node fields have depth NDIM
   //   3) the SAMRAI cell fields have depth 1
   std::vector<std::unique_ptr<ParGridFunction>> TransferToMFEM(
      const int position_id, const std::vector<int>& node_ids,
      const std::vector<int>& cell_ids);

   // transfer MFEM grid function values to SAMRAI cell values (considers cell
   // values == element averages). This method assumes the following about the
   // arguments:
   //   1) the PatchHierarchy object under the CellData<double> object is the
   //      same one passed to the MeshOps constructor (and has not changed since
   //      then)
   //   2) the finite element spaces under the MFEM grid functions were created
   //      using MeshOps::createFESpace()
   void TransferToSAMRAI(
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

   // similar to other TransferToSAMRAI method except the MFEM mesh position is
   // also transferred (assumes an external mesh grid function containing the
   // mesh topology, specified by SetMeshGridFunction, has been changed)
   void TransferToSAMRAI(int position_id,
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields)
   {
      mesh->NewNodes(*mesh_grid_function);
      node_fields.emplace_back(position_id,
         const_cast<GridFunction&>(*mesh_grid_function));
      TransferToSAMRAI(node_fields, cell_fields);
   }

private:

   /***** general utility methods *****/

   static inline SAMRAI::hier::Index ToIndex(const Vector& vector);

   static inline SAMRAI::hier::Index ToIndex(const Array<int>& array,
      const unsigned dim, const int start);

   static Vector ToVector(const SAMRAI::hier::IntVector& vector);

   static Vector GetElementDimensions(Mesh& mesh, const int element_ind);

   /***** utility classes *****/

   // an Array wrapper for data serialized (by blocks) for MPI communication
   template<typename PODType>
   class BlockArray
   {
      const unsigned block_size, num_blocks;
      Array<PODType> data;

   public:

      BlockArray(const unsigned block_size, const unsigned num_blocks);

      void SetBlock(const unsigned index, const Array<PODType> &values);

      Array<PODType> GetBlock(const unsigned index) const;

      unsigned NumBlocks() const;

      void* GetData();

      int Size() const;

      void GetElementCounts(const Array<PODType> &block_counts, Array<int> &element_counts) const;

   };

   // class storing SAMRAI patch information for MFEM mesh mirroring
   struct PatchInfo
   {
      int rank;
      int level_number;
      SAMRAI::hier::Index lower_index;
      SAMRAI::hier::Index upper_index;

      PatchInfo(const int rank_, const int level_number_,
         const SAMRAI::hier::Index lower_index_,
         const SAMRAI::hier::Index upper_index_);

      Array<int> AsArray() const;

      bool operator==(const PatchInfo& other) const;

      static unsigned Size(const unsigned dimension);

      static PatchInfo FromArray(const Array<int>& values);
   };

   // class storing MFEM element information for SAMRAI<=>MFEM data transfers
   struct ElementInfo
   {
      int level_number;
      SAMRAI::hier::Index index;

      ElementInfo(const int level_number_, const SAMRAI::hier::Index index_);

      Array<int> AsArray() const;

      static unsigned Size(const unsigned dimension);

      static ElementInfo FromArray(const Array<int>& values);
   };

   // class storing SAMRAI cell information for SAMRAI<=>MFEM data transfers
   struct CellInfo
   {
      SAMRAI::pdat::CellIndex index;
      std::shared_ptr<SAMRAI::hier::Patch> patch;
   };

   /***** MPI utility methods *****/

   // extracts the following information for the gather/scatter buffer:
   //   1) the number of values per element
   //   2) the vector dimension of each node field
   //   3) the offsets with a specified element block of buffer for each node field
   std::tuple<int,Array<int>,Array<int>> ExtractBufferInfo(
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

   void GatherGlobalPatchInfo(const std::vector<PatchInfo>& local_patch_info,
      std::vector<PatchInfo>& gathered_patch_info) const;

   using PatchLevelBounds = std::vector<std::pair<const SAMRAI::hier::Index, const SAMRAI::hier::Index>>;
   void GetGlobalPatchBounds(std::vector<PatchLevelBounds>& global_patch_bounds) const;

   /***** SAMRAI grid state update methods *****/

   void AddNewPatchesToGlobalPatchInfo();

   void RemoveOldPatchesFromGlobalPatchInfo();

   /***** MFEM mesh creation and update methods *****/

   void CreateMesh();

   void UpdateFiniteElementSpaces();

   void DerefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds);

   void RefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds);

   void CreateTransferMaps();

   /***** MPI tags *****/

   const int element_info_tag = 0;
   const int samrai_values_tag = 1;
   const int element_values_tag = 2;

   /***** SAMRAI state variables *****/

   std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy;
   std::vector<PatchInfo> global_patch_info;

   /***** SAMRAI<=>MFEM data transfer variables *****/

   const Array<SAMRAI::pdat::NodeIndex::Corner>& corners;

   const Array<SAMRAI::pdat::NodeIndex::Corner> corners1D{
      SAMRAI::pdat::NodeIndex::Left, SAMRAI::pdat::NodeIndex::Right};
   const Array<SAMRAI::pdat::NodeIndex::Corner> corners2D{
      SAMRAI::pdat::NodeIndex::LowerLeft, SAMRAI::pdat::NodeIndex::LowerRight,
      SAMRAI::pdat::NodeIndex::UpperRight, SAMRAI::pdat::NodeIndex::UpperLeft};
   const Array<SAMRAI::pdat::NodeIndex::Corner> corners3D{
      SAMRAI::pdat::NodeIndex::LLL, SAMRAI::pdat::NodeIndex::ULL,
      SAMRAI::pdat::NodeIndex::UUL, SAMRAI::pdat::NodeIndex::LUL,
      SAMRAI::pdat::NodeIndex::LLU, SAMRAI::pdat::NodeIndex::ULU,
      SAMRAI::pdat::NodeIndex::UUU, SAMRAI::pdat::NodeIndex::LUU};

   // (rank) -> {CellInfo for local cell that corresponds to element on rank}
   std::vector<std::vector<CellInfo>> local_cell_info;
   // (rank) -> {local element ind that corresponds to cell on rank}
   std::vector<std::vector<int>> local_element_inds;

   /***** MFEM variables *****/

   std::unique_ptr<ParMesh> mesh;
   std::shared_ptr<GridFunction> mesh_grid_function;
   Vector mesh_index_space_tdofs;
   H1_FECollection fe_collection_node;
   L2_FECollection fe_collection_cell;

   // maps are from field dimension to finite element space
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_cell;
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_node;

};

}

#endif
