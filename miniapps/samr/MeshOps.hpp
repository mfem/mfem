
#ifndef included_MFEMMeshOps
#define included_MFEMMeshOps

#include "boost/shared_ptr.hpp"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/pdat/NodeData.h"
#include "SAMRAI/pdat/CellData.h"

#include "mfem.hpp"

namespace mfem
{

class MeshOps
{
public:

   enum BDR_ATTRIBUTE {Ylower=1, Xupper=2, Yupper=3, Xlower=4};

   MeshOps(std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy);

   const ParMesh& getMesh() const { return mesh; }

   void setMeshGridFunction(std::shared_ptr<GridFunction> grid_function);

   std::unique_ptr<ParFiniteElementSpace> createFESpace(
      FiniteElementCollection& fe_collection, int dim=1);

   // transfer SAMRAI node positions and specified node and cell values to the
   // MFEM mesh nodes and specified grid function. This method assumes the
   // following about the arguments:
   //   1) the PatchHierarchy object under position and the SAMRAI field ids
   //      is the same one passed to the MeshOps constructor (and has not
   //      changed since then)
   //   2) the SAMRAI node fields have depth NDIM
   //   3) the SAMRAI cell fields have depth 1
   std::vector<std::unique_ptr<ParGridFunction>> transferToMFEM(
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
   void transferToSAMRAI(
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

   // similar to other trasnferToSAMRAI method except the MFEM mesh position is
   // also transfered (assumed the mesh topology has been changed by an external
   // mesh grid function specified by setMeshGridFunction)
   void transferToSAMRAI(int position_id,
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields)
   {
      mesh.NewNodes(*mesh_grid_function);
      node_fields.emplace_back(position_id,
         const_cast<GridFunction&>(*mesh_grid_function));
      transferToSAMRAI(node_fields, cell_fields);
   }

   void synchronizeToHierarchy();

private:


   /***** helper functions and utility classes *****/

   static Array<SAMRAI::pdat::NodeIndex::Corner> getCorners(const unsigned dimension);

   static inline SAMRAI::hier::Index toIndex(const Vector& vector);

   static inline SAMRAI::hier::Index toIndex(const Array<int>& array,
      const unsigned dim, const int start);

   static Vector toVector(const SAMRAI::hier::IntVector& vector);

   // extracts the following information for the gather/scatter buffer:
   //   1) the number of values per element
   //   2) the vector dimension of each node field
   //   3) the offsets with a specified element block of buffer for each node field
   std::tuple<int,Array<int>,Array<int>> ExtractBufferInfo(
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

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

   // TODO: consider creating a SerializableInfo template that is then
   // specialized for PatchInfo and ElementInfo with structs of the individual
   // information

   struct PatchInfo
   {
      // TODO: can consider removing dimension variable
      const unsigned dimension;
      int rank;
      int level_number;
      SAMRAI::hier::Index lower_index;
      SAMRAI::hier::Index upper_index;

      PatchInfo(const int rank_, const int level_number_,
         const SAMRAI::hier::Index lower_index_,
         const SAMRAI::hier::Index upper_index_);

      PatchInfo(const unsigned dimension);

      PatchInfo(const Array<int>& values);

      const Array<int>& AsArray() const;

      unsigned Size() const;

      bool operator==(const PatchInfo& other) const;

   private:

      mutable Array<int> array;
   };

   struct ElementInfo
   {
      // TODO: can consider removing dimension variable
      const int dimension;
      int level_number;
      SAMRAI::hier::Index index;

      ElementInfo(const unsigned dimension);

      ElementInfo(const Array<int>& values);

      const Array<int>& AsArray() const;

      unsigned Size() const;

   private:

      mutable Array<int> array;

   };

   void GatherGlobalPatchInfo(const std::vector<PatchInfo>& local_patch_info,
      std::vector<PatchInfo>& gathered_patch_info) const;

   using PatchLevelBounds = std::vector<std::pair<const SAMRAI::hier::Index, const SAMRAI::hier::Index>>;
   void GetGlobalPatchBounds(std::vector<PatchLevelBounds>& global_patch_bounds) const;


   // TODO: come up with better comment
   /***** member functions and variables *****/

   std::vector<PatchInfo> global_patch_info;

   void AddNewPatchesToGlobalPatchInfo();

   void RemoveOldPatchesFromGlobalPatchInfo();

   void DerefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds);

   void RefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds);

   void CreateTransferMaps();


   const int element_info_tag = 0;
   const int samrai_values_tag = 1;
   const int element_values_tag = 2;

   std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy;
   const Array<SAMRAI::pdat::NodeIndex::Corner> corners;

   ParMesh mesh;
   std::shared_ptr<GridFunction> mesh_grid_function;
   Vector mesh_index_space_tdofs;

   H1_FECollection fe_collection_node;
   L2_FECollection fe_collection_cell;
   // maps are from field dimension to finite element space
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_cell;
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_node;

   /***** transfer maps *****/

   struct CellInfo
   {
      SAMRAI::pdat::CellIndex index;
      std::shared_ptr<SAMRAI::hier::Patch> patch;
   };
   std::vector<std::vector<CellInfo>> local_cell_info; // (rank) -> {CellInfo for local cell that corresponds to element on rank}
   std::vector<std::vector<int>> local_element_inds; // (rank) -> {local element ind that corresponds to cell on rank}

};

}

#endif
