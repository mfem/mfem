
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

private:

   /***** helper functions *****/

   // extracts the following information for the gather/scatter buffer:
   //   1) the number of values per element
   //   2) the vector dimension of each node field
   //   3) the offsets with a specified element block of buffer for each node field
   std::tuple<int,Array<int>,Array<int>> ExtractBufferInfo(
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

   void CreateTransferMaps();

   static Array<SAMRAI::pdat::NodeIndex::Corner> getCorners(const unsigned dimension);

   static inline SAMRAI::hier::Index toIndex(const Vector& vector);

   static inline SAMRAI::hier::Index toIndex(const Array<int>& array,
      const unsigned dim, const int start);

   static Vector toVector(const SAMRAI::hier::IntVector& vector);

   // TODO: come up with better comment
   /***** member variables *****/

   const int element_info_tag = 0;
   const int samrai_values_tag = 1;
   const int element_values_tag = 2;

   std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy;
   const Array<SAMRAI::pdat::NodeIndex::Corner> corners;

   ParMesh mesh;
   std::shared_ptr<GridFunction> mesh_grid_function;

   H1_FECollection fe_collection_node;
   L2_FECollection fe_collection_cell;
   // maps are from field dimension to finite element space
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_cell;
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_node;

   /***** transfer maps *****/

   struct PatchInfo
   {
      const unsigned dimension;
      int rank;
      int level_number;
      SAMRAI::hier::Index lower_index;
      SAMRAI::hier::Index upper_index;

      PatchInfo(const unsigned dimension);

      PatchInfo(int rank, const Array<int>& values);

      const Array<int>& AsArray() const;

      unsigned Size() const;

   private:

      mutable Array<int> array;
   };
   std::vector<PatchInfo> global_patch_list;

   struct CellInfo
   {
      SAMRAI::pdat::CellIndex index;
      std::shared_ptr<SAMRAI::hier::Patch> patch;
   };
   std::vector<std::vector<CellInfo>> local_cell_info; // (rank) -> {CellInfo for local cell that corresponds to element on rank}
   std::vector<std::vector<int>> local_element_inds; // (rank) -> {local element ind that corresponds to cell on rank}

   /***** helper classes *****/

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

   struct ElementInfo
   {
      int level_number;
      SAMRAI::hier::Index index;

      ElementInfo(const unsigned dimension);

      void operator =(const Array<int>& values);

      const Array<int>& AsArray() const;

      unsigned Size() const;

   private:

      mutable Array<int> array;

   };

};

}

#endif
