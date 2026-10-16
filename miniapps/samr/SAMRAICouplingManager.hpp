
#ifndef MFEM_SAMRAICOUPLINGMANAGER
#define MFEM_SAMRAICOUPLINGMANAGER

#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/pdat/NodeData.h"
#include "SAMRAI/pdat/CellData.h"

#include "mfem.hpp"

namespace mfem
{

///
///

/// @brief Manages MFEM mesh and spaces that mirror a SAMRAI patch hierarchy,
/// and associated data, and transfers nodal and cell-centered data between the
/// two representations.
class SAMRAICouplingManager
{
public:

   /// Boundary attribute identifiers.
   enum BDR_ATTRIBUTE {Ylower=1, Xupper=2, Yupper=3, Xlower=4};

   /// @brief Construct a manager and create the initial MFEM mesh.
   /// @param hierarchy SAMRAI patch hierarchy that the manager will mirror.
   SAMRAICouplingManager(std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy);

   /// @brief Read-only access to the managed MFEM mesh (intended for temporary
   /// access).
   const ParMesh& GetMesh() const { return *mesh; }

   /// @brief Create an "unmanaged" finite element space on the managed MFEM
   /// mesh.
   /// @param fe_collection finite element collection for the space
   /// @param dim dimension of the space
   /// @return finite element space, owned by the caller, that becomes invalid
   /// after a call to SynchronizeMeshToHierarchy()
   std::unique_ptr<ParFiniteElementSpace> CreateFESpace(
      FiniteElementCollection* fe_collection, int dim=1)
   {
      return std::make_unique<ParFiniteElementSpace>(mesh.get(), fe_collection, dim);
   }

   /// @brief Use the provided nodal grid function to define the topology of the
   /// managed MFEM mesh (until the next call to SynchronizeMeshToHierarchy,
   /// which must currently then create a new mesh and spaces).
   /// @param grid_function grid function, which be defined on a finite
   /// element space created by CreateFESpace(), specifying the mesh topology
   void SetMeshGridFunction(std::shared_ptr<GridFunction> grid_function)
   {
      mesh->SetNodalGridFunction(grid_function.get());
      mesh_grid_function = grid_function;
   }

   /// @brief Synchronize the managed MFEM mesh and spaces objects to the
   //  current SAMRAI hierarchy.
   /// @param create_new_mesh specifies whether to create a new managed mesh
   /// (and spaces) or update the existing managed mesh and spaces (that results
   //  in all "unmanaged" spaces created by CreateFESpace() to become invalid)
   void SynchronizeMeshToHierarchy(bool create_new_mesh=false);

   /// @brief Transfer SAMRAI data to new MFEM grid functions and update the
   /// mesh topology.
   /// @param position_id SAMRAI ID for nodal data specifying node positions
   /// @param node_ids SAMRAI IDs for nodal data specifying other vector fields
   //  (currently assumed to have the same dimension as the physical space,
   //  e.g., a velocity field)
   /// @param cell_ids SAMRAI IDs for cell data specifying scalar fields
   /// @return created MFEM grid functions, owned by the caller, that become
   /// invalid after the next call to SynchronizeMeshToHierarchy()
   std::vector<std::unique_ptr<ParGridFunction>> TransferToMFEM(
                                                 const int position_id, const std::vector<int>& node_ids,
                                                 const std::vector<int>& cell_ids);

   /// @brief Transfer MFEM grid function field data to the SAMRAI patch
   /// hierarchy.
   /// @param node_fields (SAMRAI ID, grid function) pairs specifying the source
   /// MFEM vector grid function and target SAMRAI nodal data, with values
   /// interpolated at the element vertices
   /// @param cell_fields (SAMRAI ID, grid function) pairs specifying the source
   /// MFEM scalar grid function and target SAMRAI cell data, with values
   /// averaged across elements
   void TransferToSAMRAI(
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

   /// @brief Transfer MFEM grid function mesh topology and field data to the
   /// SAMRAI patch hierarchy.
   /// @param position_id SAMRAI ID for target SAMRAI nodal data specifying node
   /// positions
   /// @param node_fields (SAMRAI ID, grid function) pairs specifying the source
   /// MFEM vector grid function and target SAMRAI nodal data, with values
   /// interpolated at the element vertices
   /// @param cell_fields (SAMRAI ID, grid function) pairs specifying the source
   /// MFEM scalar grid function and target SAMRAI cell data, with values
   /// averaged across elements
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

   // General utility methods.

   /// Create a SAMRAI index from the specified MFEM @a vector .
   static inline SAMRAI::hier::Index ToIndex(const Vector& vector);

   /// Create a SAMRAI index from the specified MFEM @a array with dimension
   /// @a dim starting at index @a start .
   static inline SAMRAI::hier::Index ToIndex(const Array<int>& array,
                                             const unsigned dim, const int start);

   /// Create an MFEM vector from the specified integer SAMRAI @a vector .
   static Vector ToVector(const SAMRAI::hier::IntVector& vector);

   /// Obtain an MFEM vector specifying the lengths of the element with index
   /// @a element_ind in the specified MFEM @a mesh
   static Vector GetElementDimensions(Mesh& mesh, const int element_ind);

   // Utility classes.

   /// Array wrapper for data serialized (by blocks) for MPI communication.
   template<typename PODType>
   class BlockArray
   {
      Array<PODType> data;

   public:

      const unsigned block_size, num_blocks;

      /// Construct a block array with allocated memory for @a num_blocks
      /// blocks, each with @a block_size number of @a PODType entries.
      BlockArray(const unsigned block_size_, const unsigned num_blocks_);

      /// Set the @a index block with @a values , where @a index 2 would specify
      /// the 3rd block.
      void SetBlock(const unsigned index, const Array<PODType> &values);

      /// Get the values of the @a index block, where @a index 2 would specify
      /// the 3rd block.
      Array<PODType> GetBlock(const unsigned index) const;

      /// The number of blocks.
      unsigned NumBlocks() const;

      /// Read-write access to the underlying contiguous data (intended for MPI)
      void* GetData();

      /// The number of @a PODType entries in the underlying contiguous data
      /// (intended fore MPI)
      int Size() const;
   };

   /// Serializable description of a SAMRAI patch by rank, level, lower node
   /// index, and upper node index.
   struct PatchInfo
   {
      int rank;
      int level_number;
      SAMRAI::hier::Index lower_index;
      SAMRAI::hier::Index upper_index;

      /// Construct info for patch local to @a rank_ , on @a level_number_ ,
      /// spanning from @a lower_index_ to @a upper_index_ .
      PatchInfo(const int rank_, const int level_number_,
                const SAMRAI::hier::Index lower_index_,
                const SAMRAI::hier::Index upper_index_);

      /// Check if rank, level number, lower index, and upper index are equal.
      bool operator==(const PatchInfo& other) const;

      /// Serialize patch info as integer array (intended for MPI).
      Array<int> AsArray() const;

      /// Size of serialized @a dimension -D patch info (intended for MPI).
      static unsigned Size(const unsigned dimension);

      /// Convert serialized array @a values to patch info.
      static PatchInfo FromArray(const Array<int>& values);
   };

   /// Serializable description of an MFEM element by SAMRAI patch level and
   /// cell index.
   struct ElementInfo
   {
      int level_number;
      SAMRAI::hier::Index index;

      /// Construct info for element corresponding to level @a level_number_ and
      /// cell index @a index_ .
      ElementInfo(const int level_number_, const SAMRAI::hier::Index index_);

      /// Serialize patch info as integer array (intended for MPI).
      Array<int> AsArray() const;

      /// Size of serialized @a dimension -D element info (intended for MPI).
      static unsigned Size(const unsigned dimension);

      /// Convert serial array @a values to element info.
      static ElementInfo FromArray(const Array<int>& values);
   };

   /// Description of a SAMRAI cell by cell index and containing patch.
   struct CellInfo
   {
      SAMRAI::pdat::CellIndex index;
      std::shared_ptr<SAMRAI::hier::Patch> patch;
   };

   // MPI utility methods.

   /// Extract the MPI buffer information for transfering @a node_fields and
   /// @a cell_fields , namely the number of values per MFEM element, the
   /// dimension of each nodal field, and the buffer offsets for each nodal
   /// field.
   std::tuple<int,Array<int>,Array<int>> ExtractBufferInfo(
                                         std::vector<std::pair<int, GridFunction&>> node_fields,
                                         std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

   /// Gather (via MPI) all @a local_patch info into @a gathered_patch_info
   void GatherGlobalPatchInfo(const std::vector<PatchInfo>& local_patch_info,
                              std::vector<PatchInfo>& gathered_patch_info) const;

   /// From global patch info, extract the lower node index and upper
   /// node index for each patch into @a global_patch_bounds , organized by
   /// level (i.e., first entry of @a global_patch_bounds is all the
   /// patches on the first level)
   using PatchLevelBounds =
      std::vector<std::pair<const SAMRAI::hier::Index, const SAMRAI::hier::Index>>;
   void GetGlobalPatchBounds(std::vector<PatchLevelBounds>& global_patch_bounds)
   const;

   // SAMRAI hierarchy bookkeeping.

   /// Add SAMRAI patch info for patches that are in the patch hierarchy but not
   /// in the global patch info.
   void AddNewPatchesToGlobalPatchInfo();

   /// Remove SAMRAI patch info from global patch info for patches that are no
   /// longer in the patch hierarchy.
   void RemoveOldPatchesFromGlobalPatchInfo();

   // MFEM mesh creation and update.

   /// Create MFEM mesh in index space (not physical space) that mirrors SAMRAI
   /// patch hierarchy.
   void CreateMesh();

   /// Call Update() for each managed finite element space and the mesh grid
   /// function.
   void UpdateFiniteElementSpaces();

   /// Derefine mesh to match patch level, lower node index, and upper node
   /// index information in @a global_patch_bounds .
   void DerefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds);

   /// Refine mesh to match patch level, lower node index, and upper node index
   /// information in @a global_patch_bounds .
   void RefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds);

   /// Create mappings between local MFEM elements and SAMRAI cells and the
   /// corresponding remote SAMRAI cells and MFEM elements, respectively.
   void CreateTransferMaps();

   // MPI message tags.

   const int element_info_tag = 0;
   const int samrai_values_tag = 1;
   const int element_values_tag = 2;

   // SAMRAI hierarchy access and condensed representation as patch info.

   std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy;
   std::vector<PatchInfo> global_patch_info;

   // Arrays of SAMRAI node index flags in corresponding MFEM element vertex
   // ordering.

   const Array<SAMRAI::pdat::NodeIndex::Corner>& corners;

   const Array<SAMRAI::pdat::NodeIndex::Corner> corners1D
   {
      SAMRAI::pdat::NodeIndex::Left, SAMRAI::pdat::NodeIndex::Right};
   const Array<SAMRAI::pdat::NodeIndex::Corner> corners2D
   {
      SAMRAI::pdat::NodeIndex::LowerLeft, SAMRAI::pdat::NodeIndex::LowerRight,
      SAMRAI::pdat::NodeIndex::UpperRight, SAMRAI::pdat::NodeIndex::UpperLeft};
   const Array<SAMRAI::pdat::NodeIndex::Corner> corners3D
   {
      SAMRAI::pdat::NodeIndex::LLL, SAMRAI::pdat::NodeIndex::ULL,
      SAMRAI::pdat::NodeIndex::UUL, SAMRAI::pdat::NodeIndex::LUL,
      SAMRAI::pdat::NodeIndex::LLU, SAMRAI::pdat::NodeIndex::ULU,
      SAMRAI::pdat::NodeIndex::UUU, SAMRAI::pdat::NodeIndex::LUU};

   // Rank-indexed vector containing vectors of cell info for local cells that
   // correspond to remote elements on the given rank.
   std::vector<std::vector<CellInfo>> local_cell_info;

   // Rank-index vector containing vectors of element indices for local elements
   // that correspond to remote cells on the given rank.
   std::vector<std::vector<int>> local_element_inds;

   // Managed MFEM objects.

   std::unique_ptr<ParMesh> mesh;
   std::shared_ptr<GridFunction> mesh_grid_function;
   Vector mesh_index_space_tdofs;
   H1_FECollection fe_collection_node;
   L2_FECollection fe_collection_cell;

   // Maps field vector dimension to a corresponding finite element space.
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_cell;
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_node;

};

}

#endif
