
#ifndef MFEM_SAMRAICOUPLINGMANAGER
#define MFEM_SAMRAICOUPLINGMANAGER

#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/pdat/NodeData.h"
#include "SAMRAI/pdat/CellData.h"

#include "mfem.hpp"

namespace mfem
{

/// Manages an MFEM mesh that mirrors a SAMRAI patch hierarchy.
/// Transfers nodal and cell-centered data between the two representations.
class SAMRAICouplingManager
{
public:

   /// Boundary attribute identifiers.
   enum BDR_ATTRIBUTE {Ylower=1, Xupper=2, Yupper=3, Xlower=4};

   /// Construct a manager for @a hierarchy and create the initial mesh.
   SAMRAICouplingManager(std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy);

   /// Read-only access to the managed MFEM mesh (meant for temporary access).
   const ParMesh& GetMesh() const { return *mesh; }

   /// Create an "unmanaged" finite element space on the managed MFEM mesh.
   /** The caller owns the returned space. It is not updated after a call to
       SynchronizeMeshToHierarchy() and must not be used after that call. */
   std::unique_ptr<ParFiniteElementSpace> CreateFESpace(
      FiniteElementCollection* fe_collection, int dim=1)
   {
      return std::make_unique<ParFiniteElementSpace>(mesh.get(), fe_collection, dim);
   }

   /// Set the nodal GridFunction that defines the physical mesh positions.
   /** The GridFunction must be defined on a space created by CreateFESpace().
       A subsequent call to SynchronizeMeshToHierarchy() must create a new mesh. */
   void SetMeshGridFunction(std::shared_ptr<GridFunction> grid_function)
   {
      mesh->SetNodalGridFunction(grid_function.get());
      mesh_grid_function = grid_function;
   }

   /// Synchronize the managed MFEM mesh with the current SAMRAI hierarchy.
   /** When @a create_new_mesh is true, rebuild the MFEM mesh. Otherwise, update
       the existing mesh by derefining and refining it, after which all 
       "unmanaged" finite element spaces created by CreateFESpace() become 
       invalid. */
   void SynchronizeMeshToHierarchy(bool create_new_mesh=false);

   /// Transfer SAMRAI data to new MFEM grid functions and update the mesh position.
   /** @a position_id and every entry in @a node_ids must identify NodeData<double>
       with depth equal to the mesh dimension. Each entry in @a cell_ids must
       identify CellData<double> with depth one. All fields must belong to the
       managed patch hierarchy. The caller owns the returned grid fuctions, which
       are not updated after a call to SynchronizeMeshToHierarchy() and must not be
       used after that call. */
   std::vector<std::unique_ptr<ParGridFunction>> TransferToMFEM(
                                                 const int position_id, const std::vector<int>& node_ids,
                                                 const std::vector<int>& cell_ids);

   /// Transfer MFEM nodal and cell field values to the SAMRAI hierarchy.
   /** Each pair contains a SAMRAI patch-data identifier and its source MFEM field.
       Nodal fields must use by-node ordering. Cell fields are transferred as
       MFEM element averages. */
   void TransferToSAMRAI(
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

   /// Transfer MFEM mesh position and field values to the SAMRAI hierarchy.
   /** The position field is taken from the nodal GridFunction set by
       SetMeshGridFunction(). */
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

   static inline SAMRAI::hier::Index ToIndex(const Vector& vector);

   static inline SAMRAI::hier::Index ToIndex(const Array<int>& array,
                                             const unsigned dim, const int start);

   static Vector ToVector(const SAMRAI::hier::IntVector& vector);

   static Vector GetElementDimensions(Mesh& mesh, const int element_ind);

   // Utility classes.

   /// Array wrapper for data serialized (by blocks) for MPI communication.
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

      void GetElementCounts(const Array<PODType> &block_counts,
                            Array<int> &element_counts) const;

   };

   /// Describes a SAMRAI patch used to mirror the hierarchy in the MFEM mesh.
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

   /// Identifies an MFEM element by its corresponding SAMRAI level and cell.
   struct ElementInfo
   {
      int level_number;
      SAMRAI::hier::Index index;

      ElementInfo(const int level_number_, const SAMRAI::hier::Index index_);

      Array<int> AsArray() const;

      static unsigned Size(const unsigned dimension);

      static ElementInfo FromArray(const Array<int>& values);
   };

   /// Locates a SAMRAI cell and its owning patch for data transfer.
   struct CellInfo
   {
      SAMRAI::pdat::CellIndex index;
      std::shared_ptr<SAMRAI::hier::Patch> patch;
   };

   // MPI utility methods.

   /// Return the value # per element, nodal field dimensions, and field offsets.
   /** The offsets identify the nodal fields within one packed element record;
       cell fields follow the final nodal offset. */
   std::tuple<int,Array<int>,Array<int>> ExtractBufferInfo(
                                         std::vector<std::pair<int, GridFunction&>> node_fields,
                                         std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

   void GatherGlobalPatchInfo(const std::vector<PatchInfo>& local_patch_info,
                              std::vector<PatchInfo>& gathered_patch_info) const;

   using PatchLevelBounds =
      std::vector<std::pair<const SAMRAI::hier::Index, const SAMRAI::hier::Index>>;
   void GetGlobalPatchBounds(std::vector<PatchLevelBounds>& global_patch_bounds)
   const;

   // SAMRAI hierarchy bookkeeping.

   void AddNewPatchesToGlobalPatchInfo();

   void RemoveOldPatchesFromGlobalPatchInfo();

   // MFEM mesh creation and update.

   void CreateMesh();

   void UpdateFiniteElementSpaces();

   void DerefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds);

   void RefineMesh(const std::vector<PatchLevelBounds>& global_patch_bounds);

   void CreateTransferMaps();

   // MPI message tags.

   const int element_info_tag = 0;
   const int samrai_values_tag = 1;
   const int element_values_tag = 2;

   // SAMRAI hierarchy state.

   std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy;
   std::vector<PatchInfo> global_patch_info;

   // SAMRAI-to-MFEM data transfer state.

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

   // Indexed by rank; identifies locally owned cells for that rank's elements.
   std::vector<std::vector<CellInfo>> local_cell_info;
   // Indexed by rank; identifies local elements that correspond to that rank's cells.
   std::vector<std::vector<int>> local_element_inds;

   // Managed MFEM objects.

   std::unique_ptr<ParMesh> mesh;
   std::shared_ptr<GridFunction> mesh_grid_function;
   Vector mesh_index_space_tdofs;
   H1_FECollection fe_collection_node;
   L2_FECollection fe_collection_cell;

   // Maps field vector dimension to its managed finite element space.
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_cell;
   std::map<int,std::unique_ptr<ParFiniteElementSpace>> fe_spaces_node;

};

}

#endif
