
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
//   template <typename T>
//   using CellData = SAMRAI::pdat::CellData<T>;
//   template <typename T>
//   using NodeData = SAMRAI::pdat::NodeData<T>;
//   using PatchHierarchy = SAMRAI::hier::PatchHierarchy;
//   using NodeIndex = SAMRAI::pdat::NodeIndex;

   static Array<SAMRAI::pdat::NodeIndex::Corner> getCorners(const unsigned dimension);

   static inline SAMRAI::hier::Index toIndex(const Vector& vector);

   static inline SAMRAI::hier::Index toIndex(const Array<int>& array,
      const unsigned dim, const int start);

   static Vector toVector(const SAMRAI::hier::IntVector& vector);

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

   template<typename PODType>
   class BlockArray
   {
      const unsigned block_size, num_blocks;
      Array<PODType> data;

   public:

      BlockArray(const unsigned block_size, const unsigned num_blocks) :
         block_size(block_size),num_blocks(num_blocks),
         data(num_blocks*block_size) {}

      void SetBlock(const unsigned index, const Array<PODType> &values)
      {
         MFEM_ASSERT(values.Size() == block_size,
            "Provided array is not the correct size.");
         // TODO: consider replacing with memcpy
         for(unsigned i=0; i < block_size; i++)
         {
            data[index*block_size + i] = values[i];
         }
      }

      Array<PODType> GetBlock(const unsigned index) const
      {
         Array<PODType> values(block_size);
         // TODO: consider replacing with memcpy
         for(unsigned i=0; i < block_size; i++)
         {
            values[i] = data[index*block_size + i];
         }
         return values;
      }

      unsigned NumBlocks() const
      {
         return num_blocks;
      }

      void* GetData()
      {
         return data.GetData();
      }

      int Size() const
      {
         return data.Size();
      }

      void GetElementCounts(const Array<PODType> &block_counts, Array<int> &element_counts) const
      {
         element_counts.SetSize(block_counts.Size());
         for (int i=0; i < block_counts.Size(); i++)
            element_counts[i] = block_counts[i] * block_size;
      }

   };

   struct PatchInfo
   {
      const unsigned dimension;
      int rank;
      int level_number;
      SAMRAI::hier::Index lower_index;
      SAMRAI::hier::Index upper_index;

      PatchInfo(const unsigned dimension) : dimension(dimension), rank(-1),
         level_number(-1), lower_index(SAMRAI::tbox::Dimension(dimension)),
         upper_index(SAMRAI::tbox::Dimension(dimension))
      {
         array.SetSize(Size());
      }

      PatchInfo(int rank, const Array<int>& values) :
         dimension((values.Size()-1) / 2), rank(rank),
         lower_index(SAMRAI::tbox::Dimension(dimension)),
         upper_index(SAMRAI::tbox::Dimension(dimension))
      {
         MFEM_ASSERT((values.Size() - 1) % 2 == 0,
            "Provided array is not a valid size.")
         level_number = values[0];
         lower_index = toIndex(values, dimension, 1);
         upper_index = toIndex(values, dimension, 1+dimension);
         array.SetSize(Size());
      }

      const Array<int>& AsArray() const
      {
         array[0] = level_number;
         // TODO: consider replacing loops with memcpy
         for (int d=0; d < dimension; d++)
            array[1+d] = lower_index(d);
         for (int d=0; d < dimension; d++)
            array[1+dimension+d] = upper_index(d);
         return array;
      }

      unsigned Size() const
      {
         return 1 + 2*dimension;
      }

   private:

      mutable Array<int> array;
   };

   struct ElementInfo
   {
      int level_number;
      SAMRAI::hier::Index index;

      ElementInfo(const unsigned dimension) : dimension(dimension),
         level_number(-1), index(SAMRAI::tbox::Dimension(dimension))
      {
         array.SetSize(Size());
      }

      void operator =(const Array<int>& values)
      {
         MFEM_ASSERT(values.Size() > 1, "Provided array is not a valid size.");
         level_number = values[0];
         index = toIndex(values, dimension, 1);
      }

      const Array<int>& AsArray() const
      {
         array[0] = level_number;
         // TODO: consider replacing loops with memcpy
         for (int d=0; d < dimension; d++)
            array[1+d] = index(d);
         return array;
      }

      unsigned Size() const
      {
         return 1 + dimension;
      }

   private:

      const unsigned dimension;
      mutable Array<int> array;

   };

   struct CellInfo
   {
      SAMRAI::pdat::CellIndex index;
      std::shared_ptr<SAMRAI::hier::Patch> patch;
   };

   std::vector<std::vector<CellInfo>> local_cell_info; // (rank) -> {CellInfo for local cell that corresponds to element on rank}
   std::vector<std::vector<int>> local_element_inds; // (rank) -> {local element ind that corresponds to cell on rank}

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

   // reconstruct an L2 field u (represented by dst) from u_hat (represented by src)
   // Note currently only 2D reconstruction of L^2_1 (piecewise-linear) field from
   // L^2_0 (piecewise-constant) field is supported. The reconstruction is done
   // by enforcing
   //   1) element average: (u, psi_hat)_E = ({u_hat}, psi_hat)_E
   //   2) grad: (div[ u e_i ], psi_hat)_E = <{u_hat} (e_i dot n), psi_hat>_E
   //   3) grad^2: (div[ (e_i \otimes e_j) grad[u] ], psi_hat)_E =
   //        <(grad[u_hat] dot e_j) e_i, psi_hat>_E ... which is 0
   //   4) grad^3: (div[ (e_i \otimes e_j \otimes e_k) hessian[u] ], psi_hat)_E =
   //        <[hessian[u_hat] e_k) dot e_j] e_i, psi_hat>_E ... which is 0
   // where ( , )_E denotes area integral, < , >_E denotes the surface integral,
   // psi_hat the L^2_0 basis function on element E, and e_* is the unit vector
   // in the x_* direction
   void reconstructL2Field(const ParGridFunction& src, ParGridFunction& dst);

private:

   // extracts the following information for the gather/scatter buffer:
   //   1) the number of values per element
   //   2) the vector dimension of each node field
   //   3) the offsets with a specified element block of buffer for each node field
   std::tuple<int,Array<int>,Array<int>> extractBufferInfo(
      std::vector<std::pair<int, GridFunction&>> node_fields,
      std::vector<std::pair<int, ParGridFunction&>> cell_fields) const;

};

}

#endif
