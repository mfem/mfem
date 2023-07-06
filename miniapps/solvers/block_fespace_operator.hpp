#ifndef MFEM_BLOCK_FESPACE_OPERATOR
#define MFEM_BLOCK_FESPACE_OPERATOR

#include "mfem.hpp"

namespace mfem
{

/// @brief Operator for block systems arising from different arbitrarily many finite element spaces.
///
///        This operator can be used with FormLinearSystem to impose boundary
///        conditions for block systems arise from mixing many types of
///        finite element spaces. Each block is intended to operate on
///        L-Vectors. For example, a block may be a BilinearForm.
class BlockFESpaceOperator : public BlockOperator
{
private:
   Array<int> offsets;
   Array<int> prolongColOffsets;
   Array<int> restrictRowOffsets;
   /// @brief Maps local dofs of each block to true dofs.
   BlockOperator prolongation;
   /// @brief Maps true dofs of each block to local dofs.
   BlockOperator restriction;
   /// @brief Computes offsets for parent BlockOperator.
   static Array<int> GetBlockOffsets(const std::vector<const FiniteElementSpace*>
                                     &fespaces);
   /// @brief Computes col_offsets for prolongation operator.
   static Array<int> GetProColBlockOffsets(const
                                           std::vector<const FiniteElementSpace*> &fespaces);
   /// @brief Computes row_offsets for restriction operator.
   static Array<int> GetResRowBlockOffsets(const
                                           std::vector<const FiniteElementSpace*> &fespaces);
public:
   /// @brief Constructor for BlockFESpaceOperator.
   /// @param[in] fespaces Finite element spaces for diagonal blocks. Spaces are not owned.
   BlockFESpaceOperator(const std::vector<const FiniteElementSpace*> &fespaces);
   virtual const Operator* GetProlongation () const;
   virtual const Operator* GetRestriction () const;
};

}//namespace mfem

#endif