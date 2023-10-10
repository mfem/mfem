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
class BlockFESpaceOperator : public Operator
{
private:
   /// @brief Offsets for the square "A" operator.
   Array<int> offsets;
   /// @brief Column offsets for the prolongation operator.
   Array<int> prolongColOffsets;
   /// @brief Row offsets for the prolongation operator.
   Array<int> restrictRowOffsets;
   /// @brief The "A" part of "RAP".
   BlockOperator A;
   /// @brief Maps local dofs of each block to true dofs.
   BlockOperator prolongation;
   /// @brief Maps true dofs of each block to local dofs.
   BlockOperator restriction;
   /// @brief Computes height for parent operator.
   static int GetHeight(const std::vector<const FiniteElementSpace*>
                        &fespaces);
   /// @brief Computes offsets for A BlockOperator.
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
   const Operator* GetProlongation () const override;
   const Operator* GetRestriction () const override;
   void Mult(const Vector &x, Vector &y) const override {A.Mult(x,y);};
   /// @brief Wraps BlockOperator::SetBlock. Eventually would like this class to inherit
   /// from BlockOperator instead, but can't easily due to ownership of offset data
   /// in BlockOperator being by reference.
   void SetBlock( int   iRow,
                  int   iCol,
                  Operator *  op,
                  double   c = 1.0) {A.SetBlock(iRow, iCol, op, c);};
};

}//namespace mfem

#endif