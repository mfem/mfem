#ifndef TENSOR_CLASS
#define TENSOR_CLASS

//#include "mfem.hpp"

template<typename T>
class Tensor
{

public:

  Tensor() {}

  void AddElement(const T& elem, int row, int col)
  {
    data.emplace_back(elem); // copy construction
    row_idx.Append(row);
    col_idx.Append(col);
  }

  void AddElement(const T& elem, int dia)
  {
    AddElement(elem, dia, dia);
  }

  void Eval(std::vector<T>& data_out, 
            mfem::Array<int>& row_idx_out,
            mfem::Array<int>& col_idx_out) const
  {
    data_out    = data; // copy
    row_idx_out = row_idx;
    col_idx_out = col_idx;
  }

  void reserve(int res)
  {
    data.reserve(res);
    row_idx.Reserve(res);
    col_idx.Reserve(res);
  }

private:

  std::vector<T> data;
  mfem::Array<int> row_idx;
  mfem::Array<int> col_idx;

};

class MatrixTensorCoefficient
{

public:

  MatrixTensorCoefficient(int _vdim) : vdim{_vdim} {}

  virtual void Eval(std::vector<mfem::DenseMatrix>& data,
                    mfem::Array<int>& row_idx,
                    mfem::Array<int>& col_idx,
                    mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) const = 0;

  int GetVDim() const { return vdim; }

  virtual ~MatrixTensorCoefficient() {}

protected:

  const int vdim;

};

class VectorTensorCoefficient
{

public:

  VectorTensorCoefficient(int _vdim) : vdim{_vdim} {}

  virtual void Eval(std::vector<mfem::Vector>& data,
                    mfem::Array<int>& row_idx,
                    mfem::Array<int>& col_idx,
                    mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) const = 0;

  int GetVDim() const { return vdim; }

  virtual ~VectorTensorCoefficient() {}

protected:

  const int vdim;

};

class MatrixTensorPWConstCoefficient : public MatrixTensorCoefficient
{

public:

  MatrixTensorPWConstCoefficient(int _vdim, 
                                 const std::vector<Tensor<mfem::DenseMatrix>>& coeff)
    : MatrixTensorCoefficient{_vdim}, coefficients{coeff}
  {}

  virtual void Eval(std::vector<mfem::DenseMatrix>& data,
                    mfem::Array<int>& row_idx,
                    mfem::Array<int>& col_idx,
                    mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) const override
  {
    const int att{T.Attribute-1};
    coefficients[att].Eval(data, row_idx, col_idx);
  }

private:

  const std::vector<Tensor<mfem::DenseMatrix>>& coefficients;

};

class MatrixTensorConstantCoefficient : public MatrixTensorCoefficient
{

public:

  MatrixTensorConstantCoefficient(int _vdim, 
                                  const Tensor<mfem::DenseMatrix>& coeff)
    : MatrixTensorCoefficient{_vdim}, coefficient{coeff}
  {}

  virtual void Eval(std::vector<mfem::DenseMatrix>& data,
                    mfem::Array<int>& row_idx,
                    mfem::Array<int>& col_idx,
                    mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) const override
  {
    coefficient.Eval(data, row_idx, col_idx);
  }

private:

  const Tensor<mfem::DenseMatrix>& coefficient;

};

class VectorTensorPWConstCoefficient : public VectorTensorCoefficient
{

public:

  VectorTensorPWConstCoefficient(int _vdim, 
                                 const std::vector<Tensor<mfem::Vector>>& coeff)
    : VectorTensorCoefficient{_vdim}, coefficients{coeff}
  {}

  virtual void Eval(std::vector<mfem::Vector>& data,
                    mfem::Array<int>& row_idx,
                    mfem::Array<int>& col_idx,
                    mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) const override
  {
    const int att{T.Attribute-1};
    coefficients[att].Eval(data, row_idx, col_idx);
  }

private:

  std::vector<Tensor<mfem::Vector>> coefficients;

};

class VectorTensorConstantCoefficient : public VectorTensorCoefficient
{

public:

  VectorTensorConstantCoefficient(int _vdim,
                                  const Tensor<mfem::Vector>& coeff)
    : VectorTensorCoefficient{_vdim}, coefficient{coeff}
  {}


  virtual void Eval(std::vector<mfem::Vector>& data,
                    mfem::Array<int>& row_idx,
                    mfem::Array<int>& col_idx,
                    mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) const override
  {
    coefficient.Eval(data, row_idx, col_idx);
  }

private:

  const Tensor<mfem::Vector>& coefficient;


};  

#endif
