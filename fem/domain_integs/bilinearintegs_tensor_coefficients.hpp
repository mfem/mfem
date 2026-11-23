#ifndef BILINEARINTEGS_TENSOR_COEFFICIENTS_HPP
#define BILINEARINTEGS_TENSOR_COEFFICIENTS_HPP

/// @file
/// Structured physical fourth-order coefficients for component gradients.
/// These adapters expand to ordinary dense MFEM matrix coefficients during
/// evaluation for EA; PA can project the borrowed factors without expansion.

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace mfem {

/// Fourth-order product A_ia,jb = Q_ij K_ab, flattened as a + dimension * i.
/// Factor coefficients are borrowed and must outlive this adapter. A vector
/// field factor denotes diagonal Q; an omitted direction factor denotes I.
class BlfKroneckerTensorCoefficient : public mfem::MatrixCoefficient {
  mfem::MatrixCoefficient *field_matrix_ = nullptr;
  mfem::VectorCoefficient *field_diagonal_ = nullptr;
  mfem::MatrixCoefficient *directions_ = nullptr;
  int dimension_;

public:
  /// @param field Matrix Q with test field components in rows and trial field
  /// components in columns; rectangular field couplings are supported.
  /// @param directions Square Cartesian derivative-direction matrix K.
  BlfKroneckerTensorCoefficient(mfem::MatrixCoefficient &field,
                                mfem::MatrixCoefficient &directions)
      : mfem::MatrixCoefficient(field.GetHeight() * directions.GetHeight(),
                                field.GetWidth() * directions.GetWidth()),
        field_matrix_(&field), directions_(&directions),
        dimension_(directions.GetHeight()) {
    MFEM_VERIFY(field.GetHeight() > 0 && field.GetWidth() > 0 &&
                    dimension_ >= 1 && dimension_ <= 3 &&
                    directions.GetWidth() == dimension_,
                "direction coefficient must be square with dimension 1 to 3");
  }

  /// @param field Matrix Q coupling field components.
  /// @param dimension Cartesian space dimension; K is the identity.
  BlfKroneckerTensorCoefficient(mfem::MatrixCoefficient &field, int dimension)
      : mfem::MatrixCoefficient(field.GetHeight() * dimension,
                                field.GetWidth() * dimension),
        field_matrix_(&field), dimension_(dimension) {
    MFEM_VERIFY(field.GetHeight() > 0 && field.GetWidth() > 0 &&
                    dimension >= 1 && dimension <= 3,
                "Cartesian dimension must be between 1 and 3");
  }

  /// @param field Diagonal entries of Q, one per field component.
  /// @param directions Square Cartesian derivative-direction matrix K.
  BlfKroneckerTensorCoefficient(mfem::VectorCoefficient &field,
                                mfem::MatrixCoefficient &directions)
      : mfem::MatrixCoefficient(field.GetVDim() * directions.GetHeight(),
                                field.GetVDim() * directions.GetWidth()),
        field_diagonal_(&field), directions_(&directions),
        dimension_(directions.GetHeight()) {
    MFEM_VERIFY(field.GetVDim() > 0 && dimension_ >= 1 && dimension_ <= 3 &&
                    directions.GetWidth() == dimension_,
                "direction coefficient must be square with dimension 1 to 3");
  }

  /// @param field Diagonal entries of Q, one per field component.
  /// @param dimension Cartesian space dimension; K is the identity.
  BlfKroneckerTensorCoefficient(mfem::VectorCoefficient &field, int dimension)
      : mfem::MatrixCoefficient(field.GetVDim() * dimension),
        field_diagonal_(&field), dimension_(dimension) {
    MFEM_VERIFY(field.GetVDim() > 0 && dimension >= 1 && dimension <= 3,
                "Cartesian dimension must be between 1 and 3");
  }

  /// @return Borrowed dense field factor, or nullptr for diagonal Q.
  mfem::MatrixCoefficient *FieldMatrix() const { return field_matrix_; }
  /// @return Borrowed diagonal field factor, or nullptr for dense Q.
  mfem::VectorCoefficient *FieldDiagonal() const { return field_diagonal_; }
  /// @return Borrowed direction factor, or nullptr for identity K.
  mfem::MatrixCoefficient *Directions() const { return directions_; }

  /// @return Cartesian derivative-direction dimension of the product.
  int Dimension() const { return dimension_; }

  /// @param time Time propagated to both borrowed factor coefficients.
  void SetTime(mfem::real_t time) override {
    mfem::MatrixCoefficient::SetTime(time);
    if (field_matrix_) { field_matrix_->SetTime(time); }
    if (field_diagonal_) { field_diagonal_->SetTime(time); }
    if (directions_) { directions_->SetTime(time); }
  }

  /// Expand the factors into a dense physical gradient-action matrix.
  /// @param[out] value Resized flattened fourth-order tensor.
  /// @param transformation Element transformation with its current point set.
  /// @param point Reference point at which to evaluate the factors.
  void Eval(mfem::DenseMatrix &value,
            mfem::ElementTransformation &transformation,
            const mfem::IntegrationPoint &point) override {
    mfem::DenseMatrix field, directions;
    mfem::Vector diagonal;
    if (field_matrix_) {
      field.SetSize(field_matrix_->GetHeight(), field_matrix_->GetWidth());
      field_matrix_->Eval(field, transformation, point);
    }
    if (field_diagonal_) {
      diagonal.SetSize(field_diagonal_->GetVDim());
      field_diagonal_->Eval(diagonal, transformation, point);
    }
    if (directions_) {
      directions.SetSize(directions_->GetHeight(), directions_->GetWidth());
      directions_->Eval(directions, transformation, point);
    }
    value.SetSize(height, width);
    for (int row = 0; row < height; ++row) {
      for (int column = 0; column < width; ++column) {
        const int test_field = row / dimension_;
        const int trial_field = column / dimension_;
        const mfem::real_t field_value =
            field_matrix_ ? field(test_field, trial_field)
                          : (test_field == trial_field ? diagonal[test_field]
                                                       : 0.0);
        const int test_direction = row % dimension_;
        const int trial_direction = column % dimension_;
        const mfem::real_t direction_value =
            directions_ ? directions(test_direction, trial_direction)
                        : (test_direction == trial_direction ? 1.0 : 0.0);
        value(row, column) = field_value * direction_value;
      }
    }
  }
};

/// Fourth-order coefficient obtained from a matrix on Kelvin symmetric-gradient
/// coordinates: diagonal entries first, then (0,1), (0,2), (1,2) when present.
/// Off-diagonal coordinates are (partial_i u_j + partial_j u_i) / sqrt(2).
/// Field and derivative indices must describe the same Cartesian vector space.
/// @tparam MajorSymmetry Require a symmetric constitutive coefficient, giving
/// major symmetry in addition to the two minor symmetries. Otherwise the Kelvin
/// matrix may be nonsymmetric. The constitutive coefficient is borrowed.
template <bool MajorSymmetry>
class BlfSymmetricGradientCoefficient : public mfem::MatrixCoefficient {
public:
  using ConstitutiveCoefficient =
      std::conditional_t<MajorSymmetry, mfem::SymmetricMatrixCoefficient,
                         mfem::MatrixCoefficient>;

private:
  ConstitutiveCoefficient &constitutive_;
  int dimension_;

  /// @param field Cartesian vector-component index.
  /// @param direction Cartesian derivative-direction index.
  /// @return Index of the corresponding symmetric Kelvin coordinate.
  int KelvinIndex(int field, int direction) const {
    if (field == direction) { return field; }
    const int first = std::min(field, direction);
    const int second = std::max(field, direction);
    int index = dimension_;
    for (int row = 0; row < dimension_; ++row) {
      for (int column = row + 1; column < dimension_; ++column, ++index) {
        if (row == first && column == second) { return index; }
      }
    }
    return -1;
  }

public:
  /// @param constitutive Square Kelvin matrix of size dimension*(dimension+1)/2.
  /// For MajorSymmetry it must be an MFEM SymmetricMatrixCoefficient.
  /// @param dimension Cartesian space dimension, between 1 and 3.
  BlfSymmetricGradientCoefficient(ConstitutiveCoefficient &constitutive,
                                  int dimension)
      : mfem::MatrixCoefficient(dimension * dimension),
        constitutive_(constitutive), dimension_(dimension) {
    const int symmetric_components = dimension * (dimension + 1) / 2;
    MFEM_VERIFY(dimension >= 1 && dimension <= 3 &&
                    constitutive.GetHeight() == symmetric_components &&
                    constitutive.GetWidth() == symmetric_components,
                "constitutive coefficient has the wrong Kelvin dimensions");
  }

  /// @return Borrowed Kelvin constitutive coefficient for compact projection.
  ConstitutiveCoefficient &Constitutive() const { return constitutive_; }

  /// @return Cartesian dimension of the symmetric gradients.
  int Dimension() const { return dimension_; }

  /// @param time Time propagated to the borrowed constitutive coefficient.
  void SetTime(mfem::real_t time) override {
    mfem::MatrixCoefficient::SetTime(time);
    constitutive_.SetTime(time);
  }

  /// Expand the Kelvin matrix into a dense flattened gradient-action tensor.
  /// @param[out] value Resized dimension^2 by dimension^2 tensor matrix.
  /// @param transformation Element transformation with its current point set.
  /// @param point Reference evaluation point.
  void Eval(mfem::DenseMatrix &value,
            mfem::ElementTransformation &transformation,
            const mfem::IntegrationPoint &point) override {
    mfem::DenseMatrix constitutive(constitutive_.GetHeight(),
                                   constitutive_.GetWidth());
    constitutive_.Eval(constitutive, transformation, point);
    value.SetSize(height, width);
    const mfem::real_t shear_scale = 1.0 / std::sqrt(2.0);
    for (int row = 0; row < height; ++row) {
      const int test_field = row / dimension_;
      const int test_direction = row % dimension_;
      const mfem::real_t test_scale =
          test_field == test_direction ? 1.0 : shear_scale;
      for (int column = 0; column < width; ++column) {
        const int trial_field = column / dimension_;
        const int trial_direction = column % dimension_;
        const mfem::real_t trial_scale =
            trial_field == trial_direction ? 1.0 : shear_scale;
        value(row, column) =
            test_scale * trial_scale *
            constitutive(KelvinIndex(test_field, test_direction),
                         KelvinIndex(trial_field, trial_direction));
      }
    }
  }
};

/// Kelvin coefficient with both minor symmetries, but no required major symmetry.
using BlfMinorSymmetricTensorCoefficient = BlfSymmetricGradientCoefficient<false>;
/// Kelvin coefficient with both minor symmetries and major symmetry.
using BlfMinorMajorSymmetricTensorCoefficient = BlfSymmetricGradientCoefficient<true>;

/// Isotropic tensor lambda delta_ia delta_jb +
/// mu (delta_ij delta_ab + delta_ib delta_aj), acting on symmetric gradients.
/// Scalar coefficients are borrowed. This is the two-parameter minor-symmetric
/// isotropic family, not the unrestricted three-parameter isotropic tensor.
class BlfIsotropicSymmetricTensorCoefficient : public mfem::MatrixCoefficient {
  mfem::Coefficient &lambda_;
  mfem::Coefficient &mu_;
  int dimension_;

public:
  /// @param lambda Scalar Lame coefficient multiplying div(u) div(v).
  /// @param mu Scalar shear modulus multiplying 2 epsilon(u):epsilon(v).
  /// @param dimension Cartesian space dimension, between 1 and 3.
  BlfIsotropicSymmetricTensorCoefficient(mfem::Coefficient &lambda,
                                    mfem::Coefficient &mu, int dimension)
      : mfem::MatrixCoefficient(dimension * dimension), lambda_(lambda),
        mu_(mu), dimension_(dimension) {
    MFEM_VERIFY(dimension >= 1 && dimension <= 3,
                "Cartesian dimension must be between 1 and 3");
  }

  /// @return Borrowed scalar multiplying the trace product.
  mfem::Coefficient &Lambda() const { return lambda_; }
  /// @return Borrowed scalar multiplying the symmetric-gradient product.
  mfem::Coefficient &Mu() const { return mu_; }

  /// @return Cartesian dimension of the isotropic symmetric tensor.
  int Dimension() const { return dimension_; }

  /// @param time Time propagated to both borrowed Lame coefficients.
  void SetTime(mfem::real_t time) override {
    mfem::MatrixCoefficient::SetTime(time);
    lambda_.SetTime(time);
    mu_.SetTime(time);
  }

  /// Evaluate the isotropic tensor as a full flattened gradient-action matrix.
  /// @param[out] value Resized dimension^2 by dimension^2 tensor matrix.
  /// @param transformation Element transformation with its current point set.
  /// @param point Reference evaluation point.
  void Eval(mfem::DenseMatrix &value,
            mfem::ElementTransformation &transformation,
            const mfem::IntegrationPoint &point) override {
    const mfem::real_t lambda = lambda_.Eval(transformation, point);
    const mfem::real_t mu = mu_.Eval(transformation, point);
    value.SetSize(height, width);
    for (int row = 0; row < height; ++row) {
      const int test_field = row / dimension_;
      const int test_direction = row % dimension_;
      for (int column = 0; column < width; ++column) {
        const int trial_field = column / dimension_;
        const int trial_direction = column % dimension_;
        value(row, column) =
            lambda * (test_field == test_direction &&
                      trial_field == trial_direction) +
            mu * ((test_field == trial_field &&
                   test_direction == trial_direction) +
                  (test_field == trial_direction &&
                   test_direction == trial_field));
      }
    }
  }
};
} // namespace mfem

#endif
