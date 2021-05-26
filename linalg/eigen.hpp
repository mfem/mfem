#ifndef MFEM_EIGEN_HPP
#define MFEM_EIGEN_HPP

#include <vector>
#include <Eigen/Sparse>

#include "vector.hpp"
#include "sparsemat.hpp"
#include "densemat.hpp"

namespace mfem{
    /** @brief Eigen template specialization for vector conversion */
    template <typename T>
    struct VectorConverter {
        static Vector from(const Eigen::Matrix<T, Eigen::Dynamic, 1>& other)
        {
            Vector v(other.rows());

            for (size_t i = 0; i < v.Size(); i++)
                v(i) = other(i);

            return std::move(v);
        }

        static Eigen::Matrix<T, Eigen::Dynamic, 1> to(const Vector& other)
        {
            Eigen::Matrix<T, Eigen::Dynamic, 1> v(other.Size());

            for (size_t i = 0; i < v.Size(); i++)
                v(i) = other(i);

            return std::move(v);
        }
    };

    /** @brief Eigen template specialization for dense matrix conversion */
    template <typename T>
    struct DenseMatrixConverter {
        static DenseMatrix from(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& other)
        {
            DenseMatrix mat(other.rows(), other.cols());

            for (size_t j = 0; j < mat.Width(); j++)
                for (size_t i = 0; i < mat.Height(); i++)
                    mat(i, j) = other(i, j);

            return mat;
        }

        static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> to(const DenseMatrix& other)
        {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat(other.Height(), other.Width());

            for (size_t j = 0; j < mat.cols(); j++)
                for (size_t i = 0; i < mat.rows(); i++)
                    mat(i, j) = other(i, j);

            return mat;
        }
    };

    /** @brief Eigen template specialization for sparse matrix conversion */
    template <class T>
    struct SparseMatrixConverter {
        static SparseMatrix from(const Eigen::SparseMatrix<T, Eigen::RowMajor>& other)
        {
            return SparseMatrix(other.outerIndexPtr(), other.innerIndexPtr(), other.valuePtr(), other.rows(), other.cols());
        }

        static Eigen::SparseMatrix<T, Eigen::RowMajor> to(const SparseMatrix& other)
        {
            // MFEM memory info
            const int *I = other.GetI(), *J = other.GetJ();
            const T* Data = other.GetData();

            // Eigen triplet
            std::vector<Eigen::Triplet<double>> tripletList;
            tripletList.reserve(other.GetMemoryData().Capacity());

            for (size_t i = 0; i < other.Size(); i++) {
                for (size_t k = I[i], end = I[i + 1]; k < end; k++)
                    tripletList.push_back(Eigen::Triplet<double>(i, J[k], Data[k]));
            }

            // Create Eigen sparse matrix
            Eigen::SparseMatrix<T, Eigen::RowMajor> mat(other.Height(), other.Width());
            mat.setFromTriplets(tripletList.begin(), tripletList.end());

            return mat;
        }
    };   
}

#endif // MFEM_EIGEN_HPP