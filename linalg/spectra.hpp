#ifndef MFEM_SPECTRA_HPP
#define MFEM_SPECTRA_HPP

#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>

#include "eigen.hpp"

using namespace Spectra;

namespace mfem {
    class SpectraEigenSolver {
    public:
        SpectraEigenSolver();

        virtual ~SpectraEigenSolver();

        /// Set dimension of Krylov subspace in the Lanczos method
        SpectraEigenSolver& SetKrylov(double ncv);

        /// Set solver tolerance
        SpectraEigenSolver& SetTol(double tol);

        /// Set maximum number of iterations
        SpectraEigenSolver& SetMaxIter(int max_iter);

        /// Set the number of required eigenmodes
        SpectraEigenSolver& SetNumModes(int nev);

        /// Set operator for standard eigenvalue problem (A*x = lambda*x)
        SpectraEigenSolver& SetOperator(const Operator& A);

        /// Set operator for generalized eigenvalue problem (A*x = lambda*B*x)
        SpectraEigenSolver& SetOperators(const Operator& A, const Operator& B);

        /// Solve the eigenvalue problem for the specified number of eigenvalues
        void Solve();

        /// Get the number of converged eigenvalues
        int GetNumConverged();

        /// Get the corresponding eigenvalue
        double GetEigenvalue(unsigned int i) const;

        Eigen::VectorXd GetEigenvalues(unsigned int i = 0) const;

        /// Get the corresponding eigenvector
        Eigen::VectorXd GetEigenvector(unsigned int i) const;

        Eigen::MatrixXd GetEigenvectors(unsigned int i) const;

    protected:
        // Params
        int _nconv, _nev, _ncv, _max_iter;
        double _tol;

        // EIGEN Operators
        Eigen::SparseMatrix<double> _A_e, _B_e;

        // Spectra Operators
        SparseSymMatProd<double>* _A_s = nullptr;
        SparseCholesky<double>* _B_s = nullptr;

        // Eigenvalue solution based on Spectra
        SymEigsSolver<SparseSymMatProd<double>>* _S = nullptr;
        SymGEigsSolver<SparseSymMatProd<double>, SparseCholesky<double>, GEigsMode::Cholesky>* _G = nullptr;

        // // Eigenvalue solution based on Eigen
        // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>* _S = nullptr;
        // Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd>* _G = nullptr;
    };
} // namespace mfem

#endif // MFEM_SPECTRA_HPP