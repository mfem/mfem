#include "spectra.hpp"

#include "../fem/bilinearform.hpp"

namespace mfem
{
    SpectraEigenSolver::SpectraEigenSolver()
    {
        // Init params
        _nconv = 0;
        _nev = 1;
        _ncv = 1;
        _max_iter = 1000;
        _tol = 1e-3;
    }

    SpectraEigenSolver::~SpectraEigenSolver()
    {
        delete _A_s, _B_s, _S, _G;
    }

    /// Set dimension of Krylov subspace in the Lanczos method
    SpectraEigenSolver& SpectraEigenSolver::SetKrylov(double ncv)
    {
        _ncv = ncv;

        return *this;
    }

    /// Set solver tolerance
    SpectraEigenSolver& SpectraEigenSolver::SetTol(double tol)
    {
        _tol = tol;

        return *this;
    }

    /// Set maximum number of iterations
    SpectraEigenSolver& SpectraEigenSolver::SetMaxIter(int max_iter)
    {
        _max_iter = max_iter;

        return *this;
    }

    /// Set the number of required eigenmodes
    SpectraEigenSolver& SpectraEigenSolver::SetNumModes(int nev)
    {
        _nev = nev;

        return *this;
    }

    /// Set operator for standard eigenvalue problem (A*x = lambda*x)
    SpectraEigenSolver& SpectraEigenSolver::SetOperator(const Operator& A)
    {
        // Set EIGEN operators
        _A_e = SparseMatrixConverter<double>::to(static_cast<const mfem::BilinearForm&>(A).SpMat());
        
        // Set SPECTRA operators
        _A_s = new SparseSymMatProd<double>(_A_e);

        return *this;
    }

    /// Set operator for generalized eigenvalue problem (A*x = lambda*B*x)
    SpectraEigenSolver& SpectraEigenSolver::SetOperators(const Operator& A, const Operator& B)
    {
        // Set EIGEN operators
        _A_e = SparseMatrixConverter<double>::to(static_cast<const mfem::BilinearForm&>(A).SpMat());
        _B_e = SparseMatrixConverter<double>::to(static_cast<const mfem::BilinearForm&>(B).SpMat());
        
        // Set SPECTRA operators
        _A_s = new SparseSymMatProd<double>(_A_e);
        _B_s = new SparseCholesky<double>(_B_e);

        return *this;
    }

    /// Solve the eigenvalue problem for the specified number of eigenvalues
    void SpectraEigenSolver::Solve()
    {
        // Set the dimension of the Krilov space equal to the number of requested eigenvalues if necessary
        if (_ncv < _nev)
            _ncv = _nev;

        if (!_B_s) {
            _S = new SymEigsSolver<SparseSymMatProd<double>>(*_A_s, _nev, _ncv);
            _S->init();
            _nconv = _S->compute(SortRule::SmallestMagn, _max_iter, _tol, SortRule::SmallestMagn);
        }
        else {
            _G = new SymGEigsSolver<SparseSymMatProd<double>, SparseCholesky<double>, GEigsMode::Cholesky>(*_A_s, *_B_s, _nev, _ncv);
            _G->init();
            _nconv = _G->compute(SortRule::SmallestMagn, _max_iter, _tol, SortRule::SmallestMagn);
        }
    }

    /// Get the number of converged eigenvalues
    int SpectraEigenSolver::GetNumConverged()
    {
        return _nconv;
    }

    /// Get the corresponding eigenvalue
    double SpectraEigenSolver::GetEigenvalue(unsigned int i) const
    {
        if (!_B_s) {
            if (_S->info() == CompInfo::Successful && i < _nconv)
                return _S->eigenvalues()[i];
            else
                return 0;
        }
        else {
            if (_G->info() == CompInfo::Successful && i < _nconv)
                return _G->eigenvalues()[i];
            else
                return 0;
        }
    }

    Eigen::VectorXd SpectraEigenSolver::GetEigenvalues(unsigned int i) const
    {
        if (!_B_s) {
            if (_S->info() == CompInfo::Successful && i < _nconv)
                return _S->eigenvalues().segment(0, i);
        }
        else {
            if (_G->info() == CompInfo::Successful && i < _nconv)
                return _G->eigenvalues().segment(0, i);
        }
    }

    /// Get the corresponding eigenvector
    Eigen::VectorXd SpectraEigenSolver::GetEigenvector(unsigned int i) const
    {
        if (!_B_s) {
            if (_S->info() == CompInfo::Successful && i < _nconv)
                return _S->eigenvectors().col(i);
        }
        else {
            if (_G->info() == CompInfo::Successful && i < _nconv)
                return _G->eigenvectors().col(i);
        }
    }

    Eigen::MatrixXd SpectraEigenSolver::GetEigenvectors(unsigned int i) const
    {
        if (!_B_s) {
            if (_S->info() == CompInfo::Successful && i < _nconv)
                return _S->eigenvectors().topRows(i);
        }
        else {
            if (_G->info() == CompInfo::Successful && i < _nconv)
                return _G->eigenvectors().topRows(i);
        }
    }
}