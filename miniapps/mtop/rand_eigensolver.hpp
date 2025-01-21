#ifndef RAND_EIGENSOLVER_HPP
#define RAND_EIGENSOLVER_HPP

#include "mfem.hpp"

#include "../config/config.hpp"

namespace mfem{

class RandomizedEigenSolver{
public:

    RandomizedEigenSolver(MPI_Comm comm);

    ~RandomizedEigenSolver();

    void SetNumModes(int num_);

    void SetMaxIter(int max_it_);

    void SetTol(real_t tol_);

    void SetShift(real_t shift_);

    void SetOperator(const Operator& A_);

    void SetOperators(const Operator& A_, const Operator& B_);

    void Solve();

    int GetNumConverged();

    void GetEigenvalue(unsigned int i, real_t & lr) const;

    void GetEigenvalue(unsigned int i, real_t & lr, real_t & lc) const;

    void GetEigenvector(unsigned int i, Vector & vr) const;

    void GetEigenvector(unsigned int i, Vector & vr, Vector & vc) const;

private:

    Operator* A;
    Operator* B;

};


}
#endif // RAND_EIGENSOLVER_HPP
