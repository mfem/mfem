#ifndef RAND_EIGENSOLVER_HPP
#define RAND_EIGENSOLVER_HPP

#include "mfem.hpp"


namespace mfem{

class RandomizedSubspaceIteration
{
public:

    RandomizedSubspaceIteration()
    {
#ifdef MFEM_USE_MPI
        comm=MPI_COMM_WORLD;
#endif
        num_modes=1;
        A=nullptr;
        modes.resize(num_modes);
        symmetric=false;
    }

#ifdef MFEM_USE_MPI
    RandomizedSubspaceIteration(MPI_Comm comm_, bool symm=false)
    {
        comm=comm_;
        num_modes=1;
        A=nullptr;
        modes.resize(num_modes);
        symmetric=symm;
    }
#endif

    void SetNumModes(int num_)
    {
        num_modes=num_;
        modes.resize(num_modes);
        if(A!=nullptr)
        {
            for(int i=0;i<num_modes;i++){
                modes[i].SetSize(A->NumRows());
            }
        }
    }

    void SetNumIter(int it_){
        iter=it_;
    }

    void SetOperator(const Operator& A_, bool symm=false)
    {
        A=&A_;
        for(int i=0;i<num_modes;i++){
            modes[i].SetSize(A->NumRows());
        }
        symmetric=symm;
    }

    void Solve();

    void GetMode(int i, Vector& q){
        if((i<num_modes)&&(A!=nullptr))
        {
            q=modes[i];
        }
    }

    std::vector<Vector>& GetModes()
    {
        return modes;
    }

private:
#ifdef MFEM_USE_MPI
    MPI_Comm comm;
#endif

    int num_modes;
    int iter;
    const Operator* A;
    std::vector<Vector> modes;

    bool symmetric;

};



};
#endif // RAND_EIGENSOLVER_HPP
