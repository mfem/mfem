#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


#ifndef TWOLEVELSOLVER 
#define TWOLEVELSOLVER

namespace mfem
{

class TwoLevelAMGSolver : public Solver
{
private:
    MPI_Comm comm;
    int numProcs, myid;
    const HypreParMatrix * A = nullptr;
    const HypreParMatrix * Pc = nullptr;
    const HypreParMatrix * Pnc = nullptr;
    HypreBoomerAMG * amg = nullptr;
    HypreParMatrix * Ac = nullptr;
    Solver * Mcoarse = nullptr; // previously a mumps solver
    bool additive = false;
    int relax_type = 88;
    void Init(MPI_Comm comm_);
    void InitAMG();
    void InitCoarseSolver();
public:
    TwoLevelAMGSolver(MPI_Comm comm_);
    TwoLevelAMGSolver(const Operator & Op, const Operator & P_);
    void SetOperator(const Operator &op);
    void SetContactTransferMap(const Operator & P);
    void SetNonContactTransferMap(const Operator & P);
    void EnableAdditiveCoupling() { additive = true; }
    void EnableMultiplicativeCoupling() { additive = false; }
    void SetAMGRelaxType(int relax_type_) { relax_type = relax_type_;  }

    virtual void Mult(const Vector & y, Vector & x) const; 

    ~TwoLevelAMGSolver()
    {
        delete amg;
        delete Ac;
        delete Mcoarse;
    }
};

}

#endif
