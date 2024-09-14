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
    const HypreParMatrix * P = nullptr;
    HypreBoomerAMG * amg = nullptr;
    HypreParMatrix * Ac = nullptr;
    MUMPSSolver * M = nullptr;
    bool additive = false;
    int relax_type = 88;
    void Init(MPI_Comm comm_);
    void InitAMG();
    void InitMumps();
public:
    TwoLevelAMGSolver(MPI_Comm comm_);
    TwoLevelAMGSolver(const Operator & Op, const Operator & P_);
    void SetOperator(const Operator &op);
    void SetTransferMap(const Operator & P_);
    void EnableAdditiveCoupling() { additive = true; }
    void EnableMultiplicativeCoupling() { additive = false; }
    void SetAMGRelaxTypre(int relax_type_) { relax_type = relax_type_;  }

    virtual void Mult(const Vector & y, Vector & x) const; 

    ~TwoLevelAMGSolver()
    {
        delete amg;
        delete Ac;
        delete M;
    }
};

}

#endif