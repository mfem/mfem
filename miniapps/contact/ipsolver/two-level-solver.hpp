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
    HypreBoomerAMG * S = nullptr;

    const HypreParMatrix * Pi = nullptr;
    HypreParMatrix * Ai = nullptr;
    HypreBoomerAMG * Si = nullptr;

    const HypreParMatrix * Pc = nullptr;
    HypreParMatrix * Ac = nullptr;
    MUMPSSolver * Sc = nullptr;

    int relax_type = 88;
    void Init(MPI_Comm comm_);
    void InitAMG();
    void InitMumps();
public:
    TwoLevelAMGSolver(MPI_Comm comm_);
    TwoLevelAMGSolver(const Operator & Op, const Operator & P_);
    void SetOperator(const Operator &op);
    void SetContactTransferMap(const Operator & P);
    void SetNonContactTransferMap(const Operator & P);
    void SetAMGRelaxType(int relax_type_) { relax_type = relax_type_;  }

    virtual void Mult(const Vector & y, Vector & x) const; 

    ~TwoLevelAMGSolver()
    {
        delete S;
        delete Ai;
        delete Si;
        delete Ac;
        delete Sc;
    }
};

class TwoLevelContactSolver : public Solver
{
private:
    MPI_Comm comm;
    int numProcs, myid;
    const HypreParMatrix * K = nullptr;
    const HypreParMatrix * A = nullptr;
    const HypreParMatrix * JtDJ = nullptr;
    HypreBoomerAMG * S = nullptr;

    const HypreParMatrix * Pi = nullptr;
    HypreParMatrix * Ai = nullptr;
    HypreBoomerAMG * Si = nullptr;

    const HypreParMatrix * Pc = nullptr;
    HypreParMatrix * Ac = nullptr;
    MUMPSSolver * Sc = nullptr;

    int relax_type = 88;
    void Init(MPI_Comm comm_);
    void InitAMG();
    void InitMumps();
public:
    TwoLevelContactSolver(MPI_Comm comm_);
    TwoLevelContactSolver(const Operator & A_, const Operator & D_, const Operator & Pi_, const Operator & Pc_);
    void SetOperator(const Operator &K_);
    void SetContactTransferMap(const Operator & P);
    void SetNonContactTransferMap(const Operator & P);
    void SetAMGRelaxType(int relax_type_) { relax_type = relax_type_;  }

    virtual void Mult(const Vector & y, Vector & x) const; 

    ~TwoLevelContactSolver()
    {
        delete S;
        delete Ai;
        delete Si;
        delete Ac;
        delete Sc;
    }
};


}

#endif