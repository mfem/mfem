
#include "parproblems_util.hpp"

class ParElasticityProblem
{
private:
    MPI_Comm comm;
    ParMesh * pmesh = nullptr;
    int order;
    int ndofs;
    int gndofs;
    FiniteElementCollection * fec = nullptr;
    ParFiniteElementSpace * fes = nullptr;
    Vector lambda, mu;
    Array<int> ess_bdr, ess_tdof_list;
    ParBilinearForm *a=nullptr;
    ParLinearForm b;
    ParGridFunction x;
    HypreParMatrix A;
    Vector B,X;
    void Setup();
public:
    ParElasticityProblem(MPI_Comm comm_, const char *mesh_file , int order_ = 1) : comm(comm_), order(order_) 
    {
        Mesh * mesh = new Mesh(mesh_file,1,1);
        pmesh = new ParMesh(comm,*mesh);
        delete mesh;
        Setup();
    }

    ParMesh * GetMesh() { return pmesh; }
    ParFiniteElementSpace * GetFESpace() { return fes; }
    FiniteElementCollection * GetFECol() { return fec; }
    int GetNumDofs() { return ndofs; }
    int GetGlobalNumDofs() { return gndofs; }
    HypreParMatrix & GetOperator() { return A; }
    Vector & GetRHS() { return B; }
    ParGridFunction & GetGridFunction() {return x;};

    ~ParElasticityProblem()
    {
        delete a;
        delete fes;
        delete fec;
        delete pmesh;
    }
};


class ParContactProblem
{
private:
    ParElasticityProblem * prob1 = nullptr;
    ParElasticityProblem * prob2 = nullptr;
public:
    ParContactProblem(ParElasticityProblem * prob1_, ParElasticityProblem * prob2_);
};