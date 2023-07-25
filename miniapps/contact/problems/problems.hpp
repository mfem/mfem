#include "problems_util.hpp"


class ElasticityProblem
{
private:
    Mesh * mesh = nullptr;
    int order;
    int ndofs;
    FiniteElementCollection * fec = nullptr;
    FiniteElementSpace * fes = nullptr;
    Vector lambda, mu;
    Array<int> ess_bdr, ess_tdof_list;
    BilinearForm *a=nullptr;
    LinearForm b;
    GridFunction x;
    SparseMatrix A;
    Vector B,X;
    void Setup();
public:
    ElasticityProblem(const char *mesh_file , int order_ = 1) : order(order_) 
    {
        mesh = new Mesh(mesh_file,1,1);
        Setup();
    }

    Mesh * GetMesh() { return mesh; }
    FiniteElementSpace * GetFESpace() { return fes; }
    int GetNumDofs() { return ndofs; }
    SparseMatrix & GetOperator() { return A; }
    Vector & GetRHS() { return B; }
    GridFunction & GetGridFunction() {return x;};

    ~ElasticityProblem()
    {
        delete a;
        delete fes;
        delete fec;
        delete mesh;
    }
};


class ContactProblem
{
private:
    ElasticityProblem * prob1 = nullptr;
    ElasticityProblem * prob2 = nullptr;
public:
    ContactProblem(ElasticityProblem * prob1_, ElasticityProblem * prob2_);
};


