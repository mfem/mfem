
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

const Array<int> &GetDofMap(FiniteElementSpace &fes, int i);
Array<int> ComputeVectorFE_LORPermutation(FiniteElementSpace &fes_ho,
                                          FiniteElementSpace &fes_lor,
                                          FiniteElement::MapType type);

class LORSolver : public Solver
{

private:
   int n1;
   int n2;
   Array<int> perm;
   Array<int> p;
   Solver *solv=nullptr;
public:
   LORSolver(HypreParMatrix & A, const Array<int> p_, 
                         bool exact = true, Solver * prec = nullptr);
void SetOperator(const Operator&) { }

void Mult(const Vector &b, Vector &x) const;
};

class ComplexLORSolver : public Solver
{

private:
   int n1;
   int n2;
   Array<int> perm;
   Array<int> p;
   Solver *solv=nullptr;
public:
   ComplexLORSolver(HypreParMatrix & A, const Array<int> p_, 
                    bool exact = true, Solver * prec = nullptr);
void SetOperator(const Operator&) { }

void Mult(const Vector &b, Vector &x) const;
};