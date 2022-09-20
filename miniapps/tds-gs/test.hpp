#ifndef TEST
#define TEST

#include "mfem.hpp"
#include "initial_coefficient.hpp"
#include "plasma_model.hpp"
using namespace mfem;

/*
  Used to test saddle point calculator
 */
class TestCoefficient : public Coefficient
{
public:
  TestCoefficient() { }
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~TestCoefficient() { }
};

int test();

#endif
