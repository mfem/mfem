// Copyright A(c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "p_divW.hpp"
#include <unordered_map>

namespace mfem
{

  void PDivWForceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
						    const FiniteElement &test_fe,
						    ElementTransformation &Trans,
						    DenseMatrix &elmat)
  {
    const int dim = trial_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();

    elmat.SetSize (test_dof*dim, trial_dof);
    elmat = 0.0;
    
    DenseMatrix dshape(test_dof,dim), dshape_ps(test_dof,dim), adjJ(dim);
    Vector shape(trial_dof),divshape(dim*test_dof);
    shape = 0.0;
    dshape = 0.0;
    dshape_ps = 0.0;
    adjJ = 0.0;
    divshape = 0.0;
    
    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
                                                            Trans);
  
    for (int q = 0; q < ir->GetNPoints(); q++)
    {
      const IntegrationPoint &ip = ir->IntPoint(q);
      // Set the integration point in the face and the neighboring elements
      Trans.SetIntPoint(&ip);
    
      test_fe.CalcDShape (ip, dshape);
      trial_fe.CalcShape (ip, shape);

      CalcAdjugate(Trans.Jacobian(), adjJ);

      Mult(dshape, adjJ, dshape_ps);

      dshape_ps.GradToDiv(divshape);

      shape *= -ip.weight;

      AddMultVWt (divshape, shape, elmat);
    }
  }

  const IntegrationRule &PDivWForceIntegrator::GetRule(
						       const FiniteElement &trial_fe,
						       const FiniteElement &test_fe,
						       ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   return IntRules.Get(trial_fe.GetGeomType(), 2*order);
}

}
