// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
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

#include "volume_fractions.hpp"

using namespace std;


namespace mfem
{


  void UpdateAlpha(ParGridFunction &alpha, ParFiniteElementSpace &h1_fes, ParGridFunction &func, const int &type)
  {
    IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
    auto pfes = alpha.ParFESpace();
    const IntegrationRule &ir = IntRulesLo.Get(pfes->GetFE(0)->GetGeomType(), 25 ); // second slot is 25
    const int NE = alpha.ParFESpace()->GetNE(), nqp = ir.GetNPoints();
    for (int e = 0; e < NE; e++)
      {
	ElementTransformation &Tr = *(pfes->GetElementTransformation(e));
	double volume_1 = 0.0, volume = 0.0;
	for (int q = 0; q < nqp; q++)
	  {
	    const IntegrationPoint &ip = ir.IntPoint(q);
	    Tr.SetIntPoint(&ip);
	    double distVal = func.GetValue(Tr,ip);
	    const bool checkIntegrationPtLocation = (distVal >= 0.0) ? 1.0:0.0;   
	    volume   += ip.weight * Tr.Weight();
	    volume_1 += ip.weight * Tr.Weight() * checkIntegrationPtLocation;
	  }  
	alpha(e) = volume_1 / volume;
	if (alpha(e) == 0.0){
	  const FiniteElement *FElem = h1_fes.GetFE(e);
	  const IntegrationRule &ir = FElem->GetNodes();
	  ElementTransformation &T = *(h1_fes.GetElementTransformation(e));
	  for (int j = 0; j < ir.GetNPoints(); j++)
	    {
	      const IntegrationPoint &ip = ir.IntPoint(j);
	      double distVal = func.GetValue(T,ip);
	      const bool checkIntegrationPtLocation = (distVal >= 0.0) ? 1.0:0.0;   
	  
	      if(checkIntegrationPtLocation){
		alpha(e) = 1.0/nqp;
		break;
	      }
	    }
	}
      }
  }
}
 // namespace mfem
