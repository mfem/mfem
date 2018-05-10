// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include <sys/time.h>

#include "mfem.hpp"
#include "../../laghos_solver.hpp"

namespace mfem {
  
namespace hydrodynamics {

  // ***************************************************************************
  bool lambdaTest(ParMesh *pmesh, const int order_v, const int max_step){
    const int order_e = order_v-1;
    struct timeval st, et;
    const int nb_step = (max_step>0)?max_step:100;
    assert(order_v>0);
    assert(order_e>0);
    const int dim = pmesh->Dimension();
    const L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
    const H1_FECollection H1FEC(order_v, dim);
    RajaFiniteElementSpace L2FESpace(pmesh, &L2FEC);
    RajaFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
    RajaFiniteElementSpace H1compFESpace(H1FESpace.GetParMesh(), H1FESpace.FEColl(),1);
    const int lsize = H1FESpace.GetVSize();
    const int gsize = H1FESpace.GlobalTrueVSize();
    const int nzones = H1FESpace.GetMesh()->GetNE();
    if (rconfig::Get().Root())
      mfem::out << "Number of global dofs: " << gsize << std::endl;
    if (rconfig::Get().Root())
      mfem::out << "Number of local dofs: " << lsize << std::endl;
    const IntegrationRule &integ_rule=IntRules.Get(H1FESpace.GetMesh()->GetElementBaseGeometry(),
                                                  3*H1FESpace.GetOrder(0) + L2FESpace.GetOrder(0) - 1);
    QuadratureData quad_data(dim, nzones, integ_rule.GetNPoints());
    // RajaBilinearForm::Mult
    const int vlsize = nzones * H1compFESpace.GetLocalDofs() * H1compFESpace.GetVDim();
    RajaVector B(vlsize);
    RajaVector X(vlsize);
    /*printf("\n\033[31m[lambda] localX size = %d, GetNE=%d, GetLocalDofs=%d, GetVDim=%d\033[m",
         X.Size(),nzones,
         H1compFESpace.GetLocalDofs(),
         H1compFESpace.GetVDim());*/
    quad_data.dqMaps = RajaDofQuadMaps::Get(H1FESpace,integ_rule);
    quad_data.geom = RajaGeometry::Get(H1FESpace,integ_rule);
    RajaGridFunction d_rho(L2FESpace);
    RajaVector rhoValues; 
    d_rho = 1.0;
    d_rho.ToQuad(integ_rule, rhoValues);
    if (dim==1) { assert(false); }
    const int NUM_QUAD = integ_rule.GetNPoints();
    rInitQuadratureData(NUM_QUAD,
                        nzones,
                        rhoValues,
                        quad_data.geom->detJ,
                        quad_data.dqMaps->quadWeights,
                        quad_data.rho0DetJ0w);
    RajaOperator *massOperator;
    RajaBilinearForm bilinearForm(&H1compFESpace);
    RajaMassIntegrator &massInteg = *(new RajaMassIntegrator());
    massInteg.SetIntegrationRule(integ_rule);
    massInteg.SetOperator(quad_data.rho0DetJ0w);
    bilinearForm.AddDomainIntegrator(&massInteg);
    bilinearForm.Assemble();
    bilinearForm.FormOperator(Array<int>(), massOperator);
    // **************************************************************************
    MPI_Barrier(pmesh->GetComm());
#ifdef __NVCC__
    cudaDeviceSynchronize();
#endif
    // *************************************************************************
    // Now let go the markers
    rconfig::Get().Nvvp(true);
    // *************************************************************************
    gettimeofday(&st, NULL);
    B=1.0;
    X=0.0;
    for(int i=0;i<nb_step;i++){
      //massOperator->Mult(B, X);
      massInteg.MultAdd(B, X);
    }
    // We MUST sync after to make sure every kernel has completed
    // or play with the -sync flag to enforce it with the push/pop
#ifdef __NVCC__
    cudaDeviceSynchronize();
#endif
    gettimeofday(&et, NULL);
    const float Kalltime = (et.tv_sec-st.tv_sec)*1.0e6+(et.tv_usec - st.tv_usec);
    if (rconfig::Get().Root())
      printf("\033[32;1m[lambda] Elapsed time = %.1f us/step(%d)\33[m\n", Kalltime/nb_step,nb_step);
    return true;
  }
  
} //  namespace hydrodynamics
  
} // namespace mfem
