// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "gridfunc.hpp"
#include "bilininteg.hpp"
#include "../../fem/gridfunc.hpp"

namespace mfem
{

namespace occa
{

std::map<std::string, ::occa::kernel> gridFunctionKernels;

::occa::kernel GetGridFunctionKernel(::occa::device device,
                                     FiniteElementSpace &fespace,
                                     const mfem::IntegrationRule &ir)
{
   const int numQuad = ir.GetNPoints();

   const FiniteElement &fe = *(fespace.GetFE(0));
   const int dim  = fe.GetDim();
   const int vdim = fespace.GetVDim();

   std::stringstream ss;
   ss << ::occa::hash(device)
      << "FEColl : " << fespace.FEColl()->Name()
      << "Quad: "    << numQuad
      << "Dim: "     << dim
      << "VDim: "    << vdim;
   std::string hash = ss.str();

   // Kernel defines
   ::occa::properties props;
   props["defines/NUM_VDIM"] = vdim;

   SetProperties(fespace, ir, props);

   ::occa::kernel kernel = gridFunctionKernels[hash];
   if (!kernel.isInitialized())
   {
      const std::string &okl_path = fespace.OccaEngine().GetOklPath();
      kernel = device.buildKernel(okl_path + "gridfunc.okl",
                                  stringWithDim("GridFuncToQuad", dim),
                                  props);
   }
   return kernel;
}

void ToQuad(const IntegrationRule &ir, FiniteElementSpace &fespace, Vector &gf,
            Vector &quadValues)
{
   const Engine &engine = fespace.OccaEngine();
   ::occa::device device = engine.GetDevice();

   OccaDofQuadMaps &maps = OccaDofQuadMaps::Get(device, fespace, ir);

   const int elements = fespace.GetNE();
   const int numQuad  = ir.GetNPoints();
   quadValues.OccaResize(numQuad * elements, sizeof(double));

   ::occa::kernel g2qKernel = GetGridFunctionKernel(device, fespace, ir);
   g2qKernel(elements,
             maps.dofToQuad,
             fespace.GetLocalToGlobalMap(),
             gf.OccaMem(),
             quadValues.OccaMem());
}

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
