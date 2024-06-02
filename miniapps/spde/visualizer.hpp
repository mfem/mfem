// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#include "mfem.hpp"

namespace mfem
{
namespace spde
{

class Visualizer
{
public:
   Visualizer(ParMesh &mesh, int order, GridFunction &g1, GridFunction &g2,
              GridFunction &g3, GridFunction &g4, bool is_3D = true)
      : mesh_(&mesh),
        order_(order),
        g1_(g1),
        g2_(g2),
        g3_(g3),
        g4_(g4),
        is_3D_(is_3D) {}

   void ExportToParaView();
   void SendToGLVis() const;

private:
   ParMesh *mesh_;
   int order_;
   GridFunction &g1_;
   GridFunction &g2_;
   GridFunction &g3_;
   GridFunction &g4_;
   bool is_3D_;
};

}  // namespace spde
}  // namespace mfem
