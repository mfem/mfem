// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef LINEARFORM_EXT_HPP
#define LINEARFORM_EXT_HPP

#include "../config/config.hpp"
#include "fespace.hpp"
#include "lininteg.hpp"
#include "../general/device.hpp"

namespace mfem
{

class LinearForm;

/// Class extending the LinearForm class.
///
class LinearFormExtension
{
protected:
   LinearForm *lf; ///< Not owned

public:
   LinearFormExtension(LinearForm*);
   virtual ~LinearFormExtension();

   /// Assemble at the level given for the LinearFormExtension subclass
   virtual void Assemble() = 0;

};

/// Data and methods for fully-assembled linear forms
class FullLinearFormExtension : public LinearFormExtension
{
protected:
   const FiniteElementSpace &fes; // Not owned
   const Mesh &mesh;
   const Array<LinearFormIntegrator*> &domain_integs;
   const Array<Array<int>*> &domain_integs_marker;
   Vector marks, attributes;
   const int ne, mesh_attributes_size;

public:
   FullLinearFormExtension(LinearForm*);

   void Assemble() override;
};

} // namespace mfem

#endif // LINEARFORM_EXT_HPP
