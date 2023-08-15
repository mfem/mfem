// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CUT_INTEGRATORS_HPP
#define MFEM_CUT_INTEGRATORS_HPP

#include "mfem.hpp"
#include "marking.hpp"

namespace mfem
{

/// Cut integrator for evaluating a volume and its gradients
/// with respect to displacements of the mesh nodes
class CutVolLagrangeIntegrator: public NonlinearFormIntegrator
{

public:
    CutVolLagrangeIntegrator(int io=2)
    {
        coeff=nullptr;
        cint=nullptr;
        int_order=io;
    }

    virtual
    ~CutVolLagrangeIntegrator()
    {

    }

    void SetCutIntegration(CutIntegrationRules* cint_)
    {
        cint=cint_;
    }

    void SetCoefficient(Coefficient& cf){
        coeff=&cf;
    }

    void SetIntOrder(int io=2)
    {
        int_order=io;
    }

    /// Perform the local action of the NonlinearFormIntegrator -
    /// evaluates the gradients with respect to displacements of
    /// the mesh nodes
    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       const Vector &elfun, Vector &elvect);


    /// Compute the local energy, i.e., the volume
    virtual double GetElementEnergy(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        const Vector &elfun);

private:
    CutIntegrationRules* cint; //cut integration rulle
    Coefficient* coeff;
    int int_order;
};


}
#endif
