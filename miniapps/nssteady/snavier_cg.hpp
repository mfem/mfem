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

#ifndef MFEM_SNAVIER_CG_HPP
#define MFEM_SNAVIER_CG_HPP

#define SNAVIER_CG_VERSION 0.1

#include "mfem.hpp"

namespace mfem
{

class SNavierPicardCGSolver{
public:
    SNavierPicardCGSolver(ParMesh* mesh_,int vorder=2, int porder=1);

    ~SNavierPicardCGSolver();

    /// Set the Newton Solver
    void SetFixedPointSolver(double rtol=1e-7, double atol=1e-12,int miter=1000, int prt_level=1);

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1000);

    /// Solve the forward problem
    void FSolve();

    /// Returns the velocity solution vector.
    Vector& GetVSol(){return vsol;}

    /// Returns the pressure  solution vector.
    Vector& GetPSol(){return psol;}

    /// Returns the velocities.
    ParGridFunction& GetVelocity()
    {
        vgf.SetFromTrueDofs(vsol);
        return vgf;
    }

    /// Resturn the pressure field.
    ParGridFunction& GetPressure()
    {
        pgf.SetFromTrueDofs(psol);
        return pgf;
    }

private:
    ParMesh* pmesh;

    //velocity solution true vector
    Vector vsol;
    //pressure solution true vector
    Vector psol;
    //tmp true vectors
    Vector vtmp; //velocity size
    Vector ptmp; //pressure size


    //velocty grid function
    ParGridFunction vgf;
    //pressure grid function
    ParGridFunction pgf;

    //Newton/Fixed point solver parameters
    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    HypreBoomerAMG *prec; //preconditioner
    CGSolver *ls;  //linear solver

    // velocity FE space
    ParFiniteElementSpace* vfes;
    FiniteElementCollection* vfec;

    // pressure FE space
    ParFiniteElementSpace* pfes;
    FiniteElementCollection* pfec;

};



}

#endif

