// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_EPIC
#define MFEM_EPIC

#include "../config/config.hpp"

#ifdef MFEM_USE_EPIC

// SUNDIALS vectors
#include <nvector/nvector_serial.h>
#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <nvector/nvector_parallel.h>
#endif

#include "ode.hpp"
#include "solvers.hpp"
#include <Epic.h>

namespace mfem
{

typedef void (*JacobianFun)(const realtype t, const Vector &y, const Vector& v, Vector& Jv, void* user_data);
// ---------------------------------------------------------------------------
// Interface to the EPIC library -- exponential methods
// ---------------------------------------------------------------------------

class EPICSolver : public ODESolver
{
protected:
    EPICNumJacDelta Delta;
    Operator* Jtv;
    N_Vector temp;
    int m[2];

    bool exactJacobian;

    #ifdef MFEM_USE_MPI
    bool Parallel() const
    {
        return (N_VGetVectorID(temp) != SUNDIALS_NVEC_SERIAL);
    }
    #else
    bool Parallel() const { return false; }
    #endif

public:
    EPICSolver(bool exactJacobian, EPICNumJacDelta delta=&DefaultDelta);
    EPICSolver(MPI_Comm comm);

    static int RHS(realtype t, const N_Vector y, N_Vector ydot, void *user_data);
    static int Jacobian(N_Vector v, N_Vector Jv, realtype t,
                         N_Vector y, N_Vector fy, void *user_data, N_Vector tmp);
    virtual void Init(TimeDependentOperator &f);
    virtual void Step(Vector &x, double &t, double &dt);

    virtual ~EPICSolver() {}
};

class EPI2 : public EPICSolver
{
protected:
    Epi2_KIOPS* integrator;
public:
    EPI2(bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);
    virtual void Init(TimeDependentOperator &f);
    virtual void Step(Vector &x, double &t, double &dt);

    virtual ~EPI2();
};

class EPIRK4 : public EPICSolver
{
protected:
    EpiRK4SC_KIOPS* integrator;
public:
    EPIRK4(bool exactJacobian=true, EPICNumJacDelta delta=&DefaultDelta);
    virtual void Init(TimeDependentOperator &f);
    virtual void Step(Vector &x, double &t, double &dt);

    virtual ~EPIRK4();
};

} // namespace mfem

#endif // MFEM_USE_EPIC

#endif // MFEM_EPIC
