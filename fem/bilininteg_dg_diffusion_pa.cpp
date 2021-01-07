
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

#include "../general/forall.hpp"
#include "bilininteg.hpp"

using namespace std;

namespace mfem
{

// Setup for interior faces
void DGDiffusionIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace& fes)
{
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
    SetupPA(fes, FaceType::Interior);
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
}

// Setup for boundary faces
void DGDiffusionIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace& fes)
{
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
    SetupPA(fes, FaceType::Boundary);
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
}

// Setup for partial assembly
void DGDiffusionIntegrator::SetupPA(const FiniteElementSpace &fes, FaceType type)
{
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
    std::cout << "TODO: Correct this for DG diffusion" << std::endl;
    exit(1);
    // TODO:  Add setup
    /*
    nf = fes.GetNFbyType(type);
    if (nf==0) { return; }
    // Assumes tensor-product elements
    Mesh *mesh = fes.GetMesh();
    const FiniteElement &el =
        *fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0));
    FaceElementTransformations &T =
        *fes.GetMesh()->GetFaceElementTransformations(0);
    const IntegrationRule *ir = IntRule?
                                IntRule:
                                &GetRule(el.GetGeomType(), el.GetOrder(), T);
    const int symmDims = 4;
    const int nq = ir->GetNPoints();
    dim = mesh->Dimension();
    geom = mesh->GetFaceGeometricFactors(
                *ir,
                FaceGeometricFactors::DETERMINANTS |
                FaceGeometricFactors::NORMALS, type);
    maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
    dofs1D = maps->ndof;
    quad1D = maps->nqpt;
    pa_data.SetSize(symmDims * nq * nf, Device::GetMemoryType());
    Vector vel;
    if (VectorConstantCoefficient *c_u = dynamic_cast<VectorConstantCoefficient*>
                                        (u))
    {
        vel = c_u->GetVec();
    }
    else if (VectorQuadratureFunctionCoefficient* c_u =
                dynamic_cast<VectorQuadratureFunctionCoefficient*>(u))
    {
        // Assumed to be in lexicographical ordering
        const QuadratureFunction &qFun = c_u->GetQuadFunction();
        MFEM_VERIFY(qFun.Size() == dim * nq * nf,
                    "Incompatible QuadratureFunction dimension \n");

        MFEM_VERIFY(ir == &qFun.GetSpace()->GetElementIntRule(0),
                    "IntegrationRule used within integrator and in"
                    " QuadratureFunction appear to be different");
        qFun.Read();
        vel.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
    }
    else
    {
        vel.SetSize(dim * nq * nf);
        auto C = Reshape(vel.HostWrite(), dim, nq, nf);
        Vector Vq(dim);
        int f_ind = 0;
        for (int f = 0; f < fes.GetNF(); ++f)
        {
            int e1, e2;
            int inf1, inf2;
            fes.GetMesh()->GetFaceElements(f, &e1, &e2);
            fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
            int face_id = inf1 / 64;
            if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
                (type==FaceType::Boundary && e2<0 && inf2<0) )
            {
            FaceElementTransformations &T =
                *fes.GetMesh()->GetFaceElementTransformations(f);
            for (int q = 0; q < nq; ++q)
            {
                // Convert to lexicographic ordering
                int iq = ToLexOrdering(dim, face_id, quad1D, q);
                T.SetAllIntPoints(&ir->IntPoint(q));
                const IntegrationPoint &eip1 = T.GetElement1IntPoint();
                u->Eval(Vq, *T.Elem1, eip1);
                for (int i = 0; i < dim; ++i)
                {
                    C(i,iq,f_ind) = Vq(i);
                }
            }
            f_ind++;
            }
        }
        MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");
    }
    Vector r;
    if (rho==nullptr)
    {
        r.SetSize(1);
        r(0) = 1.0;
    }
    else if (ConstantCoefficient *c_rho = dynamic_cast<ConstantCoefficient*>(rho))
    {
        r.SetSize(1);
        r(0) = c_rho->constant;
    }
    else if (QuadratureFunctionCoefficient* c_rho =
                dynamic_cast<QuadratureFunctionCoefficient*>(rho))
    {
        const QuadratureFunction &qFun = c_rho->GetQuadFunction();
        MFEM_VERIFY(qFun.Size() == nq * nf,
                    "Incompatible QuadratureFunction dimension \n");

        MFEM_VERIFY(ir == &qFun.GetSpace()->GetElementIntRule(0),
                    "IntegrationRule used within integrator and in"
                    " QuadratureFunction appear to be different");
        qFun.Read();
        r.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
    }
    else
    {
        r.SetSize(nq * nf);
        auto C_vel = Reshape(vel.HostRead(), dim, nq, nf);
        auto n = Reshape(geom->normal.HostRead(), nq, dim, nf);
        auto C = Reshape(r.HostWrite(), nq, nf);
        int f_ind = 0;
        for (int f = 0; f < fes.GetNF(); ++f)
        {
            int e1, e2;
            int inf1, inf2;
            fes.GetMesh()->GetFaceElements(f, &e1, &e2);
            fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
            int face_id = inf1 / 64;
            if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
                (type==FaceType::Boundary && e2<0 && inf2<0) )
            {
            FaceElementTransformations &T =
                *fes.GetMesh()->GetFaceElementTransformations(f);
            for (int q = 0; q < nq; ++q)
            {
                // Convert to lexicographic ordering
                int iq = ToLexOrdering(dim, face_id, quad1D, q);

                T.SetAllIntPoints(&ir->IntPoint(q));
                const IntegrationPoint &eip1 = T.GetElement1IntPoint();
                const IntegrationPoint &eip2 = T.GetElement2IntPoint();
                double r;

                if (inf2 < 0)
                {
                    r = rho->Eval(*T.Elem1, eip1);
                }
                else
                {
                    double udotn = 0.0;
                    for (int d=0; d<dim; ++d)
                    {
                        udotn += C_vel(d,iq,f_ind)*n(iq,d,f_ind);
                    }
                    if (udotn >= 0.0) { r = rho->Eval(*T.Elem2, eip2); }
                    else { r = rho->Eval(*T.Elem1, eip1); }
                }
                C(iq,f_ind) = r;
            }
            f_ind++;
            }
        }
        MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");
    }   
    PADGDiffusionSetup(dim, dofs1D, quad1D, nf, ir->GetWeights(),
                    geom->detJ, geom->normal, r, vel,
                    alpha, beta, pa_data);
    */
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
}

static void PADGDiffusionSetup2D(const int Q1D,
                             const int NF,
                             const Array<double> &w,
                             const Vector &det,
                             const Vector &nor,
                             const Vector &rho,
                             const Vector &vel,
                             const double alpha,
                             const double beta,
                             Vector &op)
{
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
    std::cout << "TODO: Correct this for DG diffusion" << std::endl;
    exit(1);
    /*
    const int VDIM = 2;

    auto d = Reshape(det.Read(), Q1D, NF);
    auto n = Reshape(nor.Read(), Q1D, VDIM, NF);
    const bool const_r = rho.Size() == 1;
    auto R =
        const_r ? Reshape(rho.Read(), 1,1) : Reshape(rho.Read(), Q1D,NF);
    const bool const_v = vel.Size() == 2;
    auto V =
        const_v ? Reshape(vel.Read(), 2,1,1) : Reshape(vel.Read(), 2,Q1D,NF);
    auto W = w.Read();
    auto qd = Reshape(op.Write(), Q1D, 2, 2, NF);

    MFEM_FORALL(f, NF, // can be optimized with Q1D thread for NF blocks
    {
        for (int q = 0; q < Q1D; ++q)
        {
            const double r = const_r ? R(0,0) : R(q,f);
            const double v0 = const_v ? V(0,0,0) : V(0,q,f);
            const double v1 = const_v ? V(1,0,0) : V(1,q,f);
            const double dot = n(q,0,f) * v0 + n(q,1,f) * v1;
            const double abs = dot > 0.0 ? dot : -dot;
            const double w = W[q]*r*d(q,f);
            qd(q,0,0,f) = w*( alpha/2 * dot + beta * abs );
            qd(q,1,0,f) = w*( alpha/2 * dot - beta * abs );
            qd(q,0,1,f) = w*(-alpha/2 * dot - beta * abs );
            qd(q,1,1,f) = w*(-alpha/2 * dot + beta * abs );
        }
    });
    */
}

static void PADGDiffusionSetup3D(const int Q1D,
                             const int NF,
                             const Array<double> &w,
                             const Vector &det,
                             const Vector &nor,
                             const Vector &rho,
                             const Vector &vel,
                             const double alpha,
                             const double beta,
                             Vector &op)
{
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
    std::cout << "TODO: Correct this for DG diffusion" << std::endl;
    exit(1);
    /*
    const int VDIM = 3;

    auto d = Reshape(det.Read(), Q1D, Q1D, NF);
    auto n = Reshape(nor.Read(), Q1D, Q1D, VDIM, NF);
    const bool const_r = rho.Size() == 1;
    auto R =
        const_r ? Reshape(rho.Read(), 1,1,1) : Reshape(rho.Read(), Q1D,Q1D,NF);
    const bool const_v = vel.Size() == 3;
    auto V =
        const_v ? Reshape(vel.Read(), 3,1,1,1) : Reshape(vel.Read(), 3,Q1D,Q1D,NF);
    auto W = w.Read();
    auto qd = Reshape(op.Write(), Q1D, Q1D, 2, 2, NF);

    MFEM_FORALL(f, NF, // can be optimized with Q1D*Q1D threads for NF blocks
    {
        for (int q1 = 0; q1 < Q1D; ++q1)
        {
            for (int q2 = 0; q2 < Q1D; ++q2)
            {
            const double r = const_r ? R(0,0,0) : R(q1,q2,f);
            const double v0 = const_v ? V(0,0,0,0) : V(0,q1,q2,f);
            const double v1 = const_v ? V(1,0,0,0) : V(1,q1,q2,f);
            const double v2 = const_v ? V(2,0,0,0) : V(2,q1,q2,f);
            const double dot = n(q1,q2,0,f) * v0 + n(q1,q2,1,f) * v1 + n(q1,q2,2,f) * v2;
            const double abs = dot > 0.0 ? dot : -dot;
            const double w = W[q1+q2*Q1D]*r*d(q1,q2,f);
            qd(q1,q2,0,0,f) = w*( alpha/2 * dot + beta * abs );
            qd(q1,q2,1,0,f) = w*( alpha/2 * dot - beta * abs );
            qd(q1,q2,0,1,f) = w*(-alpha/2 * dot - beta * abs );
            qd(q1,q2,1,1,f) = w*(-alpha/2 * dot + beta * abs );
            }
        }
    });
    */

    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
}

static void PADGDiffusionSetup(const int dim,
                           const int D1D,
                           const int Q1D,
                           const int NF,
                           const Array<double> &W,
                           const Vector &det,
                           const Vector &nor,
                           const Vector &rho,
                           const Vector &u,
                           const double alpha,
                           const double beta,
                           Vector &op)
{
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
    switch (dim)
    {
        case 1:
            MFEM_ABORT("dim==1 not supported in PADGDiffusionSetup");
        break;
        case 2:
            PADGDiffusionSetup2D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
        break;
        case 3:
            PADGDiffusionSetup2D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
        break;
        default:
            std::cout << " dim = " << dim << std::endl;
            MFEM_ABORT("Invalid choice of dim in PADGDiffusionSetup");
        break;
    }
    std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
}

}
