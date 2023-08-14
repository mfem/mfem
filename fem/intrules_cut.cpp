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

// Implementation of Surface and Cutcell IntegrationRule(s) classes

#include "fem.hpp"
#include <cmath>

using namespace std;

namespace mfem
{

SIntegrationRule::SIntegrationRule(int q, ElementTransformation& Tr,
                                   Coefficient &levelset)
    : Array<IntegrationPoint>(), Order(q), LvlSet(levelset)
{
    nBasis = 2 * (Order + 1) + (int)(Order * (Order + 1) / 2);
    Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);

    int qorder = 0;    
    IntegrationRules irs(0, Quadrature1D::GaussLegendre);
	IntegrationRule ir = irs.Get(Trafo.GetGeometryType(), qorder);
    for (; ir.GetNPoints() <= nBasis; qorder++)
        ir = irs.Get(Trafo.GetGeometryType(), qorder);

    SetSize(ir.GetNPoints());
    for (int ip = 0; ip < Size(); ip++)
    {
        IntPoint(ip).index = ip;
        IntegrationPoint &intp = IntPoint(ip);
        intp.x = ir.IntPoint(ip).x;
        intp.y = ir.IntPoint(ip).y;
        intp.weight = 0.;
    }
}

void SIntegrationRule::ComputeWeights()
{
    Element* me = Trafo.mesh->GetElement(Trafo.ElementNo);

    DenseMatrix Mat(nBasis, GetNPoints());
    Mat = 0.;
    Vector RHS(nBasis);
    RHS = 0.;
    Vector Weights(GetNPoints());
    Weights = 0.;

    bool element_int = false;
    bool interior = true;
    Array<bool> edge_int;

    DenseMatrix PointA(me->GetNEdges(), 2);
    DenseMatrix PointB(me->GetNEdges(), 2);

    Array<int> verts;
    Trafo.mesh->GetElementVertices(Trafo.ElementNo, verts);

    for(int edge = 0; edge < me->GetNEdges(); edge++)
    {
        enum class Layout {inside, intersected, outside};
        Layout layout;
        
        const int* vert = me->GetEdgeVertices(edge);
        Vector pointA(2);
        Vector pointB(2);
        pointA(0) = (Trafo.mesh->GetVertex(verts[vert[0]]))[0];
        pointA(1) = (Trafo.mesh->GetVertex(verts[vert[0]]))[1];
        pointB(0) = (Trafo.mesh->GetVertex(verts[vert[1]]))[0];
        pointB(1) = (Trafo.mesh->GetVertex(verts[vert[1]]))[1];

        IntegrationPoint ipA;
        Trafo.TransformBack(pointA, ipA);
        IntegrationPoint ipB;
        Trafo.TransformBack(pointB, ipB);
            
        if(LvlSet.Eval(Trafo, ipA) < -1e-12
                || LvlSet.Eval(Trafo, ipB) < -1e-12)
            interior = false;

        if(LvlSet.Eval(Trafo, ipA) > -1e-12 
                && LvlSet.Eval(Trafo, ipB) > -1e-12)
            layout = Layout::inside;
        else if(LvlSet.Eval(Trafo, ipA) > 1e-15
                && LvlSet.Eval(Trafo, ipB) <= 0.)
            layout = Layout::intersected;
        else if(LvlSet.Eval(Trafo, ipA) <= 0.
                && LvlSet.Eval(Trafo, ipB) > 1e-15)
        {
            layout = Layout::intersected;
            Vector temp(pointA.Size());
            temp = pointA;
            pointA = pointB;
            pointB = temp;
        }
        else
            layout = Layout::outside;

        if(layout == Layout::intersected)
        {
            Vector pointC(pointA.Size());
            Vector mid(pointA.Size());
            pointC = pointA;
            mid = pointC;
            mid += pointB;
            mid /= 2.;

            IntegrationPoint ip;
            Trafo.TransformBack(mid, ip);

            while (LvlSet.Eval(Trafo, ip) > 1e-12 
                    || LvlSet.Eval(Trafo, ip) < -1e-12)
            {
                if(LvlSet.Eval(Trafo, ip) > 1e-12)
                    pointC = mid;
                else
                    pointB = mid;

                mid = pointC;
                mid += pointB;
                mid /= 2.;
                Trafo.TransformBack(mid, ip);
            }
            pointB = mid;
        }

        PointA.SetRow(edge, pointA);
        PointB.SetRow(edge, pointB);

        if((layout == Layout::inside || layout == Layout::intersected))
            edge_int.Append(true);   
        else
            edge_int.Append(false);
    }
if(Trafo.ElementNo == 105)
{PointA.Print();PointB.Print();}
    for(int edge = 0; edge < me->GetNEdges(); edge++)
    {
        if(edge_int[edge] && !interior)
        {
            Vector point0(2);
            Vector point1(2);
            PointA.GetRow(edge, point0);
            PointB.GetRow(edge, point1);

            element_int = true;

            const IntegrationRule *ir2 = &IntRules.Get(Geometry::SEGMENT,
                                                       2*Order+1);

            Vector normal(2);
            Array<int> edges;
            Array<int> cor;
            Trafo.mesh->GetElementEdges(Trafo.ElementNo, edges, cor);
            FaceElementTransformations* FTrans = 
                        Trafo.mesh->GetFaceElementTransformations(edges[edge]);
            FTrans->SetIntPoint(&(ir2->IntPoint(0)));  
            CalcOrtho(FTrans->Jacobian(), normal);
            if((Trafo.ElementNo == FTrans->Elem1No 
                        && Trafo.ElementNo > FTrans->Elem2No
                        && FTrans->Elem2No != -1)
                    || (Trafo.ElementNo == FTrans->Elem2No
                        && Trafo.ElementNo > FTrans->Elem1No
                        && FTrans->Elem1No != -1))
                normal *= -1.;
            normal /= normal.Norml2();

            for(int ip = 0; ip < ir2->GetNPoints(); ip++)
            {   
                Vector dist(2);
                dist = point1;
                dist -= point0;

                Vector point(2);
                point = dist;
                point *= ir2->IntPoint(ip).x;
                point += point0;

                IntegrationPoint intpoint;
                Trafo.TransformBack(point, intpoint);
                DenseMatrix shapes;
                OrthoBasis(intpoint, shapes);
                Vector grad(2);

                for(int dof = 0; dof < nBasis; dof++)
                {   
                    shapes.GetRow(dof, grad); 
                    RHS(dof) -= (grad * normal) * ir2->IntPoint(ip).weight
                               * dist.Norml2();
                }
            }
        }
    }

    if(element_int && !interior)
    {
        int elem = Trafo.ElementNo;
        H1_FECollection fec(9, 2);
        FiniteElementSpace fes(Trafo.mesh, &fec);
        GridFunction LevelSet(&fes);
        LevelSet.ProjectCoefficient(LvlSet);
        Trafo.mesh->GetElementTransformation(elem, &Trafo);

        for(int ip = 0; ip < GetNPoints(); ip++)
        {
            Vector normal(2);
            Vector physpoint(2);
            Trafo.SetIntPoint(&(IntPoint(ip)));
            Trafo.Transform(IntPoint(ip), physpoint);
            LevelSet.GetGradient(Trafo, normal);
            normal *= (-1. / normal.Norml2());

            DenseMatrix shapes;
            OrthoBasis(IntPoint(ip), shapes);

            for(int dof = 0; dof < nBasis; dof++)
            {
                Vector grad(2);
                shapes.GetRow(dof, grad);
                Mat(dof, ip) =  (grad * normal);
            }
        }

#ifdef MFEM_USE_LAPACK
        Vector temp(nBasis);
        Vector temp2(GetNPoints());
        DenseMatrixSVD SVD(Mat, 'A', 'A');
        SVD.Eval(Mat);
        SVD.LeftSingularvectors().MultTranspose(RHS, temp);
        temp2 = 0.;
        for(int i = 0; i < nBasis; i++)
            if(SVD.Singularvalue(i) > 1e-12)
                temp2(i) = temp(i) / SVD.Singularvalue(i);
        SVD.RightSingularvectors().MultTranspose(temp2, Weights);

#endif
    }

    for(int ip = 0; ip < GetNPoints(); ip++)
    {

        IntegrationPoint &intp = IntPoint(ip);
        intp.weight = Weights(ip) / Trafo.Weight();
    }
}

void SIntegrationRule::Basis(const IntegrationPoint& ip, DenseMatrix& shape)
{
    shape.SetSize(nBasis, 2);

    Vector X(2);
    X(0) = -1. + 2. * ip.x;
    X(1) = -1. + 2. * ip.y;

    for(int c = 0; c <= Order; c++)
    {
        Vector a(2);
        a = 0.;
        a(1) = pow(X(0), (double)(c));
 
        Vector b(2);
        b = 0.;
        b(0) = pow(X(1), (double)(c));

        shape.SetRow(2 * c, a);
        shape.SetRow(2 * c + 1, b);
    }

    Poly_1D poly;
    int count = 2 * Order + 2;
    for(int c = 1; c <= Order; c++)
    {
        const int* factorial = poly.Binom(c);
        for(int expo = c; expo > 0; expo--)
        {
            Vector a(2);
            a(0) = (double)(factorial[expo]) * pow(X(0), (double)(expo))
                  *  pow(X(1), (double)(c - expo));
            a(1) = -1. * (double)(factorial[expo - 1])
                  * pow(X(0), (double)(expo - 1))
                  * pow(X(1), (double)(c - expo + 1));

            shape.SetRow(count, a);
            count++;
        }
    }
}

void SIntegrationRule::OrthoBasis(const IntegrationPoint& ip, DenseMatrix& shape)
{
    const IntegrationRule *ir = &IntRules.Get(Geometry::SQUARE, 2*Order+1);

    shape.SetSize(nBasis, 2);

    DenseMatrix preshape(nBasis, 2);
    Basis(ip, shape);
    
    DenseTensor shapeMFN(nBasis, 2, ir->GetNPoints());
    for(int p = 0; p < ir->GetNPoints(); p++)
    {
        DenseMatrix shapeN(nBasis, 2);
        Basis(ir->IntPoint(p), shapeN);
        for(int i = 0; i < nBasis; i++)
            for(int j = 0; j < 2; j++)
                shapeMFN(i, j, p) = shapeN(i, j);
    }

    for(int count = 1; count < nBasis; count++)
        mGSStep(shape, shapeMFN, count);
}

void SIntegrationRule::mGSStep(DenseMatrix& shape, DenseTensor& shapeMFN,
                               int step)
{
    const IntegrationRule *ir = &IntRules.Get(Geometry::SQUARE, 2*Order+1);

    for(int count = step; count < shape.Height(); count++)
    {
        double den = 0.;
        double num = 0.;

        for(int ip = 0; ip < ir->GetNPoints(); ip++)
        {
            Vector u(2);
            Vector v(2);

            shapeMFN(ip).GetRow(count, u);
            shapeMFN(ip).GetRow(step - 1, v);

            den += v * v * ir->IntPoint(ip).weight;
            num += u * v * ir->IntPoint(ip).weight;
        }

        double coeff = num / den;

        Vector s(2);
        Vector t(2);
        shape.GetRow(step - 1, s);
        shape.GetRow(count, t);
        s *= coeff;
        t += s;
        shape.SetRow(count, t);

        for(int ip = 0; ip < ir->GetNPoints(); ip++)
        {
            shapeMFN(ip).GetRow(step - 1, s);
            shapeMFN(ip).GetRow(count, t);
            s *= coeff;
            t += s;
            shapeMFN(ip).SetRow(count, t);
        }
    }
}

void SIntegrationRule::Update(IsoparametricTransformation& Tr)
{
    ElementNo = Tr.ElementNo;
    Trafo = Tr;
    ComputeWeights();
}

////////////////////////////////////////////////////////////////////////////////

CutIntegrationRule::CutIntegrationRule(int q, ElementTransformation& Tr,
                                       Coefficient &levelset)
    : Array<IntegrationPoint>(), Order(q), LvlSet(levelset),
    SIR(NULL), SVD(NULL)
{
    nBasis = (int)((q + 1) * (q + 2) / 2);
    Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);

    SIR = new SIntegrationRule(q, Tr, levelset);

    int qorder = 0;    
    IntegrationRules irs(0, Quadrature1D::GaussLegendre);
	IntegrationRule ir = irs.Get(Trafo.GetGeometryType(), qorder);
    for (; ir.GetNPoints() < SIR->GetNPoints(); qorder++)
        ir = irs.Get(Trafo.GetGeometryType(), qorder);

    SetSize(ir.GetNPoints());
    InteriorWeights.SetSize(ir.GetNPoints());
    for (int ip = 0; ip < Size(); ip++)
    {
        IntPoint(ip).index = ip;
        IntegrationPoint &intp = IntPoint(ip);
        intp.x = ir.IntPoint(ip).x;
        intp.y = ir.IntPoint(ip).y;
        intp.weight = 0.;
        InteriorWeights(ip) = ir.IntPoint(ip).weight;
    }

    DenseMatrix Mat(nBasis, Size());
    for(int ip = 0; ip < ir.GetNPoints(); ip++)
    {
        Vector shape;
        Basis(ir.IntPoint(ip), shape);
        Mat.SetCol(ip, shape);
    }

    SVD = new DenseMatrixSVD(Mat, 'A', 'A');
    SVD->Eval(Mat);
}

void CutIntegrationRule::ComputeWeights()
{
    Element* me = Trafo.mesh->GetElement(Trafo.ElementNo);

    Vector RHS(nBasis);
    RHS = 0.;
    Vector Weights(GetNPoints());
    Weights = 0.;

    bool element_int = false;
    bool interior = true;
    Array<bool> edge_int;

    DenseMatrix PointA(me->GetNEdges(), 2);
    DenseMatrix PointB(me->GetNEdges(), 2);

    Array<int> verts;
    Trafo.mesh->GetElementVertices(Trafo.ElementNo, verts);

    for(int edge = 0; edge < me->GetNEdges(); edge++)
    {
        enum class Layout {inside, intersected, outside};
        Layout layout;
        
        const int* vert = me->GetEdgeVertices(edge);
        Vector pointA(2);
        Vector pointB(2);
        pointA(0) = (Trafo.mesh->GetVertex(verts[vert[0]]))[0];
        pointA(1) = (Trafo.mesh->GetVertex(verts[vert[0]]))[1];
        pointB(0) = (Trafo.mesh->GetVertex(verts[vert[1]]))[0];
        pointB(1) = (Trafo.mesh->GetVertex(verts[vert[1]]))[1];

        IntegrationPoint ipA;
        Trafo.TransformBack(pointA, ipA);
        IntegrationPoint ipB;
        Trafo.TransformBack(pointB, ipB);
            
        if(LvlSet.Eval(Trafo, ipA) < -1e-12
                || LvlSet.Eval(Trafo, ipB) < -1e-12)
            interior = false;

        if(LvlSet.Eval(Trafo, ipA) > -1e-12 
                && LvlSet.Eval(Trafo, ipB) > -1e-12)
            layout = Layout::inside;
        else if(LvlSet.Eval(Trafo, ipA) > 0
                && LvlSet.Eval(Trafo, ipB) <= 0.)
            layout = Layout::intersected;
        else if(LvlSet.Eval(Trafo, ipA) <= 0.
                && LvlSet.Eval(Trafo, ipB) > 0.)
        {
            layout = Layout::intersected;
            Vector temp(pointA.Size());
            temp = pointA;
            pointA = pointB;
            pointB = temp;
        }
        else
            layout = Layout::outside;

        if(layout == Layout::intersected)
        {
            Vector pointC(pointA.Size());
            Vector mid(pointA.Size());
            pointC = pointA;
            mid = pointC;
            mid += pointB;
            mid /= 2.;

            IntegrationPoint ip;
            Trafo.TransformBack(mid, ip);

            while (LvlSet.Eval(Trafo, ip) > 1e-12 
                    || LvlSet.Eval(Trafo, ip) < -1e-12)
            {
                if(LvlSet.Eval(Trafo, ip) > 1e-12)
                    pointC = mid;
                else
                    pointB = mid;

                mid = pointC;
                mid += pointB;
                mid /= 2.;
                Trafo.TransformBack(mid, ip);
            }
            pointB = mid;
        }

        PointA.SetRow(edge, pointA);
        PointB.SetRow(edge, pointB);

        if((layout == Layout::inside || layout == Layout::intersected))
            edge_int.Append(true);   
        else
            edge_int.Append(false);
    }

    for(int edge = 0; edge < me->GetNEdges(); edge++)
    {
        if(edge_int[edge] && !interior)
        {
            Vector point0(2);
            Vector point1(2);
            PointA.GetRow(edge, point0);
            PointB.GetRow(edge, point1);

            element_int = true;

            const IntegrationRule *ir2 = &IntRules.Get(Geometry::SEGMENT,
                                                       2*Order+1);

            Vector normal(2);
            Array<int> edges;
            Array<int> cor;
            Trafo.mesh->GetElementEdges(Trafo.ElementNo, edges, cor);
            FaceElementTransformations* FTrans = 
                        Trafo.mesh->GetFaceElementTransformations(edges[edge]);
            FTrans->SetIntPoint(&(ir2->IntPoint(0)));  
            CalcOrtho(FTrans->Jacobian(), normal);
            if((Trafo.ElementNo == FTrans->Elem1No 
                        && Trafo.ElementNo > FTrans->Elem2No
                        && FTrans->Elem2No != -1)
                    || (Trafo.ElementNo == FTrans->Elem2No
                        && Trafo.ElementNo > FTrans->Elem1No
                        && FTrans->Elem1No != -1))
                normal *= -1.;
            normal /= normal.Norml2();

            for(int ip = 0; ip < ir2->GetNPoints(); ip++)
            {   
                Vector dist(2);
                dist = point1;
                dist -= point0;

                Vector point(2);
                point = dist;
                point *= ir2->IntPoint(ip).x;
                point += point0;

                IntegrationPoint intpoint;
                Trafo.TransformBack(point, intpoint);
                DenseMatrix shapes;
                BasisAntiDerivative(intpoint, shapes);
                Vector grad(2);

                for(int dof = 0; dof < nBasis; dof++)
                {   
                    shapes.GetRow(dof, grad); 
                    RHS(dof) += (grad * normal) * ir2->IntPoint(ip).weight
                               * dist.Norml2() * sqrt(Trafo.Weight() / 4.);
                }
            }
        }
    }
 
    if(element_int && !interior)
    {
        int elem = Trafo.ElementNo;
        H1_FECollection fec(5, 2);
        FiniteElementSpace fes(Trafo.mesh, &fec);
        GridFunction LevelSet(&fes);
        LevelSet.ProjectCoefficient(LvlSet);
        Trafo.mesh->GetElementTransformation(elem, &Trafo);

        for(int ip = 0; ip < GetNPoints(); ip++)
        {   
            Vector normal(2);
            Vector physpoint(2);
            Trafo.SetIntPoint(&(IntPoint(ip)));
            Trafo.Transform(IntPoint(ip), physpoint);
            LevelSet.GetGradient(Trafo, normal);
            normal *= (-1. / normal.Norml2());

            DenseMatrix shapes;
            BasisAntiDerivative(IntPoint(ip), shapes);

            for(int dof = 0; dof < nBasis; dof++)
            {
                Vector adiv(2);
                shapes.GetRow(dof, adiv);
                RHS(dof) +=  (adiv * normal) * SIR->IntPoint(ip).weight
                           * sqrt(Trafo.Weight() / 4.) * Trafo.Weight();
            }
        }
#ifdef MFEM_USE_LAPACK
        Vector temp(nBasis);
        Vector temp2(GetNPoints());
        temp2 = 0.;
        SVD->LeftSingularvectors().MultTranspose(RHS, temp);
        for(int i = 0; i < nBasis; i++)
            if(SVD->Singularvalue(i) > 1e-12)
                temp2(i) = temp(i) / SVD->Singularvalue(i);
        SVD->RightSingularvectors().MultTranspose(temp2, Weights);
#endif //MFEM_USE_LAPACK
        Weights *= 1. / Trafo.Weight();
    }    
    else if(interior)
        Weights = InteriorWeights;
    else
        Weights = 0.;

    for(int ip = 0; ip < GetNPoints(); ip++)
    {
        IntegrationPoint &intp = IntPoint(ip);
        intp.weight = Weights(ip);
    }
}

void CutIntegrationRule::Basis(const IntegrationPoint& ip, Vector& shape)
{
    shape.SetSize(nBasis);

    Vector X(2);
    X(0) = -1. + 2. * ip.x;
    X(1) = -1. + 2. * ip.y;

    int count = 0;
    for(int c = 0; c <= Order; c++)
    {
        for(int expo = 0; expo <= c; expo++)
        {
            shape(count) = pow(X(0), (double)(expo))
                          * pow(X(1), (double)(c - expo));
            count++;
        }
    }
}

void CutIntegrationRule::BasisAntiDerivative(const IntegrationPoint& ip,
                                             DenseMatrix& shape)
{
    shape.SetSize(nBasis, 2);

    Vector X(2);
    X(0) = -1. + 2. * ip.x;
    X(1) = -1. + 2. * ip.y;

    int count = 0;
    for(int c = 0; c <= Order; c++)
    {
        for(int expo = 0; expo <= c; expo++)
        {
            shape(count, 0) = .5 * pow(X(0), (double)(expo + 1)) 
                             * pow(X(1), (double)(c - expo))
                             / (double)(expo + 1);
            shape(count, 1) = .5 * pow(X(0), (double)(expo))
                             * pow(X(1), (double)(c - expo + 1))
                             / (double)(c - expo + 1);
            count++;
        }
    }
}

void CutIntegrationRule::Update(IsoparametricTransformation& Tr)
{
    ElementNo = Tr.ElementNo;
    Trafo = Tr;
    if(ElementNo != SIR->GetElement())
        SIR->Update(Tr);

    ComputeWeights();
}

CutIntegrationRule::~CutIntegrationRule() { delete SVD; delete SIR; }

}