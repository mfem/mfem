// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_FE_COLLECTION
#define MFEM_FE_COLLECTION

class FiniteElementCollection
{
public:
   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const = 0;

   virtual int DofForGeometry(int GeomType) const = 0;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const = 0;

   virtual const char * Name() const { return "Undefined"; };

   int HasFaceDofs (int GeomType) const;

   virtual ~FiniteElementCollection() { };

   static FiniteElementCollection *New(const char *name);
};

class LinearFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const Linear1DFiniteElement SegmentFE;
   const Linear2DFiniteElement TriangleFE;
   const BiLinear2DFiniteElement QuadrilateralFE;
   const Linear3DFiniteElement TetrahedronFE;
   const TriLinear3DFiniteElement ParallelepipedFE;
public:
   LinearFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "Linear"; };
};

class QuadraticFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const Quad1DFiniteElement SegmentFE;
   const Quad2DFiniteElement TriangleFE;
   const BiQuad2DFiniteElement QuadrilateralFE;
   const Quadratic3DFiniteElement TetrahedronFE;
   const LagrangeHexFiniteElement ParallelepipedFE;

public:
   QuadraticFECollection() : ParallelepipedFE(2) { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "Quadratic"; };
};

class QuadraticPosFECollection : public FiniteElementCollection
{
private:
   const QuadPos1DFiniteElement   SegmentFE;
   const BiQuadPos2DFiniteElement QuadrilateralFE;

public:
   QuadraticPosFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "QuadraticPos"; };
};


class CubicFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const Cubic1DFiniteElement SegmentFE;
   const Cubic2DFiniteElement TriangleFE;
   const BiCubic2DFiniteElement QuadrilateralFE;
   const Cubic3DFiniteElement TetrahedronFE;
   const LagrangeHexFiniteElement ParallelepipedFE;

public:
   CubicFECollection() : ParallelepipedFE(3) { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "Cubic"; };
};

class CrouzeixRaviartFECollection : public FiniteElementCollection
{
private:
   const P0SegmentFiniteElement SegmentFE;
   const CrouzeixRaviartFiniteElement TriangleFE;
   const CrouzeixRaviartQuadFiniteElement QuadrilateralFE;
public:
   CrouzeixRaviartFECollection() : SegmentFE(1) { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "CrouzeixRaviart"; };
};

class RT0_2DFECollection : public FiniteElementCollection
{
private:
   const P0SegmentFiniteElement SegmentFE; // normal component on edge
   const RT0TriangleFiniteElement TriangleFE;
   const RT0QuadFiniteElement QuadrilateralFE;
public:
   RT0_2DFECollection() : SegmentFE(0) { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;
};

class RT1_2DFECollection : public FiniteElementCollection
{
private:
   const P1SegmentFiniteElement SegmentFE; // normal component on edge
   const RT1TriangleFiniteElement TriangleFE;
public:
   RT1_2DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;
};

class RT2_2DFECollection : public FiniteElementCollection
{
private:
   const P2SegmentFiniteElement SegmentFE; // normal component on edge
   const RT2TriangleFiniteElement TriangleFE;
public:
   RT2_2DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;
};

class Const2DFECollection : public FiniteElementCollection
{
private:
   const P0TriangleFiniteElement TriangleFE;
   const P0QuadFiniteElement QuadrilateralFE;
public:
   Const2DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "Const2D"; };
};

class LinearDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Linear2DFiniteElement TriangleFE;
   const BiLinear2DFiniteElement QuadrilateralFE;

public:
   LinearDiscont2DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "LinearDiscont2D"; };
};

class GaussLinearDiscont2DFECollection : public FiniteElementCollection
{
private:
   // const CrouzeixRaviartFiniteElement TriangleFE;
   const GaussLinear2DFiniteElement TriangleFE;
   const GaussBiLinear2DFiniteElement QuadrilateralFE;

public:
   GaussLinearDiscont2DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "GaussLinearDiscont2D"; };
};

class P1OnQuadFECollection : public FiniteElementCollection
{
private:
   const P1OnQuadFiniteElement QuadrilateralFE;
public:
   P1OnQuadFECollection() { };
   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;
   virtual int DofForGeometry(int GeomType) const;
   virtual int * DofOrderForOrientation(int GeomType, int Or) const;
   virtual const char * Name() const { return "P1OnQuad"; };
};

class QuadraticDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Quad2DFiniteElement TriangleFE;
   const BiQuad2DFiniteElement QuadrilateralFE;

public:
   QuadraticDiscont2DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "QuadraticDiscont2D"; };
};

class QuadraticPosDiscont2DFECollection : public FiniteElementCollection
{
private:
   const BiQuadPos2DFiniteElement QuadrilateralFE;

public:
   QuadraticPosDiscont2DFECollection() { };
   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;
   virtual int DofForGeometry(int GeomType) const;
   virtual int * DofOrderForOrientation(int GeomType, int Or) const
   { return NULL; };
   virtual const char * Name() const { return "QuadraticPosDiscont2D"; };
};

class GaussQuadraticDiscont2DFECollection : public FiniteElementCollection
{
private:
   // const Quad2DFiniteElement TriangleFE;
   const GaussQuad2DFiniteElement TriangleFE;
   const GaussBiQuad2DFiniteElement QuadrilateralFE;

public:
   GaussQuadraticDiscont2DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "GaussQuadraticDiscont2D"; };
};

class CubicDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Cubic2DFiniteElement TriangleFE;
   const BiCubic2DFiniteElement QuadrilateralFE;

public:
   CubicDiscont2DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "CubicDiscont2D"; };
};

class LinearNonConf3DFECollection : public FiniteElementCollection
{
private:
   const P0TriangleFiniteElement TriangleFE;
   const P1TetNonConfFiniteElement TetrahedronFE;
   const P0QuadFiniteElement QuadrilateralFE;
   const RotTriLinearHexFiniteElement ParallelepipedFE;

public:
   LinearNonConf3DFECollection () { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "LinearNonConf3D"; };
};

class Const3DFECollection : public FiniteElementCollection
{
private:
   const P0TetFiniteElement TetrahedronFE;
   const P0HexFiniteElement ParallelepipedFE;

public:
   Const3DFECollection () { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "Const3D"; };
};

class LinearDiscont3DFECollection : public FiniteElementCollection
{
private:
   const Linear3DFiniteElement TetrahedronFE;
   const TriLinear3DFiniteElement ParallelepipedFE;

public:
   LinearDiscont3DFECollection () { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "LinearDiscont3D"; };
};

class QuadraticDiscont3DFECollection : public FiniteElementCollection
{
private:
   const Quadratic3DFiniteElement TetrahedronFE;
   const LagrangeHexFiniteElement ParallelepipedFE;

public:
   QuadraticDiscont3DFECollection () : ParallelepipedFE(2) { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "QuadraticDiscont3D"; };
};

class RefinedLinearFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const RefinedLinear1DFiniteElement SegmentFE;
   const RefinedLinear2DFiniteElement TriangleFE;
   const RefinedBiLinear2DFiniteElement QuadrilateralFE;
   const RefinedLinear3DFiniteElement TetrahedronFE;
   const RefinedTriLinear3DFiniteElement ParallelepipedFE;

public:
   RefinedLinearFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "RefinedLinear"; };
};

class ND1_3DFECollection : public FiniteElementCollection
{
private:
   const Nedelec1HexFiniteElement HexahedronFE;
   const Nedelec1TetFiniteElement TetrahedronFE;

public:
   ND1_3DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;

   virtual const char * Name() const { return "ND1_3D"; };
};

class RT0_3DFECollection : public FiniteElementCollection
{
private:
   const P0TriangleFiniteElement TriangleFE;
   const RT0TetFiniteElement TetrahedronFE;
public:
   RT0_3DFECollection() { };

   virtual const FiniteElement *
   FiniteElementForGeometry(int GeomType) const;

   virtual int DofForGeometry(int GeomType) const;

   virtual int * DofOrderForOrientation(int GeomType, int Or) const;
};


#endif
