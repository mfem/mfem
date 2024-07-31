// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MARKING_HPP
#define MFEM_MARKING_HPP

#include "mfem.hpp"



namespace mfem{

/// Marking operations for elements, faces, dofs, etc, related to CutFEM.
class ElementMarker{
public:
    enum ElementType {INSIDE = 0, OUTSIDE = 1, CUT = 2};

    enum FaceType {UNDEFINED = 0, SURROGATE = 1, GHOSTP = 2};

    ///Defines element marker class with options to include the cut elements
    /// (include_cut=true) or to mark the cut elements as SBElementType::CUT.
    /// If use_cut=false the marking will use only INSIDE/OUTSIDE marks.
    /// The last integer argument determines the order of the surrogate H1 field
    /// for checking if an element is cut by a zero level set of an implicit
    /// material distribution.
    ElementMarker(Mesh& mesh, bool include_cut=false,
                  bool use_cut=false, int h1_order_=2)
    {
        const int dim=mesh.SpaceDimension();
        elfec=new L2_FECollection(0,dim);
        smesh=&mesh;

        elfes=new FiniteElementSpace(smesh,elfec,1);
        elgf.SetSpace(elfes);

        include_cut_elements=include_cut;
        use_cut_marks=use_cut;
        h1_order=h1_order_;
    }

    /// Destructor of the ElementMarker class
    ~ElementMarker()
    {
        delete elfes;
        delete elfec;
    }

    /// Mark elements according to the specified level-set
    /// function.
    void SetLevelSetFunction(const GridFunction& ls_fun);

    /// Mark the elements according to the specified coefficient.
    void SetLevelSetFunction(Coefficient& ls_fun);

    /// Returns the marking of all the elements
    /// in the mesh using the @a ElementType
    void MarkElements(Array<int> &elem_marker);

    /// Returns the marking of all faces in the
    /// mesh using  the @a FaceType
    void MarkFaces(Array<int> &face_marker);

    /// Returns the marking of all faces in the
    /// mesh using  the @a FaceType.
    /// The marks of all cut and faces between
    /// cut and inside elements are set to GHOSTP
    void MarkGhostPenaltyFaces(Array<int> &face_marker);

    /// Lists all inactive dofs, i.e.,
    ///  all dofs in the outside region.
    void ListEssentialTDofs(const Array<int> &elem_marker,
                            FiniteElementSpace &lfes,
                            Array<int> &ess_tdof_list) const;

protected:
    FiniteElementCollection* elfec;
    Mesh* smesh;

    FiniteElementSpace* elfes;
    GridFunction elgf;

    bool include_cut_elements;
    bool use_cut_marks;

    int h1_order; //order of the H1 FE space for level set functions defined by coefficient
};

#ifdef MFEM_USE_MPI

/// Marking operations for elements, faces, dofs, etc,
/// related to parallel implementations of CutFEM.
class ParElementMarker
{
public:
    ///Defines parallel element marker class with options to include the cut
    ///  elements (include_cut=true) or to mark the cut elements as
    /// ElementType::CUT. If use_cut=false the marking will use only
    /// INSIDE/OUTSIDE marks. The last integer argument determines the
    /// order of the surrogate H1 field for checking if an element is
    /// cut by a zero level set of an implicit material distribution.
    ParElementMarker(ParMesh& mesh,bool include_cut=false,
                  bool use_cut=false, int h1_order_=2)
    {
        pmesh=&mesh;
        const int dim=pmesh->SpaceDimension();
        elfec=new L2_FECollection(0,dim);
        elfes=new ParFiniteElementSpace(pmesh,elfec,1);
        elgf.SetSpace(elfes);

        include_cut_elements=include_cut;
        use_cut_marks=use_cut;

        h1_order=h1_order_;
    }

    /// Destructor of the ElementMarker class
    ~ParElementMarker()
    {
        delete elfes;
        delete elfec;
    }

    /// Mark elements according to the specified level-set
    /// function.
    void SetLevelSetFunction(const ParGridFunction& ls_fun);

    /// Mark the elements according to the specified coefficient.
    void SetLevelSetFunction(Coefficient& ls_fun);

    /// Returns the marking of all the elements
    /// in the mesh using the @a ElementMarker::ElementType
    void MarkElements(Array<int> &elem_marker);

    /// Returns the marking of all faces in the
    /// mesh using  the @a ElementMarker::FaceType
    void MarkFaces(Array<int> &face_marker);

    /// Returns the marking of all faces in the
    /// mesh using  the @a ElementMarker::FaceType.
    /// The marks of all cut and faces between
    /// cut and inside elements are set to GHOSTP
    void MarkGhostPenaltyFaces(Array<int> &face_marker);

    /// Lists all inactive dofs, i.e.,
    ///  all dofs in the outside region.
    void ListEssentialTDofs(const Array<int> &elem_marker,
                            ParFiniteElementSpace &lfes,
                            Array<int> &ess_tdof_list) const;

private:
    ParMesh* pmesh;

    FiniteElementCollection* elfec;
    ParFiniteElementSpace* elfes;
    ParGridFunction elgf;

    bool include_cut_elements;
    bool use_cut_marks;

    // order of the H1 FE space for level set functions
    // defined by coefficient
    int h1_order;
};



#endif




}




#endif
