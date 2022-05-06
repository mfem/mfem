// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "kdtree.hpp"

mfem::KDTree3D* mfem::BuildKDTree3D(mfem::Mesh* mesh)
{
    mfem::KDTree<int,double,3>* kdt=new mfem::KDTree<int,double,3>();

    mfem::GridFunction* nodes=mesh->GetNodes();
    int dim=mesh->SpaceDimension();
    MFEM_ASSERT(dim==3,"mfem::KDTree3D* mfem::BuildKDTree requires "
                       "the spatial dimension of the supplied mesh to be 3!");

    if(nodes==nullptr)//use vertices
    {
        for(int i=0;i<mesh->GetNV();i++)
        {
            double* v=mesh->GetVertex(i);
            kdt->AddNode(v[0],v[1],v[2],i);
        }
    }else{//use the nodes
        for(int i=0;i<nodes->Size();i=i+dim)
        {
            kdt->AddNode((*nodes)(i),(*nodes)(i+1),(*nodes)(i+2),i/dim);
        }
    }
    kdt->Sort();
    return kdt;
}

mfem::KDTree2D* mfem::BuildKDTree2D(mfem::Mesh* mesh)
{
    mfem::KDTree2D* kdt=new mfem::KDTree2D();
    mfem::GridFunction* nodes=mesh->GetNodes();
    int dim=mesh->SpaceDimension();
    MFEM_ASSERT(dim>1,"mfem::KDTree3D* mfem::BuildKDTree requires the spatial "
                      "dimension of the supplied mesh to be larger than 1!");

    if(nodes==nullptr)//use vertices
    {
        for(int i=0;i<mesh->GetNV();i++)
        {
            double* v=mesh->GetVertex(i);
            kdt->AddNode(v[0],v[1],i);
        }
    }else{//use the nodes
        for(int i=0;i<nodes->Size();i=i+dim)
        {
            kdt->AddNode((*nodes)(i),(*nodes)(i+1),i/dim);
        }
    }
    kdt->Sort();
    return kdt;
}

mfem::KDTree1D* mfem::BuildKDTree1D(mfem::Mesh* mesh)
{
    mfem::KDTree1D* kdt=new mfem::KDTree1D();
    mfem::GridFunction* nodes=mesh->GetNodes();
    int dim=mesh->SpaceDimension();
    MFEM_ASSERT(dim>0,"mfem::KDTree3D* mfem::BuildKDTree requires the spatial "
                      "dimension of the supplied mesh to be larger than 0!");

    if(nodes==nullptr)//use vertices
    {
        for(int i=0;i<mesh->GetNV();i++)
        {
            double* v=mesh->GetVertex(i);
            kdt->AddNode(v[0],i);
        }
    }else{//use the nodes
        for(int i=0;i<nodes->Size();i=i+dim)
        {
            kdt->AddNode((*nodes)(i),i/dim);
        }
    }
    kdt->Sort();
    return kdt;
}
