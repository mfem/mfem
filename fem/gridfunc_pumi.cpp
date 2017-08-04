// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of GridFunction

#include "gridfunc_pumi.hpp"
//#include "../mesh/nurbs.hpp"

#ifdef MFEM_USE_SCOREC
#ifdef MFEM_USE_MPI

#include <limits>
#include <cstring>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace mfem
{

using namespace std;


GridFunctionPumi::GridFunctionPumi(Mesh* m, apf::Mesh2* PumiM, apf::Numbering* v_num_loc
, const int mesh_order)
  //: Vector()
{
    //set to zero 
    SetDataAndSize(NULL, 0);
    int ec;
    int spDim = m->SpaceDimension();
    //needs to be modified for other orders
    if (mesh_order == 1){
        mfem_error("GridFunction::GridFunction : First order mesh!");
    }
    else if (mesh_order == 2){
        fec =  FiniteElementCollection::New("Quadratic");
    }
    else 
    {
        fec = new H1_FECollection(mesh_order, m->Dimension());
    }
    int ordering = 1; // x1y1z1/x2y2z2/...
    fes = new FiniteElementSpace(m, fec, spDim, ordering);
    int data_size = fes->GetVSize();

    //Read Pumi mesh data
    this->SetSize(data_size);
    double* PumiData = this->GetData();

    apf::MeshEntity* ent;
    apf::MeshIterator* itr;

    
    //Assume all element type are the same i.e. tetrahedral
    const FiniteElement* H1_elem = fes->GetFE(1);
    const IntegrationRule &All_nodes = H1_elem->GetNodes();
    int num_vert = m->GetElement(1)->GetNVertices();
    int nnodes = All_nodes.Size();

    //loop over elements
    apf::Field* crd_field = PumiM->getCoordinateField();
    
    int nc = apf::countComponents(crd_field);
    int iel = 0;
    itr = PumiM->begin(m->Dimension());
    while ((ent = PumiM->iterate(itr)))
    {
        Array<int> vdofs;
        fes->GetElementVDofs(iel, vdofs);
        
        //create Pumi element to interpolate
        apf::MeshElement* mE = apf::createMeshElement(PumiM, ent);
        apf::Element* elem = apf::createElement(crd_field, mE);

        //Vertices are already interpolated
        for (int ip = 0; ip < nnodes; ip++)//num_vert
        {
            //Take parametric coordinates of the node
            apf::Vector3 param;
            param[0] = All_nodes.IntPoint(ip).x;
            param[1] = All_nodes.IntPoint(ip).y;
            param[2] = All_nodes.IntPoint(ip).z;
            
            
            //Compute the interpolating coordinates
            apf::DynamicVector phCrd(nc);
            apf::getComponents(elem, param, &phCrd[0]);
            
            //Fill the nodes list
            for (int kk = 0; kk < spDim; ++kk){
              int dof_ctr = ip + kk * nnodes;
              PumiData[vdofs[dof_ctr]] = phCrd[kk];
            }
            
        }
        iel++;
        apf::destroyElement(elem);
        apf::destroyMeshElement(mE);
    }
    PumiM->end(itr);
    
    sequence = 0;      
}

}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SCOREC
