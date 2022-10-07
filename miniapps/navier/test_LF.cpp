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

#include "mfem.hpp"
#include <fstream>

int main(int argc, char *argv[])
{
    // Initialize MPI
    int num_procs, managerRank=0, MyRank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &MyRank);

       // ------------------------
    // Mesh-related definitions
    // ------------------------

    int polinomialOrder = 1;

    // Create mesh
    double Lx = 1.0;
    double Ly = 1.0;
    int NX = 4;
    int NY = 4;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(NX, NY, mfem::Element::QUADRILATERAL, true, Lx, Ly);

    // Refine mesh in serial
    int ser_refinement_ = 0;

    for (int lev = 0; lev < ser_refinement_; lev++)
    {
        if ( MyRank == managerRank ) {std::cout << "Refining the mesh in serial" << '\n'; }
        mesh.UniformRefinement();
    }

    std::shared_ptr<mfem::ParMesh> anaPMesh = std::make_shared<mfem::ParMesh>(comm, mesh);


    if (nullptr == anaPMesh->GetNodes()) {  anaPMesh->SetCurvature(1, false, -1, 0); }


    // Create finite Element Spaces for analysis mesh
    int spatialDimension = anaPMesh->SpaceDimension();


    // Create finite Element Spaces for analysis mesh
    ::mfem::H1_FECollection anaFECol_H1(polinomialOrder, anaPMesh->Dimension());
    ::mfem::ParFiniteElementSpace anaFESpace_scalar_H1(anaPMesh.get(), &anaFECol_H1, 1, mfem::Ordering::byNODES);

    ::mfem::ParFiniteElementSpace anaFESpace_vector_H1  (anaPMesh.get(), &anaFECol_H1, spatialDimension, mfem::Ordering::byNODES);


    mfem::ParGridFunction tXi(&anaFESpace_vector_H1);
    anaFESpace_scalar_H1.GetParMesh()->GetNodes(tXi);


      std::cout<<"-0---------------------start "<<std::endl;


      mfem::ConstantCoefficient oneCoef_(1.0);

            mfem::ParGridFunction tNew_1(&anaFESpace_scalar_H1);
        tNew_1 = 0.0;
        anaFESpace_scalar_H1.GetParMesh()->GetNodes(tNew_1);
        std::cout<<"-----new coords ms_1-----"<<std::endl;
        tNew_1.Print();

      std::cout<<"-coords---------------------done "<<std::endl;

        mfem::ParLinearForm volForm2(&anaFESpace_scalar_H1);
        volForm2.AddDomainIntegrator(new mfem::DomainLFIntegrator(oneCoef_));
        volForm2.Assemble();
        volForm2.Print();
         std::cout<<"-0---------------------done "<<std::endl;


        tXi(0) += 0.1;
        anaFESpace_scalar_H1.GetParMesh()->SetNodes(tXi);

        ::mfem::ParFiniteElementSpace anaFESpace_scalar_H1_1(anaPMesh.get(), &anaFECol_H1, 1, mfem::Ordering::byNODES);

        mfem::ParGridFunction tNew(&anaFESpace_scalar_H1_1);
        tNew = 0.0;
        anaFESpace_scalar_H1_1.GetParMesh()->GetNodes(tNew);
        std::cout<<"-----new coords ms-----"<<std::endl;
        tNew.Print();

        mfem::ParLinearForm volForm(&anaFESpace_scalar_H1_1);
        volForm.AddDomainIntegrator(new mfem::DomainLFIntegrator(oneCoef_));
        volForm.Assemble();
        volForm.Print();
         std::cout<<"-3---------------------done "<<std::endl;

        mfem::ParLinearForm volForm1(&anaFESpace_scalar_H1);
        volForm1.AddDomainIntegrator(new mfem::DomainLFIntegrator(oneCoef_));
        volForm1.Assemble();
        volForm1.Print();

        std::cout<<"-------------------------------done "<<std::endl;



   return 0;
}
