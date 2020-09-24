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

#include "mfem.hpp"
#include "catch.hpp"

using namespace mfem;

namespace blocknonlinearform_3d
{


double rf0(const mfem::Vector &coords)
{
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);

   double p=std::sqrt(x*x+y*y+z*z);
   double rez=0.0;
   if (p<1.0) { rez=1.0; }
   return rez;
}


double uf0(const mfem::Vector &coords)
{
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);

   double p=std::sqrt(x*x+y*y+z*z);
   double rez=0.0;
   if (p<1.5) { rez=1; }
   return rez;
}

class CExample: public BlockNonlinearFormIntegrator
{
private:
public:
   CExample() {}
   virtual ~CExample() {}

   virtual double GetElementEnergy(const mfem::Array<const FiniteElement *> &el,
                                   mfem::ElementTransformation &trans,
                                   const mfem::Array<const Vector *> &elfun)
   {
      double energy=0;
      int dof_u = el[0]->GetDof();
      int dof_r = el[1]->GetDof();
      int dim = el[0]->GetDim();

      const mfem::IntegrationRule *ir = NULL;
      int order = el[0]->GetOrder() + el[1]->GetOrder() + 1;
      ir = &mfem::IntRules.Get(el[0]->GetGeomType(), order);
      mfem::Vector shaperr(dof_r); //densities
      mfem::Vector shaperu(dof_u); //prime field

      double w;
      double c1,c2;
      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         w = ip.weight *w;
         el[0]->CalcPhysShape(trans,shaperu);
         el[1]->CalcPhysShape(trans,shaperr);

         c1=shaperr*(*elfun[1]);
         c2=shaperu*(*elfun[0]);

         energy=energy+w*c1*c2;
      }

      return energy;
   }

};

#ifdef MFEM_USE_MPI

//"[GetEnergy method]"
//"[Mixed Scalar Integrator]"


TEST_CASE("3D ParBlockNonlinearForm",
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   FunctionCoefficient u0_coef(uf0);
   FunctionCoefficient r0_coef(rf0);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      //generate mesh [0,1]x[0,1]x[0,1]
      int n=20;
      Mesh *mesh=new Mesh(n, n, n, (Element::Type)type, 0, 2.0, 2.0, 2.0);
      int dim = mesh->Dimension();
      mfem::ParMesh *pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
      delete mesh;
      pmesh->UniformRefinement();
      int uorder=3;
      int rorder=2;
      H1_FECollection ufec(uorder, dim);
      H1_FECollection rfec(rorder, dim);

      mfem::ParFiniteElementSpace ufes(pmesh, &ufec);
      mfem::ParFiniteElementSpace rfes(pmesh, &rfec);

      int udofs = ufes.GlobalTrueVSize();
      int rdofs = rfes.GlobalTrueVSize();

      mfem::Array<mfem::ParFiniteElementSpace *> fes(2);
      fes[0] = &ufes;
      fes[1] = &rfes;

      mfem::Array<int> block_trueOffsets(3);
      block_trueOffsets[0] = 0;
      block_trueOffsets[1] = ufes.TrueVSize();
      block_trueOffsets[2] = rfes.TrueVSize();
      block_trueOffsets.PartialSum();

      mfem::ParGridFunction u_gf(&ufes);
      mfem::ParGridFunction r_gf(&rfes);

      mfem::FunctionCoefficient u0_coeff(uf0);
      mfem::FunctionCoefficient r0_coeff(rf0);

      u_gf.ProjectCoefficient(u0_coeff);
      r_gf.ProjectCoefficient(r0_coeff);


      BlockVector x(block_trueOffsets);

      u_gf.GetTrueDofs(x.GetBlock(0));
      r_gf.GetTrueDofs(x.GetBlock(1));

      ParBlockNonlinearForm* nf=new ParBlockNonlinearForm(fes);
      nf->AddDomainIntegrator(new CExample());

      //compute the energy
      //integral over 1/8 sphere
      //i.e. Pi*1*1*1/6
      double A4=nf->GetEnergy(x);
      if (my_rank==0)
      {
         std::cout << my_rank << ": Checked ParBlockNonlinearForm::GetEnergy = "
                   << A4 << " Expected" << M_PI/6.0 <<" diff="<<  (A4-M_PI/6.0) <<std::endl;
      }
      REQUIRE((A4-M_PI/6.0)<1e-3);

      delete nf;
      delete pmesh;
   }
}

#endif

}
