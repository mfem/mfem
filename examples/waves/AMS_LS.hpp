#include "mfem.hpp"

using namespace mfem;
using namespace std;


// Discrete gradient matrix
HypreParMatrix* GetDiscreteGradientOp(ParFiniteElementSpace *fespace)
{
   int dim = fespace->GetMesh()->Dimension();
   // int sdim = fespace->GetMesh()->SpaceDimension();
   // const FiniteElementCollection *fec = fespace->FEColl();
   int p = 1;
   if (fespace->GetNE() > 0)
   {
      p = fespace->GetOrder(0);
   }
   ParMesh *pmesh = fespace->GetParMesh();
   FiniteElementCollection *vert_fec;
   vert_fec = new H1_FECollection(p, dim);
   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh,vert_fec);
   // generate and set the discrete gradient
   ParDiscreteLinearOperator *grad;
   grad = new ParDiscreteLinearOperator(vert_fespace, fespace);
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   HypreParMatrix *G;
   G = grad->ParallelAssemble();
   delete vert_fespace;
   delete grad;
   return G;
}



// Discrete gradient matrix
Array2D<HypreParMatrix *> GetNDInterpolationOp(ParFiniteElementSpace *fespace)
{
   int dim = fespace->GetMesh()->Dimension();
   int sdim = fespace->GetMesh()->SpaceDimension();
   // const FiniteElementCollection *fec = fespace->FEColl();
   int p = 1;
   if (fespace->GetNE() > 0)
   {
      p = fespace->GetOrder(0);
   }
   ParMesh *pmesh = fespace->GetParMesh();
   FiniteElementCollection *vert_fec;
   vert_fec = new H1_FECollection(p, dim);
   
   Array2D<HypreParMatrix *> Pi_blocks;
   ParFiniteElementSpace *vert_fespace_d
         = new ParFiniteElementSpace(pmesh, vert_fec, sdim, Ordering::byVDIM);
   ParDiscreteLinearOperator *id_ND;
   id_ND = new ParDiscreteLinearOperator(vert_fespace_d, fespace);
   id_ND->AddDomainInterpolator(new IdentityInterpolator);
   id_ND->Assemble();
   id_ND->Finalize();
   id_ND->GetParBlocks(Pi_blocks);
   //
   delete id_ND;
   delete vert_fespace_d;
   delete vert_fec;
   return Pi_blocks;
}