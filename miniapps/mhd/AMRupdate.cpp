#include "AMRupdate.hpp"

void AMRUpdate(BlockVector &S, BlockVector &S_tmp,
               Array<int> &offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j)
{
   ParFiniteElementSpace* H1FESpace = phi.ParFESpace();

   //update fem space
   H1FESpace->Update();

   int fe_size = H1FESpace->GetVSize();

   //update offset vector
   offset[0] = 0;
   offset[1] = fe_size;
   offset[2] = 2*fe_size;
   offset[3] = 3*fe_size;
   offset[4] = 4*fe_size;

   S_tmp = S;
   S.Update(offset);
    
   const Operator* H1Update = H1FESpace->GetUpdateOperator();

   H1Update->Mult(S_tmp.GetBlock(0), S.GetBlock(0));
   H1Update->Mult(S_tmp.GetBlock(1), S.GetBlock(1));
   H1Update->Mult(S_tmp.GetBlock(2), S.GetBlock(2));
   H1Update->Mult(S_tmp.GetBlock(3), S.GetBlock(3));

   phi.MakeRef(H1FESpace, S, offset[0]);
   psi.MakeRef(H1FESpace, S, offset[1]);
     w.MakeRef(H1FESpace, S, offset[2]);
     j.MakeRef(H1FESpace, S, offset[3]);

   S_tmp.Update(offset);
   H1FESpace->UpdatesFinished();
}

void AMRUpdateTrue(BlockVector &S, 
               Array<int> &true_offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j,
               ParGridFunction *pre,
               ParGridFunction *dpsidt)
{
   FiniteElementSpace* H1FESpace = phi.FESpace();

   //++++Update the GridFunctions so that they match S
   phi.SetFromTrueDofs(S.GetBlock(0));
   psi.SetFromTrueDofs(S.GetBlock(1));
   w.SetFromTrueDofs(S.GetBlock(2));

   //update fem space
   H1FESpace->Update();

   // Compute new dofs on the new mesh
   phi.Update();
   psi.Update();
   w.Update();
   
   // Note j stores data as a regular gridfunction
   j.Update();
   if (pre!=NULL) pre->Update();
   if (dpsidt!=NULL) dpsidt->Update();

   int fe_size = H1FESpace->GetTrueVSize();

   //update offset vector
   true_offset[0] = 0;
   true_offset[1] = fe_size;
   true_offset[2] = 2*fe_size;
   true_offset[3] = 3*fe_size;

   // Resize S
   S.Update(true_offset);

   // Compute "true" dofs and store them in S
   phi.GetTrueDofs(S.GetBlock(0));
   psi.GetTrueDofs(S.GetBlock(1));
     w.GetTrueDofs(S.GetBlock(2));

   H1FESpace->UpdatesFinished();
}

void AMRUpdateTrue(BlockVector &S, 
               Array<int> &true_offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j)
{
   FiniteElementSpace* H1FESpace = phi.FESpace();

   //++++Update the GridFunctions so that they match S
   phi.SetFromTrueDofs(S.GetBlock(0));
   psi.SetFromTrueDofs(S.GetBlock(1));
   w.SetFromTrueDofs(S.GetBlock(2));

   //update fem space
   H1FESpace->Update();

   // Compute new dofs on the new mesh
   phi.Update();
   psi.Update();
   w.Update();
   
   // Note j stores data as a regular gridfunction
   j.Update();

   int fe_size = H1FESpace->GetTrueVSize();

   //update offset vector
   true_offset[0] = 0;
   true_offset[1] = fe_size;
   true_offset[2] = 2*fe_size;
   true_offset[3] = 3*fe_size;

   // Resize S
   S.Update(true_offset);

   // Compute "true" dofs and store them in S
   phi.GetTrueDofs(S.GetBlock(0));
   psi.GetTrueDofs(S.GetBlock(1));
     w.GetTrueDofs(S.GetBlock(2));

   H1FESpace->UpdatesFinished();
}
