
#include "mfem.hpp"
#include "BlkSchwarzp.hpp"
#include <iterator>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


BlkParSchwarzSmoother::BlkParSchwarzSmoother(ParMesh * cpmesh_, int ref_levels_,
ParFiniteElementSpace *fespace_, Array2D<HypreParMatrix * > blockA_)
: Solver(blockA_(0,0)->Height()+blockA_(1,0)->Height(), 
         blockA_(0,1)->Width() +blockA_(1,1)->Width()), blockA(blockA_)
{
   // construct and invert the patches
   // This can be modified so only the last part of the assembly is repeated since all the 
   // matrices have the same structure
   P.SetSize(2,2);
   for (int i=0; i<2; i++)
   {
      for (int j=0; j<2; j++)
      {
         P(i,j) = new par_patch_assembly(cpmesh_,ref_levels_, fespace_, blockA(i,j));
      }
   }

   comm = fespace_->GetComm();
   nrpatch = P(0,0)->nrpatch;
   PatchMat.SetSize(nrpatch);
   host_rank.SetSize(nrpatch);
   host_rank = P(0,0)->host_rank;
   PatchInv.SetSize(nrpatch);
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (P(0,0)->PatchMat[ip])
      {
         PatchInv[ip] = new UMFPackSolver;
         PatchInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         Array<int>block_offsets(3);
         block_offsets[0] = 0;
         block_offsets[1] = P(0,0)->PatchMat[ip]->Height();
         block_offsets[2] = block_offsets[1];
         block_offsets.PartialSum();
         BlockMatrix * blockPatchMat = new BlockMatrix(block_offsets);
         for (int i=0; i<2; i++)
         {
            for (int j=0; j<2; j++)
            {
               blockPatchMat->SetBlock(i,j,P(i,j)->PatchMat[ip]);
            }
         }
         PatchMat[ip] = blockPatchMat->CreateMonolithic();
         // delete blockPatchMat;
         PatchInv[ip]->SetOperator(*PatchMat[ip]);
      }
   }
   R.SetSize(2,2);
   R(0,0) = new PatchRestriction(P(0,0));
   R(0,1) = nullptr;
   R(1,0) = nullptr;
   R(1,1) = new PatchRestriction(P(1,1));
}


void BlkParSchwarzSmoother::Mult(const Vector &r, Vector &z) const
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   z = 0.0; // initilize the correction to zero

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = r.Size()/2;
   offsets[2] = r.Size()/2;
   offsets.PartialSum();

   double * data = r.GetData();
   BlockVector rnew_test(data, offsets);
   BlockVector rnew(rnew_test);
   BlockVector znew(offsets); 
   BlockVector raux(offsets);
   BlockOperator blkA(offsets);
   blkA.SetBlock(0,0,blockA(0,0));
   blkA.SetBlock(1,0,blockA(1,0));
   blkA.SetBlock(0,1,blockA(0,1));
   blkA.SetBlock(1,1,blockA(1,1));

   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;
      Array<BlockVector * > res0;
      Array<BlockVector * > res1;
      Array<BlockVector * > res(nrpatch);
      R(0,0)->Mult(rnew.GetBlock(0),res0);
      R(1,1)->Mult(rnew.GetBlock(1),res1);

      Array<BlockVector*> sol0(nrpatch);
      Array<BlockVector*> sol1(nrpatch);
      Array<BlockVector*> sol(nrpatch);

      for (int ip=0; ip<nrpatch; ip++)
      {
         if(myid == host_rank[ip]) 
         {
            Array<int> block_offs(3);
            block_offs[0] = 0;
            block_offs[1] = res0[ip]->Size();
            block_offs[2] = res1[ip]->Size();
            block_offs.PartialSum();   
            res[ip] = new BlockVector(block_offs);
            res[ip]->SetVector(*res0[ip],0);
            res[ip]->SetVector(*res1[ip],res0[ip]->Size());

            Array<int> block_offs0(3);
            block_offs0[0] = 0;
            block_offs0[1] = res0[ip]->GetBlock(0).Size();
            block_offs0[2] = res0[ip]->GetBlock(1).Size();
            block_offs0.PartialSum();
            Array<int> block_offs1(3);
            block_offs1[0] = 0;
            block_offs1[1] = res1[ip]->GetBlock(0).Size();
            block_offs1[2] = res1[ip]->GetBlock(1).Size();
            block_offs1.PartialSum();

            sol[ip] = new BlockVector(block_offs); 
            PatchInv[ip]->Mult(*res[ip], *sol[ip]);

            sol0[ip] = new BlockVector(sol[ip]->GetBlock(0).GetData(),block_offs0);
            sol1[ip] = new BlockVector(sol[ip]->GetBlock(1).GetData(),block_offs1);
         }
      }
      R(0,0)->MultTranspose(sol0,znew.GetBlock(0));
      R(1,1)->MultTranspose(sol1,znew.GetBlock(1));
      znew *= theta;
      z += znew;
      blkA.Mult(znew,raux); 
      rnew -= raux;
   }
}



