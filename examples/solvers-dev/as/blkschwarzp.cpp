
#include "mfem.hpp"
#include "blkschwarzp.hpp"
#include <iterator>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


BlkParSchwarzSmoother::BlkParSchwarzSmoother(ParMesh * cpmesh_, int ref_levels_,
ParFiniteElementSpace *fespace_, BlockOperator * bop_)
: Solver(bop_->Height(),bop_->Width()), bop(bop_)         
{
   // construct and invert the patches
   // This can be modified so only the last part of the assembly is repeated since all the 
   // matrices have the same structure
   P.SetSize(2,2);
   blockA.SetSize(2,2);
   for (int i=0; i<2; i++)
   {
      for (int j=0; j<2; j++)
      {
         blockA(i,j) = static_cast<HypreParMatrix *>(&bop->GetBlock(i,j));
         P(i,j) = new par_patch_assembly(cpmesh_,ref_levels_, fespace_, blockA(i,j));
      }
   }

   comm = fespace_->GetComm();
   nrpatch = P(0,0)->nrpatch;
   host_rank.SetSize(nrpatch);
   host_rank = P(0,0)->host_rank;
   PatchInv.SetSize(nrpatch);
   PatchInvKLU.SetSize(nrpatch);
   PatchMat.SetSize(nrpatch);
   for (int ip = 0; ip < nrpatch; ++ip)
   {
      PatchInv[ip]=nullptr;
      PatchInvKLU[ip] = nullptr;
      PatchMat[ip]=nullptr;
      if (P(0,0)->PatchMat[ip])
      {
         
         Array<int>block_offsets(3);
         block_offsets[0] = 0;
         block_offsets[1] = P(0,0)->PatchMat[ip]->Height();
         block_offsets[2] = block_offsets[1];
         block_offsets.PartialSum();
         BlockMatrix blockPatchMat(block_offsets);
         for (int i=0; i<2; ++i)
         {
            for (int j=0; j<2; ++j)
            {
               blockPatchMat.SetBlock(i,j,P(i,j)->PatchMat[ip]);
            }
         }
         PatchMat[ip] = blockPatchMat.CreateMonolithic();
         for (int i=0; i<2; ++i)
         {
            for (int j=0; j<2; ++j)
            {
               delete P(i,j)->PatchMat[ip]; P(i,j)->PatchMat[ip] = nullptr;
            }
         }
         // PatchInv[ip] = new UMFPackSolver;
         // PatchInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         // PatchInv[ip]->SetOperator(*PatchMat[ip]);
         // PatchInv[ip]->SetPrintLevel(2);
         PatchInvKLU[ip] = new KLUSolver(*PatchMat[ip]);
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

   z = 0.0; // initialize the correction to zero

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

   for (int iter = 0; iter < maxit; ++iter)
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

      for (int ip=0; ip<nrpatch; ++ip)
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
            // PatchInv[ip]->Mult(*res[ip], *sol[ip]);
            PatchInvKLU[ip]->Mult(*res[ip], *sol[ip]);
            delete res[ip];
            sol0[ip] = new BlockVector(sol[ip]->GetBlock(0).GetData(),block_offs0);
            sol1[ip] = new BlockVector(sol[ip]->GetBlock(1).GetData(),block_offs1);
         }
      }
      R(0,0)->MultTranspose(sol0,znew.GetBlock(0));
      R(1,1)->MultTranspose(sol1,znew.GetBlock(1));
      for (int ip=0; ip<nrpatch; ++ip)
      {
         if(myid == host_rank[ip]) 
         {
            delete res0[ip];
            delete res1[ip];
            delete sol0[ip];
            delete sol1[ip];
            delete sol[ip];
         }
      }
      znew *= theta;
      z += znew;
      blkA.Mult(znew,raux); 
      rnew -= raux;
   }
}
BlkParSchwarzSmoother::~BlkParSchwarzSmoother()
{
   for (int i=0; i<2; i++)
   {
      for (int j=0; j<2; j++)
      {
         if (P(i,j)) delete P(i,j); 
         if (R(i,j)) delete R(i,j);
      }   
   }
   P.DeleteAll();
   R.DeleteAll();
   // estimate of KLU solver memory
   // double gb1 = 0.0;
   // double gb2 = 0.0;
   // double gb3 = 0.0;

   // double UMFgb1 = 0.0;
   // double UMFgb2 = 0.0;
   // double UMFgb3 = 0.0;
   // mfem::out << "nrpatch = " << nrpatch << endl;
   // for (int ip=0; ip<nrpatch; ++ip)
   // {
   //    if (PatchMat[ip]) 
   //    {
   //       int nnz1 = PatchMat[ip]->NumNonZeroElems();
   //       int n = PatchMat[ip]->Height();
   //       gb1 += nnz1 * 12.0 + (n+1.0)*4.0;
   //    }
   //    if (PatchInv[ip])
   //    {
   //       int units = PatchInv[ip]->Info[UMFPACK_SIZE_OF_UNIT];
   //       int SymGB = PatchInv[ip]->Info[UMFPACK_SYMBOLIC_PEAK_MEMORY];
   //       int NumGB = PatchInv[ip]->Info[UMFPACK_NUMERIC_SIZE_ESTIMATE];
   //       int TotGB = PatchInv[ip]->Info[UMFPACK_PEAK_MEMORY_ESTIMATE];

   //       UMFgb1 += SymGB*units;
   //       UMFgb2 += NumGB*units;
   //       UMFgb3 += TotGB*units;
   //       // int nnz2 = pow(PatchInv[ip]->Height(),4.0/3.0);
   //       // int n = PatchInv[ip]->Height();
   //       // gb2 += nnz2 * 12.0 + (n+1.0)*4.0;
   //    } 
   //    if (PatchInvKLU[ip])
   //    {
   //       int nnz3 = pow(PatchInvKLU[ip]->Height(),4.0/3.0);
   //       int n = PatchInvKLU[ip]->Height();
   //       gb3 += nnz3 * 12.0 + (n+1.0)*4.0;
   //    }  
   // }
   for (int ip=0; ip<nrpatch; ++ip)
   {
       if (PatchMat[ip])  delete PatchMat[ip];
       if (PatchInv[ip])  delete PatchInv[ip];
       if (PatchInvKLU[ip])  delete PatchInvKLU[ip];
   }
   PatchMat.DeleteAll();
   PatchInv.DeleteAll();
   PatchInvKLU.DeleteAll();

   // mfem::out << "Total storage for PatchMat       : " << gb1/pow(1024.0,3) << " GB " << endl;
   // mfem::out << "Symbolic storage for PatchInvUMF : " << UMFgb1/pow(1024.0,3) << " GB " << endl;
   // mfem::out << "Numeric storage for PatchInvUMF  : " << UMFgb2/pow(1024.0,3) << " GB " << endl;
   // mfem::out << "Total storage for PatchInvUMF    : " << UMFgb3/pow(1024.0,3) << " GB " << endl;
   // mfem::out << "Total storage for PatchInvKLU: " << gb3/pow(1024.0,3) << " GB " << endl;
}
