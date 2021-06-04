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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"

namespace mfem
{

void ParLinearForm::Update(ParFiniteElementSpace *pf)
{
   if (pf) { pfes = pf; }
   LinearForm::Update(pfes);
}

void ParLinearForm::Update(ParFiniteElementSpace *pf, Vector &v, int v_offset)
{
   pfes = pf;
   LinearForm::Update(pf,v,v_offset);
}

void ParLinearForm::MakeRef(FiniteElementSpace *f, Vector &v, int v_offset)
{
   LinearForm::MakeRef(f, v, v_offset);
   pfes = dynamic_cast<ParFiniteElementSpace*>(f);
   MFEM_ASSERT(pfes != NULL, "not a ParFiniteElementSpace");
}

void ParLinearForm::MakeRef(ParFiniteElementSpace *pf, Vector &v, int v_offset)
{
   LinearForm::MakeRef(pf, v, v_offset);
   pfes = pf;
}

void ParLinearForm::AssembleSharedFaces() 
{
	ParMesh *pmesh = pfes->GetParMesh();
	FaceElementTransformations *T;
	Array<int> vdofs1, vdofs2, vdofs_all;
	Vector elvec;

	int nfaces = pmesh->GetNSharedFaces();
	for (int i = 0; i < nfaces; i++)
	{
	   T = pmesh->GetSharedFaceTransformations(i);
	   int Elem2NbrNo = T->Elem2No - pmesh->GetNE();
	   pfes->GetElementVDofs(T->Elem1No, vdofs1);
	   pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
	   vdofs1.Copy(vdofs_all);
	   int height = pfes->GetVSize(); 
	   for (int j = 0; j < vdofs2.Size(); j++)
	   {
	      if (vdofs2[j] >= 0)
	      {
	         vdofs2[j] += height;
	      }
	      else
	      {
	         vdofs2[j] -= height;
	      }
	   }
	   vdofs_all.Append(vdofs2);
	   for (int k = 0; k < iflfi.Size(); k++)
	   {
	      iflfi[k]->AssembleRHSElementVect(*pfes->GetFE(T->Elem1No),
	      	*pfes->GetFaceNbrFE(Elem2NbrNo),
	      	*T, elvec); 
	      Vector local; 
	      local.MakeRef(elvec, 0, vdofs1.Size()); 
	      AddElementVector(vdofs1, local); 
	   }
	}

}

void ParLinearForm::Assemble() 
{
	if (iflfi.Size()>0) {
		pfes->ExchangeFaceNbrData(); 
	}
	LinearForm::Assemble(); 
	if (iflfi.Size()>0) {
		AssembleSharedFaces(); 
	}
}

void ParLinearForm::ParallelAssemble(Vector &tv)
{
   const Operator* prolong = pfes->GetProlongationMatrix();
   prolong->MultTranspose(*this, tv);
}

HypreParVector *ParLinearForm::ParallelAssemble()
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   const Operator* prolong = pfes->GetProlongationMatrix();
   prolong->MultTranspose(*this, *tv);
   return tv;
}

}

#endif
