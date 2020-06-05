#include "complex_additive_schwarzp.hpp"

// constructor
ComplexParPatchAssembly::ComplexParPatchAssembly(ParSesquilinearForm * bf_) :
   bf(bf_)
{



}


void ComplexParPatchAssembly::AssemblePatchMatrices(ParPatchDofInfo * p)
{
   // patch_mat.SetSize(nrpatch);
   // patch_bilinear_forms.SetSize(nrpatch);
   // patch_mat_inv.SetSize(nrpatch);
   // ess_tdof_list.resize(nrpatch);
   // for (int ip=0; ip<nrpatch; ++ip)
   // {
   //    patch_bilinear_forms[ip] = nullptr;
   //    patch_mat_inv[ip] = nullptr;
   //    patch_mat[ip] = nullptr;
   //    if (p->p->patch_mesh[ip])
   //    {
   //       // Define the patch bilinear form and apply boundary conditions (only the LHS)
   //       FiniteElementSpace * patch_fespace = p->patch_fespaces[ip];
   //       Mesh * patch_mesh = p->p->patch_mesh[ip];
   //       if (patch_mesh->bdr_attributes.Size())
   //       {
   //          Array<int> ess_bdr(patch_mesh->bdr_attributes.Max());
   //          ess_bdr = 1;
   //          patch_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list[ip]);
   //       }
   //       patch_bilinear_forms[ip] = new SesquilinearForm(patch_fespace, bf);
   //       patch_bilinear_forms[ip]->Assemble();
   //       OperatorPtr Alocal;
   //       // need to add the method FormSystemMatrix to complex_fem
   //       // patch_bilinear_forms[ip]->FormSystemMatrix(ess_tdof_list[ip],Alocal);
   //       patch_mat[ip] = &(SparseMatrix&)(*Alocal);
   //       patch_mat[ip]->Threshold(0.0);
   //       // Save the inverse
   //       patch_mat_inv[ip] = new KLUSolver;
   //       patch_mat_inv[ip]->SetOperator(*patch_mat[ip]);
   //    }
   // }
}


ComplexParPatchAssembly::~ComplexParPatchAssembly() {};


ComplexParPatchRestriction::ComplexParPatchRestriction(ComplexParPatchAssembly *
                                                       P_)
{}

void ComplexParPatchRestriction::Mult(const Vector & r ,
                                      std::vector<Vector > & res)
{}


void ComplexParPatchRestriction::MultTranspose(const std::vector<Vector > & sol,
                                               Vector & z)
{}


ComplexParAddSchwarz::ComplexParAddSchwarz(ParSesquilinearForm * pbf_)
   : Solver(2*pbf_->ParFESpace()->GetTrueVSize(),
            2*pbf_->ParFESpace()->GetTrueVSize())
{
   // cout << "Testing ComplexParAddSchwarz" << endl;
   // comm = pbf_->ParFESpace()->GetComm();
   // p = new ComplexParPatchAssembly(pbf_);
   // nrpatch = p->nrpatch;
   // R = new ComplexParPatchRestriction(p);
}

void ComplexParAddSchwarz::Mult(const Vector &r, Vector &z) const
{}

ComplexParAddSchwarz::~ComplexParAddSchwarz()
{
   // delete p;
   // delete R;
}



