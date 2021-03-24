#include "lor.hpp"


const Array<int> &GetDofMap(FiniteElementSpace &fes, int i)
{
   const FiniteElement *fe = fes.GetFE(i);
   auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_ASSERT(tfe != NULL, "");
   return tfe->GetDofMap();
}

Array<int> ComputeVectorFE_LORPermutation(
   FiniteElementSpace &fes_ho,
   FiniteElementSpace &fes_lor,
   FiniteElement::MapType type)
{
   // Given an index `i` of a LOR dof, `perm[i]` is the index of the
   // corresponding HO dof.
   Array<int> perm(fes_lor.GetVSize());
   Array<int> vdof_ho, vdof_lor;

   Mesh &mesh_lor = *fes_lor.GetMesh();
   int dim = mesh_lor.Dimension();
   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   for (int ilor=0; ilor<mesh_lor.GetNE(); ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      int lor_index = cf_tr.embeddings[ilor].matrix;

      int p = fes_ho.GetOrder(iho);
      int p1 = p+1;
      int ndof_per_dim = (dim == 2) ? p*p1 :
                         type == FiniteElement::H_CURL ? p*p1*p1 : p*p*p1;

      fes_ho.GetElementVDofs(iho, vdof_ho);
      fes_lor.GetElementVDofs(ilor, vdof_lor);

      const Array<int> &dofmap_ho = GetDofMap(fes_ho, iho);
      const Array<int> &dofmap_lor = GetDofMap(fes_lor, ilor);

      int off_x = lor_index % p;
      int off_y = (lor_index / p) % p;
      int off_z = (lor_index / p) / p;

      auto absdof = [](int i) { return i < 0 ? -1-i : i; };

      auto set_perm = [&](int off_lor, int off_ho, int n1, int n2)
      {
         for (int i1=0; i1<2; ++i1)
         {
            int m = (dim == 2 || type == FiniteElement::H_DIV) ? 1 : 2;
            for (int i2=0; i2<m; ++i2)
            {
               int i;
               i = dofmap_lor[off_lor + i1 + i2*2];
               int s1 = i < 0 ? -1 : 1;
               int idof_lor = vdof_lor[absdof(i)];
               i = dofmap_ho[off_ho + i1*n1 + i2*n2];
               int s2 = i < 0 ? -1 : 1;
               int idof_ho = vdof_ho[absdof(i)];
               int s3 = idof_lor < 0 ? -1 : 1;
               int s4 = idof_ho < 0 ? -1 : 1;
               int s = s1*s2*s3*s4;
               i = absdof(idof_ho);
               perm[absdof(idof_lor)] = s < 0 ? -1-absdof(i) : absdof(i);
            }
         }
      };

      int offset;

      if (type == FiniteElement::H_CURL)
      {
         // x
         offset = off_x + off_y*p + off_z*p*p1;
         set_perm(0, offset, p, p*p1);
         // y
         offset = ndof_per_dim + off_x + off_y*(p1) + off_z*p1*p;
         set_perm(dim == 2 ? 2 : 4, offset, 1, p*p1);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p1 + off_z*p1*p1;
            set_perm(8, offset, 1, p+1);
         }
      }
      else
      {
         // x
         offset = off_x + off_y*p1 + off_z*p*p1;
         set_perm(0, offset, 1, 0);
         // y
         offset = ndof_per_dim + off_x + off_y*p + off_z*p1*p;
         set_perm(2, offset, p, 0);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p + off_z*p*p;
            set_perm(4, offset, p*p, 0);
         }
      }
   }

   return perm;
}


LORSolver::LORSolver(HypreParMatrix & A, const Array<int> p_, bool exact, Solver * prec)
   : Solver(A.Height()), p(p_)
{
   if (exact)
   {
      solv = new MUMPSSolver;
      dynamic_cast<MUMPSSolver*>(solv)->SetOperator(A);
   }
   else
   {
      solv = prec;
   }
   int n = A.Height();
   n2 = p.Size();
   n1 = n - n2;
   perm.SetSize(n);
   for (int i = 0; i<n1; i++) { perm[i] = i; }
   for (int i = 0; i<n2; i++) { perm[i+n1] = p[i]; }
}

void LORSolver::Mult(const Vector &b, Vector &x) const
{
   Vector bp(b.Size());
   Vector xp(x.Size());
   
   for (int i=0; i<n1; ++i) 
   { 
      bp[i] = b[i];
   }
   
   for (int i=n1; i<n1+n2; ++i) 
   { 
      int m = perm[i] < 0 ? n1-1-perm[i] : n1+perm[i];
      bp[i] = perm[i] < 0 ? -b[m] : b[m]; 
   }

   solv->Mult(bp, xp);

   for (int i=0; i<n1; ++i)
   {
      x[i] = xp[i];
   }
   for (int i=n1; i<x.Size(); ++i)
   {
      int pi = perm[i];
      int s = pi < 0 ? -1 : 1;
      int n = pi < 0 ? n1-1-pi : n1 + pi;
      x[n] = s*xp[i];
   }
}

ComplexLORSolver::ComplexLORSolver(HypreParMatrix & A, const Array<int> p_, bool exact, Solver * prec)
   : Solver(A.Height()), p(p_)
{
   if (exact)
   {
      solv = new MUMPSSolver;
      dynamic_cast<MUMPSSolver*>(solv)->SetOperator(A);
   }
   else
   {
      solv = prec;
   }
   int n = A.Height()/2;
   n2 = p.Size();
   n1 = n - n2;
   perm.SetSize(n);
   for (int i = 0; i<n1; i++) { perm[i] = i; }
   for (int i = 0; i<n2; i++) { perm[i+n1] = p[i]; }
}

void ComplexLORSolver::Mult(const Vector &b, Vector &x) const
{
   Vector bp(b.Size());
   Vector xp(x.Size());
   
   for (int i=0; i<n1; ++i) 
   { 
      bp[i] = b[i];
      bp[n1+n2+i] = b[n1+n2+i];
   }
   
   for (int i=n1; i<n1+n2; ++i) 
   { 
      int m = perm[i] < 0 ? n1-1-perm[i] : n1+perm[i];
      bp[i] = perm[i] < 0 ? -b[m] : b[m]; 
      bp[n1+n2+i] = perm[i] < 0 ? -b[n1+n2+m] : b[n1+n2+m]; 
   }

   solv->Mult(bp, xp);

   for (int i=0; i<n1; ++i)
   {
      x[i] = xp[i];
      x[n1+n2+i] = xp[n1+n2+i];
   }

   for (int i=n1; i<n1+n2; ++i)
   {
      int pi = perm[i];
      int s = pi < 0 ? -1 : 1;
      int n = pi < 0 ? n1-1-pi : n1 + pi;
      x[n] = s*xp[i];
      x[n+n1+n2] = s*xp[i+n1+n2];
   }

}