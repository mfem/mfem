
#include "mfem.hpp"
#include <iostream>

#include "linalg/dtensor.hpp"

namespace mfem
{
// Applies the action of a Kronecker product using sum factorization
class KroneckerProduct
{
private:
   Array<DenseMatrix*> A;
   int K;               // number of matrices
   Array<int> rows;     // sizes of each matrix
   Array<int> cols;
   int N;               // total number of rows
   int M;               // total number of cols
   // Bandwidths for each matrix (0 is unset)
   Array<int> BW;
public:
   KroneckerProduct(const Array<DenseMatrix*> &A) : A(A)
   {
      K = A.Size();
      MFEM_VERIFY((K==2) || (K==3), "Must be 2 or 3 dimensions")
      rows.SetSize(K);
      cols.SetSize(K);
      BW.SetSize(K);
      for (int k = 0; k < K; k++)
      {
         rows[k] = A[k]->Height();
         cols[k] = A[k]->Width();
         BW[k] = rows[k];
      }
      N = rows.Product();
      M = cols.Product();
   }

   void Mult(const Vector &x, Vector &y) const
   {
      MFEM_VERIFY(x.Size() == M, "Input vector must have size " << M);
      y.SetSize(N);
      y = 0.0;
      if (K == 2)
      {
         SumFactor2D<false>(x, y);
      }
      else if (K == 3)
      {
         SumFactor3D<false>(x, y);
      }
   }

   void MultTranspose(const Vector &x, Vector &y) const
   {
      MFEM_VERIFY(x.Size() == N, "Input vector must have size " << N);
      y.SetSize(M);
      y = 0.0;
      if (K == 2)
      {
         SumFactor2D<true>(x, y);
      }
      else if (K == 3)
      {
         SumFactor3D<true>(x, y);
      }

   }

   template<bool transpose = false>
   void SumFactor2D(const Vector &Xv, Vector &Yv) const
   {
      // Outer/inner loop sizes
      int OY = transpose ? rows[1] : cols[1];
      int OX = transpose ? rows[0] : cols[0];
      int IY = transpose ? cols[1] : rows[1];
      int IX = transpose ? cols[0] : rows[0];
      // Accumulator(s)
      Vector sumX(IX);
      // Reshape
      const auto X = Reshape(Xv.HostRead(), OX, OY);
      const auto Y = Reshape(Yv.HostReadWrite(), IX, IY);
      // Aliases
      const DenseMatrix &Ax = *A[0];
      const DenseMatrix &Ay = *A[1];
      const int bwx = std::floor(BW[0] / 2.0); // One-sided bandwidth
      const int bwy = std::floor(BW[1] / 2.0);

      for (int oy = 0; oy < OY; oy++)
      {
         sumX = 0.0;
         for (int ox = 0; ox < OX; ox++)
         {
            const real_t x = X(ox, oy);
            const int min_ix = std::max(0, ox - bwx);
            const int max_ix = std::min(IX, ox + bwx + 1);
            for (int ix = min_ix; ix < max_ix; ix++)
            {
               const real_t ax = transpose ? Ax(ox, ix) : Ax(ix, ox);
               sumX(ix) += x * ax;
            }
         }
         const int min_iy = std::max(0, oy - bwy);
         const int max_iy = std::min(IY, oy + bwy + 1);
         for (int iy = min_iy; iy < max_iy; iy++)
         {
            const real_t y = transpose ? Ay(oy, iy) : Ay(iy, oy);
            for (int ix = 0; ix < IX; ix++)
            {
               Y(ix, iy) += sumX(ix) * y;
            }
         }
      }
   }

   template<bool transpose = false>
   void SumFactor3D(const Vector &Xv, Vector &Yv) const
   {
      // Outer/inner loop sizes
      int OZ = transpose ? rows[2] : cols[2];
      int OY = transpose ? rows[1] : cols[1];
      int OX = transpose ? rows[0] : cols[0];
      int IZ = transpose ? cols[2] : rows[2];
      int IY = transpose ? cols[1] : rows[1];
      int IX = transpose ? cols[0] : rows[0];
      // Accumulator(s)
      Vector sumXYv(IX*IY);
      Vector sumX(IX);
      // Reshape
      const auto X = Reshape(Xv.HostRead(), OX, OY, OZ);
      const auto Y = Reshape(Yv.HostReadWrite(), IX, IY, IZ);
      auto sumXY = Reshape(sumXYv.HostReadWrite(), OX, OY);
      // Aliases
      const DenseMatrix &Ax = *A[0];
      const DenseMatrix &Ay = *A[1];
      const DenseMatrix &Az = *A[2];
      const int bwx = std::floor(BW[0] / 2.0); // One-sided bandwidth
      const int bwy = std::floor(BW[1] / 2.0);
      const int bwz = std::floor(BW[2] / 2.0);

      for (int oz = 0; oz < OZ; oz++)
      {
         sumXYv = 0.0;
         for (int oy = 0; oy < OY; oy++)
         {
            sumX = 0.0;
            // oz, oy, ox, ix
            for (int ox = 0; ox < OX; ox++)
            {
               const real_t x = X(ox, oy, oz);
               const int min_ix = std::max(0, ox - bwx);
               const int max_ix = std::min(IX, ox + bwx + 1);
               for (int ix = min_ix; ix < max_ix; ix++)
               {
                  const real_t ax = transpose ? Ax(ox, ix) : Ax(ix, ox);
                  sumX(ix) += x * ax;
               }
            }
            // oz, oy, iy, ix
            const int min_iy = std::max(0, oy - bwy);
            const int max_iy = std::min(IY, oy + bwy + 1);
            for (int iy = min_iy; iy < max_iy; iy++)
            {
               const real_t y = transpose ? Ay(oy, iy) : Ay(iy, oy);
               for (int ix = 0; ix < IX; ix++)
               {
                  sumXY(ix,iy) += sumX(ix) * y;
               }
            }
         } // for (oy)

         // oz, iz, iy, ix
         const int min_iz = std::max(0, oz - bwz);
         const int max_iz = std::min(IZ, oz + bwz + 1);
         for (int iz = min_iz; iz < max_iz; iz++)
         {
            const real_t z = transpose ? Az(oz, iz) : Az(iz, oz);
            for (int iy = 0; iy < IY; iy++)
            {
               for (int ix = 0; ix < IX; ix++)
               {
                  Y(ix, iy, iz) += sumXY(ix, iy) * z;
               }
            }
         }
      } // for (oz)
   }

   DenseMatrix GetBanded(const DenseMatrix& M, int bw)
   {
      const int I = M.Height();
      const int J = M.Width();
      MFEM_VERIFY((bw > 0) && (bw <= J), "Invalid bandwidth")
      int bwx = std::floor(bw / 2.0);
      DenseMatrix banded(I, J);
      banded = 0.0;

      for (int i = 0; i < I; i++)
      {
         const int min_j = std::max(0, i - bwx);
         const int max_j = std::min(J, i + bwx + 1);
         for (int j = min_j; j < max_j; j++)
         {
            banded(i, j) = M(i, j);
         }
      }
      // Debugging
      std::cout << "original matrix norm = " << M.FNorm2() << std::endl;
      std::cout << "banded matrix (bw " << bw << ") norm = " << banded.FNorm2() << std::endl;
      return banded;
   }

   // Computes the quantity
   // || A[k] * banded(A[k]^-1, bw) - I ||_{F2} / || A[k] ||_{F2}
   // which is a metric for how well the banded inverse approximates
   // the full inverse.
   real_t BandedInverseNorm(int k, int bw)
   {
      const DenseMatrix &Ak = *A[k];
      MFEM_VERIFY(rows[k] == cols[k], "Only square matrices");
      const DenseMatrix Akb = GetBanded(Ak, bw);
      DenseMatrix Akbi(Akb);
      Akbi.Invert();
      // numerator = Ak * Akb.Inverse() - I
      DenseMatrix numerator(rows[k], rows[k]);
      mfem::Mult(Ak, Akbi, numerator);
      for (int i = 0; i < rows[k]; i++)
      {
         numerator(i, i) -= 1.0;
      }

      return numerator.FNorm2() / Ak.FNorm2();
   }

   // Set bandwidth for kth matrix such that BandedIvnerseNorm < tol
   void SetBandwidth(int k, real_t tol = 1e-5, int bw0 = 1)
   {
      MFEM_VERIFY((k >= 0) && (k < K), "Invalid matrix index");
      for (int bw = bw0; bw <= cols[k]; bw += 2)
      {
         real_t norm = BandedInverseNorm(k, bw);
         if (norm < tol)
         {
            BW[k] = bw;
            std::cout << "Set bandwidth for index " << k << " to " << bw
                 << ". norm = " << norm << std::endl;
            return;
         }
      }
   }

};


class NURBSInterpolator
{
private:
   Mesh* ho_mesh; // High-order mesh
   Mesh* lo_mesh; // Low-order mesh
   int vdim; // Vector dimension (default 1)
   int NP; // Number of patches
   int dim; // Topological dimension
   int ho_Ndof; // Number of dofs in HO mesh
   int lo_Ndof; // Number of dofs in LO mesh

   Array2D<DenseMatrix*> R; // transfer matrices from LO->HO, per patch/dimension
   Array2D<Vector*> knots; // LO knotvectors, i.e. the collocation points
   std::vector<Array<const KnotVector*>> ho_kvs; // HO knotvectors per patch
   std::vector<Array<const KnotVector*>> lo_kvs; // LO knotvectors per patch
   std::vector<Array<int>> ho_p2g; // Patch to global mapping for HO mesh
   std::vector<Array<int>> lo_p2g; // Patch to global mapping for LO mesh

   Array2D<int> orders; // Order of basis per patch/dimension

public:
   Array<KroneckerProduct*> kron; // Kronecker product actions for each patch

   NURBSInterpolator(Mesh* ho_mesh, Mesh* lo_mesh, real_t threshold = 0.0, int vdim = 1) :
      ho_mesh(ho_mesh),
      lo_mesh(lo_mesh),
      vdim(vdim),
      NP(ho_mesh->NURBSext->GetNP()),
      dim(ho_mesh->NURBSext->Dimension()),
      ho_Ndof(ho_mesh->NURBSext->GetNDof()),
      lo_Ndof(lo_mesh->NURBSext->GetNDof())
   {
      // Basic checks
      MFEM_VERIFY(ho_mesh->IsNURBS(), "HO mesh must be a NURBS mesh.")
      MFEM_VERIFY(lo_mesh->IsNURBS(), "LO mesh must be a NURBS mesh.")
      MFEM_VERIFY(NP == lo_mesh->NURBSext->GetNP(),
               "Meshes must have the same number of patches.");
      MFEM_VERIFY(dim == lo_mesh->NURBSext->Dimension(),
               "Meshes must have the same topological dimension.");

      // Collect R, and kron
      R.SetSize(NP, dim);
      orders.SetSize(NP, dim);
      knots.SetSize(NP, dim);
      kron.SetSize(NP);
      for (int p = 0; p < NP; p++)
      {
         Array<const KnotVector*> ho_kvs_(dim);
         Array<const KnotVector*> lo_kvs_(dim);
         ho_mesh->NURBSext->GetPatchKnotVectors(p, ho_kvs_);
         lo_mesh->NURBSext->GetPatchKnotVectors(p, lo_kvs_);
         ho_kvs.push_back(ho_kvs_);
         lo_kvs.push_back(lo_kvs_);
         for (int d = 0; d < dim; d++)
         {
            orders(p, d) = ho_kvs[p][d]->GetOrder();
            // Collect knots for low-order mesh (collocation points)
            knots(p, d) = new Vector;
            Vector &u = *knots(p, d);
            lo_kvs[p][d]->GetUniqueKnots(u);

            // Collect the transfer matrices
            SparseMatrix X(ho_kvs[p][d]->GetInterpolationMatrix(u));
            X.Finalize();
            R(p, d) = new DenseMatrix(*X.ToDenseMatrix());
            R(p, d)->Invert();
         }

         // Create classes for taking kron prod
         Array<DenseMatrix*> A(dim);
         R.GetRow(p, A);
         kron[p] = new KroneckerProduct(A);
      }
      if (threshold > 0)
      {
         SetBandwidths(threshold, false);
      }

      // Collect patch to global mappings
      ho_p2g.resize(NP);
      lo_p2g.resize(NP);
      for (int p = 0; p < NP; p++)
      {
         ho_mesh->NURBSext->GetPatchDofs(p, ho_p2g[p]);
         lo_mesh->NURBSext->GetPatchDofs(p, lo_p2g[p]);
      }
   }

   // Apply R using kronecker product
   void Mult(const Vector &x, Vector &y)
   {
      Vector xp, yp;
      y.SetSize(ho_Ndof);
      y = 0.0;
      for (int p = 0; p < NP; p++)
      {
         x.GetSubVector(lo_p2g[p], xp);
         kron[p]->Mult(xp, yp);
         y.SetSubVector(ho_p2g[p], yp);
      }
   }

   // Apply R^T using kronecker product
   void MultTranspose(const Vector &x, Vector &y)
   {
      Vector xp, yp;
      y.SetSize(lo_Ndof);
      y = 0.0;
      for (int p = 0; p < NP; p++)
      {
         x.GetSubVector(ho_p2g[p], xp);
         kron[p]->MultTranspose(xp, yp);
         y.SetSubVector(lo_p2g[p], yp);
      }
   }

   // bw0 = order*2 + 1 is a good initial guess
   void SetBandwidths(real_t tol = 1.0e-5, bool guess_bw0 = true)
   {
      for (int p = 0; p < NP; p++)
      {
         for (int d = 0; d < dim; d++)
         {
            const int bw0 = (guess_bw0) ? (orders(p, d) * 2 + 1) : 1;
            kron[p]->SetBandwidth(d, tol, bw0);
         }
      }
   }

   // Evaluates an analytic function at the low-order knots
   void EvaluateFunction(
      std::function<real_t(const Vector &)> &f,
      Vector &lo_x) const
   {
      lo_x = 0.0;
      const int dim = ho_mesh->Dimension();
      Vector x(dim);
      Vector xe(dim);
      Array<int> nk(dim);
      Array<int> nel(dim);
      IntegrationPoint ip;
      Array<int> el(3);
      el = 0;
      int eidx_offset = 0;

      for (int p = 0; p < NP; p++)
      {
         // patch dimensions
         for (int d = 0; d < dim; d++)
         {
            nk[d] = knots(p, d)->Size();
            nel[d] = lo_kvs[p][d]->GetNE();
         }

         for (int idx = 0; idx < lo_p2g[p].Size(); idx++)
         {
            int i = idx % nk[0];
            int j = (idx / nk[0]) % nk[1];
            int k = (idx / (nk[0] * nk[1]));
            int ijk[3] = {i, j, k};

            // patch/cartesian index
            for (int d = 0; d < dim; d++)
            {
               // collocation point in patch coordinates
               real_t u = (*knots(p, d))(ijk[d]);
               // convert to element coordinates
               int kspan = lo_kvs[p][d]->GetSpan(u);
               xe[d] = lo_kvs[p][d]->GetRefPoint(u, kspan);
               el[d] = lo_kvs[p][d]->GetUniqueSpan(u);
            }

            int eidx = el[0]
                     + el[1] * nel[0]
                     + el[2] * (nel[0] * nel[1])
                     + eidx_offset;
            ip.Set(xe.GetData(), dim);
            // convert to physical coordinates
            lo_mesh->GetNodes()->GetVectorValue(eidx, ip, x);

            const int dofidx = lo_p2g[p][idx];
            lo_x(dofidx) = f(x);
         }

         eidx_offset += nel.Product();
      }
   }

   // Interpolates an analytic function onto the high-order space
   // using the low-order knots as collocation points.
   void InterpolateFunction(
      std::function<real_t(const Vector &)> &f,
      Vector &ho_x)
   {
      // rhs is function evaluations
      Vector rhs(lo_Ndof);
      EvaluateFunction(f, rhs);

      // R = X^-1
      Mult(rhs, ho_x);
   }

};

class R_A_Rt : public Operator
{
private:
   // const Operator *R;
   NURBSInterpolator *R;
   const Operator *A;
   mutable Vector t1, t2;
   MemoryClass mem_class;

public:
   R_A_Rt(NURBSInterpolator *R, const Operator *A)
   : Operator(A->Height(), A->Width())
   , R(R), A(A)
   {
      mem_class = A->GetMemoryClass();//*C->GetMemoryClass();
      MemoryType mem_type = GetMemoryType(mem_class);
      t1.SetSize(A->Height(), mem_type);
      t2.SetSize(A->Height(), mem_type);
   }

   MemoryClass GetMemoryClass() const override { return mem_class; }

   void Mult(const Vector &x, Vector &y) const override
   { R->MultTranspose(x, t1); A->Mult(t1, t2); R->Mult(t2, y); }

   // ~R_A_Rt() override;
};

class NURBSLORPreconditioner : public Solver
{
private:
   NURBSInterpolator* R;   // transfer operator from LO->HO
   const Operator* A;      // AMG on LO dofs

   R_A_Rt* op;
   const ConstrainedOperator* opcon;

public:
   NURBSLORPreconditioner(
      NURBSInterpolator* R,
      const Array<int> & ess_tdof_list,
      const HypreBoomerAMG* A)
   : Solver(A->Height(), A->Width(), false)
   , R(R), A(A)
   {
      op = new R_A_Rt(R, A);
      opcon = new ConstrainedOperator(op, ess_tdof_list);
   }

   // y = P x = R A^-1 R^T x
   void Mult(const Vector &x, Vector &y) const
   {
      y = 0.0;
      opcon->Mult(x, y);
   }

   void SetOperator(const Operator &op) { A = &op;};

};

// utility
void Save(const char *filename, GridFunction &x)
{
   std::ofstream ofs(filename);
   ofs.precision(16);
   x.Save(ofs);
   std::cout << "Saved vector to " << filename << std::endl;
}
void Save(const char *filename, DenseMatrix *A)
{
   std::ofstream ofs(filename);
   ofs.precision(16);
   A->PrintMatlab(ofs);
   std::cout << "Saved matrix to " << filename << std::endl;
}
void Save(const char *filename, SparseMatrix *A)
{
   if (!A->Finalized()) { A->Finalize(); }
   Save(filename, A->ToDenseMatrix());
}
void Save(const char *filename, Mesh& mesh)
{
   std::ofstream ofs(filename);
   ofs.precision(8);
   mesh.Print(ofs);
}

enum class SplineIntegrationRule { FULL_GAUSSIAN, REDUCED_GAUSSIAN, };
void SetPatchIntegrationRules(const Mesh &mesh,
                              const SplineIntegrationRule &splineRule,
                              BilinearFormIntegrator * bfi);


// For each patch, sets integration rules
void SetPatchIntegrationRules(const Mesh &mesh,
                              const SplineIntegrationRule &splineRule,
                              BilinearFormIntegrator * bfi)
{
   const int dim = mesh.Dimension();
   NURBSMeshRules * patchRules  = new NURBSMeshRules(mesh.NURBSext->GetNP(), dim);
   // Loop over patches and set a different rule for each patch.
   for (int p=0; p < mesh.NURBSext->GetNP(); ++p)
   {
      Array<const KnotVector*> kv(dim);
      mesh.NURBSext->GetPatchKnotVectors(p, kv);

      std::vector<const IntegrationRule*> ir1D(dim);
      // Construct 1D integration rules by applying the rule ir to each knot span.
      for (int i=0; i<dim; ++i)
      {
         const int order = kv[i]->GetOrder();

         if ( splineRule == SplineIntegrationRule::FULL_GAUSSIAN )
         {
            const IntegrationRule* ir = &IntRules.Get(Geometry::SEGMENT, 2*order);
            ir1D[i] = ir->ApplyToKnotIntervals(*kv[i]);
            // ir1D[i] = IntegrationRule::ApplyToKnotIntervals(ir,*kv[i]);
         }
         // else if ( splineRule == SplineIntegrationRule::REDUCED_GAUSSIAN )
         // {
         //    // ir1D[i] = IntegrationRule::GetIsogeometricReducedGaussianRule(*kv[i]);
         //    MFEM_ABORT("Unknown PatchIntegrationRule1D")
         // }
         else
         {
            MFEM_ABORT("Unknown PatchIntegrationRule1D")
         }
      }

      patchRules->SetPatchRules1D(p, ir1D);
   }  // loop (p) over patches

   patchRules->Finalize(mesh);
   bfi->SetNURBSPatchIntRule(patchRules);
}


} // namespace mfem