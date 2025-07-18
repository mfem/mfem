#ifndef MFEM_REMAP_HPP
#define MFEM_REMAP_HPP

#include "mfem.hpp"
// Provide a make_unique implementation for C++11
#if __cplusplus <= 201103L
#include <memory>
#include <utility>

#include "miniapps/autodiff/admfem.hpp"

// define make_unique for C++11
namespace std
{
template<typename T, typename... Args>
std::unique_ptr<T> make_unique( Args&&... args )
{
   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}
#endif // __cplusplus <= 201103L


namespace mfem
{

class OptProblem : public OptimizationProblem
{
#ifdef MFEM_USE_HIOP
   friend class HiopOptimizationProblem;
#endif

protected:
   /// Not owned, some can remain unused (NULL).
   const Functional &objective;

public:

   /// In parallel, insize is the number of the local true dofs.
   OptProblem(Functional &obj, const Operator *C_, const Operator *D_)
      : OptimizationProblem(obj.Width(), C_, D_), objective(obj) {};

   /// Objective F(x). In parallel, the result should be reduced over tasks.
   real_t CalcObjective(const Vector &x) const override final
   {
      Vector y(1);
      objective.Mult(x, y);
      return y[0];
   }
   /// The result grad is expected to enter with the correct size.
   void CalcObjectiveGrad(const Vector &x, Vector &grad) const override final
   {
      objective.GetGradient().Mult(x, grad);
   }
};

// Mask Gradient g using normal cone conditions:
// g_i = min { g_i, 0 } if x_i < xmin_i - tol
//       max { g_i, 0 } if x_i > xmax_i + tol
//       g_i otherwise
inline real_t kkt_res(const BlockVector &x,
                      const BlockVector &xmin, const BlockVector &xmax,
                      const BlockVector &g, const real_t tol=1e-12)
{
   real_t res = 0.0;
   for (int iblock = 0; iblock < x.Size(); iblock++)
   {
      const Vector &x_block = x.GetBlock(iblock);
      const Vector &xmin_block = xmin.GetBlock(iblock);
      const Vector &xmax_block = xmax.GetBlock(iblock);
      const Vector &g_block = g.GetBlock(iblock);
      if (xmin_block.Size() == 0) { continue; } // skip when no mask
      for (int i = 0; i < x_block.Size(); i++)
      {
         real_t g = g_block[i];
         if (x_block[i] < xmin_block[i] - tol && g_block[i] < 0.0)
         {
            g = 0.0;
         }
         if (x_block[i] > xmax_block[i] + tol && g_block[i] > 0.0)
         {
            g = 0.0;
         }
         res += abs(g);
      }
   }
   return res;
}

inline std::vector<FiniteElementSpace*> par2normal(
   std::vector<ParFiniteElementSpace*> &par_spaces)
{
   std::vector<FiniteElementSpace*> fespaces(0);
   for (auto &p : par_spaces) { fespaces.push_back(p); }
   return fespaces;
}


template <typename T>
inline std::vector<T> ToRawPtrVector(const std::vector<std::unique_ptr<T>> &v)
{
   std::vector<T> raw_ptrs;
   for (const auto &ptr : v) { raw_ptrs.push_back(ptr.get()); }
   return raw_ptrs;
}

inline std::vector<Vector*> ToRawPtrVector(const std::vector<Vector> &v)
{
   std::vector<Vector*> raw_ptrs;
   for (const auto &vec : v) { raw_ptrs.push_back(const_cast<Vector*>(&vec)); }
   return raw_ptrs;
}

// Extract a single component from a vector-valued QuadratureFunction
inline void VecQF2QF(const QuadratureFunction &qf_vec,
                     const int comp,
                     QuadratureFunction &qf)
{
   MFEM_VERIFY(qf_vec.GetSpace() == qf.GetSpace(),
               "QuadratureFunction spaces do not match.");
   MFEM_VERIFY(qf_vec.GetVDim() > comp,
               "QuadratureFunction dimension is smaller than component index.");
   MFEM_VERIFY(qf.GetVDim() == 1,
               "QuadratureFunction vector dimension is not 1.");
   const int dim = qf_vec.GetVDim();
   const int N = qf_vec.GetSpace()->GetSize();
   const real_t *qf_vec_data = qf_vec.GetData();
   real_t *qf_data = qf.GetData();
   for (int i=comp, j=0; j<N; i+=dim, j++)
   {
      qf_data[j] = qf_vec_data[i];
   }
}

// Set a single component of a vector-valued QuadratureFunction
inline void QF2VecQF(const QuadratureFunction &qf,
                     const int comp,
                     QuadratureFunction &qf_vec)
{
   MFEM_VERIFY(qf_vec.GetSpace() == qf.GetSpace(),
               "QuadratureFunction spaces do not match.");
   MFEM_VERIFY(qf_vec.GetVDim() > comp,
               "QuadratureFunction dimension is smaller than component index.");
   MFEM_VERIFY(qf.GetVDim() == 1,
               "QuadratureFunction vector dimension is not 1.");
   const int dim = qf_vec.GetVDim();
   const int N = qf_vec.GetSpace()->GetSize();
   real_t *qf_vec_data = qf_vec.GetData();
   const real_t *qf_data = qf.GetData();
   for (int i=comp, j=0; j<N; i+=dim, j++)
   {
      qf_vec_data[i] = qf_data[j];
   }
}


class QuadratureDomainLFIntegrator : public LinearFormIntegrator
{
private:
   QuadratureFunction qf;
   const QuadratureSpace *qspace;
   FiniteElementSpace *fespace;
   mutable Vector qvals;
   std::vector<std::vector<std::unique_ptr<DenseMatrix>>> dof2q;
public:
   QuadratureDomainLFIntegrator(const QuadratureFunction &qf,
                                FiniteElementSpace &fes)
      : qf(qf)
      , qspace(dynamic_cast<const QuadratureSpace*>(qf.GetSpace()))
      , fespace(&fes)
   {
      MFEM_VERIFY(fes.FEColl()->GetMapType(fes.GetMesh()->Dimension())
                  == FiniteElement::MapType::VALUE,
                  "QuadratureLinearForm only supports finite element space with value maps.");
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "QuadratureLinearForm only supports scalar finite element spaces.");
      MFEM_VERIFY(qspace != nullptr,
                  "QuadratureFunction must be defined on a QuadratureSpace.");
      dof2q.resize(fes.GetMaxElementOrder()+1);
      for (int elem=0; elem<fes.GetNE(); elem++)
      {
         const FiniteElement &fe = *fespace->GetFE(elem);
         const IntegrationRule &ir = qspace->GetIntRule(elem);
         int fe_order = fe.GetOrder();
         int ir_order = ir.GetOrder();
         if (ir_order >= dof2q[fe_order].size())
         {
            dof2q[fe_order].resize(ir_order+1);
         }
         if (dof2q[fe_order][ir_order]) { continue; }

         // Create the matrix that maps quadrature points to element dofs
         const int dof = fe.GetDof();
         const int nq = ir.GetNPoints();
         dof2q[fe_order][ir_order]  = std::make_unique<DenseMatrix>(dof, nq);
         DenseMatrix &Q2E = *dof2q[fe_order][ir_order];
         Vector shape;
         for (int i=0; i<nq; i++)
         {
            Q2E.GetColumnReference(i, shape);
            fe.CalcShape(ir.IntPoint(i), shape);
         }
      }
      this->qf *= qspace->GetWeights();
   }

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      const int i = Tr.ElementNo;
      const int fe_order = el.GetOrder();
      const int dof = el.GetDof();
      elvect.SetSize(dof);

      const IntegrationRule &ir = qspace->GetIntRule(i);

      const int ir_order = ir.GetOrder();
      qf.GetValues(i, qvals);

      const DenseMatrix &Q2E = *dof2q[fe_order][ir_order];
      Q2E.Mult(qvals, elvect);
   }
};

// QuadratureFunction to dual vector, <qf, v> = \int qf v dx
// The returned vector is a T-vector in dual space.
class QuadratureLinearForm : public Operator
{
private:
   QuadratureSpace &qspace;
   FiniteElementSpace &fespace;
   // Mapping from dof 2 quadrature points
   // [max fe order]x[max ir order]
   std::vector<std::vector<std::unique_ptr<DenseMatrix>>> dof2q;
   mutable Vector Q_vec; // quadrature vector
   mutable Vector L_vec; // subdomain vector
   bool parallel;
public:
   QuadratureLinearForm(QuadratureSpace &qs, FiniteElementSpace &fes)
      : Operator(fes.GetTrueVSize(), qs.GetSize())
      , qspace(qs)
      , fespace(fes)
      , parallel(false)
   {
      MFEM_VERIFY(fes.FEColl()->GetMapType(fes.GetMesh()->Dimension())
                  == FiniteElement::MapType::VALUE,
                  "QuadratureLinearForm only supports finite element space with value maps.");
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "QuadratureLinearForm only supports scalar finite element spaces.");
      dof2q.resize(fes.GetMaxElementOrder()+1);
      for (int elem=0; elem<fes.GetNE(); elem++)
      {
         const FiniteElement &fe = *fespace.GetFE(elem);
         const IntegrationRule &ir = qspace.GetIntRule(elem);
         int fe_order = fe.GetOrder();
         int ir_order = ir.GetOrder();
         if (ir_order >= dof2q[fe_order].size())
         {
            dof2q[fe_order].resize(ir_order+1);
         }
         if (dof2q[fe_order][ir_order]) { continue; }

         // Create the matrix that maps quadrature points to element dofs
         const int dof = fe.GetDof();
         const int nq = ir.GetNPoints();
         dof2q[fe_order][ir_order]  = std::make_unique<DenseMatrix>(dof, nq);
         DenseMatrix &Q2E = *dof2q[fe_order][ir_order];
         Vector shape;
         for (int i=0; i<nq; i++)
         {
            Q2E.GetColumnReference(i, shape);
            fe.CalcShape(ir.IntPoint(i), shape);
         }
      }
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes =
         dynamic_cast<ParFiniteElementSpace*>(&fes);
      if (pfes)
      {
         parallel = true;
         L_vec.SetSize(fes.GetVSize());
      }
#endif
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(height);
      if (!parallel) { y = 0.0; L_vec.MakeRef(y, 0); }
      else { L_vec = 0.0; }

      Q_vec = x;
      Q_vec *= qspace.GetWeights();
      QuadratureFunction qf(&qspace, Q_vec.GetData());
      Array<int> vdofs;
      Vector elvect, qvals;
      for (int i=0; i<fespace.GetNE(); i++)
      {
         const FiniteElement &fe = *fespace.GetFE(i);
         fespace.GetElementVDofs(i, vdofs);
         const int fe_order = fe.GetOrder();
         const int dof = fe.GetDof();
         elvect.SetSize(dof);

         const IntegrationRule &ir = qspace.GetIntRule(i);
         const int ir_order = ir.GetOrder();
         qf.GetValues(i, qvals);

         const DenseMatrix &Q2E = *dof2q[fe_order][ir_order];
         Q2E.Mult(qvals, elvect);
         L_vec.AddElementVector(vdofs, elvect);
      }
      if (parallel)
      {
         const Operator* prolong = fespace.GetProlongationMatrix();
         prolong->MultTranspose(L_vec, y);
      }
   }
};

/// @brief A composition of analytic function f:R^n->R
/// F: (u1, ..., un) |-> int f(u1, u2, ..., un) dx - target
/// with discrete functions ui, where ui is either a QuadratureFunction or a GridFunction
/// target is defaults to 0.0, and can be set by SetTarget() and SetTargets()
///
/// User must provide the composition function f and its gradient df,
/// f(const Vector &u)->real_t and
/// df(const Vector &u, const Vector &grad_u)
/// For example, f(u1, u2) = u1 + u2
///   f(const Vector &u) { return u[0] + u[1]; }
///   df(const Vector &u, Vector &grad_u) { grad_u[0] = 1.0, grad_u[1] = 1.0; }
///
/// A quadrature space must be provided to evaluate the integral.
/// At each call of Mult() or EvalGradient(),
/// the input vector x will be parsed into multiple components,
/// and evaluated at quadrature points.
/// To avoid re-evaluating the input vector at each call,
/// EnableManualUpdate() can be called to freeze the discrete variables,
/// and Update(const Vector &x) can be called to update the variables manually.
///
/// This class also supports multiple functions and gradients,
/// where choosing the function index is done by SetIndex(int idx).
/// Also, see
///
/// The evaluation point x is a block vector ordered as ByVDim. That is,
/// x = [u1_dofs, u2_dofs, ..., un_dofs]
///
/// For evaluation, the input variables will be evaluated at quadrature points.
/// F: (u1, ..., un) = \int f(u1, u2, ..., un) dx
/// \nabla F: (u1, ..., un) = [\int f_i(u1, u2, ..., un) * v1 dx, ..., \int f_n(u1, u2, ..., un) * vn dx]
/// where f_i is the derivative of f with respect to u_i, provided by .
/// When ui is a QuadratureFunction, vi is just 1. Otherwise, vi is the FE basis function
///
/// The support for Hessian will be added in the future.
///
class ComposedFunctional : public SharedFunctional
{
   typedef std::function<real_t(const Vector &)> FuncType;
   typedef std::function<void(const Vector &, Vector &)> GradType;
public:
   /// @brief Construct an CompsedFunctional with multiple functions and gradients.
   /// This is useful for evaluating multiple functionals at once.
   /// @param funcs A vector of functions f:R^n->R
   /// @param grads A vector of gradients df:R^n->R^n
   /// @param qspace A QuadratureSpace defining the quadrature points
   /// @param fes A vector of FiniteElementSpace pointers, one for each variable
   /// @param space_idx Space index, where -1 indicates a QuadratureFunction
   /// and >= 0 indicates a FiniteElementSpace index.
   ComposedFunctional(FuncType f,
                      GradType g,
                      QuadratureSpace &qspace,
                      std::vector<FiniteElementSpace*> fes,
                      const Array<int> space_idx)
      : SharedFunctional(0)
      , f(f)
      , df(g)
      , space_idx(space_idx)
      , num_vars(space_idx.Size())
      , qspace(qspace)
      , fespace(fes)
      , gfs(fespace.size())
      , qlf(fespace.size())
      , qf(qspace)
      , qf_in(qspace, space_idx.Size())
      , qf_out(qspace, space_idx.Size())
   {
      MFEM_VERIFY(space_idx.Max() < (int)fespace.size() && space_idx.Min() >= -1,
                  "CompsedFunctional: Space index out of range.");
      Initialize();
   }

   ComposedFunctional(FuncType f,
                      GradType g,
                      QuadratureSpace &qspace,
                      std::vector<ParFiniteElementSpace*> fes,
                      const Array<int> space_idx)
      : ComposedFunctional(f, g, qspace, par2normal(fes), space_idx)
   {}

   const Array<int> &GetOffsets() const { return offsets; }
   void SetTarget(real_t target) { this->target = target; }
   void SetFunction(FuncType f, GradType g) { this->f = f; this->df = g; }

   /// Evaluate the derivative of <f, v> = \int f(u1, u2, ..., un) dx
   /// That is, y = [int f_i(u1, ..., un) * v1 dx, ..., int f_n(u1, ..., un) * vn dx]
   /// where f_i is the derivative of f with respect to u_i.
   void EvalGradientCurrent(Vector &y) const override
   {
      y.SetSize(width);
      BlockVector y_block(y, offsets);

      Vector qf_in_view((real_t*)nullptr, num_vars);
      Vector qf_out_view((real_t*)nullptr, num_vars);

      // quadrature point evaluation
      for (int qid=0; qid<qspace.GetSize(); qid++)
      {
         qf_in_view.MakeRef(qf_in, qid*num_vars, num_vars);
         qf_out_view.MakeRef(qf_out, qid*num_vars, num_vars);
         df(qf_in_view, qf_out_view);
      }

      // integration
      const Vector &w = qspace.GetWeights();
      real_t *qf_owned_data;
      qf.StealData(&qf_owned_data);
      for (int vid=0; vid<num_vars; vid++)
      {
         const int sid = space_idx[vid];
         if (sid < 0) // QuadratureFunction
         {
            // Set output as a quadarture function
            qf.MakeRef(y_block.GetBlock(vid), 0, qspace.GetSize());
            // Copy ith component to output
            VecQF2QF(qf_out, vid, qf);
            // Scale by weights
            y_block.GetBlock(vid) *= w;
         }
         else // FiniteElementSpace
         {
            // Set qf data to original data
            qf.SetData(qf_owned_data);
            // Copy ith component to quadrature function
            VecQF2QF(qf_out, vid, qf);
            // Integrate using QuadratureLinearForm
            // No need to handle parallel case
            qlf[sid]->Mult(qf, y_block.GetBlock(vid));
         }
      }
      // Restore qf data
      qf.SetDataAndSize(qf_owned_data, qspace.GetSize());
      qf.MakeDataOwner();
   }

   void MultCurrent(Vector &y) const override
   {
      y.SetSize(1);
      real_t &y0 = *y.GetData();
      y0 = 0.0;

      // view of [u1, ..., un] at given quadrature point
      Vector qf_in_view((real_t*)nullptr, num_vars);
      real_t *qf_in_data = qf_in.GetData();

      // integration
      const Vector &w = qspace.GetWeights();
      for (int i=0; i<qspace.GetSize(); i++) // for all quadrature points
      {
         // Update viewpoint
         qf_in_view.SetData(qf_in_data + i*num_vars);
         // Evaluate and accumulate with weight
         y0 += f(qf_in_view)*w[i];
      }
      if (IsParallel()) // if parallel, reduce the result
      {
#ifdef MFEM_USE_MPI
         MPI_Allreduce(MPI_IN_PLACE, &y0, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                       GetComm());
#endif
      }
      y0 -= target;
   }
protected:
   mutable FuncType f;
   mutable GradType df;
private:
   real_t target;

   Array<int> offsets;
   const Array<int> space_idx; // -1: QF, >= 0: FESpace index
   const int num_vars; // number of variables, i.e., the input size of f

   QuadratureSpace &qspace;
   std::vector<FiniteElementSpace*> fespace;
   mutable std::vector<std::unique_ptr<GridFunction>> gfs;
   mutable std::vector<std::unique_ptr<QuadratureLinearForm>> qlf; // int qf*v

   mutable QuadratureFunction qf; // for integration
   mutable QuadratureFunction qf_in; // store input quadrature function vdim = num_vars
   mutable QuadratureFunction qf_out; // store output quadrature function vdim = num_vars

   mutable bool is_input_frozen; // ignore input until FreezeInput is called
#ifdef MFEM_USE_MPI
   std::vector<ParFiniteElementSpace*> par_fespace;
#endif

   void Initialize()
   {

      is_input_frozen = false;
      target = 0.0;
#ifdef MFEM_USE_MPI
      // Check FESpaces are parallel or serial
      par_fespace.resize(0);
      for (auto &fes : fespace)
      {
         par_fespace.push_back(dynamic_cast<ParFiniteElementSpace*>(fes));
      }
      if (par_fespace.size() > 0 && par_fespace[0] != nullptr)
      {
         SetComm(par_fespace[0]->GetComm());
      }
#endif

      // for each FESpace, create a GridFunction and QuadratureLinearForm
      for (int sid=0; sid<fespace.size(); sid++)
      {
         MFEM_VERIFY(fespace[sid] != nullptr,
                     "CompsedFunctional::Initialize(): FiniteElementSpace pointer is null at index "
                     << sid);
         if (IsParallel())
         {
#ifdef MFEM_USE_MPI
            MFEM_VERIFY(par_fespace[sid] != nullptr,
                        "CompsedFunctional::Initialize(): ParFiniteElementSpace pointer is null at index "
                        << sid);
            gfs[sid] = std::make_unique<ParGridFunction>(par_fespace[sid]);
            qlf[sid] = std::make_unique<QuadratureLinearForm>(qspace, *par_fespace[sid]);
#endif
         }
         else
         {
            gfs[sid] = std::make_unique<GridFunction>(fespace[sid]);
            qlf[sid] = std::make_unique<QuadratureLinearForm>(qspace, *fespace[sid]);
         }
      }

      // Count degrees of freedom for each variable
      offsets.SetSize(0);
      offsets.Append(0);
      for (int vid=0; vid<num_vars; vid++)
      {
         const int sid = space_idx[vid];
         if (sid < 0)
         {
            offsets.Append(qspace.GetSize());
         }
         else
         {
            MFEM_VERIFY(sid < fespace.size(),
                        "CompsedFunctional::Initialize(): FiniteElementSpace index out of range.");
            offsets.Append(fespace[sid]->GetTrueVSize());
         }
      }
      offsets.PartialSum();
      width = offsets.Last();
   }

   // convert evaluation point x to quadrature functions and store in qf_in
   void ProcessX(const Vector &x) const override
   {
      // make a view of x with offsets
      // x_block is not modifiable!
      BlockVector x_block(const_cast<Vector&>(x), offsets);
      // store qf's data
      real_t *qf_owned_data;
      qf.StealData(&qf_owned_data);

      // Project if needed, and copy to qf_in
      for (int vid=0; vid<num_vars; vid++)
      {
         const int sid = space_idx[vid];
         if (sid < 0) // QuadratureFunction
         {
            // no need to project, just make a reference
            qf.SetDataAndSize(x_block.GetBlock(vid).GetData(), qspace.GetSize());
         }
         else // FiniteElementSpace
         {
            // T-Vector to L-Vector
            gfs[sid]->MakeTRef(fespace[sid], x_block.GetBlock(vid).GetData());
            gfs[sid]->SetFromTrueVector();
            // L-Vector to Q-Vector
            qf.SetDataAndSize(qf_owned_data, qspace.GetSize());
            qf.ProjectGridFunction(*gfs[sid]);
         }
         // Copy to qf_in. qf_in[i*num_vars + j] = qf[j]
         QF2VecQF(qf, vid, qf_in);
      }
      // restore the data
      qf.SetDataAndSize(qf_owned_data, qspace.GetSize());
      qf.MakeDataOwner();
   }

   void ShallowCopyProcessedX(SharedFunctional &owner) override
   {
      ComposedFunctional * owner_comp = dynamic_cast<ComposedFunctional*>(&owner);
      if (owner_comp)
      {
         qf_in.MakeRef(owner_comp->qf_in, 0);
         return;
      }
      SharedFunctional::ShallowCopyProcessedX(owner); // this will abort
   }
};

template <int n>
class ComposedADFunctional : public ComposedFunctional
{
public:
   /// @brief Construct an CompsedFunctional with multiple functions and gradients.
   /// This is useful for evaluating multiple functionals at once.
   /// @param funcs A vector of functions f:R^n->R
   /// @param grads A vector of gradients df:R^n->R^n
   /// @param qspace A QuadratureSpace defining the quadrature points
   /// @param fes A vector of FiniteElementSpace pointers, one for each variable
   /// @param space_idx Space index, where -1 indicates a QuadratureFunction
   /// and >= 0 indicates a FiniteElementSpace index.
   ComposedADFunctional(std::function<void(Vector &, ad::ADVectorType&, ad::ADVectorType&)> f,
                        QuadratureSpace &qspace,
                        std::vector<FiniteElementSpace*> fes,
                        const Array<int> space_idx)
      : ComposedFunctional(nullptr, nullptr, qspace, fes, space_idx)
      , funct(f)
      , ad_dummy(0) // dummy vector for AD param
   {
      this->f = [this](const Vector &x)
      {
         ad::ADVectorType ad_x(x);
         ad::ADVectorType y(1);
         this->funct(this->ad_dummy, ad_x, y); return y[0].value;
      };
      VectorFuncAutoDiff<1,n,0> adfunc(this->funct);
      this->df = [this, adfunc](const Vector &x, Vector &grad_x)
      {
         // Vector &xview = const_cast<Vector&>(x);
         Vector &xx = const_cast<Vector&>(x);
         grad_x.SetSize(n);
         DenseMatrix grad_mat(grad_x.GetData(), 1, n);
         adfunc.Jacobian(this->ad_dummy, xx, grad_mat);
      };
   }
private:
   std::function<void(Vector &, ad::ADVectorType&, ad::ADVectorType&)> funct;
   mutable Vector ad_dummy;
};


// A matrix operator that represents a collection of vectors as column of a matrix.
// For parallel use, each processor owns a part of each column vector.
// That is, each processor owns rows of the matrix.
// Returned vectors will be distributed when Mult() is called.
// Returned vectors will be synchronized when MultTranspose() is called
class ColArrayMatrix : public Operator
{
private:
   std::vector<const real_t*> cols;
   mutable Vector col; // column view
public:
   ColArrayMatrix(const std::vector<Vector*> &cols)
      : Operator(cols.size(), cols[0]->Size()), parallel(false)
   {
      for (const auto &col : cols) { this->cols.push_back(col->GetData()); }
   }

   ColArrayMatrix(const std::vector<std::unique_ptr<Vector>> &cols)
      : Operator(cols.size(), cols[0]->Size()), parallel(false)
   {
      for (const auto &col : cols) { this->cols.push_back(col->GetData()); }
   }

   ColArrayMatrix(const std::vector<Vector> &cols)
      : Operator(cols.size(), cols[0].Size()), parallel(false)
   {
      for (const auto &col : cols) { this->cols.push_back(col.GetData()); }
   }

   ColArrayMatrix(const DenseMatrix &mat)
      : Operator(mat.Width(), mat.Height()), parallel(false)
   {
      for (int i=0; i<mat.Width(); i++) { cols.push_back(mat.GetColumn(i)); }
   }

   // return the i-th column vector
   const Vector &GetColumn(int i) const
   {
      col.SetDataAndSize(const_cast<real_t*>(cols[i]), height);
      return col;
   }
   // return the i-th column vector
   const Vector &operator[](int i) const { return GetColumn(i); }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(height);
      y = 0.0;
      for (int i=0; i<width; i++)
      {
         col.SetDataAndSize(const_cast<real_t*>(cols[i]), height);
         y.Add(x[i], col);
      }
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      y.SetSize(width);
      y = 0.0; // clear output vector
      for (int i=0; i<width; i++)
      {
         col.SetDataAndSize(const_cast<real_t*>(cols[i]), height);
         y[i] += x * col;
      }
#ifdef MFEM_USE_MPI
      if (IsParallel())
      {
         MPI_Allreduce(MPI_IN_PLACE, y.GetData(), width,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, GetComm());
      }
#endif
   }

   // parallel support
public:
   bool IsParallel() const { return parallel; }
#ifdef MFEM_USE_MPI
   void SetComm(MPI_Comm comm) { parallel = true; this->comm = comm; }
   MPI_Comm GetComm() const { return comm; }
#endif

protected:
   bool parallel = false;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
};

inline void Woodbury(const Operator &Ainv, const real_t c,
                     const ColArrayMatrix &U,
                     const ColArrayMatrix &V,
                     const Vector &b, Vector &x)
{
   const int m = Ainv.Height();
   const int k = U.Width();

   MFEM_VERIFY(Ainv.Width() == m,
               "Ainv must be a square operator");
   MFEM_VERIFY(U.Height() == m,
               "U must have the same height as Ainv");
   MFEM_VERIFY(V.Height() == m,
               "V must have the same height as Ainv");
   MFEM_VERIFY(b.Size() == Ainv.Height(),
               "b must have the same size as Ainv height");
   MFEM_VERIFY(V.Width() == k,
               "V must have the same width as U");

   x.SetSize(b.Size());

   // 1. Solve Ay = b
   Vector y(m); // y = Ainv * b
   Ainv.Mult(b, y);

   // 2. Solve AZ = U
   std::vector<Vector> Z;
   for (int i=0; i<k; i++)
   {
      Z.emplace_back(m);
      Ainv.Mult(U[i], Z[i]);
      Z[i] *= c;
   }

   // 3. Compute K = I + V^T * Z
   // 4. Solve K w = V^T * y
   Vector Vt_y(k);
   V.MultTranspose(y, Vt_y);
   DenseMatrix K(k); // K = I + V^T * Z
   K = 0.0;
   Vector K_vec;
   for (int i=0; i<k; i++)
   {
      K.GetColumnReference(i, K_vec);
      V.MultTranspose(Z[i], K_vec);
   }
   for (int i=0; i<k; i++) { K(i, i) += 1.0; }

   DenseMatrixInverse Kinv(K);
   Vector w(k); // w = K^{-1} * V^T * y
   Kinv.Factor();
   Kinv.Mult(Vt_y, w);

   // 5. Compute x = y - Z * w
   x = y;
   for (int i=0; i<k; i++)
   {
      x.Add(-w[i], Z[i]);
   }
}
// Solve rank-k perturbed linear system, (A + c UV^T)x = b
// using the Woodbury formula:
// (A + c UV^T)^{-1} = A^{-1} - c A^{-1}U(I_k + c V^TA^{-1}U)^{-1}V^TA^{-1}
//
// We solve Ay = z for k + 1 times and K w = t for 1 time.
// We allocate O(m*k + k^2) memory for the algorithm.
//
// input:
//    Ainv - inverse of the operator A (m x m)
//    V, U - matrices of size (m, k)
//    b - right-hand side vector of size (m)
// output:
//    x - solution vector of size (m), does not need to be initialized
template <typename T>
inline void Woodbury(MPI_Comm comm, const Operator &Ainv, const real_t c,
                     const std::vector<T> &Uptr,
                     const std::vector<T> &Vptr,
                     const Vector &b, Vector &x)
{
   int rank;
   MPI_Comm_rank(comm, &rank);
   // Convert pointers to raw pointers
   std::vector<Vector*> U = ToRawPtrVector(Uptr);
   std::vector<Vector*> V = ToRawPtrVector(Vptr);
   MFEM_VERIFY(Ainv.Height() == Ainv.Width(),
               "Ainv must be a square operator");
   MFEM_VERIFY(V.size() == U.size(),
               "V and U must have the same number of columns");
   for (int i=0; i<V.size(); i++)
   {
      MFEM_VERIFY(V[i]->Size() == Ainv.Height(),
                  "V columns must have the same size as Ainv height");
      MFEM_VERIFY(U[i]->Size() == Ainv.Height(),
                  "U columns must have the same size as Ainv height");
   }
   MFEM_VERIFY(b.Size() == Ainv.Height(),
               "b (" << b.Size() << ") must have the same size as Ainv (" << Ainv.Height() <<
               ") height");

   const int m = Ainv.Height();
   const int k = V.size();
   x.SetSize(b.Size());

   // 1. Solve Ay = b
   Vector y(b); // y = Ainv * b
   Ainv.Mult(b, y);

   // 2. Solve AZ = U
   std::vector<Vector> Z;
   for (int i=0; i<k; i++)
   {
      Z.emplace_back(m);
      MPI_Barrier(comm);
      Ainv.Mult(*U[i], Z[i]);
      Z[i] *= c;
   }

   // 3. Compute K = I + V^T * Z
   DenseMatrix K(k); // K = I + V^T * Z
   if (rank == 0) { K.Diag(1.0, k); }
   else { K = 0.0; }
   // 4. Solve K w = V^T * y
   Vector Vt_y(k);
   for (int i=0; i<k; i++)
   {
      for (int j=0; j<k; j++)
      {
         K(i,j) += *V[i]*Z[j];
      }
      Vt_y[i] = *V[i]*y;
   }
   MPI_Allreduce(MPI_IN_PLACE, K.GetData(), k*k, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, Vt_y.GetData(), k, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, comm);
   Vector w(k);
   DenseMatrixInverse Kinv(K);
   Kinv.Factor();
   Kinv.Mult(Vt_y, w);

   // 5. Compute x = y - Z * w
   x = y;
   for (int i=0; i<k; i++)
   {
      x.Add(-w[i], Z[i]);
   }
}

template <typename T>
inline void Woodbury(const Operator &Ainv, const real_t c,
                     const std::vector<T> &Uptr,
                     const std::vector<T> &Vptr,
                     const Vector &b, Vector &x)
{
   // Convert pointers to raw pointers
   std::vector<Vector*> V = ToRawPtrVector(Vptr);
   std::vector<Vector*> U = ToRawPtrVector(Uptr);

   MFEM_ASSERT(Ainv.Height() == Ainv.Width(),
               "Ainv must be a square operator");
   MFEM_ASSERT(V.size() == U.size(),
               "V and U must have the same number of columns");
   for (int i=0; i<V.size(); i++)
   {
      MFEM_ASSERT(V[i]->Size() == Ainv.Height(),
                  "V columns must have the same size as Ainv height");
      MFEM_ASSERT(U[i]->Size() == Ainv.Height(),
                  "U columns must have the same size as Ainv height");
   }
   MFEM_ASSERT(b.Size() == Ainv.Height(),
               "b must have the same size as Ainv height");

   const int m = Ainv.Height();
   const int k = V.size();
   x.SetSize(b.Size());

   // 1. Solve Ay = b
   Vector y(m); // y = Ainv * b
   Ainv.Mult(b, y);

   // 2. Solve AZ = U
   std::vector<Vector> Z;
   for (int i=0; i<k; i++)
   {
      Z.emplace_back(m);
      Ainv.Mult(*U[i], Z[i]);
      Z[i] *= c;
   }

   // 3. Compute K = I + V^T * Z
   DenseMatrix K; // K = I + V^T * Z
   // 4. Solve K w = V^T * y
   Vector Vt_y(k);
   K.Diag(1.0, k);
   for (int i=0; i<k; i++)
   {
      for (int j=0; j<k; j++)
      {
         K(i,j) += *V[i]*Z[j];
      }
      Vt_y[i] = *V[i]*y;
   }

   DenseMatrixInverse Kinv(K);
   Vector w(k); // w = K^{-1} * V^T * y
   Kinv.Factor();
   Kinv.Mult(Vt_y, w);

   // 5. Compute x = y - Z * w
   x = y;
   for (int i=0; i<k; i++)
   {
      x.Add(-w[i], Z[i]);
   }

   Vector x2(x);
   ColArrayMatrix Vmat(V);
   ColArrayMatrix Umat(U);
   Woodbury(Ainv, c, Umat, Vmat, b, x2);
   MFEM_ASSERT(x2.DistanceTo(x) < 1e-10,
               "Woodbury: The result is not consistent with the original Woodbury formula.");
}

namespace remap
{
/// @brief A collection of conservative quantities that are considered in remap problems.

/// @brief int eta dx
inline real_t volume_f(const Vector &u) { return u[0]; }
inline void volume_df(const Vector &u, Vector &grad_u)
{
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = 1.0;
}

/// @brief int eta * rho dx
inline real_t mass_f(const Vector &u) { return u[0]*u[1]; }
inline void mass_df(const Vector &u, Vector &grad_u)
{
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = u[1];
   grad_u[1] = u[0];
}

/// @brief int eta * rho * e dx
inline real_t potential_f(const Vector &u) { return u[0]*u[1]*u[2]; }
inline void potential_df(const Vector &u, Vector &grad_u)
{
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = u[1]*u[2];
   grad_u[1] = u[0]*u[2];
   grad_u[2] = u[0]*u[1];

}

/// @brief int eta * rho * e + 0.5 * eta * rho * |v|^2 dx
inline real_t energy_f(const Vector &u)
{
   MFEM_ASSERT(u.Size() > 3,
               "energy_f: Energy functional requires at least 4 components: [eta, rho, e, v1, v2, ...]");
   real_t energy = u[2];
   for (int i=3; i<u.Size(); i++) { energy += u[i]*u[i]*0.5; }
   return u[0]*u[1]*energy;
}
inline void energy_df(const Vector &u, Vector &grad_u)
{
   MFEM_ASSERT(u.Size() > 3,
               "energy_df: Energy functional requires at least 4 components: [eta, rho, e, v1, v2, ...]");
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   real_t potential = u[2];
   real_t kinetic_energy = 0.0;
   for (int i=3; i<u.Size(); i++) { kinetic_energy += u[i]*u[i]*0.5; }
   real_t energy = potential + kinetic_energy;
   real_t mass = u[0]*u[1];
   grad_u[0] = u[1]*energy;
   grad_u[1] = u[0]*energy;
   grad_u[2] = u[0]*u[1];
   for (int i=3; i<u.Size(); i++)
   {
      grad_u[i] = mass*u[i];
   }
}

/// @brief int eta * rho * v[comp] dx
inline real_t momentum_f(const Vector &u, const int comp)
{
   MFEM_ASSERT(u.Size() > 3,
               "momentum_f: Momentum functional requires at least 4 components: [eta, rho, e, v1, v2, ...]");
   return u[0]*u[1]*u[3+comp];
}
inline void momentum_df(const Vector &u, Vector &grad_u, const int comp)
{
   MFEM_ASSERT(u.Size() > 3,
               "momentum_df: Momentum functional requires at least 4 components: [eta, rho, e, v1, v2, ...]");
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = u[1]*u[3+comp];
   grad_u[1] = u[0]*u[3+comp];
   grad_u[3+comp] = u[0]*u[1];
}

inline void remap_functionals(const int optType, const int dim,
                              std::vector<std::function<real_t(const Vector &)>> &f,
                              std::vector<std::function<void(const Vector &, Vector &)>> &df,
                              Array<int> &space_idx)
{
   f.resize(0);
   df.resize(0);
   space_idx.SetSize(0);

   f.push_back(volume_f);
   df.push_back(volume_df);
   space_idx.Append(-1);
   if (optType == 0) { return; }

   f.push_back(mass_f);
   df.push_back(mass_df);
   space_idx.Append(-1);
   if (optType == 1) { return; }

   if (optType == 2)
   {
      f.push_back(potential_f);
      df.push_back(potential_df);
      space_idx.Append(0); // L2
      return;
   }
   f.push_back(energy_f);
   df.push_back(energy_df);
   space_idx.Append(0); // L2
   for (int i=0; i<dim; i++)
   {
      f.push_back([i](const Vector &u) { return momentum_f(u, i); });
      df.push_back([i](const Vector &u, Vector &g) { momentum_df(u, g, i); });
      space_idx.Append(1); // H1
   }
}


/// @brief A functional that computes ||u - target||^2
/// where || || is the L2-norm.
/// Here, constraints are not considered.
/// GetGradient() is not the derivative, but the gradient of the functional.
/// that is, \nabla F = u - target
/// Riesz map can be applied to another derivatives using ApplyRieszMap()
class RemapObjectiveFunctional : public Functional
{
public:
   RemapObjectiveFunctional(QuadratureSpace &qspace,
                            std::vector<FiniteElementSpace*> fes,
                            const Vector &target,
                            const Array<int> &space_idx)
      : Functional(target.Size())
      , qspace(qspace)
      , fespace(fes)
      , target(target)
      , space_idx(space_idx)
      , num_vars(space_idx.Size())
   {
      Initialize();
   }
   Array<int> GetOffsets() const { return offsets; }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(1);
      y[0] = 0.0;
      add(x, -1.0, target, *diff);
      for (int i=0; i<num_vars; i++)
      {
         const int sid = space_idx[i];
         if (sid < 0)
         {
            y[0] += InnerProduct(diff->GetBlock(i),qspace.GetWeights());
         }
         else
         {
            // apply mass
            fe_mass[sid]->Mult(diff->GetBlock(i), fe_diff[sid]);
            // inner product
            y[0] += InnerProduct(diff->GetBlock(i), fe_diff[sid]);
         }
      }
      if (IsParallel())
      {
#ifdef MFEM_USE_MPI
         MPI_Allreduce(MPI_IN_PLACE, &y[0], 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, GetComm());
#endif
      }
      y[0] *= 0.5;
   }

   void EvalGradient(const Vector &x, Vector &y) const override
   {
      add(x, -1.0, target, y);
   }
private:
   QuadratureSpace &qspace;
   std::vector<FiniteElementSpace*> fespace;
   const Vector &target;
   Array<int> space_idx;
   const int num_vars;
   std::vector<std::unique_ptr<Operator>> fe_mass;
   Array<int> offsets;
   mutable std::unique_ptr<BlockVector> diff;
   mutable std::vector<Vector> fe_diff;
#ifdef MFEM_USE_MPI
   std::vector<ParFiniteElementSpace*> par_fespace;
#endif

   void Initialize()
   {
#ifdef MFEM_USE_MPI
      // Check FESpaces are parallel or serial
      par_fespace.resize(0);
      for (auto &fes : fespace)
      {
         par_fespace.push_back(dynamic_cast<ParFiniteElementSpace*>(fes));
      }
      if (par_fespace.size() > 0 && par_fespace[0] != nullptr)
      {
         SetComm(par_fespace[0]->GetComm());
      }
#endif

      out << "HI" << std::endl;
      // for each FESpace, create a GridFunction and QuadratureLinearForm
      fe_mass.resize(fespace.size());
      for (int sid=0; sid<fespace.size(); sid++)
      {
         MFEM_VERIFY(fespace[sid] != nullptr,
                     "CompsedFunctional::Initialize(): FiniteElementSpace pointer is null at index "
                     << sid);
         MFEM_VERIFY(fespace[sid]->GetVDim() == 1,
                     "CompsedFunctional::Initialize(): FiniteElementSpace must be a scalar FESpace");
         fe_diff.emplace_back(fespace[sid]->GetTrueVSize());
         if (IsParallel())
         {
#ifdef MFEM_USE_MPI
            MFEM_VERIFY(par_fespace[sid] != nullptr,
                        "CompsedFunctional::Initialize(): ParFiniteElementSpace pointer is null at index "
                        << sid);
            ParBilinearForm curr_mass(par_fespace[sid]);
            curr_mass.AddDomainIntegrator(new MassIntegrator());
            curr_mass.Assemble();
            curr_mass.Finalize();
            fe_mass[sid].reset(curr_mass.ParallelAssemble());
#endif
         }
         else
         {
#ifdef MFEM_USE_MPI
            BilinearForm curr_mass(fespace[sid]);
            curr_mass.AddDomainIntegrator(new MassIntegrator());
            curr_mass.Assemble();
            curr_mass.Finalize();
            fe_mass[sid].reset(curr_mass.LoseMat());
#endif
         }
      }

      // Count degrees of freedom for each variable
      offsets.SetSize(0);
      offsets.Append(0);
      for (int vid=0; vid<num_vars; vid++)
      {
         const int sid = space_idx[vid];
         if (sid < 0)
         {
            offsets.Append(qspace.GetSize());
         }
         else
         {
            MFEM_VERIFY(sid < fespace.size(),
                        "CompsedFunctional::Initialize(): FiniteElementSpace index out of range.");
            offsets.Append(fespace[sid]->GetTrueVSize());
         }
      }
      offsets.PartialSum();
      diff = std::make_unique<BlockVector>(offsets);
      width = offsets.Last();
   }
};

class MultiL2Projector : public Operator
{
private:
   QuadratureSpace &qspace;
   std::vector<ParFiniteElementSpace*> fespace;
   Array<int> offsets;
   const Array<int> space_idx;
   const int num_vars;

   std::vector<std::unique_ptr<Operator>> mass;
   std::vector<std::unique_ptr<Solver>> mass_prec;
   std::vector<std::unique_ptr<Operator>> projector;
public:
   MultiL2Projector(QuadratureSpace &qspace,
                    std::vector<ParFiniteElementSpace*> fes,
                    const Array<int> space_idx)
      : qspace(qspace)
      , fespace(fes)
      , space_idx(space_idx)
      , num_vars(space_idx.Size())
      , mass(fes.size())
      , mass_prec(fes.size())
      , projector(fes.size())
   {
      for (int i=0; i<fespace.size(); i++)
      {
         if (dynamic_cast<const L2_FECollection*>(fespace[i]->FEColl()))
         {
            ParBilinearForm curr_mass(fespace[i]);
            curr_mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));
            curr_mass.Assemble();
            curr_mass.Finalize();
            projector[i].reset(curr_mass.ParallelAssemble());
         }
         else if (fespace[i]->FEColl()->GetMapType(fes[i]->GetMesh()->Dimension()) ==
                  FiniteElement::VALUE)
         {
            ParBilinearForm curr_mass(fespace[i]);
            curr_mass.AddDomainIntegrator(new MassIntegrator());
            curr_mass.Assemble();
            curr_mass.Finalize();
            mass[i].reset(curr_mass.ParallelAssemble());
            auto prec = std::make_unique<HypreBoomerAMG>(static_cast<const HypreParMatrix&>
                        (*mass[i]));
            auto solver = std::make_unique<CGSolver>(fespace[i]->GetComm());
            prec->SetPrintLevel(0);
            solver->SetRelTol(1e-10);
            solver->SetAbsTol(1e-10);
            solver->SetMaxIter(1e06);
            solver->SetPrintLevel(0);
            solver->SetPreconditioner(*prec);
            solver->SetOperator(*mass[i]);
            mass_prec[i] = std::move(prec);
            projector[i] = std::move(solver);
         }
         else
         {
            MFEM_ABORT("MultiL2Projector: Only L2 or H1 spaces are supported.");
         }
      }
      offsets.SetSize(0);
      offsets.Append(0);
      for (auto s : space_idx)
      {
         s == -1 ? offsets.Append(qspace.GetSize())
                          : offsets.Append(fespace[s]->GetTrueVSize());
      }
      offsets.PartialSum();
      width = height = offsets.Last();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(height);
      BlockVector x_block(const_cast<Vector&>(x), offsets);
      BlockVector y_block(y, offsets);

      for (int i=0; i<num_vars; i++)
      {
         const int sid = space_idx[i];
         if (sid < 0) // QuadratureFunction
         {
            // just copy the data
            y_block.GetBlock(i) = x_block.GetBlock(i);
            y_block.GetBlock(i) /= qspace.GetWeights();
         }
         else // FiniteElementSpace
         {
            projector[sid]->Mult(x_block.GetBlock(i), y_block.GetBlock(i));
         }
      }
   }
};



} // namespace remap

} // namespace mfem

#endif // MFEM_REMAP_HPP
