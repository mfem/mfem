// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "optproblem.hpp"

namespace mfem
{

// Grouped operators: the views f, h and g. Quantity q is included in the view
// of its role (f: OBJ; h: active EQ; g: active LE and GE). A summed quantity
// (objective, or weighted constraint) adds w^T of its m_q components to the
// single row Base(q); otherwise component i goes to row Base(q) + i. GE rows
// are negated. Rows are scalars for values and derivatives, and n-blocks for
// second derivatives.

class OptProblem::GroupedSecondDerivative : public Operator
{
public:
   explicit GroupedSecondDerivative(const GroupedDerivative &d_);
   void SetPoint(const Vector &x);
   void Mult(const Vector &v, Vector &Hv) const override;
   void MultTranspose(const Vector &Y, Vector &dx) const override;

private:
   void Check() const;

   const GroupedDerivative &d;
   long d_id = -1;
   std::vector<Operator *> H;
};

class OptProblem::GroupedDerivative : public Operator
{
public:
   explicit GroupedDerivative(const GroupedOperator &f_);
   void SetPoint(const Vector &x);
   void Check() const;
   void Mult(const Vector &v, Vector &jv) const override;
   void MultTranspose(const Vector &y, Vector &dx) const override;
   Operator &GetGradient(const Vector &x) const override;

   const GroupedOperator &f;
   std::vector<Operator *> J;
   long id = 0;   ///< incremented by SetPoint

private:
   long seq = -1;
   std::unique_ptr<GroupedSecondDerivative> H;
};

class OptProblem::GroupedOperator : public Operator
{
public:
   /// @a kind: 0 = f, 1 = h, 2 = g
   GroupedOperator(const OptProblem &p_, int kind_);
   void SetHeight(int h) { height = h; }

   bool Includes(const Quantity &Q) const
   {
      switch (kind)
      {
         case 0: return Q.type == QuantityType::OBJ;
         case 1: return Q.active && Q.type == QuantityType::EQ;
         default: return Q.active && (Q.type == QuantityType::LE ||
                                         Q.type == QuantityType::GE);
      }
   }
   bool Summed(const Quantity &Q) const { return kind == 0 || Q.weighted; }
   real_t Sign(const Quantity &Q) const
   { return Q.type == QuantityType::GE ? -1.0 : 1.0; }
   int Base(int q) const
   {
      return kind == 0 ? 0 :
             (kind == 1 ? p.eq_offsets[q] : p.ineq_offsets[q]);
   }

   void Mult(const Vector &x, Vector &y) const override;
   Operator &GetGradient(const Vector &x) const override;

   const OptProblem &p;
   const int kind;

private:
   std::unique_ptr<GroupedDerivative> D;
};

// ---------------------------------------------------------------------------
// ContractedOperator
// ---------------------------------------------------------------------------

ContractedOperator::ContractedOperator(const Operator &S_, const Vector &y_)
   : Operator(S_.Width()), S(S_), y(y_)
{
   MFEM_VERIFY(S.Height() == y.Size()*width,
               "ContractedOperator: S is " << S.Height() << " x " << S.Width()
               << ", expected (" << y.Size() << "*" << width << ") x "
               << width << ".");
}

void ContractedOperator::Mult(const Vector &v, Vector &Hv) const
{
   Sv.UseDevice(v.UseDevice());
   Sv.SetSize(S.Height());
   S.Mult(v, Sv);
   Hv.SetSize(width);
   Hv = 0.0;
   const real_t *yp = y.HostRead();
   Vector blk;
   for (int i = 0; i < y.Size(); i++)
   {
      blk.MakeRef(Sv, i*width, width);
      Hv.Add(yp[i], blk);
   }
}

void ContractedOperator::MultTranspose(const Vector &v, Vector &Hv) const
{
   Sv.UseDevice(v.UseDevice());
   Sv.SetSize(S.Height());
   const real_t *yp = y.HostRead();
   Vector blk;
   for (int i = 0; i < y.Size(); i++)
   {
      blk.MakeRef(Sv, i*width, width);
      blk.Set(yp[i], v);
   }
   Hv.SetSize(width);
   S.MultTranspose(Sv, Hv);
}

// ---------------------------------------------------------------------------
// OptProblem: quantities and roles
// ---------------------------------------------------------------------------

const OptProblem::Quantity &OptProblem::Get(int q) const
{
   MFEM_VERIFY(q >= 0 && q < NumQuantities(),
               "OptProblem: quantity id " << q << " is out of range [0, "
               << NumQuantities() << ").");
   return quantities[q];
}

OptProblem::Quantity &OptProblem::Get(int q)
{
   return const_cast<Quantity &>(static_cast<const OptProblem &>(*this).Get(q));
}

int OptProblem::AddQuantity(Operator &op, real_t shift)
{
   Vector s(op.Height());
   s = shift;
   return AddQuantity(op, s);
}

int OptProblem::AddQuantity(Operator &op, const Vector &shift)
{
   MFEM_VERIFY(op.Width() == n,
               "OptProblem::AddQuantity: operator width " << op.Width()
               << " does not match the input size " << n << ".");
   MFEM_VERIFY(op.Height() > 0,
               "OptProblem::AddQuantity: operator height must be positive.");
   const int m = op.Height();
   MFEM_VERIFY(shift.Size() == m,
               "OptProblem::AddQuantity: shift size " << shift.Size()
               << " does not match the operator height " << m << ".");

   quantities.emplace_back();
   Quantity &Q = quantities.back();
   Q.op = &op;
   Q.shift = shift;
   Q.val.SetSize(m);
   UpdateLayout();
   return NumQuantities() - 1;
}

void OptProblem::SetShift(int q, real_t s)
{
   Get(q).shift = s;
}

void OptProblem::SetShift(int q, const Vector &s)
{
   Quantity &Q = Get(q);
   MFEM_VERIFY(s.Size() == Q.shift.Size(),
               "OptProblem::SetShift: shift size " << s.Size()
               << " does not match the height " << Q.shift.Size()
               << " of quantity " << q << ".");
   Q.shift = s;
}

void OptProblem::SetObjective(int q)
{
   MFEM_VERIFY(GetHeight(q) == 1,
               "OptProblem::SetObjective: quantity " << q << " has height "
               << GetHeight(q) << "; use SetObjective(q, w).");
   Vector w(1);
   w = 1.0;
   SetObjective(q, w);
}

void OptProblem::SetObjective(int q, const Vector &w)
{
   Array<int> qs({q});
   std::vector<Vector> ws(1, w);
   SetObjective(qs, ws);
}

void OptProblem::SetObjective(const Array<int> &q, const Vector &w)
{
   MFEM_VERIFY(w.Size() == q.Size(),
               "OptProblem::SetObjective: " << q.Size() << " quantities but "
               << w.Size() << " weights.");
   const real_t *wp = w.HostRead();
   std::vector<Vector> ws(q.Size());
   for (int k = 0; k < q.Size(); k++)
   {
      MFEM_VERIFY(GetHeight(q[k]) == 1,
                  "OptProblem::SetObjective: quantity " << q[k]
                  << " has height " << GetHeight(q[k]) << "; use "
                  "SetObjective(q, std::vector<Vector> w).");
      ws[k].SetSize(1);
      ws[k] = wp[k];
   }
   SetObjective(q, ws);
}

void OptProblem::SetObjective(const Array<int> &q,
                              const std::vector<Vector> &w)
{
   MFEM_VERIFY((int)w.size() == q.Size(),
               "OptProblem::SetObjective: " << q.Size() << " quantities but "
               << w.size() << " weight vectors.");
   for (int k = 0; k < q.Size(); k++)
   {
      MFEM_VERIFY(w[k].Size() == GetHeight(q[k]),
                  "OptProblem::SetObjective: weight size " << w[k].Size()
                  << " does not match the height " << GetHeight(q[k])
                  << " of quantity " << q[k] << ".");
      for (int j = 0; j < k; j++)
      {
         MFEM_VERIFY(q[j] != q[k], "OptProblem::SetObjective: quantity "
                     << q[k] << " appears more than once.");
      }
   }

   for (Quantity &Q : quantities)
   {
      if (Q.type == QuantityType::OBJ) { Q.type = QuantityType::INVALID; }
   }
   for (int k = 0; k < q.Size(); k++)
   {
      Quantity &Q = quantities[q[k]];
      Q.type = QuantityType::OBJ;
      Q.active = true;
      Q.weighted = false;
      Q.w = w[k];
   }
   UpdateLayout();
}

void OptProblem::SetConstraintType(int q, QuantityType t, bool use_weights)
{
   MFEM_VERIFY(t == QuantityType::EQ || t == QuantityType::LE ||
               t == QuantityType::GE,
               "OptProblem::SetConstraintType: type must be EQ, LE or GE.");
   Quantity &Q = Get(q);
   MFEM_VERIFY(Q.type != QuantityType::OBJ,
               "OptProblem::SetConstraintType: quantity " << q << " is part "
               "of the objective; replace the objective first.");
   MFEM_VERIFY(!use_weights || Q.w.Size() > 0,
               "OptProblem::SetConstraintType: quantity " << q << " has no "
               "stored weights; pass them explicitly.");
   Q.type = t;
   Q.weighted = use_weights;
   Q.active = true;
   UpdateLayout();
}

void OptProblem::SetConstraintType(int q, QuantityType t, const Vector &w)
{
   Quantity &Q = Get(q);
   MFEM_VERIFY(w.Size() == Q.op->Height(),
               "OptProblem::SetConstraintType: weight size " << w.Size()
               << " does not match the height " << Q.op->Height()
               << " of quantity " << q << ".");
   Q.w = w;
   SetConstraintType(q, t, true);
}

void OptProblem::SetActive(int q, bool on)
{
   Quantity &Q = Get(q);
   MFEM_VERIFY(on || Q.type != QuantityType::OBJ,
               "OptProblem::SetActive: quantity " << q << " is part of the "
               "objective and cannot be deactivated.");
   if (Q.active != on)
   {
      Q.active = on;
      UpdateLayout();
   }
}

// ---------------------------------------------------------------------------
// OptProblem: dofwise bounds
// ---------------------------------------------------------------------------

void OptProblem::SetDofBounds(const Vector &lo, const Vector &hi)
{
   MFEM_VERIFY(lo.Size() == n && hi.Size() == n,
               "OptProblem::SetDofBounds: bound sizes (" << lo.Size() << ", "
               << hi.Size() << ") do not match the input size " << n << ".");
   Vector gap(n);
   gap.UseDevice(lo.UseDevice() || hi.UseDevice());
   subtract(hi, lo, gap);
   MFEM_VERIFY(gap.Min() >= 0.0, "OptProblem::SetDofBounds: lo > hi.");
   // Copies keep the host/device side of the input.
   dof_lo.UseDevice(lo.UseDevice());
   dof_lo = lo;
   dof_hi.UseDevice(hi.UseDevice());
   dof_hi = hi;
   has_dof_bounds = true;
}

void OptProblem::SetDofBounds(real_t lo, real_t hi, bool use_device)
{
   MFEM_VERIFY(lo <= hi, "OptProblem::SetDofBounds: lo = " << lo
               << " > hi = " << hi << ".");
   dof_lo.SetSize(n);
   dof_lo.UseDevice(use_device);
   dof_lo = lo;
   dof_hi.SetSize(n);
   dof_hi.UseDevice(use_device);
   dof_hi = hi;
   has_dof_bounds = true;
}

const Vector &OptProblem::GetDofLowerBound() const
{
   MFEM_VERIFY(has_dof_bounds, "OptProblem: no dofwise bounds are set.");
   return dof_lo;
}

const Vector &OptProblem::GetDofUpperBound() const
{
   MFEM_VERIFY(has_dof_bounds, "OptProblem: no dofwise bounds are set.");
   return dof_hi;
}

void OptProblem::SetRieszMap(const Operator &R)
{
   MFEM_VERIFY(R.Height() == n && R.Width() == n,
               "OptProblem::SetRieszMap: R is " << R.Height() << " x "
               << R.Width() << ", expected " << n << " x " << n << ".");
   riesz = &R;
}

const Operator &OptProblem::GetRieszMap() const
{
   MFEM_VERIFY(riesz, "OptProblem: no Riesz map is set.");
   return *riesz;
}

// ---------------------------------------------------------------------------
// OptProblem: layout
// ---------------------------------------------------------------------------

void OptProblem::UpdateLayout()
{
   const int nq = NumQuantities();
   eq_offsets.SetSize(nq + 1);
   ineq_offsets.SetSize(nq + 1);
   eq_offsets[0] = ineq_offsets[0] = 0;
   for (int q = 0; q < nq; q++)
   {
      const Quantity &Q = quantities[q];
      const int rows = Q.weighted ? 1 : Q.op->Height();
      const bool eq = Q.active && Q.type == QuantityType::EQ;
      const bool ineq = Q.active && (Q.type == QuantityType::LE ||
                                     Q.type == QuantityType::GE);
      eq_offsets[q+1] = eq_offsets[q] + (eq ? rows : 0);
      ineq_offsets[q+1] = ineq_offsets[q] + (ineq ? rows : 0);
   }
   equality->SetHeight(eq_offsets.Last());
   inequality->SetHeight(ineq_offsets.Last());
   sequence++;
   validated = false;
}

void OptProblem::Validate() const
{
   if (validated) { return; }
   for (int q = 0; q < NumQuantities(); q++)
   {
      const Quantity &Q = quantities[q];
      MFEM_VERIFY(!Q.active || Q.type != QuantityType::INVALID,
                  "OptProblem: active quantity " << q << " has "
                  "QuantityType::INVALID; call SetConstraintType(q, t) or "
                  "Deactivate(q).");
   }
   validated = true;
}

std::tuple<int, int> OptProblem::GetEqRows(int q) const
{
   Get(q);
   return std::make_tuple(eq_offsets[q], eq_offsets[q+1]);
}

std::tuple<int, int> OptProblem::GetIneqRows(int q) const
{
   Get(q);
   return std::make_tuple(ineq_offsets[q], ineq_offsets[q+1]);
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

OptProblem::GroupedOperator::GroupedOperator(const OptProblem &p_, int kind_)
   : Operator(kind_ == 0 ? 1 : 0, p_.n), p(p_), kind(kind_),
     D(new GroupedDerivative(*this)) {}

void OptProblem::GroupedOperator::Mult(const Vector &x, Vector &y) const
{
   p.Validate();
   MFEM_VERIFY(x.Size() == width, "OptProblem: x has size " << x.Size()
               << ", expected " << width << ".");
   y.SetSize(height);
   real_t *yp = y.HostWrite();
   for (int i = 0; i < height; i++) { yp[i] = 0.0; }
   for (int q = 0; q < p.NumQuantities(); q++)
   {
      const Quantity &Q = p.quantities[q];
      if (!Includes(Q)) { continue; }
      const int b = Base(q), m = Q.op->Height();
      const real_t sgn = Sign(Q);
      Q.val.UseDevice(x.UseDevice());
      Q.op->Mult(x, Q.val);
      MFEM_ASSERT(Q.val.Size() == m, "OptProblem: quantity " << q
                  << " returned a vector of size " << Q.val.Size() << ".");
      const real_t *v = Q.val.HostRead();
      const real_t *s = Q.shift.HostRead();
      if (Summed(Q))
      {
         const real_t *w = Q.w.HostRead();
         real_t r = 0.0;
         for (int i = 0; i < m; i++) { r += w[i]*(v[i] - s[i]); }
         yp[b] += sgn*r;
      }
      else
      {
         for (int i = 0; i < m; i++) { yp[b + i] = sgn*(v[i] - s[i]); }
      }
   }
}

Operator &OptProblem::GroupedOperator::GetGradient(const Vector &x) const
{
   p.Validate();
   D->SetPoint(x);
   return *D;
}

OptProblem::GroupedDerivative::GroupedDerivative(const GroupedOperator &f_)
   : Operator(0, f_.Width()), f(f_), H(new GroupedSecondDerivative(*this)) {}

void OptProblem::GroupedDerivative::SetPoint(const Vector &x)
{
   MFEM_VERIFY(x.Size() == width, "OptProblem: x has size " << x.Size()
               << ", expected " << width << ".");
   const OptProblem &p = f.p;
   height = f.Height();
   seq = p.sequence;
   id++;
   J.assign(p.NumQuantities(), nullptr);
   for (int q = 0; q < p.NumQuantities(); q++)
   {
      const Quantity &Q = p.quantities[q];
      if (!f.Includes(Q)) { continue; }
      Operator &Jq = Q.op->GetGradient(x);
      MFEM_VERIFY(Jq.Height() == Q.op->Height() && Jq.Width() == width,
                  "OptProblem: derivative of quantity " << q << " is "
                  << Jq.Height() << " x " << Jq.Width() << ", expected "
                  << Q.op->Height() << " x " << width << ".");
      J[q] = &Jq;
   }
}

void OptProblem::GroupedDerivative::Check() const
{
   MFEM_VERIFY(seq == f.p.sequence, "OptProblem: derivative obtained before "
               "a role change; call GetGradient(x) on the view again.");
}

void OptProblem::GroupedDerivative::Mult(const Vector &v, Vector &jv) const
{
   Check();
   MFEM_VERIFY(v.Size() == width, "OptProblem: input has size " << v.Size()
               << ", expected " << width << ".");
   const OptProblem &p = f.p;
   jv.SetSize(height);
   real_t *jp = jv.HostWrite();
   for (int i = 0; i < height; i++) { jp[i] = 0.0; }
   for (int q = 0; q < p.NumQuantities(); q++)
   {
      if (!J[q]) { continue; }
      const Quantity &Q = p.quantities[q];
      const int b = f.Base(q), m = Q.op->Height();
      const real_t sgn = f.Sign(Q);
      Q.val.UseDevice(v.UseDevice());
      J[q]->Mult(v, Q.val);
      const real_t *dq = Q.val.HostRead();
      if (f.Summed(Q))
      {
         const real_t *w = Q.w.HostRead();
         real_t r = 0.0;
         for (int i = 0; i < m; i++) { r += w[i]*dq[i]; }
         jp[b] += sgn*r;
      }
      else
      {
         for (int i = 0; i < m; i++) { jp[b + i] = sgn*dq[i]; }
      }
   }
}

void OptProblem::GroupedDerivative::MultTranspose(const Vector &y,
                                                  Vector &dx) const
{
   Check();
   MFEM_VERIFY(y.Size() == height, "OptProblem: input has size " << y.Size()
               << ", expected " << height << ".");
   const OptProblem &p = f.p;
   dx.SetSize(width);
   dx = 0.0;
   const real_t *yp = y.HostRead();
   for (int q = 0; q < p.NumQuantities(); q++)
   {
      if (!J[q]) { continue; }
      const Quantity &Q = p.quantities[q];
      const int b = f.Base(q), m = Q.op->Height();
      const real_t sgn = f.Sign(Q);
      Q.val.UseDevice(dx.UseDevice());
      real_t *sd = Q.val.HostWrite();
      if (f.Summed(Q))
      {
         const real_t *w = Q.w.HostRead();
         for (int i = 0; i < m; i++) { sd[i] = sgn*yp[b]*w[i]; }
      }
      else
      {
         for (int i = 0; i < m; i++) { sd[i] = sgn*yp[b + i]; }
      }
      p.dx_q.UseDevice(dx.UseDevice());
      J[q]->MultTranspose(Q.val, p.dx_q);
      dx += p.dx_q;
   }
}

Operator &OptProblem::GroupedDerivative::GetGradient(const Vector &x) const
{
   Check();
   H->SetPoint(x);
   return *H;
}

OptProblem::GroupedSecondDerivative::GroupedSecondDerivative(
   const GroupedDerivative &d_) : Operator(0, d_.Width()), d(d_) {}

void OptProblem::GroupedSecondDerivative::SetPoint(const Vector &x)
{
   MFEM_VERIFY(x.Size() == width, "OptProblem: x has size " << x.Size()
               << ", expected " << width << ".");
   const OptProblem &p = d.f.p;
   const int n = width;
   height = d.Height()*n;
   d_id = d.id;
   H.assign(p.NumQuantities(), nullptr);
   for (int q = 0; q < p.NumQuantities(); q++)
   {
      if (!d.J[q]) { continue; }
      const int m = p.quantities[q].op->Height();
      Operator &Hq = d.J[q]->GetGradient(x);
      MFEM_VERIFY(Hq.Height() == m*n && Hq.Width() == n,
                  "OptProblem: second derivative of quantity " << q << " is "
                  << Hq.Height() << " x " << Hq.Width() << ", expected ("
                  << m << "*" << n << ") x " << n << ".");
      H[q] = &Hq;
   }
}

void OptProblem::GroupedSecondDerivative::Check() const
{
   d.Check();
   MFEM_VERIFY(d_id == d.id, "OptProblem: second derivative obtained from an "
               "earlier derivative; call GetGradient(x) on the derivative "
               "again.");
}

void OptProblem::GroupedSecondDerivative::Mult(const Vector &v,
                                               Vector &Hv) const
{
   Check();
   MFEM_VERIFY(v.Size() == width, "OptProblem: input has size " << v.Size()
               << ", expected " << width << ".");
   const OptProblem &p = d.f.p;
   const int n = width;
   Hv.SetSize(height);
   Hv = 0.0;
   Vector src, dst;
   for (int q = 0; q < p.NumQuantities(); q++)
   {
      if (!H[q]) { continue; }
      const Quantity &Q = p.quantities[q];
      const int b = d.f.Base(q), m = Q.op->Height();
      const real_t sgn = d.f.Sign(Q);
      const bool summed = d.f.Summed(Q);
      const real_t *w = summed ? Q.w.HostRead() : nullptr;
      Q.stack.UseDevice(v.UseDevice());
      Q.stack.SetSize(m*n);
      H[q]->Mult(v, Q.stack);
      for (int i = 0; i < m; i++)
      {
         src.MakeRef(Q.stack, i*n, n);
         dst.MakeRef(Hv, (summed ? b : b + i)*n, n);
         dst.Add(sgn*(summed ? w[i] : 1.0), src);
      }
   }
}

void OptProblem::GroupedSecondDerivative::MultTranspose(const Vector &Y,
                                                        Vector &dx) const
{
   Check();
   MFEM_VERIFY(Y.Size() == height, "OptProblem: input has size " << Y.Size()
               << ", expected " << height << ".");
   const OptProblem &p = d.f.p;
   const int n = width;
   dx.SetSize(n);
   dx = 0.0;
   Vector &Yv = const_cast<Vector &>(Y);   // read-only views below
   Vector src, dst;
   for (int q = 0; q < p.NumQuantities(); q++)
   {
      if (!H[q]) { continue; }
      const Quantity &Q = p.quantities[q];
      const int b = d.f.Base(q), m = Q.op->Height();
      const real_t sgn = d.f.Sign(Q);
      const bool summed = d.f.Summed(Q);
      const real_t *w = summed ? Q.w.HostRead() : nullptr;
      Q.stack.UseDevice(dx.UseDevice());
      Q.stack.SetSize(m*n);
      for (int i = 0; i < m; i++)
      {
         dst.MakeRef(Q.stack, i*n, n);
         src.MakeRef(Yv, (summed ? b : b + i)*n, n);
         dst.Set(sgn*(summed ? w[i] : 1.0), src);
      }
      p.dx_q.UseDevice(dx.UseDevice());
      H[q]->MultTranspose(Q.stack, p.dx_q);
      dx += p.dx_q;
   }
}

// ---------------------------------------------------------------------------
// OptProblem: solver side
// ---------------------------------------------------------------------------

OptProblem::OptProblem(int n_) : n(n_)
{
   MFEM_VERIFY(n >= 0, "OptProblem: input size must be non-negative.");
   dx_q.SetSize(n);
   int kind = 0;
   for (auto *view : {&objective, &equality, &inequality})
   {
      view->reset(new GroupedOperator(*this, kind++));
   }
   UpdateLayout();
}

OptProblem::~OptProblem() = default;

const Operator &OptProblem::GetObjectiveOperator() const
{
   return *objective;
}

const Operator &OptProblem::GetEqualityConstraintOperator() const
{
   return *equality;
}

const Operator &OptProblem::GetInequalityConstraintOperator() const
{
   return *inequality;
}

long OptProblem::Update(const Vector &x)
{
   MFEM_VERIFY(x.Size() == n, "OptProblem::Update: x has size " << x.Size()
               << ", expected " << n << ".");
   ++point_sequence;
   for (const Quantity &Q : quantities)
   {
      if (auto *u = dynamic_cast<UpdatableOperator *>(Q.op))
      {
         u->Update(x, point_sequence);
      }
   }
   return point_sequence;
}

} // namespace mfem
