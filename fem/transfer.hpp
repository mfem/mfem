// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TRANSFER_HPP
#define MFEM_TRANSFER_HPP

#include "../linalg/linalg.hpp"
#include "fespace.hpp"

namespace mfem
{

class TransferOperator : public Operator
{
 private:
   Operator *opr;

 public:
   TransferOperator(const FiniteElementSpace &lFESpace_,
                    const FiniteElementSpace &hFESpace_);
   virtual ~TransferOperator();
   virtual void Mult(const Vector &x, Vector &y) const override;
   virtual void MultTranspose(const Vector &x, Vector &y) const override;
};

class OrderTransferOperator : public Operator
{
 private:
   const FiniteElementSpace &lFESpace;
   const FiniteElementSpace &hFESpace;

 public:
   OrderTransferOperator(const FiniteElementSpace &lFESpace_,
                         const FiniteElementSpace &hFESpace_);
   virtual ~OrderTransferOperator();
   virtual void Mult(const Vector &x, Vector &y) const override;
   virtual void MultTranspose(const Vector &x, Vector &y) const override;
};

class TrueTransferOperator : public Operator
{
 private:
   TransferOperator *localTransferOperator;
   TripleProductOperator *opr;

 public:
   TrueTransferOperator(const FiniteElementSpace &lFESpace_,
                        const FiniteElementSpace &hFESpace_);
   ~TrueTransferOperator();
   virtual void Mult(const Vector &x, Vector &y) const override;
   virtual void MultTranspose(const Vector &x, Vector &y) const override;
};

} // namespace mfem
#endif