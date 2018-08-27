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


#ifndef MFEM_BACKENDS_PA_HOSTDOMAINKERNEL_HPP
#define MFEM_BACKENDS_PA_HOSTDOMAINKERNEL_HPP

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "util.hpp"
#include "padomainkernel.hpp"

namespace mfem
{

namespace pa
{

template <PAOp OpName>
class HostDomainKernel;

template <typename Equation, Location Device>
class QuadTensorFunc;

template <>
class HostDomainKernel<BtDB> {
private:
	template <typename Equation>
	using QFunc = QuadTensorFunc<Equation, Host>;
	// typedef QuadTensorFunc<Equation, Host> QFunc;
	const mfem::FiniteElementSpace& trial_fes, test_fes;
	Tensor<2> B;//I don't like that this is not const
	const int dim;
	const int nbElts;
	// typedef typename TensorType<TensorDim<Equation>::value, Device>::type Tensor;
	mutable Tensor<1> D_e;
public:
	template <typename Equation>
	HostDomainKernel(const Equation* eq)
		: trial_fes(eq->getTrialFESpace())
		, test_fes(eq->getTestFESpace())
		, B(eq->getNbDofs1d(), eq->getNbQuads1d())
		, dim(eq->getDim())
		, nbElts(eq->getNbElts())
		, D_e()
	{
		const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(trial_fes.GetFE(0)));
		const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
		const IntegrationRule& ir1d = eq->getIntRule1d();

		const int dofs = B.Height();
		const int quads1d = B.Width();

		// B  = Tensor(dofs, quads1d);

		mfem::Vector u(dofs);
		mfem::Vector d(dofs);
		for (int k = 0; k < quads1d; k++)
		{
			const IntegrationPoint &ip = ir1d.IntPoint(k);
			basis1d.Eval(ip.x, u, d);
			for (int i = 0; i < dofs; i++)
			{
				B(i, k) = u(i);
			}
		}
	}

	template <typename Equation>
	void evalD(const QFunc<Equation>& qfunc, Tensor<2>& D) const {
		for (int e = 0; e < qfunc.getTrialFESpace().GetNE(); ++e)
		{
			D_e.slice(D, e);
			qfunc.evalD(e, D_e);
		}
	}

	template <typename Tensor>
	void Mult(const Tensor& D, const HostVector<double>& x, HostVector<double>& y) const {
		if (dim == 1) Mult1d(D, x, y);
		else if (dim == 2 ) Mult2d(D, x, y);
		else if (dim == 3 ) Mult3d(D, x, y);
	}

	template <typename Tensor>
	void MultAdd(const Tensor& D, const HostVector<double>& x, HostVector<double>& y) const {
		if (dim == 1) MultAdd1d(D, x, y);
		else if (dim == 2 ) MultAdd2d(D, x, y);
		else if (dim == 3 ) MultAdd3d(D, x, y);
	}

private:
	void Mult1d(const Tensor<2, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d;
		Tensor<dim> BT(quads1d), DBT(quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d);
			contract(B, T, BT);
			cWiseMult(De, BT, DBT);
			contractT(B, DBT, R);
		}
	}
	void Mult2d(const Tensor<2, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), BBT(quads1d, quads1d),
		       DBT(quads1d, quads1d), BDBT(quads1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d, quads1d);
			contract(B, T, BT);
			contract(B, BT, BBT);
			cWiseMult(De, BBT, DBT);
			contractT(B, DBT, BDBT);
			contractT(B, BDBT, R);
		}
	}
	void Mult3d(const Tensor<2, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), BBT(dofs1d, quads1d, quads1d), BBBT(quads1d, quads1d, quads1d),
		       DBT(quads1d, quads1d, quads1d), BDBT(quads1d, quads1d, dofs1d), BBDBT(quads1d, dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d, quads1d, quads1d);
			contract(B, T, BT);
			contract(B, BT, BBT);
			contract(B, BBT, BBBT);
			cWiseMult(De, BBBT, DBT);
			contractT(B, DBT, BDBT);
			contractT(B, BDBT, BBDBT);
			contractT(B, BBDBT, R);
		}
	}
	template <typename Equation>
	void Mult1d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d;
		const int quads1d = B.Width();
		// const int quads   = quads1d;
		Tensor<dim> BT(quads1d), DBT(quads1d);
		Tensor<dim> De(quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			D.evalD(e, De);
			contract(B, T, BT);
			cWiseMult(De, BT, DBT);
			contractT(B, DBT, R);
		}
	}
	template <typename Equation>
	void Mult2d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), BBT(quads1d, quads1d),
		       DBT(quads1d, quads1d), BDBT(quads1d, dofs1d);
		Tensor<1> D_e(quads);
		Tensor<dim> De(D_e, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			D.evalD(e, D_e);
			contract(B, T, BT);
			contract(B, BT, BBT);
			cWiseMult(De, BBT, DBT);
			contractT(B, DBT, BDBT);
			contractT(B, BDBT, R);
		}
	}
	template <typename Equation>
	void Mult3d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), BBT(dofs1d, quads1d, quads1d), BBBT(quads1d, quads1d, quads1d),
		       DBT(quads1d, quads1d, quads1d), BDBT(quads1d, quads1d, dofs1d), BBDBT(quads1d, dofs1d, dofs1d);
		Tensor<1> D_e(quads);
		Tensor<dim> De(D_e, quads1d, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			D.evalD(e, D_e);
			contract(B, T, BT);
			contract(B, BT, BBT);
			contract(B, BBT, BBBT);
			cWiseMult(De, BBBT, DBT);
			contractT(B, DBT, BDBT);
			contractT(B, BDBT, BBDBT);
			contractT(B, BBDBT, R);
		}
	}
	void MultAdd1d(const Tensor<2, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d;
		Tensor<dim> BT(quads1d), DBT(quads1d), BDBT(dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d);
			contract(B, T, BT);
			cWiseMult(De, BT, DBT);
			contractT(B, DBT, BDBT);
			R += BDBT;
		}
	}
	void MultAdd2d(const Tensor<2, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), BBT(quads1d, quads1d),
		       DBT(quads1d, quads1d), BDBT(quads1d, dofs1d),
		       BBDBT(dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d, quads1d);
			contract(B, T, BT);
			contract(B, BT, BBT);
			cWiseMult(De, BBT, DBT);
			contractT(B, DBT, BDBT);
			contractT(B, BDBT, BBDBT);
			R += BBDBT;
		}
	}
	void MultAdd3d(const Tensor<2, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), BBT(dofs1d, quads1d, quads1d), BBBT(quads1d, quads1d, quads1d),
		       DBT(quads1d, quads1d, quads1d), BDBT(quads1d, quads1d, dofs1d), BBDBT(quads1d, dofs1d, dofs1d),
		       BBBDBT(dofs1d, dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d, quads1d, quads1d);
			contract(B, T, BT);
			contract(B, BT, BBT);
			contract(B, BBT, BBBT);
			cWiseMult(De, BBBT, DBT);
			contractT(B, DBT, BDBT);
			contractT(B, BDBT, BBDBT);
			contractT(B, BBDBT, BBBDBT);
			R += BBBDBT;
		}
	}
	template <typename Equation>
	void MultAdd1d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d;
		const int quads1d = B.Width();
		// const int quads   = quads1d;
		Tensor<dim> BT(quads1d), DBT(quads1d), BDBT(dofs1d);
		Tensor<dim> De(quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			D.evalD(e, De);
			contract(B, T, BT);
			cWiseMult(De, BT, DBT);
			contractT(B, DBT, BDBT);
			R += BDBT;
		}
	}
	template <typename Equation>
	void MultAdd2d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), BBT(quads1d, quads1d),
		       DBT(quads1d, quads1d), BDBT(quads1d, dofs1d),
		       BBDBT(dofs1d, dofs1d);
		Tensor<1> D_e(quads);
		Tensor<dim> De(D_e, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			D.evalD(e, D_e);
			contract(B, T, BT);
			contract(B, BT, BBT);
			cWiseMult(De, BBT, DBT);
			contractT(B, DBT, BDBT);
			contractT(B, BDBT, BBDBT);
			R += BBDBT;
		}
	}
	template <typename Equation>
	void MultAdd3d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), BBT(dofs1d, quads1d, quads1d), BBBT(quads1d, quads1d, quads1d),
		       DBT(quads1d, quads1d, quads1d), BDBT(quads1d, quads1d, dofs1d), BBDBT(quads1d, dofs1d, dofs1d),
		       BBBDBT(dofs1d, dofs1d, dofs1d);
		Tensor<1> D_e(quads);
		Tensor<dim> De(D_e, quads1d, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			D.evalD(e, D_e);
			contract(B, T, BT);
			contract(B, BT, BBT);
			contract(B, BBT, BBBT);
			cWiseMult(De, BBBT, DBT);
			contractT(B, DBT, BDBT);
			contractT(B, BDBT, BBDBT);
			contractT(B, BBDBT, BBBDBT);
			R += BBBDBT;
		}
	}
};

template <>
class HostDomainKernel<GtDG> {
private:
	template <typename Equation>
	using QFunc = QuadTensorFunc<Equation, Host>;
	// typedef QuadTensorFunc<Equation, Host> QFunc;
	const mfem::FiniteElementSpace& trial_fes, test_fes;
	Tensor<2> B, G;//I don't like that this is not const
	const int dim;
	const int nbElts;
	// typedef typename TensorType<TensorDim<Equation>::value, Device>::type Tensor;
	mutable Tensor<3> D_e;
public:
	template <typename Equation>
	HostDomainKernel(const Equation* eq)
		: trial_fes(eq->getTrialFESpace())
		, test_fes(eq->getTestFESpace())
		, B(eq->getNbDofs1d(), eq->getNbQuads1d())
		, G(eq->getNbDofs1d(), eq->getNbQuads1d())
		, dim(eq->getDim())
		, nbElts(eq->getNbElts())
		, D_e()
	{
		const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(trial_fes.GetFE(0)));
		const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
		const IntegrationRule& ir1d = eq->getIntRule1d();

		const int dofs = B.Height();
		const int quads1d = B.Width();

		// B  = Tensor(dofs, quads1d);

		mfem::Vector u(dofs);
		mfem::Vector d(dofs);
		for (int k = 0; k < quads1d; k++)
		{
			const IntegrationPoint &ip = ir1d.IntPoint(k);
			basis1d.Eval(ip.x, u, d);
			for (int i = 0; i < dofs; i++)
			{
				B(i, k) = u(i);
				G(i, k) = d(i);
			}
		}
	}

	template <typename Equation>
	void evalD(const QFunc<Equation>& qfunc, Tensor<4>& D) const {
		for (int e = 0; e < qfunc.getTrialFESpace().GetNE(); ++e)
		{
			D_e.slice(D, e);
			qfunc.evalD(e, D_e);
		}
	}

	template <typename Tensor>
	void Mult(const Tensor& D, const HostVector<double>& x, HostVector<double>& y) const {
		if (dim == 1) Mult1d(D, x, y);
		else if (dim == 2 ) Mult2d(D, x, y);
		else if (dim == 3 ) Mult3d(D, x, y);
	}

	template <typename Tensor>
	void MultAdd(const Tensor& D, const HostVector<double>& x, HostVector<double>& y) const {
		if (dim == 1) MultAdd1d(D, x, y);
		else if (dim == 2 ) MultAdd2d(D, x, y);
		else if (dim == 3 ) MultAdd3d(D, x, y);
	}

private:
	void Mult1d(const Tensor<4, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = G.Height();
		const int dofs    = dofs1d;
		const int quads1d = G.Width();
		const int quads   = quads1d;
		Tensor<dim> GT(quads1d), DGT(quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			Tensor<dim>       De(D.getData() + e * quads, quads1d);
			contract(G, T, GT);
			cWiseMult(De, GT, DGT);
			contractT(G, DGT, R);
		}
	}
	void Mult2d(const Tensor<4, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), GT(dofs1d, quads1d),
		       BGT(quads1d, quads1d), GBT(quads1d, quads1d),
		       D0GT(quads1d, quads1d), D1GT(quads1d, quads1d),
		       BDGT(quads1d, dofs1d), GDGT(quads1d, dofs1d),
		       GBDGT(dofs1d, dofs1d), BGDGT(dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim+2>     De(D.getData() + e *dim*dim* quads, dim, dim, quads1d, quads1d);
			contract(B, T, BT);
			contract(G, T, GT);
			contract(B, GT, BGT);
			contract(G, BT, GBT);
			cWiseMult(De, BGT, GBT, D0GT, D1GT);
			contractT(G, D0GT, GDGT);
			contractT(B, GDGT, BGDGT);
			R =  BGDGT;
			contractT(B, D1GT, BDGT);
			contractT(G, BDGT, GBDGT);
			R += GBDGT;
		}
	}
	void Mult3d(const Tensor<4, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), GT(dofs1d, dofs1d, quads1d),
		       BGT(dofs1d, quads1d, quads1d), GBT(dofs1d, quads1d, quads1d), BBT(dofs1d, quads1d, quads1d),
		       BBGT(quads1d, quads1d, quads1d), BGBT(quads1d, quads1d, quads1d), GBBT(quads1d, quads1d, quads1d),
		       D0GT(quads1d, quads1d, quads1d), D1GT(quads1d, quads1d, quads1d), D2GT(quads1d, quads1d, quads1d),
		       BD1GT(quads1d, quads1d, dofs1d), BD2GT(quads1d, quads1d, dofs1d), GDGT(quads1d, quads1d, dofs1d),
		       GBDGT(quads1d, dofs1d, dofs1d), BGDGT(quads1d, dofs1d, dofs1d), BBDGT(quads1d, dofs1d, dofs1d),
		       BGBDGT(dofs1d, dofs1d, dofs1d), BBGDGT(dofs1d, dofs1d, dofs1d), GBBDGT(dofs1d, dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim+2>     De(D.getData() + e * dim*dim*quads, dim, dim, quads1d, quads1d, quads1d);
			contract(B, T, BT);
			contract(G, T, GT);
			contract(B, GT, BGT);
			contract(G, BT, GBT);
			contract(B, BT, BBT);
			contract(B, BGT, BBGT);
			contract(B, GBT, BGBT);
			contract(G, BBT, GBBT);
			cWiseMult(De, BBGT, BGBT, GBBT, D0GT, D1GT, D2GT);
			contractT(G, D0GT, GDGT);
			contractT(B, GDGT, BGDGT);
			contractT(B, BGDGT, BBGDGT);
			R =  BBGDGT;
			contractT(B, D1GT, BD1GT);
			contractT(G, BD1GT, GBDGT);
			contractT(B, GBDGT, BGBDGT);
			R += BGBDGT;
			contractT(B, D2GT, BD2GT);
			contractT(B, BD2GT, BBDGT);
			contractT(G, BBDGT, GBBDGT);
			R += GBBDGT;
		}
	}
	template <typename Equation>
	void Mult1d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d;
		const int quads1d = B.Width();
		// const int quads   = quads1d;
		Tensor<dim> GT(quads1d), DGT(quads1d);
		Tensor<dim+2> De(dim,dim,quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			D.evalD(e, De);
			contract(G, T, GT);
			cWiseMult(De, GT, DGT);
			contractT(G, DGT, R);
		}
	}
	template <typename Equation>
	void Mult2d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), GT(dofs1d, quads1d),
		       BGT(quads1d, quads1d), GBT(quads1d, quads1d),
		       D0GT(quads1d, quads1d), D1GT(quads1d, quads1d),
		       BDGT(quads1d, dofs1d), GDGT(quads1d, dofs1d),
		       GBDGT(dofs1d, dofs1d), BGDGT(dofs1d, dofs1d);
		Tensor<3> D_e(dim,dim,quads);
		Tensor<dim+2> De(dim, dim, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			D.evalD(e, D_e);
			contract(B, T, BT);
			contract(G, T, GT);
			contract(B, GT, BGT);
			contract(G, BT, GBT);
			cWiseMult(De, BGT, GBT, D0GT, D1GT);
			contractT(G, D0GT, GDGT);
			contractT(B, GDGT, BGDGT);
			R =  BGDGT;
			contractT(B, D1GT, BDGT);
			contractT(G, BDGT, GBDGT);
			R += GBDGT;
		}
	}
	template <typename Equation>
	void Mult3d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), GT(dofs1d, dofs1d, quads1d),
		       BGT(dofs1d, quads1d, quads1d), GBT(dofs1d, quads1d, quads1d), BBT(dofs1d, quads1d, quads1d),
		       BBGT(quads1d, quads1d, quads1d), BGBT(quads1d, quads1d, quads1d), GBBT(quads1d, quads1d, quads1d),
		       D0GT(quads1d, quads1d, quads1d), D1GT(quads1d, quads1d, quads1d), D2GT(quads1d, quads1d, quads1d),
		       BD1GT(quads1d, quads1d, dofs1d), BD2GT(quads1d, quads1d, dofs1d), GDGT(quads1d, quads1d, dofs1d),
		       GBDGT(quads1d, dofs1d, dofs1d), BGDGT(quads1d, dofs1d, dofs1d), BBDGT(quads1d, dofs1d, dofs1d),
		       BGBDGT(dofs1d, dofs1d, dofs1d), BBGDGT(dofs1d, dofs1d, dofs1d), GBBDGT(dofs1d, dofs1d, dofs1d);
		Tensor<3> D_e(dim,dim,quads);
		Tensor<dim+2> De(dim, dim, quads1d, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			D.evalD(e, D_e);
			contract(B, T, BT);
			contract(G, T, GT);
			contract(B, GT, BGT);
			contract(G, BT, GBT);
			contract(B, BT, BBT);
			contract(B, BGT, BBGT);
			contract(B, GBT, BGBT);
			contract(G, BBT, GBBT);
			cWiseMult(De, BBGT, BGBT, GBBT, D0GT, D1GT, D2GT);
			contractT(G, D0GT, GDGT);
			contractT(B, GDGT, BGDGT);
			contractT(B, BGDGT, BBGDGT);
			R =  BBGDGT;
			contractT(B, D1GT, BD1GT);
			contractT(G, BD1GT, GBDGT);
			contractT(B, GBDGT, BGBDGT);
			R += BGBDGT;
			contractT(B, D2GT, BD2GT);
			contractT(B, BD2GT, BBDGT);
			contractT(G, BBDGT, GBBDGT);
			R += GBBDGT;
		}
	}
	void MultAdd1d(const Tensor<4, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = G.Height();
		const int dofs    = dofs1d;
		const int quads1d = G.Width();
		const int quads   = quads1d;
		Tensor<dim> GT(quads1d), DGT(quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			Tensor<dim>     De(D.getData() + e *dim*dim* quads, quads1d);
			contract(G, T, GT);
			cWiseMult(De, GT, DGT);
			contractT(G, DGT, R);
		}
	}
	void MultAdd2d(const Tensor<4, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), GT(dofs1d, quads1d),
		       BGT(quads1d, quads1d), GBT(quads1d, quads1d),
		       D0GT(quads1d, quads1d), D1GT(quads1d, quads1d),
		       BDGT(quads1d, dofs1d), GDGT(quads1d, dofs1d),
		       GBDGT(dofs1d, dofs1d), BGDGT(dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim+2>     De(D.getData() + e *dim*dim* quads, dim, dim, quads1d, quads1d);
			contract(B, T, BT);
			contract(G, T, GT);
			contract(B, GT, BGT);
			contract(G, BT, GBT);
			cWiseMult(De, BGT, GBT, D0GT, D1GT);
			contractT(G, D0GT, GDGT);
			contractT(B, GDGT, BGDGT);
			R += BGDGT;
			contractT(B, D1GT, BDGT);
			contractT(G, BDGT, GBDGT);
			R += GBDGT;
		}
	}
	void MultAdd3d(const Tensor<4, double>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), GT(dofs1d, dofs1d, quads1d),
		       BGT(dofs1d, quads1d, quads1d), GBT(dofs1d, quads1d, quads1d), BBT(dofs1d, quads1d, quads1d),
		       BBGT(quads1d, quads1d, quads1d), BGBT(quads1d, quads1d, quads1d), GBBT(quads1d, quads1d, quads1d),
		       D0GT(quads1d, quads1d, quads1d), D1GT(quads1d, quads1d, quads1d), D2GT(quads1d, quads1d, quads1d),
		       BD1GT(quads1d, quads1d, dofs1d), BD2GT(quads1d, quads1d, dofs1d), GDGT(quads1d, quads1d, dofs1d),
		       GBDGT(quads1d, dofs1d, dofs1d), BGDGT(quads1d, dofs1d, dofs1d), BBDGT(quads1d, dofs1d, dofs1d),
		       BGBDGT(dofs1d, dofs1d, dofs1d), BBGDGT(dofs1d, dofs1d, dofs1d), GBBDGT(dofs1d, dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim+2>     De(D.getData() + e *dim*dim* quads, dim, dim, quads1d, quads1d, quads1d);
			contract(B, T, BT);
			contract(G, T, GT);
			contract(B, GT, BGT);
			contract(G, BT, GBT);
			contract(B, BT, BBT);
			contract(B, BGT, BBGT);
			contract(B, GBT, BGBT);
			contract(G, BBT, GBBT);
			cWiseMult(De, BBGT, BGBT, GBBT, D0GT, D1GT, D2GT);
			contractT(G, D0GT, GDGT);
			contractT(B, GDGT, BGDGT);
			contractT(B, BGDGT, BBGDGT);
			R += BBGDGT;
			contractT(B, D1GT, BD1GT);
			contractT(G, BD1GT, GBDGT);
			contractT(B, GBDGT, BGBDGT);
			R += BGBDGT;
			contractT(B, D2GT, BD2GT);
			contractT(B, BD2GT, BBDGT);
			contractT(G, BBDGT, GBBDGT);
			R += GBBDGT;
		}
	}
	template <typename Equation>
	void MultAdd1d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d;
		const int quads1d = B.Width();
		// const int quads   = quads1d;
		Tensor<dim> GT(quads1d), DGT(quads1d);
		Tensor<dim+2> De(dim,dim,quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			D.evalD(e, De);
			contract(G, T, GT);
			cWiseMult(De, GT, DGT);
			contractT(G, DGT, R);
		}
	}
	template <typename Equation>
	void MultAdd2d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), GT(dofs1d, quads1d),
		       BGT(quads1d, quads1d), GBT(quads1d, quads1d),
		       D0GT(quads1d, quads1d), D1GT(quads1d, quads1d),
		       BDGT(quads1d, dofs1d), GDGT(quads1d, dofs1d),
		       GBDGT(dofs1d, dofs1d), BGDGT(dofs1d, dofs1d);
		Tensor<3> D_e(dim,dim,quads);
		Tensor<dim+2> De(dim, dim, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			D.evalD(e, D_e);
			contract(B, T, BT);
			contract(G, T, GT);
			contract(B, GT, BGT);
			contract(G, BT, GBT);
			cWiseMult(De, BGT, GBT, D0GT, D1GT);
			contractT(G, D0GT, GDGT);
			contractT(B, GDGT, BGDGT);
			R += BGDGT;
			contractT(B, D1GT, BDGT);
			contractT(G, BDGT, GBDGT);
			R += GBDGT;
		}
	}
	template <typename Equation>
	void MultAdd3d(const QFunc<Equation>& D, const HostVector<double>& U, HostVector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = B.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = B.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), GT(dofs1d, dofs1d, quads1d),
		       BGT(dofs1d, quads1d, quads1d), GBT(dofs1d, quads1d, quads1d), BBT(dofs1d, quads1d, quads1d),
		       BBGT(quads1d, quads1d, quads1d), BGBT(quads1d, quads1d, quads1d), GBBT(quads1d, quads1d, quads1d),
		       D0GT(quads1d, quads1d, quads1d), D1GT(quads1d, quads1d, quads1d), D2GT(quads1d, quads1d, quads1d),
		       BD1GT(quads1d, quads1d, dofs1d), BD2GT(quads1d, quads1d, dofs1d), GDGT(quads1d, quads1d, dofs1d),
		       GBDGT(quads1d, dofs1d, dofs1d), BGDGT(quads1d, dofs1d, dofs1d), BBDGT(quads1d, dofs1d, dofs1d),
		       BGBDGT(dofs1d, dofs1d, dofs1d), BBGDGT(dofs1d, dofs1d, dofs1d), GBBDGT(dofs1d, dofs1d, dofs1d);
		Tensor<3> D_e(dim,dim,quads);
		Tensor<dim+2> De(dim, dim, quads1d, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			D.evalD(e, D_e);
			contract(B, T, BT);
			contract(G, T, GT);
			contract(B, GT, BGT);
			contract(G, BT, GBT);
			contract(B, BT, BBT);
			contract(B, BGT, BBGT);
			contract(B, GBT, BGBT);
			contract(G, BBT, GBBT);
			cWiseMult(De, BBGT, BGBT, GBBT, D0GT, D1GT, D2GT);
			contractT(G, D0GT, GDGT);
			contractT(B, GDGT, BGDGT);
			contractT(B, BGDGT, BBGDGT);
			R += BBGDGT;
			contractT(B, D1GT, BD1GT);
			contractT(G, BD1GT, GBDGT);
			contractT(B, GBDGT, BGBDGT);
			R += BGBDGT;
			contractT(B, D2GT, BD2GT);
			contractT(B, BD2GT, BBDGT);
			contractT(G, BBDGT, GBBDGT);
			R += GBBDGT;
		}
	}
};

}

}

#endif

#endif