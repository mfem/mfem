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

#include "padomainkernel.hpp"

namespace mfem
{

namespace pa
{

template <PAOp OpName>
class HostDomainKernel;

template <typename Equation, typename Vector>
class QuadTensorFunc;

template <>
class HostDomainKernel<BtDB> {
private:
	template <typename Equation>
	using QFunc = QuadTensorFunc<Equation, Vector<double>>;
	const mfem::FiniteElementSpace& trial_fes, test_fes;
	Tensor<2> shape1d;//I don't like that this is not const
	const int dim;
	const int nbElts;
public:
	template <typename Equation>
	HostDomainKernel(const Equation& eq)
		: trial_fes(eq.getTrialFESpace())
		, test_fes(eq.getTestFESpace())
		, shape1d(eq.getNbDofs1d(), eq.getNbQuads1d())
		, dim(eq.getDim())
		, nbElts(eq.getNbElts()) {
		const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(trial_fes.GetFE(0)));
		const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
		const IntegrationRule& ir1d = eq.getIntRule1d();

		const int dofs = shape1d.Height();
		const int quads1d = shape1d.Width();

		// shape1d  = Tensor(dofs, quads1d);

		mfem::Vector u(dofs);
		mfem::Vector d(dofs);
		for (int k = 0; k < quads1d; k++)
		{
			const IntegrationPoint &ip = ir1d.IntPoint(k);
			basis1d.Eval(ip.x, u, d);
			for (int i = 0; i < dofs; i++)
			{
				shape1d(i, k) = u(i);
			}
		}
	}

	template <typename Tensor>
	void Mult(const Tensor& D, const Vector<double>& x, Vector<double>& y) const {
		if(dim == 1) Mult1d(D,x,y);
		else if(dim == 2 ) Mult2d(D,x,y);
		else if(dim == 3 ) Mult3d(D,x,y);		
	}

	template <typename Tensor>
	void MultAdd(const Tensor& D, const Vector<double>& x, Vector<double>& y) const {
		if(dim == 1) MultAdd1d(D,x,y);
		else if(dim == 2 ) MultAdd2d(D,x,y);
		else if(dim == 3 ) MultAdd3d(D,x,y);
	}

private:
	void Mult1d(const Tensor<2, double>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d;
		Tensor<dim> BT(quads1d), DBT(quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d);
			contract(shape1d, T, BT);
			cWiseMult(De, BT, DBT);
			contractT(shape1d, DBT, R);
		}
	}
	void Mult2d(const Tensor<2, double>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), BBT(quads1d, quads1d),
		       DBT(quads1d, quads1d), BDBT(quads1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d, quads1d);
			contract(shape1d, T, BT);
			contract(shape1d, BT, BBT);
			cWiseMult(De, BBT, DBT);
			contractT(shape1d, DBT, BDBT);
			contractT(shape1d, BDBT, R);
		}
	}
	void Mult3d(const Tensor<2, double>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), BBT(dofs1d, quads1d, quads1d), BBBT(quads1d, quads1d, quads1d),
		       DBT(quads1d, quads1d, quads1d), BDBT(quads1d, quads1d, dofs1d), BBDBT(quads1d, dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d, quads1d, quads1d);
			contract(shape1d, T, BT);
			contract(shape1d, BT, BBT);
			contract(shape1d, BBT, BBBT);
			cWiseMult(De, BBBT, DBT);
			contractT(shape1d, DBT, BDBT);
			contractT(shape1d, BDBT, BBDBT);
			contractT(shape1d, BBDBT, R);
		}
	}
	template <typename Equation>
	void Mult1d(const QFunc<Equation>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d;
		const int quads1d = shape1d.Width();
		// const int quads   = quads1d;
		Tensor<dim> BT(quads1d), DBT(quads1d);
		Tensor<dim> De(quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			D.evalD(e,De);
			contract(shape1d, T, BT);
			cWiseMult(De, BT, DBT);
			contractT(shape1d, DBT, R);
		}
	}
	template <typename Equation>
	void Mult2d(const QFunc<Equation>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), BBT(quads1d, quads1d),
		       DBT(quads1d, quads1d), BDBT(quads1d, dofs1d);
		Tensor<1> D_e(quads);
		Tensor<dim> De(D_e, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			D.evalD(e,D_e);
			contract(shape1d, T, BT);
			contract(shape1d, BT, BBT);
			cWiseMult(De, BBT, DBT);
			contractT(shape1d, DBT, BDBT);
			contractT(shape1d, BDBT, R);
		}
	}
	template <typename Equation>
	void Mult3d(const QFunc<Equation>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), BBT(dofs1d, quads1d, quads1d), BBBT(quads1d, quads1d, quads1d),
		       DBT(quads1d, quads1d, quads1d), BDBT(quads1d, quads1d, dofs1d), BBDBT(quads1d, dofs1d, dofs1d);
		Tensor<1> D_e(quads);
		Tensor<dim> De(D_e,quads1d, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			D.evalD(e,D_e);
			contract(shape1d, T, BT);
			contract(shape1d, BT, BBT);
			contract(shape1d, BBT, BBBT);
			cWiseMult(De, BBBT, DBT);
			contractT(shape1d, DBT, BDBT);
			contractT(shape1d, BDBT, BBDBT);
			contractT(shape1d, BBDBT, R);
		}
	}
	void MultAdd1d(const Tensor<2, double>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d;
		Tensor<dim> BT(quads1d), DBT(quads1d), BDBT(dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d);
			contract(shape1d, T, BT);
			cWiseMult(De, BT, DBT);
			contractT(shape1d, DBT, BDBT);
			R += BDBT;
		}
	}
	void MultAdd2d(const Tensor<2, double>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), BBT(quads1d, quads1d),
		       DBT(quads1d, quads1d), BDBT(quads1d, dofs1d),
		       BBDBT(dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d, quads1d);
			contract(shape1d, T, BT);
			contract(shape1d, BT, BBT);
			cWiseMult(De, BBT, DBT);
			contractT(shape1d, DBT, BDBT);
			contractT(shape1d, BDBT, BBDBT);
			R += BBDBT;
		}
	}
	void MultAdd3d(const Tensor<2, double>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), BBT(dofs1d, quads1d, quads1d), BBBT(quads1d, quads1d, quads1d),
		       DBT(quads1d, quads1d, quads1d), BDBT(quads1d, quads1d, dofs1d), BBDBT(quads1d, dofs1d, dofs1d),
		       BBBDBT(dofs1d, dofs1d, dofs1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>    De(D.getData() + e * quads, quads1d, quads1d, quads1d);
			contract(shape1d, T, BT);
			contract(shape1d, BT, BBT);
			contract(shape1d, BBT, BBBT);
			cWiseMult(De, BBBT, DBT);
			contractT(shape1d, DBT, BDBT);
			contractT(shape1d, BDBT, BBDBT);
			contractT(shape1d, BBDBT, BBBDBT);
			R += BBBDBT;
		}
	}
	template <typename Equation>
	void MultAdd1d(const QFunc<Equation>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 1;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d;
		const int quads1d = shape1d.Width();
		// const int quads   = quads1d;
		Tensor<dim> BT(quads1d), DBT(quads1d), BDBT(dofs1d);
		Tensor<dim> De(quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d);
			D.evalD(e,De);
			contract(shape1d, T, BT);
			cWiseMult(De, BT, DBT);
			contractT(shape1d, DBT, BDBT);
			R += BDBT;
		}
	}
	template <typename Equation>
	void MultAdd2d(const QFunc<Equation>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 2;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d * dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d * quads1d;
		Tensor<dim> BT(dofs1d, quads1d), BBT(quads1d, quads1d),
		       DBT(quads1d, quads1d), BDBT(quads1d, dofs1d),
		       BBDBT(dofs1d, dofs1d);
		Tensor<1> D_e(quads);
		Tensor<dim> De(D_e,quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d);
			D.evalD(e,D_e);
			contract(shape1d, T, BT);
			contract(shape1d, BT, BBT);
			cWiseMult(De, BBT, DBT);
			contractT(shape1d, DBT, BDBT);
			contractT(shape1d, BDBT, BBDBT);
			R += BBDBT;
		}
	}
	template <typename Equation>
	void MultAdd3d(const QFunc<Equation>& D, const Vector<double>& U, Vector<double>& V) const
	{
		const int dim     = 3;
		const int dofs1d  = shape1d.Height();
		const int dofs    = dofs1d * dofs1d * dofs1d;
		const int quads1d = shape1d.Width();
		const int quads   = quads1d * quads1d * quads1d;
		Tensor<dim> BT(dofs1d, dofs1d, quads1d), BBT(dofs1d, quads1d, quads1d), BBBT(quads1d, quads1d, quads1d),
		       DBT(quads1d, quads1d, quads1d), BDBT(quads1d, quads1d, dofs1d), BBDBT(quads1d, dofs1d, dofs1d),
		       BBBDBT(dofs1d, dofs1d, dofs1d);
		Tensor<1> D_e(quads);
		Tensor<dim> De(D_e,quads1d, quads1d, quads1d);
		for (int e = 0; e < nbElts; e++)
		{
			const Tensor<dim> T(U.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			Tensor<dim>       R(V.GetData() + e * dofs, dofs1d, dofs1d, dofs1d);
			D.evalD(e,D_e);
			contract(shape1d, T, BT);
			contract(shape1d, BT, BBT);
			contract(shape1d, BBT, BBBT);
			cWiseMult(De, BBBT, DBT);
			contractT(shape1d, DBT, BDBT);
			contractT(shape1d, BDBT, BBDBT);
			contractT(shape1d, BBDBT, BBBDBT);
			R += BBBDBT;
		}
	}
};


}

}

#endif

#endif