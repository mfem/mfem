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


#ifndef MFEM_BACKENDS_PA_EQUATION_HPP
#define MFEM_BACKENDS_PA_EQUATION_HPP

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "util.hpp"

namespace mfem
{

namespace pa
{

template <Location Device>
struct FESpaceType;

class HostFESpace
{
private:
	mfem::FiniteElementSpace& fes;
public:
	HostFESpace() = delete;
	HostFESpace(mfem::FiniteElementSpace& fes): fes(fes){}
	const int getNbDofs1d() const { return fes.GetFE(0)->GetOrder() + 1;}
	const int getNbElts() const { return fes.GetNE(); }
	const int getDim() const { return fes.GetFE(0)->GetDim(); }
	const Poly_1D::Basis& getBasis1d() const {
		const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fes.GetFE(0)));
		return tfe->GetBasis1D();
	}
};

template <>
struct FESpaceType<Host> {
	// typedef mfem::FiniteElementSpace& type;
	typedef HostFESpace type;
};

class CudaFESpace
{
private:
	const int nbDofs1d;
	const int nbElts;
	const int dim;
public:
	CudaFESpace(mfem::FiniteElementSpace& fes)
		: nbDofs1d(fes.GetFE(0)->GetOrder() + 1)
		, nbElts(fes.GetNE())
		, dim(fes.GetFE(0)->GetDim())
	{ }
	const int getNbDofs1d() const { return nbDofs1d;}
	const int getNbElts() const { return nbElts; }
	const int getDim() const { return dim; }
};

template <>
struct FESpaceType<CudaDevice> {
	typedef CudaFESpace type;
	// typedef mfem::FiniteElementSpace& type;
};

template <Location Device, bool IsLinear, bool IsTimeConstant>
class MeshJac;

template <>
class MeshJac<Host, false, true>
{
private:
	Tensor<4> J;
	mutable Tensor<2> locJ;
public:
	MeshJac(mfem::FiniteElementSpace& fes, const int dim, const int quads, const int nbElts, const int ir_order)
		: J(dim, dim, quads, nbElts), locJ() {
		locJ.createView(dim, dim);
		Tensor<1> J1d(J.getData(), J.length());
		EvalJacobians(dim, &fes, ir_order, J1d);
	}

	const TensorType<2,Host>& operator()(const int& e, const int& k) const {
		// static Tensor<2> J_ek(dim,dim);//workaround because 'mutable Tensor<2> locJ' does not compile
		// return J_ek.setView(&J(0, 0, k, e));
		return locJ.setView(&J(0, 0, k, e));
	}
};

template <>
class MeshJac<Host, true, true>
{
private:
	Tensor<3> J;
	mutable Tensor<2> locJ;
public:
	MeshJac(mfem::FiniteElementSpace& fes, const int dim, const int quads, const int nbElts, const int ir_order)
		: J(dim, dim, nbElts), locJ() {
		locJ.createView(dim, dim);
		//Needlessly expensive
		MeshJac<Host, false, true> Jac(fes, dim, quads, nbElts, ir_order);
		// Tensor<2> Je(J.getData(),dim,dim);
		for (int i = 0; i < nbElts; ++i)
		{
			locJ.setView(&J(0, 0, i)) = Jac(i, 0);
		}
	}

	const TensorType<2,Host>& operator()(const int& e, const int& k) const {
		return locJ.setView(&J(0, 0, e));
	}
};

template <>
class MeshJac<CudaDevice, false, true>
{
private:

public:
	__DEVICE__ MeshJac(mfem::FiniteElementSpace& fes, const int dim, const int quads, const int nbElts, const int ir_order)
	// : J(dim, dim, quads, nbElts), locJ()
	{
		// locJ.createView(dim,dim);
		// Tensor<1> J1d(J.getData(), J.length());
		// EvalJacobians(dim, &fes, ir_order, J1d);
	}

	__DEVICE__ const TensorType<2, CudaDevice>& operator()(const int& e, const int& k) const {
		// return locJ.setView(&J(0,0,e));
	}
};

//Might only be for CPU too
template <PAOp Op, Location Device, bool IsLinear = false, bool IsTimeConstant = true>
class Equation
{
public:
	typedef VectorType<Device, double> Vector;
	static const Location device = Device;
	static const PAOp OpName = Op;
protected:
	typedef typename TensorType_t<QuadDimVal<OpName>::value, Device>::type QuadTensor;
private:
	typedef typename FESpaceType<Device>::type FESpace;
	typedef typename TensorType_t<2, Device>::type JTensor;
	FESpace fes;
	const IntegrationRule& ir;//FIXME: not yet on GPU
	const IntegrationRule& ir1d;//FIXME: not yet on GPU
	MeshJac<Device, IsLinear, IsTimeConstant> Jac;
public:
	Equation(mfem::FiniteElementSpace& fes, const int ir_order)
		: fes(fes)
		, ir(IntRules.Get(fes.GetFE(0)->GetGeomType(), ir_order))
		, ir1d(IntRules.Get(Geometry::SEGMENT, ir_order))
		, Jac(fes, fes.GetFE(0)->GetDim(), getNbQuads(), getNbElts(), ir_order) {}

	__HOST__ __DEVICE__ const FESpace& getTrialFESpace() const { return fes; }
	__HOST__ __DEVICE__ const FESpace& getTestFESpace() const { return fes; }
	__HOST__ __DEVICE__ const int getNbDofs1d() const { return fes.getNbDofs1d();}
	__HOST__ __DEVICE__ const int getNbQuads() const { return ir.GetNPoints(); }
	__HOST__ __DEVICE__ const int getNbQuads1d() const { return ir1d.GetNPoints(); }
	__HOST__ __DEVICE__ const int getNbElts() const {return fes.getNbElts(); }
	__HOST__ __DEVICE__ const JTensor& getJac(const int e, const int k) const { return Jac(e, k); }
	__HOST__ __DEVICE__ const IntegrationPoint& getIntPoint(const int k) const { return ir.IntPoint(k); }
	__HOST__ __DEVICE__ const IntegrationRule& getIntRule1d() const { return ir1d; }
	__HOST__ __DEVICE__ const int getDim() const { return fes.getDim(); }
};

/**
*
*	MASS EQUATION
*
*/
template <Location Device, typename CoeffStruct = Empty>
class PAMassEq;

template <>
class PAMassEq<Host, Empty>: public Equation<BtDB, Host>
{
public:
	PAMassEq() = delete;
	PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
		D_ek = ip.weight * det(J_ek);
	}

	template <typename QuadInfo>
	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		D_ek = info.ip.weight * det(info.J_ek);
	}
};

#ifdef __NVCC__
template <>
class PAMassEq<CudaDevice, Empty>: public Equation<BtDB, CudaDevice>
{
private:
	// ConstCoefficient coef;

public:
	PAMassEq() = delete;
	PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}

	template <typename QuadInfo>
	__DEVICE__ void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		*D_ek = 1.0;//info.ip.weight * det(info.J_ek) * coeff(info);
	}
};
#endif

//Shoudl accept any coefficient that takes @info as argument and returns a Scalar
template <typename CoeffStruct>
class PAMassEq<Host, CoeffStruct>: public Equation<BtDB, Host>
{
private:
	CoeffStruct& coeff;
public:
	PAMassEq() = delete;
	PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order, CoeffStruct& coeff)
		: Equation(fes, ir_order)
		, coeff(coeff)
	{}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
		D_ek = ip.weight * det(J_ek) * coeff(dim, k, e, Tr, ip, J_ek);
	}

	template <typename QuadInfo>
	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		D_ek = info.ip.weight * det(info.J_ek) * coeff(info);
	}
};

#ifdef __NVCC__
template <typename CoeffStruct>
class PAMassEq<CudaDevice, CoeffStruct>: public Equation<BtDB, CudaDevice>
{
private:
	CoeffStruct coeff;
public:
	PAMassEq() = delete;
	PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order, CoeffStruct& coeff)
		: Equation(fes, ir_order)
		, coeff(coeff)
	{}
	// PAMassEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	// __HOST__ PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order, mfem::ConstCoefficient& coeff)
	// : Equation(fes, ir_order), coeff(coeff) {}

	// void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
	// 	D_ek = ip.weight * det(J_ek);
	// }

	template <typename QuadInfo>
	__DEVICE__ void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		*D_ek = 1.0;//info.ip.weight * det(info.J_ek) * coeff(info);
	}
};
#endif

/**
*
*	DIFFUSION EQUATION
*
*/
template <Location Device, typename CoeffStruct = Empty>
class PADiffusionEq;

template <>
class PADiffusionEq<Host, Empty>: public Equation<GtDG, Host>
{
public:
	PADiffusionEq() = delete;
	// HostMassEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	PADiffusionEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e,
	           ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
		//TODO add assert on size of D_ek (dim,dim)
		Tensor<2> Adj(dim, dim);
		adjugate(J_ek, Adj);
		double val = 0.0;
		double qval = 1.0;
		double detJ = det(J_ek);
		qval = 1.0;//args.q.Eval(*Tr, ip);
		for (int i = 0; i < dim; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				val = 0.0;
				for (int k = 0; k < dim; ++k)
				{
					val += Adj(i, k) * Adj(j, k); //Adj*Adj^T
					// val += Adj(k,i)*Adj(k,j); //Adj*Adj^T
				}
				D_ek(i, j) = ip.weight * qval / detJ * val;
			}
		}
	}

	template <typename QuadInfo>
	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		//TODO add assert on size of D_ek (dim,dim)
		const int dim = this->getDim();
		Tensor<2> Adj(dim, dim);
		adjugate(info.J_ek, Adj);
		double val = 0.0;
		double qval = 1.0;
		double detJ = det(info.J_ek);
		qval = 1.0;//args.q.Eval(*Tr, ip);
		for (int i = 0; i < dim; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				val = 0.0;
				for (int k = 0; k < dim; ++k)
				{
					val += Adj(i, k) * Adj(j, k); //Adj*Adj^T
					// val += Adj(k,i)*Adj(k,j); //Adj*Adj^T
				}
				D_ek(i, j) = info.ip.weight * qval / detJ * val;
			}
		}
	}
};

#ifdef __NVCC__
template <>
class PADiffusionEq<CudaDevice, Empty>: public Equation<GtDG, CudaDevice>
{
private:
	// ConstCoefficient coef;

public:
	PADiffusionEq() = delete;
	PADiffusionEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}
	// PADiffusionEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	// __HOST__ PADiffusionEq(mfem::FiniteElementSpace& fes, const int ir_order, mfem::ConstCoefficient& coeff)
	// : Equation(fes, ir_order), coeff(coeff) {}

	// void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
	// 	D_ek = ip.weight * det(J_ek);
	// }

	// __DEVICE__ void evalD(QuadTensor& D_ek, QuadInfo& info) const {
	// 	D_ek = info.ip.weight * det(info.J_ek) * coeff(info);
	// }
};
#endif

//Shoudl accept any coefficient that takes @info as argument and returns a Scalar
template <typename CoeffStruct>
class PADiffusionEq<Host, CoeffStruct>: public Equation<GtDG, Host>
{
private:
	CoeffStruct coeff;
public:
	PADiffusionEq() = delete;
	// HostMassEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	PADiffusionEq(mfem::FiniteElementSpace& fes, const int ir_order, CoeffStruct& coeff)
		: Equation(fes, ir_order)
		, coeff(coeff)
	{}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e,
	           ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
		//TODO add assert on size of D_ek (dim,dim)
		Tensor<2> Adj(dim, dim);
		adjugate(J_ek, Adj);
		double val = 0.0;
		double qval = 1.0;
		double detJ = det(J_ek);
		qval = coeff(dim, k, e, Tr, ip, J_ek); //1.0;//args.q.Eval(*Tr, ip);
		for (int i = 0; i < dim; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				val = 0.0;
				for (int k = 0; k < dim; ++k)
				{
					val += Adj(i, k) * Adj(j, k); //Adj*Adj^T
					// val += Adj(k,i)*Adj(k,j); //Adj*Adj^T
				}
				D_ek(i, j) = ip.weight * qval / detJ * val;
			}
		}
	}

	template <typename QuadInfo>
	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		//TODO add assert on size of D_ek (dim,dim)
		const int dim = this->getDim();
		Tensor<2> Adj(dim, dim);
		adjugate(info.J_ek, Adj);
		double val = 0.0;
		double qval = 1.0;
		double detJ = det(info.J_ek);
		qval = coeff(info);//1.0;//args.q.Eval(*Tr, ip);
		for (int i = 0; i < dim; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				val = 0.0;
				for (int k = 0; k < dim; ++k)
				{
					val += Adj(i, k) * Adj(j, k); //Adj*Adj^T
					// val += Adj(k,i)*Adj(k,j); //Adj*Adj^T
				}
				D_ek(i, j) = info.ip.weight * qval / detJ * val;
			}
		}
	}
};

#ifdef __NVCC__
template <typename CoeffStruct>
class PADiffusionEq<CudaDevice, CoeffStruct>: public Equation<GtDG, CudaDevice>
{
private:
	CoeffStruct coeff;
public:
	PADiffusionEq() = delete;
	PADiffusionEq(mfem::FiniteElementSpace& fes, const int ir_order, CoeffStruct& coeff)
		: Equation(fes, ir_order)
		, coeff(coeff)
	{}
	// PADiffusionEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	// __HOST__ PADiffusionEq(mfem::FiniteElementSpace& fes, const int ir_order, mfem::ConstCoefficient& coeff)
	// : Equation(fes, ir_order), coeff(coeff) {}

	// void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
	// 	D_ek = ip.weight * det(J_ek);
	// }

	// __DEVICE__ void evalD(QuadTensor& D_ek, QuadInfo& info) const {
	// 	D_ek = info.ip.weight * det(info.J_ek) * coeff(info);
	// }
	template <typename QuadInfo>
	__DEVICE__ void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		*D_ek = 1.0;//info.ip.weight * det(info.J_ek) * coeff(info);
	}
};
#endif

}

}

#endif

#endif