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

template <>
struct FESpaceType<Host>{
	typedef mfem::FiniteElementSpace& type;
};

class CudaFESpace
{
private:
	int nbElts;
public:
	CudaFESpace(mfem::FiniteElementSpace& fes)
	: nbElts(fes.GetNE())
	{ }

	int GetNE() { return nbElts; }
};

template <>
struct FESpaceType<CudaDevice>{
	// typedef CudaFESpace type;
	typedef mfem::FiniteElementSpace& type;
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
		locJ.createView(dim,dim);
		Tensor<1> J1d(J.getData(), J.length());
		EvalJacobians(dim, &fes, ir_order, J1d);	
	}

	const Tensor<2>& operator()(const int& e, const int& k) const {
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
		locJ.createView(dim,dim);
		//Needlessly expensive
		MeshJac<Host, false, true> Jac(fes,dim,quads,nbElts,ir_order);
		// Tensor<2> Je(J.getData(),dim,dim);
		for (int i = 0; i < nbElts; ++i)
		{
			locJ.setView(&J(0,0,i)) = Jac(i,0);
		}
	}

	const Tensor<2>& operator()(const int& e, const int& k) const {
		return locJ.setView(&J(0,0,e));
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

	__DEVICE__ const Tensor<2>& operator()(const int& e, const int& k) const {
		// return locJ.setView(&J(0,0,e));
	}	
};

//Only for CPU
struct QuadInfo
{
	int dim;
	int k;
	int e;
	ElementTransformation* tr;
	IntegrationPoint ip;
	Tensor<2>& J_ek;
};

//Might only be for CPU too
template <PAOp Op, Location Device, bool IsLinear=false, bool IsTimeConstant=true>
class Equation
{
public:
	typedef VectorType<Device,double> Vector;
	static const Location device = Device;
	static const PAOp OpName = Op;
protected:
	typedef typename TensorType<QuadDimVal<OpName>::value, Device>::type QuadTensor;
private:
	typedef typename FESpaceType<Device>::type FESpace;
	typedef typename TensorType<2, Device>::type JTensor;
	FESpace fes;
	const IntegrationRule& ir;//FIXME: not yet on GPU
	const IntegrationRule& ir1d;//FIXME: not yet on GPU
	MeshJac<Device,IsLinear,IsTimeConstant> Jac;
public:
	// Equation(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): fes(fes), dim(fes.GetFE(0)->GetDim()), ir(ir) {}
	Equation(mfem::FiniteElementSpace& fes, const int ir_order)
	: fes(fes)
	, ir(IntRules.Get(fes.GetFE(0)->GetGeomType(), ir_order))
	, ir1d(IntRules.Get(Geometry::SEGMENT, ir_order))
	, Jac(fes,fes.GetFE(0)->GetDim(),getNbQuads(),getNbElts(),ir_order) {}

	const FESpace& getTrialFESpace() const { return fes; }
	const FESpace& getTestFESpace() const { return fes; }
	const int getNbDofs1d() const { return fes.GetFE(0)->GetOrder() + 1;}
	const int getNbQuads() const { return ir.GetNPoints(); }
	const int getNbQuads1d() const { return ir1d.GetNPoints(); }
	const int getNbElts() const {return fes.GetNE(); }
	const JTensor& getJac(const int e, const int k) const { return Jac(e,k); }
	const IntegrationPoint& getIntPoint(const int k) const { return ir.IntPoint(k); }
	const IntegrationRule& getIntRule1d() const { return ir1d; }
	const int getDim() const { return fes.GetFE(0)->GetDim(); }

	// ElementInfo getQuadInfo(int e, int k) {
	// 	return {dim, k, e, fes.GetElementTransformation(e)};
	// }
};

class TestEq: public Equation<BtDB,Host>
{
public:
	TestEq() = delete;
	// TestEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	TestEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}

	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		D_ek = 1.0;
	}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {

	}
};

template <Location Device, typename CoeffStruct = Empty>
class PAMassEq;

template <>
class PAMassEq<Host, Empty>: public Equation<BtDB,Host>
{
public:
	PAMassEq() = delete;
	// HostMassEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
		D_ek = ip.weight * det(J_ek);
	}

	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		D_ek = info.ip.weight * det(info.J_ek);
	}
};

#ifdef __NVCC__
template <>
class PAMassEq<CudaDevice, Empty>: public Equation<BtDB,CudaDevice>
{
private:
	// ConstCoefficient coef;

public:
	PAMassEq() = delete;
	PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order): Equation(fes, ir_order) {}
	// PAMassEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	// __HOST__ PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order, mfem::ConstCoefficient& coeff)
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
class PAMassEq<Host, CoeffStruct>: public Equation<BtDB,Host>
{
private:
	CoeffStruct& coeff;
public:
	PAMassEq() = delete;
	// HostMassEq(mfem::FiniteElementSpace& fes, const IntegrationRule& ir): Equation(fes, ir) {}
	PAMassEq(mfem::FiniteElementSpace& fes, const int ir_order, CoeffStruct& coeff)
	: Equation(fes, ir_order)
	, coeff(coeff)
	{}

	void evalD(QuadTensor& D_ek, const int dim, const int k , const int e, ElementTransformation* Tr, const IntegrationPoint& ip, const Tensor<2>& J_ek) const {
		D_ek = ip.weight * det(J_ek) * coeff(dim,k,e,Tr,ip,J_ek);
	}

	void evalD(QuadTensor& D_ek, QuadInfo& info) const {
		D_ek = info.ip.weight * det(info.J_ek) * coeff(info);
	}
};

#ifdef __NVCC__
template <typename CoeffStruct>
class PAMassEq<CudaDevice, CoeffStruct>: public Equation<BtDB,CudaDevice>
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

	// __DEVICE__ void evalD(QuadTensor& D_ek, QuadInfo& info) const {
	// 	D_ek = info.ip.weight * det(info.J_ek) * coeff(info);
	// }
};
#endif

}

}

#endif

#endif