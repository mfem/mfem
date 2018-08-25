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


#ifndef MFEM_BACKENDS_PA_DOMAINKERNEL_HPP
#define MFEM_BACKENDS_PA_DOMAINKERNEL_HPP

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "util.hpp"
#include "hostdomainkernel.hpp"
#include "cudadomainkernel.hpp"
#include "tensor.hpp"
#include "vector.hpp"

namespace mfem
{

namespace pa
{

template <PAOp OpName>
struct QuadDimVal{
	static const int value = 1;
};

template <>
struct QuadDimVal<BtDB>{
	static const int value = 0;
};

template <>
struct QuadDimVal<GtDG>{
	static const int value = 2;
};

template <typename Equation>
struct QuadDim{
	static const int value = QuadDimVal<Equation::OpName>::value;
};

template <typename Equation>
struct EltDim{
	static const int value = QuadDim<Equation>::value + 1;
};

template <typename Equation>
struct TensorDim{
	static const int value = EltDim<Equation>::value + 1;
};

template <int Dim, Location Device>
struct TensorType;

template <int Dim>
struct TensorType<Dim,Host>{
	typedef Tensor<Dim,double> type;
};

template <int Dim>
struct TensorType<Dim,CudaDevice>{
	typedef double* type;
};

// template <>
// struct TensorType<0,HostVector<double>>{
// 	typedef double type;
// };

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

template <Location Device>
class PAMassEq;

template <>
class PAMassEq<Host>: public Equation<BtDB,Host>
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
class PAMassEq<CudaDevice>: public Equation<BtDB,CudaDevice>
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

template <typename Equation, Location Device>
class QuadTensorFunc
{
private:
	typedef typename FESpaceType<Device>::type FESpace;
	typedef typename TensorType<EltDim<Equation>::value, Device>::type Tensor; //Defines the Host/Device Tensor type for quadrature data
	typedef typename TensorType<QuadDim<Equation>::value, Device>::type QuadTensor;
	typedef typename TensorType<2, Device>::type JTensor;
	Equation* eq;//Maybe Equation needs to be moved on GPU Device<Equation,Vector>& eq? otherwise Vector could be deduced
	mutable QuadTensor D_ek;
public:
	QuadTensorFunc() = delete;
	~QuadTensorFunc(){
		delete eq;
	}
	QuadTensorFunc(Equation* eq): eq(eq), D_ek() {  }

	const FESpace& getTrialFESpace() const { return eq->getTrialFESpace(); }
	const FESpace& getTestFESpace() const { return eq->getTestFESpace(); }

	//This will not work on GPU
	__HOST__ void evalD(const int e, Tensor& D_e) const {
		ElementTransformation *Tr = eq->getTrialFESpace().GetElementTransformation(e);
		for (int k = 0; k < eq->getNbQuads(); ++k)
		{
			D_ek.slice(D_e, k);
			const JTensor& J_ek = eq->getJac(e, k);
			const IntegrationPoint &ip = eq->getIntPoint(k);
			Tr->SetIntPoint(&ip);
			eq->evalD(D_ek, eq->getDim(), k, e, Tr, ip, J_ek);
		}
	}

	#ifdef __NVCC__
	__DEVICE__ void evalD(const int e, Tensor& D_e){
		//TODO
	}
	#endif
};
//Should specialize GPU version

template <typename Equation, Location Device>
struct DomainKernel;

template <typename Equation>
struct DomainKernel<Equation,Host>
{
	typedef HostDomainKernel<Equation::OpName> type;
};

template <typename Equation>
struct DomainKernel<Equation,CudaDevice>
{
	typedef CudaDomainKernel<Equation::OpName> type;
};

/**
*	Can be used as a Matrix Free operator
*/
template <typename Equation, Location Device = Equation::device>
class PADomainIntegrator: public PAIntegrator<Device>
{
private:
	typedef VectorType<Device,double> Vector;
	typedef typename TensorType<TensorDim<Equation>::value, Device>::type Tensor;
	typedef typename TensorType<EltDim<Equation>::value, Device>::type EltTensor;
	typedef QuadTensorFunc<Equation, Device> QFunc;
	typedef typename DomainKernel<Equation, Device>::type Kernel;
	Kernel kernel;
	QFunc qfunc;//has to be pointer for GPU
	mutable EltTensor D_e;
public:
	PADomainIntegrator() = delete;
	PADomainIntegrator(Equation* eq): kernel(eq), qfunc(eq), D_e() { }

	void evalD(Tensor& D) const {
		kernel.evalD(qfunc, D);
	}	

	// __HOST__ void evalD(Tensor& D) const {
	// 	for (int e = 0; e < qfunc.getTrialFESpace().GetNE(); ++e)
	// 	{
	// 		D_e.slice(D, e);
	// 		qfunc.evalD(e, D_e);
	// 	}
	// }

	// #ifdef __NVCC
	// __DEVICE__ void evalD(Tensor& D) const {
	// 	//TODO
	// }
	// #endif

	virtual void Mult(const Vector& x, Vector& y) const {
		kernel.Mult(qfunc, x, y);
	}

	virtual void MultAdd(const Vector& x, Vector& y) const {
		kernel.MultAdd(qfunc, x, y);
	}
};

template <typename Equation>
PADomainIntegrator<Equation>* createMFDomainKernel(Equation* eq){ return new PADomainIntegrator<Equation>(eq); }

static void initFESpaceTensor(const int dim, const int nbQuads, const int nbElts, Tensor<2>& D){
	D.setSize(nbQuads,nbElts);
}
static void initFESpaceTensor(const int dim, const int nbQuads, const int nbElts, Tensor<3>& D){
	D.setSize(dim,nbQuads,nbElts);
}
static void initFESpaceTensor(const int dim, const int nbQuads, const int nbElts, Tensor<4>& D){
	D.setSize(dim,dim,nbQuads,nbElts);
}

//FOR GPU
static void initFESpaceTensor(const int dim, const int nbQuads, const int nbElts, double* D){
	
}

/**
*	Partial Assembly operator
*/
template <typename Equation, Location Device = Equation::device>
class PADomainKernel: public PAIntegrator<Device>
{
private:
	typedef VectorType<Device,double> Vector;
	typedef typename TensorType<TensorDim<Equation>::value, Device>::type Tensor;
	typedef typename DomainKernel<Equation, Device>::type Kernel;
	typedef typename FESpaceType<Device>::type FESpace;
	Kernel kernel;
	Tensor D;//has to be a pointer for GPU
	// const FESpace& trial_fes, test_fes;
public:
	PADomainKernel() = delete;
	//This should work on GPU __HOST__ __DEVICE__
	__HOST__ __DEVICE__ PADomainKernel(Equation* eq): kernel(eq), D() {
		const int dim = eq->getDim();
		const int nbQuads = eq->getNbQuads();
		const int nbElts = eq->getNbElts();
		initFESpaceTensor(dim, nbQuads, nbElts, D);
		PADomainIntegrator<Equation, Device> integ(eq);
		integ.evalD(D);
	}

	virtual void Mult(const Vector& x, Vector& y) const {
		kernel.Mult(D, x, y);
	}

	virtual void MultAdd(const Vector& x, Vector& y) const {
		kernel.MultAdd(D, x, y);
	}
};


template <typename Equation>
PADomainKernel<Equation>* createPADomainKernel(Equation* eq){ return new PADomainKernel<Equation>(eq); }

template <Location Device>
class MatrixType;

template <>
class MatrixType<Host>{
public:
	typedef mfem::Array<Tensor<2,double>> type;
};

/**
*	Local Matrices operator
*/
template <typename Equation, Location Device>
class LocMatKernel: public PAIntegrator<Device>
{
private:
	typedef VectorType<Device,double> Vector;
	typedef typename MatrixType<Device>::type MatrixSet;
	MatrixSet A;//We should find a way to accumulate those...
public:
	LocMatKernel() = delete;
	LocMatKernel(Equation& eq) {
		PADomainIntegrator<Equation, Device> qfunc(eq);
		auto trial_fes = qfunc.getTrialFES();
		auto test_fes = qfunc.getTestFES();
		//TODO
		// A = QuadBasisOut<Equation::OpOut>(test_fes) * qfunc * QuadBasisIn<Equation::OpIn>(trial_fes);
	}

	virtual void Mult(const Vector& x, Vector& y) const {
		// y = A * x;
		A.Mult(x, y);
	}

	virtual void MultAdd(const Vector& x, Vector& y) const {
		// y += A * x;
		A.MultAdd(x, y);
	}
};

template <Location Device>
class SpMatrixType;

template <>
class SpMatrixType<Host>{
public:
	typedef Tensor<2,double> type;
};

/**
*	Sparse Matrix operator
*/
template <typename Equation, Location Device>
class SpMatKernel: public PAIntegrator<Device>
{
private:
	typedef VectorType<Device,double> Vector;
	typedef typename SpMatrixType<Device>::type SpMat;
	SpMat A;
public:
	SpMatKernel() = delete;
	SpMatKernel(Equation& eq) {
		LocMatKernel<Equation, Device> locA(eq);
		A = locA;
	}

	virtual void Mult(const Vector& x, Vector& y) const {
		// y = A * x;
		A.Mult(x, y);
	}

	virtual void MultAdd(const Vector& x, Vector& y) const {
		// y += A * x;
		A.MultAdd(x, y);
	}
};

}

}

#endif

#endif