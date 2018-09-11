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
#include "equation.hpp"
#include "hostdomainkernel.hpp"
#include "cudadomainkernel.hpp"
#include "tensor.hpp"
#include "vector.hpp"

namespace mfem
{

namespace pa
{

//Should specialize GPU version
/**
*  A class to represent a quadrature point function for domain integrals (i.e. qFunction in libCEED)
*/
template <typename Equation, Location Device>
class QuadTensorFunc;

struct HostQuadInfo
{
	const int dim;
	const int k;
	const int e;
	// const ElementTransformation* Tr;
	const IntegrationPoint& ip;
	const TensorType<2, Host>& J_ek;
};

template <typename Equation>
class QuadTensorFunc<Equation, Host>
{
private:
	typedef typename FESpaceType<Host>::type FESpace;
	typedef typename TensorType_t<EltDim<Equation>::value, Host>::type Tensor; //Defines the Host/Device Tensor type for quadrature data
	typedef typename TensorType_t<QuadDim<Equation>::value, Host>::type QuadTensor;
	typedef typename TensorType_t<2, Host>::type JTensor;
	Equation* eq;//Maybe Equation needs to be moved on GPU Device<Equation,Vector>& eq? otherwise Vector could be deduced
	mutable QuadTensor D_ek;
public:
	QuadTensorFunc() = delete;
	~QuadTensorFunc(){
		delete eq;//Not sure about this
	}
	QuadTensorFunc(Equation* eq): eq(eq), D_ek() {  }

	const FESpace& getTrialFESpace() const { return eq->getTrialFESpace(); }
	const FESpace& getTestFESpace() const { return eq->getTestFESpace(); }

	void evalD(const int e, Tensor& D_e) const {
		// ElementTransformation *Tr = eq->getTrialFESpace().GetElementTransformation(e);
		for (int k = 0; k < eq->getNbQuads(); ++k)
		{
			D_ek.slice(D_e, k);
			const JTensor& J_ek = eq->getJac(e, k);
			const IntegrationPoint &ip = eq->getIntPoint(k);
			// Tr->SetIntPoint(&ip);//Should disappear
			// eq->evalD(D_ek, eq->getDim(), k, e, Tr, ip, J_ek);
			HostQuadInfo info = {eq->getDim(), k, e, ip, J_ek};
			eq->evalD(D_ek, info);
		}
	}
};

#ifdef __NVCC__
struct CudaQuadInfo
{
	
};

template <typename Equation>
class QuadTensorFunc<Equation, CudaDevice>
{
private:
	typedef typename FESpaceType<CudaDevice>::type FESpace;
	typedef typename TensorType_t<EltDim<Equation>::value, CudaDevice>::type Tensor; //Defines the Host/Device Tensor type for quadrature data
	typedef typename TensorType_t<QuadDim<Equation>::value, CudaDevice>::type QuadTensor;
	typedef typename TensorType_t<2, CudaDevice>::type JTensor;
	Equation* eq;//Maybe Equation needs to be moved on GPU Device<Equation,Vector>& eq? otherwise Vector could be deduced
	mutable QuadTensor D_ek;
public:
	QuadTensorFunc() = delete;
	~QuadTensorFunc(){
		delete eq;//Not sure about this
	}
	__HOST__ __DEVICE__ QuadTensorFunc(Equation* eq): eq(eq), D_ek() {  }

	__HOST__ __DEVICE__ const FESpace& getTrialFESpace() const { return eq->getTrialFESpace(); }
	__HOST__ __DEVICE__ const FESpace& getTestFESpace() const { return eq->getTestFESpace(); }

	__DEVICE__ void evalD(const int e, Tensor& D_e) const{
		//TODO
		int k = threadIdx.x;//we could go hemi style for generic programming
		// D_ek.slice(D_e, k);
		D_ek = D_e + k;
		// const JTensor& J_ek = eq->getJac(e, k);
		// const IntegrationPoint &ip = eq->getIntPoint(k);
		// eq->evalD(D_ek, eq->getDim(), k, e, Tr, ip, J_ek);
		CudaQuadInfo info;
		eq->evalD(D_ek, info);
	}
};
#endif

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
	typedef typename TensorType_t<TensorDim<Equation>::value, Device>::type Tensor;
	typedef typename TensorType_t<EltDim<Equation>::value, Device>::type EltTensor;
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
*	Partial Assembly operator using a qFunction to construct
*/
template <typename Equation, Location Device = Equation::device>
class PADomainKernel: public PAIntegrator<Device>
{
private:
	typedef VectorType<Device,double> Vector;
	typedef typename TensorType_t<TensorDim<Equation>::value, Device>::type Tensor;
	typedef typename DomainKernel<Equation, Device>::type Kernel;
	typedef typename FESpaceType<Device>::type FESpace;
	Kernel kernel;
	Tensor D;//has to be a pointer for GPU
	// const FESpace& trial_fes, test_fes;
public:
	PADomainKernel() = delete;
	//This should work on GPU __HOST__ __DEVICE__
	PADomainKernel(Equation* eq): kernel(eq), D() {
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