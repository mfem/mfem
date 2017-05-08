//                                MFEM Example Seg Fault
//
// Description:  This example code demonstrates a seg fault due to lack of 
//		 copy constructor in DenseMatrixEigensytem.  Also segfault
//		 on copy vector with size > 0 but null data.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
	DenseMatrix A(2,2);
	A(0, 0) = 1;
	A(0, 1) = -1;
	A(1, 0) = -1;
	A(1, 1) = 1;

	DenseMatrixEigensystem eig(A);
	DenseMatrixEigensystem eig2 = eig;

	eig.Eval();

	A(0,0) += 1;
	A(1,1) += 1;

	eig2.Eval();

	double e1 = eig.Eigenvalue(0);
	double e2 = eig2.Eigenvalue(0);

	printf("%.2f %.2f\n", e1, e2);
	//Seg fault on delete
}
