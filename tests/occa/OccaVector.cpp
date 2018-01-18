#include "mfem.hpp"
#include "occa.hpp"

void testVector();

int main(const int argc, const char **argv) {
  if (argc > 1) {
    occa::setDevice(argv[1]);
  }
  occa::settings()["verboseCompilation"] = true;
  testVector();
  return 0;
}

bool doublesAreEqualish(const double a, const double b) {
  return (((a - 1e-10) < b) && (b < (a + 1e-10)));
}

void compareVectorAndPtr(const std::string &message,
                         mfem::OccaVector &v, double *array) {

  occa::finish();
  double *tmp = new double[v.Size()];
  occa::memcpy(tmp, v.GetData(), v.Size() * sizeof(double));
  for (int i = 0; i < v.Size(); ++i) {
    OCCA_ERROR(message << '\n'
               << array[i] << " != " << tmp[i] << '\n',
               doublesAreEqualish(array[i], tmp[i]));
  }
  delete [] tmp;
}

void testVectorConstructors() {
  mfem::OccaVector v1;
  OCCA_ERROR("OccaVector::OccaVector()",
             v1.Size() == 0);

  mfem::OccaVector v2(10);
  OCCA_ERROR("OccaVector::OccaVector(const int64_t size)",
             v2.Size() == 10);
  OCCA_ERROR("OccaVector::OccaVector(const int64_t size)",
             v2.GetData().size() == (10 * sizeof(double)));

  mfem::OccaVector v3(v2);
  OCCA_ERROR("OccaVector::OccaVector(const OccaVector &other)",
             v3.Size() == v2.Size());
  OCCA_ERROR("OccaVector::OccaVector(const OccaVector &other)",
             v3.GetData().size() == v2.GetData().size());
  OCCA_ERROR("OccaVector::OccaVector(const OccaVector &other)",
             v3.GetData().ptr() != v2.GetData().ptr());

  mfem::Vector v4_(3);
  v4_(0) = 1; v4_(1) = 2; v4_(2) = 3;
  mfem::OccaVector v4(v4_);
  compareVectorAndPtr("OccaVector::OccaVector(const Vector &v)",
                      v4, v4_.GetData());
}

void testVectorResize() {
  mfem::OccaVector v1(10);
  v1.SetSize(20);
  OCCA_ERROR("void OccaVector::SetSize(const int64_t size)",
             v1.Size() == 20);

  double arr1[3] = { 3, 2, 1 };
  double arr2[3] = { 1, 2, 3 };
  double *tmp = new double[3];

  v1.SetSize(3, arr1);
  compareVectorAndPtr("void OccaVector::SetSize(const int64_t size, const void *src)",
                      v1, arr1);

  v1.SetSize(3, arr2);
  compareVectorAndPtr("void OccaVector::SetSize(const int64_t size, const void *src)",
                      v1, arr2);

  OCCA_ERROR("uint64_t OccaVector::Capacity()",
             v1.Capacity() == 20);
}

void testVectorAssignmentOperators() {
  mfem::Vector v1_(3);
  v1_(0) = 1; v1_(1) = 2; v1_(2) = 3;
  mfem::OccaVector v1 = v1_;

  OCCA_ERROR("double OccaVector::operator * (const OccaVector &v) const",
             doublesAreEqualish((v1 * v1), (1*1 + 2*2 + 3*3)));

  mfem::Vector v2_(3);
  v2_(0) = 4; v2_(1) = 5; v2_(2) = 6;
  mfem::OccaVector v2 = v2_;
  v1 = v2;
  compareVectorAndPtr("OccaVector& OccaVector::operator = (const OccaVector &v)",
                      v1, v2_.GetData());

  v1 = v2_;
  v1_(0) = 0; v1_(1) = 0; v1_(2) = 0;
  v1_ = v1;
  compareVectorAndPtr("Vector& Vector::operator = (const OccaVector &ov)",
                      v1, v1_.GetData());

  v1 *= 2;
  v1_(0) = 8; v1_(1) = 10; v1_(2) = 12;
  compareVectorAndPtr("OccaVector& OccaVector::operator *= (double value)",
                      v1, v1_.GetData());

  v1 /= 4;
  v1_(0) = 2; v1_(1) = 2.5; v1_(2) = 3;
  compareVectorAndPtr("OccaVector& OccaVector::operator /= (double value)",
                      v1, v1_.GetData());

  v1 -= 2;
  v1_(0) = 0; v1_(1) = 0.5; v1_(2) = 1;
  compareVectorAndPtr("OccaVector& OccaVector::operator -= (double value)",
                      v1, v1_.GetData());

  v1 += 2;
  v1_(0) = 2; v1_(1) = 2.5; v1_(2) = 3;
  compareVectorAndPtr("OccaVector& OccaVector::operator += (double value)",
                      v1, v1_.GetData());

  v1 -= v1;
  v1_(0) = 0; v1_(1) = 0; v1_(2) = 0;
  compareVectorAndPtr("OccaVector& OccaVector::operator -= (const OccaVector &v)",
                      v1, v1_.GetData());

  v1 += v2;
  compareVectorAndPtr("OccaVector& OccaVector::operator += (const OccaVector &v)",
                      v1, v2_.GetData());

  v1.Add(5, v2);
  v1_ = v2_;
  v1_(0) += 5 * v1_(0); v1_(1) += 5 * v1_(1); v1_(2) += 5 * v1_(2);
  compareVectorAndPtr("OccaVector& OccaVector::Add(const double a, const OccaVector &Va)",
                      v1, v1_.GetData());

  v1.Set(-1, v2);
  v1_(0) = -v2_(0); v1_(1) = -v2_(1); v1_(2) = -v2_(2);
  compareVectorAndPtr("OccaVector& OccaVector::Set(const double a, const OccaVector &Va)",
                      v1, v1_.GetData());

  v1.Neg();
  v1_(0) = -v1_(0); v1_(1) = -v1_(1); v1_(2) = -v1_(2);
  compareVectorAndPtr("void OccaVector::Neg()",
                      v1, v1_.GetData());

  mfem::Vector v3_(3);
  mfem::OccaVector v3;
  v1_(0) = 2; v1_(1) = 0; v1_(2) = -2;
  v2_(0) = 1; v2_(1) = 1; v2_(2) = 1;
  v3_(0) = 3; v3_(1) = 2; v3_(2) = -1;

  v1 = v1_;
  v2 = v2_;
  v3 = v3_;

  v2.median(v1, v3);
  v2_(0) = 2; v2_(1) = 1; v2_(2) = -1;
  compareVectorAndPtr("void OccaVector::median(const OccaVector &lo, const OccaVector &hi)",
                      v2, v2_.GetData());
}

void testVectorSubVectorMethods() {
  mfem::OccaVector v1, v2;
  mfem::Vector v1_(6), v2_(3), zero_(3);
  mfem::Array<int> indices(3);

  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  zero_(0) = 0; zero_(1) = 0; zero_(2) = 0;
  v1 = v1_;
  v2 = zero_;

  indices[0] = 3;
  indices[1] = -2;
  indices[2] = 5;

  v1.GetSubVector(indices, v2);
  v2_(0) = 3; v2_(1) = -1; v2_(2) = 5;
  compareVectorAndPtr("void OccaVector::GetSubVector("
                      "const Array<int> &dofs, "
                      "OccaVector &elemvect) const;",
                      v2, v2_.GetData());

  v2_ = zero_;
  v1.GetSubVector(indices, v2_);
  compareVectorAndPtr("void OccaVector::GetSubVector("
                      "const Array<int> &dofs, "
                      "Vector &elemvect) const;",
                      v2, v2_.GetData());

  v2_ = zero_;
  v1.GetSubVector(indices, v2_.GetData());
  compareVectorAndPtr("void OccaVector::GetSubVector("
                      "const Array<int> &dofs, "
                      "double *elem_data) const;",
                      v2, v2_.GetData());

  v1_(0) = 0; v1_(1) = 9; v1_(2) = 2;
  v1_(3) = 9; v1_(4) = 4; v1_(5) = 9;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;

  v2_(0) = 3; v2_(1) = -1; v2_(2) = 5;
  v2 = v2_;
  v1.SetSubVector(indices, v2);
  compareVectorAndPtr("void OccaVector::SetSubVector("
                      "const Array<int> &dofs, "
                      "const OccaVector &elemvect);",
                      v1, v1_.GetData());

  v1_(0) = 0; v1_(1) = 9; v1_(2) = 2;
  v1_(3) = 9; v1_(4) = 4; v1_(5) = 9;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1.SetSubVector(indices, v2_);
  compareVectorAndPtr("void OccaVector::SetSubVector("
                      "const Array<int> &dofs, "
                      "const Vector &elemvect);",
                      v1, v1_.GetData());

  v1_(0) = 0; v1_(1) = 9; v1_(2) = 2;
  v1_(3) = 9; v1_(4) = 4; v1_(5) = 9;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1.SetSubVector(indices, v2_.GetData());
  compareVectorAndPtr("void OccaVector::SetSubVector("
                      "const Array<int> &dofs, "
                      "double *elem_data);",
                      v1, v1_.GetData());

  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = -9; v1_(2) = 2;
  v1_(3) = 9; v1_(4) = 4; v1_(5) = 9;
  v1.SetSubVector(indices, 9);
  compareVectorAndPtr("void OccaVector::SetSubVector("
                      "const Array<int> &dofs, "
                      "const double value);",
                      v1, v1_.GetData());

  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1 = v1_;
  v1_(0) = 9; v1_(1) = 1; v1_(2) = 9;
  v1_(3) = 3; v1_(4) = 9; v1_(5) = 5;
  v1.SetSubVectorComplement(indices, 9);
  compareVectorAndPtr("void OccaVector::SetSubVectorComplement("
                      "const Array<int> &dofs, "
                      "const double val);",
                      v1, v1_.GetData());
}

void testVectorAddElementVectorMethods() {
  mfem::OccaVector v1, v2;
  mfem::Vector v1_(6), v2_(3), zero_(3);
  mfem::Array<int> indices(3);

  v2_(0) = 3; v2_(1) = -1; v2_(2) = 5;
  v2 = v2_;

  indices[0] = 3;
  indices[1] = -2;
  indices[2] = 5;

  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 2; v1_(2) = 2;
  v1_(3) = 6; v1_(4) = 4; v1_(5) = 10;
  v1.AddElementVector(indices, v2);
  compareVectorAndPtr("void OccaVector::AddElementVector("
                      "const Array<int> &dofs, "
                      "const OccaVector &elemvect);",
                      v1, v1_.GetData());

  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 2; v1_(2) = 2;
  v1_(3) = 6; v1_(4) = 4; v1_(5) = 10;
  v1.AddElementVector(indices, v2_);
  compareVectorAndPtr("void OccaVector::AddElementVector("
                      "const Array<int> &dofs, "
                      "const Vector &elemvect);",
                      v1, v1_.GetData());

  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 2; v1_(2) = 2;
  v1_(3) = 6; v1_(4) = 4; v1_(5) = 10;
  v1.AddElementVector(indices, v2_.GetData());
  compareVectorAndPtr("void OccaVector::AddElementVector("
                      "const Array<int> &dofs, "
                      "double *elem_data);",
                      v1, v1_.GetData());


  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 0; v1_(2) = 2;
  v1_(3) = 0; v1_(4) = 4; v1_(5) = 0;
  v1.AddElementVector(indices, -1, v2);
  compareVectorAndPtr("void OccaVector::AddElementVector("
                      "const Array<int> &dofs, "
                      "const double a, "
                      "const OccaVector &elemvect);",
                      v1, v1_.GetData());

  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 2; v1_(2) = 2;
  v1_(3) = 6; v1_(4) = 4; v1_(5) = 10;
  v1.AddElementVector(indices, 1, v2_);
  compareVectorAndPtr("void OccaVector::AddElementVector("
                      "const Array<int> &dofs, "
                      "const double a, "
                      "const Vector &elemvect);",
                      v1, v1_.GetData());

  v1_(0) = 0; v1_(1) = 1; v1_(2) = 2;
  v1_(3) = 3; v1_(4) = 4; v1_(5) = 5;
  v1 = v1_;
  v1_(0) = 0; v1_(1) = 3; v1_(2) = 2;
  v1_(3) = 9; v1_(4) = 4; v1_(5) = 15;
  v1.AddElementVector(indices, 2, v2_.GetData());
  compareVectorAndPtr("void OccaVector::AddElementVector("
                      "const Array<int> &dofs, "
                      "const double a, "
                      "double *elem_data);",
                      v1, v1_.GetData());
}

void testVectorLinalgMethods() {
  mfem::OccaVector v1, v2;
  mfem::Vector v1_(3), v2_(3);

  v1_(0) = 1; v1_(1) = -2; v1_(2) = 3;
  v2_(0) = 4; v2_(1) = 4; v2_(2) = 4;
  v1 = v1_;
  v2 = v2_;

  OCCA_ERROR("double OccaVector::Norml2() const",
             doublesAreEqualish(v1.Norml2(),
                                sqrt(1*1 + 2*2 + 3*3)));

  OCCA_ERROR("double OccaVector::Normlinf() const",
             doublesAreEqualish(v1.Normlinf(), 3));

  OCCA_ERROR("double OccaVector::Norml1() const",
             doublesAreEqualish(v1.Norml1(), 1 + 2 + 3));

  OCCA_ERROR("double OccaVector::Normlp() const",
             doublesAreEqualish(v1.Normlp(2),
                                sqrt(1*1 + 2*2 + 3*3)));

  OCCA_ERROR("double OccaVector::Normlp() const",
             doublesAreEqualish(v1.Normlp(3),
                                pow(1 - 8 + 27, 1.0/3.0)));

  OCCA_ERROR("double OccaVector::Max() const",
             doublesAreEqualish(v1.Max(), 3));

  OCCA_ERROR("double OccaVector::Min() const",
             doublesAreEqualish(v1.Min(), -2));

  OCCA_ERROR("double OccaVector::Sum() const",
             doublesAreEqualish(v1.Sum(), 2));

  OCCA_ERROR("double OccaVector::DistanceTo(const OccaVector &other) const",
             doublesAreEqualish(v1.DistanceTo(v2),
                                sqrt(3*3 + 6*6 + 1*1)));
}

void testVectorAddSubtractMethods() {
  mfem::OccaVector v1, v2, out;
  mfem::Vector zero_(3), v1_(3), v2_(3), out_(3);

  const double alpha = 2;
  const double beta = 3;

  zero_(0) = 0; zero_(1) = 0; zero_(2) = 0;
  v1_(0) = 1; v1_(1) = 2; v1_(2) = 3;
  v2_(0) = 4; v2_(1) = 5; v2_(2) = 6;
  v1 = v1_;
  v2 = v2_;


  // Set out = v1 + v2.
  out = zero_;
  out_(0) = 5; out_(1) = 7; out_(2) = 9;
  mfem::add(v1, v2, out);
  compareVectorAndPtr("void add("
                      "const OccaVector &v1, "
                      "const OccaVector &v2, "
                      "OccaVector &out);",
                      out, out_.GetData());

  // Set out = v1 + alpha * v2.
  out = zero_;
  out_(0) = 9; out_(1) = 12; out_(2) = 15;
  mfem::add(v1, alpha, v2, out);
  compareVectorAndPtr("void add("
                      "const OccaVector &v1, "
                      "const double alpha, "
                      "const OccaVector &v2, "
                      "OccaVector &out);",
                      out, out_.GetData());

  // out = alpha * (v1 + v2)
  out = zero_;
  out_(0) = 10; out_(1) = 14; out_(2) = 18;
  mfem::add(alpha, v1, v2, out);
  compareVectorAndPtr("void add("
                      "const double alpha, "
                      "const OccaVector &v1, "
                      "const OccaVector &v2, "
                      "OccaVector &out);",
                      out, out_.GetData());

  // out = alpha * v1 + beta * v2
  out = zero_;
  out_(0) = 14; out_(1) = 19; out_(2) = 24;
  mfem::add(alpha, v1, beta, v2, out);
  compareVectorAndPtr("void add("
                      "const double alpha, "
                      "const OccaVector &v1, "
                      "const double beta, "
                      "const OccaVector &v2, "
                      "OccaVector &out);",
                      out, out_.GetData());

  // Set out = v1 - v2.
  out = zero_;
  out_(0) = -3; out_(1) = -3; out_(2) = -3;
  mfem::subtract(v1, v2, out);
  compareVectorAndPtr("void subtract("
                      "const OccaVector &v1, "
                      "const OccaVector &v2, "
                      "OccaVector &out);",
                      out, out_.GetData());

  // Set out = v1 - alpha * v2.
  out = zero_;
  out_(0) = -7; out_(1) = -8; out_(2) = -9;
  mfem::subtract(v1, alpha, v2, out);
  compareVectorAndPtr("void subtract("
                      "const OccaVector &v1, "
                      "const double alpha, "
                      "const OccaVector &v2, "
                      "OccaVector &out);",
                      out, out_.GetData());

  // out = alpha * (v1 - v2)
  out = zero_;
  out_(0) = -6; out_(1) = -6; out_(2) = -6;
  mfem::subtract(alpha, v1, v2, out);
  compareVectorAndPtr("void subtract("
                      "const double alpha, "
                      "const OccaVector &v1, "
                      "const OccaVector &v2, "
                      "OccaVector &out);",
                      out, out_.GetData());

  // out = alpha * v1 - beta * v2
  out = zero_;
  out_(0) = -10; out_(1) = -11; out_(2) = -12;
  mfem::subtract(alpha, v1, beta, v2, out);
  compareVectorAndPtr("void subtract("
                      "const double alpha, "
                      "const OccaVector &v1, "
                      "const double beta, "
                      "const OccaVector &v2, "
                      "OccaVector &out);",
                      out, out_.GetData());
}

void testVector() {
  testVectorConstructors();
  testVectorResize();
  testVectorAssignmentOperators();
  testVectorSubVectorMethods();
  testVectorAddElementVectorMethods();
  testVectorLinalgMethods();
  testVectorAddSubtractMethods();
}