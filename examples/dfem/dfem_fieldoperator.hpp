#pragma once

#include <string>

namespace mfem
{

template <int FIELD_ID = -1>
class FieldOperator
{
public:
   constexpr FieldOperator(int size_on_qp = 0) :
      size_on_qp(size_on_qp) {};

   static constexpr int GetFieldId() { return FIELD_ID; }

   int size_on_qp = -1;

   int dim = -1;

   int vdim = -1;
};

template <int FIELD_ID = -1>
class None : public FieldOperator<FIELD_ID>
{
public:
   constexpr None() : FieldOperator<FIELD_ID>() {}
};

template< typename T >
struct is_none_fop
{
   static const bool value = false;
};

template <int FIELD_ID>
struct is_none_fop<None<FIELD_ID>>
{
   static const bool value = true;
};

template <typename T>
struct DisableAD
{
   T& operator()() const { return fop; }
   T fop;
};

class Weight : public FieldOperator<-1>
{
public:
   constexpr Weight() : FieldOperator<-1>() {};
};

template< typename T >
struct is_weight_fop
{
   static const bool value = false;
};

template <>
struct is_weight_fop<Weight>
{
   static const bool value = true;
};

template <int FIELD_ID = -1>
class Value : public FieldOperator<FIELD_ID>
{
public:
   constexpr Value() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_value_fop
{
   static const bool value = false;
};

template <int FIELD_ID>
struct is_value_fop<Value<FIELD_ID>>
{
   static const bool value = true;
};

template <typename T>
struct is_value_fop<DisableAD<T>>
{
   static const bool value = is_value_fop<T>::value;
};

template <int FIELD_ID = -1>
class Gradient : public FieldOperator<FIELD_ID>
{
public:
   constexpr Gradient() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_gradient_fop
{
   static const bool value = false;
};

template <int FIELD_ID>
struct is_gradient_fop<Gradient<FIELD_ID>>
{
   static const bool value = true;
};

template <int FIELD_ID = -1>
class One : public FieldOperator<FIELD_ID>
{
public:
   constexpr One() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_one_fop
{
   static const bool value = false;
};

template <int FIELD_ID>
struct is_one_fop<One<FIELD_ID>>
{
   static const bool value = true;
};

// class FieldOperator
// {
// public:
//    FieldOperator(std::string field_label = "", int size_on_qp = 0) :
//       field_label(field_label),
//       size_on_qp(size_on_qp) {};

//    std::string field_label;

//    int size_on_qp = -1;

//    int dim = -1;

//    int vdim = -1;
// };

// class None : public FieldOperator
// {
// public:
//    None(std::string field_label) :
//       FieldOperator(field_label) {}
// };

// class Weight : public FieldOperator
// {
// public:
//    Weight() : FieldOperator("quadrature_weights") {};
// };

// class Value : public FieldOperator
// {
// public:
//    Value(std::string field_label) : FieldOperator(field_label) {};
// };

// class Gradient : public FieldOperator
// {
// public:
//    Gradient(std::string field_label) : FieldOperator(field_label) {};
// };

// class Curl : public FieldOperator
// {
// public:
//    Curl(std::string field_label) : FieldOperator(field_label) {};
// };

// class Div : public FieldOperator
// {
// public:
//    Div(std::string field_label) : FieldOperator(field_label) {};
// };

// class FaceValueLeft : public FieldOperator
// {
// public:
//    FaceValueLeft(std::string field_label) : FieldOperator(field_label) {};
// };

// class FaceValueRight : public FieldOperator
// {
// public:
//    FaceValueRight(std::string field_label) : FieldOperator(field_label) {};
// };

// class FaceNormal : public FieldOperator
// {
// public:
//    FaceNormal(std::string field_label) : FieldOperator(field_label) {};
// };

// class One : public FieldOperator
// {
// public:
//    One(std::string field_label) : FieldOperator(field_label) {};
// };

// namespace BareFieldOperator
// {

// struct Base
// {
//    Base(FieldOperator &o)
//    {
//       size_on_qp = o.size_on_qp;
//       dim = o.dim;
//       vdim = o.vdim;
//    };
//    int size_on_qp = -1;
//    int dim = -1;
//    int vdim = -1;
// };

// struct None : Base
// {
//    None(FieldOperator &o) : Base(o) {}
// };

// struct Weight : Base
// {
//    Weight(FieldOperator &o) : Base(o) {}
// };

// struct Value : Base
// {
//    Value(FieldOperator &o) : Base(o) {}
// };

// struct Gradient : Base
// {
//    Gradient(FieldOperator &o) : Base(o) {}
// };

// }

} // namespace mfem
