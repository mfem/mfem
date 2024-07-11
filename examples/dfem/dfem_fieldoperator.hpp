#pragma once

#include <string>

class FieldOperator
{
public:
   FieldOperator(std::string field_label = "", int size_on_qp = 0) :
      field_label(field_label),
      size_on_qp(size_on_qp) {};

   std::string field_label;

   int size_on_qp = -1;

   int dim = -1;

   int vdim = -1;
};

class None : public FieldOperator
{
public:
   None(std::string field_label) :
      FieldOperator(field_label) {}
};

class Weight : public FieldOperator
{
public:
   Weight() : FieldOperator("quadrature_weights") {};
};

class Value : public FieldOperator
{
public:
   Value(std::string field_label) : FieldOperator(field_label) {};
};

class Gradient : public FieldOperator
{
public:
   Gradient(std::string field_label) : FieldOperator(field_label) {};
};

class Curl : public FieldOperator
{
public:
   Curl(std::string field_label) : FieldOperator(field_label) {};
};

class Div : public FieldOperator
{
public:
   Div(std::string field_label) : FieldOperator(field_label) {};
};

class FaceValueLeft : public FieldOperator
{
public:
   FaceValueLeft(std::string field_label) : FieldOperator(field_label) {};
};

class FaceValueRight : public FieldOperator
{
public:
   FaceValueRight(std::string field_label) : FieldOperator(field_label) {};
};

class FaceNormal : public FieldOperator
{
public:
   FaceNormal(std::string field_label) : FieldOperator(field_label) {};
};

class One : public FieldOperator
{
public:
   One(std::string field_label) : FieldOperator(field_label) {};
};
