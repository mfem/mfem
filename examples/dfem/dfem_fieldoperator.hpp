#pragma once

#include <string>

class FieldOperator
{
public:
   FieldOperator(std::string name) : name(name) {};

   std::string name;

   int size_on_qp = -1;

   int dim = -1;

   int vdim = -1;
};

class None : public FieldOperator
{
public:
   None(std::string name) : FieldOperator(name) {}
};

class Weight : public FieldOperator
{
public:
   Weight(std::string name) : FieldOperator(name) {};
};

class Value : public FieldOperator
{
public:
   Value(std::string name) : FieldOperator(name) {};
};

class Gradient : public FieldOperator
{
public:
   Gradient(std::string name) : FieldOperator(name) {};
};

class One : public FieldOperator
{
public:
   One(std::string name) : FieldOperator(name) {};
};
