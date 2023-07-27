// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "transport_coefs.hpp"

#ifdef MFEM_USE_MPI

using namespace std;
namespace mfem
{
using namespace common;

namespace plasma
{

namespace transport
{

TransportCoefFactory::TransportCoefFactory(
   const std::vector<std::string> & names,
   ParGridFunctionArray & pgfa)
{
   MFEM_VERIFY(names.size() == pgfa.Size(),
               "TransportCoefFactory constructor: "
               "Size mismatch in input arguments.");
   for (int i=0; i<pgfa.Size(); i++)
   {
      this->AddExternalGridFunction(names[i], *pgfa[i]);
   }
}

Coefficient *
TransportCoefFactory::GetScalarCoef(std::string &name, std::istream &input)
{
   int coef_idx = -1;
   if (name == "StateVariableConstantCoef")
   {
      double c;
      input >> c;
      coef_idx = sCoefs.Append(new StateVariableConstantCoef(c));
   }
   else if (name == "StateVariableGridFunctionCoef")
   {
      int ft;
      string gf_name;
      input >> ft >> gf_name;
      MFEM_VERIFY(ext_gf.find(gf_name) != ext_gf.end(), "TransportCoefFactory: "
                  "GridFunction named \"" << gf_name << "\" not found amongst "
                  "external GridFunctions.");
      coef_idx = sCoefs.Append(new StateVariableGridFunctionCoef(ext_gf[gf_name],
                                                                 (FieldType)ft));
   }
   else if (name == "StateVariableSumCoef")
   {
      Coefficient * ACoef = this->GetScalarCoef(input);
      Coefficient * BCoef = this->GetScalarCoef(input);

      StateVariableCoef * A = dynamic_cast<StateVariableCoef*>(ACoef);
      StateVariableCoef * B = dynamic_cast<StateVariableCoef*>(BCoef);

      MFEM_VERIFY(A != NULL, "TransportCoefFactory: first argument to "
                  "StateVariableSumCoef is not a StateVariableCoef.");
      MFEM_VERIFY(B != NULL, "TransportCoefFactory: second argument to "
                  "StateVariableSumCoef is not a StateVariableCoef.");

      coef_idx = sCoefs.Append(new StateVariableSumCoef(*A, *B));
   }
   else if (name == "StateVariableProductCoef")
   {
      Coefficient * ACoef = this->GetScalarCoef(input);
      Coefficient * BCoef = this->GetScalarCoef(input);

      StateVariableCoef * A = dynamic_cast<StateVariableCoef*>(ACoef);
      StateVariableCoef * B = dynamic_cast<StateVariableCoef*>(BCoef);

      MFEM_VERIFY(A != NULL, "TransportCoefFactory: first argument to "
                  "StateVariableProductCoef is not a StateVariableCoef.");
      MFEM_VERIFY(B != NULL, "TransportCoefFactory: second argument to "
                  "StateVariableProductCoef is not a StateVariableCoef.");

      coef_idx = sCoefs.Append(new StateVariableProductCoef(*A, *B));
   }
   else if (name == "StateVariablePowerCoef")
   {
      Coefficient * ACoef = this->GetScalarCoef(input);

      StateVariableCoef * A = dynamic_cast<StateVariableCoef*>(ACoef);

      MFEM_VERIFY(A != NULL, "TransportCoefFactory: first argument to "
                  "StateVariablePowerCoef is not a StateVariableCoef.");

      int p;
      input >> p;

      coef_idx = sCoefs.Append(new StateVariablePowerCoef(*A, p));
   }
   else if (name == "SoundSpeedCoef")
   {
      double mi;
      input >> mi;

      string TiCoefName;
      input >> TiCoefName;
      Coefficient * TiCoef = this->GetScalarCoef(TiCoefName, input);

      string TeCoefName;
      input >> TeCoefName;
      Coefficient * TeCoef = this->GetScalarCoef(TeCoefName, input);

      coef_idx = sCoefs.Append(new SoundSpeedCoef(mi, *TiCoef, *TeCoef));
   }
   else if (name == "ApproxIonizationRate")
   {
      Coefficient * TeCoef = this->GetScalarCoef(input);

      coef_idx = sCoefs.Append(new ApproxIonizationRate(*TeCoef));
   }
   else if (name == "ApproxRecombinationRate")
   {
      Coefficient * TeCoef = this->GetScalarCoef(input);

      coef_idx = sCoefs.Append(new ApproxRecombinationRate(*TeCoef));
   }
   else if (name == "ApproxChargeExchangeRate")
   {
      Coefficient * TiCoef = this->GetScalarCoef(input);

      coef_idx = sCoefs.Append(new ApproxRecombinationRate(*TiCoef));
   }
   else if (name == "IonizationSourceCoef")
   {
      Coefficient * neCoef = this->GetScalarCoef(input);
      Coefficient * nnCoef = this->GetScalarCoef(input);
      Coefficient * izCoef = this->GetScalarCoef(input);

      double nn0 = 1e10;
      input >> nn0;

      coef_idx = sCoefs.Append(new IonizationSourceCoef(*neCoef, *nnCoef,
                                                        *izCoef, nn0));
   }
   else if (name == "RecombinationSinkCoef")
   {
      Coefficient * neCoef = this->GetScalarCoef(input);
      Coefficient * niCoef = this->GetScalarCoef(input);
      Coefficient * rcCoef = this->GetScalarCoef(input);

      double ni0 = 1e10;
      input >> ni0;

      coef_idx = sCoefs.Append(new RecombinationSinkCoef(*neCoef, *niCoef,
                                                         *rcCoef, ni0));
   }
   else if (name == "ChargeExchangeSinkCoef")
   {
      Coefficient * nnCoef = this->GetScalarCoef(input);
      Coefficient * niCoef = this->GetScalarCoef(input);
      Coefficient * cxCoef = this->GetScalarCoef(input);

      coef_idx = sCoefs.Append(new RecombinationSinkCoef(*nnCoef, *niCoef,
                                                         *cxCoef));
   }
   else
   {
      return CoefFactory::GetScalarCoef(name, input);
   }
   return sCoefs[--coef_idx];
}

VectorCoefficient *
TransportCoefFactory::GetVectorCoef(std::string &name, std::istream &input)
{
   int coef_idx = -1;
   if (name == "__dummy_name__")
   {
   }
   else
   {
      return CoefFactory::GetVectorCoef(name, input);
   }
   return vCoefs[--coef_idx];
}

} // namespace transport

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
