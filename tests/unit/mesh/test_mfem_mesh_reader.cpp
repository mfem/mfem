// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

#include<algorithm>
#include<string>
#include<utility>
#include<vector>

using namespace mfem;

TEST_CASE("MFEM Mesh Named Attributes", "[MFEMData][Mesh]")
{
   const std::string fname = "compass.mesh";

   const std::string & fpath = (mfem_data_dir + "/" + fname);

   Mesh mesh(fpath);

   REQUIRE(mesh.Dimension() == 2);
   REQUIRE(mesh.GetNE() == 12);
   REQUIRE(mesh.GetNV() == 13);

   REQUIRE(mesh.attribute_sets.attr_sets.Size() == 16);
   REQUIRE(mesh.bdr_attribute_sets.attr_sets.Size() == 13);



   std::vector<std::pair<std::string, std::vector<int> > > expected_attr_sets =
   {
      {"Base",  {9}},
      {"E Even",  {16}},
      {"E Odd",  {17}},
      {"East",  {16, 17}},
      {"N Even",  {10}},
      {"N Odd",  {11}},
      {"North",  {10, 11}},
      {"Rose",  {10, 11, 12, 13, 14, 15, 16, 17}},
      {"Rose Even",  {10, 12, 14, 16}},
      {"Rose Odd",  {11, 13, 15, 17}},
      {"S Even",  {14}},
      {"S Odd",  {15}},
      {"South",  {14, 15}},
      {"W Even",  {12}},
      {"W Odd",  {13}},
      {"West",  {12, 13}}
   };


   for (auto const &attr_name_index_pair: expected_attr_sets )
   {
      REQUIRE(mesh.attribute_sets.AttributeSetExists(attr_name_index_pair.first));

      auto const &attr_set = mesh.attribute_sets.GetAttributeSet(
                                attr_name_index_pair.first);
      auto const &expected_attr_set = attr_name_index_pair.second;



      REQUIRE( attr_set.Size() == expected_attr_set.size());

      bool const elements_equal = std::equal(attr_set.begin(), attr_set.end(),
                                             expected_attr_set.begin());


      REQUIRE(elements_equal);
   }

   std::vector<std::pair<std::string, std::vector<int> > > expected_bdr_attr_sets
   =
   {

      {"Boundary",  {1, 2, 3, 4, 5, 6, 7, 8}},
      {"ENE",  { 1}},
      {"ESE",  { 8}},
      {"Eastern Boundary",  {1, 8}},
      {"NNE",  { 2}},
      {"NNW",  { 3}},
      {"Northern Boundary", {2, 3}},
      {"SSE",  { 7}},
      {"SSW",  { 6}},
      {"Southern Boundary", {6,7}},
      {"WNW",  { 4}},
      {"WSW",  { 5}},
      {"Western Boundary", {4,5}}
   };



   for (auto const &attr_bdr_name_index_pair: expected_bdr_attr_sets )
   {

      REQUIRE(mesh.bdr_attribute_sets.AttributeSetExists(
                 attr_bdr_name_index_pair.first));

      auto const &bdr_attr_set = mesh.bdr_attribute_sets.GetAttributeSet(
                                    attr_bdr_name_index_pair.first);
      auto const &expected_bdr_attr_set = attr_bdr_name_index_pair.second;

      REQUIRE( bdr_attr_set.Size() == expected_bdr_attr_set.size());

      bool const elements_equal = std::equal(bdr_attr_set.begin(), bdr_attr_set.end(),
                                             expected_bdr_attr_set.begin());


      REQUIRE(elements_equal);
   }

}
