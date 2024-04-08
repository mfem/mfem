// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

using namespace mfem;

TEST_CASE("ComplexDenseMatrix", "[ComplexDenseMatrix]")
{
   DenseMatrix A_r(
   {
      {
         7.476973198773836e-01, 5.307752092411233e-01,
         7.787714872353524e-01, 1.374015004967300e-01
      },
      {
         5.912071572460218e-01, 7.793148354527577e-01,
         8.614244090659824e-01, 3.664367109052707e-01
      },
      {
         4.707552432065591e-02, 7.498226683758581e-01,
         5.086949143815038e-02, 4.642421529285558e-01
      },
      {
         2.153684312350395e-01, 8.995081931810892e-01,
         2.838610147580374e-01, 3.462963875422466e-01
      }
   });

   DenseMatrix A_i(
   {
      {
         2.338532314577970e-01, 4.866770088208406e-01,
         6.635456790536266e-01, 5.077509109627342e-01
      },
      {
         5.921703125786201e-01, 9.612084289713472e-01,
         6.886727004080341e-01, 2.105085744039327e-01
      },
      {
         5.857974556931360e-02, 3.537664834416591e-01,
         9.653908939745925e-01, 6.672005467737580e-01
      },
      {
         1.883898354361578e-01, 6.475861201488613e-01,
         6.396282633261788e-01, 7.482565383983398e-01
      }
   });

   DenseMatrix B_r(
   {
      {
         1.401452180742794e+00, 1.912351953807087e+00,
         1.895130685031467e+00, 8.392549763377584e-01
      },
      {
         1.912351953807087e+00, 3.965697621768416e+00,
         3.118779584828659e+00, 2.188141653405000e+00
      },
      {
         1.895130685031467e+00, 3.118779584828659e+00,
         3.687368672616838e+00, 2.149180498127385e+00
      },
      {
         8.392549763377584e-01, 2.188141653405000e+00,
         2.149180498127385e+00, 1.795766264370507e+00
      }
   });

   DenseMatrix B_i(
   {
      {
         0.000000000000000e+00, 2.892905182471230e-01,
         3.377969313208506e-01, 3.550991903463218e-01
      },
      {
         -2.892905182471230e-01, 0.000000000000000e+00,
            7.792703332501396e-01, 7.993167778330725e-01
         },
      {
         -3.377969313208506e-01, -7.792703332501396e-01,
            0.000000000000000e+00, -1.901030659985157e-01
         },
      {
         -3.550991903463218e-01, -7.993167778330725e-01,
            1.901030659985157e-01,  0.000000000000000e+00
         }
   });

   DenseMatrix AB_r(
   {
      {
         4.199316358694532, 7.119508739631666,
         5.684500367417206, 3.363473388898064
      },
      {
         4.844369905372942, 8.243147426694234,
         6.525761183975376, 3.863145594383837
      },
      {
         2.651290557124633, 5.506731777039958,
         3.190755998317346, 2.503175360026021
      },
      {
         3.519699773999353, 6.664121172538803,
         4.293944745711777, 2.918008065054989
      }
   });

   DenseMatrix AB_i(
   {
      {
         2.476664090430751, 5.057308079927443,
         6.191314340700356, 4.140779508119272
      },
      {
         3.503306034281864, 6.759590101303163,
         7.988514056402222, 5.127445401059480
      },
      {
         2.749163864229835, 5.588627587485725,
         6.896492887398851, 4.702581910360381
      },
      {
         2.863512962181819, 6.124848339934783,
         7.182928158705073, 5.034994005830734
      }
   });

   DenseMatrix AtB_r(
   {
      {
         2.190427107115792, 4.263902975830449,
         4.473548538497790, 2.954315435185178
      },
      {
         3.782581973223430, 7.759823841169083,
         9.171024988758536, 6.251394254923715
      },
      {
         2.320929711857948, 4.613604134364786,
         5.842510922053375, 3.760150638280159
      },
      {
         1.511766385570270, 2.950415646350681,
         4.337120195387853, 2.758465978422851
      }
   });

   DenseMatrix AtB_i(
   {
      {
         -1.992705303802213, -3.383029459787039,
            -2.156697737612761, -1.227098261518685
         },
      {
         -4.532297784683165, -8.412647875423712,
            -5.658770233055886, -3.766073955464488
         },
      {
         -4.980449520661596, -8.451683973225684,
            -7.351440910685274, -4.331797912622623
         },
      {
         -3.392361165173878, -6.122776241469277,
            -5.289337862384023, -3.410946995052299
         }
   });

   DenseMatrix invA_r(
   {
      {
         1.284125217026929e+00,  5.566676578220062e-01,
         3.205235484188196e-01, -1.884823653328449e+00
      },
      {
         -8.923029616469982e-01, 1.334042076588671e-01,
            3.358025813557918e-01, 6.973560749734028e-01
         },
      {
         2.366798350499296e-01, -5.172682567295234e-01,
         -6.685771336109858e-01,  1.099184759019011e+00
      },
      {
         -1.623347650066281e-02,  8.311589945672611e-01,
            6.287242859617315e-02, -7.940924063262524e-01
         }
   });

   DenseMatrix invA_i(
   {
      {
         -2.049683956467907e-01,  3.816877789141072e-01,
            2.326958896869482e+00, -2.208208959084148e+00
         },
      {
         7.275898603082531e-01, -1.183531610937897e+00,
         -5.227713373718834e-01,  7.379400884151782e-01
      },
      {
         -1.255712097872317e-01, -1.944130914054951e-01,
            -1.977900692062877e+00,  1.787591531313541e+00
         },
      {
         -1.091544681586941e+00,  1.603416041973783e+00,
            1.828734546571891e+00, -2.601914242111798e+00
         }
   });

   DenseMatrix invB_r(
   {
      {
         1.609277412199292e+01, -5.725193323656868e+00,
         -1.086842033161421e+01,  1.279540669178842e+01
      },
      {
         -5.725193323656868e+00,  4.161049651113045e+00,
            2.753672305121771e+00, -5.975244397594217e+00
         },
      {
         -1.086842033161421e+01,  2.753672305121771e+00,
            9.139925521110435e+00, -9.791537312098491e+00
         },
      {
         1.279540669178842e+01, -5.975244397594217e+00,
         -9.791537312098491e+00,  1.520226505276132e+01
      }
   });

   DenseMatrix invB_i(
   {
      {
         0.000000000000000e+00,  7.582682740428248e-01,
         4.380318627231539e-02, -2.760797691635214e+00
      },
      {
         -7.582682740428248e-01,  0.000000000000000e+00,
            1.276538947979396e+00, -1.601898857634354e+00
         },
      {
         -4.380318627231539e-02, -1.276538947979396e+00,
            0.000000000000000e+00,  3.466959403556459e+00
         },
      {
         2.760797691635214e+00,  1.601898857634354e+00,
         -3.466959403556459e+00,   0.000000000000000e+00
      }
   });

   // L matrix where B = L * L^H (real part)
   DenseMatrix BL_r(
   {
      {
         1.183829455936451e+00, 0.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         1.615394805575562e+00, 1.138631338745275e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         1.600847719684717e+00, 4.066722657083971e-01,
         6.996783780831184e-01, 0.000000000000000e+00
      },
      {
         7.089323315357766e-01, 8.515800861725336e-01,
         4.506517233907201e-01, 2.564754791774774e-01
      }
   });

   // L matrix where B = L * L^H (imag part)
   DenseMatrix BL_i(
   {
      {
         0.000000000000000e+00,  0.000000000000000e+00,
         0.000000000000000e+00,  0.000000000000000e+00
      },
      {
         -2.443684069495331e-01,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00
         },
      {
         -2.853425631765863e-01, -6.231393970721737e-01,
            0.000000000000000e+00,  0.000000000000000e+00
         },
      {
         -2.999580628490323e-01, -4.285906501101729e-01,
            1.595654676419282e-01,  0.000000000000000e+00
         }
   });

   // L matrix where P * A = L * U (real part)
   DenseMatrix AL_r(
   {
      {
         1.000000000000000e+00, 0.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         8.929066668627371e-02, 1.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         8.290944409495822e-01, -6.632415969670290e-01,
         1.000000000000000e+00,  0.000000000000000e+00
      },
      {
         3.411726617549650e-01, 9.379298209889033e-01,
         -3.855119154762174e-01, 1.000000000000000e+00
      }
   });

   // L matrix where P * A = L * U (imag part)
   DenseMatrix AL_i(
   {
      {
         0.000000000000000e+00, 0.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         9.648840507821686e-03, 0.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         -4.348930483084801e-01, 2.920682426805858e-01,
            0.000000000000000e+00, 0.000000000000000e+00
         },
      {
         -2.307564472343243e-02, 1.354227054751071e-01,
            -1.114174339166530e-01, 0.000000000000000e+00
         }
   });

   // U matrix where P * A = L * U (real part)
   DenseMatrix AU_r(
   {
      {
         5.912071572460218e-01, 7.793148354527577e-01,
         8.614244090659824e-01, 3.664367109052707e-01
      },
      {
         0.000000000000000e+00, 6.895116739856960e-01,
         -1.940277529885255e-02, 4.335539383734513e-01
      },
      {
         0.000000000000000e+00, 0.000000000000000e+00,
         1.377415198958651e-02, 2.179387335957125e-01
      },
      {
         0.000000000000000e+00, 0.000000000000000e+00,
         0.000000000000000e+00,-1.073019476366396e-01
      }
   });

   // U matrix where P * A = L * U (imag part)
   DenseMatrix AU_i(
   {
      {
         5.921703125786201e-01, 9.612084289713472e-01,
         6.886727004080341e-01, 2.105085744039327e-01
      },
      {
         0.000000000000000e+00, 2.604200574416788e-01,
         8.955871026939005e-01, 6.448684064423180e-01
      },
      {
         0.000000000000000e+00, 0.000000000000000e+00,
         1.066856013356064e+00, 7.936564152043436e-01
      },
      {
         0.000000000000000e+00, 0.000000000000000e+00,
         0.000000000000000e+00, 3.515843540851109e-01
      }
   });

   // P matrix where P * A = L * U
   DenseMatrix P_r(
   {
      {
         0.0, 1.0, 0.0, 0.0
      },
      {
         0.0, 0.0, 1.0, 0.0
      },
      {
         1.0, 0.0, 0.0, 0.0
      },
      {
         0.0, 0.0, 0.0, 1.0
      }
   });

   DenseMatrix invBL_r(
   {
      {
         8.447162680277831e-01, 0.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         -1.198412712810921e+00, 8.782473887482838e-01,
            0.000000000000000e+00, 0.000000000000000e+00
         },
      {
         -1.397598658079286e+00, -5.104614729030819e-01,
            1.429228101545257e+00,  0.000000000000000e+00
         },
      {
         3.281708062547134e+00, -1.532503670075513e+00,
         -2.511289224004609e+00,  3.899008213989978e+00
      }
   });

   DenseMatrix invBL_i(
   {
      {
         0.000000000000000e+00, 0.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         1.812895550282085e-01, 0.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         -8.281949571012291e-01, 7.821744467853249e-01,
            0.000000000000000e+00, 0.000000000000000e+00
         },
      {
         7.080769108742154e-01, 4.108477771056253e-01,
         -8.891900743160057e-01, 0.000000000000000e+00
      }
   });

   DenseMatrix invAL_r(
   {
      {
         1.000000000000000e+00, 0.000000000000000e+00,
         0.000000000000000e+00, 0.000000000000000e+00
      },
      {
         -8.929066668627371e-02, 1.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00
         },
      {
         -8.911338452078618e-01, 6.632415969670290e-01,
            1.000000000000000e+00, 0.000000000000000e+00
         },
      {
         -6.529209723598358e-01, -6.497007883906214e-01,
            3.855119154762174e-01, 1.000000000000000e+00
         }
   });

   DenseMatrix invAL_i(
   {
      {
         0.000000000000000e+00,  0.000000000000000e+00,
         0.000000000000000e+00,  0.000000000000000e+00
      },
      {
         -9.648840507821686e-03,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00
         },
      {
         4.545725040280302e-01, -2.920682426805858e-01,
         0.000000000000000e+00,  0.000000000000000e+00
      },
      {
         1.201728340713420e-01, -1.741218163598231e-01,
         1.114174339166530e-01,  0.000000000000000e+00
      }
   });

   DenseMatrix invAU_r(
   {
      {
         8.443505642555830e-01, -1.940087632374593e+00,
         1.764714218183542e+00, -1.884823653328449e+00
      },
      {
         0.000000000000000e+00,  1.269246346038117e+00,
         -1.078922646843524e+00,  6.973560749734028e-01
      },
      {
         0.000000000000000e+00,  0.000000000000000e+00,
         1.209987444834138e-02,  1.099184759019011e+00
      },
      {
         0.000000000000000e+00,  0.000000000000000e+00,
         0.000000000000000e+00, -7.940924063262524e-01
      }
   });

   DenseMatrix invAU_i(
   {
      {
         -8.457261239702749e-01,  5.115617062678094e-01,
            8.563246847807503e-01, -2.208208959084148e+00
         },
      {
         0.000000000000000e+00, -4.793786948264843e-01,
         3.654075389169029e-01,  7.379400884151782e-01
      },
      {
         0.000000000000000e+00,  0.000000000000000e+00,
         -9.371773903631724e-01,  1.787591531313541e+00
      },
      {
         0.000000000000000e+00,  0.000000000000000e+00,
         0.000000000000000e+00, -2.601914242111798e+00
      }
   });

   ComplexDenseMatrix A(&A_r,&A_i,false,false);
   ComplexDenseMatrix B(&B_r,&B_i,false,false);

   SECTION("Mult")
   {
      ComplexDenseMatrix * AB = Mult(A,B);
      AB_r -= AB->real();
      AB_i -= AB->imag();

      double norm_r = AB_r.MaxMaxNorm();
      double norm_i = AB_i.MaxMaxNorm();

      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete AB;
   }
   SECTION("MultAtB")
   {
      ComplexDenseMatrix * AtB = MultAtB(A,B);
      AtB_r -= AtB->real();
      AtB_i -= AtB->imag();

      double norm_r = AtB_r.MaxMaxNorm();
      double norm_i = AtB_i.MaxMaxNorm();

      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete AtB;
   }
   SECTION("Inverse")
   {
      ComplexDenseMatrix * invA = A.ComputeInverse();
      invA_r -= invA->real();
      invA_i -= invA->imag();

      double norm_r = invA_r.MaxMaxNorm();
      double norm_i = invA_i.MaxMaxNorm();

      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete invA;
   }
   SECTION("SystemMatrix")
   {
      DenseMatrix * sA = A.GetSystemMatrix();
      sA->Invert();
      ComplexDenseMatrix * invA = A.ComputeInverse();
      DenseMatrix * sinvA = invA->GetSystemMatrix();

      *sA-=*sinvA;
      double norm = sA->MaxMaxNorm();
      REQUIRE(norm == MFEM_Approx(0.));

      delete sinvA;
      delete invA;
      delete sA;
   }

   double norm_r = 0.;
   double norm_i = 0.;
   DenseMatrix diff_r;
   DenseMatrix diff_i;
   ComplexCholeskyFactors chol(B.real().Data(),B.imag().Data());
   const int m = 4;
   chol.Factor(m);

   SECTION("ComplexCholeskyFactors::Inverse")
   {
      DenseMatrix Binv_r(m);
      DenseMatrix Binv_i(m);
      chol.GetInverseMatrix(m,Binv_r.Data(), Binv_i.Data());

      diff_r = invB_r; diff_r-=Binv_r;
      diff_i = invB_i; diff_i-=Binv_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
   }
   SECTION("ComplexCholeskyFactors::LMult")
   {
      ComplexDenseMatrix exactL(&BL_r, &BL_i, false,false);
      ComplexDenseMatrix * LA = mfem::Mult(exactL,A);

      DenseMatrix LA_r(A_r);
      DenseMatrix LA_i(A_i);
      chol.LMult(m,m,LA_r.Data(),LA_i.Data());


      diff_r = LA->real(); diff_r-=LA_r;
      diff_i = LA->imag(); diff_i-=LA_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete LA;
   }
   SECTION("ComplexCholeskyFactors::UMult")
   {
      ComplexDenseMatrix exactL(&BL_r, &BL_i, false,false);
      ComplexDenseMatrix * UA = mfem::MultAtB(exactL,A);
      DenseMatrix LtA_r(A_r);
      DenseMatrix LtA_i(A_i);
      chol.UMult(m,m,LtA_r.Data(),LtA_i.Data());

      diff_r = UA->real(); diff_r-=LtA_r;
      diff_i = UA->imag(); diff_i-=LtA_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete UA;
   }
   SECTION("ComplexCholeskyFactors::LSolve")
   {

      ComplexDenseMatrix exactinvL(&invBL_r, &invBL_i, false,false);
      ComplexDenseMatrix * invLA = mfem::Mult(exactinvL,A);

      DenseMatrix invLA_r(A_r);
      DenseMatrix invLA_i(A_i);
      chol.LSolve(m,m,invLA_r.Data(),invLA_i.Data());

      diff_r = invLA->real(); diff_r-=invLA_r;
      diff_i = invLA->imag(); diff_i-=invLA_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete invLA;
   }
   SECTION("ComplexCholeskyFactors::USolve")
   {
      ComplexDenseMatrix exactinvL(&invBL_r, &invBL_i, false,false);
      ComplexDenseMatrix * invUA = mfem::MultAtB(exactinvL,A);
      DenseMatrix invUA_r(A_r);
      DenseMatrix invUA_i(A_i);
      chol.USolve(m,m,invUA_r.Data(),invUA_i.Data());

      diff_r = invUA->real(); diff_r-=invUA_r;
      diff_i = invUA->imag(); diff_i-=invUA_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete invUA;
   }
   SECTION("ComplexCholeskyFactors::Solve")
   {
      ComplexDenseMatrix invB(&invB_r,&invB_i,false,false);
      ComplexDenseMatrix * invBA = mfem::Mult(invB,A);

      DenseMatrix invBA_r(A_r);
      DenseMatrix invBA_i(A_i);
      chol.Solve(m,m,invBA_r.Data(),invBA_i.Data());

      diff_r = invBA->real(); diff_r-=invBA_r;
      diff_i = invBA->imag(); diff_i-=invBA_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete invBA;
   }
   SECTION("ComplexCholeskyFactors::RightSolve")
   {
      ComplexDenseMatrix invB(&invB_r,&invB_i,false,false);
      ComplexDenseMatrix * AinvB = mfem::Mult(A,invB);

      DenseMatrix AinvB_r(A_r);
      DenseMatrix AinvB_i(A_i);
      chol.RightSolve(m,m,AinvB_r.Data(),AinvB_i.Data());

      diff_r = AinvB->real(); diff_r-=AinvB_r;
      diff_i = AinvB->imag(); diff_i-=AinvB_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete AinvB;
   }

   int ipiv[m];
   ComplexLUFactors lu(A.real().Data(),A.imag().Data(), ipiv);
   lu.Factor(m);

   SECTION("ComplexLUFactors::Inverse")
   {
      DenseMatrix Ainv_r(m);
      DenseMatrix Ainv_i(m);
      lu.GetInverseMatrix(m,Ainv_r.Data(), Ainv_i.Data());

      diff_r = invA_r; diff_r-=Ainv_r;
      diff_i = invA_i; diff_i-=Ainv_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
   }
   SECTION("ComplexLUFactors::LSolve")
   {
      ComplexDenseMatrix P(&P_r,nullptr, false,false);
      ComplexDenseMatrix *PB = mfem::Mult(P,B);
      ComplexDenseMatrix exactinvL(&invAL_r, &invAL_i, false,false);
      ComplexDenseMatrix * invLB = mfem::Mult(exactinvL,*PB);

      DenseMatrix invLB_r(B_r);
      DenseMatrix invLB_i(B_i);
      lu.LSolve(m,m,invLB_r.Data(),invLB_i.Data());

      diff_r = invLB->real(); diff_r-=invLB_r;
      diff_i = invLB->imag(); diff_i-=invLB_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete PB;
      delete invLB;
   }
   SECTION("ComplexLUFactors::USolve")
   {
      ComplexDenseMatrix exactinvU(&invAU_r, &invAU_i, false,false);
      ComplexDenseMatrix * invUB = mfem::Mult(exactinvU,B);
      DenseMatrix invUB_r(B_r);
      DenseMatrix invUB_i(B_i);
      lu.USolve(m,m,invUB_r.Data(),invUB_i.Data());

      diff_r = invUB->real(); diff_r-=invUB_r;
      diff_i = invUB->imag(); diff_i-=invUB_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete invUB;
   }
   SECTION("ComplexLUFactors::Solve")
   {
      ComplexDenseMatrix invA(&invA_r,&invA_i,false,false);
      ComplexDenseMatrix * invAB = mfem::Mult(invA,B);

      DenseMatrix invAB_r(B_r);
      DenseMatrix invAB_i(B_i);
      lu.Solve(m,m,invAB_r.Data(),invAB_i.Data());

      diff_r = invAB->real(); diff_r-=invAB_r;
      diff_i = invAB->imag(); diff_i-=invAB_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete invAB;
   }
   SECTION("ComplexLUFactors::RightSolve")
   {
      ComplexDenseMatrix invA(&invA_r,&invA_i,false,false);
      ComplexDenseMatrix * BinvA = mfem::Mult(B,invA);

      DenseMatrix BinvA_r(B_r);
      DenseMatrix BinvA_i(B_i);
      lu.RightSolve(m,m,BinvA_r.Data(),BinvA_i.Data());

      diff_r = BinvA->real(); diff_r-=BinvA_r;
      diff_i = BinvA->imag(); diff_i-=BinvA_i;
      norm_r = diff_r.MaxMaxNorm();
      norm_i = diff_i.MaxMaxNorm();
      REQUIRE(norm_r == MFEM_Approx(0.));
      REQUIRE(norm_i == MFEM_Approx(0.));
      delete BinvA;
   }
}
