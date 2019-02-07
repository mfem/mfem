#include <math.h>

/******************************************************************************/

void roe1d (
   double UR[3],
   double UL[3],
   double n[1],
   double F[3])
{
   double t76;
   double t64;
   double t65;
   double t46;
   double t21;
   double t22;
   double t23;
   double t5;
   double t100;
   double t29;
   double t8;
   double t1;
   double t2;
   double t30;
   double t9;
   double t10;
   double t24;
   double t25;
   double t26;
   double t79;
   double t67;
   double t68;
   double t70;
   double t47;
   double t48;
   double t49;
   double t12;
   double t14;
   double t107;
   double t88;
   double t89;
   double t19;
   double t43;
   double t86;
   double t62;
   double t81;
   double t53;
   double t35;
   double t36;
   double t56;
   double t33;
   double t34;
   double t57;
   double t59;
   double t61;
   double t73;
   t1 = UR[1];
   t2 = n[0];
   t5 = UL[1];
   t8 = UR[0];
   t9 = UL[0];
   t10 = 0.1e1 / t9;
   t12 = sqrt(t8 * t10);
   t14 = 0.1e1 / t8;
   t19 = 0.100e1 * t12 * t1 * t14 + 0.10e1 * t5 * t10;
   t21 = 0.10e1 * t12 + 0.10e1;
   t22 = 0.1e1 / t21;
   t23 = t19 * t22;
   t24 = t23 * t2;
   t25 = fabs(t24);
   t26 = t8 - t9;
   t29 = 0.10e1 * t24;
   t30 = UR[2];
   t33 = 0.4e0 * t30;
   t34 = t1 * t1;
   t35 = t14 * t34;
   t36 = 0.2000e0 * t35;
   t43 = UL[2];
   t46 = 0.4e0 * t43;
   t47 = t5 * t5;
   t48 = t10 * t47;
   t49 = 0.2000e0 * t48;
   t53 = 0.10e1 * t12 * (0.10e1 * t30 * t14 + 0.10e1 * (t33 - t36) * t14) + 0.10e1
         * t43 * t10 + 0.10e1 * (t46 - t49) * t10;
   t56 = t19 * t19;
   t57 = t21 * t21;
   t59 = t56 / t57;
   t61 = 0.40e0 * t53 * t22 - 0.2000e0 * t59;
   t62 = sqrt(t61);
   t64 = fabs(t29 + t62);
   t65 = 0.5e0 * t64;
   t67 = fabs(-t29 + t62);
   t68 = 0.5e0 * t67;
   t70 = t65 + t68 - 0.10e1 * t25;
   t73 = t1 - t5;
   t76 = 0.2000e0 * t59 * t26 - 0.40e0 * t23 * t73 + t33 - t46;
   t79 = t70 * t76 / t61;
   t81 = t65 - t68;
   t86 = -0.10e1 * t23 * t2 * t26 + t73 * t2;
   t88 = 0.1e1 / t62;
   t89 = t81 * t86 * t88;
   F[0] = 0.50e0 * t1 * t2 + 0.50e0 * t5 * t2 - 0.50e0 * t25 * t26 - 0.5e0 * t79 -
          0.5e0 * t89;
   t100 = t79 + t89;
   t107 = t81 * t76 * t88 + t70 * t86;
   F[1] = 0.50e0 * t35 * t2 + 0.50e0 * t48 * t2 + 0.5e0 * t2 *
          (t33 - t36 + t46 - t49) - 0.50e0 * t25 * t73 - 0.50e0 * t100 * t19 * t22 - 0.5e0
          * t107 * t2;
   F[2] = 0.50e0 * (0.14e1 * t30 - t36) * t1 * t14 * t2 + 0.50e0 *
          (0.14e1 * t43 - t49) * t5 * t10 * t2 - 0.50e0 * t25 * (t30 - t43) - 0.50e0 *
          t100 * t53 * t22 - 0.50e0 * t107 * t19 * t22 * t2;
   return;
}

/******************************************************************************/

void roe2d (
   double UR[4],
   double UL[4],
   double n[2],
   double F[4])
{
   double t17;
   double t46;
   double t155;
   double t23;
   double t59;
   double t35;
   double t76;
   double t47;
   double t79;
   double t49;
   double t50;
   double t51;
   double t80;
   double t2;
   double t13;
   double t81;
   double t82;
   double t22;
   double t98;
   double t99;
   double t8;
   double t85;
   double t52;
   double t55;
   double t58;
   double t37;
   double t38;
   double t39;
   double t16;
   double t69;
   double t100;
   double t90;
   double t106;
   double t105;
   double t103;
   double t102;
   double t109;
   double t108;
   double t3;
   double t5;
   double t60;
   double t61;
   double t112;
   double t111;
   double t113;
   double t94;
   double t119;
   double t122;
   double t18;
   double t19;
   double t125;
   double t41;
   double t30;
   double t97;
   double t128;
   double t130;
   double t26;
   double t64;
   double t10;
   double t134;
   double t1;
   double t137;
   double t136;
   double t143;
   double t148;
   t1 = UR[0];
   t2 = UR[1];
   t3 = 0.1e1 / t1;
   t5 = n[0];
   t8 = UR[2];
   t10 = n[1];
   t13 = 0.10e1 * t2 * t3 * t5 + 0.10e1 * t8 * t3 * t10;
   t16 = UL[0];
   t17 = UL[1];
   t18 = 0.1e1 / t16;
   t19 = t17 * t18;
   t22 = UL[2];
   t23 = t22 * t18;
   t26 = 0.10e1 * t19 * t5 + 0.10e1 * t23 * t10;
   t30 = sqrt(t1 * t18);
   t35 = 0.100e1 * t30 * t2 * t3 + 0.10e1 * t19;
   t37 = 0.10e1 * t30 + 0.10e1;
   t38 = 0.1e1 / t37;
   t39 = t35 * t38;
   t41 = 0.10e1 * t39 * t5;
   t46 = 0.100e1 * t30 * t8 * t3 + 0.10e1 * t23;
   t47 = t46 * t38;
   t49 = 0.10e1 * t47 * t10;
   t50 = t41 + t49;
   t51 = fabs(t50);
   t52 = t1 - t16;
   t55 = UR[3];
   t58 = 0.4e0 * t55;
   t59 = t2 * t2;
   t60 = t1 * t1;
   t61 = 0.1e1 / t60;
   t64 = t8 * t8;
   t69 = 0.20e0 * t1 * (0.100e1 * t59 * t61 + 0.100e1 * t64 * t61);
   t76 = UL[3];
   t79 = 0.4e0 * t76;
   t80 = t17 * t17;
   t81 = t16 * t16;
   t82 = 0.1e1 / t81;
   t85 = t22 * t22;
   t90 = 0.20e0 * t16 * (0.100e1 * t80 * t82 + 0.100e1 * t85 * t82);
   t94 = 0.10e1 * t30 * (0.10e1 * t55 * t3 + 0.10e1 * (t58 - t69) * t3) + 0.10e1 *
         t76 * t18 + 0.10e1 * (t79 - t90) * t18;
   t97 = t35 * t35;
   t98 = t37 * t37;
   t99 = 0.1e1 / t98;
   t100 = t97 * t99;
   t102 = t46 * t46;
   t103 = t102 * t99;
   t105 = 0.40e0 * t94 * t38 - 0.2000e0 * t100 - 0.2000e0 * t103;
   t106 = sqrt(t105);
   t108 = fabs(t41 + t49 + t106);
   t109 = 0.5e0 * t108;
   t111 = fabs(-t41 - t49 + t106);
   t112 = 0.5e0 * t111;
   t113 = t109 + t112 - t51;
   t119 = t2 - t17;
   t122 = t8 - t22;
   t125 = 0.4e0 * (0.500e0 * t100 + 0.500e0 * t103) * t52 - 0.40e0 * t39 * t119 -
          0.40e0 * t47 * t122 + t58 - t79;
   t128 = t113 * t125 / t105;
   t130 = t109 - t112;
   t134 = -t50 * t52 + t119 * t5 + t122 * t10;
   t136 = 0.1e1 / t106;
   t137 = t130 * t134 * t136;
   F[0] = 0.5e0 * t1 * t13 + 0.5e0 * t16 * t26 - 0.5e0 * t51 * t52 - 0.5e0 * t128 -
          0.5e0 * t137;
   t143 = t58 - t69 + t79 - t90;
   t148 = t128 + t137;
   t155 = t130 * t125 * t136 + t113 * t134;
   F[1] = 0.5e0 * t2 * t13 + 0.5e0 * t17 * t26 + 0.5e0 * t5 * t143 - 0.5e0 * t51 *
          t119 - 0.50e0 * t148 * t35 * t38 - 0.5e0 * t155 * t5;
   F[2] = 0.5e0 * t8 * t13 + 0.5e0 * t22 * t26 + 0.5e0 * t10 * t143 - 0.5e0 * t51 *
          t122 - 0.50e0 * t148 * t46 * t38 - 0.5e0 * t155 * t10;
   F[3] = 0.5e0 * (0.14e1 * t55 - t69) * t13 + 0.5e0 * (0.14e1 * t76 - t90) * t26 -
          0.5e0 * t51 * (t55 - t76) - 0.50e0 * t148 * t94 * t38 - 0.5e0 * t155 * t50;
   return;
}

/******************************************************************************/

void roe3d (
   double UR[5],
   double UL[5],
   double n[3],
   double F[5])
{
   double t105;
   double t138;
   double t8;
   double t10;
   double t44;
   double t27;
   double t28;
   double t35;
   double t186;
   double t76;
   double t55;
   double t56;
   double t58;
   double t23;
   double t24;
   double t18;
   double t125;
   double t126;
   double t128;
   double t129;
   double t131;
   double t132;
   double t134;
   double t135;
   double t137;
   double t77;
   double t78;
   double t81;
   double t48;
   double t108;
   double t113;
   double t117;
   double t120;
   double t121;
   double t122;
   double t123;
   double t149;
   double t46;
   double t47;
   double t50;
   double t63;
   double t64;
   double t2;
   double t3;
   double t152;
   double t155;
   double t158;
   double t160;
   double t96;
   double t66;
   double t67;
   double t68;
   double t165;
   double t167;
   double t168;
   double t1;
   double t69;
   double t174;
   double t13;
   double t5;
   double t89;
   double t139;
   double t179;
   double t31;
   double t32;
   double t21;
   double t22;
   double t84;
   double t15;
   double t39;
   double t75;
   double t99;
   double t72;
   double t146;
   double t100;
   double t101;
   double t102;
   t1 = UR[0];
   t2 = UR[1];
   t3 = 0.1e1 / t1;
   t5 = n[0];
   t8 = UR[2];
   t10 = n[1];
   t13 = UR[3];
   t15 = n[2];
   t18 = 0.10e1 * t2 * t3 * t5 + 0.10e1 * t8 * t3 * t10 + 0.10e1 * t13 * t3 * t15;
   t21 = UL[0];
   t22 = UL[1];
   t23 = 0.1e1 / t21;
   t24 = t22 * t23;
   t27 = UL[2];
   t28 = t27 * t23;
   t31 = UL[3];
   t32 = t31 * t23;
   t35 = 0.10e1 * t24 * t5 + 0.10e1 * t10 * t28 + 0.10e1 * t32 * t15;
   t39 = sqrt(t1 * t23);
   t44 = 0.100e1 * t39 * t2 * t3 + 0.10e1 * t24;
   t46 = 0.10e1 * t39 + 0.10e1;
   t47 = 0.1e1 / t46;
   t48 = t44 * t47;
   t50 = 0.10e1 * t48 * t5;
   t55 = 0.100e1 * t39 * t8 * t3 + 0.10e1 * t28;
   t56 = t55 * t47;
   t58 = 0.10e1 * t56 * t10;
   t63 = 0.100e1 * t39 * t13 * t3 + 0.10e1 * t32;
   t64 = t63 * t47;
   t66 = 0.10e1 * t64 * t15;
   t67 = t50 + t58 + t66;
   t68 = fabs(t67);
   t69 = t1 - t21;
   t72 = UR[4];
   t75 = 0.4e0 * t72;
   t76 = t2 * t2;
   t77 = t1 * t1;
   t78 = 0.1e1 / t77;
   t81 = t8 * t8;
   t84 = t13 * t13;
   t89 = 0.20e0 * t1 * (0.100e1 * t76 * t78 + 0.100e1 * t81 * t78 + 0.100e1 * t84 *
                        t78);
   t96 = UL[4];
   t99 = 0.4e0 * t96;
   t100 = t22 * t22;
   t101 = t21 * t21;
   t102 = 0.1e1 / t101;
   t105 = t27 * t27;
   t108 = t31 * t31;
   t113 = 0.20e0 * t21 * (0.100e1 * t100 * t102 + 0.100e1 * t105 * t102 + 0.100e1 *
                          t108 * t102);
   t117 = 0.10e1 * t39 * (0.10e1 * t72 * t3 + 0.10e1 * (t75 - t89) * t3) + 0.10e1 *
          t96 * t23 + 0.10e1 * (t99 - t113) * t23;
   t120 = t44 * t44;
   t121 = t46 * t46;
   t122 = 0.1e1 / t121;
   t123 = t120 * t122;
   t125 = t55 * t55;
   t126 = t125 * t122;
   t128 = t63 * t63;
   t129 = t128 * t122;
   t131 = 0.40e0 * t117 * t47 - 0.2000e0 * t123 - 0.2000e0 * t126 - 0.2000e0 *
          t129;
   t132 = sqrt(t131);
   t134 = fabs(t50 + t58 + t66 + t132);
   t135 = 0.5e0 * t134;
   t137 = fabs(-t50 - t58 - t66 + t132);
   t138 = 0.5e0 * t137;
   t139 = t135 + t138 - t68;
   t146 = t2 - t22;
   t149 = t8 - t27;
   t152 = t13 - t31;
   t155 = 0.4e0 * (0.500e0 * t123 + 0.500e0 * t126 + 0.500e0 * t129) * t69 - 0.40e0
          * t48 * t146 - 0.40e0 * t56 * t149 - 0.40e0 * t64 * t152 + t75 - t99;
   t158 = t139 * t155 / t131;
   t160 = t135 - t138;
   t165 = -t67 * t69 + t146 * t5 + t149 * t10 + t152 * t15;
   t167 = 0.1e1 / t132;
   t168 = t160 * t165 * t167;
   F[0] = 0.5e0 * t1 * t18 + 0.5e0 * t21 * t35 - 0.5e0 * t68 * t69 - 0.5e0 * t158 -
          0.5e0 * t168;
   t174 = t75 - t89 + t99 - t113;
   t179 = t158 + t168;
   t186 = t160 * t155 * t167 + t139 * t165;
   F[1] = 0.5e0 * t2 * t18 + 0.5e0 * t22 * t35 + 0.5e0 * t5 * t174 - 0.5e0 * t68 *
          t146 - 0.50e0 * t179 * t44 * t47 - 0.5e0 * t186 * t5;
   F[2] = 0.5e0 * t8 * t18 + 0.5e0 * t27 * t35 + 0.5e0 * t10 * t174 - 0.5e0 * t68 *
          t149 - 0.50e0 * t179 * t55 * t47 - 0.5e0 * t186 * t10;
   F[3] = 0.5e0 * t13 * t18 + 0.5e0 * t31 * t35 + 0.5e0 * t15 * t174 - 0.5e0 * t68
          * t152 - 0.50e0 * t179 * t63 * t47 - 0.5e0 * t186 * t15;
   F[4] = 0.5e0 * (0.14e1 * t72 - t89) * t18 + 0.5e0 * (0.14e1 * t96 - t113) * t35
          - 0.5e0 * t68 * (t72 - t96) - 0.50e0 * t179 * t117 * t47 - 0.5e0 * t186 * t67;
   return;
}

/******************************************************************************/

void Fi1d (double U[3], double Fi[3])
{
   double t4;
   double t1;
   double t3;
   double t6;
   Fi[0] = U[1];
   t1 = pow(Fi[0], 0.2e1);
   t3 = 0.1e1 / U[0];
   t4 = t1 * t3;
   t6 = U[2];
   Fi[1] = 0.4e1 / 0.5e1 * t4 + 0.2e1 / 0.5e1 * t6;
   Fi[2] = Fi[0] * (0.7e1 / 0.5e1 * t6 - t4 / 0.5e1) * t3;
   return;
}

/******************************************************************************/

void Fi2d (double U[4], double Fi[8])
{
   double t1;
   double t3;
   double t5;
   double t6;
   double t7;
   double t8;
   double t11;
   double t14;
   Fi[0] = U[1];
   t1 = pow(Fi[0], 0.2e1);
   t3 = 0.1e1 / U[0];
   t5 = U[3];
   t6 = 0.2e1 / 0.5e1 * t5;
   t7 = U[2];
   t8 = t7 * t7;
   t11 = (t1 + t8) * t3 / 0.5e1;
   Fi[1] = t1 * t3 + t6 - t11;
   Fi[2] = Fi[0] * t7 * t3;
   t14 = 0.7e1 / 0.5e1 * t5 - t11;
   Fi[3] = Fi[0] * t14 * t3;
   Fi[4] = t7;
   Fi[5] = Fi[2];
   Fi[6] = t8 * t3 + t6 - t11;
   Fi[7] = Fi[4] * t14 * t3;
   return;
}

/******************************************************************************/

void Fi3d (double U[5], double Fi[15])
{
   double t17;
   double t10;
   double t9;
   double t13;
   double t3;
   double t5;
   double t6;
   double t1;
   double t7;
   double t8;
   Fi[0] = U[1];
   t1 = pow(Fi[0], 0.2e1);
   t3 = 0.1e1 / U[0];
   t5 = U[4];
   t6 = 0.2e1 / 0.5e1 * t5;
   t7 = U[2];
   t8 = t7 * t7;
   t9 = U[3];
   t10 = t9 * t9;
   t13 = (t1 + t8 + t10) * t3 / 0.5e1;
   Fi[1] = t1 * t3 + t6 - t13;
   Fi[2] = Fi[0] * t7 * t3;
   Fi[3] = Fi[0] * t9 * t3;
   t17 = 0.7e1 / 0.5e1 * t5 - t13;
   Fi[4] = Fi[0] * t17 * t3;
   Fi[5] = t7;
   Fi[6] = Fi[2];
   Fi[7] = t8 * t3 + t6 - t13;
   Fi[8] = Fi[5] * t9 * t3;
   Fi[9] = Fi[5] * t17 * t3;
   Fi[10] = t9;
   Fi[11] = Fi[3];
   Fi[12] = Fi[8];
   Fi[13] = t10 * t3 + t6 - t13;
   Fi[14] = Fi[10] * t17 * t3;
   return;
}

/******************************************************************************/
void eulerF(int dim, double *U, double *F)
{
   if (dim == 1)
   {
      Fi1d(U,F);
   }
   else if (dim == 2)
   {
      Fi2d(U,F);
   }
   else if (dim == 3)
   {
      Fi3d(U,F);
   }
}
void eulerFhat(int dim, double *UR, double *UL, double *n, double *F)
{
   if (dim == 1)
   {
      roe1d(UR,UL,n,F);
   }
   else if (dim == 2)
   {
      roe2d(UR,UL,n,F);
   }
   else if (dim == 3)
   {
      roe3d(UR,UL,n,F);
   }
}