SetFactory("OpenCASCADE");

//mesh parameters
dx_divertor = DefineNumber[ 0.3, Name "parameters/dx_divertor" ];
nr_seg_th = DefineNumber[ 3, Name "parameters/nr_seg_th" ];
ni_seg_th = DefineNumber[ 5, Name "parameters/ni_seg_th" ];
dx_coil = DefineNumber[ 0.3, Name "parameters/dx_coil" ];



//first wall
Point(1)={ 6.267000, -3.046000, 0 };
Point(2)={ 7.283000, -2.257000, 0 };
Point(3)={ 7.899000, -1.342000, 0 };
Point(4)={ 8.306000, -0.421000, 0 };
Point(5)={ 8.395000, 0.633000, 0 };
Point(6)={ 8.270000, 1.681000, 0 };
Point(7)={ 7.904000, 2.464000, 0 };
Point(8)={ 7.400000, 3.179000, 0 };
Point(9)={ 6.587000, 3.894000, 0 };
Point(10)={ 5.753000, 4.532000, 0 };
Point(11)={ 4.904000, 4.712000, 0 };
Point(12)={ 4.311000, 4.324000, 0 };
Point(13)={ 4.126000, 3.582000, 0 };
Point(14)={ 4.076000, 2.566000, 0 };
Point(15)={ 4.046000, 1.549000, 0 };
Point(16)={ 4.046000, 0.533000, 0 };
Point(17)={ 4.067000, -0.484000, 0 };
Point(18)={ 4.097000, -1.500000, 0 };
Point(19)={ 4.178000, -2.306000, 0 };
Point(20)={ 3.957900, -2.538400, 0 };
Point(21)={ 4.325700, -2.651400, 0 };
Point(22)={ 4.506600, -2.941000, 0 };
Point(23)={ 4.467000, -3.280100, 0 };
Point(24)={ 4.179900, -3.884700, 0 };
Point(25)={ 4.491800, -3.909200, 0 };
Point(26)={ 4.645600, -3.746000, 0 };
Point(27)={ 4.998200, -3.741400, 0 };
Point(28)={ 5.252900, -3.985200, 0 };
Point(29)={ 5.272700, -4.263600, 0 };
Point(30)={ 5.565000, -4.555900, 0 };
Point(31)={ 5.572000, -3.896000, 0 };
Point(32)={ 5.684200, -3.526500, 0 };
Point(33)={ 5.982100, -3.282200, 0 };
Point(34)={ 6.365500, -3.244600, 0 };

Line(35)={ 1, 2 };
Line(36)={ 2, 3 };
Line(37)={ 3, 4 };
Line(38)={ 4, 5 };
Line(39)={ 5, 6 };
Line(40)={ 6, 7 };
Line(41)={ 7, 8 };
Line(42)={ 8, 9 };
Line(43)={ 9, 10 };
Line(44)={ 10, 11 };
Line(45)={ 11, 12 };
Line(46)={ 12, 13 };
Line(47)={ 13, 14 };
Line(48)={ 14, 15 };
Line(49)={ 15, 16 };
Line(50)={ 16, 17 };
Line(51)={ 17, 18 };
Line(52)={ 18, 19 };
Line(53)={ 19, 20 };
Line(54)={ 20, 21 };
Line(55)={ 21, 22 };
Line(56)={ 22, 23 };
Line(57)={ 23, 24 };
Line(58)={ 24, 25 };
Line(59)={ 25, 26 };
Line(60)={ 26, 27 };
Line(61)={ 27, 28 };
Line(62)={ 28, 29 };
Line(63)={ 29, 30 };
Line(64)={ 30, 31 };
Line(65)={ 31, 32 };
Line(66)={ 32, 33 };
Line(67)={ 33, 34 };
Line(68)={ 34, 1 };

Transfinite Curve { 35 } = ni_seg_th;
Transfinite Curve { 36 } = ni_seg_th;
Transfinite Curve { 37 } = ni_seg_th;
Transfinite Curve { 38 } = ni_seg_th;
Transfinite Curve { 39 } = ni_seg_th;
Transfinite Curve { 40 } = ni_seg_th;
Transfinite Curve { 41 } = ni_seg_th;
Transfinite Curve { 42 } = ni_seg_th;
Transfinite Curve { 43 } = ni_seg_th;
Transfinite Curve { 44 } = ni_seg_th;
Transfinite Curve { 45 } = ni_seg_th;
Transfinite Curve { 46 } = ni_seg_th;
Transfinite Curve { 47 } = ni_seg_th;
Transfinite Curve { 48 } = ni_seg_th;
Transfinite Curve { 49 } = ni_seg_th;
Transfinite Curve { 50 } = ni_seg_th;
Transfinite Curve { 50 } = ni_seg_th;
Transfinite Curve { 51 } = ni_seg_th;
Transfinite Curve { 52 } = ni_seg_th;

Line Loop(69)={35,
36,
37,
38,
39,
40,
41,
42,
43,
44,
45,
46,
47,
48,
49,
50,
51,
52,
53,
54,
55,
56,
57,
58,
59,
60,
61,
62,
63,
64,
65,
66,
67,
68 };
Plane Surface(70) = { 69 };

//solenoids
Point(35) = {0.946000, -5.415000, 0};
Point(36) = {2.446000, -5.415000, 0};
Point(37) = {0.946000, -3.606700, 0};
Point(38) = {2.446000, -3.606700, 0};
Point(39) = {0.946000, -1.798300, 0};
Point(40) = {2.446000, -1.798300, 0};
Point(41) = {0.946000, 1.818300, 0};
Point(42) = {2.446000, 1.818300, 0};
Point(43) = {0.946000, 3.626700, 0};
Point(44) = {2.446000, 3.626700, 0};
Point(45) = {0.946000, 5.435000, 0};
Point(46) = {2.446000, 5.435000, 0};
Line(70) = {35, 37};
Line(71) = {37, 39};
Line(72) = {39, 41};
Line(73) = {41, 43};
Line(74) = {43, 45};
Line(75) = {45, 46};
Line(76) = {46, 44};
Line(77) = {44, 42};
Line(78) = {42, 40};
Line(79) = {40, 38};
Line(80) = {38, 36};
Line(81) = {36, 35};
Line(82) = {38, 37};
Line(83) = {40, 39};
Line(84) = {42, 41};
Line(85) = {44, 43};

Curve Loop(70) = {81, 70, -82, 80};
Curve Loop(71) = {82, 71, -83, 79};
Curve Loop(72) = {83, 72, -84, 78};
Curve Loop(73) = {84, 73, -85, 77};
Curve Loop(74) = {85, 74, 75, 76};

Plane Surface(71) = {70};
Plane Surface(72) = {71};
Plane Surface(73) = {72};
Plane Surface(74) = {73};
Plane Surface(75) = {74};

// coils
Rectangle(76) = {2.943100, 6.824100, 0, 2.000000, 1.500000, 0};
Rectangle(77) = {7.285100, 5.789800, 0, 2.000000, 1.500000, 0};
Rectangle(78) = {10.991900, 2.525200, 0, 2.000000, 1.500000, 0};
Rectangle(79) = {10.963000, -2.983600, 0, 2.000000, 1.500000, 0};
Rectangle(80) = {7.390800, -7.476900, 0, 2.000000, 1.500000, 0};
Rectangle(81) = {3.334000, -8.216500, 0, 2.000000, 1.500000, 0};

//Transfinite Curve { 70 } = 4;
//Transfinite Curve { 71 } = 4;
//Transfinite Curve { 72 } = 8;
//Transfinite Curve { 73 } = 4;
//Transfinite Curve { 74 } = 4;
//Transfinite Curve { 75 } = 4;
//Transfinite Curve { 76 } = 4;
//Transfinite Curve { 77 } = 4;
//Transfinite Curve { 78 } = 8;
//Transfinite Curve { 79 } = 4;
//Transfinite Curve { 80 } = 4;
//Transfinite Curve { 81 } = 4;
//Transfinite Curve { 82 } = 4;
//Transfinite Curve { 83 } = 4;
//Transfinite Curve { 84 } = 4;
//Transfinite Curve { 85 } = 4;
//
//Transfinite Curve { 86 } = 4;
//Transfinite Curve { 87 } = 3;
//Transfinite Curve { 88 } = 4;
//Transfinite Curve { 89 } = 3;
//Transfinite Curve { 90 } = 4;
//Transfinite Curve { 91 } = 3;
//Transfinite Curve { 92 } = 4;
//Transfinite Curve { 93 } = 3;
//Transfinite Curve { 94 } = 4;
//Transfinite Curve { 95 } = 3;
//Transfinite Curve { 96 } = 4;
//Transfinite Curve { 97 } = 3;
//Transfinite Curve { 98 } = 4;
//Transfinite Curve { 99 } = 3;
//Transfinite Curve { 100 } = 4;
//Transfinite Curve { 101 } = 3;
//Transfinite Curve { 102 } = 4;
//Transfinite Curve { 103 } = 3;
//Transfinite Curve { 104 } = 4;
//Transfinite Curve { 105 } = 3;
//Transfinite Curve { 106 } = 4;
//Transfinite Curve { 107 } = 3;
//Transfinite Curve { 108 } = 4;
//Transfinite Curve { 109 } = 3;

//outer boundary
Point(71) = {0.0, 16.000000, 0, 1.0};
Point(72) = {0.0, 0.000000, 0, 1.0};
Point(73) = {0.0, -16.000000, 0, 1.0};
Circle(157) = {71, 72, 73};
Line(158) = {71, 73};

//set overall mesh size
//MeshSize {:} = 0.5;

Curve Loop(81) = {158, -157};
Curve Loop(82) = {70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81};
Curve Loop(83) = {89, 86, 87, 88};
Curve Loop(84) = {93, 90, 91, 92};
Curve Loop(85) = {97, 94, 95, 96};
Curve Loop(86) = {101, 98, 99, 100};
Curve Loop(87) = {103, 104, 105, 102};
Curve Loop(88) = {107, 108, 109, 106};
Curve Loop(89) = {51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};

//Outer VV
Point(200) = { 7.535451436229712, -4.650239293829089, 0, dx_divertor };
Point(201) = { 7.913701320278686, -4.170861562734443, 0, dx_divertor };
Point(202) = { 8.176060954403763, -3.67630210470504, 0, dx_divertor };
Point(203) = { 8.372525867961073, -3.231053071461125, 0, dx_divertor };
Point(204) = { 8.541890903883795, -2.8472518503504465, 0, dx_divertor };
Point(205) = { 8.689312661902681, -2.512979326568495, 0, dx_divertor };
Point(206) = { 8.923782112397348, -1.9815566331543675, 0, dx_divertor };
Point(207) = { 9.07142916908774, -1.6470000000000002, 0, dx_divertor };
Point(208) = { 9.205941080766996, -1.342, 0, dx_divertor };
Point(209) = { 9.341402672864612, -1.035, 0, dx_divertor };
Point(210) = { 9.46482393442623, -0.7280000000000001, 0, dx_divertor };
Point(211) = { 9.560626656067832, -0.421, 0, dx_divertor };
Point(212) = { 9.634640478686233, -0.06966666666666665, 0, dx_divertor };
Point(213) = { 9.673670132781513, 0.2816666666666667, 0, dx_divertor };
Point(214) = { 9.67957987616099, 0.633, 0, dx_divertor };
Point(215) = { 9.652711024440977, 0.9823333333333334, 0, dx_divertor };
Point(216) = { 9.595430716606806, 1.3316666666666668, 0, dx_divertor };
Point(217) = { 9.514129229177444, 1.681, 0, dx_divertor };
Point(218) = { 9.43718765171504, 1.9420000000000002, 0, dx_divertor };
Point(219) = { 9.347166025641025, 2.203, 0, dx_divertor };
Point(220) = { 9.242995107201759, 2.464, 0, dx_divertor };
Point(221) = { 9.133479515828679, 2.7023333333333333, 0, dx_divertor };
Point(222) = { 9.011128254690165, 2.9406666666666665, 0, dx_divertor };
Point(223) = { 8.835457058003799, 3.2406758079769005, 0, dx_divertor };
Point(224) = { 8.533659037519243, 3.6770185094497063, 0, dx_divertor };
Point(225) = { 8.138832749660736, 4.14031320067642, 0, dx_divertor };
Point(226) = { 7.65346156207998, 4.595619881373228, 0, dx_divertor };
Point(227) = { 7.1258873174726105, 4.987284702382271, 0, dx_divertor };
Point(228) = { 6.566665347426472, 5.310405112679715, 0, dx_divertor };
Point(229) = { 6.0236773332722136, 5.5432146581030315, 0, dx_divertor };
Point(230) = { 5.5421634569104565, 5.647860646207973, 0, dx_divertor };
Point(231) = { 5.071201552229415, 5.6532457322146366, 0, dx_divertor };
Point(232) = { 4.654662224200761, 5.58372652423312, 0, dx_divertor };
Point(233) = { 4.3176390369285, 5.459265425996912, 0, dx_divertor };
Point(234) = { 3.9977560140570487, 5.263410841429764, 0, dx_divertor };
Point(235) = { 3.724383804367699, 5.008271509850522, 0, dx_divertor };
Point(236) = { 3.518218884110441, 4.7207314529573825, 0, dx_divertor };
Point(237) = { 3.352276849616509, 4.351659596359502, 0, dx_divertor };
Point(238) = { 3.2692088085474618, 3.9204006323076657, 0, dx_divertor };
Point(239) = { 3.2622, 3.3454528220524877, 0, dx_divertor };
Point(240) = { 3.2622, 2.904666666666666, 0, dx_divertor };
Point(241) = { 3.2622, 2.566, 0, dx_divertor };
Point(242) = { 3.2622, 2.227, 0, dx_divertor };
Point(243) = { 3.2622, 1.888, 0, dx_divertor };
Point(244) = { 3.2622, 1.549, 0, dx_divertor };
Point(245) = { 3.2622, 1.2103333333333333, 0, dx_divertor };
Point(246) = { 3.2622, 0.8716666666666667, 0, dx_divertor };
Point(247) = { 3.2622, 0.533, 0, dx_divertor };
Point(248) = { 3.2622, 0.19400000000000003, 0, dx_divertor };
Point(249) = { 3.2622, -0.145, 0, dx_divertor };
Point(250) = { 3.2622, -0.484, 0, dx_divertor };
Point(251) = { 3.2622, -0.8226666666666667, 0, dx_divertor };
Point(252) = { 3.2622, -1.1613333333333333, 0, dx_divertor };
Point(253) = { 3.2622, -1.5, 0, dx_divertor };
Point(254) = { 3.2622, -1.8353333333333335, 0, dx_divertor };
Point(255) = { 3.2622, -2.362341635122926, 0, dx_divertor };
Point(256) = { 3.2622, -2.7056250988391275, 0, dx_divertor };
Point(257) = { 3.2622, -3.137789112671205, 0, dx_divertor };
Point(258) = { 3.2622, -3.6154606815134698, 0, dx_divertor };
Point(259) = { 3.308791973692945, -4.130581773619768, 0, dx_divertor };
Point(260) = { 3.486932842400425, -4.613324211628292, 0, dx_divertor };
Point(261) = { 3.786544259072894, -5.028120095892088, 0, dx_divertor };
Point(262) = { 4.183525057315066, -5.349659336498687, 0, dx_divertor };
Point(263) = { 4.65225018100961, -5.553334379907796, 0, dx_divertor };
Point(264) = { 5.1594489131835655, -5.626202856936588, 0, dx_divertor };
Point(265) = { 5.663199286239529, -5.6001888083146065, 0, dx_divertor };
Point(266) = { 6.141643054910025, -5.49757176569245, 0, dx_divertor };
Point(267) = { 6.582844713589526, -5.331787983376671, 0, dx_divertor };
Point(268) = { 6.978455038239984, -5.111283499786843, 0, dx_divertor };

Line(200) = { 200, 201 };
Transfinite Curve { 200 } = nr_seg_th;
Line(201) = { 201, 202 };
Transfinite Curve { 201 } = nr_seg_th;
Line(202) = { 202, 203 };
Transfinite Curve { 202 } = nr_seg_th;
Line(203) = { 203, 204 };
Transfinite Curve { 203 } = nr_seg_th;
Line(204) = { 204, 205 };
Transfinite Curve { 204 } = nr_seg_th;
Line(205) = { 205, 206 };
Transfinite Curve { 205 } = nr_seg_th;
Line(206) = { 206, 207 };
Transfinite Curve { 206 } = nr_seg_th;
Line(207) = { 207, 208 };
Transfinite Curve { 207 } = nr_seg_th;
Line(208) = { 208, 209 };
Transfinite Curve { 208 } = nr_seg_th;
Line(209) = { 209, 210 };
Transfinite Curve { 209 } = nr_seg_th;
Line(210) = { 210, 211 };
Transfinite Curve { 210 } = nr_seg_th;
Line(211) = { 211, 212 };
Transfinite Curve { 211 } = nr_seg_th;
Line(212) = { 212, 213 };
Transfinite Curve { 212 } = nr_seg_th;
Line(213) = { 213, 214 };
Transfinite Curve { 213 } = nr_seg_th;
Line(214) = { 214, 215 };
Transfinite Curve { 214 } = nr_seg_th;
Line(215) = { 215, 216 };
Transfinite Curve { 215 } = nr_seg_th;
Line(216) = { 216, 217 };
Transfinite Curve { 216 } = nr_seg_th;
Line(217) = { 217, 218 };
Transfinite Curve { 217 } = nr_seg_th;
Line(218) = { 218, 219 };
Transfinite Curve { 218 } = nr_seg_th;
Line(219) = { 219, 220 };
Transfinite Curve { 219 } = nr_seg_th;
Line(220) = { 220, 221 };
Transfinite Curve { 220 } = nr_seg_th;
Line(221) = { 221, 222 };
Transfinite Curve { 221 } = nr_seg_th;
Line(222) = { 222, 223 };
Transfinite Curve { 222 } = nr_seg_th;
Line(223) = { 223, 224 };
Transfinite Curve { 223 } = nr_seg_th;
Line(224) = { 224, 225 };
Transfinite Curve { 224 } = nr_seg_th;
Line(225) = { 225, 226 };
Transfinite Curve { 225 } = nr_seg_th;
Line(226) = { 226, 227 };
Transfinite Curve { 226 } = nr_seg_th;
Line(227) = { 227, 228 };
Transfinite Curve { 227 } = nr_seg_th;
Line(228) = { 228, 229 };
Transfinite Curve { 228 } = nr_seg_th;
Line(229) = { 229, 230 };
Transfinite Curve { 229 } = nr_seg_th;
Line(230) = { 230, 231 };
Transfinite Curve { 230 } = nr_seg_th;
Line(231) = { 231, 232 };
Transfinite Curve { 231 } = nr_seg_th;
Line(232) = { 232, 233 };
Transfinite Curve { 232 } = nr_seg_th;
Line(233) = { 233, 234 };
Transfinite Curve { 233 } = nr_seg_th;
Line(234) = { 234, 235 };
Transfinite Curve { 234 } = nr_seg_th;
Line(235) = { 235, 236 };
Transfinite Curve { 235 } = nr_seg_th;
Line(236) = { 236, 237 };
Transfinite Curve { 236 } = nr_seg_th;
Line(237) = { 237, 238 };
Transfinite Curve { 237 } = nr_seg_th;
Line(238) = { 238, 239 };
Transfinite Curve { 238 } = nr_seg_th;
Line(239) = { 239, 240 };
Transfinite Curve { 239 } = nr_seg_th;
Line(240) = { 240, 241 };
Transfinite Curve { 240 } = nr_seg_th;
Line(241) = { 241, 242 };
Transfinite Curve { 241 } = nr_seg_th;
Line(242) = { 242, 243 };
Transfinite Curve { 242 } = nr_seg_th;
Line(243) = { 243, 244 };
Transfinite Curve { 243 } = nr_seg_th;
Line(244) = { 244, 245 };
Transfinite Curve { 244 } = nr_seg_th;
Line(245) = { 245, 246 };
Transfinite Curve { 245 } = nr_seg_th;
Line(246) = { 246, 247 };
Transfinite Curve { 246 } = nr_seg_th;
Line(247) = { 247, 248 };
Transfinite Curve { 247 } = nr_seg_th;
Line(248) = { 248, 249 };
Transfinite Curve { 248 } = nr_seg_th;
Line(249) = { 249, 250 };
Transfinite Curve { 249 } = nr_seg_th;
Line(250) = { 250, 251 };
Transfinite Curve { 250 } = nr_seg_th;
Line(251) = { 251, 252 };
Transfinite Curve { 251 } = nr_seg_th;
Line(252) = { 252, 253 };
Transfinite Curve { 252 } = nr_seg_th;
Line(253) = { 253, 254 };
Transfinite Curve { 253 } = nr_seg_th;
Line(254) = { 254, 255 };
Transfinite Curve { 254 } = nr_seg_th;
Line(255) = { 255, 256 };
Transfinite Curve { 255 } = nr_seg_th;
Line(256) = { 256, 257 };
Transfinite Curve { 256 } = nr_seg_th;
Line(257) = { 257, 258 };
Transfinite Curve { 257 } = nr_seg_th;
Line(258) = { 258, 259 };
Transfinite Curve { 258 } = nr_seg_th;
Line(259) = { 259, 260 };
Transfinite Curve { 259 } = nr_seg_th;
Line(260) = { 260, 261 };
Transfinite Curve { 260 } = nr_seg_th;
Line(261) = { 261, 262 };
Transfinite Curve { 261 } = nr_seg_th;
Line(262) = { 262, 263 };
Transfinite Curve { 262 } = nr_seg_th;
Line(263) = { 263, 264 };
Transfinite Curve { 263 } = nr_seg_th;
Line(264) = { 264, 265 };
Transfinite Curve { 264 } = nr_seg_th;
Line(265) = { 265, 266 };
Transfinite Curve { 265 } = nr_seg_th;
Line(266) = { 266, 267 };
Transfinite Curve { 266 } = nr_seg_th;
Line(267) = { 267, 268 };
Transfinite Curve { 267 } = nr_seg_th;
Line(268) = { 268, 200 };
Transfinite Curve { 268 } = nr_seg_th;

Line Loop(201) = { 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268};

Curve Loop(199) = {219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218};

Plane Surface(82) = {81, 82, 83, 84, 85, 86, 87, 88, 199};
Plane Surface(202) = { 201, 89 };

Physical Surface("interior", 2000) = {82};
Physical Surface("coil1", 832) = {71};
Physical Surface("coil2", 833) = {72};
Physical Surface("coil3", 834) = {73};
Physical Surface("coil4", 835) = {74};
Physical Surface("coil5", 836) = {75};
Physical Surface("coil6", 837) = {76};
Physical Surface("coil7", 838) = {77};
Physical Surface("coil8", 839) = {78};
Physical Surface("coil9", 840) = {79};
Physical Surface("coil10", 841) = {80};
Physical Surface("coil11", 842) = {81};
Physical Surface("limiter", 1000) = {70};
Physical Surface("OuterVV", 1001) = {202};

//physical boundary
Physical Curve("boundary", 831) = {157};
Physical Curve("axis", 900) = {158};
Mesh.MshFileVersion = 2.2;
