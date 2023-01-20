import matplotlib.pyplot as plt

# SetFactory("OpenCASCADE");
# Point(1)={ 1.32, 0.0, 0 };
# Point(2)={ 1.319929875827836, 0.00856278624103039, 0 };
# Point(3)={ 1.3197195494079197, 0.01712345082879075, 0 };
# Point(4)={ 1.3193691589806256, 0.025679872635705904, 0 };
# ...
# Point(399)={ 1.3199058345166919, -0.009922589578139624, 0 };
# Point(400)={ 1.3199982311024445, -0.001359998583334122, 0 };
# 
# Line(401)={ 1, 2 };
# Line(402)={ 2, 3 };
# ...
# Line(799)={ 399, 400 };
# Line(800)={ 400, 1 };
#
# Line Loop(801)={ 
# 401,
# 402,
# ...
# 799,
# 800 };
# Plane Surface(802) = { 801 };

separated_data = 'separated_file.data'
write_file = 'meshes/iter_gen.geo'

r0 = [1.696, 1.696, 1.696, 1.696, 1.696,
      3.9431, 8.2851, 11.9919, 11.9630, 8.3908, 4.3340]
z0 = [-5.415, -3.6067, -1.7983, 1.8183, 3.6267,
      7.5741, 6.5398, 3.2752, -2.2336, -6.7269, -7.4665]

r_sn = 1.5
z_sn = 1
r_c = 2
z_c = 1.5

dr = [ r_sn, r_sn, r_sn, r_sn, r_sn,
       r_c, r_c, r_c, r_c, r_c, r_c]
dz = [ z_sn, z_sn, z_sn, z_sn, z_sn,
       z_c, z_c, z_c, z_c, z_c, z_c]
rr = [r+d/2 for r, d in zip(r0, dr)]
zr = [z+d/2 for z, d in zip(z0, dz)]
rl = [r-d/2 for r, d in zip(r0, dr)]
zl = [z-d/2 for z, d in zip(z0, dz)]

zl[:5] = [-5.415, -3.6067, -1.7983, 1.8183, 3.6267]
zr[:5] = [-3.6067, -1.7983, 1.8183, 3.6267, 5.435]

dz = [a-b for a, b in zip(zr, zl)]

R = 16

with open(separated_data, 'r') as fid:

    for line in fid:
        if "rlim(i),zlim(i)" in line:
            line = fid.readline()
            data = line.replace(" \n", "").split(" ")
            data = [eval(a) for a in data]
            rlim = data[::2]
            zlim = data[1::2]
            # plt.plot(rlim, zlim)
            # plt.show()

            with open(write_file, 'w') as gid:
                stuff = 'SetFactory("OpenCASCADE");\n\n'
                count = 1
                for i in range(len(rlim)):
                    stuff += "Point(%d)={ %f, %f, 0 };\n" % (count, rlim[i], zlim[i])
                    count += 1
                stuff += "\n"
                first_line = count
                for i in range(len(rlim)):
                    oth = i + 2
                    if oth > len(rlim):
                        oth -= len(rlim)
                    
                    stuff += "Line(%d)={ %d, %d };\n" % (count, i+1, oth)
                    count += 1
                stuff += "\n"
                stuff += "Line Loop(%d)={\n" % (count)
                key = count
                count = first_line
                for i in range(len(rlim)):
                    stuff += "%d,\n" % (count)
                    count += 1
                stuff = stuff[:-2]
                stuff += " };\n\n"
                stuff += "Plane Surface(%d) = { %d };\n" % (key+1, key)

                count = key+2

                # solenoids
                count_ = first_line
                for i in range(5):
                    stuff += "Point(%d) = {%f, %f, 0};\n" % (count_, rl[i], zl[i])
                    count_ += 1
                    stuff += "Point(%d) = {%f, %f, 0};\n" % (count_, rr[i], zl[i])
                    count_ += 1
                stuff += "Point(%d) = {%f, %f, 0};\n" % (count_, rl[i], zr[i])
                count_ += 1
                stuff += "Point(%d) = {%f, %f, 0};\n" % (count_, rr[i], zr[i])
                count_ += 1

                stuff += "Line(113) = {57, 59};\n"
                stuff += "Line(114) = {59, 61};\n"
                stuff += "Line(115) = {61, 63};\n"
                stuff += "Line(116) = {63, 65};\n"
                stuff += "Line(117) = {65, 67};\n"
                stuff += "Line(118) = {67, 68};\n"
                stuff += "Line(119) = {68, 66};\n"
                stuff += "Line(120) = {66, 64};\n"
                stuff += "Line(121) = {64, 62};\n"
                stuff += "Line(122) = {62, 60};\n"
                stuff += "Line(123) = {60, 58};\n"
                stuff += "Line(124) = {58, 57};\n"
                stuff += "Line(125) = {60, 59};\n"
                stuff += "Line(126) = {62, 61};\n"
                stuff += "Line(127) = {64, 63};\n"
                stuff += "Line(128) = {66, 65};\n"

                stuff += "Curve Loop(114) = {124, 113, -125, 123};\n"
                stuff += "Plane Surface(115) = {114};\n"
                stuff += "Curve Loop(115) = {114, -126, 122, 125};\n"
                stuff += "Plane Surface(116) = {115};\n"
                stuff += "Curve Loop(116) = {126, 115, -127, 121};\n"
                stuff += "Plane Surface(117) = {116};\n"
                stuff += "Curve Loop(117) = {127, 116, -128, 120};\n"
                stuff += "Plane Surface(118) = {117};\n"
                stuff += "Curve Loop(118) = {128, 117, 118, 119};\n"
                stuff += "Plane Surface(119) = {118};\n"

                count = 120
                for i in range(5, len(rl)):
                    stuff += "Rectangle(%d) = {%f, %f, 0, %f, %f, 0};\n" % (count, rl[i], zl[i], dr[i], dz[i])
                    count += 1

                count = 93
                stuff += "Point(%d) = {0.0, %f, 0, 1.0};\n" % (count, R)
                stuff += "Point(%d) = {0.0, %f, 0, 1.0};\n" % (count+1, 0)
                stuff += "Point(%d) = {0.0, %f, 0, 1.0};\n" % (count+2, -R)
                stuff += "Circle(157) = {%d, %d, %d};\n" % (count, count+1, count+2)

                stuff += "\n"

                stuff += "Line(158) = {95, 93};\n"
                stuff += "Curve Loop(125) = {158, 157};\n"
                stuff += "Curve Loop(126) = {123, 124, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122};\n"
                stuff += "Curve Loop(127) = {132, 129, 130, 131};\n"
                stuff += "Curve Loop(128) = {136, 133, 134, 135};\n"
                stuff += "Curve Loop(129) = {140, 137, 138, 139};\n"
                stuff += "Curve Loop(130) = {144, 141, 142, 143};\n"
                stuff += "Curve Loop(131) = {148, 145, 146, 147};\n"
                stuff += "Curve Loop(132) = {152, 149, 150, 151};\n"
                stuff += "Curve Loop(133) = {72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71};\n"
                stuff += "Plane Surface(126) = {125, 126, 127, 128, 129, 130, 131, 132, 133};\n"

                stuff += 'Physical Surface("interior", 2000) = {126};\n'
                for d in range(1, 12):
                    stuff += 'Physical Surface("coil%d", %d) = {%d};\n' % (d, 831+d, 114+d)
                stuff += 'Physical Surface("limiter", 1000) = {114};\n'
                stuff += 'Physical Curve("boundary", 831) = {157};\n'
                stuff += 'Physical Curve("axis", 900) = {158};\n'


                gid.write(stuff)

