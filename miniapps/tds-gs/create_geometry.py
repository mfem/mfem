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

rl = [ 1.0, 3.0, 6.0, 8.0,  8.0,  6.0,  3.0]
zl = [-8.5, 7.0, 6.5, 5.0, -5.0, -6.5, -7.0]
dr = [ 0.5, 0.5, 0.5, 0.5,  0.5,  0.5,  0.5]
dz = [ -zl[0]*2, 0.5, 0.5, 0.5,  -0.5,  -0.5,  -0.5]
rr = [r+d for r, d in zip(rl, dr)]
zr = [z+d for z, d in zip(zl, dz)]
R = 12

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
                for i in range(len(rl)):
                    stuff += "Rectangle(%d) = {%f, %f, 0, %f, %f, 0};\n" % (count, rl[i], zl[i], dr[i], dz[i])
                    count += 1

                count = 85
                stuff += "Point(%d) = {0.0, %f, 0, 1.0};\n" % (count, R)
                stuff += "Point(%d) = {0.0, %f, 0, 1.0};\n" % (count+1, 0)
                stuff += "Point(%d) = {0.0, %f, 0, 1.0};\n" % (count+2, -R)
                stuff += "Circle(141) = {%d, %d, %d};\n" % (count, count+1, count+2)

                stuff += "\n"

                stuff += "Line(142) = {87, 85};\n"

                stuff += "Curve Loop(121) = {142, 141};\n"
                stuff += "Curve Loop(122) = {116, 113, 114, 115};\n"
                stuff += "Curve Loop(123) = {120, 117, 118, 119};\n"
                stuff += "Curve Loop(124) = {124, 121, 122, 123};\n"
                stuff += "Curve Loop(125) = {127, 128, 125, 126};\n"
                stuff += "Curve Loop(126) = {129, 130, 131, 132};\n"
                stuff += "Curve Loop(127) = {134, 135, 136, 133};\n"
                stuff += "Curve Loop(128) = {138, 139, 140, 137};\n"
                stuff += "Curve Loop(129) = {73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72};\n"
                stuff += "Plane Surface(122) = {121, 122, 123, 124, 125, 126, 127, 128, 129};\n"
                stuff += 'Physical Surface("interior", 2000) = {122};\n'
                stuff += 'Physical Surface("coil1", 832) = {115};\n'
                stuff += 'Physical Surface("coil2", 833) = {116};\n'
                stuff += 'Physical Surface("coil3", 834) = {117};\n'
                stuff += 'Physical Surface("coil4", 835) = {118};\n'
                stuff += 'Physical Surface("coil5", 836) = {119};\n'
                stuff += 'Physical Surface("coil6", 837) = {120};\n'
                stuff += 'Physical Surface("coil7", 838) = {121};\n'
                stuff += 'Physical Surface("limiter", 1000) = {114};\n'
                stuff += 'Physical Curve("boundary", 831) = {141};\n'
                stuff += 'Physical Curve("axis", 900) = {142};\n'
                
                # Line(142) = {87, 85};
                # Curve Loop(121) = {141, 142};
                # Curve Loop(122) = {114, 115, 116, 113};
                # Curve Loop(123) = {119, 120, 117, 118};
                # Curve Loop(124) = {123, 124, 121, 122};
                # Curve Loop(125) = {127, 128, 125, 126};
                # Curve Loop(126) = {129, 130, 131, 132};
                # Curve Loop(127) = {133, 134, 135, 136};
                # Curve Loop(128) = {137, 138, 139, 140};
                # Curve Loop(129) = {57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112};
                # Surface(122) = {121, 122, 123, 124, 125, 126, 127, 128, 129};
                # Physical Surface("coil1", 143) = {115};
                # Physical Surface("coil2", 144) = {116};
                # Physical Surface("coil3", 145) = {117};
                # Physical Surface("coil4", 146) = {118};
                # Physical Surface("coil5", 147) = {119};
                # Physical Surface("coil6", 148) = {120};
                # Physical Surface("coil7", 149) = {121};
                # Physical Surface("limiter", 150) = {114};
                
                
                gid.write(stuff)

