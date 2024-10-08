using LinearAlgebra
using DelimitedFiles

Q = 4
D = 3
space_dim = 3;
num_trial_dof = D^3;
test_vdim = 1
output_op_dim = 3;

R_data = Int.(readdlm("/Users/andrej1/repos/mfem/build-debug/r_mat.mtx", ' ', Float64))
R_N = maximum(R_data[:, 1:2])
R = zeros(Float64, (R_N, R_N))
for e = 1:size(R_data, 1)
    rij = R_data[e, :, :]
    R[rij[1], rij[2]] = 1.0
end

u_l = reshape(Float64[2.345 3.345 4.345 2.345 2.345 4.595 5.595 2.345 2.845 3.845 3.345 2.345 3.47 5.095 3.97 2.345 2.345 3.97 4.97 2.345 3.095 3.1575 4.47 3.6575 2.345 3.72 3.4075], (num_trial_dof, test_vdim))
u_e = R * u_l
x_l = transpose(reshape(Float64[0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1 0.5 0 0 1 0.5 0 0.5 1 0 0 0.5 0 0.5 0 1 1 0.5 1 0.5 1 1 0 0.5 1 0 0 0.5 1 0 0.5 1 1 0.5 0 1 0.5 0.5 0.5 0 0.5 0 0.5 1 0.5 0.5 0.5 1 0.5 0 0.5 0.5 0.5 0.5 1 0.5 0.5 0.5], (space_dim, num_trial_dof)))
x_e = R * x_l

w_qp = [0.00526143468632 0.00986393947438 0.00986393947438 0.00526143468632 0.00986393947438 0.0184925420071 0.0184925420071 0.00986393947438 0.00986393947438 0.0184925420071 0.0184925420071 0.00986393947438 0.00526143468632 0.00986393947438 0.00986393947438 0.00526143468632 0.00986393947438 0.0184925420071 0.0184925420071 0.00986393947438 0.0184925420071 0.0346691208692 0.0346691208692 0.0184925420071 0.0184925420071 0.0346691208692 0.0346691208692 0.0184925420071 0.00986393947438 0.0184925420071 0.0184925420071 0.00986393947438 0.00986393947438 0.0184925420071 0.0184925420071 0.00986393947438 0.0184925420071 0.0346691208692 0.0346691208692 0.0184925420071 0.0184925420071 0.0346691208692 0.0346691208692 0.0184925420071 0.00986393947438 0.0184925420071 0.0184925420071 0.00986393947438 0.00526143468632 0.00986393947438 0.00986393947438 0.00526143468632 0.00986393947438 0.0184925420071 0.0184925420071 0.00986393947438 0.00986393947438 0.0184925420071 0.0184925420071 0.00986393947438 0.00526143468632 0.00986393947438 0.00986393947438 0.00526143468632]
B = reshape([0.80134602937 0.227784076791 -0.112196966794 -0.0597902822241 0.258444252854 0.884412890003 0.884412890003 0.258444252854 -0.0597902822241 -0.112196966794 0.227784076791 0.80134602937], (Q, D))
G = reshape([-2.72227262319 -1.67996208717 -0.32003791283 0.722272623188 3.44454524638 1.35992417434 -1.35992417434 -3.44454524638 -0.722272623188 0.32003791283 1.67996208717 2.72227262319], (Q, D))

function interpolate_value(f_e)
    vdim = size(f_e, 2)
    f_e = reshape(f_e, (D, D, D, vdim))
    f_qp = zeros(vdim, Q, Q, Q)
    s1 = zeros(Float64, D, D, Q)
    s2 = zeros(Float64, D, Q, Q)

    for vd = 1:vdim
        for dz = 1:D
            for dy = 1:D
                for qx = 1:Q
                    acc = 0.0
                    for dx = 1:D
                        acc += f_e[dx, dy, dz, vd] * B[qx, dx]
                    end
                    s1[dz, dy, qx] = acc
                end
            end
        end

        for dz = 1:D
            for qx = 1:Q
                for qy = 1:Q
                    acc = 0.0
                    for dy = 1:D
                        acc += s1[dz, dy, qx] * B[qy, dy]
                    end
                    s2[dz, qy, qx] = acc
                end
            end
        end

        for qz = 1:Q
            for qy = 1:Q
                for qx = 1:Q
                    acc = 0.0
                    for dz = 1:D
                        acc += s2[dz, qy, qx] * B[qz, dz]
                    end
                    f_qp[vd, qx, qy, qz] = acc
                end
            end
        end
    end

    return f_qp
end

function interpolate_grad(f_e)
    vdim = size(f_e, 2)
    f_e = reshape(f_e, (D, D, D, vdim))
    dudxi_qp = zeros(vdim, space_dim, Q, Q, Q)

    s1 = zeros(Float64, D, D, Q)
    s2 = zeros(Float64, D, D, Q)
    s3 = zeros(Float64, D, Q, Q)
    s4 = zeros(Float64, D, Q, Q)
    s5 = zeros(Float64, D, Q, Q)
    uvw = zeros(Float64, 3)

    for vd = 1:vdim
        for dz = 1:D
            for dy = 1:D
                for qx = 1:Q
                    uvw .= 0.0
                    for dx = 1:D
                        f = f_e[dx, dy, dz, vd]
                        uvw[1] += f * B[qx, dx]
                        uvw[2] += f * G[qx, dx]
                    end
                    s1[dz, dy, qx] = uvw[1]
                    s2[dz, dy, qx] = uvw[2]
                end
            end
        end

        for dz = 1:D
            for qy = 1:Q
                for qx = 1:Q
                    uvw .= 0.0
                    for dy = 1:D
                        uvw[1] += s2[dz, dy, qx] * B[qy, dy]
                        uvw[2] += s1[dz, dy, qx] * G[qy, dy]
                        uvw[3] += s1[dz, dy, qx] * B[qy, dy]
                    end
                    s3[dz, qy, qx] = uvw[1]
                    s4[dz, qy, qx] = uvw[2]
                    s5[dz, qy, qx] = uvw[3]
                end
            end
        end

        for qz = 1:Q
            for qy = 1:Q
                for qx = 1:Q
                    uvw .= 0.0
                    for dz = 1:D
                        uvw[1] += s3[dz, qy, qx] * B[qz, dz]
                        uvw[2] += s4[dz, qy, qx] * B[qz, dz]
                        uvw[3] += s5[dz, qy, qx] * G[qz, dz]
                    end
                    dudxi_qp[vd, 1, qx, qy, qz] = uvw[1]
                    dudxi_qp[vd, 2, qx, qy, qz] = uvw[2]
                    dudxi_qp[vd, 3, qx, qy, qz] = uvw[3]
                end
            end
        end
    end

    return dudxi_qp
end

function integrate_value(f_qp)
    r = zeros(Float64, D, D, D, test_vdim)
    f_qp = reshape(f_qp, (test_vdim, output_op_dim, Q, Q, Q))
    s1 = zeros(Float64, Q, Q, D)
    s2 = zeros(Float64, Q, D, D)

    for vd = 1:test_vdim
        for qy = 1:Q
            for dx = 1:D
                for qz = 1:Q
                    acc = 0.0
                    for qx = 1:Q
                        acc += f_qp[test_vdim, 1, qx, qy, qz] * B[qx, dx]
                    end
                    s1[qz, qy, dx] = acc
                end
            end
        end

        # for i = 1:Q
        #     for j = 1:Q
        #         for k = 1:D
        #             print(s1[i,j,k], " ")
        #         end
        #     end
        # end
        # println()

        for dy = 1:D
            for dx = 1:D
                for qz = 1:Q
                    acc = 0.0
                    for qy = 1:Q
                        acc += s1[qz, qy, dx] * B[qy, dy]
                    end
                    s2[qz, dy, dx] = acc
                end
            end
        end

        for dy = 1:D
            for dx = 1:D
                for dz = 1:D
                    acc = 0.0
                    for qz = 1:Q
                        acc += s2[qz, dy, dx] * B[qz, dz]
                    end
                    r[dx, dy, dz, vd] += acc
                end

            end
        end
    end

    return r
end

function integrate_grad(f_qp)
    r = zeros(Float64, D, D, D, test_vdim)
    f_qp = reshape(f_qp, (test_vdim, output_op_dim, Q, Q, Q))
    s0 = zeros(Float64, Q, Q, D)
    s1 = zeros(Float64, Q, Q, D)
    s2 = zeros(Float64, Q, Q, D)
    s3 = zeros(Float64, Q, D, D)
    s4 = zeros(Float64, Q, D, D)
    s5 = zeros(Float64, Q, D, D)

    for vd = 1:test_vdim
        for qz = 1:Q
            for qy = 1:Q
                for dx = 1:D
                    uvw = zeros(Float64, 3)
                    for qx = 1:Q
                        uvw[1] += f_qp[vd, 1, qx, qy, qz] * G[qx, dx]
                        uvw[2] += f_qp[vd, 2, qx, qy, qz] * B[qx, dx]
                        uvw[3] += f_qp[vd, 3, qx, qy, qz] * B[qx, dx]
                    end
                    s0[qz, qy, dx] = uvw[1]
                    s1[qz, qy, dx] = uvw[2]
                    s2[qz, qy, dx] = uvw[3]
                end
            end
        end

        for qz = 1:Q
            for dy = 1:D
                for dx = 1:D
                    uvw = zeros(Float64, 3)
                    for qy = 1:Q
                        uvw[1] += s0[qz, qy, dx] * B[qy, dy]
                        uvw[2] += s1[qz, qy, dx] * G[qy, dy]
                        uvw[3] += s2[qz, qy, dx] * B[qy, dy]
                    end
                    s3[qz, dy, dx] = uvw[1]
                    s4[qz, dy, dx] = uvw[2]
                    s5[qz, dy, dx] = uvw[3]
                end
            end
        end

        for dz = 1:D
            for dy = 1:D
                for dx = 1:D
                    uvw = zeros(Float64, 3)
                    for qz = 1:Q
                        uvw[1] += s3[qz, dy, dx] * B[qz, dz]
                        uvw[2] += s4[qz, dy, dx] * B[qz, dz]
                        uvw[3] += s5[qz, dy, dx] * G[qz, dz]
                    end
                    r[dx, dy, dz, vd] += sum(uvw)
                end
            end
        end

        for i = 1:D
            for j = 1:D
                for k = 1:D
                    print(f_qp[1, 1, i,j,k], " ")
                end
            end
        end
        println()
    end

    return r
end

function kernel(dudxi, J, w)
    transpose(inv(J)) * inv(J) * dudxi * det(J) * w
end

dudxi_qp = reshape(interpolate_grad(u_e), (space_dim, Q * Q * Q))
J_qp = reshape(interpolate_grad(x_e), (space_dim, space_dim, Q * Q * Q))
f_qp = zeros(test_vdim, output_op_dim, Q * Q * Q)

for q = 1:Q*Q*Q
    r_qp = kernel(dudxi_qp[:, q], J_qp[:, :, q], w_qp[q])
    for od = 1:output_op_dim
        f_qp[1, od, q] += r_qp[od]
    end
end

r = integrate_grad(f_qp)
# println(transpose(R) * reshape(r, (D * D * D)))

println(reshape(r, D * D * D))
