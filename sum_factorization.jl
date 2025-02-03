using LinearAlgebra
using Tullio

num_qp = 4
num_trial_dof = 4;
trial_vdim = 1
trial_op_dim = [1, 2]
test_op_dim = 2
test_vdim = 1
num_test_dof = num_trial_dof
num_rho_dof = 4;
space_dim = 2;

Bu_mem = [0.622008468 0.166666667 0.166666667 0.0446581987 0.166666667 0.622008468 0.0446581987 0.166666667 0.0446581987 0.166666667 0.166666667 0.622008468 0.166666667 0.0446581987 0.622008468 0.166666667]

Bdu_mem = [-0.788675135 -0.788675135 -0.211324865 -0.211324865 -0.788675135 -0.211324865 -0.788675135 -0.211324865 0.788675135 0.788675135 0.211324865 0.211324865 -0.211324865 -0.788675135 -0.211324865 -0.788675135 0.211324865 0.211324865 0.788675135 0.788675135 0.211324865 0.788675135 0.211324865 0.788675135 -0.211324865 -0.211324865 -0.788675135 -0.788675135 0.788675135 0.211324865 0.788675135 0.211324865]

Bu = reshape(Bu_mem, (num_qp, trial_op_dim[1], num_trial_dof))
Bdu = reshape(Bdu_mem, (num_qp, trial_op_dim[2], num_trial_dof))
Bv = Bdu;

u_e = reshape([2.345 2.345 3.595 2.345], (num_trial_dof, trial_vdim))
rho_e = reshape([0.0 1.0 2.0 1.0], (num_trial_dof, trial_vdim))
x_e = reshape([0.0 1.0 1.0 0.0 0.0 0.0 1.0 1.0], (num_trial_dof, space_dim))

rho_qpref = [0.422650 1.000000 1.000000 1.577350]
J_qpref = [1.000000 0.000000 0.000000 1.000000 1.000000 0.000000 0.000000 1.000000 1.000000 0.000000 0.000000 1.000000 1.000000 0.000000 0.000000 1.000000]
w_qpref = [0.250000 0.250000 0.250000 0.250000]
dudxi_qpref = [0.264156 0.264156 0.264156 0.985844 0.985844 0.264156 0.985844 0.985844]

rho_qp = zeros(Float64, trial_vdim, num_qp)
for v = 1:trial_vdim
    for q = 1:num_qp
        acc = 0.0
        for d = 1:num_trial_dof
            acc += Bu[q, v, d] * rho_e[d, v]
        end
        rho_qp[v, q] = acc
    end
end
println(rho_qp)

J_qp = zeros(Float64, space_dim, space_dim, num_qp)
for v = 1:space_dim
    for s = 1:space_dim
        for q = 1:num_qp
            acc = 0.0
            for d = 1:num_trial_dof
                acc += Bdu[q, s, d] * x_e[d, v]
            end
            J_qp[v, s, q] = acc
        end
    end
end
println(J_qp)

dudxi_qp = zeros(Float64, trial_vdim, space_dim, num_qp)
for v = 1:trial_vdim
    for s = 1:space_dim
        for q = 1:num_qp
            acc = 0.0
            for d = 1:num_trial_dof
                acc += Bdu[q, s, d] * u_e[d, v]
            end
            dudxi_qp[v, s, q] = acc
        end
    end
end
println(dudxi_qp)

w_qp = w_qpref

# sum factorization

u_e = reshape(Float64[2.345 2.345 2.345 3.595], (num_trial_dof, trial_vdim))
rho_e = reshape(Float64[0 1 1 2], (num_trial_dof, trial_vdim))
x_e = reshape(Float64[0 1 0 1 0 0 1 1], (num_trial_dof, space_dim))

nq1d = 2
nd1d = 2

B = reshape([0.788675135 0.211324865 0.211324865 0.788675135], (nq1d, nd1d))
G = reshape([-1 -1 1 1], (nq1d, nd1d))

function interpolate_value(u_e)
    u_e = reshape(u_e, (nd1d, nd1d))
    S2 = zeros(Float64, nq1d, nq1d)
    for v = 1:trial_vdim
        @tullio S1[qx, dy] := u_e[dx, dy] * B[qx, dx]
        @tullio S2[qx, qy] = B[qy, dy] * S1[qx, dy]
    end
    return S2
end

function interpolate_grad(u_e)
    vdim = size(u_e, 2)
    u_e = reshape(u_e, (nd1d, nd1d, vdim))
    dq0 = zeros(Float64, nd1d, nq1d)
    dq1 = zeros(Float64, nd1d, nq1d)
    dudxi_qp = zeros(nq1d, nq1d, vdim, space_dim)

    for vd = 1:vdim
        for dy = 1:nd1d
            for qx = 1:nq1d
                u = 0.0
                v = 0.0
                for dx = 1:nd1d
                    u += u_e[dx, dy, vd] * B[qx, dx]
                    v += u_e[dx, dy, vd] * G[qx, dx]
                end
                dq0[dy, qx] = u
                dq1[dy, qx] = v
            end
        end

        for qy = 1:nq1d
            for qx = 1:nq1d
                du = [0.0, 0.0]
                for dy = 1:nd1d
                    du[1] += dq1[dy, qx] * B[qy, dy]
                    du[2] += dq0[dy, qx] * G[qy, dy]
                end

                for s = 1:space_dim
                    dudxi_qp[qx, qy, vd, s] = du[s]
                end
            end
        end
    end
    return dudxi_qp
end

function integrate_grad(f_qp, r)
    dq0 = zeros(Float64, nd1d, nq1d)
    dq1 = zeros(Float64, nd1d, nq1d)
    for qy = 1:nq1d
        for dx = 1:nd1d
            u = 0.0
            v = 0.0
            for qx = 1:nq1d
                u += G[qx, dx] * f_qp[qx, qy, 1, 1]
                v += B[qx, dx] * f_qp[qx, qy, 1, 2]
            end
            dq0[dx, qy] = u
            dq1[dx, qy] = v
        end
    end

    for dy = 1:nd1d
        for dx = 1:nd1d
            u = 0.0
            v = 0.0
            for qy = 1:nq1d
                u += dq0[dx, qy] * B[qy, dy]
                v += dq1[dx, qy] * G[qy, dy]
            end
            r[dx, dy] += u + v
        end
    end
end

function integrate_value(f_qp, r)
end

function kernel(rho, J, w, dudxi)
    return rho * rho * ((inv(J) * transpose(inv(J))) * dudxi) * det(J) * w
end

rho_qp = reshape(interpolate_value(rho_e), (nq1d * nq1d,))
dudxi_qp = reshape(interpolate_grad(u_e), (nq1d * nq1d, 2))
J_qp = reshape(interpolate_grad(x_e), (nq1d * nq1d), 2, 2)
f_qp = zeros(nq1d * nq1d, 1, space_dim)
for q = 1:nq1d*nq1d
    f_qp[q, 1, :] = kernel(rho_qp[q], J_qp[q, :, :], w_qp[q], dudxi_qp[q, :])
end

w_qp = reshape(w_qp, nq1d, nq1d)
rho_qp = interpolate_value(rho_e)
dudxi_qp = reshape(interpolate_grad(u_e), (nq1d, nq1d, 2))
J_qp = interpolate_grad(x_e)
f_qp = zeros(nq1d, nq1d, 1, space_dim)
for qx = 1:nq1d
    for qy = 1:nq1d
        f_qp[qx, qy, 1, :] = kernel(rho_qp[qx, qy], J_qp[qx, qy, :, :], w_qp[qx, qy], dudxi_qp[qx, qy, :])
    end
end

r = zeros(Float64, nd1d, nd1d)
integrate_grad(f_qp, r)
println(reshape(r, nd1d * nd1d))
