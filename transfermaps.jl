using Enzyme, ForwardDiff, LinearAlgebra

num_qp = 4;
num_trial_dof = 4;
trial_vdim = 1
trial_op_dim = [1, 2]
num_test_dof = num_trial_dof
test_op_dim = 2
test_vdim = 1

Bu_u_mem = [0.622008468 0.166666667 0.166666667 0.0446581987 0.166666667 0.622008468 0.0446581987 0.166666667 0.0446581987 0.166666667 0.166666667 0.622008468 0.166666667 0.0446581987 0.622008468 0.166666667]
Bu_du_mem = [-0.788675135 -0.788675135 -0.211324865 -0.211324865 -0.788675135 -0.211324865 -0.788675135 -0.211324865 0.788675135 0.788675135 0.211324865 0.211324865 -0.211324865 -0.788675135 -0.211324865 -0.788675135 0.211324865 0.211324865 0.788675135 0.788675135 0.211324865 0.788675135 0.211324865 0.788675135 -0.211324865 -0.211324865 -0.788675135 -0.788675135 0.788675135 0.211324865 0.788675135 0.211324865]
Bu_u = reshape(Bu_u_mem, (num_qp, trial_op_dim[1], num_trial_dof))
Bu_du = reshape(Bu_du_mem, (num_qp, trial_op_dim[2], num_trial_dof))
Bu = [Bu_u, Bu_du]
Bv = Bu_du;

det(A) = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]

# kernel(u, J, w) = u * det(J) * w

# u = [1.0, 1.0];
# du = [0.0, 0.0];
J = [1 0; 0 1];
w = 0.25;

### ParamtricFunction test

# ParamtricFunction as quadrature data
pf_size_on_qp = 4
pf = zeros(num_qp * pf_size_on_qp)

residual_size_on_qp = zeros(length(pf))

function kernel(J, w)
    return J * w
end

function Trho(u, e, q, size_on_qp, num_qp)
    b = q * size_on_qp
    c = q * size_on_qp + ((e - 1) * num_qp * size_on_qp)
    return u[b-size_on_qp+1:c]
end

function TrhoT(u, uq, e, q, size_on_qp, num_qp)
    b = q * size_on_qp
    c = q * size_on_qp + ((e - 1) * num_qp * size_on_qp)
    return u[b-size_on_qp+1:c] = uq
end

for e = 1:1
    for q = 1:num_qp
        pfq = Trho(pf, e, q, pf_size_on_qp, num_qp)
        pfq[:] = kernel(J, w)
        TrhoT(pf, pfq, e, q, pf_size_on_qp, num_qp)
    end
end

# println(pf)

nqp = 4
data = reshape([1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1], (16,))
m = 2
n = 2
arg = zeros(n, m)
for q = 1:nqp
    for i = 1:m
        for j = 1:n
            arg[j, i] = data[(i * m) + j]
        end
    end
    println(arg)
end


# function kernel(u, dudxi, J, w)
#     invJ = J
#     dudx = dudxi * invJ
#     return u[1, 1] * dudx * det(J) * w * transpose(invJ)
# end

# u = ones((1, 1))
# du = [zeros((1, 1)), zeros((2, 2))]
# dudxi = zeros((2, 2))
# J = [1 0; 0 1];
# w = 0.25;

# wrap(x) = kernel(x, J, w)

# D = zeros(test_vdim, test_op_dim, trial_vdim, sum(trial_op_dim), num_qp)
# for q = 1:num_qp
#     for j = 1:trial_vdim
#         m_offset = 0
#         local trial_op_dim = size(Bu[s])[2]
#         for s = 1:length(Bu)
#             for m = 1:trial_op_dim
#                 du[s][j, m] = 1.0
#                 df = autodiff(Forward, kernel, Duplicated(u, du[1]), Duplicated(dudxi, du[2]), Const(J), Const(w))[1]
#                 du[s][j, m] = 0.0
#                 for i = 1:test_vdim
#                     for k = 1:test_op_dim
#                         D[i, k, j, m+m_offset, q] = df[i, k]
#                     end
#                 end
#             end
#             m_offset += trial_op_dim
#         end
#     end
# end

# Ae = zeros(num_test_dof, test_vdim, num_trial_dof, trial_vdim)

# for J = 1:num_trial_dof
#     for j = 1:trial_vdim
#         fhat = zeros(test_vdim, test_op_dim, num_qp)
#         m_offset = 0
#         for s = 1:length(Bu)
#             trial_op_dim = size(Bu[s])[2]
#             # precompute fhat for trial dof J column
#             for qp = 1:num_qp
#                 for i = 1:test_vdim
#                     for k = 1:test_op_dim
#                         for m = 1:trial_op_dim
#                             fhat[i, k, qp] += D[i, k, j, m+m_offset, qp] * Bu[s][qp, m, J]
#                         end
#                     end
#                 end
#             end
#             m_offset += trial_op_dim
#         end

#         # this imitates what 'map_quadrature_data_to_fields' does
#         for I = 1:num_test_dof
#             for i = 1:test_vdim
#                 for qp = 1:num_qp
#                     for k = 1:test_op_dim
#                         Ae[I, i, J, j] += fhat[i, k, qp] * Bv[qp, k, I]
#                     end
#                 end
#             end
#         end
#     end
# end

# display(reshape(Ae, (num_test_dof * test_vdim, num_trial_dof * trial_vdim)))
