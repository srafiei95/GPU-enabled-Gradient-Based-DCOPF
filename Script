using JuMP, Gurobi, LinearAlgebra, PowerModels, Base.Threads, Mosek, MosekTools
using Plots
#|||||||||||||||||||||||||||||||Loading the Data of the System||||||||||||||||||||||||||||||
#10k
#1.34611e6 quad
#1.2474631811765707e6 linear
case = "./pglib_opf_case10000_goc.m"
network_data       = PowerModels.parse_file(case)
basic_network_data = PowerModels.make_basic_network(network_data)
Result = PowerModels.solve_dc_opf(basic_network_data, Gurobi.Optimizer)

# %%
print_summary(Result["solution"]) 
#2k
#943042.0 quad
#846286.0649394316 Linear
case = "./pglib_opf_case2000_goc.m"
network_data       = PowerModels.parse_file(case)
basic_network_data = PowerModels.make_basic_network(network_data)
Result = PowerModels.solve_dc_opf(basic_network_data, Gurobi.Optimizer)

# %%
print_summary(Result["solution"]) 

# system information
nb = length(basic_network_data["bus"])
nd = length(basic_network_data["load"])
nl = length(basic_network_data["branch"])

# how many gens can produce power?
ng = 0
for (key,val) in basic_network_data["gen"]
    if val["pmax"] > 0.0
        global ng += 1
    end
end

# map buses to indices
index = 1
bus_id_list    = Int64[]
bus_index_list = Int64[]
for (key,val) in basic_network_data["bus"]
    source_id = val["source_id"][2]
    push!(bus_id_list, source_id)
    push!(bus_index_list, index)
    index += 1
end

function map_id_to_index(bus_id, bus_id_list)
    findall(bus_id_list .== bus_id)[1]
end

# get the loading at each bus!
Pd = zeros(nb)
for (key,val) in basic_network_data["load"]
    bus_id = val["source_id"][2]
    index = map_id_to_index(bus_id, bus_id_list)
    index = val["load_bus"]
    Pd[index] += val["pd"]
end

# get the generation mapping: N (nb by ng)
N = zeros(nb,ng)
Pmax = zeros(ng)
Pmin = zeros(ng)
Cost = zeros(ng)
Cost1 = zeros(ng)
Cost2 = zeros(ng)
Cost3 = zeros(ng)
Pg_soln = zeros(ng)

ii = 1
for (key,val) in basic_network_data["gen"]
    if val["pmax"] > 0.0
        gen_id = val["gen_bus"]
        index  = gen_id
        Pmax[ii] = val["pmax"]
        Pmin[ii] = val["pmin"]
        Cost1[ii] = val["cost"][1]
        Cost2[ii] = val["cost"][2]
        Cost3[ii] = val["cost"][3]
        N[index, ii] = 1
        Pg_soln[ii]  = Result["solution"]["gen"][key]["pg"]
        ii += 1
    end
end
# Cost = Cost2 ----> for LP
# branch information
b_line_list = Float64[]
to_bus_list = Int64[]
fr_bus_list = Int64[]
flow_max    = Float64[]
E = zeros(nl,nb)
ii = 1

for (key,val) in basic_network_data["branch"]
    push!(fr_bus_list, val["f_bus"])
    push!(to_bus_list, val["t_bus"])
    push!(b_line_list, imag(1/(val["br_r"] + im*val["br_x"])))
    push!(flow_max, val["rate_c"])
    E[ii,val["f_bus"]] = -1
    E[ii,val["t_bus"]] = 1
    ii = ii + 1
end

# build ptdf
Ehat = E[:,2:end]
Yl   = diagm(b_line_list)
ptdf = Yl*Ehat*inv(Ehat'*Yl*Ehat)
ptfd_full = [zeros(nl) ptdf]
# Build model
# Extract A, B, C, D
A = ones(Float64, ng)
univec = ones(length(Pd))
B = [univec'*Pd]
C=[ptfd_full*N;-ptfd_full*N;I(ng);-I(ng)]
D=[ptfd_full*Pd.+flow_max;-ptfd_full*Pd.+flow_max;Pmax;-Pmin]
#|||||||||||||||||||||||||||Solve with Gurobi-Mosek for Benchmarking||||||||||||||||||||||||||||
#------------------------------------------LP
model2 = Model(Mosek.Optimizer)
@variable(model2, Pg[1:ng])
c1 = @constraint(model2, A'*Pg .- B[1] .==0)
c2 = @constraint(model2 , C*Pg.-D .<=0)
@objective(model2, Min, Cost'*Pg)
# @constraint(model, Pmin .<= Pg .<= Pmax)
# @constraint(model, sum(Pg) == sum(Pd))
# @constraint(model, -flow_max .<= ptfd_full*(N*Pg-Pd) .<= flow_max)
# set_optimizer_attribute(model, "NonConvex", 2)
@btime 
optimize!(model2)
println(objective_value(model))
objective_value(model)

# %% solve with Gurobi without PTDF
# model2 = Model(Gurobi.Optimizer)
# @variable(model2, Pg[1:ng])
# @variable(model2, theta[1:nb])
# @constraint(model2, theta[1] == 0.0)
# 
# # %%
# v1 = N*Pg-Pd
# v2 = E'*Yl*E*theta
# 
# # %%
# @constraint(model2, v1 == v2)
# @objective(model2, Min, Cost'*Pg)
# 
# # %% 
# @constraint(model2, Pmin .<= Pg .<= Pmax)
# @constraint(model2, sum(Pg) == sum(Pd))
# @constraint(model2, -flow_max .<= Yl*E*theta .<= flow_max)
# 
# optimize!(model2)
# objective_value(model2)

# %% == #
# println(Result["objective"])
# println(objective_value(model))
# println(objective_value(model2))
# %% =====================================================================
Pg_mean = 0.5 * (Pmax .+ Pmin)
Pg_range = Pmax .- Pmin
Mσv = 0.5 * Pg_range
Mσ = diagm(0.5 * Pg_range)
mμ = Pg_mean
model3 = Model(Gurobi.Optimizer)
@variable(model3, Pg[1:ng])
@constraint(model3, A'*(Mσ*Pg + mμ).-B .==0)
@constraint(model3, C*(Mσ*Pg + mμ) .-D .<=0)
@objective(model3, Min, Cost'*(Mσ*Pg + mμ))

@time optimize!(model3)
println("Optimal g: ", JuMP.value.(Pg[1:ng]))
println("Total Cost: ", objective_value(model3))
objective_value(model3)
##==========================================
#remove the generation constraints
C_n    = C[1:nl*2,:]
D_n    = D[1:nl*2]
model4 = Model(Gurobi.Optimizer)
@variable(model4, Pg[1:ng])
@constraint(model4, A'*(Mσ*Pg + mμ).-B[1] .==0)
@constraint(model4, C_n*(Mσ*Pg + mμ) .-D_n .<=0)
@constraint(model4, -1.0 .<= Pg .<= 1.0 )
@objective(model4, Min, Cost'*(Mσ*Pg + mμ))

@time optimize!(model4)
println("Optimal g: ", JuMP.value.(Pg[1:ng]))
println("Total Cost: ", objective_value(model4))
objective_value(model4)
##================================================================================
#rebuild LD
C_n    = C[1:nl*2,:]
D_n    = D[1:nl*2]
model5 = Model(Gurobi.Optimizer)
@variable(model5, lambda)
@variable(model5, mu[1:2*nl], lower_bound = 0.0)
@variable(model5, y[1:ng])
@constraint(model5,  ((Cost'+lambda*A'+mu'*C_n)*Mσ)' .<= y)
@constraint(model5, -((Cost'+lambda*A'+mu'*C_n)*Mσ)' .<= y)
@objective(model5, Max, -sum(y) - lambda*B[1] - mu'*D_n + (Cost'*mμ+sum(lambda*A'*mμ)+mu'*C_n*mμ))
@time optimize!(model5)
LLL= JuMP.value.(lambda)
MMM=JuMP.value.(mu[1:2*nl])
println("Optimal g: ", JuMP.value.(Pg[1:ng]))
println("Total Cost: ", objective_value(model5))

#---------------------------------------QP
Pg_mean = 0.5 * (Pmax .+ Pmin)
Pg_range = Pmax .- Pmin
Mσv = 0.5 * Pg_range
Mσ  = diagm(0.5 * Pg_range)
mμ  = Pg_mean
C_n = C[1:nl*2,:]
D_n = D[1:nl*2]
a   = A'*Mσ
b   = A'*mμ .- B
c   = C_n*Mσ
d   = C_n*mμ .-D_n
cg1 = Cost2'*Mσ
cg0 = Cost2'*mμ + sum(Cost3)

model = Model(Gurobi.Optimizer)
@variable(model, Pg[1:ng])
@constraint(model, a*Pg .+ b .== 0)
@constraint(model, c*Pg .+ d .<= 0)
@constraint(model, -1.0 .<= Pg .<= 1.0 )
@objective(model, Min, cg1*Pg + cg0)
@time optimize!(model)
println("Total Cost: ", objective_value(model))

# %% add the quadratic cost! terms and normalize -- very slow :)
cg0_q = cg0 + sum(Cost1.*(mμ.^2))
cg1_q = cg1 + (2*Cost1.*(Mσ*mμ))'
cg2_q = (Mσ^2)*Cost1
model2 = Model(Mosek.Optimizer)
@variable(model2, Pg[1:ng])
@constraint(model2, A'*Pg .- B[1] .== 0)
@constraint(model2, C*Pg .- D .<= 0)
@objective(model2, Min, sum(Cost1'*(Pg.^2)) + sum(Cost2'*Pg) .+ sum(Cost3))
@time optimize!(model2)
println("Total Cost: ", objective_value(model))

model = Model(Gurobi.Optimizer)
@variable(model, Pg[1:ng])
@constraint(model, a*Pg .+ b .== 0)
@constraint(model, c*Pg .+ d .<= 0)
@constraint(model, -1.0 .<= Pg .<= 1.0 )
@objective(model, Min, cg2_q'*(Pg.^2) + cg1_q*Pg + cg0_q)
@time optimize!(model)
println("Total Cost: ", objective_value(model))

# %% append with t as the last element of Pgt, and use normalized t value: 100000*t_tilde = t
at = [a 0] 
ct = [c zeros(size(c,1),1)]
cg1_qt = [cg1_q 100000]

model = Model(Mosek.Optimizer)
@variable(model, Pgt[1:ng+1])
@constraint(model, at*Pgt .+ b .== 0)
@constraint(model, ct*Pgt .+ d .<= 0)
@constraint(model, -1.0 .<= Pgt .<= 1.0)
@objective(model, Min, cg1_qt*Pgt + cg0_q)
@constraint(model, sum(cg2_q'*(Pgt[1:end-1].^2)) .<= 100000*Pgt[end])
@time optimize!(model)
println("Total Cost: ", objective_value(model))

# %% solve using Mosek!
cg0_q = cg0 + sum(Cost1.*(mμ.^2))
cg1_q = cg1 + (2*Cost1.*(Mσ*mμ))'
cg2_q = (Mσ^2)*Cost1

at = [a 0] 
ct = [c zeros(size(c,1),1)]
cg1_qt = [cg1_q 100000]

model = Model(Gurobi.Optimizer)
@variable(model, Pgt[1:ng+1])
@constraint(model, at*Pgt .+ b .== 0)
@constraint(model, ct*Pgt .+ d .<= 0)
@constraint(model, -1.0 .<= Pgt .<= 1.0)
@objective(model, Min, cg1_qt*Pgt + cg0_q)
@constraint(model, [100000*Pgt[end]; 0.5; sqrt.(cg2_q).*Pgt[1:end-1]] in RotatedSecondOrderCone())
optimize!(model)
println("Total Cost: ", objective_value(model))

# %% solve the dual using Mosek!
using Mosek
using MosekTools

M_soc = diagm([sqrt.(cg2_q); 100000])

model = Model(Mosek.Optimizer)
@variable(model, lambda)
@variable(model, 0.00 <= sb)
@variable(model, mu[1:2*nl], lower_bound = 0.0)
@variable(model, s[1:(ng+1)])
@variable(model, y[1:(ng+1)])
@constraint(model,  ((cg1_qt + lambda*at + mu'*ct - s'*M_soc))' .<= y)
@constraint(model, -((cg1_qt + lambda*at + mu'*ct - s'*M_soc))' .<= y)

# put s in the SOC
@constraint(model, [s[end]; sb; s[1:end-1]] in RotatedSecondOrderCone())
@objective(model, Max, -sum(y) - 0.5*sb + lambda*b[1] + mu'*d + cg0_q)
@time optimize!(model)
println()
println("Total Cost: ", objective_value(model))
println(value(s[end]))
#|||||||||||||||||||||||||||||||For GPU||||||||||||||||||||||||||||||

global lambda_Mtl     = Float32(0.0)
global mu_Mtl         = MtlArray(Float32.(mu))
global grad_lambda_Mtl= Float32(0.0)
global grad_mu_Mtl    = MtlArray(Float32.(grad_mu))
global v_lambda_Mtl   = Float32.(v_lambda)
global v_mu_Mtl       = MtlArray(Float32.(v_mu))
global momentum_Mtl   = Float32.(momentum)
global alpha_Mtl      = Float32.(alpha)
at_Mtl         = MtlArray(Float32.(at))
c_Mtl          = MtlArray(Float32.(c))
d_Mtl          = MtlArray(Float32.(d))
cg1_Mtl        = MtlArray(Float32.(cg1))
cg1t_Mtl       = MtlArray(Float32.(cg1t))
sign_gamma_Mtl = MtlArray(Float32.(sign_gamma))
ct_Mtl         = MtlArray(Float32.(ct))











#|||||||||||||||||||||||||||||||GBO METHODS||||||||||||||||||||||||||||||
# Hyperparameters are set to a default zero. Please change accordingly.
#LP
#GDM
alpha = 0.0
momentum = 0.0
beta1 = 0.0
beta2 = 0.0
lr = 0.0
loop_time = @btime begin
    lambda = 0.0
    mu = zeros(2*nl)
    grad_lambda = 0.0
    grad_mu = zeros(2*nl)
    obj = zeros(n_its + 1)
    v_lambda = 0.0
    v_mu = zeros(length(mu))
    # lr = 0.999
    #CPU
    obj_CPU = gd_with_momentum_LP( obj, v_lambda, v_mu, momentum, alpha, lr,  n_its, lambda, mu, grad_lambda, grad_mu, s_gamma, at, b, c, d, cg1t, ng, nl)
    #GPU
    obj_GPU = gd_with_momentum_GPU(t, obj, v_lambda, v_mu, momentum, alpha, n_its, lambda, mu, grad_lambda, grad_mu, s_gamma, at, b, c, d, cg1t, ng, nl)


end

##AdaGrad
loop_time = @elapsed begin 
    lambda = 0.0
    mu = zeros(2*nl)
    grad_lambda = 0.0
    grad_mu = zeros(2*nl)
    obj = zeros(n_its + 1)
    r_lambda = 0.0
    r_mu = zeros(2 * nl) 
    #CPU
    obj_CPU = adagrad_LP(n_its,lambda,mu,grad_lambda,grad_mu,at,b,c,d,cg1t,cg0,alpha,epsilon,ng,nl,obj)
    #GPU
    obj_GPU = adagrad_GPU(n_its, lambda, mu, grad_lambda, grad_mu, at, b, c, d, cg1t, cg0, alpha, epsilon, ng, nl, obj)

end

##ADAM
loop_time = @elapsed begin
    lambda = 0.0
    mu = zeros(2*nl)
    grad_lambda = 0.0
    grad_mu = zeros(2*nl)
    obj = zeros(n_its + 1)
    m_lambda, v_lambda = 0.0, 0.0
    m_mu, v_mu = zeros(2 * nl), zeros(2 * nl)
    m_lambda_hat, v_lambda_hat = 0.0, 0.0
    m_mu_hat, v_mu_hat = similar(m_mu), similar(v_mu)
    #CPU
    obj = adam_LP(n_its, lambda, mu, grad_lambda, grad_mu, at, b, c, d, cg1t, cg0, alpha, beta1, beta2, epsilon, ng, nl, obj)
    #GPU
    obj_GPU = adam_GPU(n_its, lambda, mu, grad_lambda, grad_mu, at, b, c, d, cg1t, cg0, alpha, epsilon, beta1, beta2, ng, nl, obj)

end

#QP
#GDM
loop_time = @btime begin
    lambda = 0.0
    mu = zeros(2*nl)
    s  = zeros(ng+1)
    s[end] = 1.0
    grad_lambda = 0.0
    grad_mu = zeros(2*nl)
    grad_s = zeros(length(s[1:end-1]))
    obj = zeros(n_its + 1)
    v_lambda = 0.0
    v_mu = zeros(length(mu))
    v_s = zeros(ng)
    obj = gd_with_momentum_3(obj, v_lambda, v_mu, v_s, momentum, alpha, alr,mlr,  n_its, lambda, mu, s, grad_lambda, grad_mu, grad_s, a, b, c, d, cg1_qtt, ng, nl)
    end

##AdaGrad
loop_time = @btime begin 
    lambda = 0.0
    mu = zeros(2*nl)
    s  = zeros(ng+1)
    s[end] = 1.0
    grad_lambda = 0.0
    grad_mu = zeros(2*nl)
    grad_s = zeros(length(s[1:end-1]))
    obj = zeros(n_its + 1)
    r_lambda = 0.0
    r_mu = zeros(2 * nl) 
    r_s = zeros(ng)
    obj = adagrad_QP(n_its,lambda,mu,s,grad_lambda,grad_mu,grad_s,a,b,c,d,cg1_qtt,cg0,alpha,epsilon,ng,nl,obj)
    end
##ADAM
loop_time = @elapsed begin
    # @btime begin
    lambda = 0.0
    mu = zeros(2*nl)
    s  = zeros(ng+1)
    s[end] = 1.0
    grad_lambda = 0.0 
    grad_mu = zeros(2 * nl)  
    grad_s = zeros(length(s[1:end-1]))
    m_lambda, v_lambda = 0.0, 0.0
    m_mu, v_mu = zeros(2 * nl), zeros(2 * nl)
    m_lambda_hat, v_lambda_hat = 0.0, 0.0
    m_mu_hat, v_mu_hat = zeros(2 * nl), zeros(2 * nl)
    m_s,v_s = zeros(length(s[1:end-1])), zeros(length(s[1:end-1]))
    m_s_hat, v_s_hat = zeros(length(s[1:end-1])), zeros(length(s[1:end-1]))
    r_lambda = 0.0
    r_mu = zeros(2 * nl)
    r_s = zeros(length(s[1:end-1]))
    obj = zeros(n_its + 1)
    obj =  adam_QP(n_its, lambda, mu, s, grad_lambda, grad_mu, grad_s, a, b, c, d, cg1_qtt, cg0, alpha, beta1, beta2, epsilon, ng, nl, obj)
    end
 



