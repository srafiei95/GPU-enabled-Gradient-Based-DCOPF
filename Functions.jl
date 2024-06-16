#|||||||||||||||||||||||||||||||Gradient Calculation Functions||||||||||||||||||||||||||||||
lambda      = 0.0 
mu          = randn(2*nl)
grad_lambda = 0.0
grad_mu     = zeros(2*nl)
s_gamma     = zeros(ng) #L
at          = a'
cg1t        = cg1'
function obj_gradient_parallel(lambda::Float64, mu::Vector{Float64}, grad_lambda::Float64, grad_mu::Vector{Float64}, s_gamma::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Matrix{Float64}, d::Vector{Float64}, cg1::Vector{Float64}, ng::Int64, nl::Int64)
    grad_lambda  = copy(b[1])
    grad_mu     .= copy.(d)

    Threads.@threads for el in 1:ng
        cv = @view c[:,el]
        @fastmath s_gamma[el] = Float64(sign(cg1[el] + lambda*a[el] + dot(mu, cv)))
    end

    grad_lambda += -dot(s_gamma, a)
    Threads.@threads for jj = 1:2*nl
        cv = @view c[jj,:]
        grad_mu[jj] -= dot(s_gamma,cv)
    end

    return grad_lambda, grad_mu
end

lambda = 0.0*value(lambda)
mu = 0.0*value.(mu)
s  = 0.0*value.(s) 
s[end] = 1.0
grad_lambda = 0*lambda
grad_mu = 0*mu
grad_s = 0*s[1:end-1]
a = at'
c = ct
cg1_qtt = cg1_qt'
function obj_gradient_SOCP(lambda::Float64, mu::Vector{Float64}, grad_lambda::Float64, grad_mu::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Matrix{Float64}, d::Vector{Float64}, cg1_qt::Vector{Float64}, ng::Int64, nl::Int64, M_soc::Matrix{Float64}, grad_s::Vector{Float64}, s::Vector{Float64})
    grad_s      .= (-0.5).*(@view s[1:end-1])
    grad_lambda  = copy(b[1])
    grad_mu     .= copy.(d)

    for el in 1:ng
        cv = @view c[:,el]
        s_gamma = Float64(sign(cg1_qt[el] + lambda*a[el] + dot(mu, cv) - s[el]*M_soc[el,el]))

        grad_lambda += -s_gamma*a[el]
        for jj = 1:2*nl
            grad_mu[jj] += -s_gamma*c[jj,el]
        end

        grad_s[el] += s_gamma*M_soc[el,el]
    end

    return grad_lambda, grad_mu, grad_s
end
##-----------ForwardDiff for evaluation
using ForwardDiff

#LP

L(lambda_v, mu_v) = -norm(cg1 + lambda_v.*a + mu_v'*c, 1) + lambda_v*b[1] + mu_v'*d + cg0

# Function to compute gradients using ForwardDiff
function grad_val(lam_val, mu_val)
    return ForwardDiff.gradient(z -> L(z[1], z[2:end]), [lam_val; mu_val])
end

#QP
L(lambda_v, mu_v, s_v) = -norm(cg1_qt + lambda_v.*at + mu_v'*ct - s_v'*M_soc, 1) - 0.25*(s_v[1:end-1]'*s_v[1:end-1]/(s_v[end])) + lambda_v*b[1] + mu_v'*d + cg0_q
grad_val(lam_val, mu_val, s_val) = ForwardDiff.gradient(z -> L(z[1], z[1 .+ (1:2*nl)], z[(2*nl+1) .+ (1:(ng+1))]), [lam_val; mu_val; s_val])












#|||||||||||||||||||||||||||||||GBO METHODS||||||||||||||||||||||||||||||
function obj_gradient_parallel(lambda::Float64, mu::Vector{Float64}, grad_lambda::Float64, grad_mu::Vector{Float64}, s_gamma::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Matrix{Float64}, d::Vector{Float64}, cg1::Vector{Float64}, ng::Int64, nl::Int64)
    grad_lambda  = copy(b[1])
    grad_mu     .= copy.(d)

    Threads.@threads for el in 1:ng
        cv = @view c[:,el]
        @fastmath s_gamma[el] = Float64(sign(cg1[el] + lambda*a[el] + dot(mu, cv)))
    end

    grad_lambda += -dot(s_gamma, a)
    Threads.@threads for jj = 1:2*nl
        cv = @view c[jj,:]
        grad_mu[jj] -= dot(s_gamma,cv)
    end

    return grad_lambda, grad_mu
end


function obj_gradient_SOCP(lambda::Float64, mu::Vector{Float64}, grad_lambda::Float64, grad_mu::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Matrix{Float64}, d::Vector{Float64}, cg1_qt::Vector{Float64}, ng::Int64, nl::Int64, M_soc::Matrix{Float64}, grad_s::Vector{Float64}, s::Vector{Float64})
    grad_s      .= (-0.5).*(@view s[1:end-1])
    grad_lambda  = copy(b[1])
    grad_mu     .= copy.(d)

    for el in 1:ng
        cv = @view c[:,el]
        s_gamma = Float64(sign(cg1_qt[el] + lambda*a[el] + dot(mu, cv) - s[el]*M_soc[el,el]))

        grad_lambda += -s_gamma*a[el]
        for jj = 1:2*nl
            grad_mu[jj] += -s_gamma*c[jj,el]
        end

        grad_s[el] += s_gamma*M_soc[el,el]
    end

    return grad_lambda, grad_mu, grad_s
end

#LP
##GDM
function gd_with_momentum_LP(
    obj::Vector{Float64},
    v_lambda::Float64,
    v_mu::Vector{Float64},
    momentum::Float64,
    alpha::Float64,
    lr::Float64,
    n_its::Int64, lambda::Float64, mu::Vector{Float64}, grad_lambda::Float64, grad_mu::Vector{Float64}, s_gamma::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Matrix{Float64}, d::Vector{Float64}, cg1::Vector{Float64}, ng::Int64, nl::Int64)
    for ii in 1:n_its
            grad_lambda, grad_mu = obj_gradient_parallel(lambda, mu, grad_lambda, grad_mu, s_gamma, at, b, c, d, cg1t, ng, nl)
          # Update velocities
            @fastmath v_lambda = momentum * v_lambda + alpha * grad_lambda
            @fastmath v_mu     .= momentum .* v_mu     .+ alpha .* grad_mu
          # Update parameters
            @fastmath lambda += v_lambda
            @fastmath mu .+= v_mu
          #  alpha = lr*alpha
            @fastmath lambda = lambda + alpha*grad_lambda
            @fastmath mu     .= mu .+ alpha.*grad_mu
          # clip!
            mu .= max.(mu, 0.0)
            #end
            s_gamma_norm = 0.0 

            for el in 1:ng
                cv = @view c[:, el]
                 s_gamma_norm += -norm(cg1[el] + lambda * a[el] + dot(mu, cv))
            end
            final_s_gamma_norm = s_gamma_norm
            # compute!
            @fastmath obj[ii+1] = final_s_gamma_norm + lambda*b[1] + dot(mu,d) + cg0
            println("$ii     ", obj[ii+1])
            if abs(obj[ii+1]-obj[ii]) < 0.08
                # println( "Converged at iteration $ii")
                break
            end
        end
        return obj
    end

##AdaGrad

function adagrad_LP(
    n_its::Int,
    lambda::Float64,
    mu::Vector{Float64},
    grad_lambda::Float64,
    grad_mu::Vector{Float64},
    a::Vector{Float64},
    b::Vector{Float64},
    c::Matrix{Float64},
    d::Vector{Float64},
    cg1::Vector{Float64},
    cg0::Float64,
    alpha::Float64,
    epsilon::Float64,
    ng::Int,
    nl::Int,
    obj::Vector{Float64}
)
    r_lambda = 0.0
    r_mu = zeros(length(mu))
    
    for ii in 1:n_its

            grad_lambda, grad_mu = obj_gradient_parallel(lambda, mu, grad_lambda, grad_mu, s_gamma, at, b, c, d, cg1t, ng, nl)

            @fastmath  r_lambda += grad_lambda^2
            @fastmath  r_mu += grad_mu.^2

            @fastmath  lambda += (alpha / (sqrt(r_lambda) + epsilon)) * grad_lambda
            @fastmath  mu += (alpha ./ (sqrt.(r_mu) .+ epsilon)) .* grad_mu

            mu = max.(mu, 0.0)

            s_gamma_norm = Atomic{Float64}(0.0)

            @threads for el in 1:ng
                cv = @view c[:, el]
                atomic_add!(s_gamma_norm, Float64(-norm(cg1[el] + lambda * a[el] + dot(mu, cv))))
            end
            final_s_gamma_norm = s_gamma_norm[]
            obj[ii+1] = @fastmath @inbounds final_s_gamma_norm + lambda*b[1] + mu'*d + cg0
            println("$ii     ", obj[ii+1])
            if @fastmath abs(obj[ii+1]-obj[ii]) < 0.08
 
                break
            end
        end
    return obj
end
# LP on GPU
##ADAM
function adam_GPU(
    n_its::Int,
    lambda_Mtl::Float32,
    mu_Mtl::MtlArray{Float32},
    grad_lambda_Mtl::Float32,
    grad_mu_Mtl::MtlArray{Float32},
    a_Mtl::MtlArray{Float32},
    b::Vector{Float64},
    c_Mtl::MtlMatrix{Float32},
    d_Mtl::MtlArray{Float32},
    cg1_Mtl::MtlArray{Float32},
    cg0_Mtl::Float32,
    alpha_Mtl::Float32,
    epsilon::Float32,
    beta1_Mtl::Float32,
    beta2_Mtl::Float32,
    ng::Int,
    nl::Int,
    obj_Mtl::MtlArray{Float32}
)
    #  loop
    m_lam_Mtl = 0.0f0
    m_mu        = zeros(2*nl)
    m_mu_Mtl  = MtlArray(Float32.(m_mu))
    v_lam_Mtl = 0.0f0
    v_mu        = zeros(2*nl)
    v_mu_Mtl  = MtlArray(Float32.(v_mu))
    m_lam_hat_Mtl, v_mu_hat_Mtl  = similar(m_mu), similar(v_mu)
   
    for ii in 1:n_its

        # Calculate gradients
        # lambda_Mtl     = Float32(lambda)
        # mu_Mtl         = MtlArray(Float32.(mu))
        # grad_lambda_Mtl= Float32(grad_lambda)
        # grad_mu_Mtl    = MtlArray(Float32.(grad_mu))
        # at_Mtl         = MtlArray(Float32.(at))
        # a_Mtl          = MtlArray(Float32.(a))
        # b_Mtl          = MtlArray(Float32.(b))
        # c_Mtl          = MtlArray(Float32.(c))
        # d_Mtl          = MtlArray(Float32.(d))
        # cg1_Mtl        = MtlArray(Float32.(cg1))
        # cg0_Mtl        = Float32(cg0)
        # cg1t_Mtl       = MtlArray(Float32.(cg1t))
        # sign_gamma_Mtl = MtlArray(Float32.(sign_gamma))
        # ct_Mtl         = MtlArray(Float32.(ct))

        # grad_lambda, grad_mu = obj_gradient_no_loops(lambda, mu, grad_lambda, grad_mu, at, b, c, d, cg1t, sign_gamma, ng, nl, ct)

        grad_lambda_Mtl, grad_mu_Mtl = obj_gradient_update_gpu(lambda_Mtl, mu_Mtl, grad_lambda_Mtl, grad_mu_Mtl, at_Mtl, b, c_Mtl, d_Mtl, cg1t_Mtl, sign_gamma_Mtl, ng, nl, ct_Mtl);

        # lambda         = Float64(lambda_Mtl)
        # mu             = Float64.(Array(mu_Mtl))
        # grad_lambda    = Float64(grad_lambda_Mtl)
        # grad_mu    = Float64.(Array(grad_mu_Mtl))
        # # adam :)
        m_lam_Mtl  = beta1_Mtl*m_lam_Mtl + (1-beta1_Mtl)*grad_lambda_Mtl
        v_lam_Mtl  = beta2_Mtl*v_lam_Mtl + (1-beta2_Mtl)*grad_lambda_Mtl.^2
        # Apply bias correction for lambda
        m_lam_hat_Mtl = m_lam_Mtl / (1 - beta1_Mtl^ii)
        v_lam_hat_Mtl = v_lam_Mtl / (1 - beta2_Mtl^ii)
        lambda_Mtl = lambda_Mtl + alpha_Mtl* m_lam_hat_Mtl / (sqrt(v_lam_hat_Mtl) + Float32(1e-8))
        m_mu_Mtl  = beta1_Mtl*m_mu_Mtl  + (1-beta1_Mtl)*grad_mu_Mtl
        v_mu_Mtl  = beta2_Mtl*v_mu_Mtl  + (1-beta2_Mtl)*grad_mu_Mtl.^2
        # Apply bias correction for mu
        m_mu_hat_Mtl = m_mu_Mtl / (1 - beta1_Mtl^ii)
        v_mu_hat_Mtl = v_mu_Mtl / (1 - beta2_Mtl^ii)
        mu_Mtl = mu_Mtl .+ alpha_Mtl .* m_mu_hat_Mtl ./ (sqrt.(v_mu_hat_Mtl) .+ Float32(1e-8))
        # alpha = 0.99*alpha

        # clip!
        mu_Mtl = max.(mu_Mtl, 0.0f0)

        # compute!
        # obj[ii+1] = -norm(cg1 + lambda.*a + mu'*c, 1) + lambda*b[1] + mu'*d + cg0
         obj[ii+1] =  Float64( -norm(cg1_Mtl .+ lambda_Mtl.*a_Mtl .+ mu_Mtl'*c_Mtl, 1) .+ Array(lambda_Mtl .* b_Mtl)[1] .+ mu_Mtl'*d_Mtl .+ cg0_Mtl)


        println("$ii     ", obj[ii+1])
        if abs(obj[ii+1]-obj[ii]) < 0.1
            println( "Converged at iteration $ii")
            break
        end
        end
return obj
    end
#GDM
function gd_with_momentum_GPU(
    t:: Vector{Float64},
    obj::Vector{Float64},
    v_lambda::Float64,
    v_mu::Vector{Float64},
    momentum::Float64,
    alpha::Float64,
    n_its::Int64, lambda::Float64, mu::Vector{Float64}, grad_lambda::Float64, grad_mu::Vector{Float64}, s_gamma::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Matrix{Float64}, d::Vector{Float64}, cg1::Vector{Float64}, ng::Int64, nl::Int64)
    

    for ii in 1:n_its


        
            lambda_Mtl     = Float32(lambda)
            grad_lambda_Mtl= Float32(grad_lambda)
            mu_Mtl         = MtlArray(Float32.(mu))
            grad_mu_Mtl    = MtlArray(Float32.(grad_mu))
            # grad_lambda, grad_mu = obj_gradient_parallel(lambda, mu, grad_lambda, grad_mu, s_gamma, a, b, c, d, cg1, ng, nl)
             grad_lambda_Mtl, grad_mu_Mtl = obj_gradient_update_gpu(lambda_Mtl, mu_Mtl, grad_lambda_Mtl, grad_mu_Mtl, at_Mtl, b, c_Mtl, d_Mtl, cg1t_Mtl, sign_gamma_Mtl, ng, nl, ct_Mtl)
             lambda         = Float64(lambda_Mtl)
             mu             = Float64.(Array(mu_Mtl))
             grad_lambda    = Float64(grad_lambda_Mtl)
             grad_mu    = Float64.(Array(grad_mu_Mtl))

    
            @fastmath v_lambda = momentum * v_lambda + alpha * grad_lambda
            @fastmath v_mu     .= momentum .* v_mu     .+ alpha .* grad_mu
            
     
            @fastmath lambda += v_lambda
            @fastmath mu .+= v_mu

            

            #  alpha = 0.9999*alpha

            @fastmath lambda = lambda + alpha*grad_lambda
            @fastmath mu     .= mu .+ alpha.*grad_mu

            # clip!
     
            mu .= max.(mu, 0.0)
     
            s_gamma_norm = 0.0 # Atomic{Float64}(0.0)

            #@threads 
            for el in 1:ng
                cv = @view c[:, el]
               s_gamma_norm += -norm(cg1[el] + lambda * a[el] + dot(mu, cv))
            end

            final_s_gamma_norm = s_gamma_norm
      
            @fastmath obj[ii+1] = final_s_gamma_norm + lambda*b[1] + dot(mu,d) + cg0
            println("$ii     ", obj[ii+1])
            
            
            #  if sqrt(grad_lambda^2 + sum(grad_mu.^2)) < 1
            if abs(obj[ii+1]-obj[ii]) < 0.08
            # if abs(obj[ii+1]-846286.0649394316) < 6.0649394316

                # println( "Converged at iteration $ii")
                break
            end
        end
  
        return obj
    end
##AdaGrad
function adagrad_GPU(
    n_its::Int,
    lambda::Float64,
    mu::Vector{Float64},
    grad_lambda::Float64,
    grad_mu::Vector{Float64},
    a::Vector{Float64},
    b::Vector{Float64},
    c::Matrix{Float64},
    d::Vector{Float64},
    cg1::Vector{Float64},
    cg0::Float64,
    alpha::Float64,
    epsilon::Float64,
    ng::Int,
    nl::Int,
    obj::Vector{Float64}
)
    r_lambda = 0.0
    r_mu = zeros(length(mu))

    
    for ii in 1:n_its

        lambda_Mtl     = Float32(lambda)
        grad_lambda_Mtl= Float32(grad_lambda)
        mu_Mtl         = MtlArray(Float32.(mu))
        grad_mu_Mtl    = MtlArray(Float32.(grad_mu))
        # grad_lambda, grad_mu = obj_gradient_parallel(lambda, mu, grad_lambda, grad_mu, s_gamma, a, b, c, d, cg1, ng, nl)
         grad_lambda_Mtl, grad_mu_Mtl = obj_gradient_update_gpu(lambda_Mtl, mu_Mtl, grad_lambda_Mtl, grad_mu_Mtl, at_Mtl, b, c_Mtl, d_Mtl, cg1t_Mtl, sign_gamma_Mtl, ng, nl, ct_Mtl)
         lambda         = Float64(lambda_Mtl)
         mu             = Float64.(Array(mu_Mtl))
         grad_lambda    = Float64(grad_lambda_Mtl)
         grad_mu    = Float64.(Array(grad_mu_Mtl))

            @fastmath  r_lambda += grad_lambda^2
            @fastmath  r_mu += grad_mu.^2

            @fastmath  lambda += (alpha / (sqrt(r_lambda) + epsilon)) * grad_lambda
            @fastmath  mu += (alpha ./ (sqrt.(r_mu) .+ epsilon)) .* grad_mu

            mu = max.(mu, 0.0)

            s_gamma_norm = Atomic{Float64}(0.0)

            @threads for el in 1:ng
                cv = @view c[:, el]
                atomic_add!(s_gamma_norm, Float64(-norm(cg1[el] + lambda * a[el] + dot(mu, cv))))
            end

            final_s_gamma_norm = s_gamma_norm[]
            obj[ii+1] = @fastmath @inbounds final_s_gamma_norm + lambda*b[1] + mu'*d + cg0

            if @fastmath abs(obj[ii+1]-obj[ii]) < 0.08 
                break
            end
        end
        return obj
    end
 #QP
L(lambda_v, mu_v, s_v) = -norm(cg1_qt + lambda_v.*at + mu_v'*ct - s_v'*M_soc, 1) - 0.25*(s_v[1:end-1]'*s_v[1:end-1]/(s_v[end])) + lambda_v*b[1] + mu_v'*d + cg0_q
#GDM

    function gd_with_momentum_QP(
        obj::Vector{Float64},
        v_lambda::Float64,
        v_mu::Vector{Float64},
        v_s::Vector{Float64},
        momentum::Float64,
        alpha::Float64,
        alr::Float64,
        mlr::Float64,
        n_its::Int64, lambda::Float64, mu::Vector{Float64},s::Vector{Float64}, grad_lambda::Float64, grad_mu::Vector{Float64}, grad_s::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Matrix{Float64}, d::Vector{Float64}, cg1_qtt::Vector{Float64}, ng::Int64, nl::Int64)

        for ii in 1:n_its

                grad_lambda, grad_mu, grad_s = obj_gradient_SOCP(lambda, mu, grad_lambda, grad_mu, a, b, c, d, cg1_qtt, ng, nl, M_soc, grad_s, s);
    
    
                @fastmath v_lambda = momentum * v_lambda + alpha * grad_lambda
                @fastmath v_mu = momentum * v_mu .+ alpha .* grad_mu
                @fastmath v_s = momentum * v_s .+ alpha .* grad_s
            
                @fastmath lambda += v_lambda
                @fastmath mu += v_mu
                @fastmath s[1:end-1] += v_s
                #  alpha = lr*alpha
    
                @fastmath lambda +=  alpha*grad_lambda
                @fastmath mu     += alpha*grad_mu
           
                # clip!
                mu = max.(mu, 0.0)
    
                @fastmath s[1:end-1] += alpha*grad_s
                # compute!
                alpha= alpha*lr
                obj[ii+1] = L(lambda, mu, s)
                println("$ii     ", obj[ii+1])
      
                if abs(obj[ii+1] - obj[ii]) < 0.08
    
                    # println( "Converged at iteration $ii")
                    break
                end
            end
            return obj
        end

##Adagrad

function adagrad_QP(
    n_its::Int,
    lambda::Float64,
    mu::Vector{Float64},
    s::Vector{Float64},
    grad_lambda::Float64,
    grad_mu::Vector{Float64},
    grad_s::Vector{Float64},
    a::Vector{Float64},
    b::Vector{Float64},
    c::Matrix{Float64},
    d::Vector{Float64},
    cg1_qtt::Vector{Float64},
    cg0_q::Float64,
    alpha::Float64,
    epsilon::Float64,
    ng::Int,
    nl::Int,
    obj::Vector{Float64}
)
    r_lambda = 0.0
    r_mu = zeros(length(mu))
    r_s = zeros(ng)
    for ii in 1:n_its
 
            grad_lambda, grad_mu, grad_s = obj_gradient_SOCP(lambda, mu, grad_lambda, grad_mu, a, b, c, d, cg1_qtt, ng, nl, M_soc, grad_s, s);


            @fastmath  r_lambda += grad_lambda^2
            @fastmath  r_mu += grad_mu.^2
            @fastmath  r_s += grad_s.^2
        
            @fastmath  lambda += (alpha / (sqrt(r_lambda) .+ epsilon)) * grad_lambda
            @fastmath   mu += (alpha ./ (sqrt.(r_mu) .+ epsilon)) .* grad_mu
        
             # clip!
            mu = max.(mu, 0.0)
             #end
             @fastmath s[1:end-1] += (alpha ./ (sqrt.(r_s) .+ epsilon)) .* grad_s
             alpha = lr*alpha
        
             obj[ii+1] = L(lambda, mu, s)
             println("$ii     ", obj[ii+1])

            if abs(obj[ii+1] - obj[ii]) < 0.08

                break
            end
        end
    return obj
end
##ADAM

function adam_QP(
    n_its::Int,
    lambda::Float64,
    mu::Vector{Float64},
    s::Vector{Float64},
    grad_lambda::Float64,
    grad_mu::Vector{Float64},
    grad_s::Vector{Float64},
    a::Vector{Float64},
    b::Vector{Float64},
    c::Matrix{Float64},
    d::Vector{Float64},
    cg1_qtt::Vector{Float64},
    cg0_q::Float64,
    alpha::Float64,
    beta1::Float64,
    beta2::Float64,
    epsilon::Float64,
    ng::Int,
    nl::Int,
    obj::Vector{Float64}
)
    m_lambda = 0.0
    m_mu = zeros(length(mu))
    m_s = zeros(length(s[1:end-1]))
    v_lambda = 0.0
    v_mu = zeros(length(mu))
    v_s = zeros(length(s[1:end-1]))


    for ii in 1:n_its
        grad_l, grad_m, grad_s = obj_gradient_SOCP(lambda, mu, grad_lambda, grad_mu, a, b, c, d, cg1_qtt, ng, nl, M_soc, grad_s, s);

        @fastmath m_lambda = beta1 * m_lambda + (1 - beta1) * grad_l
        @fastmath v_lambda = beta2 * v_lambda + (1 - beta2) * grad_l^2
        
        @fastmath m_lambda_hat = m_lambda / (1 - beta1^ii)
        @fastmath v_lambda_hat = v_lambda / (1 - beta2^ii)

        @fastmath lambda += (alpha / (sqrt(v_lambda_hat) + epsilon)) * m_lambda_hat

        @fastmath m_mu = beta1 * m_mu + (1 - beta1) * grad_mu
        @fastmath v_mu = beta2 * v_mu + (1 - beta2) * grad_mu.^2

        @fastmath m_mu_hat = m_mu / (1 - beta1^ii)
        @fastmath v_mu_hat = v_mu / (1 - beta2^ii)

        @fastmath mu .+= (alpha ./ (sqrt.(v_mu_hat) .+ epsilon)) .* m_mu_hat

        mu = max.(mu, 0.0)

        @fastmath m_s = beta1 * m_s + (1 - beta1) * grad_s
        @fastmath v_s = beta2 * v_s + (1 - beta2) * grad_s.^2
        @fastmath m_s_hat = m_s / (1 - beta1^ii)
        @fastmath v_s_hat = v_s / (1 - beta2^ii)

        @fastmath s[1:end-1] .+= (alpha ./ (sqrt.(v_s_hat) .+ epsilon)) .* m_s_hat
        alpha = alr*alpha
        obj[ii+1] = L(lambda, mu, s)
        println("$ii     ", obj[ii+1])

        if @fastmath abs(obj[ii+1]-obj[ii]) < 0.08

            break
        end
    end
    return obj
end


    