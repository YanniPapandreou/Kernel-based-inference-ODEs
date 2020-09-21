using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots, Distributions, LinearAlgebra, StatsBase, Distances, Zygote

using Flux.Optimise: update!

function schnakenberg!(dx, x, p, t)
  x1, x2 = x
  w1, w2, w3, w4 = p
  dx[1] = dx1 = exp(w1)*(x1^2)*x2 + exp(w2) - exp(w3)*x1
  dx[2] = dx2 = -exp(w1)*(x1^2)*x2 + exp(w4)
end

function generator(prob,initial_conditions)
    n = size(initial_conditions)[2]
    function prob_func(prob,i,repeat)
        remake(prob,u0=initial_conditions[:,i])
    end
    ensembleprob = EnsembleProblem(prob,prob_func=prob_func)
    sim = solve(ensembleprob,Tsit5(),saveat=tsteps,trajectories=n,sensealg=ForwardSensitivity())
    sim_mat = Array(sim)
    for i=1:n
        if i==1
            global sim_points = transpose(sim_mat[:,:,i])[:]
        else
            sim_points = hcat(sim_points,transpose(sim_mat[:,:,i])[:])
        end
    end
    return sim_points
end

function mmdU(p,q,k)
    # compute number of sampled paths for both distributions
    m = size(p)[2]
    n = size(q)[2]

    D_pp = pairwise(Euclidean(),p,dims=2)
    D_qq = pairwise(Euclidean(),q,dims=2)
    D_pq = pairwise(Euclidean(),p,q,dims=2)

    # evaluate the kernel function for all possible pairs (pp,qq,pq)
    K_pp = k.(D_pp)
    K_qq = k.(D_qq)
    K_pq = k.(D_pq)

    # if translation_invariant
    #     # compute all the distance matrices D_pp, D_qq, D_pq
    #     D_pp = pairwise(Euclidean(),p,dims=2)
    #     D_qq = pairwise(Euclidean(),q,dims=2)
    #     D_pq = pairwise(Euclidean(),p,q,dims=2)
    #
    #     # evaluate the kernel function for all possible pairs (pp,qq,pq)
    #     K_pp = k.(D_pp)
    #     K_qq = k.(D_qq)
    #     K_pq = k.(D_pq)
    # else
    #     K_pp = kernMat(p,p,k)
    #     K_qq = kernMat(q,q,k)
    #     K_pq = kernMat(p,q,k)
    # end

    # compute estimate and return
    (1/(m*(m-1)))*(sum(K_pp) - tr(K_pp)) + (1/(n*(n-1)))*(sum(K_qq)-tr(K_qq)) - (2/(m*n))*sum(K_pq)
end

function my_custom_train!(ps,data,d,k,N_sim,opt)
  # get parameters to be trained
  ps = Flux.params(ps)

  # generate initial conditions and define loss
  sim_initials = rand(d,2,N_sim)
  function loss()
      sim_mat = generator(ps,sim_initials)
      loss = mmdU(data,sim_mat,k)
      return loss
  end
  # get training loss and gradient simultaneously using
  # Zygote.pullback
  training_loss, back = Zygote.pullback(my_loss,ps)
  gs = back(one(training_loss))

  # display training loss
  display(training_loss)

  # update parameters according to chosen optimiser opt
  update!(opt, ps, gs)
end


## Generate true data set

# Noisy Initial condition
d = truncated(Normal(1.0,0.5),0,Inf64)
x0 = rand(d,2)

# Simulation interval and intermediary points
tspan = (0.0, 1.0)
n_steps = 20
tsteps = range(tspan[1],tspan[2],length=n_steps)

# LV equation parameter. p = [α, β, δ, γ]
p_true = log.([1.0,2.0,3.0,4.0])

# Setup the ODE problem, then solve
true_prob = ODEProblem(schnakenberg!, x0, tspan,saveat=tsteps, p_true)

N = 1000 # number of trajectories
initial_conditions = rand(d,2,N)
true_data = generator(true_prob,initial_conditions)

# compute median heuristic
SqDistMat = pairwise(SqEuclidean(),true_data,dims=2)
σ₀ = 1/median(SqDistMat)
k(x) = exp(-σ₀*(x^2))


ps = [0.9534900008539118,-0.555954628240538,1.167516807284435,1.957044540973418]
prob = ODEProblem(schnakenberg!, x0, tspan,saveat=tsteps, ps)
N_sim = 100
opt = Descent(0.1)
my_custom_train!(pinit,true_data,d,k,N_sim,opt)
