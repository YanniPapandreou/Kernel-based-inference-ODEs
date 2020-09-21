using DiffEqFlux, DifferentialEquations, Plots, Flux, Optim, DiffEqSensitivity
function lotka_volterra!(du,u,p,t)
  x,y = u
  α,β,γ,δ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = δ*x*y - γ*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)

function multiplicative_noise!(du,u,p,t)
  x,y = u
  du[1] = p[5]*x
  du[2] = p[6]*y
end
p = [1.5,1.0,3.0,1.0,0.3,0.3]

prob = SDEProblem(lotka_volterra!,multiplicative_noise!,u0,tspan,p)
sol = solve(prob)
plot(sol)

using Statistics
ensembleprob = EnsembleProblem(prob)
@time sol = solve(ensembleprob,SOSRI(),saveat=0.1,trajectories=10_000)
truemean = mean(sol,dims=3)[:,:]
truevar  = var(sol,dims=3)[:,:]

function loss(p)
  tmp_prob = remake(prob,p=p)
  ensembleprob = EnsembleProblem(tmp_prob)
  tmp_sol = solve(ensembleprob,SOSRI(),saveat=0.1,trajectories=1000,sensealg=ForwardDiffSensitivity())
  arrsol = Array(tmp_sol)
  sum(abs2,truemean - mean(arrsol,dims=3)) + 0.1sum(abs2,truevar - var(arrsol,dims=3)),arrsol
end

function cb2(p,l,arrsol)
  @show p,l
  means = mean(arrsol,dims=3)[:,:]
  vars = var(arrsol,dims=3)[:,:]
  p1 = plot(sol[1].t,means',lw=5)
  scatter!(p1,sol[1].t,truemean')
  p2 = plot(sol[1].t,vars',lw=5)
  scatter!(p2,sol[1].t,truevar')
  p = plot(p1,p2,layout = (2,1))
  display(p)
  false
end

pinit = [1.2,0.8,2.5,0.8,0.1,0.1]
@time res = DiffEqFlux.sciml_train(loss,pinit,ADAM(0.05),cb=cb2,maxiters = 100)
