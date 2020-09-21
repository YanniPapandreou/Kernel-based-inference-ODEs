using Distributed
using DifferentialEquations
using Plots

addprocs()
@everywhere using DifferentialEquations


# Linear ODE which starts at 0.5 and solves from t=0.0 to t=1.0
prob = ODEProblem((u,p,t)->1.01u,0.5,(0.0,1.0))

@everywhere function prob_func(prob,i,repeat)
    remake(prob,u0=rand()*prob.u0)
end

ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
sim = solve(ensemble_prob,Tsit5(),EnsembleDistributed(),trajectories=100)

plotly()
plot(sim,linealpha=0.4)

# using multithreading
prob = ODEProblem((u,p,t)->1.01u,0.5,(0.0,1.0))
function prob_func(prob,i,repeat)
    remake(prob,u0=rand()*prob.u0)
end
ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
sim = solve(ensemble_prob,Tsit5(),EnsembleThreads(),trajectories=100)
plot(sim,linealpha=0.4)

# Pre-Determined Initial Conditions
initial_conditions = range(0,stop=1,length=100)
function prob_func(prob,i,repeat)
    remake(prob,u0=initial_conditions[i])
end
ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
sim = solve(ensemble_prob,Tsit5(),EnsembleThreads(),trajectories=100)
plot(sim,linealpha=0.4)


## Example 2: Solving an SDE with Different Parameters

function f(du,u,p,t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
end

function g(du,u,p,t)
    du[1] = p[3]*u[1]
    du[2] = p[4]*u[2]
end

p = [1.5,1.0,0.1,0.1]
prob = SDEProblem(f,g,[1.0,1.0],(0.0,10.0),p)

function prob_func(prob,i,repeat)
    x = 0.3rand(2)
    remake(prob,p=[p[1:2];x])
end

ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
sim = solve(ensemble_prob,SRIW1(),trajectories=10)
plot(sim,linealpha=0.6,color=:blue,vars=(0,1),title="Phase Space Plot")
plot!(sim,linealpha=0.6,color=:red,vars=(0,2),title="Phase Space Plot")

summ = EnsembleSummary(sim,0:0.1:10)
pyplot()
plot(summ,fillalpha=0.5)

## Example 3: Using the Reduction to Halt When Estimator is Within Tolerance

function output_func(sol,i)
    last(sol)
end

prob = ODEProblem((u,p,t)->1.01u,0.5,(0.0,1.0))

function prob_func(prob,i,repeat)
    remake(prob,u0=rand()*prob.u0)
end

function reduction(u,batch,I)
    u = append!(u,batch)
    finished = (var(u) / sqrt(last(I))) / mean(u) < 0.5
    u, finished
end

prob2 = EnsembleProblem(prob,prob_func=prob_func,output_func=output_func,reduction=reduction,u_init=Vector{Float64}())
sim = solve(prob2,Tsit5(),trajectories=10000,batch_size=20)
