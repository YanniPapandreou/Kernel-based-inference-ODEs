using Distributions, Distances, BenchmarkTools, LinearAlgebra, Flux, StatsBase

# function kernMat(p,q,k)
#     m = size(p)[2]
#     n = size(q)[2]
#
#     K = zero(Array{Float64}(undef,m,n))
#
#     for i=1:m, j=1:n
#         K[i,j] = k(p[:,i],q[:,j])
#     end
#
#     return K
# end

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

p = [-0.980015   -0.963118  -0.312737  -0.112574   0.533816    0.115972
      0.0993406  -0.895435  -0.517269   0.511351   0.0225609  -1.43622
     -0.852235   -0.145237  -0.123824   0.726614  -0.690407   -0.915984]
q = [1.55577  1.2987    1.08792  0.111985  1.88125  0.292084
     1.52505  0.796801  1.08664  1.5244    2.43204  1.57305
     1.7561   1.40077   1.77895  0.093011  1.83868  1.62734]

k(x) = exp(-(x^2))

mmdU(p,q,k)
gs = gradient((p,q) -> mmdU(p,q,k),p,q)
gs[1]
gs[2]

# @benchmark mmdU(p,q,k) setup=(p=rand(Normal(),1000,100);q=rand(Normal(1,1),1000,100))

# # general kernel
# kGen(x,y) = cos(L2dist(x,zero(similar(x))))*cos(L2dist(y,zero(similar(y))))
#
# mmdU(p,q,kGen,false)
# gs = gradient((p,q) -> mmdU(p,q,kGen,false),p,q)
