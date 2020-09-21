using Distributions, Distances, BenchmarkTools

## implementation of cdist
## store samples in the COLUMNS instead of in ROWS
function cdist!(dist_mat::Array{Float64,2},p::Array{Float64,2},q::Array{Float64,2})::Array{Float64,2}
    # m = size(p)[1]
    # n = size(q)[1]

    #dist_mat = zero(Array{Float64}(undef,m,n))
    if size(p) == size(q)
        for i=1:size(p)[2],j=i:size(q)[2]
            if i == j
                @views dist_mat[i,j] = L2dist(p[:,i],q[:,j])
                continue
            end
            @views dist_mat[i,j] = L2dist(p[:,i],q[:,j])
            @views dist_mat[j,i] = L2dist(p[:,j],q[:,i])
        end
    else
        for i=1:size(p)[2], j=1:size(q)[2]
            #@views dist_mat[i,j] = sum((p[i,:].-q[j,:]).^2)
            @views dist_mat[i,j] = L2dist(p[:,i],q[:,j])
        end
    end

    #dist_mat .= sqrt.(dist_mat)
    return dist_mat
end

d1 = Normal()
d2 = Normal(2,1)
p = rand(d1,2,10)
q = rand(d2,2,10)
dist_mat = zero(Array{Float64}(undef,size(p)[2],size(q)[2]))
cdist!(dist_mat,p,q)
for i=1:size(p)[2],j=1:size(q)[2]
    if dist_mat[i,j]!=sqrt(sum((p[:,i]-q[:,j]).^2))
        println("false")
    end
end

R = zero(Array{Float64}(undef,size(p)[2],size(q)[2]))
pairwise!(R,Euclidean(),p,q,dims=2)
R.≈dist_mat

dist_mat = zero(Array{Float64}(undef,100,100))
@benchmark cdist!(dist_mat,p,q) setup=(p=rand(Normal(),10,100);q=rand(Normal(1,1),10,100))

@benchmark pairwise(Euclidean(),p,q,dims=2) setup=(p=rand(Normal(),10,100);q=rand(Normal(1,1),10,100))
