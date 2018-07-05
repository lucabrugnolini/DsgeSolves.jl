# Luca Brugnolini
# 20 April 2016
# Based on Exercises for Practical DSGE Modelling

using Distributions
using PyPlot

type DSGE
  b::Int64
  f::Int64
  s::Int64
  A0::Array
  A1::Array
  B0::Array
  v::Distribution
end

function DSGE(b,f,s)
  A0 = zeros(b+f,b+f)
  A1 = zeros(b+f,b+f)
  B0 = zeros(b+f,s)
  v = Normal(0,1)
  return DSGE(b,f,s,A0,A1,B0,v)
end

function create_matrices(m)
  A = inv(m.A0)*m.A1
  B = inv(m.A0)*m.B0
  (λ,P) = eig(A)
  λ = real(λ)
  t = sortrows([λ P'])
  Λ = diagm(t[:,1])
  P = t[:,2:(m.b+m.f+1)]'
  pstar = inv(P)
  R = pstar*B
 return A, B, Λ, P, pstar, R
end

function create_partition(m,pstar,Λ,R)
  Λ1 = Λ[1:m.b,1:m.b]
  Λ2 = Λ[(m.b+1):(m.b+m.f),(m.b+1):(m.b+m.f)]

  P11 = pstar[1:m.b,1:m.b]
  P12 = pstar[1:m.b,(m.b+1):(m.b+m.f)]
  P21 = pstar[(m.b+1):(m.b+m.f),1:m.b]
  P22 = pstar[(m.b+1):(m.b+m.f),(m.b+1):(m.b+m.f)]

  R1 = R[1:m.s,1:m.s]
  R2 = R[(m.b+1:m.s),1:m.s]
  return Λ1, Λ2, P11, P12, P21, P22, R1, R2
end

function simulating_dsge(m,pstar,Λ,R,Σ,rep)
  (Λ1, Λ2, P11, P12, P21, P22, R1, R2) = create_partition(m,pstar,Λ,R)
    for i in 2:rep
      w[:,i] = real(inv(P11-P12*inv(P22)*P21)*Λ1*(P11-P12*inv(P22)*P21))*w[:,i-1]+real((P11-P12*inv(P22)*P21)*R1)*Σ*rand(m.v,m.s)
      y[:,i] = -real(inv(P22)*P21)*w[:,i]
    end
return w, y
end

function irf_dsge(m,pstar,Λ,R,shk,rep)
  (Λ1, Λ2, P11, P12, P21, P22, R1, R2) = create_partition(m,pstar,Λ,R)
  shock = zeros(m.s,rep)
  shock[shk,2] = 1
    for i in 2:rep
      w[:,i] = real(inv(P11-P12*inv(P22)*P21)*Λ1*(P11-P12*inv(P22)*P21))*w[:,i-1]+real((P11-P12*inv(P22)*P21)*R1)*shock[:,i]
      y[:,i] = -real(inv(P22)*P21)*w[:,i]
    end
return w, y
end


## Model Example
nb = 2 # #backwardlooking variables
nf = 2 # #forward looking variables
ns = 2 # #forcing variables
m = DSGE(nb,nf,ns)

rep = 1000
burn_in = rep/10
w = zeros(m.b,rep)
y = zeros(m.f,rep)

## Indexing
# variables
iv = 1
iu = 2
ix = 3
iπ = 4

# shock
iϵ_v = 1
iϵ_u = 2

## Calibration
# Structural parameters
β = 0.99
σ = 1
χ = 1.55
η = 0
θ = 2.064
ω = 0.5
α = 3
δ = 1.5
κ = (1-ω)*(1-β*ω)/(α*ω)

# Non-structural parameters
ρ_u = 0.5
ρ_v = 0.8
σ_v = 1
σ_u = 0.5
Σ = diagm([σ_v, σ_u])

## Filling matrices A0, A1, B0
# four variables:
#                  2 forward-looking -- output gap and inflation
#                  2 predetermined -- interest rate shock

# Equation #1 monetary policy shock
# v(t+1) = ρ_vv(t) + ϵ_v(t+1)
nequ = 1

m.A0[nequ, iv] = 1

m.A1[nequ, iv] = ρ_v

m.B0[nequ, iϵ_v] = 1


# Equation #2 cost-puch shock
# u(t+1) = ρ_uu(t) + ϵ_u(t+1)
nequ = 2

m.A0[nequ, iu] = 1

m.A1[nequ, iu] = ρ_u

m.B0[nequ, iϵ_u] = 1


# Equation #3 IS curve
# x(t+1) + σ^(-1)π(t+1) = x(t) + σ^(-1)δπ(t) + σ^(-1)v(t)
nequ = 3

m.A0[nequ, ix] = 1
m.A0[nequ, iπ] = 1/σ

m.A1[nequ, iv] = 1/σ
m.A1[nequ, ix] = 1
m.A1[nequ, iπ] = δ/σ


# Equation #4 NKPC
# βπ(t+1) = -kx(t) +π(t)
nequ = 4

m.A0[nequ, iπ] = β

m.A1[nequ, iu] = -1
m.A1[nequ, ix] = -κ
m.A1[nequ, iπ] = 1

## Solve and simulate the model
(A, B, Λ, P, pstar, R) = create_matrices(m)
(w, y) = simulating_dsge(m,pstar,Λ,R,rep)

plot(w'); hold(true); plot(y[1,1:end]'); hold(true); plot(y[2,1:end]')
plot(w[:,(burn_in+1):end]',w[:,burn_in:end-1]', color = "yellow")
plot(y[1,(burn_in+1):end]',y[1,burn_in:end-1]', color = "green")
plot(y[2,(burn_in+1):end]',y[2,burn_in:end-1]', color = "red")

# Computing stylized facts as standard deviation and correlation of series
std([w]), std([y[1,:]]), std([y[2,:]])

# Impulse response functions
shk = 2 # choose the shock number
h = 20
(irf_w,irf_y) = irf_dsge(m,pstar,Λ,R,shk,h)
plot(irf_w[:,2:h]');  hold(true); plot(irf_y[:,2:h]')

# FEVD
# Remark: this is a multi-shock concept
(irf_w,irf_y) = fevd(m,pstar,Λ,R,rep)

function fevd(m,pstar,Λ,R,rep) ##restart from this function using a for loop
                               ## with the function irf_dsge
  (Λ1, Λ2, P11, P12, P21, P22, R1, R2) = create_partition(m,pstar,Λ,R)
  irf_w = zeros(m.b,rep,m.b*m.s)
  irf_y = zeros(m.f,rep,m.f*m.s)
  for j in 1:m.s
    shock = zeros(m.s,rep)
    shock[j,2] = 1
    for i in 2:rep
      w[:,i] = real(inv(P11-P12*inv(P22)*P21)*Λ1*(P11-P12*inv(P22)*P21))*w[:,i-1]+real((P11-P12*inv(P22)*P21)*R1)*shock[:,i]
      y[:,i] = -real(inv(P22)*P21)*w[:,i]
    end
    irf_w[:,:,j] = w
    irf_y[:,:,j] = y
  end
  sum(irf_w[1,:,1].^2)Σ[j,j]/(sum(irf_w[1,:,1].^2) + sum(irf_w[1,:,2].^2))
    irf_w[1,i,1]
    den =

return irf_w, irf_y
end
