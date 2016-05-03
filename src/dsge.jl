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
  P21 = pstar[(m.b+1):(m.b+m.f)]
  P22 = pstar[(m.b+1):(m.b+m.f),(m.b+1):(m.b+m.f)]

  R1 = R[1:m.s,1:m.s]
  R2 = R[(m.b+1:m.s),1:m.s]
  return Λ1, Λ2, P11, P12, P21, P22, R1, R2
end

function simulating_dsge(m,pstar,Λ,R,rep)
  (Λ1, Λ2, P11, P12, P21, P22, R1, R2) = create_partition(m,pstar,Λ,R)
    for i in 2:rep
      w[:,i] = real(inv(P11 - P12*inv(P22)*P21)*Λ1*(P11 - P12*inv(P22)*P21))*w[:,i-1] + real((P11 - P12*inv(P22)*P21))*rand(m.v)
      y[:,i] = -real(inv(P22)*P21)*w[:,i]'
    end
return w, y
end

function irf_dsge(m,pstar,Λ,R,rep)
  (Λ1, Λ2, P11, P12, P21, P22, R1, R2) = create_partition(m,pstar,Λ,R)
  shock = zeros(1,rep)
  shock[1,2] = 1
    for i in 2:rep
      w[:,i] = real(inv(P11 - P12*inv(P22)*P21)*Λ1*(P11 - P12*inv(P22)*P21))*w[:,i-1] + real((P11 - P12*inv(P22)*P21))*shock[:,i]
      y[:,i] = -real(inv(P22)*P21)*w[:,i]'
    end
return w, y
end

m = DSGE(1,2,1)

# Model Example
# Three variables:
#                  2 forward-looking -- output gap and inflation
#                  x(t+1) + σ^(-1)π(t+1) = x(t) + σ^(-1)δπ(t) + σ^(-1)v(t)
#                  βπ(t+1) = -kx(t) +π(t)

#                  1 predetermined -- interest rate shock
#                  v(t+1) = ρv(t) + ϵ(t+1)
β = 0.99
σ = 1
χ = 1.55
η = 0
θ = 2.064
ω = 0.5
α = 3
δ = 1.5
ρ = 0.6
κ = (1-ω)*(1-β*ω)/(α*ω)

m.A0[1,1] = 1
m.A0[2,2] = 1
m.A0[2,3] = 1/σ
m.A0[3,3] = β

m.A1[1,1] = ρ
m.A1[2,1] = 1/σ
m.A1[2,2] = 1
m.A1[2,3] = δ/σ
m.A1[3,2] = -κ
m.A1[3,3] = 1

m.B0[1,1] = 1

rep = 1000
burn_in = rep/1
w = zeros(m.b,rep)
y = zeros(m.f,rep)

(A, B, Λ, P, pstar, R) = create_matrices(m)
(w, y) = simulating_dsge(m,pstar,Λ,R,rep)

plot(w'); hold(true); plot(y[1,1:end]'); hold(true); plot(y[2,1:end]')
plot(w[:,(burn_in+1):end]',w[:,burn_in:end-1]', color = "white")
plot(y[1,(burn_in+1):end]',y[1,burn_in:end-1]', color = "green")
plot(y[2,(burn_in+1):end]',y[2,burn_in:end-1]', color = "red")

# Computing stylized facts as standard deviation and correlation of series
std([w]), std([y[1,:]]), std([y[2,:]])

# Impulse response functions
(irf_w,irf_y) = irf_dsge(m,pstar,Λ,R,rep)
plot(irf_w[:,1:20]'); hold(true); plot(irf_y[1,1:20]'); hold(true); plot(irf_y[2,1:20]')

#FEVD -- need to add a shock oth it makes no sense!
