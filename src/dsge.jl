# Luca Brugnolini
# 20 April 2016
# Based on Exercises for Practical DSGE Modelling

using Distributions

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
      w[i] = real(inv(P11 - P12*inv(P22)*P21)*Λ1*(P11 - P12*inv(P22)*P21))*w[i-1] + real((P11 - P12*inv(P22)*P21))*rand(m.v)
      y[i] = -real(inv(P22)*P21)*w[i]
    end
return w, y
end

m = DSGE(1,2,1)

# Model
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

rep = 100
w = zeros(m.b,rep)
y = zeros(m.f,rep)

(A, B, Λ, P, pstar, R) = create_matrices(m)
  (w, y) = simulating_dsge(m,pstar,Λ,R,rep)









###############################################################################
# Drawing from a Gaussian
using PyPlot
x = randn(1000,1)
(n,k) = size(x)
mean_x = zeros(n,k)
var_x = zeros(n,k)
for i in 1:n
   mean_x[i] = mean(x[1:i,:])
   var_x[i] = var(x[1:i,:])
end
plot(mean_x); hold(true); plot(var_x, color = "red");

# Simulating an AR(1)
v = 0.1^0.5*randn(n,k)
y = zeros(n,k)
ρ = 1
for i in 2:n
  y[i,:] = ρ*y[i-1,:] + v[i]
end
plot(y)

# Simulating a Markow switching
prob = 0.05
A = 0.01
B = -0.01
y = zeros(n,k)
for i in 1:n
draw = rand(1,1)
if draw[1,1] > (1-prob)
  y[i] = B
else
  y[i] = A
end
end
plot(y)

# Log-linearisation
β = 0.99
σ = 1
χ = 1.55
η = 0
θ = 2.064
ω = 0.5
α = 3
δ = 1.5
ρ = 0.5

Y_bar = ((θ-1)/χ*θ)^(1/(η+σ))
κ = (1-ω)*(1-β*ω)/(α*ω)

A0 = zeros(3,3)
A0[1,1] = 1
A0[2,2] = 1
A0[2,3] = 1/σ
A0[3,3] = β

A1 = zeros(3,3)
A1[1,1] = ρ
A1[2,1] = 1/σ
A1[2,2] = 1
A1[2,3] = 1/σ*δ
A1[3,2] = -κ
A1[3,3] = 1

B0 = zeros(3,1)
B0[1,1] = 1

A = inv(A0)*A1
B = inv(A0)*B0

(val,p) = eig(A)
val = real(val)
Λ = diagm(val)
t = sortrows([val p])
Λ = diagm(t[:,1])
p = t[:,2:4]
pstar =& inv(p)

range = -10:0.1:10
stability = zeros(length(range))
for i in 1:length(range)
  δ = range[i]
  A1[2,3] = 1/σ*δ
  A = inv(A0)*A1
  B = inv(A0)*B0
  (val,p) = eig(A)
  Λ = diagm(val)
  t = sortrows([val p])
  Λ = diagm(t[:,1])
  Λ = abs(Λ)
 (Λ[1,1] < 1 && Λ[2,2] > 1 && Λ[3,3] > 1) && (stability[i] = 1)
end
plot(stability)


# FEVD
beta = 0.99; sigma = 1;     chi = 1.55
eta = 0;     theta = 2.064; omega = 0.5
alpha = 3;   delta = 1.5;   rhov = 0.5
rhou = 0.8;  sigmav = 1;    sigmau = 0.5

kappa=(1-omega)*(1-beta*omega)/(alpha*omega)
