# Luca Brugnolini
# 20 April 2016
# Based on Exercises for Practical DSGE Modelling
module DsgeSolves
using Distributions

type DSGE
  b::Int64
  f::Int64
  s::Int64
  A::Array
  B::Array
  C::Array
  v::Distribution
end

function DSGE(b,f,s)
  A = zeros(b+f,b+f)
  B = zeros(b+f,b+f)
  C = zeros(b+f,s)
  v = Normal(0,1)
  return DSGE(b,f,s,A,B,C,v)
end

function create_matrices(m)
  A = inv(m.A)*m.B
  B = inv(m.A)*m.C
  (λ,P) = real.(eig(A))
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
  w = zeros(m.b,rep)
  y = zeros(m.f,rep)
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
  w = zeros(m.b,rep)
  y = zeros(m.f,rep)
    for i in 2:rep
      w[:,i] = real(inv(P11-P12*inv(P22)*P21)*Λ1*(P11-P12*inv(P22)*P21))*w[:,i-1]+real((P11-P12*inv(P22)*P21)*R1)*shock[:,i]
      y[:,i] = -real(inv(P22)*P21)*w[:,i]
    end
return w, y
end

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

export DSGE, create_matrices, simulating_dsge, irf_dsge

end # end module
