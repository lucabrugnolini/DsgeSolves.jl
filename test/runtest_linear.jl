
## Model
## Two equations NK model on the form (1)-(2):  
# (1) A*Y(+1) = B*Y + C*X
# (2) X(+1) = D*X + E*ϵ, ϵ ~ N(0,Σ) 
# Solution has the form of (3):
# (3) Y(+1) = inv(A)*B*Y + inv(A)*C*X
# provided A full rank and eigenvalues in inv(A)*B respecting Blanchard and Kahn (1980) conditions

## Example
# NKIS: x(t+1) + σ^(-1)π(t+1) = x(t) + σ^(-1)δπ(t) + σ^(-1)v(t)
# NKPC: βπ(t+1) = -kx(t) +π(t)
using DsgeSolves, Plots
plotly()

# Declare number of variables
nb = 2 # #backwardlooking variables
nf = 2 # #forward looking variables
ns = 2 # #forcing variables
m = DSGE(nb,nf,ns)

# Preallocations and hyperparameters
rep = 1000
burn_in = 100

## Indexing variables for filling A
iv = 1
iu = 2
ix = 3
iπ = 4

# Indexing shock
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

## Filling matrices A, B, C
# four variables:
#                  2 forward-looking -- output gap and inflation
#                  2 predetermined -- interest rate shock

# Equation #1 monetary policy shock
# v(t+1) = ρ_vv(t) + ϵ_v(t+1)
nequ = 1
m.A[nequ, iv] = 1
m.B[nequ, iv] = ρ_v
m.C[nequ, iϵ_v] = 1

# Equation #2 cost-push shock
# u(t+1) = ρ_uu(t) + ϵ_u(t+1)
nequ = 2
m.A[nequ, iu] = 1
m.B[nequ, iu] = ρ_u
m.C[nequ, iϵ_u] = 1

# Equation #3 IS curve
# x(t+1) + σ^(-1)π(t+1) = x(t) + σ^(-1)δπ(t) + σ^(-1)v(t)
nequ = 3
m.A[nequ, ix] = 1
m.A[nequ, iπ] = 1/σ
m.B[nequ, iv] = 1/σ
m.B[nequ, ix] = 1
m.B[nequ, iπ] = δ/σ

# Equation #4 NKPC
# βπ(t+1) = -kx(t) +π(t)
nequ = 4
m.A[nequ, iπ] = β
m.B[nequ, iu] = -1
m.B[nequ, ix] = -κ
m.B[nequ, iπ] = 1

## Solve and simulate the model
(A, B, Λ, P, pstar, R) = create_matrices(m)
(w, y) = simulating_dsge(m,pstar,Λ,R,Σ,rep)
# Plot model simulation
pSimulation = plot(layout = grid(2,2));
plot!(pSimulation, w[1,burn_in+1:end], subplot = 1, color = "blue", title = ":ϵ_u");
plot!(pSimulation, w[2,burn_in+1:end], subplot = 2, color = "blue", title = ":ϵ_v");
plot!(pSimulation, y[1,burn_in+1:end], subplot = 3, color = "blue", title = ":x");
plot!(pSimulation, y[2,burn_in+1:end], subplot = 4, color = "blue", title = ":π");
gui(pSimulation)

# Impulse response functions
shk = 1 # choose the shock number
h = 20
(irf_w,irf_y) = irf_dsge(m,pstar,Λ,R,shk,h)
# Plot model IRFs
pIRF = plot(layout = grid(1,2));
plot!(pIRF, irf_y[1,:], subplot = 1, color = "blue", title = ":x");
plot!(pIRF, irf_y[2,:], subplot = 2, color = "blue", title = ":π");
gui(pIRF)

# Computing stylized facts as standard deviation and correlation of series
mean(w,2), mean(y,2)
std(w,2), std(y,2)
