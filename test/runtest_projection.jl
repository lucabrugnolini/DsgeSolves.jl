
###########################################################################################################################################################
# Declaring parameter and model descriptives
# Call functions to solve the model
#IMPORTANT: The model needs ONE steady state parameter for each variable

# Set Seed 
srand(1234) 
include(joinpath(Pkg.dir("DsgeSolves"),"src","projection_methods.jl"))

# Define the exogenous variables and the calibrated parameters
const Rstar = 0.01
const PFstar = 1.0
const PXstar = 1.0
const eta = 1.5
const omega = 0.25
const beta = 0.990099009901
const alpha1 = 0.15
const kappa = 0.1
const chis = 0.1
const phi0 = 0.9
const phi1 = 1.5
const Θ_s = [Rstar PFstar PXstar eta omega beta alpha1 kappa chis phi0 phi1]

# Reduced form parameters
const ρ = 0.9
const σ = 0.01
const Θ_rf = [ρ σ]

# Set out the initial steady-state values
const Blam_ss = 3.22696890071349
const C_ss = 0.45793462256871
const F_ss = 0.0
const G_ss = 0.0
const K_ss = 0.02729049017873
const L_ss = 0.74732008502373
const P_ss = 0.99999998539410
const Pk_ss = 1.0000
const R_ss = 0.0100
const S_ss = 0.99999999315490
const W_ss = 0.28812562001981
const X_ss = 0.02729049020372
const Y_ss = 0.48522511277243
const Z_ss = 1.0
const Θ_ss = [Blam_ss C_ss F_ss G_ss K_ss L_ss P_ss Pk_ss R_ss S_ss W_ss X_ss Y_ss Z_ss]

# Aproximating function
const State = 3 # number state variables
const Euler = 2  # number euler equations
const Neuron = 1 # number of neurons
const Neuronx1 = Neuron + 1
const Param = State*Euler*Neuron # number of parameters for the Neural Network function
const Start = 4
const Other = 4

# Create the shocks
const T = 1000 # length of simulated data
const K = 300  # number of simulations
const B = Int(ceil(T*0.25)) #burn-in
# const shock = initialize_shock(T,K,σ)

const Θ = Parameter(Θ_s,Θ_rf,Θ_ss)
const vDes = Descriptive(T, K, B, State, Euler, Neuron, Neuronx1, Param, Start,Other) 
const d1 = Normal(0,σ)
const m = Model(Θ,vDes,d1)

gammaf = [2.8342 -1.6406 -0.0868 0.5264 -0.0000 3.0591] # starting values
model = optimize(minimize_euler_error!, vec(gammaf), BFGS())

# Evaluate the model for the bestσlected parameters
vParam_star = reshape(Optim.minimizer(model),size(gammaf)) 
m = evaluate_model!(vParam_star,m)

# Model assessment
irf = plot_irf!(m,0.1,50,25) 
jg = judd_gaspar_stat(m) # I AM NOT SURE THIS IS CORRECT
DM = haan_marcet_statistics(m) # I AM NOT SURE THIS IS CORRECT


# @unpack endo, state_endo, euler_error, other, shock = m.Var
# (Blam,C,F,G,κ,L,P,Pk,R,S,W,X,Y,Z) = endo

# plot  