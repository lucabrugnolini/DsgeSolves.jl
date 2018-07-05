## Projection methods for global solution for non-linear models
## using neural network to approximate the policy function
## Based on Computationa Macroeconomics for the open economy
## Lim and McNelis 2008
## Chapter 2: Flexible Price Model
using Parameters, Distributions, Optim, Plots
plotly()

type Parameter
    Θ_s::Array
    Θ_rf::Array    
    Θ_ss::Array
end 

type Variable
    endo::Array
    state_endo::Array
    euler_error::Array
    other::Array
    shock::Array
end

type Descriptive
    T::Int64
    K::Int64
    B::Int64
    State::Int64
    Euler::Int64
    Neuron::Int64
    Neuronx1::Int64
    Param::Int64
    Start::Int64
    Other::Int
end

type Model
    Θ::Parameter
    Var::Variable
    Des::Descriptive
    d::Distribution
end

function Model(Θ::Parameter,Des::Descriptive,d::Distribution) 
    Var = initialize_system(Θ,Des,d)
    return Model(Θ::Parameter,Var::Variable, Des::Descriptive,d::Distribution)
end

function initialize_system(Θ::Parameter,Des::Descriptive,d::Distribution)
    @unpack T, K, B, State, Euler, Neuron, Neuronx1, Param, Start,Other = Des
    Θ_ss = Θ.Θ_ss
    (Blam_ss,C_ss,F_ss,G_ss,K_ss,L_ss,P_ss,Pk_ss,R_ss,S_ss,W_ss,X_ss,Y_ss,Z_ss) = Θ_ss
    # Endo
    endo = initialize_endo(T,K,Θ_ss)
    # State
    state_endo = initialize_state(T,K,State)  
    # Errors
    euler_error = initialize_error(T,K,Euler)
    #Other
    other = initialize_other(T,K,Other) 
    return Variable(endo,state_endo,euler_error,other,Θ,Des,d)
end

initialize_endo(T,K,Θ_ss) = [initialize_var(T,K,i)  for i in Θ_ss]

function initialize_var(T::Int64,K::Int64,x_ss::Float64)
    x = x_ss*ones(T,K)
end

initialize_state(T,K,State) = [zeros(T,K)  for i in 1:State]
initialize_error(T,K,Euler) = [zeros(T,K)  for i in 1:Euler+1]
initialize_other(T,K,Other) = [zeros(T,K)  for i in 1:Other]

function Variable(endo::Array,state_endo::Array,euler_error::Array,other::Array,Θ::Parameter,Des::Descriptive,d::Distribution)
    @unpack T, K, B, State, Euler, Neuron, Neuronx1, Param, Start,Other = Des
    @unpack Θ_s, Θ_rf, Θ_ss = Θ
    nshock = get_shock(Θ_rf)
    vStd = get_std(Θ_rf,nshock)
    shock = initialize_shock(T,K,vStd,d)
    return  Variable(endo::Array,state_endo::Array,euler_error::Array,other::Array,shock::Array)
end

get_shock(Θ_rf::Array) = Int(length(Θ_rf)/2)
get_std(Θ_rf::Array,nshock::Int64) = Θ_rf[nshock+1:end]

initialize_shock(T::Int64,K::Int64,vStd::Array,d::Distribution) = [rand(d,T,K).*σ for σ in vStd]

function demean(x::Float64,x_ss::Float64)
    x_dm = x - x_ss
end

function ar_process(x_l::Float64,x_ss::Float64,ρ::Float64,shock::Float64)
    x = ρ*log(x_l) + (1-ρ)*log(x_ss) + shock
    return exp(x)
end

function create_neuron(xstate,gamax,jj,kk,Euler,Neuron)
    neuron = zeros(1,Euler*Neuron)
    for nn = 1: Euler*Neuron
        neuron[1,nn] = 1./(1 + exp.(-xstate*gamax[jj[nn]:kk[nn]]))[1] - 0.5
    end
    return neuron
end

function create_endogenous!(m::Model,gamax::Array,jj,kk)
    @unpack T, K, B, State, Euler, Neuron, Neuronx1, Param, Start,Other = m.Des
    @unpack Θ_s, Θ_rf, Θ_ss = m.Θ
    @unpack endo, state_endo, euler_error, other, shock = m.Var
    (Blam_ss,C_ss,F_ss,G_ss,K_ss,L_ss,P_ss,Pk_ss,R_ss,S_ss,W_ss,X_ss,Y_ss,Z_ss) = Θ_ss
    (Rstar,PFstar,PXstar,eta,omega,beta,alpha1,kappa,chis,phi0,phi1)= Θ_s
    (ρ, σ) = Θ_rf
    (Blam,C,F,G,κ,L,P,Pk,R,S,W,X,Y,Z) = endo
    (ZZ, FF, RR) =  state_endo
    (error_C, error_S) = euler_error
    (Zrisk, Zinf, trade, Zder) = other
    zshock = shock[1]
    for j = 1:K
        for i = (Start+1):T
            # Create the log AR(1) shock process and exponenciate it
            Z[i,j] = ar_process(Z[i-1,j],Z_ss,ρ,zshock[i,j])
            # Demeaning the exogenous and endogenous state variables
            ZZ[i,j] = demean(Z[i,j],Z_ss)
            FF[i,j] = demean(F[i - 1,j],F_ss)
            RR[i,j] = demean(R[i - 1,j],R_ss)
            xstate = [ZZ[i,j] FF[i,j] RR[i,j]]
            # Setting the approximating functions
            neuron = create_neuron(xstate,gamax,jj,kk,Euler,Neuron)
            pea_C = neuron[1, 1: Neuron][1]
            pea_S = neuron[1, Neuron + 1:2*Neuron][1]
            C[i,j] = exp(pea_C)*C_ss # L: WHY EXPONENCIATING?
            S[i,j] = exp(pea_S)*S_ss
            # Generating the endogeous variables
            Y[i,j] = C[i,j] + G_ss + X_ss
            LL = 0.5*(1 - alpha1)*(Z[i,j]^kappa)*(Y[i,j]^(1 - kappa))*(C[i,j]^-eta)
            L[i,j] = LL^(1/(1 - kappa + omega))
            L[i,j] = real(L[i,j])
            KK = ((Y[i,j]/Z[i,j])^kappa) - (1 - alpha1)*L[i,j]^kappa
            κ[i,j] = (KK/alpha1)^(1/kappa)
            κ[i,j] = real(κ[i,j])
            mpl = (1 - alpha1)*(Z[i,j]^kappa)*(Y[i,j]/L[i,j])^(1 - kappa)
            mpk = (alpha1)*(Z[i,j]^kappa)*(Y[i,j]/κ[i,j])^(1 - kappa)
            mpl = real(mpl)
            mpk = real(mpk)
            Pk[i,j] = S[i,j]*PFstar
            P[i,j] = 2*Pk[i,j]/mpk
            W[i,j] = (L[i,j]^omega)*(C[i,j]^eta)*P[i,j]
            W[i,j] = real(W[i,j])
            Zinf[i,j] = 0.25*((P[i,j]/P[i - 4,j]) - 1)
            R[i,j] = phi0*R[i - 1,j] + (1 - phi0)*(Rstar + phi1*Zinf[i,j])
            trade[i,j] = P[i,j]*X_ss - S[i,j]*PFstar*κ[i,j]
            trade1 = trade[i,j]/S[i,j]
            F[i,j] = F[i - 1,j]*(1 + Rstar + Zrisk[i - 1,j]) - trade1
            Blam[i,j] = (C[i,j]^-eta)/P[i,j]
            Blam[i,j] = real(Blam[i,j])
            Zrisk[i,j] = sign(F[i - 1,j])*chis*(exp(abs(F[i - 1,j])) - 1)
            Zder[i,j] = chis*(exp(abs(F[i - 1,j])))
            # Obtaining the Euler errors
            MUC = Blam[i,j]*(beta*(1 + R[i - 1,j]))
            MUCLAG = Blam[i - 1,j]
            error_C[i,j] = (MUC/MUCLAG) - 1
            MUS = S[i,j]*(1 + Rstar + Zrisk[i - 1,j] + Zder[i - 1,j]*F[i - 1,j])
            MUSLAG = (1 + R[i - 1,j])*S[i - 1,j]
            error_S[i,j] = (MUS/MUSLAG) - 1
        end
    end
    endo = [Blam,C,F,G,κ,L,P,Pk,R,S,W,X,Y,Z]
    state_endo = [ZZ, FF, RR]
    euler_error = [error_C, error_S]
    other = [Zrisk, Zinf, trade, Zder]
    shock[1] = zshock
    Var = Variable(endo, state_endo, euler_error, other, shock)
    return m = Model(m.Θ,Var,m.Des,m.d)
end

minimize_euler_error!(gamax::Array) = minimize_euler_error!(gamax,m) 

function minimize_euler_error!(gamax::Array,m::Model)
    @unpack T, K, B, State, Euler, Neuron, Neuronx1, Param, Start, Other = m.Des
    jk = Param
    jj = 1:State:Param
    kk = State:State:Param
    # Recursively create the endogenous variables 
    m = create_endogenous!(m,gamax,jj,kk)
    # Defining the errors function to be minimized
    iError = create_euler_error(m)
end

function create_euler_error(m::Model)
    @unpack endo, state_endo, euler_error, other, shock = m.Var
    @unpack T, K, B, State, Euler, Neuron, Neuronx1, Param, Start,Other = m.Des
    mError = reshape_error(euler_error,T*K,1)
    iError = mean_squared_error(mError)
end

reshape_error(euler_error::Array,T::Int64,K::Int64) = hcat([reshape(error,T*K,1) for error in euler_error]...)

mean_squared_error(mError::Array) = mean([sum(mError[i,:])^2 for i in 1:size(mError,1)])

function evaluate_model!(gamax::Array,m::Model)
    @unpack T, K, B, State, Euler, Neuron, Neuronx1, Param, Start,Other = m.Des
    jk = Param
    jj = 1:State:Param
    kk = State:State:Param
    # Recursively create the endogenous variables 
    m = create_endogenous!(m,gamax,jj,kk)
end

function burn_in!(endo::Array, state_endo::Array, euler_error::Array, other::Array, shock::Array, B::Int64)
    endo = [burn_in(i,B) for i in endo]
    state_endo = [burn_in(i,B) for i in state_endo] 
    euler_error = [burn_in(i,B) for i in euler_error] 
    other = [burn_in(i,B) for i in other] 
    shock = [burn_in(i,B) for i in shock]
    return Var = Variable(endo, state_endo, euler_error, other, shock)
end

burn_in(x::Array,B::Int64) = x[B+1:end,:]

function plot_irf!(m::Model,magnitude::Float64,horizon::Int64,timing::Int64)
    compute_irf!(m,magnitude,horizon,timing)
    @unpack endo, state_endo, euler_error, other, shock = m.Var
    (Blam,C,F,G,κ,L,P,Pk,R,S,W,X,Y,Z) = endo
    plot_range = timing:timing+horizon-1
    # Plot
    irf = plot(layout = grid(2,5), legend = false)
    plot!(mean(Z[plot_range,:],2), linewidth = 2, subplot = 1, title = "Z")
    plot!(mean(C[plot_range,:],2), linewidth = 2, subplot = 2, title = "C")
    plot!(mean(S[plot_range,:],2), linewidth = 2, subplot = 3, title = "S")
    plot!(mean(Y[plot_range,:],2), linewidth = 2, subplot = 4, title = "Y")
    plot!(mean(κ[plot_range,:],2), linewidth = 2, subplot = 5, title = "K")
    plot!(mean(L[plot_range,:],2), linewidth = 2, subplot = 6, title = "L")
    plot!(mean(P[plot_range,:],2), linewidth = 2, subplot = 7, title = "P")
    plot!(mean(R[plot_range,:],2), linewidth = 2, subplot = 8, title = "R")
    plot!(mean(F[plot_range,:],2), linewidth = 2, subplot = 9, title = "F")
    plot!(mean(W[plot_range,:],2)./P[plot_range], linewidth = 2, subplot = 10, title = "W/P")
    gui()
end

function compute_irf!(m::Model,magnitude::Float64,horizon::Int64,timing::Int64)
    @unpack T, K, B, State, Euler, Neuron, Neuronx1, Param, Start,Other = m.Des
    vShock = one_time_shock(magnitude,timing,T,K)
    m.Var.shock = vShock 
    m = evaluate_model!(vParam_star,m)
end

one_time_shock(magnitude::Float64,timing::Int64,T::Int64,K::Int64) = [[zeros(timing-1,K); magnitude*ones(1,K) ; zeros(T-timing,K)]]

function judd_gaspar_stat(m::Model)
    # I AM NOT SURE THIS IS CORRECT
    @unpack endo, state_endo, euler_error, other, shock = m.Var
    (Blam,C,F,G,κ,L,P,Pk,R,S,W,X,Y,Z) = endo
    (error_C, error_S) = euler_error
    jg_c = mean(abs.(error_C)./C,2)
    jg_s = mean(abs.(error_S)./S,2)
    jg = histogram(layout = grid(2,1), legend = false)
    histogram!(jg_c, subplot = 1, title = "Error C")
    histogram!(jg_s, subplot = 2, title = "Error S")
end

function haan_marcet_statistics(m::Model)
    # I AM NOT SURE THIS IS CORRECT
    @unpack T, K, B, State, Euler, Neuron, Neuronx1, Param, Start,Other = m.Des
    @unpack endo, state_endo, euler_error, other, shock = m.Var
    DM = zeros(K,Euler)
    for j in 1:Euler
        dm = zeros(K)
        for i in 1:K
            e = euler_error[j][:,i]
            x = hcat([state_endo[j][:,i] for j in 1:State]...)
            Q = (1/T)*e'*x
            A = (1/T)sum(x*x'*e.^2)
            dm[i] = T*Q*A^(-1)*Q'
        end
        DM[:,j] = dm
    end
    return sort(DM,1)
end

function plot_dm_stat(DM::Array)
    # I AM NOT SURE THIS IS CORRECT
    K = size(DM,2)
    dm = plot(layout = grid(K,1), legend = false, show = true)
    [plot!(DM[:,i], subplot = i) for i in 1:K]
end
