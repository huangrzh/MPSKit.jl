"""
see https://arxiv.org/abs/1701.07035
"""
@with_kw struct VUMPS{F} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    orthmaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbose::Bool = Defaults.verbose
end

"
    find_groundstate(state,ham,alg,envs=environments(state,ham))

    find the groundstate for ham using algorithm alg
"

function find_groundstate(state::InfiniteMPS, H,alg::VUMPS,envs=environments(state,H))
    galerkin::Float64  = 1+alg.tol_galerkin
    iter      = 1

    temp_ACs = similar.(state.AC);
    temp_Cs = similar.(state.CR);
    E0 = 1.0
    while true
        m_time = @elapsed begin
            eigalg = Arnoldi(tol=galerkin/(4*sqrt(iter)))

            @sync for (loc,(ac,c)) in enumerate(zip(state.AC,state.CR))
                @Threads.spawn begin
                    (acvals,acvecs) = eigsolve(∂∂AC($loc,$state,$H,$envs),$ac, 1, :SR, eigalg)
                    $temp_ACs[loc] = acvecs[1];
                end

                @Threads.spawn begin
                    (crvals,crvecs) = eigsolve(∂∂C($loc,$state,$H,$envs),$c, 1, :SR, eigalg)
                    $temp_Cs[loc] = crvecs[1];
                end
            end

            for (i,(ac,c)) in enumerate(zip(temp_ACs,temp_Cs))
                QAc,_ = TensorKit.leftorth!(ac, alg=QRpos())
                Qc,_  = TensorKit.leftorth!(c, alg=QRpos())

                temp_ACs[i] = QAc*adjoint(Qc)
            end

            state = InfiniteMPS(temp_ACs,state.CR[end]; tol = alg.tol_gauge, maxiter = alg.orthmaxiter)
            recalculate!(envs,state);

            (state,envs) = alg.finalize(iter,state,H,envs) :: Tuple{typeof(state),typeof(envs)};

            galerkin   = calc_galerkin(state, envs)
        end

        E1 = real(expectation_value(state, envs))
        E1 = mean(E1)
        @printf("%4d    %.2e    %.12e     %.2e    %.2e\n", iter, galerkin, E1,  m_time, 
            (E1-E0)/abs(E0));
        E0 = E1

        if galerkin <= alg.tol_galerkin || iter>=alg.maxiter
            iter>=alg.maxiter && @warn "vumps didn't converge $(galerkin)"
            return state, envs, galerkin
        end

        iter += 1
    end
end
