using Statistics, Parameters, Printf, LinearAlgebra

@with_kw struct TEBD
    tol = 1e-12;
    maxiter = 100;
    trscheme = truncdim(100) & truncerr(1e-5);
end

function TEBDOneStep(L1, A, B, L2, G, opts::TEBD)
    @tensor W2[-2,-3;-1,-4] := L1[-1,1]*A[1,-2,5]*B[5,-3,6]*L2[6,-4] order=(1,6,5)
    @tensor gW[-1,-2;-3,-4] := G[-2,-3;2,3]*W2[2,3;-1,-4] 

    @tensor nn = conj(gW[1,2;3,4])*gW[1,2;3,4]
    gW /= sqrt(nn)
    (U,S,V,ϵ) = tsvd(gW,trunc=opts.trscheme,alg=TensorKit.SVD())
    S = sdiag_sqrt(S)
    A = U*S
    B = S*V
    B = permute(B, (1,2,), (3,))

    
    #-----------Update Tensor
    iL1 = sdiag_inv(L1)
    iL2 = sdiag_inv(L2)
    @tensor A[-1,-2;-3] := iL1[-1,1]*A[1,-2,-3]
    @tensor B[-1,-2;-3] := B[-1,-2,3]*iL2[3,-3]

    A /= norm(A)
    B /= norm(B)
    S /= norm(S)
    return A,B,S,ϵ
end

function TEBDGsE(L1, A, B, L2, Ham)
    @tensor W2[-2,-3;-1,-4] := L1[-1,1]*A[1,-2,5]*B[5,-3,6]*L2[6,-4] order=(1,6,5)
    @tensor gW[-2,-3;-1,-4] := Ham[-2,-3;2,3]*W2[2,3;-1,-4] 

    @tensor nn = conj(W2[1,2;3,4])*W2[1,2;3,4]
    @tensor E0 = conj(W2[1,2;3,4])*gW[1,2;3,4]
    E0 /= nn
    E0 = real(E0)
    return E0
end

function sdiag_inv(S::AbstractTensorMap)
    toret = similar(S);
    if sectortype(S) == Trivial
        copyto!(toret.data,LinearAlgebra.pinv(S.data));
    else
        for (k,b) in blocks(S)
            copyto!(blocks(toret)[k],LinearAlgebra.pinv(b));
        end
    end
    toret
end

function sdiag_sqrt(S::AbstractTensorMap)
    toret = similar(S);
    if sectortype(S) == Trivial
        copyto!(toret.data,LinearAlgebra.diagm(LinearAlgebra.diag(S.data).^(1/2)));
    else
        for (k,b) in blocks(S)
            copyto!(blocks(toret)[k],LinearAlgebra.diagm(LinearAlgebra.diag(b).^(1/2)));
        end
    end
    toret
end


function simple_tebd(G::TensorMap, As::AbstractArray, Ls::AbstractArray, Ham::TensorMap, opts::TEBD)
    nsite = length(As)
    Err1 = 0.0
    Ls0 = Ls
    EnS0 = 1.0
    istep0 = 10
    for istep = 1:opts.maxiter
        Err0 = 0.0
        for i = [1:2:nsite...,2:2:nsite...]
            A,B,S,ϵ = TEBDOneStep(Ls[i-1], As[i], As[i+1], Ls[i+1], G, opts)
            
            Err0 = max(Err0,ϵ)
            As[i] = A
            As[i+1] = B

            #@show S
            Ls[i] = S
        end

        Sp = convert(Array,Ls[1])
        Sp = real(diag(Sp))
        Sp = Sp.^4
        Sp = Sp / sum(Sp)
        Sp = Sp[Sp.>1e-16]
        Sp = Sp / sum(Sp)
        EnS = -sum(Sp.*log.(Sp))

        Srr = abs(EnS-EnS0)/EnS0
        Db = maximum(dim.(domain.(Ls)))
        if mod(istep,istep0)==0
            E0 = zeros(nsite)
            for i = 1:nsite
                im = mod1(i-1,nsite)
                ip = mod1(i+1,nsite)
                Ei = TEBDGsE(Ls[im],As[i],As[ip],Ls[ip],Ham)
                E0[i] = real(Ei)
            end
            @show E0
            @printf("%4d    %.2e    %.2e    %.2e    %4d\n", istep, EnS, Err0, Srr, Db);
        end
        if (Srr<opts.tol) & (istep>400)
            @printf("%4d    %.2e    %.2e    %.2e    %4d\n", istep, EnS, Err0, Srr, Db);
            break;
        end
        Err1 = max(Err1,Err0)
        EnS0 = EnS
    end 
    
    return As,Ls,Err1
end


function example_tebd()
    opts = TEBD(
        tol = 1e-6,
        maxiter = 200,
        trscheme = truncerr(1e-5)&truncdim(100)
    )

    _,Ham = MPSKit.xxz_ham(g=1.0, group="U1")
    vp = U1Space(1//2 => 1, -1//2 => 1)
    vi = U1Space(0=>1)
    st = InfiniteMPS([vp,vp],[vp,vi])
    As = st.AL
    Ls = st.CR
    
    τ = 0.1
    G = exp(-τ*Ham)
    As, Ls, Err1 = simple_tebd(G, As, Ls, Ham, opts)
end