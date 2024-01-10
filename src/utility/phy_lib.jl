using TensorKit, MPSKit, LinearAlgebra

function PauliMatrix()
    X = zeros(ComplexF64, 2, 2)
    Y = zeros(ComplexF64, 2, 2)
    Z = zeros(ComplexF64, 2, 2)

    X[1, 2] = 1.0
    X[2, 1] = 1.0

    Y[1, 2] = -im
    Y[2, 1] = +im

    Z[1, 1] = +1.0
    Z[2, 2] = -1.0

    V=ℂ
    X = TensorMap(X, V^2, V^2)
    Y = TensorMap(Y, V^2, V^2)
    Z = TensorMap(Z, V^2, V^2)

    return X, Y, Z
end

function _pauliterm(spin, i, j)
    1 <= i <= 2 * spin + 1 || return 0.0
    1 <= j <= 2 * spin + 1 || return 0.0
    return sqrt((spin + 1) * (i + j - 1) - i * j) / 2.0
end

"""
    spinmatrices(spin [, eltype])

the spinmatrices according to [Wikipedia](https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins).
"""
function spinmatrices(s::Union{Rational{Int},Int}, elt=ComplexF64)
    N = Int(2 * s)

    Sx = zeros(elt, N + 1, N + 1)
    Sy = zeros(elt, N + 1, N + 1)
    Sz = zeros(elt, N + 1, N + 1)

    for row in 1:(N + 1)
        for col in 1:(N + 1)
            term = _pauliterm(s, row, col)

            if (row + 1 == col)
                Sx[row, col] += term
                Sy[row, col] -= 1im * term
            end

            if (row == col + 1)
                Sx[row, col] += term
                Sy[row, col] += 1im * term
            end

            if (row == col)
                Sz[row, col] += s + 1 - row
            end
        end
    end
    
    v=ℂ^(N+1)
    Sx = TensorMap(Sx, v, v)
    Sy = TensorMap(Sy, v, v)
    Sz = TensorMap(Sz, v, v)

    return Sx, Sy, Sz
end



# charge: 0,1,...,n-1
function Zn_Potts_Operator(n::Int)
    F=ComplexF64
    V=ℂ
    Sz1 = zeros(F, n, n)
    Sz = zeros(F, n, n)
    Sx = zeros(F, n, n)
    for ie = 1:n
        Sz[ie, ie] = -1
        iem = mod1(ie - 1, n)
        Sx[iem, ie] = 1
        Sz1[ie,ie] = exp(im*(ie-1)*2*pi/n)
    end
    Sz[1, 1] = n - 1
    Si = Matrix{F}(1.0 * I, n, n)
    X = TensorMap(Sx, V^n, V^n)
    Xad = TensorMap(copy(adjoint(Sx)), V^n, V^n)
    Z = TensorMap(Sz, V^n, V^n)
    Z1 = TensorMap(Sz1, V^n, V^n)
    Id = TensorMap(Si, V^n, V^n)
    return X, Z, Id, Xad,Z1
end

# H = -J*(XX^+ + X^+X) - hZ 
function potts_ham(h::Float64, group::String="none"; J=1.0)
    Ham = nothing
    if group=="none"
        X,Z,Id,Xad,_ = Zn_Potts_Operator(3)
        @tensor XX1[-1,-2;-3,-4] := X[-1;-3] * Xad[-2;-4]
        @tensor XX2[-1,-2;-3,-4] := Xad[-1;-3] * X[-2;-4]
        Ham = MPOHamiltonian(-J*XX1-J'*XX2) + MPOHamiltonian(-h * Z)
    elseif group=="Z3"
        Z,XX1,XX2 = potts_op_z3()
        Ham = MPOHamiltonian(-J*XX1-J'*XX2) + MPOHamiltonian(-h * Z)
    elseif group=="z2"
        Z,XX,_ = potts_op_CZ2()
        Ham = MPOHamiltonian(-J*XX) + MPOHamiltonian(-h * Z)
    else
        error("Do not support", group)
    end

    return Ham
end

function potts_op_z3()
    vp = Rep[ℤ₃](0 => 1, 1 => 1, 2 => 1)
    vi = Rep[ℤ₃](1 => 1)

    Z = TensorMap(zeros, ComplexF64, vp, vp)
    blocks(Z)[ℤ₃(0)] .= 2
    for i = 1:2
        blocks(Z)[ℤ₃(i)] .= -1
    end

    X1 = TensorMap(ones, ComplexF64, vp * vi, vp)
    X2 = TensorMap(ones, ComplexF64, vp, vi * vp)
    @tensor XX1[-1 -2; -3 -4] := X1[-1 1; -3] * X2[-2; 1 -4]
    @tensor XX2[-1 -2; -3 -4] := X2[-1; 1 -3] * X1[-2 1; -4]

    return Z, XX1, XX2
end

# charge conjugation 
function potts_op_CZ2()
    X, Z, Id, Xad,Sz = Zn_Potts_Operator(3)
    U = Matrix(I,3,3)*(1.0+0.0im)
    U[2,2] = 1/sqrt(2)
    U[2,3] = 1/sqrt(2)
    U[3,2] = 1/sqrt(2)
    U[3,3] = -1/sqrt(2)
    Z1 = U'*Z.data*U
    Sz1 = U'*Sz.data*U
    M1 = U'*X.data*U
    P1 = copy(M1')
    @tensor xx1[-1,-2;-3,-4] := M1[-1,-3]*P1[-2,-4]
    @tensor xx2[-1,-2;-3,-4] := P1[-1,-3]*M1[-2,-4]
    xx = xx1 + xx2 
    xx = reshape(xx, 9, 9)
    
    vp = Rep[ℤ₂](0 => 2, 1 => 1)
    Z = TensorMap(ones, ComplexF64, vp, vp)
    blocks(Z)[ℤ₂(0)] = Z1[1:2,1:2]
    blocks(Z)[ℤ₂(1)] .= Z1[3,3]

    Sz = TensorMap(ones, ComplexF64, vp, vp)
    blocks(Sz)[ℤ₂(0)] = Sz1[1:2,1:2]
    blocks(Sz)[ℤ₂(1)] .= Sz1[3,3]
    
    XX = TensorMap(zeros, ComplexF64, vp*vp, vp*vp)
    m0 = zeros(ComplexF64)
    blocks(XX)[ℤ₂(0)] = xx[[1,2,4,5,9],[1,2,4,5,9]]
    blocks(XX)[ℤ₂(1)] = xx[[3,6,7,8],[3,6,7,8]]
    
    return Z, XX, Sz
end

function cluster_ising_mpo(h::Float64; hx::Float64=Inf, group="Z2")
    if group == "Z2"
        vp = Rep[ℤ₂](0=>1,1=>1);
        vi = Rep[ℤ₂](1=>1);
        Z = TensorMap(zeros,ComplexF64,vp,vp);
        blocks(Z)[ℤ₂(1)].=-1;
        blocks(Z)[ℤ₂(0)].=+1;
        X1 = TensorMap(ones,ComplexF64,vp*vi,vp);
        X2 = TensorMap(ones,ComplexF64,vp,vi*vp);
        @tensor XZX[-1 -2 -3;-4 -5 -6] := X1[-1 1;-4]*Z[-2;-5]*X2[-3; 1 -6];
        @tensor XX[-1 -2;-3 -4] := X1[-1 1;-3]*X2[-2;1 -4];
        Ham = MPOHamiltonian(-XX) + MPOHamiltonian(-h*XZX)
        return Ham, XX, XZX
    else
        X,_,Z = PauliMatrix()
        @tensor XZX[-1 -2 -3;-4 -5 -6] := X[-1 1;-4]*Z[-2;-5]*X[-3; 1 -6];
        @tensor XX[-1 -2;-3 -4] := X[-1 1;-3]*X[-2;1 -4];
        Ham = MPOHamiltonian(-XX) + MPOHamiltonian(-h*XZX)
        if isfinite(hx)
            Ham = MPOHamiltonian(-XX) + MPOHamiltonian(-h*XZX) + MPOHamiltonian(hx*X)
        end
        return Ham, XX, XZX
    end
end


# H = -XX - h Z
function qising_ham_z2(; h::Float64=1.0)
    vp = Rep[ℤ₂](0 => 1, 1 => 1)
    vi = Rep[ℤ₂](1 => 1)

    Z = TensorMap(zeros, ComplexF64, vp, vp)
    blocks(Z)[ℤ₂(1)] .= -1
    blocks(Z)[ℤ₂(0)] .= +1

    X1 = TensorMap(ones, ComplexF64, vp * vi, vp)
    X2 = TensorMap(ones, ComplexF64, vp, vi * vp)
    @tensor XX[-1 -2; -3 -4] := X1[-1 1; -3] * X2[-2; 1 -4]
    Ham = MPOHamiltonian(-XX) + MPOHamiltonian(-h * Z)
    return Ham
end


function qising_ham(; h::Float64=1.0, V=ℂ)
    X, _, Z = PauliMatrix()
    @tensor ZZ[-1 -2; -3 -4] := Z[-1; -3] * Z[-2; -4]
    Ham = MPOHamiltonian(-ZZ) + MPOHamiltonian(-h * X)
    return Ham
end

# H = XX + YY
function xx_ham(group::String, nsite::Int=1)
    H = nothing
    if group == "none"
        X, Y, Z = PauliMatrix()
        Id = zeros(ComplexF64, 2, 2)
        Id[1, 1] = 1.0
        Id[2, 2] = 1.0
        Id = TensorMap(Id, ℂ^2, ℂ^2)

        @tensor XX[-1 -2; -3 -4] := X[-1; -3] * X[-2; -4]
        @tensor YY[-1 -2; -3 -4] := Y[-1; -3] * Y[-2; -4]
        H = -XX + YY
        Ham = MPOHamiltonian(H)
    elseif group == "U1"
        vp = U1Space(1//2 => 1, -1//2 => 1)
        vi = U1Space(1 => 1)
        Sp = TensorMap(ones, ComplexF64, vp, vp * vi)
        Sm = TensorMap(ones, ComplexF64, vp * vi, vp)

        @tensor Hpm[-1 -2; -3 -4] := Sp[-1; -3 1] * Sm[-2 1; -4]
        @tensor Hmp[-1 -2; -3 -4] := Sm[-1 1; -3] * Sp[-2; -4 1]
        H = -2 * (Hpm + Hmp)
        Ham = MPOHamiltonian(MPSKit.decompose_localmpo(MPSKit.add_util_leg(H)))
    else
        error("Do not support", group)
    end

    if nsite > 1
        Ham = repeat(Ham, nsite)
    end

    return Ham, H
end


# H = g1 Z  + g2 XZX 
function x_zxz_ham(group::String, g1, g2)
    if group == "none"
        X, Y, Z = PauliMatrix()
        @tensor XZX[-1 -2 -3;-4 -5 -6] := X[-1;-4]*Z[-2;-5]*X[-3;-6];
        Ham = MPOHamiltonian(g1*Z) + MPOHamiltonian(g2*XZX)
    elseif group == "Z2"
        vp = Rep[ℤ₂](0 => 1, 1 => 1)
        vi = Rep[ℤ₂](1 => 1)

        Z = TensorMap(zeros, ComplexF64, vp, vp)
        blocks(Z)[ℤ₂(1)] .= -1
        blocks(Z)[ℤ₂(0)] .= +1

        X1 = TensorMap(ones, ComplexF64, vp * vi, vp)
        X2 = TensorMap(ones, ComplexF64, vp, vi * vp)
        @tensor XX[-1 -2; -3 -4] := X1[-1 1; -3] * X2[-2; 1 -4]
        @tensor XZX[-1 -2 -3;-4 -5 -6] := X1[-1 1;-4]*Z[-2;-5]*X2[-3; 1 -6];
        Ham = MPOHamiltonian(g1*Z) + MPOHamiltonian(g2*XZX)
    else
        error("Do not support", group)
    end

    return Ham
end

# H = XX + YY + g*ZZ
function xxz_ham(;g::Float64=0.0,group::String="none")
    H = nothing
    if group == "none"
        X, Y, Z = PauliMatrix()
        Id = zeros(ComplexF64, 2, 2)
        Id[1, 1] = 1.0
        Id[2, 2] = 1.0
        Id = TensorMap(Id, ℂ^2, ℂ^2)

        @tensor XX[-1 -2; -3 -4] := X[-1; -3] * X[-2; -4]
        @tensor YY[-1 -2; -3 -4] := Y[-1; -3] * Y[-2; -4]
        @tensor ZZ[-1 -2; -3 -4] := Z[-1; -3] * Z[-2; -4]
        #@tensor ZI[-1 -2;-3 -4] := Z[-1;-3]*Id[-2; -4];
        H = -XX + YY - g * ZZ
        Ham = MPOHamiltonian(H)
    elseif group == "U1"
        vp = U1Space(1/2 => 1, -1/2 => 1)
        vi = U1Space(1 => 1)
        Sp = TensorMap(ones, ComplexF64, vp, vp * vi)
        Sm = TensorMap(ones, ComplexF64, vp * vi, vp)
        Sz = TensorMap(ones, ComplexF64, vp, vp)
        blocks(Sz)[Irrep[U₁](1/2)] = zeros(ComplexF64, 1, 1) .+ 0.5
        blocks(Sz)[Irrep[U₁](-1/2)] = zeros(ComplexF64, 1, 1) .- 0.5
        @tensor Hpm[-1 -2; -3 -4] := Sp[-1; -3 1] * Sm[-2 1; -4]
        @tensor Hmp[-1 -2; -3 -4] := Sm[-1 1; -3] * Sp[-2; -4 1]
        @tensor Hzz[-1 -2; -3 -4] := Sz[-1; -3] * Sz[-2; -4]
        H = 2.0 * (Hpm + Hmp) + (4.0*g) * Hzz
        Ham = MPOHamiltonian(MPSKit.decompose_localmpo(MPSKit.add_util_leg(H)))
    elseif group=="CU1"
        X, Z, Id, Xad = Zn_Potts_Operator(3)
        U = Matrix(I,3,3)*(1.0+0.0im)
        U[2,2] = 1/sqrt(2)
        U[2,3] = 1/sqrt(2)
        U[3,2] = 1/sqrt(2)
        U[3,3] = -1/sqrt(2)
        Z1 = U'*Z.data*U
        M1 = U'*X.data*U
        P1 = copy(M1')
        @tensor xx1[-1,-2;-3,-4] := M1[-1,-3]*P1[-2,-4]
        @tensor xx2[-1,-2;-3,-4] := P1[-1,-3]*M1[-2,-4]
        xx = xx1 + xx2 
        xx = reshape(xx, 9, 9)

        vp = CU1Space((1/2,2) => 1)
        Z = TensorMap(ones, ComplexF64, vp, vp)
        blocks(Z)[ℤ₂(0)] = Z1[1:2,1:2]
        blocks(Z)[ℤ₂(1)] .= Z1[3,3]
        
        XX = TensorMap(zeros, ComplexF64, vp*vp, vp*vp)
        m0 = zeros(ComplexF64)
        blocks(XX)[ℤ₂(0)] = xx[[1,2,4,5,9],[1,2,4,5,9]]
        blocks(XX)[ℤ₂(1)] = xx[[3,6,7,8],[3,6,7,8]]


        Sp = TensorMap(ones, ComplexF64, vp, vp * vi)
        Sm = TensorMap(ones, ComplexF64, vp * vi, vp)
        Sz = TensorMap(ones, ComplexF64, vp, vp)
        blocks(Sz)[Irrep[U₁](1/2)] = zeros(ComplexF64, 1, 1) .+ 0.5
        blocks(Sz)[Irrep[U₁](-1/2)] = zeros(ComplexF64, 1, 1) .- 0.5
        @tensor Hpm[-1 -2; -3 -4] := Sp[-1; -3 1] * Sm[-2 1; -4]
        @tensor Hmp[-1 -2; -3 -4] := Sm[-1 1; -3] * Sp[-2; -4 1]
        @tensor Hzz[-1 -2; -3 -4] := Sz[-1; -3] * Sz[-2; -4]
        H = 2.0 * (Hpm + Hmp) + (4.0*g) * Hzz
        Ham = MPOHamiltonian(MPSKit.decompose_localmpo(MPSKit.add_util_leg(H)))
    else
        error("Do not support", group)
    end

    return Ham, H
end

# S is a diagonal non-negative matrix
function entanglement_in_S(S::TensorMap)
    EnSp = []
    Qs = []
    for (q, blk) in blocks(S)
        push!(Qs, q)
        push!(EnSp, diag(blk))
    end

    E0 = sort(diag(convert(Array, S)), rev=true)
    E0 = E0 .^ 2
    E0 = E0 / sum(E0)
    Sv = -real(sum(E0 .* log.(E0)))
    return Sv, Qs, EnSp
end

function KW_mpo()
    CZ = zeros(2,2,2,2) # ip,jp,i,j
    for i = 1:2
        for j=1:2
            if i==2 && j==2
                CZ[i,j,i,j] = -1
            else
                CZ[i,j,i,j] = 1
            end
        end
    end
    CZ = TensorMap(CZ, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    u,s,vt,ε = tsvd(CZ, (1,3,),(2,4,), trunc=truncbelow(1e-12))
    u = u*s
    gplus = TensorMap(ones, ComplexF64, ℂ^2, ℂ^1)
    gad = copy(gplus');
    @tensor g[-1,-2,-3,-4] := vt[-2,2,-3]*u[-1,2,-4]  
    @tensor g2[-1,-2,-3,-4,-10,-11] := gplus[1,-10]*g[-1,-2,1,3]*g[2,3,-3,-4]*gad[-11,2]
    g2mat = convert(Array, g2)
    g2mat = reshape(g2mat, (2,2,2,2))

    gKW = TensorMap(g2mat, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    return gKW
end

function KW_mpo2()
    CZ = zeros(2,2,2,2) # ip,jp,i,j
    for i = 1:2
        for j=1:2
            if i==2 && j==2
                CZ[i,j,i,j] = -1
            else
                CZ[i,j,i,j] = 1
            end
        end
    end
    CZ = TensorMap(CZ, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    u,s,vt,ε = tsvd(CZ, (1,3,),(2,4,), trunc=truncbelow(1e-12))
    u = u*s
    gplus = TensorMap(ones, ComplexF64, ℂ^2, ℂ^1)
    gad = copy(gplus');
    @tensor g[-1,-2,-3,-4] := vt[-2,2,-3]*u[-1,2,-4]  
    @tensor g2[-1,-2,-3,-4,-10,-11] := gplus[1,-10]*g[-1,-2,1,3]*g[2,3,-3,-4]*gad[-11,2]
    g2mat = convert(Array, g2)
    g2mat = reshape(g2mat, (2,2,2,2))

    gKW = TensorMap(g2mat, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    return gKW
end

function KW_act_MPS(st::InfiniteMPS; gKW=KW_mpo())
#gKW = KW_mpo()
N = length(st.AL)

As = PeriodicArray{Any,1}(nothing, N)
ps = similar(As)
qs = similar(As)
for i = 1:N
    @tensor T[-2,-10,-1;-3,-11] := gKW[-1,-2,1,-3]*st.AL[i][-10,1,-11];
    ps[i],r = leftorth(T, (1,2,), (3,4,5))
    T,qs[i] = rightorth(r, (1,2,), (3,4))
    As[i] = permute(T, (1,2), (3,))
end

#merge bonds
for i = 1:N
    @tensor pq[-1,-2] := qs[i][-1,1,2]*ps[i+1][1,2,-2] 
    @tensor As[i][-1,-2;-3] := As[i][-1,-2,3]*pq[3,-3]
end
As1 = similar(st.AL)
for i = 1:N
    As1[i] = As[i]
end
st = InfiniteMPS(As1)

return st   
end