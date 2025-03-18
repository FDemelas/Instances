
abstract type abstractInstanceFeaturesType end


abstract type abstractGraphInstanceFeaturesType end


struct noCrGraphInstanceFeaturesType <: abstractGraphInstanceFeaturesType end


function size_ins_features(ϕ::LagrangianFunctionMCND,fmt::noCrGraphInstanceFeaturesType)
    return 1
end

function features_matrix(ϕ::LagrangianFunctionMCND,fmt::abstractInstanceFeaturesType)
    s_cv = get_number_cv(ϕ) 
    s_iv = get_number_iv(ϕ)
    s_rc = get_number_rc(ϕ)
    s_kc = get_number_kc(ϕ)
    f = zeros(4*size_ins_features(ϕ,fmt),s_cv+s_rc+s_kc+s_iv)
    idx = 1
    for i in 1:s_rc
        f[1,idx] = get_rhs_rc(ϕ,i)
        idx += 1
    end
    for i in 1:s_kc
        f[2,idx] = get_rhs_kc(ϕ,i)
        idx += 1
    end
    for i in 1:s_cv
        f[3,idx] = get_costs_cv(ϕ,i)
        idx += 1
    end
    for i in 1:s_iv
        f[4,idx] = get_costs_iv(ϕ,i)
        idx += 1
    end
    return f
end

function from_couple_to_idx(j::Int,i::Int,maxI::Int)
    return Int64(( j - 1 ) * maxI + i )
end

function from_idx_to_couple(ji::Int,maxI::Int)
    i = ( ji % maxI ) 
    i = i ≈ 0 ? maxI : i
    j = ( ji - i ) / maxI + 1
    return Int64(j), Int64(i) 
end

function get_costs_cv(ϕ::LagrangianFunctionMCND, i::Int)
    e,k = from_idx_to_couple(i,sizeK(ϕ.inst))
    return ϕ.inst.r[ k , e ]
end


function get_costs_iv(ϕ::LagrangianFunctionMCND, i::Int)
    return ϕ.inst.f[i]
end


function get_rhs_rc(ϕ::LagrangianFunctionMCND, i::Int)
    k,v = from_idx_to_couple(i,sizeK(ϕ.inst))
    return b(ϕ.inst,k,v)
end

function get_rhs_kc(ϕ::LagrangianFunctionMCND, i::Int)
    return ϕ.inst.c[i]
end

function get_number_cv(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst)*sizeK(ϕ.inst)
end

function get_number_iv(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst)
end

function get_number_rc(ϕ::LagrangianFunctionMCND)
    return sizeK(ϕ.inst)*sizeV(ϕ.inst)
end

function get_number_kc(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst)
end

function get_number_c(ϕ::LagrangianFunctionMCND) 
    return get_number_rc(ϕ) + get_number_kc(ϕ)
end

function number_non_zeros_coefficients(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst) * (1 + 3 * sizeK(ϕ.inst))
end

function preprocess_weight(fmt::abstractInstanceFeaturesType, weight::Real)
    return exp(weight)
end

function get_coefficient(ϕ::LagrangianFunctionMCND,i_c::Int,i_v::Int)
    if i_c > get_number_c(ϕ)
        println(" Something wrong with indexes !")
        return 0.0
    else
        if i_v <= get_number_c(ϕ)
            println(" Something wrong with indexes !")
            return 0.0
        else
            if i_v <= get_number_c(ϕ) + get_number_cv(ϕ)
                if i_c <= get_number_rc(ϕ)
                    v_c, k_c = from_idx_to_couple(i_c,sizeK(ϕ.inst))
                    idx_v = i_v - get_number_c(ϕ)
                    e_v, k_v = from_idx_to_couple(idx_v,sizeK(ϕ.inst))
                    coeff = tail(ϕ.inst,e_v) == v_c ? -1.0 : ( head(ϕ.inst,e_v) == v_c ? 1.0 : 0.0 ) 
                    return k_c == k_v ? coeff : 0.0
                else
                    e_c = i_c - get_number_rc(ϕ)
                    idx_v = i_v - get_number_rc(ϕ) - get_number_kc(ϕ) 
                    e_v,_ = from_idx_to_couple(idx_v,sizeInputSpace(ϕ)[1])
                    return e_v == e_c ? 1.0 : 0.0
                end
            else
                if i_c <= get_number_rc(ϕ)
                   return 0.0
                else
                    idx_v = i_v - get_number_rc(ϕ) - get_number_kc(ϕ) - get_number_cv(ϕ)
                    e_v,_ = from_idx_to_couple(idx_v,sizeK(ϕ))
                    e_c = i_c - get_number_rc(ϕ)
                    return e_v == e_c ? - ϕ.inst.c[e_c] : 0.0
                end   
            end 
        end
    end
end

function featuresExtraction( ϕ::LagrangianFunctionMCND, fmt::abstractGraphInstanceFeaturesType)
	s_cv = get_number_cv(ϕ) 
    s_iv = get_number_iv(ϕ)
    s_rc = get_number_rc(ϕ)
    s_kc = get_number_kc(ϕ)
    
    lnfc = s_rc + 1
    lnkc = lnfc + s_kc
	lnx = lnkc + s_cv
	lny = lnx + s_iv

	nodesRC = collect( 1:(lnfc - 1) )
	nodesKC = collect( lnfc:(lnkc - 1) )
	nodesx  = collect( lnkc:(lnx - 1) )
	nodesy  = collect( lnx:(lny - 1) )

	sizeBiArcs = 2 * number_non_zeros_coefficients(ϕ)
	tails = zeros(Int64, sizeBiArcs)
	heads = zeros(Int64, sizeBiArcs)
	weightsE = zeros(Float32, sizeBiArcs)

	tmp = 1
    for i in union(nodesRC,nodesKC)
        for j in union(nodesx,nodesy)
            s,e=i,j
            weight = get_coefficient(ϕ,i,j)
            if !(weight ≈ 0)
                tails[tmp] = s
                heads[tmp] = e
                weightsE[tmp] = preprocess_weight(fmt, weight)
                tmp += 1

            	tails[tmp] = e
    	        heads[tmp] = s
    	        weightsE[tmp] = preprocess_weight(fmt, weight)
    	        tmp += 1 
            end
        end
    end
    
	f = features_matrix(ϕ, fmt)

	g = GNNGraph(tails, heads, weightsE, ndata = f, gdata = [sizeInputSpace(ϕ)])

	return add_self_loops(g)
end