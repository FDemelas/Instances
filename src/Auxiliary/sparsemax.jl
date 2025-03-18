"""
Sparse-max function.

Presented in:

`From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification`
André F. T. Martins, Ramón Fernandez Astudillo
"""
function sparsemax(γi; dims=1)
	#if dims == 1
	#    return device(sparsemax_1(γi))
	#end
	z = sort(γi; dims, rev = true)
	cs = cumsum(z; dims)
	rng = device(dims == 2 ? repeat(collect(1:size(z, 2))', size(z, 1)) : repeat(collect(1:size(z, 1))', size(z, 2))')
	is_gt = (rng .* z .+ 1 .> cs)
	k = maximum(rng .* is_gt; dims)

	τ = (sum(z .* is_gt ; dims) .- 1) ./ k	
    return relu(γi .- τ)
end

"""
Backward pass for the sparsemax function.
"""
function ChainRulesCore.rrule(::typeof(sparsemax), γi; dims=1)
	#if dims == 1
	#    return ChainRulesCore.rrule(typeof(sparsemax_1), γi)
	#end
	z = sort(γi; dims, rev = true)
	cs = cumsum(z; dims)
	rng = device(dims == 2 ? repeat(collect(1:size(z, 2))', size(z, 1)) : repeat(collect(1:size(z, 1))', size(z, 2))')
	is_gt = (rng .* z .+ 1 .> cs)
	k = maximum(rng .* is_gt; dims)

	τ = (sum(z .* is_gt ; dims) .- 1) ./ k

	val = relu(γi .- τ)
	function loss_pullback(dl)
		non_zeros = device(val .!= 0)
		return (NoTangent(), dl .- sum(dl * non_zeros'; dims) ./ sum(non_zeros; dims), NoTangent())
	end
	return device(val), device(loss_pullback)
end
