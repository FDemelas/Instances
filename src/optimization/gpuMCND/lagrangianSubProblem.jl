"""
# Arguments:

- `ins`: gpuMCNDinstance of the problem
- `π`: a Lagrangian multipliers Vector

Compute LR using the GPU.
regY is the regularization vector for y and α a multiplicative parameter.
It does not requires the solution vector x,y as input and it return them as output.
"""
function LR(ins::gpuMCNDinstance, z)

	z = gpu(z)
	# compute the costs for the sub-problem
	c̃ = (gpu(ins.r) + z * ins.MHT) .* ins.M01

	# sort the demands in an increasing way based to the value of
	# the costs c̃ 
	demands = hcat(sortperm.(eachcol(c̃))...)

	# compute a matrix with sizes |demands| x |arcs| with
	# the volume of the demands sorted considering the matrix
	# demands previously defined from c̃
	SQ = ins.q[demands]

	#create the cumulative sum matrix for the volumes 
	sumQ = cumsum(SQ, dims = 1)

	# construct the x that we'll be fully remplited
	# in this case their value will be the volume
	isFull = -sumQ .+ gpu(ins.c') .>= 0
	xFull = zeros(Float32, size(isFull)) |> gpu
	xFull[demands.+ins.eIDX] = isFull .* SQ

	#construct the x that cannot be fully remplited
	# in this case their value will be the residual capacity 
	isFullp1 = circshift(isFull, 1)
	isFullp1[1, :] .= 1
	xPart = zeros(Float32, size(isFull)) |> gpu
	xPart[demands.+ins.eIDX] = (isFullp1 - isFull) .* (-sumQ .+ gpu(ins.c') + SQ)

	# final flow solution composed of the two vector xFull
	# and xPart previously defined
	x = xFull + xPart

	# put to zero in the flow solution all the component
	# with non-negative objective cos
	x .*= (c̃ .< 0)

	# compute the objective value of all sub-problem
	# in the case in which we activate the arc
	f = gpu(ins.f')
	LRe = sum(c̃ .* x, dims = 1) + f

	# we active an arc if and only if the cost of the associated sub-problem 
	# is negative
	y = LRe .< 0

	x .*= y

	#compute the final objective function
	# as the difference of all sub-problems with negative objective value
	#¯and the constant Lagrangian bound
	LReValues = LRe .* (LRe .<= 0)

	LReValue = sum(LReValues)
	constantBound = sum(ins.B .* z)
	obj = LReValue - constantBound
	return obj, x, y
end
