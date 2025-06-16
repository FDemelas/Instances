struct cpuTUCinstanceFactory <: abstractInstanceFactory end

mutable struct TUC <: abstractInstance
	I::Int64
	T::Int64
	ts::Vector{Float32}
	D::Vector{Float32}
	MinPower::Vector{Float32}
	MaxPower::Vector{Float32}
	DeltaRampUp::Vector{Float32}
	DeltaRampDown::Vector{Float32}
	QuadTerm::Matrix{Float32}
	StartUpCost::Vector{Float32}
	LinearTerm::Matrix{Float32}
	ConstTerm::Matrix{Float32}
	InitialPower::Vector{Float32}
	InitUpDownTime::Vector{Float32}
	MinUpTime::Vector{Float32}
	MinDownTime::Vector{Float32}
end

nT(uc::TUC) = return length(uc.ts)

function read_nUC(path_dsp)
	# Open dataset
	ds = Dataset(path_dsp)
	# Access the group
	gg = ds.group["Block_0"]
	D = gg["ActivePowerDemand"][:]

	MinPower = Float32[]
	MaxPower = Float32[]
	DeltaRampUp = Float32[]
	DeltaRampDown = Float32[]
	QuadTerm = Float32[]
	StartUpCost = Float32[]
	LinearTerm = Float32[]
	ConstTerm = Float32[]
	InitialPower = Float32[]
	InitUpDownTime = Float32[]
	MinUpTime = Float32[]
	MinDownTime = Float32[]


	T = gg.dim["TimeHorizon"]
	I = gg.dim["NumberUnits"]
	ts = [i for i in 1:length(D)]

	for i in keys(gg.group)[2:end]
		g = gg.group[i]
		append!(MinPower, g["MinPower"][:])
		append!(MaxPower, g["MaxPower"][:])
		append!(DeltaRampUp, g["DeltaRampUp"][:])
		append!(DeltaRampDown, g["DeltaRampDown"][:])
		append!(QuadTerm, ((length(g["QuadTerm"][:]) > 1) ? g["QuadTerm"][:] : Float32[g["QuadTerm"][1] for i in ts]))
		append!(StartUpCost, g["StartUpCost"][:])
		append!(LinearTerm, ((length(g["LinearTerm"][:]) > 1) ? g["LinearTerm"][:] : Float32[g["LinearTerm"][1] for i in ts]))
		append!(ConstTerm, ((length(g["ConstTerm"][:]) > 1) ? g["ConstTerm"][:] : Float32[g["ConstTerm"][1] for i in ts]))
		append!(InitialPower, g["InitialPower"][:])
		append!(InitUpDownTime, g["InitUpDownTime"][:])
		append!(MinUpTime, g["MinUpTime"][:])
		append!(MinDownTime, g["MinDownTime"][:])

	end
	QuadTerm = reshape(QuadTerm, (I, length(ts)))
	LinearTerm = reshape(LinearTerm, (I, length(ts)))
	ConstTerm = reshape(ConstTerm, (I, length(ts)))
	close(ds)
	return TUC(I, T, ts, D, MinPower, MaxPower, DeltaRampUp, DeltaRampDown, QuadTerm, StartUpCost, LinearTerm, ConstTerm, InitialPower, InitUpDownTime, MinUpTime, MinDownTime)
end

function read_1UC(path_dsp)
	# Open dataset
	ds = Dataset(path_dsp)
	# Access the group
	g = ds.group["Block_0"]
	D = [100]#g["ActivePowerDemand"][:]


	T = g.dim["TimeHorizon"]

	ts = [i for i in (T/g.dim["NumberIntervals"]):(T/g.dim["NumberIntervals"]):T]

	MinPower = g["MinPower"][:]
	MaxPower = g["MaxPower"][:]
	DeltaRampUp = g["DeltaRampUp"][:]
	DeltaRampDown = g["DeltaRampDown"][:]
	QuadTerm = (length(g["QuadTerm"][:]) > 1) ? g["QuadTerm"][:] : Float32[g["QuadTerm"][1] for i in ts]
	StartUpCost = g["StartUpCost"][:]
	LinearTerm = (length(g["LinearTerm"][:]) > 1) ? g["LinearTerm"][:] : Float32[g["LinearTerm"][1] for i in ts]
	ConstTerm = (length(g["ConstTerm"][:]) > 1) ? g["ConstTerm"][:] : Float32[g["ConstTerm"][1] for i in ts]
	InitialPower = g["InitialPower"][:]
	InitUpDownTime = g["InitUpDownTime"][:]
	MinUpTime = g["MinUpTime"][:]
	MinDownTime = g["MinDownTime"][:]
	close(ds)
	return TUC(1, T, ts, D, MinPower, MaxPower, DeltaRampUp, DeltaRampDown, QuadTerm', StartUpCost, LinearTerm', ConstTerm', InitialPower, InitUpDownTime, MinUpTime, MinDownTime)
end


function correct_size(x,ts,I)
	T = length(ts)
	return length(x) == T ?  repeat(x',I) : ( length(x) == I ? repeat(x,1,T) : reshape(x,(I,T)) )
end

function read_json(path::String)
	f = open(path, "r")
	res = JSON.parse(f)
	close(f)
	I = maximum(res["Thermal_Unit"])
	T = round(Int64, maximum(res["Time"]) / 24)
	ts = res["Time"]
	D = res["Load"]
	InitUpDownTime = res["storia0"]
	MinUpTime = res["ton"]
	MinDownTime = res["toff"]
	MinPower = res["inf"]
	MaxPower = res["sup"]
	QuadTerm = res["a_comb"]
	LinearTerm = res["b_comb"]
	ConstTerm = res["c_comb"]
	InitialPower = res["P0"]
	DeltaRampUp = res["ramp_up_str"]
	DeltaRampDown = res["ramp_dwn_str"]
	StartUpCost = res["costof"]
	return TUC(I,
		T,
		ts,
		D,
		MinPower,
		MaxPower,
		DeltaRampUp,
		DeltaRampDown,
		correct_size(QuadTerm,ts,I),
		StartUpCost,
		correct_size(LinearTerm,ts,I),
		correct_size(ConstTerm,ts,I),
		InitialPower,
		InitUpDownTime,
		MinUpTime,
		MinDownTime)
end