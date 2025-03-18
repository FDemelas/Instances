using Instances

function my_read_dat_json(path)
    f = JSON.open(path, "r")
    data = JSON.parse(f)
    close(f)
    goldV = data["labels"]["Ld"]

    data_ins = data["instance"]

    n = data_ins["N"]
    commodities = Tuple{Int64, Int64, Int64}[(data_ins["o"][i] + 1, data_ins["d"][i] + 1, data_ins["q"][i]) for i in 1:length(data_ins["q"])]
    edges = Tuple{Int64, Int64}[(data_ins["tail"][i] + 1, data_ins["head"][i] + 1) for i in 1:length(data_ins["head"])]

    fc = Float32.(data_ins["f"])
    c = Float32.(data_ins["c"])
    r = Float32.(hcat(data_ins["r"]...))'

    return Instances.cpuInstanceMCND(n, edges, commodities, fc, r, c), Float32(goldV)
end

function my_read_dat(path)
f = open(path, "r")
sep = "\t"
line = split(readline(f), sep; keepempty = false)

n = parse(Int64, line[1])
E = parse(Int64, line[2])
K = parse(Int64, line[3])
commodities = Tuple{Int64, Int64, Int64}[]
edges = Tuple{Int64, Int64}[]

fc = zeros(E)
c = zeros(E)
r = zeros(K, E)

for e in 1:E
    line = split(readline(f), sep; keepempty = false)
    push!(edges, (parse(Int64, line[2]), parse(Int64, line[1])))
    fc[e] = parse(Int64, line[3])
    c[e] = parse(Int64, line[4])
    for k in 1:K
        line = split(readline(f), sep; keepempty = false)
        r[k, e] = parse(Int64, line[2])
    end
end
for k in 1:(K)
    lineD = split(readline(f), sep; keepempty = false)
    lineO = split(readline(f), sep; keepempty = false)
    if parse(Int64, lineD[3]) >= 0
        push!(commodities, (parse(Int64, lineO[2]), parse(Int64, lineD[2]), parse(Int64, lineD[3])))
    else
        push!(commodities, (parse(Int64, lineD[2]), parse(Int64, lineO[2]), parse(Int64, lineO[3])))
    end
end
close(f)
return Instances.cpuInstanceMCND(n, edges, commodities, fc, r, c)
end
