using Instances
using Documenter
using Literate

DocMeta.setdocmeta!(Instances, :DocTestSetup, :(using Instances); recursive = true)

makedocs(;
    modules = [Instances],
    authors = "F. Demelas, M. Lacroix, J. Le Roux, A. Parmentier",
    repo = "https://github.com/FDemelas/Instances",
    sitename = "Instances.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/FDemelas/Instances/-/blob/main/src/Instances.jl",
        assets = String[]),
    pages = [
        "Home" => "index.md",
        "Problem statement" => ["MCND" => "MCND.md","GA"=>"GA.md", "CWL"=>"CWL.md", "UC"=>"UC.md"],
        "Lagrangian Sub-Problem" => ["MCND" => "lagrangianSubProblem_MCND.md", "GA" => "lagrangianSubProblem_GA.md", "CWL"=>"lagrangianSubProblem_CWL.md","UC"=>"lagrangianSubProblem_UC.md"],
	"File Format" => "instanceFormat.md",
        "API reference" => "api.md"
    ]
)



deploydocs(;
    repo = "https://github.com/FDemelas/Instances.git",
    devbranch = "master"
)
