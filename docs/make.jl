using Documenter, DiscreteEntropy

makedocs(;
    modules=[DiscreteEntropy],
    sitename="DiscreteEntropy.jl",
    highlightsig=true,
    checkdocs=:none,
    format=Documenter.HTML(;
        prettyurls=true,
        edit_link="main",
        assets=String[],
        size_threshold=nothing
    ),
    pages=[
        "Overview" => "index.md",
        "Data" => "data.md",
        "Estimators" => "estimators.md",
    ]
)

deploydocs(
    repo = "github.com/kellino/DiscreteEntropy.jl.git",
)
