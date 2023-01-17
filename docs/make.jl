using Documenter, DiscreteEntropy

makedocs(
    modules=[DiscreteEntropy],
    sitename="DiscreteEntropy.jl",
    highlightsig=true,
    pages=[
        "Home" => "index.md"
    ]
)
