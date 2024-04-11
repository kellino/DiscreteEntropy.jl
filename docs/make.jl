using Documenter, DiscreteEntropy

makedocs(;
    modules=[DiscreteEntropy],
    sitename="DiscreteEntropy.jl",
    highlightsig=true,
    format=Documenter.HTML(;
        prettyurls=true,
        edit_link="main",
        assets=String[],
        size_threshold=nothing
    ),
    pages=[
        "Overview" => "index.md",
        "Data" => "data.md",
        "Estimate_H" => "est_h.md",
        "Estimators" => "estimators.md",
        "Utility Functions" => "utilities.md",
        "Divergence and Distance" => "divergence.md",
        "Mutual Information and Conditional Entropy" => "mutual.md"

    ]
)

deploydocs(
    repo = "github.com/kellino/DiscreteEntropy.jl.git",
)
