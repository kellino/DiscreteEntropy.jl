var documenterSearchIndex = {"docs":
[{"location":"est_h/#Estimate_H","page":"Estimate_H","title":"Estimate_H","text":"","category":"section"},{"location":"est_h/","page":"Estimate_H","title":"Estimate_H","text":"The main entry point for the library.","category":"page"},{"location":"est_h/","page":"Estimate_H","title":"Estimate_H","text":"estimate_h","category":"page"},{"location":"est_h/#DiscreteEntropy.estimate_h","page":"Estimate_H","title":"DiscreteEntropy.estimate_h","text":"estimate_h(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}\nestimate_h(data::CountData, ::Type{JackknifeMLE}; corrected=false)\nestimate_h(data::CountData, ::Type{Schurmann}, xi=nothing)\n\nestimate_h(data::CountVector, ::Type{SchurmannGeneralised}, xis::XiVector)\n\nReturn the estimate in nats of Shannon entropy of data using estimator.\n\nExample\n\nimport Random # hide\nRandom.seed!(1) # hide\n\nX = rand(1:10, 1000)\nestimate_h(from_data(X, Samples), Schurmann)\n\nNote: while most calls to estimate_h take a CountData struct, this is not true for every estimator, especially those that work directly over samples, or need to original structure of the histogram.\n\nThis function is a wrapper indended to make using the libary easier. For finer control over some of the estimators, it is advisable to call them directly, rather than through this function.\n\n\n\n\n\n","category":"function"},{"location":"divergence/#Divergence-and-Distance","page":"Divergence and Distance","title":"Divergence and Distance","text":"","category":"section"},{"location":"divergence/","page":"Divergence and Distance","title":"Divergence and Distance","text":"cross_entropy\nkl_divergence\njensen_shannon_divergence\njensen_shannon_distance\njeffreys_divergence","category":"page"},{"location":"divergence/#DiscreteEntropy.cross_entropy","page":"Divergence and Distance","title":"DiscreteEntropy.cross_entropy","text":" cross_entropy(P::CountVector, Q::CountVector, ::Type{T}) where {T<:MaximumLikelihood}\n\nH(PQ) = - sum_x(P(x) log(Q(x)))\n\nCompute the cross entropy of P and Q, given an estimator of type T. P and Q must be the same length. Both vectors are normalised. The cross entropy of a probability distribution, P with itself is equal to its entropy, ie H(P P) = H(P).\n\nExample\n\nP = cvector([1,2,3,4,3,2]) Q = cvector([2,5,5,4,3,4])\n\nce = cross_entropy(P, Q, MaximumLikelihood) 1.778564897565542\n\nNote: not every estimator is currently supported.\n\n\n\n\n\n","category":"function"},{"location":"divergence/#DiscreteEntropy.kl_divergence","page":"Divergence and Distance","title":"DiscreteEntropy.kl_divergence","text":"kl_divergence(p::AbstractVector, q::AbstractVector)::Float64\n\nD_KL(P  Q) = sum_x in X P(x) log left( fracP(x)Q(x) right)\n\nCompute the Kullback-Lebler Divergence between two discrete distributions. Both distributions needs to be defined over the same space, so length(p) == length(q). If the distributions are not normalised, they will be.\n\n\n\n\n\n","category":"function"},{"location":"divergence/#DiscreteEntropy.jensen_shannon_divergence","page":"Divergence and Distance","title":"DiscreteEntropy.jensen_shannon_divergence","text":"jensen_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector)\njensen_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector, estimator::Type{T}) where {T<:NonParamterisedEstimator}\njensen_shannon_divergence(countsP::AbstractVector, countsQ::AbstractVector, estimator::Type{Bayes}, α)\n\nCompute the Jensen Shannon Divergence between discrete distributions P and q, as represented by their histograms. If no estimator is specified, it defaults to MaximumLikelihood.\n\nwidehatJS(p q) = hatHleft(fracp + q2 right) - left( fracH(p) + H(q)2 right)\n\n\n\n\n\n\n","category":"function"},{"location":"divergence/#DiscreteEntropy.jensen_shannon_distance","page":"Divergence and Distance","title":"DiscreteEntropy.jensen_shannon_distance","text":"jensen_shannon_distance(P::AbstractVector, Q::AbstractVector, estimator)\n\nCompute the Jensen Shannon Distance\n\n\n\n\n\n","category":"function"},{"location":"divergence/#DiscreteEntropy.jeffreys_divergence","page":"Divergence and Distance","title":"DiscreteEntropy.jeffreys_divergence","text":"jeffreys_divergence(p, q)\n(link)[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7516653/]\n\nJ(p q) = D_KL(p Vert q) + D_KL(q Vert p)\n\n\n\n\n\n","category":"function"},{"location":"utilities/#Utility-Functions","page":"Utility Functions","title":"Utility Functions","text":"","category":"section"},{"location":"utilities/","page":"Utility Functions","title":"Utility Functions","text":"logx\nxlogx\nto_bits\nto_bans\nmarginal_counts\nbins\nmultiplicities\nfrom_csv","category":"page"},{"location":"utilities/#DiscreteEntropy.logx","page":"Utility Functions","title":"DiscreteEntropy.logx","text":"logx(x)::Float64\n\nReturns natural logarithm of x, or 0.0 if x is zero\n\n\n\n\n\n","category":"function"},{"location":"utilities/#DiscreteEntropy.xlogx","page":"Utility Functions","title":"DiscreteEntropy.xlogx","text":"xlogx(x::Float64)\n\nReturns x * log(x) for x > 0, or 0.0 if x is zero\n\n\n\n\n\n","category":"function"},{"location":"utilities/#DiscreteEntropy.to_bits","page":"Utility Functions","title":"DiscreteEntropy.to_bits","text":"to_bits(x::Float64)\n\nReturn frachlog(2) where h is in nats\n\n\n\n\n\n","category":"function"},{"location":"utilities/#DiscreteEntropy.to_bans","page":"Utility Functions","title":"DiscreteEntropy.to_bans","text":"to_bans(x::Float64)\n\nReturn frachlog(10) where h is in nats\n\n\n\n\n\n","category":"function"},{"location":"utilities/#DiscreteEntropy.marginal_counts","page":"Utility Functions","title":"DiscreteEntropy.marginal_counts","text":"marginal_counts(contingency_matrix::Matrix, dim; normalise=false)\n\nReturn the marginal counts of contingency_matrix along dimension dim.\n\nIf normalised = true, return as probability distribution.\n\n\n\n\n\n","category":"function"},{"location":"utilities/#DiscreteEntropy.bins","page":"Utility Functions","title":"DiscreteEntropy.bins","text":"bins(x::CountData)\n\nReturn the bins (top row) of x.multiplicities\n\n\n\n\n\n","category":"function"},{"location":"utilities/#DiscreteEntropy.multiplicities","page":"Utility Functions","title":"DiscreteEntropy.multiplicities","text":"bins(x::CountData)\n\nReturn the bin multiplicities (bottom row) of x.multiplicities\n\n\n\n\n\n","category":"function"},{"location":"utilities/#DiscreteEntropy.from_csv","page":"Utility Functions","title":"DiscreteEntropy.from_csv","text":"from_csv(file::String, field, ::Type{T}; remove_zeros=false, header=nothing) where {T<:EntropyData}\n\nSimple wrapper around *CSV.File() which returns a CountData object. For more complex requirements, it is best to call CSV directly.\n\n\n\n\n\n","category":"function"},{"location":"utilities/#Non-exported-functions","page":"Utility Functions","title":"Non-exported functions","text":"","category":"section"},{"location":"utilities/","page":"Utility Functions","title":"Utility Functions","text":"DiscreteEntropy.xFx\nDiscreteEntropy.gammalndiff","category":"page"},{"location":"utilities/#DiscreteEntropy.xFx","page":"Utility Functions","title":"DiscreteEntropy.xFx","text":" xFx(f::Function, x)\n\nReturns x * f(x) for x > 0, or 0.0 if x is zero\n\n\n\n\n\n","category":"function"},{"location":"utilities/#DiscreteEntropy.gammalndiff","page":"Utility Functions","title":"DiscreteEntropy.gammalndiff","text":" gammalndiff(x::Float64, dx::Float64)\n\nReturn log Gamma(x + δx) - log Γ(x)\n\n\n\n\n\n","category":"function"},{"location":"data/#DataTypes","page":"Data","title":"DataTypes","text":"","category":"section"},{"location":"data/","page":"Data","title":"Data","text":"EntropyData\nDiscreteEntropy.AbstractCounts\nCountData\nfrom_counts\nfrom_data\nfrom_samples","category":"page"},{"location":"data/#DiscreteEntropy.EntropyData","page":"Data","title":"DiscreteEntropy.EntropyData","text":"abstract type EntropyData\nHistogram <: EntropyData\nSamples <: EntropyData\n\nIt is very easy, when confronted with a vector such as 123454 to forget whether it represents samples from a distribution, or a histogram of a (discrete) distribution. DiscreteEntropy.jl attempts to make this a difficult mistake to make by enforcing a type difference between a vector of samples and a vector of counts.\n\nSee svector and cvector.\n\n\n\n\n\n","category":"type"},{"location":"data/#DiscreteEntropy.AbstractCounts","page":"Data","title":"DiscreteEntropy.AbstractCounts","text":"AbstractCounts{T<:Real,V<:AbstractVector{T}} <: AbstractVector{T}\n\nEnforced type incompatibility between vectors of samples, vectors of counts, and vectors of xi.\n\nCountVector\n\nA vector representing a histogram\n\nSampleVector\n\nA vector of samples\n\nXiVector\n\nA vector of xi values for use with the schurmann_generalised estimator.\n\n\n\n\n\n","category":"type"},{"location":"data/#DiscreteEntropy.CountData","page":"Data","title":"DiscreteEntropy.CountData","text":"CountData\n\nFields\n\nmultiplicities::Matrix{Float64}  : multiplicity representation of data\nN::Float64 : total number of samples\nK::Int64   : observed support size\n\nMultiplicities\n\nAll of the estimators operate over a multiplicity representation of raw data. Raw data takes the form either of a vector of samples, or a vector of counts (ie a histogram).\n\nGiven histogram = [1,2,3,2,1,4], the multiplicity representation is\n\nbeginpmatrix\n4  2  3  1 \n1  2  1  2\nendpmatrix\n\nThe top row represents bin contents, and the bottom row the number of bins. We have 1 bin with a 4 elements, 2 bins with 2 elements, 1 bin with 3 elements and 2 bins with only 1 element.\n\nThe advantages of the multiplicity representation are compactness and efficiency. Instead of calculating the surprisal of a bin of 2 twice, we can calculate it once and multiply by the multiplicity. The downside of the representation may be floating point creep due to multiplication.\n\nConstructor\n\nCountData is not expected to be called directly, nor is it advised to directly manipulate the fields. Use either from_data, from_counts or from_samples instead.\n\n\n\n\n\n","category":"type"},{"location":"data/#DiscreteEntropy.from_counts","page":"Data","title":"DiscreteEntropy.from_counts","text":" from_counts(counts::AbstractVector; remove_zeros::Bool=true)\n from_counts(counts::CountVector, remove_zeros::Bool)\n\nReturn a CountData object from a vector or CountVector. Many estimators cannot handle a histogram with a 0 value bin, so there are filtered out unless remove_zeros is set to false.\n\n\n\n\n\n","category":"function"},{"location":"data/#DiscreteEntropy.from_data","page":"Data","title":"DiscreteEntropy.from_data","text":"from_data(data::AbstractVector, ::Type{T}; remove_zeros=true) where {T<:EntropyData}\n\nCreate a CountData object from a vector or matrix. The function is parameterised on whether the vector contains samples or the histogram.\n\nWhile remove_zeros defaults to true, this might not be the desired behaviour for Samples. A 0 value in the histgram causes problems for the estimators, but a 0 value in a vector of samples may be perfectly legitimate.\n\n\n\n\n\n","category":"function"},{"location":"data/#DiscreteEntropy.from_samples","page":"Data","title":"DiscreteEntropy.from_samples","text":" from_samples(sample::SampleVector, remove_zeros::Bool)\n\nReturn a CountData object from a vector of samples.\n\n\n\n\n\n","category":"function"},{"location":"data/#Vector-Types","page":"Data","title":"Vector Types","text":"","category":"section"},{"location":"data/","page":"Data","title":"Data","text":"cvector\nsvector\nxivector","category":"page"},{"location":"data/#DiscreteEntropy.cvector","page":"Data","title":"DiscreteEntropy.cvector","text":" cvector(vs::AbstractVector{<:Integer})\n cvector(vs::AbstractVector{<:Real}) = CountVector(vs)\n cvector(vs::AbstractArray{<:Real}) = CountVector(vec(vs))\n\nConvert an AbstractVector into a CountVector. A CountVector represents the frequency of sampled values.\n\n\n\n\n\n","category":"function"},{"location":"data/#DiscreteEntropy.svector","page":"Data","title":"DiscreteEntropy.svector","text":"svector(vs::AbstractVector{<:Integer})\nsvector(vs::AbstractVector{<:Real})\nsvector(vs::AbstractArray{<:Real})\n\nConvert an AbstractVector into a SampleVector. A SampleVector represents a sequence of sampled values.\n\n\n\n\n\n","category":"function"},{"location":"data/#DiscreteEntropy.xivector","page":"Data","title":"DiscreteEntropy.xivector","text":" xivector(vs::AbstractVector{<:Real})\n xivector(vs::AbstractArray{<:Real})\n\nConvert an AbstractVector{Real} into a XiVector. Exclusively for use with schurmann_generalised.\n\n\n\n\n\n","category":"function"},{"location":"estimators/#Estimators","page":"Estimators","title":"Estimators","text":"","category":"section"},{"location":"estimators/","page":"Estimators","title":"Estimators","text":"We split the estimators into two broad categories, which we call Frequentist and Bayesian. We also have a few composite estimators that either take an averaging or resampling approach to estimation.","category":"page"},{"location":"estimators/","page":"Estimators","title":"Estimators","text":"AbstractEstimator\nNonParameterisedEstimator\nParameterisedEstimator","category":"page"},{"location":"estimators/#DiscreteEntropy.AbstractEstimator","page":"Estimators","title":"DiscreteEntropy.AbstractEstimator","text":"AbstractEstimator\n\nSupertype for NonParameterised and Parameterised entropy estimators.\n\n\n\n\n\n","category":"type"},{"location":"estimators/#DiscreteEntropy.NonParameterisedEstimator","page":"Estimators","title":"DiscreteEntropy.NonParameterisedEstimator","text":"NonParameterisedEstimator\n\nType for NonParameterised  entropy estimators.\n\n\n\n\n\n","category":"type"},{"location":"estimators/#DiscreteEntropy.ParameterisedEstimator","page":"Estimators","title":"DiscreteEntropy.ParameterisedEstimator","text":"ParameterisedEstimator\n\nType for Parameterised  entropy estimators.\n\n\n\n\n\n","category":"type"},{"location":"estimators/#Frequentist-Estimators","page":"Estimators","title":"Frequentist Estimators","text":"","category":"section"},{"location":"estimators/","page":"Estimators","title":"Estimators","text":"maximum_likelihood\njackknife_mle\nmiller_madow\ngrassberger\nschurmann\nschurmann_generalised\nchao_shen\nzhang\nbonachela\nshrink\nchao_wang_jost","category":"page"},{"location":"estimators/#DiscreteEntropy.maximum_likelihood","page":"Estimators","title":"DiscreteEntropy.maximum_likelihood","text":"maximum_likelihood(data::CountData)::Float64\n\nCompute the maximum likelihood estimation of Shannon entropy of data in nats.\n\nhatH_tinyML = - sum_i=1^K p_i log(p_i)\n\nor equivalently\n\nhatH_tinyML = log(N) - frac1N sum_i=1^Kh_i log(h_i)\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.jackknife_mle","page":"Estimators","title":"DiscreteEntropy.jackknife_mle","text":"jackknife_mle(data::CountData; corrected=false)::Tuple{AbstractFloat, AbstractFloat}\n\nCompute the jackknifed maximum_likelihood estimate of data and the variance of the jackknifing (not the variance of the estimator itself).\n\nIf corrected is true, then the variance is scaled with n-1, else it is scaled with n\n\nExternal Links\n\nEstimation of the size of a closed population when capture probabilities vary among animals\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.miller_madow","page":"Estimators","title":"DiscreteEntropy.miller_madow","text":"miller_madow(data::CountData)\n\nCompute the Miller Madow estimation of Shannon entropy, with a positive bias based on the total number of samples seen (N) and the support size (K).\n\nhatH_tinyMM = hatH_tinyML + fracK - 12N\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.grassberger","page":"Estimators","title":"DiscreteEntropy.grassberger","text":"grassberger(data::CountData)\n\nCompute the Grassberger (1988) estimation of Shannon entropy of data in nats\n\nhatH_tinyGr88 = sum_i frach_iH left(log(N) - psi(h_i) - frac(-1)^h_in_i + 1  right)\n\nEquation 13 from Finite sample corrections to entropy and dimension estimate\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.schurmann","page":"Estimators","title":"DiscreteEntropy.schurmann","text":"schurmann(data::CountData, ξ::Float64 = ℯ^(-1/2))\n\nCompute the Schurmann estimate of Shannon entropy of data in nats.\n\nhatH_SHU = psi(N) - frac1N sum_i=1^K  h_i left( psi(h_i) + (-1)^h_i _0^frac1xi - 1 fract^h_i-11+tdt right)\n\n\nThis is no one ideal value for xi, however the paper suggests e^(-12) approx 06\n\nExternal Links\n\nschurmann\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.schurmann_generalised","page":"Estimators","title":"DiscreteEntropy.schurmann_generalised","text":"schurmann_generalised(data::CountVector, xis::XiVector{T}) where {T<:Real}\n\nschurmann_generalised\n\nhatH_tinySHU = psi(N) - frac1N sum_i=1^K  h_i left( psi(h_i) + (-1)^h_i _0^frac1xi_i - 1 fract^h_i-11+tdt right)\n\n\nCompute the generalised Schurmann entropy estimation, given a countvector data and a xivector xis, which must both be the same length.\n\nschurmann_generalised(data::CountVector, xis::Distribution, scalar=false)\n\nComputes the generalised Schurmann entropy estimation, given a countvector data and a vector of xi values.\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.chao_shen","page":"Estimators","title":"DiscreteEntropy.chao_shen","text":"chao_shen(data::CountData)\n\nCompute the Chao-Shen estimate of the Shannon entropy of data in nats.\n\nhatH_CS = - sum_i=i^K frachatp_i^CS log hatp_i^CS1 - (1 - hatp_i^CS)\n\nwhere\n\nhatp_i^CS = (1 - frac1 - hatp_i^MLN) hatp_i^ML\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.zhang","page":"Estimators","title":"DiscreteEntropy.zhang","text":"zhang(data::CountData)\n\nCompute the Zhang estimate of the Shannon entropy of data in nats.\n\nThe recommended definition of Zhang's estimator is from Grabchak et al.\n\nhatH_Z = sum_i=1^K hatp_i sum_v=1^N - h_i frac1v _j=0^v-1 left( 1 + frac1 - h_iN - 1 - j right)\n\nThe actual algorithm comes from Fast Calculation of entropy with Zhang's estimator by Lozano et al..\n\nExernal Links\n\nEntropy estimation in turing's perspective\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.bonachela","page":"Estimators","title":"DiscreteEntropy.bonachela","text":"bonachela(data::CountData)\n\nCompute the Bonachela estimator of the Shannon entropy of data in nats.\n\nhatH_B = frac1N+2 sum_i=1^K left( (h_i + 1) sum_j=n_i + 2^N+2 frac1j right)\n\nExternal Links\n\nEntropy estimates of small data sets\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.shrink","page":"Estimators","title":"DiscreteEntropy.shrink","text":"shrink(data::CountData)\n\nCompute the Shrinkage, or James-Stein estimator of Shannon entropy for data in nats.\n\nhatH_tinySHR = - sum_i=1^K hatp_x^tinySHR log(hatp_x^tinySHR)\n\nwhere\n\nhatp_x^tinySHR = lambda t_x + (1 - lambda) hatp_x^tinyML\n\nand\n\nlambda = frac 1 - sum_x=1^K (hatp_x^tinySHR)^2(n-1) sum_x=1^K (t_x - hatp_x^tinyML)^2\n\nwith\n\nt_x = 1  K\n\nNotes\n\nBased on the implementation in the R package entropy\n\nExternal Links\n\nEntropy Inference and the James-Stein Estimator\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.chao_wang_jost","page":"Estimators","title":"DiscreteEntropy.chao_wang_jost","text":"chao_wang_jost(data::CountData)\n\nCompute the Chao Wang Jost Shannon entropy estimate of data in nats.\n\nhatH_tinyCWJ = sum_1 leq h_i leq N-1 frach_iN left(sum_k=h_i^N-1 frac1k right) +\nfracf_1N (1 - A)^-N + 1 left - log(A) - sum_r=1^N-1 frac1r (1 - A)^r right\n\nwith\n\nA = begincases\nfrac2 f_2(N-1) f_1 + 2 f_2   textif  f_2  0 \nfrac2(N-1)(f_1 - 1) + 1   textif  f_2 = 0  f_1 neq 0 \n1  textif  f_1 = f_2 = 0\nendcases\n\nwhere f_1 is the number of singletons and f_2 the number of doubletons in data.\n\nNotes\n\nThe algorithm is slightly modified port of that used in the entropart R library.\n\nExternal Links\n\nEntropy and the species accumulation curve: a novel entropy estimator via discovery rates of new species\n\n\n\n\n\n","category":"function"},{"location":"estimators/#Bayesian-Estimators","page":"Estimators","title":"Bayesian Estimators","text":"","category":"section"},{"location":"estimators/","page":"Estimators","title":"Estimators","text":"bayes\njeffrey\nlaplace\nschurmann_grassberger\nminimax\nnsb\nansb\npym","category":"page"},{"location":"estimators/#DiscreteEntropy.bayes","page":"Estimators","title":"DiscreteEntropy.bayes","text":"bayes(data::CountData, α::AbstractFloat; K=nothing)\n\nCompute an estimate of Shannon entropy given data and a concentration parameter α. If K is not provided, then the observed support size in data is used.\n\nhatH_textBayes = - sum_k=1^K hatp_k^textBayes  log hatp_k^textBayes\n\nwhere\n\np_k^textBayes = fracK + αn + A\n\nand\n\nA = sum_x=1^K α_x\n\nIn addition to setting your own α, we have the following suggested choices\n\njeffrey : α = 0.5\nlaplace: α = 1.0\nschurmann_grassberger: α = 1 / K\nminimax: α = √{n} / K\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.jeffrey","page":"Estimators","title":"DiscreteEntropy.jeffrey","text":" jeffrey(data::CountData; K=nothing)\n\nCompute bayes estimate of entropy, with α = 05\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.laplace","page":"Estimators","title":"DiscreteEntropy.laplace","text":" laplace(data::CountData; K=nothing)\n\nCompute bayes estimate of entropy, with α = 10\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.schurmann_grassberger","page":"Estimators","title":"DiscreteEntropy.schurmann_grassberger","text":" schurmann_grassberger(data::CountData; K=nothing)\n\nCompute bayes estimate of entropy, with α = frac1K. If K is nothing, then use data.K\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.minimax","page":"Estimators","title":"DiscreteEntropy.minimax","text":" minimax(data::CountData; K=nothing)\n\nCompute bayes estimate of entropy, with α = fracdataN where K = data.K if K is nothing.\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.nsb","page":"Estimators","title":"DiscreteEntropy.nsb","text":"nsb(data, K=data.K)\n\nReturns the Bayesian estimate of Shannon entropy of data, using the Nemenman, Shafee, Bialek algorithm\n\nhatH^textNSB = frac int_0^ln(K) dxi  rho(xi textbfn) langle H^m rangle_beta (xi)  \n                             int_0^ln(K) dxi  rho(ximid n)\n\nwhere\n\nrho(xi mid textbfn) =\n    mathcalP(beta (xi)) frac Gamma(kappa(xi))Gamma(N + kappa(xi))\n    prod_i=1^K fracGamma(n_i + beta(xi))Gamma(beta(xi))\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.ansb","page":"Estimators","title":"DiscreteEntropy.ansb","text":"ansb(data::CountData; undersampled::Float64=0.1)::Float64\n\nReturn the Asymptotic NSB estimation of the Shannon entropy of data in nats.\n\nSee Asymptotic NSB estimator (equations 11 and 12)\n\nhatH_tinyANSB = (C_gamma - log(2)) + 2 log(N) - psi(Delta)\n\nwhere C_gamma is Euler's Gamma (approx 057721), psi_0 is the digamma function and Delta the number of coincidences in the data.\n\nThis is designed for the extremely undersampled regime (K ~ N) and diverges with N when well-sampled. ANSB requires that NK  0, which we set to be NK  01 by default\n\nExternal Links\n\nAsymptotic NSB estimator (equations 11 and 12)\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.pym","page":"Estimators","title":"DiscreteEntropy.pym","text":"pym(mm::Vector{Int64}, icts::Vector{Int64})::Float64\n\nA more or less faithful port of the original matlab code to Julia\n\n\n\n\n\n","category":"function"},{"location":"estimators/#Mixed-Estimators","page":"Estimators","title":"Mixed Estimators","text":"","category":"section"},{"location":"estimators/","page":"Estimators","title":"Estimators","text":"pert\njackknife\nbayesian_bootstrap","category":"page"},{"location":"estimators/#DiscreteEntropy.pert","page":"Estimators","title":"DiscreteEntropy.pert","text":"pert(data::CountData, estimator::Type{T}) where {T<:AbstractEstimator}\npert(data::CountData, e1::Type{T}, e2::Type{T}) where {T<:AbstractEstimator}\n\nA Pert estimate of entropy, where\n\na = best estimate\nb = most likely estimate\nc = worst case estimate\n\nH = \\frac{a + 4b + c}{6}\n\nwhere the default estimators are: a = maximum_likelihood, c = ANSB and b is the most likely value = ChaoShen\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.jackknife","page":"Estimators","title":"DiscreteEntropy.jackknife","text":"jackknife(data::CountData, statistic::Function; corrected=false)\n\nCompute the jackknifed estimate of statistic on data.\n\n\n\n\n\n","category":"function"},{"location":"estimators/#DiscreteEntropy.bayesian_bootstrap","page":"Estimators","title":"DiscreteEntropy.bayesian_bootstrap","text":" bayesian_bootstrap(samples::SampleVector, estimator::Type{T}, reps, seed, concentration) where {T<:AbstractEstimator}\n\n\n\n\n\n","category":"function"},{"location":"","page":"Overview","title":"Overview","text":"CurrentModule = DiscreteEntropy\nDocTestSetup = quote\n    using DiscreteEntropy\nend\n","category":"page"},{"location":"#index","page":"Overview","title":"DiscreteEntropy","text":"","category":"section"},{"location":"#Summary","page":"Overview","title":"Summary","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"DiscreteEntropy is a Julia package to estimate the Shannon entropy of discrete data.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"DiscreteEntropy implements a large collection of entropy estimators.","category":"page"},{"location":"#installing-DiscreteEntropy","page":"Overview","title":"Installing DiscreteEntropy","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"If you have not done so already, install Julia. Julia 1.8 and","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"higher are supported. Nightly is not (yet) supported.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"Install DiscreteEntropy using","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"using Pkg; Pkg.add(\"DiscreteEntropy\")","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"or ","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"] add DiscreteEntropy","category":"page"},{"location":"#Basic-Usage","page":"Overview","title":"Basic Usage","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"using DiscreteEntropy\n\ndata = [1,2,3,4,3,2,1]\n\n# treating data as a vector of counts\nh = estimate_h(from_data(data, Histogram), ChaoShen)\n\n# treating data as a vector of samples\nh = estimate_h(from_data(data, Samples), ChaoShen)","category":"page"},{"location":"mutual/#Mutual-Information-and-Conditional-Information","page":"Mutual Information and Conditional Entropy","title":"Mutual Information and Conditional Information","text":"","category":"section"},{"location":"mutual/","page":"Mutual Information and Conditional Entropy","title":"Mutual Information and Conditional Entropy","text":"mutual_information\nconditional_entropy","category":"page"},{"location":"mutual/#DiscreteEntropy.mutual_information","page":"Mutual Information and Conditional Entropy","title":"DiscreteEntropy.mutual_information","text":" mutual_information(X::CountData, Y::CountData, XY::CountData, estimator::Type{T}) where {T<:AbstractEstimator}\n mutual_information(joint::Matrix{I}, estimator::Type{T}) where {T<:AbstractEstimator, I<:Real}\n\nI(XY) = sum_y in Y sum_x in X p(x y) log left(fracp_XY(xy)p_X(x) p_Y(y)right)\n\nBut we use the identity\n\nI(XY) = H(X) + H(Y) - H(XY)\n\nwhere H(XY) is the entropy of the joint distribution\n\n\n\n\n\n","category":"function"},{"location":"mutual/#DiscreteEntropy.conditional_entropy","page":"Mutual Information and Conditional Entropy","title":"DiscreteEntropy.conditional_entropy","text":"conditional_entropy(X::CountData, XY::CountData, estimator::Type{T}) where {T<:NonParameterisedEstimator}\nconditional_entropy(joint::Matrix{R}, estimator::Type{NSB}; dim=1, guess=false, KJ=nothing, KX=nothing) where {R<:Real}\nconditional_entropy(joint::Matrix{R}, estimator::Type{Bayes}, α; dim=1, KJ=nothing, KX=nothing) where {R<:Real}\n\nCompute the conditional entropy of Y conditioned on X\n\nH(Y mid X) = - sum_x in X y in Y p(x y) ln fracp(x y)p(x)\n\nCompute the estimated conditional entropy of Y given X, from counts of X, and (X,Y) and estimator estimator\n\nhatH(Y mid X) = hatH(X Y) - hatH(X)\n\n\n\n\n\n\n\n","category":"function"}]
}
