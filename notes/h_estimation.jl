using Distributions
using EmpiricalDistributions
using StatsBase: fit, Histogram as hist
using FreqTables
using StatsBase
using SimpleRandom
using Serialization
using Metrics
using Dates
using Random


function entropy_estimation(X_n::Integer, runs=1000)
    # Generate arbitrary Probability Distribution (PMF) of one random variable (X)
    # (Random generation of array of decimal values)

    # support set (sample space) of X  -> length of PMF array
    n = X_n

    # number of runs (characterised by different seeds)
    n_runs = runs

    # shannon measurement's Ground Truth and Estimation (for n_runs)
    n_runs_hX = []
    n_runs_hX_est = []

    # Saving setting
    root = "out/h_estimation/"
    array_size = string(n)
    current_datetime = now()
    formatted_datetime = Dates.format(current_datetime, "yyyymmdd_HHMMSS")

    working_dir = root * array_size * "_" * string(n_runs) * "_" * formatted_datetime * "/"

    # create root directory
    if isdir(root) == false
        mkdir(root)
    end

    # create working directory
    if isdir(working_dir) == false
        mkdir(working_dir)
    end

    println("Support (sample space) X:" * " " * string(n))
    
    for id_run in 1:n_runs

        Random.seed!(id_run)

        # shannon measurement's Ground Truth and Estimation (for single run)
        hX = []
        hX_est = []

        println("*********** Run" * " " * string(id_run) * " ***********")

        # set of sample size
        ss_lst = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        # ------------------------Random Probability Distribution-------------------------#
        
        # Generate n random numbers
        pmfX = rand(n)
        # Normalize the array to have sum 1.0 (probability rule of sum)
        pmfX /= sum(pmfX)

        pX = pmfX

        # Check probability rule of sum
        sum_pX = sum(pX)
        print_data("Sum probabilities X:", sum_pX)

        # ----------------------------------Ground Truth----------------------------------#
        
        # Ground Truth Entropy
        gt = _entropy(pX)
        push!(hX, gt)
        print_data("Ground truth H(X):", gt)

        for ss in ss_lst
            
            # --------------------Sampling from Probability Distribution----------------------#
            
            global n_s = 0
            samples = []

            println("Sample size " * string(ss))

            while n_s < ss
                global n_s += 1
                sample = random_choice(pX)
                push!(samples, sample)
            end

            f = freqtable(samples)
            # Probability distribution sample
            pX_s = prop(f)
            
            data_X = from_samples(svector([x for x in samples]), true)
            
            # --------------------------------Entropy Estimation------------------------------#
            
            println("-----------H(X)-----------")
            hXe = h_estimations(data_X)
            push!(hX_est, hXe)
        end 

        push!(n_runs_hX, hX)
        push!(n_runs_hX_est, hX_est)
    end

    f = serialize(string(working_dir)*"hX.dat", n_runs_hX)
    f = serialize(string(working_dir)*"hX_est.dat", n_runs_hX_est)
end

