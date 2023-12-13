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
    # Generate arbitrary Probability Distribution of one random variables (X)
    # -> Random generation of array of decimal values

    # support set for X -> length of array
    n = X_n

    # number of runs (characterised by different seeds)
    n_runs = runs

    # ground truth and estimation of Shannon measurements (for n_runs)
    n_runs_h = []
    n_runs_h_est = []

    # Saving setting
    root = "out/"
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


    for id_run in 1:n_runs

        Random.seed!(id_run)

        # arrays for shannon measurements' Ground Truth and Estimation (for single run)
        h = []
        h_est = []

        println("Input Space" * " " * string(n))
        println("Run" * " " * string(id_run))

        # set of sample size
        ss_lst = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        # ------------------------Random Probability Distribution-------------------------#
        
        # Generate n random numbers
        array = rand(n)
        # Normalize the array to have a sum of 1
        array /= sum(array)

        pX = array

        # Check sum probabilities = 1
        sum_pX = sum(pX)
        print_data(sum_pX)

        # ----------------------------------Ground Truth----------------------------------#
        # Ground Truth Entropy
        push!(h, _entropy(pX))


        for ss in ss_lst
            
            # --------------------Sampling from Probability Distribution----------------------#
            global n_s = 0
            samples = []

            println("Sample size")
            println(ss)

            while n_s < ss
                global n_s += 1
                sample = random_choice(pX)
                push!(samples, sample)
            end

            f = freqtable(samples)
            p = prop(f)
            
            data_X = from_samples(svector([x for x in samples]), true)
            
            # --------------------------------Entropy Estimation------------------------------#
            he = h_estimations(data_X)
            println("entropy X")
            push!(h_est, he)

        end 

        push!(n_runs_h, h)
        push!(n_runs_h_est, h_est)

        f = serialize(string(working_dir)*"hX.dat", n_runs_h)
        f = serialize(string(working_dir)*"hX_est.dat", n_runs_h_est)

    end
end