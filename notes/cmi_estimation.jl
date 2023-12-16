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


function conditional_mutual_information_estimation(X_n::Integer, Y_n::Integer, Z_n::Integer, runs=1000)
    # Generate arbitrary Joint Probability Distribution of three random variables (X, Y, Z)
    # (Random generation of matrices of decimal values)

    # support set (sample space) of X, Y and Z -> number of rows (r) and colums (c) of contingency matrices
    r = X_n
    c = Y_n
    n = Z_n

    # number of runs (characterised by different seeds)
    n_runs = runs

    # shannon measurements' Ground Truth and Estimation (for n_runs)
    n_runs_hXZ = []
    n_runs_hYZ = []
    n_runs_hXYZ = [] 
    n_runs_hZ = []
    n_runs_cmi = []
    n_runs_hXZ_est = []
    n_runs_hYZ_est = []
    n_runs_hXYZ_est = []
    n_runs_hZ_est = []
    n_runs_cmi_est = []

    function add_rule_of_sum(_matrices)
        single_sums = []
        total_sum = []

        # Scale matrices to ensure the sum of values within each matrix is greater than zero and lower than 1
        scale_factor = 0.9
        for i in 1:length(_matrices)
            _matrices[i] *= scale_factor
            # Calculate the sum of each matrix
            push!(single_sums, sum(_matrices[i]))
        end
    
        # Normalize matrices to make the sum among the matrices equal to 1.0 (probability rule of sum)
        total_sum = sum(single_sums)
        for i in 1:length(_matrices)
            _matrices[i] /= total_sum
        end
    
        return _matrices
    end

    # Saving setting
    root = "out/cmi_estimation/"
    matrix_size = string(r) * "x" * string(c) * "x" * string(n)
    current_datetime = now()
    formatted_datetime = Dates.format(current_datetime, "yyyymmdd_HHMMSS")

    working_dir = root * matrix_size * "_" * string(n_runs) * "_" * formatted_datetime * "/"

    # create root directory
    if isdir(root) == false
        mkdir(root)
    end

    # create working directory
    if isdir(working_dir) == false
        mkdir(working_dir)
    end

    println("Support X:" * " " * string(r))
    println("Support Y:" * " " * string(c))
    println("Support Z:" * " " * string(n))
    println("Sample space" * " " * string(r) * "x" * string(c) * "x" * string(n))

    for id_run in 1:n_runs

        Random.seed!(id_run)

        single_sum = 0
        pXZ_temp = []
        pXZ = []
        pYZ_temp = []
        pYZ = []
        pZ = []
        pXYZ_temp = []
        pXYZ = []    

        # shannon measurements' Ground Truth and Estimation (for single run)
        hXZ = []
        hYZ = []
        hXYZ = []
        hZ = []
        cmi = []
        hXZ_est = []
        hYZ_est = []
        hXYZ_est = []
        hZ_est = []
        cmi_est = []

        println("*********** Run" * " " * string(id_run) * " ***********")

        # set of sample size
        ss_lst = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        # --------------------Random Joint Probability Distribution----------------------#
        
        # Generate n random contingency matrices r*c
        matrices = []
        single_sum = []

        for i in 1:n
            matrix = rand(Float64, r, c)
            push!(matrices, matrix)
        end 
        matrices = add_rule_of_sum(matrices)

        # Check probability rule of sum
        for i in 1:length(matrices)
            push!(single_sum, sum(matrices[i]))
        end
        print_data("Sum joint probabilities XYZ:", sum(single_sum))

        # Marginal joint probability distributions
        for i in 1:length(matrices)
            XZ_matrix = marginal_counts(Matrix{Float64}(matrices[i]), 1)
            push!(pXZ_temp, XZ_matrix)
        end
        for i in 1:length(pXZ_temp)
            push!(pXZ, getindex.(pXZ_temp,i))
        end
        
        for i in 1:length(matrices)
            YZ_matrix = marginal_counts(Matrix{Float64}(matrices[i]), 2)
            push!(pYZ_temp, YZ_matrix)
        end
        for i in 1:length(pYZ_temp)
            push!(pYZ, getindex.(pYZ_temp,i))
        end
        
        for i in 1:length(matrices)
            push!(pZ, sum(matrices[i]))
        end

        for i in 1:length(matrices)
            push!(pXYZ_temp, vec(matrices[i]))
        end
        pXYZ = collect(Iterators.flatten(pXYZ_temp))

        # ----------------------------------Ground Truth----------------------------------#
        
        # Ground Truth Entropy
        gthXZ = _entropy(collect(Iterators.flatten(pXZ)))
        push!(hXZ, gthXZ)
        gthYZ = _entropy(collect(Iterators.flatten(pYZ)))
        push!(hYZ, gthYZ)
        gthXYZ = _jointentropy(collect(Iterators.flatten(pXYZ)))
        push!(hXYZ, gthXYZ)
        gthZ = _entropy([i for i in pZ])
        push!(hZ, gthZ)

        # Ground Truth Conditional Mutual Information
        gt = _conditional_mutual_information(collect(Iterators.flatten(pXZ)), collect(Iterators.flatten(pYZ)), collect(Iterators.flatten(pXYZ)), [i for i in pZ])
        push!(cmi, gt)
        print_data("Ground truth I(X;Y|Z):", gt)

        for ss in ss_lst
            
            # ----------------------Sampling from Joint Probability Distribution----------------------#
            global n_s = 0
            pos_samples_XZ = []
            pos_samples_YZ = []
            pos_samples_XYZ = []
            pos_samples_Z = []

            println("Sample size " * string(ss))

            while n_s < ss
                global n_s += 1
                sampleXZ = random_choice(collect(Iterators.flatten(pXZ)))
                push!(pos_samples_XZ, sampleXZ)
        
                sampleYZ = random_choice(collect(Iterators.flatten(pYZ)))
                push!(pos_samples_YZ, sampleYZ)
        
                sampleXYZ = random_choice(collect(Iterators.flatten(pXYZ)))
                push!(pos_samples_XYZ, sampleXYZ)
        
                sampleZ = random_choice(collect(Iterators.flatten(pZ)))
                push!(pos_samples_Z, sampleZ)
            end
            
            # positional samples -> e.g. [1, 2, 3, 3, 4, 4, 4] 
            #       Y=0    Y=1
            # X=0 (1)0.1 (3)0.3
            # X=1 (2)0.2 (4)0.4
            pos_samples_XZ = sort(pos_samples_XZ)
            pos_samples_XZ = [x for x in pos_samples_XZ]
        
            pos_samples_YZ = sort(pos_samples_YZ)
            pos_samples_YZ = [x for x in pos_samples_YZ]
        
            pos_samples_XYZ = sort(pos_samples_XYZ)
            pos_samples_XYZ = [x for x in pos_samples_XYZ]
        
            pos_samples_Z = sort(pos_samples_Z)
            pos_samples_Z = [x for x in pos_samples_Z]
            
            data_XZ = from_samples(svector([s for s in pos_samples_XZ]), true)
            data_YZ = from_samples(svector([s for s in pos_samples_YZ]), true)
            data_XYZ = from_samples(svector([s for s in pos_samples_XYZ]), true)
            data_Z = from_samples(svector([s for s in pos_samples_Z]), true)
            
            # ---------------------------Mutual Information Estimation-------------------------#
            println("-----------I(X;Y|Z)-----------")
            res = cmi_estimations(data_XZ, data_YZ, data_XYZ, data_Z)
            push!(cmi_est, res)
            println("-----------H(X,Z)-----------")
            hXZe = h_estimations(data_XZ)
            push!(hXZ_est, hXZe)
            println("-----------H(Y,Z)-----------")
            hYZe = h_estimations(data_YZ)
            push!(hYZ_est, hYZe)
            println("-----------H(X,Y,Z)-----------")
            hXYZe = h_estimations(data_XYZ)
            push!(hXYZ_est, hXYZe)
            println("-----------H(Z)-----------")
            hZe = h_estimations(data_Z)
            push!(hZ_est, hZe)
        end 

        push!(n_runs_hXZ, hXZ)
        push!(n_runs_hYZ, hYZ)
        push!(n_runs_hXYZ, hXYZ)
        push!(n_runs_hZ, hZ)
        push!(n_runs_cmi, cmi)
        push!(n_runs_hXZ_est, hXZ_est)
        push!(n_runs_hYZ_est, hYZ_est)
        push!(n_runs_hXYZ_est, hXYZ_est)
        push!(n_runs_hZ_est, hZ_est)
        push!(n_runs_cmi_est, cmi_est)
    end

    f = serialize(string(working_dir)*"hXZ.dat", n_runs_hXZ)
    f = serialize(string(working_dir)*"hYZ.dat", n_runs_hYZ)
    f = serialize(string(working_dir)*"hXYZ.dat", n_runs_hXYZ)
    f = serialize(string(working_dir)*"hZ.dat", n_runs_hZ)
    f = serialize(string(working_dir)*"cmi.dat", n_runs_cmi)
    f = serialize(string(working_dir)*"hXZ_est.dat", n_runs_hXZ_est)
    f = serialize(string(working_dir)*"hYZ_est.dat", n_runs_hYZ_est)
    f = serialize(string(working_dir)*"hXYZ_est.dat", n_runs_hXYZ_est)
    f = serialize(string(working_dir)*"cmi_est.dat", n_runs_cmi_est)
end