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
    # -> Random generation of matrix of decimal values

    # support set for X, Y and Z -> number of rows (r) and colums (c) of matrix
    r = X_n
    c = Y_n
    n = Z_n

    # number of runs (characterised by different seeds)
    n_runs = runs

    # ground truth and estimation of Shannon measurements (for n_runs)
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

    function add_rule_of_sum(matrices)
        matrices = []
        single_sums = []
        total_sum = []

        # Scale matrices to ensure the sum of values within each matrix is greater than zero and lower than 1
        scale_factor = 0.9
        for i in 1:length(matrices)
            matrices[i] *= scale_factor
            # Calculate the sum of each matrix
            push!(single_sums, sum(matrices[i]))
        end
    
        # Normalize matrices to make the sum among the three matrices equal to 1.0
        total_sum = sum(single_sums)
        for i in 1:length(matrices)
            matrices[i] /= total_sum
        end
    
        return matrices
    end

    # Saving setting
    root = "out/"
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


    for id_run in 1:n_runs

        Random.seed!(id_run)

        single_sum = 0
        jXZ_temp = []
        jXZ = []
        jYZ_temp = []
        jYZ = []
        m_Z = []
        jXYZ_temp = []
        jXYZ = []    

        # arrays for shannon measurements' Ground Truth and Estimation (for single run)
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

        println("Matrix" * " " * string(r) * "x" * string(c)) * "x" * string(n)
        println("Run" * " " * string(id_run))

        # set of sample size
        ss_lst = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        # --------------------Random Joint Probability Distribution----------------------#
        
        # Generate n random matrices
        for i in 1:n
            matrix = rand(Float64, r, c)
            push!(matrices, matrix)
        end 
        matrices = add_rule_of_sum(matrices)

        # Check sum probabilities = 1
        for i in 1:length(matrices)
            push!(single_sum, sum(matrices[i]))
        end
        print_data("\nSum among the" * string(n) * "matrices:", sum(single_sum))


        # Marginal joint probability distributions
        for i in 1:length(matrices)
            m_matrix = marginal_counts(Matrix{Float64}(matrices[i]), 1)
            push!(jXZ_temp, m_matrix)
        end
        for i in 1:length(jXZ_temp)
            push!(jXZ, getindex.(jXZ_temp,i))
        end
        
        for i in 1:length(matrices)
            m_matrix = marginal_counts(Matrix{Float64}(matrices[i]), 2)
            push!(jYZ_temp, m_matrix)
        end
        for i in 1:length(jYZ_temp)
            push!(jYZ, getindex.(jYZ_temp,i))
        end
        
        for i in 1:length(matrices)
            push!(m_Z, sum(matrices[i]))
        end

        for i in 1:length(matrices)
            push!(jXYZ_temp, vec(matrices[i]))
        end
        jXYZ = collect(Iterators.flatten(jXYZ_temp))

        # ----------------------------------Ground Truth----------------------------------#
        # Ground Truth Entropy
        push!(hXZ, _entropy(collect(Iterators.flatten(jXZ))))
        push!(hYZ, _entropy(collect(Iterators.flatten(jYZ))))
        push!(hXYZ, _jointentropy(collect(Iterators.flatten(jXYZ))))
        push!(hZ, _entropy([i for i in m_Z]))

        # Ground Truth Conditional Mutual Information
        push!(cmi, _conditional_mutual_information(collect(Iterators.flatten(jXZ)), collect(Iterators.flatten(jYZ)), collect(Iterators.flatten(jXYZ)), [i for i in m_Z]))


        for ss in ss_lst
            
            # ----------------------Sampling from Joint Distribution----------------------#
            global n_s = 0
            pos_samples_XZ = []
            pos_samples_YZ = []
            pos_samples_XYZ = []
            pos_samples_Z = []

            println("Sample size")
            println(ss)

            while n_s < ss
                global n_s += 1
                sample = random_choice(collect(Iterators.flatten(jXZ)))
                push!(pos_samples_XZ, sample)
        
                sample = random_choice(collect(Iterators.flatten(jYZ)))
                push!(pos_samples_YZ, sample)
        
                sample = random_choice(collect(Iterators.flatten(jXYZ)))
                push!(pos_samples_XYZ, sample)
        
                sample = random_choice(collect(Iterators.flatten(m_Z)))
                push!(pos_samples_Z, sample)
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
            hXZe = H_estimation(data_XZ)
            println("entropy XZ")
            push!(hXZ_est, hXZe)
            hYZe = H_estimation(data_YZ)
            println("entropy YZ")
            push!(hYZ_est, hYZe)
            hXYZe = H_estimation(data_XYZ)
            println("entropy XYZ")
            push!(hXYZ_est, hXYZe)
            hZe = H_estimation(data_Z)
            println("entropy Z")
            push!(hZ_est, hZe)
            res = cmi_estimations(data_XZ, data_YZ, data_XYZ, data_Z)
            push!(cmi_est, res)

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
end