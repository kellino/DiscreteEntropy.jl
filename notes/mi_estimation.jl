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


function mutual_information_estimation(X_n::Integer, Y_n::Integer, runs=1000)
    # Generate arbitrary Joint Probability Distribution of two random variables (X, Y)
    # -> Random generation of matrix of decimal values

    # support set for X and Y -> number of rows (r) and colums (c) of matrix
    r = X_n
    c = Y_n

    # number of runs (characterised by different seeds)
    n_runs = runs

    # ground truth and estimation of Shannon measurements (for n_runs)
    n_runs_hX = []
    n_runs_hY = []
    n_runs_hXY = []
    n_runs_mi = []
    n_runs_hX_est = []
    n_runs_hY_est = []
    n_runs_hXY_est = []
    n_runs_mi_est = []

    # Saving setting
    root = "out/"
    matrix_size = string(r) * "x" * string(c)
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

        # arrays for shannon measurements' Ground Truth and Estimation (for single run)
        hX = []
        hY = []
        hXY = []
        mi = []
        hX_est = []
        hY_est = []
        hXY_est = []
        mi_est = []

        println("Matrix" * " " * string(r) * "x" * string(c))
        println("Run" * " " * string(id_run))

        # set of sample size
        ss_lst = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        # --------------------Random Joint Probability Distribution----------------------#
        matrix_joint = Matrix{Float64}(undef, r, c)
        v = rand(r * c)
        matrix_joint = reshape((v / sum(v)), (r, c))
        j_XY = matrix_joint

        # Check sum probabilities = 1
        sum_j_XY = sum(j_XY)
        print_data(sum_j_XY)

        # Marginal probability distributions
        m_X = marginal_counts(Matrix{Float64}(j_XY), 1)
        m_Y = marginal_counts(Matrix{Float64}(j_XY), 2)

        # ----------------------------------Ground Truth----------------------------------#
        # Ground Truth Entropy
        push!(hX, _entropy(m_X))
        push!(hY, _entropy(m_Y))
        push!(hXY, _jointentropy(vec(j_XY)))

        # Ground Truth Mutual Information
        push!(mi, _mutual_information(vec(m_X), vec(m_Y), vec(j_XY)))

        # Cartesian product
        x = []
        y = []
        for i in 1:r
            for j in 1:c
                push!(x,i)
                push!(y,j)
            end
        end

        for ss in ss_lst
            
            # ----------------------Sampling from Joint Distribution----------------------#
            global n_s = 0
            pos_samples = []

            println("Sample size")
            println(ss)

            while n_s < ss
                global n_s += 1
                sample = random_choice(vec(j_XY))
                push!(pos_samples, sample)
            end
            
            # positional samples -> e.g. [1, 2, 3, 3, 4, 4, 4] 
            #       Y=0    Y=1
            # X=0 (1)0.1 (3)0.3
            # X=1 (2)0.2 (4)0.4
            pos_samples = sort(pos_samples)
            pos_samples = [x for x in pos_samples]
            
            f = freqtable(pos_samples)
            p = prop(f)
            
            # Association of positional samples to actual n-tuple samples (n=2 -> joint samples (X,Y))
            d = Dict()
            global count = 0
            for i in 1:r
                for j in 1:c
                    global count += 1
                    merge!(d, Dict(count=>(j,i)))
                end
            end
            
            XY_samples = []
            X_samples = []
            Y_samples = []
            for s in pos_samples
                # samples (X,Y)
                push!(XY_samples, d[s])
                # samples X, Y (X and Y coordinates)
                push!(X_samples, d[s][1])
                push!(Y_samples, d[s][2])
            end

            matrix_sample = freqtable(X_samples, Y_samples)
            j_XY = prop(matrix_sample)
            m_X = marginal_counts(Matrix{Float64}(j_XY), 1)
            m_Y = marginal_counts(Matrix{Float64}(j_XY), 2)
            
            data_X = from_samples(svector([x for x in X_samples]), true)
            data_Y = from_samples(svector([y for y in Y_samples]), true)
            data_XY = from_samples(svector([s for s in pos_samples]), true)
            
            # ---------------------------Mutual Information Estimation-------------------------#
            hXe = h_estimations(data_X)
            println("entropy X")
            push!(hX_est, hXe)
            hYe = h_estimations(data_Y)
            println("entropy Y")
            push!(hY_est, hYe)
            hXYe = h_estimations(data_XY)
            println("joint entropy XY")
            push!(hXY_est, hXYe)
            #res = mi_estimations(Matrix(matrix_sample))
            res = mi_estimations(data_X, data_Y, data_XY)
            push!(mi_est, res)

        end 

        push!(n_runs_hX, hX)
        push!(n_runs_hY, hY)
        push!(n_runs_hXY, hXY)
        push!(n_runs_mi, mi)
        push!(n_runs_hX_est, hX_est)
        push!(n_runs_hY_est, hY_est)
        push!(n_runs_hXY_est, hXY_est)
        push!(n_runs_mi_est, mi_est)

        f = serialize(string(working_dir)*"hX.dat", n_runs_hX)
        f = serialize(string(working_dir)*"hY.dat", n_runs_hY)
        f = serialize(string(working_dir)*"hXY.dat", n_runs_hXY)
        f = serialize(string(working_dir)*"mi.dat", n_runs_mi)
        f = serialize(string(working_dir)*"hX_est.dat", n_runs_hX_est)
        f = serialize(string(working_dir)*"hY_est.dat", n_runs_hY_est)
        f = serialize(string(working_dir)*"hXY_est.dat", n_runs_hXY_est)
        f = serialize(string(working_dir)*"mi_est.dat", n_runs_mi_est)

    end
end