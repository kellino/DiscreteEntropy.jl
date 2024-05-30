function unseen(data::CountData)
    finger = sortslices(data.multiplicities, dims=2)[2, :]
    grid_factor = 1.05
    alpha = 0.5

    x_lp_min = 1 / (data.K * max(10, data.K))

    println(finger)
    min_i = minimum(filter(x -> x > 0, finger))


    if min_i > 1
        x_lp_min = min_i / data.K
    end

    max_lp_iters = 1000

    x = 0
    histx = 0

    f_lp = zeros(length(finger))

    for i in 1:length(finger)
        if finger[i] > 0
            wind = [ max(1, i - ceil(sqrt(i))), min(i + ceil(sqrt(i)) , length(finger))  ]
            if finger[convert(Integer, wind[1])] + finger[convert(Integer, wind[2])] < sqrt(i)
                x = [x, i / data.K]
                histx = [histx, finger[i]]
                f_lp[i] = 0
            else
                f_lp[i] = finger[convert(Integer, i)]
            end
        end
    end

    f_max = maximum(f_lp)
    println(f_max)

end
# Input matrix
# matrix = [4 2 6; 1 5 3; 9 7 8]

# # Column to sort by
# column = 2

# # Sort the matrix based on the specified column
# sorted_matrix = sortrows(matrix, by = x -> x[column])
