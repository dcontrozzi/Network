
import numpy as np

def similation_average(model, sample_size_list, nb_simulations):
    """

    :param model: simulation model, requires a function similate()
    :param sample_size_list: list of sample size to test
    :param nb_simulations: number of simulation for each sample size
    :return:
    """

    mean_dict_mean = {}
    mean_dict_std = {}

    var_dict_mean = {}
    var_dict_std = {}

    diff_mean_dict_mean = {}
    diff_mean_dict_std = {}

    diff_var_dict_mean = {}
    diff_var_dict_std = {}

    # Test different values of N
    for j in sample_size_list:
        var = []
        mean = []
        diff_var = []
        diff_mean = []
        # Run nb_simulations and average
        for i in range(nb_simulations):
            # Use N = T
            S = model.simulate(j, j)
            var.append(np.var(S))
            mean.append(np.mean(S))

            S_diff = np.diff(S)
            diff_var.append(np.var(S_diff))
            diff_mean.append(np.mean(S_diff))

        var_dict_mean[j] = np.mean(var)
        var_dict_std[j] = np.std(var)

        mean_dict_mean[j] = np.mean(mean)
        mean_dict_std[j] = np.std(mean)

        diff_var_dict_mean[j] = np.mean(diff_var)
        diff_var_dict_std[j] = np.std(diff_var)

        diff_mean_dict_mean[j] = np.mean(diff_mean)
        diff_mean_dict_std[j] = np.std(diff_mean)

        print('levels')
        print('number of points ', j, ' exact mean ', model.mean(j), ' calculated mean ', mean_dict_mean[j], ' mean std ',  mean_dict_std[j] )
        print('number of points ', j, 'exact val ', model.var(j), ' calculated std ', var_dict_mean[j], ' var std ', var_dict_std[j])

        print('diff')
        print('number of points ', j, ' exact mean ', model.diff_mean(j), ' calculated mean ', diff_mean_dict_mean[j], ' mean std ',  diff_mean_dict_std[j] )
        print('number of points ', j, 'exact val ', model.diff_var(j), ' calculated std ', diff_var_dict_mean[j], ' var std ', diff_var_dict_std[j])

        pass