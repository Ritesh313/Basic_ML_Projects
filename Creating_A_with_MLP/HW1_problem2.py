from matplotlib import pyplot as plt
import numpy as np
from random import random
import math
import random as rand
import cv2


y = []


def plotting(matrix, network=None, title="Prediction Matrix"):

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("i1")
    ax.set_ylabel("i2")

    if network != None:
        map_min = -2.0
        map_max = 2.0
        y_res = 0.01
        x_res = 0.01
        ys = np.arange(map_min, map_max, y_res)
        xs = np.arange(map_min, map_max, x_res)
        zs = []
        for cur_y in np.arange(map_min, map_max, y_res):
            for cur_x in np.arange(map_min, map_max, x_res):
                zs.append(predict(network, [cur_x, cur_y, 1]))
        xs, ys = np.meshgrid(xs, ys)
        zs = np.array(zs)
        zs = zs.reshape(xs.shape)
        cp = plt.contourf(
            xs, ys, zs, levels=[-2, -1, 0, 1, 2], colors=('b', 'r'), alpha=0.1)

    c1_data = [[], []]
    c0_data = [[], []]
    for i in range(len(matrix)):
        cur_i1 = matrix[i][0]
        cur_i2 = matrix[i][1]
        cur_y = matrix[i][-1]
        if cur_y == 1:
            c1_data[0].append(cur_i1)
            c1_data[1].append(cur_i2)
        else:
            c0_data[0].append(cur_i1)
            c0_data[1].append(cur_i2)
    plt.xticks(np.arange(-2.0, 2.0, 1))
    plt.yticks(np.arange(-2.0, 2.0, 1))
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)

    c0s = plt.scatter(c0_data[0], c0_data[1], s=40.0, c='r', label='Class -1')
    c1s = plt.scatter(c1_data[0], c1_data[1], s=40.0, c='b', label='Class 1')

    plt.legend(fontsize=10, loc=1)
    plt.show()
    return


def create_network(n_inputs, layers_hidden, hidden_neurons: list, n_output):
    # The list input for hidden neurons is in case there is more than one hidden layers.
    network = list()
    for i in range(layers_hidden):
        layer = []
        for j in range(hidden_neurons[i]):
            layer.append({"weights": [random() for j in range(n_inputs+1)]})
        network.append(layer)

    layer = []

    for j in range(n_output):
        layer.append({"weights": [random()
                                  for j in range(hidden_neurons[-1]+1)]})
    network.append(layer)
    return network


def training(network, matrix, epochs, learning_rate, n_output, visualize_plot=True, error_plot=True):
    # The network should be a list of weights of different layers with appropriate number of layers and neurons in the layers.
    # Example a network with 2 input nodes (with 1 bias), 1 hidden layer with 2 nodes (with 1 bias) and one 1 node will look like:
    # [[{'weights': [0.16696199275834855, 0.9074274294852825, 0.7977612543253025]}, {'weights': [0.8403945031023915, 0.3699665961711708, 0.41002603124474046]}], [{'weights': [0.06221525252990556, 0.8261995420912167, 0.017695313965993886]}]]
    if(visualize_plot):
        f1 = plt.figure(1)
        ax1 = f1.add_subplot(111)
    if(error_plot):
        f2 = plt.figure(2)
        ax2 = f2.add_subplot(111)
    error_plot_values = []
    for epoch in range(epochs):
        sum_error = 0
        for data_row in matrix:
            final_output = forward_propagate(network, data_row)
            true_value = []
            for i in range(n_output):
                true_value.append(0)
            true_value[data_row[-1]] = 1
            for i in range(len(true_value)):
                sum_error += sum([(true_value[i]-final_output[i])**2])
            backprop(network, true_value)
            update_paramters(network, data_row, learning_rate)
        error_plot_values.append(sum_error)
        if(visualize_plot):
            x_axis = np.linspace(-3, 3)
            layer = network[0]
            for idx, node in enumerate(layer):
                weights = node['weights']
                y_value = -((weights[0]/weights[1]) *
                            x_axis + weights[-1]/weights[1])
                if idx == 0:
                    ax1.plot(x_axis, y_value, '-g')
                if idx == 1:
                    ax1.plot(x_axis, y_value, '-b')
                if idx == 2:
                    ax1.plot(x_axis, y_value, '-y')
                if idx == 3:
                    ax1.plot(x_axis, y_value, '-r')
            ax1.set_title('Weights Plot')
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")

            # ax1.legend()
            # f1.show()
        print('>epoch=%d, lrate=%.3f, error=%.3f' %
              (epoch, learning_rate, sum_error))

    if(error_plot):

        epoch_plot_values = [i for i in range(1, epochs+1)]
        ax2.plot(epoch_plot_values, error_plot_values, '-b', label='Error')
        # plt.title('Training Error')
        # plt.xlabel('Epochs')
        # plt.ylabel('Error')
        ax2.set_title('Training Error')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Error")
        plt.legend()
        # f2.show()
    if(error_plot or visualize_plot):
        plt.show()
    # plotting(matrix,network)


def forward_propagate(network, data):
    # This function handles the complete forward propagation of network for one set of data points.

    data = data[:-1]

    for layer in network:
        layer_output = []
        for node in layer:
            node_output = single_forward_pass(node['weights'], data)
            node['Output'] = node_output
            layer_output.append(node_output)
        data = layer_output
    # print("Final output", data)
    return data


def single_forward_pass(node, data):
    # This function is to calculate final output of a single neuron
    weighted_sum = node[-1]
    for w, x in zip(node[:-1], data):
        # print("w = {} and x = {}" .format(w,x))
        weighted_sum += w*x
    layer_output = sigmoid_activation_function(weighted_sum)
    return layer_output


def backprop(network, true_value):
    depth = len(network)
    for i in reversed(range(depth)):
        layer = network[i]
        layer_len = len(layer)
        update_term = []
        if i < depth-1:
            # Delta for the hidden layers
            for j in range(layer_len):
                error = 0
                for next_layer_node in network[i+1]:
                    error += next_layer_node['update_term'] * \
                        next_layer_node['weights'][j]
                update_term.append(
                    error * derivative_for_sigmoid(layer[j]['Output']))
        else:
            # Delta for the last layer
            for j in range(layer_len):
                error = true_value[j] - layer[j]['Output']
                update_term.append(
                    error * derivative_for_sigmoid(layer[j]['Output']))
        for k in range(layer_len):
            layer[k]['update_term'] = update_term[k]


def update_paramters(network, input_data, learning_rate):
    for idx, layer in enumerate(network):
        input_to_the_layer = input_data
        if idx != 0:
            input_to_the_layer = [node["Output"] for node in network[idx-1]]
        for node in layer:
            for j in range(len(input_to_the_layer)):
                node['weights'][j] += learning_rate * \
                    node['update_term'] * input_to_the_layer[j]
            # For the bias term
            node['weights'][-1] += learning_rate * node['update_term']
    return network


def derivative_for_sigmoid(layer_output):
    return layer_output * (1.0 - layer_output)


def signum_activation_function(weighted_sum):
    if weighted_sum > 0:
        return 1
    elif weighted_sum < 0:
        return -1
    return 0


def sigmoid_activation_function(weighted_sum):
    return 1.0 / (1.0 + math.exp(-weighted_sum))


def predict(network, input_data):
    output_val = forward_propagate(network, input_data)
    return output_val.index(max(output_val))


problem1_network = [[[-3.0, 1.0, -3.0], [-3.0, 1.0, -2.5], [3.0, 1.0, -3.0], [3.0, 1.0, -2.5],
                     [0.0, 1.0, 1.0], [0.0, 1.0, 1.25]], [[-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, -2.0],
                                                          [-1.0, 0.0, -1.0,
                                                              1.0, -1.0, 1.0, -2.0],
                                                          [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 0.0]]]


best_initial_network = [[{'weights': [0.9841698946226802, 0.7443485956429766, 0.39033739874241846]}, {'weights': [0.7673593727535573, 0.3685434829913632, 0.9572486431551711]}, {'weights': [0.055382569601113874, 0.964813054007715, 0.44590963589993826]}, {'weights': [0.8279780984770803,0.19282227643805672, 0.5041359264246169]}], [{'weights': [0.18592780528717245, 0.6300980308882045, 0.8275516186965673, 0.581243397370875, 0.411158347707138]}, {'weights': [0.024129688526639992, 0.9063010720675685, 0.5613600951334111, 0.022899930594609375, 0.44146418007664934]}]]


trained_1000_epoch_network = [[{'weights': [-0.7621764006364317, 13.413159802195512, -4.205275583522489]}, {'weights': [21.399783490503765, -0.1459231768571693, 3.9502944765046233]}, {'weights': [0.26071688141565813, 28.353012964178795, 2.5259659073280694]}, {'weights': [22.76489662560973, 0.20763140581019643, -1.7934700058672997]}], [{'weights': [8.217085471783731, -8.90876832545036, -9.992744739662633, 9.971211949020903, 5.040115989257821]}, {'weights': [-8.218256402038454, 8.909878380203223, 9.99421872733163, -9.9727231560631, -5.040812974236946]}]]
trained_10000_epoch_network = [[{'weights': [-0.9592563011978409, 14.786706148824658, -4.70046446583266]}, {'weights': [24.239787129483556, -0.07607190170847078, 4.03496481324339]}, {'weights': [0.32474503242357666, 32.11287764808236, 3.0821490712316764]}, {'weights': [31.805523981691188, 0.18425852730265568, -1.889651919738324]}], [{'weights': [11.528377483909567, -13.033110700624434, -13.29532893994785, 14.18828147971293, 6.776882877598519]}, {'weights': [-11.528519071159739, 13.033275271867767, 13.2955026246674, -14.18847264132986, -6.776973161427623]}]]

#The netowrk below is trained on the right data(the square of size 2 instead of 4.)
trained_30000_epoch_network = [[{'weights': [1.3891935982406474, 75.18593286318277, -7.4924786545146285]}, {'weights': [58.73439693450513, 1.2984993826531128, 1.3402682791591916]}, {'weights': [2.0504858458313255, 42.91101701224852, 3.181033972810449]}, {'weights': [
    77.75051558095807, -1.7815769164149757, -3.99637373568659]}], [{'weights': [30.179943858189624, -30.113777504492592, -14.495115552006803, 35.059920105222105, 7.520987908400877]}, {'weights': [-30.17997592451851, 30.113808047554617, 14.49514016633849, -35.05995868401386, -7.5210042287980166]}]]

network_4_neurons_2_layers = [[{'weights': [0.9841698946226802, 0.7443485956429766, 0.39033739874241846]}, {'weights': [0.7673593727535573, 0.3685434829913632, 0.9572486431551711]}, {'weights': [0.055382569601113874, 0.964813054007715, 0.44590963589993826]}, {'weights': [0.8279780984770803,0.19282227643805672, 0.5041359264246169]}], [{'weights': [0.18592780528717245, 0.6300980308882045, 0.8275516186965673, 0.581243397370875, 0.411158347707138]}, {'weights': [0.024129688526639992, 0.9063010720675685, 0.5613600951334111, 0.022899930594609375, 0.44146418007664934]}], [{'weights': [0.18592780528717245, 0.6300980308882045, 0.8275516186965673, 0.581243397370875, 0.411158347707138]}, {'weights': [0.024129688526639992, 0.9063010720675685, 0.5613600951334111, 0.022899930594609375, 0.44146418007664934]}]]

network_5_neurons = [[{'weights': [0.9841698946226802, 0.7443485956429766, 0.39033739874241846]}, {'weights': [0.7673593727535573, 0.3685434829913632, 0.9572486431551711]}, {'weights': [0.055382569601113874, 0.964813054007715, 0.44590963589993826]}, {'weights': [0.8279780984770803,0.19282227643805672, 0.5041359264246169]}, {'weights': [-0.9592563011978409, 14.786706148824658, -4.70046446583266]}], [{'weights': [0.18592780528717245, 0.6300980308882045, 0.8275516186965673, 0.581243397370875, 0.411158347707138]}, {'weights': [0.024129688526639992, 0.9063010720675685, 0.5613600951334111, 0.022899930594609375, 0.44146418007664934]}]]

network_6_neurons = [[{'weights': [0.9841698946226802, 0.7443485956429766, 0.39033739874241846]}, {'weights': [0.7673593727535573, 0.3685434829913632, 0.9572486431551711]}, {'weights': [0.055382569601113874, 0.964813054007715, 0.44590963589993826]}, {'weights': [0.8279780984770803,0.19282227643805672, 0.5041359264246169]}, {'weights': [-0.9592563011978409, 14.786706148824658, -4.70046446583266]},{'weights': [-0.9592563011978409, 0.786706148824658, -4.70046446583266]}], [{'weights': [0.18592780528717245, 0.6300980308882045, 0.8275516186965673, 0.581243397370875, 0.411158347707138]}, {'weights': [0.024129688526639992, 0.9063010720675685, 0.5613600951334111, 0.022899930594609375, 0.44146418007664934]}]]

trained_10000_epoch_network = [[{'weights': [-0.9592563011978409, 14.786706148824658, -4.70046446583266]}, {'weights': [24.239787129483556, -0.07607190170847078, 4.03496481324339]}, {'weights': [0.32474503242357666, 32.11287764808236, 3.0821490712316764]}, {'weights': [31.805523981691188, 0.18425852730265568, -1.889651919738324]}], [{'weights': [11.528377483909567, -13.033110700624434, -13.29532893994785, 14.18828147971293, 6.776882877598519]}, {'weights': [-11.528519071159739, 13.033275271867767, 13.2955026246674, -14.18847264132986, -6.776973161427623]}]]
x_value = 5.0
x_neg_value = -5.0
y_value = 5.0
y_neg_value = -5.0

[[[-3.0, 1.0, -3.0], [-3.0, 1.0, -2.5], [3.0, 1.0, -3.0], [3.0, 1.0, -2.5],
  [0.0, 1.0, 1.0], [0.0, 1.0, 1.25], [1.0, 0.0, 1.39], [
      1.0, 0.0, x_value], [1.0, 0.0, -1.39],
  [1.0, 0.0, x_neg_value], [0.0, 1.0, y_neg_value], [0.0, 1.0, -2.75]],
    [[-2.0, 2.0, -2.0, 2.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 2.0, -2.0, -1.0]]]
