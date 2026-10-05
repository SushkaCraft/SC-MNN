#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
// #include <random>
#include <fstream>
#include <string>
#include <cstdlib>

namespace MiniNN {

    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double sigmoid_derivative(double output) {
        return output * (1.0 - output);
    }

    // ==========================================
    // НЕЙРОН
    // ==========================================
    class Neuron {
    public:
        std::vector<double> weights;
        double bias;
        double output;
        double delta;

        Neuron(size_t input_size, bool randomize = true) {
            bias = randomize ? ((double)rand() / RAND_MAX - 0.5) : 0.0; // [-0.5, 0.5]
            delta = 0.0;
            output = 0.0;
            for (size_t i = 0; i < input_size; ++i) {
                weights.push_back(randomize ? ((double)rand() / RAND_MAX - 0.5) : 0.0);
            }
        }

        double forward(const std::vector<double>& inputs) {
            double sum = bias;

            if (inputs.size() == weights.size()){
                for (size_t i = 0; i < inputs.size(); ++i) {
                    sum += inputs[i] * weights[i];
                }
                output = sigmoid(sum);
                return output;
            } else {
                printf("Inputs quantities != weights quantities!\n inputs.size() = %zu, weights.size() = %zu\n", inputs.size(), weights.size());
            }
        }

        void update_weights(const std::vector<double>& inputs, double learning_rate) {
            if (inputs.size() >= weights.size()){
                for (size_t i = 0; i < weights.size(); ++i) {
                    weights[i] += learning_rate * delta * inputs[i];
                }
                bias += learning_rate * delta;
            } else {
                printf("Inputs quantities < weights quantities!\n inputs.size() = %zu, weights.size() = %zu\n", inputs.size(), weights.size());
            }
        }
    };

    // ==========================================
    // СЛОЙ
    // ==========================================
    class Layer {
    public:
        std::vector<Neuron> neurons;

        Layer() = default;

        Layer(size_t neuron_count, size_t input_size, bool randomize = true) {
            for (size_t i = 0; i < neuron_count; ++i) {
                neurons.emplace_back(input_size, randomize);
            }
        }

        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> outputs;
            for (auto& neuron : neurons) {
                outputs.push_back(neuron.forward(inputs));
            }
            return outputs;
        }
    };

    // ==========================================
    // СЕТЬ
    // ==========================================
    class Network {
    private:
        std::vector<Layer> layers;

    public:
        const std::vector<Layer>& getLayers() const { return layers; }

        Network() = default;

        void addLayer(const Layer& l) {
            layers.push_back(l);
        }

        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> current_inputs = inputs;
            for (auto& layer : layers) {
                current_inputs = layer.forward(current_inputs);
            }
            return current_inputs;
        }

        void backward(const std::vector<double>& inputs, const std::vector<double>& targets, double learning_rate) {
            Layer& outputLayer = layers.back();
            for (size_t i = 0; i < outputLayer.neurons.size(); ++i) {
                double error = targets[i] - outputLayer.neurons[i].output;
                outputLayer.neurons[i].delta = error * sigmoid_derivative(outputLayer.neurons[i].output);
            }

            for (int l = (int)layers.size() - 2; l >= 0; --l) {
                Layer& currentLayer = layers[l];
                Layer& nextLayer = layers[l + 1];

                for (size_t i = 0; i < currentLayer.neurons.size(); ++i) {
                    double error = 0.0;
                    for (size_t j = 0; j < nextLayer.neurons.size(); ++j) {
                        error += nextLayer.neurons[j].delta * nextLayer.neurons[j].weights[i];
                    }
                    currentLayer.neurons[i].delta = error * sigmoid_derivative(currentLayer.neurons[i].output);
                }
            }

            std::vector<double> layer_input = inputs;
            for (size_t l = 0; l < layers.size(); ++l) {
                for (auto& neuron : layers[l].neurons) {
                    neuron.update_weights(layer_input, learning_rate);
                }
                layer_input = layers[l].forward(layer_input);
            }
        }

        // ==========================================
        // СОХРАНЕНИЕ МОДЕЛИ
        // ==========================================
        void save(const std::string& filename) const {
            std::ofstream out(filename);
            if (!out.is_open()) {
                throw std::runtime_error("Не удалось открыть файл для записи: " + filename);
            }

            out.precision(10);

            out << layers.size() << "\n";
            for (const auto& layer : layers) {
                if (layer.neurons.empty()) continue;
                
                size_t neuron_count = layer.neurons.size();
                size_t weight_count = layer.neurons[0].weights.size();
                out << neuron_count << " " << weight_count << "\n";

                for (const auto& neuron : layer.neurons) {
                    out << neuron.bias;
                    for (double w : neuron.weights) {
                        out << " " << w;
                    }
                    out << "\n";
                }
            }
            out.close();
            std::cout << "[MiniNN] Модель сохранена в " << filename << "\n";
        }

        // ==========================================
        // ЗАГРУЗКА МОДЕЛИ
        // ==========================================
        void load(const std::string& filename) {
            std::ifstream in(filename);
            if (!in.is_open()) {
                throw std::runtime_error("Не удалось открыть файл для чтения: " + filename);
            }

            layers.clear();

            size_t num_layers;
            in >> num_layers;

            for (size_t l = 0; l < num_layers; ++l) {
                size_t neuron_count, weight_count;
                in >> neuron_count >> weight_count;

                Layer layer;
                for (size_t n = 0; n < neuron_count; ++n) {
                    Neuron neuron(weight_count, false);
                    
                    in >> neuron.bias;
                    for (size_t w = 0; w < weight_count; ++w) {
                        in >> neuron.weights[w];
                    }
                    layer.neurons.push_back(neuron);
                }
                layers.push_back(layer);
            }
            in.close();
            std::cout << "[MiniNN] Модель загружена из " << filename << "\n";
        }

        bool empty() const { return layers.empty(); }
    };

}