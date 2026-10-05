#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "NeuralNetwork.h"
#include "TableBuilder.h"

// ---------- Вспомогательная функция: визуальный "прогресс-бар" ----------
static std::string progressBar(double value, int width = 20) {
    if (value < 0.0) value = 0.0;
    if (value > 1.0) value = 1.0;

    int filled = static_cast<int>(value * width + 0.5);
    std::string result;
    for (int i = 0; i < filled; ++i)   result += "█";
    for (int i = filled; i < width; ++i) result += "░";
    return result;
}

int main() {
    system("chcp 65001");
    // Инициализация генератора случайных чисел
    // std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // ============================================================
    // 1. АРХИТЕКТУРА СЕТИ
    // ============================================================
    MiniNN::Network net;

    // net.addLayer(MiniNN::Layer(2, 2)); // скрытый слой: 2 нейрона, у каждого 2 входа
    // net.addLayer(MiniNN::Layer(1, 2)); // выходной слой: 1 нейрон, 2 входа

    // Скрытый слой: 2 нейрона, у каждого 2 входа.
    // randomize = false — отключает случайную инициализацию.
    MiniNN::Layer hidden(2, 2, false);
    
    // Нейрон 0 скрытого слоя: логическое И (AND)
    hidden.neurons[0].weights = { 20.0, 20.0 };
    hidden.neurons[0].bias    = -10.0;
    
    // Нейрон 1 скрытого слоя: логическое ИЛИ (OR)
    hidden.neurons[1].weights = { -20.0, -20.0 };
    hidden.neurons[1].bias    = 30.0;
    
    // Выходной слой: 1 нейрон, 2 входа.
    MiniNN::Layer output(1, 2, false);
    
    // Выход: h2 - h1 (это даёт XOR)
    output.neurons[0].weights = { 20.0, 20.0 };
    output.neurons[0].bias    = -30.0;
    
    net.addLayer(hidden);
    net.addLayer(output);

    std::cout << "=================================================\n";
    std::cout << "  Нейросеть для операции XOR (архитектура 2-2-1) \n";
    std::cout << "=================================================\n\n";

    std::cout << "--- Архитектура сети ---\n";
    MiniNN::TableBuilder arch;
    arch.addColumn("№ слоя",           7,  true)
        .addColumn("Тип",             12,  false)
        .addColumn("Нейронов",        10,  true)
        .addColumn("Входов на нейрон", 18, true);

    const std::vector<MiniNN::Layer>& layers = net.getLayers();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& neurons = layers[i].neurons;
        if (neurons.empty()) continue;

        std::string type = (i + 1 == layers.size()) ? "Выходной" : "Скрытый";
        arch.addRow({
            MiniNN::fmtInt(static_cast<long long>(i + 1)),
            type,
            MiniNN::fmtInt(static_cast<long long>(neurons.size())),
            MiniNN::fmtInt(static_cast<long long>(neurons[0].weights.size()))
        });
    }
    arch.print();
    std::cout << "\n";

    // ============================================================
    // 2. ДАННЫЕ ДЛЯ ОБУЧЕНИЯ
    // ============================================================
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    std::cout << "--- Обучающая выборка XOR ---\n";
    MiniNN::TableBuilder dataTable;
    dataTable.addColumn("A",            3,  true)
             .addColumn("B",            3,  true)
             .addColumn("A XOR B (цель)", 15, true);

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        dataTable.addRow({
            MiniNN::fmtInt(static_cast<long long>(inputs[i][0])),
            MiniNN::fmtInt(static_cast<long long>(inputs[i][1])),
            MiniNN::fmtInt(static_cast<long long>(targets[i][0]))
        });
    }
    dataTable.print();
    std::cout << "\n";

    // ============================================================
    // 3. ОБУЧЕНИЕ
    // ============================================================
    const double learning_rate = 0.8;
    const int    epochs        = 25000;

    std::cout << "--- Прогресс обучения ---\n";

    MiniNN::TableBuilder progress;
    progress.addColumn("Эпоха",            8,  true)
            .addColumn("Суммарная ошибка", 18, true)
            .addColumn("Прогресс",         22, false);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> output = net.forward(inputs[i]);

            double error = targets[i][0] - output[0];
            total_error += error * error;

            net.backward(inputs[i], targets[i], learning_rate);
        }

        if (epoch % 5000 == 0) {
            // Нормируем ошибку для прогресс-бара: чем меньше ошибка, тем полнее бар
            double quality = 1.0 - (total_error / 1.0);
            progress.addRow({
                MiniNN::fmtInt(static_cast<long long>(epoch)),
                MiniNN::fmt(total_error, 6),
                progressBar(quality)
            });
        }
    }

    progress.print();
    std::cout << "\n";

    // ============================================================
    // 4. ИТОГОВЫЕ РЕЗУЛЬТАТЫ
    // ============================================================
    std::cout << "--- Итоговые результаты ---\n";

    MiniNN::TableBuilder results;
    results.addColumn("A",           3,  true)
           .addColumn("B",           3,  true)
           .addColumn("Выход",       10, true)
           .addColumn("Округлённо",  12, true)
           .addColumn("Ожидалось",   10, true)
           .addColumn("Верно?",      8,  false);

    int correct = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> output = net.forward(inputs[i]);
        double prediction = (output[0] > 0.5) ? 1.0 : 0.0;

        bool ok = (prediction == targets[i][0]);
        if (ok) ++correct;

        results.addRow({
            MiniNN::fmtInt(static_cast<long long>(inputs[i][0])),
            MiniNN::fmtInt(static_cast<long long>(inputs[i][1])),
            MiniNN::fmt(output[0], 6),
            MiniNN::fmtInt(static_cast<long long>(prediction)),
            MiniNN::fmtInt(static_cast<long long>(targets[i][0])),
            ok ? "да" : "нет"
        });
    }

    results.print();

    std::cout << "\nТочность: " << correct << " / " << inputs.size()
              << " (" << (100.0 * correct / inputs.size()) << "%)\n\n";

    // ============================================================
    // 5. СОХРАНЕНИЕ МОДЕЛИ
    // ============================================================
    try {
        net.save("xor_model.txt");
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка сохранения: " << ex.what() << "\n";
    }

    return 0;
}