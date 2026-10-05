#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "NeuralNetwork.h"
#include "TableBuilder.h"

// ==========================================
// Настройки
// ==========================================
const int    WINDOW_SIZE      = 7;    // длина окна входа и коррекции
const int    HIDDEN1_NEURONS  = 7;    // первый скрытый слой
const int    HIDDEN2_NEURONS  = 5;    // корректирующий слой
const int    STATS_WINDOW     = 10;   // окно для оценки частоты смены выбора
const int    QUALITY_WINDOW   = 10;   // окно для оценки точности предсказаний
const double LEARNING_RATE    = 0.8;
const int    MAX_VISIBLE_ROWS = 25;

// ==========================================
// Очистка экрана
// ==========================================
void clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

// ==========================================
// Метрики поведения и качества
// ==========================================
// Доля переходов 0<->1 среди последних n вводов (0..1)
double computeSwitchRate(const std::vector<int>& h, int n) {
    int end   = (int)h.size();
    int start = std::max(0, end - n);
    if (end - start < 2) return 0.0;
    int changes = 0;
    for (int i = start + 1; i < end; ++i)
        if (h[i] != h[i - 1]) ++changes;
    return (double)changes / (end - start - 1);
}

// Точность сети за последние n предсказаний (0..1).
// p[i] предсказывает h[WINDOW_SIZE + i]
double computeQuality(const std::vector<int>& h,
                      const std::vector<int>& p, int n) {
    int total = 0, ok = 0;
    int end   = std::min((int)p.size(), (int)h.size() - WINDOW_SIZE);
    int start = std::max(0, end - n);
    for (int i = start; i < end; ++i) {
        int actual_idx = WINDOW_SIZE + i;
        if (actual_idx >= (int)h.size()) break;
        ++total;
        if (p[i] == h[actual_idx]) ++ok;
    }
    return total ? (double)ok / total : 0.5;   // нет данных — нейтрально
}

// ==========================================
// Сеть с корректирующим слоем
// ==========================================
class SelfCorrectingNetwork {
public:
    MiniNN::Layer L1;
    MiniNN::Layer L2;
    MiniNN::Layer L3;

    std::vector<double> l1_out;
    std::vector<double> corr_input;   // L1_out + last_inputs + last_preds + [switch, quality]
    std::vector<double> l2_out;

    SelfCorrectingNetwork(size_t window, size_t h1, size_t h2)
        : L1(h1, window),
          L2(h2, h1 + 2 * window + 2),  // +2 мета-признака
          L3(1, h2) {}

    std::vector<double> forward(const std::vector<double>& window,
                                 const std::vector<double>& last_inputs,
                                 const std::vector<double>& last_preds,
                                 double switch_rate,
                                 double quality) {
        l1_out = L1.forward(window);

        corr_input.clear();
        corr_input.insert(corr_input.end(), l1_out.begin(),      l1_out.end());
        corr_input.insert(corr_input.end(), last_inputs.begin(), last_inputs.end());
        corr_input.insert(corr_input.end(), last_preds.begin(),  last_preds.end());
        corr_input.push_back(switch_rate);   // мета-признак 1
        corr_input.push_back(quality);       // мета-признак 2

        l2_out = L2.forward(corr_input);
        return L3.forward(l2_out);
    }

    void backward(const std::vector<double>& window, double target, double lr) {
        // Выходной слой
        auto& outN = L3.neurons[0];
        outN.delta = (target - outN.output) * MiniNN::sigmoid_derivative(outN.output);

        // Корректирующий слой
        for (size_t i = 0; i < L2.neurons.size(); ++i) {
            double e = 0.0;
            for (size_t j = 0; j < L3.neurons.size(); ++j)
                e += L3.neurons[j].delta * L3.neurons[j].weights[i];
            L2.neurons[i].delta = e * MiniNN::sigmoid_derivative(L2.neurons[i].output);
        }

        // Первый скрытый слой — только первые HIDDEN1_NEURONS входов L2 идут из L1
        for (size_t i = 0; i < L1.neurons.size(); ++i) {
            double e = 0.0;
            for (size_t j = 0; j < L2.neurons.size(); ++j)
                e += L2.neurons[j].delta * L2.neurons[j].weights[i];
            L1.neurons[i].delta = e * MiniNN::sigmoid_derivative(L1.neurons[i].output);
        }

        // Обновление весов
        for (auto& n : L1.neurons) n.update_weights(window,     lr);
        for (auto& n : L2.neurons) n.update_weights(corr_input, lr);
        for (auto& n : L3.neurons) n.update_weights(l2_out,     lr);
    }
};

// ==========================================
// Печать структуры сети
// ==========================================
void printNetworkStructure(const SelfCorrectingNetwork& net) {
    std::cout << "\n=== Структура сети (с корректирующим слоем) ===\n";

    MiniNN::TableBuilder t;
    t.addColumn("Слой", 6, false)
     .addColumn("Тип", 16, false)
     .addColumn("Нейронов", 10, true)
     .addColumn("Входов/нейрон", 14, true)
     .addColumn("Весов", 8, true)
     .addColumn("Смещений", 10, true);

    struct Info { const char* name; const char* type; const MiniNN::Layer* L; };
    Info infos[] = {
        {"L1", "скрытый",        &net.L1},
        {"L2", "корректирующий", &net.L2},
        {"L3", "выходной",       &net.L3},
    };

    std::size_t tw = 0, tb = 0;
    for (auto& info : infos) {
        std::size_t n = info.L->neurons.size();
        std::size_t w = n ? info.L->neurons[0].weights.size() : 0;
        tw += n * w;
        tb += n;
        t.addRow({ info.name, info.type, std::to_string(n), std::to_string(w),
                   std::to_string(n * w), std::to_string(n) });
    }
    t.addRow({ "ИТОГО", "", "", "", std::to_string(tw), std::to_string(tb) });
    t.print();

    std::cout << "  L2 (корректирующий) получает: "
              << HIDDEN1_NEURONS << " (L1) + "
              << WINDOW_SIZE     << " (вводы) + "
              << WINDOW_SIZE     << " (предсказания) + "
              << "2 (switch_rate, quality) = "
              << (HIDDEN1_NEURONS + 2 * WINDOW_SIZE + 2) << " входов\n";
}

// ==========================================
// Печать весов по слоям (команда 's')
// ==========================================
void printNetworkParams(const SelfCorrectingNetwork& net) {
    auto printLayer = [](const std::string& name, const MiniNN::Layer& L) {
        std::cout << "\n" << name << " (" << L.neurons.size() << " нейронов, "
                  << (L.neurons.empty() ? 0 : L.neurons[0].weights.size())
                  << " входов):\n";
        MiniNN::TableBuilder wt;
        wt.addColumn("Нейрон", 8, false).addColumn("bias", 12, true);
        std::size_t wc = L.neurons.empty() ? 0 : L.neurons[0].weights.size();
        for (std::size_t w = 0; w < wc; ++w)
            wt.addColumn("w" + std::to_string(w + 1), 12, true);

        for (std::size_t n = 0; n < L.neurons.size(); ++n) {
            std::vector<std::string> row;
            row.push_back("N" + std::to_string(n + 1));
            row.push_back(MiniNN::fmt(L.neurons[n].bias, 4));
            for (double w : L.neurons[n].weights)
                row.push_back(MiniNN::fmt(w, 4));
            wt.addRow(row);
        }
        wt.print();
    };
    printLayer("L1 (скрытый)",        net.L1);
    printLayer("L2 (корректирующий)", net.L2);
    printLayer("L3 (выходной)",       net.L3);
}

// ==========================================
// Печать таблицы анализа
// ==========================================
void printAnalysisTable(const std::vector<std::vector<std::string>>& rows) {
    MiniNN::TableBuilder t;
    t.addColumn("Шаг",   4, false)
     .addColumn("Окно входа", 16, false)
     .addColumn("P(1)",   7, true)
     .addColumn("Пр",     4, true)
     .addColumn("Вв",     4, true)
     .addColumn("Ош",     7, true)
     .addColumn("Switch", 7, true)
     .addColumn("Qual",   7, true)
     .addColumn("Самоконтроль L2", 18, false)
     .addColumn("Итог",   6, false);

    std::size_t start = 0;
    if ((int)rows.size() > MAX_VISIBLE_ROWS)
        start = rows.size() - MAX_VISIBLE_ROWS;
    for (std::size_t i = start; i < rows.size(); ++i)
        t.addRow(rows[i]);
    t.print();
}

// ==========================================
// main
// ==========================================
int main() {
    system("chcp 65001 > nul");
    srand(42);

    SelfCorrectingNetwork net(WINDOW_SIZE, HIDDEN1_NEURONS, HIDDEN2_NEURONS);

    printNetworkStructure(net);

    std::cout << "\n=== Нейросеть-предсказатель с самокоррекцией ===\n"
              << "Вводите 0 или 1. После накопления " << WINDOW_SIZE
              << " значений сеть предскажет следующий ввод.\n"
              << "L2 видит: L1-выход, посл. " << WINDOW_SIZE << " вводов, "
              << "посл. " << WINDOW_SIZE << " предсказаний,\n"
              << "          switch_rate (частота смены выбора, окно "
              << STATS_WINDOW << ")\n"
              << "          quality     (точность за посл. "
              << QUALITY_WINDOW << " предсказаний).\n"
              << "Команды: q — выход, s — структура и веса.\n\n";

    std::vector<int> history;        // все вводы пользователя
    std::vector<int> pred_history;   // все предсказания сети (0/1)
    std::vector<std::vector<std::string>> rows;

    while (true) {
        std::cout << "Введите 0/1 (или q/s): ";
        std::string line;
        if (!std::getline(std::cin, line)) break;

        if (line == "q" || line == "Q") break;

        if (line == "s" || line == "S") {
            printNetworkStructure(net);
            printNetworkParams(net);
            std::cout << "\n";
            continue;
        }

        if (line != "0" && line != "1") {
            std::cout << "Введите 0, 1, q или s.\n";
            continue;
        }

        int input = line[0] - '0';
        int step  = (int)history.size() + 1;

        std::vector<std::string> row;
        row.push_back(std::to_string(step));

        // --- мета-признаки (считаются по уже накопленной истории) ---
        double switch_rate = computeSwitchRate(history, STATS_WINDOW);
        double quality     = computeQuality(history, pred_history, QUALITY_WINDOW);

        if ((int)history.size() >= WINDOW_SIZE) {
            // --- Окно последних WINDOW_SIZE вводов ---
            std::vector<double> window;
            for (int i = (int)history.size() - WINDOW_SIZE;
                 i < (int)history.size(); ++i)
                window.push_back((double)history[i]);

            std::vector<double> last_inputs = window;

            // --- Последние WINDOW_SIZE предсказаний ---
            std::vector<double> last_preds;
            for (int k = 0; k < WINDOW_SIZE; ++k) {
                int idx = (int)pred_history.size() - WINDOW_SIZE + k;
                if (idx < 0) last_preds.push_back(0.5);
                else         last_preds.push_back((double)pred_history[idx]);
            }

            // --- Forward с двумя мета-признаками ---
            std::vector<double> out = net.forward(window, last_inputs, last_preds,
                                                  switch_rate, quality);
            double p1        = out[0];
            int    predicted = (p1 >= 0.5) ? 1 : 0;
            double error     = (double)input - p1;

            net.backward(window, (double)input, LEARNING_RATE);

            // --- Строка таблицы ---
            std::string win_str;
            for (std::size_t i = 0; i < window.size(); ++i) {
                if (i) win_str += " ";
                win_str += std::to_string((int)window[i]);
            }

            std::string ctrl_str;
            int matches = 0, tot = 0;
            for (int k = 0; k < WINDOW_SIZE; ++k) {
                int idx = (int)pred_history.size() - WINDOW_SIZE + k;
                if (idx < 0) continue;
                ctrl_str += std::to_string(pred_history[idx]);
                ++tot;
                if (pred_history[idx] == (int)window[k]) ++matches;
            }
            if (ctrl_str.empty()) ctrl_str = "-";
            else ctrl_str += " (" + std::to_string(matches) + "/" +
                             std::to_string(tot) + ")";

            row.push_back(win_str);
            row.push_back(MiniNN::fmt(p1, 3));
            row.push_back(std::to_string(predicted));
            row.push_back(std::to_string(input));
            row.push_back(MiniNN::fmt(error, 3));
            row.push_back(MiniNN::fmt(switch_rate, 2));
            row.push_back(MiniNN::fmt(quality, 2));
            row.push_back(ctrl_str);
            row.push_back(predicted == input ? "OK" : "FAIL");
        } else {
            row.push_back("(накопление)");
            row.push_back("-");
            row.push_back("-");
            row.push_back(std::to_string(input));
            row.push_back("-");
            row.push_back(MiniNN::fmt(switch_rate, 2));
            row.push_back("-");
            row.push_back("-");
            row.push_back("-");
        }

        rows.push_back(row);
        history.push_back(input);
        if ((int)history.size() > WINDOW_SIZE)
            pred_history.push_back(std::stoi(rows.back()[3]));

        clearScreen();
        printNetworkStructure(net);
        std::cout << "\n=== Анализ предсказаний ===\n";
        printAnalysisTable(rows);
    }

    int total = 0, ok = 0;
    for (const auto& r : rows) {
        if (r.size() >= 10 && (r[9] == "OK" || r[9] == "FAIL")) {
            ++total;
            if (r[9] == "OK") ++ok;
        }
    }
    std::cout << "\n=== Итоги ===\n";
    std::cout << "Всего предсказаний: " << total << "\n";
    std::cout << "Угадано: " << ok << "\n";
    if (total > 0)
        std::cout << "Точность: " << MiniNN::fmt(100.0 * ok / total, 2) << " %\n";

    return 0;
}