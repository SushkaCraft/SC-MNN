#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <cstddef>

namespace MiniNN {

    // Количество отображаемых (не байтовых!) символов в UTF-8 строке
    inline std::size_t utf8Length(const std::string& s) {
        std::size_t count = 0;
        for (unsigned char c : s) {
            if ((c & 0xC0) != 0x80) ++count;
        }
        return count;
    }

    // Дополнение строки пробелами до нужной визуальной ширины
    inline std::string padTo(const std::string& s, std::size_t width, bool right) {
        std::size_t len = utf8Length(s);
        if (len >= width) return s;
        std::string pad(width - len, ' ');
        return right ? (pad + s) : (s + pad);
    }

    // Форматирование числа в строку фиксированной точности
    inline std::string fmt(double value, int precision = 4) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(precision) << value;
        return ss.str();
    }

    inline std::string fmtInt(long long value) {
        return std::to_string(value);
    }

    // ==========================================
    // Универсальный построитель таблиц
    // ==========================================
    class TableBuilder {
    public:
        TableBuilder& addColumn(const std::string& header,
                                std::size_t width,
                                bool right = false) {
            columns_.push_back({header, width, right});
            return *this;
        }

        TableBuilder& addRow(const std::vector<std::string>& cells) {
            rows_.push_back(cells);
            return *this;
        }

        void print(std::ostream& os = std::cout) const {
            if (columns_.empty()) return;

            // Заголовок
            for (std::size_t i = 0; i < columns_.size(); ++i) {
                if (i > 0) os << " | ";
                os << padTo(columns_[i].header, columns_[i].width, columns_[i].right);
            }
            os << "\n";

            // Разделитель
            std::size_t total = 0;
            for (std::size_t i = 0; i < columns_.size(); ++i) {
                total += columns_[i].width;
                if (i > 0) total += 3; // " | "
            }
            os << std::string(total, '-') << "\n";

            // Строки данных
            for (const auto& row : rows_) {
                for (std::size_t i = 0; i < columns_.size(); ++i) {
                    if (i > 0) os << " | ";
                    std::string cell = (i < row.size()) ? row[i] : std::string();
                    os << padTo(cell, columns_[i].width, columns_[i].right);
                }
                os << "\n";
            }
        }

    private:
        struct Column {
            std::string header;
            std::size_t width;
            bool        right;
        };
        std::vector<Column>                    columns_;
        std::vector<std::vector<std::string>>  rows_;
    };
}