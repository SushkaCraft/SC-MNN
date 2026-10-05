# MiniNN — мини-библиотека нейросетей и табличного вывода

Два header-only файла для C++ в пространстве имён `MiniNN`. Не требуют сборки отдельных `.cpp` — достаточно подключить заголовки.

## Состав

| Файл | Назначение |
|------|------------|
| `NeuralNetwork.h` | Простая полносвязная нейронная сеть с сигмоидной активацией |
| `TableBuilder.h` | Построение аккуратных таблиц в консоли с поддержкой UTF-8 |

---

## NeuralNetwork.h

**Что это:** заголовочная реализация простой полносвязной нейросети. Подходит для учебных целей, прототипов и небольших задач.

**Основные сущности:**
- `sigmoid`, `sigmoid_derivative` — функции активации;
- `Neuron` — веса, bias, output, delta, методы `forward`, `update_weights`;
- `Layer` — слой нейронов, метод `forward`;
- `Network` — стек слоёв, методы `addLayer`, `forward`, `backward`, `save`, `load`, `empty`.

**Для чего может понадобиться:**
- обучение на XOR, простых классификаторах, аппроксимации функций;
- быстрый прототип без сторонних зависимостей;
- встраивание в C++ проект как header-only;
- сохранение и загрузка обученных весов в текстовый файл;
- демонстрация работы обратного распространения ошибки.

**Пример:**
```cpp
#include "NeuralNetwork.h"
using namespace MiniNN;

int main() {
    Network net;
    net.addLayer(Layer(4, 2)); // 4 нейрона, 2 входа
    net.addLayer(Layer(1, 4)); // 1 нейрон, 4 входа

    std::vector<double> input = {0, 1};
    std::vector<double> target = {1};

    for (int i = 0; i < 10000; ++i) {
        net.forward(input);
        net.backward(input, target, 0.5);
    }

    auto out = net.forward(input);
    net.save("model.txt");
}
```

**Ограничения:**
- только сигмоида;
- `rand()` без автоматического seed;
- нет батчей, оптимизаторов, dropout и регуляризации;
- при несовпадении размеров входов и весов поведение `Neuron::forward` может быть некорректным — следите за размерами.

---

## TableBuilder.h

**Что это:** утилита для красивого вывода таблиц в консоль с учётом UTF-8.

**Возможности:**
- `addColumn(header, width, right)`
- `addRow({...})`
- `print()`
- `fmt`, `fmtInt`, `utf8Length`, `padTo`

**Для чего может понадобиться:**
- вывод метрик обучения по эпохам;
- логи, отчёты, сравнение экспериментов;
- аккуратное консольное представление данных.

**Пример:**
```cpp
#include "TableBuilder.h"
using namespace MiniNN;

TableBuilder tb;
tb.addColumn("Epoch", 8, true)
  .addColumn("Loss", 12, true);

tb.addRow({fmtInt(1), fmt(0.1234)});
tb.addRow({fmtInt(2), fmt(0.0456)});
tb.print();
```

---

## Подключение

Оба файла header-only. Достаточно положить их в проект и подключить:

```cpp
#include "NeuralNetwork.h"
#include "TableBuilder.h"
```

## СБорка и запуск проекта

**Через PowerShell запустить файл `vs.ps1` в рабочей иректории и выполнить почередно команды**:
1. cl /c /EHcs .\main.cpp | link .\main.obj /out:main.exe
2. .\main.exe