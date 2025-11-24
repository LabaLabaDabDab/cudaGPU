# Лабораторная работа №2

## Задание:
"Реализовать программу для накладывания фильтров на изображения. Возможные фильтры: размытие, выделение границ, избавление от шума. Изменить время
Для работы с графическими файлами рекомендуется использовать libpng (man libpng). Примеры использования библиотеки в /usr/share/doc/libpng12-dev/examples/
Считать изображение из файла, преобразовать в массив, отправить на CUDA Device, провести на устройстве обработку фильтром, вернуть изображение в память хоста, преобразовать в картинку, сохранить файл."

## Характеристики устройства
```
Device name: NVIDIA GeForce RTX 5070 Ti
Number of multiprocessors: 70
Global memory size: 17094475776 bytes
Max threads per block: 1024
Max grid size: 2147483647 x 65535 x 65535
Max block dimensions: 1024 x 1024 x 64
```

## Результаты работы
### Blur filter 
```
Loaded image: test_img.png (1024 x 1536), channels = 4
Host -> Device copy time: 0.384 ms
Kernel execution time: 0.096 ms
Device -> Host copy time: 2.528 ms
Total GPU time: 3.888 ms
```

### Edge filter
```
Loaded image: test_img.png (1024 x 1536), channels = 4
Host -> Device copy time: 0.391 ms
Kernel execution time: 0.087 ms
Device -> Host copy time: 2.429 ms
Total GPU time: 4.098 ms
```

### Denoise filter
```
Loaded image: test_img.png (1024 x 1536), channels = 4
Host -> Device copy time: 0.370 ms
Kernel execution time: 0.074 ms
Device -> Host copy time: 2.350 ms
Total GPU time: 3.898 ms
```

## Вывод
### Время
Время вычислений на GPU составляет менее 3% от общего времени выполнения программы. Основным узким местом является передача изображений между хостом и устройством, которая занимает в среднем 3 ms. Выполнение свёрточного фильтра занимает ~0.08 ms благодаря высокой степени параллелизма, поскольку каждый поток обрабатывает один пиксель.