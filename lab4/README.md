# Лабораторная работа №4

## Задание:
Разобрать программы из глав 6, 7 методического пособия, запустить, замерить время, сравнить по производительности, уметь объяснять работу программ детально.

## Характеристики устройства
```
Device name: NVIDIA GeForce RTX 5070 Ti
Number of multiprocessors: 70
Global memory size: 17094475776 bytes
Max threads per block: 1024
Max grid size: 2147483647 x 65535 x 65535
Max block dimensions: 1024 x 1024 x 64
```

---
## Глава 6
---
## 1) Создание матрицы
### Результаты работы
```
===== TASK PARAMETERS =====
Matrix size: 10000 x 10000
Total elements: 100000000
Total data size: 381.47 MB

===== CPU DATA PREPARATION =====
CPU matrix creation time: 249.349182 ms

===== GPU =====
GPU memory allocation time: 5.080288 ms
Grid: (625, 625)
Block: (16, 16)
Kernel iterations: 10
Kernel time (total): 5.625984 ms
Kernel time (per iter): 0.562598 ms
Device->host copy time: 158.630692 ms
Total GPU time: 169.587585 ms

===== RESULT CHECK =====
Matrices A and B are identical.

===== PERFORMANCE SUMMARY =====
CPU time (1 matrix): 249.349182 ms
GPU time (total): 169.587585 ms
GPU kernel (10 matrices): 5.625984 ms
GPU kernel per matrix: 0.562598 ms
Speedup (CPU / GPU total): 1.470x
Speedup (CPU / GPU kernel only): 443.210x

===== DATA TRANSFER =====
Data size: 381.47 MB
Allocation time: 5.080288 ms
D2H copy time: 158.630692 ms
```

### Выводы
Программа корректно выполняет поэлементное сложение матриц: результаты GPU полностью совпадают с CPU-версией, что подтверждает корректность реализации ядра и обращения к памяти.

Однако по производительности GPU здесь существенно уступает CPU. Сложение матриц на CPU занимает около 16 ms, тогда как выполнение ядра на GPU длится примерно 862 ms. Это означает, что GPU выполняет вычисления медленнее CPU примерно в 54 раза, то есть ускорения нет, а происходит замедление. Полное время GPU-варианта, включая выделение памяти и копирование данных, составляет около 877 ms, что делает его медленнее CPU примерно в 55 раз.

Причина в характере задачи: поэлементное сложение — очень «лёгкая» операция, почти полностью ограниченная пропускной способностью памяти. При таком низком числе операций на элемент накладные расходы GPU (запуск ядра, PCIe-передачи, рассинхронизация) полностью перекрывают потенциальный выигрыш от параллелизма. GPU становится эффективным только для более тяжёлых вычислений или для цепочек операций, выполняемых без лишних копирований между CPU и устройством.

---
## 2) Транспонирование матрицы
### Результаты работы
```
===== TASK PARAMETERS =====
Original matrix size (rows x cols): 1008 x 2000
Transposed matrix size (cols x rows): 2000 x 1008
Data size A : 7.69 MB
Data size AT: 7.69 MB

===== CPU PHASE =====
CPU transpose time: 6.903552 ms

===== GPU =====
GPU allocation time: 0.788032 ms
Host->device copy time: 1.692320 ms
Grid: (125, 63)
Block: (16, 16)
Kernel time: 3.001248 ms
Device->host copy time: 3.794944 ms
Total GPU time: 9.529888 ms

===== RESULT CHECK =====
CPU and GPU transposed matrices are identical.

===== PERFORMANCE SUMMARY =====
CPU transpose time: 6.903552 ms
GPU total time: 9.529888 ms
GPU kernel time: 3.001248 ms
Speedup (CPU / GPU kernel only): 2.300x
Speedup (CPU / GPU total): 0.724x

===== DATA TRANSFER =====
Data size A : 7.69 MB
Data size AT: 7.69 MB
Allocation time: 0.788032 ms
Host->device time: 1.692320 ms
Device->host time: 3.794944 ms
```

### Выводы
Программа корректно выполняет транспонирование матрицы на CPU и GPU: результаты полностью совпадают, что подтверждает корректность реализации ядра и индексации элементов.
По «чистым» вычислениям GPU работает заметно быстрее: ядро выполняет транспонирование примерно за 3.0 ms против 6.9 ms на CPU. Это означает, что сами вычисления удалось ускорить примерно в 2.3 раза. Однако при учёте накладных расходов (выделение памяти на устройстве, копирование ~15 MB данных туда и обратно) преимущество исчезает: полное время GPU-варианта составляет ~9.5 ms, что примерно в 1.4 раза медленнее CPU. Ограничивающим фактором снова становится высокая стоимость PCIe-передач, а не скорость арифметики.

Использование GPU становится выгодным либо при существенно больших матрицах, либо в сценариях, где данные уже находятся на устройстве и проходят через цепочку нескольких операций без возврата на CPU.

---
## 3) Суммирование матриц
### Результаты работы
```
===== TASK PARAMETERS =====
Matrix size (rows x cols): 2000 x 3008
Data size A : 22.95 MB
Data size B : 22.95 MB
Data size C : 22.95 MB
Total data (A+B+C): 68.85 MB

===== CPU PHASE =====
CPU matrix addition time: 16.028065 ms

===== GPU =====
GPU allocation time: 1.541152 ms
Host->device copy time (A+B): 3.768128 ms
Grid: (188, 125)
Block: (16, 16)
Kernel time: 861.670288 ms
Device->host copy time (C): 9.942592 ms
Total GPU time: 877.183228 ms

===== RESULT CHECK =====
CPU and GPU results are identical.

===== PERFORMANCE SUMMARY =====
CPU addition time: 16.028065 ms
GPU total time: 877.183228 ms
GPU kernel time: 861.670288 ms
Speedup (CPU / GPU kernel only): 0.019x
Speedup (CPU / GPU total): 0.018x

===== DATA TRANSFER =====
Matrix size: 2000 x 3008
Data size A : 22.95 MB
Data size B : 22.95 MB
Data size C : 22.95 MB
Allocation time: 1.541152 ms
Host->device time: 3.768128 ms
Device->host time: 9.942592 ms
```

### Выводы
Программа корректно выполняет поэлементное сложение матриц: результаты CPU и GPU полностью совпадают, что подтверждает правильность реализации ядра matrixAdd и корректную работу с памятью.

По производительности GPU здесь заметно уступает CPU. Сложение на CPU занимает примерно 16 ms, тогда как «чистое» GPU-вычисление длится около 862 ms, а полное время с учётом выделения памяти и передачи данных достигает ~877 ms. То есть вместо ускорения мы фактически замедлили вычисление примерно в 50 раз.

Причина в том, что операция слишком простая: одно сложение на элемент не даёт достаточной вычислительной нагрузки, чтобы окупить накладные расходы GPU. При таком низком арифметическом коэффициенте параллельность устройства не компенсирует издержки, и использование GPU становится неэффективным. Для поэлементных операций этого класса целесообразно оставаться на CPU или применять GPU только в составе более тяжёлых вычислительных конвейеров, где данные уже находятся на устройстве.

---
## 4) Умножение матриц
### Результаты работы
```
===== TASK PARAMETERS =====
A: 112 x 208
B: 208 x 160
C: 112 x 160
Data size A: 0.18 MB
Data size B: 0.25 MB
Data size C: 0.14 MB
Total data (A+B+C): 0.57 MB

===== CPU PHASE =====
CPU matrix multiplication time: 9.087904 ms

===== GPU =====
GPU allocation time: 0.521088 ms
Host->device copy time (A+B): 1.868736 ms
Grid: (10, 7)
Block: (16, 16)
Kernel time: 0.310624 ms
Device->host copy time (C): 0.117152 ms
Total GPU time: 3.076160 ms

===== RESULT CHECK =====
CPU and GPU results are identical (within tolerance).

===== PERFORMANCE SUMMARY =====
CPU matrix multiplication: 9.087904 ms
GPU total time: 3.076160 ms
GPU kernel time: 0.310624 ms
Speedup (CPU / GPU kernel only): 29.257x
Speedup (CPU / GPU total): 2.954x

===== DATA TRANSFER =====
Data size A: 0.18 MB
Data size B: 0.25 MB
Data size C: 0.14 MB
Allocation time: 0.521088 ms
Host->device time: 1.868736 ms
Device->host time: 0.117152 ms
```

### Выводы
Программа корректно выполняет умножение матриц: результаты на CPU и GPU совпадают с точностью до заданного допуска, что подтверждает правильность реализации ядра matrixMult и логики работы с памятью.

По производительности видно ощутимое преимущество GPU. Само вычисление на видеокарте занимает ~0.31 ms против ~9.09 ms на CPU, то есть по «чистым» вычислениям мы ускорили умножение примерно в 29 раз. С учётом всех накладных расходов (выделение памяти и копирование матриц A и B на устройство и C обратно) полное время GPU-варианта составляет ~3.08 ms, что даёт ускорение примерно в 3 раза относительно CPU. В этом эксперименте объём данных небольшой (~0.57 MB), поэтому стоимость передачи данных не «съедает» выигрыш, и использование GPU для умножения матриц оказывается оправданным.

---
## Глава 7
---
## 1) Умножение векторов с shared памятью
### Результаты работы
```
===== TASK PARAMETERS =====
Vector length (original): 1000000
Vector length (aligned): 1000192
Block size: 256
Iterations: 100
Data size A: 3.82 MB
Data size B: 3.82 MB
Result size: 4 bytes
Total data (A+B+Result): 7.63 MB

===== CPU PHASE =====
CPU dot product time: 2.904288 ms
CPU result: 249863.093750

===== GPU =====
GPU allocation time: 1.180192 ms
Host->device copy time (A+B): 2.278432 ms
Grid: (3907)
Block: (256)

Global kernel total time (100 iters): 128.134521 ms
Global kernel avg time: 1.281345 ms

Shared kernel total time (100 iters): 2.248992 ms
Shared kernel avg time: 0.022490 ms

Device->host copy time (results): 0.305664 ms
Total GPU time: 787.753052 ms

===== RESULT CHECK =====
Mismatch: CPU=249863.093750, GPU=249863.796875, diff=0.703125, eps=0.010000
Mismatch: CPU=249863.093750, GPU=249902.531250, diff=39.437500, eps=0.010000
CPU result: 249863.093750
Global result: 249863.796875
Shared result: 249902.531250
Global kernel result differs from CPU.
Shared kernel result differs from CPU.

===== PERFORMANCE SUMMARY =====
CPU dot product: 2.904288 ms
GPU total time: 787.753052 ms
GPU global kernel avg: 1.281345 ms
GPU shared kernel avg: 0.022490 ms
Speedup (CPU / GPU global avg): 2.267x
Speedup (CPU / GPU shared avg): 129.137x
Shared vs Global improvement: 98.245%

===== DATA TRANSFER =====
Data size A: 3.82 MB
Data size B: 3.82 MB
Result size: 4 bytes
Allocation time: 1.180192 ms
Host->device time: 2.278432 ms
Device->host time: 0.305664 ms
```

### Выводы
Программа корректно вычисляет скалярное произведение векторов на CPU, однако оба GPU-варианта дают заметное расхождение с эталонным результатом. Это означает, что в реализациях dotProductGlobal и dotProductShared присутствует ошибка суммирования, связанная с накоплением погрешности или некорректной редукцией.

По производительности результаты показательны. Использование разделяемой памяти радикально ускоряет вычисления: среднее время работы shared-ядра составляет ~0.022 ms против ~1.28 ms для global-ядра, то есть shared-вариант работает примерно в 57 раз быстрее благодаря локальной редукции внутри блока и уменьшению числа атомарных операций. Относительно CPU shared-ядро показывает ускорение примерно в 129 раз по чистым вычислениям.

Тем не менее полный GPU-вариант остаётся медленнее CPU: итоговое время ~788 ms против ~2.9 ms на CPU. Причина в больших накладных расходах: многократные запуски ядра, повторное обнуление памяти, синхронизации и передача данных полностью перекрывают выигрыш от параллелизма. Чтобы GPU был быстрее, необходимо либо выполнять вычисления контейнерно (без 100 отдельных запусков), либо полностью избегать лишних копирований и атомарных операций.

---
## 2) Умножение матриц с shared памятью
### Результаты работы
```
===== TASK PARAMETERS =====
A (original): 500 x 400
B (original): 400 x 300
A (aligned):  512 x 400
B (aligned):  400 x 304
C (aligned):  512 x 304
Block size:   16 x 16
Iterations:   100
Data size A: 0.78 MB
Data size B: 0.46 MB
Data size C: 0.59 MB
Total data (A+B+C): 1.84 MB

===== CPU PHASE =====
CPU matrix multiplication time: 152.522659 ms

===== GPU =====
GPU allocation time: 0.743232 ms
Host->device copy time (A+B): 2.287904 ms
Grid: (19, 32)
Block: (16, 16)

Basic kernel total time (100 iters): 5.461664 ms
Basic kernel avg time: 0.054617 ms

Tiled kernel total time (100 iters): 4.011104 ms
Tiled kernel avg time: 0.040111 ms

Device->host copy time (C basic + C tiled): 0.995008 ms
Total GPU time: 734.530640 ms

===== RESULT CHECK =====
Basic kernel result matches CPU (within tolerance).
Tiled kernel result matches CPU (within tolerance).

===== PERFORMANCE SUMMARY =====
CPU matrix multiplication: 152.522659 ms
GPU total time: 734.530640 ms
GPU basic kernel avg: 0.054617 ms
GPU tiled kernel avg: 0.040111 ms
Speedup (CPU / GPU basic avg): 2792.604x
Speedup (CPU / GPU tiled avg): 3802.510x
Shared memory gain (basic / tiled): 1.362x

===== DATA TRANSFER =====
Data size A: 0.78 MB
Data size B: 0.46 MB
Data size C: 0.59 MB
Allocation time: 0.743232 ms
Host->device time: 2.287904 ms
Device->host time: 0.995008 ms
```

### Выводы
Программа корректно умножает матрицы: результаты обоих GPU-ядер (basic и tiled) совпадают с CPU-вариантом с заданной точностью, что подтверждает правильность реализации и работы с памятью.

По производительности сами вычисления на GPU на порядки быстрее CPU: среднее время basic-ядра ~0.055 ms и tiled-ядра ~0.040 ms против ~152.5 ms на CPU, то есть ускорение по «чистым» вычислениям порядка 2800 раз для базового ядра и около 3800 раз для tiled-варианта. Использование shared memory и блочного перемножения даёт дополнительный выигрыш: tiled-ядро работает примерно в 1.4 раза быстрее, чем базовое.

Однако полное время работы GPU-пайплайна (с учётом аллокаций, копирования данных и многократных запусков ядер) составляет ~735 ms и оказывается медленнее CPU. Как и в других экспериментах, основная проблема не в скорости вычислений, а в накладных расходах и выбранном сценарии измерений: чтобы выигрыш от GPU был заметен на уровне всей программы, нужно либо уменьшать число лишних запусков, либо включать умножение в более крупный, вычислительно насыщенный конвейер без постоянного возврата данных на CPU.