# YOLOv8 со сложением признаков

В этом эксперименте объединение признаков через конкатенацию заменено
сложением в neck, блоках C2f и SPPF. Замена касается архитектуры; объединение
выходов детектора при постобработке рассматривается отдельно.

## Конфигурации

| Конфигурация | Объединение масштабов в neck | Объединение в C2f | Объединение в SPPF |
| --- | --- | --- | --- |
| `configs/yolov8/yolov8_s_add_neck_only.py` | Add | Concat | Concat |
| `configs/yolov8/yolov8_s_add.py` | Add | Add | Add |
| `configs/yolov8/yolov8_s_add_fine_tune.py` | Add | Add | Add |
| `configs/yolov8/yolov8_s_add_gray_hard.py` | Add | Add | Add |
| `configs/yolov8/yolov8_s_add_gray_hard_fine_tune.py` | Add | Add | Add |
| `configs/yolov8/yolov8_s_add_surgery2.py` | Add | Add | Add |
| `configs/yolov8/yolov8_s_add_surgery2_fine_tune.py` | Add | Add | Add |
| `configs/yolov8/yolov8_n_focus_add_surgery2.py` | Add | Add | Add |

Первые две конфигурации наследуют существующий конфиг обучения YOLOv8-s,
включая настройки локального датасета. Вариант дообучения использует обучающую
и валидационную выборки `data/small_people_v3/`, прежнюю предобработку
(`std=1`, как в исходном checkpoint) и расписание на 100 эпох.
Все стадии backbone и их BatchNorm разморожены. Незарегистрированный
`ForceBNToEvalHook` удалён; для скорости обучения используется один хук
YOLOv5 с косинусным расписанием. Перед запуском проверьте пути к датасету.

`YOLOv8AddPAFPN` преобразует признаки верхних уровней свёрткой 1x1 до
увеличения разрешения. Свёртки с шагом 2 сразу формируют нужное число каналов
для сложения в направлении снизу вверх. Число каналов и шаги выходов neck
соответствуют конфигурации, поэтому существующая голова детектора совместима
с ними.

В C2f сохранена последовательность bottleneck-блоков, но их выходы
суммируются вместо хранения всех ветвей для конкатенации. Финальная свёртка
1x1 получает `mid_channels` вместо `(2 + num_blocks) * mid_channels`.
В SPPF складываются ветви пулинга; при стандартном последовательном пулинге
число входных каналов финальной свёртки уменьшается с `4 * mid_channels`
до `mid_channels`.

В существующих классах по умолчанию используется `fusion_mode='concat'`.
Исходные конфигурации сохраняют прежнюю архитектуру и размеры весов.

## Обучение

Обучение модели со сложением во всех перечисленных блоках с нуля:

```bash
python tools/train.py configs/yolov8/yolov8_s_add.py \
  --work-dir work_dirs/yolov8_s_add
```

Для первого эксперимента с меньшим числом изменений используйте
`yolov8_s_add_neck_only.py` и отдельный рабочий каталог. Сравнивайте с исходной
моделью на одинаковом датасете, с одинаковой предобработкой, расписанием
обучения и настройками оценки.

Чтобы перенести совместимые веса из обученной модели MMYOLO:

```bash
python tools/model_converters/yolov8_add_checkpoint.py \
  configs/yolov8/yolov8_s_add_fine_tune.py \
  work_dirs/yolov8_s_big_dataset_2/best_coco_bbox_mAP_epoch_210.pth \
  work_dirs/yolov8_s_add/warm_start.pth

python tools/train.py configs/yolov8/yolov8_s_add_fine_tune.py \
  --work-dir work_dirs/yolov8_s_add_fine_tune \
  --cfg-options load_from=work_dirs/yolov8_s_add/warm_start.pth
```

Конвертер переносит только тензоры с совпадающими именами и размерами,
выводя список пропущенных и новых тензоров. Изменённые свёртки и новые
проекции сохраняют начальную инициализацию. Состояния оптимизатора, счётчик
эпох и EMA не переносятся. Для такого старта используйте `load_from`,
а не `--resume`. Checkpoint Ultralytics сначала нужно преобразовать
в формат MMYOLO существующим конвертером.

Для других размеров модели используйте те же поля архитектуры с её
коэффициентами ширины, глубины и числом каналов последней стадии.
Настройки каналов backbone, neck и головы должны быть согласованы.
Для отдельных компонентов можно оставить конкатенацию:
`model.backbone.csp_fusion_mode='concat'`,
`model.backbone.spp_fusion_mode='concat'` или
`model.neck.csp_fusion_mode='concat'` в neck со сложением между масштабами.

## Экспорт

Экспорт обученного checkpoint существующим инструментом:

```bash
python projects/easydeploy/tools/export_onnx.py \
  configs/yolov8/yolov8_s_add_fine_tune.py \
  work_dirs/yolov8_s_add_fine_tune/best_coco_bbox_mAP_epoch_100.pth \
  --model-only --export-type MMYOLO --model-surgery 0 \
  --work-dir work_dirs/yolov8_s_add_export --img-size 640 640 --opset 11
```

Замените путь на фактический лучший checkpoint после обучения.
Для этой конфигурации экспорт только модели выдаёт шесть выходов:
логиты классов и распределений координат. Генерация prototxt использует
имена выходов графа напрямую и работает без узлов `Concat`.
Экспорт начальных перенесённых весов полезен для проверки или компиляции
графа, но не является обученным детектором с новой архитектурой.

## ReLU и surgery 2

Конфигурации `yolov8_s_add_surgery2.py` и
`yolov8_s_add_surgery2_fine_tune.py` сохраняют трёхканальный RGB-вход.
В backbone, neck и голове используются ReLU; замены на HSwish или
Hardsigmoid нет. Для оценок классов сохранён обычный sigmoid, но экспорт
без постобработки возвращает исходные логиты, до sigmoid.

Обучение с нуля:

```bash
python tools/train.py configs/yolov8/yolov8_s_add_surgery2.py \
  --model-surgery 2 --work-dir work_dirs/yolov8_s_add_surgery2
```

Экспорт в ONNX и TorchScript:

```bash
CONFIG=configs/yolov8/yolov8_s_add_surgery2.py
CHECKPOINT=$(cat work_dirs/yolov8_s_add_surgery2/last_checkpoint)

python projects/easydeploy/tools/export_onnx.py "$CONFIG" "$CHECKPOINT" \
  --model-only --model-surgery 2 --img-size 640 640 --opset 11 \
  --work-dir work_dirs/yolov8_s_add_surgery2/onnx --device cpu

python projects/easydeploy/tools/export_pt.py "$CONFIG" "$CHECKPOINT" \
  --model-only --model-surgery 2 --img-size 640 640 \
  --work-dir work_dirs/yolov8_s_add_surgery2/pt --device cpu
```

Оба экспортёра возвращают три тензора NCHW:
`(1,65,80,80)`, `(1,65,40,40)` и `(1,65,20,20)`.
В каждом выходе канал 0 содержит логит единственного класса, каналы 1..64
содержат исходные распределения DFL в порядке left, top, right, bottom,
по 16 значений на сторону. Sigmoid, DFL-декодирование и NMS выполняются
снаружи модели. Имена выходов ONNX: `583`, `584`, `585`.
Архитектурные объединения используют Add; три конечных узла Concat
объединяют каналы класса и координат и остаются в графе намеренно.

Расширение `.pt` при `--model-surgery 2` означает TorchScript:
загружайте файл через `torch.jit.load`. Это не архив `.pt2` для
`torch.export.load`; преобразование в PT2 и квантизация выполняются отдельно.
При экспорте без `--model-surgery` PT-экспортёр сохраняет Python-модель
через `torch.save`, это другой формат загрузки.

## YOLOv8n с внешним Focus4

`yolov8_n_focus_add_surgery2.py` задаёт новую nano-архитектуру для
упакованного grayscale-входа. В отличие от преобразования старых весов
в `quantize_model/pt2_quant`, две исходные свёртки удалены:
обычный stem и свёртка с шагом 2 в первой стадии backbone.
Перед первым Split остаётся одна свёртка 3x3 с 16 входными и 32 выходными
каналами, шагом 1 и padding 1:

```text
Grayscale (1,1,640,640)
  -> внешний Focus4 (1,16,160,160)
  -> Conv 3x3, 16->32, stride=1 -> BN -> ReLU
  -> Split -> C2f со сложением ветвей
```

В обучении и валидации `YOLOv5FocusDetDataPreprocessor` сначала выполняет
обычную нормализацию и дополнение изображения, затем `pixel_unshuffle(..., 4)`.
Пайплайны, включая Mosaic и заключительную стадию без Mosaic, используют
grayscale. Координаты аннотаций и метаданные сохраняют исходный масштаб
640x640, а не размер упакованного тензора 160x160.

Это изменение вычислений модели, а не точная перепараметризация обученных
свёрток. В конфигурации установлены `load_from=None` и `resume=False`:
обучайте вариант с нуля в отдельном каталоге.

```bash
python tools/train.py configs/yolov8/yolov8_n_focus_add_surgery2.py \
  --model-surgery 2 --work-dir work_dirs/yolov8_n_focus_add_surgery2

CONFIG=configs/yolov8/yolov8_n_focus_add_surgery2.py
CHECKPOINT=$(cat work_dirs/yolov8_n_focus_add_surgery2/last_checkpoint)

python projects/easydeploy/tools/export_onnx.py "$CONFIG" "$CHECKPOINT" \
  --model-only --model-surgery 2 --img-size 640 640 --opset 11 \
  --work-dir work_dirs/yolov8_n_focus_add_surgery2/onnx --device cpu

python projects/easydeploy/tools/export_pt.py "$CONFIG" "$CHECKPOINT" \
  --model-only --model-surgery 2 --img-size 640 640 \
  --work-dir work_dirs/yolov8_n_focus_add_surgery2/pt --device cpu
```

Аргумент `--img-size 640 640` задаёт размер **до Focus**. Экспортёры читают
`deploy_cfg.input_spatial_divisor=4` и формируют вход модели
`(1,16,160,160)` типа float32. Focus не входит в граф инференса: перед
запуском преобразуйте изображение в grayscale, выполните letterbox
до 640x640 с заполнением 114, нормализуйте делением на 255 и упакуйте
каналы на CPU. Их порядок соответствует `pixel_unshuffle`:

```python
packed[n, 4 * r + c, y, x] = gray[n, 0, 4 * y + r, 4 * x + c]
# r, c = 0, 1, 2, 3
```

Настройки предобработки записываются в метаданные ONNX под ключом
`preprocess`, а в TorchScript сохраняются как дополнительный файл
`preprocess.json`. Эти записи описывают требуемую предобработку, но не
выполняют её автоматически. Prototxt использует координаты исходного
изображения 640x640. Порядок каналов и формы трёх выходов совпадают
с описанным выше вариантом RGB/ReLU.

Размеры входа, упаковка, один Conv до первого Split, обучение после
surgery 2 и экспорт проверяются в `test_yolov8_focus_add.py`.
Сохранение точности требует полноценного обучения и оценки на датасете;
проверочный шаг обучения и численное сравнение экспортов не заменяют mAP.

## Одноканальный вход и hard sigmoid

Конфигурации grayscale/hard используют настоящий одноканальный вход
и заменяют SiLU на HSwish в backbone, neck и голове:

```text
hard_sigmoid(x) = clip((x + 3) / 6, 0, 1)
HSwish(x) = x * hard_sigmoid(x)
```

Загрузчик сразу декодирует изображение в grayscale. `ToGrayscale` также
обрабатывает BGR-массивы, переданные через API инференса MMDetection.
Все пайплайны обучения, валидации, тестирования, TTA и загрузки Mosaic
используют один канал. Цветовые аугментации HSV и оттенка удалены;
яркость, контраст, размытие и геометрические преобразования сохранены.
Последние десять эпох используют grayscale-пайплайн без Mosaic.
Конфигурация `yolov8_s_add_gray_hard.py` на 500 эпох наследует исходную
`yolov8_s_syncbn_fast_8xb16-500e_coco.py`: размер батча из базового конфига,
LR 0.01,
аугментации для тепловизионных изображений, `mean=[0]`, `std=[255]`.
Нормализация соответствует сохранённой в
`work_dirs/yolov8_s_small_dataset_v3/best_coco_bbox_mAP_epoch_499.pth`.
Вход ONNX: нормализованное grayscale-изображение типа float, форма
`N x 1 x H x W`, значения в диапазоне 0..1. Преобразование в grayscale,
изменение размера и нормализация выполняются до подачи в ONNX.

Отдельная конфигурация `yolov8_s_add_gray_hard_fine_tune.py` на 100 эпох
сохраняет `std=[1]` для более раннего checkpoint `yolov8_s_big_dataset_2`.
Её вход ONNX использует диапазон 0..255. Для каждой конфигурации используйте
соответствующие ей нормализацию и checkpoint.

Обучение с нуля с исходным расписанием на 500 эпох:

```bash
python tools/train.py configs/yolov8/yolov8_s_add_gray_hard.py \
  --work-dir work_dirs/yolov8_s_add_gray_hard_scratch
```

В этой конфигурации заданы `load_from=None` и `resume=False`.
Для инициализации из модели, обученной 499 эпох, сначала подготовьте
её веса с учётом изменённых слоёв:

```bash
python tools/model_converters/yolov8_add_checkpoint.py \
  configs/yolov8/yolov8_s_add_gray_hard.py \
  work_dirs/yolov8_s_small_dataset_v3/best_coco_bbox_mAP_epoch_499.pth \
  work_dirs/yolov8_s_add_gray_hard_from_499/warm_start.pth --rgb-to-gray

python tools/train.py configs/yolov8/yolov8_s_add_gray_hard.py \
  --work-dir work_dirs/yolov8_s_add_gray_hard_from_499_train \
  --cfg-options load_from=work_dirs/yolov8_s_add_gray_hard_from_499/warm_start.pth
```

Для оценок классов hard sigmoid используется только при инференсе.
Назначение целей при обучении сохраняет обычный sigmoid: начальные смещения
классов YOLOv8 ниже -3, где hard sigmoid даёт нули и обнуляет веса
положительных целей. Ошибка классификации по-прежнему рассчитывается
численно устойчивым BCE с логитами. Обучающий sigmoid и функция потерь
не входят в граф инференса.
При экспорте только модели три выхода классов уже содержат значения
после hard sigmoid, а выходы координат сохраняют исходные логиты DFL.
Не применяйте sigmoid к выходам классов повторно.
В создаваемом prototxt устанавливается `score_converter=IDENTITY`.
Экспорт с постобработкой также использует hard sigmoid для оценок классов.
Исходные конфигурации RGB/sigmoid сохраняют прежний смысл выходов.

Если обучение с прежней реализацией hard sigmoid в assigner привело
к обнулению bbox/DFL loss, начните обучение с нуля в новом рабочем каталоге
с исправленным кодом. Не возобновляйте обучение из такого checkpoint.
Уже работающий процесс нужно перезапустить, чтобы он загрузил исправление.

Для более ранней конфигурации дообучения на 100 эпох используйте команды:

```bash
python tools/model_converters/yolov8_add_checkpoint.py \
  configs/yolov8/yolov8_s_add_gray_hard_fine_tune.py \
  work_dirs/yolov8_s_big_dataset_2/best_coco_bbox_mAP_epoch_210.pth \
  work_dirs/yolov8_s_add_gray_hard/warm_start.pth --rgb-to-gray

python tools/train.py configs/yolov8/yolov8_s_add_gray_hard_fine_tune.py \
  --work-dir work_dirs/yolov8_s_add_gray_hard_fine_tune \
  --cfg-options load_from=work_dirs/yolov8_s_add_gray_hard/warm_start.pth
```

`--rgb-to-gray` суммирует ядра первой свёртки по трём входным каналам.
При совпадающей нормализации это сохраняет отклик RGB-входного блока
на grayscale-изображение, повторённое в трёх каналах. Отклик исходной
RGB-модели на произвольные цветные изображения не сохраняется.
Изменение входа и активаций требует дообучения; затем подберите пороги
уверенности на валидационной выборке.

Hard sigmoid не эквивалентен обычному sigmoid: он меняет оценки уверенности
и обрезает логиты вне диапазона от -3 до 3 до нуля или единицы.
Сохранение точности не гарантируется. Сравните один обученный checkpoint
с обеими выходными функциями по mAP, precision и recall. Изменение порога
не восстанавливает оценки, обрезанные до нуля, и различия между оценками,
обрезанными до единицы.

Для grayscale используйте приведённую выше команду экспорта с нужной
конфигурацией и её обученным checkpoint. Экспортёр определяет число входных
каналов по backbone. При одинаковых размерах и типе данных входной буфер
уменьшается в три раза; это не означает трёхкратного уменьшения памяти
всей сети или времени инференса.

## Проверка

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
  -s tests/test_models -p test_yolov8_add.py -v
python -m unittest discover \
  -s tests/test_tools -p test_yolov8_add_checkpoint.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
  -s tests/test_models -p test_yolov8_gray_hard.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
  -s tests/test_models -p test_yolov8_add_surgery2.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
  -s tests/test_models -p test_yolov8_focus_add.py -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
  -s tests/test_tools -p test_export_pt.py -v
```

Тесты проверяют градиенты, прямоугольные входы, проекции каналов, сборку
конфигураций, выходы головы и перенос весов. Отдельно проверяются назначение
положительных целей и ненулевые bbox/DFL loss после начальной инициализации.
При установленном ONNX тесты экспортируют backbone и neck и проверяют число
архитектурных узлов `Concat`: 13 для исходной модели, 9 при замене только
в neck и 0 при полной замене. Итоговый экспорт детектора с постобработкой
может содержать необходимые конкатенации выходов.
Также проверяются экспорт без постобработки и сопоставление выходов prototxt.
При наличии ONNX Runtime выходы экспортированной модели сравниваются
с выходами PyTorch. Если установлен новый protobuf, несовместимый со старыми
сгенерированными proto-файлами репозитория, перед командами проверки,
обучения и конвертации установите
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

Отказ от конкатенации меняет модель и сам по себе не гарантирует ускорение
или сохранение точности. Обучите модель и оцените mAP, затем измерьте
задержку и пиковое потребление SRAM целевым компилятором на устройстве.
Увеличенные выходы свёрток с шагом 2 и дополнительные проекции также имеют
свою стоимость. Для INT8-квантизации нужна калибровка или QAT с учётом
изменившихся распределений активаций.
