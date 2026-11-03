# Квантование пруненой YOLOv8 Focus + Add через XNNPACK PT2E

Скрипт `projects/easydeploy/tools/quantize.py` выполняет статическое PTQ:
загружает обученные веса, применяет model surgery 2, собирает статистику
на изображениях и сохраняет графы `.pt2` и `.jit.pt` с помощью
`XNNPACKQuantizer`. 

## Команда для p20_r16 после дообучения

Из корня `/mmyolo` в контейнере:

```bash
OUT=work_dirs/yolov8_s_focus_add_spp_concat_surgery2_pruned_p20_r16_surgery2

python projects/easydeploy/tools/quantize.py \
  --config "$OUT/finetune_100e_lr001/model.py" \
  --checkpoint "$OUT/finetune_100e_lr001/best_coco_bbox_mAP_epoch_97.pth" \
  --calib-dir data/small_people_v3/train/images \
  --num-calib-images 300 \
  --img-size 640 640 \
  --raw-output-format combined \
  --duplicate-cls-channels 1 \
  --work-dir "$OUT/pt_xnnpack"
```

Укажите существующий каталог изображений в `--calib-dir`. Желательно брать
репрезентативную выборку из обучающих данных, покрывающую размеры объектов,
яркость и условия реального применения.

## Калибровка и формат

- Изображения загружаются через OpenCV в grayscale.
- Геометрия: `YOLOv5KeepRatioResize` и `LetterResize`, как в MMYOLO,
  без `CenterCrop`; заполнение по умолчанию 114.
- Нормализация берётся из `data_preprocessor` сохранённого конфига:
  для этих моделей это деление на 255 ровно один раз.
- Затем применяется `pixel_unshuffle(4)`. Вход графа при размере 640 x 640:
  `float32 [1, 16, 160, 160]`. Упаковка и letterbox выполняются вне графа.
- Калибровка идёт с batch 1, без отбрасывания последнего батча.
- По умолчанию три сырых выхода: `[1, 65, 80, 80]`,
  `[1, 65, 40, 40]`, `[1, 65, 20, 20]` для модели с одним классом.
  Порядок каналов: cls logits, затем 64 канала DFL-регрессии.
  Выходы float32 после деквантования, декодирование и NMS остаются снаружи.

`--raw-output-format separate` включает раздельные ветки cls/bbox.
`--duplicate-cls-channels 16` включает копирование единственного cls-фильтра
в 16 каналов. Для обычного квантования эти опции не нужны: значения
`combined` и `1` сохраняют три объединённых выхода без дублирования.

## Выбор свёрток для квантования

По умолчанию квантование применяется ко всем поддерживаемым паттернам.
Для сохранения последних предсказывающих свёрток в FP32 добавьте:

```bash
--keep-last-convs-fp32
```

Для трёх масштабов это шесть последних Conv2d: три cls и три reg.
Остальные свёртки головы остаются кандидатами на INT8-квантование.
У исключённых свёрток сохраняются FP32-веса; их входы могут быть
деквантованы из предшествующих INT8-участков. Финальный Concat сырых
выходов не квантует значения защищённых предсказаний повторно.

Можно выбирать отдельные свёртки или целые блоки по именам:

```bash
# Только последние cls-свёртки, без исключения reg.
--exclude-modules 'bbox_head.head_module.cls_preds.*.2'

# Вся голова детекции.
--exclude-modules bbox_head

# Несколько блоков; опцию можно повторять.
--exclude-modules backbone.stage1 neck.top_down_layers.0
```

Поддерживаются shell-style шаблоны (`*`, `?`, `[0-9]`), которые нужно
заключать в кавычки от раскрытия оболочкой. Имя блока включает его
дочерние свёртки. Это выбор свёрток; распределение вспомогательных
операторов внутри FX-блоков по точности определяет PT2E.
Неизвестный шаблон или модуль без Conv2d вызывают ошибку,
а не игнорируются. Опция не предназначена для выбора отдельно взятого
BatchNorm после слияния Conv+BN.

Чтобы получить точные имена после surgery, без калибровки и сохранения:

```bash
python projects/easydeploy/tools/quantize.py \
  --config "$OUT/model.py" --checkpoint "$OUT/model.pth" \
  --list-conv-modules
```

Пример для обученной модели с входом 640 x 480:

```bash
OUT=work_dirs/yolov8_s_focus_add_last_spp_concat_surgery2

python projects/easydeploy/tools/quantize.py \
  --config "$OUT/yolov8_s_focus_add_last_spp_concat_surgery2.py" \
  --checkpoint "$OUT/best_coco_bbox_mAP_epoch_500.pth" \
  --calib-dir data/small_people_v3/train/images \
  --num-calib-images 300 --img-size 480 640 \
  --maxpool-kernel-size 5 --keep-last-convs-fp32 \
  --work-dir "$OUT/pt_xnnpack_480x640"
```

В имени результатов появится `_fp32last`. Для произвольного набора
исключений добавляется `_fp32sel_<hash>`, чтобы не перезаписать полное
квантование или другую политику. JSON содержит `fp32_modules`,
`fp32_convolutions` и фактическую `convolution_precisions` каждого
оператора Conv2d: имя модуля, признак квантования весов и тип весов.
Перед сохранением проверяется, что выбранные свёртки действительно
остались FP32.

## Выбор пулингов при квантовании

- `--maxpool-kernel-size 5` (или `--maxpool-5x5`) сохраняет три исходных
  MaxPool 5 x 5 в SPPF, отключая их автоматическое разложение в surgery.
- `--maxpool-kernel-size 3` (или `--maxpool-3x3`) заменяет их шестью
  последовательными MaxPool 3 x 3. Ветви SPPF и один Concat сохраняются.
- Без этих опций остаётся прежнее поведение установленной версии surgery.
  Для воспроизводимого выбора всегда указывайте размер явно.

Пример получения обоих вариантов p30_r16 в одном каталоге:

```bash
OUT=work_dirs/yolov8_s_focus_add_spp_concat_surgery2_pruned_p30_r16_surgery2

for K in 5 3; do
  python projects/easydeploy/tools/quantize.py \
    --config "$OUT/finetune_100e_lr001/model.py" \
    --checkpoint "$OUT/finetune_100e_lr001/best_coco_bbox_mAP_epoch_98.pth" \
    --calib-dir data/small_people_v3/train/images \
    --num-calib-images 300 --seed 0 \
    --raw-output-format combined --duplicate-cls-channels 1 \
    --maxpool-kernel-size "$K" \
    --work-dir "$OUT/pt_xnnpack"
done
```

В именах файлов будут суффиксы `_pool5x5` и `_pool3x3` перед расширением
`.pt2`, `.jit.pt` и `.json`, поэтому результаты не перезапишут друг друга.
В JSON записываются запрошенный `maxpool_kernel_size` и фактический список
`maxpool_kernels` из квантованного графа. Формы входа и выходов не меняются.

## Заменить Add на Concat + Conv без переобучения

В `quantize.py` и `export_onnx.py` доступен флаг
`--replace-add-with-concat-conv`. Он заменяет сложения двух одинаковых
по форме floating-point тензоров NCHW на Concat по каналам и плотную
Conv 1 x 1 без bias, активации и нормализации:

```text
Add(a, b) = Conv1x1(Concat(a, b), W=[I, I])
```

Для C каналов Concat даёт 2C каналов, а Conv возвращает C:
`out[c] = a[c] + b[c]`. Веса фиксированы и не обучаются. В PT2E также
сохраняется коэффициент `alpha` у исходного Add, если он есть.
Математически результат тот же.

Для квантования добавьте к своей команде:

```bash
--replace-add-with-concat-conv
```

Пример ONNX model-only для 640 x 480:

```bash
OUT=work_dirs/yolov8_s_focus_add_last_spp_concat_surgery2_480
python projects/easydeploy/tools/export_onnx.py \
  "$OUT/yolov8_s_focus_add_last_spp_concat_surgery2.py" \
  "$OUT/best_coco_bbox_mAP_epoch_497.pth" \
  --model-only --model-surgery 2 --img-size 480 640 \
  --replace-add-with-concat-conv \
  --opset 11 --device cpu --work-dir "$OUT/onnx_addconcatconv"
```

Флаг также работает с NMS: вместо `--model-only` используйте
`--export-type YOLOv5`. Заменяются только подходящие 4D-сложения признаков,
а скалярные, broadcast-сложения и арифметика декодирования других рангов
сохраняются. Формы и имена выходов не меняются. ONNX-замена выполняется
после simplify, чтобы именно Concat + Conv присутствовали в сохраняемом
файле; сторонний компилятор может снова оптимизировать этот шаблон.

Файлы получают суффикс `_addconcatconv` (перед суффиксами пулингов и
FP32-политики). Список замен записывается в JSON квантования и metadata
ONNX как `add_concat_conv_replacements`. Для пруненных моделей используйте
соответствующие `model.py` и дообученные веса.

## Результаты и проверка

В каталоге `pt_xnnpack` появятся:

```text
best_coco_bbox_mAP_epoch_97_int8_combined.pt2
best_coco_bbox_mAP_epoch_97_int8_combined.jit.pt
best_coco_bbox_mAP_epoch_97_int8_combined.json
```

В `.pt2` и `.jit.pt` встроен `preprocess.json`, а отдельный JSON содержит
формы входа и выходов, параметры и файлы калибровки, версию PyTorch и число
операторов квантования. Скрипт проверяет наличие операторов квантования,
сохранение форм выходов и отсутствие NaN/Inf. 

Существующие результаты не перезаписываются без `--force`. Для
воспроизводимой выборки используйте `--seed` (по умолчанию 0).

Поддерживается PT2E из `torchao`, а при его отсутствии используется
`torch.ao` из установленного PyTorch 2.7. Для имеющегося контейнера
с PyTorch 2.7.1 установка `torchao` не требуется. При переносе `.pt2`
между средами предпочтительна та же версия PyTorch, что при сохранении.

Если контейнер при загрузке пруненой модели выдаёт ошибку protobuf
`Descriptors cannot be created directly`, перед запуском установите:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Это обход несовместимости существующего сгенерированного `*_pb2.py`
с установленной версией protobuf; он не меняет вычисления модели.

## Выбор пулингов при экспорте ONNX

В `export_onnx.py` доступны те же варианты:

- `--maxpool-kernel-size 5` или `--maxpool-5x5`: сохранить три MaxPool 5 x 5.
- `--maxpool-kernel-size 3` или `--maxpool-3x3`: получить шесть MaxPool 3 x 3.

Явный выбор отключает автоматическое разложение пулингов в surgery,
но сохраняет остальные преобразования. Без выбора размера применяется
прежнее поведение установленной версии surgery: отсутствие флага
`--maxpool-3x3` само по себе не гарантирует сохранения 5 x 5.

Пример для изображения шириной 640 и высотой 480, с surgery 2:

```bash
OUT=work_dirs/yolov8_s_focus_add_last_spp_concat_surgery2

python projects/easydeploy/tools/export_onnx.py \
  "$OUT/yolov8_s_focus_add_last_spp_concat_surgery2.py" \
  "$OUT/best_coco_bbox_mAP_epoch_500.pth" \
  --model-surgery 2 --maxpool-kernel-size 5 \
  --model-only --raw-output-format combined \
  --img-size 480 640 --device cpu --opset 11 --simplify \
  --work-dir "$OUT/onnx_480x640"
```

Для варианта 3 x 3 замените значение `5` на `3`. Файлы `.onnx` и
`.prototxt` получают суффикс `_pool5x5` или `_pool3x3` перед расширением,
поэтому оба варианта можно экспортировать в один каталог.
Для указанной модели вход имеет форму `(1,16,120,160)`, а сырые выходы:
`(1,65,60,80)`, `(1,65,30,40)`, `(1,65,15,20)`.

