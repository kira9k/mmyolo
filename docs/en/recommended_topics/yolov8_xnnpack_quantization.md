# Квантование пруненой YOLOv8 Focus + Add через XNNPACK PT2E

Скрипт `projects/easydeploy/tools/quantize.py` выполняет статическое PTQ:
загружает обученные веса, применяет model surgery 2, собирает статистику
на изображениях и сохраняет графы `.pt2` и `.jit.pt` с помощью
`XNNPACKQuantizer`. Это не артефакты TIDL и не скомпилированный делегат
ExecuTorch XNNPACK. Само сохранение PT2E-графа не гарантирует ускорения
в обычном PyTorch: скорость зависит от среды исполнения и lowering.

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

Нужен сохранённый конфиг именно пруненой модели с `model.pruning_structure`.
Исходный конфиг непрореженной архитектуры не подходит. Веса загружаются
строго, чтобы несовпадение архитектуры не осталось незамеченным.
Surgery 2 применяется скриптом автоматически, отдельный флаг не требуется.

Укажите существующий каталог изображений в `--calib-dir`. Желательно брать
репрезентативную выборку из обучающих данных, покрывающую размеры объектов,
яркость и условия реального применения. Не используйте итоговую тестовую
выборку для подбора калибровки. Список выбранных файлов сохраняется в отчёте.

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

Каждый вариант калибруется отдельно на одинаковой выбранной выборке.
Разложение выполняется до PT2E-калибровки; изменять пулинги в уже
квантованном графе вместо повторной калибровки не следует. Эта опция
работает и с конфигами `focus_add_last_spp_concat_surgery2`.

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
сохранение форм выходов и отсутствие NaN/Inf. Это не заменяет оценку
точности: перед использованием сравните FP32 и INT8 на одинаковой
валидационной выборке с одинаковыми декодированием, NMS и порогами.

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

## ONNX с пулингами 3 x 3

Чтобы явно заменить каждый MaxPool 5 x 5 в SPPF двумя последовательными
MaxPool 3 x 3, добавьте в ONNX-экспорт флаг `--maxpool-3x3`:

```bash
OUT=work_dirs/yolov8_s_focus_add_spp_concat_surgery2_pruned_p30_r16_surgery2

python projects/easydeploy/tools/export_onnx.py \
  "$OUT/finetune_100e_lr001/model.py" \
  "$OUT/finetune_100e_lr001/best_coco_bbox_mAP_epoch_98.pth" \
  --model-surgery 2 --maxpool-3x3 \
  --model-only --raw-output-format combined \
  --img-size 640 640 --device cpu --opset 11 --simplify \
  --work-dir "$OUT/onnx_model_only_pool3x3_surgery2"
```

Флаг работает и при экспорте с NMS: вместо `--model-only
--raw-output-format combined` укажите `--export-type YOLOv5` и нужные
пороги NMS. Используйте другой `--work-dir`, чтобы сохранить предыдущие
экспорты рядом.

Замена выполняется до model surgery и не зависит от версии
`edgeai_torchmodelopt`. Она затрагивает только MaxPool с ядром 5 x 5,
stride 1, padding 2, dilation 1, без ceil mode и возврата индексов.
Три вызова пулинга в SPPF становятся шестью вызовами 3 x 3, формы выходов
и FP32-результаты сохраняются. Без флага поведение экспорта не меняется.
