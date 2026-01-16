# merge_coco_30percent.py
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

random.seed(42)  # для воспроизводимости

def merge_with_30percent(
    dataset1_dir,           # data/train_val_2/          ← старый, берём 100%
    dataset1_train_ann,
    dataset1_val_ann,

    dataset2_dir,           # data/train_val_fine-tune/  ← новый, берём только 30%
    dataset2_train_ann,
    dataset2_val_ann,

    output_dir="data/combined_30percent_2025_v2",
    prefix2="small_",       # префикс при конфликте имён
    keep_ratio=0.30         # ← сколько брать из нового датасета
):
    output_dir = Path(output_dir)
    (output_dir / "train/images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val/images").mkdir(parents=True, exist_ok=True)

    def load_ann(p): return json.load(open(p))
    def save_ann(data, p): json.dump(data, open(p, 'w'), indent=2)

    # ==================== TRAIN ====================
    print(f"Объединяем TRAIN (берём {keep_ratio*100:.0f}% из нового датасета)...")
    ann1 = load_ann(Path(dataset1_dir) / dataset1_train_ann)
    ann2_full = load_ann(Path(dataset2_dir) / dataset2_train_ann)

    # --- Берём только 30% случайных изображений из нового датасета ---
    num_to_keep = max(1, int(len(ann2_full['images']) * keep_ratio))
    selected_images = random.sample(ann2_full['images'], num_to_keep)
    selected_img_ids = {img['id'] for img in selected_images}
    
    ann2 = {
        "images": selected_images,
        "annotations": [a for a in ann2_full['annotations'] if a['image_id'] in selected_img_ids],
        "categories": ann2_full['categories']
    }
    print(f"Из нового датасета взяли {len(ann2['images'])} / {len(ann2_full['images'])} изображений")

    # Защита от коллизий имён
    existing_names = {img['file_name'] for img in ann1['images']}
    for img in ann2['images']:
        if img['file_name'] in existing_names:
            img['file_name'] = f"{prefix2}{img['file_name']}"

    # Перебиваем ID
    max_img_id = max(img['id'] for img in ann1['images'] + [dict(id=0)])
    max_ann_id = max(a['id'] for a in ann1['annotations'] + [dict(id=0)])

    for img in ann2['images']:
        img['id'] += max_img_id + 1
    for ann in ann2['annotations']:
        ann['image_id'] += max_img_id + 1
        ann['id'] += max_ann_id + 1

    # Объединяем
    combined_train = {
        "images": ann1['images'] + ann2['images'],
        "annotations": ann1['annotations'] + ann2['annotations'],
        "categories": ann1['categories'],
    }

    # Копируем все картинки
    for img_info in tqdm(combined_train['images'], desc="копируем train"):
        if img_info['id'] <= max_img_id:
            src = Path(dataset1_dir) / "train/images" / img_info['file_name']
            if not src.exists():
                src = Path(dataset1_dir) / "train" / img_info['file_name']
        else:
            old_name = img_info['file_name'].replace(prefix2, "", 1) if img_info['file_name'].startswith(prefix2) else img_info['file_name']
            src = Path(dataset2_dir) / "train/images" / old_name
            if not src.exists():
                src = Path(dataset2_dir) / "train" / old_name

        dst = output_dir / "train/images" / img_info['file_name']
        if src.exists():
            shutil.copy2(src, dst)

    save_ann(combined_train, output_dir / "train/ann.json")

    # ==================== VAL (аналогично 30%) ====================
    print(f"Объединяем VAL (тоже {keep_ratio*100:.0f}%)...")
    ann1v = load_ann(Path(dataset1_dir) / dataset1_val_ann)
    ann2v_full = load_ann(Path(dataset2_dir) / dataset2_val_ann)

    num_to_keep_v = max(1, int(len(ann2v_full['images']) * keep_ratio))
    selected_val_images = random.sample(ann2v_full['images'], num_to_keep_v)
    selected_val_ids = {img['id'] for img in selected_val_images}

    ann2v = {
        "images": selected_val_images,
        "annotations": [a for a in ann2v_full['annotations'] if a['image_id'] in selected_val_ids],
        "categories": ann2v_full['categories']
    }

    existing_val_names = {img['file_name'] for img in ann1v['images']}
    for img in ann2v['images']:
        if img['file_name'] in existing_val_names:
            img['file_name'] = f"{prefix2}{img['file_name']}"

    max_img_id_v = max(img['id'] for img in ann1v['images'] + [dict(id=0)])
    max_ann_id_v = max(a['id'] for a in ann1v['annotations'] + [dict(id=0)])

    for img in ann2v['images']:
        img['id'] += max_img_id_v + 1
    for ann in ann2v['annotations']:
        ann['image_id'] += max_img_id_v + 1
        ann['id'] += max_ann_id_v + 1

    combined_val = {
        "images": ann1v['images'] + ann2v['images'],
        "annotations": ann1v['annotations'] + ann2v['annotations'],
        "categories": ann1v['categories'],
    }

    for img_info in tqdm(combined_val['images'], desc="копируем val"):
        if img_info['id'] <= max_img_id_v:
            src = Path(dataset1_dir) / "vid16_test/images" / img_info['file_name']
            if not src.exists():
                src = Path(dataset1_dir) / "val/images" / img_info['file_name']
        else:
            old_name = img_info['file_name'].replace(prefix2, "", 1) if img_info['file_name'].startswith(prefix2) else img_info['file_name']
            src = Path(dataset2_dir) / "val/images" / old_name
            if not src.exists():
                src = Path(dataset2_dir) / "val" / old_name

        dst = output_dir / "val/images" / img_info['file_name']
        if src.exists():
            shutil.copy2(src, dst)

    save_ann(combined_val, output_dir / "val/ann.json")
    print(f"ГОТОВО! Датасет → {output_dir}")
    print(f"   train: {len(ann1['images'])} + {len(ann2['images'])} = {len(combined_train['images'])}")
    print(f"   val:   {len(ann1v['images'])} + {len(ann2v['images'])} = {len(combined_val['images'])}")

# ======================= ЗАПУСК =======================
if __name__ == "__main__":
    merge_with_30percent(
        dataset2_dir="data/train_val_2/",
        dataset2_train_ann="train/ann.json",
        dataset2_val_ann="val/ann.json",

        dataset1_dir="data/train_val_small_v2/",
        dataset1_train_ann="train/ann.json",
        dataset1_val_ann="val/ann.json",

        output_dir="data/data/combined_100_percent_2025_v2",
        prefix2="old_",
        keep_ratio=1   # ← можно поставить 0.2, 0.4 и т.д.
    )