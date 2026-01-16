import onnxruntime as ort
import numpy as np
import cv2
import json


class OnnxModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def inference(self, input_tensor):
        return self.session.run(None, {'images': input_tensor})


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Ошибка при открытии видеоисточника")
        
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def preprocess(frame, target_size=(640, 640)):
        """Препроцессинг изображения для YOLO"""
        image_resized = cv2.resize(frame, target_size)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(image_rgb, (2, 0, 1))  
        tensor = np.expand_dims(tensor, axis=0)  
        tensor = tensor.astype(np.float32) / 255.0
        return tensor
    
    @staticmethod
    def postprocess(results, frame, frame_num, conf_threshold=0.25):
        """
        Постпроцессинг с парсингом в JSON
        """
        detections = results[0] 
        annotated_frame = frame.copy()
        h_orig, w_orig = frame.shape[:2]
        
        scale_x = w_orig / 640
        scale_y = h_orig / 640
        
        # Структура для JSON
        frame_data = {
            'num_frame': frame_num,
            'score': [],
            'boxes': []
        }
        
        for detection in detections[0]:
            if len(detection) < 6 or detection[5] == -1:
                continue
                
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            
            if confidence < conf_threshold:
                continue
            
            # Масштабирование координат к оригинальному размеру
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            class_id = int(class_id)
            
            # Добавление в JSON структуру
            frame_data['score'].append(float(confidence))
            frame_data['boxes'].append([float(x1), float(y1), float(x2), float(y2)])
            
            # Рисование на кадре
            color = (0, 255, 0) 
            cv2.rectangle(annotated_frame, (x1_scaled, y1_scaled), 
                         (x2_scaled, y2_scaled), color, 2)
            
            # Добавление текста
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1_scaled, y1_scaled - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
        return annotated_frame, frame_data


def main(video_path, model_path, output_json='detections.json'):
    video_proc = VideoProcessor(video_path)
    yolo_model = OnnxModel(model_path)
    
    frame_count = 0
    all_detections = []
    
    while True:
        frame = video_proc.read_frame()
        if frame is None:
            print("Видео завершено")
            break

        frame_count += 1
        
        preprocessed = video_proc.preprocess(frame)
        results = yolo_model.inference(preprocessed)
        
        annotated_frame, frame_data = video_proc.postprocess(
            results, frame, frame_count, conf_threshold=0.25
        )
        
        # Добавляем данные кадра в список
        all_detections.append(frame_data)
        
        # Показываем количество детекций на текущем кадре
        num_detections = len(frame_data['score'])
        cv2.putText(annotated_frame, f"Frame: {frame_count} | Detections: {num_detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_proc.release()
    
    # Сохранение результатов в JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)
    
    print(f"\nОбработано кадров: {frame_count}")
    print(f"Результаты сохранены в: {output_json}")
    
    # Вывод статистики
    total_detections = sum(len(fd['score']) for fd in all_detections)
    print(f"Всего детекций: {total_detections}")
    if all_detections:
        avg_detections = total_detections / len(all_detections)
        print(f"Среднее количество детекций на кадр: {avg_detections:.2f}")


if __name__ == "__main__":
    main(
        video_path='/home/kira9k/ieos/test_human_datasets/videos/mp4/Georg_0.mp4',
        model_path='/home/kira9k/ieos/mmyolo/work_dirs/onnx_export/yolov8_s_mmyolo.onnx',
        output_json='detections.json'
    )
