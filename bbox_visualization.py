import cv2
import json

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_bboxes(json_path, image_path, output_path="debu.jpg"):
    # JSON 데이터와 이미지를 불러옵니다.
    data = load_json(json_path)
    image = cv2.imread(image_path)
    shapes = data["shapes"]
    if len(shapes) == 0:
        cv2.imwrite(output_path, image)

    for shape in shapes:
        bbox = shape.get("bbox")
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])
        label = shape.get("label", "N/A")
        
    
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    json_path = f"./train/1241218144156467_1_Outer.json"
    image_path = f"./train/1241218144156467_1_Outer.bmp"
    visualize_bboxes(json_path, image_path, output_path=f"11241218144156467_1_Outer.jpg")
    
    for i in range(0, 5):
        json_path = f"./total_dataset/train_dataset/1241218144156467_1_Outer_patch_{i}.json"
        image_path = f"./total_dataset/train_dataset/1241218144156467_1_Outer_patch_{i}.bmp"
        visualize_bboxes(json_path, image_path, output_path=f"11241218144156467_1_Outer_patch_{i}.jpg")
