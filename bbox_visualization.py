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
    visualize_bboxes("./train/1241219023343044_1_Outer.json", "./train/1241219023343044_1_Outer.bmp", output_path=f"1241219023343044_1_Outer.jpg")
    visualize_bboxes("./augmented_non_stabbed/1241219023343044_1_Outer_non_stabbed.json", "./augmented_non_stabbed/1241219023343044_1_Outer_non_stabbed.bmp", output_path="1241219023343044_1_Outer_non_stabbed.jpg")

    
    for i in range(2,10):
        json_path = f"./train_augmented/1241219023343044_1_Outer_non_stabbed_{i}.json"
        image_path = f"./train_augmented/1241219023343044_1_Outer_non_stabbed_{i}.bmp"
        visualize_bboxes(json_path, image_path, output_path=f"1241219023343044_1_Outer_non_stabbed_{i}.jpg")
