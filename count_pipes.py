import argparse
from ultralytics import YOLO
import cv2

def count_objects(model_path, image_path):
    model = YOLO(model_path)

    results = model(image_path)

    num_objects = len(results[0].boxes)

    image_cv = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Pipes count: {num_objects}"
    font_scale = 0.9
    thickness = 2
    color = (255, 0, 0)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = image_cv.shape[1] - text_size[0] - 10
    text_y = image_cv.shape[0] - 10
    cv2.putText(image_cv, text, (text_x, text_y), font, font_scale, color, thickness)

    cv2.imshow("Detected pipes", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count pipes in an image using a YOLO model.")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to the YOLO model file.")
    parser.add_argument("--image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    count_objects(args.model_path, args.image_path)
