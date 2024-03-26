from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')
# Train the model
results = model.train(data='polyp491.yaml', epochs=100, imgsz=640)
