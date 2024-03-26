from ultralytics import YOLO

# Load a model
model = YOLO('polyp491.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='polyp491.yaml', epochs=100, imgsz=640)
