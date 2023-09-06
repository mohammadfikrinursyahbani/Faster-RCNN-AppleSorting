import streamlit as st
import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
import torchvision.transforms as T

class_labels = ["Background", "apple", "damaged_apple"]

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = len(class_labels)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('model/Last-epoch-Model49.pth', map_location=torch.device('cpu')))
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_objects(image):
    # Preprocess the image
    image = preprocess(image).unsqueeze(0)
    
    # Perform object detection
    with torch.no_grad():
        prediction = model(image)
    
    # Get the bounding boxes, labels, and scores
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    
    # Create a copy of the image to draw bounding boxes on
    drawn_image = image.squeeze(0).permute(1, 2, 0).numpy()
    drawn_image = (drawn_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    drawn_image = (drawn_image * 255).astype('uint8')
    drawn_image = Image.fromarray(drawn_image)
    draw = ImageDraw.Draw(drawn_image)
    
    # Iterate over detected objects
    for box, label, score in zip(boxes, labels, scores):
        if score >= 0.5:  # You can adjust the confidence threshold here
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="blue", width=2)
            label_text = f"Label: {label.item()}, Confidence: {int(score.item() * 100)}%"
            draw.text((box[0], box[1]), label_text, fill="green")
            
            # Print label and confidence to Streamlit
            st.write(label_text)
    
    return drawn_image

# Streamlit app
st.title("Object Detection with Faster R-CNN")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Objects"):
        # Perform object detection
        detected_image = detect_objects(image)
        
        # Display the detected image with bounding boxes
        st.image(detected_image, caption="Detected Objects", use_column_width=True)