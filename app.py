import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the complete model
@st.cache_data()
def load_model(num_classes):
    model = Net(num_classes)
    # model = torch.load('./models/medical_net.pth', map_location=torch.device('cpu'))
    model.load_state_dict(torch.load('./models/medical_net.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_huggingface_model():
    model_name = "facebook/bart-large-cnn"  # Using a summarization model for better generation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model
# Prediction function
def predict(image, model,num_classes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        diagnosis = classes[predicted.item()]
    return diagnosis

# Generate personalized response using Hugging Face model
def generate_response(diagnosis, tokenizer, model):
    input_text = f"The patient is diagnosed with {diagnosis}. Provide medical advice."
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("MedAlgen")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Load models
    num_classes = 3  # Number of classes: breast_cancer, pneumonia, healthy
    classes = ['Brain Tumor', 'pneumonia', 'healthy']
    pytorch_model = load_model(num_classes)
    tokenizer, huggingface_model = load_huggingface_model()
    
    # Make a prediction
    diagnosis = predict(image, pytorch_model, classes)
    st.write(f"Diagnosis: {diagnosis}")
    
    # Generate personalized response
    response = generate_response(diagnosis, tokenizer, huggingface_model)
    st.write(f"Personalized Response: {response}")
