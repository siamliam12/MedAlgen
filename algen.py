import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Authenticate and create the PyDrive client
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and automatically handles authentication
drive = GoogleDrive(gauth)

# Function to list all files in a folder
def list_files_in_folder(folder_id):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    return file_list

# Function to get direct download link
def get_direct_link(file):
    return f"https://drive.google.com/uc?id={file['id']}"
# Replace with your actual folder IDs
yes_folder_id = '1pj0E5ju9KdbXA3sI3zJBstLcYsrm9l2Q'
no_folder_id = '1Hlqbr6qpxowYCBlUGdm57jS1i_u'

# List and get direct links
yes_files = list_files_in_folder(yes_folder_id)
no_files = list_files_in_folder(no_folder_id)

yes_links = [get_direct_link(file) for file in yes_files]
no_links = [get_direct_link(file) for file in no_files]

class MedicalDataset(Dataset):
    def __init__(self,image_path,labels,transform=None):
        self.image_path = image_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self,idx):
        img_path = self.image_path[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def load_images_from_folder(folder, label):
    image_paths = []
    labels = []
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        try:
            image_paths.append(img_path)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    return image_paths, labels

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

# Define directories
# yes_dir = './yes'
# no_dir = './no'

# # Load images and labels
# yes_image_paths, yes_labels = load_images_from_folder(yes_dir, label=1)
# no_image_paths, no_labels = load_images_from_folder(no_dir, label=0)

# # Combine the data
# all_image_paths = yes_image_paths + no_image_paths 
# all_labels = yes_labels + no_labels 

# # Split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(all_image_paths, all_labels, test_size=0.3, random_state=42)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets
# train_dataset = MedicalDataset(X_train, y_train, transform=transform)
# test_dataset = MedicalDataset(X_test, y_test, transform=transform)

# # Create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# # Training loop
# for epoch in range(10):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
# Training loop (simplified for example)
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for link in yes_links + no_links:
        img = load_image_from_url(link)
        label = 1 if link in yes_links else 0
        img = transform(img).unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)
        
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(yes_links + no_links)}")
# Evaluate the model
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f'Accuracy: {100 * correct / total}%')
# Evaluate the model (simplified)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for link in yes_links + no_links:
        img = load_image_from_url(link)
        label = 1 if link in yes_links else 0
        img = transform(img).unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)
        
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
print(f'Accuracy: {100 * correct / total}%')

# Load the pre-trained model
# generator = pipeline('text-generation', model='distilgpt2')

# Load the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForCausalLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def generate_response(diagnosis):
    prompt = f"The patient has been diagnosed as {diagnosis}. Please give details of the diagnosis and prescribe a personalized treatment plan."
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load the trained model
model.eval()
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
    
st.title("MedAlgen")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    # Make a prediction
    prediction = predict(image)
    diagnosis = "sick" if prediction == 1 else "healthy"
    st.write(f"Diagnosis: {diagnosis}")
    
    # Generate personalized response
    response = generate_response(diagnosis)
    st.write(f"Personalized Response: {response}")