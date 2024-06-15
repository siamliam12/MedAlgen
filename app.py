import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import TrainingArguments, Trainer

# Define directories
train_dir = 'chest_xray/train'
test_dir = 'chest_xray/test'
categories = ['NORMAL', 'PNEUMONIA']

def load_images_from_brain_folder(folder):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))  # Resize to 224x224
            images.append(np.array(img))
            labels.append(1 if folder.endswith('yes') else 0)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    return images, labels

def load_images_form_chest_folder(folder,categories):
    images = []
    labels = []
    for category in categories:
        class_num = categories.index(category)
        path = os.path.join(folder,category)
        for img in tqdm(os.listdir(path)):
            try:
                img_path = os.path.join(path, img)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))  # Resize to 224x224
                images.append(np.array(img))
                labels.append(class_num)
            except Exception as e:
                pass
    return images, labels


yes_images, yes_labels = load_images_from_brain_folder('.\\braintumor\\brain_tumor_dataset\yes')
no_images, no_labels = load_images_from_brain_folder('.\\braintumor\\brain_tumor_dataset\\no')

# Combine the data
images = np.array(yes_images + no_images)
labels = np.array(yes_labels + no_labels)

# Normalize the images
images = images / 255.0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load pre-trained model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)

# Preprocess the images
def preprocess_images(images):
    return feature_extractor(images, return_tensors='pt')['pixel_values']

X_train_processed = preprocess_images(X_train)
X_test_processed = preprocess_images(X_test)

#train the model
#define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=(X_train_processed, y_train),
    eval_dataset=(X_test_processed, y_test)
)
trainer.train()

#evaluate the results
eval_result = trainer.evaluate()
print(f"Test accuracy: {eval_result['eval_accuracy']:.2f}")