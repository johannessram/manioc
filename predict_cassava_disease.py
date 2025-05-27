from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn



class_names = ['Cassava__bacterial_blight', 'Cassava__brown_streak_disease', 'Cassava__green_mottle', 'Cassava__healthy', 'Cassava__mosaic_disease']

def load_model(path='cassava_model.pth'):
    # load model saved earlier 
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(path))
    # model = model.to(device)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Taille compatible avec ResNet
    transforms.ToTensor(),
])


def predict_image(image_path, model, transform, class_names=class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    print(f"Prediction: {class_names[predicted.item()]}")
    return str(class_names[predicted.item()])

# model = load_model()
# predict_image('test/test.jpeg')
