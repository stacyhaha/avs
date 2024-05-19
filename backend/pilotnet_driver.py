import torch
from pilotnet.model import DrivingModel
from torchvision import transforms
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PilotNetDriver():
    def __init__(self, model_path):
        self.model_path = model_path 
        self.model = DrivingModel()
        checkpoint = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([66, 200]),
        ])
        self.model.eval()


    def inference(self, image:Image):
        image_tensor = self.transform(image)
        with torch.no_grad():
            image_tensor = torch.unsqueeze(image_tensor.to(device), dim=0)
            outputs = self.model(image_tensor)
        return outputs[0].tolist()        
