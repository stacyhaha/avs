
from torch.utils.data import DataLoader
import torch
from data import VehicleDataset
from model import DrivingModel
from torchvision import datasets, transforms
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def test(model, image:Image):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
])

    image_tensor = transform(image)
    model.eval()
    with torch.no_grad():
        image_tensor = torch.unsqueeze(image_tensor.to(device), dim=0)
        outputs = model(image_tensor)
    return outputs[0].tolist()

if __name__ == "__main__":
    image_path = "collect_data/collect_images/129.9261646270752.jpg"
    txt_path = "collect_data/collect_images/1.33597993850708.txt"
    checkpoint_path = "model.pth"
    image = Image.open(image_path)
    image.show()
    

    model = DrivingModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    res = test(model, image)
    print(res)


    import os 
    images = os.listdir("collect_data/collect_images")
    images = sorted(list(filter(lambda x:x.endswith("jpg"), images)))
    for i in images:
        image_path = os.path.join("collect_data/collect_images", i)
        image = Image.open(image_path)
        image.show()

        res = test(model, image)
        print("predict: ")
        print(res)
        print()
        print("true label")
        with open(os.path.join("collect_data/collect_images", i[:-4]+".txt"), "r") as f:
            print(f.read())
        import pdb;pdb.set_trace()
