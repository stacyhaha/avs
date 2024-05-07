import os
from itertools import islice
from torch import nn
import torch
from torch.serialization import save
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import sys
from model import DrivingModel
from data import VehicleDataset
from transformers import get_scheduler
from functools import reduce
import logging
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from test import test



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def should_resume():
    return "--resume" in sys.argv or "-r" in sys.argv


def save_checkpoint(model: DrivingModel, optimizer: optim.Optimizer):
    torch.save(model.state_dict(), CHECKPOINT_PATH)


def load_checkpoint(model: DrivingModel, optimizer: optim.Optimizer = None):
    checkpoint = torch.load(CHECKPOINT_PATH)

    model.load_state_dict(checkpoint)


dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
    transforms.RandomCrop(size=(66,200))
])

def train(model: DrivingModel, optimizer: optim.Optimizer):
    quit = False

    def on_quit():
        nonlocal quit
        quit = True


    criterion = nn.MSELoss()
    
    data_dirs = os.listdir(DATA_PATH)
    data_dirs = list(filter(lambda x: os.path.isdir(os.path.join(DATA_PATH, x)), data_dirs))
    dataset = []
    for i in data_dirs:
        logger.info(f"load dataset {i}")
        dataset.append(VehicleDataset(os.path.join(DATA_PATH, i), dataset_transforms))
    dataset = reduce(lambda x,y: x+y, dataset)
    logger.info(f"full dataset {len(dataset)}")
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=EPOCH*len(train_loader))
    
    logger.info("****** start training *******")
    model = model.to(device)

    writer = SummaryWriter()
    global_step = 0
    for epoch in range(EPOCH):


        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            if quit:
                save_checkpoint(model, optimizer)
                return
            
            model.train()
            # get the inputs; data is a list of [inputs, labels]
            image, command_label = data
            image = image.to(device)
            command_label = command_label.to(device)
   
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            loss = criterion(outputs, command_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            # print statistics
            running_loss += loss.item()
            
        print('epoch [%d] loss: %.3f' % (epoch + 1, running_loss / (len(train_loader))))
        writer.add_scalar("train_loss", running_loss / len(train_loader),  epoch)
        current_lr = lr_scheduler.get_last_lr()[0]
        writer.add_scalar("learning_rate", current_lr, epoch)
        running_loss = 0.0

        save_checkpoint(model, optimizer)
        image_path = "/home/featurize/data/collect_data/collect_images/1.33597993850708.jpg"
        txt_path = "/home/featurize/data/collect_data/collect_images/1.33597993850708.txt"
        image = Image.open(image_path)
        pred = test(model, image)
        writer.add_text("pred_command", str(pred), global_step)
        writer.add_text("true_command", str([160, 90, 0]), global_step)

    print('Finished Training')


if __name__ == "__main__":
    CHECKPOINT_PATH = "model.pth"
    DATA_PATH = "/home/featurize/data/collect_data"
    EPOCH = 1200
    clip_value = 1.0  # 梯度裁剪阈值


    model = DrivingModel()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    if should_resume():
        load_checkpoint(model, optimizer)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train(True)

    train(model, optimizer)