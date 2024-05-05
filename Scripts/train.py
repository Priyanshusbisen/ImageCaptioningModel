from utils import save_checkpoint, load_checkpoint
from torchvision import transforms
from dataloader import get_loader
from model import Seq2Seq
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
RESIZE = 356
CROP = 299

def train():

    # Train the model
    batch_size=64
    
    transform = transforms.Compose(
        [
            transforms.Resize((RESIZE, RESIZE)),
            transforms.RandomCrop((CROP, CROP)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_loader, dataset = get_loader(
        './Data/Flickr_Data/Flickr_Data/Images',
        './Data/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt',
        transform=transform,
        num_workers=1,
        batch_size=batch_size,
        shuffle=True,
        isTrain=True
    )

    # Hyperparameters
    embed_dim = 256
    hidden_layers = 1
    decoder_dim = 256
    vocab_size = len(dataset.vocab)
    learning_rate = 4e-04
    num_epochs = 10
    load_model = False
    save_model = True

    step = 0

    # initialize model, loss etc
    model = Seq2Seq(embed_dim, vocab_size, decoder_dim, hidden_layers ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    if load_model:
        #step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
        step = load_checkpoint(torch.load("./models/my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)
            torch.save(model.state_dict(), 'puremodel.pth.tar')
        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
        
            optimizer.step()
        
        print('Epoch {} completed with loss {}'.format(epoch+1, loss))

if __name__ == '__main__':
    train()