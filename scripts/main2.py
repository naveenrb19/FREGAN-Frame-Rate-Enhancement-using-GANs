import torch
import torch.nn as nn
from label_smoothing import smooth_negative_labels,smooth_positive_labels
import os
from Disc import Discriminator
from Gen import Generator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import torchvision
import torch.optim as optim
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import transforms
transform_norm = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(0,1)])
def save_checkpoint(epochstate,model, optimizer,loss,frame_number,scaler,filename):
    print("----------------------------------------------------Saving checkpoint----------------------------------------------------")
    checkpoint={
            'epoch':epochstate,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':loss,
            'frame_number':frame_number,
            'scaler_dict':scaler.state_dict()
            }
    torch.save(checkpoint, filename)


def save_mdoel(model,path):
    torch.save(model,path)
def load_model(path):
    return torch.load(path)
gen_p='./Models/gen1.pth'
disc_p='./Models/disc1.pth'
def load_checkpoint(checkpoint_file, model, optimizer, lr,scaler):
    print("----> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch=checkpoint['epoch']
    loss=checkpoint['loss']
    frame=checkpoint['frame_number']
    scaler.load_state_dict(checkpoint['scaler_dict'])
    # print(f'Model loaded: {model}')
    return epoch,loss,frame 

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.BatchNorm2d,nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data,0.0,0.2)

l=0
epoch=0
num_epochs=500
lera=0.0001
gen=Generator(6,64).to(device)
disc=Discriminator(9,16).to(device)
opt_gen=optim.Adam(gen.parameters(),lr=lera,betas=(0,0.95))
opt_disc=optim.Adam(disc.parameters(),lr=lera,betas=(0,0.95)) 
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
genckptsave=str()
discckptsave=str()
genckptload=str()
discckptload=str()
genmodelsave=str()
discmodelsave=str()
genmodelload=str()
discmodelload=str()
# initialize_weights(gen)
# initialize_weights(disc)
from torch.utils.tensorboard import SummaryWriter
train_mode=int(input('Enter training mode: (0:continue from check_point) \n (1:Start fresh)'))
if train_mode==0:
    genmodelload=input('Enter generator model name to load (gen.pth): ')
    discmodelload=input('Enter discriminator model name to load (disc.pth): ')
    if os.path.exists(f'./Models/{genmodelload}') and os.path.exists(f'./Models/{discmodelload}'):
        gen=torch.load('./Models/gen1.pth').to(device)
        print('Loaded Generator')
        disc=torch.load('./Models/disc1.pth').to(device)
        print('Loaded Discriminator')
    
    genckptload=input('Enter generator checkpoint name to load (genepoch0.pth): ')
    discckptload=input('Enter discriminator checkpoint name to load (discepoch0.pth): ')
    for i in reversed(range(0,num_epochs,10)):
        if os.path.exists(f'./Checkpoints/{genckptload}_{epoch}.pth') and os.path.exists(f'./Checkpoints/{discckptload}_{epoch}.pth'):
            epoch,loss_gen,l=load_checkpoint(f'./Checkpoints/genepoch{i}.pth',gen, opt_gen, lera,g_scaler)
            print(f'Generator checkpoint loaded:  genepoch{i}.pth')
            epoch,loss_disc,l=load_checkpoint(f'./Checkpoints/discepoch{i}.pth',disc, opt_disc, lera,d_scaler)
            print(f'Discriminator checkpoints loaded:  discepoch{i}.pth')
            break
    print(epoch,l)
    gen.train()
    disc.train()
    num_epochs=num_epochs-epoch
    print(f'Remaining epochs: {num_epochs}')
    print('Checkpoints loaded')
else:
    print('Training from start.....')
    genckptsave=input('Enter generator checkpoint name to save: ')
    discckptsave=input('Enter discriminator checkpoint name to save: ')
    genmodelsave=input('Enter generator model name to save: ')
    discmodelsave=input('Enter discriminator model name to save: ')
    
writer_real=SummaryWriter(f'logs/real')
writer_fake=SummaryWriter(f'logs/fake')
bce=nn.BCEWithLogitsLoss()
L1=nn.SmoothL1Loss(beta=0.5)

for epoch in range(num_epochs-epoch):
    for i in tqdm(range(80),desc ="Folders"):
        img_lst=[]
        img_lst.clear()
        for j in range(999):
            pat='E:/GAN_video/'
            file=pat+str(i)+'/'+str(j)+'.jpg'
            im = Image.open(file)
            img=transform_norm(im)
#             img=img.permute(2, 0, 1)
            img_lst.append(img)
       

        i1=[]
        i2=[]
        i3=[]
        i1.clear()
        i2.clear()
        i3.clear()
        step=0
        for x in range(0,len(img_lst),3):
            i1.append(img_lst[x])
            i2.append(img_lst[x+1])
            i3.append(img_lst[x+2])
        io=[t.numpy() for t in i1]
        it=[t.numpy() for t in i2]
        ith=[t.numpy() for t in i3]
        for y in range(0,len(io),9):
            i1b=torch.tensor(np.array(io[y:y+9])).to(device)
            i2b=torch.tensor(np.array(it[y:y+9])).to(device)
            i3b=torch.tensor(np.array(ith[y:y+9])).to(device)
            f13=torch.cat((i1b,i3b),1).to(device)
            
            with torch.cuda.amp.autocast():
                fake_img=gen(f13)
                disc_fake_input=torch.cat((f13,fake_img),1).to(device)
                disc_real_input=torch.cat((f13,i2b),1)
                
                disc_real=disc(disc_real_input)
                real_label=torch.ones_like(disc_real)
                real_label=smooth_positive_labels(real_label.cpu().detach())
                loss_disc_real=bce(disc_real,real_label.to(device))

                disc_fake=disc(disc_fake_input)
                fake_label=torch.ones_like(disc_fake)
                fake_label=smooth_negative_labels(fake_label.cpu().detach())
                loss_disc_fake=bce(disc_fake,fake_label.to(device))
                loss_disc=(loss_disc_fake+loss_disc_real)
            
            d_scaler.scale(loss_disc).backward(retain_graph=True)
            d_scaler.step(opt_disc)
            d_scaler.update()
            opt_disc.zero_grad()
            
            with torch.cuda.amp.autocast():
                disc_fake = disc(disc_fake_input)
                G_fake_loss = bce(disc_fake, real_label.to(device))
                l1loss = L1(fake_img, i2b) 
                loss_gen = G_fake_loss + l1loss

            
            g_scaler.scale(loss_gen).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            opt_gen.zero_grad()
            if y % 10 == 0:
                if i%10==0:
                    if epoch%10==0:
                        gn=f'C:/Users/navee/Documents/Frame_rate/Checkpoints/{genckptsave}_{epoch}.pth'
                        dn=f'C:/Users/navee/Documents/Frame_rate/Checkpoints/{discckptsave}_{epoch}.pth'
                    save_checkpoint(epoch,disc,opt_disc,loss_disc,l,d_scaler,dn)
                    save_checkpoint(epoch,gen,opt_gen,loss_gen,l,g_scaler,gn)
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {y}/{len(io)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(f13)
                    # take out (up to) 32 examples
                    # img_grid_real = torchvision.utils.make_grid(
                        # i2b[:9], normalize=True
                    # )
                    # img_grid_fake = torchvision.utils.make_grid(
                        # fake[:9], normalize=True
                    # )

                    # writer_real.add_image("Real", img_grid_real, global_step=step)
                    # writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    if i%10==0:
                        imgs=fake[0].cpu()
                        imgs=imgs.permute(1,2,0)
                        imgs=imgs.numpy()
                        print(imgs.shape)
                        print(type(imgs))
                        im = Image.fromarray((imgs * 255).astype(np.uint8))
                        im.save(f"E:/Frames/{l}.jpg")
                    l=l+1
                step += 1
    
    save_mdoel(gen,f'./Models/{genmodelsave}')
    save_mdoel(disc,f'./Models/{discmodelsave}')
