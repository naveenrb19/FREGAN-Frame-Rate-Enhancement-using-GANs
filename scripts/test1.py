import torch
from PIL import Image
import torch.optim as optim
import numpy as np
from torchvision import transforms
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=torch.load('gen1.pth').to(device)
model.eval()
transform_norm = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(0,1)])
import torch
import os
import cv2
import numpy as np


path='E:/GAN_video/Test/0/'
files=os.listdir(path)
pthlst=[]
for i in range(999):
    pthlst.append(path+str(i)+'.jpg')

x=0
for i in range(999):
    img=Image.open(pthlst[i])
    img.save(f"E:/GAN_video/Test\SRC/{x}.jpg")
    x=x+2
p=0
img_lst=[]
img_lst.clear()
for j in range(998):
            pat='E:/GAN_video/Test/SRC/'
            file=pat+str(p)+'.jpg'
            im = Image.open(file)
            im=transform_norm(im)
            im=torch.unsqueeze(im, 0)
            # img=img.permute(0,3,1,2)
            img_lst.append(im)
            p=p+2
            print(p)
print(len(img_lst))
i1=[]
i3=[]
i1.clear()
i3.clear()
i=1
for x in range(0,998):
    

# for y in range(0,499):
    f13=torch.cat((img_lst[x],img_lst[x+1]),1).to(device)
    pred=model(f13)
    print(f'Out: {pred.shape}')
    imgs=pred[0].cpu().detach()
    imgs=imgs.permute(1,2,0)
    imgs=imgs.numpy()
    im = Image.fromarray((imgs * 255).astype(np.uint8))
    im.save(f"E:/GAN_video/Test/Final/{i}.jpg")
    i=i+2