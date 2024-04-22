from PIL import Image
import numpy as np
import os
import torch
import cv2
import time
import imageio

import torchvision.transforms as transforms

from net import MODEL as net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')


model = net(in_channel=2)

model_path = "/kaggle/working/MATR/model_10.pth"
use_gpu = torch.cuda.is_available()


if use_gpu:

    model = model.cuda()
    model.cuda()

    model.load_state_dict(torch.load(model_path))

else:

    state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)

path1 = '/kaggle/working/MATR/PET.bmp'
path2 = '/kaggle/working/MATR/MRI.bmp'
img1 = Image.open(path1).convert('L')
img2 = Image.open(path2).convert('L')  
def fusion():

    for num in range(1):
        tic = time.time()

        path1 = '/kaggle/working/MATR/11.png'
        path2 = '/kaggle/working/MATR/MRI.bmp'

        img1 = cv2.imread(path1)
        img2 = Image.open(path2).convert('L')
        img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        yuv_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        y = yuv_image[:,:,0]
        u = yuv_image[:,:,1]
        v = yuv_image[:,:,2]
        img1_org = y
        img2_org = img2

        tran = transforms.ToTensor()

        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)
        if use_gpu:
            input_img = input_img.cuda()
        else:
            input_img = input_img

        model.eval()
        out = model(input_img)

        d = np.squeeze(out.detach().cpu().numpy())
        result = (d* 255).astype(np.uint8)
        merged_image = cv2.merge([result, u, v])
        rgb_image = cv2.cvtColor(merged_image, cv2.COLOR_YUV2RGB)
        imageio.imwrite('/kaggle/working/{}.bmp'.format(num),
                        rgb_image)


        toc = time.time()
        print('end  {}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))
      



if __name__ == '__main__':

    fusion()
