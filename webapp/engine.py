import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from u2net import utils, model
import cv2

model_path = './ckpt/u2net_bce_itr_107900_train_0.040223_tar_0.001831.pth'
model_pred = model.U2NET(3, 1)
model_pred.load_state_dict(torch.load(model_path, map_location="cpu"))
model_pred.eval()


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def remove_bg(image, resize=False):
    sample = preprocess(np.array(image))

    with torch.no_grad():
        inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, _, _, _, _, _, _ = model_pred(inputs_test)
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred).squeeze().cpu().detach().numpy()
        img_out = Image.fromarray(predict * 255).convert("RGB")
        img_out = img_out.resize(image.size, resample=Image.BILINEAR)
        empty_img = Image.new("RGBA", image.size, 0)
        img_out = Image.composite(image, empty_img, img_out.convert("L"))
        del d1, pred, predict, inputs_test, sample

        return img_out


def blur_bg(image, threshold, resize=False):
    sample = preprocess(np.array(image))

    with torch.no_grad():
        inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, _, _, _, _, _, _ = model_pred(inputs_test)
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred).squeeze().cpu().detach().numpy()
        img_out = Image.fromarray(predict * 255).convert("RGB")
        img_out = img_out.resize(image.size, resample=Image.BILINEAR)

        img_out = img_out.point(lambda p: 255 - p)
        blurred_img = image.filter(ImageFilter.GaussianBlur(threshold))
        result_img = Image.composite(blurred_img, image, img_out.convert("L"))

        del d1, pred, predict, inputs_test, sample

        return result_img


def create_mask(image):
    sample = preprocess(np.array(image))
    with torch.no_grad():
        inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())
        d1, _, _, _, _, _, _ = model_pred(inputs_test)
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred).squeeze().cpu().detach().numpy()
        img_out = Image.fromarray(predict * 255).convert("RGB")
        image = img_out.resize(image.size, resample=Image.BILINEAR)

    return image


def remove_bg_mult(image):
    img_out = image.copy()
    for _ in range(4):
        img_out = create_mask(img_out)

    img_out = img_out.resize(image.size, resample=Image.BILINEAR)
    empty_img = Image.new("RGBA", image.size, 0)
    img_out = Image.composite(image, empty_img, img_out)
    return img_out


def change_background(image, background):
    background = background.resize(image.size, resample=Image.BILINEAR)
    img_out = Image.alpha_composite(background, image)
    return img_out
