from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image is <= 400 pixels in the x-y dims.'''
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def imshow(img):
    plt.figure(1)
    plt.imshow(img)
    plt.show()

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv0',
                  '5': 'conv5', 
                  '10': 'conv10', 
                  '19': 'conv19',   ## content representation
                  '21': 'conv21',
                  '28': 'conv28'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, n_features, h, w = tensor.size()
    tensor = tensor.view(n_features, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    vgg.to(device)

    content = load_image('images/mer.jpg').to(device)
    style = load_image('images/peinture1.jpg', shape=content.shape[-2:]).to(device)

    imshow(im_convert(content))
    imshow(im_convert(style))

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    target = content.clone().requires_grad_(True).to(device)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    style_layers = {'conv0', 'conv5', 'conv10', 'conv19', 'conv21', 'conv28'}

    optimizer = optim.Adam([target], lr=0.003)
    for i in range(500): 
        target_features = get_features(target, vgg)

        content_loss = torch.mean((target_features['conv19'] - content_features['conv19'])**2)

        style_loss = 0
        target_grams = {layer: gram_matrix(target_features[layer]) for layer in target_features}
        for layer in style_layers:
            style_loss += torch.mean((style_grams[layer] - target_grams[layer])**2)
        style_loss /= len(style_layers)

        total_loss = 0.5 * content_loss + 0.5 * style_loss  # Adjust the weights

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f'Itertation {i} /500, press ctrl c to stop')
        if i % 100 == 0:
            print(f'Iteration {i}, Total loss: {total_loss.item()}, Content loss: {content_loss.item()}, Style loss: {style_loss.item()}')
            imshow(im_convert(target))

        if i % 499 == 0:
            print(f'Iteration {i}, Total loss: {total_loss.item()}, Content loss: {content_loss.item()}, Style loss: {style_loss.item()}')
            print('last image')
            imshow(im_convert(target))