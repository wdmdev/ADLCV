import torch
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloader(batch_size, classes=[3, 7]):

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # select two classes 
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    testset, _ = torch.utils.data.random_split(testset, [1000, 1000])

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    return testloader, testset

def main():
    # Load the ViT model from a file
    print('Loading model...')
    model = torch.load('models/model1.pt')
    model.eval()
    model.to(device)
    print('Model loaded.')

    #get dataloaders
    print('Loading dataloader...')
    batch_size = 16
    classes = [3,7]
    testloader, testset = prepare_dataloader(batch_size=batch_size, classes=classes)

    # Load the image and convert it to a tensor
    for image, img_name in testloader:
        image = testset[0][0]
        image = image.to(device)

        # Forward pass through the model
        print('Running forward pass...')
        with torch.no_grad():
            output = model(image).argmax(dim=1)
        print('Forward pass done.')

        # Get the attention weights from the last attention layer
        print('Getting attention weights...')
        attention_weights = output['attentions'][-1]

        # Save the attention weights as an image
        print('Saving attention weights...')
        to_pil_image(attention_weights[0, 0]).save(f'plots/{img_name}_attention_weights.png')

        # Map the attention weights onto the input image
        attention_map = resize(attention_weights[0, 0], image.size, interpolation=Image.NEAREST)
        attention_map = to_pil_image(attention_map)
        attention_map.putalpha(128)

        print('Saving image with attention weights...')
        image.paste(attention_map, (0, 0), attention_map)

        # Save the resulting image as a .png file
        image.save(f'plots/{img_name}_image_with_attention.png')
        print('Done.')
        break

if __name__ == '__main__':
    main()
