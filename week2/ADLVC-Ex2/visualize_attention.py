import PIL
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import numpy as np
from einops import rearrange
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keys_list = []

def get_attention_hook(module, input, output, ph=4, pw=4):
    # We get the attention using the keys of the last attention block
    x = input[0]
    keys    = module.k_projection(x)
    queries = module.q_projection(x)
    values  = module.v_projeciton(x)

    # Rearrange keys, queries and values 
    # from batch_size x seq_len x embed_dim to (batch_size x num_head) x seq_len x head_dim
    keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=module.num_heads, d=module.head_dim)
    queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=module.num_heads, d=module.head_dim)

    attention_logits = torch.matmul(queries, keys.transpose(1, 2))
    attention = F.softmax(attention_logits, dim=-1)
    attention = attention[:, 0, 1:]
    attention = torch.mean(attention, dim=0)

    keys_list.append(attention)


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
    patch_h, patch_w = 4, 4
    num_patches_h, num_patches_w = 8, 8

    # Load the ViT model from a file
    print('Loading model...')
    model = torch.load('models/model1.pt')
    model.to(device)
    model.eval()
    print('Model loaded.')

    #get dataloaders
    print('Loading dataloader...')
    batch_size = 1 
    classes = [3,7]
    testloader, testset = prepare_dataloader(batch_size=batch_size, classes=classes)

    # Load the image and convert it to a tensor
    for i, (image, label) in enumerate(testloader):
        image = image.to(device)
        img_name = f'img{i+1}'

        # Forward pass through the model
        hook = model.transformer_blocks[-1].attention.register_forward_hook(get_attention_hook)
        print('Getting attention weights...')
        with torch.no_grad():
            output = model(image)
            attention = keys_list[0]
            patches = rearrange(image, 'b c (nph ph) (npw pw) -> b (nph npw) c ph pw', ph=patch_h, pw=patch_w) 
            attention_patches = patches * attention.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            attention_weights = rearrange(attention_patches, 'b (nph npw) c ph pw -> b c (nph ph) (npw pw)', nph=num_patches_h, npw=num_patches_w, ph=patch_h, pw=patch_w)
            hook.remove()

        #Save plot of attention weights
        print('Saving attention weights...')
        attention_weights = attention_weights[0].cpu()
        attention_weights = torchvision.utils.make_grid(attention_weights, nrow=8, normalize=True, pad_value=0.9)
        attention_weights = to_pil_image(attention_weights)
        attention_weights = resize(attention_weights, (256, 256))
        attention_weights.save(f'plots/{img_name}_attention_weights.png')

        print('Done.')
        if i > 5:
            break

if __name__ == '__main__':
    main()
