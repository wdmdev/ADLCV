import torch
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
import torchvision.transforms as transforms
import torchvision
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keys_list = []

def get_attention_hook(module, input, output, batch_size=8):
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

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(seed=1)
    model_name = 'model_heads_6'
    num_heads = 6
    patch_h, patch_w = 4, 4
    num_patches_h, num_patches_w = 8,8 

    # Load the ViT model from a file
    print('Loading model...')
    model = torch.load(f'models/{model_name}.pt')
    model.to(device)
    model.eval()
    print('Model loaded.')

    #get dataloaders
    print('Loading dataloader...')
    batch_size = 8
    classes = [3,7]
    testloader, testset = prepare_dataloader(batch_size=batch_size, classes=classes)

    # Load the image and convert it to a tensor
    for i, (image, label) in enumerate(testloader):
        image = image.to(device)

        # Forward pass through the model
        hook = model.transformer_blocks[-1].attention.register_forward_hook(get_attention_hook)
        print('Getting attention weights...')
        with torch.no_grad():
            output = model(image)
            attention_mask = keys_list[0]
            attention_mask = rearrange(attention_mask, '(b h) s -> h b s', b=batch_size, h=num_heads)
            attention_mask = torch.mean(attention_mask, dim=0)
            #threshold attention weights
            one_mask = torch.ones_like(image)
            image_patches = rearrange(image, 'b c (nph ph) (npw pw) -> b (nph npw) c ph pw', ph=patch_h, pw=patch_w) 
            one_patches = rearrange(one_mask, 'b c (nph ph) (npw pw) -> b (nph npw) c ph pw', ph=patch_h, pw=patch_w) 
            attention_patches = attention_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            one_attention_patches = one_patches * attention_patches
            image_attention_patches = image_patches * attention_patches
            one_attention_weights = rearrange(one_attention_patches, 'b (nph npw) c ph pw -> b c (nph ph) (npw pw)', nph=num_patches_h, npw=num_patches_w, ph=patch_h, pw=patch_w)
            image_attention_weights = rearrange(image_attention_patches, 'b (nph npw) c ph pw -> b c (nph ph) (npw pw)', nph=num_patches_h, npw=num_patches_w, ph=patch_h, pw=patch_w)
            image_attention_weights[image_attention_weights < 10e-5] = 0
            hook.remove()

        #Save plot of attention weights
        print('Saving attention weights...')
        one_attention_weights = one_attention_weights.cpu()
        image_attention_weights = image_attention_weights.cpu()
        #create image grid
        img_grid = torchvision.utils.make_grid(image.cpu(), nrow=batch_size, normalize=True, pad_value=0.9)
        #create attention grid
        one_attention_grid = torchvision.utils.make_grid(one_attention_weights, nrow=batch_size, normalize=True, pad_value=0.9)
        #create image attention grid
        # img_attention_grid = torchvision.utils.make_grid(image_attention_weights, nrow=batch_size, normalize=True, pad_value=0.9)

        #vertical stack image and attention grid
        img_grid = torch.cat((img_grid, one_attention_grid), dim=1)

        one_attention_weights = to_pil_image(img_grid)
        one_attention_weights.save(f'plots/new_{model_name}_attention_weights.png')

        print('Done.')
        break

if __name__ == '__main__':
    main()
