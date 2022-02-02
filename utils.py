import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("/home/semi/Machine_Learning/Mnist/Images/image_0.png").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: five ")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab)).split(' ')[0]
    )
    test_img2 = transform(
        Image.open("/home/semi/Machine_Learning/Mnist/Images/image_1.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT:  zero ")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab)).split(' ')[0]
    )
    test_img3 = transform(Image.open("/home/semi/Machine_Learning/Mnist/Images/image_2.png").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: four ")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab)).split(' ')[0]
    )
    test_img4 = transform(
        Image.open("/home/semi/Machine_Learning/Mnist/Images/image_3.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: one ")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab)).split(' ')[0]
    )
    test_img5 = transform(
        Image.open("/home/semi/Machine_Learning/Mnist/Images/image_4.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: Nine")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab)).split(' ')[0]
    )
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step