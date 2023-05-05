from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomVerticalFlip
from modeling_pix2pix import GeneratorUNet
import torch, math
from PIL import Image

# Configure dataloaders
transform = Compose(
        [
            Resize((512, 512), Image.BICUBIC),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

def nearest_pow2(n):
    return 2 ** round(math.log2(n))


def center_crop(img):
    MIN = min(img.size)
    width, height = img.size
    left = (width - MIN) / 2
    top = (height - MIN) / 2
    right = (width + MIN) / 2
    bottom = (height + MIN) / 2
    crop = img.crop((left, top, right, bottom))
    return crop.resize((nearest_pow2(MIN), nearest_pow2(MIN)), Image.ANTIALIAS)


img = Image.open("cmp_b0004.png").convert("RGB")
img = center_crop(img).convert("RGB")
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)
generator = GeneratorUNet()
generator.load_state_dict(torch.load("saved_models/generator.pth"))
fake_B = generator(img_tensor)
save_image(fake_B, "OUT.png", normalize=True)
