from torchvision.io import read_image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

img = read_image("zoo.jpg")

# Step 1: init model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

preprocess = weights.transforms()
img_transformed = preprocess(img).unsqueeze(0)

prediction = model(img_transformed).squeeze(0).softmax(0)

print('weights', weights.meta["categories"])

