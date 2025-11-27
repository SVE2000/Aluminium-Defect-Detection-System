from models import *

def return_fasterrcnn_mobilenetv3_large_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_mobilenetv3_large_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

create_model = {
    'fasterrcnn_mobilenetv3_large_fpn': return_fasterrcnn_mobilenetv3_large_fpn,
}