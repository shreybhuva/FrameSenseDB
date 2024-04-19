import mongo_setup
import save_data
from data_model import datapt

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import cv2
import time

def main():
    mongo_setup.global_init()
    datapt.drop_collection()
    
    '''
The entire code that will loop through the video and generate scene and object dictionary
The code will also have an iterating integer variable for frame number
Code below will be inside the loop and will store the the generated data to a database
    '''

    cap = cv2.VideoCapture('testvideo.mp4')
    if not cap.isOpened():
        print("video not captured")

    frame_num=1
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        #will break if iterated through each frame
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        scene = detect_scene(img)
        objdict = detect_object(img)

        #dummy variables
        # video_name = 'Video.mp4'
        # frame_num = 10
        # scene = 'lawn'
        # objdict = {'Horse':5}

        save_data.save_info(frame_num,scene,objdict)

        if frame_num%10 == 0:
            print(f'Processed frame: {frame_num} / {total_frame} in {time.time()-start_time: .4f} seconds')
            start_time = time.time()

        frame_num+=1

def detect_scene(img):
    input_img = V(centre_crop(img).unsqueeze(0))

    logit = model_sc_det.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    return classes[idx[0]]

def detect_object(img,threshold=0.5):
    dist = {}
    for i in model_objdet.names.values():
        dist[i] = 0

    results = model_objdet(img)
    results.xyxy[0] = results.xyxy[0][results.xyxy[0][:,4] > threshold]

    for i in results.xyxy[0]:
        dist[model_objdet.names[i[-1].item()]]+=1

    return dist


start_time = time.time()
# Load YOLOv5s
model_name_objdet = 'yolov5s'
model_objdet = torch.hub.load('ultralytics/yolov5', model_name_objdet, pretrained=True)

# Load Places365 CNN
arch = 'resnet18'
model_name_sc_det = '%s_places365.pth.tar' % arch

model_sc_det = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_name_sc_det, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model_sc_det.load_state_dict(state_dict)
model_sc_det.eval()

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

file_name = 'categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)
# Places365 CNN loaded

print(f"Models loaded in {time.time() - start_time:.4f} seconds")

main()