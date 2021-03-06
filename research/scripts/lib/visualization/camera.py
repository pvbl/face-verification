import cv2
from datetime import datetime
from array import array
import os
from PIL import Image
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch


from helpers.env import load_config
from models.predicts import load_model, predict_model, predict_over_dataset
from data.data import SiameseNetworkDataset


def generate_date_now():
    return datetime.now().strftime("%Y%m%d_%H%M_%Ss")

def mkdir_folder(path_root):
    path = generate_date_now() + "_pics"
    out_path = os.path.join(path_root,path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return out_path


def file_paths_from_name(name,ids="ids",pics="pics",path=None,path_root=None):
    if not path:
        path = mkdir_folder(path_root)

    id_pic = os.path.join(path,"{0}_c1.png".format(name))
    pers_pic = os.path.join(path,"{0}_c2.png".format(name))
    return id_pic, pers_pic

def check_exist_files_from_person(id_pic,pers_pic):
    assert os.path.exists(id_pic) and os.path.exists(pers_pic),  "needs both pics"


def take_a_snapshot(file_name,window_name="ID-picture"):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow(window_name)
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow(window_name, frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed

            cv2.imwrite(file_name, frame)
            print("{} written!".format(file_name))
            img_counter += 1
            break

    cam.release()

    cv2.destroyAllWindows()


def init_data(data_folder):
    folder_dataset = dset.ImageFolder(root=data_folder)
    s_data = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
    return s_data





def predict_photos_similarity(model_id, id_pic, pers_pic,path_root):
    net = load_model(model_id)
    s_data = init_data(path_root)

    x0 = s_data.process_img(id_pic).unsqueeze(0)
    x1 = s_data.process_img(pers_pic).unsqueeze(0)

    return predict_model(net,x0,x1,plot=True)


def take_photos(name,main_model,path=None,path_root=None):
    id_pic, pers_pic = file_paths_from_name(name,path=path,path_root=path_root)
    print("Picture of ID...")
    take_a_snapshot(file_name=id_pic)
    print("Selfie...")
    take_a_snapshot(file_name=pers_pic,window_name="selfie")
    check_exist_files_from_person(id_pic, pers_pic)
    preds = predict_photos_similarity(main_model, id_pic, pers_pic,path_root=path_root)
    print("SIMILARITY MODEL...")
    print("Predictions are ..{0}".format(preds))
    return id_pic, pers_pic




class predictOverVideoSim:
    def __init__(self,data_folder,model,id_pic):
        self.s_data = init_data(data_folder)
        self.model = model
        self.x0 = self.s_data.process_img(id_pic).unsqueeze(0)
    def frame_to_pil(self,frame):

        return Image.fromarray(frame).convert("L")

    def predict(self,frame):
        frame = self.frame_to_pil(frame)
        x1 = self.s_data.process_img(frame).unsqueeze(0)
        preds = predict_model(self.model,self.x0,x1,plot=False)
        return preds


def write_text_over_img(img, text, font = cv2.FONT_HERSHEY_SIMPLEX, bottomLeftCornerOfText = (0,100),fontScale  = 2,
               fontColor = (255,255,255),lineType= 2):


                return cv2.putText(img,text,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)



def take_a_video(pred_vids,window_name="ID-picture"):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow(window_name)
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break


        pred = pred_vids.predict(frame)

        _=write_text_over_img(frame,"disimilarity {0:.2f}".format(pred))
        cv2.imshow(window_name, frame)


        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed

            #cv2.imwrite(file_name, frame)
            #print("{} written!".format(file_name))
            img_counter += 1
            break

    cam.release()

    cv2.destroyAllWindows()


#import threading
#import tempfile


#def printit(frame,time=5.0):
#    threading.Timer(time, printit).start()
#    file_name = temp_name = next(tempfile._get_candidate_names())
#    cv2.imwrite(file_name, frame)
#    return img_plot_result(img_comp,[face_info_to_json(f2[0])],thickness=10)
