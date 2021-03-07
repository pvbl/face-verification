import mlflow
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torchvision
from visualization.image import imshow

def load_model(run_id):
    model_uri = "runs:/{}/model".format(run_id)
    loaded_model = mlflow.pytorch.load_model(model_uri)
    return loaded_model


def predict_model(model,x0,x1,plot=True):
    concatenated = torch.cat((x0,x1),0)
    output1, output2 = model(x0,x1)
    euclidean_distance = F.pairwise_distance(output1, output2).item()
    if plot:
        print("disimilarity {:.2f}".format(euclidean_distance))
        imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {0:.2f} label {1}'.format(euclidean_distance,""))
    return euclidean_distance

def predict_over_dataset(model,siamese_dataset,x0,plot=False,cuda = False):
    disimilarities = []
    labels=[]
    for x1,l2 in siamese_dataset.iter_over_all_img_dataset():
        x1 = x1.unsqueeze(0)
        concatenated = torch.cat((x0,x1),0)
        if cuda:
            output1,output2 = model(Variable(x0).cuda(),Variable(x1).cuda())
        else:
            output1,output2 = model(Variable(x0),Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        print(euclidean_distance)
        disimilarity = euclidean_distance.item()
        if plot:
            imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {0:.2f} label {1}'.format(disimilarity,l2))
        disimilarities.append(disimilarity)
        labels.append(l2)
    return disimilarities,labels
