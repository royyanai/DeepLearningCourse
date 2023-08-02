import torch
import torchvision
import PIL.Image as Image
import scipy.linalg as lin
import numpy as np
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(13)


FEATURE_EXTRACTOR = torch.hub.load('pytorch/vision:v0.10.0',
                                   'inception_v3', weights='Inception_V3_Weights.IMAGENET1K_V1')
FEATURE_EXTRACTOR.to(device)
FEATURE_EXTRACTOR.eval()


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = torchvision.transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(torchvision.transforms.ToTensor()(resized_img))
    output = torch.stack(arr)
    output = output.to(device)
    return output


def extract_features(self, batch):

    batch=interpolate(batch)

    batch = self.Conv2d_1a_3x3(batch)
    # N x 32 x 149 x 149
    batch = self.Conv2d_2a_3x3(batch)
    # N x 32 x 147 x 147
    batch = self.Conv2d_2b_3x3(batch)
    # N x 64 x 147 x 147
    batch = self.maxpool1(batch)
    # N x 64 x 73 x 73
    batch = self.Conv2d_3b_1x1(batch)
    # N x 80 x 73 x 73
    batch = self.Conv2d_4a_3x3(batch)
    # N x 192 x 71 x 71
    batch = self.maxpool2(batch)
    # N x 192 x 35 x 35
    batch = self.Mixed_5b(batch)
    # N x 256 x 35 x 35
    batch = self.Mixed_5c(batch)
    # N x 288 x 35 x 35
    batch = self.Mixed_5d(batch)
    # N x 288 x 35 x 35
    batch = self.Mixed_6a(batch)
    # N x 768 x 17 x 17
    batch = self.Mixed_6b(batch)
    # N x 768 x 17 x 17
    batch = self.Mixed_6c(batch)
    # N x 768 x 17 x 17
    batch = self.Mixed_6d(batch)
    # N x 768 x 17 x 17
    batch = self.Mixed_6e(batch)
    # N x 768 x 17 x 17
    batch = self.Mixed_7a(batch)
    # N x 1280 x 8 x 8
    batch = self.Mixed_7b(batch)
    # N x 2048 x 8 x 8
    batch = self.Mixed_7c(batch)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    batch = self.avgpool(batch)
    # N x 2048 x 1 x 1
    batch = self.dropout(batch)
    # N x 2048 x 1 x 1
    batch = torch.flatten(batch, 1)
    # N x 2048

    output=batch.numpy(force=True)
    return output


def matrix_sqrt(matrix):
    output = lin.sqrtm(matrix)
    return output

def compute_mean(input):
    return torch.mean(input,dim=0)

def compute_cov_matrix(input):
    B,D=input.shape
    
    mean=compute_mean(input)
    means_matrix=mean.reshape(B,1)*mean.reshape(1,B)

    products_tensor= input.reshape(B,D,1)*input.reshape(B,1,D)

    cov=compute_mean(products_tensor)-means_matrix
    return cov

def compute_fid(mean_1,mean_2,cov_1,cov_2):
    diff=mean_1-mean_2
    
    mean_diff_term=np.sum(diff**2)

    cov_2_sqrt=matrix_sqrt(cov_2)
    sqrt_covs_term=np.trace(matrix_sqrt(np.matmul(cov_2_sqrt,np.matmul(cov_1,cov_2_sqrt))))

    return mean_diff_term+np.trace(cov_1)+np.trace(cov_2)-2*sqrt_covs_term


def compute_fid_from_feature_files(gen_path,or_path):

    generated_features=[]
    original_features=[]

    while True:
        with open(gen_path ,'br') as f:
            try:
                features=pickle.load(f)
                generated_features.append(features)
            except:
                break    

    while True:
        with open(or_path ,'br') as f:
            try:
                features=pickle.load(f)
                original_features.append(features)
            except:
                break    

    original_features=np.stach(original_features)
    generated_features=np.stack(generated_features)

    or_mean=compute_mean(original_features)
    or_cov=compute_cov_matrix(original_features)

    gen_mean=compute_mean(generated_features)
    gen_cov=compute_cov_matrix(generated_features)

    print(compute_fid(or_mean,gen_mean,or_cov,gen_cov))



