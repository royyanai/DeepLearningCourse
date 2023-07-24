import torch
import torchvision
import PIL.Image as Image
import scipy.linalg as lin


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(13)


FEATURE_EXTRACTOR = torch.hub.load('pytorch/vision:v0.10.0',
                                   'inception_v3', weights='Inception_V3_Weights.IMAGENET1K_V1')
FEATURE_EXTRACTOR.to(device)
FEATURE_EXTRACTOR.eval()


def interpolate(batch, size):
    arr = []
    for img in batch:
        pil_img = torchvision.transforms.ToPILImage()(img)
        resized_img = pil_img.resize((size, size), Image.BILINEAR)
        arr.append(torchvision.transforms.ToTensor()(resized_img))
    output = torch.stack(arr)
    output = output.to(device)
    return output


def extract_features(self, x):
    x = self.Conv2d_1a_3x3(x)
    # N x 32 x 149 x 149
    x = self.Conv2d_2a_3x3(x)
    # N x 32 x 147 x 147
    x = self.Conv2d_2b_3x3(x)
    # N x 64 x 147 x 147
    x = self.maxpool1(x)
    # N x 64 x 73 x 73
    x = self.Conv2d_3b_1x1(x)
    # N x 80 x 73 x 73
    x = self.Conv2d_4a_3x3(x)
    # N x 192 x 71 x 71
    x = self.maxpool2(x)
    # N x 192 x 35 x 35
    x = self.Mixed_5b(x)
    # N x 256 x 35 x 35
    x = self.Mixed_5c(x)
    # N x 288 x 35 x 35
    x = self.Mixed_5d(x)
    # N x 288 x 35 x 35
    x = self.Mixed_6a(x)
    # N x 768 x 17 x 17
    x = self.Mixed_6b(x)
    # N x 768 x 17 x 17
    x = self.Mixed_6c(x)
    # N x 768 x 17 x 17
    x = self.Mixed_6d(x)
    # N x 768 x 17 x 17
    x = self.Mixed_6e(x)
    # N x 768 x 17 x 17
    x = self.Mixed_7a(x)
    # N x 1280 x 8 x 8
    x = self.Mixed_7b(x)
    # N x 2048 x 8 x 8
    x = self.Mixed_7c(x)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    x = self.avgpool(x)
    # N x 2048 x 1 x 1
    x = self.dropout(x)
    # N x 2048 x 1 x 1
    x = torch.flatten(x, 1)
    # N x 2048
    return x


def matrix_sqrt(matrix):
    matrix = matrix.to('cpu')
    matrix = matrix.detach()
    matrix = matrix.numpy()
    sqrt = lin.sqrtm(matrix)
    output = torch.from_numpy(sqrt)
    output = output.to(device)
    return output


def compute_FID(ground_truth_distribution, genrated_distribution):
    ground_truth_feature_distribution = extract_features(FEATURE_EXTRACTOR,
                                                         ground_truth_distribution)
    generated_feature_distribution = extract_features(
        FEATURE_EXTRACTOR, genrated_distribution)

    ground_truth_feature_distribution = torch.transpose(
        ground_truth_feature_distribution, dim0=0, dim1=1)
    generated_feature_distribution = torch.transpose(
        generated_feature_distribution, dim0=0, dim1=1)

    ground_truth_covariance = torch.cov(ground_truth_feature_distribution)
    # checking for nan
    nan_sum = torch.sum(torch.isnan(ground_truth_covariance))
    if nan_sum > 0:
        print("nan found in ground truth covariance")
    generated_distribution_covariance = torch.cov(
        generated_feature_distribution)
    # checking for nan
    nan_sum = torch.sum(torch.isnan(generated_distribution_covariance))
    if nan_sum > 0:
        print("nan found in generated distribution covariance")

    ground_truth_mean = torch.mean(ground_truth_feature_distribution, dim=1)
    generated_distribution_mean = torch.mean(
        generated_feature_distribution, dim=1)
    # checking for nan
    nan_sum = torch.sum(torch.isnan(
        ground_truth_mean+generated_distribution_mean))
    if nan_sum > 0:
        print("nan found in mean")

    sqrt = matrix_sqrt(
        torch.mm(ground_truth_covariance, generated_distribution_covariance))
    # checking for nan
    nan_sum = torch.sum(torch.isnan(sqrt))
    if nan_sum > 0:
        print("nan found in sqrt")

    trace = torch.trace(ground_truth_covariance +
                        generated_distribution_covariance-2*sqrt)

    FID = torch.sqrt(torch.abs(torch.norm(ground_truth_mean -
                     generated_distribution_mean, p=2)+trace))

    return FID
