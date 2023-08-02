import torch
import torch.nn as nn
import numpy as np
import random as rn
import pickle
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import My_FID as fid


rn.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


TRANSFORM = transforms.Compose(
    (transforms.ToTensor(), transforms.RandomHorizontalFlip()))


TRAIN_SET = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=TRANSFORM)
TEST_SET = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=TRANSFORM)


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dimension, dropout=0.1):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_layer_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')

        self.conv_layer_2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same')

        if in_channels != out_channels:
            self.compatability_layer = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')

        self.time_embedding_layer = nn.Linear(
            time_embedding_dimension, out_channels)

        self.batch_norm_1 = nn.GroupNorm(32, in_channels)
        self.batch_norm_2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout2d(dropout)

        self.activation = nn.SiLU()

    def forward(self, input, time_emb):
        B = input.size(0)

        time_emb = self.time_embedding_layer(time_emb)
        time_emb = self.activation(time_emb)

        out = self.activation(self.batch_norm_1(input))
        out = self.conv_layer_1(out)
        out = out+time_emb.view(B, -1, 1, 1)
        out = self.activation(self.batch_norm_2(out))
        out = self.dropout(out)
        out = self.conv_layer_2(out)
        if self.in_channels == self.out_channels:
            skip = input
        else:
            skip = self.compatability_layer(input)
        out = out+skip

        return out


class AttensionBlock(nn.Module):
    def __init__(self, channels):
        super(AttensionBlock, self).__init__()
        self.Q = nn.Conv2d(channels, channels, 1)
        self.K = nn.Conv2d(channels, channels, 1)
        self.V = nn.Conv2d(channels, channels, 1)
        self.output_layer = nn.Conv2d(channels, channels, 1)

        self.softmax = nn.Softmax(dim=3)
        self.batchnorm = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, input):
        B, C, H, W = input.size()
        out = self.activation(self.batchnorm(input))
        q = self.Q(out)
        k = self.K(out)
        v = self.V(out)

        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * (int(C) ** (-0.5))
        w = w.view(B, H, W, H*W)
        w = self.softmax(w)
        w = w.view(B, H, W, H, W)

        h = torch.einsum('bhwHW,bcHW->bchw', w, v)
        h = self.output_layer(h)

        out = h+input
        return out


class KnnBlock(nn.Module):
    def __init__(self, in_chanels, latent_space_dim, nn_number, sqrt_num_patches, patch_side_len):
        super(KnnBlock, self).__init__()
        self.out_size_len = sqrt_num_patches
        self.patch_side_len = patch_side_len
        self.nn_number = nn_number
        self.latent_space_dim = latent_space_dim

        self.patch_encoding_layers = nn.Sequential(nn.Conv2d(in_channels=in_chanels, out_channels=in_chanels, kernel_size=patch_side_len, stride=patch_side_len),
                                                   nn.BatchNorm2d(num_features=in_chanels), nn.ReLU(), nn.Conv2d(in_channels=in_chanels, out_channels=latent_space_dim, kernel_size=1, stride=1))
        self.temperature_layers = nn.Sequential(nn.Conv2d(in_channels=in_chanels, out_channels=1, kernel_size=patch_side_len, stride=patch_side_len),
                                                nn.BatchNorm2d(num_features=1), nn.ReLU())
        self.softmax = nn.Softmax(dim=3)

    def forward(self, input):
        # assigning constants used for numerical stability in logarithem and division operations
        log_epsilon = torch.tensor(0.1**45, dtype=torch.float32, device=device)
        dev_epsilon = torch.tensor(0.025, dtype=torch.float32, device=device)

        # metric processing
        metric = self.patch_encoding_layers(input)

        B, C, H, W = metric.size()

        # compute similarity measure tensor
        similarity_tensor = torch.sum(metric.view(
            B, C, H, W, 1, 1)-metric.view(B, C, 1, 1, H, W), dim=1)
        similarity_tensor = -(similarity_tensor**2)

        # creating tensor with large negative numbers on the diagonal to prevent first nearest neighbor to be self
        anti_self_tensor = torch.zeros((H, W, H, W), device=device)
        for i in range(H):
            for j in range(W):
                anti_self_tensor[i, j, i, j] = -1000
        anti_self_tensor = torch.broadcast_to(
            anti_self_tensor, (B, H, W, H, W))

        similarity_tensor = similarity_tensor + anti_self_tensor

        # change dimension for normalisation purpuses and for summing over the different patches at the end
        similarity_tensor = similarity_tensor.view(B, H, W, H*W)

        # perform nearest neighbor distribution calculation
        """here we use bothe the previously defined epsilon values for numerical stability"""
        temperatures = self.temperature_layers(input)
        temperatures = temperatures.view(B, H, W, 1)

        nearest_neighbor_tensors = []
        for i in range(self.nn_number):
            nearest_neighbor_tensor = self.softmax(
                similarity_tensor/(dev_epsilon+temperatures))
            nearest_neighbor_tensors.append(nearest_neighbor_tensor)
            if i < (self.nn_number-1):
                similarity_tensor = similarity_tensor + \
                    torch.log(1 - nearest_neighbor_tensor.detach()+log_epsilon)

        output = tuple(nearest_neighbor_tensors)
        return output


class Model(nn.Module):
    def __init__(self, time_embedding_dim=None, temporal_depth=1000, input_side_len=32, num_levels=4, base_channel_expension=64, channel_multiplier_list=[1, 2, 4, 8], latent_patch_space=30, attension_resolutions_indicies=[], K=2, patch_size=8):
        super(Model, self).__init__()
        self.num_levels = num_levels
        self.temporal_depth = temporal_depth
        self.base_channels = base_channel_expension
        self.latent_patch_space = latent_patch_space
        self.input_image_side_length = 2**((num_levels+1))
        self.patch_size = patch_size
        self.attension_resolution_indicies = attension_resolutions_indicies
        if time_embedding_dim == None:
            self.time_emb_dim = 4*base_channel_expension
        else:
            self.time_emb_dim = time_embedding_dim

        # Noising parameters
        self.variance_schedual = np.array(
            [0.0001+el*(0.02-0.0001)/1000 for el in range(1, 1001)])
        self.mean_schedual = np.array(
            [np.sqrt(1-var) for var in self.variance_schedual])
        self.alpha_bar = np.array(
            [np.prod(1-self.variance_schedual[:el+1]) for el in range(temporal_depth)])

        self.variance_schedual_tensor = torch.tensor(
            self.variance_schedual, dtype=torch.float32, device=device, requires_grad=False)
        self.mean_schedual_tensor = torch.tensor(
            self.mean_schedual, dtype=torch.float32, device=device, requires_grad=False)
        self.alpha_bar_tensor = torch.tensor(
            self.alpha_bar, dtype=torch.float32, device=device, requires_grad=False)

        # layers
        self.time_emb_layers = nn.Sequential(nn.Linear(base_channel_expension, self.time_emb_dim), nn.SiLU(
        ), nn.Linear(self.time_emb_dim, self.time_emb_dim), nn.SiLU())

        self.knn_time_emb_layers = nn.Sequential(nn.Linear(
            self.time_emb_dim, (K+1)*3), nn.Tanh())

        # did not exist in previous model
        self.final_avg_time_emb_layer = nn.Sequential(nn.Linear(
            self.time_emb_dim, 1), nn.Sigmoid())
        # did not exist in previous model
        self.channel_expension_layer = nn.Conv2d(
            in_channels=3, out_channels=base_channel_expension, kernel_size=3, padding='same')

        for i in range(num_levels):
            mul = channel_multiplier_list[i]
            setattr(self, f"going_down_resblock_1_level_{i+1}", ResBlock(
                mul*base_channel_expension, mul*base_channel_expension, self.time_emb_dim))
            setattr(self, f"going_down_resblock_2_level_{i+1}", ResBlock(
                mul*base_channel_expension, mul*base_channel_expension, self.time_emb_dim))
            if i in attension_resolutions_indicies:
                setattr(
                    self, f"going_down_attension_block_level_{i+1}", AttensionBlock(mul*base_channel_expension))

        self.knn_block = KnnBlock(in_chanels=3, latent_space_dim=latent_patch_space,
                                  nn_number=K, sqrt_num_patches=input_side_len//(patch_size), patch_side_len=patch_size)

        self.knn_avg_layers = nn.Sequential(nn.Conv2d((K+1)*3, (K+1)*3, kernel_size=3, padding='same'), nn.BatchNorm2d((K+1)*3), nn.SiLU(),
                                            nn.Conv2d(
            (K+1)*3, (K)*3, kernel_size=3, padding='same'), nn.BatchNorm2d((K)*3), nn.SiLU(),
            nn.Conv2d((K)*3, (K-1)*3, kernel_size=3, padding='same'), nn.BatchNorm2d(
            (K-1)*3), nn.SiLU(),
            nn.Conv2d((K-1)*3, 3, kernel_size=3, padding='same'), nn.Tanh())

        for i in range(num_levels-1):
            mul = channel_multiplier_list[i]
            setattr(self, f"downsample_layer_{i+1}", nn.Sequential(nn.Conv2d(
                mul*base_channel_expension, mul*2*base_channel_expension, kernel_size=2, stride=2)))

        self.first_bottom_res_block = ResBlock(
            channel_multiplier_list[-1]*base_channel_expension,
            channel_multiplier_list[-1]*base_channel_expension, time_embedding_dimension=self.time_emb_dim)
        self.bottom_attension_block = AttensionBlock(
            channel_multiplier_list[-1]*base_channel_expension)
        self.second_bottom_res_block = ResBlock(
            channel_multiplier_list[-1]*base_channel_expension,
            channel_multiplier_list[-1]*base_channel_expension, time_embedding_dimension=self.time_emb_dim)

        for i in range(num_levels-1):
            mul = list(reversed(channel_multiplier_list))[i]
            setattr(self, f"upsample_layer_{i+1}", nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(
                mul*base_channel_expension, (mul//2)*base_channel_expension, kernel_size=3, padding='same')))

        for i in range(num_levels):
            mul = list(reversed(channel_multiplier_list))[i]
            setattr(self, f"going_up_resblock_1_level_{num_levels-i}", ResBlock(
                mul*2*base_channel_expension, mul*base_channel_expension, self.time_emb_dim))
            setattr(self, f"going_up_resblock_2_level_{num_levels-i}", ResBlock(
                mul*2*base_channel_expension, mul*base_channel_expension, self.time_emb_dim))
            setattr(self, f"going_up_resblock_3_level_{num_levels-i}", ResBlock(
                mul*2*base_channel_expension, mul*base_channel_expension, self.time_emb_dim))
            if num_levels-1-i in attension_resolutions_indicies:
                setattr(self, f"going_up_attension_block_level_{num_levels-i}", AttensionBlock(
                    mul*base_channel_expension))

        self.normalize = nn.GroupNorm(32, base_channel_expension)
        self.output_layers = nn.Conv2d(
            in_channels=base_channel_expension, out_channels=3, kernel_size=3, padding='same')

        self.silu = nn.SiLU()

        # loss function
        self.l2 = nn.MSELoss()

    # TEMPORAL ENCODING METHODS:

    def initial_temporal_encoding(self, time_list):
        """recieves a list of time values as argument and returns a vanilla temporal encodings (one for each image in the batch),
        similar to the one in the famous papaer 'attention is all you need' only here since our data is a two dimensional image we multiply two sine/cosine waves
        one for the hight and one for the width. Notice the output does nt contain the input image."""
        B = len(time_list)
        num_channels = self.base_channels
        # initialze empty lists to be turned into 2d tensors of the different time encodings for each example in the batch
        # these lists will be turnd to tensors and multiplied to produce the final output
        temporal_encodings_list = []

        for time in time_list:
            # initialize and compute current example  time encoding encoding
            temporal_encoding = []
            for i in range(num_channels):
                if i % 2 == 0:
                    temporal_encoding.append(
                        np.sin(time/(10000**(i/num_channels))))
                else:
                    temporal_encoding.append(
                        np.cos(time/((10000**((i-1)/num_channels)))))
            temporal_encodings_list.append(temporal_encoding)
        # turn encodign lists into tnesors
        temporal_encoding_tensor = torch.tensor(
            temporal_encodings_list, device=device, dtype=torch.float32)

        # rearange and multipy the hight and width encodings to produce a single encoding
        temporal_encoding_tensor = temporal_encoding_tensor.view(
            B, num_channels)

        return temporal_encoding_tensor

    def generate_nearest_neighbors(self, input, nearest_neighbor_tensors, patch_size):
        """for the patch avereging portion of the computation it is convenient to unfoled our image so that each patch is a single vector"""
        num_nearest_neighbors = len(nearest_neighbor_tensors)
        B, H, W, L = nearest_neighbor_tensors[0].size()
        input_side_len = input.size(2)
        channels = input.size(1)

        Unfolder = nn.Unfold(kernel_size=(
            patch_size, patch_size), stride=patch_size)
        Folder = nn.Fold(output_size=(input_side_len, input_side_len), kernel_size=(
            patch_size, patch_size), stride=patch_size)

        # performing nearest neighbor averaging
        unfolded_input = Unfolder(input)

        nearest_neighbors = []
        for i in range(num_nearest_neighbors):
            weight_tensor = nearest_neighbor_tensors[i]
            neighbor = torch.einsum(
                'bvl,bHWl->bvHW', unfolded_input, weight_tensor)
            neighbor = neighbor.view(B, channels*(patch_size**2), H*W)
            neighbor = Folder(neighbor)
            nearest_neighbors.append(neighbor)

        # concatinate output
        output = torch.cat(nearest_neighbors, dim=1)

        return output

    def estimate_noise(self, input, time_list):
        B = input.size(0)
        alpha_bars = [self.alpha_bar[t] for t in time_list]
        alpha_bars = torch.tensor(
            alpha_bars, device=device, dtype=torch.float32)
        alpha_bars = alpha_bars.view(B, 1, 1, 1)

        main_temporal_encoding = self.initial_temporal_encoding(
            time_list=time_list)

        main_temporal_encoding = self.time_emb_layers(main_temporal_encoding)

        knn_temporal_encoding = self.knn_time_emb_layers(
            main_temporal_encoding).view(B, -1, 1, 1)

        skip_connctions = []
        out = self.channel_expension_layer(input)
        skip_connctions.append(out)

        for i in range(self.num_levels):
            out = self.__dict__[
                '_modules'][f'going_down_resblock_1_level_{i+1}'].forward(out, main_temporal_encoding)
            skip_connctions.append(out)
            out = self.__dict__[
                '_modules'][f'going_down_resblock_2_level_{i+1}'].forward(out, main_temporal_encoding)
            skip_connctions.append(out)

            if i in self.attension_resolution_indicies:
                out = self.__dict__[
                    '_modules'][f'going_down_attension_block_level_{i+1}'].forward(out)
            if i != self.num_levels-1:
                out = self.__dict__['_modules'][f'downsample_layer_{i+1}'](out)
                skip_connctions.append(out)

        out = self.first_bottom_res_block(out, main_temporal_encoding)
        out = self.bottom_attension_block.forward(out)
        out = self.second_bottom_res_block(out, main_temporal_encoding)

        for i in range(self.num_levels):
            out = torch.cat((out, skip_connctions.pop()), dim=1)
            out = self.__dict__[
                '_modules'][f'going_up_resblock_1_level_{self.num_levels-i}'].forward(out, main_temporal_encoding)
            out = torch.cat((out, skip_connctions.pop()), dim=1)
            out = self.__dict__[
                '_modules'][f'going_up_resblock_2_level_{self.num_levels-i}'].forward(out, main_temporal_encoding)
            out = torch.cat((out, skip_connctions.pop()), dim=1)
            out = self.__dict__[
                '_modules'][f'going_up_resblock_3_level_{self.num_levels-i}'].forward(out, main_temporal_encoding)
            if self.num_levels-1-i in self.attension_resolution_indicies:
                out = self.__dict__[
                    '_modules'][f'going_up_attension_block_level_{self.num_levels-i}'].forward(out)
            if i != self.num_levels-1:
                out = self.__dict__['_modules'][f'upsample_layer_{i+1}'](out)

        out = self.silu(self.normalize(out))
        out = self.output_layers(out)
        out = (input-torch.sqrt(1-alpha_bars)*out)/torch.sqrt(alpha_bars)
        skip = out

        nearest_neighbor_weights = self.knn_block.forward(out)
        nearest_neighbors = self.generate_nearest_neighbors(
            out, nearest_neighbor_weights, self.patch_size)
        out = torch.cat((out, nearest_neighbors), dim=1)
        out = out+knn_temporal_encoding*out
        out = self.knn_avg_layers(out)

        final_time_emb = self.final_avg_time_emb_layer(
            main_temporal_encoding).view(B, 1, 1, 1)
        out = final_time_emb*out+(1-final_time_emb)*skip

        out = (input-torch.sqrt(alpha_bars)*out)/torch.sqrt(1-alpha_bars)
        return out

    # IMAGE SAMPLING METHODS:

    def scale_images(self, input):
        """transforms image tensors from [0,1] to [-1,1]"""
        return (2*(input-0.5))

    def rescale_images(self, input):
        """transforms image tensors from [-1,1] to [0,1]"""
        return (input/2)+0.5

    def sample_noised_images(self, input_images, time_list):
        """given a batch of clean images, sample noised images at time t for each t in the time list.
        Returns a tuple of the noised images as the first element and the noise used to generate them as the second"""
        B, _, side_len, _ = input_images.size()

        gaussian_noise = torch.normal(torch.zeros(
            (B, 3, side_len, side_len), device=device))
        # gaussian_noise = torch.clip(gaussian_noise, min=-10, max=10) suspended!
        alpha_bar_tensor = [self.alpha_bar[t] for t in time_list]
        alpha_bar_tensor = torch.tensor(
            alpha_bar_tensor, device=device, dtype=torch.float32).view(B, 1, 1, 1)

        noised_images_tensor = (torch.sqrt(
            alpha_bar_tensor)*input_images)+(torch.sqrt(1-alpha_bar_tensor)*gaussian_noise)
        return (noised_images_tensor, gaussian_noise)

    def denoise_images(self, input_images, time):
        """REQUIRES INPUT TO BE NOISED UNIFORMLY! i.e. a single time for all images
        Returns denoised image tensor of an entire batch of images"""
        self.eval()
        B, _, side_len, _ = input_images.size()
        current_step_tensor = input_images
        # testing
        noise_nans = 0
        current_step_nan = 0
        for t in reversed(range(time)):
            current_time_list = [t]*B
            estimated_noise_tensor = self.estimate_noise(
                current_step_tensor, current_time_list)
            estimated_noise_tensor = estimated_noise_tensor.detach()
            noise_nans = torch.sum(torch.isnan(estimated_noise_tensor))
            current_step_tensor = (1/self.mean_schedual[t])*(current_step_tensor-(self.variance_schedual_tensor[t]/torch.sqrt(
                1-self.alpha_bar_tensor[t]))*estimated_noise_tensor)
            current_step_nan = torch.sum(torch.isnan(current_step_tensor))
            if t != 0:
                current_step_tensor += torch.normal(mean=torch.zeros(
                    (B, 3, side_len, side_len), device=device, requires_grad=False), std=torch.sqrt(self.variance_schedual_tensor[t]))

            current_step_tensor = current_step_tensor.detach()
            if noise_nans > 0 and current_step_nan == 0:
                print(f"nan first apears in noise at time {t}")
            if noise_nans == 0 and current_step_nan > 0:
                print(f"nan first apears in current step at time {t}")

        return torch.clip(current_step_tensor, min=-1, max=1)



def generate_test_data():
    """this is a temporary function for testing purpuses"""
    test_loader = DataLoader(TEST_SET, batch_size=100,
                             shuffle=False)
    data = None
    for X, Y in test_loader:
        data = X
    return data


def generate_train_data():
    """this is a temporary function for testing purpuses"""
    test_loader = DataLoader(TRAIN_SET, batch_size=100,
                             shuffle=False)
    data = None
    for X, Y in test_loader:
        data = X
    return data


def display_images(model, time, data):
    num_images = data.size(0)
    original_images = data.to(device)
    original_images = model.scale_images(original_images)
    time_list = [time]*num_images
    noised_images, _ = model.sample_noised_images(original_images, time_list)
    cleaned_images = model.denoise_images(
        noised_images, time)
    original_images = original_images.to('cpu')
    noised_images = noised_images.to('cpu')
    cleaned_images = cleaned_images.to('cpu')

    rescaled_originals = model.rescale_images(original_images).numpy()
    rescaled_originals = rescaled_originals.transpose(0, 2, 3, 1)

    rescaled_noised = model.rescale_images(noised_images).numpy()
    rescaled_noised = rescaled_noised.transpose(0, 2, 3, 1)

    rescaled_cleaned = model.rescale_images(cleaned_images).numpy()
    rescaled_cleaned = rescaled_cleaned.transpose(0, 2, 3, 1)

    for i in range(num_images):
        print("original image")
        plt.imshow(rescaled_originals[i, :, :, :])
        plt.show()
        print("noised image")
        plt.imshow(rescaled_noised[i, :, :, :])
        plt.show()
        print("cleaned image")
        print(
            f"cleaned image maximum pixel value is: {np.max(rescaled_cleaned[i, :, :, :])}")
        plt.imshow(rescaled_cleaned[i, :, :, :])
        plt.show()


def display_images_from_noise(model, num_images):
    mean = torch.zeros((num_images, 3, 32, 32))
    original_images = torch.normal(mean=mean, std=1)
    original_images = original_images.to(device)
    time_list = [999]*num_images
    cleaned_images = model.denoise_images(
        original_images, 999)
    cleaned_images = cleaned_images.to('cpu')

    rescaled_cleaned = model.rescale_images(cleaned_images).numpy()
    rescaled_cleaned = rescaled_cleaned.transpose(0, 2, 3, 1)
    for i in range(num_images):
        plt.imshow(rescaled_cleaned[i, :, :, :])
        plt.show()

def save_fid_features(model,batch_size,generated_path,original_data_path):
    feature_extractor=fid.FEATURE_EXTRACTOR
    data_loader = DataLoader(TRAIN_SET, batch_size=batch_size)


    for X,_ in data_loader:
        #generated images part of loop
        mean = torch.zeros((batch_size, 3, 32, 32))
        original_images = torch.normal(mean=mean, std=1)
        original_images = original_images.to(device)
        cleaned_images = model.denoise_images(
        original_images, 999)

        features=fid.extract_features(feature_extractor,cleaned_images)
        with open(generated_path,'ba') as f:
            print('generated features saved')
            pickle.dump(features,f)

        #dataset part of loop
        features=fid.extract_features(feature_extractor,X)
        with open(original_data_path,'ba') as f:
            print('original features saved')
            pickle.dump(features,f)

            

        

def train_and_evaluate(num_epochs, lr, batch_size,  model, wd=0, state_dict_paths=None):
   # initialize dataloaders
    train_loader = DataLoader(TRAIN_SET, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    # initize model loss function and optimizer and ema
    model.to(device)
    if state_dict_paths:
        model.load_state_dict(torch.load(state_dict_paths[0]))

    noise_criterion = model.l2
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=wd)

    ema = EMA(0.999)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    for epoch in range(num_epochs):
        # beggin training phase
        epoch_losses = []
        model.train()
        for batch_index, (X, Y) in enumerate(train_loader):
            time_list = []
            X = X.to(device)
            X = X.to(torch.float32)
            X = model.scale_images(X)
            Y = Y.to(device)
            for i in range(batch_size):
                random_time = rn.randint(1, 999)
                time_list.append(random_time)
            noised_images, noise = model.sample_noised_images(
                input_images=X, time_list=time_list)
            estimated_noise = model.estimate_noise(
                noised_images, time_list)
            loss = noise_criterion(estimated_noise, noise)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            optimizer.step()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema(name, param.data)

        torch.save(model.state_dict(), state_dict_paths[0])
        avg_loss = sum(epoch_losses)/len(epoch_losses)
        print(f'avg losss on epoch no. {epoch} was {avg_loss}')

    model.eval()
    display_images_from_noise(model, 30)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = ema.shadow[name]

    torch.save(model.state_dict(), state_dict_paths[1])
    print('ema model pictures')
    display_images_from_noise(model, 30)



