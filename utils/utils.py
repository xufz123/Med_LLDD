import time, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils import data
import cv2
from . import sr_imgproc
from networks.sr_networks import SRCNN, VDSR, EDSR, S_CTSRN
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import re


class TensorDataset(Dataset):
    def __init__(self, images, labels):         # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class MultiStaticSharedDataset(Dataset):
    def __init__(self, static, dynamic, hallucinator):
        self.static = static.detach().float()
        self.dynamic = dynamic.detach().float()
        self.hallucinator = hallucinator[0]
        self.spc, _, _, _ = static.shape  # (spc, 1, 512, 512)
        self.n_c, self.dpc = dynamic.shape # (num_class, dpc)
        self.samples = [[] for c in range(self.n_c)]
        with torch.no_grad():
            for c in range(self.n_c):
                self.samples[c] = self.hallucinator(self.static.unsqueeze(0), self.dynamic[c])  # Add a batch dimension.
                # print(torch.max(self.samples[c]))
                # w/o fidlity
                # self.samples[c] = torch.clamp(self.samples[c], min=0, max=1)  # Clamp to the CT intensity range [0, 1].
                # w fidlity
                self.samples[c] = torch.clamp(self.static[c%self.spc]+self.samples[c], min=0, max=1)

        self.vpc = self.samples[0].shape[0]
        # print(len(self.samples))  # 6
        # print(self.samples[0].shape)  # [vpc, 1, 512, 512]

    def __getitem__(self, index):
        label = index//self.vpc
        idx = index%self.vpc
        sample = self.samples[label][idx]
        return sample, label
    
    def __len__(self):
        return self.n_c*self.vpc
    

class MultiStaticSharedDataset_3D(Dataset):
    def __init__(self, static, dynamic, hallucinator):
        """
            static  (spc, num_slices, 512, 512)
            dynamic (num_class, dpc)
            hallucinator (in_channel=args.spc*args.dpc, out_channel=args.vpc): process one image at a time.
        """
        self.static = static.detach().float()
        self.dynamic = dynamic.detach().float()
        self.hallucinator = hallucinator[0]
        self.spc, self.num_slices, _, _ = static.shape   # (spc, num_slices, 512, 512)
        self.n_c, self.dpc = dynamic.shape # (num_class, dpc)
        self.samples = [[] for c in range(self.n_c)]
        with torch.no_grad():
            for c in range(self.n_c):
                slices_out = []
                for s in range(self.num_slices):
                    static_slice = self.static[:, s:s+1, :, :]   # (spc, 1, 512, 512)
                    out_slice = self.hallucinator(static_slice.unsqueeze(0), self.dynamic[c])  # Add a batch dimension.
                    slices_out.append(out_slice)
                sample_c = torch.cat(slices_out, dim=1)
                # print(torch.max(self.samples[c]))
                # w/o fidlity
                # self.samples[c] = torch.clamp(self.samples[c], min=0, max=1)  # Clamp to the CT intensity range [0, 1].
                # w fidlity
                self.samples[c] = torch.clamp(self.static[c%self.spc]+sample_c, min=0, max=1)

        self.vpc = self.samples[0].shape[0]
        # print(len(self.samples))      # 6
        print(self.samples[0].shape)  # [vpc, num_slices, 512, 512]

    def __getitem__(self, index):
        label = index//self.vpc
        idx = index%self.vpc
        sample = self.samples[label][idx]
        return sample, label            # (num_slices, 512, 512)
    
    def __len__(self):
        return self.n_c*self.vpc


class dataset_ct(data.Dataset):
    """
        Dataset of (high-resolution image, class/subject label) pairs.
    """
    def __init__(self, data_path, mode, upscale_factor):
        self.image_file_names = [os.path.join(data_path, image_file_name) for image_file_name in os.listdir(data_path)]
        self.image_size = 512
        parts = data_path.split('/')[-2]
        if parts=='medical_ldct':
            self.image_size = 256
        _ = upscale_factor
        self.mode = mode
        self.mappinglist = ["L004", "L006", "L014", "L019", "L033", "L049", "L056", "L057", "L058", "L064", "L299", "L277"]
        if parts == 'medical_ct_mix':
            self.mappinglist = ["L000", "L004", "L006", "L014", "L019", "L033", "L049", "L056", "L057", "L058", "L064", "L299", "L277"]

    def __getitem__(self, index):
        image_name = self.image_file_names[index]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        image = image[:, :, np.newaxis]  # shape:(512, 512, 1)
        try:
            className = os.path.basename(image_name).split("_")[0]
            if className in ["L299", "L277"]:
                class_index = -1  # Test
            else:
                class_index = self.mappinglist.index(className)
        except ValueError:
            print(f"Subject {os.path.basename(image_name).split('_')[0]} is not in the mapping list")
        if self.mode == "Train":
            hr_image = sr_imgproc.random_crop(image, self.image_size)
        elif self.mode == "Test":
            hr_image = sr_imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Test`.")
        
        # lr_image = sr_imgproc.image_resize(hr_image, 1 / self.upscale_factor)
        # lr_image = sr_imgproc.image_resize(lr_image, self.upscale_factor)

        # Convert image data into Tensor stream format (PyTorch).
        # lr_y_tensor = sr_imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_y_tensor = sr_imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return hr_y_tensor, class_index  # hr, class(0~9)
        
    def __len__(self):
        return len(self.image_file_names)


class dataset_ct_all(data.Dataset):
    """
        Dataset of (high-resolution image, class/subject label) pairs.
    """
    def __init__(self, data_path, mode, upscale_factor):
        self.image_file_names = [os.path.join(data_path, image_file_name) for image_file_name in os.listdir(data_path)]
        self.image_size = 512
        parts = data_path.split('/')[-2]
        if parts=='medical_ldct':
            self.image_size = 256
        _ = upscale_factor
        self.mode = mode
        self.mappinglist = ['L004', 'L006', 'L014', 'L019', 'L033', 'L049', 'L056', 'L057', 'L058', 'L064',
                            'L071', 'L072', 'L075', 'L077', 'L081', 'L107', 'L110', 'L114', 'L116', 'L123',
                            'L125', 'L131', 'L134', 'L145', 'L148', 'L150', 'L160', 'L170', 'L175', 'L178',
                            'L179', 'L186', 'L187', 'L193', 'L203', 'L209', 'L210', 'L212', 'L219', 'L220',
                            'L221', 'L229', 'L232', 'L237', 'L241', 'L248', 'L266', 'L273', 'L277', 'L299']

    def __getitem__(self, index):
        image_name = self.image_file_names[index]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        image = image[:, :, np.newaxis]  # shape:(512, 512, 1)
        try:
            className = os.path.basename(image_name).split("_")[0]
            if className in ["L299", "L277"]:
                class_index = -1  # Test
            else:
                class_index = self.mappinglist.index(className)
        except ValueError:
            print(f"Subject {os.path.basename(image_name).split('_')[0]} is not in the mapping list")
        if self.mode == "Train":
            hr_image = sr_imgproc.random_crop(image, self.image_size)
        elif self.mode == "Test":
            hr_image = sr_imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Test`.")
        
        # lr_image = sr_imgproc.image_resize(hr_image, 1 / self.upscale_factor)
        # lr_image = sr_imgproc.image_resize(lr_image, self.upscale_factor)

        # Convert image data into Tensor stream format (PyTorch).
        # lr_y_tensor = sr_imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_y_tensor = sr_imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return hr_y_tensor, class_index  # hr, class(0~9)
        
    def __len__(self):
        return len(self.image_file_names)


def natural_sort_key(s):
    """
    Helper function for natural sorting.
    Sort filenames by numeric value, so 'L004_2.png' comes before 'L004_10.png'.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


class dataset_ct_3d(data.Dataset):
    """
        Dataset of (high-resolution image, class/subject label) pairs.
    """
    def __init__(self, data_path, mode, upscale_factor, num_slices=10, stride=5):
        """
        Args:
            data_path: Dataset directory
            mode: "Train" or "Test"
            upscale_factor: Super-resolution scale factor (retained for interface compatibility)
            num_slices: Number of 2D slices in a 3D volume (the number of network input channels)
            stride: Sliding-window stride for extracting volumes; a stride of 1 gives maximum overlap for data augmentation.
        """
        self.image_file_names = [os.path.join(data_path, image_file_name) for image_file_name in os.listdir(data_path)]
        self.image_size = 512
        parts = data_path.split('/')[-2]
        if parts=='medical_ldct':
            self.image_size = 256
        _ = upscale_factor
        self.mode = mode
        self.num_slices = num_slices
        self.mappinglist = ["L004", "L006", "L014", "L019", "L033", "L049", "L056", "L057", "L058", "L064", "L299", "L277"]
        if parts == 'medical_ct_mix':
            self.mappinglist = ["L000", "L004", "L006", "L014", "L019", "L033", "L049", "L056", "L057", "L058", "L064", "L299", "L277"]

        # 1. Naturally sort filenames to order slices along the Z axis, assuming filenames encode slice order.
        all_files = os.listdir(data_path)
        all_files.sort(key=natural_sort_key)
        # 2. Group slices by subject (className).
        patient_files = {}
        for f in all_files:
            try:
                className = f.split("_")[0]
                if className not in patient_files:
                    patient_files[className] = []
                patient_files[className].append(os.path.join(data_path, f))
            except Exception as e:
                print(f"Skipping filename that could not be parsed: {f}, error: {e}")
        # 3. Build a list of 3D volume samples using a sliding window.
        self.volume_samples = []
        for className, file_paths in patient_files.items():
            # Determine the label index.
            if className in ["L299", "L277"]:
                class_index = -1  # Test subjects or subjects that do not require classification
            else:
                try:
                    class_index = self.mappinglist.index(className)
                except ValueError:
                    print(f"Warning: subject {className} is not in mappinglist; skipping.")
                    continue
            
            # Build volumes only when the subject has at least num_slices slices.
            if len(file_paths) >= self.num_slices:
                for i in range(0, len(file_paths) - self.num_slices + 1, stride):
                    # Select num_slices consecutive slice paths to form one 3D sample.
                    volume_paths = file_paths[i : i + self.num_slices]
                    self.volume_samples.append((volume_paths, class_index))

        print(f"Dataset initialized: mode={self.mode}, generated {len(self.volume_samples)} 3D samples ({self.num_slices} consecutive slices per sample).")

    def __getitem__(self, index):
        volume_paths, class_index = self.volume_samples[index]
        # 1. Read consecutive 2D slices.
        volume_list = []
        for path in volume_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Unable to read image: {path}")
            img = img.astype(np.float32) / 255.
            volume_list.append(img)
        # 2. Stack into a 3D array with shape (H, W, num_slices), e.g. (512, 512, 16).
        volume_array = np.stack(volume_list, axis=-1) 
        # 3. Apply spatially consistent cropping to preserve the 3D structure.
        h, w, c = volume_array.shape
        th, tw = self.image_size, self.image_size
        if self.mode == "Train":
            # Choose a random crop origin and apply it to all channels (slices) in the volume.
            if h > th and w > tw:
                i = np.random.randint(0, h - th + 1)
                j = np.random.randint(0, w - tw + 1)
                volume_array = volume_array[i:i+th, j:j+tw, :]
            elif h == th and w == tw:
                pass # The dimensions already match; no cropping is needed.
            else:
                raise ValueError(f"Image dimensions ({h}, {w}) are smaller than the target crop size ({th}, {tw})")
        elif self.mode == "Test":
            # Apply the same center crop to all slices.
            if h > th and w > tw:
                i = (h - th) // 2
                j = (w - tw) // 2
                volume_array = volume_array[i:i+th, j:j+tw, :]
        else:
            raise ValueError("Unsupported mode; use 'Train' or 'Test'.")

        # 4. Convert to a PyTorch tensor with shape (C, H, W).
        # C (the number of channels) equals num_slices.
        volume_array = np.transpose(volume_array, (2, 0, 1)) 
        hr_y_tensor = torch.from_numpy(volume_array).contiguous()
        return hr_y_tensor, class_index
        
    def __len__(self):
        # The dataset length is the total number of 3D volumes.
        return len(self.volume_samples)
    

class dataset_lidc(data.Dataset):
    """
        Dataset of (high-resolution image, class/subject label) pairs.
    """
    def __init__(self, data_path, mode, upscale_factor):
        self.image_file_names = [os.path.join(data_path, image_file_name) for image_file_name in os.listdir(data_path)]
        self.image_size = 512
        _ = upscale_factor
        self.mode = mode
        self.mappinglist = ["LIDC-IDRI-0028", "LIDC-IDRI-0029", "LIDC-IDRI-0030", "LIDC-IDRI-0031", "LIDC-IDRI-0032",
                            "LIDC-IDRI-0033", "LIDC-IDRI-0034","LIDC-IDRI-0035","LIDC-IDRI-0036","LIDC-IDRI-0037"]

    def __getitem__(self, index):
        image_name = self.image_file_names[index]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        image = image[:, :, np.newaxis]  # shape:(512, 512, 1)
        try:
            className = os.path.basename(image_name).split("_")[0]
            if className in ["LIDC-IDRI-0001"]:
                class_index = -1  # Test
            else:
                class_index = self.mappinglist.index(className)
        except ValueError:
            print(f"Subject {os.path.basename(image_name).split('_')[0]} is not in the mapping list")
        if self.mode == "Train":
            hr_image = sr_imgproc.random_crop(image, self.image_size)
        elif self.mode == "Test":
            hr_image = sr_imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Test`.")
        
        # lr_image = sr_imgproc.image_resize(hr_image, 1 / self.upscale_factor)
        # lr_image = sr_imgproc.image_resize(lr_image, self.upscale_factor)

        # Convert image data into Tensor stream format (PyTorch).
        # lr_y_tensor = sr_imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_y_tensor = sr_imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return hr_y_tensor, class_index  # hr, class(0~9)
        
    def __len__(self):
        return len(self.image_file_names)


class dataset_mri(data.Dataset):
    """
        Dataset of (high-resolution image, class/subject label) pairs.
    """
    def __init__(self, data_path, mode, upscale_factor):
        self.image_file_names = [os.path.join(data_path, image_file_name) for image_file_name in os.listdir(data_path)]
        self.image_size = 256
        _ = upscale_factor
        self.mode = mode
        self.mappinglist = ["CC0001", "CC0002", "CC0003", "CC0004", "CC0005", "CC0006", "CC0007", "CC0008", "CC0009", "CC0010", "CC0011", "CC0012"]

    def __getitem__(self, index):
        image_name = self.image_file_names[index]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        image = image[:, :, np.newaxis]  # shape:(512, 512, 1)
        try:
            className = os.path.basename(image_name).split("_")[0]
            if className in ["CC0011", "CC0012"]:
                class_index = -1  # Test
            else:
                class_index = self.mappinglist.index(className)
        except ValueError:
            print(f"Subject {os.path.basename(image_name).split('_')[0]} is not in the mapping list")
        
        hr_y_tensor = sr_imgproc.image2tensor(image, range_norm=False, half=False)

        return hr_y_tensor, class_index  # hr, class(0~9)
        
    def __len__(self):
        return len(self.image_file_names)


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def get_dataset(dataset, data_path):
    if dataset == 'medical_CT':
        channel     = 1
        im_size     = (512, 512)
        num_classes = 10    # 10 categories refer to 10 subjects , 10 for normal, 11 for mix
        mean        = []
        std         = []
        dst_train   = dataset_ct(data_path + "/train", "Train", 2)
        dst_test    = dataset_ct(data_path + "/test", "Test", 2)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'medical_mri':
        channel     = 1
        im_size     = (256, 256)
        num_classes = 10
        mean        = []
        std         = []
        dst_train   = dataset_mri(data_path + "/train", "Train", 2)
        dst_test    = dataset_mri(data_path + "/test", "Test", 2)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'medical_ldct':  # restoration task
        channel     = 1
        im_size     = (256, 256)
        num_classes = 10
        mean        = []
        std         = []
        dst_train   = dataset_ct(data_path + "/train", "Train", 2)
        dst_test    = dataset_ct(data_path + "/test", "Test", 2)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'medical_ct_all':
        channel     = 1
        im_size     = (512, 512)
        num_classes = 48
        mean        = []
        std         = []
        dst_train   = dataset_ct_all(data_path + "/train", "Train", 2)
        dst_test    = dataset_ct_all(data_path + "/test", "Test", 2, 10, 10)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'medical_CT_3D':
        mean        = []
        std         = []
        channel     = 0
        im_size     = (512, 512)
        num_classes = 10      
        dst_train   = dataset_ct_3d(data_path + "/train", "Train", 2)
        dst_test    = dataset_ct_3d(data_path + "/test", "Test", 2)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'LIDC_IDRI_CT':
        mean        = []
        std         = []
        channel     = 0
        im_size     = (512, 512)
        num_classes = 10
        dst_train   = dataset_lidc(data_path + "/train", "Train", 2)
        dst_test    = dataset_lidc(data_path + "/test", "Test", 2)
        class_names = [str(c) for c in range(num_classes)]
    else:
        exit('unknown dataset: %s'%dataset)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=10, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader


def get_network(model, channel, args, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    if model == 'SRCNN':
        net = SRCNN()
    elif model == 'VDSR':
        net = VDSR()
    elif model == 'EDSR':
        net = EDSR(args)
    elif model == 'S_CTSRN':
        net = S_CTSRN()
    else:
        net = None
        exit('unknown model: %s'%model)
    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net


def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 5:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)
    elif args.dis_metric == 'l1':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum(torch.abs(gw_syn_vec - gw_real_vec))
    elif args.dis_metric == 'huber':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        # Use the built-in PyTorch Huber loss (equivalent to Smooth L1 when delta=1).
        # delta is a tunable hyperparameter, typically set to 1.0.
        dis = torch.nn.functional.huber_loss(gw_syn_vec, gw_real_vec, reduction='sum', delta=1.0)
    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, transform, savef=False):
    num_exp, loss_avg, psnr_val, ssim_val = 0, 0, 0, 0
    #psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3), reduction='elementwise_mean').to(args.device)  # Average the per-image PSNR values within each batch.
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    net = net.to(args.device)
    criterion = criterion.to(args.device)
    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        # print(i_batch)
        img_hr = datum[0].float().to(args.device)  # 0~1  shape(5,1,512,512)
        img_lr = transform(img_hr)
        n_b = img_hr.shape[0]
        output = net(img_lr)
        loss = criterion(output, img_hr)
        with torch.no_grad():
            # Disable gradient tracking for PSNR and SSIM to avoid retaining metric computation graphs.
            psnr_val += psnr_metric(output, img_hr).cpu().data.numpy()*n_b  # Weight the batch PSNR by the batch size.
            ssim_val += ssim_metric(output, img_hr).cpu().data.numpy()*n_b
            if mode=='test' and savef:
                os.makedirs(args.save_img_path, exist_ok=True)
                for i in range(output.shape[0]):
                    filename = os.path.join(args.save_img_path, f'batch_{i_batch}_img_{i}.png')
                    save_image(output[i], filename)
        loss_avg += loss.item()*n_b
        num_exp += n_b
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    loss_avg /= num_exp
    psnr_val /= num_exp
    ssim_val /= num_exp
    return loss_avg, psnr_val, ssim_val


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def evaluate_synset(it_eval, net, images_hr_train, labels_train, testloader, args, transform):
    net = net.to(args.device)
    images_hr_train = images_hr_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    if args.model=='EDSR' or args.model=='S_CTSRN':
        criterion = nn.L1Loss().to(args.device)
    else:
        criterion = nn.MSELoss().to(args.device)
    dst_train = TensorDataset(images_hr_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, psnr_train, ssim_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=False, transform=transform)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
    time_train = time.time() - start
    loss_test, psnr_test, ssim_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False, transform=transform)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train psnr = %.4f, train ssim = %.4f, test psnr = %.4f, test ssim = %.4f' 
          % (get_time(), it_eval, Epoch, int(time_train), loss_train, psnr_train, ssim_train, psnr_test, ssim_test ))
    return net, psnr_train, ssim_train, psnr_test, ssim_test
    

def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['strategy'] = 'none'

    return dc_aug_param


def evaluate_synset_s2d(it_eval, net, images_train, labels_train, testloader, args, transform, mode='hallucinator', return_loss=False, test_freq=None):
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.MSELoss().to(args.device)
    if args.model=='EDSR' or args.model=='S_CTSRN' or args.model=='GHASR':
        criterion = nn.L1Loss().to(args.device)
    if mode == 'none':
        dst_train = TensorDataset(images_train, labels_train)
    elif mode == 'multi-static':
        dst_train = MultiStaticSharedDataset(images_train[0], images_train[1], images_train[2])
    elif mode == 'multi-static-3d':
        dst_train = MultiStaticSharedDataset_3D(images_train[0], images_train[1], images_train[2])
    else:
        raise NotImplementedError
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
    
    start = time.time()
    
    for ep in range(Epoch + 1):
        loss_train, psnr_train, ssim_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=False, transform=transform)
        if ep in lr_schedule:
            lr *= 0.1
            # print('lr = %.6f'%lr)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    time_train = time.time() - start

    with torch.no_grad():
        # Redefine the transform when training on LDCT data and testing on SRCT data.
        # transform = transforms.Compose([
        #     transforms.Resize((int(512/4), int(512/4))),
        #     transforms.Resize((int(512), int(512))),
        # ])
        loss_test, psnr_test, ssim_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False, transform=transform, savef=args.savef)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train psnr = %.4f, train ssim = %.4f, test psnr = %.4f, test ssim = %.4f' 
          % (get_time(), it_eval, Epoch, int(time_train), loss_train, psnr_train, ssim_train, psnr_test, ssim_test ))
    return net, psnr_train, ssim_train, psnr_test, ssim_test


def evaluate_synset_conv(it_eval, net, images_hr_train, labels_train, hals, testloader, args, transform):
    net = net.to(args.device)
    images_hr_train = images_hr_train.to(args.device)  # shape(spc, 1, 512, 512)
    # print(f'images_hr_train-shape:{images_hr_train.shape}')
    labels_train = labels_train.to(args.device) # torch.Size([50])
    # Generate images for each subject label.
    output_list = []
    for i, conv_layer in enumerate(hals):
        img_index = i % args.spc
        input_image = images_hr_train[img_index]
        input_image = input_image.unsqueeze(0) # (1, 1, 512, 512)
        output = conv_layer(input_image) # (1, ipc, 512, 512)
        # output = torch.clamp(output, 0.0, 1.0)
        min_val = output.min()
        max_val = output.max()
        if max_val - min_val == 0:   # Avoid division by zero.
            return torch.zeros_like(output)
        output = (output - min_val) / (max_val - min_val)  # Normalize to [0, 1].
        output_reshaped = output.view(args.ipc, 1, 512, 512)
        output_list.append(output_reshaped)
    final_array = torch.cat(output_list, dim=0)  # final_array has shape (num * ipc, 1, 512, 512).
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    if args.model=='EDSR' or args.model=='S_CTSRN':
        criterion = nn.L1Loss().to(args.device)
    else:
        criterion = nn.MSELoss().to(args.device)
    dst_train = TensorDataset(final_array, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, psnr_train, ssim_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=False, transform=transform)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
    time_train = time.time() - start
    loss_test, psnr_test, ssim_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False, transform=transform)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train psnr = %.4f, train ssim = %.4f, test psnr = %.4f, test ssim = %.4f' 
          % (get_time(), it_eval, Epoch, int(time_train), loss_train, psnr_train, ssim_train, psnr_test, ssim_test ))
    return net, psnr_train, ssim_train, psnr_test, ssim_test


class Conv2DNet(nn.Module):
    # in_channel=spc*dpc; out_channel=vpc 
    def __init__(self, in_channel=4, out_channel=10, img_size=512, kernel_size=3, mode='mul'):
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        if mode == 'add':
            in_channel = 3
        self.encoder = nn.Conv2d(in_channel, out_channel, kernel_size, padding=1)

    def forward(self, static, dynamic):
        # static (bz, spc, 1, 512, 512)
        # dynamic (dpc)
        spc = static.size(1)
        dpc = dynamic.size(0)

        if self.mode == 'mul':
            dynamic = dynamic.view(1, 1, dpc, 1, 1)
            x = static * dynamic  # (bz, spc, dpc, 512, 512)
            x = x.reshape(1, spc*dpc, self.img_size, self.img_size)
        else:
            raise NotImplementedError
        x = self.encoder(x).unsqueeze(2)  # bz, vpc, 1, 512, 512
        return x.squeeze(0)


class Conv2DNet_per_patient(nn.Module):
    # input (1, 1, 512, 512); output (ipc, 1, 512, 512)
    # Each class has a convolutional layer that maps the initial hqimg to that class's hqimg.
    def __init__(self, in_channel=1, out_channel=5, kernel_size=3):
        super().__init__()
        self.encoder = nn.Conv2d(in_channel, out_channel, kernel_size, padding=1)

    def forward(self, hqimg_init):
        x = self.encoder(hqimg_init)
        return x


if __name__ == '__main__':
    medical_data = dataset_ct("datasets/medical_ct/train", "Train", 2)
    medical_dataloader = torch.utils.data.DataLoader(medical_data, batch_size=10, shuffle=True, num_workers=0)
