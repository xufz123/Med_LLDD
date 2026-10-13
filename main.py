"""
Med_LLDD For CT Super-Resolution + static with dynamic
"""
import os, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.utils import *


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='LLDD')
    parser.add_argument('--dataset', type=str, default='medical_CT', help='dataset')
    parser.add_argument('--model', type=str, default='SRCNN', help='model SRCNN/EDSR')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')                                             # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')                                 # 3
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')     # 20 3 5
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')    # 300 for ct; 600 for mri
    parser.add_argument('--Iteration', type=int, default=2000, help='training iterations')                                  # 1000
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')            # 0.1  2000
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=8, help='batch size for real data')                               # 256
    parser.add_argument('--batch_train', type=int, default=8, help='batch size for training networks')                      # 256
    parser.add_argument('--init', type=str, default='onePeo', help='noise/real/onePeo: initialize synthetic images from random noise or randomly sampled real images.')  # noise
    parser.add_argument('--data_path', type=str, default='datasets/medical_ct', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='cos', help='distance metric')                                    # ours
    # s2d
    parser.add_argument('--spc', type=int, default=5, help='static memory(s) per class')
    parser.add_argument('--dpc', type=int, default=2, help='dynamic memory(s) per class')
    parser.add_argument('--vpc', type=int, default=1, help='virtual images per class')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--n_hal',type=int,default=1,help='number of hallucinators')
    parser.add_argument('--train_static', action='store_false', help='train static memory')                       # store_false ->True
    parser.add_argument('--path_static', type=str, default=None, help='path to pretrained static memory')
    parser.add_argument('--lr_static', type=float, default=100, help='learning rate for updating synthetic static memory')
    parser.add_argument('--lr_dynamic', type=float, default=0.01, help='learning rate for updating synthetic dynamic memory')
    parser.add_argument('--train_lr', action='store_true', help='train the learning rate')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_hal',type=float,default=0.01,help='learning rate for updating hallucinator')
    # EDSR
    parser.add_argument('--scale', type=str, default='4', help='super resolution scale')                                    # EDSR needs super-resolution setting
    parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
    parser.add_argument('--n_colors', type=int, default=1, help='number of color channels to use')
    parser.add_argument('--savef', action='store_true', help='save image results')                                          # store_true -> False (default)
    args = parser.parse_args()
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    eval_it_pool = np.arange(0, args.Iteration+1, 200).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)  # Evaluate every 200 iterations using num_eval randomly initialized networks trained on synthetic data.
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    if args.batch_syn is None:
        args.batch_syn = num_classes *args.vpc
    # record performances of all experiments
    psnrs_all_exps = dict() 
    ssims_all_exps = dict() 
    for key in model_eval_pool:
        psnrs_all_exps[key] = []
        ssims_all_exps[key] = []


    """
        Loop 0: run multiple experiments
    """
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        imgs_hr_all   = []
        labels_all    = []
        indices_class = [[] for c in range(num_classes)]
        for i in range(len(dst_train)):
            sample = dst_train[i]
            imgs_hr_all.append(torch.unsqueeze(sample[0], dim=0))   # [0, 1]
            labels_all.append(sample[1])                            # The subject ID to which the image belongs
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)                            # The data list corresponding to the subject
        imgs_hr_all = torch.cat(imgs_hr_all, dim=0).to(args.device)
        labels_all  = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(imgs_hr_all[:, ch]), torch.std(imgs_hr_all[:, ch])))
        
        def get_images(c, n, mode='random', step=5, start_idx=None):
            """
            Get n image slices for the specified category/subject (c).
            
            Args:
            c: int, Class/subject index
            n: int, Number of images/slices to retrieve
            mode: str, Sampling mode ('random': random slices, 'adjacent': consecutive slices, 'spaced': evenly spaced slices)
            step: int, Slice spacing (only used when mode='spaced')
            start_idx: int, Starting index. If None, choose a random starting point within the valid range.
            """
            valid_indices = np.sort(indices_class[c])                           # The indices of the current category are arranged in slice order (physical order)
            total_slices  = len(valid_indices)

            if mode == 'random':
                idx_selected = np.random.permutation(valid_indices)[:n]         # Randomly shuffle and take the first n
            elif mode == 'adjacent':
                if total_slices < n:
                    raise ValueError(f"Total number of slices ({total_slices}) for category {c} is less than the requested amount ({n})")
                if start_idx is None:
                    # Randomly generate a valid starting index to prevent slice out-of-bounds
                    max_start = total_slices - n
                    start_idx = np.random.randint(0, max_start + 1)             
                idx_selected = valid_indices[start_idx : start_idx+n]           # Take adjacent consecutive slices
            elif mode == 'spaced':
                span = (n - 1) * step + 1                                       # Calculate the required total span
                if total_slices < span:
                    raise ValueError(f"Category {c} does not have enough slices to get {n} images with a step size of {step}")
                if start_idx is None:
                    max_start = total_slices - span
                    start_idx = np.random.randint(0, max_start + 1)             # Randomly generate a valid starting index
                idx_selected = valid_indices[start_idx:start_idx+span:step]
            else:
                raise ValueError(f"Unsupported mode: {mode}. Please use 'random', 'adjacent', or 'spaced'")

            return imgs_hr_all[idx_selected]


        ''' initialize the synthetic data '''
        # Random initialization
        # static_syn = torch.rand(size=(args.spc, 1, im_size[0], im_size[1]), dtype=torch.float)    # The random range is [0, 1)
        
        # Initialize based on subject
        static_syn = torch.randn(size=(args.spc, 1, im_size[0], im_size[1]), dtype=torch.float)     # (spc, 1, 512, 512), original initialization
        static_syn.data[:args.spc] = get_images(0, args.spc).detach().data                          # Initialize static memory from subject 0; change the subject index to use subject 2.
        dynamic_syn = torch.randn(size=(num_classes, args.dpc), dtype=torch.float)                  # (num_class, dpc)
        
        # initialize the hallucinator
        hals   = nn.ModuleList([Conv2DNet(in_channel=args.spc*args.dpc, out_channel=args.vpc, img_size=im_size[0]) for _ in range(args.n_hal)])
        syn_lr = torch.tensor(args.lr_teacher)
        if args.path_static is not None:
            static_syn = torch.load(args.path_static)["image"]
            print('load static memory from %s' % args.path_static)
            print('static_syn shape: ', static_syn.shape)
        static_syn  = static_syn.detach().to(args.device).requires_grad_(False) if args.train_static else static_syn.detach().to(args.device).requires_grad_(True)
        dynamic_syn = dynamic_syn.detach().to(args.device).requires_grad_(True)
        hals        = hals.to(args.device)
        syn_lr      = syn_lr.detach().to(args.device).requires_grad_(args.train_lr)
        optimizer_static  = None if args.train_static else torch.optim.SGD([static_syn], lr=args.lr_static, momentum=0.95)
        optimizer_dynamic = torch.optim.SGD([dynamic_syn], lr=args.lr_dynamic, momentum=0.95)
        optimizer_hals    = torch.optim.SGD(hals.parameters(), lr=args.lr_hal, momentum=0.95)
        optimizer_lr      = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.9) if args.train_lr else None

        scale_factor = args.scale[0]
        if args.model in ('SRCNN', 'VDSR'):
            criterion = nn.MSELoss().to(args.device)
            transform = transforms.Compose([
                transforms.Resize((
                    int(im_size[0]/scale_factor), int(im_size[0]/scale_factor)
                )),
                transforms.Resize((
                    int(im_size[0]), int(im_size[0])
                )),
            ])
        elif args.model in ('EDSR'):
            criterion = nn.L1Loss().to(args.device)
            transform = transforms.Compose([
                transforms.Resize((
                    int(im_size[0]/scale_factor), int(im_size[0]/scale_factor)
                )),
            ])
        else:
            print("no matching transform")
        print('%s training begins' % get_time())

        """
            LOOP 1: randomly initialize the network
        """
        for it in range(args.Iteration+1):
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    psnrs = []
                    ssims = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, args, im_size).to(args.device)  # get a random model
                        static_syn_eval = copy.deepcopy(static_syn.detach())
                        dynamic_syn_eval = copy.deepcopy(dynamic_syn.detach())
                        hal_eval = copy.deepcopy(hals)
                        args.lr_net = syn_lr.detach()
                        _, psnr_train, ssim_train, psnr_test, ssim_test = evaluate_synset_s2d(it_eval, net_eval, [static_syn_eval, dynamic_syn_eval, hal_eval],None, testloader, args, transform, mode='multi-static')
                        psnrs.append(psnr_test)
                        ssims.append(ssim_test)
                    print('Evaluate %d random %s, psnr_mean = %.4f psnr_std = %.4f\n-------------------------'%(len(psnrs), model_eval, np.mean(psnrs), np.std(psnrs)))
                    print('Evaluate %d random %s, ssim_mean = %.4f ssim_std = %.4f\n-------------------------'%(len(ssims), model_eval, np.mean(ssims), np.std(ssims)))
                    if it == args.Iteration:  # record the final results
                        psnrs_all_exps[model_eval] += psnrs
                        ssims_all_exps[model_eval] += ssims
                ''' visualize and save '''
                with torch.no_grad():
                    image_save = static_syn.detach()
                    dynamic_save = dynamic_syn.detach()  # flatten(0, 1)
                    save_dir = args.save_path
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_save.cpu(),   os.path.join(save_dir, "exp{}_statics_{}.pt".format(exp, it)))
                    torch.save(hals.state_dict(),  os.path.join(save_dir, "exp{}_hal_{}.pt".format(exp, it)))
                    torch.save(dynamic_save.cpu(), os.path.join(save_dir, "exp{}_dynamic_{}.pt".format(exp, it)))

            ''' Train synthetic data '''
            net = get_network(args.model, channel, args, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net  = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()
            loss_avg = 0
            syn_static  = static_syn
            syn_dynamic = dynamic_syn
            """
                LOOP 2: network training
            """
            for ol in range(args.outer_loop):
                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.
                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): # BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                ''' update synthetic data '''
                hal = hals[0]
                loss = torch.tensor(0.0).to(args.device)
                """
                    LOOP 3: different subjects
                """
                for c in range(num_classes):
                    # real hr image and real lr image
                    img_hr_real = get_images(c, args.batch_real)  # torch.Size([batch_real, 1, 512, 512])
                    img_lr_real = transform(img_hr_real)

                    # syn hr image and syn lr image
                    img_hr_syn = hal(syn_static.unsqueeze(0), syn_dynamic[c])  # Add a dimension to simulate batch_size
                    # w/o fidlity
                    # img_hr_syn = torch.clamp(img_hr_syn, min=0, max=1)
                    # w fidlity Add a residual connection
                    img_hr_syn = torch.clamp(syn_static[c%args.spc]+img_hr_syn, min=0, max=1)
                    img_lr_syn = transform(img_hr_syn)

                    # cal real gw
                    img_output_real = net(img_lr_real)
                    loss_real = criterion(img_output_real, img_hr_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    # cal syn gw
                    img_output_syn = net(img_lr_syn)
                    loss_syn = criterion(img_output_syn, img_hr_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    # match loss
                    loss += match_loss(gw_syn, gw_real, args)
        
                if not args.train_static:
                    optimizer_static.zero_grad()
                optimizer_dynamic.zero_grad()
                optimizer_hals.zero_grad()
                if args.train_lr:
                    optimizer_lr.zero_grad()
                loss.backward()
                if not args.train_static:
                    optimizer_static.step()  
                optimizer_dynamic.step()
                optimizer_hals.step()
                if args.train_lr:
                    optimizer_lr.step()
                    syn_lr.data = syn_lr.data.clip(min=0.001)
                loss_avg += loss.item()
                if ol == args.outer_loop - 1:
                    break

                ''' update network '''
                static_syn_eval = copy.deepcopy(static_syn.detach())
                dynamic_syn_eval = copy.deepcopy(dynamic_syn.detach())
                hal_eval = copy.deepcopy(hals)
                images_train = [static_syn_eval, dynamic_syn_eval, hal_eval]
                dst_syn_train = MultiStaticSharedDataset(images_train[0], images_train[1], images_train[2])
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=False, transform=transform)
    
            loss_avg /= (num_classes*args.outer_loop)
            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        psnrs = psnrs_all_exps[key]
        ssims = ssims_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(psnrs), key, np.mean(psnrs), np.std(psnrs)))
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(ssims), key, np.mean(ssims)*100, np.std(ssims)*100))


if __name__ == '__main__':
    main()
