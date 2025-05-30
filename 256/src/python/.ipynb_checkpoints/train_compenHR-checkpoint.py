'''
Training and testing script for CompenHD (journal extension of cvpr'19 and iccv'19 papers)

This script trains/tests CompenHD on different dataset specified in 'data_list' below.
The detailed training options are given in 'train_option' below.

1. We start by setting the training environment to GPU (if any).
2. Setups are listed in 'data_list'.
3. We set number of training images to 500 and loss function to l1+l2+ssim. Other training options are specified in 'train_option'.
4. The training data 'train_data' and validation data 'valid_data', are loaded in RAM using function 'loadData'. 
5. The training and validation results are both updated in Visdom window (`http://server:8098`) and console.
6. Once the training is finished, we can compensate the desired image. The compensation images 'prj_cmp_test' can then be projected to the surface.

Example:
    python train_compenHD.py

See Models.py for CompenHD structure.
See trainNetwork.py for detailed training process.
See utils.py for helper functions.

Citation:

    @inproceedings{wang2023compenhd,
        author = {Yuxi Wang and Bingyao Huang and Haibin Ling},
        title = {CompenHR: Efficient Full Compensation for High-resolution Projector},
        booktitle = {IEEE Virtual Reality and 3D User Interfaces (VR)},
        month = {April},
        year = {2023} }


'''
if __name__ == '__main__':
    # %% Set environment
    import os

    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device_ids = [0]
    from thop import profile
    from torchstat import stat
    from torchsummary import summary
    from ptflops import get_model_complexity_info
    from fvcore.nn import FlopCountAnalysis,parameter_count_table
    from trainNetwork import *
    import Models
    # from RFDN import *
    # from rlfn import *
    from utils import *

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if torch.cuda.device_count() >= 1:
        print('Train with', len(device_ids), 'GPUs!')
    else:
        print('Train with CPU!')


    dataset_root = fullfile(os.getcwd(), '/mnt/data/huang-lab/mingjie/1024')
    data_list = [
        # 'bubble/1',
        # 'bubble/2',
        # 'bubble/3',
        # 'cloud/1',
        # 'cloud/2',
        # 'cloud/3',
        # 'cube/1',
        # 'cube/2',
        # 'cube/3',
        # 'curve/1',
        # 'curve/2',
        # 'curve/3',
        # 'lavender/1',
        # 'lavender/2',
        # 'lavender/3',
        # 'stripes/1',
        # 'stripes/2',
        # 'stripes/3',
        # 'water/1',
        # 'water/2',
        'stripes_np/1',
        'rock_np/1',
        'lemon_np/1',
        'leaf_np/1',
        'flower_np/1',
    ]
    loss_list = ['l1+l2+ssim']
    num_train_list = [500]
    model_list = ['CompenHD']

    # default training options
    train_option_default = {'max_iters': 2000,
                            'batch_size': 12,
                            'lr_cmp': 3e-3,  # learning rate
                            'lr_warp': 3e-3,  # learning rate
                            'lr_drop_ratio': 0.5,
                            'lr_drop_rate': 800,  # adjust this according to max_iters (lr_drop_rate < max_iters)
                            'loss': '',  # loss will be set to one of the loss functions in loss_list later
                            'l2_reg': 1e-4,  # l2 regularization
                            'device': device,
                            'pre-trained': True,
                            'pretrain_csr': '', # pre-trained model
                            'pretrain_cmp': '', # pre-trained photometric compensation model
                            'plot_on': True,  # plot training progress using visdom (disable for faster training)
                            'train_plot_rate': 100,  # training and visdom plot rate (increase for faster training)
                            'valid_rate': 100}  # validation and visdom plot rate (increase for faster training)

    # log file
    from time import localtime, strftime

    log_dir = '../../log'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file_name = strftime('%Y-%m-%d_%H_%M_%S', localtime()) + '.txt'
    log_file = open(fullfile(log_dir, log_file_name), 'w')
    title_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<30}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
    log_file.write(title_str.format('data_name', 'model_name', 'loss_function',
                                    'num_train', 'batch_size', 'max_iters','time_lapse',
                                    'uncmp_psnr', 'uncmp_rmse', 'uncmp_ssim','uncmp_diff',
                                    'valid_psnr', 'valid_rmse', 'valid_ssim','valid_diff','valid_speed'))
    log_file.close()

    # resize the input images if input_size is not None

    input_size = (1024,1024) # we can also use a low-res input to reduce memory usage and speed up training/testing with a sacrifice of precision

    resetRNGseed(0)
    # torch.manual_seed(3407)
    # create a CompenNeSt

    # stats for different setups
    for data_name in data_list:
        torch.cuda.empty_cache()
        # load training and validation data
        data_root = fullfile(dataset_root, data_name)
        cam_surf, cam_train, cam_valid, prj_train, prj_valid, mask_corners = loadData(dataset_root, data_name, input_size, CompenNeSt_only=False)

        # surface image for training and validation
        cam_surf_train = cam_surf.expand_as(cam_train)
        cam_surf_valid = cam_surf.expand_as(cam_valid)


        # validation data, 200 image pairs
        valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)

        # stats for different #Train
        for num_train in num_train_list:
            train_option = train_option_default.copy()
            train_option['num_train'] = num_train


            # select a subset to train
            train_data = dict(cam_surf=cam_surf_train[:num_train, :, :, :], cam_train=cam_train[:num_train, :, :, :],
                              prj_train=prj_train[:num_train, :, :, :])

            # stats for different models
            for model_name in model_list:

                train_option['model_name'] = model_name.replace('/', '_')
                # stats for different loss functions
                for loss in loss_list:
                    log_file = open(fullfile(log_dir, log_file_name), 'a')

                    # set seed of rng for repeatability
                    resetRNGseed(0)

                    # torch.manual_seed(3407)
                    # create a GANet
                    ga_net = Models.GANet(out_size=input_size)
                    # flops2, params2 = profile(ga_net, inputs=(input1,))
                    # print(flops2, params2)
                    # initialize GANet with affine transformation (remember grid_sample is inverse warp, so src is the the desired warp
                    src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
                    dst_pts = np.array(mask_corners[0][0:3]).astype(np.float32)
                    affine_mat = cv.getAffineTransform(src_pts, dst_pts)
                    ga_net.set_affine(affine_mat.flatten())
                    if torch.cuda.device_count() >= 1: ga_net = nn.DataParallel(ga_net, device_ids=device_ids).to(device)


                    if model_name == 'CompenHD':
                        pa_net = Models.PANet()
                        # flops1, params1 = profile(pa_net, inputs=(input1, input2,))
                        # print(flops1, params1)
                        if torch.cuda.device_count() >= 1: pa_net = nn.DataParallel(pa_net, device_ids=device_ids).to(device)
                        # sr_net = Models.SRNet()
                        # if torch.cuda.device_count() >= 1: sr_net = nn.DataParallel(sr_net, device_ids=device_ids).to(
                        #     device)
                        # model_path = os.path.join('/home/mingjie/code/code/CompenHR-main/src/python', 'epoch_300.pth')
                        # sr_net = RLFN()
                        # # sr_net.load_state_dict(torch.load(model_path), strict=True)
                        # model_dict = load_state_dict(model_path)
                        # sr_net.load_state_dict(model_dict['model_state_dict'], strict=True)
                        # sr_net.eval()
                        # for k, v in sr_net.named_parameters():
                        #     v.requires_grad = False
                        # # sr_net = sr_net.to('cuda')
                        # if torch.cuda.device_count() >= 1: sr_net = nn.DataParallel(sr_net, device_ids=device_ids).to(
                        #     device)

                        compen_hd = Models.CompenHD(ga_net, pa_net)

                        # if torch.cuda.device_count() >= 1: compen_hd = nn.DataParallel(compen_hd, device_ids=device_ids).to(device)
                        # def get_parameter_number(model):
                        #     total_num = sum(p.numel() for p in model.parameters())
                        #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        #     return {'Total': total_num, 'Trainable': trainable_num}
                        # result1 = get_parameter_number(compen_hd)
                        # flops=FlopCountAnalysis(compen_hd,(input1,input2,))
                        # print("Flops:",flops.total())
                        # print(parameter_count_table(compen_hd))
                        # input1 = torch.randn(4, 3, 1024, 1024).cuda()
                        # input2 = torch.randn(4, 3, 1024, 1024).cuda()
                        # flops, params = profile(compen_hd, inputs=(input1,input2,))
                        # print(flops, params)
                        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
                        # print('Params = ' + str(params / 1000 ** 2) + 'M')


                        if torch.cuda.device_count() >= 1: compen_hd = nn.DataParallel(compen_hd,device_ids=device_ids).to(device)
                        # input1 = torch.randn(1, 3, 1024, 1024).to(device)
                        # input2 = torch.randn(1, 3, 1024, 1024).to(device)
                        # summary(compen_hd, input_size=[[3, 1024,1024], [3,1024,1024]])
                        # flops = FlopCountAnalysis(compen_hd, (input1, input2,))
                        # print("Flops:", flops.total())
                        # print(parameter_count_table(compen_hd))

                        # flops, params = profile(compen_hd, inputs=(input1, input2,))
                        # print('flops: ', flops, 'params: ', params)
                        if train_option['pretrain_csr'] !='':
                            print(train_option['pretrain_csr'])
                            compen_hd.load_state_dict(torch.load(train_option['pretrain_csr']))



                    # train option for current configuration, i.e., data name and loss function
                    train_option['data_name'] = data_name.replace('/', '_')
                    train_option['loss'] = loss

                    print('-------------------------------------- Training Options -----------------------------------')
                    print('\n'.join('{}: {}'.format(k, v) for k, v in train_option.items()))
                    print('------------------------------------ Start training {:s} ---------------------------'.format(model_name))

                    # train model
                    compen_hd, valid_psnr, valid_rmse, valid_ssim,valid_diff,valid_speed,time_lapse = trainModel(compen_hd, train_data, valid_data, train_option)

                    uncmp_psnr = 0.0
                    uncmp_rmse=0.0
                    uncmp_ssim = 0.0
                    uncmp_diff = 0.0


                        # %%

                    # save results to log file
                    ret_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<30}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
                    log_file.write(ret_str.format(data_name, model_name, loss, num_train, train_option['batch_size'], train_option['max_iters'],time_lapse,
                                                  uncmp_psnr, uncmp_rmse, uncmp_ssim,uncmp_diff,
                                                  valid_psnr, valid_rmse ,valid_ssim,valid_diff,valid_speed))
                    log_file.close()

                    del compen_hd, ga_net
                    torch.cuda.empty_cache()
                    print('-------------------------------------- Done! ---------------------------\n')
            del train_data
        del cam_surf, cam_train, cam_valid, prj_train, prj_valid, mask_corners,valid_data

    print('All dataset done!')

