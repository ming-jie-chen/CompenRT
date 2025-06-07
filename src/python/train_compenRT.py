'''
Training and testing script for CompenRT

This script trains/tests CompenRT on different dataset specified in 'data_list' below.
The detailed training options are given in 'train_option' below.

1. We start by setting the training environment to GPU (if any).
2. Setups are listed in 'data_list'.
3. We set number of training images to 500 and loss function to l1+l2+ssim+diff. Other training options are specified in 'train_option'.
4. The training data 'train_data' and validation data 'valid_data', are loaded in RAM using function 'loadData'. 
5. The training and validation results are both updated in Visdom window (`http://server:8098`) and console.
6. Once the training is finished, we can compensate the desired image. The compensation images 'prj_cmp_test' can then be projected to the surface.

Example:
    python train_compenRT.py

See Models.py for CompenRT structure.
See trainNetwork.py for detailed training process.
See utils.py for helper functions.




'''
if __name__ == '__main__':
    # %% Set environment
    import os
    from profile1 import *

    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device_ids = [0]
    from trainNetwork import *
    import Models
    from utils import *

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        print('Train with', len(device_ids), 'GPUs!')
    else:
        print('Train with CPU!')


    dataset_root = fullfile(os.getcwd(), '../../data/1024')
    data_list = [
        'bubble/1',
        'bubble/2',
        'bubble/3',
        'cloud/1',
        'cloud/2',
        'cloud/3',
        'cube/1',
        'cube/2',
        'cube/3',
        'curve/1',
        'curve/2',
        'curve/3',
        'lavender/1',
        'lavender/2',
        'lavender/3',
        'stripes/1',
        'stripes/2',
        'stripes/3',
        'water/1',
        'water/2',
    ]
    loss_list = ['l1+l2+ssim+diff']
    num_train_list = [500]
    model_list = ['CompenRT (256->1024)']
    #model_list = ['CompenRT (512->1024)']

    # default training options
    train_option_default = {'max_iters': 2000,
                            'batch_size': 4,
                            'lr_cmp': 1e-3,  # learning rate
                            'lr_warp': 1e-3,  # learning rate
                            'lr_dis':1e-3,
                            'lr_drop_ratio': 0.5,
                            'lr_drop_rate': 900,  # adjust this according to max_iters (lr_drop_rate < max_iters)
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
    title_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<30}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
    log_file.write(title_str.format('data_name', 'model_name', 'loss_function',
                                    'num_train', 'batch_size', 'max_iters','time_lapse',
                                    'uncmp_psnr', 'uncmp_rmse', 'uncmp_ssim','uncmp_diff',
                                    'valid_psnr', 'valid_rmse', 'valid_ssim','valid_diff','valid_lpips','valid_speed'))
    log_file.close()

    # resize the input images if input_size is not None

    input_size = (1024,1024) # we can also use a low-res input to reduce memory usage and speed up training/testing with a sacrifice of precision
    input_lr_size= (256,256)
    resetRNGseed(0)


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
                    # create a GDNet
                    gd_net = Models.GDNet(out_size=input_lr_size) #The outsize parameter is used to adjust the size of the output resolution of the geometry correction.
                    # initialize GDNet with affine transformation (remember grid_sample is inverse warp, so src is the the desired warp
                    src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
                    dst_pts = np.array(mask_corners[0][0:3]).astype(np.float32)
                    affine_mat = cv.getAffineTransform(src_pts, dst_pts)
                    gd_net.set_affine(affine_mat.flatten())
                    if torch.cuda.device_count() >= 1: gd_net = nn.DataParallel(gd_net, device_ids=device_ids).to(device)

                    #  CompenRT (256->1024)
                    if model_name == 'CompenRT (256->1024)':
                        pu_net = Models.PUNet1()
                        if torch.cuda.device_count() >= 1: pu_net = nn.DataParallel(pu_net, device_ids=device_ids).to(device)


                        compen_rt = Models.CompenRTFast(gd_net, pu_net)
                    #  CompenRT (512->1024)
                    if model_name == 'CompenRT (512->1024)':
                        pu_net = Models.PUNet2()
                        if torch.cuda.device_count() >= 1: pu_net = nn.DataParallel(pu_net, device_ids=device_ids).to(device)


                        compen_rt = Models.CompenRT(gd_net, pu_net)
                    if torch.cuda.device_count() >= 1: compen_rt = nn.DataParallel(compen_rt,device_ids=device_ids).to(device)

                    if train_option['pretrain_csr'] !='':
                        print(train_option['pretrain_csr'])
                        compen_rt.load_state_dict(torch.load(train_option['pretrain_csr']))


                    # train option for current configuration, i.e., data name and loss function
                    train_option['data_name'] = data_name.replace('/', '_')
                    train_option['loss'] = loss

                    print('-------------------------------------- Training Options -----------------------------------')
                    print('\n'.join('{}: {}'.format(k, v) for k, v in train_option.items()))
                    print('------------------------------------ Start training {:s} ---------------------------'.format(model_name))

                    # train model
                    compen_rt, valid_psnr, valid_rmse, valid_ssim,valid_diff,valid_speed,valid_lpips,time_lapse = trainModel(compen_rt, train_data, valid_data, train_option)

                    uncmp_psnr = 0.0
                    uncmp_rmse=0.0
                    uncmp_ssim = 0.0
                    uncmp_diff = 0.0


                        # %%

                    # save results to log file
                    ret_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<30}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
                    log_file.write(ret_str.format(data_name, model_name, loss, num_train, train_option['batch_size'], train_option['max_iters'],time_lapse,
                                                  uncmp_psnr, uncmp_rmse, uncmp_ssim,uncmp_diff,
                                                  valid_psnr, valid_rmse ,valid_ssim,valid_diff,valid_lpips,valid_speed))
                    log_file.close()

                    del compen_rt, gd_net
                    torch.cuda.empty_cache()
                    print('-------------------------------------- Done! ---------------------------\n')
            del train_data
        del cam_surf, cam_train, cam_valid, prj_train, prj_valid, mask_corners,valid_data

    print('All dataset done!')

