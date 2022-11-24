import time
from collections import deque
from itertools import chain
import numpy as np
import torch
import sys
import os
import copy
from scipy import stats
import torch.nn as nn
from src.utils import get_argparser
from src.encoders_ICA import NatureCNN

import pandas as pd
import datetime
from src.lstm_attn import subjLSTM
from src.All_Architecture import combinedModel

# import torchvision.models.resnet_conv1D as models
# from tensorboardX import SummaryWriter

# from src.pyg_class import pyg_data_creation, Net, Net2

from src.graph_the_works_fMRI import the_works_trainer



def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):



    start_time = time.time()










    # ID = args.script_ID + 3
    ID = args.script_ID - 1
    JobID = args.job_ID

    print(JobID)
    # return
    ID = 4
    print('ID = ' + str(ID))
    print('exp = ' + args.exp)
    print('pretraining = ' + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = '_' + str(JobID) + '_' + str(ID)

    Name = args.exp + 'HCP' + args.pre_training + 'DECENNT_Region'
    dir = 'run-' + d1 + d2 + Name
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)

    wdb1 = 'wandb_new'
    wpath1 = os.path.join(os.getcwd(), wdb1)


    p = 'UF'
    dir = 'run-2019-09-1223:36:31' + '-' + str(ID) + 'FPT_ICA_COBRE'
    p_path = os.path.join(os.getcwd(), p)
    p_path = os.path.join(p_path, dir) 
    args.p_path = p_path
    # os.mkdir(fig_path)
    # hf = h5py.File('../FBIRN_AllData.h5', 'w')
    tfilename = str(JobID) + 'HCP_region' + Name + str(ID)

    output_path = os.path.join(os.getcwd(), 'Output')
    output_path = os.path.join(output_path, tfilename)
    # output_text_file = open(output_path, "w+")
    # writer = SummaryWriter('exp-1')
    ntrials = args.ntrials
    ngtrials = 10
    best_auc = 0.
    best_gain = 0
    current_gain=0
    tr_sub_SZ = [15, 25, 50, 75, 425, 125] #425, 319, 142
    tr_sub_HC = [15, 25, 50, 75, 329, 125] #329, 247, 134



    # With 16 per sub val, 10 WS working, MILC default
    if args.exp == 'FPT':
        gain = [0.45, 0.05, 0.05, 0.15, 0.85]  # FPT
    elif args.exp == 'UFPT':
        gain = [3, 3, 3, 3, 3, 3]  # UFPT
    else:
        gain = [1, 1, 1, 1, 1.65, 1]  # NPT

    sub_per_class_SZ = tr_sub_SZ[ID]
    sub_per_class_HC = tr_sub_HC[ID]
    current_gain = gain[ID]
    args.gain = current_gain
    sample_x = 200
    sample_y = 1
    subjects = 942 # 752
    tc = 1200

    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 1200
    # samples_per_subject = int((tc - sample_y)+1)
    ntest_samples_perclass_SZ = 53 # 106#53
    ntest_samples_perclass_HC = 41 # 82#41
    if ID == 5:
        nval_samples_perclass = 15
    else:
        nval_samples_perclass_SZ = 53 # 106#53
        nval_samples_perclass_HC = 41 #82#41
    test_start_index = 0
    test_end_index = test_start_index + ntest_samples_perclass_SZ
    window_shift = 1

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        print(torch.cuda.device_count())
        device = torch.device("cuda:0")
        device_one = torch.device("cuda:1")
        device_two = torch.device("cuda:2")
        device_extra = torch.device("cuda:0")
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device_one = torch.device("cpu")
    print('device = ', device)



    n_good_comp = 53
    n_regions = 200


    # data = []
    with open('../DataandLabels/HCP_fMRI_200ShaeferAtlas_onlytimeserieszscored.npz', 'rb') as file:
        data1=(np.load(file))

    with open('../DataandLabels/HCP_fMRI_remaining190_200ShaeferAtlas_onlytimeserieszscored.npz', 'rb') as file:
        data2=(np.load(file))

    data = np.concatenate((data1,data2),axis=0)
    print('combined data shape ', data.shape)

    # filename = '../DataandLabels/HCP_indices_752.csv'
    # df = pd.read_csv(filename, header=None, dtype=int)
    # indices_752 = (df.values).reshape(-1).nonzero()[0]
    # indices_752 = indices_752.reshape(-1)
    #
    # print('752 shape ', indices_752.shape)
    #
    # filename = '../DataandLabels/HCP_indices_190.csv'
    # df = pd.read_csv(filename, header=None, dtype=int)
    # indices_190 = (df.values).reshape(-1).nonzero()[0]
    # indices_190 = indices_190.reshape(-1)
    #
    # print('190 shape ', indices_190.shape)
    # data = np.zeros((subjects, n_regions, tc))
    #
    # data[indices_752, :, :] = data1
    # data[indices_190, :, :] = data2
    #
    # print('combined data shape ', data.shape)




    # data = np.concatenate(data,axis=0)
    # print('shape is', data.shape)
    # with open('../DataandLabels/HCP_fMRI_200ShaeferAtlas_onlytimeserieszscored.npz', 'wb') as filesim:
    #     np.save(filesim, data)
    #
    # return






    data[data != data] = 0
    # for t in range(subjects):
    #     for r in range(n_regions):
    #         data[t, r, :] = stats.zscore(data[t, r, :])
    data = torch.from_numpy(data).float()
    finalData = np.zeros((subjects, samples_per_subject, n_regions, sample_y))
    for i in range(subjects):
        for j in range(samples_per_subject):
            #if j != samples_per_subject-1:
            finalData[i, j, :, :] = data[i, :, (j * window_shift):(j * window_shift) + sample_y]
            #else:
                #finalData[i, j, :, :17] = data[i, :, (j * window_shift):]


    finalData2 = torch.from_numpy(finalData).float()
    # selected = np.arange(311) != 73
    # finalData2 = finalData2[selected,:,:,:]#torch.cat((finalData2[0:73,:,:,:], finalData2[74:,:,:,:]), dim=0)
    finalData2[finalData2 != finalData2] = 0

    start_path = '../Atlases2'
    count = 0;

    # print('starting to read data')
    # with open('../fMRI/FBIRN/fMRI_complete_FBIRN.npz', 'rb') as file:
    #     finalData2 = np.load(file)
    # print('data loaded')
    # finalData2 = torch.from_numpy(finalData2).float()
    # finalData2 = finalData2.permute(0,4,1,2,3)
    # region_data = np.zeros((311,116,140))
    # print(finalData2.shape)
    # index = 0
    # size = 0
    # for dirpath, dirnames, filenames in os.walk(start_path):
    #     for f in sorted(filenames):
    #
    #         img = nib.load(dirpath +'/'+f)
    #         # img_np = np.array(img.dataobj)
    #         img_np = img.get_fdata(caching='unchanged')
    #         img_np = torch.from_numpy(img_np)
    #         size = size + (img_np != 0).nonzero().shape[0]
    #
    #         img_np = img_np != 0
    #         output = torch.masked_select(finalData2, img_np).reshape(311, 140, -1)
    #         output = output.sum(dim=2)
    #         region_data[:, index, :] = output
    #         index = index + 1
    #         print('index = ',index)
    #
    # print('size = ', size)
    # with open('../fMRI/FBIRN/fMRI_complete_atlas_regions_116.npz', 'wb') as filesim:
    #     np.save(filesim, region_data)
    # print('file saved')
    # return

    # filename = '../DataandLabels/correct_indices_GSP.csv'
    # df = pd.read_csv(filename, header=None)
    # c_indices = df.values
    # c_indices = torch.from_numpy(c_indices).int()
    # c_indices = c_indices.view(53)
    # c_indices = c_indices - 1
    # finalData2 = finalData2[:, :, c_indices.long(), :]
    # n_regions = 53





    # filename = '../DataandLabels/index_array_labelled_HCP_Gender.csv'
    # df = pd.read_csv(filename, header=None)
    # index_array = df.values
    # index_array = torch.from_numpy(index_array).long()
    # index_array = index_array.view(subjects)

    filename = '../DataandLabels/HCPlabelsIhave.csv'
    df = pd.read_csv(filename, header=None)
    all_labels1 = df.values
    all_labels1 = torch.from_numpy(all_labels1).int()
    all_labels1 = all_labels1.view(752)

    filename2 = '../DataandLabels/labels_HCP_remaining_190_subjects.csv'
    df2 = pd.read_csv(filename2, header=None)
    all_labels2 = df2.values
    all_labels2 = torch.from_numpy(all_labels2).int()
    all_labels2 = all_labels2.view(190)

    all_labels = torch.cat((all_labels1,all_labels2),dim=0).int()
    all_labels = all_labels.view(subjects)

    # all_labels = np.zeros((subjects))
    #
    # all_labels[indices_752] = all_labels1
    # all_labels[indices_190] = all_labels2
    # all_labels = torch.from_numpy(all_labels.reshape(subjects)).int()
    #
    # print('combined label shape ', all_labels.shape)

    # return



    # all_labels = all_labels - 1
    # all_labels = all_labels[selected]

    #finalData2 = finalData2[]
#    index_array = torch.randperm(311)
#     finalData2 = finalData2[:, :1200, :,:]
#     all_labels = all_labels[index_array]
    tc = 1200
    print(finalData2.shape)
    # return

    # finalData2_copy = torch.clone(finalData2)

    # finalData2_copy = torch.squeeze(finalData2_copy)
    # finalData2_copy = finalData2_copy.permute(0,2,1)
    # cor, rho = stats.spearmanr(finalData2[0,0,:,:],axis=1)
    # print(cor.shape)
    # return

    # FNC = np.zeros((subjects - 1, 4950))  # 6670
    # corrM = np.zeros((subjects-1, n_regions, n_regions))
    # for i in range(subjects - 1):
    #     corrM[i, :, :] = np.corrcoef(finalData2_copy[i])
    #     M = corrM[i, :, :]
    #     FNC[i, :] = M[np.triu_indices(n_regions, k=1)]
    # corrM = torch.from_numpy(corrM).float()
    # print (corrM[0,0,:])
    # print(FNC.shape)

    # report = poly(FNC, all_labels, n_folds=19, exclude=['RBF SVM'])
    # # Plot results
    # report.plot_scores()
    # return

    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)
    # test_indices_HC = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136]
    # test_indices_SZ = [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 151]

    # test_indices_HC = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    # test_indices_SZ = [0, 34, 68, 102, 136, 170, 204, 238, 272, 306, 340, 374, 408]

    # test_indices_HC = [0, 25, 50, 75, 100, 125]
    # test_indices_SZ = [0, 26, 52, 78, 104, 130]

    # test_indices_HC = [0, 78, 156, 234]
    # test_indices_SZ = [0, 110, 220, 330]

    # test_indices_HC = [0, 31, 62, 93, 124, 155, 186, 217, 248, 279]
    # test_indices_SZ = [0, 44, 88, 132, 176, 220, 264, 308, 352, 396]

    test_indices_HC = [0, 41, 82, 123, 164, 205, 246, 287, 328, 369]
    test_indices_SZ = [0, 53, 106, 159, 212, 265, 318, 371, 424, 477]

    # test_indices_HC = [0, 82, 164, 246, 328]
    # test_indices_SZ = [0, 106, 212, 318, 424]


    #
    # test_indices = [0, 16, 32, 48, 64, 80, 96, 112, 128]

    # dataset = TensorDataset(finalData2, all_labels)
    #
    # dataset = pyg_data_creation('../', dataset, file_name="FBIRN_region_pyg_dataset")
    #
    # dataset = dataset.shuffle()
    # train_dataset = dataset[:200]
    # val_dataset = dataset[200:232]
    # test_dataset = dataset[232:296]
    # print(len(train_dataset))
    # print(len(val_dataset))
    # print(len(test_dataset))
    #
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    # val_loader = DataLoader(val_dataset, batch_size=32)
    # test_loader = DataLoader(test_dataset, batch_size=64)
    #
    # model = Net()
    #
    # trainer = graph_trainer(model,device)
    # trainer.start_train(train_loader,val_loader,test_loader,args.epochs)
    #

    number_of_test_folds_to_run = args.n_test_folds_to_run

    output_text_file = open(output_path, "a+")
    output_text_file.write("number of CV set = %d nTrial = %d\r\n" % (number_of_test_folds_to_run, ntrials))
    output_text_file.close()


    n_regions_output = n_regions
    tc_after_encoder = 1200
    HC_index, SZ_index = find_indices_of_each_class(all_labels)
    print(HC_index.shape)
    print(SZ_index.shape)
    # return
    total_test_size = ntest_samples_perclass_HC + ntest_samples_perclass_SZ
    results = torch.zeros(ntrials * number_of_test_folds_to_run, 6)
    # adjacency_matrices_FNC = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size, n_regions_output,
    #                                          n_regions_output)
    adjacency_matrices_learned = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size, n_regions_output,
                                             n_regions_output)
    # LR_top_adjacency_matrices_learned = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size, n_regions_output, n_regions_output)
    # LR_bottom_adjacency_matrices_learned = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size, n_regions_output, n_regions_output)
    #
    adjacency_matrices_learned_sum = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size, n_regions_output,
                                                 n_regions_output)
    # attention_region = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size, n_regions_output,
    #                                              tc_after_encoder)
    attention_time = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size, tc_after_encoder)
    result_targets = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size)
    # attention_components = torch.zeros(ntrials * number_of_test_folds_to_run, total_test_size, n_regions_output)

    # attention_time_embedding = torch.zeros(ntrials * number_of_test_folds_to_run, ntest_samples_perclass * 2, tc_after_encoder)
    # test_targets = torch.zeros(ntrials * number_of_test_folds_to_run, ntest_samples_perclass * 2)
    # test_pred = torch.zeros(ntrials * number_of_test_folds_to_run, ntest_samples_perclass * 2)
    # regions_selected = torch.zeros(ntrials * number_of_test_folds_to_run, ntest_samples_perclass * 2 * 13) # 23 is the number of regions left after last pooling layer
    result_counter = 0
    for test_ID in range(number_of_test_folds_to_run):
        # test_ID = 1
        # index_array = torch.randperm(311)
        # finalData2 = finalData2[index_array, :, :, :]
        # all_labels = all_labels[index_array]
        # if test_ID == 0:
        #     test_ID = test_ID + 0
        # else:
        test_ID = test_ID  + args.starting_test_fold
        print('test Id =', test_ID)

        test_start_index_SZ = test_indices_SZ[test_ID]
        test_start_index_HC = test_indices_HC[test_ID]
        test_end_index_SZ = test_start_index_SZ + ntest_samples_perclass_SZ
        test_end_index_HC = test_start_index_HC + ntest_samples_perclass_HC
        total_HC_index_tr_val = torch.cat([HC_index[:test_start_index_HC], HC_index[test_end_index_HC:]])
        total_SZ_index_tr_val = torch.cat([SZ_index[:test_start_index_SZ], SZ_index[test_end_index_SZ:]])

        HC_index_test = HC_index[test_start_index_HC:test_end_index_HC]
        SZ_index_test = SZ_index[test_start_index_SZ:test_end_index_SZ]

        total_HC_index_tr = total_HC_index_tr_val[:(total_HC_index_tr_val.shape[0] - nval_samples_perclass_HC)]
        total_SZ_index_tr = total_SZ_index_tr_val[:(total_SZ_index_tr_val.shape[0] - nval_samples_perclass_SZ)]

        HC_index_val = total_HC_index_tr_val[(total_HC_index_tr_val.shape[0] - nval_samples_perclass_HC):]
        SZ_index_val = total_SZ_index_tr_val[(total_SZ_index_tr_val.shape[0] - nval_samples_perclass_SZ):]

        auc_arr = torch.zeros(ngtrials, 1)
        avg_auc = 0.
        for trial in range(ntrials):
                print ('trial = ', trial)

            # writer.add_scalar('trial', trial)
            # current_gain = (trial+1) * 0.05
            # args.gain = current_gain
            # for g_trial in range (ngtrials):
                g_trial=1
                output_text_file = open(output_path, "a+")
                output_text_file.write("Test fold number = %d Trial = %d\r\n" % (test_ID,trial))
                output_text_file.close()
                # Get subject_per_class number of random values
                HC_random = torch.randperm(total_HC_index_tr.shape[0])
                SZ_random = torch.randperm(total_SZ_index_tr.shape[0])
                HC_random = HC_random[:sub_per_class_HC]
                SZ_random = SZ_random[:sub_per_class_SZ]
                # HC_random = torch.randint(high=len(total_HC_index_tr), size=(sub_per_class,))
                # SZ_random = torch.randint(high=len(total_SZ_index_tr), size=(sub_per_class,))
                #

                # Choose the subject_per_class indices from HC_index_val and SZ_index_val using random numbers

                HC_index_tr = total_HC_index_tr[HC_random]
                SZ_index_tr = total_SZ_index_tr[SZ_random]

                # ID = ID * ntest_samples
                # val_indexs = ID-1;
                # val_indexe = ID+200

                tr_index = torch.cat((HC_index_tr, SZ_index_tr))
                val_index = torch.cat((HC_index_val, SZ_index_val))
                test_index = torch.cat((HC_index_test, SZ_index_test))

                tr_index = tr_index.view(tr_index.size(0))
                val_index = val_index.view(val_index.size(0))
                test_index = test_index.view(test_index.size(0))

                # tr_eps = finalData2[0:200, :, :, :]
                # val_eps = finalData2[200:280, :, :, :]
                # test_eps = finalData2[280:296, :, :, :]
                #
                # tr_labels = all_labels[0:200]
                # val_labels = all_labels[200:280]
                # test_labels = all_labels[280:296]


                tr_eps = finalData2[tr_index.long(), :, :, :]
                val_eps = finalData2[val_index.long(), :, :, :]
                test_eps = finalData2[test_index.long(), :, :, :]
                # indexx = torch.tensor(np.array([0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]))
                # test_eps = test_eps[indexx.long(), :, :, :]

                # tr_eps2 = finalData2_copy[tr_index.long(), :, :, :]
                # val_eps2 = finalData2_copy[val_index.long(), :, :, :]
                # test_eps2 = finalData2_copy[test_index.long(), :, :, :]



                # tr_FNC = corrM[tr_index.long(), :, :]
                # val_FNC = corrM[val_index.long(), :, :]
                # test_FNC = corrM[test_index.long(), :, :]

                tr_labels = all_labels[tr_index.long()]
                val_labels = all_labels[val_index.long()]
                test_labels = all_labels[test_index.long()]
                # test_labels = test_labels[indexx.long()]



                #tr_eps = torch.from_numpy(tr_eps).float()
                #val_eps = torch.from_numpy(val_eps).float()
                #test_eps = torch.from_numpy(test_eps).float()

                # tr_labelsf = torch.cat((tr_labels, tr_labels))
                # val_labelsf = torch.cat((val_labels, val_labels))
                # test_labelsf = torch.cat((test_labels, test_labels))

                tr_labels = tr_labels.to(device)
                val_labels = val_labels.to(device)
                test_labels = test_labels.to(device)



                # tr_eps = tr_eps.to(device)
                # val_eps = val_eps.to(device)
                # test_eps = test_eps.to(device)

                # tr_eps = torch.cat((tr_eps,tr_eps),dim=0)
                # val_eps = torch.cat((val_eps, val_eps),dim=0)
                # test_eps = torch.cat((test_eps, test_eps),dim=0)

                # tr_eps = torch.cat((tr_eps, tr_eps),dim=0)
                # val_eps = torch.cat((val_eps, val_eps),dim=0)
                # test_eps = torch.cat((test_eps, test_eps),dim=0)


                # tr_epsf = torch.cat((tr_eps, tr_eps2),dim=0)
                # val_epsf = torch.cat((val_eps, val_eps2),dim=0)
                # test_epsf = torch.cat((test_eps, test_eps2),dim=0)

                # tr_epsf = tr_epsf.permute(0,2,1,3).contiguous().reshape(292*100,160,1)



                # tr_eps = tr_eps.to(device)
                # val_eps = val_eps.to(device)
                # test_eps = test_eps.to(device)

                # print(tr_epsf.shape)
                # tr_epsf = tr_epsf.permute(0, 2, 1, 3).contiguous().reshape(292 * 100, 160, 1)
                # packed = tn.pack_sequence(tr_epsf, enforce_sorted=False)
                # return

                print(tr_eps.shape)
                print(val_eps.shape)
                print(test_eps.shape)

                print(tr_labels.shape)
                print(val_labels.shape)
                print(test_labels.shape)

                # tr_FNC = tr_FNC.to(device)
                # val_FNC = val_FNC.to(device)
                # test_FNC = test_FNC.to(device)



                # asdfasdf = tr_eps.to(device)
                # print('tr2', tr_eps.device, tr_eps.dtype, type(tr_eps), tr_eps.type())
                # print('asdf',asdfasdf.device, asdfasdf.dtype, type(asdfasdf), asdfasdf.type())
                # return;
                # print("index_arrayshape", index_array.shape)
                #     print("trainershape", tr_eps.shape)
                #     print("valshape", val_eps.shape)
                #     print("testshape", test_eps.shape)
                #     print('ID = ', args.script_ID)

                # print(tr_labels)
                # print(test_labels)


                observation_shape = finalData2.shape
                L=""
                lmax=""
                number_of_graph_channels = 1
                if args.model_type == "graph_the_works":
                    encoder = NatureCNN(1, args)
                    encoder.to(device_two)
                    lstm_model = subjLSTM(device, sample_y, args.lstm_size, num_layers=args.lstm_layers,
                                          freeze_embeddings=True, gain=current_gain,bidrection=True)
                    # lstm_model_within_window = subjLSTM(device, 1, args.lstm_size_within_window, num_layers=args.lstm_layers,
                    #                       freeze_embeddings=True, gain=current_gain,bidrection=False)
                    # graph_model = Net()
                    # graph2=Net2()
                    # graph_list = nn.ModuleList(
                    #     [Net(device) for _ in range(number_of_graph_channels)])
                    dir = ""
                    lstm_model.to(device)
                    if args.pre_training == "milc-fMRI":
                        dir = 'Pre_Trained/BrainGNNLSTMG2Gx4tox6/model.pt'
                        args.oldpath = wpath1 + '/Pre_Trained/BrainGNNLSTMG2Gx4tox6'





                complete_model = combinedModel(encoder,lstm_model, samples_per_subject, gain=current_gain, PT=args.pre_training, exp=args.exp, device_one=device_one, oldpath=args.oldpath,n_regions=n_regions,device_two=device_two,device_zero=device,device_extra=device_extra )

                # complete_model.to(device)
                # if args.exp in ['UFPT', 'FPT']:
                #    model_dict = torch.load(path, map_location=device)
                #    complete_model.load_state_dict(model_dict)

                # torch.set_num_threads(1)
                config = {}
                config.update(vars(args))
                # print("trainershape", os.path.join(wandb.run.dir, config['env_name'] + '.pt'))
                config['obs_space'] = observation_shape  # weird hack
                if args.method == "graph_the_works":
                    trainer = the_works_trainer(complete_model, config, device=device_two, device_encoder=device,
                                                tr_labels=tr_labels,
                                          val_labels=val_labels, test_labels=test_labels, trial=str(trial),
                                                crossv=str(test_ID),gtrial=str(g_trial))

                else:
                    assert False, "method {} has no trainer".format(args.method)
                # xindex = (ntrials * test_ID) + trial
                results[result_counter][0], results[result_counter][1], results[result_counter][2], \
                results[result_counter][3],results[result_counter][4],\
                results[result_counter][5] = trainer.train(tr_eps, val_eps, test_eps)
                result_counter = result_counter + 1
                tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
                np.savetxt(tresult_csv, results.numpy(), delimiter=",")

# , LR_top_adjacency_matrices_learned[result_counter, :, :, :], \
#   LR_bottom_adjacency_matrices_learned[result_counter, :, :, :]
            # auc_arr[g_trial] = trainer.train(tr_eps, val_eps, test_eps)
            # return

        # avg_auc = auc_arr.mean()
        # if avg_auc > best_auc:
        #   best_auc = avg_auc
        #   best_gain = current_gain
    np_results = results.numpy()
    auc = np_results[:,1]
    print(np.mean(auc[:]))
    # np_result_targets = result_targets.numpy()

    # np_adjacency_matrices = adjacency_matrices_learned.numpy()
    # np_LR_top_adjacency_matrices_learned = LR_top_adjacency_matrices_learned.numpy()
    # np_LR_bottom_adjacency_matrices_learned = LR_bottom_adjacency_matrices_learned.numpy()
    # np_adjacency_matrices_sum = adjacency_matrices_learned_sum.numpy()
    # np_attention_region = attention_region.numpy()
    # np_attention_time = attention_time.numpy()
    # np_attention_components = attention_components.numpy()
    # np_attention_time_embedding = attention_time_embedding.numpy()
    # np_adjacency_matrices_FNC = adjacency_matrices_FNC.numpy()
    # np_test_targets = test_targets.numpy()
    # np_test_pred = test_pred.numpy()
    # np_regions_selected = regions_selected.numpy()
    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")

    # with open('../fMRI/HCP/testtargets'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_result_targets)
    #
    # with open('../fMRI/HCP/adjacencymatrix'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_adjacency_matrices)
    # with open('../fMRI/FBIRN/AdjacencyMatrices/LR/np_top_adjacency_matrices_learned'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_LR_top_adjacency_matrices_learned)
    # with open('../fMRI/FBIRN/AdjacencyMatrices/LR/np_bottom_adjacency_matrices_learned'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_LR_bottom_adjacency_matrices_learned)
    #
    # with open('../fMRI/FBIRN/AdjacencyMatrices/LR/adjacencymatrix_sum'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_adjacency_matrices_sum)

    # with open('../fMRI/FBIRN/AdjacencyMatrices/np_attention_region'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_attention_region)

    # with open('../fMRI/FBIRN/AdjacencyMatrices/LR/np_attention_time'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_attention_time)

    # with open('../fMRI/FBIRN/AdjacencyMatrices/np_attention_components'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_attention_components)

    # with open('../fMRI/FBIRN/AdjacencyMatrices/np_attention_time_embedding'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_attention_time_embedding)
    # with open('../fMRI/FBIRN/AdjacencyMatrices/adjacencymatrix_FNC'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_adjacency_matrices_FNC)
    # with open('../fMRI/FBIRN/AdjacencyMatrices/testtargets'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_test_targets)
    # with open('../fMRI/FBIRN/AdjacencyMatrices/testpred'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_test_pred)
    # with open('../fMRI/FBIRN/AdjacencyMatrices/regions'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_regions_selected)

    #
    # return encoder
    # print ('best gain = ', best_gain)
    # output_text_file = open(output_path, "a+")
    # output_text_file.write("best gain = %f \r\n" % (best_gain))
    # output_text_file.close()
    elapsed = time.time() - start_time
    print('total time = ', elapsed);


if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = "1"
    # torch.manual_seed(33)
    # np.random.seed(33)
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
