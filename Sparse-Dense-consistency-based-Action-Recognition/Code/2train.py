# Cosine_loss for semi, ramp_weight_epochwise
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time
import os
import numpy as np
from model import *
import parameters as params
import configuration as cfg
from clean_dlv2 import *
import sys, traceback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import cv2
from torch.utils.data import DataLoader
import math
import argparse
from jsd_loss_function import *

if torch.cuda.is_available(): 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_epoch(run_id, epoch, data_loader,num_training_steps, lw, model, criterion, criterion_b, optimizer, writer, use_cuda,lr_array):
    print('train at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr']=lr_array[epoch]
        writer.add_scalar('Learning Rate', lr_array[epoch], epoch)  

        print("Learning rate is: {}".format(param_group['lr']))
  
    losses, weighted_losses = [], []
    loss_mini_batch = 0
    self_sup_losses, semi_sup_losses, sup_losses = [], [], [] 
    optimizer.zero_grad()

    model.train()
   
    
    for i, (full_clip, trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path) in enumerate(data_loader):
        full_clip = full_clip.permute(0,4,1,2,3) # BS x c x frames x h x w
        trimmed_clip = trimmed_clip.permute(0,4,1,2,3)
        # print(f'label is {label}')
        if use_cuda:
            full_clip = full_clip.cuda()
            trimmed_clip = trimmed_clip.cuda()
            label = torch.from_numpy(np.asarray(label)).cuda()
        _, pretext_full, full_rep = model(full_clip)
        class_pred, pretext_trim, trim_rep = model(trimmed_clip)
    
        sup_loss = criterion(class_pred,label) # CAN BE EXTENDED TO FULL CLIP AND TRIMMED CLIP
        
        # self_sup_loss = 0.5*(criterion(pretext_trim, torch.zeros(len(lu_bit)).to(torch.long)) + criterion(pretext_full, torch.ones(len(lu_bit)).to(torch.long)))
        # self_sup_loss = 5*(criterion_b(pretext_trim, torch.zeros(len(lu_bit),1).to(torch.float)) + criterion_b(pretext_full, torch.ones(len(lu_bit),1).to(torch.float)))
        self_sup_loss = 0.5*(criterion_b(pretext_trim, torch.Tensor([0,1]).repeat(len(lu_bit),1).to(torch.float)) +\
                             criterion_b(pretext_full, torch.Tensor([1,0]).repeat(len(lu_bit),1).to(torch.float)))
        
        # Rep dimension [bs, 512, 4, 7, 7]
        semi_sup_loss = calc_consistency_loss(full_rep.permute(0,2,1,3,4), trim_rep.permute(0,2,1,3,4),
                                        temporal_span_full, temporal_span_trimmed,
                                        full_clip, trimmed_clip).cuda() 

        self_sup_losses.append(self_sup_loss.item())
        semi_sup_losses.append(semi_sup_loss.item())
        sup_losses.append(sup_loss.item())
        
        # lw= [sup_w, self_w, semi_w]

        loss = torch.mul(self_sup_loss, lw[1][epoch]) + \
                torch.mul(sup_loss, lw[0][epoch]) + \
                torch.mul(semi_sup_loss, lw[2][epoch])

        loss.backward()

        optimizer.step()

        losses.append(loss.item())
        if i % 100 == 0: 
            print(f'batch {i}')
            print(f'weight of sup, self, semi are {lw[0][epoch]},{lw[1][epoch]}, {lw[2][epoch]}')
            print(f'self_sup_loss is {self_sup_loss.item()}')
            print(f'sup_loss is {sup_loss.item()}')
            print(f'semi_sup_loss is {semi_sup_loss.item()}')
            print() 
        del trimmed_clip, full_clip, self_sup_loss, sup_loss, semi_sup_loss       

    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('Self_sup_losses', np.mean(self_sup_losses), epoch)
    writer.add_scalar('Semi_sup_losses', np.mean(semi_sup_losses), epoch)
    writer.add_scalar('Sup_losses', np.mean(sup_losses), epoch)


    print(f'epoch {epoch}')
    print(f'weight of sup, self, semi are {lw[0][epoch]},{lw[1][epoch]}, {lw[2][epoch]}')
    print(f'self_sup_loss is {np.mean(self_sup_losses)}')
    print(f'sup_loss is {np.mean(sup_losses)}')
    print(f'semi_sup_loss is {np.mean(semi_sup_losses)}')
    print() 


    del loss

    return model

def val_epoch(run_id, epoch, data_loader, model, criterion, writer, use_cuda):
    print('validation at epoch {}'.format(epoch))
    
    model.eval()

    losses = []
    predictions, ground_truth = [], []

    for i, (trimmed_clip, label, _, _, _, _) in enumerate(data_loader):
        inputs = trimmed_clip.permute(0,4,1,2,3)
        ground_truth.extend(label)

        if use_cuda:
            inputs = inputs.cuda()
            label = torch.from_numpy(np.asarray(label)).cuda()
        output, _, _ = model(inputs)

        loss = criterion(output,label)

        losses.append(loss.item())


        predictions.extend(output.cpu().data.numpy())

        if i % 100 == 0:
            print("Validation Epoch ", epoch , " Batch ", i, "- Loss : ", loss.item())
        
    del inputs, output, label, loss 

    ground_truth = np.asarray(ground_truth)
    c_pred = np.argmax(predictions,axis=1).reshape(len(predictions))
    print(c_pred[0])
    correct_count = np.sum(c_pred==ground_truth)
    print(f'Correct Count is {correct_count}')
    accuracy = float(correct_count)/len(c_pred)
    print(f'Accuracy for Epoch {epoch} is {accuracy*100 :.3f} %')
    results_actions = precision_recall_fscore_support(ground_truth, c_pred, average=None)
    f1_scores, precision, recall = results_actions[2], results_actions[0], results_actions[1]
    print(f'Epoch {epoch}, F1-Score {np.mean(f1_scores)}')
    print(f'Epoch {epoch}, Prec {np.mean(precision)}, Recall {np.mean(recall)}')

    print(f'Epoch {epoch} Precision {precision}')
    print(f'Epoch {epoch} Recall {recall}')




    writer.add_scalar('Validation Loss', np.mean(losses), epoch)
    writer.add_scalar('Validation F1-Score', np.mean(f1_scores), epoch)
    writer.add_scalar('Validation Precision', np.mean(precision), epoch)
    writer.add_scalar('Validation Recall', np.mean(recall), epoch)
    writer.add_scalar('Validation Accuracy', np.mean(accuracy), epoch)

    return np.mean(f1_scores)
    
def train_classifier(run_id):
    use_cuda = True
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = build_r2plus1d_model(num_classes=102)

    l=params.learning_rate
    l1= l
    l2= l/10

    # lr_array = [l*10 for kk in range(10)]+[l for kk in range(15)]+ [l/2 for kk in range(15)] + [l/10 for kk in range(200)]
    # lr_array = [l for kk in range(25)] + [l/2 for kk in range(25)]+ [l/5 for kk in range(25)]+ [l/10 for kk in range(130)]

    lr_array =  [kk*(l2-l1)/50 +l1 for kk in range(50)] + [l for kk in range(25)] + [kk*(l2-l1)/75 +l1 for kk in range(75)]
    fraction_array = [1 for kk in range(int(params.num_epochs/3))]+[0 for kk in range(int(params.num_epochs*2/3))]
    percent_array = [0.25 for kk in range(int(params.num_epochs/3))]+[1 for kk in range(int(params.num_epochs*2/3))]
    validation_array = [5*kk for kk in range(10)]+ [kk for kk in range(50,150)]

    print(lr_array[:50])



    criterion= nn.CrossEntropyLoss(ignore_index=-1)
    criterion_b = nn.BCELoss()

    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        model=nn.DataParallel(model)
        model.cuda()
        criterion.cuda()
        criterion_b.cuda()
    else:
        print('Only 1 GPU is available')
        model.cuda()
        criterion.cuda()
        criterion_b.cuda()

    optimizer=optim.Adam(model.parameters(),lr=params.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.80)

    train_dataset = pa3_train_dataset_generator(fraction=fraction_array[0], transform=None, shuffle=True, data_percentage=percent_array[0])
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn1)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
    num_training_steps=int(len(train_dataset)/params.batch_size)
    validation_dataset = pa3_test_dataset_generator(transform=None, shuffle=True, data_percentage=1.0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params.val_bs, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
    print(f'Validation dataset length: {len(validation_dataset)}')
    print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.val_bs}')

    sup_w, self_w, semi_w = ramp_weight_epochwise(int(params.num_epochs))
    lw = [sup_w, self_w, semi_w]


    for epoch in range(params.num_epochs):
        
        print(f'Epoch {epoch} started')
        start=time.time()
        try:
            model = train_epoch(run_id, epoch, train_dataloader,num_training_steps,lw, model, criterion, criterion_b, optimizer, writer, use_cuda,lr_array)
            if epoch in validation_array:
                validation_dataset = pa3_test_dataset_generator(transform=None, shuffle=True, data_percentage=1.0)
                validation_dataloader = DataLoader(validation_dataset, batch_size=params.val_bs, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
                f1_score=val_epoch(run_id, epoch, validation_dataloader, model, criterion, writer, use_cuda)
                if f1_score > best_score:
                    print('++++++++++++++++++++++++++++++')
                    print(f'Epoch {epoch} is the best model till now for {run_id}!')
                    print('++++++++++++++++++++++++++++++')
                    save_dir = os.path.join(cfg.saved_models_dir, run_id)
                    save_file_path = os.path.join(save_dir, 'model_{}_best_{}.pth'.format(epoch, f1_score))
                    states = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(states, save_file_path)
                    best_score = f1_score
                elif epoch % 5==0:
                    save_dir = os.path.join(cfg.saved_models_dir, run_id)
                    save_file_path = os.path.join(save_dir, 'model_e{}_{}.pth'.format(epoch, f1_score))
                    states = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                torch.save(states, save_file_path)

            # scheduler.step()
        except:
            print("Epoch ", epoch, " failed")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue

        train_dataset = pa3_train_dataset_generator(fraction=fraction_array[epoch+1], transform=None, shuffle=True, data_percentage=percent_array[epoch+1])
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn1)

        # validation_dataset = pa3_test_dataset_generator(transform=None, shuffle=False, data_percentage=0.1)
        # validation_dataloader = DataLoader(validation_dataset, batch_size=params.val_bs, shuffle=False, num_workers=params.num_workers, collate_fn=collate_fn2)
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=True,
                        help='run_id')
    args = parser.parse_args()
    run_id = args.run_id
    train_classifier(str(run_id))



        


