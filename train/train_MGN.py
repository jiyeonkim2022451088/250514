import torch
import torch.nn.functional as F
import numpy as np
from models.MGN import MGN_shared
from models.MGN_UCVME import MGN_shared_UCVME
from models.MGN_TWIN import MGN_TWIN
from models.Common import Decoder_F, Decoder_G 
from tqdm import tqdm
#from torch_geometric.nn.unpool import knn_interpolate
from utils.knn_interpolate import knn_interpolate
import os
from torch.cuda.amp import autocast, GradScaler




def train_MGN_comp(device, train_data1, train_data2, test_data, dir, num_exp=5, y_input_size=1, pos_input_size=2, edge_input_size=4, hidden_size=30, output_size=1, depth=3,  residual=True, ib_n=True, ib_e=True, verbose=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    scaler = GradScaler()
    for i in range(num_exp):

        model_shared=MGN_shared(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        model_F= Decoder_F(hidden_size, output_size, residual).to(device)
        model_G= Decoder_G(hidden_size, output_size, residual).to(device)
        optimizer=torch.optim.Adam(list(model_shared.parameters())+list(model_F.parameters())+list(model_G.parameters()), lr=1e-3)
        
        loss_list=[]
        test_loss_list=[]

       
    
        for epoch in tqdm(range(5000)):
            mean_loss=[]
            for _ in (range(len(train_data2)+len(train_data1))):
           
                with autocast():

                    idx3=np.random.choice(list(range(len(train_data2))),1)[0]
                    idx12=np.random.choice(list(range(len(train_data1))),2,replace=False)
                    idx1=idx12[0]
                    idx2=idx12[1]
                        
                    l_pos1, l_y1, l_e1, h_pos1, h_e1, y1=train_data1[idx1]
                    l_pos2, l_y2, l_e2, h_pos2, h_e2, y2=train_data1[idx2]
                    l_pos3, l_y3, l_e3, h_pos3, h_e3=train_data2[idx3]
                    optimizer.zero_grad()
                    emb1=model_shared(l_pos1, l_y1, l_e1,  h_pos1, h_e1)
                    emb2=model_shared(l_pos2, l_y2, l_e2,  h_pos2, h_e2)
                    emb3=model_shared(l_pos3, l_y3, l_e3,  h_pos3, h_e3)
                    out1=model_F(emb1,l_y1, l_pos1, h_pos1)
                    out2=model_F(emb2, l_y2, l_pos2, h_pos2)
                    out3=model_F(emb3, l_y3, l_pos3, h_pos3)
                    out12=model_G(emb1, l_y1, l_pos1, h_pos1, emb2, l_y2, l_pos2, h_pos2)
                    out23=model_G(emb2, l_y2, l_pos2, h_pos2, emb3, l_y3, l_pos3, h_pos3)
                    out31=model_G(emb3, l_y3, l_pos3, h_pos3, emb1, l_y1, l_pos1, h_pos1)
                        
                    loss_F = F.mse_loss(out1,y1)+F.mse_loss(out2,y2)+F.mse_loss(out3, (out31+knn_interpolate(y1,h_pos1,h_pos3)).detach())+F.mse_loss(out3, knn_interpolate(y2-out23, h_pos2,h_pos3).detach())              
                    loss_G = F.mse_loss(out12,y1-knn_interpolate(y2,h_pos2,h_pos1))+F.mse_loss(out31, out3.detach()-knn_interpolate(y1,h_pos1,h_pos3))+F.mse_loss(out23,y2-knn_interpolate(out3,h_pos3,h_pos2).detach())
                    loss=loss_F+loss_G
                
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                            
                    mean_loss.append(loss.item())

            mean_loss=np.mean(mean_loss)        
            mean_test_loss=[]

            
            with torch.no_grad():
                
                for l_pos, l_y, l_e, h_pos, h_e,  y in test_data:
                    with autocast():
                        emb=model_shared(l_pos, l_y, l_e, h_pos, h_e)
                        out = model_F(emb, l_y, l_pos, h_pos)
                        loss = F.mse_loss(out,y)
                        mean_test_loss.append(loss.item())

                   
                mean_test_loss=np.mean(mean_test_loss)
            
            if epoch%5==0 and verbose:    
                print("epoch {} train loss {} test loss {}".format(epoch, mean_loss, mean_test_loss))
            
            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)
            
            if epoch> 15:
                if -(np.mean(test_loss_list[-15:])-np.mean(test_loss_list[-16:-1]))/np.mean(test_loss_list[-16:-1])<0.01:
                    break
        
        np.save(dir+"/MGN_comp_{}_{}_{}_{}.npy".format(len(train_data1), len(train_data2)+len(train_data1), str(ib_n)[0]+str(ib_e)[0], i), test_loss_list)
    return None





def train_MGN_sup(device, train_data, test_data, dir, num_exp=5, y_input_size=1, pos_input_size=2, edge_input_size=4, hidden_size=30, output_size=1, depth=3, residual=True,  ib_n=True, ib_e=True, verbose=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    scaler = GradScaler()
    
    for i in range(num_exp):
        model_shared=MGN_shared(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        model_F= Decoder_F(hidden_size, output_size, residual).to(device)
        optimizer=torch.optim.Adam(list(model_shared.parameters())+list(model_F.parameters()), lr=1e-3)
        
        loss_list=[]
        test_loss_list=[]
    
        for epoch in tqdm(range(5000)):
            mean_loss=[]
            for _ in (range(len(train_data))):
                with autocast():
                    idx=np.random.choice(list(range(len(train_data))),1)[0]
                    l_pos, l_y, l_e, h_pos, h_e, y=train_data[idx]
            
                    optimizer.zero_grad()
                    emb=model_shared(l_pos, l_y, l_e,  h_pos, h_e)
                    out=model_F(emb, l_y, l_pos, h_pos)
                    loss = F.mse_loss(out,y)                    

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    mean_loss.append(loss.item())
            mean_loss=np.mean(mean_loss)
                    
            mean_test_loss=[]
            with torch.no_grad():
                for l_pos, l_y, l_e, h_pos, h_e, y in test_data:
                    with autocast():
                        emb=model_shared(l_pos, l_y, l_e, h_pos, h_e)
                        out = model_F(emb, l_y, l_pos, h_pos)
                        loss = F.mse_loss(out,y)
                        mean_test_loss.append(loss.item())
                mean_test_loss=np.mean(mean_test_loss)
            
            if epoch%5==0 and verbose:    
                print("epoch {} train loss {} test loss {}".format(epoch, mean_loss, mean_test_loss))
            
            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)
            
            if epoch> 15:
                if -(np.mean(test_loss_list[-15:])-np.mean(test_loss_list[-16:-1]))/np.mean(test_loss_list[-16:-1])<0.01:
                    break
        
        np.save(dir+"/MGN_sup_{}_{}_{}_{}.npy".format(len(train_data), len(train_data), str(ib_n)[0]+str(ib_e)[0], i), test_loss_list)
    return None


def train_MGN_UCVME(device, train_data1, train_data2, test_data, dir, num_exp=5, y_input_size=1, pos_input_size=2, edge_input_size=4, hidden_size=30, output_size=1, depth=3,  residual=True, ib_n=True, ib_e=True, verbose=True, pred_time=5):
    if not os.path.exists(dir):
        os.mkdir(dir)
    scaler = GradScaler()
    for i in range(num_exp):

        model_shared_a=MGN_shared_UCVME(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        model_shared_b=MGN_shared_UCVME(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        model_F_mean_a= Decoder_F(hidden_size, output_size, residual=True).to(device)
        model_F_mean_b= Decoder_F(hidden_size, output_size, residual=True).to(device)
        model_F_std_a= Decoder_F(hidden_size, output_size, residual=False).to(device)
        model_F_std_b= Decoder_F(hidden_size, output_size, residual=False).to(device)
        optimizer1=torch.optim.Adam(list(model_shared_a.parameters())+list(model_F_mean_a.parameters())+list(model_F_std_a.parameters()), lr=1e-3)
        optimizer2=torch.optim.Adam(list(model_shared_b.parameters())+list(model_F_mean_b.parameters())+list(model_F_std_b.parameters()), lr=1e-3)
        
        loss_list=[]
        test_loss_list=[]

       
    
        for epoch in tqdm(range(5000)):
            mean_loss=[]
            for _ in (range(len(train_data2)+len(train_data1))):
           
                with autocast():

                    idx3=np.random.choice(list(range(len(train_data2))),1)[0]
                    idx12=np.random.choice(list(range(len(train_data1))),2,replace=False)
                    idx1=idx12[0]
                    idx2=idx12[1]
                        
                    l_pos1, l_y1, l_e1, h_pos1, h_e1, y1=train_data1[idx1]
                    l_pos2, l_y2, l_e2, h_pos2, h_e2, y2=train_data1[idx2]
                    l_pos3, l_y3, l_e3, h_pos3, h_e3=train_data2[idx3]
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    emb1=model_shared_a(l_pos1, l_y1, l_e1,  h_pos1, h_e1)
                    emb2=model_shared_b(l_pos2, l_y2, l_e2,  h_pos2, h_e2)

                    y_hat1a=model_F_mean_a(emb1, l_y1, l_pos1, h_pos1)
                    z_hat1a=torch.mean(model_F_std_a(emb1, l_y1, l_pos1, h_pos1))
                    y_hat1b=model_F_mean_b(emb1, l_y1, l_pos1, h_pos1)
                    z_hat1b=torch.mean(model_F_std_b(emb1, l_y1, l_pos1, h_pos1))

                    y_hat2a=model_F_mean_a(emb2, l_y2, l_pos2, h_pos2)
                    z_hat2a=torch.mean(model_F_std_a(emb2, l_y2, l_pos2, h_pos2))
                    y_hat2b=model_F_mean_b(emb2, l_y2, l_pos2, h_pos2)
                    z_hat2b=torch.mean(model_F_std_b(emb2, l_y2, l_pos2, h_pos2))

                    y_hat3a=[]
                    z_hat3a=[]
                    y_hat3b=[]
                    z_hat3b=[]
                    for _ in range(pred_time):
                        emb3=model_shared_a(l_pos3, l_y3, l_e3,  h_pos3, h_e3)
                        y_hata=model_F_mean_a(emb3, l_y3, l_pos3, h_pos3)
                        z_hata=torch.mean(model_F_std_a(emb3, l_y3, l_pos3, h_pos3))
                        y_hatb=model_F_mean_b(emb3, l_y3, l_pos3, h_pos3)
                        z_hatb=torch.mean(model_F_std_b(emb3, l_y3, l_pos3, h_pos3))
                        y_hat3a.append(y_hata.unsqueeze(0))
                        z_hat3a.append(z_hata)
                        y_hat3b.append(y_hatb.unsqueeze(0))
                        z_hat3b.append(z_hatb)
                    y_tilda3=torch.mean(torch.cat((y_hat3a),0),0)/2+torch.mean(torch.cat((y_hat3b),0),0)/2
                    z_tilda3=torch.mean(torch.tensor(z_hat3a))/2+torch.mean(torch.tensor(z_hat3b))/2
                
                    
                    L12_reg=1/2*torch.mean((((y_hat1a-y1)**2)/(2*torch.exp(z_hat1a))+z_hat1a/2)+(((y_hat1b-y1)**2)/(2*torch.exp(z_hat1b))+z_hat1b/2))+1/2*torch.mean((((y_hat2a-y2)**2)/(2*torch.exp(z_hat2a))+z_hat2a/2)+(((y_hat2b-y2)**2)/(2*torch.exp(z_hat2b))+z_hat2b/2))                
                    L12_unc=1/2*torch.mean(((z_hat1a-z_hat1b)**2)+((z_hat2a-z_hat2b)**2))
                    L3_reg=torch.mean((((torch.mean(torch.cat((y_hat3a),0),0)-y_tilda3)**2)/(2*torch.exp(z_tilda3))+z_tilda3/2)+(((torch.mean(torch.cat((y_hat3b),0),0)-y_tilda3)**2)/(2*torch.exp(z_tilda3))+z_tilda3/2))
                    L3_unc=torch.mean(((torch.mean(torch.tensor(z_hat3a))-z_tilda3)**2)+((torch.mean(torch.tensor(z_hat3b))-z_tilda3)**2))
                    loss=(L12_reg+L12_unc+L3_reg+L3_unc)       
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer1)
                    scaler.step(optimizer2)
                    scaler.update()
                                                
                    mean_loss.append(loss.item())

            mean_loss=np.mean(mean_loss)        
            mean_test_loss=[]

            
            with torch.no_grad():
                
                for l_pos, l_y, l_e, h_pos, h_e,  y in test_data:
                    with autocast():
                        y_hat3a=[]
                        z_hat3a=[]
                        y_hat3b=[]
                        z_hat3b=[]
                        for _ in range(pred_time):
                            emb=model_shared_a(l_pos, l_y, l_e,  h_pos, h_e)
                            y_hata=model_F_mean_a(emb, l_y, l_pos, h_pos)
                            z_hata=torch.mean(model_F_std_a(emb, l_y, l_pos, h_pos))
                            y_hatb=model_F_mean_b(emb, l_y, l_pos, h_pos)
                            z_hatb=torch.mean(model_F_std_b(emb, l_y, l_pos, h_pos))
                            y_hat3a.append(y_hata.unsqueeze(0))
                            z_hat3a.append(z_hata)
                            y_hat3b.append(y_hatb.unsqueeze(0))
                            z_hat3b.append(z_hatb)
                        y_tilda3=torch.mean(torch.cat((y_hat3a),0),0)/2+torch.mean(torch.cat((y_hat3b),0),0)/2
                        
                        loss = F.mse_loss(y_tilda3,y)
                        mean_test_loss.append(loss.item())
                mean_test_loss=np.mean(mean_test_loss)
            
            if epoch%5==0 and verbose:    
                print("epoch {} train loss {} test loss {}".format(epoch, mean_loss, mean_test_loss))
            
            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)
            
            if epoch> 15:
                if -(np.mean(test_loss_list[-15:])-np.mean(test_loss_list[-16:-1]))/np.mean(test_loss_list[-16:-1])<0.01:
                    break
        
        np.save(dir+"/MGN_UCVME_{}_{}_{}_{}.npy".format(len(train_data1), len(train_data2)+len(train_data1), str(ib_n)[0]+str(ib_e)[0], i), test_loss_list)
    return None


def train_MGN_twin(device, train_data1, train_data2, test_data, dir, num_exp=5, y_input_size=1, pos_input_size=2, edge_input_size=4, hidden_size=30, output_size=1, depth=3,  residual=True, ib_n=True, ib_e=True, verbose=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    scaler = GradScaler()
    for i in range(num_exp):

        model=MGN_TWIN(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        optimizer=torch.optim.Adam(list(model.parameters()), lr=1e-3)
        
        loss_list=[]
        test_loss_list=[]

       
    
        for epoch in tqdm(range(5000)):
            mean_loss=[]
            for _ in (range(len(train_data2)+len(train_data1))):
           
                with autocast():

                    idx3=np.random.choice(list(range(len(train_data2))),1)[0]
                    idx12=np.random.choice(list(range(len(train_data1))),2,replace=False)
                    idx1=idx12[0]
                    idx2=idx12[1]
                        
                    l_pos1, l_y1, l_e1, h_pos1, h_e1, y1=train_data1[idx1]
                    l_pos2, l_y2, l_e2, h_pos2, h_e2, y2=train_data1[idx2]
                    l_pos3, l_y3, l_e3, h_pos3, h_e3=train_data2[idx3]
                    optimizer.zero_grad()
                    
                    out12 = model(l_pos1, l_y1, l_e1,  h_pos1, h_e1, l_pos2, l_y2, l_e2,  h_pos2, h_e2)
                    out23 = model(l_pos2, l_y2, l_e2, h_pos2, h_e2, l_pos3, l_y3, l_e3, h_pos3, h_e3)
                    out31 = model(l_pos3, l_y3, l_e3, h_pos3, h_e3, l_pos1, l_y1, l_e1, h_pos1, h_e1)
                        
                    loss = F.mse_loss(out12,y1-knn_interpolate(y2,h_pos2,h_pos1))+torch.mean((out12+knn_interpolate(out23,h_pos2,h_pos1)+knn_interpolate(out31,h_pos3,h_pos1))**2)
                
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                            
                    mean_loss.append(loss.item())

            mean_loss=np.mean(mean_loss)        
            mean_test_loss=[]

            
            with torch.no_grad():

                for l_pos2, l_y2, l_e2, h_pos2, h_e2, y2 in test_data:
                    with autocast():
                        idx1=np.random.choice(list(range(len(train_data1))),1)[0]
                        l_pos1, l_y1, l_e1, h_pos1, h_e1, y1 =train_data1[idx1]

                        out = model(l_pos1, l_y1, l_e1,  h_pos1, h_e1, l_pos2, l_y2, l_e2,  h_pos2, h_e2)
                        loss = F.mse_loss(y1-out,knn_interpolate(y2,h_pos2,h_pos1))

                        mean_test_loss.append(loss.item())
                mean_test_loss=np.mean(mean_test_loss)
                
                
            
            if epoch%5==0 and verbose:    
                print("epoch {} train loss {} test loss {}".format(epoch, mean_loss, mean_test_loss))
            
            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)
            
            if epoch> 15:
                if -(np.mean(test_loss_list[-15:])-np.mean(test_loss_list[-16:-1]))/np.mean(test_loss_list[-16:-1])<0.01:
                    break
        
        np.save(dir+"/MGN_twin_{}_{}_{}_{}.npy".format(len(train_data1), len(train_data2)+len(train_data1), str(ib_n)[0]+str(ib_e)[0], i), test_loss_list)
    return None


def train_MGN_MT(device, train_data1, train_data2, test_data, dir, num_exp=5, y_input_size=1, pos_input_size=2, edge_input_size=4, hidden_size=30, output_size=1, depth=3,  residual=True, ib_n=True, ib_e=True, verbose=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    scaler = GradScaler()
    for i in range(num_exp):

        model_shared_s=MGN_shared_UCVME(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        model_F_s= Decoder_F(hidden_size, output_size, residual).to(device)
        model_shared_t=MGN_shared_UCVME(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        model_F_t= Decoder_F(hidden_size, output_size, residual).to(device)
        optimizer=torch.optim.Adam(list(model_shared_s.parameters())+list(model_F_s.parameters()), lr=1e-3)
        
        loss_list=[]
        test_loss_list=[]

    
        for epoch in tqdm(range(5000)):
            mean_loss=[]
            for _ in (range(len(train_data2)+len(train_data1))):
           
                with autocast():

                    idx3=np.random.choice(list(range(len(train_data2))),1)[0]
                    idx12=np.random.choice(list(range(len(train_data1))),2,replace=False)
                    idx1=idx12[0]
                    idx2=idx12[1]
                        
                    l_pos1, l_y1, l_e1, h_pos1, h_e1, y1=train_data1[idx1]
                    l_pos2, l_y2, l_e2, h_pos2, h_e2, y2=train_data1[idx2]
                    l_pos3, l_y3, l_e3, h_pos3, h_e3=train_data2[idx3]
                    optimizer.zero_grad()


                    emb1s=model_shared_s(l_pos1, l_y1, l_e1,  h_pos1, h_e1)
                    emb2s=model_shared_s(l_pos2, l_y2, l_e2,  h_pos2, h_e2)
                    emb3s=model_shared_s(l_pos3, l_y3, l_e3,  h_pos3, h_e3)
                    out1s=model_F_s(emb1s,l_y1, l_pos1, h_pos1)
                    out2s=model_F_s(emb2s, l_y2, l_pos2, h_pos2)
                    out3s=model_F_s(emb3s, l_y3, l_pos3, h_pos3)

                    emb1t=model_shared_t(l_pos1, l_y1, l_e1,  h_pos1, h_e1)
                    emb2t=model_shared_t(l_pos2, l_y2, l_e2,  h_pos2, h_e2)
                    emb3t=model_shared_t(l_pos3, l_y3, l_e3,  h_pos3, h_e3)
                    out1t=model_F_t(emb1t,l_y1, l_pos1, h_pos1).detach()
                    out2t=model_F_t(emb2t, l_y2, l_pos2, h_pos2).detach()
                    out3t=model_F_t(emb3t, l_y3, l_pos3, h_pos3).detach()
                    
                    loss1 = F.mse_loss(out1s,y1)+F.mse_loss(out2s,y2)
                    loss2 = F.mse_loss(out1s,out1t)+F.mse_loss(out2s,out2t)+F.mse_loss(out3s,out3t)
                    loss=loss1+loss2    

                    
                    
                
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    for param_t, param_s in zip(model_shared_t.parameters(), model_shared_s.parameters()):
                        param_t.data.mul_(0.99).add_(param_s.data, alpha=0.01)
                    
                    for param_t, param_s in zip(model_F_t.parameters(), model_F_s.parameters()):
                        param_t.data.mul_(0.99).add_(param_s.data, alpha=0.01)
                            
                    mean_loss.append(loss.item())

            mean_loss=np.mean(mean_loss)        
            mean_test_loss=[]

            
            with torch.no_grad():
                
                for l_pos, l_y, l_e, h_pos, h_e,  y in test_data:
                    with autocast():
                        emb=model_shared_s(l_pos, l_y, l_e, h_pos, h_e)
                        out = model_F_s(emb, l_y, l_pos, h_pos)
                        loss = F.mse_loss(out,y)
                        mean_test_loss.append(loss.item())

                   
                mean_test_loss=np.mean(mean_test_loss)
            
            if epoch%5==0 and verbose:    
                print("epoch {} train loss {} test loss {}".format(epoch, mean_loss, mean_test_loss))
            
            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)
            
            if epoch> 15:
                if -(np.mean(test_loss_list[-15:])-np.mean(test_loss_list[-16:-1]))/np.mean(test_loss_list[-16:-1])<0.01:
                    break
        
        np.save(dir+"/MGN_MT_{}_{}_{}_{}.npy".format(len(train_data1), len(train_data2)+len(train_data1), str(ib_n)[0]+str(ib_e)[0], i), test_loss_list)
    return None