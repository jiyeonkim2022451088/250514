from tqdm import tqdm
import numpy as np
import torch



class GraphDataset_paired(torch.utils.data.Dataset):
    def __init__(self, index_list, data_dir, device, SRGNN=False):
        self.index_list=index_list
        self.l_pos=[]
        self.l_e=[]
        self.l_y=[]   
        self.h_pos=[]
        self.h_e=[]
        self.h_y=[]
        self.SRGNN=SRGNN
        if SRGNN:
            self.n_i=[]
            self.c_e=[]

        
        for n in tqdm((index_list)):
            l_pos=np.load(data_dir+str(n)+"/L_mesh_geometry.npy")
            l_topology=np.load(data_dir+str(n)+"/L_mesh_topology.npy")
            l_y=np.load(data_dir+str(n)+"/L_y.npy")
            h_pos=np.load(data_dir+str(n)+"/H_mesh_geometry.npy")
            h_topology=np.load(data_dir+str(n)+"/H_mesh_topology.npy")
            h_y=np.load(data_dir+str(n)+"/H_y.npy")
      
            l_e1=np.concatenate([l_topology[:,0].reshape(-1,1),l_topology[:,1].reshape(-1,1)], 1)
            l_e2=np.concatenate([l_topology[:,1].reshape(-1,1),l_topology[:,2].reshape(-1,1)], 1)
            l_e3=np.concatenate([l_topology[:,2].reshape(-1,1),l_topology[:,0].reshape(-1,1)], 1)
            l_e4=np.concatenate([l_topology[:,1].reshape(-1,1),l_topology[:,0].reshape(-1,1)], 1)
            l_e5=np.concatenate([l_topology[:,2].reshape(-1,1),l_topology[:,1].reshape(-1,1)], 1)
            l_e6=np.concatenate([l_topology[:,0].reshape(-1,1),l_topology[:,2].reshape(-1,1)], 1)
            l_e=np.concatenate([l_e1,l_e2,l_e3,l_e4,l_e5,l_e6], 0)
            if SRGNN:
                node_info=l_topology.reshape(-1)
                c_e1=np.concatenate([np.arange(0,3*len(l_topology),3).reshape(1,-1), np.arange(1,3*len(l_topology),3).reshape(1,-1)],0)
                c_e2=np.concatenate([np.arange(1,3*len(l_topology),3).reshape(1,-1), np.arange(2,3*len(l_topology),3).reshape(1,-1)],0)
                c_e3=np.concatenate([np.arange(2,3*len(l_topology),3).reshape(1,-1), np.arange(0,3*len(l_topology),3).reshape(1,-1)],0)
                c_e4=np.concatenate([np.arange(1,3*len(l_topology),3).reshape(1,-1), np.arange(0,3*len(l_topology),3).reshape(1,-1)],0)
                c_e5=np.concatenate([np.arange(2,3*len(l_topology),3).reshape(1,-1), np.arange(1,3*len(l_topology),3).reshape(1,-1)],0)
                c_e6=np.concatenate([np.arange(0,3*len(l_topology),3).reshape(1,-1), np.arange(2,3*len(l_topology),3).reshape(1,-1)],0)
                c_e=np.concatenate([c_e1,c_e2,c_e3,c_e4,c_e5,c_e6], -1)
                                
            h_e1=np.concatenate([h_topology[:,0].reshape(-1,1),h_topology[:,1].reshape(-1,1)], 1)
            h_e2=np.concatenate([h_topology[:,1].reshape(-1,1),h_topology[:,2].reshape(-1,1)], 1)
            h_e3=np.concatenate([h_topology[:,2].reshape(-1,1),h_topology[:,0].reshape(-1,1)], 1)
            h_e4=np.concatenate([h_topology[:,1].reshape(-1,1),h_topology[:,0].reshape(-1,1)], 1)
            h_e5=np.concatenate([h_topology[:,2].reshape(-1,1),h_topology[:,1].reshape(-1,1)], 1)
            h_e6=np.concatenate([h_topology[:,0].reshape(-1,1),h_topology[:,2].reshape(-1,1)], 1)
            h_e=np.concatenate([h_e1,h_e2,h_e3,h_e4,h_e5,h_e6], 0)
            
            l_e=torch.transpose(torch.LongTensor(l_e),0,1)
            h_e=torch.transpose(torch.LongTensor(h_e),0,1)

            if SRGNN:
                node_info=torch.LongTensor(node_info)
                c_e=torch.LongTensor(c_e)

            self.l_pos.append(torch.FloatTensor((l_pos-np.min(l_pos,0))/(np.max(l_pos,0)-np.min(l_pos,0))).to(device))
            self.l_e.append(l_e.to(device))
            self.l_y.append(torch.FloatTensor(((l_y-np.min(l_y))/(np.max(l_y)-np.min(l_y))).reshape(-1,1)).to(device))
            
            if SRGNN:
                self.n_i.append(node_info.to(device))
                self.c_e.append(c_e.to(device))
            self.h_pos.append(torch.FloatTensor((h_pos-np.min(h_pos,0))/(np.max(h_pos,0)-np.min(h_pos,0))).to(device))
            self.h_e.append(h_e.to(device))
            self.h_y.append(torch.FloatTensor(((h_y-np.min(l_y))/(np.max(l_y)-np.min(l_y))).reshape(-1,1)).to(device))
           
           
    def __len__(self):
        return len(self.index_list)
    
        
    def __getitem__(self,idx):
        if self.SRGNN:
            return self.l_pos[idx], self.l_y[idx], self.l_e[idx], self.h_pos[idx], self.h_e[idx], self.h_y[idx], self.n_i[idx], self.c_e[idx]
        else:
            return self.l_pos[idx], self.l_y[idx], self.l_e[idx], self.h_pos[idx], self.h_e[idx], self.h_y[idx]
    



class GraphDataset_unpaired(torch.utils.data.Dataset):
    def __init__(self, index_list, data_dir, device, SRGNN=False):
        self.index_list=index_list
        self.l_pos=[]
        self.l_e=[]
        self.l_y=[]
        self.h_pos=[]
        self.h_e=[]
        self.SRGNN=SRGNN
        if SRGNN:
            self.n_i=[]
            self.c_e=[]
        
           
        
        for n in tqdm((index_list)):
            l_pos=np.load(data_dir+str(n)+"/L_mesh_geometry.npy")
            l_topology=np.load(data_dir+str(n)+"/L_mesh_topology.npy")
            l_y=np.load(data_dir+str(n)+"/L_y.npy")
            h_pos=np.load(data_dir+str(n)+"/H_mesh_geometry.npy")
            h_topology=np.load(data_dir+str(n)+"/H_mesh_topology.npy")
            h_y=np.load(data_dir+str(n)+"/H_y.npy")
      
            l_e1=np.concatenate([l_topology[:,0].reshape(-1,1),l_topology[:,1].reshape(-1,1)], 1)
            l_e2=np.concatenate([l_topology[:,1].reshape(-1,1),l_topology[:,2].reshape(-1,1)], 1)
            l_e3=np.concatenate([l_topology[:,2].reshape(-1,1),l_topology[:,0].reshape(-1,1)], 1)
            l_e4=np.concatenate([l_topology[:,1].reshape(-1,1),l_topology[:,0].reshape(-1,1)], 1)
            l_e5=np.concatenate([l_topology[:,2].reshape(-1,1),l_topology[:,1].reshape(-1,1)], 1)
            l_e6=np.concatenate([l_topology[:,0].reshape(-1,1),l_topology[:,2].reshape(-1,1)], 1)
            l_e=np.concatenate([l_e1,l_e2,l_e3,l_e4,l_e5,l_e6], 0)
            if SRGNN:
                node_info=l_topology.reshape(-1)
                c_e1=np.concatenate([np.arange(0,3*len(l_topology),3).reshape(1,-1), np.arange(1,3*len(l_topology),3).reshape(1,-1)],0)
                c_e2=np.concatenate([np.arange(1,3*len(l_topology),3).reshape(1,-1), np.arange(2,3*len(l_topology),3).reshape(1,-1)],0)
                c_e3=np.concatenate([np.arange(2,3*len(l_topology),3).reshape(1,-1), np.arange(0,3*len(l_topology),3).reshape(1,-1)],0)
                c_e4=np.concatenate([np.arange(1,3*len(l_topology),3).reshape(1,-1), np.arange(0,3*len(l_topology),3).reshape(1,-1)],0)
                c_e5=np.concatenate([np.arange(2,3*len(l_topology),3).reshape(1,-1), np.arange(1,3*len(l_topology),3).reshape(1,-1)],0)
                c_e6=np.concatenate([np.arange(0,3*len(l_topology),3).reshape(1,-1), np.arange(2,3*len(l_topology),3).reshape(1,-1)],0)
                c_e=np.concatenate([c_e1,c_e2,c_e3,c_e4,c_e5,c_e6], -1)

            h_e1=np.concatenate([h_topology[:,0].reshape(-1,1),h_topology[:,1].reshape(-1,1)], 1)
            h_e2=np.concatenate([h_topology[:,1].reshape(-1,1),h_topology[:,2].reshape(-1,1)], 1)
            h_e3=np.concatenate([h_topology[:,2].reshape(-1,1),h_topology[:,0].reshape(-1,1)], 1)
            h_e4=np.concatenate([h_topology[:,1].reshape(-1,1),h_topology[:,0].reshape(-1,1)], 1)
            h_e5=np.concatenate([h_topology[:,2].reshape(-1,1),h_topology[:,1].reshape(-1,1)], 1)
            h_e6=np.concatenate([h_topology[:,0].reshape(-1,1),h_topology[:,2].reshape(-1,1)], 1)
            h_e=np.concatenate([h_e1,h_e2,h_e3,h_e4,h_e5,h_e6], 0)
            
            l_e=torch.transpose(torch.LongTensor(l_e),0,1)
            h_e=torch.transpose(torch.LongTensor(h_e),0,1)

            if SRGNN:
                node_info=torch.LongTensor(node_info)
                c_e=torch.LongTensor(c_e)
           
            self.l_pos.append(torch.FloatTensor((l_pos-np.min(l_pos,0))/(np.max(l_pos,0)-np.min(l_pos,0))).to(device))
            self.l_e.append(l_e.to(device))
            self.l_y.append(torch.FloatTensor(((l_y-np.min(l_y))/(np.max(l_y)-np.min(l_y))).reshape(-1,1)).to(device))

            if SRGNN:
                self.n_i.append(node_info.to(device))
                self.c_e.append(c_e.to(device))
            
            self.h_pos.append(torch.FloatTensor((h_pos-np.min(h_pos,0))/(np.max(h_pos,0)-np.min(h_pos,0))).to(device))
            self.h_e.append(h_e.to(device))
        
           
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self,idx):
        if self.SRGNN:
            return self.l_pos[idx], self.l_y[idx], self.l_e[idx], self.h_pos[idx], self.h_e[idx], self.n_i[idx], self.c_e[idx]
        else:
            return self.l_pos[idx], self.l_y[idx], self.l_e[idx], self.h_pos[idx], self.h_e[idx]