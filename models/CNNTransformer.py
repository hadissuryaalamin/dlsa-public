import torch
import torch.nn as nn


class CNN_Block(nn.Module):
    def __init__(self, in_filters=1, out_filters=8, normalization=True, filter_size=2):
        super(CNN_Block, self).__init__()  
        self.in_filters = in_filters
        self.out_filters = out_filters
        
        self.conv1 = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=filter_size,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=out_filters, out_channels=out_filters, kernel_size=filter_size,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)
        self.left_zero_padding = nn.ConstantPad1d((filter_size-1,0),0)
        
        self.normalization1 = nn.InstanceNorm1d(in_filters)
        self.normalization2 = nn.InstanceNorm1d(out_filters)
        self.normalization = normalization
       
    def forward(self, x): #x and out have dims (N,C,T) where C is the number of channels/filters
        if self.normalization:
            x = self.normalization1(x)
        out = self.left_zero_padding(x)
        out = self.conv1(out)
        out = self.relu(out)
        if self.normalization: 
            out = self.normalization2(out)
        out = self.left_zero_padding(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out + x.repeat(1,int(self.out_filters/self.in_filters),1)   
        return out 

class CNNTransformer(nn.Module):
    def __init__(self, 
                 logdir,
                 random_seed = 0, 
                 lookback = 30,
                 device = "cpu", # other options for device are e.g. "cuda:0"
                 normalization_conv = True, 
                 filter_numbers = [1,8], 
                 attention_heads = 4, 
                 use_convolution = True,
                 hidden_units = 2*8, 
                 hidden_units_factor = 2,
                 dropout = 0.25, 
                 filter_size = 2, 
                 use_transformer = True,
                 use_gnn = False,
                 gnn_type = "fully_connected",
                 gnn_hidden_dim = 8,
                 gnn_layers = 1):
        
        super(CNNTransformer, self).__init__()
        self.logdir = logdir
        self.random_seed = random_seed 
        torch.manual_seed(self.random_seed)
        self.device = torch.device(device)
        self.filter_numbers = filter_numbers
        self.use_transformer = use_transformer
        self.use_convolution = use_convolution and len(filter_numbers) > 0
        self.use_gnn = use_gnn
        self.gnn_type = gnn_type
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_layers = gnn_layers
        self.is_trainable = True
        cnn_output_dim = filter_numbers[-1]
        model_dim = gnn_hidden_dim if use_gnn else cnn_output_dim

        if hidden_units and hidden_units_factor and hidden_units != hidden_units_factor * model_dim:
            raise Exception(f"`hidden_units` conflicts with `hidden_units_factor`; provide one or the other, but not both.")
        if hidden_units_factor:
            hidden_units = hidden_units_factor * model_dim
        
        self.convBlocks = nn.ModuleList()
        for i in range(len(filter_numbers)-1):
            self.convBlocks.append(
                CNN_Block(filter_numbers[i],filter_numbers[i+1],normalization=normalization_conv,filter_size=filter_size))

        if self.use_gnn:
            if self.gnn_type != "fully_connected":
                raise Exception(f"Invalid gnn_type '{self.gnn_type}'")
            self.gnn_layers_list = nn.ModuleList()
            for i in range(gnn_layers):
                in_dim = cnn_output_dim if i == 0 else gnn_hidden_dim
                self.gnn_layers_list.append(nn.Linear(2 * in_dim, gnn_hidden_dim))
            self.gnn_relu = nn.ReLU(inplace=True)

        self.encoder = nn.TransformerEncoderLayer(d_model=model_dim, nhead=attention_heads, dim_feedforward=hidden_units, dropout=dropout)
        self.linear = nn.Linear(model_dim,1)
        #self.softmax = nn.Sequential(nn.Linear(filter_numbers[-1],num_classes))#,nn.Softmax(dim=1))

    def _apply_convolution(self, x):
        if self.use_convolution:
            for i in range(len(self.filter_numbers)-1):
                x = self.convBlocks[i](x) #(N,C,T), C is the number of channels/features
        return x

    def _apply_fully_connected_gnn(self, x, valid_mask):
        # x has shape (B,L,N,C); valid_mask has shape (B,N)
        node_mask = valid_mask.to(dtype=x.dtype, device=x.device).unsqueeze(1).unsqueeze(-1)
        x = x * node_mask

        for layer in self.gnn_layers_list:
            masked_x = x * node_mask
            total = torch.sum(masked_x, dim=2, keepdim=True)
            counts = torch.sum(node_mask, dim=2, keepdim=True).clamp_min(1.0)

            neighbor_counts = (counts - node_mask).clamp_min(1.0)
            neighbor_mean = (total - masked_x) / neighbor_counts
            neighbor_mean = neighbor_mean * (counts > 1).to(dtype=x.dtype)

            x = layer(torch.cat((x, neighbor_mean.expand_as(x)), dim=-1))
            x = self.gnn_relu(x)
            x = x * node_mask

        return x

    def _forward_flat(self,x): #x has dimension (N,T)
        N,T = x.shape
        x = x.reshape((N,1,T))  #(N,1,T)
        x = self._apply_convolution(x)
        x = x.permute(2,0,1)
        if self.use_transformer:
            x = self.encoder(x) #the input of the transformer is (T,N,C)
        return self.linear(x[-1,:,:]).squeeze() #this outputs the weights #self.softmax(x[-1,:,:]) #(N,num_classes)

    def _forward_gnn(self, x, valid_mask):
        # x has dimension (B,N,T), valid_mask has dimension (B,N)
        if valid_mask is None:
            raise Exception("valid_mask must be provided when use_gnn=True")
        B,N,T = x.shape
        valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)

        x_valid = x[valid_mask]
        x_cnn_valid = torch.zeros(
            (x_valid.shape[0], self.filter_numbers[-1], T),
            dtype=x.dtype,
            device=x.device,
        )
        if x_valid.shape[0] > 0:
            x_valid = x_valid.reshape((x_valid.shape[0],1,T))
            x_cnn_valid = self._apply_convolution(x_valid)

        x_full = torch.zeros(
            (B, N, self.filter_numbers[-1], T),
            dtype=x.dtype,
            device=x.device,
        )
        x_full[valid_mask] = x_cnn_valid
        x_full = x_full.permute(0,3,1,2)  #(B,T,N,C)
        x_full = self._apply_fully_connected_gnn(x_full, valid_mask)

        x_full = x_full.permute(0,2,1,3).reshape(B*N, T, self.gnn_hidden_dim)
        x_full = x_full.permute(1,0,2)  #(T,B*N,C)
        if self.use_transformer:
            x_full = self.encoder(x_full)
        out = self.linear(x_full[-1,:,:]).reshape(B,N)
        return out.masked_fill(~valid_mask, 0)

    def forward(self,x, valid_mask=None):
        if self.use_gnn:
            return self._forward_gnn(x, valid_mask)
        return self._forward_flat(x)
