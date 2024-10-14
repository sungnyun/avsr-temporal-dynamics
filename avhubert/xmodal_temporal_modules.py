import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFlowPredictor(nn.Module):
    def __init__(self, feat_dim, temp_len=1, temp_hidden_dim=1024, use_aggregator=False, aggregator_kernel_size=3):
        super().__init__()
        self.use_aggregator = use_aggregator
        if self.use_aggregator:
            assert temp_len == 1
            self.aggregator = nn.Conv1d(in_channels=feat_dim,
                                        out_channels=(temp_hidden_dim // aggregator_kernel_size),  ## for # of params compensation
                                        kernel_size=aggregator_kernel_size)  ## 3 or 5
            self.relu = nn.GELU()
            self.fc2 = nn.Linear(2 * self.aggregator.out_channels, 1)
        else:
            self.fc1 = nn.Linear(feat_dim, temp_hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(2 * temp_len * temp_hidden_dim, 1)
        self.temp_len = temp_len
    
    def forward(self, x, random=True):
        if self.use_aggregator:
            x = self.relu(self.aggregator(x.transpose(1, 2))).transpose(1, 2)
        else:
            x = self.relu(self.fc1(x))
        len_x = x.size(1) - (self.temp_len-1)

        if random:
            if len_x % 2 == 1:
                len_x -= 1
            perm = torch.randperm(len_x).view(-1, 2)
            perm = perm.sort(dim=-1)[0]

            before = x[:, perm[:,0]]
            after = x[:, perm[:,1]]
            for i in range(1, self.temp_len):
                before = torch.cat([before, x[:, perm[:,0] + i]], dim=-1)
                after = torch.cat([after, x[:, perm[:,1] + i]], dim=-1)
                
            forward_flow  = torch.cat([before, after], dim=-1)  # B * T' * (2 * temp_len * D)
            backward_flow = torch.cat([after, before], dim=-1)  # B * T' * (2 * temp_len * D)
        else:
            raise NotImplementedError
        
        x = torch.stack([forward_flow, backward_flow], dim=0)  # 2 * B * T' * (2 * temp_len * D)

        return self.fc2(x)

class TemporalVAFlowPredictor(nn.Module):
    def __init__(self, feat_dim, temp_len=1, temp_hidden_dim=1024, use_aggregator=False, aggregator_kernel_size=3):
        super().__init__()
        self.use_aggregator = use_aggregator
        if self.use_aggregator:
            assert temp_len == 1
            self.video_aggregator = nn.Conv1d(in_channels=feat_dim,
                                        out_channels=(temp_hidden_dim // aggregator_kernel_size),  ## for # of params compensation
                                        kernel_size=aggregator_kernel_size)  ## 3 or 5
            self.audio_aggregator = nn.Conv1d(in_channels=feat_dim,
                                        out_channels=(temp_hidden_dim // aggregator_kernel_size),  ## for # of params compensation
                                        kernel_size=aggregator_kernel_size)  ## 3 or 5            
            self.relu = nn.GELU()
            self.fc2 = nn.Linear(2 * self.video_aggregator.out_channels, 1)
        else:
            self.fc1 = nn.Linear(feat_dim, temp_hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(2 * temp_len * temp_hidden_dim, 1)
        self.temp_len = temp_len
    
    def forward(self, x, y, random=True):
        # x for video
        # y for audio
        if self.use_aggregator:
            x = self.relu(self.video_aggregator(x.transpose(1, 2))).transpose(1, 2)
            y = self.relu(self.audio_aggregator(y.transpose(1, 2))).transpose(1, 2)
        else:
            x = self.relu(self.fc1(x))
            y = self.relu(self.fc1(y))
        len_x = x.size(1) - (self.temp_len-1)
        len_y = y.size(1) - (self.temp_len-1)

        assert len_x == len_y
        if random:
            if len_x % 2 == 1:
                len_x -= 1
                len_y -= 1
            perm = torch.randperm(len_x).view(-1, 2)
            perm = perm.sort(dim=-1)[0]

            ## Ground Truth : video then audio
            before_video = x[:, perm[:,0]]
            after_audio = y[:, perm[:,1]]

            for i in range(1, self.temp_len):
                before_video = torch.cat([before_video, x[:, perm[:,0] + i]], dim=-1)
                after_audio = torch.cat([after_audio, y[:, perm[:,1] + i]], dim=-1)
                
            forward_flow  = torch.cat([before_video, after_audio], dim=-1)  # B * T' * (2 * temp_len * D)
            
            ## Ground Truth : audio then video
            after_video = x[:, perm[:,1]]
            before_audio = y[:, perm[:,0]]

            for i in range(1, self.temp_len):
                after_video = torch.cat([after_video, x[:, perm[:,1] + i]], dim=-1)
                before_audio = torch.cat([before_audio, y[:, perm[:,0] + i]], dim=-1)

            backward_flow = torch.cat([after_video, before_audio], dim=-1)  # B * T' * (2 * temp_len * D)
        else:
            raise NotImplementedError
        
        x = torch.stack([forward_flow, backward_flow], dim=0)  # 2 * B * T' * (2 * temp_len * D)

        return self.fc2(x)

class TemporalDirectionPredictor(nn.Module):
    def __init__(self, feat_dim, temp_len=1, temp_hidden_dim=1024, use_aggregator=False, aggregator_kernel_size=3):
        super().__init__()
        self.use_aggregator = use_aggregator
        self.aggregator_kernel_size = aggregator_kernel_size
        if self.use_aggregator:
            if temp_len == 1:
                temp_len = aggregator_kernel_size
            self.aggregator = nn.Conv1d(in_channels=feat_dim,
                                        out_channels=(temp_hidden_dim // aggregator_kernel_size),  ## for # of params compensation
                                        kernel_size=aggregator_kernel_size)  ## 3 or 5
            self.relu = nn.GELU()
            self.fc2 = nn.Linear(temp_len * self.aggregator.out_channels, 1)
        else:
            self.fc1 = nn.Linear(feat_dim, temp_hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(temp_len * temp_hidden_dim, 1)
        self.temp_len = temp_len
        assert self.temp_len > 1

    def forward(self, x):
        if self.use_aggregator:
            x = self.relu(self.aggregator(x.transpose(1, 2))).transpose(1, 2)
        else:
            x = self.relu(self.fc1(x))
        len_x = x.size(1) - (self.temp_len-1)
        forward, backward = x[:, :len_x], x[:, :len_x]
        for i in range(1, self.temp_len):
            forward = torch.cat([forward, x[:, i:i+len_x]], dim=-1)
            backward = torch.cat([x[:, i:i+len_x], backward], dim=-1)
        x = torch.stack([forward, backward], dim=0)
        return self.fc2(x)
    
class TemporalSpeedPredictor(nn.Module):
    def __init__(self, feat_dim, temp_len=1, temp_hidden_dim=1024, speed=2, use_aggregator=False, aggregator_kernel_size=3):
        super().__init__()
        self.use_aggregator = use_aggregator
        self.aggregator_kernel_size = aggregator_kernel_size
        if self.use_aggregator:
            if temp_len == 1:
                temp_len = aggregator_kernel_size
            self.aggregator = nn.Conv1d(in_channels=feat_dim,
                                        out_channels=(temp_hidden_dim // aggregator_kernel_size),  ## for # of params compensation
                                        kernel_size=aggregator_kernel_size)  ## 3 or 5
            self.relu = nn.GELU()
            self.fc2 = nn.Linear(temp_len * self.aggregator.out_channels, 1)
        else:
            self.fc1 = nn.Linear(feat_dim, temp_hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(temp_len * temp_hidden_dim, 1)
        self.temp_len = temp_len
        self.speed = speed
        assert self.temp_len > 1

    def forward(self, x):
        if self.use_aggregator:
            x = self.relu(self.aggregator(x.transpose(1, 2))).transpose(1, 2)
        else:
            x = self.relu(self.fc1(x))
        len_x = x.size(1) - (self.speed * (self.temp_len-1))
        speed_x1, speed_x2 = x[:, :len_x], x[:, :len_x]
        for i in range(1, self.temp_len):
            speed_x1 = torch.cat([speed_x1, x[:, i:i+len_x]], dim=-1)
            speed_x2 = torch.cat([speed_x2, x[:, (i*self.speed):(i*self.speed)+len_x]], dim=-1)
        x = torch.stack([speed_x1, speed_x2], dim=0)
        return self.fc2(x)