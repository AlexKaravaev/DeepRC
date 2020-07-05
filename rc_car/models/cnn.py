from torch import nn
from rc_car.models.utils import crop_input

class CNNAutoPilot(nn.Module):
    '''
    Nvidia CNN end-to-end network
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    '''
    def __init__(self, input_shape=(120,160,3), roi_crop=(0,0)):
        super(CNNAutoPilot, self).__init__()

        self.input_shape = crop_input(input_shape, roi_crop)
        self.dropout_prob = 0.2

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 24,
                      kernel_size=(5,5),
                      stride=(2,2),
                      ),

            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels = 24,
                      out_channels = 32,
                      kernel_size=(5,5),
                      stride=(2,2),
                      ),

            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
 
            
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels = 32,
                      out_channels = 64,
                      kernel_size=(5,5),
                      stride=(2,2),
                      )
        )
        
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                      out_channels = 64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      ),

            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        
        self.layer_5 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                      out_channels = 64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      ),

            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        self.layer_6 = nn.Flatten()
        self.layer_7 = nn.Sequential(
            nn.Linear(in_features  = 8*13*64,
                      out_features = 100),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        self.layer_8 = nn.Sequential(

            nn.Linear(in_features=100,
                      out_features=50),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        self.angle_out = nn.Sequential(
            nn.Linear(in_features=50,
                      out_features=1)
           
        )

        self.throttle_out = nn.Sequential(
            nn.Linear(in_features=50,
                      out_features=1)
           
        )

    
    def forward(self, x):

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
    
        return self.angle_out(x).view(-1).double(), self.throttle_out(x).view(-1).double()

 
 
 
 