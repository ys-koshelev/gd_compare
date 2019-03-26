# adopted from https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import torch
from tqdm.autonotebook import tqdm
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torch import nn
from torch import empty

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro");
    return metric_fn(true_y, pred_y);
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
    pass
                           

def measure_scores(model, val_loader, cuda_avail):
    model.train(False);
    val_batches = len(val_loader);
    loss_function = torch.nn.CrossEntropyLoss();
    torch.cuda.empty_cache();

    val_losses = 0;
    precision, recall, f1, accuracy = [], [], [], [];

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if cuda_avail:
                X, y = data[0].cuda(), data[1].cuda().squeeze(-1).long();
            else:
                X, y = data[0], data[1].squeeze(-1).long();

            outputs = torch.sigmoid(model(X));
            val_losses += loss_function(outputs, y.long());

            predicted_classes = torch.max(torch.sigmoid(outputs), 1)[1];

            for acc, metric in zip((precision, recall, f1, accuracy), 
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(calculate_metric(metric, y.cpu(), predicted_classes.cpu()));
    print_scores(precision, recall, f1, accuracy, val_batches);
    model.train(True);
    return val_losses/val_batches, sum(accuracy)/val_batches;

def train(model, optimizer, train_loader, val_loader, epochs=1, alpha=0.9, stop_accuracy=0.9940, cuda_avail=True):
    start_ts = time.time();
    losses = [];
              
    train_loss = [];
    nll_loss_function = torch.nn.NLLLoss();
    batches = len(train_loader);
              
    # training loop + eval loop
    for epoch in range(epochs):
        total_loss = 0;
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches);
        model.train(True);

        for i, data in progress:
            
            if cuda_avail:
                X, y, fakes = data[0].cuda(), data[1].cuda().squeeze(-1).long(), data[2].cuda();
            else:
                X, y, fakes = data[0], data[1].squeeze(-1).long(), data[2];
            
            model.zero_grad();
            outputs = model(X);
            
            # prepare loss
            loss = 0;
            
            # computing loss on fake samples
            if not torch.all(fakes == 0):
                mask = torch.zeros((y.shape[0], 10)).byte();
                for i in range(y.shape[0]):
                    mask[i, torch.abs(y[i])] = 1;
                loss = alpha*torch.sum(torch.log1p(torch.exp(torch.sigmoid(outputs[mask][fakes]))));
            
            # computing loss on real samples  
            if not torch.all(fakes == 1):
                loss += (1-alpha)*nll_loss_function(torch.softmax(torch.sigmoid(outputs[1-fakes]), -1), y[1-fakes]);
            
            total_loss += loss.detach().data;
            train_loss.append(copy.deepcopy(loss.detach().data.cpu().numpy()));
            loss.backward()
            optimizer.step()
            progress.set_description("Loss: {:.4f}".format(loss.item()))

        val_loss, val_accuracy = measure_scores(model, val_loader, cuda_avail);
        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_loss}")
        losses.append(total_loss/batches)
        
        # early stopping with a threshold
        if val_accuracy >= stop_accuracy:
            break
    return train_loss;
              
class MnistDataset(Dataset):
    def __init__(self, root_dir='data/', training='train', train_samples='all'):
        self.imgs = [];
        self.labels = [];
        self.fakes = [];
        
        if training == 'train' or training == 'validate':
            x, y = torch.load(root_dir + 'MNIST/processed/training.pt');
        else:
            x, y = torch.load(root_dir + 'MNIST/processed/test.pt');
        if train_samples == 'all' and training == 'train':
            for i in range(50000):
                self.imgs.append(x[i, ...].float()/255);
                self.labels.append(y[i].long());
                self.fakes.append(torch.ByteTensor([0])[0])
        elif training == 'train':
            for i in range(train_samples):
                self.imgs.append(x[i, ...].float()/255);
                self.labels.append(y[i].long());
                self.fakes.append(torch.ByteTensor([0])[0]);
        elif training == 'validate':
            for i in range(1, 10001):
                self.imgs.append(x[-i, ...].float()/255);
                self.labels.append(y[-i].long());
                self.fakes.append(torch.ByteTensor([0])[0]);
        elif training == 'test':
            for i in range(y.shape[0]):
                self.imgs.append(x[i, ...].float()/255);
                self.labels.append(y[i].long());
                self.fakes.append(torch.ByteTensor([0])[0]);
                
    def __len__(self):
        return len(self.labels);

    def __getitem__(self, idx):
        x = self.imgs[idx];
        y = self.labels[idx];
        fake = self.fakes[idx];
        
        return (x.unsqueeze(0)-0.5)/(0.5)*0.6, y.float().unsqueeze(0), fake;
    
    def add_artificial(self, X):
        for i in range(X.shape[0]):
            self.imgs.append(X[i, 0, ...].detach().cpu());
            self.labels.append(self.labels[0].new(1).fill_(-i)[0]);
            self.fakes.append(torch.ByteTensor([1])[0]);
        pass;
    
def get_data_loaders(train_batch_size, val_batch_size, test_batch_size, train_size='all'):
    MNIST(download=True, train=True, root=".").train_data.float();

    train_loader = DataLoader(MnistDataset(root_dir='', training='train', train_samples=train_size),
                              batch_size=train_batch_size, shuffle=True);

    val_loader = DataLoader(MnistDataset(root_dir='', training='validate'),
                            batch_size=val_batch_size, shuffle=False);
    
    test_loader = DataLoader(MnistDataset(root_dir='', training='test'),
                            batch_size=test_batch_size, shuffle=False);
    return train_loader, val_loader, test_loader;
              
              
#residual architecture for MNIST classification. Adopted from PyTorch ResNet code
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
              
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def to_synth(self):
        return self.forward(self.X);

def MnistResNet():
    """Constructs a kind of ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2]);
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False);
    model.fc = nn.Linear(256, 10, bias=True);
    return model