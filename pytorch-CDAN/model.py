
from backBone import network_dict

import torch
import tqdm
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from utils import ConditionalDomainAdversarialLoss


def train_process(model, sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader,DEVICE,imageSize,args):

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.l2_Decay, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.5, 0.999),
    #                             weight_decay=args.l2_Decay)

    learningRate = LambdaLR(optimizer, lambda x: (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        model.discriminator, entropy_conditioning=args.entropy,
        num_classes=args.n_labels, features_dim=model.classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(DEVICE)

    lenSourceDataLoader = len(sourceDataLoader)

    path = args.savePath

    writer = SummaryWriter(log_dir=path)

    base_epoch = 0
    if args.ifload:
        # path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        base_epoch = checkpoint['epoch']
    t_correct=0
    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        domain_adv.train()

        correct = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.to(DEVICE), sourceLabel.to(DEVICE)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.to(DEVICE), targetLabel.to(DEVICE)
                break

            # compute output
            x = torch.cat((sourceData, targetData), dim=0)

            y, f = model(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            source_pre = y_s.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()

            cls_loss = F.cross_entropy(y_s, sourceLabel.long())
            transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
            loss = cls_loss + transfer_loss * args.trade_off


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learningRate.step(epoch)


            if batch_idx % args.logInterval == 0:
                print(
                    '\ncls_loss: {:.4f},  transfer_loss: {:.4f}'.format(
                        cls_loss.item(), transfer_loss.item()))

            writer.add_scalar(tag="loss_train/cls_loss",scalar_value=cls_loss,
                              global_step=(epoch-1) * lenSourceDataLoader * batch_idx)
            writer.add_scalar(tag="loss_train/transfer_loss", scalar_value=cls_loss,
                              global_step=(epoch - 1) * lenSourceDataLoader * batch_idx)
            writer.add_scalar(tag="loss_train/loss", scalar_value=cls_loss,
                              global_step=(epoch - 1) * lenSourceDataLoader * batch_idx)


        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Train Accuracy: {}/{} ({:.2f}%)'.format(
            correct, (lenSourceDataLoader * args.batchSize), acc_train))

        test_correct, test_acc=test_process(model, sourceTestDataLoader,taragetTestDataLoader, DEVICE, args)
        if test_correct > t_correct:
            t_correct = test_correct
        print("max correct:" , t_correct)
        # if epoch % args.logInterval == 0:
        #     model_feature_tSNE(model, sourceTestDataLoader, taragetTestDataLoader, 'epoch' + str(epoch), DEVICE,
        #                        args.model_name)

        writer.add_scalar(tag="acc_train", scalar_value=acc_train, global_step=epoch)
        writer.add_scalar(tag="acc_test", scalar_value=test_acc, global_step=epoch)


        if args.ifsave:
            if not os.path.exists(path):
                os.makedirs(path)

            if args.if_saveall:
                state = {
                    'epoch': args.epoch,
                    'net': model,
                    'optimizer': optimizer,

                }
            else:
                state = {
                    'epoch': args.epoch,
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),

                }

            if epoch % 50 == 0:
                path+='epoch'+str(epoch)+'.pth'
                torch.save(state, path)


def test_process(model,sourceTestDataLoader,taragetTestDataLoader, device, args):
    model.eval()


    # source Test
    correct = 0
    testLoss = 0
    with torch.no_grad():
        for data, suorceLabel in sourceTestDataLoader:
            if args.n_dim == 0:
                data, suorceLabel = data.to(args.device), suorceLabel.to(args.device)
            elif args.n_dim > 0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.to(device)
                suorceLabel = suorceLabel.to(device)
            Output = model(data)[0]
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), suorceLabel.long(),
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(suorceLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(sourceTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, source Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(sourceTestDataLoader.dataset),
            100. * correct / len(sourceTestDataLoader.dataset)))


    if taragetTestDataLoader==None:
        return
    # target Test
    correct = 0
    testLoss = 0
    with torch.no_grad():
        for data, targetLabel in taragetTestDataLoader:
            if args.n_dim == 0:
                data, targetLabel = data.to(args.device), targetLabel.to(args.device)
            elif args.n_dim > 0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.to(device)
                targetLabel = targetLabel.to(device)
            Output = model(data)[0]
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), targetLabel.long(),
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
        target_acc = 100. * correct / len(taragetTestDataLoader.dataset)
    return correct, target_acc




class ADDAModel(nn.Module):

    def __init__(self, args):
        super(ADDAModel, self).__init__()
        self.args=args
        if args.data_name == 'Digits':
            self.backbone = network_dict['Net1d'](args.n_dim)
            if args.bottleneck:
                self.bottleneck = nn.Sequential(
                    nn.Linear(800, args.bottleneck_dim),
                    nn.BatchNorm1d(args.bottleneck_dim),
                    nn.ReLU()
                )
                self.classifier_feature_dim = args.bottleneck_dim
            else:
                self.classifier_feature_dim = 2560

            self.classifier = nn.Sequential(
                nn.Linear(self.classifier_feature_dim, 50),
                nn.ReLU(),
                nn.Linear(50, args.n_labels),
            )
        # elif args.data_name == 'office':
        #     self.backbone = network_dict['ResNet50']()
        #     if args.bottleneck:
        #         self.bottleneck = nn.Sequential(
        #             nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #             nn.Flatten(),
        #             nn.Linear(self.backbone.out_features, args.bottleneck_dim),
        #             nn.BatchNorm1d(args.bottleneck_dim),
        #             nn.ReLU(),
        #         )
        #         self.classifier_feature_dim = args.bottleneck_dim
        #     else:
        #         self.classifier_feature_dim = self.backbone.out_features
        #     self.classifier = nn.Sequential(
        #
        #         nn.Linear(self.classifier_feature_dim, args.n_labels)
        #     )


        # D
        if args.randomized:
            in_feature=args.randomized_dim
        else:
            in_feature =self.classifier_feature_dim * args.n_labels
        hidden_size=args.hidden_size

        if args.batch_norm:
            self.discriminator = nn.Sequential(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def forward(self,data):

        feature = self.backbone(data)
        if self.args.bottleneck:
            feature=self.bottleneck(feature)
        label=self.classifier(feature)


        return label,feature
