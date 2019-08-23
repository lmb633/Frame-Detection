import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from config import device, grad_clip, print_freq, loss_ratio
from data_gen import FrameDetectionDataset
from models import FrameDetectionModel
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_iou
from demo import visual_img


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        print("from beginning ")
        model = FrameDetectionModel()
        # model = nn.DataParallel(model)

    else:
        print("load checkpoint ")
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(0))
        model = checkpoint['model']
        # optimizer = checkpoint['optimizer']
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    print('optimizer.lr ', optimizer.state_dict()['param_groups'][0]['lr'])

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.L1Loss().to(device)

    # Custom dataloaders
    train_dataset = FrameDetectionDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_dataset = FrameDetectionDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('Train_Loss', train_loss, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterion=criterion,
                           logger=logger)

        writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)
        print('optimizer.lr ', optimizer.state_dict()['param_groups'][0]['lr'])
        visual_img(model)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # print(img.shape)
        # print(label.shape)
        # Move to GPU, if available
        img = img.to(device)
        label = label.type(torch.FloatTensor).to(device)  # [N, 8]

        # Forward prop.
        output = model(img)  # embedding => [N, 8]
        # print(output.shape)
        # print(label)
        # Calculate loss
        loss = criterion(output, label)
        loss = loss * loss_ratio

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'
                        .format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, criterion, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.type(torch.FloatTensor).to(device)  # [N, 8]

        # Forward prop.
        output = model(img)  # embedding => [N, 8]

        # Calculate loss
        loss = criterion(output, label)
        loss = loss * loss_ratio

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('Validation: Loss {0:.4f}\n'.format(losses.avg))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
