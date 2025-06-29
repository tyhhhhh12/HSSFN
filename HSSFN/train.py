import argparse
import auxil
import torch
import torch.nn.parallel
from HSIDataset import HSIDataset
import numpy as np
from auxil import str2bool
from model.model_PSFEM_Freqfusion_FocalLoss import UNet
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,recall_score,cohen_kappa_score,accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from FoctalLoss import FocalLoss
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def set_seed(seed=42):
    random.seed(seed)               # Python内置随机模块
    np.random.seed(seed)            # NumPy
    torch.manual_seed(seed)         # PyTorch CPU
    torch.cuda.manual_seed(seed)    # PyTorch GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def display_sampling(label,y_train,y_test,y_all):
    sample_report = f"{'class': ^10}{'train_num':^10}{'test_num': ^10}{'total': ^10}\n"
    for i in np.unique(y_all):
        sample_report += f"{i: ^10}{(y_train==i).sum(): ^10}{(y_test==i).sum(): ^10}{(y_all==i).sum(): ^10}\n"
    sample_report += f"{'total': ^10}{np.count_nonzero(y_train): ^10}{np.count_nonzero(y_test): ^10}{np.count_nonzero(y_all): ^10}"
    print(sample_report)
    fp = open(os.path.join('result', 'SA_FinalModel_0.05' + 'sample_report.txt'), 'w+')
    fp.writelines(sample_report)
    fp.close()

def load_hyper(args):

    data, label, numclass,class_name = auxil.loadData(args.dataset, num_components=args.components)
    data_shape = data.shape
    print(data_shape)

    PATCH_LENGTH = int((args.spatial_size - 1) // 2)

    padded_data = np.pad(data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)), 'constant',
                             constant_values=0)



    bands = data_shape[-1]
    number_class = np.max(label)
    print("bands: {}, number_class: {}".format(bands, number_class))


    labels = label.reshape((label.shape[0] * label.shape[1]))
    labels = labels.astype(np.int32)
    print(labels.shape)



    train_indices, test_indices = auxil.sampling(1 - args.tr_percent, labels)
    _, total_indices = auxil.sampling(1, labels)
    total_size, train_size, test_size = len(total_indices), len(train_indices), len(test_indices)
    x_all, x_train, x_test, y_all, y_train, y_test = auxil.generate_data(train_size, train_indices, test_size,
                                                                         test_indices, total_size, total_indices,
                                                                         data, PATCH_LENGTH, padded_data, bands, labels)
    print(
        "total_size: {}, train_size: {}, test_size: {}\ntotal_data_shape: {}, train_data_shape: {}, test_data_shape: {}".format(
            total_size, train_size, test_size, x_all.shape, x_train.shape, x_test.shape
        ))
    display_sampling(labels, y_train, y_test, y_all)


    train_hyper = HSIDataset((np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32), y_train), None)
    test_hyper = HSIDataset((np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32), y_test), None)
    full_hyper = HSIDataset((np.transpose(x_all, (0, 3, 1, 2)).astype(np.float32), y_all), None)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(train_hyper,
                                                batch_size=args.tr_bsize,
                                                shuffle=True,
                                                generator=g,
                                                **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper,
                                              batch_size=args.te_bsize,
                                              shuffle=False,
                                              **kwargs)
    full_loader = torch.utils.data.DataLoader(full_hyper,
                                              batch_size=args.te_bsize,
                                              shuffle=False,
                                              generator=g,
                                              **kwargs)
    return labels, full_loader, train_loader, test_loader, number_class, bands, data_shape, total_indices,class_name,y_train

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.5 ** max(0, (epoch + 1) // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, ce_criterion,  optimizer,epoch, use_cuda, args):
    model.train()
    accs   = np.ones((len(train_loader))) * -1000.0
    losses = np.ones((len(train_loader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}')):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda().long()
        outputs= model(inputs)
        loss1 = ce_criterion(outputs, targets)
        loss = loss1
        losses[batch_idx] = loss.item()
        accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (np.average(losses), np.average(accs))

def test(testloader, model, ce_criterion,epoch, use_cuda, args):
    model.eval()
    accs   = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda().long()
            outputs = model(inputs)
            losses[batch_idx] = ce_criterion(outputs, targets).item()
            accs[batch_idx] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))

def predict(fullloader, model, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(fullloader):
            if use_cuda: inputs = inputs.cuda()
            pred = model(inputs)
            predicted.extend(pred.cpu().numpy())
    return np.array(predicted)


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(12, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, cbar=False)

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning - rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--components', default=30, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='Di', type=str, help='')
    parser.add_argument('--tr_percent', default=0.3, type=float)
    parser.add_argument('--tr_bsize', default=32, type=int)
    parser.add_argument('--te_bsize', default=32, type=int)
    parser.add_argument('--spatial_size',  default=15, type=int,)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume', type=str2bool, default='false')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    patchesLabels, full_loader, train_loader, test_loader, num_classes, n_bands, data_shape, total_indices,class_name,y_train= load_hyper(args)
    print('[i] Dataset finished!')

    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True


    model_name = "HFBNet_Di_LDA_10_0.3_FC_{}_".format(args.spatial_size)
    model = UNet(num_classes,n_bands)

    if use_cuda: model = model.cuda()

    all_labels = []
    for  targets in y_train:
        all_labels.extend(targets.flatten())
    unique_classes = np.unique(all_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
    gamma = 2  # 聚焦指数（根据需求调整）
    alpha = class_weights
    ce_criterion = FocalLoss(alpha=alpha, gamma=gamma)


    if use_cuda:
        ce_criterion = ce_criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                nesterov=True)


    best_acc = -1
    init_epoch = 0
    if args.resume:
        checkpoint = torch.load(
            'current_checkpoints/' + str(args.dataset) + '_tr-' + str(args.tr_percent) + '_' + model_name + '.pth')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(init_epoch, args.epochs):
        # 调整学习率
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc = train(train_loader, model, ce_criterion, optimizer, epoch,
                                      use_cuda, args)
        with torch.no_grad():
            test_loss, test_acc = test(test_loader, model, ce_criterion, epoch, use_cuda, args)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        print("EPOCH", epoch + 1, "Train Loss", train_loss, "Train Accuracy", train_acc, end=', ')
        print("Test Loss", test_loss, "Test Accuracy", test_acc)

        # save model
        torch.save(state, 'current_checkpoints/' + str(args.dataset) + '_tr-' + str(
            args.tr_percent) + '_' + model_name + '.pth')
        if test_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, 'best_checkpoints/' + str(args.dataset) + '_tr-' + str(
                args.tr_percent) + '_' + model_name + '.pth')
            best_acc = test_acc

    # test
    checkpoint = torch.load(
        'best_checkpoints/' + str(args.dataset) + '_tr-' + str(args.tr_percent) + '_' + model_name + '.pth',weights_only=False)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        test_loss, test_acc = test(test_loader, model, ce_criterion, start_epoch, use_cuda, args)
    pred = predict(full_loader, model, use_cuda)
    prediction = np.argmax(pred, axis=1)
    print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)


    de_map = np.zeros(patchesLabels.shape, dtype=np.int32)
    de_map[total_indices] = prediction + 1
    de_map = np.reshape(de_map, (data_shape[0], data_shape[1]))

    w, h = de_map.shape
    plt.figure(figsize=[h / 100.0, w / 100.0])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.axis('off')


    plt.imshow(de_map, cmap='jet')
    plt.savefig(os.path.join('result/' + str(args.dataset) + '_tr-' + str(
        args.tr_percent) + '_' + model_name + '_classification-map_' + '.png'), format='png')
    plt.close()


    plt.figure(figsize=(15, 5))


    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss', color='blue', linewidth=2)
    plt.plot(test_loss_list, label='Test Loss', color='red', linewidth=2)
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)



    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(test_acc_list, label='Test Accuracy', color='red', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join('result',str(args.dataset)+f'{model_name}_loss_accuracy.png'))
    plt.show()


    patchesLabels = np.reshape(patchesLabels, (data_shape[0], data_shape[1]))
    valid_indices = patchesLabels != 0
    test_pred = de_map[valid_indices]
    test_true = patchesLabels[valid_indices]


    OA = accuracy_score(test_true, test_pred)
    AA = recall_score(test_true, test_pred, average='macro')
    kappa = cohen_kappa_score(test_true, test_pred)
    report_log = F"OA: {OA}\nAA: {AA}\nKappa: {kappa}\n"
    report_log += classification_report(
        test_true, test_pred,
        target_names=class_name,
        digits=4,
    )
    print(report_log)
    fp = open(os.path.join('result', str(args.dataset) + model_name +'classfication_report.txt'), 'w+')
    fp.writelines(report_log)
    fp.close()


    cm = confusion_matrix(test_true, test_pred)
    plot_confusion_matrix(cm, classes=class_name[1:] if 0 in class_name else class_name,  # 排除背景类
                          normalize=True,
                          title=f'Confusion Matrix (OA: {OA:.2%})')

    plt.savefig(os.path.join('result',  str(args.dataset)+f'{model_name}_confusion_matrix.png'),
                bbox_inches='tight', dpi=300)
    plt.close()



if __name__ == '__main__':
    set_seed(42)
    torch.set_num_threads(1)
    main()


