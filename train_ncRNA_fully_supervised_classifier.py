import os
import sys
import torch
import argparse
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

from supervised_encoder_models.task_dataloader import TaskFolder, ncRNA_all_classes, \
    read_ncRNA_fasta, ncRNA_train_datapath, ncRNA_test_datapath
from supervised_encoder_models.supervised_encoder_model import FULL_ENC_Model
import lib.plot_utils, lib.logger

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='mlp')
parser.add_argument('--hidden_size', type=eval, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--mode', type=str, default='lstm')
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--epoch', type=int, default=100)


def evaluate(loader):
    all_loss = 0.
    size = loader.size
    all_preds = []
    all_label = []

    with torch.no_grad():
        for batch_input, batch_label in loader:
            # compute various metrics
            ret_dict = model(batch_input, batch_label)
            all_loss += ret_dict['loss'].item()

            all_preds.extend(np.argmax(ret_dict['preds'], axis=-1))
            all_label.extend(batch_label)

    all_loss /= size
    acc = np.sum(np.array(all_preds) == np.array(all_label)) / size
    f1_macro = f1_score(all_preds, all_label, average='macro')
    f1_micro = f1_score(all_preds, all_label, average='micro')
    mcc = matthews_corrcoef(all_preds, all_label)

    return all_loss, acc, f1_macro, f1_micro, mcc


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    preprocess_type = args.mode
    input_size = 128  # latent dimension
    output_size = len(ncRNA_all_classes)
    train_val_split_ratio = 0.1

    train_seq, train_label = read_ncRNA_fasta(ncRNA_train_datapath)
    test_seq, test_label = read_ncRNA_fasta(ncRNA_test_datapath)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    valid_idx = []
    for label in range(output_size):
        label_idx = np.where(train_label == label)[0]
        nb_label = len(label_idx)
        valid_idx.extend(np.random.choice(label_idx, int(nb_label * train_val_split_ratio), replace=False))
    valid_idx = np.array(valid_idx)
    train_idx = np.setdiff1d(np.arange(len(train_label)), valid_idx)

    valid_seq = np.array(train_seq)[valid_idx]
    valid_label = np.array(train_label)[valid_idx]
    val_size = len(valid_seq)

    train_seq = np.array(train_seq)[train_idx]
    train_label = np.array(train_label)[train_idx]
    train_size = len(train_label)

    save_dir = os.path.join('full-ncRNA-classification', args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    outfile = open(os.path.join(save_dir, '%s.out' % (args.save_dir)), "w")
    sys.stdout = outfile
    sys.stderr = outfile

    model = FULL_ENC_Model(input_size, args.hidden_size, output_size, device=device,
                           vae_type=preprocess_type, loss_type='ce').to(device)
    print(model)
    import torch.nn as nn

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        elif param.dim() >= 2:
            nn.init.xavier_normal_(param)

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    from importlib import reload

    reload(lib.plot_utils)
    lib.plot_utils.set_output_dir(save_dir)
    lib.plot_utils.suppress_stdout()

    all_fields = ['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc',
                  'train_f1_macro', 'valid_f1_macro', 'train_f1_micro', 'valid_f1_micro',
                  'train_mcc', 'valid_mcc']

    logger = lib.logger.CSVLogger('run.csv', save_dir, all_fields)

    best_valid_loss = np.inf
    best_valid_weight_path = None

    train_loader = TaskFolder(train_seq, train_label, args.batch_size, shuffle=True,
                              preprocess_type=preprocess_type, num_workers=8)
    valid_loader = TaskFolder(valid_seq, valid_label, args.batch_size, shuffle=False,
                              preprocess_type=preprocess_type, num_workers=8)
    test_loader = TaskFolder(test_seq, test_label, args.batch_size, shuffle=False,
                             preprocess_type=preprocess_type, num_workers=8)
    last_improved = 0
    last_5_epochs = []
    for epoch in range(1, args.epoch + 1):
        if last_improved >= 20:
            print('Have\'t improved for %d epochs' % (last_improved))
            break

        # training loop
        model.train()
        for batch_input, batch_label in train_loader:
            model.zero_grad()
            ret_dict = model(batch_input, batch_label)
            loss = ret_dict['loss'] / ret_dict['nb_preds']
            # print(sum(np.argmax(ret_dict['preds'], axis=-1) == np.array(batch_label)) / ret_dict['nb_preds'])
            loss.backward()
            optimizer.step()

        model.eval()
        # validation loop
        train_loss, train_acc, train_f1_macro, train_f1_micro, train_mcc = evaluate(train_loader)
        valid_loss, valid_acc, valid_f1_macro, valid_f1_micro, valid_mcc = evaluate(valid_loader)

        lib.plot_utils.plot('train_loss', train_loss)
        lib.plot_utils.plot('train_acc', train_acc)
        lib.plot_utils.plot('train_f1_macro', train_f1_macro)
        lib.plot_utils.plot('train_f1_micro', train_f1_micro)
        lib.plot_utils.plot('train_mcc', train_mcc)

        lib.plot_utils.plot('valid_loss', valid_loss)
        lib.plot_utils.plot('valid_acc', valid_acc)
        lib.plot_utils.plot('valid_f1_macro', valid_f1_macro)
        lib.plot_utils.plot('valid_f1_micro', valid_f1_micro)
        lib.plot_utils.plot('valid_mcc', valid_mcc)

        lib.plot_utils.set_xlabel_for_tick(index=0, label='epoch')
        lib.plot_utils.flush()
        lib.plot_utils.tick(index=0)

        print(
            'Epoch %d, train_loss: %.2f, train_acc: %2f, train_f1_macro: %.2f, train_f1_micro: %.2f, train_mcc: %.2f, '
            'valid_loss: %.2f, valid_acc: %.2f, valid_f1_macro: %.2f, valid_f1_micro: %.2f, valid_mcc: %.2f' %
            (epoch, train_loss, train_acc, train_f1_macro, train_f1_micro, train_mcc,
             valid_loss, valid_acc, valid_f1_macro, valid_f1_micro, valid_mcc,))

        logger.update_with_dict({
            'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss,
            'train_f1_macro': train_f1_macro, 'train_f1_micro': train_f1_micro,
            'train_mcc': train_mcc, 'valid_f1_macro': valid_f1_macro,
            'valid_f1_micro': valid_f1_micro, 'valid_mcc': valid_mcc,
            'train_acc': train_acc, 'valid_acc': valid_acc
        })

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if len(last_5_epochs) >= 5:
                to_remove_epoch = last_5_epochs.pop(0)
                os.remove(os.path.join(save_dir, "model.epoch-" + str(to_remove_epoch)))
            last_5_epochs.append(epoch)
            best_valid_weight_path = os.path.join(save_dir, "model.epoch-" + str(epoch))
            torch.save(
                {'model_weights': model.state_dict(),
                 'opt_weights': optimizer.state_dict()},
                best_valid_weight_path)
            print('Validation loss improved, saving current weights to path:', best_valid_weight_path)
            last_improved = 0
        else:
            last_improved += 1

    if best_valid_weight_path is not None:
        print('Loading best weights from: %s' % (best_valid_weight_path))
        model.load_state_dict(torch.load(best_valid_weight_path)['model_weights'])

    model.eval()
    test_loss, test_acc, test_f1_macro, test_f1_micro, test_mcc = evaluate(test_loader)
    print('Test acc:', test_acc)

    logger.close()
