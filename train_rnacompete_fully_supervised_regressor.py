import os
import sys
import torch
import argparse
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr

from supervised_encoder_models.task_dataloader import TaskFolder, rnacompete_all_rbps, \
    read_rnacompete_datafile, rnacompete_train_datapath, rnacompete_test_datapath
from supervised_encoder_models.supervised_encoder_model import FULL_ENC_Model
import lib.plot_utils, lib.logger

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='mlp')
parser.add_argument('--rbp_name', type=str, default='PTB')
parser.add_argument('--hidden_size', type=eval, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--mode', type=str, default='lstm')
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--normalize_target', type=eval, default=True, choices=[True, False])


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

            all_preds.append(ret_dict['preds'])
            all_label.extend(batch_label)

    all_loss /= size
    pearson_corr = pearsonr(np.array(all_label)[:, 0], np.concatenate(all_preds, axis=0)[:, 0])[0]

    return all_loss, pearson_corr


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)
    rbp_name = args.rbp_name
    if rbp_name == 'all':
        pass
        all_rbps = rnacompete_all_rbps
    else:
        assert rbp_name in rnacompete_all_rbps
        all_rbps = [rbp_name]

    preprocess_type = args.mode
    input_size = 128  # latent dimension
    output_size = 1
    train_val_split_ratio = 0.1

    for rbp_name in all_rbps:

        train_datapath_filled = rnacompete_train_datapath.format(rbp_name)
        test_datapath_filled = rnacompete_test_datapath.format(rbp_name)

        train_seq, train_targets = read_rnacompete_datafile(train_datapath_filled)
        test_seq, test_targets = read_rnacompete_datafile(test_datapath_filled)
        train_targets = np.array(train_targets)[:, None]
        test_targets = np.array(test_targets)[:, None]

        if args.normalize_target is True:
            print('Note: normalizing training targets to [0, 1]')
            offset = np.min(train_targets)
            diff = np.max(train_targets) - offset

            train_targets = (np.array(train_targets) - offset) / diff
            test_targets = (np.array(test_targets) - offset) / diff
            loss_type = 'binary_ce'
        else:
            loss_type = 'mse'

        valid_idx = np.random.choice(np.arange(len(train_targets)), int(len(train_targets) * train_val_split_ratio),
                                     replace=False)
        valid_idx = np.array(valid_idx)
        train_idx = np.setdiff1d(np.arange(len(train_targets)), valid_idx)

        valid_seq = np.array(train_seq)[valid_idx]
        valid_targets = np.array(train_targets)[valid_idx]
        val_size = len(valid_seq)

        train_seq = np.array(train_seq)[train_idx]
        train_targets = np.array(train_targets)[train_idx]
        train_size = len(train_seq)

        save_dir = os.path.join('full-rnacompete-regressor', args.save_dir, rbp_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            continue

        outfile = open(os.path.join(save_dir, '%s.out' % (args.save_dir)), "w")
        sys.stdout = outfile
        sys.stderr = outfile

        model = FULL_ENC_Model(input_size, args.hidden_size, output_size, device=device,
                               vae_type=preprocess_type, loss_type=loss_type).to(device)
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

        all_fields = ['epoch', 'train_loss', 'valid_loss', 'train_pearson_corr', 'valid_pearson_corr']

        logger = lib.logger.CSVLogger('run.csv', save_dir, all_fields)

        best_valid_loss = np.inf
        best_valid_weight_path = None

        train_loader = TaskFolder(train_seq, train_targets, args.batch_size, shuffle=True,
                                  preprocess_type=preprocess_type, num_workers=8)
        valid_loader = TaskFolder(valid_seq, valid_targets, args.batch_size, shuffle=False,
                                  preprocess_type=preprocess_type, num_workers=8)
        test_loader = TaskFolder(test_seq, test_targets, args.batch_size, shuffle=False,
                                 preprocess_type=preprocess_type, num_workers=8)
        last_improved = 0
        last_2_epochs = []
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
            train_loss, train_pearson_corr = evaluate(train_loader)
            valid_loss, valid_pearson_corr = evaluate(valid_loader)

            lib.plot_utils.plot('train_loss', train_loss)
            lib.plot_utils.plot('train_pearson_corr', train_pearson_corr)
            lib.plot_utils.plot('valid_loss', valid_loss)
            lib.plot_utils.plot('valid_pearson_corr', valid_pearson_corr)

            lib.plot_utils.set_xlabel_for_tick(index=0, label='epoch')
            lib.plot_utils.flush()
            lib.plot_utils.tick(index=0)

            print(
                'Epoch %d, train_loss: %.2f, train_pearson_corr: %2f, '
                'valid_loss: %.2f, valid_pearson_corr: %.2f' %
                (epoch, train_loss, train_pearson_corr,
                 valid_loss, valid_pearson_corr))

            logger.update_with_dict({
                'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss,
                'train_pearson_corr': train_pearson_corr, 'valid_pearson_corr': valid_pearson_corr
            })

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if len(last_2_epochs) >= 2:
                    to_remove_epoch = last_2_epochs.pop(0)
                    os.remove(os.path.join(save_dir, "model.epoch-" + str(to_remove_epoch)))
                last_2_epochs.append(epoch)
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
        test_loss, test_pearson_corr = evaluate(test_loader)
        print('Test pearson corr:', test_pearson_corr)

        logger.close()
