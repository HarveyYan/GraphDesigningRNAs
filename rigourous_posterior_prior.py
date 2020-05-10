import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import trange

from baseline_models.FlowLSTMVAE import LSTMVAE, BasicLSTMVAEFolder
import baseline_models.baseline_metrics
from baseline_models.baseline_metrics import evaluate_prior, evaluate_posterior
import lib.plot_utils, lib.logger

if __name__ == "__main__":

    save_dir = 'lstm_baseline_output/rigorosity-[20200429-222820-flow-prior-resumed-5e-4-1e-2]/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lib.plot_utils.set_output_dir(save_dir)
    lib.plot_utils.suppress_stdout()
    logger = lib.logger.CSVLogger('run.csv', save_dir,
                                  ['Epoch',
                                   'Validation_recon_acc_with_reg', 'Validation_post_valid_with_reg',
                                   'Validation_post_fe_deviation_with_reg',
                                   'Validation_recon_acc_no_reg', 'Validation_post_valid_no_reg',
                                   'Validation_post_fe_deviation_no_reg',
                                   'Validation_recon_acc_no_reg_greedy', 'Validation_post_valid_no_reg_greedy',
                                   'Validation_post_fe_deviation_no_reg_greedy',
                                   'Prior_valid_with_reg', 'Prior_fe_deviation_with_reg', 'Prior_valid_no_reg',
                                   'Prior_fe_deviation_no_reg',
                                   'Prior_valid_no_reg_greedy', 'Prior_fe_deviation_no_reg_greedy',
                                   'Prior_uniqueness_no_reg_greedy'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # save_dir = '/home/zichao/scratch/JTRNA/lstm_baseline_output/20200429-223941-flow-prior-limited-data-10-1e-4-1e-2'
    expr_dir = 'lstm_baseline_output/20200429-222820-flow-prior-resumed-5e-4-1e-2'

    epochs_to_load = []
    for dirname in os.listdir(expr_dir):
        if dirname.startswith('model'):
            epochs_to_load.append(int(dirname.split('-')[-1]))
    epochs_to_load = list(np.sort(epochs_to_load))
    print(epochs_to_load)
    mp_pool = Pool(20)

    for enc_epoch_to_load in epochs_to_load:
        model = LSTMVAE(512, 128, 2, device=device, use_attention=True).to(device)

        weight_path = os.path.join(expr_dir, 'model.epoch-%d'%(enc_epoch_to_load))
        print('Loading', weight_path)
        model.load_state_dict(
            torch.load(weight_path, map_location=device)['model_weights'])

        baseline_models.baseline_metrics.model = model
        valid_batch_size = 512
        loader = BasicLSTMVAEFolder('data/rna_jt_32-512/validation-split', valid_batch_size, num_workers=2)
        nb_iters = 20000 // valid_batch_size  # 20000 is the size of the validation set
        total = 0
        # bar = trange(nb_iters, desc='', leave=True)
        # loader = loader.__iter__()
        nb_encode, nb_decode = 5, 5

        recon_acc, post_valid, post_fe_deviation = 0, 0, 0.
        recon_acc_noreg, post_valid_noreg, post_fe_deviation_noreg = 0, 0, 0.
        recon_acc_noreg_det, post_valid_noreg_det, post_fe_deviation_noreg_det = 0, 0, 0.

        with torch.no_grad():

            # for i in bar:
            for i, (original_data, batch_sequence, batch_label, batch_fe) in enumerate(loader):

                # original_data, batch_sequence, batch_label, batch_fe = next(loader)
                latent_vec = model.encode(batch_sequence)

                batch_recon_acc, batch_post_valid, batch_post_fe_deviation = \
                    evaluate_posterior(list(np.array(original_data)[:, 0]), list(np.array(original_data)[:, 1]),
                                       latent_vec, mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                       enforce_rna_prior=True)

                total += nb_encode * nb_decode * valid_batch_size
                recon_acc += np.sum(batch_recon_acc)
                post_valid += np.sum(batch_post_valid)
                post_fe_deviation += np.sum(batch_post_fe_deviation)

                batch_recon_acc, batch_post_valid, batch_post_fe_deviation = \
                    evaluate_posterior(list(np.array(original_data)[:, 0]), list(np.array(original_data)[:, 1]),
                                       latent_vec, mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                       enforce_rna_prior=False)

                recon_acc_noreg += np.sum(batch_recon_acc)
                post_valid_noreg += np.sum(batch_post_valid)
                post_fe_deviation_noreg += np.sum(batch_post_fe_deviation)

                batch_recon_acc, batch_post_valid, batch_post_fe_deviation = \
                    evaluate_posterior(list(np.array(original_data)[:, 0]), list(np.array(original_data)[:, 1]),
                                       latent_vec, mp_pool, nb_encode=nb_encode, nb_decode=nb_decode, prob_decode=False,
                                       enforce_rna_prior=False)

                recon_acc_noreg_det += np.sum(batch_recon_acc)
                post_valid_noreg_det += np.sum(batch_post_valid)
                post_fe_deviation_noreg_det += np.sum(batch_post_fe_deviation)

            #     bar.set_description(
            #         'streaming recon acc: %.2f, streaming post valid: %.2f, streaming post free energy deviation: %.2f'
            #         % (recon_acc / total * 100, post_valid / total * 100, post_fe_deviation / post_valid))
            #
            # bar.refresh()

            # posterior decoding with enforced RNA regularity
            lib.plot_utils.plot('Validation_recon_acc_with_reg', recon_acc / total * 100)
            lib.plot_utils.plot('Validation_post_valid_with_reg', post_valid / total * 100)
            lib.plot_utils.plot('Validation_post_fe_deviation_with_reg', post_fe_deviation / post_valid)

            # posterior decoding without RNA regularity
            lib.plot_utils.plot('Validation_recon_acc_no_reg', recon_acc_noreg / total * 100)
            lib.plot_utils.plot('Validation_post_valid_no_reg', post_valid_noreg / total * 100)
            lib.plot_utils.plot('Validation_post_fe_deviation_no_reg', post_fe_deviation_noreg / post_valid_noreg)

            # posterior decoding without RNA regularity and deterministic
            lib.plot_utils.plot('Validation_recon_acc_no_reg_greedy', recon_acc_noreg_det / total * 100)
            lib.plot_utils.plot('Validation_post_valid_no_reg_greedy', post_valid_noreg_det / total * 100)
            lib.plot_utils.plot('Validation_post_fe_deviation_no_reg_greedy', post_fe_deviation_noreg_det / post_valid_noreg_det)

            prior_valid_reg_sto, prior_fe_deviation_reg_sto = 0., 0.
            prior_valid_noreg_sto, prior_fe_deviation_noreg_sto = 0., 0.
            prior_valid_noreg_det, prior_fe_deviation_noreg_det, prior_uniqueness_noreg_det = 0., 0., 0

            ####################### sampling from the prior ########################
            sampled_latent_prior = torch.as_tensor(np.random.randn(10000, 128).astype(np.float32)).to(
                device)
            if True:
                sampled_latent_prior = model.latent_cnf(sampled_latent_prior, None, reverse=True).view(
                    *sampled_latent_prior.size())

            ######################## evaluate prior with regularity constraints ########################
            prior_valid, prior_fe_deviation, _, _ = evaluate_prior(sampled_latent_prior, 10000, 10, mp_pool,
                                                                   enforce_rna_prior=True)
            prior_valid_reg_sto += np.sum(prior_valid)
            prior_fe_deviation_reg_sto += np.sum(prior_fe_deviation)

            ######################## evaluate prior without regularity constraints ########################
            prior_valid, prior_fe_deviation, _, _ = evaluate_prior(sampled_latent_prior, 10000, 10, mp_pool,
                                                                   enforce_rna_prior=False)

            prior_valid_noreg_sto += np.sum(prior_valid)
            prior_fe_deviation_noreg_sto += np.sum(prior_fe_deviation)

            ######################## evaluate prior without regularity constraints and greedy ########################
            prior_valid, prior_fe_deviation, decoded_seq, _ = evaluate_prior(sampled_latent_prior, 10000, 1, mp_pool,
                                                                             enforce_rna_prior=False, prob_decode=False)

            prior_valid_noreg_det += np.sum(prior_valid)
            prior_fe_deviation_noreg_det += np.sum(prior_fe_deviation)
            prior_uniqueness_noreg_det += len(set(decoded_seq))

            lib.plot_utils.plot('Prior_valid_with_reg', prior_valid_reg_sto / 1000)
            lib.plot_utils.plot('Prior_fe_deviation_with_reg', prior_fe_deviation_reg_sto / prior_valid_reg_sto)

            lib.plot_utils.plot('Prior_valid_no_reg', prior_valid_noreg_sto / 1000)
            lib.plot_utils.plot('Prior_fe_deviation_no_reg', prior_fe_deviation_noreg_sto / prior_valid_noreg_sto)

            lib.plot_utils.plot('Prior_valid_no_reg_greedy', prior_valid_noreg_det / 1000)
            lib.plot_utils.plot('Prior_fe_deviation_no_reg_greedy', prior_fe_deviation_noreg_det / prior_valid_noreg_det)
            lib.plot_utils.plot('Prior_uniqueness_no_reg_greedy', prior_uniqueness_noreg_det / 100)

            tocsv = {'Epoch': enc_epoch_to_load}
            for name, val in lib.plot_utils._since_last_flush.items():
                tocsv[name] = list(val.values())[0]
            logger.update_with_dict(tocsv)

            lib.plot_utils.set_xlabel_for_tick(index=0, label='epoch')
            lib.plot_utils.flush()
            lib.plot_utils.tick(index=0)

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
