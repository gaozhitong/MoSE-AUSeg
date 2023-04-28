import torch
import numpy as np
import os
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_ece
from utils import get_conf_map_accumulate,m_scheduler,get_model_size


class Engine:
    def __init__(self, exp_config, logger=None, tensorboard=True, args = None):
        self.exp_config = exp_config
        self.logger = logger
        self.args = args

        # Setup Model
        self.net = torch.nn.DataParallel(self.exp_config.net).cuda()
        size = get_model_size(self.net)
        self.logger.info("model size: {:.2f} / MB".format(size))

        # Setup Optimizer and Schedulers.
        self.optimizer = torch.optim.Adam(self.net.module.parameters(), lr=exp_config.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', min_lr=exp_config.scheduler_options['min_lr'], verbose=True,
            patience=exp_config.scheduler_options['patience'],
            factor = exp_config.scheduler_options['factor'])

        self.setup_param_schedulers(exp_config)

        self.load_model_optimizer_weight(exp_config, args)

        # Setup tensorboard
        if tensorboard:
            run_dir = os.path.join(exp_config.run_root, exp_config.log_dir_name, exp_config.experiment_name)
            if os.path.exists(run_dir) and args.overwrite:
                shutil.rmtree(run_dir,  ignore_errors=True)
            self.validation_writer = SummaryWriter(log_dir=run_dir)


    def train(self, data):
        self.net.train()
        self.logger.info('Starting training.')

        self.avg_terms = {}
        self.best_terms = {'loss': np.inf, 'ged': np.inf, 'kl_loss':np.inf,'seg_loss':np.inf}
        losses_list = []

        start_epoch = self.exp_config.start_epoch if hasattr(self.exp_config, 'start_epoch') else 0
        self.epoch = start_epoch

        self.validate(data)

        for self.epoch in tqdm(range(start_epoch, self.exp_config.epochs)):

            for ii,(patch_arrangement, masks_arrangement, prob_gt) in tqdm(enumerate(data.train)):
                patch_arrangement = patch_arrangement.cuda()
                masks_arrangement = masks_arrangement.cuda()
                prob_gt = prob_gt.cuda()
                losses = self.net.forward(patch_arrangement, masks_arrangement, prob_gt)
                loss = losses[0].nanmean()
                with torch.no_grad():
                    losses = [loss_.nanmean().item() for loss_ in losses]
                    losses_list.append(losses)
                del patch_arrangement, masks_arrangement, prob_gt, losses

                self.optimizer.zero_grad()
                try:
                    loss.backward()
                    self.optimizer.step()
                except RuntimeError as e:
                    # Check if memory error occurs
                    if 'CUDA out of memory' in str(e):
                        print('CUDA out of memory error: reducing batch size to {}'.format(self.exp_config.train_batch_size-4))
                        self.exp_config.train_batch_size = max(self.exp_config.train_batch_size-4, 1)
                        data = self.exp_config.data_loader(exp_config=self.exp_config)
                        break
                    else:
                        raise e
                del loss

            mean_losses = np.array(losses_list).mean(0)
            self.scheduler.step(mean_losses[0])
            self.update_param_schedulers(mean_losses)

            # self.logger.info('- Gamma: ' + str(self.net.module.loss_fn.gamma))
            if (self.epoch) % self.exp_config.logging_frequency == 0 and self.epoch > 1:
                # self.logger.info('epoch {} Loss {}'.format(self.epoch, losses[0])) \
                self.logger.info(' - Mean Train Loss: ' + str(mean_losses))
                self.avg_terms = {'Mean_Train_Loss_{}'.format(i): mean_loss for i, mean_loss in enumerate(mean_losses)}
                self.avg_terms.update({'Learning_Rate': self.scheduler._last_lr[0],
                                       'gamma': self.exp_config.loss_fn.gamma,
                                       'G': self.exp_config.loss_fn.G})

            if (self.epoch) % self.exp_config.validation_frequency == 0:
                self.validate(data)
                self.tensorboard_summary()

        self.logger.info('Best GED score! (%.3f)' %  self.best_terms['ged'])
        self.logger.info('Finished training.')

    def validate(self, data, vis_batch_size = 3):
        self.logger.info('Validation for step {}'.format(self.epoch))
        self.net.eval()

        with torch.no_grad():
            metrics_list = []
            for ii,(patch_arrangement, masks_arrangement, prob_gt) in enumerate(data.validation):
                patch_arrangement = patch_arrangement.cuda()
                masks_arrangement = masks_arrangement.cuda()
                prob_gt = prob_gt.cuda()
                metrics,prediction,prob = self.net.forward(patch_arrangement, masks_arrangement, prob_gt, val = True)
                if isinstance(metrics, list):
                    metrics = [metric_.nanmean().item() for metric_ in metrics]  # get mean if parallel is used
                else:
                    metrics = [metrics.nanmean().item()]
                metrics_list.append(metrics)

            del patch_arrangement, masks_arrangement, prob_gt,prediction
            mean_metrics = torch.tensor(metrics_list).mean(0)
            self.logger.info(' - Mean Val Metrics:  ' + str(mean_metrics))
            self.avg_terms['Mean_Val_GED'] = mean_metrics[0].item()

            if mean_metrics[0] <= self.best_terms['ged']:
                self.best_terms['ged'] = mean_metrics[0]
                self.logger.info('New best GED score! (%.3f)' %  self.best_terms['ged'])
                self.save_model(savename='best_ged')
                self.save_optim(savename='best_ged')

        self.net.train()

    def test(self, dataloader, log_dir):
        self.net.eval()
        with torch.no_grad():
            metrics_list = []
            all_pixel_conf, all_gt_conf = None, None

            for ii, (patch_arrangement, masks_arrangement, prob_gt) in enumerate(tqdm(dataloader)):

                patch_arrangement = patch_arrangement.cuda()
                masks_arrangement = masks_arrangement.cuda()
                prob_gt = prob_gt.cuda()
                metrics,prediction,prob = self.net.forward(patch_arrangement, masks_arrangement, prob_gt, val=True)
                metrics_list.append([metric_.nanmean().item() for metric_ in metrics])  # get mean if parallel is used
                all_pixel_conf, all_gt_conf = get_conf_map_accumulate(prediction, prob, masks_arrangement, prob_gt,
                                                               self.exp_config.n_classes, all_pixel_conf, all_gt_conf)
                del patch_arrangement,masks_arrangement,prob_gt, metrics, prediction
                torch.cuda.empty_cache()
            # calculate ECE
            ece = calculate_ece(all_pixel_conf.transpose(0,1), all_gt_conf, n_bins=10, label_range = self.exp_config.eval_class_ids)

            mean_metrics = torch.tensor(metrics_list).nanmean(0)
            mean_metrics =  torch.cat([mean_metrics, torch.tensor(ece).unsqueeze(0)])

            self.logger.info(' - Mean GED: {:.3f}'.format(mean_metrics[0]))
            self.logger.info(' - Mean M-IoU: {:.3f}'.format(mean_metrics[1]))
            self.logger.info(' - Mean ECE: {:.3%}'.format(mean_metrics[2]))

            sample_num = self.net.module.eval_sample_num
            sample_per_mode = self.net.module.sample_per_mode
            eval_name = 'sample_{}'.format(sample_num) if sample_num is not None else 'weighted_{}'.format(
                sample_per_mode)
            np.save(os.path.join(log_dir, '{}_mean_metrics.npy'.format(eval_name)),  np.array(mean_metrics))

    def load_model_optimizer_weight(self,exp_config, args):

        # inference time, select loading model
        if args.demo != '':
            model_selection = exp_config.experiment_name + '_best_ged.pth'

            model_path = os.path.join(
                exp_config.log_root,
                exp_config.log_dir_name,
                exp_config.experiment_name,
                model_selection)

            self.net.load_state_dict(torch.load(model_path), strict=True)
            self.optimizer.load_state_dict(torch.load(model_path.replace('.pth', '_optim.pth')))
            self.logger.info('Loading model {}'.format(model_path))

        # Training with pretrained model
        elif hasattr(exp_config, 'pretrained_model_full_path') and exp_config.pretrained_model_full_path!= '':
            if os.path.exists(exp_config.pretrained_model_full_path):
                self.logger.info('Loading pretrained model {}'.format(exp_config.pretrained_model_full_path))
                self.net.load_state_dict(torch.load(exp_config.pretrained_model_full_path), strict=True)
                self.optimizer.load_state_dict(torch.load(exp_config.pretrained_model_full_path.replace('.pth', '_optim.pth')))
            else:
                self.logger.info('The file {} does not exist. Starting training without pretrained net.'
                                 .format(exp_config.pretrained_model_full_path))\

    def setup_param_schedulers(self, exp_config):
        if hasattr(exp_config, 'gamma_scheduler_options'):
            self.gamma_scheduler = m_scheduler(exp_config.loss_fn, 'gamma',
                                           exp_config.gamma_scheduler_options['updated_values'],
                                           exp_config.gamma_scheduler_options['patience'], logger=self.logger)
        else:
            self.gamma_scheduler = None

        if hasattr(exp_config, 'G_scheduler_options'):
            self.G_scheduler = m_scheduler(exp_config.loss_fn, 'G',
                                               exp_config.G_scheduler_options['updated_values'],
                                               exp_config.G_scheduler_options['patience'], logger=self.logger)
        else:
            self.G_scheduler = None
        if hasattr(exp_config, 'S_scheduler_options'):
            self.S_scheduler = m_scheduler(exp_config.net, 'sample_per_mode',
                                           exp_config.S_scheduler_options['updated_values'],
                                           exp_config.S_scheduler_options['patience'], logger=self.logger)
        else:
            self.S_scheduler = None

    def update_param_schedulers(self,mean_losses):
        if self.gamma_scheduler is not None:
            self.gamma_scheduler.step(mean_losses[0])
        if self.G_scheduler is not None:
            self.G_scheduler.step(mean_losses[-1])
        if self.S_scheduler is not None:
            self.S_scheduler.step(mean_losses[0])

    def save_model(self, savename):
        model_name = self.exp_config.experiment_name + '_' + savename + '.pth'

        log_dir = os.path.join(self.exp_config.log_root, self.exp_config.log_dir_name, self.exp_config.experiment_name)
        save_model_path = os.path.join(log_dir, model_name)
        torch.save(self.net.state_dict(), save_model_path)
        self.logger.info('saved model to .pth file in {}'.format(save_model_path))

    def save_optim(self, savename):
        model_name = self.exp_config.experiment_name + '_' + savename + '_optim.pth'
        log_dir = os.path.join(self.exp_config.log_root, self.exp_config.log_dir_name, self.exp_config.experiment_name)
        save_optim_path = os.path.join(log_dir, model_name)
        torch.save(self.optimizer.state_dict(), save_optim_path)

    def tensorboard_summary(self):
        # plot scalars
        for key in self.avg_terms:
            self.validation_writer.add_scalar(key, self.avg_terms[key], global_step=self.epoch)