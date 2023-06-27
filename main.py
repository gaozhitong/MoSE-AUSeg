import os
import shutil
from engine import Engine
from utils import utils
from config.parse_arg import args
from importlib.machinery import SourceFileLoader

if __name__ == '__main__':
    utils.random_init(0)

    # Config file
    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, config_file).load_module()

    log_dir = os.path.join(exp_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)
    utils.makefolder(log_dir)

    if args.demo == '':
        basic_logger = utils.setup_logger('basic_logger', log_dir + '/training_log.log')
    else:
        sample_num = exp_config.net.eval_sample_num
        basic_logger = utils.setup_logger('basic_logger',
                                          log_dir + '/inference_{}_{}_log.log'.format(args.demo, sample_num))

    basic_logger.info('********Running Experiment: %s********', exp_config.experiment_name)

    # Prepare Data
    data = exp_config.data_loader(exp_config=exp_config)

    # Start Inference or Training
    if args.demo != '':
        model = Engine(exp_config, logger=basic_logger, args=args, tensorboard=False)
        if args.demo == 'train':
            data_demo = data.train
        elif args.demo == 'val':
            data_demo = data.validation
        elif args.demo == 'test':
            data_demo = data.test

        save_path = log_dir + '/{}'.format(args.demo)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model.test(data_demo, save_path)

    else:
        model = Engine(exp_config, logger=basic_logger, args=args, tensorboard=True)
        shutil.copy(exp_config.__file__, log_dir)
        model.train(data)
        model.save_model('last')
