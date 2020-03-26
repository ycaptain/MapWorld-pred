import argparse
import os
import sys
import torch

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

import trainer as trains
from utils import init
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    data_loader, model, criterion, metrics = init(config)

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # checkpoint = torch.load(config.resume)
    # state_dict = checkpoint['state_dict']
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)

    tester = config.init_obj('trainer', trains, model, criterion, metrics, config, data_loader)
    total_loss, total_metrics = tester.test()

    # prepare model for testing
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # model.eval()

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metrics))

    # with torch.no_grad():
    #     for i, (image_ids, data, target) in enumerate(tqdm(data_loader)):
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #
    #         for image_id, logit in zip(image_ids, output):
    #             filename = os.path.join(logit_dir, image_id + ".npy")
    #             np.save(filename, logit.cpu().numpy())

            # # img = Image.fromarray(labmap.astype(np.uint8))
            # # img.save("saved/test/"+str(i)+".tif")
            # Image.fromarray(np.moveaxis(data.cpu().numpy().astype(np.uint8)[0], 0, -1)).show()
            # tmp = Image.fromarray(labmap.astype(np.uint8))
            # Image.eval(tmp, lambda a: 255 if a >= 1 else 0).show()
            # if j == 10:
            #     break
            # j+=1
            # j = 0
            # for label in output:
            #     label = label.float().cpu().numpy()
            #     label = Image.fromarray(label.astype(np.uint8))
            #     label.save("saved/test/"+str(j)+".tif")
            #     j += 1

            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            # batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
