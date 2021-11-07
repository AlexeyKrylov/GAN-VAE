import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()    # get training options
    dataset = create_dataset(opt)   # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)     # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)       # create a model given opt.model and other options
    model.setup(opt)                # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)    # create a visualizer that display/save images and plots in neptune.ai
    total_iters = 0                 # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + 1):     # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        lossG, lossD = 0, 0
        for i, data in enumerate(dataset): # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)       # unpack data from dataset and apply preprocessing
            tmp_lossG, tmp_lossD = model.optimize_parameters() # calculate loss functions, get gradients, update network weights
            lossG += tmp_lossG
            lossD += tmp_lossD
            iter_data_time = time.time()

        if epoch % opt.print_freq == 0:
            model.compute_visuals(epoch)
            visualizer.upload_current_visuals(epoch)
            losses = model.get_current_losses([lossG, lossD], epoch_iter)
            visualizer.print_current_losses(epoch, losses)
            visualizer.upload_current_losses(losses)
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks(epoch)