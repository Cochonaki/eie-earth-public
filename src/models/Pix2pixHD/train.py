import time
import os
import numpy as np
import torch
from collections import OrderedDict
import math

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import util.util as util

def lcm(a, b): return abs(a * b) // math.gcd(a, b) if a and b else 0

# Print environment info
print(f"torch.version = {torch.__version__}")
print(f"torch.version.cuda = {torch.version.cuda}")
print(f"torch.backends.cudnn.version() = {torch.backends.cudnn.version()}")
print(f"torch.cuda.memory_summary(): \n {torch.cuda.memory_summary()}")

# Parse options
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

# Resume logic
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except Exception:
        start_epoch, epoch_iter = 1, 0
    print(f"Resuming from epoch {start_epoch} at iteration {epoch_iter}")
else:
    start_epoch, epoch_iter = 1, 0

# Adjust frequencies for batchSize
opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

# Load data
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(f"#training images = {dataset_size}")

# Create and wrap model
model, optimizer_G, optimizer_D = create_model(opt)

# Optionally use FP16 training with Apex
if opt.fp16:
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [optimizer_G, optimizer_D], opt_level='O1')

visualizer = Visualizer(opt)
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

# Training loop
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()

        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # Handle L1 loss decay (if enabled)
        if epoch > opt.l1_pre_l1_loss_decay_epoch and opt.lambda_l1 > 0:
            e_lambda_l1 = max(opt.lambda_l1 - opt.lambda_l1_decay * (epoch - opt.l1_pre_l1_loss_decay_epoch), 0)
        else:
            e_lambda_l1 = opt.lambda_l1

        # Adjust sampling probability (if enabled)
        if epoch > opt.l1_pre_l1_loss_decay_epoch:
            e_l1_pre_prob = max(1 - opt.l1_pre_prob_decay * (epoch - opt.l1_pre_l1_loss_decay_epoch), 0.5)
        else:
            e_l1_pre_prob = 1

        save_fake = total_steps % opt.display_freq == display_delta

        # Forward
        label = data['label'].to(opt.gpu_ids[0], non_blocking=True)
        inst  = data['inst'].to(opt.gpu_ids[0],    non_blocking=True)
        img   = data['image'].to(opt.gpu_ids[0],   non_blocking=True)
        feat  = data['feat'].to(opt.gpu_ids[0],    non_blocking=True)
        losses, generated = model(label, inst, img, feat, infer=save_fake)

        # Loss aggregation
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + \
                 loss_dict.get('G_VGG', 0) + loss_dict.get('G_LPIPS', 0) + \
                 loss_dict.get('G_L1_pre', 0)

        # Backward G
        optimizer_G.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_G.backward()
        optimizer_G.step()

        # Backward D
        optimizer_D.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_D.backward()
        optimizer_D.step()

        # Print errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        # Visual display
        if save_fake:
            visuals = OrderedDict([
                ('input_mask', util.tensor2label(data['mask'][0], opt.label_nc)),
                ('synthesized_post-flood', util.tensor2im(generated.data[0])),
                ('input_pre-flood', util.tensor2im(data['image_A'][0])),
                ('real_post-flood', util.tensor2im(data['image'][0]))
            ])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # Save latest
        if total_steps % opt.save_latest_freq == save_delta:
            print(f"Saving the latest model (epoch {epoch}, total_steps {total_steps})")
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.1f} sec')

    # Save at epoch
    if epoch % opt.save_epoch_freq == 0:
        print(f"Saving the model at the end of epoch {epoch}, total_steps {total_steps}")
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    # Fix parameters if needed
    if opt.niter_fix_global != 0 and epoch == opt.niter_fix_global:
        model.module.update_fixed_params()

    # Decay learning rate
    if epoch > opt.niter:
        model.module.update_learning_rate()
