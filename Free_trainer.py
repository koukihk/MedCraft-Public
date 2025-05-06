import os
import shutil
import time
from utils.utils import dice, resample_3d
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from models.TumorGAN import TGAN
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
import torch.nn as nn
from monai.data import decollate_batch
from monai.losses import DiceLoss
from utils.mixup import mixup


def syn_data(images, labels, Tgan, args):
    b = images.size()[0]
    syn_images, syn_labels = [], []
    mode = 'generate'
    for i in range(b):
        img = images[i, :, :, :, :].unsqueeze(0)
        lab = labels[i, :, :, :, :].unsqueeze(0)

        ls = list(torch.unique(lab))

        if (2 not in ls) and (1 in ls):
            # Only unlabeled data (without tumor labels) for synthesis !!!
            _, _, _, _, syn_img, syn_lab = Tgan(img, lab, mode, args)
        else:
            syn_img, syn_lab = img, lab

        syn_images.append(syn_img)
        syn_labels.append(syn_lab)

    syn_images = torch.concat(syn_images, dim=0)
    syn_labels = torch.concat(syn_labels, dim=0)
    return syn_images, syn_labels


def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, loss_func, args, Tgan):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        image, label = batch_data["image"], batch_data["label"]
        image, label = image.cuda(), label.cuda()

        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            if args.task == 'freesyn':
                with torch.no_grad():
                    syn_image, syn_label = syn_data(image, label, Tgan, args)
            else:
                syn_image, syn_label = image, label

            if args.mixup:
                syn_image, syn_label = mixup([syn_image, syn_label])

            logits = model(syn_image)
            loss = loss_func(logits, syn_label)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        length = len(loader) // 4
        if args.rank == 0 and (idx + 1) % length == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "lr: {:.8f}".format(lr),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None

    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()

    cls_num = 3
    num = np.zeros(cls_num-1)

    all_dice = None

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(), target.cuda()

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            dice_list_sub = []
            val_outputs = logits.argmax(1)[0].data.cpu().numpy().astype(np.uint8)
            val_labels = target.data.cpu().numpy()[0, 0, :, :, :]
            from utils.utils import dice

            for i in range(1, cls_num):
                num[i - 1] += (np.sum(val_labels == i) > 0).astype(np.uint8)
                organ_Dice = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)

            if all_dice is None:
                all_dice = (np.asarray(dice_list_sub)).copy()
            else:
                all_dice = all_dice + np.asarray(dice_list_sub)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            # if args.rank == 0:
            #     avg_acc = np.mean(run_acc.avg)
            #
            #     print(
            #         "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            #         "acc",
            #         (all_dice / num)[-1],
            #         "time {:.2f}s".format(time.time() - start_time),
            #     )
            start_time = time.time()
    torch.cuda.empty_cache()

    tumor_acc = all_dice / num
    tumor_acc = tumor_acc[-1]
    return tumor_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        args,
        model_inferer=None,
        scheduler=None,
        start_epoch=0,
        post_label=None,
        post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0

    Tgan = TGAN(args).cuda()
    model_dict = torch.load(args.TGAN_checkpoint, map_location=torch.device('cpu'))
    state_dict = model_dict["state_dict"]
    Tgan.load_state_dict(state_dict, strict=True)
    Tgan.cuda(args.gpu)
    Tgan.eval()
    print('Load trained TumorGAN')

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, Tgan=Tgan,
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "Final tumor acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
