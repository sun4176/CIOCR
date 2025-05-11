import torch
import torch.utils.data as data
import libs.utils.logger as logger
import os
import cv2






def test_adaptive_memory(testloader, model, use_cuda, model_name, opt):
    data_time = AverageMeter()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames, masks, objs, infos = data

            if use_cuda:
                frames = frames.squeeze(0).to(device)
                masks = masks.squeeze(0).to(device)

            load_number = int(frames.size(0) / opt.sampled_frames) if (frames.size(
                0) % opt.sampled_frames) == 0 else int((frames.size(0) / opt.sampled_frames) + 1)
            num_objects, info, max_obj = objs[0], infos[0], (masks.shape[1] - 1)


            T, _, H, W = frames.shape
            pred, keys, vals, scales, keys_dict, vals_dict = [], [], [], [], {}, {}

            for C in range(1, load_number):
                Clip_first_idx, Clip_last_idx = (C - 1) * opt.sampled_frames, (C) * opt.sampled_frames
                Clip_Frame, Clip_Mask = frames[Clip_first_idx:Clip_last_idx, :, :, :], masks[
                                                                                       Clip_first_idx:Clip_last_idx, :,
                                                                                       :, :]
                clip_key, clip_val, r4 = model(frame=Clip_Frame, mask=Clip_Mask, num_objects=num_objects)
                keys.append(clip_key)
                vals.append(clip_val)
                keys_dict[C] = clip_key
                vals_dict[C] = clip_val

                tmp_key, tmp_val = torch.cat(keys, dim=0), torch.cat(vals, dim=0)
                logit_list, _ = model(frame=Clip_Frame,
                                                                                               keys=tmp_key,
                                                                                               values=tmp_val,
                                                                                               num_objects=num_objects,
                                                                                               max_obj=max_obj,
                                                                                               opt=opt,
                                                                                               Clip_idx=C,
                                                                                               keys_dict=keys_dict,
                                                                                               vals_dict=vals_dict,
                                                                                               patch=2)
                for l in range(len(logit_list)):
                    logit = logit_list[l]
                    out = torch.softmax(logit, dim=1)
                    pred.append(out)

            if (frames.size(0) - Clip_last_idx) > 0:
                Clip_Frame, Clip_Mask = frames[-int(opt.sampled_frames):, :, :, :], masks[-int(opt.sampled_frames):, :, :, :]
                clip_key, clip_val, r4 = model(frame=Clip_Frame, mask=Clip_Mask, num_objects=num_objects)
                keys.append(clip_key)
                vals.append(clip_val)
                keys_dict[C + 1] = clip_key
                vals_dict[C + 1] = clip_val

                tmp_key, tmp_val = torch.cat(keys, dim=0), torch.cat(vals, dim=0)
                logit_list, _ = model(frame=Clip_Frame,
                                                                                               keys=tmp_key,
                                                                                               values=tmp_val,
                                                                                               num_objects=num_objects,
                                                                                               max_obj=max_obj,
                                                                                               opt=opt,
                                                                                               Clip_idx=C + 1,
                                                                                               keys_dict=keys_dict,
                                                                                               vals_dict=vals_dict,
                                                                                               patch=2)

                if frames.size(0) % opt.sampled_frames != 0:
                    logit_list = logit_list[-1 * int(frames.size(0) % opt.sampled_frames):]

                for l in range(len(logit_list)):
                    logit = logit_list[l]
                    out = torch.softmax(logit, dim=1)
                    pred.append(out)

            pred = torch.cat(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            assert num_objects == 1
            write_mask(pred, info, opt, directory=opt.output_dir, model_name='{}'.format(model_name))
    return data_time.sum



