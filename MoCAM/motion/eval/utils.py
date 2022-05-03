import torch
import torch.nn.functional as F
import torch

def get_correct_num_labels_topk(model, motions, texts, full_label, valid_length, topk=[1,5], gpu=None, verbose=True):

    topk_corrects = {k : 0 for k in topk}
    max_k = max(topk)
    softk_corrects = {max_k: 0}

    for motion, text, val_len in zip(motions, texts, valid_length):
        if text in full_label:
            correct_label = full_label.index(text)

        encoded_text = model.text_tokenizer(full_label, return_tensors='pt', padding=True)
        for k in list(encoded_text.keys()):
            if gpu is not None:
                encoded_text[k] = encoded_text[k].cuda(gpu)
            else:
                encoded_text[k] = encoded_text[k].cpu()
        text_feat_dict = model.text_encoder(**encoded_text)

        if gpu is not None:
            motion_in = motion[None, :, :].cuda(gpu)
        else:
            motion_in = motion[None, :, :].cpu()
        
        motion_feat_dict = model.motion_encoder(motion_in,  val_len[None])

        # pooling
        text_feat   = text_feat_dict['pooler_output']
        motion_feat = motion_feat_dict['pooler_output']

        # joint multimodal embedding
        text_feat = F.normalize(text_feat, dim=-1, p=2)
        motion_feat = F.normalize(motion_feat, dim=-1, p=2)

        # scaled pairwise cosine similarities
        logit_scale = model.logit_scale.exp()
        logits_per_motion = logit_scale * motion_feat @ text_feat.t()
        probs = logits_per_motion.softmax(dim=0)

        topk_indices = torch.topk(probs, k=max_k)[1].cpu().tolist()
        topk_probs = torch.topk(probs, k=max_k)[0].cpu().tolist()

        if verbose:
            print("Answer: ", text, f"/ Top {max_k} Prediction (sorted): ", [full_label[i] for i in topk_indices])

        for k in topk:
            if (text in full_label) and (correct_label in topk_indices[:k]):
                topk_corrects[k] += 1

        soft_answers = []
        for label_ind in topk_indices[:max_k]:
            soft_answers.append(full_label[label_ind]) # original answer
            soft_answers.append(full_label[label_ind].replace(" ", ""))  # ignore space
            soft_answers.extend(full_label[label_ind].split(" "))  # admit substring

        if set(text.split()).issubset(set(soft_answers)):
            softk_corrects[max_k] += 1
            
    return topk_corrects, softk_corrects, (topk_probs, topk_indices, full_label)


def calc_acc(epoch, log_df, loader, dataset, configs, model, full_proc_label_list, topk, gpu, verbose):
    correct_k = {k: 0 for k in topk}
    with torch.no_grad():
        for batch_in in loader:
            texts = batch_in['labels']
            if(configs['train']['motion_format'] == 'rot6d'):
                motions = batch_in['rotation_6d_pose_list']
                valid_length = batch_in['valid_length_list']
                B,T,J,C = motions.shape
                motions = motions.view(B,T,J*C)
            hardk_corrects, _, topk_info = get_correct_num_labels_topk(model, motions, texts, full_proc_label_list, valid_length, topk=topk, gpu=gpu, verbose=verbose)
            for k in topk:
                correct_k[k] += hardk_corrects[k]
    acc = {k: v/len(dataset) for k, v in correct_k.items()}
    print(f"Hard Accuracy: {acc} / Number of Testing Samples: {len(dataset)}")
    log_info = {"epoch": epoch}
    log_info.update({f"top_{k}_acc": acc[k] for k in topk})
    log_df = log_df.append(log_info, ignore_index=True)
    return log_df