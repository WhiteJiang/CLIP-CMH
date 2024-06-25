from src.cmdataset import CMDataset
import torch
import nets as models
import numpy as np
import scipy
import torch.nn as nn
import clip


def eval(data_loader, image_model, text_model, clip_model):
    imgs, txts, labs = [], [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            images, texts, idx, targets = data['image'], data['text'], data['index'], data['label']
            images, texts, idx, targets = images.cuda(), texts.cuda(), idx.cuda(), targets.cuda()
            with torch.no_grad():
                texts = [clip_model.encode_text(txt) for txt in texts]

            images_outputs = image_model(images)
            texts = torch.cat(texts)
            texts_outputs = text_model(texts.float())

            imgs.append(images_outputs)
            txts.append(texts_outputs)
            labs.append(targets)

        imgs = torch.cat(imgs).sign_().cpu().numpy()
        txts = torch.cat(txts).sign_().cpu().numpy()
        labs = torch.cat(labs).cpu().numpy()
    return imgs, txts, labs


def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = dist.shape[1]
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)[0: k]

        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)


if __name__ == '__main__':
    retrieval_dataset = CMDataset(
        path='ImageData/Flickr',
        partition='retrieval'
    )
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    test_dataset = CMDataset(
        path='ImageData/Flickr',
        partition='test'
    )
    query_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # load model
    backbone = models.__dict__['vgg11'](pretrained=True, feature=True).cuda()
    fea_net = models.__dict__['ImageNet'](y_dim=4096, bit=128, hiden_layer=3).cuda()
    image_model = nn.Sequential(backbone, fea_net).cuda()
    text_model = models.__dict__['TextNet'](y_dim=1024, bit=128,
                                            hiden_layer=2).cuda()
    # load parameters without knowledge distillation
    state_dict = torch.load('model.t7')
    image_model.load_state_dict(state_dict['image_model_state_dict'], strict=False)
    text_model.load_state_dict(state_dict['text_model_state_dict'])

    image_model = image_model.cuda().eval()
    text_model = text_model.cuda().eval()
    clip_model, process = clip.load('RN50')
    clip_model = clip_model.cuda().eval()

    # eval data set
    (retrieval_imgs, retrieval_txts, retrieval_labs) = eval(retrieval_loader, image_model, text_model, clip_model)
    query_imgs, query_txts, query_labs = retrieval_imgs[0: 100], retrieval_txts[0: 100], retrieval_labs[
                                                                                         0: 100]
    retrieval_imgs, retrieval_txts, retrieval_labs = retrieval_imgs[0: 100], retrieval_txts[
                                                                             0: 100], retrieval_labs[0: 100]
    i2t_without_knowledge = fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0,
                                                     metric='hamming')
    t2i_without_knowledge = fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0,
                                                     metric='hamming')

    avg_without_knowledge = (i2t_without_knowledge + t2i_without_knowledge) / 2.
    print('The MAP@100 results for without_knowledge: i2t: %.4f, t2i: %.4f, avg: %.4f' % (
    i2t_without_knowledge, t2i_without_knowledge, avg_without_knowledge))

    state_dict = torch.load('model_with_knowledge.t7')
    image_model.load_state_dict(state_dict['image_model_state_dict'], strict=False)
    text_model.load_state_dict(state_dict['text_model_state_dict'])

    image_model = image_model.cuda().eval()
    text_model = text_model.cuda().eval()

    # eval data set
    (retrieval_imgs, retrieval_txts, retrieval_labs) = eval(retrieval_loader, image_model, text_model, clip_model)
    query_imgs, query_txts, query_labs = retrieval_imgs[0: 100], retrieval_txts[0: 100], retrieval_labs[
                                                                                         0: 100]

    retrieval_imgs, retrieval_txts, retrieval_labs = retrieval_imgs[0: 100], retrieval_txts[
                                                                             0: 100], retrieval_labs[0: 100]
    i2t_knowledge = fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0,
                                             metric='hamming')
    t2i_knowledge = fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0,
                                             metric='hamming')

    avg_knowledge = (i2t_knowledge + t2i_knowledge) / 2.
    print('The MAP@100 results for with_knowledge: i2t: %.4f, t2i: %.4f, avg: %.4f' % (i2t_knowledge, t2i_knowledge,
                                                                                   avg_knowledge))
    print('The improvement of MAP@100 results: %.4f' % ((avg_knowledge - avg_without_knowledge) * 100), '%')
