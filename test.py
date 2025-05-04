


import torch


def test(test_loader, model, device, args):
    result = {}
    for i, data in enumerate(test_loader):
        feature, data_video_name = data
        feature = feature.to(device)
        seq_len = torch.sum(torch.max(torch.abs(feature), dim=2)[0] > 0, 1)

        with torch.no_grad():

            _, element_logits, v_features = model(feature, seq_len, is_training=False)


            element_logits = torch.mean(element_logits, dim=0)
        element_logits = element_logits.cpu().data.numpy().reshape(-1)

        result[data_video_name[0]] = element_logits
    return result



