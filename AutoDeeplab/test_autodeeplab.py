import sys
import torch
import argparse
from auto_deeplab import AutoDeeplab
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF

from dataloaders import make_data_loader
from utils.metrics import AverageMeter

def test(model, device, test_loader):
    acc = AverageMeter()
    iou = AverageMeter()
    latency = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warm_up for latency calculation
        rand_img = torch.rand(1, 3, 640, 640).to(device)
        for _ in range(10):
            _ = model(rand_img)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        for batch_idx, (data, target) in enumerate(test_loader):
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            starter.record()

            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1)

            output = model(data)

            target = target.long()
            tp, fp, fn, tn = smp.metrics.get_stats(output, target, 'binary', threshold=0.5)
            iou_value = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

            ori_image = data * 255.0
            output_image = output * 255.0

            ori_image[:, 0] = output_image[:, 0]

            # 두 배열을 RGB 이미지로 결합
            for n, (ori_img, out_img) in enumerate(zip(ori_image, output_image)):
                index = batch_idx * (len(ori_image))
                ori_img = TF.to_pil_image(ori_img.squeeze().byte(), mode='RGB')
                ori_img.save("overlap/output_image" + str(index + n) + ".jpg")

                out_img = TF.to_pil_image(out_img.squeeze().byte(), mode='L')
                out_img.save("results/output_image" + str(index + n) + ".jpg")

            acc.update(accuracy)
            iou.update(iou_value)

            ender.record()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            latency_time = starter.elapsed_time(ender) / data.size(0)    # μs ()
            torch.cuda.empty_cache()

            latency.update(latency_time)

    sys.stdout.write('\r')
    sys.stdout.write("Test | Accuracy: %.4f, IoU: %.4f, Latency: %.2f\n" % (acc.avg, iou.avg, latency.avg))
    sys.stdout.flush()

    return acc.avg, iou.avg, latency.avg

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--dataset', type=str, default='sealer',
                        choices=['pascal', 'coco', 'cityscapes', 'sealer'],
                        help='dataset name (default: sealer)')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N', 
                        help='input batch size for training (default: auto)')
    args = parser.parse_args()

    torch.manual_seed(0)
    device = 'cuda:0'
    device = torch.device(device)
    torch.cuda.set_device(device)

    criterion = smp.losses.DiceLoss('binary')

    model = AutoDeeplab(1, 6, criterion, crop_size=128)
    model = model.to(device)

    checkpoint = torch.load('run/sealer/deeplab-resnet/experiment_10/checkpoint.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    train_loader1, train_loader2, val_loader,  nclass = make_data_loader(args)

    test_acc, test_iou, latency = test(model, device, val_loader)
    print("FPS:{:.2f}".format(1000./latency))
    print("Latency:{:.2f}ms / {:.4f}s".format(latency, (latency/1000.)))

if __name__ == "__main__":
   main()