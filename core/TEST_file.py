import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
import multiprocessing
import threading, queue

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):

    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    #queue1 = queue.Queue()
    #queue2 = queue.Queue()

    def upsample_flow(self, flow, mask, queue2):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        #return up_flow.reshape(N, 2, 8*H, 8*W).detach()
        queue2.put((up_flow.reshape(N, 2, 8*H, 8*W).detach()))

    def update_block_process(self, net, inp, corr, flow, queue1):
        net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
        #queue1 = queue.Queue()
        queue1.put((net, up_mask, delta_flow))


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        queue1 = queue.Queue()
        queue2 = queue.Queue()

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim


        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)


        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        net = net.requires_grad_(True)  # set requires_grad to True
        inp = inp.requires_grad_(True)  # set requires_grad to True


        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []

        coords1 = coords1.detach()
        corr = corr_fn(coords1)  # index correlation volume

        flow = coords1 - coords0

        with autocast(enabled=self.args.mixed_precision):
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

        coords1 = coords1 + delta_flow

        for itr in range(iters - 1):


            queue1 = queue.Queue()

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = threading.Thread(target=self.upsample_flow,
                                                  args=(coords1 - coords0, up_mask, queue2))
                flow_up.start()
                flow_up.join()


            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0


            with autocast(enabled=self.args.mixed_precision):
                p2 = threading.Thread(target=self.update_block_process, args=(net, inp, corr, flow, queue1))
                p2.start()
                p2.join()



            net, up_mask, delta_flow = queue1.get()
            flow_up = queue2.get()

            delta_flow = delta_flow.requires_grad_(True)

            coords1 = coords1.detach() + delta_flow
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions







