# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

import os

import torch
from executorch.exir import to_edge, EdgeCompileConfig

#from executorch.extension.training.examples.XOR.model_cifar10 import Net, TrainingNet
from model_cifar10 import ConvNet, Net, TrainingNet

from torch.export import export
from torch.export.experimental import _export_forward_backward

BATCH_SIZE = 32

def _export_model():
    net = TrainingNet(ConvNet())
    x = torch.randn(BATCH_SIZE, 3, 32, 32)

    # Captures the forward graph. The graph will look similar to the model definition now.
    # Will move to export_for_training soon which is the api planned to be supported in the long term.
    ep = export(net, (x, torch.ones(BATCH_SIZE, dtype=torch.int64)), strict=True)
    print("FORWARD")
    print(ep.graph_module.graph)

    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    print("BACKWARD")
    print(ep.graph_module.graph)

    # Lower the graph to edge dialect.    
    print("TO_EDGE")
    #edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
    #ep = to_edge(ep, compile_config=edge_compile_config)  

    ep = to_edge(ep)  

    # Lower the graph to executorch.
    print("TO_EXECUTORCH")
    ep = ep.to_executorch()

    print("EXPORTED MODEL EXPECTED INPUT SIZE", x.shape)
    return ep


def main() -> None:
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(
        prog="export_model",
        description="Exports an nn.Module model to ExecuTorch .pte files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Path to the directory to write xor.pte files to",
    )
    args = parser.parse_args()

    ep = _export_model()

    # Write out the .pte file.
    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, "cifar10.pte")
    with open(outfile, "wb") as fp:
        fp.write(
            ep.buffer,
        )
    print(f"WROTE TO {outfile}")


if __name__ == "__main__":
    main()
