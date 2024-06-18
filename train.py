import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py

from models.modeling import CrossEntropyWrapper
from models.bert import SimpleBERT

from checkfreq.cf_checkpoint import CFCheckpoint
from checkfreq.cf_iterator import CFIterator
from checkfreq.cf_manager import CFManager, CFMode


class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            (
                torch.from_numpy(input_[index].astype(np.int64))
                if indice < 5
                else torch.from_numpy(np.asarray(input_[index].astype(np.int64)))
            )
            for indice, input_ in enumerate(self.inputs)
        ]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero(as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        # print("nsp_label",next_sentence_labels)
        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]


DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# train_files = [
#     os.path.join(args.data_dir, f)
#     for f in os.listdir(args.data_dir)
#     if os.path.isfile(os.path.join(args.data_dir, f)) and "training" in f
# ]
train_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if os.path.isfile(os.path.join(DATA_DIR, f)) and "training" in f
]
data_file = train_files[0]
# train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
train_data = pretraining_dataset(data_file, 20)

BATCH_SIZE = 16

train_loader = DataLoader(
    train_data,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)
train_loader = iter(train_loader)

bert_model = SimpleBERT()
bert_model = bert_model.cuda()

optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-4)

criterion = CrossEntropyWrapper(30528)

TRAIN_ITERS = 1000
CKPT_PATH = "/home/user/yzy/pipedream-pipedream/runtime/image_classification/check"
CKPT_INTERVAL = 10
# ! Enable CheckFreq mechanism
ENABLE_CF = True

if __name__ == "__main__":

    # * ===== CheckFreq =====
    if ENABLE_CF:
        chk = CFCheckpoint(model=bert_model, optimizer=optimizer)
        cf_manager = CFManager(
            chk_dir=os.path.join(CKPT_PATH, "cf"),
            chk=chk,
            mode=CFMode.AUTO,
        )
        train_loader = CFIterator(
            train_loader,
            bs=BATCH_SIZE,
            dali=False,
            epoch=1,
            arch="bert",
            chk_freq=CKPT_INTERVAL,
            steps_to_run=TRAIN_ITERS,
            cf_manager=cf_manager,
        )
    # * ===== CheckFreq =====

    for step in range(TRAIN_ITERS):
        print(f"[Step: {step}] | ", end="")
        s = time.time()
        input_ = next(train_loader)
        input0, input1, input2, target, _ = input_
        input0, input1, input2, target = input0.cuda(), input1.cuda(), input2.cuda(), target.cuda()

        input2 = input2.unsqueeze(1).unsqueeze(2)
        input2 = input2.to(dtype=torch.float32)  # fp16 compatibility
        input2 = (1.0 - input2) * -10000.0

        optimizer.zero_grad()
        output = bert_model(input0, input1, input2)
        loss = criterion(output, target)
        print(f"Loss: {loss} | ", end="")
        loss.backward()
        if ENABLE_CF:
            cf_manager.weight_update()
        else:
            optimizer.step()
        if not ENABLE_CF:
            if step % CKPT_INTERVAL == 0:
                torch.save(bert_model.state_dict(), os.path.join(CKPT_PATH, f"bert_model.pth"))
        t = time.time()
        print(f"Iter Time: {(t - s) * 1000}ms")
        if not ENABLE_CF:
            if step % CKPT_INTERVAL == 0:
                print(f"Model of step {step} saved at {CKPT_PATH}")
