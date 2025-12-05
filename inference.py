import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import data_loader
import utils
from utils import decode
from model import Model

from transformers import AutoTokenizer

tokenizner = AutoTokenizer.from_pretrained("vinai/phobert-base")

vocab = tokenizner.get_vocab()
id2token = {v: k for k, v in vocab.items()}


def decode_instance(outputs, tokens, cfg):
    ti = outputs.get("ti", None)
    tc = outputs.get("tc", None)
    ai = outputs.get("ai", None)
    ac = outputs.get("ac", None)

    events = []

    if tc is not None:
        for row in tc:
            b, s, e, ev_id = row.tolist()
            if b != 0:
                continue
            span_tokens = None
            if tokens is not None and 0 <= s < len(tokens) and 0 <= e < len(tokens):
                span_tokens = tokens[s:e + 1]
            if hasattr(cfg, "tri_id2label"):
                ev_label = cfg.tri_id2label.get(int(ev_id), str(int(ev_id)))
            elif hasattr(cfg, "vocab") and hasattr(cfg.vocab, "tri_id2label"):
                ev_label = cfg.vocab.tri_id2label.get(int(ev_id), str(int(ev_id)))
            else:
                ev_label = str(int(ev_id))
            events.append(
                {
                    "start": int(s),
                    "end": int(e),
                    "event_type_id": int(ev_id),
                    "event_type": ev_label,
                    "trigger_tokens": span_tokens,
                    "arguments": []
                }
            )

    if ai is not None:
        for row in ai:
            b, s, e, ev_id = row.tolist()
            if b != 0:
                continue
            span_tokens = None
            if tokens is not None and 0 <= s < len(tokens) and 0 <= e < len(tokens):
                span_tokens = tokens[s:e + 1]
            for ev in events:
                if ev["event_type_id"] == int(ev_id):
                    ev.setdefault("arg_spans", []).append(
                        {
                            "start": int(s),
                            "end": int(e),
                            "arg_tokens": span_tokens
                        }
                    )

    if ac is not None:
        for row in ac:
            b, arg_pos, ev_id, role_id = row.tolist()
            if b != 0:
                continue
            role_label = None
            if hasattr(cfg, "rol_id2label"):
                role_label = cfg.rol_id2label.get(int(role_id), str(int(role_id)))
            elif hasattr(cfg, "vocab") and hasattr(cfg.vocab, "rol_id2label"):
                role_label = cfg.vocab.rol_id2label.get(int(role_id), str(int(role_id)))
            else:
                role_label = str(int(role_id))

            for ev in events:
                if ev["event_type_id"] != int(ev_id):
                    continue
                if "arg_spans" not in ev:
                    continue
                best_span = None
                for span in ev["arg_spans"]:
                    if span["start"] <= arg_pos <= span["end"]:
                        best_span = span
                        break
                if best_span is None:
                    continue
                ev["arguments"].append(
                    {
                        "start": best_span["start"],
                        "end": best_span["end"],
                        "role_id": int(role_id),
                        "role": role_label,
                        "tokens": best_span["arg_tokens"]
                    }
                )

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/bkee.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="model.pt")

    parser.add_argument("--bert_hid_size", type=int)
    parser.add_argument("--tri_hid_size", type=int)
    parser.add_argument("--eve_hid_size", type=int)
    parser.add_argument("--arg_hid_size", type=int)
    parser.add_argument("--node_type_size", type=int)
    parser.add_argument("--event_sample", type=int)
    parser.add_argument("--layers", type=int)

    parser.add_argument("--dropout", type=float)
    parser.add_argument("--graph_dropout", type=float)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warm_epochs", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--grad_clip_norm", type=float)
    parser.add_argument("--gamma", type=float)

    parser.add_argument("--bert_name", type=str)
    parser.add_argument("--bert_learning_rate", type=float)

    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    cfg = config.Config(args)
    logger = utils.get_logger(cfg.dataset + "_infer")
    logger.info(cfg)
    cfg.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    if cfg.seed >= 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    datasets = data_loader.load_data(cfg)
    train_dataset, dev_dataset, test_dataset = datasets

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        collate_fn=data_loader.collate_fn,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=1,
        collate_fn=data_loader.collate_fn,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        collate_fn=data_loader.collate_fn,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    logger.info("Building Model")
    model = Model(cfg)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    logger.info(f"Loading checkpoint from {args.ckpt}")
    state_dict = torch.load(
        args.ckpt,
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        # for loader in [dev_loader, test_loader, train_loader]:
        for loader in [test_loader]:
            i = 0
            logger.info(f"==== set ====")
            for batch_id, data_batch in enumerate(loader):
                data_batch_cuda = [d.to(device) for d in data_batch[:-2]] + [data_batch[-2], data_batch[-1]]

                inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, event_idx, tuple_labels, raw = data_batch_cuda

                outputs = model(
                    inputs,
                    att_mask,
                    word_mask1d,
                    word_mask2d,
                    triu_mask2d,
                    tri_labels,
                    arg_labels,
                    role_labels
                )
                
                logger.info(f'Size outputs: {len(outputs)}')
                logger.info(f'Type outputs: {type(outputs)}')
                
                
                _inputs = inputs.cpu().numpy()[0]
                _inputs = [id2token[id] for id in _inputs]
                
                logger.info(f'\nInputs: {_inputs}')
                logger.info(f'\nOutputs: {outputs}')
                logger.info(f'\nDecoded: {decode(outputs, tuple_labels, cfg.tri_args)}')
                sent_tokens = None
                if isinstance(raw, (list, tuple)) and len(raw) > 0:
                    sent_tokens = raw[0]
                events = decode_instance(outputs, sent_tokens, cfg)

                if sent_tokens is not None:
                    logger.info("Tokens: " + " ".join(sent_tokens))
                logger.info(f"Predicted {len(events)} events for instance {batch_id}")
                for idx, ev in enumerate(events):
                    logger.info(f'Event: {events}')
                    trig_text = " ".join(ev["trigger_tokens"]) if ev["trigger_tokens"] is not None else ""
                    logger.info(f"[Event {idx}] type_id={ev['event_type_id']} type={ev['event_type']} trigger_span=({ev['start']},{ev['end']}) trigger=\"{trig_text}\"")
                    for arg in ev["arguments"]:
                        arg_text = " ".join(arg['tokens']) if arg['tokens'] is not None else ""
                        logger.info(f"    Arg span=({arg['start']},{arg['end']}) role_id={arg['role_id']} role={arg['role']} text=\"{arg_text}\"")

                # i += 1
                # if i == 5:
                #     break


if __name__ == "__main__":
    main()
