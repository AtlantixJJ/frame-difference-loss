"""
Helper for reproducing experiments in the paper.
Usage: python run.py <train/eval> <other arguments>
"""
import argparse, sys, os, time, glob


def find_model_from_dir(model, style_name):
    init_dir = f"{model}_none_{style_name}_interpolate-detach"
    init_model = glob.glob(f"exprs/{init_dir}/*.model")
    if len(init_model) != 1:
        print(f"!> Fail to find initialization needed for {model} {style_name} from exprs/{init_dir}")
        exit(0)
    init_model.sort()
    return init_model[0]


def run_commands(gpus, cmds):
    gpus = gpus.split(",")
    slots = [[] for _ in range(len(gpus))]
    for i, c in enumerate(cmds):
        gpu = gpus[i % len(gpus)]
        slots[i % len(gpus)].append(f"CUDA_VISIBLE_DEVICES={gpu} {c}")
    for s in slots:
        if len(s) == 0:
            continue
        print(" && ".join(s) + " &")
        os.system(" && ".join(s) + " &")


def get_train_command(model, loss, style, init_model, pad_type, find_init=False):
    script_name = "baseline.py"
    style_name = style[style.rfind("/") + 1 : -4]
    model_name = f"{model}_{loss}_{style_name}_{pad_type}"
    model_dir = f"exprs/{model_name}"
    if find_init:
        init_model = find_model_from_dir(model, style_name)
    if loss == "ofb":
        script_name = "ofb.py"
    elif loss in ["p-fdb", "c-fdb"]:
        script_name = "fdb.py"
    os.makedirs(model_dir, exist_ok=True)
    s = f"python -u {script_name} train --pad-type {pad_type} --style-image {style} --save-model-dir {model_dir}"
    if loss != "none":
        s = s + f" --model-type {model} --init-model {init_model}"
    if loss == "p-fdb":
        s = s + f" --time-strength1 400.0 --time-strength2 0.0"
    s = s + f" > {model_dir}/train.log"
    return s


def get_eval_command(model, loss, style, init_model, pad_type, find_init=True):
    style_name = style[style.rfind("/") + 1 : -4]
    model_name = f"{model}_{loss}_{style_name}_{pad_type}"
    model_dir = f"exprs/{model_name}"
    if find_init:
        init_model = find_model_from_dir(model, style_name)
    return f"python -u baseline.py eval --model-path {init_model} --model-type {model} --model-name {model_name} --pad-type {pad_type}"


def evaluate(args):
    if args.style != "ALL":
        return [get_eval_command(            
            model=args.model,
            loss=args.temp_loss,
            style=args.style,
            init_model=args.model_path,
            pad_type=args.pad_type,
            find_init=False)]
    cmds = []
    styles = glob.glob("data/styles/*")
    styles.sort()
    for style in styles:
        cmds.append(get_eval_command(
            model=args.model,
            loss=args.temp_loss,
            style=style,
            init_model=args.model_path,
            pad_type=args.pad_type,
            find_init=args.temp_loss != "none"))
    run_commands(args.gpus, cmds)


def train(args):
    if args.style != "ALL":
        return [get_train_command(            
            model=args.model,
            loss=args.temp_loss,
            style=args.style,
            init_model=args.model_path,
            pad_type=args.pad_type,
            find_init=False)]
    cmds = []
    styles = glob.glob("data/styles/*")
    styles.sort()
    for style in styles:
        cmds.append(get_train_command(
            model=args.model,
            loss=args.temp_loss,
            style=style,
            init_model=args.model_path,
            pad_type=args.pad_type,
            find_init=args.temp_loss != "none"))
    run_commands(args.gpus, cmds)


def main():
    main_parser = argparse.ArgumentParser(description="Parser for the script running helper. Models will be trained / evaluated using default settings.")
    subparsers = main_parser.add_subparsers(
        title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train",
        help="parser for training models")
    train_parser.add_argument("--gpus",
        default="0,1,2,3,4,5,6,7",
        help="All the available gpus.")
    train_parser.add_argument("--model",
        default="sfn",
        choices=["rnn", "sfn"],
        help="The architecture type.")
    train_parser.add_argument("--temp-loss",
        default="none",
        choices=["none", "c-fdb", "p-fdb", "ofb"],
        help="Train models using temporal loss with default settings.")
    train_parser.add_argument("--pad-type",
        default="interpolate-detach",
        choices=["interpolate-detach", "reflect-start", "none", "reflect", "replicate", "zero"],
        help="Train models using default paddings.")
    train_parser.add_argument("--style",
        default="ALL",
        help="Path to the style image. If set to ``ALL'', all the four styles in data/styles are used.")
    train_parser.add_argument("--model-path",
        default="",
        help="If the argument ``style'' is set to ``ALL'', this is ignored and models are automatically found in expr directory. Otherwise, it specifies the model path.")

    eval_parser = subparsers.add_parser("eval",
        help="parser for evaluating models")
    eval_parser.add_argument("--gpus",
        default="0,1,2,3,4,5,6,7",
        help="All the available gpus.")
    eval_parser.add_argument("--model",
        default="sfn",
        choices=["rnn", "sfn"],
        help="The architecture type.")
    eval_parser.add_argument("--temp-loss",
        default="none",
        choices=["none", "c-fdb", "p-fdb", "ofb"],
        help="Train models using temporal loss with default settings.")
    eval_parser.add_argument("--pad-type",
        default="interpolate-detach",
        choices=["interpolate-detach", "reflect-start", "none", "reflect", "replicate", "zero"],
        help="Train models using default paddings.")
    eval_parser.add_argument("--style",
        default="ALL",
        help="Path to the style image. If set to ``ALL'', all the four styles in data/styles are used.")
    eval_parser.add_argument("--model-path",
        default="",
        help="If the argument ``style'' is set to ``ALL'', this is ignored and models are automatically found in expr directory. Otherwise, it specifies the model path.")

    args = main_parser.parse_args()
    if args.subcommand == "train":
        train(args)
    elif args.subcommand == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()