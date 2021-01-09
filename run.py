"""
Helper for running scripts.
Usage: python run.py <>
"""
import argparse, sys, os, time, glob


def get_train_command(model, loss, style, expr_dir, init_model, pad_type, find_init=False):
    script_name = "baseline.py"
    style_name = style[style.rfind("/") + 1 : -4]
    model_name = f"{model}_{loss}_{style_name}_{pad_type}"
    model_dir = f"{expr_dir}/{model_name}"
    if find_init:
        init_dir = f"{model}_none_{style_name}_interpolate-detach"
        init_model = glob.glob(f"{expr_dir}/{init_dir}/*.model")
        if len(init_model) != 1:
            print(f"!> Fail to find initialization needed for {model_name} from {model_dir}")
            exit(0)
        init_model.sort()
        init_model = init_model[0]
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


def get_eval_command(model, loss, style, expr_dir, init_model, pad_type, find_init=False):
    evalcmd = "python -u fdb.py eval --input-dir %s --output-dir %s --model-dir %s --model-type %s --model-name %s --pad-type %s"
    style_name = style[style.rfind("/") + 1 : -4]
    model_name = f"{model}_{loss}_{style_name}_{pad_type}"
    model_dir = f"{expr_dir}/{model_name}"

def evalute(args):
    if args.style != "ALL":
        return [get_eval_command(            
            model=args.model,
            loss=args.temp_loss,
            style=args.style,
            expr_dir=args.expr_dir,
            init_model=args.model_path,
            pad_type=args.pad_type,
            find_init=False)]

    evalcmd = "CUDA_VISIBLE_DEVICES=%d python -u %s eval --input-dir %s --output-dir %s --model-dir %s --model-type %s --model-name %s --pad-type %s"

def train(args):
    if args.style != "ALL":
        return [get_train_command(            
            model=args.model,
            loss=args.temp_loss,
            style=args.style,
            expr_dir=args.expr_dir,
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
            expr_dir=args.expr_dir,
            init_model=args.model_path,
            pad_type=args.pad_type,
            find_init=args.temp_loss != "none"))
    gpus = args.gpus.split(",")
    slots = [[] for _ in range(len(gpus))]
    for i, c in enumerate(cmds):
        gpu = gpus[i % len(gpus)]
        slots[i % len(gpus)].append(f"CUDA_VISIBLE_DEVICES={gpu} {c}")
    for s in slots:
        print(" && ".join(s) + " &")
        os.system(" && ".join(s) + " &")


def main():
    main_parser = argparse.ArgumentParser(description="Parser for the script running helper. Models will be trained / evaluated using default settings.")
    subparsers = main_parser.add_subparsers(
        title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train",
        help="parser for training models")
    train_parser.add_argument("--gpus",
        default="0,1,2,3,4,5,6,7",
        help="All the available gpus.")
    train_parser.add_argument("--expr-dir",
        default="exprs",
        help="The experiment directory to store the results.")
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
        help="Train models using temporal loss with default settings.")
    train_parser.add_argument("--style",
        default="ALL",
        help="Path to the style image. If set to ``ALL'', all the four styles in data/styles are used.")
    train_parser.add_argument("--model-path",
        default="",
        help="For temp loss ``none'', this is left blank. Otherwise, this model is used for initialization. If the argument ``style'' is set to ``ALL'', this is ignored and models are automatically found in expr directory.")


    args = main_parser.parse_args()
    if args.subcommand == "train":
        train(args)
    elif args.subcommand == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()