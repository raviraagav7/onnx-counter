import logging
import argparse
from pathlib import Path
from thop import OnnxProfile, FlopsOpt
from tqdm import tqdm
from rich.live import Live

# create logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)
import yaml

def load_model_list():
    with open('model_list.yaml') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            logger.error("yaml parser: %s", exception)
        
        return data


def parse():
    parser = argparse.ArgumentParser(description='Flops Counter.')
    parser.add_argument('base_path', type=str, help='Directory path to where the models exist.')
    parser.add_argument('--all', default=False, action='store_true', help='Run Profiling on all Models.')
    parser.add_argument('--summary', default=False, action='store_true', help='Creates a summary Table.')
    parser.add_argument('--simplify', default=False, action='store_true', help='Simplifies the model.')
    parser.add_argument('--decimal_roundoff', default=2, type=int, help='The decimal value upto which you want to round off.')
    parser.add_argument('--flops', type=lambda flops_i: FlopsOpt[flops_i], choices=list(FlopsOpt), default=FlopsOpt.MEGA, help='Displays Flops in appropriate scale.')
    parser.add_argument('--live', default=False, action='store_true', help='Display the table live.')

    args = parser.parse_args()

    return args


def main(args):
    yaml_data = load_model_list()
    data = yaml_data.get('models', None)
    if data:
        if args.all:
            model_list = data.get('all', [])
        else:
            model_list = data.get('test', [])
    else:
        logger.error('Yaml file does not contain any models.')

    base_path = Path(args.base_path)
    
    o_onnxPro = OnnxProfile(flops_opt=args.flops)
    
    if args.live:
        with Live(o_onnxPro.generate_table(table_name=''), refresh_per_second=4) as live:
            for model_name in tqdm(model_list, desc='Models'):
                o_onnxPro.calculate_macs_agg(f"{base_path/model_name}.onnx", simplify=args.simplify)
                live.update(o_onnxPro.generate_table(table_name=model_name))
                o_onnxPro.overall_stat(model_name)
    else:
        for model_name in tqdm(model_list, desc='Models'):
                o_onnxPro.console.rule(f"Report Generated for {model_name}")
                o_onnxPro.calculate_macs_agg(f"{base_path/model_name}.onnx", simplify=args.simplify)
                o_onnxPro.console.print(o_onnxPro.generate_table(table_name=model_name))
                o_onnxPro.overall_stat(model_name)

    if args.summary:
        o_onnxPro.export_to_csv(round_decimal=args.decimal_roundoff)
        o_onnxPro.console.save_text("regression_output_local.txt")


if __name__ == '__main__':
    args = parse()
    main(args)