import argparse
from pathlib import Path
from thop import OnnxProfile
from rich.console import Console

def parse():
    parser = argparse.ArgumentParser(description='Flops Counter.')
    parser.add_argument('--file_name', type=str, default='data/sample-onnx/alexnet.onnx')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    onnxp = OnnxProfile()
    onnxp.calculate_macs_agg(args.file_name)
    console = Console()
    console.print(onnxp.generate_table(table_name=str(Path(args.file_name).stem)))
    onnxp.overall_stat(str(Path(args.file_name).stem))
    # onnxp.export_to_csv(save_path=f'{str(Path(args.file_name).stem)}.xlsx')