import os
import logging
import argparse
from pathlib import Path
from settings import API_KEY
from rich.console import Console
from artifactory import ArtifactoryPath
from thop import OnnxProfile, FlopsOpt


URL = 'https://boartifactory.micron.com:443/artifactory/ndcg-generic-dev-local/MDLA-ModelZoo/models-simplified'
# create logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)


def parse():
    parser = argparse.ArgumentParser(description='Flops Counter.')
    parser.add_argument('--summary', default=False, action='store_true', help='Creates a summary Table.')
    parser.add_argument('--simplify', default=False, action='store_true', help='Simplifies the model.')
    parser.add_argument('--decimal_roundoff', default=2, type=int, help='The decimal value upto which you want to round off.')
    parser.add_argument('--flops', type=lambda flops_i: FlopsOpt[flops_i], choices=list(FlopsOpt), default=FlopsOpt.MEGA, help='Displays Flops in appropriate scale.')
    args = parser.parse_args()

    return args


def main(args):
    artifactory_path = ArtifactoryPath(URL, apikey=API_KEY)
    
    o_onnxPro = OnnxProfile(flops_opt=args.flops)
    console = Console()
    with console.status("[bold green]Fetching and Processing data...") as status:
        for path in artifactory_path.glob("**/*.onnx"):
            try:
                o_onnxPro.console.rule(f"Report Generated for {Path(path).stem}")
                o_onnxPro.calculate_macs_agg(path, simplify=args.simplify)
                o_onnxPro.console.print(o_onnxPro.generate_table(table_name=str(Path(path).stem)))
                o_onnxPro.overall_stat(str(Path(path).stem))
            except Exception as e:
                logger.error("Exception Occurs! : %s", e, exc_info=True)
            console.log(f"[green]Finish fetching and processing data[/green] {Path(path).stem}")

        console.log(f'[bold][red]Done!')

    if args.summary:
        o_onnxPro.export_to_csv(round_decimal=args.decimal_roundoff, save_path='regression_flops_output_cloud.xlsx')
        o_onnxPro.console.save_text("regression_output_cloud.txt")
        


if __name__ == '__main__':
    args = parse()
    main(args)