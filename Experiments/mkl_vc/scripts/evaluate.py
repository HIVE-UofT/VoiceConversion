"""Evaluate MKL-VC conversion quality using shared metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_evaluate import make_parser, run_evaluation

if __name__ == '__main__':
    parser = make_parser(
        method_name='MKL-VC',
        default_converted_dir=os.path.join(os.path.dirname(__file__), '..', 'converted'))
    args = parser.parse_args()
    run_evaluation(args.converted_dir, args.pre_dir, args.post_dir,
                   args.method_name, args.skip_f0)
