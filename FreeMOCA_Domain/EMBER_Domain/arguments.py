import argparse

def _parse_args():
    parser = argparse.ArgumentParser(description='Domain-IL (Month Split)')
    
    # --- Training Hyperparameters ---
    parser.add_argument('-e', '--epochs', default=50, type=int, help="Epochs per month")
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    
    # --- Model Structure ---
    # Domain-IL: Fixed output size (e.g., 100 malware families)
    parser.add_argument('--init_classes', type=int, default=100) 
    
    # These are kept for compatibility but ignored in Domain-IL logic
    parser.add_argument('--final_classes', type=int, default=100)
    parser.add_argument('--n_inc', type=int, default=0)

    parser.add_argument('--seed_', default=20, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)

    parser.add_argument('-a', '--alpha', default=None, type=float)
    parser.add_argument('-m', '--method', default='spline', type=str, choices=['linear', 'spline', 'polynomial'])

    # --- CRITICAL: DATA PATHS ---
    
    # 1. Path to the PARENT folder containing '2018-01', '2018-02', etc.
    # Update this to your actual path
    parser.add_argument('--data_root', default="/path/to/data", 
                        help='Root path containing the month folders')

    # 2. Define the SEQUENCE of months (Tasks)
    # You can add more: '2018-03', '2018-04', etc.
    parser.add_argument('--domains', nargs='+', 
                        default=['2018-01', '2018-02', '2018-03','2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12' ],
                        help='List of folder names to be used as sequential tasks')

    return parser.parse_args()

    # CUDA_VISIBLE_DEVICES=7 python main.py -a 0.1 -m spline