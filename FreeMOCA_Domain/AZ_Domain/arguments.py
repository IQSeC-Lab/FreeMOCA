import argparse

def _parse_args():
    parser = argparse.ArgumentParser(description='Domain-IL (Year Split)')
    
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
                        default=['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016'],
                        help='List of folder names to be used as sequential tasks')

    return parser.parse_args()