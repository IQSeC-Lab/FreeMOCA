import argparse

def _parse_args():
    parser = argparse.ArgumentParser(
        description="FreeMOCA Continual Learning (Class-IL & Domain-IL)"
    )

    # =====================================================
    # 1. EXPERIMENT MODE
    # =====================================================
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['class', 'domain'],
        required=True,
        help="Continual learning scenario"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['EMBER', 'AZ'],
        required=True,
        help="Dataset name"
    )

    parser.add_argument('--seed_', type=int, default=20)
    parser.add_argument('--use_cuda', type=bool, default=True)

    # =====================================================
    # 2. TRAINING HYPERPARAMETERS
    # =====================================================
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    # =====================================================
    # 3. CONTINUAL LEARNING STRUCTURE
    # =====================================================
    parser.add_argument(
        '--nb_task',
        type=int,
        required=True,
        help="Number of sequential tasks"
    )

    parser.add_argument(
        '--init_classes',
        type=int,
        required=True,
        help="Number of classes at task 0"
    )

    # ---- Class-IL only ----
    parser.add_argument(
        '--n_inc',
        type=int,
        default=0,
        help="Number of new classes per task (Class-IL only)"
    )

    parser.add_argument(
        '--final_classes',
        type=int,
        default=None,
        help="Total number of classes (Class-IL only)"
    )

    # =====================================================
    # 4. DATA PATHS
    # =====================================================
    # Used by:
    # - Class-IL (EMBER, AZ)
    # - EMBER-Domain (train/test split inside each domain)
    parser.add_argument(
        '--train_data',
        type=str,
        default=None,
        help="Path to training data (Class-IL or EMBER-Domain)"
    )

    parser.add_argument(
        '--test_data',
        type=str,
        default=None,
        help="Path to test data (Class-IL or EMBER-Domain)"
    )

    # =====================================================
    # 5. DOMAIN-IL SPECIFIC
    # =====================================================
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help="Root directory containing domain subfolders (Domain-IL only)"
    )

    parser.add_argument(
        '--domains',
        nargs='+',
        default=None,
        help="Ordered list of domains (Domain-IL only)"
    )

    # =====================================================
    # 6. FREEMOCA INTERPOLATION
    # =====================================================
    parser.add_argument('--lambda_min', type=float, default=0.3)
    parser.add_argument('--lambda_max', type=float, default=0.7)

    args = parser.parse_args()

    # =====================================================
    # 7. DATASET-SPECIFIC DEFAULTS (IMPORTANT)
    # =====================================================
    if args.scenario == 'domain' and args.domains is None:
        if args.dataset == 'EMBER':
            args.domains = [
                '2018-01', '2018-02', '2018-03', '2018-04',
                '2018-05', '2018-06', '2018-07', '2018-08',
                '2018-09', '2018-10', '2018-11', '2018-12'
            ]
        elif args.dataset == 'AZ':
            args.domains = [
                '2008', '2010', '2011', '2012',
                '2013', '2014', '2015', '2016'
            ]

    # =====================================================
    # 8. SAFETY CHECKS (FAIL FAST)
    # =====================================================
    # =====================================================
# 8. SAFETY CHECKS (FAIL FAST)
# =====================================================
    if args.scenario == 'class':
        assert args.n_inc > 0, "Class-IL requires --n_inc > 0"
        assert args.final_classes is not None, "Class-IL requires --final_classes"
        assert args.train_data is not None, "Class-IL requires --train_data"
        assert args.test_data is not None, "Class-IL requires --test_data"

    if args.scenario == 'domain':
        assert args.data_root is not None, "Domain-IL requires --data_root"
        assert args.n_inc == 0, "Domain-IL must have n_inc = 0"
    

    return args
