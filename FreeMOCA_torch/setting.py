import torch


def configurate(args, config):
    """
    Populate config from args and derive scenario-dependent values.
    """

    # --------------------------------------------------
    # Copy all arguments into config
    # --------------------------------------------------
    for k, v in vars(args).items():
        config.set(k, v)

    # --------------------------------------------------
    # Derive number of tasks (CRITICAL)
    # --------------------------------------------------
    if args.scenario == "class":
        # Expected:
        # nb_task = ((final_classes - init_classes) / n_inc) + 1
        assert args.final_classes is not None
        assert args.n_inc > 0

        diff = args.final_classes - args.init_classes
        assert diff % args.n_inc == 0, (
            f"(final_classes - init_classes) must be divisible by n_inc "
            f"({args.final_classes} - {args.init_classes}) % {args.n_inc} != 0"
        )

        nb_task = diff // args.n_inc + 1

        # Sanity check with known experimental setup
        assert nb_task == 11, (
            f"Expected 11 tasks for Class-IL, got {nb_task}"
        )

    elif args.scenario == "domain":
        # Domain-IL: number of tasks = number of domains
        assert args.domains is not None
        nb_task = len(args.domains)

        # Dataset-specific sanity checks
        if args.dataset == "EMBER":
            assert nb_task == 12, (
                f"EMBER-Domain must have 12 tasks, got {nb_task}"
            )
        elif args.dataset == "AZ":
            assert nb_task == 8, (
                f"AZ-Domain must have 8 tasks, got {nb_task}"
            )

    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    config.nb_task = nb_task

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    if args.use_cuda and torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    # --------------------------------------------------
    # Misc (kept for backward compatibility)
    # --------------------------------------------------
    config.ls_a = []
  


def torch_setting(config):
    """
    Torch environment setup and diagnostics.
    """
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA not available, using CPU")

    torch.cuda.empty_cache()
