"""The command demo: a first run that audits the real sample map shipped with the package, or a made-up one."""


def cmd_demo(args):
    from oe_inferencex.demo import run
    return run(out=args.out, seed=args.seed, made_up=args.made_up)


def add_demo_parser(sub):
    d = sub.add_parser("demo", help="first run: audit the real sample map shipped with the package (or a made-up one) and draw the result")
    d.add_argument("--out", default="oe_inferencex_demo", help="output directory (default oe_inferencex_demo)")
    d.add_argument("--seed", type=int, default=0, help="seed of the made-up scene and of the random pick")
    d.add_argument("--made-up", action="store_true", help="audit a small synthetic water map instead of the real sample tile")
    d.set_defaults(func=cmd_demo)
