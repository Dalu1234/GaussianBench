from gaussian_cli import parse_cli


def test_parse_cli_defaults_do_not_import_simulator():
    args = parse_cli([])

    assert args.solver == "ul_mlsmpm"
    assert args.scene == "single"
    assert args.shape is None
    assert args.custom_obj is None
    assert args.n_per_entity == 203930


def test_parse_cli_custom_obj_and_known_args():
    args = parse_cli([
        "--shape", "custom",
        "--custom-obj", "assets/example.obj",
        "--n-per-entity", "120",
        "--unknown-from-caller", "ok",
    ])

    assert args.shape == "custom"
    assert args.custom_obj == "assets/example.obj"
    assert args.n_per_entity == 120