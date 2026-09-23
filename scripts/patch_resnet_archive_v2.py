#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"Could not locate patch anchor: {label}")
    return text.replace(old, new, 1)


def patch_preserve():
    src = SCRIPTS / "eval_resnet18_localisation_attack_preserve_cls_full.py"
    dst = SCRIPTS / "eval_resnet18_localisation_attack_preserve_cls_full_archive_v2.py"
    text = src.read_text()

    text = replace_once(
        text,
        "import eval_resnet18_gradcam_clean_rma as clean_rma\n",
        "import eval_resnet18_gradcam_clean_rma as clean_rma\n"
        "from resnet_exact_archive import save_exact_bundle\n",
        "preserve import",
    )

    text = replace_once(
        text,
        '/ "resnet18_localisation_attack_preserve_cls_full"\n',
        '/ "resnet18_localisation_attack_preserve_cls_full_archive_v2"\n',
        "preserve output root",
    )

    text = replace_once(
        text,
        'OUT_CAMS = OUT_ROOT / "localisation_attack_full_layer4_maps.npz"\n',
        'OUT_CAMS = OUT_ROOT / "localisation_attack_full_layer4_maps.npz"\n'
        'EXACT_ARCHIVE_ROOT = OUT_ROOT / "exact_archive"\n',
        "preserve archive root",
    )

    anchor = '''                saved_probability = float(
                    row[
                        "saved_attack_probability"
                    ]
                )

                record = {
'''
    inserted = '''                saved_probability = float(
                    row[
                        "saved_attack_probability"
                    ]
                )

                exact_archive = save_exact_bundle(
                    repo_root=ROOT,
                    archive_root=EXACT_ARCHIVE_ROOT,
                    row=row,
                    info=batch_infos[j],
                    clean_input=batch_x[j],
                    adv_input=adv_x[j],
                    clean_cam=clean_eval["full_cam"][j],
                    adv_cam=adv_eval["full_cam"][j],
                    clean_probability=clean_p,
                    adv_probability=adv_p,
                    clean_margin=clean_margin[j].item(),
                    adv_margin=adv_margin[j].item(),
                    attack_metadata={
                        "attack_type": "classification_preserving_gradcam_E",
                        "epsilon_pixel": EPSILON_PIXEL,
                        "epsilon_255": EPSILON_PIXEL * 255.0,
                        "alpha_pixel": ALPHA_PIXEL,
                        "alpha_255": ALPHA_PIXEL * 255.0,
                        "steps": ATTACK_STEPS,
                        "bisection_steps": BISECTION_STEPS,
                        "classification_margin_floor": CLASSIFICATION_MARGIN_FLOOR,
                        "objective_tol": OBJECTIVE_TOL,
                        "initialisation": "clean_image",
                        "random_restart": False,
                        "attack_support": "document_content_only",
                        "padding_frozen": True,
                    },
                )

                record = {
'''
    text = replace_once(
        text,
        anchor,
        inserted,
        "preserve bundle call",
    )

    anchor = '''                    "cam_change_defined":
                        bool(
                            (not clean_zero)
                            and
                            (not adv_zero)
                        ),
                }
'''
    inserted = '''                    "cam_change_defined":
                        bool(
                            (not clean_zero)
                            and
                            (not adv_zero)
                        ),

                    "exact_bundle_path": exact_archive["exact_bundle_path"],
                    "exact_bundle_sha256": exact_archive["exact_bundle_sha256"],
                    "exact_bundle_bytes": exact_archive["exact_bundle_bytes"],
                    "exact_input_dtype": exact_archive["exact_input_dtype"],
                    "exact_cam_dtype": exact_archive["exact_cam_dtype"],
                    "exact_cam_height": exact_archive["exact_cam_height"],
                    "exact_cam_width": exact_archive["exact_cam_width"],
                    "exact_archive_schema": exact_archive["exact_archive_schema"],
                }
'''
    text = replace_once(
        text,
        anchor,
        inserted,
        "preserve CSV archive fields",
    )

    compile(text, str(dst), "exec")
    dst.write_text(text)
    return dst


def patch_pgd():
    src = SCRIPTS / "eval_resnet18_adversarial_localisation_pgd_full.py"
    dst = SCRIPTS / "eval_resnet18_adversarial_localisation_pgd_full_archive_v2.py"
    text = src.read_text()

    text = replace_once(
        text,
        "import eval_resnet18_gradcam_clean_rma as clean_rma\n",
        "import eval_resnet18_gradcam_clean_rma as clean_rma\n"
        "from resnet_exact_archive import save_exact_bundle\n",
        "PGD import",
    )

    text = replace_once(
        text,
        '/ "resnet18_adversarial_localisation_pgd_eps1_full"\n',
        '/ "resnet18_adversarial_localisation_pgd_eps1_full_archive_v2"\n',
        "PGD output root",
    )

    text = replace_once(
        text,
        'OUT_CAMS = OUT_ROOT / "pgd_eps1_layer4_maps.npz"\n',
        'OUT_CAMS = OUT_ROOT / "pgd_eps1_layer4_maps.npz"\n'
        'EXACT_ARCHIVE_ROOT = OUT_ROOT / "exact_archive"\n',
        "PGD archive root",
    )

    anchor = '''                adv_zero = bool(
                    cam_change[
                        "adv_zero"
                    ][
                        j
                    ]
                    .item()
                )

                record = {
'''
    inserted = '''                adv_zero = bool(
                    cam_change[
                        "adv_zero"
                    ][
                        j
                    ]
                    .item()
                )

                exact_archive = save_exact_bundle(
                    repo_root=ROOT,
                    archive_root=EXACT_ARCHIVE_ROOT,
                    row=row,
                    info=batch_infos[j],
                    clean_input=batch_x[j],
                    adv_input=adv_x[j],
                    clean_cam=clean_result["full_cam"][j],
                    adv_cam=adv_result["full_cam"][j],
                    clean_probability=clean_p,
                    adv_probability=adv_p,
                    clean_margin=clean_margin[j].item(),
                    adv_margin=adv_margin[j].item(),
                    attack_metadata={
                        "attack_type": "targeted_bonafide_pgd",
                        "epsilon_pixel": EPSILON_PIXEL,
                        "epsilon_255": EPSILON_PIXEL * 255.0,
                        "alpha_pixel": ALPHA_PIXEL,
                        "alpha_255": ALPHA_PIXEL * 255.0,
                        "steps": PGD_STEPS,
                        "random_start": True,
                        "attack_target": "bonafide_class_0",
                        "attack_support": "document_content_only",
                        "padding_frozen": True,
                    },
                )

                record = {
'''
    text = replace_once(
        text,
        anchor,
        inserted,
        "PGD bundle call",
    )

    anchor = '''                    "adv_content_cam_mass":
                        cam_change[
                            "adv_mass"
                        ][
                            j
                        ]
                        .item(),
                }
'''
    inserted = '''                    "adv_content_cam_mass":
                        cam_change[
                            "adv_mass"
                        ][
                            j
                        ]
                        .item(),

                    "exact_bundle_path": exact_archive["exact_bundle_path"],
                    "exact_bundle_sha256": exact_archive["exact_bundle_sha256"],
                    "exact_bundle_bytes": exact_archive["exact_bundle_bytes"],
                    "exact_input_dtype": exact_archive["exact_input_dtype"],
                    "exact_cam_dtype": exact_archive["exact_cam_dtype"],
                    "exact_cam_height": exact_archive["exact_cam_height"],
                    "exact_cam_width": exact_archive["exact_cam_width"],
                    "exact_archive_schema": exact_archive["exact_archive_schema"],
                }
'''
    text = replace_once(
        text,
        anchor,
        inserted,
        "PGD CSV archive fields",
    )

    compile(text, str(dst), "exec")
    dst.write_text(text)
    return dst


def main():
    helper = SCRIPTS / "resnet_exact_archive.py"
    if not helper.exists():
        raise RuntimeError(
            "Missing scripts/resnet_exact_archive.py. "
            "Copy the supplied helper there first."
        )

    preserve = patch_preserve()
    pgd = patch_pgd()

    print("Created:")
    print(f"  {preserve.relative_to(ROOT)}")
    print(f"  {pgd.relative_to(ROOT)}")
    print()
    print("Original scripts were not modified.")


if __name__ == "__main__":
    main()
