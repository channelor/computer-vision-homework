import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="SAM prompt demo with point or box prompts.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint.")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs" / "sam_demo"))
    parser.add_argument("--point", nargs=2, type=int, action="append", help="Positive point prompt: x y")
    parser.add_argument("--negative-point", nargs=2, type=int, action="append", help="Negative point prompt: x y")
    parser.add_argument("--box", nargs=4, type=int, help="Box prompt: x1 y1 x2 y2")
    parser.add_argument("--multimask", action="store_true", help="Save multiple SAM candidate masks.")
    return parser.parse_args()


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_prompt_arrays(args):
    point_coords = []
    point_labels = []

    for point in args.point or []:
        point_coords.append(point)
        point_labels.append(1)
    for point in args.negative_point or []:
        point_coords.append(point)
        point_labels.append(0)

    if point_coords:
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
    else:
        point_coords = None
        point_labels = None

    box = np.array(args.box) if args.box else None
    if point_coords is None and box is None:
        raise ValueError("At least one --point, --negative-point, or --box prompt is required.")

    return point_coords, point_labels, box


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.55])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax):
    if coords is None:
        return
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points):
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color="lime", marker="*", s=160, edgecolor="white")
    if len(neg_points):
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=160, edgecolor="white")


def show_box(box, ax):
    if box is None:
        return
    x1, y1, x2, y2 = box
    ax.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            edgecolor="yellow",
            facecolor=(0, 0, 0, 0),
            linewidth=2,
        )
    )


def save_mask_outputs(output_dir, image, masks, scores, point_coords, point_labels, box):
    records = []
    image_area = image.shape[0] * image.shape[1]

    for idx, (mask, score) in enumerate(zip(masks, scores)):
        mask_area = int(mask.sum())
        area_ratio = mask_area / image_area
        output_path = output_dir / f"mask_{idx}.png"

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(point_coords, point_labels, plt.gca())
        show_box(box, plt.gca())
        plt.title(f"Mask {idx}, score={score:.3f}, area={area_ratio:.3f}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        records.append(
            {
                "mask_id": idx,
                "score": float(score),
                "mask_area": mask_area,
                "area_ratio": area_ratio,
                "file": output_path.name,
            }
        )

    with (output_dir / "mask_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mask_id", "score", "mask_area", "area_ratio", "file"])
        writer.writeheader()
        writer.writerows(records)


def write_prompt_config(output_dir, args):
    with (output_dir / "prompt_config.txt").open("w") as f:
        f.write(f"image={args.image}\n")
        f.write(f"checkpoint={args.checkpoint}\n")
        f.write(f"model_type={args.model_type}\n")
        f.write(f"point={args.point}\n")
        f.write(f"negative_point={args.negative_point}\n")
        f.write(f"box={args.box}\n")
        f.write(f"multimask={args.multimask}\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(args.image)
    point_coords, point_labels, box = build_prompt_arrays(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=args.multimask,
    )

    save_mask_outputs(output_dir, image, masks, scores, point_coords, point_labels, box)
    write_prompt_config(output_dir, args)
    print(f"SAM results saved to: {output_dir}")


if __name__ == "__main__":
    main()
