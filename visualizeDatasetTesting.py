import torch
import matplotlib.pyplot as plt
from brats import get_datasets_independent, BraTSBezierAug3DDataset
from utils.augment import DataAugmenter
from utils.all_utils import listdir, get_brats_folder

def inspect_and_visualize_dataset(
    dataset_folder,
    mode,
    target_size=(128, 128, 128),
    version="brats2020",
    modality="t2",
    use_augmenter = False
):
    print(f"Inspecting dataset with modality '{modality}' in mode '{mode}'...")

    dataset = get_datasets_independent(
        dataset_folder=dataset_folder,
        mode=mode,
        target_size=target_size,
        version=version,
        modality=modality
    )

    print(f"Total samples in the dataset: {len(dataset)}")

    sample_idx = 2
    sample = dataset[sample_idx]

    image = sample["image"].unsqueeze(0)
    label = sample["label"].unsqueeze(0)

    print(f"Sample ID: {sample['patient_id']}")
    print(f"Image shape: {image.shape} (Expected: [D, H, W])")
    print(f"Label shape: {label.shape} (Expected: [3, D, H, W])")

    if use_augmenter:
        augmenter = DataAugmenter()
        image, label = augmenter(image, label)
        print("Augmentation applied to the dataset.")

    image = image.squeeze(0)
    label = label.squeeze(0)

    print(f"Non-zero indexes: {sample['nonzero_indexes']}")
    print(f"Box slice (cropping info): {sample['box_slice']}")
    print(f"Pad list (padding info): {sample['pad_list']}")

    print(f"Image tensor min value: {image.min().item()}, max value: {image.max().item()}")
    print(f"Label tensor unique values: {torch.unique(label)}")

    middle_idx = image.shape[1] // 2
    middle_image_slices = [image[c, middle_idx].cpu().numpy() for c in range(image.shape[0])]
    middle_label_slices = [label[c, middle_idx].cpu().numpy() for c in range(label.shape[0])]

    plt.figure(figsize=(12, 8))

    for i, image_slice in enumerate(middle_image_slices):
        plt.subplot(1, len(middle_image_slices) + len(middle_label_slices), i + 1)
        plt.imshow(image_slice, cmap="gray")
        plt.title(f"Image Modality {i} - Slice {middle_idx}")
        plt.axis("off")

    for i, label_slice in enumerate(middle_label_slices):
        plt.subplot(1, len(middle_image_slices) + len(middle_label_slices), len(middle_image_slices) + i + 1)
        plt.imshow(label_slice, cmap="jet")
        plt.title(f"Label Channel {i} - Slice {middle_idx}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig("visualization_output.png")

########################################################

def inspect_and_visualize_dataset(
    dataset_folder,
    mode,
    target_size = (128, 128, 128),
    version = 'brats2020',
    modality = 't2',
    use_augmenter = False,
    use_bezier = False,
    domain_label = 0,
    sample_id = 2
):
    print(f"Inspecting dataset with modality '{modality}' in mode '{mode}'...")

    if use_bezier:
        print(f"Using BraTSBezierAugDataset with domain_label = {domain_label}, augment = True")
        dataset_folder_actual = get_brats_folder(dataset_folder, mode, version = version)
        patient_ids = [x for x in listdir(dataset_folder_actual)]

        dataset = BraTSBezierAug3DDataset(
            patients_dir = dataset_folder_actual,
            patient_ids=patient_ids,
            mode=mode,
            target_size=target_size,
            version=version,
            modality=modality,
            domain_label=domain_label,
            augment=True,
            seed = 42
        )
    else:
        print(f"Using standard BraTSIndependent dataset (no Bezier)")
        dataset = get_datasets_independent(
            dataset_folder=dataset_folder,
            mode=mode,
            target_size=target_size,
            version=version,
            modality=modality
        )

    print(f"Total samples in the dataset: {len(dataset)}")

    sample_idx = sample_id
    sample = dataset[sample_idx]

    image = sample["image"].unsqueeze(0)
    label = sample["label"].unsqueeze(0)

    print(f"Sample ID: {sample['patient_id']}")
    print(f"Image shape (including batch dim): {image.shape}")
    print(f"Label shape (including batch dim): {label.shape}")

    if use_augmenter:
        augmenter = DataAugmenter()
        image, label = augmenter(image, label)
        print("Additional Augmenter applied on top of dataset logic.")

    image = image.squeeze(0)
    label = label.squeeze(0)

    if 'domain_label' in sample:
        print(f"Domain Label in sample: {sample['domain_label']}")

    if "nonzero_indexes" in sample:
        print(f"Non-zero indexes: {sample['nonzero_indexes']}")
    if "box_slice" in sample:
        print(f"Box slice (cropping info): {sample['box_slice']}")
    if "pad_list" in sample:
        print(f"Pad list (padding info): {sample['pad_list']}")

    print(f"Image min: {image.min().item():.4f}, max: {image.max().item():.4f}")
    print(f"Label unique values: {torch.unique(label)}")

    depth = image.shape[1]
    middle_idx = depth // 2

    middle_image_slices = [image[c, middle_idx].cpu().numpy() for c in range(image.shape[0])]
    middle_label_slices = [label[c, middle_idx].cpu().numpy() for c in range(label.shape[0])]

    plt.figure(figsize = (12, 8))

    total_plots = len(middle_image_slices) + len(middle_label_slices)
    for i, im_slice in enumerate(middle_image_slices):
        plt.subplot(1, total_plots, i + 1)
        plt.imshow(im_slice, cmap="gray")
        plt.title(f"Image Ch {i} Slice {middle_idx}")
        plt.axis("off")

    for i, lb_slice in enumerate(middle_label_slices):
        plt.subplot(1, total_plots, len(middle_image_slices) + i + 1)
        plt.imshow(lb_slice, cmap="jet")
        plt.title(f"Label Ch {i} Slice {middle_idx}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig("visualization_output.png")


#########################################################

if __name__ == "__main__":
    dataset_folder = "/home/monetai/Desktop/dillan/dataBratsFormatted"

    inspect_and_visualize_dataset(
        dataset_folder = dataset_folder,
        mode = 'train',
        modality = 'flair',
        use_bezier = True,
        domain_label = 2,
        use_augmenter = False,
        sample_id = 2
    )


