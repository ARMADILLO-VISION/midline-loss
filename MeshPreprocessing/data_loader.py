import os
import shutil
import trimesh
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


class DatasetPreparer:
    def __init__(self):
        self.patient_mapping = []  # List to map patient IDs to folder paths
        self.xukun_data = False    # Flag to determine whether to use Xukun-specific data
        self.fine_tuning = False   # Flag to determine if the data is for fine tuning

    def load_data(self):
        """Load all patients' model paths and annotations."""
        self.patient_mapping = []
        patient_data = []

        if self.xukun_data:
            # Logic for Xukun dataset
            patient_data = self.load_xukun_data()
        else:
            # Logic for default patient dataset
            patient_data = self.load_patient_data()

        print(f"Total patients loaded: {len(self.patient_mapping)}")
        return patient_data

    def load_xukun_data(self):
        """Load and process data from the Xukun dataset."""
        patient_data = []
        # List all patients by iterating over the 'livermesh' directory
        for patient_file in sorted(os.listdir(os.path.join(self.dataset_dir, "livermesh"))):
            patient_name = os.path.splitext(patient_file)[0]
            model_path = os.path.join(self.dataset_dir, "livermesh", f"{patient_name}.obj")
            edges_path = os.path.join(self.dataset_dir, "edges", f"{patient_name}.edges")
            seg_path = os.path.join(self.dataset_dir, "seg", f"{patient_name}.eseg")

            if os.path.isfile(model_path) and os.path.isfile(edges_path) and os.path.isfile(seg_path):
                annotations = self._process_xukun_patient(model_path, edges_path, seg_path)
                patient_data.append({
                    "model_path": model_path,
                    "annotations": annotations,
                    "patient_folder": patient_name
                })
                self.patient_mapping.append(patient_name)
            else:
                print(f"Skipping patient {patient_name} due to missing files.")
        return patient_data

    def _process_xukun_patient(self, patient_path, edges_path, seg_path):
        contours = {}
        edges = self._load_edges(edges_path)
        eseg_annotations = self._load_eseg(seg_path)
        contours['edges'] = edges
        contours['annotations'] = eseg_annotations
        return contours

    def load_patient_data(self):
        """Load and process data from the default patient dataset."""
        patient_data = []
        for patient_folder in sorted(os.listdir(self.dataset_dir)):
            patient_path = os.path.join(self.dataset_dir, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            model_path = self._get_model_path(patient_path)
            annotations = self._process_patient(patient_path)

            if not model_path or not annotations:
                print(f"Skipping patient {patient_folder} due to missing model or annotation files.")
                continue

            patient_data.append({
                "model_path": model_path,
                "annotations": annotations,  # This is a dict mapping XML filenames -> list of contours
                "patient_folder": patient_folder
            })
            self.patient_mapping.append(patient_folder)
        return patient_data

    def _get_model_path(self, patient_path):
        """Retrieve the path to the liver.obj file for a patient."""
        model_path = os.path.join(patient_path, "model", "liver.obj")
        if os.path.isfile(model_path):
            return model_path
        else:
            print(f"No model file found at {model_path}")
            return None

    def _load_edges(self, filepath):
        """Load edges from the .edges file."""
        with open(filepath, 'r') as f:
            edges = [list(map(int, line.strip().split())) for line in f.readlines()]
        return np.array(edges)

    def _load_eseg(self, filepath):
        """Load edge segmentation labels from the .eseg file."""
        with open(filepath, 'r') as f:
            annotations = np.array([int(line.strip()) for line in f.readlines()])
        return annotations

    def _process_patient(self, patient_path):
        """Process each patient's folder and extract 3D contours from XML files."""
        patient_data = {}
        contours_folder = os.path.join(patient_path, "2D-3D_contours")
        if not os.path.isdir(contours_folder):
            print(f"No contours folder for {patient_path}")
            return patient_data

        for xml_file in os.listdir(contours_folder):
            if not xml_file.endswith('.xml'):
                continue
            xml_path = os.path.join(contours_folder, xml_file)
            contours = self._parse_3d_contours(xml_path)
            # Only add the annotation if there are contours present
            if contours:
                patient_data[xml_file] = contours
        return patient_data

    def _parse_3d_contours(self, xml_path):
        """Parse 3D contour data from an XML file."""
        contours = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for contour in root.findall('contour'):
                contour_type = contour.find('contourType').text if contour.find('contourType') is not None else 'Unknown'
                model_points_element = contour.find('modelPoints/vertices')
                if model_points_element is None or model_points_element.text is None:
                    continue
                model_points = [int(point) for point in model_points_element.text.split(',')]
                contours.append({
                    'type': contour_type,
                    'model_points': model_points
                })
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {xml_path}: {e}")

        return contours

    def split_and_save_data(self, patient_data, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        Split data into train, validation, and test sets, then save NPZ files.

        For Xukun data, the original logic is retained.
        For default (patient) data, if there are patient folders named exactly
        "patient 4" or "patient 11", they will be used exclusively for the test set.
        All other patients are split between training and validation.
        """
        parent_dir = 'tuning_data_split' if self.fine_tuning else 'data_split'
        if os.path.exists(parent_dir):
            shutil.rmtree(parent_dir)
        os.makedirs(parent_dir)

        output_dirs = ['train', 'val', 'test']
        for dir_name in output_dirs:
            os.makedirs(os.path.join(parent_dir, dir_name), exist_ok=True)

        # If using Xukun data, use the original random split
        if self.xukun_data:
            train_data, temp_data = train_test_split(patient_data, train_size=train_size, random_state=42)
            val_data, test_data = train_test_split(temp_data, train_size=val_size / (val_size + test_size), random_state=42)
        else:
            # For default patient data, assign test set exclusively based on folder names
            test_data = [p for p in patient_data if p["patient_folder"] in ["patient4", "patient11"]]
            train_val_data = [p for p in patient_data if p["patient_folder"] not in ["patient4", "patient11"]]
            # Split remaining patients into training and validation sets using relative proportions
            total_prop = train_size + val_size
            if total_prop <= 0:
                raise ValueError("train_size and val_size cannot sum to 0.")
            relative_train_size = train_size / total_prop
            train_data, val_data = train_test_split(train_val_data, train_size=relative_train_size, random_state=42)

        def save_data(data, folder):
            for patient in data:
                mesh = trimesh.load(patient['model_path'], process=False)
                vertices = mesh.vertices
                if self.xukun_data:
                    # Original logic for Xukun data
                    labels = self.create_labels(vertices, patient['annotations'])
                    patient_file = os.path.join(folder, f'{patient["patient_folder"]}.npz')
                    np.savez_compressed(patient_file, vertices=vertices, labels=labels)
                    print(f"Saved Xukun data NPZ: {patient_file}")
                else:
                    # For default patient data, iterate over each XML annotation
                    for xml_filename, contours in patient['annotations'].items():
                        # Check if the XML contains valid 3D points for Ridge or Ligament
                        valid = any(contour['type'] in ['Ridge', 'Ligament'] and contour['model_points'] for contour in contours)
                        if not valid:
                            print(f"Skipping annotation {xml_filename} for patient {patient['patient_folder']} due to missing valid 3D points.")
                            continue

                        labels = np.zeros(vertices.shape[0], dtype=int)
                        for contour in contours:
                            contour_type = contour['type']
                            model_points = contour['model_points']
                            if contour_type == "Ridge":
                                for idx in model_points:
                                    if 0 <= idx < len(labels):
                                        labels[idx] = 1
                            elif contour_type == "Ligament":
                                for idx in model_points:
                                    if 0 <= idx < len(labels):
                                        labels[idx] = 2

                        base_xml = os.path.splitext(xml_filename)[0]
                        patient_file = os.path.join(folder, f"{base_xml}.npz")
                        np.savez_compressed(patient_file, vertices=vertices, labels=labels)
                        print(f"Saved annotation NPZ file: {patient_file}")

        save_data(train_data, os.path.join(parent_dir, 'train'))
        save_data(val_data, os.path.join(parent_dir, 'val'))
        save_data(test_data, os.path.join(parent_dir, 'test'))

        print(f"Data split into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test patients.")
        return train_data, val_data, test_data

    def augment_training_data(self, train_data, output_dir, num_augmentations=1, lowest_frequency_altered=5,
                              number_of_frequencies_to_alter=1, min_perturbation=-0.5, max_perturbation=0.5,
                              number_eigenvectors=100, visualise=False, spectral_augment=True):
        """
        Augment training data and save augmented point clouds.

        For default patient data, each XML annotation is augmented separately.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        if not train_data:
            print("No training data available for augmentation.")
            return

        print(f"Starting augmentation for {len(train_data)} training patients.")

        for patient_id, patient_info in enumerate(train_data):
            mesh = trimesh.load(patient_info["model_path"], process=False)
            vertices = mesh.vertices

            if self.xukun_data:
                labels = self.create_labels(vertices, patient_info["annotations"])
                for i in range(num_augmentations):
                    print(f"Applying augmentation {i + 1}/{num_augmentations} for Patient {patient_id}...")
                    base_vertices = vertices
                    augmented_vertices = self._augment_point_cloud(base_vertices)
                    augmented_filename = f"{patient_info['patient_folder']}_aug{i + 1}.npz"
                    output_file = os.path.join(output_dir, augmented_filename)
                    np.savez_compressed(output_file, vertices=augmented_vertices, labels=labels)
                    print(f"Saved augmented point cloud to {output_file}.")

                    if visualise:
                        self.visualise_augmented_patient(patient_id, vertices, labels, augmented_vertices)
            else:
                for xml_filename, contours in patient_info["annotations"].items():
                    valid = any(contour['type'] in ['Ridge', 'Ligament'] and contour['model_points'] for contour in contours)
                    if not valid:
                        print(f"Skipping augmentation for annotation {xml_filename} for patient {patient_info['patient_folder']} due to missing valid 3D points.")
                        continue

                    labels = np.zeros(vertices.shape[0], dtype=int)
                    for contour in contours:
                        contour_type = contour['type']
                        model_points = contour['model_points']
                        if contour_type == "Ridge":
                            for idx in model_points:
                                if 0 <= idx < len(labels):
                                    labels[idx] = 1
                        elif contour_type == "Ligament":
                            for idx in model_points:
                                if 0 <= idx < len(labels):
                                    labels[idx] = 2

                    for i in range(num_augmentations):
                        print(f"Applying augmentation {i + 1}/{num_augmentations} for Patient {patient_id}, annotation {xml_filename}...")
                        base_vertices = vertices
                        augmented_vertices = self._augment_point_cloud(base_vertices)
                        base_xml = os.path.splitext(xml_filename)[0]
                        augmented_filename = f"{patient_info['patient_folder']}_{base_xml}_aug{i + 1}.npz"
                        output_file = os.path.join(output_dir, augmented_filename)
                        np.savez_compressed(output_file, vertices=augmented_vertices, labels=labels)
                        print(f"Saved augmented annotation point cloud to {output_file}.")

                        if visualise:
                            self.visualise_augmented_patient(patient_id, vertices, labels, augmented_vertices)
        print("Data augmentation completed successfully.")

    def create_labels(self, vertices, annotations):
        """Original label creation for Xukun data."""
        labels = np.zeros(vertices.shape[0], dtype=int)  # Default all labels to 0 (background)

        if self.xukun_data:
            edges = annotations.get('edges', [])
            eseg_labels = annotations.get('annotations', [])
            if len(edges) != len(eseg_labels):
                print("Warning: Number of edges does not match number of annotations from .eseg.")
            else:
                for edge, ann in zip(edges, eseg_labels):
                    for vertex_index in edge:
                        if 0 <= vertex_index < len(labels):
                            labels[vertex_index] = ann
            corrected_labels = np.zeros_like(labels)
            corrected_labels[labels == 1] = 0  # Liver remains 0
            corrected_labels[labels == 2] = 2  # Ligament
            corrected_labels[labels == 3] = 1  # Ridge
            labels = corrected_labels
        else:
            print("Creating labels for default data is now handled per annotation.")
        return labels

    def visualise_augmented_patient(self, patient_id, original_vertices, labels, augmented_vertices):
        """Visualise original and augmented point clouds with labels for a single patient."""
        print(f"Visualising Patient {patient_id}...")
        fig = plt.figure(figsize=(12, 6))

        # Original Data
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(
            original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2],
            c=labels, cmap='viridis', s=5
        )
        ax1.set_title(f"Original Point Cloud (Patient {patient_id})")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # Augmented Data
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(
            augmented_vertices[:, 0], augmented_vertices[:, 1], augmented_vertices[:, 2],
            c=labels, cmap='viridis', s=5
        )
        ax2.set_title(f"Augmented Point Cloud (Patient {patient_id})")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        plt.tight_layout()
        plt.show()

    def _scale_point_cloud(self, points, scale_range=(0.65, 1.45)):
        """Randomly scales the point cloud by a factor within scale_range."""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return points * scale_factor

    def _rotate_point_cloud(self, points, rotation_range=(-360, 360)):
        """Rotates the point cloud around the z-axis by a random angle in degrees."""
        angle = np.radians(np.random.uniform(rotation_range[0], rotation_range[1]))
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval,  cosval, 0],
                                    [0,       0,      1]], dtype=np.float32)
        return np.dot(points, rotation_matrix.T)
    
    def _jitter_point_cloud(self, points, sigma=0.02, clip=0.035):
        """Add random Gaussian noise to each point."""
        noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
        return points + noise

    def _augment_point_cloud(self, points):
        """Applies scaling and rotation augmentation to the point cloud."""
        augmented_points = points.copy()
        augmented_points = self._scale_point_cloud(augmented_points)
        augmented_points = self._rotate_point_cloud(augmented_points)
        augmented_points = self._rotate_point_cloud(augmented_points)
        # Optionally, jitter can be added:
        # augmented_points = self._jitter_point_cloud(augmented_points)
        return augmented_points


def run_augmentation(preparer, data, args, dataset_type="train"):
    """Run augmentation on specified dataset (train/val)."""
    parent_dir = 'tuning_data_split' if preparer.fine_tuning else 'data_split'
    output_dir = os.path.join(parent_dir, dataset_type)
    preparer.augment_training_data(
        train_data=data,
        output_dir=output_dir,
        num_augmentations=args.num_augmentations,
        lowest_frequency_altered=args.lowest_frequency_altered,
        number_of_frequencies_to_alter=args.number_of_frequencies_to_alter,
        min_perturbation=args.min_perturbation,
        max_perturbation=args.max_perturbation,
        number_eigenvectors=args.number_eigenvectors,
        visualise=args.visualise,
        spectral_augment=args.spectral_augment
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and augment dataset into train, validation, and test sets")

    parser.add_argument('--dataset_dir', type=str, required=True, help="Directory where the dataset is located")
    parser.add_argument('--train_size', type=float, default=0.6, help="Proportion of data to be used for training")
    parser.add_argument('--val_size', type=float, default=0.2, help="Proportion of data to be used for validation")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of data to be used for testing")

    # Flag for Xukun data
    parser.add_argument('--xukun_data', action='store_true', help="Flag to use the Xukun dataset processing logic")
    # Fine tuning flag
    parser.add_argument('--fine_tuning', action='store_true', help="Flag to indicate if the data is for fine tuning (saves to tuning_data_split)")

    # Augmentation arguments
    parser.add_argument('--num_augmentations', type=int, default=1, help="Number of augmentations per patient/annotation")
    parser.add_argument('--visualise', action="store_true", help="Flag to visualise the augmentations")

    args = parser.parse_args()

    total = args.train_size + args.val_size + args.test_size
    if not np.isclose(total, 1.0):
        raise ValueError("The sum of train_size, val_size, and test_size must sum to 1.")

    dataset_preparer = DatasetPreparer()
    dataset_preparer.dataset_dir = args.dataset_dir
    dataset_preparer.xukun_data = args.xukun_data
    dataset_preparer.fine_tuning = args.fine_tuning

    patient_data = dataset_preparer.load_data()

    if not patient_data:
        print("No patient data found. Exiting.")
        exit(1)

    train_data, val_data, test_data = dataset_preparer.split_and_save_data(
        patient_data,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size
    )

    if args.num_augmentations > 0:
        print("Starting data augmentation on training data...")
        run_augmentation(dataset_preparer, train_data, args, dataset_type="train")
        # run_augmentation(dataset_preparer, val_data, args, dataset_type="val")
    else:
        print("No augmentations requested. Skipping augmentation step.")

    print("Data loading, splitting, and augmentation completed successfully.")
