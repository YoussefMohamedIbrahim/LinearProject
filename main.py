"""
PCA Eigenfaces Project
Face Recognition using Principal Component Analysis (PCA)
Uses only NumPy and PIL as requested
"""

import numpy as np
from PIL import Image
import os


def load_images(dataset_path, image_size=None):
    """
    Load all face images from the dataset directory.
    
    Args:
        dataset_path: Path to the Dataset folder containing subject folders (s1, s2, ...)
        image_size: Optional tuple (width, height) to resize images
    
    Returns:
        images: numpy array of shape (num_images, height * width)
        labels: numpy array of subject labels
    """
    images = []
    labels = []
    
    # Get all subject folders (s1, s2, ..., s40)
    subject_folders = sorted([f for f in os.listdir(dataset_path) 
                              if os.path.isdir(os.path.join(dataset_path, f)) and f.startswith('s')])
    
    for subject_folder in subject_folders:
        subject_path = os.path.join(dataset_path, subject_folder)
        # Extract subject number as label
        label = int(subject_folder[1:])
        
        # Get all .pgm images in the subject folder
        image_files = sorted([f for f in os.listdir(subject_path) if f.endswith('.pgm')])
        
        for image_file in image_files:
            image_path = os.path.join(subject_path, image_file)
            
            # Load image using PIL
            img = Image.open(image_path)
            
            # Resize if specified
            if image_size is not None:
                img = img.resize(image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and flatten
            img_array = np.array(img, dtype=np.float64).flatten()
            
            images.append(img_array)
            labels.append(label)
    
    return np.array(images), np.array(labels)


def compute_mean_face(images):
    """
    Compute the mean face from all images.
    
    Args:
        images: numpy array of shape (num_images, num_pixels)
    
    Returns:
        mean_face: numpy array of shape (num_pixels,)
    """
    return np.mean(images, axis=0)


def center_images(images, mean_face):
    """
    Center images by subtracting the mean face.
    
    Args:
        images: numpy array of shape (num_images, num_pixels)
        mean_face: numpy array of shape (num_pixels,)
    
    Returns:
        centered_images: numpy array of shape (num_images, num_pixels)
    """
    return images - mean_face


def compute_eigenfaces(centered_images, num_components=None):
    """
    Compute eigenfaces using PCA.
    Uses the trick: compute eigenvectors of A^T * A instead of A * A^T
    when number of images << number of pixels.
    
    Args:
        centered_images: numpy array of shape (num_images, num_pixels)
        num_components: number of eigenfaces to keep (None = all)
    
    Returns:
        eigenfaces: numpy array of shape (num_components, num_pixels)
        eigenvalues: numpy array of eigenvalues
    """
    num_images, num_pixels = centered_images.shape
    
    # Use the computational trick when images > pixels
    # Compute A^T * A (smaller covariance matrix)
    # L = A^T * A (num_images x num_images) instead of A * A^T (num_pixels x num_pixels)
    L = np.dot(centered_images, centered_images.T) / num_images
    
    # Compute eigenvalues and eigenvectors of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Sort by eigenvalue (descending order)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Convert eigenvectors of L to eigenvectors of covariance matrix
    # eigenfaces = A * v (project back to original space)
    eigenfaces = np.dot(centered_images.T, eigenvectors).T
    
    # Normalize eigenfaces
    for i in range(eigenfaces.shape[0]):
        norm = np.linalg.norm(eigenfaces[i])
        if norm > 0:
            eigenfaces[i] = eigenfaces[i] / norm
    
    # Keep only the specified number of components
    if num_components is not None:
        eigenfaces = eigenfaces[:num_components]
        eigenvalues = eigenvalues[:num_components]
    
    return eigenfaces, eigenvalues


def project_to_eigenspace(images, mean_face, eigenfaces):
    """
    Project images onto the eigenface space.
    
    Args:
        images: numpy array of shape (num_images, num_pixels) or (num_pixels,)
        mean_face: numpy array of shape (num_pixels,)
        eigenfaces: numpy array of shape (num_components, num_pixels)
    
    Returns:
        weights: numpy array of shape (num_images, num_components) or (num_components,)
    """
    centered = images - mean_face
    return np.dot(centered, eigenfaces.T)


def reconstruct_face(weights, mean_face, eigenfaces):
    """
    Reconstruct a face from its weights in eigenspace.
    
    Args:
        weights: numpy array of weights
        mean_face: numpy array of shape (num_pixels,)
        eigenfaces: numpy array of shape (num_components, num_pixels)
    
    Returns:
        reconstructed: numpy array of shape (num_pixels,)
    """
    return mean_face + np.dot(weights, eigenfaces)


def recognize_face(test_image, training_weights, training_labels, mean_face, eigenfaces):
    """
    Recognize a face by finding the nearest neighbor in eigenspace.
    
    Args:
        test_image: numpy array of shape (num_pixels,)
        training_weights: numpy array of shape (num_training, num_components)
        training_labels: numpy array of training labels
        mean_face: numpy array of shape (num_pixels,)
        eigenfaces: numpy array of shape (num_components, num_pixels)
    
    Returns:
        predicted_label: predicted subject label
        min_distance: distance to the nearest neighbor
    """
    # Project test image to eigenspace
    test_weights = project_to_eigenspace(test_image, mean_face, eigenfaces)
    
    # Find nearest neighbor using Euclidean distance
    distances = np.linalg.norm(training_weights - test_weights, axis=1)
    min_idx = np.argmin(distances)
    
    return training_labels[min_idx], distances[min_idx]


def array_to_image(array, image_shape):
    """
    Convert a flattened array back to an image.
    
    Args:
        array: numpy array of shape (num_pixels,)
        image_shape: tuple (height, width)
    
    Returns:
        PIL Image object
    """
    # Normalize to 0-255 range
    array = array.reshape(image_shape)
    array = array - array.min()
    if array.max() > 0:
        array = array * 255 / array.max()
    return Image.fromarray(array.astype(np.uint8))


def split_data(images, labels, train_ratio=0.7):
    """
    Split data into training and testing sets.
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
        train_ratio: ratio of training data
    
    Returns:
        train_images, train_labels, test_images, test_labels
    """
    unique_labels = np.unique(labels)
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    for label in unique_labels:
        # Get all images for this subject
        mask = labels == label
        subject_images = images[mask]
        
        # Split images for this subject
        num_train = int(len(subject_images) * train_ratio)
        
        train_images.extend(subject_images[:num_train])
        train_labels.extend([label] * num_train)
        test_images.extend(subject_images[num_train:])
        test_labels.extend([label] * (len(subject_images) - num_train))
    
    return (np.array(train_images), np.array(train_labels), 
            np.array(test_images), np.array(test_labels))


def evaluate_recognition(test_images, test_labels, train_weights, train_labels, mean_face, eigenfaces):
    """
    Evaluate face recognition accuracy.
    
    Args:
        test_images: numpy array of test images
        test_labels: numpy array of test labels
        train_weights: numpy array of training weights
        train_labels: numpy array of training labels
        mean_face: mean face
        eigenfaces: eigenfaces matrix
    
    Returns:
        accuracy: recognition accuracy
    """
    # Project all test images at once (vectorized - much faster)
    test_weights = project_to_eigenspace(test_images, mean_face, eigenfaces)
    
    # Find nearest neighbor for each test image using broadcasting
    correct = 0
    for i in range(len(test_images)):
        distances = np.linalg.norm(train_weights - test_weights[i], axis=1)
        predicted_label = train_labels[np.argmin(distances)]
        if predicted_label == test_labels[i]:
            correct += 1
    
    return correct / len(test_images)


def save_visualization(images_list, titles, filename, image_shape, cols=5):
    """
    Save a visualization of multiple images as a grid.
    
    Args:
        images_list: list of flattened image arrays
        titles: list of titles for each image
        filename: output filename
        image_shape: tuple (height, width)
        cols: number of columns in the grid
    """
    num_images = len(images_list)
    rows = (num_images + cols - 1) // cols
    
    height, width = image_shape
    margin = 5
    title_height = 20
    
    # Create output image
    out_width = cols * (width + margin) + margin
    out_height = rows * (height + margin + title_height) + margin
    output = Image.new('L', (out_width, out_height), 255)
    
    for idx, (img_array, title) in enumerate(zip(images_list, titles)):
        row = idx // cols
        col = idx % cols
        
        x = margin + col * (width + margin)
        y = margin + row * (height + margin + title_height) + title_height
        
        img = array_to_image(img_array, image_shape)
        output.paste(img, (x, y))
    
    output.save(filename)
    print(f"Saved visualization to {filename}")


def main():
    print("=" * 60)
    print("PCA EIGENFACES - Face Recognition Project")
    print("=" * 60)
    
    # Configuration
    dataset_path = "Dataset"
    num_eigenfaces = 50  # Number of eigenfaces to use
    train_ratio = 0.7    # 70% training, 30% testing
    
    # Step 1: Load images
    print("\n[1] Loading face images...")
    images, labels = load_images(dataset_path)
    num_images, num_pixels = images.shape
    
    # Determine image shape (assuming ORL dataset: 112x92)
    # Try to infer from first image
    sample_img = Image.open(os.path.join(dataset_path, "s1", "1.pgm"))
    image_shape = (sample_img.height, sample_img.width)
    print(f"    Loaded {num_images} images")
    print(f"    Image shape: {image_shape[1]}x{image_shape[0]} pixels")
    print(f"    Number of subjects: {len(np.unique(labels))}")
    
    # Step 2: Split data
    print("\n[2] Splitting data into training and testing sets...")
    train_images, train_labels, test_images, test_labels = split_data(
        images, labels, train_ratio)
    print(f"    Training images: {len(train_images)}")
    print(f"    Testing images: {len(test_images)}")
    
    # Step 3: Compute mean face
    print("\n[3] Computing mean face...")
    mean_face = compute_mean_face(train_images)
    
    # Step 4: Center images
    print("\n[4] Centering images (subtracting mean face)...")
    centered_train = center_images(train_images, mean_face)
    
    # Step 5: Compute eigenfaces
    print(f"\n[5] Computing eigenfaces using PCA...")
    eigenfaces, eigenvalues = compute_eigenfaces(centered_train, num_eigenfaces)
    print(f"    Computed {len(eigenfaces)} eigenfaces")
    
    # Calculate variance explained
    total_variance = np.sum(eigenvalues)
    variance_explained = np.cumsum(eigenvalues) / total_variance * 100
    print(f"    Variance explained by {num_eigenfaces} components: {variance_explained[-1]:.2f}%")
    
    # Step 6: Project training images to eigenspace
    print("\n[6] Projecting training images to eigenface space...")
    train_weights = project_to_eigenspace(train_images, mean_face, eigenfaces)
    
    # Step 7: Evaluate recognition
    print("\n[7] Evaluating face recognition...")
    accuracy = evaluate_recognition(test_images, test_labels, train_weights, 
                                   train_labels, mean_face, eigenfaces)
    print(f"    Recognition Accuracy: {accuracy * 100:.2f}%")
    
    # Step 8: Test with different numbers of eigenfaces
    print("\n[8] Accuracy vs Number of Eigenfaces:")
    print("    " + "-" * 40)
    
    # Pre-compute all test weights once
    all_test_weights = project_to_eigenspace(test_images, mean_face, eigenfaces)
    
    for n_comp in [10, 20, 30, 40, 50, 75, 100]:
        if n_comp <= len(eigenfaces):
            tw_subset = train_weights[:, :n_comp]
            test_w_subset = all_test_weights[:, :n_comp]
            
            correct = 0
            for i in range(len(test_images)):
                distances = np.linalg.norm(tw_subset - test_w_subset[i], axis=1)
                pred = train_labels[np.argmin(distances)]
                if pred == test_labels[i]:
                    correct += 1
            
            acc = correct / len(test_images)
            print(f"    {n_comp:3d} eigenfaces: {acc * 100:.2f}%")
    
    # Step 9: Face reconstruction demo
    print("\n[9] Face reconstruction demonstration...")
    test_idx = 0
    original = test_images[test_idx]
    
    print("    Reconstruction quality with different numbers of eigenfaces:")
    for n_comp in [10, 25, 50, num_eigenfaces]:
        if n_comp <= len(eigenfaces):
            weights = project_to_eigenspace(original, mean_face, eigenfaces[:n_comp])
            reconstructed = reconstruct_face(weights, mean_face, eigenfaces[:n_comp])
            mse = np.mean((original - reconstructed) ** 2)
            print(f"    {n_comp:3d} eigenfaces: MSE = {mse:.2f}")
    
    # Step 10: Save visualizations
    print("\n[10] Saving visualizations...")
    
    # Save mean face
    mean_img = array_to_image(mean_face, image_shape)
    mean_img.save("mean_face.png")
    print("    Saved mean_face.png")
    
    # Save top eigenfaces
    eigenface_images = [eigenfaces[i] for i in range(min(10, len(eigenfaces)))]
    eigenface_titles = [f"Eigenface {i+1}" for i in range(len(eigenface_images))]
    save_visualization(eigenface_images, eigenface_titles, "eigenfaces.png", image_shape, cols=5)
    
    # Save reconstruction comparison
    n_components_list = [5, 10, 25, 50, num_eigenfaces]
    recon_images = [original]
    recon_titles = ["Original"]
    
    for n_comp in n_components_list:
        if n_comp <= len(eigenfaces):
            weights = project_to_eigenspace(original, mean_face, eigenfaces[:n_comp])
            reconstructed = reconstruct_face(weights, mean_face, eigenfaces[:n_comp])
            recon_images.append(reconstructed)
            recon_titles.append(f"n={n_comp}")
    
    save_visualization(recon_images, recon_titles, "reconstruction.png", image_shape, cols=6)
    
    print("\n" + "=" * 60)
    print("Project completed successfully!")
    print("=" * 60)
    print("\nOutput files generated:")
    print("  - mean_face.png: The average face of all training images")
    print("  - eigenfaces.png: The top 10 eigenfaces (principal components)")
    print("  - reconstruction.png: Face reconstruction with varying eigenfaces")
    
    return {
        'mean_face': mean_face,
        'eigenfaces': eigenfaces,
        'eigenvalues': eigenvalues,
        'train_weights': train_weights,
        'train_labels': train_labels,
        'accuracy': accuracy,
        'image_shape': image_shape
    }


if __name__ == "__main__":
    results = main()
