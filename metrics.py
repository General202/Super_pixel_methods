import time
import numpy as np
import scipy.io
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation

def load_bsds500_gt(mat_path):
    """
    Завантажує ground truth розмітку з .mat файла датасету BSDS500.
    Повертає список масивів з сегментаціями від різних експертів.
    """
    mat = scipy.io.loadmat(mat_path)
    gt_data = mat['groundTruth']
    num_gt = gt_data.shape[1]
    
    segmentations = []
    boundaries = []
    
    for i in range(num_gt):
        seg = gt_data[0, i]['Segmentation'][0, 0]
        bound = gt_data[0, i]['Boundaries'][0, 0]
        segmentations.append(seg)
        boundaries.append(bound)
        
    return segmentations, boundaries


def compute_boundary_recall(gt_boundaries, superpixel_labels, tolerance=2):
    """
    Boundary Recall (BR): вимірює частку межевих пікселів із Ground Truth, 
    які знаходяться в межах r пікселів (tolerance) від меж суперпікселів.
    """
    sp_boundaries = find_boundaries(superpixel_labels, mode='inner')
    
    # Розширюємо межі суперпікселів на величину tolerance
    dilated_sp_bounds = binary_dilation(sp_boundaries, iterations=tolerance)
    
    # Рахуємо, скільки пікселів істинної межі потрапили в розширені межі суперпікселів
    matched_gt = np.logical_and(gt_boundaries > 0, dilated_sp_bounds)
    
    total_gt_boundary_pixels = np.sum(gt_boundaries > 0)
    if total_gt_boundary_pixels == 0:
        return 0.0
        
    return np.sum(matched_gt) / total_gt_boundary_pixels


def compute_undersegmentation_error(gt_segmentation, superpixel_labels):
    """
    Undersegmentation Error (UE): вимірює витік суперпікселів за межі істинних сегментів.
    """
    total_pixels = gt_segmentation.size
    error_sum = 0
    
    sp_ids = np.unique(superpixel_labels)
    gt_ids = np.unique(gt_segmentation)
    
    for gt_id in gt_ids:
        gt_mask = (gt_segmentation == gt_id)
        # Знаходимо всі суперпікселі, які перетинаються з цим об'єктом GT
        intersecting_sps = np.unique(superpixel_labels[gt_mask])
        
        for sp_id in intersecting_sps:
            sp_mask = (superpixel_labels == sp_id)
            sp_size = np.sum(sp_mask)
            overlap = np.sum(np.logical_and(gt_mask, sp_mask))
            
            # Помилка: беруться суперпікселі, що виходять за межі GT-сегмента
            error_sum += min(overlap, sp_size - overlap)
            
    return error_sum / total_pixels


def benchmark_algorithm(algo_fn, image, **kwargs):
    """
    Вимірює час виконання алгоритму та повертає карти міток (labels) і час у секундах.
    """
    start_time = time.time()
    labels = algo_fn(image, **kwargs)
    exec_time = time.time() - start_time
    
    return labels, exec_time