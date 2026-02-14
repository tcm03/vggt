import numpy as np

REAL_EPS: float = 1e-6

def eval_batch(
    pred_axes_dir: np.ndarray,
    pred_axes_orig: np.ndarray,
    pred_joint_logits: np.ndarray,
    gt_axes_dir: np.ndarray,
    gt_axes_orig: np.ndarray,
    gt_joint_types: np.ndarray,
    nonpad_masks: np.ndarray
):
    assert pred_axes_dir.shape == gt_axes_dir.shape, f"expected same shape, got {pred_axes_dir.shape} != {gt_axes_dir.shape}"
    assert pred_axes_dir.ndim == 3 and pred_axes_dir.shape[2] == 3, f"unexpected pred_pose.shape = {pred_axes_dir.shape}"
    assert pred_axes_orig.shape == gt_axes_orig.shape, f"expected same shape, got {pred_axes_orig.shape} != {gt_axes_orig.shape}"
    assert pred_axes_orig.ndim == 3 and pred_axes_orig.shape[2] == 3, f"unexpected pred_pose.shape = {pred_axes_orig.shape}"
    
    B, S, _ = pred_axes_dir.shape
    total_joint_axis_err = 0.
    total_joint_origin_err = 0.
    num_incorrect = 0
    num_revolute = 0
    for batch_idx in range(B):
        for frame_idx in range(S):
            if nonpad_masks[batch_idx, frame_idx]:
                pred_label = np.argmax(pred_joint_logits[batch_idx, frame_idx])
                gt_label = gt_joint_types[batch_idx, frame_idx]
                if pred_label != gt_label:
                    num_incorrect += 1

                pred_axis_dir: np.ndarray = pred_axes_dir[batch_idx, frame_idx]
                gt_axis_dir: np.ndarray = gt_axes_dir[batch_idx, frame_idx]
                pred_axis_orig: np.ndarray = pred_axes_orig[batch_idx, frame_idx]
                gt_axis_orig: np.ndarray = gt_axes_orig[batch_idx, frame_idx]
                result = eval_one(pred_axis_dir, gt_axis_dir, pred_axis_orig, gt_axis_orig)
                total_joint_axis_err += result["joint_axis_err"]
                if gt_joint_types[batch_idx, frame_idx] == 0: # [2026-02-12] @tcm: revolute
                    num_revolute += 1
                    total_joint_origin_err += result["joint_origin_err"]
    return total_joint_axis_err, total_joint_origin_err, num_revolute, num_incorrect

def eval_one(
    pred_axis_dir: np.ndarray,
    gt_axis_dir: np.ndarray,
    pred_axis_orig: np.ndarray,
    gt_axis_orig: np.ndarray
):
    assert pred_axis_dir.ndim in [1, 2], f"expect of shape (D,) or (1, D)"
    if pred_axis_dir.ndim == 2:
        assert pred_axis_dir.shape[0] == 1, f"expect of shape (1, D)"
    assert pred_axis_dir.shape == gt_axis_dir.shape, f"expect same shape, got {pred_axis_dir.shape} != {gt_axis_dir.shape}"
    
    res = {}
    pred_axis_dir = pred_axis_dir / (np.linalg.norm(pred_axis_dir) + REAL_EPS)
    gt_axis_dir = gt_axis_dir / (np.linalg.norm(gt_axis_dir) + REAL_EPS)
    axis_err = min(
        np.arccos(np.dot(pred_axis_dir, gt_axis_dir) / (np.linalg.norm(pred_axis_dir) * np.linalg.norm(gt_axis_dir))),
        np.arccos(np.dot(pred_axis_dir, -gt_axis_dir) / (np.linalg.norm(pred_axis_dir) * np.linalg.norm(gt_axis_dir)))
    )
    res['joint_axis_err'] = axis_err
    
    orig_diff = pred_axis_orig - gt_axis_orig
    orig_err = np.linalg.norm(np.dot(orig_diff, np.cross(pred_axis_dir, gt_axis_dir))) / np.linalg.norm(np.cross(pred_axis_dir, gt_axis_dir))
    res['joint_origin_err'] = orig_err 

    return res