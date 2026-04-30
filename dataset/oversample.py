import random
from collections import Counter


def balance_binary_labels(
        segment_ids,
        labels,
        random_seed = 42
):
    """
    对二分类标签进行过采样，使 label=0 和 label=1 数量相等。

    Args:
        segment_ids: 原始样本 ID 列表
        labels: 对应的二分类标签列表（仅含 0 和 1）
        random_seed: 随机种子

    Returns:
        (balanced_segment_ids, balanced_labels): 平衡后的两个列表
    """
    # 检查是否为二分类
    unique_labels = set(labels)
    assert unique_labels <= {0, 1}, f"Only binary labels (0/1) supported, got {unique_labels}"
    assert len(segment_ids) == len(labels), "segment_ids and labels must have same length"

    # 分离索引
    idx_0 = [i for i, y in enumerate(labels) if y == 0]
    idx_1 = [i for i, y in enumerate(labels) if y == 1]
    n_0, n_1 = len(idx_0), len(idx_1)

    print(f"  Original: label=0 → {n_0}, label=1 → {n_1}")

    if n_0 == n_1:
        balanced_indices = list(range(len(labels)))
    elif n_1 < n_0:
        # 过采样少数类 label=1
        rng = random.Random(random_seed)
        extra_idx = rng.choices(idx_1, k=n_0 - n_1)
        balanced_indices = idx_0 + idx_1 + extra_idx
    else:
        # 过采样 label=0（理论上少见）
        rng = random.Random(random_seed)
        extra_idx = rng.choices(idx_0, k=n_1 - n_0)
        balanced_indices = idx_1 + idx_0 + extra_idx

    # 打乱顺序
    rng = random.Random(random_seed)
    rng.shuffle(balanced_indices)

    # 构建新列表
    balanced_segment_ids = [segment_ids[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]

    final_counts = Counter(balanced_labels)
    print(f"  Balanced: {dict(final_counts)} | Total samples: {len(balanced_labels)}")

    return balanced_segment_ids, balanced_labels