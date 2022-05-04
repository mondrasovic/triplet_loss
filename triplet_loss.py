import torch
from torch import nn
from torch.nn import functional as F


def _pairwise_l2_dist(
    embs: torch.Tensor,
    *,
    squared: bool = False,
    eps: float = 1e-16
) -> torch.Tensor:
    """Computes a 2D matrix of L2 or squared L2 distances between all the
    embeddings.
    Args:
        embs (torch.Tensor): embeddings of shape [B,E]
        squared (bool, optional): If True, the output is the pairwise squared L2
        distance, if False, then standard L2 distance is computed. Defaults to
        False.
        eps (float, optional): Small value to add to the zero distances to
        prevent infinite gradients after applying sqrt(). Defaults to 1e-16.
    Returns:
        torch.Tensor: A 2D distance matrix of shape [B,B].
    """
    dot_product = torch.matmul(embs, embs.T)  # [B,B]
    square_norm = torch.diag(dot_product)  # [B,]

    # Apply the l2 norm formula using the dot product:
    # ||A - B||^2 = ||A||^2 - 2<A,B> + ||B||^2
    distances = (
        torch.unsqueeze(square_norm, dim=1) -  # [B,1]
        (2 * dot_product) +  # [B,B]
        torch.unsqueeze(square_norm, dim=0)  # [1,B]
    )

    # Due to potential errors caused by numerical instability, some values may
    # have become negative. Thus, we have to make sure the min. value is zero.
    zero = torch.tensor(0.0)
    distances = torch.maximum(distances, zero)  # [B,B]

    if not squared:
        # Since the gradient of sqrt(0) is infinite, we, therefore, have to
        # add a small epsilon to the zero terms to prevent this.
        zeroes_mask = ((distances - zero) < eps).float()  # [B,B]
        distances += zeroes_mask * eps

        distances = torch.sqrt(distances)  # [B,B]

        # Set all the "zero" values back to zero after adding the epsilon value.
        distances *= (1.0 - zeroes_mask)

    return distances


def _get_anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    """Generates a 2D mask where M[a,p] is True iff anchor (a) and positive (p)
    have identical labels but distinct indices, i.e., belong to different
    objects.
    Args:
        labels (torch.Tensor): Labels of shape [B,].
    Returns:
        torch.Tensor: 2D boolean mask of shape [B,B].
    """
    labels_eq_mask = (labels[..., None] == labels[None, ...])  # [B,B]
    idxs_neq_mask = ~torch.eye(
        len(labels), dtype=torch.bool, device=labels.device
    )  # [B,B]
    anchor_positive_mask = (labels_eq_mask & idxs_neq_mask)  # [B,B]

    return anchor_positive_mask


def _get_anchor_negative_mask(labels: torch.Tensor) -> torch.Tensor:
    """Generates a 2D mask where M[a,n] is True iff anchor (a) and negative (n)
    have distinct labels.
    Args:
        labels (torch.Tensor): Labels of shape [B,].
    Returns:
        torch.Tensor: 2D boolean mask of shape [B,B].
    """
    anchor_negative_mask = (labels[..., None] != labels[None, ...])  # [B,B]

    return anchor_negative_mask


def _get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Generates a 3D mask where M[a,p,n] is True iff anchor (a) and positive(p)
    are distinct objects but have the same class, whereas anchor (a) and
    negative (p) are also distinct objects but of different class.
    Args:
        labels (torch.Tensor): Labels of shape [B,].
    Returns:
        torch.Tensor: 3D boolean mask of shape [B,B,B].
    """
    idxs_neq_mask = ~torch.eye(
        len(labels), dtype=torch.bool, device=labels.device
    )  # [B,B]
    idx_i_neq_j_mask = torch.unsqueeze(idxs_neq_mask, dim=2)  # [B,B,1]
    idx_i_neq_k_mask = torch.unsqueeze(idxs_neq_mask, dim=1)  # [B,1,B]
    idx_j_neq_k_mask = torch.unsqueeze(idxs_neq_mask, dim=0)  # [1,B,B]
    triplet_idxs_neq_mask = (
        idx_i_neq_j_mask & idx_i_neq_k_mask & idx_j_neq_k_mask
    )  # [B,B,B]

    labels_eq_mask = (labels[..., None] == labels[None, ...])  # [B,B]
    label_i_eq_j = torch.unsqueeze(labels_eq_mask, dim=2)  # [B,B,1]
    label_i_neq_k = ~torch.unsqueeze(labels_eq_mask, dim=1)  # [B,1,B]
    triplet_labels_valid_mask = (label_i_eq_j & label_i_neq_k)  # [B,B,B]

    triplet_mask = (
        triplet_idxs_neq_mask & triplet_labels_valid_mask
    )  # [B,B,B]

    return triplet_mask


class BatchAllTripletLoss(nn.Module):
    """Triplet loss with 'batch all' mining strategy.

    Equation (6) from the paper:
        Hermans, A., Beyer, L., & Leibe, B. (2017).
        In defense of the triplet loss for person re-identification.
    """
    def __init__(self, margin: float = 1.0, squared: bool = True) -> None:
        """Constructor.
        Args:
            margin (float, optional): Margin of separation between positive and
            negative samples. Defaults to 1.0.
            squared (bool, optional): If True, the  the pairwise squared L2
            distance is used, if False, then standard L2 distance is computed.
            Defaults to True.
        """
        super().__init__()

        self.margin: float = margin
        self.squared: bool = squared

    def forward(self, embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generates all possible valid triplets but computes the loss over the
        positive ones in terms of the loss function value.
        Args:
            embs (torch.Tensor): Embeddings of shape [B,E].
            labels (torch.Tensor): Labels of shape [B,].
        Returns:
            torch.Tensor: Triplet loss scalar.
        """
        pairwise_dist = _pairwise_l2_dist(embs, squared=self.squared)  # [B,B]

        anchor_positive_dist = torch.unsqueeze(pairwise_dist, dim=2)  # [B,B,1]
        anchor_negative_dist = torch.unsqueeze(pairwise_dist, dim=1)  # [B,1,B]

        triplet_loss = (
            anchor_positive_dist - anchor_negative_dist + self.margin
        )  # [B,B,B]

        # Put zero to invalid triplets.
        mask = _get_triplet_mask(labels)  # [B,B,B]
        triplet_loss *= mask.float()  # [B,B,B]

        # Remove triplets that incurred negative loss, i.e., were too easy.
        triplet_loss = torch.clamp(triplet_loss, min=0)  # [B,B,B]

        eps = 1e-16
        loss_positive_triplets_mask = triplet_loss > eps  # [B,B,B]
        n_loss_positive_triplets = torch.sum(loss_positive_triplets_mask)  # [c]

        triplet_loss = (
            torch.sum(triplet_loss) / (n_loss_positive_triplets + eps)
        )  # [c]

        return triplet_loss


class SemiHardTripletLoss(nn.Module):
    """Triplet loss with 'batch hard' negative mining. These triplets are
    considered "semi hard" (moderate) since they are selected only within a
    specific minibatch.

    Equation (5) from the paper:
        Hermans, A., Beyer, L., & Leibe, B. (2017).
        In defense of the triplet loss for person re-identification.
    """
    def __init__(self, margin: float = 1.0, squared: bool = True) -> None:
        """Constructor.
        Args:
            margin (float, optional): Margin of separation between positive and
            negative samples. Defaults to 1.0.
            squared (bool, optional): If True, the  the pairwise squared L2
            distance is used, if False, then standard L2 distance is computed.
            Defaults to True.
        """
        super().__init__()

        self.margin: float = margin
        self.squared: bool = squared

    def forward(self, embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes the loss between all the hard triplets. For a given anchor
        sample, a hard triplet is formed by finding the hardest positive, i.e,
        a different sample with the same label which has the maximum distance;
        and also the hardest negative, .e., a different sample with a distinct
        label which has the minimum distance.
        Args:
            embs (torch.Tensor): Embeddings of shape [B,E].
            labels (torch.Tensor): Labels of shape [B,].
        Returns:
            torch.Tensor: Triplet loss scalar.
        """
        pairwise_dist = _pairwise_l2_dist(embs, squared=self.squared)  # [B,B]

        anchor_positive_mask = _get_anchor_positive_mask(labels).float(
        )  # [B,B]
        anchor_positive_dist = pairwise_dist * anchor_positive_mask  # [B,B]
        hardest_positive_dist = torch.amax(
            anchor_positive_dist, dim=1, keepdim=True
        )  # [B,1]

        anchor_negative_mask = _get_anchor_negative_mask(labels).float(
        )  # [B,B]
        max_anchor_negative_dist = torch.amax(
            pairwise_dist, dim=1, keepdim=True
        )  # [B,1]
        anchor_negative_dist = (
            pairwise_dist +
            (1 - anchor_negative_mask) * max_anchor_negative_dist
        )  # [B,B]
        hardest_negative_dist = torch.amin(
            anchor_negative_dist, dim=1, keepdim=True
        )  # [B,1]

        triplet_loss = torch.clamp(
            hardest_positive_dist - hardest_negative_dist + self.margin, min=0
        )  # [B,1]
        triplet_loss = torch.mean(triplet_loss)  # [c]

        return triplet_loss
