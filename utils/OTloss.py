import torch
import torch.nn as nn
import torch.nn.functional as F

class OT_loss(nn.Module):
    """
    The proposed OT-based loss.
    """
    def __init__(self,  cost_fn ='iou', beta = 1, gamma0 = 1/2, G0 = 1., num_chunks=1):
        super(OT_loss, self).__init__()
        self.cost_fn = cost_fn
        self.beta = beta
        self.gamma = gamma0
        self.G = G0
        self.start = True
        self.num_chunks = num_chunks

    def forward(self, gt_arr, sample_arr, prob, prob_gt, sample_shape):
        self.B, self.N, self.C, self.H, self.W = sample_arr.shape
        self.M = gt_arr.shape[1]
        self.K, self.S = sample_shape

        cost = self.get_cost_matrix(sample_arr.flatten(3), gt_arr.flatten(2), self.cost_fn, num_chunks=self.num_chunks) # self.B, self.N, self.M
        with torch.no_grad():
            P = self.get_coupling_matrix(cost, prob_gt)
        seg_loss = (P.detach() * cost).sum([1, 2]).mean(0)
        kl_loss = self.get_kl_loss(P, prob, sample_arr)

        loss = seg_loss + self.beta * kl_loss
        return loss, seg_loss, kl_loss

    def get_kl_loss(self, P, prob, sample_arr = None):
        """
        Get the KL divergence loss.
        """
        target_prob = P.sum(-1).detach()

        if self.G < 1: # Apply soft gradient. We use it only for the cityscapes dataset.
            prob, target_prob = self.get_soft_gradients(sample_arr, prob, target_prob)

        kl_loss = F.kl_div(torch.log(prob + 1e-8), target_prob)
        # print(target_prob_grouped[0])
        return kl_loss

    def get_cost_matrix(self, sample_arr, gt_arr, cost_fn=None, gt_onehot=True, num_chunks=1, use_symmetric=False, ):
        """
        Calculate the pair wise cost matrix between sample_arr and gt_arr
        using the pair-wise cost function.
        """
        if gt_onehot:
            gt_arr = torch.nn.functional.one_hot(gt_arr.long(), self.C).permute(0, 1, 3, 2)

        N, M = sample_arr.shape[1], gt_arr.shape[1]

        # Fast matrix operation, need more space. An alternative implementation is to use iteration.
        sample_arr_repeat = sample_arr.expand(M, -1, -1, -1, -1).permute(1, 2, 0, 3, 4)
        if use_symmetric:
            gt_arr_repeat = sample_arr_repeat.transpose(1, 2)
        else:
            gt_arr_repeat = gt_arr.expand(N, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)

        sample_chunks = sample_arr_repeat.chunk(num_chunks, dim=1)
        gt_chunks = gt_arr_repeat.chunk(num_chunks, dim=1)

        costs = []
        for sample_chunk, gt_chunk in zip(sample_chunks, gt_chunks):
            if cost_fn == 'ce':
                loss_fn = torch.nn.LogSoftmax(dim=3)
                negative_logsoftmax = -loss_fn(sample_chunk)
                del sample_chunk
                cost = (negative_logsoftmax.mul(gt_chunk)).sum(-2)[:, :, :, 1:].mean(-1)  # exclude bg
                del gt_chunk
            elif cost_fn == 'iou':
                intersection = (sample_chunk * gt_chunk).sum(-1)
                union = (sample_chunk.sum(-1) + gt_chunk.sum(-1)) - intersection
                del sample_chunk, gt_chunk
                cost = 1.0 - ((intersection + 1) / (union + 1))[:, :, :, 1:].mean(-1)  # exclude bg', 1-iou
            costs.append(cost)
        cost = torch.cat(costs, dim=1)
        return cost

    def get_coupling_matrix(self, cost, prob_gt):
        """
        Solve the coupling matrix in our problem.
        """
        # greedy algorithm O( self.N)
        if self.gamma == 1:
            # moving the mass of each ground truth label to its nearest prediction
            P = (torch.nn.functional.one_hot(cost.argmin(-2),  self.N) * prob_gt.expand( self.N, -1, -1)
                  .permute(1, 2,0)).transpose(-1, -2)

        # A standard linear programming problem.
        else:
            # We adopt a greedy strategy which emPrically works fine. O( self.N self.Mlog self.N)
            P = torch.zeros_like(cost)
            sort_gt_ind = cost.min(-2).values.argsort()
            uniform = torch.ones_like(prob_gt) * (1.0 / self.M)

            for b in range(self.B):
                for i in sort_gt_ind[b]:

                    j_list = cost[b, :, i].argsort()  # for gt i, sort  its cost with each sample.
                    for j in j_list:
                        if P[b, j].sum() < self.gamma :
                            P[b, j, i] = min(self.gamma - P[b, j].sum(), uniform[b,i])
                            uniform[b, i] -=  P[b, j, i]
                            if uniform[b,i] == 0:
                                break
                        continue
            P =  P * (prob_gt * self.M).expand(self.N, -1, -1).permute(1, 0, 2)
        return P

    def get_soft_gradients(self, sample, prob, target_prob):
        """
        [In] sample S ∈ (B,N1,N2,C,HW), cost_matrix ∈ (B,N1*N2,M)
        [out] Grouped probability p'∈ (B,G), G is the number of groupings.
        [Algorithm]
        # Constraint 1:  IOU > threshold
        >> Constraint_IOU = (get_IOU_per_pair(S,S) > threshold) # (B,N,N)
        # Constraint 2: has the same nearest gt.
        >> Constraint_NGT = (get_nearest_gt(S) == get_nearest_gt(S).T)
        # Overal Constraint = constrain 1 & constraint 2
        >> Constraint = Constraint_IOU & Constraint_NGT (B,N,N)
        # Convert constraint to transition matrix.
        >> M = Normalized (Constraint)
        >> p' = BMM(p,M) #(M ∈ (B,N,G)))
        """
        B, N, C, HW = self.B, self.N, self.C, self.H * self.W
        K, S = self.K, self.S
        device = sample.device

        with torch.no_grad():
            sample = torch.nn.functional.one_hot(sample.argmax(2).detach(), self.C).permute(0, 1, -1, 2, 3)

            # Get constraint matrix (B,N1,N1)
            get_IOU_per_sample_pair = 1 - self.get_cost_matrix(sample.flatten(3), sample.flatten(3), cost_fn='iou',
                                                               gt_onehot=False, num_chunks=self.num_chunks, use_symmetric = True)
            Constraint = (get_IOU_per_sample_pair.view(B, K, S, K, S).mean([2, 4]) > self.G).triu(diagonal=1)

            # Transfer the bipartite matrix to grouping matrix.
            expert_id = torch.arange(K, device=device).repeat(B, 1)
            for pair in Constraint.nonzero():
                expert_id[pair[0], pair[2]] = expert_id[pair[0], pair[1]]
            grouping_matrix = torch.nn.functional.one_hot(expert_id, K).float().transpose(1, -1)

        prob = torch.bmm(grouping_matrix, prob.reshape(-1, K, S).sum(-1, keepdim=True)).squeeze(-1)
        target_prob = torch.bmm(grouping_matrix, target_prob.reshape(-1, K, S).sum(-1, keepdim=True)).squeeze(-1)

        return prob, target_prob
