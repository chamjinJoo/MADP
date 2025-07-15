import torch
import torch.nn.functional as F
from collections import defaultdict
from ACDrepo.ACD.model.utils import kl_latent, kl_categorical

def forward_pass_and_eval(
    args,
    encoder,
    decoder,
    data,
    rel_rec,
    rel_send,
    nagents,
    hard=False,
    data_encoder=None,
    data_decoder=None,
    relations=None,
    log_prior=None,
    epoch=None,
    batch_idx=None,
):
    """
    우리의 MultiAgentActorCritic/ACD_SCM 구조에 맞춘 forward+loss 계산 함수
    """
    losses = defaultdict(lambda: torch.zeros((), device=data.device))
    if data_encoder is None:
        data_encoder = data
    if data_decoder is None:
        data_decoder = data

    # --- ENCODER ---
    logits = encoder(data_encoder, rel_rec, rel_send)
    edges = torch.softmax(logits, dim=-1) if not hard else torch.argmax(logits, dim=-1)
    prob = torch.softmax(logits, dim=-1)

    # --- DECODER ---
    target = data_decoder[:, :, 1:, :]  # [batch, n_agents, T-1, obs_dim+1]
    output = decoder(data_decoder, edges, rel_rec, rel_send)

    # --- LOSSES ---
    losses['loss_nll'] = F.mse_loss(output, target)
    edge_types = 2
    edge_types_expanded = edge_types * 4  
    log_prior = torch.log(torch.ones(edge_types_expanded) / edge_types_expanded).to(prob.device)

    # shape: (edge_types_expanded,)
    predicted_atoms = nagents  # 또는 args.num_agents
    losses['loss_kl'] = kl_categorical(prob, log_prior, predicted_atoms)
    losses['loss'] = losses['loss_nll'] + losses['loss_kl']

    # --- 기타 metric ---
    losses['acc'] = (edges.argmax(-1) == (relations.argmax(-1) if relations is not None else edges.argmax(-1))).float().mean()
    losses['auroc'] = torch.tensor(0.0, device=data.device)  # 필요시 구현

    return losses, output, edges
