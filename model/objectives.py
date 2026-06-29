import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_constrative(
    image_features,
    text_features,
    sim_targets,
    logit_scale,
    logit_bias,
    use_sigmoid,
):
    """
    Compute contrastive loss for image-text pairs.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)
    
    #    # ==========================================
    # # 🕵️ DIAGNOSTIC DES CLONES SÉMANTIQUES (TEXTE vs TEXTE)
    # # ==========================================
    # with torch.no_grad(): 
    #     # 1. On compare LES TEXTES ENTRE EUX !
    #     text_sim_matrix = text_features @ text_features.t()
        
    #     # 2. On isole UNIQUEMENT les paires négatives (IDs différents)
    #     negatives_only = text_sim_matrix.clone()
    #     # sim_targets == 1 signifie que c'est la même personne, on cache ces valeurs
    #     negatives_only[sim_targets == 1] = -1.0 
        
    #     # 3. On cherche la description d'un INTRU qui ressemble le plus à la nôtre
    #     hardest_text_clones, _ = negatives_only.max(dim=1)

    #     print("\n" + "="*50)
    #     print("🚨 ANALYSE DES CLONES SÉMANTIQUES (TEXTE vs TEXTE)")
    #     print(f"Score moyen des pires clones (Intrus) : {hardest_text_clones.mean().item():.3f}")
    #     print(f"⚠️ Pire clone absolu (Même description, personne diff.) : {hardest_text_clones.max().item():.3f}")
        
    #     # Tolérance de 85% de ressemblance textuelle
    #     nb_faux_negatifs = (hardest_text_clones > 0.85).sum().item()
    #     print(f"🔥 Nombre d'intrus avec >85% de similarité textuelle : {nb_faux_negatifs} sur {hardest_text_clones.shape[0]}")
    #     print("="*50 + "\n")
    # # ==========================================
    
    logit_t2i = logit_scale * text_features @ image_features.t() + logit_bias
    logit_i2t = logit_scale * image_features @ text_features.t() + logit_bias

    if use_sigmoid:
        loglik = F.logsigmoid(logit_t2i * sim_targets)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
    else:
        loss_i2t = -torch.sum(F.log_softmax(logit_i2t, dim=1) * sim_targets, dim=1)
        loss_t2i = -torch.sum(F.log_softmax(logit_t2i, dim=1) * sim_targets, dim=1)
        loss = (loss_i2t.mean() + loss_t2i.mean()) / 2

    return loss


def compute_simclr(
    image_features_1,
    image_features_2,
    temperature=0.07,
):
    """
    Contrastive learning loss using SimCLR.
    """
    device = image_features_1.device
    batch_size = image_features_1.shape[0]

    image_features_1 = F.normalize(image_features_1, dim=-1, p=2)
    image_features_2 = F.normalize(image_features_2, dim=-1, p=2)

    labels = torch.arange(start=0, end=batch_size, device=device)

    sim_ab = (image_features_1 @ image_features_2.t()) / temperature
    sim_ba = sim_ab.t()

    mask = torch.where(F.one_hot(labels, batch_size) == 0, 0, float("-inf"))
    sim_aa = (image_features_1 @ image_features_1.t()) / temperature + mask
    sim_bb = (image_features_2 @ image_features_2.t()) / temperature + mask

    sim_a = torch.cat((sim_ab, sim_aa), dim=1)
    sim_b = torch.cat((sim_ba, sim_bb), dim=1)

    loss_a = F.cross_entropy(sim_a, labels)
    loss_b = F.cross_entropy(sim_b, labels)

    return (loss_a + loss_b) / 2


def compute_citc(
    image_features,
    text_features,
    logit_scale,
    logit_bias,
    inmodal_weight,
    intermodal_weight,
):
    """
    Compute cyclic image-text contrastive loss.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    sim_i2i = logit_scale * image_features @ image_features.t() + logit_bias
    sim_t2t = logit_scale * text_features @ text_features.t() + logit_bias

    sim_i2t = logit_scale * image_features @ text_features.t() + logit_bias
    sim_t2i = sim_i2t.t()

    inmodal_cyclic_loss = (sim_i2i - sim_t2t).square().mean() / (
        logit_scale * logit_scale
    )
    intermodal_cyclic_loss = (sim_i2t - sim_t2i).square().mean() / (
        logit_scale * logit_scale
    )

    return (
        inmodal_weight * inmodal_cyclic_loss
        + intermodal_weight * intermodal_cyclic_loss
    )


def compute_cross_modal_circle(image_features, text_features, pids, m=0.25, gamma=128):
    """
    Circle Loss between 2 modalities.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    sim_mat = torch.matmul(image_features, text_features.t())

    pids = pids.view(-1, 1)

    pos_mask = torch.eq(pids, pids.t()).float()
    neg_mask = 1 - pos_mask

    s_p = sim_mat[pos_mask.bool()]
    s_n = sim_mat[neg_mask.bool()]

    if s_p.numel() == 0 or s_n.numel() == 0:
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + m, min=0.0)
    alpha_n = torch.clamp_min(s_n.detach() + m, min=0.0)

    delta_p = 1 - m
    delta_n = m

    logit_p = - gamma * alpha_p * (s_p - delta_p)
    logit_n = gamma * alpha_n * (s_n - delta_n)

    soft_plus = nn.Softplus()
    loss = soft_plus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0))

    return loss

def compute_adaptive_cross_modal_circle(image_features, text_features, pids, m=0.25, gamma=128):
    """
    Circle Loss with Semantic Adaptive Margin for Text-Based Person Search.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    # 1. Matrice de similarité Image-Texte classique (ce que le modèle cherche à optimiser)
    sim_mat = torch.matmul(image_features, text_features.t())

    # ==========================================
    # 🌟 NOUVEAUTÉ : CALCUL DE LA MARGE DYNAMIQUE
    # ==========================================
    with torch.no_grad(): # Pas de gradient ici, la sémantique guide juste la Loss
        # Similarité entre toutes les descriptions textuelles du batch
        text_sim = torch.matmul(text_features, text_features.t())
        
        # On ignore les similarités négatives (textes opposés). 
        # On ne s'intéresse qu'à la ressemblance (de 0 à 1).
        text_sim_clamped = torch.clamp(text_sim, min=0.0, max=1.0)
        
        # Création de la matrice de marges.
        # Si similarité = 1 (textes très proches), marge_ij tend vers 0.
        # Si similarité = 0 (textes distincts), marge_ij reste à m (ex: 0.25).
        margin_matrix = m * (1.0 - text_sim_clamped)
    # ==========================================

    pids = pids.view(-1, 1)

    pos_mask = torch.eq(pids, pids.t()).float()
    neg_mask = 1 - pos_mask

    # Extraction des scores pour les paires Positives et Négatives
    s_p = sim_mat[pos_mask.bool()]
    s_n = sim_mat[neg_mask.bool()]
    
    # 🌟 EXTRACTION DES MARGES : On récupère exactement les marges adaptées à chaque paire négative
    m_n = margin_matrix[neg_mask.bool()]

    if s_p.numel() == 0 or s_n.numel() == 0:
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    # ------------------------------------------
    # APPLICATION DES MARGES
    # ------------------------------------------
    
    # Positifs : Marge FIXE. 
    # (Si c'est la même personne, on veut toujours que la distance soit minimale, sans exception)
    delta_p = 1 - m
    alpha_p = torch.clamp_min(-s_p.detach() + 1 + m, min=0.0)
    
    # Négatifs : Marge DYNAMIQUE (m_n est un tenseur, chaque paire a sa propre marge !)
    delta_n = m_n
    alpha_n = torch.clamp_min(s_n.detach() + m_n, min=0.0)

    # Calcul des logits
    logit_p = - gamma * alpha_p * (s_p - delta_p)
    logit_n = gamma * alpha_n * (s_n - delta_n)

    soft_plus = nn.Softplus()
    loss = soft_plus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0))

    return loss



def compute_asymmetric_circle_loss(
    image_features, 
    text_features, 
    pids, 
    m_i2t=0.25, 
    m_t2i=0.40, 
    gamma=128
    ):
    """
    Asymmetric Cross-Modal Circle Loss
    m_i2t : Marge pour la recherche Image -> Texte
    m_t2i : Marge pour la recherche Texte -> Image (souvent plus élevée)
    """
    device = image_features.device
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    # Matrice de similarité [Batch_size, Batch_size]
    sim_mat = torch.matmul(image_features, text_features.t())

    pids = pids.view(-1, 1)
    pos_mask = torch.eq(pids, pids.t()).float()
    neg_mask = 1.0 - pos_mask

    def get_directional_loss(sim_scores, pos_mask, neg_mask, m):
        """
        Calcule la Circle Loss directionnelle (Lignes = Ancres, Colonnes = Candidats)
        """
        # --- 1. Gestion des Positifs ---
        alpha_p = torch.clamp_min(-sim_scores.detach() + 1 + m, min=0.0)
        delta_p = 1 - m
        logit_p = -gamma * alpha_p * (sim_scores - delta_p)
        
        # On masque les négatifs en les mettant à -l'infini pour qu'ils disparaissent du logsumexp
        logit_p = torch.where(pos_mask.bool(), logit_p, torch.tensor(-1e9, device=device))
        lse_p = torch.logsumexp(logit_p, dim=1)

        # --- 2. Gestion des Négatifs ---
        alpha_n = torch.clamp_min(sim_scores.detach() + m, min=0.0)
        delta_n = m
        logit_n = gamma * alpha_n * (sim_scores - delta_n)
        
        # On masque les positifs en les mettant à -l'infini
        logit_n = torch.where(neg_mask.bool(), logit_n, torch.tensor(-1e9, device=device))
        lse_n = torch.logsumexp(logit_n, dim=1)

        # --- 3. Combinaison par ancre (Softplus) ---
        loss_per_anchor = F.softplus(lse_p + lse_n)
        
        # On retourne la moyenne sur tout le batch
        return loss_per_anchor.mean()

    # ==========================================
    # CALCUL ASYMÉTRIQUE
    # ==========================================
    
    # A. Sens Image -> Texte (Les ancres sont les images, on lit les lignes)
    loss_i2t = get_directional_loss(sim_mat, pos_mask, neg_mask, m=m_i2t)

    # B. Sens Texte -> Image (Les ancres sont les textes, on transpose pour lire les colonnes)
    loss_t2i = get_directional_loss(sim_mat.t(), pos_mask.t(), neg_mask.t(), m=m_t2i)

    # Moyenne finale
    loss = (loss_i2t + loss_t2i) / 2.0

    return loss

def compute_true_mining_circle_loss(image_features, text_features, pids, threshold=0.90, m=0.25, gamma=128):
    """
    Circle Loss avec True Mining (Élimination des faux négatifs sémantiques).
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    sim_mat = torch.matmul(image_features, text_features.t())

    # --- TRUE MINING ---
    with torch.no_grad():
        text_sim = torch.matmul(text_features, text_features.t())
        # On repère les intrus qui ont une description trop proche
        semantic_clones = (text_sim > threshold).float()

    pids = pids.view(-1, 1)
    
    # Masque des vrais positifs (Même identité)
    pos_mask = torch.eq(pids, pids.t()).float()
    
    # Masque des vrais négatifs :
    # Avant c'était (1 - pos_mask). 
    # Maintenant, on exclut aussi les clones sémantiques de la liste des ennemis.
    true_pos_and_clones = torch.max(pos_mask, semantic_clones)
    neg_mask = 1.0 - true_pos_and_clones

    # Extraction des scores (seuls les VRAIS ennemis sont dans s_n)
    s_p = sim_mat[pos_mask.bool()]
    s_n = sim_mat[neg_mask.bool()]

    if s_p.numel() == 0 or s_n.numel() == 0:
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    # Calcul classique de la Circle Loss
    alpha_p = torch.clamp_min(-s_p.detach() + 1 + m, min=0.0)
    alpha_n = torch.clamp_min(s_n.detach() + m, min=0.0)

    delta_p = 1 - m
    delta_n = m

    logit_p = - gamma * alpha_p * (s_p - delta_p)
    logit_n = gamma * alpha_n * (s_n - delta_n)

    soft_plus = nn.Softplus()
    loss = soft_plus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0))

    return loss


def compute_symmetric_mining_circle_loss(
    image_features, 
    text_features, 
    pids, 
    m=0.25, 
    gamma=128,
    tm_threshold=0.90,   # Seuil du True Mining (Immunité)
    adapt_scale=0.15     # Force de réduction de la marge pour les cas difficiles
):
    """
    Circle Loss Globale (Symétrique) avec filtrage des clones et marge adaptative.
    """
    # --- 1. Normalisation ---
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    # Matrice de similarité Cross-Modal (Image vs Texte)
    sim_mat = torch.matmul(image_features, text_features.t())

    # --- 2. L'Arbitre (Texte vs Texte) ---
    with torch.no_grad():
        text_sim = torch.matmul(text_features, text_features.t())
        
        # A. True Mining : Repérer les clones parfaits pour les masquer
        semantic_clones = (text_sim > tm_threshold).float()

        # B. Adaptive Margin : Calcul de la matrice de marges dynamiques
        # Marge de base (m) - une fraction de la ressemblance textuelle
        adaptive_margin_matrix = m - (adapt_scale * text_sim)
        
        # Sécurité : on ne descend jamais sous une marge stricte minimale (ex: 0.05)
        adaptive_margin_matrix = torch.clamp_min(adaptive_margin_matrix, min=0.05)

    # --- 3. Masques ---
    pids = pids.view(-1, 1)
    pos_mask = torch.eq(pids, pids.t()).float()
    
    # Les "vrais" ennemis (Ni même identité, ni clone textuel)
    true_pos_and_clones = torch.max(pos_mask, semantic_clones)
    neg_mask = 1.0 - true_pos_and_clones

    # --- 4. Extraction des scores ---
    # On aplatit les matrices pour ne garder que les valeurs qui nous intéressent
    s_p = sim_mat[pos_mask.bool()]
    s_n = sim_mat[neg_mask.bool()]

    # MAGIE ICI : On extrait la marge dynamique correspondante EXACTE pour chaque négatif !
    m_n = adaptive_margin_matrix[neg_mask.bool()]

    # Sécurité si un batch est "parfait" (aucun positif ou aucun négatif)
    if s_p.numel() == 0 or s_n.numel() == 0:
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    # --- 5. Calcul de la Loss ---
    # Positifs : On garde la marge de base stricte (m)
    alpha_p = torch.clamp_min(-s_p.detach() + 1 + m, min=0.0)
    delta_p = 1 - m
    logit_p = -gamma * alpha_p * (s_p - delta_p)

    # Négatifs : On utilise la marge dynamique sur-mesure (m_n)
    alpha_n = torch.clamp_min(s_n.detach() + m_n, min=0.0)
    delta_n = m_n
    logit_n = gamma * alpha_n * (s_n - delta_n)

    # Fusion globale (Formule classique de l'ICIP)
    soft_plus = nn.Softplus()
    loss = soft_plus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0))

    return loss


def compute_ultimate_circle_loss(
    image_features, 
    text_features, 
    pids, 
    m_i2t=0.25, 
    m_t2i=0.40, 
    gamma=128,
    tm_threshold=0.90,   # Seuil pour le True Mining (Faux Négatifs)
    adapt_scale=0.15     # Force de l'Adaptive Margin
):
    """
    La version ultime combinant : Asymétrie, True Mining et Adaptive Margin.
    """
    device = image_features.device
    
    # --- 1. Normalisation ---
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    # Matrice de similarité Cross-Modal (Celle qu'on optimise)
    sim_mat = torch.matmul(image_features, text_features.t())

    # --- 2. L'Arbitre (Intra-Modal Texte) ---
    with torch.no_grad(): # VITAL pour ne pas saturer la VRAM !
        text_sim = torch.matmul(text_features, text_features.t())
        
        # True Mining : On repère les clones sémantiques absolus
        semantic_clones = (text_sim > tm_threshold).float()

    # --- 3. Masques de base ---
    pids = pids.view(-1, 1)
    pos_mask = torch.eq(pids, pids.t()).float()
    
    # Les "vrais" ennemis sont ceux qui n'ont ni le même ID, ni une description quasi-identique
    true_pos_and_clones = torch.max(pos_mask, semantic_clones)
    neg_mask = 1.0 - true_pos_and_clones

    # --- 4. Moteur de calcul directionnel ---
    def get_directional_loss(sim_scores, pos_mask, neg_mask, base_m, text_sim_matrix):
        """
        Calcule la perte avec une Marge Adaptative sous forme de matrice.
        """
        # --- Positifs ---
        # On exige toujours la même rigueur (m_base) pour rapprocher les vrais positifs
        alpha_p = torch.clamp_min(-sim_scores.detach() + 1 + base_m, min=0.0)
        delta_p = 1 - base_m
        logit_p = -gamma * alpha_p * (sim_scores - delta_p)
        
        logit_p = torch.where(pos_mask.bool(), logit_p, torch.tensor(-1e9, device=device))
        lse_p = torch.logsumexp(logit_p, dim=1)

        # --- Négatifs (ADAPTIVE MARGIN) ---
        # On adoucit la marge si le texte de l'intrus ressemble au nôtre
        # (sans jamais descendre en dessous d'une marge minimale de sécurité, ex: 0.05)
        adaptive_m = base_m - (adapt_scale * text_sim_matrix)
        adaptive_m = torch.clamp_min(adaptive_m, min=0.05)

        # On applique cette matrice de marges dynamiques
        alpha_n = torch.clamp_min(sim_scores.detach() + adaptive_m, min=0.0)
        delta_n = adaptive_m
        
        logit_n = gamma * alpha_n * (sim_scores - delta_n)
        
        logit_n = torch.where(neg_mask.bool(), logit_n, torch.tensor(-1e9, device=device))
        lse_n = torch.logsumexp(logit_n, dim=1)

        # Softplus par ancre
        return F.softplus(lse_p + lse_n).mean()

    # ==========================================
    # CALCUL ASYMÉTRIQUE FINAL
    # ==========================================
    
    # Sens Image -> Texte (Ancres = Images)
    loss_i2t = get_directional_loss(
        sim_scores=sim_mat, 
        pos_mask=pos_mask, 
        neg_mask=neg_mask, 
        base_m=m_i2t, 
        text_sim_matrix=text_sim
    )

    # Sens Texte -> Image (Ancres = Textes, tout est transposé !)
    loss_t2i = get_directional_loss(
        sim_scores=sim_mat.t(), 
        pos_mask=pos_mask.t(), 
        neg_mask=neg_mask.t(), 
        base_m=m_t2i, 
        text_sim_matrix=text_sim.t()
    )

    return (loss_i2t + loss_t2i) / 2.0