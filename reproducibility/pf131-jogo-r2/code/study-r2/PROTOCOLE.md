# PF131 — Ablation et sensibilité LJ75 / LJ102, refine-all — R2

## Amendement demandé après le lancement de R1

L'utilisateur remplace **LJ98 par LJ102** et **le raffinement top-5 par
`--refine-all` dans la boucle PF**. LJ75 est conservé. La campagne R1 top-5
LJ75/LJ98 et ses tentatives ne sont ni supprimées ni utilisées dans les nouvelles
statistiques. Le nouveau dossier est `PF131_ETUDES_LJ75_LJ102_REFINE_ALL_R2`.

Le manuscrit fourni (`PFO_LJ_JOGO_SN_R2(1).pdf`, sections 4.2–4.3, tableaux 7–10)
étudiait LJ75/LJ98 ; sa section 3.7.1 et son tableau 1 indiquent top-5. R2 garde
les **configurations, les grilles et les répétitions**, mais change les tailles
et le raffinement. Il s'agit d'un **protocole expérimental amendé**, pas de la
reproduction à l'identique des expériences ou des pourcentages du manuscrit.
Aucune conclusion nouvelle de R2 ne doit être attribuée à LJ98.

## Ce qui ne change pas

Le C++ est **identique octet pour octet à la copie instrumentée du kit R1**.
Il sait déjà traiter LJ102 et `--refine-all`. Les changements opérationnels sont
limités au lanceur Python, à son plan, aux noms de sortie et à l'arrêt ciblé de R1.
Le binaire conserve donc la version `1.3.1-article-study-r1` et l'interface
`PF131_ARTICLE_STUDY_R1` ; le plan R2 a l'identifiant distinct
`PF131_LJ75_LJ102_REFINE_ALL_R2`.

La préparation réutilise les exécutables gelés de R1 après vérification de leur
SHA-256, de la source figée et de l'interface. **Aucune recompilation et aucune
installation de dépendance n'est nécessaire par cette voie.** Une divergence
fait arrêter la préparation, sans modifier l'ancienne installation.

## Raffinement effectif

Pour `full`, les ablations conservant la recherche locale et toutes les lignes
de sensibilité, le lanceur passe **`--refine-all`**. Dans le C++, cette option a
priorité sur `refine_top_k` pour la sélection effectuée à chaque itération :
**toutes les particules courantes sont raffinées**. Le nombre de particules
courant peut décroître et recevoir des injections, comme dans PF131 R1.

`--refine-top-k 5` reste également enregistré pour préserver la sémantique native
PF131 : un **ultime raffinement des cinq meilleurs** reste possible après la
boucle si la cible n'a pas été atteinte. Cela ne limite PAS les raffinements de
la boucle à cinq. R2 ne change pas discrètement cette phase finale du C++.

Les variantes `no_local` et `pf_only` ont `local_method=none`,
`refine_top_k=0`, `refine_all=false`, `final_refinement=false` : **aucun appel
local**, ni dans la boucle ni à la fin. Cette exception est nécessaire à l'ablation.

## Table 7 — 30 runs par configuration et par taille

|Configuration|Intervention par rapport au nouveau contrôle refine-all|
|---|---|
|full|Contrôle complet de cette étude, raffinement de toutes les particules courantes|
|no_ipf|Transport IPF désactivé, ordre par clusters et jitter conservés|
|no_sharing|Poids sharing nul et calcul omis|
|no_novelty|Poids novelty nul et calcul omis|
|no_kicks|Kick et adaptation associée du rayon désactivés|
|no_island_exchange|Échange périodique désactivé ; injection périodique conservée|
|no_local|Aucun raffinement, y compris final|
|no_adaptive_annealing|Beta constant à 1e-4 par défaut|
|pf_local_only|Sans IPF, sharing, novelty, kicks, échanges/injections d'îles, injection de diversité et réinitialisation ESS ; recherche locale refine-all conservée|
|pf_only|Même configuration réduite, sans aucune recherche locale|

Les conventions de beta constant et des configurations réduites, insuffisamment
précisées dans le manuscrit, restent les conventions déclarées de R1. Elles ne
sont pas présentées comme des détails historiques retrouvés.

## Tables 8–10 — 10 runs par valeur et par taille

* Np initial : 100, 200, 300, 400, 500.
* Beta initial : 1e-5, 1e-4, 1e-3, 1e-2.
* Alpha initial : 0.95, 0.98, 0.99, 1.00 ; alpha final : 0.05.

Une seule valeur est modifiée à la fois. Les lignes beta=1e-4 et alpha=0.98
réutilisent les runs 1–10 du **nouveau** contrôle `full`, sans choisir les meilleurs.
Total : **820 calculs distincts (410 par taille), 860 observations de tables**.
Les 600 calculs de la table 7 comprennent 120 calculs sans recherche locale.
Les autres 700 calculs utilisent refine-all.

## Paramètres communs

- LJ75 : Np par défaut 281, cible -397.492331.
- LJ102 : Np par défaut 362, cible -569.363653.
- Itérations PF 1000 ; plafond de temps mural 36000 s par run ; pas de plafond
  d'évaluations (`max_evals=0`) ; arrêt pour stagnation désactivé.
- Ipopt/MUMPS, tolérance locale 1e-8, maximum local 5000.
- Un run actif, 10 threads batch, 8 workers locaux, BLAS à un thread.
- Succès : objectif <= cible + 1e-4 ; ce n'est pas un certificat de stationnarité.
- Graines base 2024, MD5 et offsets 0/1000 inchangés.
- Populations natives gelées par (N,Np,run), réutilisées entre configurations
  comparables. Elles changent entre répétitions, contrairement à la vieille
  campagne principale à population unique par taille.
- Flux PF131 inchangé : la population est générée depuis seed_pf ; seed_init
  décrit x0, non utilisé dans la boucle.
- Alpha linéaire à extrémités exactes de R1, de alpha0 à 0.05 ; pas de retour à la
  formule historique qui commençait à 1.03.

Les autres différences connues entre PF131 instrumenté et les équations du
manuscrit (recuit, kicks, sharing, novelty, échanges, énergie adoucie,
contraintes et backend CPU/MUMPS) restent celles décrites dans
`reference/PROTOCOLE_R1_HISTORIQUE.md`. Elles ne sont pas corrigées implicitement.

## Mesures, reprise et conservation

Temps mural interne : `pf.json.seconds`. Temps CLI et dates UTC :
`EXECUTION.json`. Compteurs f PF/local conservés et contrôlés. Jamais de division
par le nombre de workers. Les résultats sont validés avant création de DONE.json.
Les résultats top-5 antérieurs ne sont pas importés dans cette campagne.

`stop_campaign.py` ne cible que le lanceur `study.py run --out <R1>` et les
processus de sa campagne. SIGTERM laisse le gestionnaire R1 consigner l'arrêt et
fermer son groupe solveur/worker. Les fichiers restent en place. Le script ne
réutilise pas le PID historique 314360 et n'utilise aucun `pkill` général.

L'arrêt demandé peut interrompre un calcul en cours : il reste une tentative
interrompue, pas un échec scientifique. Aucun nouveau résultat n'est inventé.
La campagne R2 refuse de démarrer si R1 est encore détectée comme active.

L'archive finale inclut le plan amendé, sources, exécutables utilisés, populations,
JSON complets, XYZ, historiques CSV, logs, commandes, configurations, dates,
rapports et manifestes SHA-256. Aucun calcul n'est relancé par la commande bundle.
Une archive avant complétude est explicitement nommée PARTIEL.

## Limites des conclusions

Refine-all augmente le nombre de résolutions locales par itération et peut
changer les probabilités de succès et les nombres d'itérations. Remplacer LJ98
par LJ102 ne fournit aucune garantie de durée totale ou de succès sur les variantes.
Les résultats de cette étude devront être interprétés uniquement pour les tailles,
paramètres et budgets effectivement utilisés ; pas comme une ablation sur LJ98.
