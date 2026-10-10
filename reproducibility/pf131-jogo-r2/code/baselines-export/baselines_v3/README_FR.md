# Baselines JOGO v3.0.0 — corrections de baselines_v2

**Les cinq méthodes, les huit tailles, les 30 runs et les huit budgets sont conservés.**
Le moteur de comptage/exécution est commun. Il ne faut pas mélanger ces nouveaux
résultats avec ceux produits par la v2. L'archive reçue reste intacte dans `archive/`.
Aucune figure ni aucun tableau du manuscrit n'est supprimé par ce paquet.

## Statut

Voir `validation/VALIDATION_REPORT.json` pour les tests effectivement exécutés.
Les essais joints sont des contrôles techniques à petits budgets, **pas les
résultats de la campagne scientifique**. Aucune performance, durée totale ou
probabilité de succès des cinq baselines n'est annoncée à partir de ces essais.
Le chemin CuPy reste optionnel et exige le précontrôle sur le GPU cible ;
il n'y a ni repli CPU silencieux ni installation automatique d'une roue CUDA.

## Installation et vérification (sans hypothèse sur le nom Conda)

Déposer `baselines_v3.tar.gz` dans `/data/projects/PF_GlobalOptimization`.
Ce bloc n'affecte pas les options de votre shell interactif. Utiliser Python >=3.11.
Une installation dédiée évite de modifier les versions des environnements HPF.

```bash
(
  cd /data/projects/PF_GlobalOptimization &&
  tar -xzf baselines_v3.tar.gz &&
  cd baselines_v3 &&
  python3 -m venv .venv &&
  . .venv/bin/activate &&
  python -m pip install -r requirements.txt &&
  python -m pytest -q &&
  python preflight.py --backend cpu --report validation/preflight_local_cpu.json &&
  unset LJ_USE_GPU &&
  DRY=1 BACKEND=cpu bash launch_baselines.sh
) && BASELINE_RC=0 || BASELINE_RC=$?
printf '\nVALIDATION_RC=%s -- terminal conserve\n' "$BASELINE_RC"
```

Le lancement à blanc doit annoncer **1200 jobs**. Il ne lance aucun run de recherche.
Le lanceur utilise l'interpréteur actif et n'impose pas `pfo_optim` ni un autre nom.
Ne pas extraire sur un répertoire v3 modifié : utiliser une installation neuve ou
archiver l'ancien répertoire avant remplacement. Les sorties de campagne sont séparées.

## Campagne CPU

```bash
(
  cd /data/projects/PF_GlobalOptimization/baselines_v3 &&
  . .venv/bin/activate &&
  unset LJ_USE_GPU &&
  JOBS=4 BACKEND=cpu OUTDIR=/data/projects/PF_GlobalOptimization/results_baselines_v3_cpu \
    bash launch_baselines.sh
) && BASELINE_RC=0 || BASELINE_RC=$?
printf '\nCAMPAGNE_RC=%s -- terminal conserve\n' "$BASELINE_RC"
```

Quatre runs indépendants en parallèle par défaut, un thread BLAS/OpenMP par processus.
Augmenter `JOBS` explicitement après contrôle des ressources ; cela change la concurrence
et doit rester stable pour les temps rapportés. Le précontrôle est exécuté automatiquement
avant la campagne. Tout échec d'un enfant donne un code de campagne non nul.
Les sorties standards sont dans `_logs/<horodatage>/`; chaque run sauvegarde les
coordonnées, énergie auditée, compteurs, options effectives, seeds et raison d'arrêt.

Le plafond global conservé est **36000 s par run**, contrôlé de façon coopérative
aux appels d'oracle ; il peut être dépassé du temps d'un appel/lot déjà engagé.
Ce n'est pas un échec après consommation du budget lorsque le temps coupe avant.
`JOB_SECONDS=0` désactive ce plafond, **change le protocole**, et requiert un autre
répertoire de résultats. Aucun restart temporel de 900 s n'est imposé par défaut.
Les fins naturelles d'algorithme sont suivies de redémarrages à seeds distincts sous le même
budget et le même meilleur point. Toutes les tentatives sont enregistrées.

## CuPy, uniquement après validation sur la machine cible

Installer CuPy adapté à votre pile CUDA dans cet environnement, puis :

```bash
(
  cd /data/projects/PF_GlobalOptimization/baselines_v3 &&
  . .venv/bin/activate &&
  python preflight.py --backend cupy --report validation/preflight_local_cupy.json &&
  JOBS=1 BACKEND=cupy OUTDIR=/data/projects/PF_GlobalOptimization/results_baselines_v3_cupy \
    bash launch_baselines.sh
) && BASELINE_RC=0 || BASELINE_RC=$?
printf '\nCAMPAGNE_GPU_RC=%s -- terminal conserve\n' "$BASELINE_RC"
```

Le GPU évalue les lots d'énergie en **float64**. Les tirages aléatoires, les règles des
méthodes et L-BFGS-B restent côté CPU ; BH-OJ est essentiellement scalaire CPU. Il ne
s'agit pas de cinq nouveaux optimiseurs GPU. Les différences de réduction flottante
peuvent modifier les trajectoires : aucune parité bit à bit CPU/GPU n'est affirmée.
`--allow-shared-gpu` est nécessaire pour plusieurs jobs sur un même GPU ; alors les
mesures de temps incluent cette contention. Ne pas agréger CPU et CuPy dans une cellule.

## Un seul essai / paramètres explicites

```bash
python run_one_baseline.py --method SMC --atoms 98 --run 1 \
  --eval-config baseline_eval_limits_v3.json --backend cpu \
  --outdir /data/projects/PF_GlobalOptimization/results_baselines_v3_cpu
```

Les cinq noms sont `BH-OJ GA PSO CLPSO SMC`. Les paramètres effectifs sont dans
`identity.method_params` et `method_params` du JSON. `--params fichier.json` accepte
un dictionnaire par méthode ; toute clé inconnue est refusée, pas filtrée silencieusement.
`beta_min` est appliqué par SMC. CLPSO utilise `w_min`, `w_max` et `inertia_schedule`.
Un `--max-evals` explicite sert aux pilotes techniques, est enregistré, et n'est pas
la campagne de référence aux budgets fournis. Ne pas mélanger ces pilotes et la campagne.

## Reprise

Relancer exactement la même commande : seuls les runs déjà terminés, audités, dont
l'empreinte code/configuration/environnement correspond, sont sautés. Une modification
de budget, options, backend, nombre de jobs (provenance temporelle), code ou versions
n'est pas ignorée. Utiliser un nouveau dossier pour un nouveau protocole.

`--force` archive les fichiers remplacés sous `_replaced/` ; il ne les efface pas.
Un run interrompu/erroné exige inspection et reprise explicite avec `--force`.
Il n'y a pas de reprise au milieu d'un état d'optimiseur : la nouvelle tentative
recommence avec le seed enregistré. Les verrous empêchent deux écritures concurrentes.

## Résumé et tableaux

Pendant la campagne :

```bash
python aggregate_baselines.py --results-dir /data/projects/PF_GlobalOptimization/results_baselines_v3_cpu --preview
```

Cela génère `summary_preview.*`, marqué **PREVIEW_NOT_FOR_MANUSCRIPT**.
À la fin :

```bash
python aggregate_baselines.py --results-dir /data/projects/PF_GlobalOptimization/results_baselines_v3_cpu
python generate_tables.py \
  --summary /data/projects/PF_GlobalOptimization/results_baselines_v3_cpu/_summary/summary.json \
  --output /data/projects/PF_GlobalOptimization/results_baselines_v3_cpu/_tables
```

Par défaut, le résumé final exige 1200 runs uniques valides et aucun arrêt censuré.
Il bloque sur doublon, fichier manquant, succès incohérent, erreur numérique, budget
incohérent ou mélange de protocoles. Il **recalcule** les énergies finales sur les
coordonnées. S'il reste des `time_limit`, lire le rapport : `--allow-censored` autorise
un export explicitement **VALIDATED_WITH_CENSORING**, avec les effectifs séparés.
Ce n'est pas une conversion de ces runs en échecs au budget d'évaluations épuisé.
Les écarts-types sont ceux de l'échantillon (`ddof=1`), les médianes sont usuelles.
Les cinq fichiers LaTeX fournissent les colonnes baselines ; aucune colonne HPF n'est
inventée. Les statistiques de temps/énergie incluent tous les runs valides, échecs compris.

## Lecture du code

- `lj_oracle.py` : potentiel, gradient, batch double et audit scalaire compensé.
- `evaluation.py` : budget global, meilleur point, dérivées séparées et arrêts.
- `baselines.py` : les cinq méthodes et L-BFGS-B réellement appelé via SciPy.
- `run_one_baseline.py` : un run, redémarrages, empreinte, JSON et journal des progrès.
- `campaign.py` / `launch_baselines.sh` : plan, précontrôle et exécution parallèle.
- `aggregate_baselines.py`, `record_schema.py`, `generate_tables.py` : validation et export.
- `PROTOCOL_BASELINES_V3.md` : définitions exactes à utiliser pour actualiser l'article.
- `CHANGELOG.md` : corrections et modifications explicites depuis v2.
- `validation/` : preuves techniques livrées, séparées des données de recherche.

## Limites à ne pas masquer

L'oracle par défaut correspond à la **formule du source PF131 fourni**, pas au potentiel
standard avec sigma=1 aux mêmes coordonnées. L'appariement exact aux populations initiales
réellement chargées par la campagne HPF nécessite ces fichiers. La tolérance à une cible
énergétique ne constitue pas un certificat d'optimalité globale. Les essais techniques
ne démontrent ni compétitivité scientifique ni parité complète avec la campagne C++.
