# Baselines v3.1.0 — réparation ciblée du raffinement BH-OJ / GA

## Décision opérationnelle

1. Interrompre le lanceur v3 par Ctrl+C et attendre le retour au prompt. Il transmet SIGINT aux enfants, puis conserve les sorties terminées et les traces disponibles.
2. **Ne supprimer aucun résultat v3, et ne modifier aucun fichier v3 en place.**
3. Extraire ce paquet dans le nouveau répertoire `baselines_v3_1`.
4. Tester la réparation sur la machine. Les tests incluent le cas réel enregistré de faux arrêt local.
5. Exécuter d'abord un run LJ38 par méthode affectée (BH-OJ / GA) dans un dossier de qualification séparé. Inspecter les diagnostics locaux avant la relance des 60 runs LJ38. Le résultat global n'est pas un critère d'acceptation des tests techniques.
6. Conserver les résultats PSO / CLPSO / SMC. Reprendre leurs seuls runs manquants avec le code v3 original.

Les lignes `OK ... rc=0` ne sont pas des succès d'optimisation. Elles signifient que le processus a rendu une sortie sans erreur fatale selon le runner. La cible atteinte est dans `success` / `stop_reason`.

## Installation (environnement v3 existant)

Déposer `baselines_v3_1_LOCALFIX_R1.tar.gz` dans `/data/projects/PF_GlobalOptimization`.

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  PY="$BASE/baselines_v3/.venv/bin/python" &&
  test -x "$PY" &&
  tar -xzf "$BASE/baselines_v3_1_LOCALFIX_R1.tar.gz" -C "$BASE" &&
  cd "$BASE/baselines_v3_1" &&
  unset LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS LJ_USE_GPU &&
  export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 &&
  "$PY" tools/verify_bundle.py &&
  "$PY" -m pytest -q &&
  "$PY" preflight.py --backend cpu --report validation/preflight_workstation.json
) && FIX_RC=0 || FIX_RC=$?
printf '\nVALIDATION_RC=%s -- terminal conserve\n' "$FIX_RC"
```

Aucun nouvel environnement Conda, aucune mise à niveau de SciPy, aucun remplacement du potentiel ou des budgets.

## Qualification sur la workstation : deux runs, pas 1 200

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  cd "$BASE/baselines_v3_1" &&
  PY="$BASE/baselines_v3/.venv/bin/python" &&
  test -x "$PY" &&
  unset LJ_USE_GPU &&
  PYTHON="$PY" JOBS=2 BACKEND=cpu ATOMS="38" METHODS="BH-OJ GA" RUNS=1 \
    OUTDIR="$BASE/qualification_baselines_v3_1_localfix" bash launch_baselines.sh &&
  "$PY" tools/local_summary.py \
    --results-dir "$BASE/qualification_baselines_v3_1_localfix"
) && FIX_RC=0 || FIX_RC=$?
printf '\nQUALIFICATION_RC=%s -- terminal conserve\n' "$FIX_RC"
```

Ce dossier est distinct des données de production. Les deux tests inclus dans le bundle ont épuisé leur budget sans succès global. BH-OJ conserve 29 sorties locales à grand résidu (dont 25 au plafond local de 1 200 itérations et quatre départs avec collisions exactes); elles sont signalées comme non résolues, jamais déclarées stationnaires. GA n'en conserve aucune dans son pilote. Le correctif répare le cas enregistré et la fausse classification; il ne garantit pas la résolution de chaque sous-problème local. Voir `validation/VALIDATION_REPORT.md`.

## Relance LJ38 : seulement les deux méthodes affectées

À exécuter après réussite du précontrôle, sans autre campagne concurrente pour garder le même nombre de workers.

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  cd "$BASE/baselines_v3_1" &&
  PY="$BASE/baselines_v3/.venv/bin/python" &&
  test -x "$PY" &&
  unset LJ_USE_GPU &&
  PYTHON="$PY" JOBS=4 BACKEND=cpu ATOMS="38" METHODS="BH-OJ GA" RUNS=30 \
    OUTDIR="$BASE/results_baselines_v3_1_localfix_cpu" bash launch_baselines.sh
) && FIX_RC=0 || FIX_RC=$?
printf '\nRELANCE_LJ38_RC=%s -- terminal conserve\n' "$FIX_RC"
```

Ne pas utiliser `--force` sur les anciens résultats. La relance est dans un autre répertoire. Une nouvelle trajectoire n'est pas une continuation de l'état interne d'un run interrompu.

## Diagnostic de la nouvelle version

Ce bloc est purement en lecture, hors écriture du résumé provisoire.

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  cd "$BASE/baselines_v3_1" &&
  PY="$BASE/baselines_v3/.venv/bin/python" &&
  "$PY" aggregate_baselines.py \
    --results-dir "$BASE/results_baselines_v3_1_localfix_cpu" \
    --methods BH-OJ GA --preview &&
  "$PY" tools/local_summary.py --results-dir "$BASE/results_baselines_v3_1_localfix_cpu"
) && FIX_RC=0 || FIX_RC=$?
printf '\nDIAGNOSTIC_LOCAL_RC=%s -- terminal conserve\n' "$FIX_RC"
```

`stationary` signifie que le résidu projeté **non redimensionné** vérifie `local_gtol`. `function_tolerance` n'affirme pas cela. `unresolved_nonstationary` identifie une sortie encore à grand résidu malgré les reprises bornées. Les détails figurent dans `local_anomaly_examples`; ils ne sont ni cachés, ni transformés en succès global.

Le nombre `zero_displacement_nonstationary` peut inclure des élites déjà quasi stationnaires à la précision fonctionnelle. Le sous-compteur `zero_displacement_large_residual` distingue les cas de la famille défectueuse reproduite. L'indicateur `unresolved_large_residual` doit être inspecté avec le message et le plafond d'itérations, avant d'étendre la relance aux sept autres tailles. Une limite locale atteinte n'est pas la même chose qu'un faux succès; aucun seuil arbitraire de succès global n'est exigé. Les pilotes et leurs limites sont donnés dans le rapport.

## Reprendre les trois méthodes non affectées sans les recalculer

Après retour au prompt du lanceur interrompu :

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  PY="$BASE/baselines_v3/.venv/bin/python" &&
  "$PY" "$BASE/baselines_v3_1/tools/prepare_resume_v3.py" \
    --results-dir "$BASE/results_baselines_v3_cpu" --apply &&
  cd "$BASE/baselines_v3" &&
  unset LJ_USE_GPU &&
  PYTHON="$PY" JOBS=4 BACKEND=cpu METHODS="PSO CLPSO SMC" \
    OUTDIR="$BASE/results_baselines_v3_cpu" bash launch_baselines.sh
) && RESUME_RC=0 || RESUME_RC=$?
printf '\nREPRISE_NONLOCALE_RC=%s -- terminal conserve\n' "$RESUME_RC"
```

Le préparateur refuse une campagne ou un run encore verrouillés. Il archive seulement les fichiers de runs PSO/CLPSO/SMC portant `interrupted`, avec empreintes et journal. Les runs achevés restent inchangés et sont sautés par le runner original après vérification d'identité. Il ne traite pas silencieusement une véritable erreur numérique. Ne pas lancer cette reprise en même temps que les quatre nouveaux workers BH-OJ/GA si l'on veut garder une concurrence de quatre pour les temps rapportés.

## Extension après inspection de LJ38

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  cd "$BASE/baselines_v3_1" &&
  PY="$BASE/baselines_v3/.venv/bin/python" &&
  unset LJ_USE_GPU &&
  PYTHON="$PY" JOBS=4 BACKEND=cpu ATOMS="102 103 104 75 76 77 98" METHODS="BH-OJ GA" RUNS=30 \
    OUTDIR="$BASE/results_baselines_v3_1_localfix_cpu" bash launch_baselines.sh
) && FIX_RC=0 || FIX_RC=$?
printf '\nRELANCE_AUTRES_TAILLES_RC=%s -- terminal conserve\n' "$FIX_RC"
```

Les résultats des deux versions doivent rester identifiés par méthode. Ne pas fusionner les dossiers ni modifier leurs empreintes pour contourner le refus de campagnes mélangées. La consolidation finale doit garder la provenance v3.0 pour les méthodes inchangées et v3.1 pour BH-OJ/GA.

## Portée

La réparation est numérique et locale. Elle ne garantit pas le minimum global et ne promet pas que BH-OJ ou GA vont atteindre 100 % de succès. Les tests et pilotes techniques ne remplacent pas les 30 runs de production. Les figures et tableaux du manuscrit ne sont pas modifiés par ce paquet.
