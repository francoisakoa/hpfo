# HPFO_TRX_REVIEW_R1_FIX1 — correction du lecteur et reprise

Le défaut du R1 est l'absence du champ `n_func_evals` dans le lecteur de
résultats baseline. Les champs `method`, `n_atoms`, `run_id`, `energy`,
`wall_time`, `stop_reason`, `target_energy` et `target_tolerance` restent
lus dans les enregistrements d'origine. Les identités numériques peuvent être
des entiers ou des chaînes de chiffres (sans déduction à partir des chemins).

La correction n'écrit pas dans les sources C++, les campagnes ou les exports.
Les JSON sont lus tels quels. Le champ de durée retenu reste `wall_time`, pas
`elapsed_including_audit`. Le succès est recalculé à partir de la cible et de la
tolérance enregistrées, pas supposé à partir d'un statut générique de solveur.

## Reprendre le dossier temporaire identifié sur la TRX

```bash
D=/data/projects/PF_GlobalOptimization
python3 -B create_hpfo.py \
  --stage "$D/HPFO_PF131_REVIEW_f97xexav" \
  --baselines-dir "$D/EXPORT_BASELINES_COMPACT_20261010T141633Z_y3yzr2g9" \
  --baselines-index "$D/results_baselines_nohup_R1/results_index.json" \
  --resume-work "$D/.hpfo-preparation-r479kt8g" \
  --destination "$D/hpfo"
```

La reprise vérifie les empreintes du dossier initial, des copies déjà présentes
et des archives. Elle relit et contrôle les fichiers déjà extraits mais ne
recopie ni ne recompresse les grosses archives. Elle reconstruit les tableaux
sans optimisation, garde le diagnostic d'origine dans `.local/`, puis renomme
le dossier temporaire en `hpfo` uniquement après les contrôles finaux.
Un conflit de contenu, un doublon ou un résultat manquant arrête la procédure.
Un dossier `hpfo` déjà existant n'est jamais écrasé.

Une création neuve est possible en omettant `--resume-work`, mais ce n'est pas
le chemin recommandé pour la préparation déjà réalisée sur cette TRX.

## Après réussite

```bash
cd /data/projects/PF_GlobalOptimization/hpfo
python3 -B scripts/verify_package.py --assets
```

Les sources du solveur sont dans
`reproducibility/pf131-jogo-r2/code/main-pf131/`.
Leur compilation et les tests C++ sont une étape distincte, documentée dans
`docs/BUILD_AND_REPLAY.md`. Ils ne sont pas exécutés par l'assemblage.
Aucune publication GitHub n'est effectuée par ce kit.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
```

`tests/reference/HPFO_TRX_REVIEW_R1.zip` conserve le préparateur précédent
à l'identique pour reproduire le défaut, pas une version alternative du
solveur. Tous les résultats utilisés par les tests sont synthétiques. Les noms
de champs sont ceux du diagnostic fourni ; les valeurs expérimentales réelles
ne sont ni simulées ni présentées comme vérifiées dans ces tests.
