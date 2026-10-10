# Les cinq baselines sous nohup — orchestration R1

## Objet

BH-OJ et GA utilisent **baselines_v3_1**. PSO, CLPSO et SMC utilisent **baselines_v3**.
Le total nominal reste 5 méthodes × 8 tailles × 30 répétitions = 1200 résultats.
Ce kit ne lance ni HPF, ni IPOPT multistart, ni l'ablation. Il ne modifie aucun solveur.
Il règle l'exécution parallèle, le détachement du terminal, la reprise et la provenance.

## Concurrence

Une seule file équilibrée entre les cinq méthodes, avec **22 runs CPU simultanés au total** par défaut.
Un thread BLAS/OpenMP par run. Quand toutes les files restent alimentées, environ 4–5 places par méthode;
les places libres sont redistribuées. Le nombre 22 est une limite, pas 22 par méthode.
Les cellules de plus grand budget sont programmées d'abord, sans utiliser les résultats pour choisir les seeds.
Le script vérifie la topologie disponible. Un nombre dépassant les cœurs physiques exige `--allow-smt`.
Sur une machine partagée, les autres calculs restent à prendre en compte : ils ne sont jamais arrêtés par ce kit.

## Avant lancement

1. Décompresser ce kit dans `/data/projects/PF_GlobalOptimization/BASELINES_5_NOHUP_R1`.
2. Dans le terminal du lanceur actuel, `Ctrl+C`, puis attendre son retour au prompt.
   Les résultats achevés restent inchangés. Les runs effectivement interrompus repartiront
   depuis leur seed : aucun checkpoint algorithmique exact n'est inventé.
3. Ne pas relancer les anciens `launch_baselines.sh` en parallèle du nouveau superviseur.
   Le kit refuse de démarrer si un ancien lanceur/worker est encore actif; il n'en tue aucun.

Aucune nouvelle dépendance : on utilise `/data/projects/PF_GlobalOptimization/baselines_v3/.venv/bin/python`.
Aucune modification de NumPy/SciPy/Conda pendant les calculs. Pas de reboot : nohup ne survit pas au reboot.

## Lancement

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  PY="$BASE/baselines_v3/.venv/bin/python" &&
  test -x "$PY" &&
  cd "$BASE/BASELINES_5_NOHUP_R1" &&
  BASE="$BASE" PYTHON="$PY" JOBS=22 bash launch_nohup.sh
) && LANCEMENT_RC=0 || LANCEMENT_RC=$?
printf '\nLANCEMENT_RC=%s -- terminal conserve\n' "$LANCEMENT_RC"
```

Le lanceur exécute réellement `nohup ... </dev/null > journal 2>&1 &`.
Il attend une confirmation `READY`, et pas seulement l'existence d'un PID.
Après `NOHUP_READY`, fermer cette connexion SSH ne doit pas arrêter le superviseur.
Les workers sont des sous-processus à session séparée et héritent de la protection SIGHUP.
Le log et le PID sont indiqués à l'écran et enregistrés dans `last_log.txt` et `last_launcher.pid`.
Un lancement répété est refusé par verrou; ne pas multiplier les copies du kit.

## Résultats antérieurs et nouvelles phases

Lecture des résultats historiques dans :
- `results_baselines_v3_1_localfix_cpu` : **BH-OJ, GA seulement**;
- `results_baselines_v3_cpu` : **PSO, CLPSO, SMC seulement**.

Chaque résultat est contrôlé par le validateur de sa version, y compris réévaluation des coordonnées,
empreintes et identité du protocole. Les versions de code et tous les paramètres scientifiques doivent
correspondre aux bundles figés. Une différence de Python/NumPy/SciPy/OS bloque la reprise automatique.
Les champs de ressources d'exécution sont conservés comme cohortes distinctes, jamais falsifiés.

Les résultats valides achevés ne sont pas recalculés, y compris les arrêts temporels déclarés.
Les `interrupted` sont consignés séparément et restent à leur emplacement initial; leurs essais
reprennent à froid dans une nouvelle phase. Les vraies erreurs numériques, résultats illisibles,
empreintes divergentes ou doublons bloquent la préparation : il faut les examiner, pas les forcer.

Les nouveaux résultats sont écrits sous :
`results_baselines_nohup_R1/phase_.../cohort_..._j22/results/baselines_v3[_1]/LJ.../...`
Les anciennes sources et les anciens résultats ne sont ni copiés dans un faux protocole, ni rescellés,
ni écrasés. Des fichiers de verrou de coordination peuvent être créés dans les métadonnées des anciens dossiers.

Changer `JOBS` en plein dossier ancien aurait été refusé par l'identité complète du runner.
Ici chaque phase a son propre répertoire. `results_index.json` référence toutes les sorties retenues
avec chemins, empreintes, version et environnement. **Ne pas appliquer l'ancien agrégateur global
à un mélange brut v3/v3.1/cohortes.** La consolidation finale doit rester consciente de ces cohortes.

Les métriques de temps doivent être séparées ou accompagnées de l'allocation de ressources. Les
arrêts temporels ne deviennent pas des budgets d'évaluations épuisés. Les coûts des essais interrompus
sont des dépenses de migration, pas de nouvelles répétitions indépendantes.

## Suivi

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  "$BASE/baselines_v3/.venv/bin/python" \
    "$BASE/BASELINES_5_NOHUP_R1/supervise.py" status --base "$BASE"
) && SUIVI_RC=0 || SUIVI_RC=$?
printf '\nSUIVI_RC=%s -- terminal conserve\n' "$SUIVI_RC"
```

Le suivi affiche toutes les méthodes, les résultats retenus et actifs, les PID, une mesure CPU entre
observations et la dernière amélioration connue. Les événements ne sont pas un compteur continu :
leur numéro d'évaluation est une borne inférieure du travail déjà fait, pas le pourcentage exact courant.
Le snapshot est actualisé environ toutes les dix secondes. Un `tail -f` du journal peut être arrêté par
Ctrl+C sans arrêter le superviseur.

```bash
BASE=/data/projects/PF_GlobalOptimization
tail -n 60 -f "$(cat "$BASE/results_baselines_nohup_R1/last_log.txt")"
```

## Augmenter la concurrence si le débit est insuffisant

Sur la TRX50 à 24 cœurs physiques / 48 logiques, tester 32 places implique un partage SMT.
Il ne faut pas promettre 32 cœurs physiques ou une accélération linéaire. Conserver la même machine
et documenter la timeline de ressources; les limites temporelles peuvent être affectées par la contention.

```bash
(
  BASE=/data/projects/PF_GlobalOptimization &&
  "$BASE/baselines_v3/.venv/bin/python" \
    "$BASE/BASELINES_5_NOHUP_R1/supervise.py" set-jobs \
    --base "$BASE" --jobs 32 --allow-smt
) && PARALLELE_RC=0 || PARALLELE_RC=$?
printf '\nPARALLELE_RC=%s -- terminal conserve\n' "$PARALLELE_RC"
```

Le plafond change pour les prochains départs; aucun worker actif n'est tué.
Une baisse du plafond laisse finir le surplus sans en lancer de nouveaux.
Un nouvel épisode de ressources est inscrit dans le journal, et les nouvelles sorties vont dans
un nouveau sous-dossier de cohorte. Revenir à 22 s'effectue avec `--jobs 22` sans `--allow-smt`.

## Objectif d'une semaine

Le kit suit une cible de sept jours **sans interrompre les runs au septième jour** et sans modifier
les budgets d'évaluations, 30 répétitions, ou le plafond individuel existant de 36000 secondes.
22 places sur sept jours donnent 3696 heures-run; 1042 runs restants doivent alors durer en moyenne
moins de 3,55 heures pour tenir (hors pertes de capacité et autres travaux). C'est une condition de charge,
pas une prévision. 48 processeurs logiques ne doublent pas la puissance de 24 cœurs physiques.
Le suivi affiche le débit nécessaire et le débit observé, mais pas une date de fin artificiellement précise.
Si le débit mesuré ne suffit pas même après réallocation, il faudra plus de capacité ou plus de temps;
les statistiques scientifiques ne sont pas modifiées pour fabriquer une fin dans le délai.

## Arrêt et reprise

`supervise.py stop --base ...` demande l'arrêt propre **de ce superviseur seulement**.
Il transmet SIGTERM à ses workers, dont le runner enregistre les résultats interrompus.
Il n'envoie jamais de SIGKILL à des processus scientifiques. Attendre le retour/fin des PID avant reprise.
La même commande de lancement reprend les résultats valides de toutes les phases; elle ne reconstitue pas
l'état RNG interne d'un run interrompu. Aucune donnée n'est supprimée.

## Validation réellement effectuée

- 22 tests unitaires : sélection des cinq méthodes, 1200 identités, file équilibrée, verrous,
  exclusions, erreurs, budgets, concurrence, reprise, lecture partielle d'événements.
- 10 petits runs réels CPU (2 tailles × 5 méthodes), **61 évaluations chacun**, tous validés.
- Démarrage avec nohup et envoi de SIGHUP au superviseur pendant les calculs : poursuite réussie.
- Changement en cours de route de 2 à 3 workers : sorties des deux cohortes conservées.
- Relance : les dix résultats sont réutilisés, zéro nouvelle optimisation; empreintes inchangées.
- Refus d'un ancien worker simulé encore actif; rejet des résultats techniques comme données de production.

Environnement de validation : Python 3.13, NumPy 2.3.5, SciPy 1.17.0, CPU Linux.
**Aucune campagne scientifique à budget complet, aucun test de débit à 22/32 workers,
aucune mesure de vitesse sur la TRX50 et aucun test GPU n'ont été réalisés ici.**
Les anciens tests des solveurs ne sont pas revendiqués comme de nouveaux tests de ce kit.
