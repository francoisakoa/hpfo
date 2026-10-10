# LJ75 / LJ102 — refine-all — R2

Kit d'ablation et sensibilité suivant les grilles des tableaux 7–10, avec les deux
amendements demandés : LJ102 remplace LJ98 et toutes les particules courantes
sont raffinées dans la boucle PF. Lire `PROTOCOLE.md`.

## Préparation sans compilation

Décompresser ce kit dans un nouveau dossier, puis :

```bash
bash prepare_from_r1.sh /data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ98_R1
```

Cette commande active `pf_cpp_110`, vérifie l'identité des sources et des binaires,
copie les binaires gelés et lance les tests du plan et du gestionnaire d'arrêt.
Seules les commandes `--version` / `--study-schema` sont exécutées sur le solveur :
pas de résolution scientifique. Aucun fichier de R1 n'est remplacé.

## Arrêter l'ancienne campagne

```bash
python3 stop_campaign.py --out /data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ98_R1 --stop
```

Attendre `ANCIENNE_CAMPAGNE_ARRETEE=OUI`. L'arrêt est ciblé, sans PID ancien supposé
encore valide. Les fichiers des runs terminés et interrompus restent conservés.
Le message d'interruption / code 130 de l'ancien lanceur est attendu.

## Plan et lancement

```bash
OUT=/data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ102_REFINE_ALL_R2 bash run_study.sh plan
OUT=/data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ102_REFINE_ALL_R2 bash run_study.sh run
```

820 calculs distincts, 860 observations. 30 répétitions par ablation et 10 par valeur
de sensibilité, un calcul actif. Nouveau contrôle calculé ; pas de recyclage de R1.
Le mode `status` ne fait que lire le rapport actuel, sans le réécrire.

```bash
OUT=/data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ102_REFINE_ALL_R2 bash run_study.sh status
OUT=/data/projects/PF_GlobalOptimization/PF131_ETUDES_LJ75_LJ102_REFINE_ALL_R2 bash run_study.sh bundle
```

Bundle automatique en fin de campagne. `bundle --allow-partial` permet un dossier
intermédiaire explicitement partiel, seulement lorsque la campagne n'est pas active.

## Réutilisation et repli

Le C++ est inchangé par rapport au kit instrumenté R1 : seule la configuration
change. En cas de divergence de source ou de binaire, la réutilisation est refusée.
Le script `build.sh` reste disponible pour une compilation explicite dans cette
nouvelle installation, mais n'est pas nécessaire dans le chemin normal ci-dessus.
Ne pas recompiler l'installation historique ni mélanger les résultats.

## Validation

Voir `TEST_REPORT.txt` et `evidence/`. Les validations du lanceur utilisent des
fixtures synthétiques. Elles ne sont pas des résultats LJ ni une estimation de
performance. Les archives scientifiques doivent être créées par les calculs locaux.
