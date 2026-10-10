# Validation PF C++ 1.3.1 — 16 septembre 2026

## Changements vérifiés

1. Ordonnancement METIS explicite pour LDL et MUMPS natifs, réutilisation de
   l'analyse symbolique et traçabilité dans les workers/JSON.
2. Conservation par particule du meilleur candidat local admissible évalué,
   indépendamment du statut de convergence. En cas d'interruption, restitution
   du candidat et de son objectif mémorisé avant propagation de l'arrêt.
3. Comptage des améliorations trouvées dans les appels interrompus.

## Configurations et preuves

| Configuration | Tests | Journal |
|---|---:|---|
| GCC 13.3, native, OpenMP, Ipopt 3.14.11, MUMPS 5.4.1, CoinMETIS API 4 | 11/11 | `ctest_native_final.log` |
| Portable, OpenMP, METIS 5.1.0 à indices 64 bits, sans Ipopt/MUMPS | 9/9 | `ctest_metis5_final.log` |
| Portable, OpenMP, sans METIS/Ipopt/MUMPS, sélection pertinente | 4/4 | `ctest_no_metis_final.log` |

Les bibliothèques natives proviennent de l'installation CasADi de cet
environnement et utilisent des en-têtes correspondants. Le METIS 5 du second
build a été compilé depuis les sources embarquées dans PyMetis 2025.2.2, pour
valider aussi cette API/largeur d'indices. Les dépendances et binaires ne sont
pas redistribués dans ce bundle. `configuration_*.txt` et `version_*.txt`
consignent les réglages. La configuration exacte de la machine TRX50 de
l'utilisateur n'a pas été exécutée ici.

`local_incumbent.log` détaille 228 contrôles : limite d'une itération sur LJ2,
budget d'évaluations, objectif/coordonnées cohérents, faisabilité, cache de
valeurs des particules interrompues, retour global, compteurs distincts de la
convergence, série et deux workers. Les chemins CG, LDL, METIS+LDL, MUMPS,
METIS+MUMPS et Ipopt/MUMPS sont inclus. Un problème analytique oscillant force
un échec de recherche linéaire après évaluation d'un point améliorant.

Les contrôles linéaires (`linear_*metis*.log`) portent sur un système symétrique
9×9, plusieurs valeurs numériques dont zéro→non-zéro, les résidus, l'inertie,
le rejet singulier, la reprise après échec et des minima LJ2 analytiques.
MUMPS confirme `INFOG(7)=1`, c'est-à-dire l'utilisation de la permutation
METIS externe. LDL est testé avec METIS 4 et 5 ; MUMPS+METIS est testé avec
METIS 4 seulement. Les tests sans METIS vérifient le rejet explicite d'une
demande d'ordonnancement indisponible.

## LJ38 et LJ76 : portée limitée, entrées figées

`lj_metis_final/` conserve les commandes, populations, JSON et journaux du
script `tests/validate_metis.py`. Pour chacune des deux tailles, il reprend
les **deux premières particules** de la campagne figée 1.3.0. Ce n'est pas une
campagne PF complète, ni une comparaison statistique sur 30 runs.

Pour chaque backend METIS natif, les exécutions à un et deux workers produisent
les mêmes coordonnées, objectifs et nombres d'évaluations/factorisations.
Les fichiers de départ sont identiques octet pour octet. Les compteurs
confirment les analyses METIS et, pour MUMPS, l'ordre utilisateur effectivement
accepté. Le témoin Ipopt/MUMPS reçoit les mêmes deux départs.

L'audit indépendant des points retournés passe sur les deux points de chaque
taille/backend natif dans ce test. Pour Ipopt, il échoue au seuil strict sur
les points énergétiquement meilleurs retenus : le meilleur candidat évalué
peut précéder l'itéré terminal. Cela ne justifie pas de rejeter son amélioration
ni de conclure à une supériorité générale du solveur natif.

Les temps de `summary.json` sont des mesures courtes dans un environnement
partagé ; aucun facteur d'accélération de référence n'est revendiqué.
La Hessienne LJ complète reste structurellement dense. Cette version ne
garantit ni l'exclusion des selles ni l'obtention des optima globaux LJ38/LJ76.

## Traçabilité historique

`previous_v1.3.0/` contient uniquement les anciennes validations, conservées
comme historique. Les journaux sans suffixe `final` et `lj_metis/` sont des
contrôles intermédiaires du développement 1.3.1 ; les conclusions ci-dessus
utilisent les journaux `final` et le test d'incumbent final.

MA27 reste raccordé dynamiquement mais n'a pas été exécuté ici, faute de HSL.
Les tests de graines/flux NumPy et de noyaux Python passent ; ils ne prouvent
pas une identité de trajectoire complète PF Python/C++.
