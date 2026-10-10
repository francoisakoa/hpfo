# Validation PF C++ 1.3.0

Validation sous Linux, GCC 13.3, Eigen 3.4, Ipopt 3.14.11 et MUMPS séquentiel
5.4.1, OpenMP, Python 3.12, NumPy 2.3.5. CPU AMD EPYC 9V74, quota de 8 CPU.
La machine de validation n'est pas le TRX50 de l'utilisateur.
Toutes les quantités réelles C++ et interfaces numériques sont en `double`
IEEE 754 binary64, sans fast-math ni contraction FMA. Les réglages précis sont
conservés dans `validation/environment.json`.

## Tests de correction

- Build natif, Ipopt et MUMPS activés : **8/8 groupes CTest réussis**.
- Build portable, Ipopt et MUMPS désactivés : **7/7 groupes réussis**.
- Cœur : 89 vérifications dans le build Ipopt.
- Frontière mémoire Eigen : 304 cycles d'allocation/redimensionnement/destruction.
- Nouveau Newton : 43 contrôles, dont gradient, Hessienne assemblée, H*v contre
  l'oracle dense et les différences finies, mode de translation, minima analytiques
  avec boules actives, coïncidences atomiques, LJ4, budget et rejeu par threads.
- LDL/MUMPS : systèmes définis positifs, résidus, détection d'inertie négative et
  de singularité, refactorisation après singularité, réutilisation de l'analyse,
  deux minima LJ analytiques. Résidus relatifs maximaux : 1.19e-16 et 1.68e-16.
- Intégration CLI : génération/rejeu, JSON, précision, budgets stricts en parallèle,
  exécution CG/LDL par threads, MUMPS par processus, absence HSL explicite et
  refus de l'option Ipopt `ldl` inexistante.
- Parité NumPy : 123 cas MD5, 800 000 normales identiques bit à bit,
  60 initialisations LJ38/LJ76, 2 534 220 coordonnées de départ exactes,
  6 itérations PF contrôlées exactes ; 467 assertions / 3 444 615 valeurs.
- Parité des opérateurs Python : 11 comparaisons, source chargée par AST.

Les journaux `validation/ctest_native.txt`, `validation/ctest_portable.txt` et
`validation/numpy_random_*.json` consignent les résultats. Aucun avertissement
n'est présent dans les compilations finales retenues.
**MA27 : interface implémentée, exécution non validée ici**, HSL étant absent.
La commande `pf_linear_check --backend ma27 --hsl-library ...` vérifie la
bibliothèque réelle chez l'utilisateur. Aucun gain MA27 n'est revendiqué.

## Benchmark local : points fournis identiques

Huit départs par taille, base 2024/run 1, première tranche du flux PF NumPy,
rayon 5, contraintes identiques, tolérance demandée 1e-8 et limite locale 5000.
Trois répétitions dans un ordre alterné, une et quatre tâches concurrentes ;
24 exécutions. Le tableau donne les médianes du temps du lot de huit départs.
Le temps inclut le pool et sa fermeture, exclut préparation, écritures et audits.
Les énergies médianes et le nombre de points passant l'audit indépendant
`||x-Projection(x-gradient)||_inf <= 1e-8` sont publiés avec les temps.

| Atomes | Méthode | Workers | Médiane (s) | Audit | Énergie médiane |
|---:|---|---:|---:|---:|---:|
| 38 | lj-newton | 1 | 0.102533 | 8/8 | -156.695861446 |
| 38 | lj-newton | 4 | 0.041912 | 8/8 | -156.695861446 |
| 38 | ipopt | 1 | 4.767616 | 2/8 | -154.406954482 |
| 38 | ipopt | 4 | 1.464588 | 2/8 | -154.406954482 |
| 76 | lj-newton | 1 | 0.541000 | 8/8 | -377.141436611 |
| 76 | lj-newton | 4 | 0.187327 | 8/8 | -377.141436611 |
| 76 | ipopt | 1 | 13.250205 | 0/8 | -381.531871631 |
| 76 | ipopt | 4 | 4.641109 | 0/8 | -381.531871631 |

CG utilise les dérivées exactes avec son propre arrêt non mis à l'échelle.
Ipopt utilise son arrêt interne avec mise à l'échelle et le portage conserve
son meilleur point admissible évalué par callback, qui peut différer de l'itéré
terminal. `local_converged` Ipopt ne signifie donc pas que ce point enregistré
passe l'audit plus strict. Les minima locaux diffèrent ; notamment l'énergie
médiane Ipopt est plus basse sur ces départs LJ76. Ces mesures ne sont pas une
comparaison à énergie finale ou critère d'arrêt interne identique.

Sur LJ38, rapport des médianes Ipopt/CG séquentiels : 46.5 ; accélération CG 1→4 workers : 2.45.

Sur LJ76, rapport des médianes Ipopt/CG séquentiels : 24.5 ; accélération CG 1→4 workers : 2.89.

Tous les points fournis sont identiques octet par octet entre méthodes.
Les points finaux sont identiques entre répétitions et entre 1/4 workers pour
chaque méthode. Ils ne sont pas identiques entre méthodes. Le solveur natif
projette un départ extérieur aux boules ; Ipopt garde sa propre initialisation.

Preuves : `validation/local_benchmark_final/summary.json`, commandes, journaux,
JSON par exécution, `.initial.pfpop` et `.final.pfpop`. Le script reproductible
est `tests/benchmark_local.py` ; les commandes complètes sont dans
`docs/INSTALL_TEST_COMMANDS.md`.

## Variantes directes LDL/MUMPS

Quatre départs par taille, LJ38/LJ76, chaque backend en séquentiel et avec quatre
workers : les 32 points résultats passent le même audit à 1e-8. Les résultats
sont identiques entre 1/4 workers pour chaque backend, et aucun secours CG
n'a été utilisé. Les compteurs de factorisation prouvent l'utilisation des
backends. Ces essais servent à la validation fonctionnelle ; leurs temps
n'ont pas été répétés dans un protocole isolé et aucun classement de vitesse
LDL/MUMPS n'en est déduit. Voir `validation/direct_checks.json` et les résultats
`validation/direct_*.json`.

## Intégration PF sur LJ38 et LJ76

Une exécution de vingt tours par méthode/taille, base 2024/run 1, top-k 5,
quatre workers, quatre threads globaux, sans cible ni arrêt de stagnation.
Populations conformes au défaut Python : 170/284. Les départs sont figés et
identiques pour les deux variantes PF. Tous les essais terminent les vingt tours.

| Atomes | Raffinement PF | Particules | Énergie finale | Temps observé (s) |
|---:|---|---:|---:|---:|
| 38 | lj-newton | 170 | -168.284124146 | 0.646 |
| 38 | ipopt | 170 | -171.792291554 | 22.695 |
| 76 | lj-newton | 284 | -395.508506026 | 3.410 |
| 76 | ipopt | 284 | -394.153104297 | 94.532 |

Ces quatre essais sont des contrôles d'intégration, non une campagne de
performance répétée ni une preuve de supériorité globale. Les trajectoires
se séparent dès que les minima locaux diffèrent. Fichiers :
`validation/pf_checks.json` et `validation/pf_LJ*/`.

## Limites scientifiques

La source Python archivée conserve son identité et ses réglages d'origine ;
sa capacité à reproduire la référence LJ76 revendiquée n'est pas établie.
Le nouveau solveur n'est pas présenté comme une reproduction des trajectoires
Python/Ipopt. Aucune campagne complète de trente runs ni aucun certificat
d'optimalité globale n'a été produit. Aucune validation numérique de MA27
réel ni campagne ASan/UBSan n'est revendiquée pour cette version.
