# PF C++ 1.3.1 — METIS+LDL et METIS+MUMPS, float64

Bibliothèque et exécutables C++17 pour le filtre particulaire PF/HPF, avec
un nouveau raffinement local Newton spécialisé pour Lennard–Jones 3D.
Le solveur natif calcule les dérivées exactes du premier et du second ordre,
respecte les bornes et les boules atomiques, et ne dépend pas d'Ipopt.
Les exécutions n'appellent pas Python ; Python intervient dans les tests.

## Nouveautés 1.3.1

- Tout candidat local évalué, admissible et améliorant est conservé, même sans
  convergence ou après rejet par la recherche linéaire/région de confiance.
  Le statut du solveur reste indépendant ; les budgets conservent le meilleur
  point global déjà évalué. Voir `docs/LOCAL_INCUMBENT.md`.
- `--newton-linear-ordering metis` avec `--newton-linear-solver ldl|mumps`.
- LDL : permutation METIS, `Eigen::SimplicialLDLT<double>` sans second
  réordonnancement, permutation du second membre et retour à l'ordre initial.
  Le mode `auto` conserve le LDLT dense de la 1.3.0.
- MUMPS : permutation METIS externe fournie dans `PERM_IN`, analyse séquentielle,
  contrôle explicite `INFOG(7)=1`. Aucun remplacement silencieux par un autre ordre.
- Analyse symbolique réutilisée entre les factorisations d'un même raffinement.
  La permutation du graphe LJ complet est mise en cache par dimension/processus.
- `linear_metis_analyses` et `mumps_user_ordering_analyses` prouvent le passage
  effectif dans ces chemins, y compris avec les workers.
- METIS est une dépendance optionnelle : `PF_WITH_METIS=ON`. Interfaces METIS 4
  et METIS 5 prises en charge ; les en-têtes doivent correspondre à la bibliothèque.

La Hessienne LJ exacte est structurellement dense. Aucune interaction n'est
supprimée pour fabriquer un graphe creux ; METIS ne garantit donc pas un gain
de temps sur LJ38/LJ76. Les dérivées et les factorisations restent en binary64.
Cette option porte sur le solveur Newton natif ; elle ne pilote pas l'ordre
interne du MUMPS utilisé par Ipopt. Le témoin Ipopt/MUMPS reste disponible.

La 1.3.1 ajoute l'ordonnancement et renforce la conservation des améliorations.
Elle ne corrige pas les limites de convergence
identifiées dans la 1.3.0 : stationnarité de premier ordre insuffisante pour
exclure une selle, et échec PF+Newton-CG LJ38 face à la réussite PF+Ipopt. Aucun nouveau
résultat d'optimalité globale n'est revendiqué.

Voir [docs/METIS.md](docs/METIS.md) et les commandes complètes dans
[docs/INSTALL_TEST_COMMANDS.md](docs/INSTALL_TEST_COMMANDS.md).

## Nouveautés 1.3.0

- `--local-refine lj-newton` : Newton avec région de confiance, produits
  Hessienne-vecteur analytiques par paires, préconditionnement, détection de
  courbure négative et garde par pas de Cauchy projeté.
- `--newton-linear-solver cg|ldl|ma27|mumps` : CG matriciellement implicite,
  ou Newton régularisé avec factorisation directe. CG est le choix par défaut
  du nouveau solveur ; la méthode locale historique par défaut reste Ipopt.
- `--hsl-library FILE` : même bibliothèque MA27 pour le backend natif et Ipopt.
- `--workers W` : threads OpenMP pour CG/LDL ; processus persistants isolés
  pour MA27/MUMPS natifs et pour Ipopt. `--threads T` règle les phases PF.
- `pf_local_bench` conserve chaque départ et chaque point renvoyé, chronomètre la
  minimisation locale et audite indépendamment la stationnarité/faisabilité.
- `pf_linear_check` vérifie la résolution linéaire, l'inertie et les cas LJ
  analytiques, y compris sur la bibliothèque HSL installée par l'utilisateur.
- Contrôles de compilation IEEE 754 **binary64** : toutes les quantités réelles
  utilisent `double`, équivalent du `float64` NumPy. Aucun mode float32 ou calcul
  mixte n'est présent dans le C++. `fast-math` est interdit et FMA non contracté.

| Backend local | CG | LDL (Eigen) | MA27 | MUMPS |
|---|---:|---:|---:|---:|
| Newton LJ natif | Oui | Oui | Interface HSL dynamique | Option de compilation |
| Ipopt | Non | Pas de backend autonome `ldl` | Selon build/HSL | Selon build |

Les chemins LDL et MUMPS sont validés ici ; MA27 est raccordé mais n'a pas été
exécuté faute de bibliothèque HSL dans l'environnement de validation. HSL n'est
pas redistribué. Les détails de l'algorithme, des contraintes et de son domaine
figurent dans [docs/LJ_NEWTON.md](docs/LJ_NEWTON.md).

## Installer et tester

Toutes les commandes pour l'archive placée dans
`/data/projects/PF_GlobalOptimization`, l'environnement `pf_cpp_110`, les
backends, les tests LJ38/LJ76 et les comparaisons depuis les mêmes points sont
regroupées dans **[docs/INSTALL_TEST_COMMANDS.md](docs/INSTALL_TEST_COMMANDS.md)**.

Compilation dans un environnement préparé :

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DPF_NATIVE=ON -DPF_WITH_OPENMP=ON \
  -DPF_WITH_IPOPT=ON -DPF_WITH_MUMPS=ON -DPF_WITH_METIS=ON \
  -DPF_MUMPS_INCLUDE="$CONDA_PREFIX/include" \
  -DPF_MUMPS_LIBRARY="$CONDA_PREFIX/lib/libdmumps_seq.so"
cmake --build build_release --parallel 8
ctest --test-dir build_release --output-on-failure
./build_release/pf_solver --version
```

Sans dépendance Ipopt/MUMPS, utiliser `PF_WITH_IPOPT=OFF` et `PF_WITH_MUMPS=OFF`.
CG/LDL restent disponibles. Avec le Makefile, les options correspondantes sont
`PF_WITH_IPOPT=0`, `PF_WITH_MUMPS=0` ; l'activation de MUMPS prend les paramètres
`PF_MUMPS_INCLUDE`, `PF_MUMPS_LIBDIR`, `PF_MUMPS_LIB=dmumps_seq`.
Linux/WSL, GCC/Clang et entiers non signés 128 bits sont nécessaires au générateur.
`PF_NATIVE=OFF` produit une compilation portable pour l'architecture cible.

Test PF, puis baseline multistart Ipopt depuis sa population initiale :

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
./build_release/pf_solver --atoms 38 76 --runs 1 --seed 2024 \
  --max-iter 1000 --max-time 3600 --no-improve-stop 400 \
  --threads 4 --workers 4 --refine-top-k 5 --local-max-iter 5000 --local-tol 1e-8 \
  --local-refine lj-newton --newton-linear-solver cg \
  --include-ipopt --ipopt-linear-solver mumps --output-dir results_newton130
```

Le dossier de sortie doit être nouveau. `--include-ipopt` ajoute un multistart
Ipopt avec son propre budget ; ce n'est pas PF utilisant Ipopt à chaque tour.
Ne pas fixer `--particles` pour utiliser le défaut Python `max(3N+56,50)`,
soit 170 pour LJ38 et 284 pour LJ76. Dans l'API, la taille est un paramètre
explicite de `Options` et garde son défaut historique 50.

## Graines, points initiaux et provenance

Les garanties de la version 1.2.0 sont conservées et retestées :

- Deux graines `int(md5(f"{base}_{N}_{run}_{offset}").hexdigest()[:8],16)`,
  offsets 0 et 1000, base 2024 et premier run 1.
- PCG64/SeedSequence, uniformes, normales Ziggurat, entiers et sélections
  compatibles avec NumPy 2.3.5 dans les cas testés.
- `x0` Python avec marge est reproduit. Dans la source fournie, PF ne l'utilise
  pas comme population : ses départs proviennent de `seed_pf` dans toute la boîte.
  Cette population est elle aussi reproduite, sans confusion entre les deux.
- `initialization.json` conserve les graines et `x0` ; `initial.pfpop` conserve
  la population effectivement fournie aux solveurs. `--population FILE`
  permet son rejeu en conservant la continuation correcte du flux PF.
- Les tirages ont lieu sur le coordinateur et les agrégations sont ordonnées.
  Changer le nombre de workers conserve les résultats des essais non limités ;
  un budget contraignant peut interrompre des tâches différentes.

La source archivée reste `ReleasePFJOGO-v1.0-jogo-r2.tar.gz`, SHA256
`b4ea0bb0d27b4de634ced239eda27994fc6d59df2c87f59a88a040b8c97c922d`.
Les quatre fichiers sont inchangés dans `reference/` et décrits par
`reference/SOURCES.json`. L'identité de cette source comme version scientifique
ayant reproduit LJ76 n'est **pas établie**. Le passage intégral en double
concerne le C++ ; les réglages float32 du Python archivé restent visibles comme
preuve de provenance et ne sont pas actifs dans le C++.

La parité des graines/départs ne signifie pas parité des trajectoires complètes :
le nouveau raffinement change les minima locaux et le Python contient des flux
auxiliaires non initialisés. Lire [docs/RANDOM_PROTOCOL.md](docs/RANDOM_PROTOCOL.md)
et [docs/PORTAGE.md](docs/PORTAGE.md) pour les écarts historiques.

## Résultats, contraintes et budgets

Les fichiers `pf.json`, `pf_iterations.csv`, `pf.xyz`, `initial.pfpop` et
`initialization.json` sont écrits pour chaque taille/run. Le multistart ajoute
`ipopt_multistart.*`. Les nombres sont écrits avec 17 chiffres significatifs.
Le JSON consigne les paramètres, compteurs, historique, backend et précision.

`--max-evals` borne toutes les évaluations d'objectif, y compris les locales.
Gradients, Hessiennes, produits H*v et opérations linéaires ont leurs compteurs
séparés. La réservation par tâche empêche un dépassement du budget d'objectif.
Les audits finaux sont hors budget et comptés séparément. Les limites de temps
sont vérifiées entre noyaux ; un noyau en cours n'est pas préempté.

Les phases PF agissent sur la boîte ; les raffinements locaux imposent aussi
les boules atomiques, conformément à la source. Un meilleur point PF peut être
extérieur aux boules. `--no-local-ball`/`--no-ipopt-ball` permet explicitement
les bornes seules. Le diagnostic PF historique `projected_gradient_inf` porte
sur la boîte ; `pf_local_bench` audite le domaine local complet avec
`||x-Projection(x-gradient)||_inf`.

`target_reached` n'est pas un certificat global. Le critère de convergence
Newton est du premier ordre ; l'utilisation de dérivées secondes exactes ne
constitue pas une certification de minimum strict. Les statuts de limite sont
exploitables (code 0) ; les erreurs de configuration/exécution ont le code 2.

## Validation et limites des performances

[docs/VALIDATION.md](docs/VALIDATION.md) publie les tests, les temps, les énergies
et les audits. Les journaux et populations sont livrés dans `validation/`.
Le gain local observé pour CG ne prouve pas une supériorité sur toutes les
campagnes PF et tous les backends. LJ38 et LJ76 ont aussi été exécutés sur
20 tours PF avec les populations par défaut et des départs identiques.
Cette livraison ne revendique pas l'atteinte de l'optimum global LJ76 ni une
validation statistique sur 30 runs.

Le correctif mémoire Eigen 1.1.1 est conservé : `-march=native` est une option
**PUBLIC** de la cible CMake `pf`, commune à la bibliothèque, aux exécutables,
aux workers et aux tests. Le test de frontière mémoire couvre 304 cycles.
Une application CMake doit lier `pf` pour hériter des paramètres ABI ; une
liaison manuelle de `libpf.a` doit garder les mêmes paramètres Eigen/SIMD.
